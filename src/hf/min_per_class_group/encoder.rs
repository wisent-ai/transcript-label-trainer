use super::*;

impl Encoder {
    pub(crate) fn load(arch: Architecture, config: &Value, vb: &VarBuilder) -> Result<Self> {
        match arch {
            Architecture::DistilBert => {
                let cfg: DistilBertConfig = serde_json::from_value(config.clone())
                    .map_err(|err| Error(format!("unusable distilbert config.json: {err}")))?;
                Ok(Self::DistilBert(DistilBert::load(&cfg, vb)?))
            }
            Architecture::Bert => {
                let cfg: BertConfig = serde_json::from_value(config.clone())
                    .map_err(|err| Error(format!("unusable bert config.json: {err}")))?;
                Ok(Self::Bert(Bert::load(&cfg, vb)?))
            }
        }
    }

    pub(crate) fn forward(&self, ids: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::DistilBert(model) => model.forward(ids, padding),
            Self::Bert(model) => model.forward(ids, padding),
        }
    }
}

// ---------------------------------------------------------------------------
// classifier
// ---------------------------------------------------------------------------

pub(crate) struct Classifier {
    pub(crate) encoder: Encoder,
    pub(crate) pooler: Linear,
    pub(crate) dropout: HeadDropout,
    pub(crate) head: Linear,
}

impl Classifier {
    /// Assemble the encoder plus the classification head over a checkpoint.
    ///
    /// `trainable` turns every float parameter into a candle variable and
    /// creates the head weights a base checkpoint does not carry; without it
    /// the checkpoint has to be complete, which is what a saved artifact is.
    pub(crate) fn load(
        arch: Architecture,
        config: &Value,
        raw: Vec<(String, Tensor)>,
        num_labels: usize,
        device: &Device,
        trainable: bool,
        rng: &mut ChaCha8Rng,
    ) -> Result<(Self, Vec<(String, Var)>)> {
        let hidden = arch.hidden_size(config)?;
        let mut tensors = normalize(raw, config, arch, device)?;
        if tensors.is_empty() {
            return Err(Error(
                "the checkpoint holds no parameter this architecture recognises; the model id \
                 and its config.json disagree"
                    .into(),
            ));
        }

        let mut vars: Vec<(String, Var)> = Vec::new();
        if trainable {
            let std = config
                .get("initializer_range")
                .and_then(Value::as_f64)
                .unwrap_or(0.02);
            // A classifier head sized for someone else's labels is useless
            // here, so it is re-initialised rather than reshaped.
            let mismatched = tensors
                .get("classifier.weight")
                .is_some_and(|weight| weight.dims() != [num_labels, hidden]);
            if mismatched {
                tensors.remove("classifier.weight");
                tensors.remove("classifier.bias");
            }
            let pooler = arch.pooler_name();
            init_linear(&mut tensors, pooler, hidden, hidden, std, device, rng)?;
            init_linear(
                &mut tensors,
                "classifier",
                num_labels,
                hidden,
                std,
                device,
                rng,
            )?;

            let mut names: Vec<String> = tensors.keys().cloned().collect();
            names.sort_unstable();
            for name in names {
                let tensor = tensors.get(&name).expect("name came from this map");
                let var = Var::from_tensor(tensor)
                    .map_err(|err| Error(format!("could not make {name} trainable: {err}")))?;
                tensors.insert(name.clone(), var.as_tensor().clone());
                vars.push((name, var));
            }
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        let encoder = Encoder::load(arch, config, &vb)?;
        let pooler = candle_nn::linear(hidden, hidden, vb.pp(arch.pooler_name()))
            .map_err(|err| Error(format!("could not load {}: {err}", arch.pooler_name())))?;
        let head = candle_nn::linear(hidden, num_labels, vb.pp("classifier"))
            .map_err(|err| Error(format!("could not load the classification head: {err}")))?;

        Ok((
            Self {
                encoder,
                pooler,
                dropout: HeadDropout {
                    probability: arch.head_dropout(config) as f64,
                    rng: std::cell::RefCell::new(ChaCha8Rng::seed_from_u64(rng.next_u64())),
                },
                head,
            },
            vars,
        ))
    }

    /// Logits for one batch. `mask` is the HuggingFace attention mask: 1 for a
    /// real token, 0 for padding.
    pub(crate) fn forward(&self, ids: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
        let logits = || -> candle_core::Result<Tensor> {
            // (batch, 1, 1, tokens), 1 where the attention must not look.
            let padding = mask.eq(0u32)?.unsqueeze(1)?.unsqueeze(1)?;
            let hidden = self.encoder.forward(ids, &padding)?;
            // Slicing row 0 out of every sequence leaves a strided view, which
            // the Metal matmul refuses; the head needs it packed.
            let cls = hidden.i((.., 0))?.contiguous()?;
            let pooled = match self.encoder {
                Encoder::DistilBert(_) => self.pooler.forward(&cls)?.relu()?,
                Encoder::Bert(_) => self.pooler.forward(&cls)?.tanh()?,
            };
            self.head.forward(&self.dropout.forward(&pooled, train)?)
        };
        logits().map_err(|err| Error(format!("the forward pass failed: {err}")))
    }
}

/// Inverted dropout over the pooled state.
///
/// candle's own `Dropout` draws from the device RNG, which `Device::set_seed`
/// cannot seed on CPU, so a fine-tune would not be reproducible there. This
/// draws from the run's own seeded stream instead, on every device.
pub(crate) struct HeadDropout {
    pub(crate) probability: f64,
    pub(crate) rng: std::cell::RefCell<ChaCha8Rng>,
}

impl HeadDropout {
    pub(crate) fn forward(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        if !train || self.probability <= 0.0 {
            return Ok(xs.clone());
        }
        let (rows, columns) = xs.dims2()?;
        let keep = (1.0 - self.probability) as f32;
        let mut rng = self.rng.borrow_mut();
        let mask: Vec<f32> = (0..rows * columns)
            .map(|_| {
                if rng.random::<f64>() < self.probability {
                    0.0
                } else {
                    1.0 / keep
                }
            })
            .collect();
        xs.mul(&Tensor::from_vec(mask, (rows, columns), xs.device())?)
    }
}

/// A checkpoint under this file's parameter names: the `<model_type>.` prefix
/// a task-headed checkpoint carries is stripped, integer buffers and other
/// task heads are dropped, and everything left is float32 on `device`.
pub(crate) fn normalize(
    raw: Vec<(String, Tensor)>,
    config: &Value,
    arch: Architecture,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let prefix = config
        .get("model_type")
        .and_then(Value::as_str)
        .map(|model_type| format!("{model_type}."))
        .unwrap_or_default();
    let mut out = HashMap::new();
    for (name, tensor) in raw {
        let name = rename_legacy(name.strip_prefix(prefix.as_str()).unwrap_or(&name));
        if !arch.roots().iter().any(|root| name.starts_with(root)) {
            continue;
        }
        if !tensor.dtype().is_float() {
            continue;
        }
        let tensor = tensor
            .to_device(device)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
            .map_err(|err| Error(format!("could not place {name} on the device: {err}")))?;
        out.insert(name, tensor);
    }
    Ok(out)
}

/// The parameter rename transformers performs on load: checkpoints published
/// before the TensorFlow port was retired spell layer-norm parameters `gamma`
/// and `beta`. `bert-base-multilingual-cased` is one of them.
pub(crate) fn rename_legacy(name: &str) -> String {
    match name.strip_suffix(".gamma") {
        Some(stem) => format!("{stem}.weight"),
        None => match name.strip_suffix(".beta") {
            Some(stem) => format!("{stem}.bias"),
            None => name.to_string(),
        },
    }
}

/// Create a linear layer's parameters when the checkpoint has none: normal
/// weights at the config's `initializer_range` and a zero bias, the
/// transformers initialisation for a fresh head.
///
/// The samples come from the run's own seeded stream rather than
/// `Tensor::randn`, because candle's CPU device RNG cannot be seeded and a
/// fine-tune has to start from the same head on every device.
pub(crate) fn init_linear(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    std: f64,
    device: &Device,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let weight = format!("{name}.weight");
    if !tensors.contains_key(&weight) {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| normal(rng, std) as f32)
            .collect();
        let value = Tensor::from_vec(values, (out_dim, in_dim), device)
            .map_err(|err| Error(format!("could not initialise {weight}: {err}")))?;
        tensors.insert(weight, value);
    }
    let bias = format!("{name}.bias");
    if !tensors.contains_key(&bias) {
        let value = Tensor::zeros(out_dim, DType::F32, device)
            .map_err(|err| Error(format!("could not initialise {bias}: {err}")))?;
        tensors.insert(bias, value);
    }
    Ok(())
}

/// One sample from a zero-mean normal distribution, Box-Muller over the
/// stream's uniforms. `rand` alone has no normal distribution and this file is
/// not worth a dependency on `rand_distr` for two head matrices.
pub(crate) fn normal(rng: &mut ChaCha8Rng, std: f64) -> f64 {
    let uniform = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let angle = std::f64::consts::TAU * rng.random::<f64>();
    std * (-2.0 * uniform.ln()).sqrt() * angle.cos()
}

// ---------------------------------------------------------------------------
// hub, tokenizer, batching
// ---------------------------------------------------------------------------

/// The files a fine-tune starts from, resolved through the hub cache.
pub(crate) struct HubFiles {
    pub(crate) config: PathBuf,
    pub(crate) tokenizer: PathBuf,
    pub(crate) safetensors: Option<PathBuf>,
    pub(crate) pytorch: Option<PathBuf>,
}
