use super::*;

impl HubFiles {
    pub(crate) fn fetch(model_id: &str) -> Result<Self> {
        let api = Api::new()
            .map_err(|err| Error(format!("could not reach the HuggingFace hub: {err}")))?;
        let repo = api.model(model_id.to_string());
        let config = repo
            .get("config.json")
            .map_err(|err| Error(format!("could not download {model_id}/config.json: {err}")))?;
        let tokenizer = repo.get("tokenizer.json").map_err(|err| {
            Error(format!(
                "could not download {model_id}/tokenizer.json ({err}); fine-tuning needs a model \
                 that ships a fast tokenizer, such as distilbert-base-multilingual-cased"
            ))
        })?;
        let safetensors = repo.get("model.safetensors").ok();
        let pytorch = match safetensors {
            Some(_) => None,
            None => Some(repo.get("pytorch_model.bin").map_err(|err| {
                Error(format!(
                    "{model_id} publishes neither model.safetensors nor pytorch_model.bin ({err})"
                ))
            })?),
        };
        Ok(Self {
            config,
            tokenizer,
            safetensors,
            pytorch,
        })
    }

    pub(crate) fn weights(&self, device: &Device) -> Result<Vec<(String, Tensor)>> {
        if let Some(path) = &self.safetensors {
            let loaded = candle_core::safetensors::load(path, device)
                .map_err(|err| Error(format!("could not read {}: {err}", path.display())))?;
            return Ok(loaded.into_iter().collect());
        }
        let path = self.pytorch.as_ref().expect("one of the two is always set");
        candle_core::pickle::read_all(path)
            .map_err(|err| Error(format!("could not read {}: {err}", path.display())))
    }
}

pub(crate) fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|err| Error(format!("could not read {}: {err}", path.display())))
}

/// Truncation and padding as the Python path asked for them: truncate at
/// `max_length`, pad the whole encoded split to its longest member.
pub(crate) fn prepare_tokenizer(tokenizer: &mut Tokenizer, max_length: usize) -> Result<()> {
    let pad_token = ["[PAD]", "<pad>"]
        .into_iter()
        .find(|token| tokenizer.token_to_id(token).is_some())
        .unwrap_or("[PAD]");
    let pad_id = tokenizer.token_to_id(pad_token).unwrap_or(0);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|err| Error(format!("could not configure truncation: {err}")))?
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id,
            pad_token: pad_token.to_string(),
            ..Default::default()
        }));
    Ok(())
}

/// One encoded split, padded to a single width so any subset of its rows can
/// be stacked into a batch.
pub(crate) struct Batches {
    pub(crate) ids: Vec<u32>,
    pub(crate) mask: Vec<u32>,
    pub(crate) labels: Vec<usize>,
    pub(crate) width: usize,
    pub(crate) rows: usize,
    pub(crate) device: Device,
}

impl Batches {
    pub(crate) fn encode(
        tokenizer: &Tokenizer,
        texts: &[String],
        label_ids: &[usize],
        index: &[usize],
        device: &Device,
    ) -> Result<Self> {
        if index.is_empty() {
            return Ok(Self {
                ids: Vec::new(),
                mask: Vec::new(),
                labels: Vec::new(),
                width: 0,
                rows: 0,
                device: device.clone(),
            });
        }
        let batch: Vec<&str> = index.iter().map(|row| texts[*row].as_str()).collect();
        let encodings = tokenizer
            .encode_batch(batch, true)
            .map_err(|err| Error(format!("tokenization failed: {err}")))?;
        let width = encodings[0].get_ids().len();
        let mut ids = Vec::with_capacity(width * encodings.len());
        let mut mask = Vec::with_capacity(width * encodings.len());
        for encoding in &encodings {
            ids.extend_from_slice(encoding.get_ids());
            mask.extend_from_slice(encoding.get_attention_mask());
        }
        Ok(Self {
            ids,
            mask,
            labels: index.iter().map(|row| label_ids[*row]).collect(),
            width,
            rows: encodings.len(),
            device: device.clone(),
        })
    }

    /// `(input_ids, attention_mask, targets)` for the rows named by `rows`.
    pub(crate) fn take(&self, rows: &[usize]) -> Result<(Tensor, Tensor, Tensor)> {
        let mut ids = Vec::with_capacity(rows.len() * self.width);
        let mut mask = Vec::with_capacity(rows.len() * self.width);
        let mut targets = Vec::with_capacity(rows.len());
        for row in rows {
            let start = row * self.width;
            ids.extend_from_slice(&self.ids[start..start + self.width]);
            mask.extend_from_slice(&self.mask[start..start + self.width]);
            targets.push(self.labels[*row] as u32);
        }
        let shape = (rows.len(), self.width);
        let ids = Tensor::from_vec(ids, shape, &self.device)
            .map_err(|err| Error(format!("could not build the input batch: {err}")))?;
        let mask = Tensor::from_vec(mask, shape, &self.device)
            .map_err(|err| Error(format!("could not build the attention mask: {err}")))?;
        let targets = Tensor::from_vec(targets, rows.len(), &self.device)
            .map_err(|err| Error(format!("could not build the target batch: {err}")))?;
        Ok((ids, mask, targets))
    }
}

// ---------------------------------------------------------------------------
// artifact
// ---------------------------------------------------------------------------

/// `save_pretrained`'s output, as candle writes it: the weights under the names
/// this file loads them back by, the base config carrying the label mapping
/// this fine-tune learned, and the fast tokenizer.
pub(crate) fn save_artifact(
    out_dir: &Path,
    base_config: &Value,
    arch: Architecture,
    classes: &[String],
    vars: &[(String, Var)],
    tokenizer: &Tokenizer,
) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(vars.len());
    for (name, var) in vars {
        let tensor = var
            .as_tensor()
            .contiguous()
            .and_then(|tensor| tensor.to_device(&Device::Cpu))
            .map_err(|err| Error(format!("could not read {name} back: {err}")))?;
        tensors.insert(name.clone(), tensor);
    }
    let weights = out_dir.join("model.safetensors");
    candle_core::safetensors::save(&tensors, &weights)
        .map_err(|err| Error(format!("could not write {}: {err}", weights.display())))?;

    let mut config = base_config.clone();
    let object = config
        .as_object_mut()
        .ok_or_else(|| Error("the model's config.json is not an object".into()))?;
    object.insert("architectures".into(), json!([arch.architecture_name()]));
    object.insert("num_labels".into(), json!(classes.len()));
    object.insert(
        "id2label".into(),
        Value::Object(
            classes
                .iter()
                .enumerate()
                .map(|(index, label)| (index.to_string(), json!(label)))
                .collect(),
        ),
    );
    object.insert(
        "label2id".into(),
        Value::Object(
            classes
                .iter()
                .enumerate()
                .map(|(index, label)| (label.clone(), json!(index)))
                .collect(),
        ),
    );
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&config)? + "\n",
    )?;

    let path = out_dir.join("tokenizer.json");
    tokenizer
        .save(&path, true)
        .map_err(|err| Error(format!("could not write {}: {err}", path.display())))?;
    Ok(())
}

/// The classes an artifact was fine-tuned on, in label-id order.
pub(crate) fn classes_from_config(config: &Value) -> Result<Vec<String>> {
    let id2label = config
        .get("id2label")
        .and_then(Value::as_object)
        .ok_or_else(|| Error("the artifact's config.json has no id2label mapping".into()))?;
    let mut classes = vec![String::new(); id2label.len()];
    for (index, label) in id2label {
        let position: usize = index.parse().map_err(|_| {
            Error(format!(
                "the artifact's id2label has a non-numeric key '{index}'"
            ))
        })?;
        let label = label
            .as_str()
            .ok_or_else(|| Error("the artifact's id2label holds a non-string label".into()))?;
        *classes
            .get_mut(position)
            .ok_or_else(|| Error(format!("the artifact's id2label skips index {position}")))? =
            label.to_string();
    }
    Ok(classes)
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// `re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("-")`, the directory name
/// the artifact layout has always used.
pub(crate) fn sanitize_model_id(model_id: &str) -> String {
    let mut out = String::with_capacity(model_id.len());
    let mut in_run = false;
    for ch in model_id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
            out.push(ch);
            in_run = false;
        } else if !in_run {
            out.push_str("--");
            in_run = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// CPU by default, Metal when candle reports it available: the shape of the
/// Python `_hf_device`, which asked torch for MPS and fell back to CPU.
/// `metal_if_available` is candle's own report, true only when candle-core was
/// compiled with its `metal` feature on a machine that has a Metal device.
pub(crate) fn pick_device() -> (Device, &'static str) {
    let device = Device::metal_if_available(0).unwrap_or(Device::Cpu);
    let name = if device.is_metal() { "metal" } else { "cpu" };
    (device, name)
}

/// Per-class counts ordered by class, the `_class_counts` of the Python path.
pub(crate) fn class_counts(values: &[String]) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for value in values {
        *counts.entry(value.clone()).or_insert(0usize) += 1;
    }
    counts
}
