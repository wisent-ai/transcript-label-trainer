use super::*;

/// Accuracy and mean loss over the in-training slice. `None` for both when the
/// slice came out empty, which is what the Python metrics carried when the
/// Trainer reported no such key.
pub(crate) fn evaluate_slice(
    model: &Classifier,
    batch: &Batches,
    batch_size: usize,
) -> Result<(Option<f64>, Option<f64>)> {
    if batch.rows == 0 {
        return Ok((None, None));
    }
    let mut correct = 0usize;
    let mut total_loss = 0f64;
    let mut seen = 0usize;
    let index: Vec<usize> = (0..batch.rows).collect();
    for chunk in index.chunks(batch_size.max(1)) {
        let (ids, mask, targets) = batch.take(chunk)?;
        let logits = model.forward(&ids, &mask, false)?;
        let batch_loss = loss::cross_entropy(&logits, &targets)
            .map_err(|err| Error(format!("evaluation loss failed: {err}")))?;
        total_loss += scalar(&batch_loss)? as f64 * chunk.len() as f64;
        for (row, predicted) in argmax(&logits)?.into_iter().enumerate() {
            if predicted as usize == batch.labels[chunk[row]] {
                correct += 1;
            }
        }
        seen += chunk.len();
    }
    Ok((
        Some(correct as f64 / seen as f64),
        Some(total_loss / seen as f64),
    ))
}

/// `(value, confidence)` per text, batched the way `_hf_predict` batched.
pub(crate) fn infer(
    model: &Classifier,
    tokenizer: &Tokenizer,
    texts: &[String],
    classes: &[String],
    device: &Device,
) -> Result<Vec<(String, f64)>> {
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(PREDICT_BATCH) {
        let index: Vec<usize> = (0..chunk.len()).collect();
        let labels = vec![0usize; chunk.len()];
        let batch = Batches::encode(tokenizer, chunk, &labels, &index, device)?;
        let (ids, mask, _) = batch.take(&index)?;
        let logits = model.forward(&ids, &mask, false)?;
        let probabilities = ops::softmax(&logits, D::Minus1)
            .and_then(|p| p.to_vec2::<f32>())
            .map_err(|err| Error(format!("could not read the prediction: {err}")))?;
        for row in probabilities {
            let (best, confidence) =
                row.iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |best, (index, value)| {
                        if *value > best.1 {
                            (index, *value)
                        } else {
                            best
                        }
                    });
            out.push((classes[best].clone(), confidence as f64));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// architecture
// ---------------------------------------------------------------------------

/// The encoders this file can fine-tune. The head on top of each is the one
/// transformers puts there: distilbert pools the `[CLS]` state through
/// `pre_classifier` + ReLU, bert through its pretrained `pooler.dense` + tanh.
#[derive(Clone, Copy)]
pub(crate) enum Architecture {
    DistilBert,
    Bert,
}

impl Architecture {
    pub(crate) fn from_config(config: &Value) -> Result<Self> {
        match config.get("model_type").and_then(Value::as_str) {
            Some("distilbert") => Ok(Self::DistilBert),
            Some("bert") => Ok(Self::Bert),
            Some(other) => Err(Error(format!(
                "model_type '{other}' is not one of the sequence-classification architectures \
                 this trainer can fine-tune (distilbert, bert); try \
                 distilbert-base-multilingual-cased"
            ))),
            None => Err(Error(
                "the model's config.json has no model_type, so its architecture cannot be \
                 identified; pick a standard HuggingFace encoder such as \
                 distilbert-base-multilingual-cased"
                    .into(),
            )),
        }
    }

    /// Parameter prefixes worth keeping out of a checkpoint: the encoder, plus
    /// the head names this file writes. Everything else in a base checkpoint
    /// (the masked-LM head, most often) is dropped, exactly as
    /// `AutoModelForSequenceClassification` dropped it.
    pub(crate) fn roots(self) -> &'static [&'static str] {
        match self {
            Self::DistilBert => &[
                "embeddings.",
                "transformer.",
                "pre_classifier.",
                "classifier.",
            ],
            Self::Bert => &["embeddings.", "encoder.", "pooler.", "classifier."],
        }
    }

    pub(crate) fn pooler_name(self) -> &'static str {
        match self {
            Self::DistilBert => "pre_classifier",
            Self::Bert => "pooler.dense",
        }
    }

    pub(crate) fn architecture_name(self) -> &'static str {
        match self {
            Self::DistilBert => "DistilBertForSequenceClassification",
            Self::Bert => "BertForSequenceClassification",
        }
    }

    pub(crate) fn hidden_size(self, config: &Value) -> Result<usize> {
        let key = match self {
            Self::DistilBert => "dim",
            Self::Bert => "hidden_size",
        };
        config
            .get(key)
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| Error(format!("the model's config.json has no numeric '{key}'")))
    }

    pub(crate) fn max_positions(self, config: &Value) -> Result<usize> {
        config
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| {
                Error("the model's config.json has no numeric 'max_position_embeddings'".into())
            })
    }

    /// The dropout transformers applies between the pooled state and the
    /// classifier: `seq_classif_dropout` for distilbert, `classifier_dropout`
    /// falling back to `hidden_dropout_prob` for bert.
    pub(crate) fn head_dropout(self, config: &Value) -> f32 {
        let configured = match self {
            Self::DistilBert => config.get("seq_classif_dropout").and_then(Value::as_f64),
            Self::Bert => config
                .get("classifier_dropout")
                .and_then(Value::as_f64)
                .or_else(|| config.get("hidden_dropout_prob").and_then(Value::as_f64)),
        };
        configured.unwrap_or(0.1) as f32
    }
}

#[derive(Deserialize)]
pub(crate) struct DistilBertConfig {
    pub(crate) vocab_size: usize,
    pub(crate) dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_heads: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) max_position_embeddings: usize,
    #[serde(default = "default_gelu")]
    pub(crate) activation: String,
}

#[derive(Deserialize)]
pub(crate) struct BertConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub(crate) type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub(crate) layer_norm_eps: f64,
    #[serde(default = "default_gelu")]
    pub(crate) hidden_act: String,
}

pub(crate) fn default_gelu() -> String {
    "gelu".to_string()
}

pub(crate) fn default_type_vocab_size() -> usize {
    2
}

pub(crate) fn default_layer_norm_eps() -> f64 {
    DEFAULT_LAYER_NORM_EPS
}

/// The feed-forward activations these two architectures name. HuggingFace's
/// plain `gelu` is the exact error-function form; the tanh approximation is
/// what `gelu_new` and `gelu_pytorch_tanh` ask for.
#[derive(Clone, Copy)]
pub(crate) enum Activation {
    Gelu,
    GeluTanh,
    Relu,
}

impl Activation {
    pub(crate) fn parse(name: &str) -> Result<Self> {
        match name {
            "gelu" => Ok(Self::Gelu),
            "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => Ok(Self::GeluTanh),
            "relu" => Ok(Self::Relu),
            other => Err(Error(format!(
                "the model's activation '{other}' is not supported; \
                 gelu, gelu_new and relu are"
            ))),
        }
    }

    pub(crate) fn apply(self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            Self::GeluTanh => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------

/// Layer normalisation out of primitive ops.
///
/// `candle_nn::LayerNorm` dispatches contiguous input to a fused kernel with
/// no backward pass, which silently truncates the autograd graph; see the
/// module documentation. `ops::layer_norm_slow` is the same maths, and it
/// differentiates.
pub(crate) struct LayerNorm {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
    pub(crate) eps: f64,
}

impl LayerNorm {
    pub(crate) fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            bias: vb.get(size, "bias")?,
            eps,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        ops::layer_norm_slow(xs, &self.weight, &self.bias, self.eps as f32)
    }
}
