//! The HuggingFace fine-tune backend, ported from `model.py`'s `_train_hf`,
//! `_hf_predict` and `_infer_hf` onto candle. Compiled only under the `hf`
//! feature; when it is off, `model.rs` refuses `--model` with the actionable
//! rebuild message and never reaches this file.
//!
//! What the Python path did with torch + transformers this does with candle:
//! pull the base model from the hub, put a sequence-classification head on it,
//! fine-tune the whole thing with AdamW, and save weights, tokenizer and the
//! metrics fragment into `<aspect dir>/hf-<sanitized model id>/`.
//!
//! Two architectures load, the two an operator training mixed Polish/English
//! transcripts would reach for: `distilbert` (the documented default,
//! `distilbert-base-multilingual-cased`) and `bert`. Anything else fails with
//! a sentence naming what is supported instead of loading garbage weights.
//!
//! The encoders are spelled out here rather than taken from
//! candle-transformers, and the reason is load-bearing: that crate's blocks
//! normalise through `candle_nn::LayerNorm`, which dispatches to a fused
//! kernel registered with `apply_op3_no_bwd`. A tensor produced by that kernel
//! carries no backward op, so `loss.backward()` stops at the first layer norm
//! and the gradient never reaches the encoder. Measured against
//! `distilbert-base-multilingual-cased`: 4 of 104 parameters received a
//! gradient, and every encoder weight came out of a fine-tune bit-identical to
//! the checkpoint. The same blocks written out of primitive ops (including
//! `ops::layer_norm_slow`) differentiate end to end.
//!
//! `model.rs` owns everything around this: the base metrics, the frozen
//! evaluation split, `holdout_evaluation`, the job block, and writing
//! `metrics.json` into the directory `train` returns.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{loss, ops, AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW, VarBuilder};
use hf_hub::api::sync::Api;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::util::{Error, Result, TrainFailure};

/// HF fine-tuning needs 2 sessions per class on the training side so the
/// stratified in-training split keeps every class on both sides. The floor of
/// 8 labeled sessions overall lives in `model.rs`, where both backends share
/// it; this one is the HF path's own.
const MIN_PER_CLASS: usize = 2;

/// Inference batch size, the 8 the Python `_hf_predict` hardcoded.
const PREDICT_BATCH: usize = 8;

/// The `TrainingArguments` defaults the Python path inherited by passing none
/// of them: seed 0, linear learning-rate decay to zero with no warmup, no
/// weight decay, gradients clipped at a global norm of 1.
const SEED: u64 = 0;
const MAX_GRAD_NORM: f64 = 1.0;
const WEIGHT_DECAY: f64 = 0.0;

/// Both architectures normalise at eps 1e-12; bert lets its config say so.
const DEFAULT_LAYER_NORM_EPS: f64 = 1e-12;

/// The `in_training_eval` note, verbatim from the Python metrics.
const IN_TRAINING_NOTE: &str = "stratified slice of the training side, resplit every run";

/// The hyperparameters `train` was invoked with, plus the aspect, which only
/// appears in the too-few-sessions-per-class message.
pub struct TrainConfig<'a> {
    pub aspect: &'a str,
    pub model_id: &'a str,
    pub epochs: f64,
    pub batch_size: usize,
    pub lr: f64,
    pub max_length: usize,
}

/// A finished fine-tune: where it landed, and the part of `metrics.json` this
/// backend owns. `model.rs` merges the fragment after its own base fields and
/// writes the file.
pub struct Trained {
    pub dir: PathBuf,
    pub metrics: Map<String, Value>,
}

/// Fine-tune `config.model_id` on the training side of a plan.
///
/// `aspect_dir` is `<training root>/models/<out name>`; the artifact goes into
/// its `hf-<sanitized model id>` subdirectory, which this creates.
pub fn train(
    aspect_dir: &Path,
    texts: &[String],
    values: &[String],
    config: &TrainConfig<'_>,
) -> Result<Trained, TrainFailure> {
    let counts = class_counts(values);
    let too_small: Vec<(&String, usize)> = counts
        .iter()
        .filter(|(_, n)| **n < MIN_PER_CLASS)
        .map(|(value, n)| (value, *n))
        .collect();
    if !too_small.is_empty() {
        let detail = too_small
            .iter()
            .map(|(value, n)| format!("'{value}' has {n}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(TrainFailure::NotEnoughData(format!(
            "aspect '{}': HF fine-tuning requires at least {MIN_PER_CLASS} sessions per class \
             on the training side ({detail}). Add labels with 'transcript-lake label add' \
             and retry.",
            config.aspect
        )));
    }

    // `sorted(counts)` in Python: the class order the label ids are assigned in.
    let classes: Vec<String> = counts.keys().cloned().collect();
    let label_ids: Vec<usize> = values
        .iter()
        .map(|value| {
            classes
                .iter()
                .position(|class| class == value)
                .expect("every value was just counted")
        })
        .collect();

    Ok(fine_tune(aspect_dir, texts, &label_ids, &classes, config)?)
}

/// `(value, confidence)` per text from a saved artifact directory. This is the
/// inference path: it loads the fine-tune back, it never trains.
pub fn predict(
    artifact_dir: &Path,
    texts: &[String],
    max_length: usize,
) -> Result<Vec<(String, f64)>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let config: Value = serde_json::from_slice(&std::fs::read(artifact_dir.join("config.json"))?)?;
    let classes = classes_from_config(&config)?;
    let (device, _) = pick_device();

    let weights = artifact_dir.join("model.safetensors");
    if !weights.is_file() {
        return Err(Error(format!(
            "artifact {} has no model.safetensors; retrain it with \
             'transcript-label-trainer train --model <id>'",
            artifact_dir.display()
        )));
    }
    let raw = candle_core::safetensors::load(&weights, &device)
        .map_err(|err| Error(format!("could not read {}: {err}", weights.display())))?
        .into_iter()
        .collect::<Vec<_>>();

    let arch = Architecture::from_config(&config)?;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let (model, _) =
        Classifier::load(arch, &config, raw, classes.len(), &device, false, &mut rng)?;
    let mut tokenizer = load_tokenizer(&artifact_dir.join("tokenizer.json"))?;
    prepare_tokenizer(&mut tokenizer, arch.max_positions(&config)?.min(max_length))?;

    infer(&model, &tokenizer, texts, &classes, &device)
}

// ---------------------------------------------------------------------------
// training
// ---------------------------------------------------------------------------

fn fine_tune(
    aspect_dir: &Path,
    texts: &[String],
    label_ids: &[usize],
    classes: &[String],
    config: &TrainConfig<'_>,
) -> Result<Trained> {
    let (device, device_name) = pick_device();

    let files = HubFiles::fetch(config.model_id)?;
    let base_config: Value = serde_json::from_slice(&std::fs::read(&files.config)?)?;
    let arch = Architecture::from_config(&base_config)?;
    // A model with 512 learned positions cannot look up token 512, so the
    // requested max length is capped by the base model's own window. The
    // request is still what the hyperparameters record, exactly as the Python
    // metrics recorded whatever was passed.
    let max_length = arch.max_positions(&base_config)?.min(config.max_length);

    let mut tokenizer = load_tokenizer(&files.tokenizer)?;
    prepare_tokenizer(&mut tokenizer, max_length)?;

    // A stratified slice of the TRAINING side, resplit on every run, so the
    // fine-tune has a loss curve to watch. It is not the frozen holdout: that
    // one is the same sessions every run and no backend ever trains on it.
    let n_test = std::cmp::max(classes.len(), (texts.len() as f64 * 0.2).round() as usize);
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let (train_index, eval_index) = stratified_split(label_ids, classes.len(), n_test, &mut rng);

    let train_batch = Batches::encode(&tokenizer, texts, label_ids, &train_index, &device)?;
    let eval_batch = Batches::encode(&tokenizer, texts, label_ids, &eval_index, &device)?;

    let raw = files.weights(&device)?;
    let (model, vars) =
        Classifier::load(arch, &base_config, raw, classes.len(), &device, true, &mut rng)?;
    let stepped: Vec<Var> = vars.iter().map(|(_, var)| var.clone()).collect();

    let steps_per_epoch = train_batch.rows.div_ceil(config.batch_size.max(1)).max(1);
    let total_steps = ((steps_per_epoch as f64 * config.epochs).ceil() as usize).max(1);
    let mut optimizer = AdamW::new(
        stepped.clone(),
        ParamsAdamW {
            lr: config.lr,
            weight_decay: WEIGHT_DECAY,
            ..Default::default()
        },
    )
    .map_err(|err| Error(format!("could not build the AdamW optimizer: {err}")))?;

    let epochs = (config.epochs.ceil() as usize).max(1);
    let mut order: Vec<usize> = (0..train_batch.rows).collect();
    let mut step = 0usize;
    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let mut epoch_loss = 0f64;
        let mut epoch_steps = 0usize;
        for chunk in order.chunks(config.batch_size.max(1)) {
            // Linear decay to zero over the whole run, no warmup.
            optimizer.set_learning_rate(config.lr * (1.0 - step as f64 / total_steps as f64));
            let (ids, mask, targets) = train_batch.take(chunk)?;
            let logits = model.forward(&ids, &mask, true)?;
            let batch_loss = loss::cross_entropy(&logits, &targets)
                .map_err(|err| Error(format!("loss failed at step {step}: {err}")))?;
            let mut grads = batch_loss
                .backward()
                .map_err(|err| Error(format!("backward pass failed at step {step}: {err}")))?;
            clip_grads(&mut grads, &stepped, MAX_GRAD_NORM)?;
            optimizer
                .step(&grads)
                .map_err(|err| Error(format!("optimizer step {step} failed: {err}")))?;
            epoch_loss += scalar(&batch_loss)? as f64;
            epoch_steps += 1;
            step += 1;
            if step >= total_steps {
                break;
            }
        }
        if epoch_steps > 0 {
            eprintln!(
                "epoch {}/{epochs}: train_loss={:.4} ({epoch_steps} step(s) of {total_steps}, \
                 device={device_name})",
                epoch + 1,
                epoch_loss / epoch_steps as f64,
            );
        }
        if step >= total_steps {
            break;
        }
    }

    let (eval_accuracy, eval_loss) = evaluate_slice(&model, &eval_batch, config.batch_size)?;

    let out_dir = aspect_dir.join(format!("hf-{}", sanitize_model_id(config.model_id)));
    std::fs::create_dir_all(&out_dir)?;
    save_artifact(&out_dir, &base_config, arch, classes, &vars, &tokenizer)?;

    let mut metrics = Map::new();
    metrics.insert("base_model".into(), json!(config.model_id));
    metrics.insert(
        "hyperparameters".into(),
        json!({
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "max_length": config.max_length,
        }),
    );
    metrics.insert("device".into(), json!(device_name));
    metrics.insert(
        "in_training_eval".into(),
        json!({
            "accuracy": eval_accuracy.map(round4),
            "loss": eval_loss.map(round4),
            "sessions": eval_batch.rows,
            "note": IN_TRAINING_NOTE,
        }),
    );
    metrics.insert("model_path".into(), json!(out_dir.display().to_string()));

    Ok(Trained {
        dir: out_dir,
        metrics,
    })
}

/// Accuracy and mean loss over the in-training slice. `None` for both when the
/// slice came out empty, which is what the Python metrics carried when the
/// Trainer reported no such key.
fn evaluate_slice(
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
fn infer(
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
enum Architecture {
    DistilBert,
    Bert,
}

impl Architecture {
    fn from_config(config: &Value) -> Result<Self> {
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
    fn roots(self) -> &'static [&'static str] {
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

    fn pooler_name(self) -> &'static str {
        match self {
            Self::DistilBert => "pre_classifier",
            Self::Bert => "pooler.dense",
        }
    }

    fn architecture_name(self) -> &'static str {
        match self {
            Self::DistilBert => "DistilBertForSequenceClassification",
            Self::Bert => "BertForSequenceClassification",
        }
    }

    fn hidden_size(self, config: &Value) -> Result<usize> {
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

    fn max_positions(self, config: &Value) -> Result<usize> {
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
    fn head_dropout(self, config: &Value) -> f32 {
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
struct DistilBertConfig {
    vocab_size: usize,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    hidden_dim: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_gelu")]
    activation: String,
}

#[derive(Deserialize)]
struct BertConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_gelu")]
    hidden_act: String,
}

fn default_gelu() -> String {
    "gelu".to_string()
}

fn default_type_vocab_size() -> usize {
    2
}

fn default_layer_norm_eps() -> f64 {
    DEFAULT_LAYER_NORM_EPS
}

/// The feed-forward activations these two architectures name. HuggingFace's
/// plain `gelu` is the exact error-function form; the tanh approximation is
/// what `gelu_new` and `gelu_pytorch_tanh` ask for.
#[derive(Clone, Copy)]
enum Activation {
    Gelu,
    GeluTanh,
    Relu,
}

impl Activation {
    fn parse(name: &str) -> Result<Self> {
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

    fn apply(self, xs: &Tensor) -> candle_core::Result<Tensor> {
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
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            bias: vb.get(size, "bias")?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        ops::layer_norm_slow(xs, &self.weight, &self.bias, self.eps as f32)
    }
}

/// Scaled dot-product attention over already-projected queries, keys and
/// values of shape (batch, tokens, hidden). `padding` is 1 at padding
/// positions, shaped (batch, 1, 1, tokens) so it broadcasts over the heads.
fn attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    heads: usize,
    padding: &Tensor,
) -> candle_core::Result<Tensor> {
    let (batch, tokens, hidden) = query.dims3()?;
    let head_dim = hidden / heads;
    let split = |xs: &Tensor| -> candle_core::Result<Tensor> {
        xs.reshape((batch, tokens, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()
    };
    let query = (split(query)? / (head_dim as f64).sqrt())?;
    let key = split(key)?;
    let value = split(value)?;

    let scores = query.matmul(&key.transpose(2, 3)?.contiguous()?)?;
    // Padding keys are scored at -inf so softmax gives them no weight.
    let blocked = Tensor::new(f32::NEG_INFINITY, scores.device())?.broadcast_as(scores.shape())?;
    let scores = padding
        .broadcast_as(scores.shape())?
        .where_cond(&blocked, &scores)?;
    let weights = ops::softmax(&scores, D::Minus1)?;

    weights
        .matmul(&value)?
        .transpose(1, 2)?
        .reshape((batch, tokens, hidden))
}

struct DistilBertLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_out: Linear,
    attention_norm: LayerNorm,
    lin1: Linear,
    lin2: Linear,
    output_norm: LayerNorm,
    heads: usize,
    activation: Activation,
}

impl DistilBertLayer {
    fn load(
        config: &DistilBertConfig,
        activation: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let dim = config.dim;
        let attention = vb.pp("attention");
        let ffn = vb.pp("ffn");
        Ok(Self {
            query: candle_nn::linear(dim, dim, attention.pp("q_lin"))?,
            key: candle_nn::linear(dim, dim, attention.pp("k_lin"))?,
            value: candle_nn::linear(dim, dim, attention.pp("v_lin"))?,
            attention_out: candle_nn::linear(dim, dim, attention.pp("out_lin"))?,
            attention_norm: LayerNorm::load(
                dim,
                DEFAULT_LAYER_NORM_EPS,
                vb.pp("sa_layer_norm"),
            )?,
            lin1: candle_nn::linear(dim, config.hidden_dim, ffn.pp("lin1"))?,
            lin2: candle_nn::linear(config.hidden_dim, dim, ffn.pp("lin2"))?,
            output_norm: LayerNorm::load(
                dim,
                DEFAULT_LAYER_NORM_EPS,
                vb.pp("output_layer_norm"),
            )?,
            heads: config.n_heads,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let context = attention(
            &self.query.forward(xs)?,
            &self.key.forward(xs)?,
            &self.value.forward(xs)?,
            self.heads,
            padding,
        )?;
        let attended = self
            .attention_norm
            .forward(&self.attention_out.forward(&context)?.add(xs)?)?;
        let ffn = self
            .lin2
            .forward(&self.activation.apply(&self.lin1.forward(&attended)?)?)?;
        self.output_norm.forward(&ffn.add(&attended)?)
    }
}

struct DistilBert {
    words: Embedding,
    positions: Embedding,
    norm: LayerNorm,
    layers: Vec<DistilBertLayer>,
}

impl DistilBert {
    fn load(config: &DistilBertConfig, vb: &VarBuilder) -> Result<Self> {
        let activation = Activation::parse(&config.activation)?;
        let embeddings = vb.pp("embeddings");
        let layers = vb.pp("transformer").pp("layer");
        let build = || -> candle_core::Result<Self> {
            Ok(Self {
                words: candle_nn::embedding(
                    config.vocab_size,
                    config.dim,
                    embeddings.pp("word_embeddings"),
                )?,
                positions: candle_nn::embedding(
                    config.max_position_embeddings,
                    config.dim,
                    embeddings.pp("position_embeddings"),
                )?,
                norm: LayerNorm::load(
                    config.dim,
                    DEFAULT_LAYER_NORM_EPS,
                    embeddings.pp("LayerNorm"),
                )?,
                layers: (0..config.n_layers)
                    .map(|index| {
                        DistilBertLayer::load(config, activation, layers.pp(index.to_string()))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?,
            })
        };
        build().map_err(|err| Error(format!("could not load the distilbert encoder: {err}")))
    }

    fn forward(&self, ids: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let (_, tokens) = ids.dims2()?;
        let positions = Tensor::arange(0u32, tokens as u32, ids.device())?;
        let mut hidden = self.norm.forward(
            &self
                .words
                .forward(ids)?
                .broadcast_add(&self.positions.forward(&positions)?)?,
        )?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, padding)?;
        }
        Ok(hidden)
    }
}

struct BertLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_out: Linear,
    attention_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_norm: LayerNorm,
    heads: usize,
    activation: Activation,
}

impl BertLayer {
    fn load(
        config: &BertConfig,
        activation: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        let attention = vb.pp("attention");
        let self_attention = attention.pp("self");
        let attention_output = attention.pp("output");
        let output = vb.pp("output");
        Ok(Self {
            query: candle_nn::linear(hidden, hidden, self_attention.pp("query"))?,
            key: candle_nn::linear(hidden, hidden, self_attention.pp("key"))?,
            value: candle_nn::linear(hidden, hidden, self_attention.pp("value"))?,
            attention_out: candle_nn::linear(hidden, hidden, attention_output.pp("dense"))?,
            attention_norm: LayerNorm::load(hidden, eps, attention_output.pp("LayerNorm"))?,
            intermediate: candle_nn::linear(
                hidden,
                config.intermediate_size,
                vb.pp("intermediate").pp("dense"),
            )?,
            output: candle_nn::linear(config.intermediate_size, hidden, output.pp("dense"))?,
            output_norm: LayerNorm::load(hidden, eps, output.pp("LayerNorm"))?,
            heads: config.num_attention_heads,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let context = attention(
            &self.query.forward(xs)?,
            &self.key.forward(xs)?,
            &self.value.forward(xs)?,
            self.heads,
            padding,
        )?;
        let attended = self
            .attention_norm
            .forward(&self.attention_out.forward(&context)?.add(xs)?)?;
        let intermediate = self
            .activation
            .apply(&self.intermediate.forward(&attended)?)?;
        self.output_norm
            .forward(&self.output.forward(&intermediate)?.add(&attended)?)
    }
}

struct Bert {
    words: Embedding,
    positions: Embedding,
    token_types: Embedding,
    norm: LayerNorm,
    layers: Vec<BertLayer>,
}

impl Bert {
    fn load(config: &BertConfig, vb: &VarBuilder) -> Result<Self> {
        let activation = Activation::parse(&config.hidden_act)?;
        let embeddings = vb.pp("embeddings");
        let layers = vb.pp("encoder").pp("layer");
        let hidden = config.hidden_size;
        let build = || -> candle_core::Result<Self> {
            Ok(Self {
                words: candle_nn::embedding(
                    config.vocab_size,
                    hidden,
                    embeddings.pp("word_embeddings"),
                )?,
                positions: candle_nn::embedding(
                    config.max_position_embeddings,
                    hidden,
                    embeddings.pp("position_embeddings"),
                )?,
                token_types: candle_nn::embedding(
                    config.type_vocab_size,
                    hidden,
                    embeddings.pp("token_type_embeddings"),
                )?,
                norm: LayerNorm::load(
                    hidden,
                    config.layer_norm_eps,
                    embeddings.pp("LayerNorm"),
                )?,
                layers: (0..config.num_hidden_layers)
                    .map(|index| BertLayer::load(config, activation, layers.pp(index.to_string())))
                    .collect::<candle_core::Result<Vec<_>>>()?,
            })
        };
        build().map_err(|err| Error(format!("could not load the bert encoder: {err}")))
    }

    fn forward(&self, ids: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let (_, tokens) = ids.dims2()?;
        let positions = Tensor::arange(0u32, tokens as u32, ids.device())?;
        // Every session is a single segment, so the token type is always 0.
        let token_types = ids.zeros_like()?;
        let embedded = self
            .words
            .forward(ids)?
            .add(&self.token_types.forward(&token_types)?)?
            .broadcast_add(&self.positions.forward(&positions)?)?;
        let mut hidden = self.norm.forward(&embedded)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, padding)?;
        }
        Ok(hidden)
    }
}

enum Encoder {
    DistilBert(DistilBert),
    Bert(Bert),
}

impl Encoder {
    fn load(arch: Architecture, config: &Value, vb: &VarBuilder) -> Result<Self> {
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

    fn forward(&self, ids: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::DistilBert(model) => model.forward(ids, padding),
            Self::Bert(model) => model.forward(ids, padding),
        }
    }
}

// ---------------------------------------------------------------------------
// classifier
// ---------------------------------------------------------------------------

struct Classifier {
    encoder: Encoder,
    pooler: Linear,
    dropout: HeadDropout,
    head: Linear,
}

impl Classifier {
    /// Assemble the encoder plus the classification head over a checkpoint.
    ///
    /// `trainable` turns every float parameter into a candle variable and
    /// creates the head weights a base checkpoint does not carry; without it
    /// the checkpoint has to be complete, which is what a saved artifact is.
    fn load(
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
            init_linear(&mut tensors, "classifier", num_labels, hidden, std, device, rng)?;

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
    fn forward(&self, ids: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
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
struct HeadDropout {
    probability: f64,
    rng: std::cell::RefCell<ChaCha8Rng>,
}

impl HeadDropout {
    fn forward(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
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
fn normalize(
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
fn rename_legacy(name: &str) -> String {
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
fn init_linear(
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
        let values: Vec<f32> = (0..out_dim * in_dim).map(|_| normal(rng, std) as f32).collect();
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
fn normal(rng: &mut ChaCha8Rng, std: f64) -> f64 {
    let uniform = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let angle = std::f64::consts::TAU * rng.random::<f64>();
    std * (-2.0 * uniform.ln()).sqrt() * angle.cos()
}

// ---------------------------------------------------------------------------
// hub, tokenizer, batching
// ---------------------------------------------------------------------------

/// The files a fine-tune starts from, resolved through the hub cache.
struct HubFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    safetensors: Option<PathBuf>,
    pytorch: Option<PathBuf>,
}

impl HubFiles {
    fn fetch(model_id: &str) -> Result<Self> {
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

    fn weights(&self, device: &Device) -> Result<Vec<(String, Tensor)>> {
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

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|err| Error(format!("could not read {}: {err}", path.display())))
}

/// Truncation and padding as the Python path asked for them: truncate at
/// `max_length`, pad the whole encoded split to its longest member.
fn prepare_tokenizer(tokenizer: &mut Tokenizer, max_length: usize) -> Result<()> {
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
struct Batches {
    ids: Vec<u32>,
    mask: Vec<u32>,
    labels: Vec<usize>,
    width: usize,
    rows: usize,
    device: Device,
}

impl Batches {
    fn encode(
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
    fn take(&self, rows: &[usize]) -> Result<(Tensor, Tensor, Tensor)> {
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
fn save_artifact(
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
fn classes_from_config(config: &Value) -> Result<Vec<String>> {
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
fn sanitize_model_id(model_id: &str) -> String {
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
fn pick_device() -> (Device, &'static str) {
    let device = Device::metal_if_available(0).unwrap_or(Device::Cpu);
    let name = if device.is_metal() { "metal" } else { "cpu" };
    (device, name)
}

/// Per-class counts ordered by class, the `_class_counts` of the Python path.
fn class_counts(values: &[String]) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for value in values {
        *counts.entry(value.clone()).or_insert(0usize) += 1;
    }
    counts
}

/// A stratified holdout of `n_test` rows with every class present on both
/// sides: what `train_test_split(..., stratify=label_ids, random_state=0)`
/// produced for the in-training slice.
fn stratified_split(
    label_ids: &[usize],
    n_classes: usize,
    n_test: usize,
    rng: &mut ChaCha8Rng,
) -> (Vec<usize>, Vec<usize>) {
    let mut per_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (row, label) in label_ids.iter().enumerate() {
        per_class[*label].push(row);
    }
    for group in per_class.iter_mut() {
        group.shuffle(rng);
    }

    let fraction = n_test as f64 / label_ids.len().max(1) as f64;
    // Every class keeps at least one row on each side; that is what the extra
    // minimum of 2 sessions per class buys.
    let mut take: Vec<usize> = per_class
        .iter()
        .map(|group| {
            ((group.len() as f64 * fraction).round() as usize)
                .clamp(1, group.len().saturating_sub(1).max(1))
        })
        .collect();

    // Per-class rounding rarely lands on exactly n_test, so single rows move
    // between the sides, largest class first, until it does or nothing can
    // move without emptying a side.
    loop {
        let total: usize = take.iter().sum();
        if total == n_test {
            break;
        }
        let grow = total < n_test;
        let candidate = (0..n_classes)
            .filter(|class| {
                if grow {
                    take[*class] + 1 < per_class[*class].len()
                } else {
                    take[*class] > 1
                }
            })
            .max_by_key(|class| per_class[*class].len());
        match candidate {
            Some(class) if grow => take[class] += 1,
            Some(class) => take[class] -= 1,
            None => break,
        }
    }

    let mut test = Vec::with_capacity(n_test);
    let mut train = Vec::with_capacity(label_ids.len().saturating_sub(n_test));
    for (class, group) in per_class.iter().enumerate() {
        let (held, kept) = group.split_at(take[class].min(group.len()));
        test.extend_from_slice(held);
        train.extend_from_slice(kept);
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

/// Clip the gradients to a global L2 norm, the `max_grad_norm=1.0` every
/// transformers `Trainer` applies by default.
fn clip_grads(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[Var],
    max_norm: f64,
) -> Result<()> {
    let mut squares = Vec::with_capacity(vars.len());
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let square = grad
                .sqr()
                .and_then(|grad| grad.sum_all())
                .map_err(|err| Error(format!("could not measure a gradient: {err}")))?;
            squares.push(square);
        }
    }
    if squares.is_empty() {
        return Err(Error(
            "no parameter received a gradient; the fine-tune would not change the model".into(),
        ));
    }
    let total = Tensor::stack(&squares, 0)
        .and_then(|squares| squares.sum_all())
        .map_err(|err| Error(format!("could not measure the gradient norm: {err}")))?;
    let norm = (scalar(&total)? as f64).sqrt();
    if !norm.is_finite() {
        return Err(Error(
            "the gradient norm is not finite; the fine-tune diverged, lower --lr and retry".into(),
        ));
    }
    if norm <= max_norm {
        return Ok(());
    }
    let scale = max_norm / (norm + 1e-6);
    let mut scaled = Vec::with_capacity(vars.len());
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let clipped =
                (grad * scale).map_err(|err| Error(format!("could not clip a gradient: {err}")))?;
            scaled.push((var.as_tensor().clone(), clipped));
        }
    }
    for (tensor, grad) in scaled {
        grads.insert(&tensor, grad);
    }
    Ok(())
}

/// The winning class per row of a logits tensor.
fn argmax(logits: &Tensor) -> Result<Vec<u32>> {
    logits
        .argmax(D::Minus1)
        .and_then(|best| best.to_vec1::<u32>())
        .map_err(|err| Error(format!("could not read the predicted classes: {err}")))
}

fn scalar(tensor: &Tensor) -> Result<f32> {
    tensor
        .to_scalar::<f32>()
        .map_err(|err| Error(format!("could not read a scalar back from the device: {err}")))
}

/// Python's `round(value, 4)`, which is what the metrics have always carried.
fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}
