use super::*;

/// Training decays the learning rate linearly to zero with no warmup; the
/// seed, weight decay, gradient clip and in-training slice are the caller's.

/// Both architectures normalise at eps 1e-12; bert lets its config say so.
// DistilBERT hard-codes nn.LayerNorm(..., eps=1e-12) and BertConfig defaults layer_norm_eps=1e-12:
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/configuration_bert.py
pub(crate) const DEFAULT_LAYER_NORM_EPS: f64 = 1e-12;

/// The `in_training_eval` note, verbatim from the Python metrics.
pub(crate) const IN_TRAINING_NOTE: &str =
    "stratified slice of the training side, resplit every run";

/// The hyperparameters `train` was invoked with, plus the aspect, which only
/// appears in the too-few-sessions-per-class message.
pub struct TrainConfig<'a> {
    pub aspect: &'a str,
    pub model_id: &'a str,
    pub epochs: f64,
    pub batch_size: usize,
    pub lr: f64,
    pub max_length: usize,
    pub seed: u64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub in_training_eval_share: f64,
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
    // The stratified in-training split keeps every class on both of its
    // sides, so a class only one session holds cannot be split.
    let counts = class_counts(values);
    let single: Vec<&String> = counts
        .iter()
        .filter(|(_, n)| **n == 1)
        .map(|(value, _)| value)
        .collect();
    if !single.is_empty() {
        let detail = single
            .iter()
            .map(|value| format!("'{value}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(TrainFailure::NotEnoughData(format!(
            "aspect '{}': HF fine-tuning needs more than one session of every class on the \
             training side, one for each side of its in-training split; only one session \
             holds {detail}. Add labels with 'transcript-lake label add' and retry.",
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
    batch_size: usize,
    seed: u64,
) -> Result<Vec<(String, f64)>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let config: Value = serde_json::from_slice(&std::fs::read(artifact_dir.join("config.json"))?)?;
    let classes = classes_from_config(&config)?;
    let (device, _) = pick_device()?;

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
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (model, _) = Classifier::load(arch, &config, raw, classes.len(), &device, false, &mut rng)?;
    let mut tokenizer = load_tokenizer(&artifact_dir.join("tokenizer.json"))?;
    prepare_tokenizer(&mut tokenizer, arch.max_positions(&config)?.min(max_length))?;

    infer(&model, &tokenizer, texts, &classes, &device, batch_size)
}

// ---------------------------------------------------------------------------
// training
// ---------------------------------------------------------------------------

pub(crate) fn fine_tune(
    aspect_dir: &Path,
    texts: &[String],
    label_ids: &[usize],
    classes: &[String],
    config: &TrainConfig<'_>,
) -> Result<Trained> {
    let (device, device_name) = pick_device()?;

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
    let n_test = std::cmp::max(
        classes.len(),
        (texts.len() as f64 * config.in_training_eval_share).round() as usize,
    );
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let (train_index, eval_index) = stratified_split(label_ids, classes.len(), n_test, &mut rng);

    let train_batch = Batches::encode(&tokenizer, texts, Some(label_ids), &train_index, &device)?;
    let eval_batch = Batches::encode(&tokenizer, texts, Some(label_ids), &eval_index, &device)?;

    let raw = files.weights(&device)?;
    let (model, vars) = Classifier::load(
        arch,
        &base_config,
        raw,
        classes.len(),
        &device,
        true,
        &mut rng,
    )?;
    let stepped: Vec<Var> = vars.iter().map(|(_, var)| var.clone()).collect();

    let batch_size = NonZeroUsize::new(config.batch_size)
        .ok_or_else(|| Error("the batch size must hold at least one session".to_string()))?;
    if train_index.is_empty() {
        return Err(Error(
            "the training side holds no session to fine-tune on".to_string(),
        ));
    }
    if !(config.epochs.is_normal() && config.epochs.is_sign_positive()) {
        return Err(Error(format!(
            "epochs must be a positive number, not {}",
            config.epochs
        )));
    }
    let steps_per_epoch = train_batch.rows.div_ceil(batch_size.get());
    let total_steps = (steps_per_epoch as f64 * config.epochs).ceil() as usize;
    let epochs = config.epochs.ceil() as usize;
    let mut optimizer = AdamW::new(
        stepped.clone(),
        ParamsAdamW {
            lr: config.lr,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )
    .map_err(|err| Error(format!("could not build the AdamW optimizer: {err}")))?;

    // One position per encoded training session, shuffled every epoch.
    let mut order: Vec<usize> = (0..train_batch.rows).collect();
    // Every step's loss: its count is the step reached, and an epoch's mean
    // is taken over the steps it added.
    let mut losses: Vec<f64> = Vec::with_capacity(total_steps);
    let mut epoch_means: Vec<f64> = Vec::with_capacity(epochs);
    while losses.len() < total_steps {
        order.shuffle(&mut rng);
        let epoch_start = losses.len();
        for chunk in order.chunks(batch_size.get()) {
            let step = losses.len();
            // Linear decay to zero over the whole run, no warmup.
            optimizer
                .set_learning_rate(config.lr * (total_steps - step) as f64 / total_steps as f64);
            let (ids, mask, targets) = train_batch.take(chunk)?;
            let logits = model.forward(&ids, &mask, true)?;
            let batch_loss = loss::cross_entropy(&logits, &targets)
                .map_err(|err| Error(format!("loss failed at step {step}: {err}")))?;
            let mut grads = batch_loss
                .backward()
                .map_err(|err| Error(format!("backward pass failed at step {step}: {err}")))?;
            clip_grads(&mut grads, &stepped, config.max_grad_norm)?;
            optimizer
                .step(&grads)
                .map_err(|err| Error(format!("optimizer step {step} failed: {err}")))?;
            losses.push(f64::from(scalar(&batch_loss)?));
            if losses.len() >= total_steps {
                break;
            }
        }
        let epoch_losses = &losses[epoch_start..];
        let mean = epoch_losses.iter().sum::<f64>() / epoch_losses.len() as f64;
        epoch_means.push(mean);
        eprintln!(
            "epoch {}/{epochs}: train_loss={mean:.4} ({} step(s) of {total_steps}, \
             device={device_name})",
            epoch_means.len(),
            epoch_losses.len(),
        );
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
            "seed": config.seed,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "in_training_eval_share": config.in_training_eval_share,
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
