//! The window's `train`: POST /api/train with what `train` takes —
//! `aspect`, `min_labeled_sessions`, `model` (`tfidf-logreg` or a HuggingFace
//! model id), `eval_split` ({fraction, seed} or false) and `training`, the
//! backend's settings as the text a person typed for each key. The holdout is
//! checked by the validator `run` uses, the settings by the reader behind
//! `train`'s flags, and the run is `model::train`, the implementation behind
//! `train`. Guarded like the upload (session token and the listener's exact
//! Origin); answered with the metrics the run wrote, or the refusal sentence
//! with whether labels were too few.

use super::*;

use reqwest::StatusCode as Status;

use crate::{jobs, model};

pub(crate) fn train(mut request: Request, origin: &str, token: &str) -> Result<()> {
    if !authorized(&request, token) {
        return unauthorized(request);
    }
    if header(&request, "Origin") != Some(origin) {
        return respond_json(
            request,
            Status::FORBIDDEN.as_u16(),
            &json!({"ok": false, "error": "mutation Origin does not match the GUI listener"}),
        );
    }
    let mut raw = String::new();
    request.as_reader().read_to_string(&mut raw)?;
    match run(&raw) {
        Ok(metrics) => respond_json(request, Status::OK.as_u16(), &json!({"ok": true, "metrics": metrics})),
        Err((not_enough_data, error)) => respond_json(
            request,
            Status::UNPROCESSABLE_ENTITY.as_u16(),
            &json!({"ok": false, "not_enough_data": not_enough_data, "error": error}),
        ),
    }
}

/// The trained metrics, or the refusal and whether it was too few labels.
fn run(raw: &str) -> std::result::Result<Value, (bool, String)> {
    let refused = |error: String| (false, error);
    let body: serde_yaml::Mapping = serde_json::from_str(raw)
        .map_err(|error| refused(format!("the train request is not a JSON object: {error}")))?;
    let text = |key: &str| {
        jobs::get(&body, key)
            .and_then(serde_yaml::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| refused(format!("{key} is required")))
    };
    let aspect = text("aspect")?;
    let backend = text("model").map_err(|_| {
        refused(format!("model is required: {} or a HuggingFace model id", jobs::SKLEARN_MODEL))
    })?;
    let min_sessions = jobs::get(&body, "min_labeled_sessions")
        .and_then(serde_yaml::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .and_then(std::num::NonZeroUsize::new)
        .ok_or_else(|| refused("min_labeled_sessions is required: a positive whole number".to_string()))?;
    let eval_split = jobs::eval_split(&body).map_err(|error| refused(error.to_string()))?;
    let settings = match jobs::get(&body, "training") {
        Some(serde_yaml::Value::Mapping(settings)) => settings,
        _ => return Err(refused("training is required: the backend's settings, one text per key".to_string())),
    };
    let typed = |flag: &str| {
        let key = flag.trim_start_matches('-').replace('-', "_");
        jobs::get(settings, &key).and_then(serde_yaml::Value::as_str)
    };
    let training = jobs::training_from_flags(backend, typed).map_err(|error| refused(error.to_string()))?;
    let eval_split = serde_json::to_value(&eval_split).map_err(|error| refused(error.to_string()))?;
    let model_id = (backend != jobs::SKLEARN_MODEL).then_some(backend);
    model::train(aspect, model_id, &training, &eval_split, min_sessions.get()).map_err(|failure| {
        let not_enough_data = matches!(failure, crate::util::TrainFailure::NotEnoughData(_));
        (not_enough_data, failure.to_string())
    })
}
