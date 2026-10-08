//! `lifecycle-evaluate-gguf`: the quantized lifecycle model measured the way
//! production serves it — `llama-server` (`crate::serving`) answering Oko's
//! chat contract with decoding constrained to the checked-in decision schema,
//! narrowed per request to that request's candidate references. A request
//! that fails is a failure naming its row. The answer's length is bounded by
//! the schema's grammar, not by a token budget.

use std::path::PathBuf;

use serde_json::{json, Map};

use super::*;

/// What one evaluation run is given; every field is the caller's. How the
/// server sizes its slots, context and offload is its own (`crate::serving`).
pub struct GgufEvaluation {
    pub model: PathBuf,
    pub dataset: PathBuf,
    pub predictions: PathBuf,
    pub metrics: PathBuf,
    pub server: PathBuf,
    pub server_log: PathBuf,
    pub output_schema: PathBuf,
}

impl GgufEvaluation {
    /// How this run serves its model.
    fn serving(&self) -> crate::serving::Serving {
        crate::serving::Serving {
            server: self.server.clone(),
            model: self.model.clone(),
            server_log: self.server_log.clone(),
        }
    }
}

/// The checked-in schema with `goal_ref` narrowed from its pattern to `refs`.
fn narrowed(schema: &Value, refs: &[String]) -> Result<Value> {
    let variants = schema["oneOf"].as_array().ok_or("the output schema has no oneOf")?;
    let mut narrowed = Vec::with_capacity(variants.len());
    for variant in variants {
        let mut variant = variant.clone();
        let goal_ref = variant
            .pointer_mut("/properties/goal_ref")
            .and_then(Value::as_object_mut)
            .ok_or("an output schema variant has no goal_ref property")?;
        if goal_ref.remove("pattern").is_some() {
            goal_ref.insert("enum".to_string(), json!(refs));
        }
        narrowed.push(variant);
    }
    Ok(json!({ "oneOf": narrowed }))
}

/// The candidate references of `row` the schema's pattern variants may name:
/// every candidate except the references the schema itself fixes (a new goal).
fn existing_refs(row: &TrainingRow, schema: &Value) -> Result<Vec<String>> {
    let fixed: Vec<&str> = schema["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|variant| variant.pointer("/properties/goal_ref/const").and_then(Value::as_str))
        .collect();
    let envelope = input_envelope(row)?;
    let candidates = envelope["candidates"].as_array().ok_or_else(|| Error(format!("{} has no candidates", row.id)))?;
    Ok(candidates
        .iter()
        .filter_map(|candidate| candidate["ref"].as_str())
        .filter(|reference| !fixed.contains(reference))
        .map(str::to_string)
        .collect())
}

/// Whether `value` satisfies one narrowed variant: exactly its required
/// string properties, each matching its `const` or `enum`.
fn satisfies(value: &Map<String, Value>, variant: &Value) -> bool {
    let Some(properties) = variant["properties"].as_object() else { return false };
    if value.len() != properties.len() || !value.keys().all(|key| properties.contains_key(key)) {
        return false;
    }
    properties.iter().all(|(key, rule)| {
        let Some(actual) = value.get(key) else { return false };
        if !actual.is_string() || rule.get("pattern").is_some() {
            return false;
        }
        rule.get("const").map_or(true, |constant| constant == actual)
            && rule.get("enum").and_then(Value::as_array).map_or(true, |allowed| allowed.contains(actual))
    })
}

/// The decision in `text` when it satisfies one variant of `schema`.
fn decision(text: &str, schema: &Value) -> Option<Decision> {
    let value = parse_json_object(text).ok()?;
    let object = value.as_object()?;
    schema["oneOf"].as_array()?.iter().any(|variant| satisfies(object, variant)).then(|| serde_json::from_value(value.clone()).ok())?
}

/// The reference decision of `row`, its title blanked: titles belong to Oko's title model.
fn target(row: &TrainingRow, schema: &Value) -> Result<Decision> {
    let content = &row.messages.iter().find(|message| message.role == "assistant").ok_or_else(|| Error(format!("{} has no assistant message", row.id)))?.content;
    let mut value = parse_json_object(content)?;
    value["title"] = json!("");
    decision(&value.to_string(), schema).ok_or_else(|| Error(format!("{} has a reference decision outside the output schema", row.id)))
}

/// Ask the server for `row`'s decision; `(raw answer, decision when it satisfies the schema)`.
fn classify(server: &crate::serving::Server, schema: &Value, row: &TrainingRow) -> Result<(String, Option<Decision>)> {
    let user = &row.messages.iter().find(|message| message.role == "user").ok_or_else(|| Error(format!("{} has no user message", row.id)))?.content;
    let narrowed = narrowed(schema, &existing_refs(row, schema)?)?;
    let constraint = json!({ "type": "json_schema", "json_schema": { "name": "oko_goal_lifecycle", "strict": true, "schema": narrowed } });
    let raw = server.answer(&row.id, SYSTEM_PROMPT.trim(), user, Some(("response_format", constraint)))?;
    let parsed = decision(&raw, &narrowed);
    Ok((raw, parsed))
}

mod scoring;
pub use scoring::evaluate_gguf;
