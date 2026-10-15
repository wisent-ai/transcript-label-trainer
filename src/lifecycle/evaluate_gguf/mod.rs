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
            // Oko asks the lifecycle model greedily.
            decoding: crate::serving::Decoding::Greedy,
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
    let variants = schema["oneOf"].as_array().ok_or("the output schema has no oneOf")?;
    let fixed: Vec<&str> = variants
        .iter()
        .filter_map(|variant| variant.pointer("/properties/goal_ref/const").and_then(Value::as_str))
        .collect();
    let envelope = input_envelope(row)?;
    let candidates = envelope["candidates"].as_array().ok_or_else(|| Error(format!("{} has no candidates", row.id)))?;
    let mut references = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let reference = candidate["ref"]
            .as_str()
            .ok_or_else(|| Error(format!("{} has a candidate without a text ref: {candidate}", row.id)))?;
        if !fixed.contains(&reference) {
            references.push(reference.to_string());
        }
    }
    Ok(references)
}

/// Whether `value` satisfies one narrowed variant, or why not: exactly its
/// properties, each a string matching its `const` or `enum` when the variant
/// states one.
fn satisfies(value: &Map<String, Value>, variant: &Value) -> std::result::Result<(), String> {
    let properties = variant["properties"].as_object().ok_or("a schema variant states no properties")?;
    if value.len() != properties.len() || !value.keys().all(|key| properties.contains_key(key)) {
        let expected: Vec<&String> = properties.keys().collect();
        return Err(format!("its keys are not the variant's {expected:?}"));
    }
    for (key, rule) in properties {
        let actual = value.get(key).ok_or_else(|| format!("{key} is missing"))?;
        if !actual.is_string() {
            return Err(format!("{key} is not a string"));
        }
        if rule.get("pattern").is_some() {
            return Err(format!("{key} is still a pattern in the narrowed schema"));
        }
        if let Some(constant) = rule.get("const") {
            if constant != actual {
                return Err(format!("{key} is {actual}, not {constant}"));
            }
        }
        if let Some(allowed) = rule.get("enum") {
            let allowed = allowed.as_array().ok_or_else(|| format!("{key}'s enum is not a list"))?;
            if !allowed.contains(actual) {
                return Err(format!("{key} is {actual}, none of {allowed:?}"));
            }
        }
    }
    Ok(())
}

/// The decision in `text`, or why it is not one: not a JSON object, in no
/// variant of `schema`, or not a decision.
fn decision(text: &str, schema: &Value) -> std::result::Result<Decision, String> {
    let value = parse_json_object(text).map_err(|error| error.to_string())?;
    let object = value.as_object().ok_or("the answer is not a JSON object")?;
    let variants = schema["oneOf"].as_array().ok_or("the output schema has no oneOf")?;
    let mut refusals = Vec::with_capacity(variants.len());
    for variant in variants {
        match satisfies(object, variant) {
            Ok(()) => {
                return serde_json::from_value(value.clone())
                    .map_err(|error| format!("it satisfies the schema but is not a decision: {error}"));
            }
            Err(why) => refusals.push(why),
        }
    }
    Err(format!("it satisfies no variant: {}", refusals.join("; ")))
}

/// The reference decision of `row`, its title blanked: titles belong to Oko's title model.
fn target(row: &TrainingRow, schema: &Value) -> Result<Decision> {
    let content = &row.messages.iter().find(|message| message.role == "assistant").ok_or_else(|| Error(format!("{} has no assistant message", row.id)))?.content;
    let mut value = parse_json_object(content)?;
    value["title"] = json!("");
    decision(&value.to_string(), schema)
        .map_err(|why| Error(format!("{} has a reference decision outside the output schema: {why}", row.id)))
}

/// Ask the server for `row`'s decision; `(raw answer, decision when it
/// satisfies the schema)`. An answer outside the schema is scored as invalid
/// and its reason written on stderr beside the row.
fn classify(server: &crate::serving::Server, schema: &Value, row: &TrainingRow) -> Result<(String, Option<Decision>)> {
    let user = &row.messages.iter().find(|message| message.role == "user").ok_or_else(|| Error(format!("{} has no user message", row.id)))?.content;
    let narrowed = narrowed(schema, &existing_refs(row, schema)?)?;
    let constraint = json!({ "type": "json_schema", "json_schema": { "name": "oko_goal_lifecycle", "strict": true, "schema": narrowed } });
    let raw = server.answer(&row.id, SYSTEM_PROMPT.trim(), user, Some(("response_format", constraint)))?;
    let parsed = match decision(&raw, &narrowed) {
        Ok(decision) => Some(decision),
        Err(why) => {
            eprintln!("transcript-label-trainer: {} answered outside the output schema: {why}", row.id);
            None
        }
    };
    Ok((raw, parsed))
}

mod scoring;
pub use scoring::evaluate_gguf;
