use super::*;

pub(crate) const SYSTEM_PROMPT: &str =
    include_str!("../../training/lifecycle-model/lifecycle-system-prompt.txt");

/// The decision contract: one variant per action, each naming the goal
/// references and the evidence that action allows.
const OUTPUT_SCHEMA: &str =
    include_str!("../../training/lifecycle-model/lifecycle-output-schema.json");

/// The input and decision contract Oko asks the lifecycle model under: the
/// envelope Oko sends, without a candidate score, and the decision schema.
pub(crate) const LIFECYCLE_CONTRACT: &str = "oko-goal-lifecycle-v2";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TrainingRow {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) split_day: String,
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(crate) metadata: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Decision {
    pub(crate) action: String,
    pub(crate) goal_ref: String,
    pub(crate) title: String,
    pub(crate) lifecycle_evidence: String,
}

pub(crate) fn read_rows(path: &Path) -> Result<Vec<TrainingRow>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .filter(|line| !line.as_ref().is_ok_and(|value| value.trim().is_empty()))
        .map(|line| {
            let line = line?;
            serde_json::from_str(&line).map_err(Error::from)
        })
        .collect()
}

/// The JSON object `answer` is, read whole: a stored decision is the object
/// this crate wrote, and a served answer is decoded under the output schema,
/// so neither is searched for an object inside other text.
pub(crate) fn parse_json_object(answer: &str) -> Result<Value> {
    serde_json::from_str(answer.trim()).map_err(|error| {
        Error(format!(
            "lifecycle answer is not one JSON object ({error}): {answer}"
        ))
    })
}

pub(crate) fn input_envelope(row: &TrainingRow) -> Result<Value> {
    let content = row
        .messages
        .iter()
        .find(|message| message.role == "user")
        .ok_or_else(|| Error(format!("{} has no user message", row.id)))?
        .content
        .trim();
    serde_json::from_str(content)
        .map_err(|error| Error(format!("{} has invalid user envelope: {error}", row.id)))
}

/// The values one schema property allows: its `const`, else its `enum`; none
/// when it constrains the value another way (a pattern).
fn allowed(property: &Value) -> Vec<&str> {
    match property.get("const") {
        Some(value) => value.as_str().into_iter().collect(),
        None => property["enum"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect(),
    }
}

/// The reviewer's decision, held to the declared output schema: the variant
/// its action names fixes which goal references and evidence it may carry,
/// and its goal reference must be one of the row's candidates. The title is
/// cleared, since the contract keeps it empty.
pub(crate) fn validate_decision(row: &TrainingRow, value: Value) -> Result<Decision> {
    let mut decision: Decision = serde_json::from_value(value)
        .map_err(|error| Error(format!("{} has invalid reviewer decision: {error}", row.id)))?;
    let envelope = input_envelope(row)?;
    let refs = envelope
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|candidate| candidate.get("ref").and_then(Value::as_str))
        .collect::<Vec<_>>();
    let schema: Value = serde_json::from_str(OUTPUT_SCHEMA)?;
    let variants: Vec<&Value> = schema["oneOf"].as_array().into_iter().flatten().collect();
    // The references a variant fixes by value start a goal; every other
    // variant names an existing candidate.
    let new_refs: Vec<&str> = variants
        .iter()
        .flat_map(|variant| allowed(&variant["properties"]["goal_ref"]))
        .collect();
    let properties = &variants
        .iter()
        .find(|variant| {
            allowed(&variant["properties"]["action"]).contains(&decision.action.as_str())
        })
        .ok_or_else(|| Error(format!("{} has unknown action {}", row.id, decision.action)))?
        ["properties"];
    if !refs.contains(&decision.goal_ref.as_str()) {
        return Err(Error(format!(
            "{} has unknown goal_ref {}",
            row.id, decision.goal_ref
        )));
    }
    let fixed = allowed(&properties["goal_ref"]);
    let reference_fits = if fixed.is_empty() {
        !new_refs.contains(&decision.goal_ref.as_str())
    } else {
        fixed.contains(&decision.goal_ref.as_str())
    };
    if !reference_fits {
        return Err(Error(format!(
            "{} uses goal_ref {} for {}, which the output schema does not allow",
            row.id, decision.goal_ref, decision.action
        )));
    }
    if !allowed(&properties["lifecycle_evidence"]).contains(&decision.lifecycle_evidence.as_str()) {
        return Err(Error(format!(
            "{} gives lifecycle_evidence {} for {}, which the output schema does not allow",
            row.id, decision.lifecycle_evidence, decision.action
        )));
    }
    decision.title.clear();
    Ok(decision)
}

/// The function the reviewer answers by: its one parameter is the decision,
/// held to the declared output schema, so the decision is the reviewer's
/// answer itself. The schema's root is a `oneOf`, which a function's
/// parameters cannot be, so it is the schema of that one property.
const DECISION_FUNCTION: &str = "lifecycle_decision";

fn decision_parameters() -> Result<Value> {
    let mut schema: Value = serde_json::from_str(OUTPUT_SCHEMA)?;
    if let Some(root) = schema.as_object_mut() {
        // A dialect is declared once, by a document's root.
        root.remove("$schema");
    }
    Ok(serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["decision"],
        "properties": { "decision": schema },
    }))
}

pub(crate) fn classify(client: &BramaClient, model: &str, row: &TrainingRow) -> Result<Decision> {
    let envelope = input_envelope(row)?;
    let request = [
        Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        },
        Message {
            role: "user".to_string(),
            content: serde_json::to_string(&envelope)?,
        },
    ];
    let description = "The lifecycle decision for the envelope, under the output schema.";
    client
        .call(
            model,
            &request,
            DECISION_FUNCTION,
            description,
            decision_parameters()?,
        )
        .and_then(|mut arguments| {
            arguments.remove("decision").ok_or_else(|| {
                Error(format!(
                    "the call of {DECISION_FUNCTION} carries no decision"
                ))
            })
        })
        .and_then(|value| validate_decision(row, value))
        .map_err(|error| Error(format!("{}: {error}", row.id)))
}

pub(crate) fn reviewed_row(
    mut row: TrainingRow,
    decision: Decision,
    model: &str,
    split: &str,
) -> Result<Value> {
    row.messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.trim().to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: serde_json::to_string(&input_envelope(&row)?)?,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: serde_json::to_string(&decision)?,
        },
    ];
    let mut value = serde_json::to_value(row)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| Error("serialized lifecycle row is not an object".to_string()))?;
    object.insert("split".to_string(), Value::String(split.to_string()));
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(Default::default()));
    let metadata = metadata
        .as_object_mut()
        .ok_or_else(|| Error("lifecycle metadata is not an object".to_string()))?;
    metadata.insert("reviewedBy".to_string(), Value::String(model.to_string()));
    metadata.insert("reviewedAt".to_string(), Value::String(now_iso()));
    metadata.insert(
        "contract".to_string(),
        Value::String(LIFECYCLE_CONTRACT.to_string()),
    );
    Ok(value)
}

pub(crate) fn reviewed_ids(path: &Path) -> Result<HashSet<String>> {
    if !path.is_file() {
        return Ok(HashSet::new());
    }
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .filter(|line| !line.as_ref().is_ok_and(|value| value.trim().is_empty()))
        .map(|line| {
            let line = line?;
            let value: Value = serde_json::from_str(&line)?;
            value
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| Error("reviewed lifecycle row has no id".to_string()))
        })
        .collect()
}
