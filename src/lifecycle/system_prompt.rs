use super::*;

pub(crate) const SYSTEM_PROMPT: &str = include_str!("../../training/lifecycle-model/lifecycle-system-prompt.txt");

pub(crate) const WORKERS: usize = 16;

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
        .filter(|line| line.as_ref().map_or(true, |value| !value.trim().is_empty()))
        .map(|line| {
            let line = line?;
            serde_json::from_str(&line).map_err(Error::from)
        })
        .collect()
}

pub(crate) fn parse_json_object(answer: &str) -> Result<Value> {
    let trimmed = answer.trim();
    let candidate = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        trimmed
    } else {
        let start = trimmed
            .find('{')
            .ok_or_else(|| Error("lifecycle reviewer returned no JSON object".to_string()))?;
        let end = trimmed
            .rfind('}')
            .ok_or_else(|| Error("lifecycle reviewer returned incomplete JSON".to_string()))?;
        &trimmed[start..=end]
    };
    serde_json::from_str(candidate).map_err(Error::from)
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
    if !["startGoal", "continueCurrent", "finishGoal", "ignore"].contains(&decision.action.as_str())
    {
        return Err(Error(format!(
            "{} has unknown action {}",
            row.id, decision.action
        )));
    }
    if !refs.contains(&decision.goal_ref.as_str()) {
        return Err(Error(format!(
            "{} has unknown goal_ref {}",
            row.id, decision.goal_ref
        )));
    }
    if !["none", "explicit_open", "explicit_completion"]
        .contains(&decision.lifecycle_evidence.as_str())
    {
        return Err(Error(format!(
            "{} has unknown lifecycle_evidence {}",
            row.id, decision.lifecycle_evidence
        )));
    }
    decision.title.clear();
    if decision.action == "startGoal" {
        if decision.goal_ref != "NEW_GOAL" {
            return Err(Error(format!("{} has invalid startGoal ref", row.id)));
        }
    } else if decision.goal_ref == "NEW_GOAL" {
        return Err(Error(format!(
            "{} uses NEW_GOAL for {}",
            row.id, decision.action
        )));
    }
    if decision.action == "finishGoal" && decision.lifecycle_evidence != "explicit_completion" {
        return Err(Error(format!(
            "{} finishGoal lacks explicit completion evidence",
            row.id
        )));
    }
    if decision.lifecycle_evidence == "explicit_completion" && decision.action != "finishGoal" {
        return Err(Error(format!(
            "{} completion evidence has non-finish action",
            row.id
        )));
    }
    Ok(decision)
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
    let mut last = Error("lifecycle reviewer did not run".to_string());
    for attempt in 0..3 {
        match client
            .chat(model, &request)
            .and_then(|answer| parse_json_object(&answer))
            .and_then(|value| validate_decision(row, value))
        {
            Ok(decision) => return Ok(decision),
            Err(error) => last = error,
        }
        thread::sleep(Duration::from_secs(1 << attempt));
    }
    Err(Error(format!("{}: {last}", row.id)))
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
        Value::String("oko-goal-lifecycle-v1".to_string()),
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
        .filter(|line| line.as_ref().map_or(true, |value| !value.trim().is_empty()))
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
