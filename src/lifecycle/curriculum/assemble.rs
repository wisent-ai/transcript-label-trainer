//! `lifecycle-assemble-curriculum`: the lifecycle training and evaluation
//! splits, assembled from the real reviewed rows and the generated curriculum
//! an independent review agreed with.
//!
//! Every row's one reviewed decision is checked against the decision contract
//! (`validate_decision`) and written back with its title cleared, so moving
//! title generation out of this model changes no reviewed action. A
//! curriculum row is kept only when the reviewed action is the one the
//! curriculum intended; every disagreement is counted as
//! `intended->reviewed`. The evaluation curriculum must hold at least the
//! stated number of accepted rows for every action the decision schema
//! declares, ids are unique in each split and never shared between them, and
//! each split is written whole or not at all.

use std::collections::BTreeMap;

use super::super::*;

/// The decision schema every lifecycle answer is held to; its `oneOf`
/// branches declare the actions.
const OUTPUT_SCHEMA: &str =
    include_str!("../../../training/lifecycle-model/lifecycle-output-schema.json");

/// What one run assembles from and writes to.
pub struct CurriculumAssembly<'a> {
    pub base_train: &'a Path,
    pub base_eval: &'a Path,
    pub reviewed_train: &'a Path,
    pub reviewed_eval: &'a Path,
    pub output_train: &'a Path,
    pub output_eval: &'a Path,
    /// Accepted evaluation rows each action needs, as the caller states it.
    pub minimum_eval_per_action: usize,
}

/// The actions the decision schema declares, in its order.
pub(super) fn declared_actions() -> Result<Vec<String>> {
    let schema: Value = serde_json::from_str(OUTPUT_SCHEMA)?;
    let actions: Vec<String> = schema["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|branch| {
            branch
                .pointer("/properties/action/const")
                .and_then(Value::as_str)
        })
        .map(str::to_string)
        .collect();
    if actions.is_empty() {
        return Err(Error(
            "the lifecycle output schema declares no action".to_string(),
        ));
    }
    Ok(actions)
}

/// The row with its reviewed decision checked and its title cleared, and
/// that decision's action.
fn reviewed(mut value: Value) -> Result<(Value, String)> {
    let row: TrainingRow = serde_json::from_value(value.clone())?;
    let answers: Vec<&ChatMessage> = row
        .messages
        .iter()
        .filter(|message| message.role == "assistant")
        .collect();
    let [answer] = answers.as_slice() else {
        return Err(Error(format!(
            "{} must contain one reviewed assistant decision",
            row.id
        )));
    };
    let decision = validate_decision(&row, parse_json_object(&answer.content)?)?;
    let content = serde_json::to_string(&decision)?;
    for message in value["messages"].as_array_mut().into_iter().flatten() {
        if message["role"] == "assistant" {
            message["content"] = Value::String(content.clone());
        }
    }
    Ok((value, decision.action))
}

fn read_reviewed(path: &Path) -> Result<Vec<(Value, String)>> {
    let file = File::open(path).map_err(|error| Error(format!("{}: {error}", path.display())))?;
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| Error(format!("{}: unreadable row: {error}", path.display())))?;
        rows.push(reviewed(value).map_err(|error| Error(format!("{}: {error}", path.display())))?);
    }
    Ok(rows)
}

/// The curriculum rows whose review agrees with their intended action, and
/// the disagreements counted by `intended->reviewed`.
fn accepted(rows: Vec<(Value, String)>) -> (Vec<(Value, String)>, BTreeMap<String, usize>) {
    let mut kept = Vec::new();
    let mut rejected = Vec::new();
    for (row, action) in rows {
        let intended = row
            .pointer("/metadata/intended_action")
            .and_then(Value::as_str)
            .map(str::to_string);
        match intended {
            Some(intended) if intended == action => kept.push((row, action)),
            Some(intended) => rejected.push(format!("{intended}->{action}")),
            None => rejected.push(format!("none stated->{action}")),
        }
    }
    (kept, tally(&rejected))
}

/// How many times each key occurs.
fn tally(keys: &[String]) -> BTreeMap<String, usize> {
    keys.iter()
        .map(|key| {
            (
                key.clone(),
                keys.iter().filter(|other| *other == key).count(),
            )
        })
        .collect()
}

fn action_counts(rows: &[(Value, String)]) -> BTreeMap<String, usize> {
    let actions: Vec<String> = rows.iter().map(|(_, action)| action.clone()).collect();
    tally(&actions)
}

fn write_rows(path: &Path, rows: &[(Value, String)]) -> Result<()> {
    crate::util::write_jsonl(path, rows.iter().map(|(row, _)| row))
}

fn unique_ids(rows: &[(Value, String)], split: &str) -> Result<HashSet<String>> {
    let mut seen = HashSet::new();
    for (row, _) in rows {
        let id = row
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| {
                Error(format!(
                    "assembled {split} split holds a row without a nonblank string id: {row}"
                ))
            })?
            .to_string();
        if !seen.insert(id.clone()) {
            return Err(Error(format!(
                "assembled {split} split contains the id {id} more than once"
            )));
        }
    }
    Ok(seen)
}

/// Assemble both splits, write them and answer what was kept and refused.
pub fn assemble_curriculum(run: &CurriculumAssembly) -> Result<Value> {
    let actions = declared_actions()?;
    let mut train_rows = read_reviewed(run.base_train)?;
    let mut eval_rows = read_reviewed(run.base_eval)?;
    let (train_curriculum, train_rejected) = accepted(read_reviewed(run.reviewed_train)?);
    let (eval_curriculum, eval_rejected) = accepted(read_reviewed(run.reviewed_eval)?);

    let missing: BTreeMap<&str, usize> = actions
        .iter()
        .filter_map(|action| {
            let have = eval_curriculum
                .iter()
                .filter(|(_, reviewed)| reviewed == action)
                .count();
            (have < run.minimum_eval_per_action)
                .then(|| (action.as_str(), run.minimum_eval_per_action - have))
        })
        .collect();
    if !missing.is_empty() {
        return Err(Error(format!(
            "reviewed evaluation curriculum is underrepresented: {}",
            serde_json::to_string(&missing)?
        )));
    }

    let (accepted_train, accepted_eval) = (train_curriculum.len(), eval_curriculum.len());
    train_rows.extend(train_curriculum);
    eval_rows.extend(eval_curriculum);
    let train_ids = unique_ids(&train_rows, "training")?;
    let eval_ids = unique_ids(&eval_rows, "evaluation")?;
    if let Some(shared) = train_ids.intersection(&eval_ids).next() {
        return Err(Error(format!("assembled splits overlap at {shared}")));
    }

    write_rows(run.output_train, &train_rows)?;
    write_rows(run.output_eval, &eval_rows)?;
    Ok(serde_json::json!({
        "accepted_eval_curriculum": accepted_eval,
        "accepted_train_curriculum": accepted_train,
        "eval_actions": action_counts(&eval_rows),
        "eval_rows": eval_rows.len(),
        "rejected_eval_curriculum": eval_rejected,
        "rejected_train_curriculum": train_rejected,
        "train_actions": action_counts(&train_rows),
        "train_rows": train_rows.len(),
    }))
}
