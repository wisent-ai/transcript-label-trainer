//! `lifecycle-split-by-day`: a fresh day-held-out lifecycle split from the
//! reviewed rows of an earlier split.
//!
//! The rows of both earlier files are merged by id (one id holding two
//! different rows is refused), every row of the named evaluation day goes to
//! evaluation and every other row to training, ordered by id, and each split
//! must hold every action the decision schema declares. Rows that put a goal
//! title into the reviewed decision are not made here: the decision contract
//! keeps the title empty, because a separate title model names new goals.

use std::collections::BTreeMap;

use serde_json::json;

use super::super::*;
use super::assemble::declared_actions;

/// What one run splits and writes.
pub struct DaySplit<'a> {
    pub train: &'a Path,
    pub eval: &'a Path,
    /// The `split_day` whose rows are held out for evaluation.
    pub eval_day: &'a str,
    pub output_train: &'a Path,
    pub output_eval: &'a Path,
}

/// One split being built: its rows and how many carry each action.
#[derive(Default)]
struct Split {
    rows: Vec<Value>,
    actions: BTreeMap<String, usize>,
}

impl Split {
    fn push(&mut self, mut row: Value, name: &str) -> Result<()> {
        *self.actions.entry(action_of(&row)?).or_default() += 1;
        row["split"] = json!(name);
        self.rows.push(row);
        Ok(())
    }

    /// Refused when an action the schema declares has no row here.
    fn covers(&self, actions: &[String], name: &str) -> Result<()> {
        let missing: Vec<&str> = actions
            .iter()
            .map(String::as_str)
            .filter(|action| !self.actions.contains_key(*action))
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error(format!(
                "{name} split is missing actions: {}",
                missing.join(", ")
            )))
        }
    }
}

fn action_of(row: &Value) -> Result<String> {
    let content = row["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|message| message["role"] == "assistant")
        .and_then(|message| message["content"].as_str())
        .ok_or_else(|| Error(format!("{} has no reviewed decision", row["id"])))?;
    let decision: Value = serde_json::from_str(content)?;
    decision["action"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error(format!("{} has a decision without an action", row["id"])))
}

fn write_split(path: &Path, rows: &[Value]) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = std::path::PathBuf::from(temporary);
    let mut out = BufWriter::new(File::create(&temporary)?);
    for row in rows {
        writeln!(out, "{}", serde_json::to_string(row)?)?;
    }
    out.into_inner()
        .map_err(|error| Error(error.to_string()))?
        .sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn merged(sources: &[&Path]) -> Result<BTreeMap<String, Value>> {
    let mut by_id: BTreeMap<String, Value> = BTreeMap::new();
    for source in sources {
        let file =
            File::open(source).map_err(|error| Error(format!("{}: {error}", source.display())))?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(&line)?;
            let id = row["id"].as_str().unwrap_or_default().to_string();
            match by_id.get(&id) {
                Some(previous) if previous != &row => {
                    return Err(Error(format!("conflicting reviewed rows for {id}")));
                }
                Some(_) => {}
                None => {
                    by_id.insert(id, row);
                }
            }
        }
    }
    Ok(by_id)
}

/// Split the merged rows by day, write both splits and answer the counts.
pub fn split_by_day(run: &DaySplit) -> Result<Value> {
    let (mut train, mut eval) = (Split::default(), Split::default());
    for (_, row) in merged(&[run.train, run.eval])? {
        if row["split_day"] == run.eval_day {
            eval.push(row, "eval")?;
        } else {
            train.push(row, "train")?;
        }
    }
    let actions = declared_actions()?;
    train.covers(&actions, "training")?;
    eval.covers(&actions, "evaluation")?;
    write_split(run.output_train, &train.rows)?;
    write_split(run.output_eval, &eval.rows)?;
    Ok(json!({
        "eval_actions": eval.actions,
        "eval_day": run.eval_day,
        "eval_rows": eval.rows.len(),
        "train_actions": train.actions,
        "train_rows": train.rows.len(),
    }))
}
