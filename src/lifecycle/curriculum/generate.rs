//! `lifecycle-generate-curriculum`: deterministic hard-case lifecycle turns
//! for Brama review, built from real reviewed envelopes.
//!
//! Every family in `templates.json` gets the stated number of rows. Each row
//! copies a real envelope that has an active candidate (the same-session
//! candidate with the highest score, else any), writes one of the family's
//! sentences into it with that candidate's title (and, for a switch, the
//! title of a candidate the envelope does not hold), and records the family
//! and its intended action, which the assembler later holds the review to.
//! The turn index of a generated row follows every source turn, so no
//! generated turn sits among real ones. The envelope carries only what Oko
//! sends: any `lifecycle_features` an older source row still holds is
//! dropped, since Oko no longer computes them and the model must not learn
//! from a field production never sends. The same seed always gives the same
//! rows.

use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::json;

use super::super::*;

const TEMPLATES: &str = include_str!("../../../training/lifecycle-model/curriculum/templates.json");
/// The candidate reference that starts a goal rather than naming one.
const NEW_GOAL: &str = "NEW_GOAL";

#[derive(Deserialize)]
struct Family {
    family: String,
    intended_action: String,
    templates: Vec<String>,
}

#[derive(Deserialize)]
struct Declared {
    #[allow(dead_code)]
    why: String,
    families: Vec<Family>,
}

/// What one run generates from and writes to.
pub struct CurriculumGeneration<'a> {
    pub source: &'a Path,
    pub output: &'a Path,
    /// Rows written for each family, as the caller states it.
    pub per_family: usize,
    /// The seed every choice is drawn from, as the caller states it.
    pub seed: u64,
}

fn user_envelope(row: &Value) -> Result<Value> {
    let content = row["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|message| message["role"] == "user")
        .and_then(|message| message["content"].as_str())
        .ok_or_else(|| Error(format!("{} has no user message", row["id"])))?;
    Ok(serde_json::from_str(content)?)
}

fn score(candidate: &Value) -> f64 {
    candidate["score"].as_f64().unwrap_or_default()
}

/// The candidate the turn is about: the best-scored same-session candidate,
/// else the best-scored of all, never the new-goal placeholder.
fn active_candidate(envelope: &Value) -> Option<&Value> {
    let candidates: Vec<&Value> = envelope["candidates"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|candidate| candidate["ref"] != NEW_GOAL)
        .collect();
    let same: Vec<&Value> = candidates
        .iter()
        .copied()
        .filter(|candidate| candidate["same_session"] == true)
        .collect();
    let pool = if same.is_empty() { candidates } else { same };
    pool.into_iter()
        .max_by(|left, right| score(left).total_cmp(&score(right)))
}

fn title_of(candidate: &Value) -> String {
    candidate["title"].as_str().unwrap_or_default().to_string()
}

/// The title of an active candidate from another envelope that this one
/// does not hold, drawn in the seeded order; none when every row shares it.
fn unrelated_title(
    envelopes: &[Value],
    held: &HashSet<String>,
    rng: &mut ChaCha20Rng,
) -> Option<String> {
    let mut order: Vec<usize> = (0..envelopes.len()).collect();
    order.shuffle(rng);
    order.into_iter().find_map(|index| {
        let title = active_candidate(&envelopes[index]).map(title_of)?;
        (!title.is_empty() && !held.contains(&title.to_lowercase())).then_some(title)
    })
}

fn lower_first(title: &str) -> String {
    let mut chars = title.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Generate every family's rows, write them whole and answer the counts.
pub fn generate_curriculum(run: &CurriculumGeneration) -> Result<Value> {
    let declared: Declared = serde_json::from_str(TEMPLATES)?;
    let file = File::open(run.source)
        .map_err(|error| Error(format!("{}: {error}", run.source.display())))?;
    let mut sources: Vec<(Value, Value)> = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let envelope = user_envelope(&row)?;
        if active_candidate(&envelope).is_some() {
            sources.push((row, envelope));
        }
    }
    if sources.is_empty() {
        return Err(Error(
            "source dataset has no lifecycle candidates".to_string(),
        ));
    }
    let last_turn = sources
        .iter()
        .filter_map(|(_, envelope)| envelope["turn_index"].as_i64())
        .max()
        .unwrap_or_default();
    let envelopes: Vec<Value> = sources
        .iter()
        .map(|(_, envelope)| envelope.clone())
        .collect();
    let mut rng = ChaCha20Rng::seed_from_u64(run.seed);
    let mut order: Vec<usize> = (0..sources.len()).collect();
    order.shuffle(&mut rng);

    let mut generated: Vec<Value> = Vec::new();
    let mut families = serde_json::Map::new();
    for (cursor, (family, index)) in declared
        .families
        .iter()
        .flat_map(|family| (0..run.per_family).map(move |index| (family, index)))
        .enumerate()
    {
        let (source, envelope) = &sources[order[cursor % order.len()]];
        let mut envelope = envelope.clone();
        let title = active_candidate(&envelope)
            .map(title_of)
            .unwrap_or_default();
        let held: HashSet<String> = envelope["candidates"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|candidate| title_of(candidate).to_lowercase())
            .collect();
        let template = &family.templates[index % family.templates.len()];
        let filled = if template.contains("{new_title") {
            let new_title = unrelated_title(&envelopes, &held, &mut rng).ok_or_else(|| {
                Error(format!(
                    "no source row holds a title unrelated to {title:?} for {}",
                    family.family
                ))
            })?;
            template
                .replace("{new_title_lower}", &lower_first(&new_title))
                .replace("{new_title}", &new_title)
        } else {
            template.clone()
        };
        let text = filled.replace("{title}", &title);
        let row_id = format!(
            "lifecycle-curriculum-{}-{}-{:04}",
            run.seed,
            family.family,
            index + 1
        );
        let object = envelope.as_object_mut().ok_or_else(|| {
            Error(format!(
                "{} has a user envelope that is not an object",
                source["id"]
            ))
        })?;
        object.remove("lifecycle_features");
        object.insert("prompt_id".to_string(), json!(row_id));
        object.insert(
            "local_day".to_string(),
            json!(format!("curriculum-{}", run.seed)),
        );
        object.insert("text".to_string(), json!(text));
        object.insert(
            "turn_index".to_string(),
            json!(last_turn + 1 + cursor as i64),
        );
        generated.push(json!({
            "id": row_id,
            "split_day": format!("curriculum-{}", run.seed),
            "messages": [{"role": "user", "content": serde_json::to_string(&envelope)?}],
            "metadata": {
                "synthetic": true,
                "curriculum_family": family.family,
                "intended_action": family.intended_action,
                "source_id": source["id"],
                "seed": run.seed,
            },
        }));
        families.insert(family.family.clone(), json!(index + 1));
    }

    if let Some(parent) = run
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = run.output.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = std::path::PathBuf::from(temporary);
    let mut out = BufWriter::new(File::create(&temporary)?);
    for row in &generated {
        writeln!(out, "{}", serde_json::to_string(row)?)?;
    }
    out.into_inner()
        .map_err(|error| Error(error.to_string()))?
        .sync_all()?;
    fs::rename(&temporary, run.output)?;
    Ok(json!({
        "families": families,
        "output": run.output.display().to_string(),
        "rows": generated.len(),
    }))
}
