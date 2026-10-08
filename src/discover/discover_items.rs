use super::*;

/// Propose aspect dimensions from recent sessions via the Brama teacher.
pub fn discover(
    limit: Option<i64>,
    brama_model: Option<&str>,
    best_review: bool,
    max_aspects: Option<i64>,
    runtime: Option<&str>,
) -> Result<Value> {
    let model_id = brama_model
        .filter(|value| !value.is_empty())
        .unwrap_or(brama::DEFAULT_MODEL)
        .to_string();
    let client = brama::BramaClient::from_env()?;

    let mut sessions = lake::all_sessions()?;
    if let Some(runtime) = runtime {
        sessions.retain(|session| session.runtime.as_deref() == Some(runtime));
    }
    let mut targets: Vec<String> = sessions
        .into_iter()
        .map(|session| session.session_id)
        .collect();
    // Every session unless the caller names how many of the newest to read.
    if let Some(session_limit) = limit.map(|value| value.max(0) as usize) {
        if targets.len() > session_limit {
            // The lake lists oldest first; the newest sessions are the ones the
            // operator's current habits are visible in.
            targets = targets.split_off(targets.len() - session_limit);
        }
    }
    let texts = lake::session_texts(&targets)?;

    let mut merged: BTreeMap<String, Merged> = BTreeMap::new();
    let mut failures: Vec<Value> = Vec::new();
    let mut chunks = 0usize;
    let mut sampled = 0usize;
    let excerpts: Vec<(String, String)> = targets
        .iter()
        .filter_map(|session_id| {
            let entry = texts.get(session_id)?;
            let text = entry.text.trim();
            if text.is_empty() {
                return None;
            }
            Some((session_id.clone(), text.to_string()))
        })
        .collect();
    let mut answers = Vec::new();
    if !excerpts.is_empty() {
        teacher_answers(&client, &model_id, &excerpts, &mut answers);
    }
    for (sessions, answer) in answers {
        chunks += 1;
        sampled += sessions;
        let answer = match answer {
            Ok(answer) => answer,
            Err(error) => {
                failures.push(json!({
                    "chunk": chunks,
                    "error": error.0,
                }));
                continue;
            }
        };
        let Some(proposals) = parse_proposals(&answer) else {
            failures.push(json!({
                "chunk": chunks,
                "error": format!("unparseable proposals: {answer}"),
            }));
            continue;
        };
        for proposal in proposals {
            let Some(aspect) = proposal.get("aspect").and_then(Value::as_str) else {
                continue;
            };
            let aspect = kebab(aspect);
            if aspect.is_empty() {
                continue;
            }
            let values: Vec<String> = proposal
                .get("values")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(kebab)
                        .filter(|value| !value.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            if values.len() < 2 {
                continue;
            }
            let description = proposal
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let evidence: Vec<String> = proposal
                .get("evidence")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let entry = merged.entry(aspect).or_insert_with(|| Merged {
                values: Vec::new(),
                description: description.clone(),
                support: 0,
                evidence: Vec::new(),
            });
            entry.support += 1;
            for value in values {
                if !entry.values.contains(&value) {
                    entry.values.push(value);
                }
            }
            for session_id in evidence {
                if !entry.evidence.contains(&session_id) {
                    entry.evidence.push(session_id);
                }
            }
            if entry.description.is_empty() {
                entry.description = description;
            }
        }
    }

    let mut ranked: Vec<(String, Merged)> = merged.into_iter().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .support
            .cmp(&left.1.support)
            .then_with(|| left.0.cmp(&right.0))
    });
    // Every merged proposal unless the caller names how many to keep.
    if let Some(keep) = max_aspects.map(|value| value.max(0) as usize) {
        ranked.truncate(keep);
    }

    let mut proposals: Vec<Value> = Vec::new();
    let mut rejected: Vec<Value> = Vec::new();
    let mut review_failed = 0usize;
    for (aspect, entry) in ranked {
        let mut review = None;
        if best_review {
            let prompt = review_prompt(&aspect, &entry.values, &entry.description);
            match client.chat(brama::BEST_MODEL, &prompt) {
                Ok(answer) => match brama::parse_answer(&answer, &REVIEW_VALUES[..]) {
                    Some((verdict, _exact)) if verdict == REVIEW_VALUES[1] => {
                        rejected.push(json!({
                            "aspect": aspect,
                            "values": entry.values,
                            "best_review": verdict,
                        }));
                        continue;
                    }
                    Some((verdict, _exact)) => review = Some(verdict),
                    None => {
                        review_failed += 1;
                        failures.push(json!({
                            "aspect": aspect,
                            "error": format!("unparseable best review: {answer}"),
                        }));
                        continue;
                    }
                },
                Err(error) => {
                    review_failed += 1;
                    failures.push(json!({
                        "aspect": aspect,
                        "error": format!("best review failed: {}", error.0),
                    }));
                    continue;
                }
            }
        }
        let values_csv = entry.values.join(",");
        let mut proposal = Map::new();
        proposal.insert("aspect".to_string(), Value::String(aspect.clone()));
        proposal.insert(
            "values".to_string(),
            Value::Array(entry.values.iter().cloned().map(Value::String).collect()),
        );
        proposal.insert(
            "description".to_string(),
            Value::String(entry.description),
        );
        proposal.insert("support".to_string(), Value::Number(entry.support.into()));
        proposal.insert(
            "evidence_sessions".to_string(),
            Value::Array(entry.evidence.into_iter().map(Value::String).collect()),
        );
        if let Some(review) = review {
            proposal.insert("best_review".to_string(), Value::String(review));
        }
        proposal.insert(
            "next".to_string(),
            json!([
                format!(
                    "transcript-label-trainer autolabel --aspect {aspect} --values {values_csv} --best"
                ),
                format!(
                    "transcript-label-trainer train --aspect {aspect} \
                     --training scikit-learn --eval-split scikit-learn"
                ),
            ]),
        );
        proposals.push(Value::Object(proposal));
    }

    let mut summary = Map::new();
    summary.insert("brama_model".to_string(), Value::String(model_id));
    summary.insert(
        "sessions_sampled".to_string(),
        Value::Number(sampled.into()),
    );
    summary.insert("chunks".to_string(), Value::Number(chunks.into()));
    summary.insert(
        "best_review".to_string(),
        json!({
            "enabled": best_review,
            "model": if best_review {
                Value::String(brama::BEST_MODEL.to_string())
            } else {
                Value::Null
            },
            "accepted": proposals.len(),
            "rejected_nonsensical": rejected.len(),
            "failed": review_failed,
            "sensible": !best_review || (rejected.is_empty() && review_failed == 0),
        }),
    );
    summary.insert("proposals".to_string(), Value::Array(proposals));
    summary.insert("rejected".to_string(), Value::Array(rejected));
    summary.insert("failures".to_string(), Value::Array(failures));
    Ok(Value::Object(summary))
}

/// Ask the teacher about every session of `batch` in one call. When Brama
/// answers that the prompt does not fit the routed model, ask about each half
/// instead, so the model's own context decides how many sessions one call
/// shows. A single session that does not fit alone is a failure naming it.
fn teacher_answers(
    client: &brama::BramaClient,
    model: &str,
    batch: &[(String, String)],
    answers: &mut Vec<(usize, Result<String>)>,
) {
    match client.answer(model, &teacher_prompt(batch)) {
        Ok(brama::ChatAnswer::Content(answer)) => answers.push((batch.len(), Ok(answer))),
        Ok(brama::ChatAnswer::DoesNotFit(_)) if batch.len() > 1 => {
            let (first, second) = batch.split_at(batch.len() / 2);
            teacher_answers(client, model, first, answers);
            teacher_answers(client, model, second, answers);
        }
        Ok(brama::ChatAnswer::DoesNotFit(detail)) => answers.push((
            1,
            Err(crate::util::Error(format!(
                "session {} alone does not fit {model}: {detail}",
                batch[0].0
            ))),
        )),
        Err(error) => answers.push((batch.len(), Err(error))),
    }
}
