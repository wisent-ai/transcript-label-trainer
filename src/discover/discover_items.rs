use super::*;

/// A count a command option names, refused when it is negative.
fn count_option(flag: &str, value: Option<i64>) -> Result<Option<usize>> {
    match value {
        Some(value) => Ok(Some(usize::try_from(value).map_err(|error| {
            crate::util::Error(format!("{flag} must be a count, not {value}: {error}"))
        })?)),
        None => Ok(None),
    }
}

/// Propose aspect dimensions from recent sessions via the Brama teacher.
pub fn discover(
    limit: Option<i64>,
    brama_model: Option<&str>,
    best_review: bool,
    max_aspects: Option<i64>,
    runtime: Option<&str>,
) -> Result<Value> {
    let model_id = match brama_model {
        Some(model) if model.trim().is_empty() => {
            return Err(crate::util::Error(
                "--brama-model is empty; name a Brama alias or leave it out".to_string(),
            ));
        }
        Some(model) => model.trim().to_string(),
        None => brama::DEFAULT_MODEL.to_string(),
    };
    let session_limit = count_option("--limit", limit)?;
    let keep = count_option("--max-aspects", max_aspects)?;
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
    if let Some(session_limit) = session_limit {
        if targets.len() > session_limit {
            // The lake lists oldest first; the newest sessions are the ones the
            // operator's current habits are visible in.
            targets = targets.split_off(targets.len() - session_limit);
        }
    }
    let texts = lake::session_texts(&targets)?;

    let mut merged: BTreeMap<String, Merged> = BTreeMap::new();
    let mut failures: Vec<Value> = Vec::new();
    let excerpts: Vec<(String, String)> = targets
        .iter()
        .filter_map(|session_id| {
            let text = texts.get(session_id)?.text.trim();
            (!text.is_empty()).then(|| (session_id.clone(), text.to_string()))
        })
        .collect();
    let mut answers = Vec::new();
    if !excerpts.is_empty() {
        teacher_answers(&client, &model_id, &excerpts, &mut answers);
    }
    let sampled: usize = answers.iter().map(|(sessions, _)| sessions).sum();
    let chunks = answers.len();
    for (chunk, (_, answer)) in answers.into_iter().enumerate() {
        let proposals = match answer.map(called_proposals) {
            Ok(Ok(proposals)) => proposals,
            Ok(Err(why)) => {
                failures.push(json!({ "chunk": chunk, "error": why }));
                continue;
            }
            Err(error) => {
                failures.push(json!({ "chunk": chunk, "error": error.0 }));
                continue;
            }
        };
        for proposal in proposals {
            let proposal = match read_proposal(&proposal) {
                Ok(proposal) => proposal,
                Err(why) => {
                    failures.push(json!({ "chunk": chunk, "error": why }));
                    continue;
                }
            };
            let entry = merged.entry(proposal.aspect).or_insert_with(|| Merged {
                values: Vec::new(),
                description: proposal.description.clone(),
                supported_by: Vec::new(),
                evidence: Vec::new(),
            });
            entry.supported_by.push(chunk);
            for value in proposal.values {
                if !entry.values.contains(&value) {
                    entry.values.push(value);
                }
            }
            for session_id in proposal.evidence {
                if !entry.evidence.contains(&session_id) {
                    entry.evidence.push(session_id);
                }
            }
            if entry.description.is_empty() {
                entry.description = proposal.description;
            }
        }
    }

    let mut ranked: Vec<(String, Merged)> = merged.into_iter().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .supported_by
            .len()
            .cmp(&left.1.supported_by.len())
            .then_with(|| left.0.cmp(&right.0))
    });
    // Every merged proposal unless the caller names how many to keep.
    if let Some(keep) = keep {
        ranked.truncate(keep);
    }

    let mut proposals: Vec<Value> = Vec::new();
    let mut rejected: Vec<Value> = Vec::new();
    let mut review_failed: Vec<String> = Vec::new();
    for (aspect, entry) in ranked {
        let mut review = None;
        if best_review {
            match client.choose(
                brama::BEST_DECISION_MODEL,
                &review_state(&aspect, &entry.values, &entry.description),
                "Is this a sensible aspect to classify every coding-agent session on?",
                REVIEW_LABELS,
            ) {
                Ok(verdict) if verdict == NONSENSICAL => {
                    rejected.push(json!({
                        "aspect": aspect,
                        "values": entry.values,
                        "best_review": verdict,
                    }));
                    continue;
                }
                Ok(verdict) => review = Some(verdict),
                Err(error) => {
                    failures.push(json!({
                        "aspect": aspect,
                        "error": format!("best review failed: {}", error.0),
                    }));
                    review_failed.push(aspect);
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
        proposal.insert("description".to_string(), Value::String(entry.description));
        proposal.insert("support".to_string(), json!(entry.supported_by.len()));
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
    summary.insert("sessions_sampled".to_string(), json!(sampled));
    summary.insert("chunks".to_string(), json!(chunks));
    summary.insert(
        "best_review".to_string(),
        json!({
            "enabled": best_review,
            "model": if best_review {
                Value::String(brama::BEST_DECISION_MODEL.to_string())
            } else {
                Value::Null
            },
            "accepted": proposals.len(),
            "rejected_nonsensical": rejected.len(),
            "failed": review_failed.len(),
            "sensible": !best_review || (rejected.is_empty() && review_failed.is_empty()),
        }),
    );
    summary.insert("proposals".to_string(), Value::Array(proposals));
    summary.insert("rejected".to_string(), Value::Array(rejected));
    summary.insert("failures".to_string(), Value::Array(failures));
    Ok(Value::Object(summary))
}

/// How many parts a batch that does not fit is split into.
const BISECTION_PARTS: usize = 2; // https://en.wikipedia.org/wiki/Bisection_method "repeatedly bisecting the interval" — halving keeps the number of calls logarithmic in the batch

/// Ask the teacher about every session of `batch` in one required call of
/// [`PROPOSALS`]. When Brama answers that the prompt does not fit the routed
/// model, ask about each half instead, so the model's own context decides how
/// many sessions one call shows. A single session that does not fit alone is
/// a failure naming it.
fn teacher_answers(
    client: &brama::BramaClient,
    model: &str,
    batch: &[(String, String)],
    answers: &mut Vec<(usize, Result<Map<String, Value>>)>,
) {
    let called = client.called(
        model,
        &teacher_prompt(batch),
        PROPOSALS,
        "The aspects proposed from the sessions shown, each grounded in them.",
        proposals_schema(),
    );
    match called {
        Ok(brama::Called::Arguments(arguments)) => answers.push((batch.len(), Ok(arguments))),
        Ok(brama::Called::DoesNotFit(detail)) => match batch {
            [only] => answers.push((
                batch.len(),
                Err(crate::util::Error(format!(
                    "session {} alone does not fit {model}: {detail}",
                    only.0
                ))),
            )),
            _ => {
                for half in batch.chunks(batch.len().div_ceil(BISECTION_PARTS)) {
                    teacher_answers(client, model, half, answers);
                }
            }
        },
        Err(error) => answers.push((batch.len(), Err(error))),
    }
}
