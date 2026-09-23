use super::*;

pub(crate) const PRODUCT_ID: &str = "transcript-label-trainer";

pub(crate) const JOURNEY_ID: &str = "first-use";

pub(crate) const STATE_SCHEMA: &str = "transcript-label-trainer.onboarding-state.v1";

pub(crate) const CORPUS_FACT: &str = "labeled_corpus_adopted";

pub(crate) const TRAINED_FACT: &str = "aspect_classifier_trained";

pub(crate) const FIRST_SUCCESS_FACT: &str = "label_suggestion_emitted";

/// The published definition, embedded at build time from the file Echo's
/// publisher discovers at `origin/main`.
pub(crate) const DEFINITION: &str = include_str!("../../onboarding_first_use.json");

/// Suggestions the walkthrough asks for: enough to read on one screen, and
/// the same code path `infer --limit` takes.
pub(crate) const WALKTHROUGH_LIMIT: i64 = 5;

/// The published contract's own screen ceiling, used here only to keep a walk
/// over a republished definition finite.
pub(crate) const MAX_STEPS: usize = 128;

pub fn onboarding(
    reset: bool,
    yes: bool,
    json_output: bool,
    corpus_path: Option<&str>,
    skip_corpus: bool,
) -> Result<i32> {
    // Machine output has no reader to press Enter, so it never prompts.
    let unattended = yes || json_output;
    let definition = canonical_definition()?;
    let revision = format!("{PRODUCT_ID}-{}", env!("CARGO_PKG_VERSION"));
    let mut state = load_or_start_state(&definition, &revision, reset)?;
    let mut report = Report::new(&definition, reset, json_output);

    if state.get("status").and_then(Value::as_str) == Some("completed") {
        report.finish(
            "completed",
            &state,
            "A trained classifier already suggested labels on this machine. \
             Continue with: transcript-label-trainer infer --help",
        );
        return report.emit();
    }

    loop {
        let screen_id = string_field(&state, "current_screen_id")
            .ok_or_else(|| Error("onboarding state has no current screen".into()))?;
        let screen = screen_by_id(&definition, &screen_id)?.clone();
        report.render(&screen);

        match screen.get("screen_kind").and_then(Value::as_str) {
            Some("first_action") => match screen_fact(&screen)? {
                CORPUS_FACT => {
                    if skip_corpus {
                        report.note(
                            "No corpus was adopted. The trainer remains empty and usable; \
                             corpus-adopt or live Transcript Lake labels can be added later.",
                        );
                        report.finish(
                            "skipped",
                            &state,
                            "Adopt later with: transcript-label-trainer corpus-adopt <bundle.json>",
                        );
                        return report.emit();
                    }
                    if let Some(path) = corpus_path {
                        let adoption = corpus::adopt(std::path::Path::new(path))?;
                        report.note(&format!(
                            "Selected corpus {} for aspect '{}' ({} imported, {} unchanged, \
                             {} conflicting, {} rejected).",
                            string_field(&adoption, "corpusId").unwrap_or_default(),
                            string_field(&adoption, "aspect").unwrap_or_default(),
                            adoption.get("imported").and_then(Value::as_u64).unwrap_or(0),
                            adoption.get("unchanged").and_then(Value::as_u64).unwrap_or(0),
                            adoption.get("conflicting").and_then(Value::as_u64).unwrap_or(0),
                            adoption.get("rejected").and_then(Value::as_u64).unwrap_or(0),
                        ));
                    } else if corpus::selected_bundle_path()?.is_none() {
                        report.note(
                            "No adopted corpus is selected. Supply an existing canonical \
                             dataset bundle, or explicitly skip this optional step.",
                        );
                        report.finish(
                            "awaiting_corpus",
                            &state,
                            "Run: transcript-label-trainer onboarding --corpus <bundle.json> \
                             (or --skip-corpus)",
                        );
                        return report.emit();
                    } else {
                        report.note("The previously adopted corpus remains selected.");
                    }
                    let evidence = fact(CORPUS_FACT);
                    advance(&definition, &screen, &mut state, &evidence, &revision)?
                        .ok_or_else(|| {
                            Error(
                                "an adopted corpus does not satisfy the published journey".into(),
                            )
                        })?;
                }
                TRAINED_FACT => {
                    let Some(aspect) = trained_aspect()? else {
                        report.note(
                            "No aspect is trained under the training root yet, so there is \
                             nothing to suggest labels with.",
                        );
                        report.finish(
                            "awaiting_training",
                            &state,
                            "Train one aspect, then run: transcript-label-trainer onboarding",
                        );
                        return report.emit();
                    };
                    report.note(&format!(
                        "Aspect '{aspect}' has a trained classifier under the training root."
                    ));
                    let evidence = fact(TRAINED_FACT);
                    advance(&definition, &screen, &mut state, &evidence, &revision)?
                        .ok_or_else(|| {
                            Error(
                                "a trained classifier does not satisfy the published journey"
                                    .into(),
                            )
                        })?;
                }
                other => {
                    return Err(Error(format!(
                        "unsupported first-action completion fact '{other}'"
                    )))
                }
            },
            Some("first_success") => {
                let aspect = trained_aspect()?
                    .ok_or_else(|| Error("no trained aspect to suggest labels for".into()))?;
                report.note(&format!(
                    "Running: transcript-label-trainer infer --aspect {aspect} \
                     --limit {WALKTHROUGH_LIMIT}"
                ));
                // The product's own inference path, which is what records the
                // first-success fact when it emits suggestions.
                let suggestions = match model::infer(&aspect, None, Some(WALKTHROUGH_LIMIT)) {
                    Ok(suggestions) => suggestions,
                    Err(error) => {
                        report.note(&format!("Suggestions could not be emitted: {error}"));
                        report.finish(
                            "awaiting_lake",
                            &state,
                            "Make the lake readable (transcript-lake on PATH, LAKE_DATA set), \
                             then run: transcript-label-trainer onboarding",
                        );
                        return report.emit();
                    }
                };
                let records = suggestions.as_array().map(Vec::as_slice).unwrap_or_default();
                if records.is_empty() {
                    report.finish(
                        "awaiting_sessions",
                        &state,
                        "Every session the lake holds already carries this label; capture more \
                         sessions, then run: transcript-label-trainer onboarding",
                    );
                    return report.emit();
                }
                report.rows(records);
                wait_for_enter(unattended, "Press Enter to finish onboarding. ")?;
                // Inference recorded the fact where the suggestions were
                // emitted; read that back rather than asserting it here.
                state = recorded_state(&definition)?;
                if state.get("status").and_then(Value::as_str) != Some("completed") {
                    return Err(Error(
                        "emitted suggestions did not record the published first-success fact"
                            .into(),
                    ));
                }
                report.finish(
                    "completed",
                    &state,
                    "Review the records, then apply the accepted ones through the lake's \
                     labeler: transcript-lake label add <session-id> --aspect <name> \
                     --value <v> --source model",
                );
                return report.emit();
            }
            Some(_) => {
                wait_for_enter(unattended, "Press Enter to continue. ")?;
                advance(&definition, &screen, &mut state, &Map::new(), &revision)?.ok_or_else(
                    || Error("published journey has no eligible next screen".into()),
                )?;
            }
            None => return Err(Error("published onboarding screen has no kind".into())),
        }
    }
}

/// Record the first real result this product produces: label suggestions
/// emitted by a trained classifier over reconstructed session text.
///
/// Called from `model::infer` after the records exist, so the fact is written
/// where the effect happens and not where a walkthrough asks about it. A run
/// that emitted nothing is not a first success, and progress that cannot be
/// written is a warning on stderr: onboarding state never fails an `infer`.
pub fn record_first_success(aspect: &str, suggestions: usize) {
    if suggestions == 0 {
        return;
    }
    if let Err(error) = record(aspect, suggestions) {
        lake::warn(&format!("first-use progress was not recorded: {error}"));
    }
}

pub(crate) fn record(aspect: &str, suggestions: usize) -> Result<()> {
    let definition = canonical_definition()?;
    let revision = format!(
        "{PRODUCT_ID}-{}:infer:{aspect}:{suggestions}",
        env!("CARGO_PKG_VERSION")
    );
    let mut state = load_or_start_state(&definition, &revision, false)?;
    if state.get("status").and_then(Value::as_str) == Some("completed") {
        return Ok(());
    }
    // Suggestions in hand satisfy training and first success. The source
    // selection step is satisfied only when an adopted corpus really exists.
    let mut evidence = fact(TRAINED_FACT);
    if corpus::selected_bundle_path()?.is_some() {
        evidence.insert(CORPUS_FACT.to_string(), Value::Bool(true));
    }
    evidence.insert(FIRST_SUCCESS_FACT.to_string(), Value::Bool(true));

    for _ in 0..MAX_STEPS {
        let screen_id = string_field(&state, "current_screen_id")
            .ok_or_else(|| Error("onboarding state has no current screen".into()))?;
        let screen = screen_by_id(&definition, &screen_id)?.clone();
        if transitions(&screen).is_empty() {
            if !complete(&screen, &mut state, &evidence, &revision)? {
                return Err(Error(
                    "published first-success evidence was not satisfied".into(),
                ));
            }
            return Ok(());
        }
        advance(&definition, &screen, &mut state, &evidence, &revision)?
            .ok_or_else(|| Error("published journey has no eligible next screen".into()))?;
    }
    Err(Error("published journey does not reach a final screen".into()))
}

/// The aspect the walkthrough demonstrates. Any aspect `info` reports a
/// trained artifact for will do, but an artifact whose backend this build
/// cannot load is a demonstration that cannot run, so one it can load wins.
pub(crate) fn trained_aspect() -> Result<Option<String>> {
    let entries = model::info()?;
    let trained: Vec<(&str, &Value)> = entries
        .iter()
        .filter_map(|entry| Some((entry.get("active")?.as_str()?, entry)))
        .collect();
    let chosen = trained
        .iter()
        .find(|(backend, _)| model::loadable(backend))
        .or_else(|| trained.first());
    Ok(chosen.and_then(|(_, entry)| string_field(entry, "aspect")))
}

pub(crate) fn screen_fact(screen: &Value) -> Result<&str> {
    screen
        .get("completion_evidence")
        .and_then(|rule| rule.get("fact"))
        .and_then(Value::as_str)
        .ok_or_else(|| Error("first-action screen has no completion fact".into()))
}

/// One fact, asserted true: the only evidence shape the shipped journey uses.
pub(crate) fn fact(name: &str) -> Map<String, Value> {
    Map::from_iter([(name.to_string(), Value::Bool(true))])
}
