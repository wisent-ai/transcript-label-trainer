//! The journey graph itself: the embedded definition checked before a screen
//! is shown, the transitions between screens, and the evidence a screen
//! requires before the walk may move past it.

use super::*;

/// The embedded definition, checked for the identity and the graph this
/// command relies on before a single screen is shown. It is embedded at build
/// time, so it is always the file this code was written with: who it is for,
/// that every screen has an id, a kind, a title and a body, and that every
/// transition names an existing target and a priority.
pub(crate) fn canonical_definition() -> Result<Value> {
    let definition: Value = serde_json::from_str(DEFINITION)?;
    if definition.get("product_id").and_then(Value::as_str) != Some(PRODUCT_ID)
        || definition.get("journey_id").and_then(Value::as_str) != Some(JOURNEY_ID)
        || definition.get("first_success_fact").and_then(Value::as_str) != Some(FIRST_SUCCESS_FACT)
    {
        return Err(Error(
            "canonical onboarding journey identity mismatch".into(),
        ));
    }
    let entry = string_field(&definition, "entry_screen_id")
        .ok_or_else(|| Error("canonical onboarding journey has no entry screen".into()))?;
    let screens = definition
        .get("screens")
        .and_then(Value::as_array)
        .ok_or_else(|| Error("canonical onboarding journey has no screens".into()))?;
    let mut ids: Vec<&str> = Vec::with_capacity(screens.len());
    for screen in screens {
        let id = screen
            .get("screen_id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error("canonical onboarding screen has no id".into()))?;
        if ids.contains(&id) {
            return Err(Error(format!(
                "duplicate canonical onboarding screen id: {id}"
            )));
        }
        if screen.get("screen_kind").and_then(Value::as_str).is_none() {
            return Err(Error(format!(
                "canonical onboarding screen is incomplete: {id}"
            )));
        }
        presentation_text(screen, id, "title")?;
        presentation_text(screen, id, "body")?;
        ids.push(id);
    }
    if !ids.contains(&entry.as_str()) {
        return Err(Error(
            "canonical onboarding entry screen does not exist".into(),
        ));
    }
    for screen in screens {
        let id = screen
            .get("screen_id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error("canonical onboarding screen has no id".into()))?;
        let mut priorities = std::collections::HashSet::new();
        for transition in transitions(screen)? {
            let (next, priority) = edge(id, transition)?;
            if !ids.contains(&next) {
                return Err(Error(format!(
                    "canonical onboarding transition target does not exist: {next}"
                )));
            }
            if !priorities.insert(priority) {
                return Err(Error(format!(
                    "canonical onboarding transitions of {id} share priority {priority}; the journey contract requires them unique"
                )));
            }
        }
    }
    Ok(definition)
}

/// One text a screen's presentation must carry, refused naming the screen.
pub(crate) fn presentation_text<'a>(screen: &'a Value, id: &str, field: &str) -> Result<&'a str> {
    screen
        .get("presentation")
        .and_then(|presentation| presentation.get(field))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error(format!(
                "canonical onboarding screen {id} has no presentation {field}"
            ))
        })
}

/// A screen's transitions. A screen that lists none ends the walk; one whose
/// transitions are not a list is refused rather than read as an end.
pub(crate) fn transitions(screen: &Value) -> Result<&[Value]> {
    match screen.get("transitions") {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(list)) => Ok(list),
        Some(other) => Err(Error(format!(
            "canonical onboarding screen's transitions are not a list: {other}"
        ))),
    }
}

/// One transition's target and priority, refused naming the screen and the
/// target when either is missing or the priority is not a non-negative whole
/// number, the journey contract's type for it.
fn edge<'a>(screen_id: &str, transition: &'a Value) -> Result<(&'a str, u64)> {
    let target = transition
        .get("next_screen_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error(format!(
                "canonical onboarding transition of {screen_id} has no target"
            ))
        })?;
    let priority = transition
        .get("priority")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            Error(format!(
                "canonical onboarding transition from {screen_id} to {target} declares no non-negative whole-number priority"
            ))
        })?;
    Ok((target, priority))
}

pub(crate) fn screen_by_id<'a>(definition: &'a Value, screen_id: &str) -> Result<&'a Value> {
    definition
        .get("screens")
        .and_then(Value::as_array)
        .and_then(|screens| {
            screens
                .iter()
                .find(|screen| screen.get("screen_id").and_then(Value::as_str) == Some(screen_id))
        })
        .ok_or_else(|| {
            Error(format!(
                "published onboarding screen is unavailable: {screen_id}"
            ))
        })
}

/// The published edge out of this screen: the lowest priority leads, as the
/// journey contract and the control plane's own routing order transitions.
/// Priorities are unique per screen, which the definition check enforces.
pub(crate) fn next_screen_id(screen: &Value) -> Result<Option<&str>> {
    let screen_id = screen
        .get("screen_id")
        .and_then(Value::as_str)
        .ok_or_else(|| Error("canonical onboarding screen has no id".into()))?;
    let mut chosen: Option<(u64, &str)> = None;
    for transition in transitions(screen)? {
        let (target, rank) = edge(screen_id, transition)?;
        if chosen.is_none_or(|(best, _)| rank < best) {
            chosen = Some((rank, target));
        }
    }
    Ok(chosen.map(|(_, target)| target))
}

/// Whether the evidence gathered satisfies what the definition requires of the
/// screen. A screen with no rule is satisfied by arriving.
pub(crate) fn evidence_satisfied(screen: &Value, evidence: &Map<String, Value>) -> Result<bool> {
    let Some(rule) = screen
        .get("completion_evidence")
        .filter(|value| !value.is_null())
    else {
        return Ok(true);
    };
    if rule.get("kind").and_then(Value::as_str) != Some("fact")
        || rule.get("operator").and_then(Value::as_str) != Some("eq")
    {
        return Err(Error(
            "unsupported canonical onboarding evidence rule".into(),
        ));
    }
    let name = rule
        .get("fact")
        .and_then(Value::as_str)
        .ok_or_else(|| Error("canonical onboarding evidence rule has no fact".into()))?;
    let expected = rule
        .get("value")
        .ok_or_else(|| Error("canonical onboarding evidence rule has no expected value".into()))?;
    Ok(evidence.get(name) == Some(expected))
}
