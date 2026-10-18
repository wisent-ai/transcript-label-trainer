use super::*;

/// The verdicts the best review chooses between, with what each means.
pub(crate) const SENSIBLE: &str = "sensible";
pub(crate) const NONSENSICAL: &str = "nonsensical";
pub(crate) const REVIEW_LABELS: &[(&str, &str)] = &[
    (SENSIBLE, "judgeable from a transcript alone, with mutually exclusive values that would divide real sessions"),
    (NONSENSICAL, "not judgeable from a transcript, values that overlap, or a dimension that lands on one value always"),
];

pub(crate) fn teacher_prompt(excerpts: &[(String, String)]) -> Vec<brama::Message> {
    let mut transcripts = String::new();
    for (session_id, text) in excerpts {
        transcripts.push_str(&format!("=== session {session_id} ===\n{text}\n\n"));
    }
    vec![
        brama::Message {
            role: "system".to_string(),
            content: "You design label taxonomies for coding-agent session \
                      transcripts. From the sessions shown you propose ASPECTS: \
                      independent dimensions worth classifying every session on, \
                      grounded in what the user actually asked for and how the \
                      agent actually answered — recurring behaviors, failure \
                      modes, user reactions. Rules: an aspect must be judgeable \
                      from a transcript alone; at least two mutually exclusive \
                      values in lowercase-kebab-case; no aspect that would be \
                      true of every session or of almost none; no restating of \
                      obvious metadata (language, length, runtime). Answer by \
                      calling propose_aspects with every aspect you propose."
                .to_string(),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!("Propose aspects grounded in these sessions.\n\n{transcripts}"),
        },
    ]
}

/// What the best review decides about one proposal.
pub(crate) fn review_state(aspect: &str, values: &[String], description: &str) -> String {
    format!(
        "Aspect: {aspect}\nValues: {}\nDescription: {description}",
        values.join(", ")
    )
}

pub(crate) fn kebab(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for character in name.trim().to_lowercase().chars() {
        if character.is_ascii_alphanumeric() {
            out.push(character);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

/// The function the teacher answers by: its parameters are the proposals, as
/// a strict JSON Schema, so they are the teacher's answer itself and never
/// JSON looked for in its prose. `read_proposal` still holds each one to what
/// the schema cannot state (kebab names, two values or more).
pub(crate) const PROPOSALS: &str = "propose_aspects";

pub(crate) fn proposals_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["aspect", "values", "description", "evidence"],
                    "properties": {
                        "aspect": { "type": "string", "description": "The aspect's name in lowercase-kebab-case." },
                        "values": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Its mutually exclusive values, in lowercase-kebab-case.",
                        },
                        "description": { "type": "string", "description": "One sentence." },
                        "evidence": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Session ids from the input the aspect is grounded in.",
                        },
                    },
                },
            },
        },
    })
}

/// The proposals list of the teacher's call.
pub(crate) fn called_proposals(
    arguments: Map<String, Value>,
) -> std::result::Result<Vec<Value>, String> {
    if let Some(Value::Array(proposals)) = arguments.get("proposals") {
        return Ok(proposals.clone());
    }
    Err(format!(
        "the teacher's call of {PROPOSALS} holds no proposals list: {}",
        Value::Object(arguments)
    ))
}

pub(crate) struct Merged {
    pub(crate) values: Vec<String>,
    pub(crate) description: String,
    /// The teacher answers (by their index) that proposed this aspect.
    pub(crate) supported_by: Vec<usize>,
    pub(crate) evidence: Vec<String>,
}

/// One proposal as the teacher stated it.
pub(crate) struct Proposal {
    pub(crate) aspect: String,
    pub(crate) values: Vec<String>,
    pub(crate) description: String,
    pub(crate) evidence: Vec<String>,
}

/// Read one proposal field by field, or say why it is not one: a proposal
/// needs an aspect name, at least two values a session can fall on, a
/// description and the session ids it is grounded in.
pub(crate) fn read_proposal(proposal: &Value) -> std::result::Result<Proposal, String> {
    let aspect = proposal
        .get("aspect")
        .and_then(Value::as_str)
        .map(kebab)
        .filter(|aspect| !aspect.is_empty())
        .ok_or_else(|| format!("a proposal names no aspect: {proposal}"))?;
    let values = proposal
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("proposal {aspect} has no values list: {proposal}"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(kebab)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("proposal {aspect} has a value that is not a name: {value}"))
        })
        .collect::<std::result::Result<Vec<String>, String>>()?;
    // A dimension classifies only when a session can fall on more than one value.
    if !crate::model::at_least_two(values.iter()) {
        return Err(format!(
            "proposal {aspect} offers fewer than two values: {proposal}"
        ));
    }
    let description = proposal
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .ok_or_else(|| format!("proposal {aspect} has no description: {proposal}"))?
        .to_string();
    let evidence = proposal
        .get("evidence")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("proposal {aspect} names no evidence sessions: {proposal}"))?
        .iter()
        .map(|id| {
            id.as_str().map(str::to_string).ok_or_else(|| {
                format!("proposal {aspect} has an evidence id that is not text: {id}")
            })
        })
        .collect::<std::result::Result<Vec<String>, String>>()?;
    Ok(Proposal {
        aspect,
        values,
        description,
        evidence,
    })
}
