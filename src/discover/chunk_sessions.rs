use super::*;

/// Sessions shown to the teacher per call: enough contrast to see a dimension,
/// small enough that every transcript excerpt stays readable.
pub(crate) const CHUNK_SESSIONS: usize = 5;

/// Per-session excerpt cap inside a teacher prompt.
pub(crate) const SESSION_CHARS: usize = 4000;

/// A proposal list is a few hundred tokens of JSON, not one word.
pub(crate) const PROPOSAL_MAX_TOKENS: u32 = 900;

pub(crate) const DEFAULT_SESSION_LIMIT: usize = 60;

pub(crate) const DEFAULT_MAX_ASPECTS: usize = 8;

pub(crate) const REVIEW_VALUES: [&str; 2] = ["sensible", "nonsensical"];

/// Values kept per merged aspect; a dimension needing more is a free-text
/// field, not a classification aspect.
pub(crate) const MAX_VALUES: usize = 8;

pub(crate) const MAX_EVIDENCE: usize = 12;

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
                      from a transcript alone; 2-6 mutually exclusive values in \
                      lowercase-kebab-case; no aspect that would be true of \
                      every session or of almost none; no restating of obvious \
                      metadata (language, length, runtime). Answer with strict \
                      JSON only: an array of at most 4 objects, each \
                      {\"aspect\": \"kebab-case-name\", \"values\": [..], \
                      \"description\": \"one sentence\", \
                      \"evidence\": [\"session ids from the input\"]}."
                .to_string(),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Propose aspects grounded in these sessions.\n\n{transcripts}"
            ),
        },
    ]
}

pub(crate) fn review_prompt(aspect: &str, values: &[String], description: &str) -> Vec<brama::Message> {
    vec![
        brama::Message {
            role: "system".to_string(),
            content: "You audit proposed aspect dimensions for coding-agent \
                      session transcripts. A sensible aspect is judgeable from \
                      a transcript alone, has mutually exclusive values, and \
                      would divide real sessions rather than land on one value \
                      always. Answer with exactly one word: sensible or \
                      nonsensical."
                .to_string(),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Aspect: {aspect}\nValues: {}\nDescription: {description}\n\n\
                 Is this a sensible aspect to classify every session on? \
                 Answer sensible or nonsensical.",
                values.join(", ")
            ),
        },
    ]
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

/// The teacher's JSON, with a possible ```json fence stripped.
pub(crate) fn parse_proposals(answer: &str) -> Option<Vec<Value>> {
    let trimmed = answer
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str::<Value>(trimmed)
        .ok()?
        .as_array()
        .cloned()
}

pub(crate) struct Merged {
    pub(crate) values: Vec<String>,
    pub(crate) description: String,
    pub(crate) support: usize,
    pub(crate) evidence: Vec<String>,
}
