use super::*;

pub(crate) const SYSTEM_PROMPT: &str = include_str!("../../training/goal-model/goal-system-prompt.txt");

pub(crate) const REVIEW_VALUES: [&str; 2] = ["sensible", "nonsensical"];

/// Curation decides which rows become training labels, so it uses the
/// strongest operator-approved route. It named `wisent-backend/chat/primary`
/// until 2026-08-18 — an unrelated product model with no business judging
/// task labels.
pub(crate) const CURATION_REVIEW_MODEL: &str = BEST_MODEL;

pub(crate) const AUDIT_VALUES: [&str; 4] = [
    "both-sensible",
    "label-nonsensical",
    "student-nonsensical",
    "both-nonsensical",
];

pub(crate) const WORKERS: usize = 24;

pub(crate) const AUDIT_WORKERS: usize = 4;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GoalRow {
    pub session_id: String,
    pub runtime: String,
    pub message: String,
    pub goal: Option<String>,
    pub goal_source: Option<String>,
    #[serde(default)]
    pub gold: bool,
    #[serde(default)]
    pub reviewed_by: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) enum ParsedGoal {
    NoTask,
    Task(String),
}

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) index: usize,
    pub(crate) row: GoalRow,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Prediction {
    pub(crate) session_id: String,
    pub(crate) message: String,
    pub(crate) goal: String,
    pub(crate) student: String,
}

pub(crate) fn text(row: &Value, key: &str) -> String {
    row.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

pub(crate) fn normalized_digest(value: &str) -> String {
    let normalized = value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    hex::encode(Sha256::digest(normalized.as_bytes()))
}

pub(crate) fn obvious_no_task(value: &str) -> bool {
    let normalized = value
        .trim()
        .to_lowercase()
        .trim_matches(|character: char| {
            character.is_whitespace() || character.is_ascii_punctuation()
        })
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    [
        "ok",
        "okay",
        "okej",
        "tak",
        "yes",
        "no",
        "nie",
        "continue",
        "kontynuuj",
        "dalej",
        "go ahead",
        "proceed",
        "carry on",
        "keep going",
        "yes continue",
        "ok continue",
        "okay continue",
        "okej kontynuuj",
        "tak kontynuuj",
        "yes do that",
        "ok do that",
        "okay do that",
        "okej zrób to",
        "tak zrób to",
        "do it",
        "zrób to",
        "sounds good",
        "looks good",
        "that works",
        "perfect",
        "great",
        "super",
        "fine",
        "agreed",
        "zgoda",
        "jasne",
        "dobrze",
        "got it",
        "understood",
        "rozumiem",
        "thanks",
        "thank you",
        "dzięki",
        "dziekuje",
        "dziękuję",
        "hi",
        "hello",
        "hey",
        "hej",
        "ready",
        "gotowe",
    ]
    .contains(&normalized.as_str())
}

pub(crate) fn safe_message(value: &str) -> bool {
    let trimmed = value.trim();
    if !(3..=4_000).contains(&trimmed.chars().count()) || trimmed.starts_with('/') {
        return false;
    }
    let lower = trimmed.to_lowercase();
    ![
        "[masked:",
        "authorization: bearer",
        "begin private key",
        "begin rsa private key",
        "github_pat_",
        "gho_",
        "sk-ant-",
        "sk-proj-",
        "wisent_app_agent_auth_secret",
        "api_key=",
        "token=",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(crate) fn task_message(value: &str) -> bool {
    if !safe_message(value) {
        return false;
    }
    let lower = value.trim().to_lowercase();
    let synthetic_prefixes = [
        "you are a strict gate that",
        "you are swiatowid's prompt-to-goal classifier",
        "<system-reminder>",
        "on your first completion attempt",
        "decide whether the user message below",
        "you translate developer-tool ui strings",
        "use the read tool",
        "use bash to run exactly",
        "use the monitor tool",
        "use the notebookedit tool",
        "use the edit tool",
    ];
    !synthetic_prefixes
        .iter()
        .any(|prefix| lower.starts_with(prefix))
        && !lower.contains("api validation error:")
}

pub(crate) fn gold_rows() -> Result<Vec<GoalRow>> {
    let sql = r#"
WITH users AS (
  SELECT session_id, text,
         row_number() OVER (PARTITION BY session_id ORDER BY ts) AS rank
  FROM events
  WHERE runtime = 'omp' AND event_type = 'user' AND text IS NOT NULL
), titles AS (
  SELECT session_id, text,
         row_number() OVER (
           PARTITION BY session_id
           ORDER BY CASE json_extract_string(try_cast(extra AS JSON), '$.kind')
                      WHEN 'title_change' THEN 0 ELSE 1 END, ts
         ) AS rank
  FROM events
  WHERE runtime = 'omp' AND event_type = 'meta' AND text IS NOT NULL
    AND json_extract_string(try_cast(extra AS JSON), '$.kind') IN ('title', 'title_change')
)
SELECT users.session_id, 'omp' AS runtime, users.text AS message, titles.text AS goal
FROM users JOIN titles USING (session_id)
WHERE users.rank = 1 AND titles.rank = 1
"#;
    Ok(lake::query(sql)?
        .into_iter()
        .filter_map(|value| {
            let message = text(&value, "message");
            let goal = text(&value, "goal");
            if !task_message(&message) || goal.is_empty() {
                return None;
            }
            Some(GoalRow {
                session_id: text(&value, "session_id"),
                runtime: "omp".to_string(),
                message,
                goal: Some(goal),
                goal_source: Some("transcript-lake:omp-title".to_string()),
                gold: true,
                reviewed_by: None,
            })
        })
        .collect())
}

pub(crate) fn unlabeled_rows(limit: usize) -> Result<Vec<GoalRow>> {
    let fetch = limit.saturating_mul(20).max(limit);
    let sql = format!(
        r#"
WITH users AS (
  SELECT session_id, runtime, text, ts,
         row_number() OVER (PARTITION BY runtime, session_id ORDER BY ts) AS message_rank
  FROM events
  WHERE runtime IN ('omp', 'claude', 'codex', 'droid', 'kimi')
    AND event_type = 'user' AND text IS NOT NULL
)
SELECT session_id, runtime, text AS message
FROM users
WHERE message_rank <= 10 AND length(text) BETWEEN 3 AND 4000
ORDER BY ts DESC
LIMIT {fetch}
"#
    );
    let mut rows: Vec<GoalRow> = lake::query(&sql)?
        .into_iter()
        .filter_map(|value| {
            let message = text(&value, "message");
            let no_task = obvious_no_task(&message);
            if !no_task && !task_message(&message) {
                return None;
            }
            Some(GoalRow {
                session_id: text(&value, "session_id"),
                runtime: text(&value, "runtime"),
                message,
                goal: None,
                goal_source: no_task.then(|| "contract:no-task-v1".to_string()),
                gold: false,
                reviewed_by: None,
            })
        })
        .collect();
    rows.sort_by_key(|row| row.goal_source.is_none());
    Ok(rows)
}
