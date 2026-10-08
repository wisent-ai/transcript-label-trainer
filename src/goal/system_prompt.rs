use super::*;

pub(crate) const SYSTEM_PROMPT: &str = include_str!("../../training/goal-model/goal-system-prompt.txt");

pub(crate) const REVIEW_VALUES: [&str; 2] = ["sensible", "nonsensical"];

/// Curation decides which rows become training labels, so it uses the
/// strongest operator-approved route rather than a product-serving route.
pub(crate) const CURATION_REVIEW_MODEL: &str = BEST_MODEL;

pub(crate) const AUDIT_VALUES: [&str; 4] = [
    "both-sensible",
    "label-nonsensical",
    "student-nonsensical",
    "both-nonsensical",
];

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
            if goal.is_empty() {
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

/// Every user message in Transcript Lake, newest first.
pub(crate) fn unlabeled_rows() -> Result<Vec<GoalRow>> {
    let sql = r#"
SELECT session_id, runtime, text AS message
FROM events
WHERE event_type = 'user' AND text IS NOT NULL
ORDER BY ts DESC
"#;
    Ok(lake::query(sql)?
        .into_iter()
        .map(|value| GoalRow {
            session_id: text(&value, "session_id"),
            runtime: text(&value, "runtime"),
            message: text(&value, "message"),
            goal: None,
            goal_source: None,
            gold: false,
            reviewed_by: None,
        })
        .filter(|row| !row.message.trim().is_empty())
        .collect())
}
