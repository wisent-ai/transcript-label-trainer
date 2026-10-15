use super::*;

pub(crate) const SYSTEM_PROMPT: &str =
    include_str!("../../training/goal-model/goal-system-prompt.txt");

/// Curation decides which rows become training labels, so it uses the
/// strongest operator-approved route rather than a product-serving route.
pub(crate) const CURATION_REVIEW_MODEL: &str = BEST_MODEL;

/// The curation review's verdicts, with what each means to the reviewer.
pub(crate) const SENSIBLE: &str = "sensible";
pub(crate) const REVIEW_LABELS: &[(&str, &str)] = &[
    (SENSIBLE, "the goal is faithful to the user's self-contained task, imperative, 3-7 words, keeps product names and identifiers and invents no work; or it is empty and the text holds no self-contained actionable task"),
    ("nonsensical", "the goal misreads, invents or loses the task, breaks the 3-7 word form, or is empty although the text holds a task, or is non-empty for small talk or a continuation that depends on missing context"),
];

/// The final audit's verdicts on a reference label and a student output.
pub(crate) const BOTH_SENSIBLE: &str = "both-sensible";
pub(crate) const AUDIT_LABELS: &[(&str, &str)] = &[
    (BOTH_SENSIBLE, "the reference label and the student output are both sensible goals for the user text"),
    ("label-nonsensical", "only the reference label has a material semantic error, invented or lost action, malformed output or wrong no-task decision"),
    ("student-nonsensical", "only the student output has a material semantic error, invented or lost action, malformed output or wrong no-task decision"),
    ("both-nonsensical", "both the reference label and the student output have such an error"),
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

/// A text field `row` must carry, trimmed, or an error naming it and the row.
pub(crate) fn text(row: &Value, key: &str) -> Result<String> {
    row.get(key)
        .and_then(Value::as_str)
        .map(|value| value.trim().to_string())
        .ok_or_else(|| Error(format!("a goal row has no text field `{key}`: {row}")))
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
    let mut rows = Vec::new();
    for value in lake::query(sql)? {
        // A session whose title is blank has no goal to hold out as gold.
        let goal = text(&value, "goal")?;
        if goal.is_empty() {
            continue;
        }
        rows.push(GoalRow {
            session_id: text(&value, "session_id")?,
            runtime: text(&value, "runtime")?,
            message: text(&value, "message")?,
            goal: Some(goal),
            goal_source: Some("transcript-lake:omp-title".to_string()),
            gold: true,
            reviewed_by: None,
        });
    }
    Ok(rows)
}

/// Every user message in Transcript Lake, newest first.
pub(crate) fn unlabeled_rows() -> Result<Vec<GoalRow>> {
    let sql = r#"
SELECT session_id, runtime, text AS message
FROM events
WHERE event_type = 'user' AND text IS NOT NULL
ORDER BY ts DESC
"#;
    let mut rows = Vec::new();
    for value in lake::query(sql)? {
        let message = text(&value, "message")?;
        if message.is_empty() {
            continue;
        }
        rows.push(GoalRow {
            session_id: text(&value, "session_id")?,
            runtime: text(&value, "runtime")?,
            message,
            goal: None,
            goal_source: None,
            gold: false,
            reviewed_by: None,
        });
    }
    Ok(rows)
}
