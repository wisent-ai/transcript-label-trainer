use super::*;

/// Screens as they are shown, plus the terminal verdict. Human output is
/// printed as the walk happens; `--json` collects the same walk into one
/// object, because a machine reader wants one document, not a transcript.
/// A journey version or current screen nothing stated is `null` in that
/// object, not an empty name.
pub(crate) struct Report {
    pub(crate) journey_version: Option<String>,
    pub(crate) reset: bool,
    pub(crate) json: bool,
    pub(crate) steps: Vec<Value>,
    pub(crate) status: &'static str,
    pub(crate) current_screen_id: Option<String>,
    pub(crate) next: String,
}

impl Report {
    pub(crate) fn new(definition: &Value, reset: bool, json: bool) -> Self {
        Self {
            journey_version: string_field(definition, "journey_version"),
            reset,
            json,
            steps: Vec::new(),
            status: "in_progress",
            current_screen_id: None,
            next: String::new(),
        }
    }

    /// Show one screen. The definition check already required its id, title
    /// and body, so a screen without them is refused here by the same reader.
    pub(crate) fn render(&mut self, screen: &Value) -> Result<()> {
        let id = screen
            .get("screen_id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error("canonical onboarding screen has no id".into()))?;
        let title = presentation_text(screen, id, "title")?;
        let body = presentation_text(screen, id, "body")?;
        let command = screen
            .get("presentation")
            .and_then(|value| value.get("command"))
            .and_then(Value::as_str);
        if self.json {
            self.steps.push(json!({
                "screen_id": id,
                "screen_kind": screen.get("screen_kind"),
                "title": title,
                "body": body,
                "command": command,
                "actions": screen.get("actions"),
            }));
            return Ok(());
        }
        println!("\n== {title} ==\n{body}");
        if let Some(command) = command {
            println!("command: {command}");
        }
        Ok(())
    }

    /// A line about what this machine actually holds, printed under the screen
    /// it qualifies. In `--json` it belongs to the step it followed.
    pub(crate) fn note(&mut self, text: &str) {
        if !self.json {
            println!("{text}");
            return;
        }
        if let Some(step) = self.steps.last_mut() {
            let text = Value::String(text.to_string());
            match step.get_mut("notes").and_then(Value::as_array_mut) {
                Some(notes) => notes.push(text),
                None => step["notes"] = Value::Array(vec![text]),
            }
        }
    }

    /// The suggestion records inference emitted: the first result of the
    /// product, shaped exactly like the lake's label-store records.
    pub(crate) fn rows(&mut self, rows: &[Value]) {
        if self.json {
            if let Some(step) = self.steps.last_mut() {
                step["suggestions"] = Value::Array(rows.to_vec());
            }
            return;
        }
        println!("{} suggestion(s):", rows.len());
        for row in rows {
            println!("  {row}");
        }
    }

    pub(crate) fn finish(&mut self, status: &'static str, state: &Value, next: &str) {
        self.status = status;
        self.current_screen_id = string_field(state, "current_screen_id");
        self.next = next.to_string();
    }

    pub(crate) fn emit(self) -> Result<i32> {
        if !self.json {
            println!("\nstatus: {}", self.status);
            println!("state: {}", state_path().display());
            println!("next: {}", self.next);
            return Ok(crate::cli::EXIT_SUCCESS);
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "product_id": PRODUCT_ID,
                "journey_id": JOURNEY_ID,
                "journey_version": self.journey_version,
                "status": self.status,
                "current_screen_id": self.current_screen_id,
                "first_success_fact": FIRST_SUCCESS_FACT,
                "state_path": state_path().display().to_string(),
                "reset": self.reset,
                "steps": self.steps,
                "next": self.next,
            }))?
        );
        Ok(crate::cli::EXIT_SUCCESS)
    }
}

pub(crate) fn advance(
    definition: &Value,
    screen: &Value,
    state: &mut Value,
    evidence: &Map<String, Value>,
    revision: &str,
) -> Result<Option<String>> {
    if !evidence_satisfied(screen, evidence)? {
        return Ok(None);
    }
    let Some(next) = next_screen_id(screen)?.map(str::to_string) else {
        return Ok(None);
    };
    screen_by_id(definition, &next)?;
    set(state, "current_screen_id", Value::String(next.clone()))?;
    set(state, "revision", Value::String(revision.to_string()))?;
    save_state(state)?;
    Ok(Some(next))
}

pub(crate) fn complete(
    screen: &Value,
    state: &mut Value,
    evidence: &Map<String, Value>,
    revision: &str,
) -> Result<bool> {
    if !evidence_satisfied(screen, evidence)? {
        return Ok(false);
    }
    set(state, "status", Value::String("completed".into()))?;
    set(state, "completed_at", Value::String(now_iso()))?;
    set(state, "revision", Value::String(revision.to_string()))?;
    save_state(state)?;
    Ok(true)
}
