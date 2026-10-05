use super::*;

impl BramaClient {
    pub fn new(url: &str, agent_id: String, secret: String, token: Option<String>) -> Result<Self> {
        // No clock on a request: it ends with Brama's answer or a transport
        // error, and reqwest's built-in 30 s limit is switched off.
        let http = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .map_err(|error| Error(format!("could not build the Brama HTTP client: {error}")))?;
        Ok(BramaClient {
            url: url.trim_end_matches('/').to_string(),
            agent_id,
            secret,
            token: token.filter(|value| !value.is_empty()),
            http,
        })
    }

    /// The signing identity and the two secrets come from the environment or
    /// from the vault roles the environment declares; nothing names an agent
    /// or an item here.
    pub fn from_env() -> Result<Self> {
        let agent_id = env_trimmed("WISENT_APP_AGENT_ID");
        if agent_id.is_empty() {
            bail!("WISENT_APP_AGENT_ID is required: the agent identity Brama verifies the signature against")
        }
        let mut secret = env_trimmed("WISENT_APP_AGENT_AUTH_SECRET");
        if secret.is_empty() {
            secret = read_role("TLT_BRAMA_AGENT_ROLE", "the agent's signing secret")?;
        }
        let mut token = env_trimmed("BRAMA_TOKEN");
        if token.is_empty() {
            token = read_role("TLT_BRAMA_TOKEN_ROLE", "the Brama bearer")?;
        }
        let url = resolve_url()?;
        Self::new(&url, agent_id, secret, Some(token))
    }

    /// The identity headers jeden signs with: the canonical string is
    /// `"{agent_id}:{timestamp}:{body_hash}"`, keyed by the shared secret.
    pub(crate) fn auth_headers(&self, body: &str) -> Vec<(&'static str, String)> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0)
            .to_string();
        let body_hash = if body.is_empty() {
            String::new()
        } else {
            hex::encode(Sha256::digest(body.as_bytes()))
        };
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(self.secret.as_bytes())
            .expect("HMAC accepts a key of any length");
        mac.update(format!("{}:{timestamp}:{body_hash}", self.agent_id).as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        let mut headers = vec![
            ("content-type", "application/json".to_string()),
            ("x-agent-id", self.agent_id.clone()),
            ("x-agent-timestamp", timestamp),
            ("x-agent-body-sha256", body_hash),
            ("x-agent-signature", signature),
        ];
        if let Some(token) = &self.token {
            headers.push(("authorization", format!("Bearer {token}")));
        }
        headers
    }

    /// One non-streaming chat completion; returns the message content. No
    /// output budget is sent: how long an answer may be is the routed model's
    /// own limit, which Brama's route carries, not a count chosen here.
    pub fn chat(&self, model: &str, messages: &[Message]) -> Result<String> {
        match self.answer(model, messages)? {
            ChatAnswer::Content(content) => Ok(content),
            ChatAnswer::DoesNotFit(detail) => bail!("Brama answered HTTP 400: {detail}"),
        }
    }

    /// One chat completion that tells a prompt the routed model cannot take
    /// (Brama's `context_length_exceeded`) apart from every other failure, so
    /// a caller can ask about less instead of guessing a size up front.
    pub(crate) fn answer(&self, model: &str, messages: &[Message]) -> Result<ChatAnswer> {
        let body = serde_json::to_string(&serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": 0,
        }))?;
        let mut request = self.http.post(format!("{}/v1/chat/completions", self.url));
        for (name, value) in self.auth_headers(&body) {
            request = request.header(name, value);
        }
        let response = match request.body(body).send() {
            Ok(response) => response,
            Err(error) => bail!("Brama unreachable at {}: {error}", self.url),
        };
        let status = response.status();
        let text = response.text().unwrap_or_default();
        if status.as_u16() != 200 {
            let detail = text.trim();
            let detail = if detail.is_empty() {
                "(empty body)"
            } else {
                detail
            };
            let code = serde_json::from_str::<serde_json::Value>(detail)
                .ok()
                .and_then(|payload| payload.pointer("/error/code")?.as_str().map(str::to_string));
            if code.as_deref() == Some("context_length_exceeded") {
                return Ok(ChatAnswer::DoesNotFit(detail.to_string()));
            }
            bail!("Brama answered HTTP {}: {detail}", status.as_u16())
        }
        let content = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|payload| {
                payload
                    .get("choices")?
                    .get(0)?
                    .get("message")?
                    .get("content")?
                    .as_str()
                    .map(str::to_string)
            });
        match content {
            Some(content) => Ok(ChatAnswer::Content(content.trim().to_string())),
            None => bail!(
                "Brama response was not an OpenAI chat completion: {}",
                text.trim()
            ),
        }
    }
}

/// What Brama answered one chat completion.
pub(crate) enum ChatAnswer {
    /// The model's message content.
    Content(String),
    /// The prompt does not fit the routed model; Brama's answer body.
    DoesNotFit(String),
}

/// Compact single-label classification prompt.
pub fn build_prompt(aspect: &str, values: &[String], text: &str) -> Vec<Message> {
    let allowed = values.join(", ");
    vec![
        Message::new(
            "system",
            "You classify coding-agent session transcripts. Answer with \
             exactly one of the allowed values and nothing else."
                .to_string(),
        ),
        Message::new(
            "user",
            format!(
                "Aspect: {aspect}\n\
                 Allowed values: {allowed}\n\n\
                 Classify this session transcript into one allowed value.\n\n\
                 Transcript:\n{text}"
            ),
        ),
    ]
}

/// Map a raw model answer to an allowed value.
///
/// `Some((value, exact))` where `exact == false` means the value was recovered
/// from a longer answer. `None` is a parse failure: apply nothing.
pub fn parse_answer<S: AsRef<str>>(answer: &str, values: &[S]) -> Option<(String, bool)> {
    let cleaned = answer
        .trim()
        .trim_matches(|c| c == '"' || c == '`' || c == '\'' || c == ' ' || c == '.');
    for value in values {
        let value = value.as_ref();
        if cleaned.eq_ignore_ascii_case(value) || cleaned.to_lowercase() == value.to_lowercase() {
            return Some((value.to_string(), true));
        }
    }
    let haystack = answer.to_lowercase();
    let mut found = values
        .iter()
        .map(AsRef::as_ref)
        .filter(|value| haystack.contains(&value.to_lowercase()));
    match (found.next(), found.next()) {
        (Some(only), None) => Some((only.to_string(), false)),
        _ => None,
    }
}
