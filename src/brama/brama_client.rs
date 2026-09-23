use super::*;

impl BramaClient {
    pub fn new(url: &str, agent_id: String, secret: String, token: Option<String>) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
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

    pub fn from_env() -> Result<Self> {
        let agent_id = match env_trimmed("WISENT_APP_AGENT_ID") {
            value if value.is_empty() => "wisent-app".to_string(),
            value => value,
        };
        let mut secret = env_trimmed("WISENT_APP_AGENT_AUTH_SECRET");
        if secret.is_empty() {
            let item = match env_trimmed("TLT_BRAMA_AGENT_ITEM") {
                value if value.is_empty() => DEFAULT_AGENT_ITEM.to_string(),
                value => value,
            };
            secret = skarbiec_read(&item, "value");
        }
        if secret.is_empty() {
            bail!(
                "no Brama signing secret: set WISENT_APP_AGENT_AUTH_SECRET, or make \
                 Skarbiec item '{DEFAULT_AGENT_ITEM}' readable (vault-first via \
                 ~/.stado/bin/skarbiec, fallback 'stado credentials get')"
            )
        }
        let mut token = env_trimmed("BRAMA_TOKEN");
        if token.is_empty() {
            let item = match env_trimmed("TLT_BRAMA_TOKEN_ITEM") {
                value if value.is_empty() => DEFAULT_TOKEN_ITEM.to_string(),
                value => value,
            };
            token = skarbiec_read(&item, "token");
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

    /// One non-streaming chat completion; returns the message content.
    pub fn chat(&self, model: &str, messages: &[Message]) -> Result<String> {
        self.chat_limited(model, messages, ANSWER_MAX_TOKENS)
    }

    /// One non-streaming chat completion with a caller-chosen output budget,
    /// for answers that are a JSON document rather than one word.
    pub fn chat_limited(
        &self,
        model: &str,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<String> {
        let body = serde_json::to_string(&serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
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
            let detail = truncate_chars(&text, 300);
            let detail = detail.trim();
            let detail = if detail.is_empty() {
                "(empty body)"
            } else {
                detail
            };
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
            Some(content) => Ok(content.trim().to_string()),
            None => bail!(
                "Brama response was not an OpenAI chat completion: {}",
                truncate_chars(&text, 300)
            ),
        }
    }
}

/// The first `limit` characters, the way Python's `text[:limit]` slices.
pub fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit).collect()
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
