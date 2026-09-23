//! Brama client: HMAC-signed OpenAI-compatible chat completions.
//!
//! Brama is Wisent's authenticated, provider-neutral LLM gateway. All LLM
//! inference goes through it — never direct provider keys.
//!
//! Auth contract mirrored from jeden (rust/model_router.rs::hmac_headers,
//! rust/control_plane/brama.rs, scripts/run-with-stado.sh):
//!
//! - POST {BRAMA_URL}/v1/chat/completions, OpenAI-compatible body;
//! - HMAC identity headers: x-agent-id, x-agent-timestamp (unix seconds),
//!   x-agent-body-sha256 (hex SHA-256 of the raw body, empty string for an
//!   empty body), x-agent-signature (hex HMAC-SHA256 of
//!   "{agent_id}:{timestamp}:{body_hash}" keyed by the shared secret);
//! - bearer Authorization when the gateway demands one (the fleet gateway does).
//!
//! Credential resolution mirrors jeden's launcher: environment first
//! (WISENT_APP_AGENT_AUTH_SECRET / BRAMA_TOKEN), then Skarbiec — the signing
//! secret is item `agent:wisent-app` field `value` (vault-first, stado
//! fallback), the bearer is item `jeden-model-router` field `token`. Secret
//! values are only ever held in memory — never printed, never logged.
//!
//! Endpoint resolution: BRAMA_URL, then JEDEN_BRAMA_URL, then the BRAMA_URL
//! line of jeden's own config (~/.jeden/.env — the same value jeden uses), then
//! Stado's service directory, which derives the address from where the gateway
//! is placed.
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::bail;
use crate::util::{home_dir, Error, Result};

mod default_stado_bin;
mod brama_client;

pub use default_stado_bin::*;
pub use brama_client::*;
