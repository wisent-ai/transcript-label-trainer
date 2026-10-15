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
//! Credential resolution: environment first (WISENT_APP_AGENT_AUTH_SECRET /
//! BRAMA_TOKEN), then the vault role each is declared under —
//! `TLT_BRAMA_AGENT_ROLE` and `TLT_BRAMA_TOKEN_ROLE`, each `ROLE#FIELD`, read
//! with `stado credentials get --role`. No item and no agent is named in
//! code. Secret values are only ever held in memory — never printed, never
//! logged. A variable that is set but empty is refused by name, never read
//! as unset.
//!
//! Endpoint resolution: BRAMA_URL when the caller names it, otherwise Stado's
//! service directory, which derives the address from where the gateway is
//! placed; when Stado cannot place Brama the refusal carries why (stado not
//! run, Stado's own reason, a reply that is not JSON or names no url).
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::bail;
use crate::util::{Error, Result};

mod allowance;
mod default_stado_bin;
mod brama_client;

pub use allowance::allowance;
pub use default_stado_bin::*;
pub use brama_client::*;
