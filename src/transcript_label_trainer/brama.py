"""Brama client: HMAC-signed OpenAI-compatible chat completions.

Brama is Wisent's authenticated, provider-neutral LLM gateway. All LLM
inference goes through it — never direct provider keys.

Auth contract mirrored from jeden (rust/model_router.rs::hmac_headers,
rust/control_plane/brama.rs, scripts/run-with-stado.sh):

- POST {BRAMA_URL}/v1/chat/completions, OpenAI-compatible body;
- HMAC identity headers: x-agent-id, x-agent-timestamp (unix seconds),
  x-agent-body-sha256 (hex SHA-256 of the raw body, empty string for an
  empty body), x-agent-signature (hex HMAC-SHA256 of
  "{agent_id}:{timestamp}:{body_hash}" keyed by the shared secret);
- bearer Authorization when the gateway demands one (the fleet gateway does).

Credential resolution mirrors jeden's launcher: environment first
(WISENT_APP_AGENT_AUTH_SECRET / BRAMA_TOKEN), then Skarbiec — the signing
secret is item ``agent:wisent-app`` field ``value`` (vault-first, stado
fallback), the bearer is item ``jeden-model-router`` field ``token``. Secret
values are only ever held in memory — never printed, never logged.

Endpoint resolution: BRAMA_URL, then JEDEN_BRAMA_URL, then the BRAMA_URL line
of jeden's own config (~/.jeden/.env — the same value jeden uses), then Stado's
service directory, which derives the address from where the gateway is placed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import time
from pathlib import Path

import requests

DEFAULT_STADO_BIN = "stado"
DEFAULT_AGENT_ITEM = "agent:wisent-app"
DEFAULT_TOKEN_ITEM = "jeden-model-router"

# Being listed by /v1/models is not the same as being servable: that list is the
# public models.dev catalogue, several thousand ids wide, and on 2026-08-09 only
# 59 of 6244 came back with ``available: true`` for this agent. The previous
# default, ``302ai/claude-haiku-4-5``, was chosen off that list and answered 503
# ``direct '302ai' credential is unavailable`` on every call, because the fleet
# vault has never held a 302ai credential.
#
# This one is billed to an existing subscription rather than per-token credits,
# handles the mixed Polish/English transcripts, and was measured answering
# through this client. When the subscription is exhausted, the free local route
# is ``wisent-backend/chat/primary``. Override either way with --brama-model.
DEFAULT_MODEL = "codex/gpt-5.6-sol"

REQUEST_TIMEOUT = 120
STADO_TIMEOUT = 20
ANSWER_MAX_TOKENS = 16


class BramaError(Exception):
    """Raised when Brama cannot be reached, authenticated, or parsed."""


def _stado_url() -> str:
    """Where Stado says Brama is actually placed, or an empty string.

    ``stado service directory connect brama`` derives the address from the
    placement rather than from a name someone wrote down, and verifies that
    something answers there before reporting it.
    """
    stado = os.environ.get("TLT_STADO_BIN", DEFAULT_STADO_BIN)
    try:
        done = subprocess.run(
            [stado, "service", "directory", "connect", "--json", "brama"],
            capture_output=True,
            text=True,
            timeout=STADO_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if done.returncode:
        return ""
    try:
        return str(json.loads(done.stdout).get("url", "")).strip()
    except json.JSONDecodeError:
        return ""


def _resolve_url() -> str:
    for var in ("BRAMA_URL", "JEDEN_BRAMA_URL"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    jeden_env = Path.home() / ".jeden" / ".env"
    if jeden_env.is_file():
        for line in jeden_env.read_text(encoding="utf-8").splitlines():
            if line.startswith("BRAMA_URL="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    # There used to be a public default here, `https://brama.wisent.com`. That
    # hostname resolves to the company website, so every call that reached this
    # line got a 404 page from Vercel and a parse error blaming Brama. The fleet
    # keeps the real placement in Stado's service directory, which is also the
    # one answer that follows the gateway when it moves.
    url = _stado_url()
    if url:
        return url
    raise BramaError(
        "no Brama endpoint: set BRAMA_URL, or make `stado service directory "
        "connect brama` resolve (it reports where the gateway is placed)"
    )


def _skarbiec_read(item: str, field: str) -> str:
    """Read one field of one Skarbiec item; empty string when unavailable.

    Vault-first (the gateway verifies against the vault's current revision),
    then the managed `stado credentials get` path as fallback — the same order
    as jeden's run-with-stado.sh.
    """
    home = Path.home()
    vault = os.environ.get("SKARBIEC_VAULT_FILE", str(home / ".stado" / "skarbiec.vault.json"))
    skarbiec = os.environ.get("TLT_SKARBIEC_BIN", str(home / ".stado" / "bin" / "skarbiec"))
    if Path(skarbiec).is_file() and Path(vault).is_file():
        env = dict(os.environ, SKARBIEC_VAULT_FILE=vault)
        done = subprocess.run(
            [skarbiec, "get", item], capture_output=True, text=True, env=env
        )
        if done.returncode == 0:
            try:
                fields = json.loads(done.stdout).get("fields") or {}
            except json.JSONDecodeError:
                fields = {}
            value = str(fields.get(field, "")).strip()
            if value:
                return value
    stado = os.environ.get("TLT_STADO_BIN", DEFAULT_STADO_BIN)
    env = dict(
        os.environ,
        WC_SKARBIEC_CONSUMER=os.environ.get("TLT_SKARBIEC_CONSUMER", "local-operator"),
        WC_SKARBIEC_TOKEN_FILE=os.environ.get(
            "TLT_SKARBIEC_TOKEN_FILE", str(home / ".stado" / "local-operator-skarbiec-token")
        ),
    )
    done = subprocess.run(
        [stado, "credentials", "get", "--field", field, item],
        capture_output=True,
        text=True,
        env=env,
    )
    if done.returncode == 0:
        return done.stdout.strip()
    return ""


class BramaClient:
    def __init__(self, url: str, agent_id: str, secret: str, token: str | None):
        self.url = url.rstrip("/")
        self.agent_id = agent_id
        self._secret = secret
        self._token = token or None

    @classmethod
    def from_env(cls) -> "BramaClient":
        agent_id = os.environ.get("WISENT_APP_AGENT_ID", "wisent-app").strip() or "wisent-app"
        secret = os.environ.get("WISENT_APP_AGENT_AUTH_SECRET", "").strip()
        if not secret:
            secret = _skarbiec_read(
                os.environ.get("TLT_BRAMA_AGENT_ITEM", DEFAULT_AGENT_ITEM), "value"
            )
        if not secret:
            raise BramaError(
                "no Brama signing secret: set WISENT_APP_AGENT_AUTH_SECRET, or make "
                f"Skarbiec item '{DEFAULT_AGENT_ITEM}' readable (vault-first via "
                "~/.stado/bin/skarbiec, fallback 'stado credentials get')"
            )
        token = os.environ.get("BRAMA_TOKEN", "").strip()
        if not token:
            token = _skarbiec_read(
                os.environ.get("TLT_BRAMA_TOKEN_ITEM", DEFAULT_TOKEN_ITEM), "token"
            )
        return cls(_resolve_url(), agent_id, secret, token)

    def _auth_headers(self, body: str) -> dict:
        timestamp = str(int(time.time()))
        body_hash = hashlib.sha256(body.encode()).hexdigest() if body else ""
        signature = hmac.new(
            self._secret.encode(),
            f"{self.agent_id}:{timestamp}:{body_hash}".encode(),
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "content-type": "application/json",
            "x-agent-id": self.agent_id,
            "x-agent-timestamp": timestamp,
            "x-agent-body-sha256": body_hash,
            "x-agent-signature": signature,
        }
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    def chat(self, model: str, messages: list[dict]) -> str:
        """One non-streaming chat completion; returns the message content."""
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": ANSWER_MAX_TOKENS,
            }
        )
        try:
            response = requests.post(
                f"{self.url}/v1/chat/completions",
                headers=self._auth_headers(body),
                data=body,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BramaError(f"Brama unreachable at {self.url}: {exc}")
        if response.status_code != 200:
            detail = response.text[:300].strip()
            raise BramaError(
                f"Brama answered HTTP {response.status_code}: {detail or '(empty body)'}"
            )
        try:
            payload = response.json()
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (ValueError, KeyError, IndexError, TypeError):
            raise BramaError(
                f"Brama response was not an OpenAI chat completion: {response.text[:300]}"
            )


def build_prompt(aspect: str, values: list[str], text: str) -> list[dict]:
    """Compact single-label classification prompt."""
    allowed = ", ".join(values)
    return [
        {
            "role": "system",
            "content": (
                "You classify coding-agent session transcripts. Answer with "
                "exactly one of the allowed values and nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Aspect: {aspect}\n"
                f"Allowed values: {allowed}\n\n"
                "Classify this session transcript into one allowed value.\n\n"
                f"Transcript:\n{text}"
            ),
        },
    ]


def parse_answer(answer: str, values: list[str]) -> tuple[str | None, bool]:
    """Map a raw model answer to an allowed value.

    Returns (value, exact). exact=False means the value was recovered from a
    longer answer. (None, False) is a parse failure: apply nothing.
    """
    cleaned = answer.strip().strip('"`\' .')
    for value in values:
        if cleaned.lower() == value.lower():
            return value, True
    found = [value for value in values if value.lower() in answer.lower()]
    if len(found) == 1:
        return found[0], False
    return None, False
