"""Where training runs and where lake data lives — resolved from Stado.

Stado owns the canonical compute-target registry, and that registry is the
authority on placement:

- ``targets[<this machine>].transcript_lake.root`` — the lake data root, i.e.
  the storage root this trainer reads labels and session text out of;
- ``targets[<host>].training`` — the host that trains label models, with
  ``models_dir`` as the artifact root on that host.

Resolution order, strongest first:

1. an explicit CLI flag (``--training-root`` / ``--storage-root``),
2. the environment (``TLT_HOME`` / ``LAKE_DATA``),
3. the Stado registry declarations above,
4. a local fallback under ``~``.

The fallback is an exception, not a default. Anything that stops Stado from
answering — the binary absent, the registry unreachable, no declaration for
this machine, training placed on a host that is not this one — degrades to the
local path and writes the reason into :attr:`Placement.detail`, which ``info``
prints. Resolution never raises: a broken control plane must not stop a local
run, it must only stop being invisible.

:attr:`Placement.source` reports the *weakest* layer any root depended on, so
one silent local fallback cannot hide behind a root that did resolve.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Registry keys read here, grounded in `stado registry pull` output. Neither
# block is modelled by stado's own Rust structs: both ride in the per-target
# `extra` map, which the registry loader round-trips verbatim.
TARGETS_KEY = "targets"
LAKE_KEY = "transcript_lake"
LAKE_ROOT_KEY = "root"
TRAINING_KEY = "training"
TRAINING_ENABLED_KEY = "enabled"
TRAINING_KINDS_KEY = "kinds"
TRAINING_ROOT_KEY = "models_dir"

# The training kind this trainer claims; a host may be declared for others.
TRAINING_KIND = "label-model"

# Historical defaults, kept only as the local exception path.
LOCAL_TRAINING_ROOT = Path.home() / ".transcript-label-trainer"
LOCAL_STORAGE_ROOT = Path.home() / ".transcript-lake"

STADO_BIN = "stado"
STADO_TIMEOUT_SECONDS = 20

# Weakest to strongest. `Placement.source` is the weakest one in play.
SOURCE_ORDER = ("local-fallback", "stado", "env", "flag")

_override: dict[str, Path] = {}
_cache: "Placement | None" = None


@dataclass(frozen=True)
class Placement:
    """The resolved answer to "where does this run, and out of what data"."""

    training_host: str | None
    training_root: Path
    storage_root: Path
    source: str
    detail: str


def set_override(training_root: str | None = None, storage_root: str | None = None) -> None:
    """Record explicit CLI roots — the strongest layer — and drop the cache."""
    global _cache
    if training_root:
        _override["training_root"] = Path(training_root).expanduser()
    if storage_root:
        _override["storage_root"] = Path(storage_root).expanduser()
    _cache = None


def resolve_placement() -> Placement:
    """Resolve placement once per process, per set of overrides."""
    global _cache
    if _cache is None:
        _cache = _resolve()
    return _cache


def _resolve() -> Placement:
    declared, why_not = _stado_declaration()

    training_root, training_source, training_note = _pick(
        "training_root", "TLT_HOME", declared.get("training_root"), LOCAL_TRAINING_ROOT, why_not
    )
    storage_root, storage_source, storage_note = _pick(
        "storage_root", "LAKE_DATA", declared.get("storage_root"), LOCAL_STORAGE_ROOT, why_not
    )
    return Placement(
        training_host=declared.get("training_host"),
        training_root=training_root,
        storage_root=storage_root,
        source=min((training_source, storage_source), key=SOURCE_ORDER.index),
        detail=f"training root {training_note}; storage root {storage_note}",
    )


def _pick(
    key: str, env_var: str, declared: Path | None, local: Path, why_not: str
) -> tuple[Path, str, str]:
    """One root through the four layers, plus the reason it landed there."""
    flagged = _override.get(key)
    if flagged is not None:
        return flagged, "flag", f"{flagged} from the command line"
    from_env = os.environ.get(env_var, "").strip()
    if from_env:
        path = Path(from_env).expanduser()
        return path, "env", f"{path} from ${env_var}"
    if declared is not None:
        return declared, "stado", f"{declared} declared in the Stado registry"
    reason = why_not or "the Stado registry declares no root for it"
    return local, "local-fallback", f"{local} — local fallback because {reason}"


def _stado_declaration() -> tuple[dict, str]:
    """The registry's placement declarations, or why there are none.

    Returns ``(declared, why_not)``. ``declared`` carries whichever of
    ``training_host`` / ``training_root`` / ``storage_root`` Stado answered
    for; ``why_not`` states why the rest are absent. Never raises.
    """
    registry, failure = _run_stado_json(["registry", "pull"])
    if registry is None:
        return {}, failure
    self_name, failure = _self_target()
    if self_name is None:
        return {}, failure

    targets = registry.get(TARGETS_KEY)
    if not isinstance(targets, list):
        return {}, f"the Stado registry carries no {TARGETS_KEY!r} list"
    by_name = {t.get("name"): t for t in targets if isinstance(t, dict)}

    declared: dict = {}
    reasons: list[str] = []

    storage_root = _declared_lake_root(by_name.get(self_name))
    if storage_root is None:
        reasons.append(f"Stado target {self_name!r} declares no {LAKE_KEY}.{LAKE_ROOT_KEY}")
    else:
        declared["storage_root"] = storage_root

    training_host, training_root = _declared_training(by_name)
    if training_host is None:
        reasons.append(
            f"no Stado target declares {TRAINING_KEY}.{TRAINING_KINDS_KEY} "
            f"containing {TRAINING_KIND!r}"
        )
    else:
        declared["training_host"] = training_host
        if training_host != self_name:
            reasons.append(
                f"Stado places {TRAINING_KIND} training on {training_host} at "
                f"{training_root}, and this machine is {self_name}"
            )
        elif training_root is None:
            reasons.append(
                f"Stado target {training_host!r} declares {TRAINING_KEY} without "
                f"{TRAINING_ROOT_KEY}"
            )
        else:
            declared["training_root"] = training_root

    return declared, "; ".join(reasons)


def _declared_lake_root(target: dict | None) -> Path | None:
    if not isinstance(target, dict):
        return None
    block = target.get(LAKE_KEY)
    if not isinstance(block, dict):
        return None
    root = block.get(LAKE_ROOT_KEY)
    return Path(str(root)).expanduser() if root else None


def _declared_training(by_name: dict[str, dict]) -> tuple[str | None, Path | None]:
    """The one host Stado places label-model training on."""
    for name, target in by_name.items():
        block = target.get(TRAINING_KEY)
        if not isinstance(block, dict) or not block.get(TRAINING_ENABLED_KEY):
            continue
        kinds = block.get(TRAINING_KINDS_KEY)
        if isinstance(kinds, list) and TRAINING_KIND not in kinds:
            continue
        root = block.get(TRAINING_ROOT_KEY)
        return name, Path(str(root)).expanduser() if root else None
    return None, None


def _self_target() -> tuple[str | None, str]:
    """Which registry target this machine is, per ``stado registry self``."""
    out, failure = _run_stado(["registry", "self"])
    if out is None:
        return None, failure
    name = out.split("\t", 1)[0].strip() if out.strip() else ""
    if not name:
        return None, "'stado registry self' did not name this machine"
    return name, ""


def _run_stado(args: list[str]) -> tuple[str | None, str]:
    """Run a stado subcommand. Returns (stdout, "") or (None, why not)."""
    command = [STADO_BIN, *args]
    printable = " ".join(command)
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=STADO_TIMEOUT_SECONDS
        )
    except FileNotFoundError:
        return None, f"the {STADO_BIN!r} CLI is not on PATH"
    except subprocess.TimeoutExpired:
        return None, f"{printable!r} timed out after {STADO_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return None, f"{printable!r} could not run: {exc}"
    if done.returncode != 0:
        detail = (done.stderr.strip() or done.stdout.strip() or "no output").splitlines()[0]
        return None, f"{printable!r} failed: {detail}"
    return done.stdout, ""


def _run_stado_json(args: list[str]) -> tuple[dict | None, str]:
    out, failure = _run_stado(args)
    if out is None:
        return None, failure
    printable = " ".join([STADO_BIN, *args])
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as exc:
        return None, f"{printable!r} returned unparseable JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, f"{printable!r} returned a {type(parsed).__name__}, not an object"
    return parsed, ""


def as_dict() -> dict:
    """The resolved placement, shaped for ``info --json``."""
    resolved = resolve_placement()
    return {
        "source": resolved.source,
        "training_host": resolved.training_host,
        "training_root": str(resolved.training_root),
        "storage_root": str(resolved.storage_root),
        "detail": resolved.detail,
    }
