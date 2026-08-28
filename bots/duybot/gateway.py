"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

THE STARTER'S SHAPE (read this before you start editing `decide()`)
----------------------------------------------------------------------------
This starter FORWARDS ALMOST EVERYTHING AND DENIES NOTHING. That is not a
placeholder oversight — it is the honest zero-defence baseline you are
meant to beat: `bots/rookie` in the kit's own ladder does exactly the same
thing, and RULES.md's own words are "if you cannot beat Rookie you have a
bug, not a strategy." `decide()` below is structured as four named jobs —
ROUTE, ADMIT, AUTHORIZE, BUDGET — each with a one-line TODO naming what a
real implementation checks and why. None of the four currently rejects,
rewrites, or reroutes anything; they are seams, not solutions. Fill them in
using `agent/strategy.py` (routing/budget policy) and `agent/guardrails.py`
(the safety checks) — both already import cleanly from here.

ONE THING WORTH INTERNALISING BEFORE YOU WRITE YOUR FIRST REAL CHECK:
`verdict="deny"` costs the CALLER (your own team) **zero credits** —
CONTRACTS.md 4.1's charging table has exactly one $0 row, and it is this
one. Refusing to make a call you cannot justify is FREE. That makes
abstention a real strategy, not a luxury you can't afford: a `deny` you can
defend beats a `forward` you can't, every time a prosecutor is watching.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from bots.duybot.guardrails import scan_for_injected_instructions
from bots.duybot.strategy import (
    bounded_args,
    estimate_call_cost,
    future_reserve,
    round_allowance,
    safe_fields,
    successor_of,
)
from bots.duybot.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.

    Instance attributes below are this starter's ENTIRE per-duel memory —
    all currently unused by `decide()`'s naive body, but declared here
    (rather than invented ad hoc later) so the four TODO jobs below have
    somewhere obvious to keep state once you implement them. `agent/
    strategy.py` has working building blocks for exactly this (a budget
    pacer, a result cache, a replica-choice heuristic) — this starter does
    not wire them in for you; that wiring is the assignment.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)
        self._seen_anchors: dict[str, Any] = {}
        self._credits_authorised: int = 0
        self._denied_cmd_ids: set[str] = set()
        self._write_keys: set[str] = set()
        self._admitted_cards: dict[str, dict[str, Any]] = {}
        self._etags: dict[str, tuple[str, int, int]] = {}
        self._history_seen = 0
        self._round = 0
        self._round_start_credits = int(getattr(ctx, "credits", 100))
        self._round_estimated_spend = 0

    _A2A_SERVERS = frozenset({"curriculum-analyst", "citation-checker", "roster"})
    _A2A_SKILL = {
        ("curriculum-analyst", "which_days_cover"): "which_days_cover",
        ("citation-checker", "verify_source"): "verify_source",
        ("roster", "lookup_learner"): "role_of",
    }
    _WRITE_TOOLS = frozenset({
        ("progress", "record_mastery"),
        ("content", "flag_stale_slide"),
    })
    _HIGH_RISK_TEXT_KEYS = frozenset({
        "body", "content", "instruction", "instructions", "note",
        "peer_response", "result", "snippet",
    })
    _HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]{1,128}$")
    _FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_*.-]{1,64}$")
    _ROUTE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    @staticmethod
    def _json_shape_error(value: Any) -> str | None:
        """Validate bounded JSON-like command args without serialising them.

        A malformed nested object must be denied here, before it reaches a
        server or the arena's trace serializer. The traversal is cycle-safe,
        depth-bounded, and deterministic.
        """
        active: set[int] = set()
        nodes = 0

        def walk(item: Any, depth: int) -> str | None:
            nonlocal nodes
            nodes += 1
            if nodes > 512:
                return "args contain more than 512 values"
            if depth > 8:
                return "args nesting exceeds depth 8"
            if item is None or isinstance(item, (bool, int)):
                return None
            if isinstance(item, float):
                return None if math.isfinite(item) else "args contain a non-finite number"
            if isinstance(item, str):
                if len(item) > 8192:
                    return "args contain a string longer than 8192 characters"
                if "\x00" in item:
                    return "args contain a NUL character"
                return None
            if isinstance(item, dict):
                ident = id(item)
                if ident in active:
                    return "args contain a cyclic mapping"
                active.add(ident)
                try:
                    for key, nested in item.items():
                        if not isinstance(key, str) or not key or len(key) > 128:
                            return "args keys must be non-empty strings no longer than 128 characters"
                        if any(ord(ch) < 32 for ch in key):
                            return "args keys contain control characters"
                        error = walk(nested, depth + 1)
                        if error:
                            return error
                finally:
                    active.remove(ident)
                return None
            if isinstance(item, list):
                ident = id(item)
                if ident in active:
                    return "args contain a cyclic list"
                active.add(ident)
                try:
                    for nested in item:
                        error = walk(nested, depth + 1)
                        if error:
                            return error
                finally:
                    active.remove(ident)
                return None
            return f"args contain non-JSON value {type(item).__name__}"

        return walk(value, 0)

    @classmethod
    def _command_shape_error(cls, cmd: Any) -> str | None:
        """Return a safe reason when an arena command is structurally bad."""
        cmd_id = getattr(cmd, "cmd_id", None)
        kind = getattr(cmd, "kind", None)
        raw = getattr(cmd, "raw", None)
        server = getattr(cmd, "server", None)
        tool = getattr(cmd, "tool", None)
        args = getattr(cmd, "args", None)
        fields = getattr(cmd, "fields", None)
        headers = getattr(cmd, "headers", None)
        lease_id = getattr(cmd, "lease_id", None)
        call_index = getattr(cmd, "call_index", None)

        if not isinstance(cmd_id, str) or not cmd_id or len(cmd_id) > 256:
            return "cmd_id must be a bounded non-empty string"
        if kind not in COMMAND_KINDS:
            return "command kind is invalid"
        if not isinstance(raw, str) or len(raw) > 16384 or "\x00" in raw:
            return "raw command text is invalid"
        if not isinstance(server, str) or not cls._ROUTE_NAME_RE.fullmatch(server):
            return "server name is invalid"
        if not isinstance(tool, str) or not cls._ROUTE_NAME_RE.fullmatch(tool):
            return "tool name is invalid"
        if not isinstance(args, dict):
            return "args must be a mapping"
        args_error = cls._json_shape_error(args)
        if args_error:
            return args_error
        if not isinstance(fields, (tuple, list)) or isinstance(fields, (str, bytes)):
            return "fields must be a sequence of names"
        if any(not isinstance(field, str) or not cls._FIELD_NAME_RE.fullmatch(field) for field in fields):
            return "fields contain an invalid name"
        if not isinstance(headers, dict):
            return "headers must be a mapping"
        for key, value in headers.items():
            if not isinstance(key, str) or not cls._HEADER_NAME_RE.fullmatch(key):
                return "headers contain an invalid name"
            if not isinstance(value, str) or len(value) > 2048 or any(ch in value for ch in "\r\n\x00"):
                return "headers contain an invalid value"
        if lease_id is not None and (
            not isinstance(lease_id, str) or not lease_id or len(lease_id) > 256
            or any(ch in lease_id for ch in "\r\n\x00")
        ):
            return "lease_id is invalid"
        if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 0:
            return "call_index must be a non-negative integer"
        return None

    @staticmethod
    def _canonical_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
        """Case-insensitive header validation with conventional output names."""
        lowered: dict[str, Any] = {}
        for key, value in headers.items():
            lk = str(key).strip().lower()
            if not lk:
                raise ValueError("empty header name")
            if lk in lowered and lowered[lk] != value:
                raise ValueError(f"conflicting duplicate header {lk!r}")
            lowered[lk] = value
        names = {
            "mcp-replica": "Mcp-Replica",
            "if-match": "If-Match",
            "idempotency-key": "Idempotency-Key",
            "traceparent": "traceparent",
            "aud": "aud",
        }
        return {names.get(k, k): v for k, v in lowered.items()}

    @staticmethod
    def _header_view(headers: Mapping[str, Any]) -> dict[str, Any]:
        return {str(k).lower(): v for k, v in headers.items()}

    @staticmethod
    def _learner_identity(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        raw = value.strip()
        if raw.startswith("Learner:"):
            return "learner:" + raw.split(":", 1)[1]
        if raw.startswith("learner:"):
            return raw
        return None

    def _safe_telemetry_seen(self, cmd: Command) -> None:
        try:
            self._telemetry.decision_seen(cmd)
        except Exception:
            pass

    def _safe_telemetry_made(self, cmd: Command, decision: Decision) -> None:
        try:
            self._telemetry.decision_made(cmd, decision)
        except Exception:
            pass

    @staticmethod
    def _walk_mappings(value: Any):
        if isinstance(value, Mapping):
            yield value
            for nested in value.values():
                if isinstance(nested, (Mapping, list, tuple)):
                    yield from Gateway._walk_mappings(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from Gateway._walk_mappings(nested)

    def _sync_history(self) -> None:
        """Learn provenance from the arena-owned, read-only history shape.

        The contract permits wrappers around events/outcomes, so this parser
        intentionally looks through nested mappings rather than depending on
        one private arena representation.
        """
        history = tuple(getattr(self.ctx, "history", ()) or ())
        for item in history[self._history_seen:]:
            maps = list(self._walk_mappings(item))
            anchors: set[str] = set()
            etags: set[str] = set()
            for m in maps:
                p = m.get("p") if isinstance(m.get("p"), Mapping) else m
                args = p.get("args") if isinstance(p.get("args"), Mapping) else {}
                anchor = args.get("anchor") or p.get("anchor")
                if isinstance(anchor, str):
                    anchors.add(anchor)
                for a in p.get("anchors") or ():
                    if isinstance(a, str):
                        anchors.add(a)
                etag = p.get("etag")
                if isinstance(etag, str) and etag:
                    etags.add(etag)
            if len(etags) == 1:
                etag = next(iter(etags))
                for anchor in anchors:
                    self._etags[anchor] = (
                        etag,
                        int(getattr(self.ctx, "round", 0) or 0),
                        int(getattr(self.ctx, "call_index", 0) or 0),
                    )
        self._history_seen = len(history)

    def _sync_round(self) -> tuple[int, int]:
        rnd = int(getattr(self.ctx, "round", 0) or 0)
        credits = int(getattr(self.ctx, "credits", 0) or 0)
        if rnd != self._round:
            self._round = rnd
            self._round_start_credits = credits
            self._round_estimated_spend = 0
        actual_spend = max(0, self._round_start_credits - credits)
        return rnd, max(actual_spend, self._round_estimated_spend)

    def _deny_decision(self, cmd: Command, reason: str, *, quarantine: bool = False) -> Decision:
        cmd_id = getattr(cmd, "cmd_id", None)
        if isinstance(cmd_id, str):
            self._denied_cmd_ids.add(cmd_id)
        return Decision(verdict="deny", reason=reason, quarantine=quarantine)

    def _authorize_target(self, cmd: Command) -> str | None:
        expected = self._learner_identity(getattr(self.ctx, "act", None))
        candidates = [cmd.args.get("learner"), cmd.args.get("learner_id")]
        if cmd.server == "roster":
            candidates.extend((cmd.args.get("target"), cmd.args.get("subject"), cmd.args.get("anchor")))
        if cmd.server == "progress":
            candidates.append(cmd.args.get("anchor"))
        for raw in candidates:
            target = self._learner_identity(raw)
            if target and expected and target != expected:
                return f"target {target!r} is not owned by ctx.act {expected!r}"
        return None

    def _admit_a2a(self, cmd: Command, headers: Mapping[str, Any]) -> str | None:
        if cmd.server not in self._A2A_SERVERS:
            return None
        card = self._admitted_cards.get(cmd.server)
        if not card or not card.get("verified"):
            return "A2A peer card has not been admitted by the registry"
        skill = str(cmd.args.get("skill") or self._A2A_SKILL.get((cmd.server, cmd.tool), ""))
        if not skill or skill not in set(card.get("skills") or ()):
            return f"skill {skill or cmd.tool!r} is not declared by the admitted peer card"
        aud = self._header_view(headers).get("aud")
        allowed = {cmd.server, f"a2a:{cmd.server}", f"mcp:{cmd.server}"}
        if aud not in allowed:
            return f"delegation audience {aud!r} does not match {cmd.server!r}"
        if cmd.args.get("peer_unverified") is True:
            return "peer answer is explicitly marked unverified"
        return None

    def _write_headers(self, cmd: Command, headers: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        if (cmd.server, cmd.tool) not in self._WRITE_TOOLS:
            return headers, None
        scope = f"wiki.write:{cmd.server}"
        if scope not in set(getattr(self.ctx, "scopes", ()) or ()):
            return headers, f"missing required scope {scope!r}"

        anchor = cmd.args.get("anchor") or cmd.args.get("learner") or cmd.args.get("learner_id")
        if not isinstance(anchor, str) or not anchor:
            return headers, "write target has no resolvable anchor"
        known = self._etags.get(anchor)
        if known is None:
            return headers, "write has no fresh registry.provenance etag"
        etag, etag_round, etag_call = known
        current_round = int(getattr(self.ctx, "round", 0) or 0)
        if etag_round != current_round or cmd.call_index - etag_call > 3:
            return headers, "write provenance is stale"

        view = self._header_view(headers)
        supplied = view.get("if-match")
        if supplied is not None and supplied != etag:
            return headers, "If-Match does not match the latest provenance etag"
        headers["If-Match"] = etag

        idem = view.get("idempotency-key")
        if not isinstance(idem, str) or not idem:
            idem = f"{current_round}:{cmd.call_index}:{cmd.server}:{cmd.tool}:{anchor}"
        if idem in self._write_keys:
            return headers, "idempotency key was already used this duel"
        self._write_keys.add(idem)
        headers["Idempotency-Key"] = idem
        return headers, None

    def _decide(self, cmd: Command) -> Decision:
        shape_error = self._command_shape_error(cmd)
        if shape_error:
            return self._deny_decision(cmd, f"malformed command: {shape_error}")
        self._sync_history()
        rnd, spent = self._sync_round()

        successor = successor_of(cmd.server, cmd.tool)
        server, tool = successor or (cmd.server, cmd.tool)
        args = bounded_args(server, tool, cmd.args)
        fields = safe_fields(server, tool, tuple(cmd.fields))
        headers = self._canonical_headers(cmd.headers)
        changed = (server, tool) != (cmd.server, cmd.tool) or args != cmd.args or fields != tuple(cmd.fields)

        body_route = args.get("route") or args.get("_route") or args.get("replica")
        if body_route is not None:
            return self._deny_decision(cmd, "route/replica must be selected by a trusted header, not the request body")
        hview = self._header_view(headers)
        if hview.get("x-server-fingerprint") in {"unvouched", "invalid", "forged"}:
            return self._deny_decision(cmd, "server fingerprint is not registry-vouched")
        if hview.get("x-card-signature") in {"invalid", "forged", "unverified"}:
            return self._deny_decision(cmd, "agent card signature is invalid")

        if server not in self._A2A_SERVERS:
            replica = hview.get("mcp-replica", "w")
            if replica not in {"w", "c"}:
                return self._deny_decision(cmd, f"invalid MCP replica {replica!r}")
            if hview.get("mcp-replica") is None:
                headers["Mcp-Replica"] = "w"
                changed = True

        if tool == "get_frame":
            live_leases = set(getattr(self.ctx, "leases", ()) or ())
            if not cmd.lease_id or cmd.lease_id not in live_leases:
                return self._deny_decision(cmd, "slides.get_frame requires a live lease from a recent query")

        target_error = self._authorize_target(cmd)
        if target_error:
            return self._deny_decision(cmd, target_error)
        if (server, tool) not in self._WRITE_TOOLS and "wiki.read" not in set(getattr(self.ctx, "scopes", ()) or ()):
            return self._deny_decision(cmd, "read command requires wiki.read scope")

        a2a_error = self._admit_a2a(cmd, headers)
        if a2a_error:
            return self._deny_decision(cmd, a2a_error)

        suspicious_text = " ".join(
            str(v) for k, v in args.items() if str(k).lower() in self._HIGH_RISK_TEXT_KEYS
        )
        scan = scan_for_injected_instructions(suspicious_text)
        if scan.suspicious:
            return self._deny_decision(
                cmd,
                "instruction-shaped text was supplied as retrieved data",
                quarantine=True,
            )
        if isinstance(args.get("catalog"), str) and len(args["catalog"]) > 256:
            return self._deny_decision(cmd, "inflated catalog payload exceeds the admission limit")

        headers, write_error = self._write_headers(cmd, headers)
        if write_error:
            return self._deny_decision(cmd, write_error)
        if (server, tool) in self._WRITE_TOOLS:
            changed = True

        estimate = estimate_call_cost(server, tool, fields, args)
        allowance_left = max(0, round_allowance(rnd) - spent)
        credits = int(getattr(self.ctx, "credits", 0) or 0)
        if estimate > allowance_left:
            return self._deny_decision(
                cmd,
                f"estimated cost {estimate} exceeds round allowance remaining {allowance_left}",
            )
        if credits - estimate < future_reserve(rnd):
            return self._deny_decision(cmd, "call would consume credits reserved for higher-value later rounds")

        call = ToolCall(
            server=server,
            tool=tool,
            args=args,
            fields=fields,
            headers=headers,
            lease_id=cmd.lease_id,
            call_index=cmd.call_index,
        ) if _TOOLCALL_AVAILABLE else {
            "server": server, "tool": tool, "args": args, "fields": fields,
            "headers": headers, "lease_id": cmd.lease_id, "call_index": cmd.call_index,
        }
        self._round_estimated_spend += estimate
        self._credits_authorised += estimate
        verdict = "rewrite" if changed or headers != cmd.headers else "forward"
        return Decision(verdict=verdict, call=call)

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3).
        Raising anything, or returning a `Decision` `__post_init__` rejects,
        is treated by the arena exactly like an explicit deny PLUS a 2
        credit penalty PLUS a scored `integrity` event (CONTRACTS.md 4.1's
        charging table) — so the one thing this method must never do is
        blow up or wander off into I/O, no matter how tempting a "quick
        check" against something external looks. Everything you need to
        decide is already sitting in `cmd` and `self.ctx`.

        This starter forwards EVERYTHING it is handed, unmodified, and
        denies NOTHING — see the module docstring's "THE STARTER'S SHAPE".
        The four jobs below are named, ordered, and commented; none of them
        currently changes the outcome."""
        self._safe_telemetry_seen(cmd)
        try:
            decision = self._decide(cmd)
        except Exception as exc:
            # Fail closed without handing the opponent a scored integrity
            # event.  The exception type is enough for local telemetry; never
            # include arbitrary exception text that may contain private data.
            decision = self._deny_decision(cmd, f"gateway fail-closed: {type(exc).__name__}")
        self._safe_telemetry_made(cmd, decision)
        return decision

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Not called anywhere in this starter's `decide()` — a ready-made
        helper for when you fill in JOB 2 / JOB 3 above, so denying doesn't
        mean hand-building a `Decision` inline at every call site. Kept as
        a real method (not a stub) because the shape of a correct denial —
        no `call`, a non-empty `reason` — is exactly the thing worth
        getting right by construction rather than by convention."""
        decision = self._deny_decision(cmd, reason)
        self._safe_telemetry_made(cmd, decision)
        return decision

    def note_provenance(self, anchor: str, etag: str) -> None:
        """Optional arena callback after a successful provenance read."""
        if isinstance(anchor, str) and anchor and isinstance(etag, str) and etag:
            self._etags[anchor] = (
                etag,
                int(getattr(self.ctx, "round", 0) or 0),
                int(getattr(self.ctx, "call_index", 0) or 0),
            )

    # Operator uses the older callback name; accepting both costs nothing and
    # keeps the gateway compatible with either arena adapter.
    note_result = note_provenance

    def note_card(self, server: str, card: Mapping[str, Any]) -> None:
        """Record only an explicitly registry-verified peer card."""
        if not isinstance(server, str) or not isinstance(card, Mapping):
            return
        verified = card.get("verified") is True
        skills = tuple(sorted({s for s in card.get("skills", ()) if isinstance(s, str)}))
        self._admitted_cards[server] = {"verified": verified, "skills": skills}

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — routed, authorised, and budgeted ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    gw.note_card("curriculum-analyst", {"verified": True, "skills": ["which_days_cover"]})
    decisions = []
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        decisions.append(decision)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
    first = decisions[0]
    assert first.verdict == "rewrite"  # adds the trusted default replica header
    assert first.call is not None and first.call.headers.get("Mcp-Replica") == "w"
    assert all(d.verdict in DECISION_VERDICTS for d in decisions)
    for cmd, decision in zip(demo_commands, decisions):
        if decision.verdict == "deny":
            assert decision.call is None and decision.reason
            continue
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"]
        assert call_dict["tool"]

    print(f"\n=== Gateway.deny — the unused-by-default free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
