"""agent/telemetry.py — thin wrappers around `GatewayContext.emit` (CONTRACTS.md 4.2).

READ THIS TWICE BEFORE YOU WRITE YOUR FIRST `self._telemetry.note(...)` CALL:

    EVERYTHING THIS FILE RECORDS GOES TO `own_telemetry` L1 EVENTS
    (CONTRACTS.md 5.2). `own_telemetry` IS:

      - NEVER SCORED.       No referee detector, no rubric class, no claim
                             gate anywhere in CONTRACTS.md section 6 reads
                             an `own_telemetry` event. Emitting one cannot
                             earn you credit, and — just as importantly —
                             a *missing* one cannot cost you any either.

      - NEVER SHOWN TO THE OPPONENT. CONTRACTS.md 5.4, verbatim: "The
        opponent's L1 events only, with `own_telemetry` removed." Your
        prosecutor never sees a single note you write here, and neither
        does the team across the table.

      - PRODUCER-STAMPED `student`, PERMANENTLY YOURS. CONTRACTS.md 5.1:
        "Only `arena` and `referee` may emit L1/L2. A `student`-produced L1
        event is an `integrity` violation" — `own_telemetry` is the one
        L1 event type a student producer is allowed to write at all, and
        it stays that way because nothing downstream ever treats it as an
        authoritative fact about what happened. The AUTHORITATIVE record
        of what your gateway did lives in the `command` / `decision` /
        `enforced` / `tool_call` / `tool_result` events the ARENA writes
        around your `Decision` (see agent/gateway.py's module docstring) —
        those are the ones a prosecutor reads and the ones that decide
        `enforcement_failure`. This file's events are never a substitute
        for that; they are commentary alongside it.

So why write any of this at all? Because "never scored, never shown" is
exactly what makes it safe to be HONEST in. What you record here is what
YOU can prove to YOURSELF, after the fact, about why your gateway did what
it did — which command looked suspicious and why you let it through anyway,
which round your budget got tight, which anchor you decided to trust
without re-verifying. None of that helps you in the moment it happens; all
of it helps you find your own bugs before a prosecutor finds them for you,
and it is the raw material for `eval/prosecute.py` development against
runs you own (FINAL-PLAN.md section 5.3) before you ever point a prosecutor
at someone else's trace.

Stdlib only. No network, no randomness, no wall-clock reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = [
    "Emits",
    "Telemetry",
    "RecordingGatewayContext",
]


@runtime_checkable
class Emits(Protocol):
    """The one method this whole file actually needs from a context object:
    `emit(name, **payload)` (CONTRACTS.md 4.2's `GatewayContext.emit`).
    Declared locally, rather than importing `agent.gateway.GatewayContext`,
    so this module has zero dependency on that file — `agent/gateway.py`
    imports `agent/telemetry.py`, not the other way around, and keeping
    this file leaf-level means editing your gateway can never break your
    own telemetry wrapper's imports."""

    def emit(self, name: str, **payload: Any) -> None: ...


class Telemetry:
    """A small, named-method wrapper around `ctx.emit`. Nothing here is
    magic — every method is a one-line call to `self._ctx.emit(...)` with a
    stable event name and a documented payload shape, so a later pass over
    your own runs (or a quick `grep` through them) finds "every place I
    decided something" by event NAME instead of by re-deriving it from
    scattered ad hoc `ctx.emit(...)` calls with slightly different keys
    each time. Add your own methods here as your gateway grows more
    reasoning worth remembering — the four below cover this starter's own
    call sites in `agent/gateway.py` and are a reasonable base, not a
    closed set."""

    def __init__(self, ctx: Emits) -> None:
        self._ctx = ctx

    def event(self, name: str, **payload: Any) -> None:
        """The escape hatch: any event name, any payload. Every other
        method on this class is a thin, named convenience over this one."""
        self._ctx.emit(name, **payload)

    def decision_seen(self, cmd: Any) -> None:
        """Call at the TOP of `decide()`, before any of your four jobs run
        — "here is what I was asked to rule on". `cmd` is duck-typed
        (`.cmd_id`/`.kind`/`.server`/`.tool`/`.call_index`) rather than
        type-hinted against `agent.gateway.Command`, for the same
        leaf-module reason `Emits` above is declared locally rather than
        imported."""
        self._ctx.emit(
            "gateway.command_seen",
            cmd_id=getattr(cmd, "cmd_id", None),
            kind=getattr(cmd, "kind", None),
            server=getattr(cmd, "server", None),
            tool=getattr(cmd, "tool", None),
            call_index=getattr(cmd, "call_index", None),
        )

    def decision_made(self, cmd: Any, decision: Any) -> None:
        """Call once `decide()` has settled on a `Decision` — records the
        verdict you returned and why, from YOUR side, independent of
        whatever the arena's own authoritative `decision`/`enforced` events
        end up saying. Comparing the two after a duel (this vs. the
        authoritative trace) is the fastest way to notice your own gateway
        doing something you did not intend."""
        self._ctx.emit(
            "gateway.decision",
            cmd_id=getattr(cmd, "cmd_id", None),
            verdict=getattr(decision, "verdict", None),
            reason=getattr(decision, "reason", None),
            quarantine=getattr(decision, "quarantine", None),
            note=getattr(decision, "note", None),
        )

    def budget_snapshot(self, *, round: int, credits_left: int, spent_this_round: int) -> None:
        """A cheap habit worth forming early: call this once per round (or
        once per `decide()`, if you want finer granularity) so a post-duel
        pass over your own telemetry can plot your spend curve without you
        having had to reconstruct it from the authoritative `tool_call`
        events after the fact."""
        self._ctx.emit(
            "gateway.budget_snapshot",
            round=round,
            credits_left=credits_left,
            spent_this_round=spent_this_round,
        )

    def note(self, message: str, **extra: Any) -> None:
        """Free-text commentary — "this smelled like a poisoned Note, I let
        it through anyway because I had no budget left to verify it", "the
        drift heuristic disagreed with itself here", whatever you actually
        thought at the time. `message` is truncated nowhere by this
        wrapper, but remember CONTRACTS.md 5.3: large payloads over 4 KB get
        blob-referenced by the real event pipeline, so keep it a note, not a
        dump of a whole tool result."""
        self._ctx.emit("gateway.note", message=message, **extra)


@dataclass
class RecordingGatewayContext:
    """A concrete, structurally-`GatewayContext`-shaped (CONTRACTS.md 4.2)
    object you can build in your own tests and demos WITHOUT a real duel —
    `emit(...)` just appends to `self.events` instead of writing to a real
    trace. This is what `agent/gateway.py`'s own `__main__` demo builds a
    `Gateway` against, and it is a reasonable starting point for your own
    local tests: the real arena hands your `Gateway.__init__` something
    that satisfies the SAME shape (CONTRACTS.md 4.2: "read-only,
    arena-provided" — the fields below are yours to READ, and this
    recording version is deliberately permissive about writes so a test
    can drive a duel forward round by round; the live arena's own
    implementation is under no such obligation).

    NOT frozen, on purpose: a realistic test needs to advance `round` /
    `call_index` / `credits` / `history` between calls the same way a real
    duel does — `agent/gateway.py`'s `GatewayContext` docstring covers why
    the real thing is a live view rather than a frozen snapshot."""

    act: str
    sub: str
    scopes: frozenset[str]
    credits: int
    round: int
    call_index: int
    leases: tuple[str, ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, name: str, **payload: Any) -> None:
        self.events.append({"name": name, "payload": dict(payload)})

    def events_named(self, name: str) -> list[dict[str, Any]]:
        """Convenience for tests/demos: every recorded payload dict for one
        event `name`, in emission order."""
        return [ev["payload"] for ev in self.events if ev["name"] == name]


if __name__ == "__main__":
    print("=== agent.telemetry: RecordingGatewayContext + Telemetry ===\n")

    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
    )
    assert isinstance(ctx, Emits)
    print(f"  fresh RecordingGatewayContext: events={ctx.events}")
    assert ctx.events == []

    tel = Telemetry(ctx)

    class _FakeCmd:
        cmd_id = "cmd:0001"
        kind = "mcp"
        server = "slides"
        tool = "get_frame"
        call_index = 0

    class _FakeDecision:
        verdict = "forward"
        reason = None
        quarantine = False
        note = None

    tel.decision_seen(_FakeCmd())
    tel.decision_made(_FakeCmd(), _FakeDecision())
    tel.budget_snapshot(round=1, credits_left=91, spent_this_round=9)
    tel.note("first round looked clean", suspicious_anchors=0)

    print(f"\n  {len(ctx.events)} events recorded:")
    for ev in ctx.events:
        print(f"    {ev['name']:26} {ev['payload']}")
    assert len(ctx.events) == 4
    assert [ev["name"] for ev in ctx.events] == [
        "gateway.command_seen",
        "gateway.decision",
        "gateway.budget_snapshot",
        "gateway.note",
    ]

    print("\n=== events_named() filters by event name ===")
    seen_events = ctx.events_named("gateway.command_seen")
    print(f"  events_named('gateway.command_seen') -> {seen_events}")
    assert seen_events == [{"cmd_id": "cmd:0001", "kind": "mcp", "server": "slides", "tool": "get_frame", "call_index": 0}]

    print("\n=== reminder: none of this is scored, none of it reaches the opponent ===")
    print("  (CONTRACTS.md 5.1: producer='student' is only ever legal for own_telemetry;")
    print("   CONTRACTS.md 5.4: the opponent's handed-over trace strips every own_telemetry event.)")

    print("\nAll agent/telemetry.py demos passed.")
