"""gate ③ decision-layer probe — 75% self-consistency recall @ evicted, LIVE endpoint only.

Measures the decision layer: given a resident view where the answer fact lives in an
EVICTED P0 (full text not visible), does the main model, when it has memory_recall,
actually (a) decide to recall and (b) recover the fact? This is the "does the model
know it should look" number — the real bottleneck behind "防失忆" (@researcher lane).

NO-FABRICATION GUARD (3 hard preconditions, mechanical — @reviewer ①②③):
  P1  commit anchor: 75% must run against a COMMIT that contains live_endpoint
      (checked via git log; refuse if the anchor commit isn't in this tree's history).
  P2  real model name: resolve_live_endpoint() already RuntimeError's on a placeholder
      model ('main' / unset); re-assert the resolved id is a real model name.
  P3  same endpoint: the decision-layer probe AND the digest generator must share the
      SAME LiveEndpoint object (self-consistency precondition) — built once, injected both.

A test double (FakeEndpoint) or a missing/placeholder endpoint => REFUSE, not a
fabricated 75% (circular-argument discipline, @reviewer-backed). 75% is model-specific:
valid only for the resolved main-model config; a config change re-derives it.

Run:
  python bench/decision_layer_probe.py            # precondition check only (no token spend)
  python bench/decision_layer_probe.py --run N=8  # fire N live decision probes, report rate
"""
from __future__ import annotations
import os, re, sys, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LIVE_ANCHOR = "ebdd2ed"          # commit that lands live_endpoint.py + resolve tests
TOPIC, MV = "bench", "m1"
PLACEHOLDER_MODELS = ("", "main", None)


def _head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _head_log() -> str:
    try:
        return subprocess.check_output(["git", "log", "--format=%h"], cwd=ROOT, text=True)
    except Exception:
        return ""


class ProbeRefused(RuntimeError):
    pass


def preconditions():
    """Return {anchor, model, endpoint} — raise ProbeRefused on any failure (no fabrication)."""
    from sc.live_endpoint import resolve_live_endpoint

    head, log = _head(), _head_log()
    # P1: live_endpoint commit must be in this tree's history
    if LIVE_ANCHOR not in log:
        raise ProbeRefused(
            f"P1 fail: live_endpoint anchor {LIVE_ANCHOR} not in history (HEAD={head[:8]})")
    # P2 + live endpoint resolution (refuses on missing endpoint OR placeholder model)
    try:
        ep = resolve_live_endpoint()
    except RuntimeError as e:
        raise ProbeRefused(f"P2/live fail — 75% not producible: {e}")
    if ep.model in PLACEHOLDER_MODELS:
        raise ProbeRefused(f"P2 fail: model id placeholder {ep.model!r} — cannot attribute 75%")

    return {"anchor": head, "model": ep.model, "endpoint": ep}


# ── reach evicted state by driving the REAL Distiller (non-trivial prerequisite) ──
def build_evicted(tmp, ep):
    from sc.schema import Segment
    from sc.store import JsonlStore
    from sc.distill import Window, WindowSeg, Distiller
    from sc.aux_provider import AuxProvider

    store = JsonlStore(tmp)
    aux = AuxProvider(ep, max_concurrency=1)   # P3: SAME endpoint object -> digest + probe
    user = store.append(Segment(tier="P0", provenance="user",
        original_text="user: ship v1 and keep /srv/a1", session_topic=TOPIC,
        model_version=MV, structural_p0=True))
    target = store.append(Segment(tier="P0", provenance="model",
        original_text="decision: auth service binds port 8080 to 10.0.0.5",
        session_topic=TOPIC, model_version=MV, structural_p0=False))
    aux.ensure_digest(target)
    assert target.digest_ready, "non-trivial prereq: target P0 digest must be ready"
    p1s = [store.append(Segment(tier="P1", provenance="model",
        original_text=f"ctx block {i} reasoning about module {i}",
        session_topic=TOPIC, model_version=MV)) for i in range(3)]
    p2s = [store.append(Segment(tier="P2", provenance="model",
        original_text=f"raw dump {i} verbose tool output",
        session_topic=TOPIC, model_version=MV)) for i in range(3)]

    w = Window(16, [WindowSeg(s) for s in [user, target] + p1s + p2s])
    Distiller(w, store, aux).run()
    target_ws = [ws for ws in w.segs if ws.segment is target][0]
    valid = target_ws.form == "digest_only" and store.get(target.segment_id) is not None
    return store, w, target, aux, valid


def decision_probe(ep, store, window, question, target):
    """One decision-layer instance. Returns (decided_to_recall, recovered_fact, raw)."""
    from sc.recall import Recaller
    view = "\n".join(f"- [{ws.segment.tier}] {ws.excerpt()[:80]}" for ws in window.segs)
    prompt = (
        "You are an agent with a memory_recall(query) tool that retrieves facts "
        f"evicted from your window. Question: {question}\n"
        f"Visible window context:\n{view}\n"
        "If the answer is NOT in the visible context, reply exactly: RECALL <query>. "
        "If you can answer without recalling, reply exactly: ANSWER <answer>."
    )
    raw = ep.call("decision_probe", prompt).strip()
    decided = raw.upper().startswith("RECALL")
    recovered = False
    if decided:
        q = re.sub(r"^RECALL\s*", "", raw, flags=re.I)
        hits = Recaller(store, window).search(q, k=3)
        recovered = any(h.seg.segment_id == target.segment_id for h in hits)
    return decided, recovered, raw


def main(argv):
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "bench.jsonl")

    try:
        pre = preconditions()
    except ProbeRefused as e:
        print(f"REFUSED (no-fabrication guard fired):\n  {e}")
        print(f"  HEAD={_head()[:8]}  anchor_in_history={LIVE_ANCHOR in _head_log()}")
        print("  75% NOT emitted — a 75% without live endpoint + real model + commit anchor "
              "is a fabricated number. CORRECT refusal.")
        return

    m = next((re.match(r"--run\s+(\d+)", a) for a in argv if a.startswith("--run")), None)
    n = int(m.group(1)) if m else 0
    if n <= 0:
        print(f"preconditions OK: anchor={pre['anchor'][:8]} model={pre['model']}")
        print(f"  live endpoint resolved -> {pre['endpoint'].base_url} ({pre['model']})")
        print("  preconditions-only mode (no token spend). Run with --run N to fire the probes.")
        return

    # full run: N decision probes, commit-anchored, model-specific
    store, window, target, aux, valid = build_evicted(tmp, pre["endpoint"])
    ep = pre["endpoint"]
    if not valid:
        print("  evicted-arm INVALID this run — 75% NOT counted (arm-validity gate).")
        return

    question = "What IP and port does the auth service bind to?"
    decided = recovered = 0
    for _ in range(n):
        d, r, _raw = decision_probe(ep, store, window, question, target)
        decided += d
        recovered += r
    rate = recovered / n
    print(f"\nRESULT @ {pre['anchor'][:8]}  model={pre['model']}  (model-specific, re-derive on config change)")
    print(f"  evicted-arm valid={valid}  n={n}")
    print(f"  decided-to-recall = {decided}/{n}   (decision rate)")
    print(f"  recovered fact@3  = {recovered}/{n}  = {rate:.0%}   <- KPI #3 decision-layer recall")
    print(f"  GATE 75%: {'PASS' if rate >= 0.75 else 'NOT MET'} @ evicted, self-consistency precondition holds "
          f"(probe+digest share one LiveEndpoint).")


if __name__ == "__main__":
    main(sys.argv[1:])
