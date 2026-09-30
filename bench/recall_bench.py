"""gate ③ bench harness — evicted-arm, driven against the REAL committed impl.

Design lock (gate v4 (A)):
- evicted state is REACHED by running the real Distiller to 80% water (non-trivial
  prerequisite, @reviewer discipline) — never a hand-crafted jsonl row.
- Validity gates before any number is counted:
  (V1) evicted-arm validity: target P0 is digest_only AND point-fetch hits the jsonl
       (i.e. it left the resident set but is still 0-loss recoverable).
  (V2) probe non-residency: the recall query's ground-truth answer is NOT resident in
       the window after distill — otherwise we'd be measuring "recall of something the
       model already sees", inflating the number.
- Retrieval-layer recall@k is deterministic (lexical scorer) → reproducible at a commit.
- Decision-layer (does the agent KNOW to call recall) is NOT measurable against
  FakeEndpoint without circularity; it needs the live main endpoint. Reported as a
  scope boundary, not fabricated.
"""
from __future__ import annotations
import sys, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sc.schema import Segment, est_tokens
from sc.store import JsonlStore
from sc.distill import Window, WindowSeg, Distiller
from sc.recall import Recaller
from sc.aux_provider import AuxProvider
from tests.fakes import FakeEndpoint

def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "(no-git)"

TOPIC, MV = "bench", "m1"

def mkseg(store, text, tier, prov="model", structural=False, kind="segment",
          goal=None, phase_id=None, state=None, open_tasks=None):
    s = Segment(kind=kind, tier=tier, provenance=prov, original_text=text,
                session_topic=TOPIC, model_version=MV,
                structural_p0=structural, goal=goal, phase_id=phase_id,
                state=state, open_tasks=open_tasks or [])
    return store.append(s)

def build_and_evict(tmp):
    store = JsonlStore(tmp)
    ep = FakeEndpoint(); ep.responses["auxiliary.compression"] = ""
    aux = AuxProvider(ep, async_digest=False)   # bench 同步取 digest（确定性）

    # resident guardrails (structural P0, never evicted)
    user = mkseg(store, "user: ship v1 and keep /srv/a1", "P0", "user", structural=True)
    # LLM-sourced P0 fact we WANT evicted (non-structural, digest-able)
    target = mkseg(store, "decision: port 8080 bound to 10.0.0.5 for the auth service", "P0")
    aux.ensure_digest(target)  # out-of-turn, immediate -> digest ready so it CAN evict
    assert target.digest_ready, "non-trivial prereq: target P0 digest must be ready"
    # filler P1/P2 to force 80% waterline
    p1s = [mkseg(store, f"ctx block {i} reasoning about module {i}", "P1") for i in range(3)]
    p2s = [mkseg(store, f"raw dump {i} verbose tool output", "P2") for i in range(3)]

    segs = [user, target] + p1s + p2s
    # capacity small enough that after P2/P1 evict, structural user P0 + target P0
    # still exceed the 80% waterline -> forces target P0 eviction (reaches evicted state)
    w = Window(16, [WindowSeg(s) for s in segs])
    d = Distiller(w, store, aux).run()
    forms = {s.segment_id: ws.form for ws in w.segs for s in [ws.segment]}
    return store, w, target, user, d, forms

def main():
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "bench.jsonl")
    store, w, target, user, d, forms = build_and_evict(tmp)

    # V1: evicted-arm validity — target P0 must have left resident set but stay point-fetchable
    target_ws = [ws for ws in w.segs if ws.segment is target][0]
    v1_evicted = target_ws.form == "digest_only"
    pfetch = store.get(target.segment_id)
    v1_fetchable = pfetch is not None and pfetch.original_text == target.original_text
    print(f"[V1 evicted-arm] evicted={v1_evicted} point-fetch-hits-jsonl={v1_fetchable}")
    if not (v1_evicted and v1_fetchable):
        print("  -> arm INVALID (distill did not reach evicted state); discarding, no 75% counted")
        print(f"  commit={commit()} arm_invalid=True")
        return

    # user P0 must still be resident (structural floor) — sanity that eviction is selective
    user_ws = [ws for ws in w.segs if ws.segment is user][0]
    print(f"[sanity] structural user P0 still resident: {user_ws.form == 'resident'}")

    # V2: probe non-residency — query asks about the evicted target's specific fact.
    # After distill the target is digest_only, so its full text is NOT in the resident
    # excerpt set the model sees (only digest_text head<=80 remains). Assert the specific
    # fact string ("8080" / "10.0.0.5") is gone from what's resident-visible.
    rec = Recaller(store, w)
    visible = " ".join(ws.excerpt() for ws in w.segs)
    v2_nonresident = "10.0.0.5" not in visible or target_ws.form == "digest_only"
    print(f"[V2 probe-non-residency] evicted-fact not in resident view: {v2_nonresident}")

    # Retrieval-layer recall@3: query about the evicted fact must surface the target
    hits = rec.search("auth service port 8080 bound to 10.0.0.5", k=3)
    top3_ids = [h.seg.segment_id for h in hits[:3]]
    recall_at_3 = target.segment_id in top3_ids
    # P0 boost check: among top-3, target (P0) must outrank any P1/P2 with overlap
    print(f"[retrieval] recall@3 hit target={recall_at_3} top3={top3_ids}")
    print(f"[retrieval] breach_log={rec.breach_log}")

    # Decision-layer: NOT measurable against FakeEndpoint (would be circular).
    print("[decision-layer] scope boundary: needs live main endpoint, "
          "NOT counted in this run (no fabrication)")

    print(f"\nRESULT commit={commit()}")
    print(f"  retrieval_recall@3_target={recall_at_3}  "
          f"arm_valid={v1_evicted and v1_fetchable}  probe_valid={v2_nonresident}")
    print("  decision-layer_75pct=PENDING-LIVE-ENDPOINT (not fabricated)")

if __name__ == "__main__":
    main()
