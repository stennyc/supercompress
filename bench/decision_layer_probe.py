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

Re-run after a model/config change (option ③, advisory — 不 block build):
  LIVE_ENDPOINT_MODEL=<new-model-name> && .venv/Scripts/python.exe bench/decision_layer_probe.py --run 8
  (resolve 链: LIVE_ENDPOINT_* → OPENAI_* → RuntimeError; 占位模型名拒出数; 网关不可达 → REFUSED 不产出。)

Stamp contract — bench/.last_75_stamp (单行 JSON, 与出数同 commit, 显式 git add):
  {"head": "<7位短id>", "model": "<真模型名>", "n": <N>, "pass": <k>}
  双锚 = head + model (75% 是 model-scoped 静态门, 换档双锚都变、整发重跑)。
Preflight 断言三态 (③ 只读, advisory, 失配动作 = WARN + 打上面重跑命令, 绝不 block):
  ① 双锚匹配            → PASS (可引用该 75%)
  ② head 或 model 失配  → WARN + 拒引用 (旧数对新代码/新模型档无效, 打重跑命令)
  ③ 文件缺失/不可读     → WARN "本 checkout 无 75% 出数记录, 引用前先跑探针" (不是 vacuous pass)
"""
from __future__ import annotations
import os, re, sys, subprocess, tempfile, time, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LIVE_ANCHOR = "ebdd2ed"          # commit that lands live_endpoint.py + resolve tests
TOPIC, MV = "bench", "m1"
PLACEHOLDER_MODELS = ("", "main", None)
# the specific fact the question asks about. For a decision-layer probe to be
# meaningful it must be GENUINELY HIDDEN from the model's view after eviction;
# if the head-80 digest still shows it, the arm is INVALID (fact not actually
# evicted from the view) and 75% is refused — no forced number.
TARGET_FACT = "10.0.0.5:8080"


def _view(window, target_id=None) -> str:
    """EXACT view fed to the model = per-seg tier + excerpt capped at the digest
    (head-80 in the reference impl). A digest_only segment shows its DIGEST HEAD,
    never its full original text — that gap is the whole point of the probe.
    Build in ONE place so the 'hidden' check and the model-view cannot drift
    apart (drift = fabricated 75%)."""
    return "\n".join(f"- [{ws.segment.tier}] {ws.excerpt()[:80]}" for ws in window.segs)


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
    """Reach evicted state via the REAL Distiller, with the target fact GENUINELY hidden.

    The reference impl's digest is `original_text[:80]` (a head excerpt), so to make the
    fact truly evicted-from-view we put it at the TAIL of a >80-char source. Then:
      - digest (head 80) does NOT contain TARGET_FACT  -> model must RECALL to get it
      - full source (in jsonl, retrievable) DOES contain it -> 0-loss holds
    """
    from sc.schema import Segment
    from sc.store import JsonlStore
    from sc.distill import Window, WindowSeg, Distiller
    from sc.aux_provider import AuxProvider

    store = JsonlStore(tmp)
    aux = AuxProvider(ep, max_concurrency=1)   # P3: SAME endpoint object -> digest + probe
    user = store.append(Segment(tier="P0", provenance="user",
        original_text="user: ship v1 and keep /srv/a1", session_topic=TOPIC,
        model_version=MV, structural_p0=True))
    # long source; fact buried past the 80-char digest head, as the LITERAL "10.0.0.5:8080"
    long_src = ("decision log: the auth service configuration was reviewed in detail "
                "and after the incident postmortem the team decided that the binding "
                + TARGET_FACT + " is authoritative and must be kept in the runbook "
                "for the whole quarter before any further deployment changes.")
    target = store.append(Segment(tier="P0", provenance="model",
        original_text=long_src, session_topic=TOPIC, model_version=MV, structural_p0=False))
    # bounded transient-retry on digest generation (@reviewer's structural finding:
    # Sub2API intermittently force-closes cold connections, so the FIRST ensure_digest
    # in a fresh process can hit a reset; AuxProvider._generate swallows it (transient=True)
    # and digest_ready stays False. Retry while transient; stop on success or hard fail.
    for _attempt in range(3):
        aux.ensure_digest(target)
        if target.digest_ready:
            break
        if not aux.transient:
            break                      # not transient -> retrying won't help
        time.sleep(2.0)
    if not target.digest_ready:
        raise RuntimeError(
            "digest 生成 3 次重试后仍未 ready（网关瞬态持续）——本发 abort、不写 stamp；"
            "网关稳了重跑即可（REFUSED 正确行为，非故障）")
    # HIDDEN INVARIANT: the digest head the model sees must NOT reveal the fact
    assert TARGET_FACT not in (target.digest_text or ""), \
        f"arm INVALID: digest head still shows {TARGET_FACT!r} — fact not actually hidden"
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


def _call_retry(ep, task, prompt, tries=3, delay=2.0):
    """Bounded retry for TRANSIENT gateway errors only (@reviewer's structural finding:
    Sub2API intermittently force-closes cold connections -> ConnectionResetError/TimeoutError).

    - HTTP 5xx / 429 -> transient, sleep + retry;
    - HTTP 4xx (auth/param) -> FATAL, re-raise immediately (no retry, no silent 'miss');
    - OSError family (ConnectionReset / Timeout / URLError) -> transient, sleep + retry;
    - all retries exhausted -> raise last, caller aborts the run (stamp NOT written).
    This is bench-face only: zero production impact (LiveEndpoint/aux_provider unchanged)."""
    import urllib.error
    last = None
    for _ in range(tries):
        try:
            return ep.call(task, prompt)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                last = e
                time.sleep(delay)
                continue
            raise                     # 4xx = not transient, do not retry
        except OSError as e:         # ConnectionReset / Timeout / URLError (transient)
            last = e
            time.sleep(delay)
    raise last


def decision_probe(ep, store, window, question, target):
    """One decision-layer instance. Returns (decided_to_recall, recovered_fact, raw)."""
    from sc.recall import Recaller
    view = _view(window)   # digest heads for all segs incl. target; full text only via recall
    prompt = (
        "You are an agent with a memory_recall(query) tool that retrieves facts "
        f"evicted from your window. Question: {question}\n"
        f"Visible window context:\n{view}\n"
        "If the answer is NOT in the visible context, reply exactly: RECALL <query>. "
        "If you can answer without recalling, reply exactly: ANSWER <answer>."
    )
    raw = _call_retry(ep, "decision_probe", prompt).strip()
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

    # accept both "--run 8" (two tokens) and "--run8" (one token)
    n = 0
    for i, a in enumerate(argv):
        m1 = re.match(r"^--run(\d+)$", a)
        if m1:
            n = int(m1.group(1)); break
        if a == "--run" and i + 1 < len(argv):
            n = int(argv[i + 1]); break
    if n <= 0:
        print(f"preconditions OK: anchor={pre['anchor'][:8]} model={pre['model']}")
        print(f"  live endpoint resolved -> {pre['endpoint'].base_url} ({pre['model']})")
        print("  preconditions-only mode (no token spend). Run with --run N to fire the probes.")
        return

    # full run: N decision probes, commit-anchored, model-specific
    try:
        store, window, target, aux, valid = build_evicted(tmp, pre["endpoint"])
    except RuntimeError as e:
        print(f"ABORTED — {e}")
        print(f"  75% NOT emitted, stamp NOT written (no-fabrication discipline holds).")
        return
    ep = pre["endpoint"]
    # hard invariant: the fact must be GENUINELY hidden from the model's view, else
    # the whole 75% is measuring 'model reads what's in front of it', not recall.
    view = _view(window)
    if TARGET_FACT in view:
        print(f"  arm INVALID: target fact {TARGET_FACT!r} still visible in the "
              f"digest-head view (reference digest = original_text[:80]). "
              f"Fact not actually evicted — 75% NOT counted, refusing to force a number.")
        return
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

    # ③: persist the model-specific stamp (HEAD + model double-anchored, same chain as the number).
    # ONLY reachable on a VALID full run — REFUSED / arm-invalid paths return before this,
    # so a stamp can never be written for a number that was never actually produced.
    import json
    stamp_path = os.path.join(ROOT, "bench", ".last_75_stamp")
    with open(stamp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"head": pre["anchor"], "model": pre["model"],
                            "n": n, "pass": recovered}) + "\n")
    print(f"  stamp written -> bench/.last_75_stamp (head={pre['anchor'][:8]}, model={pre['model']})")


if __name__ == "__main__":
    main(sys.argv[1:])
