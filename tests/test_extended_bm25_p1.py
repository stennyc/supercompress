"""precision P1 · extended-BM25 三招钉死面（bench/extended_bm25_bench.py 收口为 pytest 门禁）。

对应 plan §2 P1 DoD：
- ① 高信号 exact-match boost + ② 字段加权 + ③ 时间衰减接线面 → 任务精准相关段
  （AC 数字 / 文件路径 / 函数名）top-k 排名上移（bench 出 before/after 数字）；
- 确定性基座不破（G4）：同 query 两次全等（含 decay 启用面）；
- 零新依赖（铁律①）：sc/recall.py 纯标准库 import 面静态钉死。
"""
import re

from sc.recall import Recaller, Segment

from bench.extended_bm25_bench import (
    CTRL_Q, CTRL_SEGS, CTRL_T, DECAY_LAM, DECAY_Q, DECAY_SEGS, DECAY_T,
    after_topk, before_topk, determinism_selfcheck, run_all,
)


def test_bench_gates_pass():
    """三招门禁全过：F 族全上移 + decay 接线上移 + margin/对照不劣 + 确定性两遍全等。"""
    g = run_all()["gate"]
    assert g["G1_field_rescue"], "族 F 字段/高信号救援未全上移（before→after rank）"
    assert g["G2_decay_wired"], "族 C 时间衰减接线未生效（λ>0 后 target 应上移）"
    assert g["G3_margin_not_worse"], "族 M margin 面劣化（rank1 面被伤）"
    assert g["G4_ctrl_not_worse"], "对照钉死面劣化（无锚点普通段被三招误伤）"
    assert determinism_selfcheck(), "bench 非确定（两遍输出不等）"


def test_families_rank_numbers():
    """DoD 数字面钉死（before/after 具体排名，防语料漂移改变结论）：
    F 三族（path/func/numeric）before=3 → after=1；C 族 λ=0 rank3 → λ=0.5 rank1。"""
    r = run_all()
    for row in r["f_rows"]:
        assert row["before"] == 3 and row["after"] == 1, f"{row['case']} 排名漂移 {row}"
    d = r["decay"]
    assert d["lam0"] == 3 and d["lam_on"] == 1, f"decay 族排名漂移 {d}"
    for row in r["m_rows"]:
        assert row["before"] == 1 and row["after"] == 1, f"{row['case']} margin 面漂移 {row}"
    c = r["ctrl"]
    assert c["after"] <= c["before"], f"对照面劣化 {c}"


def test_decay_default_zero_pinned_face_unchanged():
    """招③ 默认 0 钉死面：Recaller 缺省构造 + 缺省 search ≡ 显式 decay_lambda=0。"""
    from sc.distill import Window, WindowSeg
    segs = CTRL_SEGS
    w = Window(2000, [WindowSeg(s) for s in segs])
    a = [(h.seg.seq, round(h.score, 9)) for h in Recaller(None, w).search(CTRL_Q, k=5)]
    b = [(h.seg.seq, round(h.score, 9)) for h in Recaller(None, w).search(CTRL_Q, k=5, decay_lambda=0.0)]
    c = [(h.seg.seq, round(h.score, 9))
         for h in Recaller(None, w, decay_lambda=0.0).search(CTRL_Q, k=5)]
    assert a == b == c, "缺省 decay_lambda=0 钉死面被扰动"
    assert Recaller(None, w).decay_lambda == 0.0, "实例默认 decay_lambda 应为 0.0"


def test_decay_positive_twice_identical():
    """确定性负例（DoD#2 扩展面）：decay_lambda>0 启用面下同 query 两次全等。"""
    from sc.distill import Window, WindowSeg
    segs = DECAY_SEGS
    w = Window(2000, [WindowSeg(s) for s in segs])
    a = [(h.seg.seq, round(h.score, 9)) for h in Recaller(None, w).search(DECAY_Q, k=3, decay_lambda=DECAY_LAM)]
    b = [(h.seg.seq, round(h.score, 9)) for h in Recaller(None, w).search(DECAY_Q, k=3, decay_lambda=DECAY_LAM)]
    assert a == b, "decay>0 启用面非确定（两次结果不等）"
    # 同分下越早 seq 越多被降：最新段（seq=2）得分不被降权面压过最早段
    assert a[0][0] == 2, f"λ={DECAY_LAM} 应最新段排前，实际 {a}"


def test_engine_wiring_present():
    """招③ 引擎接线面：engine.py 三处（默认常量 / 实例读取 / Recaller 构造传参）。"""
    from pathlib import Path
    eng = Path(__file__).resolve().parents[1] / "engine.py"
    src = eng.read_text(encoding="utf-8")
    assert "DEFAULT_RECALL_DECAY_LAMBDA = 0.0" in src, "engine 缺 decay 默认常量（钉死面 0）"
    assert 'self.recall_decay_lambda: float = _get("recall_decay_lambda", DEFAULT_RECALL_DECAY_LAMBDA)' in src
    assert "Recaller(self.store, window, decay_lambda=self.recall_decay_lambda)" in src, \
        "Recaller 构造未传 decay_lambda（招③ 未接线）"


def test_recall_stdlib_only_no_new_deps():
    """铁律①静态面：sc/recall.py import 全为标准库 + 仓内 sc 相对面（0 第三方、0 模型）。"""
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "sc" / "recall.py"
    imports = re.findall(r"^(?:import|from)\s+([\w.]+)", p.read_text(encoding="utf-8"), re.M)
    allowed = {"__future__", "math", "re"}    # 纯标准库面
    for mod in imports:
        head = mod.split(".")[0]
        assert head in allowed or head.startswith("sc") or head == "", \
            f"sc/recall.py 引入非标准库依赖 {mod!r}（铁律① 违）"
    assert "sc.recall" not in " ".join(imports), "自引用 import（异常）"
    # 0 新模型面：无 onnx/embedding/cross-encoder 类字样
    body = p.read_text(encoding="utf-8")
    for bad in ("onnx", "embedding", "transformers", "sentence_transformers", "openai"):
        assert bad not in body, f"sc/recall.py 出现新模型面字样 {bad!r}（铁律①② 违）"
