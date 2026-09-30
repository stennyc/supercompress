"""S-Phase 1：per-session 独立 store + carry_over 交接（@coder own）。

DoD（references/supercompress-session-redesign.md S-Phase 1）：
- 双 session 并行 0 泄漏：session A 的信息 0 泄漏进 B（独立文件 <sid>.jsonl）
- carry_over（确认轮转事件 old_session_id + carry_over=True）→ materialize(old→new, reopen_from=old)
  激活 store.py:137 死代码，P1 中间推理自动 stale 标记（recall 降权 ×0.5 即刻生效）
- 自动路径（无 carry_over）默认 OFF：旧 session 归档字节不动，新 session 全新空文件
- 同 session 冷重开（新实例、文件已存在）→ P1 内存标 stale，文件字节不动（byte-stable 保证）
- 幂等：同实例同 sid 重入 on_session_start → no-op
- hermes_home 落点：<hermes_home>/supercompress_store/<sid>.jsonl（profile 隔离白捡）

注意：JsonlStore 在 __init__ 只 makedirs 目录，session 文件在首次 append 时才落盘；
故「绑定到正确路径」用 store.path 断言，文件存在性断言放在 append 之后。

跑法（repo 根）：
  python -m pytest tests/test_session_store.py -v
"""
from __future__ import annotations

from pathlib import Path

from engine import SupercompressContextEngine
from sc.schema import Segment


def _mkengine(tmp_path: Path) -> SupercompressContextEngine:
    """独立隔离的 engine 实例（external_store_path → tmp，不碰全局 active.jsonl）。"""
    return SupercompressContextEngine(context_length=256000,
                                       external_store_path=str(tmp_path))


def _append_user(engine: SupercompressContextEngine, text: str) -> Segment:
    """直接往 engine 当前 store 落一条 user P0 段（不经宿主 turn 链，隔离数据面）。"""
    seg = Segment(kind="segment", tier="P0", provenance="user",
                  original_text=text, session_topic="", model_version="m1")
    return engine.store.append(seg)


def _append_p1(engine: SupercompressContextEngine, text: str) -> Segment:
    seg = Segment(kind="segment", tier="P1", provenance="model",
                  original_text=text, session_topic="", model_version="m1")
    return engine.store.append(seg)


def test_fresh_session_binds_own_path(tmp_path):
    e = _mkengine(tmp_path)
    e.on_session_start("s_alpha")  # 无 hermes_home → legacy 目录（= tmp_path）
    target = str(tmp_path / "s_alpha.jsonl")
    assert e.store.path == target, "store 必须绑定到 per-session 路径"
    assert e.store.line_count == 0, "fresh session 窗口为空"
    # 首条 append 后才落盘
    _append_user(e, "alpha marker")
    assert Path(target).exists(), "append 后 session 文件落盘"


def test_no_cross_session_leak(tmp_path):
    """DoD 核心：session A 的内容 0 泄漏进 session B。"""
    e_a = _mkengine(tmp_path)
    e_a.on_session_start("s_alpha")
    _append_user(e_a, "ALPHA-SECRET fact only in alpha")
    e_a.on_session_end("s_alpha", [])
    assert (tmp_path / "s_alpha.jsonl").exists()

    e_b = _mkengine(tmp_path)
    e_b.on_session_start("s_beta")
    assert e_b.store.path == str(tmp_path / "s_beta.jsonl")
    # B 的窗口里绝不能有 A 的段
    all_text = " ".join(s.original_text for s in e_b.store.by_hash.values())
    assert "ALPHA-SECRET" not in all_text, "跨 session 泄漏：A 的段进了 B 的窗口"
    assert e_b.store.line_count == 0, "B 应为独立空窗口"
    _append_user(e_b, "beta marker")
    b_text = " ".join(s.original_text for s in e_b.store.by_hash.values())
    assert "ALPHA-SECRET" not in b_text, "B append 后仍不得含 A 的内容"


def test_carry_over_materializes_old_to_new(tmp_path):
    """确认轮转（old_session_id + carry_over=True）→ materialize 交接。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_old")
    _append_user(e, "goal: v1 上线, 关掉重开记忆还在")
    p1 = _append_p1(e, "P1 中间推理")
    e.on_session_reset()  # /new 前 flush，保证 old 归档字节完整
    old_lines = (tmp_path / "s_old.jsonl").read_text(encoding="utf-8")

    e.on_session_start("s_new", old_session_id="s_old", carry_over=True)
    new_lines = (tmp_path / "s_new.jsonl").read_text(encoding="utf-8")
    # materialize = 字节稳定拷贝（逐行 verbatim）
    assert new_lines.strip() == old_lines.strip(), "carry_over 必须字节稳定物化 old→new"
    # P1 中间推理被 stale 标记（reopen_from 语义，recall ×0.5 降权）
    p1_replayed = e.store.get(p1.segment_id)
    assert p1_replayed is not None
    assert p1_replayed.stale is True, "carry_over 后旧 P1 应 stale（跨期重开）"
    assert p1_replayed.reopened_from == "s_old"


def test_auto_path_default_off_no_carry(tmp_path):
    """无 carry_over 的自动路径（old_session_id 存在但不带 carry）→ 新 session 全新，不动旧归档。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_old")
    _append_user(e, "goal: auto-path marker")
    old_bytes = (tmp_path / "s_old.jsonl").read_text(encoding="utf-8")

    e.on_session_start("s_new", old_session_id="s_old")  # 无 carry_over kwarg → 默认 OFF
    assert e.store.path == str(tmp_path / "s_new.jsonl")
    assert e.store.line_count == 0, "自动路径默认 OFF：new 不得携带 old 的段"
    # 旧归档字节一字不动
    assert (tmp_path / "s_old.jsonl").read_text(encoding="utf-8") == old_bytes


def test_cold_reopen_same_sid_marks_p1_stale(tmp_path):
    """同 sid 冷重开（新实例、文件已存在）→ P1 内存标 stale，文件字节不动。"""
    e1 = _mkengine(tmp_path)
    e1.on_session_start("s_x")
    p1 = _append_p1(e1, "cold P1")
    e1.on_session_end("s_x", [])  # flush P1 落盘
    file_bytes = (tmp_path / "s_x.jsonl").read_text(encoding="utf-8")
    assert p1.segment_id in e1.store.index

    e2 = _mkengine(tmp_path)
    e2.on_session_start("s_x")  # 冷重开
    assert e2.store.line_count == 1
    p1_re = e2.store.get(p1.segment_id)
    assert p1_re is not None
    assert p1_re.stale is True, "冷重开后旧 P1 应 stale"
    # byte-stable：重开不重写文件（stale 是内存标记，非行字段变更）
    assert (tmp_path / "s_x.jsonl").read_text(encoding="utf-8") == file_bytes


def test_on_session_start_idempotent_same_instance(tmp_path):
    """同实例同 sid 重入 → no-op（不重复物化、不重置窗口）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_idem")
    _append_user(e, "idem marker")
    assert e.store.line_count == 1
    e.on_session_start("s_idem")  # 幂等重入
    assert e._bound_session == "s_idem"
    assert e.store.line_count == 1, "重入不得清空已有段"


def test_on_session_reset_flushes_and_resets_session_state(tmp_path):
    """/new 交接原子点：flush 在途 P1/P2 + 重置实态字段 + 清 _force_evict_mode（reviewer 挂账 #3/#4）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_r")
    _append_p1(e, "in-flight P1")  # 进 lazy 批（未 flush）
    # 模拟旧 session 末 turn 升级态
    e._force_evict_mode = True
    e.last_prompt_tokens = 1234
    e.last_completion_tokens = 56
    e.compression_count = 2

    e.on_session_reset()

    # P1 已落盘（交接原子点：归档字节完整）
    assert (tmp_path / "s_r.jsonl").exists()
    assert e.store.line_count == 1, "reset 后 P1 已 durable 进 store 视图"
    # 实态字段对齐宿主基类重置集（agent/context_engine.py:187-205）
    assert e.last_prompt_tokens == 0
    assert e.last_completion_tokens == 0
    assert e.last_total_tokens == 0
    assert e.compression_count == 0
    # 会话级升级态与 on_session_start 对称清零（不得泄漏进新 session 首 turn）
    assert e._force_evict_mode is False, "reset 必须清 _force_evict_mode（reviewer 挂账 #4）"


def test_hermes_home_profiles_isolated(tmp_path):
    """hermes_home 按 profile 解析 → 不同 profile 各自 store 目录（隔离白捡，同 sid 不撞）。"""
    prof_a = tmp_path / "profiles" / "default"
    prof_b = tmp_path / "profiles" / "coder"

    e_a = _mkengine(tmp_path)
    e_a.on_session_start("s1", hermes_home=str(prof_a))
    e_b = _mkengine(tmp_path)
    e_b.on_session_start("s1", hermes_home=str(prof_b))

    # 同一 sid 绑定到两个不同 profile 目录（store.path 断言，未 append 前文件未落盘）
    assert e_a.store.path == str(prof_a / "supercompress_store" / "s1.jsonl")
    assert e_b.store.path == str(prof_b / "supercompress_store" / "s1.jsonl")

    _append_user(e_a, "DEFAULT-only fact")
    _append_user(e_b, "CODER-only fact")
    assert (prof_a / "supercompress_store" / "s1.jsonl").exists()
    assert (prof_b / "supercompress_store" / "s1.jsonl").exists()
    # 0 跨 profile 泄漏
    a_text = " ".join(s.original_text for s in e_a.store.by_hash.values())
    b_text = " ".join(s.original_text for s in e_b.store.by_hash.values())
    assert "DEFAULT-only" in a_text and "DEFAULT-only" not in b_text
    assert "CODER-only" in b_text and "CODER-only" not in a_text
