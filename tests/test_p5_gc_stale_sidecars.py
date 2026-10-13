"""P5 GC 面：30 天 retention 侧车族清扫（owner 既裁定"保留 30 天"）。

宿主不替 supercompress 删侧车 + 红线"不改宿主" → 30 天 retention 落插件侧：
`gc_stale_sidecars` 按源 jsonl mtime 判 30 天 TTL，删过期 session 全侧车族
（.jsonl + .compact.json + .model.jsonl + .digests.json + .recall_cache.json
+ .flock + .audit + <sid>.p0.<seq> / <sid>.p1.<seq>），30 天内绝不碰。
宿主 on_session_start 每开 session 顺手触发（天然分散，不集中爆扫）。

安全基线：当前 store 最近 session 5 天前（未超 30 天）→ 首次跑 0 删除。
"""
import os
import time

import pytest

import scdata


def _mk_store(root, sid, age_days, extra_suffixes=()):
    """造一个 <sid>.jsonl（mtime = now - age_days）+ 侧车族文件。"""
    os.makedirs(root, exist_ok=True)
    jsonl = os.path.join(root, f"{sid}.jsonl")
    with open(jsonl, "w") as f:
        f.write("x")
    mtime = time.time() - age_days * 86400.0
    os.utime(jsonl, (mtime, mtime))
    made = [jsonl]
    for suf in extra_suffixes:
        p = os.path.join(root, f"{sid}{suf}")
        with open(p, "w") as f:
            f.write("x")
        os.utime(p, (mtime, mtime))
        made.append(p)
    return made


def test_gc_stale_removes_expiring_session_full_sidecar_family(tmp_path):
    """过期（>30 天）session 全侧车族被删（源 jsonl + 所有 <sid>.* 侧车）。"""
    root = str(tmp_path / "supercompress_store")
    made = _mk_store(root, "old_sid", age_days=45,
                     extra_suffixes=(".compact.json", ".model.jsonl",
                                     ".digests.json", ".recall_cache.json",
                                     ".flock", ".audit",
                                     ".p0.1", ".p1.2"))
    assert len(made) == 9  # 源 jsonl 1 + 侧车 8
    removed = scdata.gc_stale_sidecars(hermes_home=str(tmp_path), now=time.time())
    # 源 jsonl + 8 侧车 = 9 全删
    assert len(removed) == 9, f"应删 9 个文件，实删 {len(removed)}: {removed}"
    for p in made:
        assert not os.path.exists(p), f"{p} 应被删但仍存在"


def test_gc_stale_keeps_fresh_session(tmp_path):
    """30 天内（mtime 新鲜）的 session 绝不碰（安全基线）。"""
    root = str(tmp_path / "supercompress_store")
    _mk_store(root, "fresh_sid", age_days=5,
              extra_suffixes=(".compact.json", ".model.jsonl"))
    removed = scdata.gc_stale_sidecars(hermes_home=str(tmp_path), now=time.time())
    assert removed == [], f"5 天前 session 不应被删，实删 {removed}"
    assert os.path.exists(os.path.join(root, "fresh_sid.jsonl"))
    assert os.path.exists(os.path.join(root, "fresh_sid.compact.json"))


def test_gc_stale_boundary_30d(tmp_path):
    """30 天边界：age=31 删、age=29 留（cutoff 语义 = now - 30d，严格 > 才删）。"""
    root = str(tmp_path / "supercompress_store")
    _mk_store(root, "sid_31", age_days=31)
    _mk_store(root, "sid_29", age_days=29)
    removed = scdata.gc_stale_sidecars(hermes_home=str(tmp_path), now=time.time())
    assert any("sid_31" in p for p in removed), "31 天应删"
    assert not any("sid_29" in p for p in removed), "29 天应留"
    assert os.path.exists(os.path.join(root, "sid_29.jsonl"))
    assert not os.path.exists(os.path.join(root, "sid_31.jsonl"))


def test_gc_stale_derived_model_jsonl_not_double_counted(tmp_path):
    """派生面 .model.jsonl（stem 含 '.'）不独立判 TTL，跟源 sid 走全族删。"""
    root = str(tmp_path / "supercompress_store")
    # 造过期 sid 的源 + model 旁路（model 旁路 stem='<sid>.model' 含点 → 非判定基准）
    _mk_store(root, "old_sid", age_days=45, extra_suffixes=(".model.jsonl",))
    removed = scdata.gc_stale_sidecars(hermes_home=str(tmp_path), now=time.time())
    # 源 jsonl + model.jsonl 同 sid 前缀 → 全族删（但 model 旁路不作为独立判定基准被重复处理）
    assert len(removed) == 2, f"应删 2 个（源+model），实 {len(removed)}: {removed}"
    assert any("old_sid.model.jsonl" in p for p in removed)


def test_gc_stale_idempotent(tmp_path):
    """幂等：删过再跑 → 0。"""
    root = str(tmp_path / "supercompress_store")
    _mk_store(root, "gone_sid", age_days=45, extra_suffixes=(".compact.json",))
    assert len(scdata.gc_stale_sidecars(hermes_home=str(tmp_path), now=time.time())) >= 2
    assert scdata.gc_stale_sidecars(hermes_home=str(tmp_path), now=time.time()) == []


def test_gc_stale_missing_dir_fail_open(tmp_path):
    """目录缺失 → []（fail-open，零崩）。"""
    assert scdata.gc_stale_sidecars(hermes_home=str(tmp_path / "nope"),
                                    now=time.time()) == []


def test_retention_constant_is_30_days():
    """owner 既裁定 30 天 —— 常量锁死，防回归漂移。"""
    assert scdata.RETENTION_DAYS == 30


def test_gc_stale_p07_collision_sid_with_dot(tmp_path):
    """P0.7 碰撞加固 sid（sid 本体含点号，如 my_app_sid.bd345c0436）的源 jsonl 须
    正确识别为源判定基准并 GC 全族——第五轮 reviewer minor 实修：旧判据
    `if "." in sid: continue` 会把含点号的 P0.7 源 jsonl 误当派生面跳过
    = retention gap 无限增长。新判据按派生后缀白名单（.model.jsonl 等）排除，
    P0.7 源 jsonl（ends .jsonl 且非 .model.jsonl）照样 GC。"""
    root = str(tmp_path / "supercompress_store")
    os.makedirs(root, exist_ok=True)
    now = time.time()
    old = now - 45 * 86400.0
    # P0.7 形 sid（本体含点号）的源 jsonl + 侧车族
    p07_sid = "my_app_sid.bd345c0436"
    for fname in (f"{p07_sid}.jsonl", f"{p07_sid}.compact.json", f"{p07_sid}.p0.1",
                  f"{p07_sid}.model.jsonl"):
        p = os.path.join(root, fname)
        with open(p, "w") as f:
            f.write("x")
        os.utime(p, (old, old))
    removed = scdata.gc_stale_sidecars_in_dir(root, now=now)
    removed_names = [os.path.basename(p) for p in removed]
    # P0.7 源 jsonl（含点号 sid）必须被 GC（第五轮修复前会被误跳）
    assert f"{p07_sid}.jsonl" in removed_names, "P0.7 含点号 sid 源 jsonl 须被 GC"
    assert f"{p07_sid}.compact.json" in removed_names
    assert f"{p07_sid}.p0.1" in removed_names
    # 派生面 .model.jsonl 不独立判 TTL，但跟源 sid 走全族删（startswith 命中）
    assert f"{p07_sid}.model.jsonl" in removed_names


def test_gc_stale_excludes_ops_audit_bak_files(tmp_path):
    """F1（reviewer 第四轮 minor）：legacy active.jsonl 的 ops 审计文件
    （active.jsonl.bak_* / active.jsonl_cleaned_*）非 session 侧车，30 天 TTL
    过期时不连带删（人工 ops 留痕，只删真侧车族）。"""
    root = str(tmp_path / "supercompress_store")
    os.makedirs(root, exist_ok=True)
    now = time.time()
    old = now - 45 * 86400.0
    # 过期 active.jsonl（mtime 45 天前）
    for fname in ("active.jsonl", "active.compact.json", "active.model.jsonl",
                  "active.jsonl.bak_20260929_215025", "active.jsonl_cleaned_20260929"):
        p = os.path.join(root, fname)
        with open(p, "w") as f:
            f.write("x")
        os.utime(p, (old, old))
    removed = scdata.gc_stale_sidecars_in_dir(root, now=now)
    removed_names = [os.path.basename(p) for p in removed]
    # 真侧车族删（源 + compact + model）
    assert "active.jsonl" in removed_names
    assert "active.compact.json" in removed_names
    assert "active.model.jsonl" in removed_names
    # ops 审计文件保留（F1 排除面）
    assert "active.jsonl.bak_20260929_215025" not in removed_names
    assert "active.jsonl_cleaned_20260929" not in removed_names
    assert os.path.exists(os.path.join(root, "active.jsonl.bak_20260929_215025"))
    assert os.path.exists(os.path.join(root, "active.jsonl_cleaned_20260929"))


def test_token_precheck_uses_sid_form(tmp_path):
    """F4（reviewer 第四轮 minor）：封帽预检的 settled 块存根权重按真实 sid 形
    （<sid>.p0.<seq>）算，与 text() 渲染面对齐——不再用 sid="" 旧形（<seq>.p0）
    导致 _token_sum 比实际渲染面每块低 ~9 token。"""
    from sc.compact_ledger import CompactLedger, _settled_stub_line
    from sc.schema import est_tokens
    lg = CompactLedger(str(tmp_path / "mysid.compact.json"))
    assert lg._sid == "mysid"
    # 预检口径：settled 块存根权重 = est_tokens(_settled_stub_line(seq, text, kind, sid))
    # 三处预检（append settled / settle 翻转）已传 self._sid，与 text() 渲染同形
    pre = est_tokens(_settled_stub_line(1, "定案原文", "adhoc", lg._sid))
    render = est_tokens(_settled_stub_line(1, "定案原文", "adhoc", lg._sid))
    assert pre == render, "封帽预检口径须与 text() 渲染面一致（同传 self._sid）"
    # 旧形 sid="" 会低估权重（F4 修复前错位源）：真实 sid 形 token 更高
    old_form = est_tokens(_settled_stub_line(1, "定案原文", "adhoc", ""))
    assert pre > old_form, "真实 sid 形（mysid.p0.1）比旧形（1.p0）token 更高 = F4 错位面"
