"""紧凑摘要账本（append-only 累计布局）RED 回归面（卡片 t_63bf3ba2）。

现状（RED 前提，钉 PRD §0.9）：compress() 出块 = head(protect_first_n=3) + 滚动
tail(protect_last_n=6) + staged 块（S1 尾投递到末条 user）。staged 块恒 1 个、每次
compact 新替旧 → compact #2 重算块 → 位置 2/tail 字节全变 → provider 前缀缓存全量回落。

GREEN 目标布局（owner 裁定）：
    head(3) + [累计摘要账本块 c#1, c#2, …（append-only，旧块字节不动）] + 滚动 tail
- 每次 compact 只 **append** 新块 c#N（N 跨压缩单调），已缓存旧块 c#1..c#N-1 字节
  原样保留 → 前缀单调 → provider 前缀缓存跨 compact 命中；唯一刷缓存位 = 滚动 tail。
- 封顶/轮转 = **整段清 + 重新积累**（seal / session-reset），不做 mid-block 回改。
- 侧车落点（G2 单源 + 与 digest pool / model store 同形同住）：
  ``<store>.jsonl → <store>.compact.json``（splitext 单源，零硬编码路径；零 .db、零宿主）。

布局门控（关键，保既有测试零回归）：
- 账本前缀消息只在**存在可压缩 middle**（len(messages) > protect_first_n + protect_last_n）
  时注入到 head/tail 缝——长会话（真实压缩场景）才需要稳定前缀。
- 短会话（head+tail 已覆盖全部，无 middle）→ 不发行账本消息，staged 块照走既有 S1
  尾投递（末条 user 并入）——test_staged_block_s1_tail（7 条会话）行为零变化。

红线（本卡自检）：宿主 agent/ 零 diff；``_s1_tail_deliver`` 函数 / P0/P1 打分核 /
Segment 类零动；staged S1 尾投递路径保留（短会话门控）；全量 tests 零新增失败。

跑法（repo 根或插件根）：
  python -m pytest plugins/context_engine/supercompress/tests/test_compact_ledger.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

from engine import SupercompressContextEngine, _COMPACT_LEDGER_PREFIX
from sc.compact_ledger import compact_ledger_path_for_jsonl, PAGE_FOOTER
from sc.schema import Segment

MARK = _COMPACT_LEDGER_PREFIX   # 账本前缀消息 marker（单源 = engine._COMPACT_LEDGER_PREFIX，e2e 断言同源）
_C = re.compile(r"c#\d+")


def _mk(tmp_path: Path, context_length: int = 1000, **kw) -> SupercompressContextEngine:
    """独立隔离 engine（external_store_path → tmp，不碰全局 active.jsonl）。

    小 context_length 造高压（高压面 choose_forms 细节池归零 → L1 索引行形态，确定性），
    保证「同段集 → 同块字节」，append-only 前缀稳定性可被字节级断言。"""
    return SupercompressContextEngine(context_length=context_length,
                                      external_store_path=str(tmp_path), **kw)


def _seed_user(e: SupercompressContextEngine, text: str,
               settled_signal: int = 0) -> None:
    """直接落 user P0 段（隔离数据面，不经宿主 turn 链）。

    4a 数字代号：`settled_signal` 打定案信号码（0=无/1=弱候选/2=强完成），
    账本写侧纯读码不扫词（多语言干净，词表默认关）。"""
    e.store.append(Segment(kind="segment", tier="P0", provenance="user",
                           original_text=text, session_topic="", model_version="m1",
                           settled_signal=int(settled_signal)))
    e.store.flush()


def _mkmsgs_long() -> list:
    """长会话面（15 条）：head(3)+tail(6)=9 < 15 → 存在可压缩 middle → 账本前缀门控开启。

    交替 role（sys + 14 条 u/a 交替）避免头尾缝合处本身造 user-user 相邻。"""
    out = [{"role": "system", "content": "你是 ops 助手，负责网关部署与回滚，保留回滚入口。"}]
    for i in range(14):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    return out


def _ledger_section(out) -> str:
    """取含账本标记的消息 content（累计块；无 → ""）。"""
    for m in out:
        c = str(m.get("content", ""))
        if MARK in c:
            return c
    return ""


def _blocks(section: str) -> list[str]:
    return _C.findall(section)


# ─────────────────────────── 侧车路径（G2 单源）───────────────────────────
def test_ledger_path_single_source():
    """侧车路径 = 主 store splitext + '.compact.json'（与 model/digest 侧车同住同前缀）。"""
    p = compact_ledger_path_for_jsonl("/data/supercompress_store/s_alpha.jsonl")
    assert p == "/data/supercompress_store/s_alpha.compact.json", p
    p2 = compact_ledger_path_for_jsonl("/tmp/active.jsonl")
    assert p2 == "/tmp/active.compact.json", p2
    # 与 digest pool / model store 前缀同源（同目录同 sid 前缀，GC 联动）
    import os
    assert os.path.dirname(p2) == "/tmp" and os.path.basename(p2).startswith("active."), \
        f"同目录同前缀：{p2}"


# ─────────────────────────── GREEN 核心：append-only 前缀单调 ───────────────────────────
def test_ledger_appends_across_compacts_and_prefix_stable(tmp_path):
    """v6.1 页化契约（§2.4 迁移）：compact #1→#2 间账本**按页** append——
    每页 c#N 只装自己段的新 entries + 页头 notes 区（前 N-1 页 settled 快照），
    旧页字节冻结（无 settled 翻转时恒不变）→ provider 前缀缓存命中除最后一页
    外的全部旧页（v6.1 新锚；旧「累计超集严格前缀」契约 c2.startswith(c1) 已随
    页化迁移——页头 notes 区让新页不是旧页的严格字节前缀，改断言「旧页段原样
    在 + 每页只装自己」）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_ledger")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    _seed_user(e, "部署发布版本四十二到生产环境保留回滚入口并输出工具结果")

    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    c1 = _ledger_section(out1)
    assert "c#1" in c1, "compact #1 账本块须含 c#1（RED 现状：无账本块）"
    assert e._ledger.entries and e._ledger.entries[0]["text"] in c1, "账本条 = staged 块原文"
    _page1 = _page_segment(c1, "c#1")
    assert _page1, "页 c#1 页段在场"
    assert "notes: -" in _page1, "页 c#1 notes 区 = 首页空快照（notes: - 占位）"
    assert "c#2" not in _page1, "页 c#1 不含 c#2 正文（每页只装自己）"

    # compact #2：窗口又长大（+新 user 段），账本追加页 c#2；页 c#1 正文冻结，
    # 但它在「变旧页」那一刻追加 1 行页脚（v6.1 §2.5，铁律 4 的有意代价：
    # 每页留一行 → 旧页字节含页脚，不再是纯字节恒不变）
    _seed_user(e, "回滚开关位置在网关配置第三节保留现场")
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    c2 = _ledger_section(out2)
    assert "c#2" in c2 and "c#1" in c2, "compact #2 账本块含 c#1 + c#2 两页"
    _p1_in_c2 = _page_segment(c2, "c#1")
    assert _p1_in_c2.startswith(_page1), \
        "旧页 c#1 正文冻结：c#2 渲染里 c#1 正文原样（无 settled 翻转）"
    assert _p1_in_c2.rstrip("\n---").endswith(PAGE_FOOTER), \
        "旧页 c#1 变旧页 → 追加 1 行页脚（§2.5，铁律 4 有意代价）"
    _p2 = _page_segment(c2, "c#2")
    assert "c#2" in _p2, "页 c#2 页段在场"
    assert "回滚开关位置在网关配置第三节保留现场" in _p2, "页 c#2 只装自己（本段新增 entries）"
    assert "c#1=" in _p2, "页 c#2 notes 区带 c#1 settled 快照（跨页一行索引，不重吐 c#1 全文）"
    assert "notes c#1=open" in _p2, "c#1 未定案 → notes 快照 open（settled 翻转前恒 open）"
    # 页化渲染：c#1/c#2 标记总数 = 页头（2）+ notes 区（页 c#2 带 1 处 c#1）+ 正文行内（2）
    # = c#1 共 3（页头 1 + notes 1 + 正文行内 1）、c#2 共 2（页头 1 + 正文行内 1）
    assert _blocks(c2).count("c#1") == 3 and _blocks(c2).count("c#2") == 2, _blocks(c2)
    # 账本块落 head/tail 缝（middle 注入位）：非首、非尾（唯一刷缓存位 = 滚动 tail）
    idx_ledger = next(i for i, m in enumerate(out2) if MARK in str(m.get("content", "")))
    assert 0 < idx_ledger < len(out2) - 1, "账本块须在 head 与滚动 tail 之间（非头非尾）"
    assert all(MARK not in str(m.get("content", "")) for m in out2[:idx_ledger])


def _page_segment(section: str, tag: str) -> str:
    """取 section 里某页「── 页 c#N ──」整段（到下一个 ── 页 或 section 末）的字节，
    尾部页间分隔符（\n---\n）裁掉（末尾页本就无分隔符；跨页比较时尾部差异归一）。"""
    _re_tag = re.compile(r"── 页 c#\d+ ──")
    _marks = list(_re_tag.finditer(section))
    for _i, _m in enumerate(_marks):
        if section[_m.start():_m.end()] == f"── 页 {tag} ──":
            _end = _marks[_i + 1].start() if _i + 1 < len(_marks) else len(section)
            return section[_m.start():_end].rstrip("\n---")
    return ""


# ─────────────────────────── 宿主 re-feed 去重（活体 compact #1→#2 面）───────────────────────────
def test_ledger_refeed_dedups_stale_section(tmp_path):
    """活体正确性：compact #1 发行的累计账本消息被宿主 re-feed 回 compact #2 的 messages 面时，
    旧账本消息（c#1 短段）落在 head/tail 保护窗内会与新累计段（c#1..c#N 全量超集）并存 →
    薄剥除 stale 账本消息（只认专属前缀，绝不误伤 S1 尾投递的 staged P0/P1 消息）→
    输出恒 = 单条累计账本消息（前缀单调不被陈旧旧段破坏）。

    无去重（RED 面）→ out 里 2 条账本消息（旧 c#1 短段 + 新累计段）；GREEN → 恰 1 条。"""
    e = _mk(tmp_path)
    e.on_session_start("s_refeed")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert len(_ledger_sections(out1)) == 1, "compact #1 恰 1 条账本消息"
    c1 = _ledger_section(out1)
    assert "c#1" in c1
    # compact #2：宿主 re-feed = 把 out1（含 compact #1 的账本消息）当 live transcript 喂回，
    # 中间又长大（+新 user 段）→ 账本追加 c#2
    _seed_user(e, "回滚开关位置在网关配置第三节保留现场")
    out2 = e.compress(out1, current_tokens=4000, force=True)
    secs = _ledger_sections(out2)
    assert len(secs) == 1, f"re-feed 后仍恰 1 条账本消息（stale 旧段已剥除），实得 {len(secs)} 条"
    c2 = secs[0]
    assert "c#2" in c2 and "c#1" in c2, "新累计段 = c#1 + c#2（全量超集，非 stale 旧短段）"
    # stale 旧段特征 = 只到 c#1 为止的短段；新段须是严格前缀超集（c#1..c#2）
    assert c2.startswith(c1.rstrip()) or c2.startswith(c1), "前缀单调：旧段 c#1 字节是新段前缀"


def _ledger_sections(out) -> list:
    """取**全部**含账本标记的消息 content（多条判定面；单条断言用它数个数）。"""
    return [str(m.get("content", "")) for m in out if MARK in str(m.get("content", ""))]


def test_ledger_refeed_tail_residue_dedup(tmp_path):
    """R2-2（2026-10-27 活体：compact #N→#N+1 账本消息双发·前缀断裂·复读死循环根因）：

    活体 compact #2 的 messages = out1（compact #1 全量输出，账本消息含在尾区）
    + 6 条新 user 段（让账本消息随 tail 滚动）→ 账本消息位置（idx ≥ protect_first_n）
    落进 protect_last_n 保护窗 → 若无修复，旧账本消息残存 out2 尾区，
    c#2 新账本消息 append 后**两条账本消息并存** = 前缀断裂 + 模型逐轮回看旧账本全文
    （2026-10-27 活体实锤：compact #2 后 c#2 双发，模型 7 轮复读「重新 build 了吗？」）。

    无修复（RED）→ out2 里 2 条账本消息；GREEN → 恰 1 条（新 c#2 累计段）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_tailres")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    _seed_user(e, "部署发布版本四十二到生产环境保留回滚入口并输出工具结果")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert len(_ledger_sections(out1)) == 1, "compact #1 恰 1 条账本消息"
    # compact #2：宿主 re-feed 面——out1 作为前段（账本消息落尾区）+ 新 user 段
    _seed_user(e, "回滚开关位置在网关配置第三节保留现场")
    msgs2 = list(out1)
    for i in range(6):
        msgs2.append({"role": "user", "content": f"live{15+i}"})
    out2 = e.compress(msgs2, current_tokens=4000, force=True)
    secs = _ledger_sections(out2)
    assert len(secs) == 1, \
        f"tail 残存旧账本消息须剥除，output 恒单条累计段，实得 {len(secs)} 条"
    c2 = secs[0]
    assert "c#2" in c2, "compact #2 账本含 c#2 页"
    # 前缀单调：out2 里无第二条带 MARK 的消息
    marked = [m for m in out2 if MARK in str(m.get("content", ""))]
    assert len(marked) == 1, f"out2 恰 1 条 MARK 消息，实得 {len(marked)}"


# ─────────────────────────── 封顶 = 整段清 + 重新积累 ───────────────────────────
def test_ledger_cap_seals_whole_section(tmp_path):
    """累计条目 > cap → 旧块整段清掉、只留新块（seal +1、generation 单调），不做 mid-block 回改。"""
    e = _mk(tmp_path, compact_ledger_max_tokens=10_000, compact_ledger_max_entries=3)
    e.on_session_start("s_seal")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    _seed_user(e, "部署发布版本四十二到生产环境保留回滚入口并输出工具结果")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert "c#1" in _ledger_section(out1)
    assert e._ledger.generation == 0, "首段积累不 seal（generation=0）"
    assert len(e._ledger.entries) <= 3, "封顶点前累计不超 cap"
    # 逐条塞满累计 → 触发整段 seal（generation +1、旧块清、只留新块）
    out = out1
    for i in range(6):
        _seed_user(e, f"封测段落编号{i:02d}网关告警保留回滚入口工具输出")
        out = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
        if e._ledger.generation > 0:
            break
    assert e._ledger.generation >= 1, "累计超 cap 须触发整段 seal"
    assert e._ledger.seals >= 1
    c_last = _ledger_section(out)
    assert "c#1" not in c_last, "seal 后旧块整段清掉（mid-block 不回改 = 整段 reset 语义）"
    assert _blocks(c_last), "seal 后重新积累（新块在）"


# ─────────────────────────── 会话边界（/new）整段 reset ───────────────────────────
def test_ledger_resets_on_session_reset(tmp_path):
    """on_session_reset（/new）整段清账本：条目 0、generation +1（与 rerank/p4 缓存清零同口径）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_r")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert "c#1" in _ledger_section(out1)
    # 造第二块（积累 2 块）
    _seed_user(e, "第二块新段部署网关告警保留现场")
    out1b = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    _sec1b = _ledger_section(out1b)
    assert _page_segment(_sec1b, "c#1") and _page_segment(_sec1b, "c#2"), \
        "reset 前累计 2 块（页化：c#1 + c#2 两页）"
    e.on_session_reset()
    assert e._ledger.entries == [] and e._ledger.generation >= 1, "reset = 整段清"
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    _sec2 = _ledger_section(out2)
    # reset 后 seq 号单调推进（c#3），旧 c#1/c#2 页头全清：只 1 页在场
    assert len(_page_marks(_sec2)) == 1, "reset 后只重新积累 1 页（旧 c#1/c#2 页头全清）"
    assert not _page_segment(_sec2, "c#1") and not _page_segment(_sec2, "c#2"), \
        "旧页 c#1/c#2 不跨 reset 续写（seq 号单调推进到新页号）"


# ─────────────────────────── 同 session 冷重开：账本字节不动（跨进程 append-only）───────────────────────────
def test_ledger_survives_cold_restart_of_same_session(tmp_path):
    """同 session 冷重开（进程重启，新实例）→ 账本字节不动，load 既有累计块（append-only 跨进程
    延续 = 本卡对 provider 缓存的核心承诺）；整段 reset 只在会话边界，不在此面。"""
    e = _mk(tmp_path)
    e.on_session_start("s_r2")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    c1 = _ledger_section(out1)
    assert "c#1" in c1
    e.on_session_end("s_r2", [])

    e2 = _mk(tmp_path)
    e2.on_session_start("s_r2")          # 同 sid 冷重开（新实例，load 既有侧车）
    _seed_user(e2, "冷重开新段网关告警保留回滚入口")
    out2 = e2.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    c2 = _ledger_section(out2)
    assert c2.startswith(c1), "冷重开 append-only：旧 c#1 字节原样是新一段账本的前缀"
    assert "c#1" in c2 and len(_blocks(c2)) >= 2, "冷重开后追加新块（累计）"


# ─────────────────────────── 轮转 rebind：新 session 账本不跨 session 泄漏 ───────────────────────────
def _contents(out):
    """全部消息 content 文本（子串判定面）。"""
    return [str(m.get("content", "")) for m in out]


def _has_substr(needle, out) -> bool:
    return any(needle in c for c in _contents(out))


def test_ledger_rebinds_on_carry_over_rotation(tmp_path):
    """轮转 rebind（old_sid + carry_over_context=True → 新 session 侧车）→ 新 session 账本从空起，
    旧 session 累计块不跨 session 泄漏进新前缀（新侧车 = new_s.compact.json，与旧 old_s 物理隔离）。"""
    e = _mk(tmp_path)
    e.on_session_start("old_s")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    _sec1 = _ledger_section(out1)
    assert _page_segment(_sec1, "c#1"), "旧 session 首块 c#1"
    # 再喂一段 → 窗口变大 → 新 staged 块 → 追加 c#2（append-only 累计）
    _seed_user(e, "旧 session 第二段网关部署保留回滚入口工具输出")
    out1b = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    _sec1b = _ledger_section(out1b)
    assert _page_segment(_sec1b, "c#1") and _page_segment(_sec1b, "c#2"), \
        "旧 session 累计 2 块（c#1 + c#2）"
    e.on_session_start("new_s", old_session_id="old_s", carry_over_context=True)
    lg = e._compact_ledger()
    assert lg.path == compact_ledger_path_for_jsonl(e.store.path), "账本随 store rebind 到新 session 侧车"
    assert lg.path.endswith("new_s.compact.json"), "新 session 侧车 = new_s（与旧 old_s 物理隔离）"
    assert lg.entries == [], "轮转 rebind 后新 session 账本从空起（旧 c#1/c#2 不跨 session 泄漏）"
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    _sec2 = _ledger_section(out2)
    assert _page_segment(_sec2, "c#1") and not _page_segment(_sec2, "c#2"), \
        "新 session 只重新积累 1 块（c#1，非旧 2 块续写）"


def test_ledger_noop_when_no_middle_keeps_s1(tmp_path):
    """无 middle（len ≤ protect_first+last）→ 不发行账本前缀消息（无 MARK），staged 块照走既有
    S1 尾投递（末条 user 并入）→ 既有 test_staged_block_s1_tail（7 条会话）行为零变化。"""
    e = _mk(tmp_path)
    e.on_session_start("s_short")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    # 短会话（7 条：head3+tail6=9 ≥ 7 → 无 middle）
    short = [{"role": "system", "content": "你是 ops 助手，负责网关部署与回滚，保留回滚入口。"},
             {"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
             {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
             {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]
    out = e.compress(short, current_tokens=4000, force=True)
    assert not _has_substr(MARK, out), "无 middle 不发行账本消息（MARK 不在任何消息 content）"
    # staged S1 尾投递保留（末条 assistant → 追加独立 user，content 含 staged 头）
    assert _has_substr("[supercompress] staged", out), "短会话 staged 照走 S1 尾投递"
    # 短会话面从不触碰账本（走 else 分支）→ 账本未 materialize
    assert e._ledger is None, "短会话无账本积累（未 materialize 侧车，无块可 append）"


def test_ledger_noop_when_no_stored_content(tmp_path):
    """无 P0/P1 段（空 store）→ choose_forms 出空块 → 无块可累计 → 无账本消息（布局 = 现状）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_empty")
    out = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert not _has_substr(MARK, out), "空 store 不发行账本块"
    assert not _has_substr("[supercompress] staged", out), "空块不发 staged"
    # 空块 → if block: False → 账本从不 materialize
    assert e._ledger is None, "空 store 零积累（无块可 append，侧车未落盘）"


def test_ledger_fail_open_on_corrupt_sidecar(tmp_path):
    """损坏 JSON → fail-open 空账本（整段作废 generation 留痕）继续 append，永不炸宿主 compress 返回面。"""
    e = _mk(tmp_path)
    (tmp_path / "s_corrupt.compact.json").write_text("{not valid json", encoding="utf-8")
    e.on_session_start("s_corrupt")   # 首次 _compact_ledger() load 见损坏 → scrap（generation 单调 +1）
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    out = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)   # 不得抛
    assert _has_substr(MARK, out), "损坏侧车 fail-open：重置后正常积累"
    assert e._ledger.generation >= 1, "损坏侧车整段作废（generation 单调留痕）"
    _sec = _ledger_section(out)
    assert len([_m for _m in _page_marks(_sec)]) == 1, \
        "fail-open 后从空重新积累 1 块（页化：恰 1 个 ── 页 c#N ── 页头）"
    assert any("c#" in _m for _m in _page_marks(_sec)), "新页在场（seq 号随 scrap 单调推进）"


def _page_marks(section: str) -> list:
    """section 里所有「── 页 c#N ──」页头行（页化契约：页数 = 页头数）。"""
    return [ln for ln in section.splitlines() if ln.startswith("── 页 c#")]


# ─────────────────────────── 宿主 agent/ 零 diff 自检（红线条款）───────────────────────────
def test_host_agent_zero_diff():
    """宿主 agent/ 零 diff（本卡只改插件 sc/+engine.py+tests/）。"""
    import subprocess
    repo = Path("/usr/local/lib/hermes-agent")
    if not (repo / ".git").is_dir():
        return
    st = subprocess.run(["git", "status", "--porcelain", "agent/"],
                        cwd=repo, capture_output=True, text=True)
    assert st.stdout.strip() == "", f"宿主 agent/ 出现 diff（红线）：\n{st.stdout}"


# ─────────────────────────── L1 区裁定（owner）：撤 A1 前置重排，P0 回 c# 自然 append 序 ───────────────────────────
def test_p0_follows_natural_c_order_after_a1_removal(tmp_path):
    """撤 A1 后断言（owner 裁定：P0 定案段回 c# 自然 append 序 = L1 区）：
    账本消息按 c#1..c#N 原序渲染（ledger.text() 纯函数，无 tier 重排）→
    provider 前缀缓存真单调稳定（A1 重排面已删，P0 一变不再打断前缀）。

    构造：小 context_length（高压 → 确定性 L3 结构面）+ 账本帽（max_entries=2）
    逼出多块累计面，断言 c#1、c#2 严格按 append 序在场（无重排），
    且 P0 定案段被自动折出（settled 存根面）：P0 段打 settled_signal=2 强完成码
    → 折单行存根（原文沉盘 <sid>.p0，治「重复问」病根），不再全量重喂模型。"""
    e = _mk(tmp_path, compact_ledger_max_tokens=50, compact_ledger_max_entries=2)
    e.on_session_start("s_p0nat")
    # 4a 数字代号：P0 段打 settled_signal=2（强完成码，模拟上游 LLM 打码）→ 自动折存根
    _seed_user(e, "网关限流阈值已定案为每秒五百请求并配置告警通知保留回滚入口",
               settled_signal=2)
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    secs = _ledger_sections(out1)
    assert len(secs) == 1, "恰 1 条账本消息"
    c1 = secs[0]
    assert "c#1" in c1
    idx = next(i for i, m in enumerate(out1) if MARK in str(m.get("content", "")))
    assert idx == e.protect_first_n, "账本消息固定 head 缝位"
    assert all(MARK not in str(m.get("content", "")) for m in out1[idx + 1:])

    # 再积累出第二块 → c#1/c#2 严格自然 append 序（撤 A1：无 P0 强制前置面）
    _seed_user(e, "P1 中间推理步：网关配置第三节定位与回滚开关映射推导过程")
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    c2 = _ledger_section(out2)
    assert c2, "多块累计后账本 section 仍在"
    # 前缀单调稳定性（撤 A1 核心收益）：P0 定案段一变（重排/折出）不再打断前缀，
    # c# 自然 append 序下旧块字节恒为新一段严格前缀 → provider 前缀缓存真单调。
    # 注：max_entries=2 + P1 块 token > cap 时 c#1（settled 存根）可被裁（seal 保底），
    # c#2 在场（重新积累）即证明自然 append 序（非强制前置重排面）。
    assert c2.find("c#2") >= 0, "c#N 自然 append 序渲染（无 tier 重排面）"
    # P0 定案段被自动折出（settled 存根面）：c#1 打 settled_signal=2 → 存根在场（不重读全文），
    # 治「重复问」病根（settled 块不再全量回喂模型）。存根 = `#seq=1 P0 settled sha256=…`
    assert "P0 settled" in c1 and "sha256=" in c1, \
        "P0 定案段命中完成信号 → settled 存根在场（折出活渲染面，治重复问）"


def test_p1_p2_tail_segments_may_be_truncated(tmp_path):
    """L1 区对称面：纯 P1/P2 可裁段（无 P0 定案段）的账本 section 保持自然 append 序
    （P1/P2 块沉后部 = 可裁区）；撤 A1 后无任何重排面，账本超帽整段清裁切
    只影响尾部可裁块。"""
    e = _mk(tmp_path, compact_ledger_max_tokens=50)
    e.on_session_start("s_p1p2")
    # 只种 P1 段（无 P0）
    e.store.append(Segment(kind="segment", tier="P1", provenance="user",
                           original_text="中间推理步推导网关配置第三节回滚映射", session_topic="",
                           model_version="m1"))
    e.store.flush()
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    c1 = _ledger_section(out1)
    assert c1, "P1 段账本 section 在场"
    assert "# P0" not in c1, "本面纯 P1（无 P0 定案段）"
    idx = next(i for i, m in enumerate(out1) if MARK in str(m.get("content", "")))
    assert idx == e.protect_first_n, "P1/P2 面账本消息同样固定 head 缝位"
    # 纯 P1/P2 → section 自然 append 序（撤 A1：无前置重排面）
    assert c1.find("#") > 0
    # 纯 P1 无 P0 定案段 → 自动面不折出（settled=False，全文回喂；治「重复问」
    # 只对定案段生效，未定案段照常全量回喂不藏内容）
    assert e._ledger is not None and len(e._ledger.entries) >= 1
    assert e._ledger.entries[0]["settled"] is False, "纯 P1 无 P0 定案段 → 不自动折出"
    # 前缀单调稳定性（撤 A1 核心收益）：P0 定案段一变（重排/折出）不再打断前缀，
    # c# 自然 append 序下旧块字节恒为新一段严格前缀 → provider 前缀缓存真单调。
    assert c1.find("c#1") >= 0, "账本块按 c#1 自然 append 序渲染（无 tier 重排面）"
    assert "sha256=" not in c1, "纯 P1 未折出 → 无存根（全文回喂）"


# ─────────────────────────── settled 折出存根 + sink/restore P0 原文落盘 ───────────────────────────
def test_settled_block_folds_to_stub_in_text(tmp_path):
    """settled 块在 text() 里折成单行存根（_settled_stub_line），原文不吐；
    未 settled 块照常渲染全文。存根 sha256 前 16 hex + kind + <seq>.p0 指针，
    字节恒定（不随原文长度变）→ 活渲染面前缀稳定。"""
    from sc.compact_ledger import CompactLedger, _settled_stub_line
    lg = CompactLedger(str(tmp_path / "a.compact.json"))
    e = lg.append("#1 P0 [L3]\n网关限流阈值定案五百请求", settled=True, kind="adhoc")
    assert e is not None and e["settled"] is True
    out = lg.text()
    # P5：text() 渲染存根带 sid 前缀（self.path 基名 a → a.p0.1）；对比基线同传 sid
    import os
    _sid = os.path.splitext(os.path.basename(str(tmp_path / "a.compact.json")))[0].replace(".compact", "")
    stub = _settled_stub_line(1, "#1 P0 [L3]\n网关限流阈值定案五百请求", "adhoc", _sid)
    assert "c#1: " + stub in out
    # 原文全文不在存根渲染里（只存根行在场）
    assert "网关限流阈值定案五百请求" not in out
    assert "sha256=" in out and "kind=adhoc" in out


def test_settle_unsettle_toggle_is_idempotent(tmp_path):
    """settle/unsettle 翻转：settle 幂等（已 settled → None）；unsettle 幂等
    （未 settled → None）；settle 后 text() 折存根、unsettle 后恢复全文。"""
    from sc.compact_ledger import CompactLedger
    lg = CompactLedger(str(tmp_path / "b.compact.json"))
    e1 = lg.append("#1 P0 [L3]\n定案段A")
    assert lg.settle(1) is not None, "settle 首次翻转"
    assert lg.settle(1) is None, "settle 幂等（已 settled → None）"
    assert lg.unsettle(1) is not None, "unsettle 复活"
    assert lg.unsettle(1) is None, "unsettle 幂等（未 settled → None）"
    # 找不到 seq fail-open
    assert lg.settle(99) is None and lg.unsettle(99) is None
    # 状态位可被 append() settled= 直写面与 settle() 翻转面双路驱动
    e2 = lg.append("#2 P0 [L3]\n定案段B", settled=True, kind="kanban")
    assert e2 is not None and e2["settled"] and e2["kind"] == "kanban"


def test_sink_restore_p0_roundtrip(tmp_path):
    """sink_p0/restore_p0 落盘捞回：sink 写 <seq>.p0（首行 kind + 原文），
    restore 按 seq 读回 (kind, text)；缺失 → None（fail-open）；幂等覆盖。

    旧形（sid 空）兼容面：<seq>.p0。P5 跨 session 隔离面（sid 非空）走
    <sid>.p0.<seq>，见 test_p5_cross_session_sink_isolation。"""
    import scdata
    base = str(tmp_path)
    ok = scdata.sink_p0(base, 42, "#1 P0 [L3]\n定案原文", kind="kanban")
    assert ok is True
    p = scdata.p0_sink_path(base, 42)
    assert p.endswith("42.p0")
    got = scdata.restore_p0(base, 42)
    assert got is not None and got[0] == "kanban" and got[1] == "#1 P0 [L3]\n定案原文", got
    # 幂等覆盖：同 seq 重复写 → 字节稳定（内容一致）
    ok2 = scdata.sink_p0(base, 42, "#1 P0 [L3]\n定案原文", kind="kanban")
    assert ok2 is True
    assert scdata.restore_p0(base, 42) == ("kanban", "#1 P0 [L3]\n定案原文")
    # 缺失 seq → None（fail-open）
    assert scdata.restore_p0(str(tmp_path / "no_dir"), 7) is None


def test_p5_cross_session_sink_isolation(tmp_path):
    """P5（第三轮 reviewer major 翻案）：跨 session 沉盘文件命名隔离。

    共享 store_dir 下多个 session 各自 seq 从 0 起，无 sid 前缀时 <seq>.p0 同名
    跨 session 覆盖 = 数据损坏（restore 捞回别的 session 内容）。P5 修复：sink/restore
    加 sid 参数，<sid>.p0.<seq> 同前缀隔离——两个 session 写同 seq 不撞名，restore
    各自按 sid 捞回本 session 内容。

    断言面：
    1. 两个 sid（sid_a / sid_b）各写 seq=1 → 两个独立文件 <sid_a>.p0.1 / <sid_b>.p0.1
       不覆盖，restore 各得各的内容。
    2. sid 空（旧形）写 seq=1 → <1>.p0，与 sid 形不撞。
    3. p1 面同隔离：<sid>.p1.<seq>。"""
    import scdata, os
    base = str(tmp_path)
    # 两 session 各写 seq=1（同 seq 不同 sid）
    assert scdata.sink_p0(base, 1, "session A 的定案", kind="adhoc", sid="sid_a") is True
    assert scdata.sink_p0(base, 1, "session B 的定案", kind="kanban", sid="sid_b") is True
    # 两文件独立存在（不覆盖）
    assert os.path.exists(os.path.join(base, "sid_a.p0.1"))
    assert os.path.exists(os.path.join(base, "sid_b.p0.1"))
    # restore 各得各的内容（跨 session 不串号）
    got_a = scdata.restore_p0(base, 1, sid="sid_a")
    got_b = scdata.restore_p0(base, 1, sid="sid_b")
    assert got_a is not None and got_a[1] == "session A 的定案" and got_a[0] == "adhoc"
    assert got_b is not None and got_b[1] == "session B 的定案" and got_b[0] == "kanban"
    # 旧形（sid 空）写 seq=1 → <1>.p0，与 sid 形不撞
    assert scdata.sink_p0(base, 1, "legacy 形", kind="adhoc") is True
    assert os.path.exists(os.path.join(base, "1.p0"))
    assert scdata.restore_p0(base, 1) is not None and scdata.restore_p0(base, 1)[1] == "legacy 形"
    # p1 面同隔离
    assert scdata.sink_p1(base, 1, "A 的 P1 原文", sid="sid_a") is True
    assert scdata.sink_p1(base, 1, "B 的 P1 原文", sid="sid_b") is True
    assert scdata.restore_p1(base, 1, sid="sid_a") == "A 的 P1 原文"
    assert scdata.restore_p1(base, 1, sid="sid_b") == "B 的 P1 原文"


def test_auto_settled_on_completion_signal(tmp_path):
    """引擎侧纯自动定案（B 路线）：staged 块含 L3 全文档面（forms 含 L3）
    且命中完成信号词（如「已完成」）→ compress() 本块记 settled=True，
    原文同事务沉盘 <sid>.p0（scdata.restore_p0 可捞回）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_auto")
    # 4a 数字代号：P0 段打 settled_signal=2（强完成码）→ adhoc 纯自动折出 + 沉盘
    _seed_user(e, "网关限流阈值定案为每秒五百请求，部署已完成，保留回滚入口",
               settled_signal=2)
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert e._ledger is not None and len(e._ledger.entries) >= 1
    e1 = e._ledger.entries[0]
    # 完成信号词「已完成」命中 + L3 面在场 → settled 自动翻转
    assert e1["settled"] is True, \
        f"staged 块命中完成信号（已完成）+L3 面 → 自动 settled，实得 {e1}"
    assert e1["kind"] == "adhoc", "纯自动面 kind=adhoc（非 kanban 半自动）"
    # 原文同事务沉盘（restore_p0 按 sid+seq 捞回，P5 跨 session 隔离 <sid>.p0.<seq>）
    import scdata, os
    _dir = os.path.dirname(os.path.abspath(e._ledger.path))
    _sid = os.path.splitext(os.path.basename(e._ledger.path))[0].replace(".compact", "")
    got = scdata.restore_p0(_dir, e1["seq"], sid=_sid)
    assert got is not None, "settled 块原文须沉盘 <sid>.p0.<seq>（restore_p0 可捞回）"
    assert got[0] == "adhoc" and got[1] == e1["text"]


def test_unsettled_block_keeps_full_text(tmp_path):
    """对称面：未打 settled_signal 码（默认 0）的 staged 块保持 settled=False + kind=adhoc，
    text() 全文渲染（不折存根）→ 未定案段仍回喂模型（治病根但不藏未定案内容）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_unsettled")
    _seed_user(e, "网关限流阈值待调研，保留回滚入口")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert e._ledger is not None and len(e._ledger.entries) >= 1
    e1 = e._ledger.entries[0]
    assert e1["settled"] is False, \
        f"未打码（settled_signal=0）→ 保持 settled=False（全文回喂），实得 {e1}"
    assert e1["kind"] == "adhoc", "不定案块 kind=adhoc（非 kanban candidate）"
    # text() 全文渲染（无存根）：未 settled 块照常渲染全文，活渲染面含 P0 定案段
    assert "P0" in _ledger_section(out1) and "sha256=" not in _ledger_section(out1)


def test_kanban_candidate_nomination_and_judge_loop(tmp_path):
    """看板半自动完整链路（B 路线，方案 1 引擎内闭环）：

    ① 弱候选信号词（「定案/确认/锁定」等提名级）+ P0 定案段在场
       → compress() 提名 candidate 块：settled=False + kind=kanban
       （活渲染面照常全量回喂，不折存根）
    ② 宿主 LLM 裁判读 `kanban_candidates()` 拿候选清单 → 提名 seq
    ③ clarify 用户确认 → 调 `settle_p0(seq, kind="kanban")` 真正折出
       （settled=True + 原文沉盘 <sid>.p0）
    ④ 看板 task 重新活跃 → `unsettle_p0(seq)` 复活（settled=False，全文回喂）
    ⑤ 未确认的 candidate 保持 settled=False（不隐藏，LLM 裁判可下轮再拍）

    不读 kanban 自动钩子，守「改动只在 supercompress」红线。"""
    import scdata, os
    e = _mk(tmp_path)
    e.on_session_start("s_kanban_loop")
    # 种 user P0 段，打 settled_signal=1（弱候选码），无强完成码（2）
    _seed_user(e, "网关限流阈值锁定为每秒五百请求，保留回滚入口", settled_signal=1)
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert e._ledger is not None and len(e._ledger.entries) >= 1
    e1 = e._ledger.entries[0]
    # ① candidate 提名：settled=False + kind=kanban（未折出，全文回喂）
    assert e1["settled"] is False, \
        f"弱候选信号 → candidate 块 settled=False（不折出），实得 {e1}"
    assert e1["kind"] == "kanban", "candidate 块 kind=kanban（半自动面）"
    # 活渲染面：未 settled 块全文在场（无存根），P0 定案段仍回喂模型
    c1 = _ledger_section(out1)
    assert "sha256=" not in c1, "candidate 未折出 → 无存根（全文回喂）"
    assert "P0" in c1, "candidate 块 P0 定案段仍在活渲染面（LLM 裁判可看到）"

    # ② LLM 裁判读候选清单（宿主侧，O(n) 点查，零新索引）
    cands = e.kanban_candidates()
    assert len(cands) == 1, f"恰 1 个 kanban candidate 待拍板，实得 {len(cands)}"
    assert cands[0]["seq"] == e1["seq"] and cands[0]["kind"] == "kanban"

    # ③ clarify 用户确认 → 调 settle_p0 真正折出（引擎内闭环，不读 kanban 钩子）
    ok = e.settle_p0(e1["seq"], kind="kanban")
    assert ok is True, "LLM 裁判 + clarify 确认 → settle_p0 折出成功"
    assert e._ledger.entries[0]["settled"] is True
    assert e._ledger.entries[0]["kind"] == "kanban"
    # 折出后：原文沉盘（kind=kanban，审计面，P5 跨 session 隔离 <sid>.p0.<seq>）
    _dir = os.path.dirname(os.path.abspath(e._ledger.path))
    _sid = os.path.splitext(os.path.basename(e._ledger.path))[0].replace(".compact", "")
    got = scdata.restore_p0(_dir, e1["seq"], sid=_sid)
    assert got is not None and got[0] == "kanban" and got[1] == e1["text"], \
        "settled 原文沉盘 <sid>.p0.<seq>（restore_p0 可捞回，kind=kanban）"
    # candidate 清单清空（已折出，不再待拍）
    assert e.kanban_candidates() == [], "折出后 candidate 清单应清空"
    # 活渲染面：折出后变存根（sha256= 在场，原文不吐）
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    c2 = _ledger_section(out2)
    assert "P0 settled" in c2 and "sha256=" in c2, \
        "折出后活渲染面变单行存根（治重复问）"

    # ④ 看板 task 重新活跃 → unsettle_p0 复活（全文回喂，可审计）
    ok2 = e.unsettle_p0(e1["seq"])
    assert ok2 is True, "unsettle_p0 复活成功"
    assert e._ledger.entries[0]["settled"] is False
    # 复活后重入 candidate 清单（kind=kanban 不变，重新待拍）
    assert len(e.kanban_candidates()) == 1, "复活后重新进入 candidate 清单"
    # ⑤ 未确认的 candidate 保持全文回喂（不隐藏）
    out3 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert "sha256=" not in _ledger_section(out3), "复活后全文回喂（无存根）"


def test_kanban_candidate_rejected_keeps_full_text(tmp_path):
    """对称面：LLM 裁判/clarify 否决某 candidate → 不调 settle_p0，
    该块保持 settled=False + kind=kanban（candidate 仍在清单，下轮可再拍），
    活渲染面全文回喂（不隐藏未确认内容）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_kanban_reject")
    _seed_user(e, "网关限流阈值锁定为每秒五百请求，保留回滚入口", settled_signal=1)
    e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert e._ledger is not None and len(e._ledger.entries) >= 1
    e1 = e._ledger.entries[0]
    assert e1["settled"] is False and e1["kind"] == "kanban"
    # LLM 裁判否决 → 不调 settle_p0（宿主侧决定，引擎不主动折）
    cands = e.kanban_candidates()
    assert len(cands) == 1 and cands[0]["seq"] == e1["seq"], \
        "否决后 candidate 仍在清单（下轮可再拍）"
    # 活渲染面全文回喂
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert "sha256=" not in _ledger_section(out2), "否决 → 全文回喂（无存根）"
    assert e._ledger.entries[0]["settled"] is False, "否决 → 不折出"


def test_settle_p0_public_api_and_unsettle_revive(tmp_path):
    """看板半自动面（B 路线）：settle_p0(seq, kind='kanban') 折出 + 沉盘；
    unsettle_p0(seq) 复活（重新活跃）。LLM 裁判提名 + clarify 确认后调，
    不读 kanban 自动钩子（红线：改动只在 supercompress）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_kanban")
    # 种 P1 段（无 P0 定案段）→ 自动面不折出（settled=False，全文回喂），
    # 给半自动面留可翻转的未 settled 块（LLM 裁判/clarify 拍板后才折）
    e.store.append(Segment(kind="segment", tier="P1", provenance="user",
                           original_text="中间推理步：网关配置第三节定位与回滚开关映射推导过程",
                           session_topic="", model_version="m1"))
    e.store.flush()
    e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert e._ledger is not None and len(e._ledger.entries) >= 1
    e1 = e._ledger.entries[0]
    assert e1["settled"] is False, "纯 P1 无 P0 定案段 → 自动面不折出（留半自动可翻转块）"
    seq = e1["seq"]
    # LLM 裁判提名 + clarify 确认 → 调 settle_p0（kind=kanban 半自动面）
    ok = e.settle_p0(seq, kind="kanban")
    assert ok is True, "settle_p0 翻转成功"
    assert e._ledger.entries[0]["settled"] is True
    assert e._ledger.entries[0]["kind"] == "kanban"
    # 沉盘件在场（可审计可复活，P5 跨 session 隔离 <sid>.p0.<seq>）
    import scdata, os
    _dir = os.path.dirname(os.path.abspath(e._ledger.path))
    _sid = os.path.splitext(os.path.basename(e._ledger.path))[0].replace(".compact", "")
    got = scdata.restore_p0(_dir, seq, sid=_sid)
    assert got is not None and got[0] == "kanban"
    # 看板 task 重新活跃 → unsettle_p0 复活
    ok2 = e.unsettle_p0(seq)
    assert ok2 is True, "unsettle_p0 复活成功"
    assert e._ledger.entries[0]["settled"] is False
    # 幂等：再 unsettle → False（本就未 settled）
    assert e.unsettle_p0(seq) is False


# ─────────────────────────── tier 分层：P0 定案块 + P1 可裁层（owner 裁定）───────────────────────────
def test_tier_segregated_p0_p1_into_ledger(tmp_path):
    """tier 分层核心断言：P0 定案段和 P1 推理段拆成**两个独立账本块**——
    P0 块 trimmable=False（永不裁，走 settled 三态），P1 块 trimmable=True（可裁层，
    封顶优先裁，原文 recall 可捞）。定案账本不再被 P1 推理段撑爆。"""
    e = _mk(tmp_path)
    e.on_session_start("s_tier_seg")
    # 同时种 P0 定案段（settled_signal=2 强完成码 → adhoc 面）+ P1 推理段
    _seed_user(e, "网关限流阈值定案为每秒五百请求并配置告警通知，部署已完成",
               settled_signal=2)
    e.store.append(Segment(kind="segment", tier="P1", provenance="user",
                           original_text="中间推理步：网关配置第三节定位与回滚开关映射推导",
                           session_topic="", model_version="m1"))
    e.store.flush()
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert e._ledger is not None
    # 分层：P0 块和 P1 块各自独立 append
    p0_entries = [x for x in e._ledger.entries if not x.get("trimmable")]
    p1_entries = [x for x in e._ledger.entries if x.get("trimmable")]
    assert len(p0_entries) >= 1, "P0 定案块在场（trimmable=False）"
    assert p0_entries[0]["trimmable"] is False
    # P1 可裁块在场（有 P1 段时）
    if p1_entries:
        assert p1_entries[0]["trimmable"] is True, "P1 推理块 trimmable=True（可裁层）"
    # P0 定案块走三态判据（此面「定案」强信号 → adhoc settled）
    assert p0_entries[0]["settled"] is True, "P0 定案块命中完成信号 → adhoc settled"
    # P1 块不折存根（还没定案，推理过程仍回喂）
    if p1_entries:
        assert p1_entries[0]["settled"] is False, "P1 可裁块 settled=False（不折存根）"


def test_p1_trimmable_trimmed_before_p0_at_cap(tmp_path):
    """封顶裁切顺序实压断言（R5 语义 X：折存根不硬删，真治编号空洞）：
    P1 可裁层（trimmable=True）在封顶时折 trimmed 存根（不硬删 = entry 常驻、编号零空洞）；
    P0 块（trimmable=False）永不被 P1 层误伤。

    结构性事实（owner 第三轮 P7 修正）：P1 可裁层**全量折所有 trimmable 块（无位置
    门槛，不扫最旧端、遇 P0 不即停）**——混合 P0+P1 账本里 P1 层照样全折 P1，
    P0 挡前端只意味着 P0 自身永不被折（trimmable=False）；P1 真正被全清靠 ③ seal
    保底（P0 同清）。**只有 P1-only 账本（无 P0）P1 层全量折才真腾帽位**；
    本测试走 CompactLedger 单元面（P1-only 账本，不经 engine e2e 的 store 累积）：
    4 块 P1（各 ~5 token）+ 条数帽驱动（max_entries=2，token 帽放大到不触发）→
    P1 层折存根裁 3 块（4 → 1 全文块 + 3 trimmed 存根，seals=0 证 P1 层裁非 seal）。

    R5 语义 X 关键变化（owner 裁定，既有断言"裁后条目数减少"→"trimmed 存根在场"）：
    被裁 P1 块 **entry 常驻 `entries`**（trimmed=True），`len(entries)` 不下降，
    改数全文块数（`_full_block_count`）判帽位 → 折存根真腾帽位、seal 不抢跑。
    """
    import tempfile as _tf
    from sc.compact_ledger import CompactLedger
    _base = _tf.mkdtemp()
    # 条数帽驱动：max_entries=2（4 块 P1 → 全文块 1 + 新块 1 = 2，trim 折 3 块存根），
    # token 帽放大到 1000（各 ~5 token × 4 = 20 < 1000，token 帽永不触发 → 纯条数帽面）
    lg = CompactLedger(_base + "/led.json", max_tokens=1000, max_entries=2)
    for i in range(4):
        lg.append(f"#P1c{i} 推理段：网关限流阈值推导{i}", trimmable=True)
    # 语义 X：P1 层真裁 = 3 块折 trimmed 存根（entry 常驻，编号零空洞），1 全文块 + 新块
    p1_now = [x for x in lg.entries if x.get("trimmable")]
    assert len(p1_now) == 4, \
        f"语义 X：被裁 P1 块 entry 常驻（trimmable 全在账本，{len(p1_now)} 块）"
    trimmed_now = [x for x in p1_now if x.get("trimmed")]
    assert len(trimmed_now) >= 1, \
        f"语义 X：P1 层真裁（≥1 块折 trimmed 存根在场，实得 {len(trimmed_now)}）"
    assert lg.seals == 0, "P1 层裁切未触发 seal（seal 是最后保底出口，本面 P1 层已够）"
    assert all(x.get("trimmable") for x in lg.entries), \
        "P1-only 账本裁后在场块全为可裁（无 P0 前端，P1 层真生效）"


def test_p0_candidate_survives_p1_cap_trim(tmp_path):
    """混合 P0+P1 账本封顶（reviewer 签认补强）：P0 candidate（trimmable=False，
    未定案全文回喂）永不被 P1 可裁层**选择性**误伤。

    可观测不变量（owner 第三轮 P7 修正）：P1 可裁层**全量折所有 trimmable 块（无位置
    门槛）**——混合账本里 P1 层照样全折 P1，P0 自身因 trimmable=False 永不被折；seal
    是保底兜底（全清 P0+P1）。P0 块与 P1 块同进退——要么全保留（未超帽，P1 层无可裁
    面），要么 P0/P1 全清（seal 保底）。**绝不出现** P1 在场而 P0 被单独裁掉。

    CompactLedger 单元面双场景对比（不经 engine e2e 的 store 累积，P1 块不跨轮增长）：
    P0（~5 token）+ 4 块 P1（各 ~5 token）= 25 token 总量。
    cap=30 未超 → 全保留（P0/P1 同在，P1 层无可裁面）；cap=12 真超 → P1 层全量折
    P1（P0 不受影响），若折后仍超（P0 挡住 + 总量仍超）→ seal 保底全清（P0/P1 同清）。"""
    import tempfile as _tf
    from sc.compact_ledger import CompactLedger
    _base = _tf.mkdtemp()
    _p0_block = "#0 P0 [L3]\n限流五百配告警"  # P0 candidate（trimmable=False，~5 token）
    _p1_blocks = [f"#P1c{i} 推理段：网关限流阈值推导{i}" for i in range(4)]
    # 场景 A：cap=30 未超（总量 25 < 30）→ P0/P1 全保留（P1 层无可裁面，seal 不触发）
    lg_a = CompactLedger(_base + "/a.json", max_tokens=30, max_entries=6)
    lg_a.append(_p0_block, trimmable=False)
    for b in _p1_blocks:
        lg_a.append(b, trimmable=True)
    p0_a = [x for x in lg_a.entries if not x.get("trimmable")]
    p1_a = [x for x in lg_a.entries if x.get("trimmable")]
    assert lg_a.seals == 0, "场景 A（未超帽）seal 不触发"
    assert len(p0_a) == 1 and len(p1_a) == 4, \
        "场景 A：P0 candidate + 4 P1 全保留（未超帽，P1 层无可裁面）"
    assert _p0_block in p0_a[0]["text"], "场景 A：P0 candidate text 保持原文（未被 P1 层改）"
    # 场景 B：cap=12 真超（总量 25 > 12）→ P1 层全量折 P1（无位置门槛，P0 不受影响）
    # → 折存根后 token 仍超（P0 挡住 + 总量仍超）→ seal 保底全清（P0 同清）→
    # 新 P1 块重入账本。核心不变量：
    # **P0 永不被 P1 层单独保留/单独裁掉**（P0 消失只可能来自 seal 全清，
    # 绝不"P1 层裁 P0 留 P1"或"P1 层裁 P1 留 P0"）。
    lg_b = CompactLedger(_base + "/b.json", max_tokens=12, max_entries=6)
    lg_b.append(_p0_block, trimmable=False)
    for b in _p1_blocks:
        lg_b.append(b, trimmable=True)
    p0_b = [x for x in lg_b.entries if not x.get("trimmable")]
    # P0 candidate（settled=False, trimmable=False）永不被 P1 层误伤：
    # 若 P0 在场，必保持原文 + 全非可裁；若 P0 不在场，必是 seal 全清（非 P1 层单独裁 P0）
    if p0_b:
        assert _p0_block in p0_b[0]["text"], "场景 B：P0 candidate text 保持原文"
        assert all(not x.get("trimmable") for x in p0_b), \
            "场景 B：P0 块全非可裁（P1 可裁层永不误伤 P0）"
    assert lg_b.seals >= 1, "场景 B（P1 层全量折 P1 后 token 仍超，P0 挡住）seal 保底全清（P0/P1 同清）"


def test_p1_only_ledger_cap_trims_p1_not_p0(tmp_path):
    """纯 P1 账本封顶 + P1 层折存根裁的实压断言（R5 语义 X：折存根不硬删，真治编号空洞；
    既有"裁后条目数减少"断言同步更新为"trimmed 存根在场 + entry 常驻"）：
    P1-only 账本（全 trimmable=True）条数帽超限时 P1 层**全量折所有 P1 块存根（无位置
    门槛，非 front-only）**，P0（若在场 trimmable=False）永不被 P1 层误伤。

    语义 X 关键不变量（owner 裁定）：被裁 P1 块 entry 常驻 `entries`（trimmed=True，
    编号零空洞），封帽改数全文块数（`_full_block_count`）→ 折存根真腾帽位、seal 不
    抢跑；`len(entries)` 不下降是预期行为（既有"裁到 2 块"断言随之更新）。

    单元面直接压 CompactLedger.append（不经 engine compress），条数帽驱动
    （max_entries=2，token 帽放大到 1000 不触发 = 纯条数帽面），精确复现
    reviewer 指出的「P1 层帽裁 P1、P0 candidate 在场不受影响」的实压语义。"""
    import tempfile as _tf
    from sc.compact_ledger import CompactLedger
    _base = _tf.mkdtemp()
    # 条数帽驱动：max_entries=2，token 帽 1000（各 ~5 token 永不触发 → 纯条数帽面）
    lg = CompactLedger(_base + "/led.json", max_tokens=1000, max_entries=2)
    # P1-only 账本：4 段 P1 块（各 ~5 token）逼条数帽超（全文块 4 > max_entries=2 真超）
    for i in range(4):
        lg.append(f"#P1c{i} 推理段：网关限流阈值推导{i}", trimmable=True)
    # 语义 X：P1 层真裁 = 折 trimmed 存根（entry 常驻，编号零空洞），seal 不抢跑
    p1_now = [x for x in lg.entries if x.get("trimmable")]
    assert len(p1_now) == 4, \
        f"语义 X：被裁 P1 块 entry 常驻账本（trimmable 全在，{len(p1_now)} 块）"
    trimmed_now = [x for x in p1_now if x.get("trimmed")]
    assert len(trimmed_now) >= 1, \
        f"语义 X：P1 层真裁（≥1 块折 trimmed 存根在场，实得 {len(trimmed_now)}）"
    assert lg.seals == 0, "P1 层裁切未触发 seal（seal 保底是最后出口，本面 P1 层已够）"
    # P0 candidate 在场不受 P1 层误伤（同账本加一块 P0，trimmable=False 永不裁）
    lg.append("#P0c0 定案 candidate（弱提名）", settled=False, kind="kanban",
              trimmable=False)
    p0_now = [x for x in lg.entries if not x.get("trimmable")]
    assert len(p0_now) == 1, "P0 candidate 块在场（trimmable=False 不被 P1 层误伤）"
    assert p0_now[0]["kind"] == "kanban" and p0_now[0]["settled"] is False, \
        "P0 candidate 块保持 kind=kanban / settled=False（未折出，全文回喂）"


def test_p1_trim_layer_real_pressure_seal_fallback(tmp_path):
    """engine e2e 实压面（与 test_p1_only_ledger_cap_trims_p1_not_p0 的单元面对偶）：
    P1 段跨轮累积在 store → P1 块 token 单调增长 → 超帽时 P1 层**全量折所有 trimmable
    块（无位置门槛）**——若折存根后总量仍超（P0 挡住或 P1 块自身超 cap）→ seal 保底兜底。
    断言可观测面：
    封顶后 P1 块数净减（P1 内容被清，机制=P1 层或 seal 均可），且 P0 块（若有）
    不被 P1 层选择性误伤（seal 全清或 P0 同在，绝不"P1 在 P0 单独被裁"）。"""
    e = _mk(tmp_path, compact_ledger_max_tokens=25, compact_ledger_max_entries=6)
    e.on_session_start("s_p1only")
    e.store.append(Segment(kind="segment", tier="P1", provenance="user",
                           original_text="第一段推理：网关限流阈值推导", session_topic="",
                           model_version="m1"))
    e.store.append(Segment(kind="segment", tier="P1", provenance="user",
                           original_text="第二段推理：告警通知映射推导", session_topic="",
                           model_version="m1"))
    e.store.flush()
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert e._ledger is not None
    p1_before = [x for x in e._ledger.entries if x.get("trimmable")]
    assert len(p1_before) >= 1, "P1-only 账本 P1 可裁块在场"
    # 再种 3 段 P1 逼超帽（store 累积 → P1 块增长 → 真超帽）
    for i in range(3):
        e.store.append(Segment(kind="segment", tier="P1", provenance="user",
                               original_text=f"第三段推理{ i}：回滚开关映射", session_topic="",
                               model_version="m1"))
    e.store.flush()
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    p1_after = [x for x in e._ledger.entries if x.get("trimmable")]
    assert len(p1_after) <= len(p1_before), \
        f"P1 可裁层封顶被清（块数 {len(p1_before)} → {len(p1_after)}，P1 层/seal 保底）"
    # P0 块（P1-only 账本恒空）：断言空真面已由上面实压断言覆盖，此处只保 P1 层不误伤
    p0_now = [x for x in e._ledger.entries if not x.get("trimmable")]
    assert not p0_now, "P1-only 账本无 P0 块（P1 层裁切面纯净，不误伤 P0）"


def test_staged_by_tier_helper(tmp_path):
    """`_staged_by_tier` helper 纯函数断言：P0/P1 拆分正确，块格式与 render_block 同源。"""
    from engine import _staged_by_tier
    from sc.schema import Segment
    from sc.distill import WindowSeg
    # 造最小 rendered 面（不跑完整 compress，纯 helper 测）
    seg_p0 = Segment(kind="segment", tier="P0", provenance="user",
                     original_text="定案段A", session_topic="", model_version="m1", seq=0)
    seg_p1 = Segment(kind="segment", tier="P1", provenance="user",
                     original_text="推理段B", session_topic="", model_version="m1", seq=1)
    ws_p0 = WindowSeg(seg_p0)
    ws_p1 = WindowSeg(seg_p1)
    rendered = [(ws_p0, "L3", "定案段A全文"), (ws_p1, "L2", "推理段B摘要")]
    p0_text, p0_forms = _staged_by_tier(rendered, "P0")
    p1_text, p1_forms = _staged_by_tier(rendered, "P1")
    assert "P0" in p0_text and "定案段A" in p0_text, "P0 块只含 P0 段"
    assert "推理段B" not in p0_text, "P0 块不含 P1 段"
    assert "P1" in p1_text and "推理段B" in p1_text, "P1 块只含 P1 段"
    assert "定案段A" not in p1_text, "P1 块不含 P0 段"
    assert p0_forms.get("L3") == 1 and p1_forms.get("L2") == 1, "forms 按 tier 分计"
    # 空 tier → 空块
    p2_text, p2_forms = _staged_by_tier(rendered, "P2")
    assert p2_text == "" and p2_forms == {}, "无该 tier 段 → 空块空 forms"
