"""R5（2026-10-13 · owner 批次 A）：compact-ledger 写侧 6 条确定性缺陷修复。

批次 A（纯确定性、零 LLM、低成本）裁定 6 条，对应 owner 诊断账本 #2/#3/#4/#5/#6/#8：

(a) #2 编号空洞 → 封帽裁量改"折存根行"：trimmable/settled 层裁块时不再硬删 entry，
    改为折叠成单行存根（settled 块既有 sha256 存根形态；trimmable P1 块折
    `#seq=P1 trimmed sha256=… (full text on disk: recall)` 存根）——seq 单调不回退，
    账本行无空洞，被裁块可经存根定位（settled→sink <seq>.p0；P1→recall 点查）。
(b) #3 同编号双行 → L1/L3/L2 行加 `prov=<user|model>` 标记（`#seq P0 prov=model ts=… · fact`），
    消除 user/model 段同号视觉重复；前缀缓存契约零扰动（渲染面追加字段，段字节不变）。
(c) #4 头重复 → engine.py 账本消息头去掉 generation/seals（单源 = ledger.text() 首行
    MARK 行），壳不渲染账本元信息（壳薄度原则）。
(d) #5 c#1: 前缀混淆 → engine._staged_by_tier 的块内段序号改**块内相对编号**
    行形 `#<rel> P0 prov=model seq=<全局seq> ts=… · fact`，账本面只剩 c#N 块号一层。
(e) #6 占位符混入 → **T1 折叠契约不动**（owner 裁定零回归，小围栏照折），
    治理改在记账面：sc/model_store._is_small_code_placeholder + engine 的
    _expand_small_code_placeholders（staged 块进账本前把 [code: ×N lines]
    N<20 的小占位符段展开原文，N≥20 维持残句）——治 [code: ×8 lines] 混入账本。
(f) #8 语义去重门 → CompactLedger.append 前近重复跳过：新块**开首 N=40 字符**与
    账本内任一未 settled 块开首 N 字符前缀碰撞（互为前缀 / 相同）→ 跳过不 append
    （旧块通常是信息超集，保留；零写盘）。诚实边界（owner 已知）：零 LLM 只能拦
    "近重复"（同 compact 事件重生成 / 同模板清点），"同任务详版+精简版"（开首不同
    字面）拦不住——那是 LLM 裁判面（批次 B），不在本批。
"""
from __future__ import annotations

import pytest

from sc.compact_ledger import CompactLedger
from sc.inject import _l1_line
from sc.model_store import ModelStore, _fold_code_fences
from sc.schema import Segment
from sc.distill import WindowSeg


def _seg(seq: int, tier: str, prov: str, text: str, ts: float = 0.0) -> Segment:
    s = Segment(tier=tier, provenance=prov, original_text=text)
    s.seq = seq
    s.ts = ts
    s.key_fact = "fact"
    return s


class TestL1LineTsRendering:
    def test_normal_ts_renders_as_before(self):
        line = _l1_line(WindowSeg(_seg(7, "P0", "model", "y" * 20, ts=1791822710.0)))
        # R5 新行形：#<rel> tier prov=… seq=<全局> ts=… · fact
        assert line == "#1 P0 prov=model seq=7 ts=1791822710 · fact", line

    def test_zero_ts_omitted_not_rendered(self):
        # 回放段 ts=0（不可信）→ 省略 ts 段，不再渲染误导性共用 ts
        line = _l1_line(WindowSeg(_seg(7, "P0", "model", "y" * 20, ts=0.0)))
        assert line == "#1 P0 prov=model seq=7 · fact", line
        assert "ts=" not in line


# ───────────────────────── (c) 账本消息头去重（engine 壳面）─────────────────────────
def test_ledger_message_header_single_source():
    """engine.py 消息头不再双渲染 generation/seals（单源 = ledger.text() 首行 MARK）。"""
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "engine.py"  # 安装可移植（原 /usr/local 字面量为 POSIX 部署假设）
    src = p.read_text(encoding="utf-8")
    # #4 修复的锚点：账本消息 content = 前缀 + section（首行 MARK 已含 generation/seals）
    assert '"content": _COMPACT_LEDGER_PREFIX + "\\n" + section' in src, \
        "账本消息 content 构造位未找到（应为 PREFIX + '\\n' + section 单源拼接）"
    # 渲染面断言（进模型上下文的 content 构造）不引用 ledger generation/seals：
    # 只卡 `_COMPACT_LEDGER_PREFIX + f"...generation...` 这种渲染面拼接形式
    # （日志观测面 logger.info 引用 ledger.generation 合法，不进模型上下文）。
    assert '_COMPACT_LEDGER_PREFIX + f' not in src, "账本消息头仍 f-string 渲染 generation/seals（#4 未修）"


# ───────────────────────── (b)+(d) L1 行 prov 标记 + 相对编号 ─────────────────────────
class TestL1LineProvAndRelSeq:
    def test_l1_line_has_prov_marker(self):
        ws = WindowSeg(_seg(7, "P0", "model", "x" * 20, ts=1791822710.0))
        line = _l1_line(ws)
        # 行形：#rel#<i> P0 prov=model [seq=<全局>] ts=… · fact
        assert "prov=model" in line, line
        assert "seq=7" in line, "全局 seq 须保留可定位（prov 标记后）"
        assert "ts=1791822710" in line

    def test_l1_line_user_prov(self):
        ws = WindowSeg(_seg(9, "P1", "user", "x" * 20, ts=1791822710.0))
        line = _l1_line(ws)
        assert "prov=user" in line, line

    def test_l1_rel_seq_param(self):
        """rel# 相对号由调用面传入（render_block 逐段计数）；缺省 1。"""
        ws = WindowSeg(_seg(7, "P0", "model", "x" * 20, ts=1791822710.0))
        line = _l1_line(ws, rel=3)
        assert line.startswith("#3 "), line
        assert "seq=7" in line, "相对号 #3 与全局 seq=7 并存可定位"


# ───────────────────────── (a) 封帽裁量折存根（编号无空洞）─────────────────────────
class TestTrimLeavesStubs:
    def test_trimmable_trim_leaves_stub_line(self, tmp_path):
        # 封帽触发 ① 裁量层：max_entries=2，trimmable P1 块占位后再 append
        # → 条数帽超限（len(entries)=2 已达 max_entries）→ _trim_trimmable_oldest
        # 把所有 P1 折存根（全量折，reviewer P7：无位置门槛，P0 挡前端不影响）
        # → 编号无空洞、无 seal
        led = CompactLedger(str(tmp_path / "l.compact.json"),
                            max_tokens=999999, max_entries=2)
        led.append("P1推理块一" * 20, trimmable=True)   # c#1 P1 可裁
        e2 = led.append("P0定案块一")                    # c#2：达帽（2 块）
        e3 = led.append("P0定案块二")                     # c#3 超限 → 折最旧 P1
        assert e2 is not None and e3 is not None
        assert led.seals == 0, "P1 折存根只降 token，条数帽由折存根后新块替代 → 不该 seal"
        text = led.text()
        # 被折的 P1 块须以 trimmed 存根行在场（非消失、非硬删）
        assert "trimmed" in text, "P1 块被裁后须折 trimmed 存根行（不产生编号空洞）"
        # 编号恒连续（无跳号）
        import re
        rels = re.findall(r"^c#(\d+):", text, re.M)
        nums = [int(x) for x in rels]
        assert nums == sorted(nums) and len(nums) == len(set(nums)), f"块序号须连续：{nums}"

    def test_settled_trim_keeps_existing_stub(self, tmp_path):
        led = CompactLedger(str(tmp_path / "l2.compact.json"), max_tokens=200, max_entries=2)
        led.append("旧P0定案长块" * 50, settled=True, kind="adhoc")
        led.append("新P0块" * 50)
        # settled 块在渲染面恒折既有存根（_settled_stub_line）——现状已成立，回归锚定
        text = led.text()
        assert "settled sha256=" in text, "settled 块存根形态回归"


# ───────────────────────── (e) 占位符混入：T1 折叠契约不变 + 记账面展开 ─────────────────────────
class TestFenceContractUnchanged:
    """owner 裁定 #6 零回归：T1 折叠契约（_fold_code_fences）不变，小围栏照折。"""

    def test_small_fence_still_folds(self):
        """1-3 行小围栏仍折 [code: ×N lines]（既有 7 条 normalize 测试锁死语义）。"""
        small = "intro\n```python\na=1\nb=2\n```\n"
        out = _fold_code_fences(small)
        assert out is not None and "[code:python ×2 lines]" in out, "小围栏须照旧折叠（T1 契约）"

    def test_one_line_fence_folds(self):
        out = _fold_code_fences("```\na=1\n```")
        assert out is not None and "×1 lines" in out, "1 行围栏也折（既有契约）"

    def test_no_fence_passthrough(self):
        assert _fold_code_fences("no fence here") is None, "无围栏原样"

    def test_small_placeholder_detection(self):
        """记账面：[code: ×8 lines] 占位符（8 < 20）→ 判小，账本应展开原文。"""
        from sc.model_store import _is_small_code_placeholder
        assert _is_small_code_placeholder("[code:python ×8 lines] def foo():") is True

    def test_large_placeholder_kept(self):
        """[code: ×25 lines]（25 ≥ 20）→ 判大，维持残句进账本。"""
        from sc.model_store import _is_small_code_placeholder
        assert _is_small_code_placeholder("[code:python ×25 lines] x") is False

    def test_non_placeholder_not_matched(self):
        from sc.model_store import _is_small_code_placeholder
        assert _is_small_code_placeholder("普通结论段") is False

    def test_engine_expands_small_placeholder_in_ledger(self):
        """engine._expand_small_code_placeholders：小占位符段 → 原文；大占位符 → 残句。"""
        from engine import _expand_small_code_placeholders
        from sc.schema import Segment
        from sc.distill import WindowSeg
        small_seg = Segment(tier="P1", provenance="model", original_text="a=1\nb=2")
        small_seg.seq = 1
        big_seg = Segment(tier="P1", provenance="model",
                          original_text="\n".join(f"l{i}" for i in range(30)))
        big_seg.seq = 2
        rendered = [
            (WindowSeg(small_seg), "L3", "[code:python ×2 lines] a=1"),
            (WindowSeg(big_seg), "L3", "[code:python ×30 lines] l0..."),
        ]
        out = _expand_small_code_placeholders(rendered)
        assert out[0][2] == "a=1\nb=2", "小占位符段展开原文进账本"
        assert out[1][2] == "[code:python ×30 lines] l0...", "大占位符段维持残句"


# ───────────────────────── (f) 近重复跳过去重门 ─────────────────────────
class TestNearDupSkip:
    def test_same_head_prefix_skipped(self, tmp_path):
        """R5 批次 A #8 近重复门（sha256 全文哈希前 16 字节强判据版）：
        整块字面重生成（同 compact 事件重算 staging 块、同模板清点原文逐字不变）→ 跳过；
        "同任务详版+精简版"开首同但整体字面不同 → 不拦（sha256 不同 = 合法新块，照入账）。
        既有的 40 字符开首前缀碰撞判据（owner 批次 A 初版）误杀 staging 窗口滚动前
        起点未动时重算的 P0 块（开首字面稳定、整体不同 → 被误跳 c#2 永不 append），
        改全文 sha256 前 16 字节：整体字面不变 = 真近重复（拦），整体不同 = 合法新块
        （不拦），误杀面归零。"""
        led = CompactLedger(str(tmp_path / "l3.compact.json"), max_tokens=99999, max_entries=8)
        # 整块字面重生成（原文逐字不变）→ sha256 碰撞 → 跳过（真近重复，幂等快退同源）
        e1 = led.append("全部就位。迁移完成，收尾清点：共 36 个 py 文件，含 15 plugin。")
        e2 = led.append("全部就位。迁移完成，收尾清点：共 36 个 py 文件，含 15 plugin。")
        assert e1 is not None
        assert e2 is None, "整块字面重生成须跳过（sha256 全文哈希前缀碰撞，旧块信息超集照留）"
        assert len(led.entries) == 1
        # "同任务详版+精简版"开首同但整体字面不同 → sha256 不同 → 合法新块，不拦
        e3 = led.append("全部就位。迁移完成，收尾清点：共 36 个 py 文件，含 15 plugin + 精简备注。")
        assert e3 is not None, "同任务详版/精简版开首同但整体不同字面 → 合法新块，近重复门不拦"
        assert len(led.entries) == 2

    def test_distinct_heads_both_kept(self, tmp_path):
        led = CompactLedger(str(tmp_path / "l4.compact.json"), max_tokens=99999, max_entries=8)
        assert led.append("结论A：全量 133 passed。") is not None
        assert led.append("结论B：全量 218 passed。") is not None
        assert len(led.entries) == 2

    def test_dedup_only_against_unsettled(self, tmp_path):
        # 已 settled 块折存根后开首变了——去重门只看未 settled 块的原文开首
        led = CompactLedger(str(tmp_path / "l5.compact.json"), max_tokens=99999, max_entries=8)
        head = "同一任务清点："
        led.append(head + " v1", settled=True)
        e = led.append(head + " v2")
        assert e is not None, "settled 块已折存根，去重不对比它（v2 照入账）"


# ───────────────────────── (f1) 跨进程 flock 写路径（owner 既定决议 #4）─────────────────────────
class TestLedgerFlock:
    """F1（reviewer 实测 + owner 决议 #4）：`_persist` 的 read-RMW-write（tmp+os.replace）
    在专属 flock（`<dir>/.ledger.flock`）临界区内，跨进程并发 append 串行化；
    锁面 fail-open（锁文件 open/flock 失败 → 退化裸 RMW，不炸宿主）。"""

    def test_persist_creates_dedicated_flock_file(self, tmp_path):
        import sc.compact_ledger as cl
        led_path = str(tmp_path / "s1.compact.json")
        led = cl.CompactLedger(led_path, max_tokens=99999, max_entries=8)
        led.append("块A 网关限流阈值调至五百请求并配置告警")
        lock_path = tmp_path / ".ledger.flock"
        assert lock_path.exists(), \
            "F1：_persist 须创建专属 flock 文件 <dir>/.ledger.flock（不进账本 JSON）"

    def test_append_serialized_across_instances(self, tmp_path):
        """同进程内两个独立实例（模拟双 provider 并发）交替 append：
        flock 互斥 + 共享 load() 语义下 seq 单调、entries 无丢、无 seal 抢跑。"""
        import sc.compact_ledger as cl
        led_path = str(tmp_path / "s2.compact.json")
        a = cl.CompactLedger(led_path, max_tokens=99999, max_entries=64)
        b = cl.CompactLedger(led_path, max_tokens=99999, max_entries=64)  # 共享同盘
        # 交替 append（每步 re-load 对方盘状态，模拟宿主 compress 各进程视角）
        for i in range(4):
            a.append(f"块{i:02d} 部署网关限流版本{i}保留回滚入口")
            b.load()   # 同步 a 的写
            b.append(f"块b{i:02d} 回滚开关位置网关配置第三节")
            a.load()
        # seq 单调（a/b 共享 seq 面，后 load 者续号）：entries 全在、无洞
        a.load()
        seqs = [e["seq"] for e in a.entries]
        assert seqs == sorted(seqs), f"seq 须单调（无回退空洞）：{seqs}"
        assert len(a.entries) == 8, f"双实例 4×2=8 块全在（flock 串行化无丢块）：{len(a.entries)}"
        assert a.seals == 0, "无封帽（token/条数帽 99999/64 远未超）seal 不抢跑"

    def test_fail_open_when_lock_unavailable(self, tmp_path, monkeypatch):
        """锁文件 open 失败 → fail-open 退化裸 RMW（零行为回归，不炸宿主）。"""
        import sc.compact_ledger as cl
        led_path = str(tmp_path / "s3.compact.json")
        led = cl.CompactLedger(led_path, max_tokens=99999, max_entries=8)
        # 让 _flock_ledger 的 open 抛 OSError（模拟只读目录/权限拒绝）
        import builtins
        real_open = builtins.open
        def _denied(path, *a, **kw):
            if ".ledger.flock" in str(path):
                raise PermissionError("simulated: lock file open denied")
            return real_open(path, *a, **kw)
        monkeypatch.setattr(builtins, "open", _denied)
        try:
            led.append("块X fail-open 写路径照跑（锁不可用不炸）")
        finally:
            monkeypatch.undo()
        assert led.entries and led.entries[-1]["text"].startswith("块X"), \
            "F1 fail-open：锁不可用退化裸 RMW，append 仍落（零宿主回归）"


# ───────────────────────── (p1) 跨进程 load 竞态补全（reviewer 二次复盘 P1）─────────────────────────
class TestLedgerLoadFlock:
    """P1（reviewer 二次复盘 major）：第一轮 F1 只给 `_persist` 加了 flock，`load()`
    读面零锁 → 双宿主冷重开 `load→内存 append→flock persist` 时后 load 者读到
    stale 盘状态、append 到旧 seq 上、flock 写盘时覆盖先写者 = 丢块（reviewer P1
    实测"磁盘只剩 B1、A1 丢失"）。P1 修法：`load()` 的读盘+解析同样进专属 flock
    临界区（LOCK_EX 短临界区），读侧 stale read 面归零。

    **合法部署形态 = 先后 compress（双宿主 / 双 provider 不**同时** compress 同一
    session）**：flock 串行化"先后"的两个宿主，B 的 load 读到 A 写好的完整盘状态、
    续 seq、不丢块。跨进程**真并发同时** compress 同一 session 是**非法部署形态**
    （宿主契约：一个 session 一个 provider 在 compress，engine 会话面保证），不在
    ledger 层为它做跨调用持锁——同 session 单 provider 由宿主层保证。"""

    def test_sequential_two_processes_no_loss(self, tmp_path):
        """先后（非同时）双宿主 compress 同一 session：B load 读到 A 写好的盘、
        续 seq、8 块全在、无封帽抢跑。flock 护 load+persist 把"先后"竞态治住。"""
        import subprocess, sys, os, textwrap
        led_path = str(tmp_path / "conc.compact.json")
        script = textwrap.dedent(f"""
            import sys, os
            sys.path.insert(0, {os.getcwd()!r})
            from sc.compact_ledger import CompactLedger
            l = CompactLedger({led_path!r}, max_tokens=999999, max_entries=64)
            # 本宿主 load()（flock 护读面）读到既有盘状态 → 内存 append 4 块
            # → flock persist（写面串行化）。返回写后 seq。
            for i in range(4):
                l.append(os.environ.get("PROCTAG", "?") + str(i))
            print(l.seq, end="")
        """)
        # 先后（sequential）跑两宿主：A 完整跑完再跑 B（B load 读到 A 的 4 块）
        seqs = []
        for tag in ("A", "B"):
            env = dict(os.environ, PROCTAG=tag)
            pr = subprocess.Popen([sys.executable, "-c", script],
                                  env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
            out, _ = pr.communicate(timeout=30)
            assert pr.returncode == 0, f"{tag} 宿主 append 不崩"
            seqs.append(int(out.strip()))
        assert seqs == [4, 8], \
            f"先后 compress：A 写后 seq=4、B load 读 A 续到 seq=8（flock 护 load 面）：{seqs}"
        # 主进程 reload 盘上状态（flock 护读面）：8 块全在、seq 单调、无 seal
        led = CompactLedger(led_path, max_tokens=999999, max_entries=64)
        n_a = sum(1 for e in led.entries if e["text"].startswith("A"))
        n_b = sum(1 for e in led.entries if e["text"].startswith("B"))
        assert n_a == 4 and n_b == 4, \
            f"先后双宿主 8 块全在（flock 护 load+persist，无丢块）：A={n_a} B={n_b}"
        seqs2 = [e["seq"] for e in led.entries]
        assert seqs2 == sorted(seqs2), f"seq 须单调无回退空洞：{seqs2}"
        assert led.seals == 0, "条数帽 64 远未超，seal 不抢跑"

    def test_concurrent_same_session_is_illegal_deployment(self, tmp_path):
        """**文档面（非行为断言）**：跨进程**同时** compress 同一 session 是非法
        部署形态（宿主契约：一个 session 单 provider compress，engine 会话面保证）。
        本测试不构造并发（那需要 spawn 同时跑，非本面责任），只断言 load docstring
        已明写该边界——防止后人误以为 ledger 的 flock 能护跨进程真并发。"""
        import inspect
        src_doc = inspect.getsource(CompactLedger.load)
        assert ("非法" in src_doc or "单 provider" in src_doc
                or "宿主层" in src_doc or "先后" in src_doc), \
            "P1：load docstring 须明写 flock 护先后 compress、跨进程真并发由宿主层" \
            "同 session 单 provider 保证（非 ledger 层责任）"


# ───────────────────────── (p2) 长会话 ledger tier-split 用展开面（reviewer 二次复盘 P2）─────────────
class TestLedgerLongSessionUsesExpanded:
    """P2（reviewer 二次复盘 major）：R5 #6 占位符展开只接进短会话 S1 面（`_staged`
    用 `_expanded`），长会话 ledger tier-split（`_staged_by_tier`）误传未展开的
    `rendered` → 长会话账本 P1 块的小代码占位符 `[code: ×N lines]` 照样混入
    （#6 在长会话面漏修）。修法：engine.compress() 长会话路径 tier-split 改传
    `_expanded`（已构造，零新计算）。本测试锁 engine 侧：tier-split 入参是展开后
    的 rendered（小占位符段 = original_text 全文，非占位符残句）。"""

    def test_long_session_tier_split_uses_expanded_rendered(self):
        """engine.compress() 长会话路径的 tier-split 必须喂 _expanded（展开面），
        不能喂 rendered（未展开面）——源码面断言（锁定调用面，防回退）。"""
        import re
        src = open("engine.py", encoding="utf-8").read()
        # 定位 _staged_by_tier 调用：两处（P0 + P1）都须传 _expanded
        p0 = re.search(r"_staged_by_tier\(\s*(_\w+)\s*,\s*\"P0\"\s*\)", src)
        p1 = re.search(r"_staged_by_tier\(\s*(_\w+)\s*,\s*\"P1\"\s*\)", src)
        assert p0 and p1, "engine 须有 _staged_by_tier P0/P1 两处调用"
        assert p0.group(1) == "_expanded", \
            f"P2：P0 tier-split 须喂展开面 _expanded（实得 {p0.group(1)}，未展开 rendered = 占位符漏修）"
        assert p1.group(1) == "_expanded", \
            f"P2：P1 tier-split 须喂展开面 _expanded（实得 {p1.group(1)}，未展开 rendered = 占位符漏修）"

    def test_small_placeholder_expands_for_ledger_not_residue(self):
        """行为面：小代码占位符段（[code: ×N lines]，N < 阈值）经
        _expand_small_code_placeholders 后 = original_text 全文（非残句占位符）。
        ledger 长会话路径喂 _expanded → P1 块里小代码段是全文，诊断账本 #6 的
        残句占位符不进账本。"""
        from engine import _expand_small_code_placeholders
        from sc.distill import WindowSeg
        from sc.schema import Segment
        import time
        seg = Segment(session_topic="t", original_text="def f():\n    return 1\n\ndef g():\n    pass",
                      tier="P1", provenance="model", seq=1, ts=time.time(),
                      content_hash="x"*32, kind="adhoc")
        # L3 渲染面给小代码占位符（模拟 T1 折叠产物：N=2 < 阈值 20）
        placeholder = "[code:python ×2 lines] def f():"
        rendered = [(WindowSeg(seg), "L3", placeholder)]
        out = _expand_small_code_placeholders(rendered)
        assert out[0][2].startswith("def f()"), \
            "小代码占位符段须展开为 original_text 全文（非残句）：实得 %r" % out[0][2][:40]
