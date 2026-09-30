"""Block ⑤ user 内容分级（@reviewer 裁定 2026-09-29）：
- user 短指令（<100 chars）→ P0 structural（确定性地板，短指令不误伤）；
- user 长内容（≥100 chars，group chat 转发 / compaction summary / 注入块）→ 非 structural
  （P0 non-structural，日志/技能提示词 P2，compaction P1）；
- 注入类内容 0% structural → push cap 可实际逐出（根治 100% structural 钉死）。

数据面：active.jsonl 实测 360/360 structural、73.4% 字符量为 group chat 转发，
全部打 provenance="user"——本 block 修复新写入判定 + retag.py 修复存量。
"""
import json

from sc.schema import Segment
from sc.tagger import Tagger, classify_user
from sc.distill import Distiller
from helpers import mkseg, complete_digest, mk_window, TOPIC, MV


class TestClassifyUser_Rule:
    def test_short_user_instruction_kept_structural(self):
        tier, structural = classify_user("ping")
        assert (tier, structural) == ("P0", True), "<100 短指令 = structural 地板"

    def test_99_chars_boundary_still_structural(self):
        tier, structural = classify_user("x" * 99)
        assert (tier, structural) == ("P0", True)

    def test_100_chars_boundary_long_content(self):
        tier, structural = classify_user("x" * 100)
        assert (tier, structural) == ("P0", False), "≥100 → 可逐出（tier 不变，仅解除钉死）"

    def test_long_group_chat_injection_p0_non_structural(self):
        text = "[Group chat: \"room\"] You are @coder..." + " msg " * 100
        tier, structural = classify_user(text)
        assert tier == "P0" and structural is False, "注入块不钉死，push cap 可逐出"

    def test_compaction_summary_long_p1(self):
        text = "<compaction-summary> " + "goal state decisions " * 30
        tier, structural = classify_user(text)
        assert tier == "P1" and structural is False, "已压缩摘要原文冗余 → P1"

    def test_skill_update_prompt_long_p2(self):
        text = "Review the conversation above and update the skill library. " + "notes " * 30
        tier, structural = classify_user(text)
        assert tier == "P2" and structural is False, "系统噪声 → P2"

    def test_traceback_long_p2(self):
        text = "Traceback (most recent call last):\n  File ..." + " frame " * 30
        tier, structural = classify_user(text)
        assert tier == "P2" and structural is False


class TestTaggerClassify_Integration:
    def test_user_short_via_classify(self, aux):
        tier, structural, regime, _ = Tagger(aux, TOPIC).classify("看看这个 PR", "user")
        assert (tier, structural, regime) == ("P0", True, "normal")

    def test_user_long_via_classify(self, aux):
        tier, structural, regime, _ = Tagger(aux, TOPIC).classify("y" * 5000, "user")
        assert tier == "P0" and structural is False, "长 user 内容非 structural（确定性，不过 aux）"

    def test_user_classify_never_calls_aux(self, aux):
        ep = aux.main_endpoint
        Tagger(aux, TOPIC).classify("z" * 5000, "user")
        assert not ep.calls, "user 分级是纯确定性规则，不发起 LLM 调用（零外呼）"


class TestDistill_EvictablePushCap:
    def test_long_user_p0_evicted_under_push_cap(self, store, aux):
        """回归根因：长 user P0 曾被钉死 → push cap 无法逐出 → compress 膨胀被拒。
        修复后 digest ready 的长 user P0 可被 push cap 逐出（non-structural）。"""
        w = mk_window(100, [
            mkseg(store, "group chat dump " * 100, "P0"),   # 360 tok ≈ P0 常驻
        ])
        w.segs[0].segment.digest_ref = "pending"
        complete_digest(aux, w.segs[0].segment)
        d = Distiller(w, store, aux).run()
        assert w.segs[0].form == "digest_only", "长 user 注入块已可逐出（不再钉死）"

    def test_short_user_structural_still_never_evicted(self, store, aux):
        u = mkseg(store, "ship v1", "P0", "user", structural=True)
        lp0 = mkseg(store, "llm p0 " * 30, "P0")
        complete_digest(aux, lp0)
        p1 = mkseg(store, "p1 ctx " * 30, "P1")
        p2 = mkseg(store, "p2 dump " * 30, "P2")
        w = mk_window(100, [u, lp0, p1, p2])
        d = Distiller(w, store, aux).run()
        forms = {s.segment_id: s.form for s in w.segs}
        assert forms[u.segment_id] == "resident", "短指令 structural 地板仍永不逐出"


class TestRetag_ExistingEntries:
    """D：存量重打分（retag.py 单一入口，规则与 classify_user 同源）。"""

    def test_retag_module_exists_and_shares_rule(self):
        import retag
        assert callable(getattr(retag, "retag_file", None)), "retag.py 暴露 retag_file()"

    def test_retag_flips_long_user_p0(self, tmp_path):
        import retag
        p = tmp_path / "active.jsonl"
        lines = []
        for text in ["ping", "ship it", "group chat dump " * 100, "y" * 200]:
            d = {"tier": "P0", "provenance": "user", "original_text": text,
                 "structural_p0": True, "seq": len(lines), "kind": "segment",
                 "ts": 0.0, "session_topic": "", "digest_ref": "none",
                 "model_version": "m1", "reopened_from": None, "stale": False,
                 "goal": None, "phase_id": None, "state": None,
                 "decisions": [], "open_tasks": [], "guardrail": None}
            import unicodedata, hashlib
            d["content_hash"] = hashlib.sha256(unicodedata.normalize("NFC", text).encode()).hexdigest()
            lines.append(json.dumps(d, ensure_ascii=False))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = retag.retag_file(str(p))
        assert report["changed"] >= 1, "长 user P0 必须被翻转为 non-structural"
        new = [json.loads(l) for l in p.read_text().splitlines() if l]
        short = [x for x in new if len(x["original_text"]) < 100]
        long_ = [x for x in new if len(x["original_text"]) >= 100]
        assert all(x["structural_p0"] is True for x in short), "短指令保持 structural"
        assert all(x["structural_p0"] is False for x in long_), "长内容解除钉死"
        # 注入类内容 0% structural（@reviewer 验收门 #2）
        assert not any(x["structural_p0"] for x in long_), "注入类 0% structural"
