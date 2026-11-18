"""R2（2026-10-23 data-engineer 复盘项 #3）：`i'm ` 独白词批量修复面。

背景：2026-10-23 tagger REASONING_MONOLOGUE_CUES 加 "i'm " 独白词后，
**历史** store jsonl 里已有的 model P0 段（开首 "I'm checking/noticing…"）
仍按旧 tagger 留在 P0 注入面，而新入库段会被降 P2 → 同 session 内新旧
tier 断层（data-engineer 结论 R2：可观测、可 batch 修复）。

修法（owner 极简偏好，零新状态面）：sc/retag.py 加 `retag_monologue` ——
对 store 里 P0 非结构 model 段重跑 `is_reasoning_monologue`，命中者降 P2
并落盘（tier 是可变字段，content_hash 只验 original_text，replay 不重跑
tiering → 批量重打是唯一一致修复面）。dry_run 默认安全（只报不写）。
"""
from __future__ import annotations

import json

from helpers import mkseg
from sc.tagger import is_reasoning_monologue

# 层级校验面：sc/retag.py 模块级 require_install_root()——独立部署（非 repo
# plugins/context_engine/supercompress 层级）下 import 即 RuntimeError → 整文件 skip。
import pytest
try:
    from sc import retag as _sc_retag  # noqa: F401  触发层级校验
except RuntimeError as _e:
    pytest.skip(f"安装层级校验不适用（独立部署）: {_e}", allow_module_level=True)


class TestRetagMonologue:
    def test_demotes_monologue_p0_model_to_p2(self, store, tmp_path):
        """历史 P0 非结构 model 段开首命中独白词 → retag_monologue 降 P2 并落盘。"""
        s = mkseg(store, "I'm checking whether the regex works on the ledger line",
                  "P0", "model")
        path = store.path
        from sc.retag import retag_monologue
        flipped = retag_monologue(path, dry_run=False)
        assert len(flipped) == 1, f"应翻转 1 段（独白 P0 model），实得 {len(flipped)}"
        # 落盘重读验证 tier 已变
        reloaded = json.loads(open(path).read().strip().splitlines()[-1])
        assert reloaded["tier"] == "P2", f"落盘 tier 须 P2，实得 {reloaded['tier']}"

    def test_conclusion_segment_not_demoted(self, store):
        """开首结论句（i'm + 结论词 / 非独白开首）不降档。"""
        ok1 = mkseg(store, "I'm sure the terminal fix is done and verified", "P0", "model")
        ok2 = mkseg(store, "Terminal 修复完成，APK 已出", "P0", "model")
        from sc.retag import retag_monologue
        flipped = retag_monologue(store.path, dry_run=False)
        assert flipped == [], f"结论句不得降档，实得 {flipped}"

    def test_user_and_structural_untouched(self, store):
        """user 段 / structural P0 段免疫（独白词只对 model 非结构 P0 生效）。"""
        u = mkseg(store, "I'm checking the thing", "P0", "user")
        struct = mkseg(store, "I'm checking the thing", "P0", "model", structural=True)
        from sc.retag import retag_monologue
        flipped = retag_monologue(store.path, dry_run=False)
        assert flipped == [], f"user/structural 段免疫，实得 {flipped}"

    def test_dry_run_reports_without_writing(self, store):
        """dry_run（默认）：返回翻转清单但不写盘（字节不变）。"""
        s = mkseg(store, "I'm noticing a mismatch in the forms", "P0", "model")
        before = open(store.path).read()
        from sc.retag import retag_monologue
        flipped = retag_monologue(store.path)  # dry_run 默认 True
        assert len(flipped) == 1
        after = open(store.path).read()
        assert before == after, "dry_run 不得写盘"

    def test_idempotent_second_run_empty(self, store):
        """幂等：已降 P2 的段第二次跑不再翻转。"""
        mkseg(store, "I'm checking whether X", "P0", "model")
        from sc.retag import retag_monologue
        assert len(retag_monologue(store.path, dry_run=False)) == 1
        assert retag_monologue(store.path, dry_run=False) == []
