"""T 卡（t_d3a6c58e）：入库 gate 归一（裸 JSON / 代码围栏）+ append-only supersede 存量清洗契约测试。

覆盖（T1/T2 钉死面；LF 新文件，sc 层契约面确定性 = 零 LLM / 零 wall-clock）：
- gate_normalize_top_level_json：裸 JSON dict/list 回复 → 入库段 original_text = 归一行
  （key 索引 + 第一段可读 verdict），不含原始 JSON 全文；key_fact = "norm:json"；
  JSON 值内嵌 ``` 围栏 → 反引号中和（脏水特征不穿透 verdict 单行）；
- gate_normalize_code_fence：含 ``` 围栏回复 → 折叠为单行引用 [code:<lang> ×N lines] 首行，
  围栏体不全文入库（围栏外散文同折叠丢弃 = 卡片「单行引用」语义）；key_fact = "norm:fence"；
- gate_fail_open：JSON 解析失败 / 围栏不配对 / 无特征 → 原文入库（fail-open，不阻塞、不改写）；
- gate_deterministic：同一输入两次归一结果逐字节相同（零 LLM 纯函数）；
- append_only_supersede：存量清洗产物 = 单物理行新段（行内 supersedes 标记）+ 原段行字节不变
  （重放 jsonl 验证原段哈希不变、行数只增不减、冷重开后原段出召回池、幂等不重复 append）；
- regression_existing_gate：现有 gate 行为（空文本 / dedup / len<200 无 marker / marker 恒持久 /
  普通长文原文入库）不变。

跑法（repo 根）：
  python -m pytest tests/test_model_store_normalize.py -v
"""
from __future__ import annotations

import json

from sc.model_store import ModelStore, normalize_model_response
from sc.schema import Segment, nfc_hash
from sc.store import JsonlStore

MIN = 200  # 默认 model_min_chars（与 test_model_store.py 同口径）


def _model_path(tmp_path, sid: str = "s") -> str:
    return str(tmp_path / f"{sid}.model.jsonl")


def _pad(text: str) -> str:
    """补齐到 ≥ MIN 字符（gate 的 chars 支；不引入 JSON/围栏特征）。"""
    if len(text) >= MIN:
        return text
    return text + " pad" + "z" * (MIN - len(text) - 4)


def _read_rows(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in f.read().splitlines() if ln.strip()]


def _ingest_json_response(ms: ModelStore, payload, min_chars: int = MIN) -> tuple:
    return ms.ingest_assistant(json.dumps(payload, ensure_ascii=False),
                               min_chars=min_chars, model_version="m1", main_by_hash={})


# ═══════════════════════ T1(a) 顶层裸 JSON 归一 ═══════════════════════


class TestGateNormalizeTopLevelJson:
    """裸 JSON dict/list 回复 → 只持久化「key 名索引 + 第一段可读 verdict」单行，不存原文全文。"""

    def test_dict_response_collapses_to_key_index_plus_verdict(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        payload = {
            "bug_confirmed": True,
            "mechanism": "主 system prompt 走 in-list：turn_context.build_api_messages 在 "
                        "L1285-1286 把 effective_system 拼进 api_messages 头部",
            "anchors": ["engine.py:1285", "turn_context.py:33"],
        }
        ok, gate = _ingest_json_response(ms, payload)
        assert ok and gate in ("chars", "marker")
        seg = ms.live_segments()[0]
        assert seg.original_text == "[model-json] keys=bug_confirmed,mechanism,anchors · bug_confirmed=true"
        assert seg.original_text.startswith("[model-json]")
        # 原始 JSON 全文不入库（长值片段 / 嵌套 list 值必须消失，只留首段标量 verdict）
        assert "turn_context.build_api_messages" not in seg.original_text
        assert "engine.py:1285" not in seg.original_text
        assert "bug_confirmed=true" in seg.original_text
        assert seg.key_fact == "norm:json", "轻标记（卡片预案 B：零 schema 变更，key_fact 运行期字段）"
        # 入库段 content_hash = 归一后字节的 NFC 哈希（行字节自洽）
        assert nfc_hash(seg.original_text) == seg.content_hash
        assert "\n" not in seg.original_text, "单行不变式"

    def test_list_response_key_index(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        # 顶层 list 裸 JSON（≥200 字符才过 chars gate；不 pad——pad 会破坏顶层 json.loads）
        payload = [
            "first-item-diagnostic-" + "a" * 30,
            42,
            "third-item-" + "b" * 150,
        ]
        raw = json.dumps(payload, ensure_ascii=False)
        assert len(raw) >= MIN
        ok, _ = ms.ingest_assistant(raw, min_chars=MIN, model_version="m1", main_by_hash={})
        assert ok
        seg = ms.live_segments()[0]
        assert seg.original_text.startswith("[model-json]")
        assert "3 items" in seg.original_text           # list 索引形态
        assert "first-item-diagnostic" in seg.original_text   # 第一段可读 scalar verdict
        assert raw not in seg.original_text             # 原始数组全文不入库
        assert "\n" not in seg.original_text
        assert seg.key_fact == "norm:json"
        assert nfc_hash(seg.original_text) == seg.content_hash

    def test_verdict_line_truncated_to_cap(self, tmp_path):
        """整行 ≤ 200 字符，超则截断加 ...（卡片 verdict 行 ≤200 字符）。"""
        ms = ModelStore(_model_path(tmp_path), sync=True)
        payload = {f"field_{i:02d}_name": i for i in range(30)}  # 30 键 → 整行 > 200 必截断
        raw = json.dumps(payload, ensure_ascii=False)
        assert len(raw) >= MIN
        _ingest_json_response(ms, payload)
        seg = ms.live_segments()[0]
        assert len(seg.original_text) <= 203, "整行截断到 cap（≤200 + '...'）"
        assert seg.original_text.endswith("...")
        assert seg.original_text.startswith("[model-json]")
        assert "\n" not in seg.original_text

    def test_no_scalar_values_keys_only(self, tmp_path):
        """全嵌套/null 值 = 无首段可读标量 verdict → 只留 key 索引（无 ' · ' verdict 段）。"""
        ms = ModelStore(_model_path(tmp_path), sync=True)
        payload = {"k1": {"nested": "x" * 300}, "k2": None}   # 原始 JSON ≥200，但顶层无标量值
        raw = json.dumps(payload, ensure_ascii=False)
        assert len(raw) >= MIN
        ms.ingest_assistant(raw, min_chars=MIN, model_version="m1", main_by_hash={})
        seg = ms.live_segments()[0]
        assert seg.original_text == "[model-json] keys=k1,k2"
        assert " · " not in seg.original_text, "无首段可读标量 → 无 verdict 段"
        assert "xxxx" not in seg.original_text, "嵌套内容不入库"
        assert seg.key_fact == "norm:json"

    def test_json_wins_over_embedded_fences(self, tmp_path):
        """脏水重叠形态：顶层 JSON 的 string 值内嵌 ``` 围栏 → JSON 归一获胜（审计 4 段重叠口径），
        不存原始 JSON 全文；形态对齐审计示例（首键 = 短布尔 → verdict 短读）。"""
        ms = ModelStore(_model_path(tmp_path), sync=True)
        payload = {"ok": True, "code": "```python\n" + "def f():\n    pass\n" * 12, "note": "z" * 150}
        raw = json.dumps(payload, ensure_ascii=False)
        assert len(raw) >= MIN
        ms.ingest_assistant(raw, min_chars=MIN, model_version="m1", main_by_hash={})
        seg = ms.live_segments()[0]
        assert seg.original_text == "[model-json] keys=ok,code,note · ok=true"
        assert raw not in seg.original_text, "原始 JSON 全文不入库"
        assert "```" not in seg.original_text
        assert "\n" not in seg.original_text
        assert seg.key_fact == "norm:json"

    def test_newline_value_flattened_to_single_line(self, tmp_path):
        """JSON 值内嵌换行 + 围栏 → 折行压平 + 反引号中和，verdict 恒单行（不穿透单行不变式）。"""
        ms = ModelStore(_model_path(tmp_path), sync=True)
        payload = {"report": "line1\nline2\n```sql\nSELECT x\n```" + " filler" * 30, "k": 1}
        raw = json.dumps(payload, ensure_ascii=False)
        assert len(raw) >= MIN
        ms.ingest_assistant(raw, min_chars=MIN, model_version="m1", main_by_hash={})
        seg = ms.live_segments()[0]
        assert seg.original_text.startswith("[model-json] keys=report,k")
        assert "line1 line2 sql SELECT x" in seg.original_text  # 折行压平 + 反引号中和
        assert "```" not in seg.original_text and "\n" not in seg.original_text


# ═══════════════════════ T1(b) 代码围栏折叠 ═══════════════════════


class TestGateNormalizeCodeFence:
    """含 ``` 围栏回复 → 折叠为单行引用（语言标注 + 首行 + 行数），围栏体不全文入库。"""

    def test_fence_collapses_to_single_line_reference(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        text = _pad("结论先行：\n```python\nif effective_system:\n    api_messages = build(effective_system)\n    return api_messages\n```\n收尾说明")
        ok, gate = ms.ingest_assistant(text, min_chars=MIN, model_version="m1", main_by_hash={})
        assert ok
        seg = ms.live_segments()[0]
        assert seg.original_text == "[code:python ×3 lines] if effective_system:"
        assert "[code:python ×3 lines]" in seg.original_text
        assert "if effective_system:" in seg.original_text          # 首行保留
        assert "build(effective_system)" not in seg.original_text   # 围栏体不入库
        # 卡片「折叠为单行引用」：围栏外散文同折叠丢弃（单行 = 整段变一行）
        assert "结论先行" not in seg.original_text
        assert "收尾说明" not in seg.original_text
        assert "\n" not in seg.original_text
        assert seg.key_fact == "norm:fence"
        assert nfc_hash(seg.original_text) == seg.content_hash

    def test_multi_fences_fold_to_referenced_joined(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        text = _pad("A\n```sql\nSELECT 1\n```\nB\n```ts\nconst x = 1\n```\nC")
        ms.ingest_assistant(text, min_chars=MIN, model_version="m1", main_by_hash={})
        seg = ms.live_segments()[0]
        assert seg.original_text == "[code:sql ×1 lines] SELECT 1 [code:ts ×1 lines] const x = 1"
        assert "[code:sql ×1 lines] SELECT 1" in seg.original_text
        assert "[code:ts ×1 lines] const x = 1" in seg.original_text
        assert "\n" not in seg.original_text
        assert seg.key_fact == "norm:fence"

    def test_unlabeled_fence_plain(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        ms.ingest_assistant(_pad("```plain-ish\nline one\nline two\n```"),
                            min_chars=MIN, model_version="m1", main_by_hash={})
        seg = ms.live_segments()[0]
        assert seg.original_text == "[code:plain-ish ×2 lines] line one"
        assert "line two" not in seg.original_text


# ═══════════════════════ fail-open ═══════════════════════


class TestGateFailOpen:
    """失败（JSON 解析失败 / 围栏不配对 / 无特征）→ 原文入库（fail-open，不阻塞、不改写）。"""

    def test_invalid_json_ingests_verbatim(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        broken = '{"unclosed": [' + "x" * 300
        assert len(broken) >= MIN
        ok, gate = ms.ingest_assistant(broken, min_chars=MIN, model_version="m1", main_by_hash={})
        assert ok and gate == "chars"
        seg = ms.live_segments()[0]
        assert seg.original_text == broken          # 原文入库
        assert seg.key_fact == ""                  # 未归一 = 无标记

    def test_unpaired_fence_ingests_verbatim(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        unpaired = "```python\ndef f():\n    pass\n" + "w" * 300   # 无闭合围栏（奇数 marker）
        ok, gate = ms.ingest_assistant(unpaired, min_chars=MIN, model_version="m1", main_by_hash={})
        assert ok and gate == "chars"
        seg = ms.live_segments()[0]
        assert seg.original_text == unpaired
        assert seg.key_fact == ""

    def test_normal_prose_ingests_verbatim_no_marker(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        prose = "plain long answer " + "p" * 200
        ok, gate = ms.ingest_assistant(prose, min_chars=MIN, model_version="m1", main_by_hash={})
        assert ok and gate == "chars"
        seg = ms.live_segments()[0]
        assert seg.original_text == prose and seg.key_fact == ""


# ═══════════════════════ 确定性 ═══════════════════════


class TestGateDeterministic:
    """零 LLM 纯函数：同一输入两次归一结果逐字节相同。"""

    def test_same_input_byte_identical(self):
        t_json = json.dumps({"a": 1, "b": "x" * 300}, ensure_ascii=False)
        t_fence = _pad("P\n```python\nl1\nl2\n```\nQ")
        for t in (t_json, t_fence):
            n1, k1 = normalize_model_response(t)
            n2, k2 = normalize_model_response(t)
            assert (n1, k1) == (n2, k2), "逐字节 + 逐 kind 相同"

    def test_normalize_vs_ingest_agree(self, tmp_path):
        """入库字节 = normalize_model_response 归一字节（gate 面与纯函数面同源，零 LLM）。"""
        ms = ModelStore(_model_path(tmp_path), sync=True)
        payload = {"bug_confirmed": True, "mechanism": "m" * 300}
        raw = json.dumps(payload, ensure_ascii=False)
        ms.ingest_assistant(raw, min_chars=MIN, model_version="m1", main_by_hash={})
        seg = ms.live_segments()[0]
        expected, kind = normalize_model_response(raw)
        assert seg.original_text == expected
        assert kind == "json"


# ═══════════════════════ T3 append-only supersede ═══════════════════════


class TestAppendOnlySupersede:
    """T3 语义：存量清洗产物 = 单物理行新段（行内 supersedes 标记）+ 原段行字节不变（append-only 不变式）。"""

    def _dirty_segment(self, ms: ModelStore, dirty_text: str) -> Segment:
        """模拟存量脏段（pre-deploy 原文入库行）：绕过 gate 直接 append 原始围栏文。"""
        return ms.store.append(Segment(tier="P0", provenance="model",
                                       original_text=dirty_text, structural_p0=False))

    def test_supersede_appends_single_row_and_preserves_original_bytes(self, tmp_path):
        p = _model_path(tmp_path)
        ms = ModelStore(p, sync=True)
        dirty = "```sql\nSELECT id FROM t\n```\n" + "w" * 300
        orig = self._dirty_segment(ms, dirty)
        pre_rows = _read_rows(p)
        assert len(pre_rows) == 1

        norm, kind = normalize_model_response(dirty)
        assert kind == "fence"
        new_seg = ms.supersede(orig.segment_id, norm, model_version="m1")
        assert new_seg is not None
        assert new_seg.segment_id != orig.segment_id
        assert nfc_hash(norm) == new_seg.content_hash

        post_rows = _read_rows(p)
        # append-only：原段行字节不变（前缀逐行对照）+ 只增 1 行
        assert post_rows[:len(pre_rows)] == pre_rows
        assert len(post_rows) == len(pre_rows) + 1
        new_row = json.loads(post_rows[-1])
        assert new_row["supersedes"] == orig.segment_id, "行级 supersedes 标记（原段行绝不改写）"
        assert new_row["content_hash"] == nfc_hash(norm)

        # 原段出召回池（live_segments 过滤）但保留：L0 点查仍命中 + 磁盘行仍在
        assert all(s.segment_id != orig.segment_id for s in ms.live_segments())
        assert ms.get(orig.segment_id) is orig
        assert ms.line_count == 2

    def test_supersede_rebuilds_from_rows_on_cold_reopen(self, tmp_path):
        p = _model_path(tmp_path)
        ms = ModelStore(p, sync=True)
        dirty = "```ts\nconst y = 2\n```\n" + "v" * 300
        orig = self._dirty_segment(ms, dirty)
        ms.supersede(orig.segment_id, normalize_model_response(dirty)[0], model_version="m1")

        ms2 = ModelStore(p, sync=True)  # 冷重开：_superseded 从行键重建
        assert orig.segment_id in ms2._superseded
        assert all(s.segment_id != orig.segment_id for s in ms2.live_segments())
        assert ms2.line_count == 2       # 行数只增不减
        assert ms2.get(orig.segment_id) is not None, "原段行保留（点查面不变）"

    def test_supersede_is_idempotent(self, tmp_path):
        p = _model_path(tmp_path)
        ms = ModelStore(p, sync=True)
        dirty = "```python\nl1\n```\n" + "q" * 300
        orig = self._dirty_segment(ms, dirty)
        norm = normalize_model_response(dirty)[0]
        ms.supersede(orig.segment_id, norm, model_version="m1")
        ms2 = ModelStore(p, sync=True)
        res = ms2.supersede(orig.segment_id, norm, model_version="m1")  # 归一段已在池 → 幂等不重 append
        assert res is not None
        assert ms2.line_count == 2, "重复清洗 = 0 新增行（nfc_hash 去重，幂等）"

    def test_supersede_missing_original_is_noop(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        assert ms.supersede("no-such-segment-0", "anything", model_version="") is None
        assert ms.line_count == 0


# ═══════════════════════ 既有 gate 回归 ═══════════════════════


class TestRegressionExistingGate:
    """现有 gate 行为不变（空文本 / dedup / len<200 无 marker / marker 恒持久 / chars 长文）。"""

    def test_blank_and_short_gates_unchanged(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        assert ms.ingest_assistant("", min_chars=MIN, model_version="m1", main_by_hash={}) == (False, "blank")
        assert ms.ingest_assistant("  \n\t", min_chars=MIN, model_version="m1", main_by_hash={}) == (False, "blank")
        assert ms.ingest_assistant("x" * 50, min_chars=MIN, model_version="m1", main_by_hash={}) == \
            (False, "below_min_no_marker")

    def test_dedup_semantics_unchanged(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        assert ms.ingest_assistant("A" * 300, min_chars=MIN, model_version="m1", main_by_hash={}) == (True, "chars")
        assert ms.ingest_assistant("A" * 300, min_chars=MIN, model_version="m1", main_by_hash={}) == (False, "dedup")
        main = JsonlStore(str(tmp_path / "s.jsonl"))
        main.append(Segment(tier="P0", provenance="user", original_text="B" * 300))
        assert ms.ingest_assistant("B" * 300, min_chars=MIN, model_version="m1",
                                   main_by_hash=main.by_hash) == (False, "dedup")

    def test_marker_gate_still_always_persists(self, tmp_path):
        ms = ModelStore(_model_path(tmp_path), sync=True)
        short = "<compaction-summary> tiny "
        assert len(short) < MIN
        assert ms.ingest_assistant(short, min_chars=MIN, model_version="m1", main_by_hash={}) == (True, "marker")
        tb = "Traceback (most recent call last): KeyError"
        assert len(tb) < MIN
        assert ms.ingest_assistant(tb, min_chars=MIN, model_version="m1", main_by_hash={}) == (True, "marker")

    def test_persisted_row_shape_unchanged_for_clean_text(self, tmp_path):
        """非特征长文：行信封键 / tier / prov / structural_p0 与 B 卡契约逐键一致（归一零副作用面）。"""
        ms = ModelStore(_model_path(tmp_path), sync=True)
        ms.ingest_assistant("clean long model answer " + "k" * 250,
                            min_chars=MIN, model_version="m1", main_by_hash={})
        row = json.loads(_read_rows(_model_path(tmp_path))[0])
        assert row["tier"] == "P0" and row["provenance"] == "model"
        assert row["structural_p0"] is False and row["model_version"] == "m1"
        assert "supersedes" not in row, "常规入库行不带 supersedes 键（仅清洗行带）"

    def test_post_normalize_dedup_seals_double_write(self, tmp_path):
        """归一后同文双写封死：两段**不同原文**归一到同一归一行 → 第二段 dedup（store 以归一 hash 为键）。
        围栏折叠只看「语言 + 行数 + 首行」，故首行/行数相同而后续行不同的两段折叠为同一归一行。"""
        ms = ModelStore(_model_path(tmp_path), sync=True)
        f1 = _pad("X\n```python\nfirst_line\nsame\n```\n")
        f2 = _pad("Y\n```python\nfirst_line\ndiff\n```\n")
        assert f1 != f2
        ok1, _ = ms.ingest_assistant(f1, min_chars=MIN, model_version="m1", main_by_hash={})
        ok2, why2 = ms.ingest_assistant(f2, min_chars=MIN, model_version="m1", main_by_hash={})
        assert ok1 and ok2 is False and why2 == "dedup", "归一后同文（首行+行数同）第二段 dedup"
        assert ms.line_count == 1
