"""R2-v2（2026-10-10 · 语言无关化）e2e 回归。

owner option c·b + X1 拍板后落地：
- is_agent_self_report(text, provenance="") 加 provenance 硬门：
  非 model（user/""/缺省）→ 退回 v1 纯 marker 兼容面（不误删 user 段）；
  model → 命中 _SELF_REF_CUES（多语言自指结构词）∪ v1 marker 任一 → 逐出。
- _AGENT_SELF_REPORT_MARKERS 删 7 条"引用拍板继续推进"宽词（X1），保留 33 条中/英子集。
- rerank EXPLICIT_SEARCH_KEYWORDS 多语化（18 词，中英日韩德西法俄）。

红线（owner）：零数据变更 / 可回滚 / 不误删 user 段（provenance 门保护）/
宁漏勿错。宿主侧 conversation_loop.py 等原生文件不得碰。
"""
from __future__ import annotations

import pytest

from engine import is_agent_self_report, _SELF_REF_CUES, _AGENT_SELF_REPORT_MARKERS
from sc.rerank import EXPLICIT_SEARCH_KEYWORDS


class TestProvenanceGateProtectsUser:
    """provenance != 'model' → user 段含任何自指词都不误删。"""

    @pytest.mark.parametrize("prov", ["user", "", None, "other"])
    def test_user_segment_with_self_ref_cues_not_cut(self, prov):
        # 含时序自指词（earlier/上一轮）但 provenance=user → False
        assert is_agent_self_report("Please reconsider the earlier threshold you set", prov) is False
        assert is_agent_self_report("上一轮的方案我再改改", prov) is False
        assert is_agent_self_report("我刚才说的阈值你按 0.92 改", prov) is False

    def test_default_empty_provenance_is_v1_compat_face(self):
        # 缺省 ""（旧调用点/旧测试）= v1 纯 marker 面（向后兼容）
        assert is_agent_self_report("先认个账", "") is True
        assert is_agent_self_report("我刚才说的阈值你按 0.92 改", "") is False


class TestModelSelfReport:
    """provenance == 'model' → agent 自述（认账/复述/多语言时序自指）逐出。"""

    @pytest.mark.parametrize("text", [
        "先认个账：这一轮 supercompress 把我前两轮的复述当\"已确认\"回放给我了",
        "I just restated my earlier recap to the user",
        "전 단계의 확인 방향을 다시 재생했습니다",
        "die bisherige Bestätigung habe ich erneut abgespielt",
        "la confirmación anterior la reproduce otra vez",
        "前回の指示をまたリプレイした",
    ])
    def test_model_self_report_cut(self, text):
        assert is_agent_self_report(text, "model") is True

    def test_normal_model_segment_not_cut(self):
        # 正常 agent 段（无自指/认账）→ 不删（宁漏勿错）
        assert is_agent_self_report("Let me search the repo for the confirmed flag", "model") is False
        # X1：正常 agent 引用拍板方向继续推进 → 不删（删掉的宽词生效）
        assert is_agent_self_report("好，按你拍板的四点出第一版方案", "model") is False
        assert is_agent_self_report("这版按你确认的做", "model") is False


class TestX1BroadWordsRemoved:
    """X1：7 条'引用拍板继续推进'宽词已从 marker 表删除（可追溯）。"""

    def test_broad_words_not_in_markers(self):
        removed = ("跟你拍板", "跟你确认", "跟你定",
                   "这版按你", "按你拍板", "按你确认", "按你拍")
        for w in removed:
            assert w not in _AGENT_SELF_REPORT_MARKERS, f"{w} 不应在 marker 表（X1 删除）"

    def test_marker_table_size(self):
        # v1 原表 40 条（中英 marker），X1 删 7 条宽词（"引用拍板继续推进"类）后 = 33 条。
        # 表实际长度钉死在 33（防未来误加宽词回到"宁误删"面）。
        assert len(_AGENT_SELF_REPORT_MARKERS) == 33, len(_AGENT_SELF_REPORT_MARKERS)


class TestSelfRefCuesMultilingual:
    """_SELF_REF_CUES 多语言自指结构词表（结构判定，非逐语言堆 marker）。"""

    def test_covers_major_languages(self):
        joined = " ".join(_SELF_REF_CUES)
        # 中 / 英 / 日 / 韩 / 德 / 西 各至少 1 个高频自指词
        assert "上一轮" in joined  # 中
        assert "earlier" in joined  # 英
        assert "前の" in joined or "前回の" in joined or "さきほど" in joined  # 日
        assert "이전" in joined or "방금" in joined  # 韩
        assert "vorherige" in joined or "bisherige" in joined  # 德
        assert "anterior" in joined or "antecedente" in joined  # 西


class TestRerankMultilingualKeywords:
    """rerank gate② 显式搜索词多语化（修单语言下其他语言搜索意图漏触发）。"""

    def test_covers_languages(self):
        joined = " ".join(EXPLICIT_SEARCH_KEYWORDS)
        assert "search" in joined            # 英
        assert "検索" in joined             # 日
        assert "검색" in joined             # 韩
        assert "durchsuchen" in joined      # 德
        assert "buscar" in joined           # 西
        assert "recherche" in joined        # 法
        assert "поиск" in joined            # 俄

    def test_superset_of_v1(self):
        # 原 3 词仍在（向后兼容）
        for kw in ("搜一下", "搜下", "search"):
            assert kw in EXPLICIT_SEARCH_KEYWORDS
