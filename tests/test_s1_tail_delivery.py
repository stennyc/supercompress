"""T3 · S1 尾投递 + P0 垫底（Plan C Phase 1）单测面。

对应卡 t_a7c9bff4 T3 + 验收 §3.3 的 7 条硬断言：
  s1_merge_last_user            末条 user → recall 并入该 user content（不新增 user 消息）
  s1_standalone_tail_on_tool    末条 tool → 末位追加独立 user（role=user）
  p0_at_tail_of_block           被召回 P0 段渲染在 recall 块最末（P1/P2/Pn 背景在前）
  openai_wire_layout            OpenAI wire assembled request 无第 2 条 in-list system（仅主 system 1 条）
  anthropic_no_second_inlist_system  模拟 Anthropic 路由组装（调 convert_messages_to_anthropic）：
                                     wire system 参数 = 主 prompt（非被 recall 覆盖），in-list system 数 = 1
  no_user_user_adjacent         wire dump 无 [user, user] 独立相邻对
  staged_block_s1_tail          compress() 出块面 staged 块同走 S1 尾投递（不在位置 2 发 system）

红线（@user）：只 import engine / sc，测试替身（FakeEndpoint）住 tests/；
不动 sc/* 生产、不动宿主 agent/*、不动 jsonl store。B 块（compress 选档/render）零改，
本测试仅断言其**投递位置**（system→user），不断言其内容/渲染逻辑。
"""
import os
import sys

import pytest

from sc.schema import Segment
from fakes import FakeEndpoint
from engine import SupercompressContextEngine


# ── 主 system prompt + 已证可召回的 P0/P1 段 + 强重叠 query（gate① 默认 3.0 不触发 LLM 面）──
_MAIN_PROMPT = "你是 ops 助手，负责网关部署与回滚，保留回滚入口。"
_P0_TEXT = "网关限流阈值调至每秒五百请求并配置告警"
_P1_TEXT = "部署发布版本四十二到生产环境保留回滚入口并输出工具结果"
_QUERY = "网关限流 部署版本 工具输出"


def _mkseg(text, tier="P1", seq=0):
    return Segment(tier=tier, provenance="model", original_text=text, seq=seq, ts=0.0)


def _seed_recallable(e):
    """seed 一条 P0 + 一条 P1（均对 _QUERY >0 分）→ 召回块 = [P1 背景, P0 垫底]。"""
    e.store.append(_mkseg(_P0_TEXT, tier="P0", seq=0))
    e.store.append(_mkseg(_P1_TEXT, tier="P1", seq=1))
    e.store.flush()   # P1 落盘 → _window_fresh replay 可见（P0 即时 fsync 已在盘）


def _recalled_texts(out):
    """取含 [supercompress recall] 标记的消息 content（S1 尾投递面），去标记按 \\n\\n 切段序。"""
    block = [m for m in out if "[supercompress recall]" in str(m.get("content", ""))][0]
    body = str(block["content"]).split("[supercompress recall]", 1)[1]
    return [t for t in body.strip().split("\n\n") if t]


@pytest.fixture()
def engine(tmp_path):
    e = SupercompressContextEngine(external_store_path=str(tmp_path), auto_recall=True)
    e._aux.main_endpoint = FakeEndpoint()   # 默认 fail-safe（无 LLM 重排 → 确定性）
    return e


# ─────────────────────────── S1 尾投递两种形态（select_context 出块面）───────────────────────────
class TestS1TailDelivery:
    def test_s1_merge_last_user(self, engine):
        """末条 user → recall 并入该 user content（合并，不新增 user 消息）。"""
        _seed_recallable(engine)
        req = [{"role": "system", "content": _MAIN_PROMPT},
               {"role": "user", "content": "x"},
               {"role": "assistant", "content": "ok"},
               {"role": "user", "content": "继续"}]
        out = engine.select_context(req, incoming_message={"role": "user", "content": _QUERY})
        assert out is not None, "召回非空应产出替换请求"
        assert len(out) == len(req), "S1 合并面：不新增 user 消息（len 不变）"
        assert out[-1]["role"] == "user"
        assert "[supercompress recall]" in out[-1]["content"], "recall 并入末条 user content"
        assert "继续" in out[-1]["content"], "原 user content 保留（合并非替换）"
        assert _recalled_texts(out), "recall 块非空"

    def test_s1_standalone_tail_on_tool(self, engine):
        """末条 tool → 末位追加一条独立 user（role=user，实际形态 mid-turn [tool→user]）。"""
        _seed_recallable(engine)
        req = [{"role": "user", "content": "x"},
               {"role": "assistant", "content": "call tool"},
               {"role": "tool", "content": "result-zzz"}]
        out = engine.select_context(req, incoming_message={"role": "user", "content": _QUERY})
        assert out is not None
        assert len(out) == len(req) + 1, "S1 独立尾面：追加一条 user 消息（len +1）"
        assert out[-1]["role"] == "user"
        assert out[-1]["content"].startswith("[supercompress recall]"), "独立 user 尾 = recall 块"
        assert out[-2]["role"] == "tool", "原 tool 消息保留在前（未改写）"

    def test_p0_at_tail_of_block(self, engine):
        """被召回 P0 段渲染在 recall 块最末（P1 背景在前 = recency 权重最强位给 P0）。"""
        _seed_recallable(engine)
        out = engine.select_context([{"role": "user", "content": "x"}],
                                   incoming_message={"role": "user", "content": _QUERY})
        assert out is not None
        texts = _recalled_texts(out)
        assert _P0_TEXT in texts, "P0 段被召回"
        assert _P1_TEXT in texts, "P1 背景段被召回"
        assert texts.index(_P0_TEXT) > texts.index(_P1_TEXT), \
            "P0 段须垫底（块尾）；P1 背景在前"

    def test_openai_wire_layout(self, engine):
        """OpenAI wire：assembled request 无第 2 条 in-list system（仅主 system 1 条）。"""
        _seed_recallable(engine)
        req = [{"role": "system", "content": _MAIN_PROMPT},
               {"role": "user", "content": "x"}]
        out = engine.select_context(req, incoming_message={"role": "user", "content": _QUERY})
        assert out is not None
        sys_msgs = [m for m in out if m.get("role") == "system"]
        assert len(sys_msgs) == 1, "S1：仅主 system 1 条（recall 不再注位置 2 第 2 条 in-list system）"
        assert sys_msgs[0]["content"] == _MAIN_PROMPT, "主 system 内容不变（未被 recall 覆盖）"

    def test_no_user_user_adjacent(self, engine):
        """wire dump 断言无 [user, user] 独立相邻对（S1 两种形态均不产生 user-user 独立相邻）。"""
        _seed_recallable(engine)
        # 形态①：末条 tool → 追加独立 user = [.., tool, user]（非 [user, user]）
        req_tool = [{"role": "user", "content": "x"},
                    {"role": "assistant", "content": "call"},
                    {"role": "tool", "content": "r"}]
        out1 = engine.select_context(req_tool, incoming_message={"role": "user", "content": _QUERY})
        assert out1 is not None
        _assert_no_user_user_adjacent(out1)
        # 形态②：末条 user → 合并（不新增）
        req_user = [{"role": "user", "content": "x"},
                    {"role": "assistant", "content": "call"},
                    {"role": "user", "content": "next"}]
        out2 = engine.select_context(req_user, incoming_message={"role": "user", "content": _QUERY})
        assert out2 is not None
        _assert_no_user_user_adjacent(out2)


def _assert_no_user_user_adjacent(out):
    roles = [m.get("role") for m in out]
    for i in range(len(roles) - 1):
        assert not (roles[i] == "user" and roles[i + 1] == "user"), \
            f"S1 构造破相邻：[user,user] 独立相邻对 at idx {i} (roles={roles})"


# ─────────────────────────── Anthropic wire 组装（宿主路由模拟）───────────────────────────
class TestAnthropicWire:
    def test_anthropic_no_second_inlist_system(self, engine):
        """模拟 Anthropic 路由组装（调 convert_messages_to_anthropic）：
        wire system 参数 = 主 prompt 内容（非被 recall 块覆盖），in-list system 消息数 = 1。"""
        conv = _import_anthropic_converter()
        if conv is None:
            pytest.skip("宿主 Anthropic 转换器不可达（非 host 环境）—— 路由冒烟面，非本卡硬门")
        _seed_recallable(engine)
        req = [{"role": "system", "content": _MAIN_PROMPT},
               {"role": "user", "content": "x"}]
        out = engine.select_context(req, incoming_message={"role": "user", "content": _QUERY})
        assert out is not None
        # in-list system 数 = 1（主 prompt；recall 走 user 面 → 无第 2 条 in-list system）
        in_list_systems = [m for m in out if m.get("role") == "system"]
        assert len(in_list_systems) == 1, "in-list system 消息数须 = 1（主 prompt）"
        system_param, _result = conv(out)
        sys_text = _as_text(system_param)
        assert _MAIN_PROMPT in sys_text, "wire system 参数须 = 主 prompt 内容"
        assert "[supercompress recall]" not in sys_text, \
            "wire system 参数不可被 recall 块覆盖（Anthropic 覆盖赋值 bug 免疫）"


# ─────────────────────────── compress 出块面（staged 块同走 S1）───────────────────────────
class TestStagedBlockS1:
    def test_staged_block_s1_tail(self, engine):
        """compress() 出块面：staged 块不再注位置 2 的第 2 条 in-list system，改走 S1 尾投递（user 面）。

        输入用交替 role 的真实会话面（system + user/assistant 交替，protect_first_n/last_n 缝合处
        不产 user-user 相邻）——「无 [user,user] 独立相邻对」断言度量的是 **S1 构造**未新造相邻对，
        非输入历史自带的相邻对（S1 保证：末条 user 并入=不新增；末条非 user 追加独立 user=追加在
        非 user 之后，两者都不产生 user-user 独立相邻）。"""
        _seed_recallable(engine)
        # 交替 role（7 条）：protect_first_n=3 → [system, u0, a0]；protect_last_n=6 → [u0,a0,u1,a1,u2,a2]
        # 缝合处 = ...a0, u0...（user 跟 assistant 后）；全 wire 无 user-user 相邻（S1 追加/并入前的基准面）
        msgs = [{"role": "system", "content": _MAIN_PROMPT},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"}]
        out = engine.compress(msgs, current_tokens=200000, force=True)
        assert out is not None
        # 无第 2 条 in-list system：除主 system 外无任何 role=system
        sys_msgs = [m for m in out if m.get("role") == "system"]
        assert len(sys_msgs) == 1, "S1：staged 块不注位置 2 system（仅保留主 system 1 条）"
        assert sys_msgs[0]["content"] == _MAIN_PROMPT, "主 system 内容不变"
        # staged 块走 S1 尾投递：投在 user 面（末条 user 并入 / 独立 user 追加），非 system
        staged = [m for m in out if "[supercompress] staged" in str(m.get("content", ""))]
        assert staged, "staged 块非空（P0/P1 段在注入池）"
        assert all(m.get("role") == "user" for m in staged), \
            "staged 块须投 user 面（S1 尾投递），绝不再发 in-list system"
        # 相邻对护城河：交替基准面 + S1 末位追加独立 user（追加在 a2 之后）→ 全 wire 仍无 user-user 独立相邻
        _assert_no_user_user_adjacent(out)


# ─────────────────────────── 宿主转换器 import（hermes root 反推 + fail-open）───────────────────────────
def _import_anthropic_converter():
    """宿主 agent.anthropic_message_convert.convert_messages_to_anthropic。

    宿主 repo root（含 agent/ + hermes_yaml.py）可能不在 sys.path（editable finder 只映射 agent
    包，hermes_yaml 是顶层 .py 模块不在 MAPPING）→ 从 engine.__file__ 反推 hermes root 补进
    sys.path。反推失败或 import 失败 → 返回 None（调用方 pytest.skip，不炸宿主）。"""
    import engine as _eng
    d = os.path.dirname(os.path.abspath(_eng.__file__))
    root = None
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "agent")) and \
           os.path.isfile(os.path.join(d, "hermes_yaml.py")):
            root = d
            break
        d = os.path.dirname(d)
    if root is None:
        return None
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from agent.anthropic_message_convert import convert_messages_to_anthropic
        return convert_messages_to_anthropic
    except Exception:
        return None


def _as_text(x):
    """wire system 参数可能是 str（无 cache_control）或 block list（含 cache_control）→ 归一文本。"""
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join(p.get("text", "") for p in x if isinstance(p, dict))
    return str(x)
