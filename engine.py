"""Hermes host adapter shell — 薄接线层（PM 路①定稿：壳只做接线，零新业务逻辑）。

宿主契约（hermes-agent 宿主仓实码核清，@coder 直查 2026-09-27；宿主仓位置因安装而异，
本插件不引用宿主仓绝对路径——安装根从 __file__ 反推，数据根走 HERMES_HOME env）：
- loader: plugins/plugin_loader.py `load_plugin_module` → `instance_from_module`
  （`mod.register(collector)` 优先，collector._EngineCollector `register_context_engine(engine)` 收实例；
  register 失败则回退扫描模块属性找 ContextEngine 子类无参构造）。`probe_availability` 调
  实例 `is_available()`（若有）。
- 宿主调用面: select_context / update_from_response / should_compress / compress（宿主按签名过滤
  kwargs）/ on_turn_complete / on_session_start|end / on_session_reset。
- 失败模式封死: 壳加载失败 → loader warn + None → 宿主静默回落内置 `compressor`。壳必须让
  probe_availability 返回 active（@reviewer 静态验收门 #1，tests/test_hermes_shell.py 复现）。

零外呼: aux 端点默认 `_NoEndpoint`（纯本地）→ tagger LLM tiering 异常 → 结构 P0 地板；
历史内网外部端点依赖已整类移除（旧代外部调用面不复存在，运行时不发起任何外部服务调用）。
Phase 1 起：`update_model()` 由宿主注入 agent 运行时（call_llm + main_runtime），aux 自动
升级为 `_CallLlmEndpoint`（两级路由：aux 通道短超时 → 主模型兜底）——**生产路径零 env
依赖**，`LIVE_ENDPOINT_*`/`OPENAI_*` 仅活在 bench 出数面（sc/live_endpoint.py，非生产调用链）。
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
import json
import hashlib
from typing import Any, Dict, List, Optional

# ── 宿主路径引导（壳的唯一环境接线，非业务逻辑）──────────────────────────
# 本项目按 install.sh 约定以「repo 根在 PYTHONPATH」为运行前提（`import sc` /
# `import scdata` 即通）。Hermes loader（plugins/plugin_loader.py）用 qualified
# namespace exec 插件模块，NOT 把插件目录挂进 sys.path——sc/*.py 的裸 `import scdata`
# 会在此断链 → loader warn+None → 宿主静默回落内置 compressor。引导一行补上：
# 把本插件目录（= 交付面 repo 根）加入 sys.path，使 `sc` / `scdata` 按设计 top-level 可导入。
_PLUGDIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGDIR not in sys.path:
    sys.path.insert(0, _PLUGDIR)

import sc
from sc import (
    AuxProvider,
    Distiller,
    JsonlStore,
    Segment,
    StoreError,
    Window,
    WindowSeg,
    est_tokens,
    nfc_hash,
)
# 数据子目录名单一事实源（G2）：telemetry_report_path / on_session_start 的 legacy 回退
# 各带 env/kwarg 分支（合法独立规则），但目录名字面引用共享常量，杜绝「改一处漏一处」。
from sc._bootstrap import STORE_SUBDIR
from sc.inject import choose_forms, render_block, block_tokens
from sc.recall import Recaller, tokenize
from sc.rerank import RerankGate, rerank_hits
from sc.semantic import BoundaryExpander, BOUNDARY_TASK
from sc.tagger import Tagger
from sc.drift import DriftTracker
from sc.telemetry import TelemetryCollector, filler_form_share
from sc.version import build_id, desync_warning
from sc import digest_pool as sc_digest_pool
from sc import model_store as sc_model_store
from sc.compact_ledger import CompactLedger, compact_ledger_path_for_jsonl  # t_63bf3ba2 append-only 累计布局（与 cache-aware 卡共用本侧车模块）
import scdata  # scdata.sink_p0/restore_p0：settled P0 原文沉盘（侧车同目录 <seq>.p0，可复活可审计）



def _get_call_llm():
    """惰性导入 call_llm（避免循环依赖）。"""
    try:
        from agent.auxiliary_client import call_llm
        return call_llm
    except ImportError:
        return None

logger = logging.getLogger(__name__)

# 配置默认值（远端 config.yaml `compression.supercompress` 段实查存在；旧 key 全部兼容）
DEFAULT_CONTEXT_LENGTH = 256000
DEFAULT_COMPACT_RATIO = 0.5
DEFAULT_FORCE_COMPACT_RATIO = 0.75
DEFAULT_MAX_TAIL_TURNS = 15
DEFAULT_MAX_TAIL_TOKENS = 32000
# P3 梯级水位：主 compact 触发默认挪 0.8→0.55（55–80% compact 带提前预摘要，band 内最优点由
# P2 遥测定案，先给 0.55 默认——G5 数据闸门；P2 首读 fill_share 0.789 / 触发点随 1–2 周 sink 校准）。
# 语义边界：本常量 = engine 侧「何时触发 compress」的水线；sc/distill.py 内部 evict-to 软水位
# （80% cap，R3 逃生门钉死面）与 90% 非 P0 逐出档在组件内独立定义，壳不重算算法。
DISTILL_WATER_FRAC = 0.55
# 紧凑摘要账本（t_63bf3ba2 · append-only 累计布局）消息面 marker（单源，G2）：长会话
# compress() 在 head/tail 缝发行的累计账本消息 content 前缀。宿主 re-feed 去重（剥除
# 陈旧旧段）与发行两处共用此常量——前缀判据绝不与文案漂移。注意与 S1 尾投递的
# `[supercompress] staged P0/P1 segments` 头文案**不同**（S1 消息不含 compact-ledger 词）：
# 只认本专属前缀才不误伤 S1 尾投递面。
_COMPACT_LEDGER_PREFIX = "[supercompress] staged compact-ledger"
# P1 extended-BM25 招③：召回时间衰减接线面（Recaller 构造默认 decay_lambda）。
# 默认 0.0 = 不启用（钉死面零扰动）；启用 = 同分下越早 seg.seq 越多被降（纯函数、无 wall-clock）。
DEFAULT_RECALL_DECAY_LAMBDA = 0.0
# cache-aware 决策面默认值（owner 指令「1 要」：last_cache_hit_ratio 参与压缩/召回决策）。
# 全默认 = 现状零变化（fail-safe）：enabled 默认 False（关），开启后阈值/权重走
# config 嵌套块 compression.supercompress.cache_aware_compress.*（owner 指定键名，
# 不硬编码进决策路径——决策面只认实例属性，实例属性由下方 _get 逐键带默认规整）。
DEFAULT_CACHE_AWARE_ENABLED = False          # 关（默认）= 决策面恒走现状路径（零变化）
DEFAULT_CACHE_AWARE_HIT_MIN = 0.5           # 高命中判据：last_cache_hit_ratio ≥ 此值 → 前缀判稳
DEFAULT_CACHE_AWARE_DELAY_FRAC = 0.10       # 推迟量：触发线抬高 base×此比例（省 re-prime 成本）
DEFAULT_CACHE_AWARE_RECALL_K = 1           # 高命中时 recall top-K 收窄值（少换 recall 内容）
# block-② recall cache（t_12f38672）：query 未实质变化判据阈值——本轮 query 归一化 token 集
# 与上轮指纹的 Jaccard 相似度 ≥ 此值判「未实质变化」→ 复用上轮 recall 结果（块字节跨轮稳定，
# 减少 head 动态块对 provider 前缀缓存的断裂）。走 config 嵌套键 recall_cache_threshold
# （缺失 = 安全默认，零变化 fail-safe）。判定异常 / 上轮指纹缺失（冷启动）/ 复用段已被
# compact 淘汰 → 一律回落全量重算（现状路径零变化）。
#
# A1 配套（R1 · 2026-10-10）：默认 0.85 → 0.92。原 0.85 对 CJK 单字指纹过宽——两条主题
# 完全不同的中文 query 仅靠日常单字（你/我/修/代/码…）重叠即 ≥0.85 → 误判「未实质变化」
# → 上轮整块逐字节复放（= 本 bug 主放大器：agent 自述段跨主题误命中后又被固化回放）。
# 0.92 落在「cosmetic 变化（删一两个词）仍判复用 / 换主题必判重算」的空档中位，
# 配合 _recall_token_fingerprint 剔 CJK 单字（指纹面只留双字+拉丁+路径 run），
# 让「未实质变化」真正反映主题同。可回滚：改回 0.85 + 指纹全 token。
DEFAULT_RECALL_CACHE_THRESHOLD = 0.92
# P0.4 rotation advisory 阈值（引擎侧纯信号，零注入、零遥测、零 cache 影响；宿主契约 fail-safe）：
ROTATION_PROMPT_RATIO = 0.85   # T_prompt：连续压力达标置位线
ROTATION_RESET_RATIO = 0.75    # < 此值 = 已 compress 过，清 advisory（引擎侧等价「轮转前重读 pressure 二次确认防误轮转」）
ADVISORY_DEBOUNCE_TURNS = 2    # 连续 N turn 超阈值才置位（session-redesign S-Phase 6 防抖口径）


def _default_store_root() -> str:
    """默认 store 根（安装可移植，零宿主改动）：委托 ``sc._bootstrap.store_dir()`` 单源解析。

    优先级：显式 ``external_store_path`` 配置（__init__ 消费）> ``HERMES_HOME`` env
    （生产 gateway 进程 per-profile 自带，指向 profile 数据根）> ``~/.hermes``（平台默认）。
    生产轮转 rebind 缺 ``hermes_home`` kwarg 时，env 回退即落对 profile 目录
    （reviewer S1 Major#1 效果达成，宿主 conversation_compression.py 零改动）；
    任意安装环境设 ``HERMES_HOME`` 指数据根即可，无 /root、/usr/local 写死。

    单源铁律（安装可移植性的可维护性）：store 子目录名 + HERMES_HOME 解析只在
    ``sc/_bootstrap.store_dir()`` 一处定义；本壳不再内联复写该规则（避免「改一处漏一处」
    的漂移——CLI 脚本与 engine 数据面共用同一解析，任意安装目录成立）。惰性 import
    与 CLI 脚本口径一致（``from sc._bootstrap import store_dir``），壳 import sc 但不
    import 宿主。
    """
    from sc._bootstrap import store_dir
    return store_dir()


def sc_context_engine_base():
    """宿主 ContextEngine 基类惰性解析：宿主/验收环境 = agent.context_engine.ContextEngine（真基类）；
    独立环境 = object（壳可独立 import）。真静态验收门由 tests/test_hermes_shell.py 用真 loader 链裁决。"""
    try:
        from agent.context_engine import ContextEngine
        return ContextEngine
    except Exception:
        return object


class _NoEndpoint:
    """aux 无端点降级替身（纯本地零外呼）：任何 call 抛异常 → aux transient → tagger 落结构 P0 地板。

    非测试替身（FakeEndpoint 住 tests/fakes.py）——这是生产路径的显式降级件：主模型端点未注入时
    召回/蒸馏照常工作，LLM tiering 退化为确定性结构地板（sc.tagger 契约：异常 → P0 structural，
    永不当 P2 丢）。
    """

    def call(self, task: str, text: str) -> str:
        raise RuntimeError(
            "supercompress shell: no aux main-model endpoint injected (update_model() "
            "尚未由宿主调用) — LLM tiering degrades to structural P0 floor; "
            "recall/distill proceed local-only")


class _CallLlmEndpoint:
    """通过注入的 call_llm 函数调用主模型端点（与 compressor 共享同一 API 路径）。

    不复制 provider/model/base_url/api_key，直接引用 agent 的运行时配置。
    """

    def __init__(self, call_llm_func, main_runtime: dict,
                 aux_timeout: float = 10.0, main_fallback_timeout: float = 30.0,
                 digest_max_tokens: int = 256):
        self.call_llm = call_llm_func
        self.main_runtime = main_runtime  # {"model", "provider", "base_url", "api_key", "api_mode"}
        self.model = main_runtime.get("model", "")
        self.provider = main_runtime.get("provider", "")
        self.base_url = main_runtime.get("base_url", "")
        self.api_key = main_runtime.get("api_key", "")
        self.api_mode = main_runtime.get("api_mode", "")
        # 两级短超时链（用户契约：aux 短超时 → 超时/aux 未配置 → 主模型）
        self.aux_timeout = aux_timeout                    # tier-1：agent 共享 aux 通道（短，fail fast）
        self.main_fallback_timeout = main_fallback_timeout  # tier-2：主模型直连（稍宽）
        self.digest_max_tokens = digest_max_tokens         # digest 短生成，不跑长篇摘要

    def call(self, task: str, text: str) -> str:
        """两级路由：
        tier-1 aux 通道（task 原样，agent 共享 API 路由，短超时）；
        超时/异常/aux 未配置 → tier-2 主模型（task=None + main_runtime → 宿主解析主
        provider/model，同 compressor 路径）。两级都失败 → 抛异常（aux transient →
        冷却窗 + force_evict 兜底，见 sc.aux_provider）。
        """
        is_digest = task == "auxiliary.compression"
        last_err: Exception | None = None
        for i, route in enumerate((
            # tier-1：aux 通道（显式短 timeout 盖过宿主默认 30s 与任何 compression floor）
            dict(task=task, main_runtime=self.main_runtime, timeout=self.aux_timeout,
                 **(dict(max_tokens=self.digest_max_tokens) if is_digest else {})),
            # tier-2：主模型直连（main_runtime 由 update_model 注入，共享 agent 配置，零复制）
            dict(task=None, main_runtime=self.main_runtime, timeout=self.main_fallback_timeout,
                 **(dict(max_tokens=self.digest_max_tokens) if is_digest else {})),
        )):
            try:
                response = self.call_llm(**route, messages=[{"role": "user", "content": text}])
            except Exception as e:
                last_err = e
                if i == 0:
                    logger.info("supercompress: aux channel task=%s short-timeout/failed (%s) "
                                "-> falling back to main model route", task, e)
                continue
            # 兼容 dict 和 namespace 响应
            try:
                if isinstance(response, dict):
                    # 思考模型（enable_thinking 未关 / aux 未继承 extra_body）：
                    # 思维链走 message.reasoning_content，正文才是 content；
                    # 若 content 为空但 reasoning 非空 → 只取 content 会拿到空串 →
                    # 下游 parse 全部 fail-safe（rerank 日志实证：sc.rerank not parseable）。
                    _msg = response["choices"][0]["message"]
                    _c = _msg.get("content") or ""
                    if not _c.strip() and (_msg.get("reasoning_content") or "").strip():
                        logger.warning("supercompress: task=%s LLM returned empty content "
                                       "with non-empty reasoning_content (%d chars) — "
                                       "thinking-model merge suspected; downstream parse will fail-safe",
                                       task, len(_msg.get("reasoning_content") or ""))
                    return _c
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"call_llm failed (aux+main routes): {last_err}")


def _load_config() -> Dict[str, Any]:
    """宿主配置读取（宿主环境有 hermes_cli；测试环境无 → 空 dict 默认值，壳永不因配置缺失炸）。"""
    try:
        from hermes_cli.config import cfg_get, load_config
    except Exception:
        return {}
    try:
        return cfg_get(load_config(), "compression", "supercompress") or {}
    except Exception:
        return {}


class _AuxMainEndpointProxy:
    """P4 边界 LLM 端点解耦（G3：宿主主模型，0 外部 / 0 ONNX / 0 模型名硬编码）。

    持有 engine 引用、**调用时刻**活读 `engine._aux.main_endpoint` → `update_model()` 重建
    `self._aux`（新 `_CallLlmEndpoint`）后 P4 自动骑上当前宿主主模型通道（不留 stale 端点、
    不引入第二 provider、不建模型名/provider 块）；端点两级路由/瞬态失败语义由 aux 侧统一
    承担，P4 层只管自己的 2 档退避 + per-epoch 缓存（见 sc/semantic.py）。测试替身注入
    （`e._aux.main_endpoint = FakeEndpoint()`）同样经本代理活读生效，零改动测试面。
    """

    def __init__(self, engine) -> None:
        self._engine = engine

    def call(self, task: str, text: str) -> str:
        return self._engine._aux.main_endpoint.call(task, text)


def _make_segment(msg: Dict[str, Any], tagger: Tagger, ts: float) -> Segment:
    """宿主 message → sc.Segment（接线：role→provenance；classify→tier/structural，全委托 sc.tagger）。

    4a 数字代号：model 段经 LLM tagging 面顺带打定案信号码（settled_signal 0/1/2）；
    user 段纯规则面恒 0（不打码，宁漏勿错——错折会把未定案段藏掉）。"""
    provenance = "user" if msg.get("role") == "user" else "model"
    text = str(msg.get("content") or "")
    _res = tagger.classify(text, provenance)
    tier, structural, _regime, _reason = _res[:4]
    settled_signal = _res[4] if len(_res) > 4 else 0
    return Segment(tier=tier, provenance=provenance, original_text=text,
                   ts=ts, structural_p0=structural, settled_signal=settled_signal)


# Room-skeleton 单源谓词（宿主 ingress/egress 共用；与 file-memory 的 META_MARKERS /
# is_room_meta_turn 同一 single-source 规则，开首测试，不拦正文引用）：bot-mode
# 群聊回合把房间转写骨架当 user_content 传入（`[Group chat: "…"] You are @devops,
# one participant in a group chat…`）。该骨架存成 P0 user 段后，被后续任意会话
# 的 select_context 按词重叠召回注入 = 跨会话污染（2026-09-30 事故实证：全局 store
# 208 条 group 段、203 条 P0、零 session/scope 字段）。开首判据：lstrip 后以
# `[Group chat` 起始才算骨架；真实用户消息正文中引用 "[Group chat" 不拦（M1 边界）。
def is_room_skeleton(text: str) -> bool:
    s = (text or "").lstrip()
    return s.startswith("[Group chat")


# R2（2026-10-10 · A1 配套）：agent 自述段 egress 谓词——把"agent 把自己前几轮复述
# 当'已确认'回放"这段 text 逐出自动注入面（select_context recall + compress staged 双点）。
# 根因（2026-10-10 事故实证）：model 旁路 P0 段（provenance=model，agent 自己的认账/复述/
# 推理）被 P0_BOOST + 预算豁免 + 每 turn recall 自动投末条 user → 模型（及人）误当"用户拍板"。
# 本谓词**只拦"归属错乱"那段 text 形态**（复述/认账/回放自指 + "把你拍板的 X 反了"类），
# 不拦 agent 正常推理段（实证取证/方案输出）——宁漏勿错：漏 = 原样注入不误删，
# 误删面只剩 marker 词面命中这一个入口（纯字面、零 LLM、确定性、同 is_room_skeleton 模式）。
# 双语言词面（中英）：只拦 **agent 自述**（agent 认账/复述/回放自己前几轮）的形态。
# 宁漏勿错红线：
#   - 不拦用户句里正常的自我指涉（"我刚才说的阈值你按 X 改"——这是用户在纠正，
#     属正常 user 段，绝不能误删）；故"我刚才说的"/"我刚才那个"/"我上一轮"等
#     以"我"开头的用户自我指涉一律不入表。
#   - 只保留"把/上轮/本轮 + agent 输出自指"这类**agent 复述自己前轮输出**的形态
#     （"把我自己..."是 agent 把自己当对象，正常 user 消息不会出现；"认个账"
#     是 agent 认账惯用语，user 极少用）。
#   误删面收窄到"agent 复述自指"单一入口；漏 = 原样注入不误删（宁漏勿错）。
# R2-v2（2026-10-10 · 语言无关化）：
#   - 本 33 条中英 marker 保留作**中/英子集**（向后兼容、零数据变更）；
#   - 新增 _SELF_REF_CUES（多语言"时序自指"结构词，各语言 1-2 个高频词即可，
#     不逐语言堆 marker）+ provenance 硬门（seg.provenance=="model" 才参与判定，
#     user 段天然免疫误删）——见 is_agent_self_report 的 v2 判定式。
_AGENT_SELF_REPORT_MARKERS = (
    "认个账", "先认个账", "认账",
    "把我自己", "把我前", "我前两",
    "当\"已确认\"", "当'已确认'", "当成你", "当成用户",
    "旧默认值", "旧回放", "上轮回放", "上轮复述", "上轮把", "本轮把",
    "不再拿旧", "不再拿上轮",
    "supercompress 把我", "recall 把我", "把\"你确认", "当\"你确认",
    "restated", "playback", "confirmed direction",
    "my earlier recap", "my earlier restatement", "my previous reply",
    "as if you", "as your confirmation", "i just acknowledged",
    "i first acknowledged", "acknowledge first",
)
# X1（owner 2026-10-10）：从 v1 词表删掉的宽词（"引用拍板继续推进"形态，非 agent
# 自述复述，不该删）——记录在此便于追溯，不入表：
#   "跟你拍板","跟你确认","跟你定","这版按你","按你拍板","按你确认","按你拍"
#   事故面"把拍板方向当已确认回放"仍由 认账/把我前/旧回放/当"已确认"/旧默认值 等
#   真·自述 marker + R3(降权)/R4(gate drop) 语言无关兜底，不靠这类宽词硬扛。

# R2-v2（2026-10-10）：多语言"时序自指结构"词表（agent 指向前轮输出的结构特征，
# 跨语言稳定——比自然语言惯用语窄、可追加，避免逐语言堆 marker 词表的老毛病）。
# 判"agent 复述/回放自己前轮"的结构 = 时序自指词（前一轮/之前的/earlier/previous）
# 或认账/重述语气（reconsider/corrección/認め直す）。每语言 1-3 个高频词，够用即可；
# 命中须叠加 provenance=="model" 硬门（user 段含这些词不误删）。大小写不敏感。
_SELF_REF_CUES = (
    # 时序自指（前轮/之前输出）：中 / 英 / 日 / 韩 / 德 / 西
    "上一轮", "前几轮", "前两轮", "本轮的复述", "本轮的推理",
    "earlier", "previous reply", "previous turn", "my last",
    "前の", "前回の", "さきほど",            # 日
    "이전", "전 단계", "방금",                # 韩
    "vorherigen", "vorherige", "bisherige",  # 德
    "anterior", "antecedente",  # 西
    # 认账 / 重述语气（附加低权重信号，不单独触发，须叠 provenance 门）：
    "reconsider", "reconsidered", "rectif", "correction", "改正", "修正",
)


def is_agent_self_report(text: str, provenance: str = "") -> bool:
    """R2-v2 egress 谓词：判定"agent 自述段"（归属错乱形态）是否逐出自动注入面。

    v2 判定式（语言无关结构 + provenance 硬门，owner 2026-10-10 拍 option c·b）：
      is_self_report = (provenance == "model")
                       AND ( self_ref_cue_hit(text)  OR  legacy_marker_hit(text) )
    - **provenance 硬门**：非 model 段（user 拍板/指令）直接 False——user 段含
      "earlier/上一轮" 等词不误删（修掉现状"我刚才说的阈值…"误杀 user 段的面）。
      缺省 provenance=""（未传）→ 退回纯 marker 面（兼容旧调用点 / 测试，行为 = v1）。
    - **model 段**：命中"时序自指结构词"(_SELF_REF_CUES，多语言) 或"中英 marker"
      (_AGENT_SELF_REPORT_MARKERS，v1 子集) 任一 → 逐出（agent 复述自己前轮 = 归属错乱）。
    纯字面子串判定（零 LLM / 零 wall-clock / 确定性）；大小写不敏感。宁漏勿错：
    漏 = 原样注入不误删（不引入新风险）。
    """
    if not text:
        return False
    low = text.lower()
    # provenance 硬门：非 model（user/""/其他）→ 只有显式 provenance="model" 才参与判定。
    # 缺省 ""（未传 provenance 的旧调用面）→ 走纯 marker 兼容面（v1 行为，不误删）。
    if provenance != "model":
        # 兼容面（v1）：无 provenance 信息时退回纯 marker 判定（既有调用点 / 既有测试）。
        return any(m.lower() in low for m in _AGENT_SELF_REPORT_MARKERS)
    # model 段：自指结构词（多语言）∪ 中英 marker 任一命中 → 逐出
    if any(c.lower() in low for c in _SELF_REF_CUES):
        return True
    return any(m.lower() in low for m in _AGENT_SELF_REPORT_MARKERS)


def _safe_session_name(session_id: str) -> str:
    """session id → 安全文件名（宿主 sid 形如 20260926_111839_1e31d0；防御非法字符）。

    P0.7（Nit#3 碰撞加固）：净化改变原串（含非法字符或边界 _）→ 追加原串短 hash
    后缀，不同原始 sid（`a/b` vs `a_b`）不再共写一文件；宿主生成器的干净 sid
    （时间戳+hex）identity 通过 → 零行为变化（既有测试/生产落点不变）。
    """
    raw = session_id or ""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("_") or "unnamed"
    if base != raw:
        base = f"{base[:64]}.{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"
    return base


# ── L1 区裁定（owner）：撤 A1 前置重排，P0 回 c# 自然 append 序 ──
# A1（P0 定案段恒居 section 前部）会把 P0 一重排/一变就打断 provider 前缀缓存
# （cache-hit 丢失命门）；回到自然 append 序，前缀才真单调稳定。P0 定案段的
# “不被重读”改由 settled 折出存根面承担（sc.compact_ledger.settle/unsettle +
# scdata.sink_p0/restore_p0），不再靠 tier 重排。
#
# 纯自动定案判定（owner 裁定 B 路线：引擎侧、不读 kanban 钩子）：
# staged 块 forms 含 P0 段（tier P0 定案段）且 P0 段定案信号码（settled_signal）
# = 强完成（2）→ 本块记 adhoc settled（下轮 append 起折存根，原文沉盘 <sid>.p0 可复活）。
# 4a 数字代号（owner 裁定，替代旧硬编码中英文词表——多语言不友好 + 误匹配面宽）：
# 定案信号码由 LLM 蒸馏面（Tagger.classify → main_endpoint "tagging" 任务）顺手打码：
#   SETTLED_SIGNAL_NONE      = 0  无完成信号 → 全文回喂（不定案，治重复问只对真定案段生效）
#   SETTLED_SIGNAL_CANDIDATE = 1  弱候选信号 → kanban 半自动（LLM 裁判 + clarify 拍板才 settle）
#   SETTLED_SIGNAL_SETTLED   = 2  强完成信号 → adhoc 纯自动直接折存根 + 沉盘
# 账本写侧纯读码（_staged_signal 取渲染 P0 段最高码），零词表、零 re、零 LLM 参与本判据。
# 宁漏勿错红线：默认码 0（全文回喂）；错折（把未定案段藏掉）面只剩上游 LLM 打 2 这一个入口。
SETTLED_SIGNAL_NONE: int = 0
SETTLED_SIGNAL_CANDIDATE: int = 1
SETTLED_SIGNAL_SETTLED: int = 2


def _staged_signal(rendered: list) -> int:
    """取渲染 P0 段里的最高定案信号码（2 > 1 > 0）。零 LLM、纯函数。

    `rendered` = [(ws, form, text)]（choose_forms 产物）：只读 `ws.segment.tier == "P0"`
    段的 `settled_signal` 码；无 P0 段 / 全 0 → 返回 0（不定案）。
    """
    sig = SETTLED_SIGNAL_NONE
    for ws, _f, _t in (rendered or []):
        seg = getattr(ws, "segment", None)
        if getattr(seg, "tier", None) != "P0":
            continue
        s = getattr(seg, "settled_signal", SETTLED_SIGNAL_NONE)
        if not isinstance(s, int):
            try:
                s = int(s)
            except (TypeError, ValueError):
                s = SETTLED_SIGNAL_NONE
        if s not in (SETTLED_SIGNAL_NONE, SETTLED_SIGNAL_CANDIDATE, SETTLED_SIGNAL_SETTLED):
            s = SETTLED_SIGNAL_NONE
        if s > sig:
            sig = s
    return sig


def _staged_p0_decision(staged_block: str, forms: Optional[Dict[str, int]],
                       rendered: list) -> str:
    """staged 块 P0 定案段三态判据（纯自动 adhoc / 半自动 kanban candidate / 不定案）。

    4a 数字代号驱动（纯读码，不扫词）：
    ① 渲染 P0 段最高信号码 = SETTLED_SIGNAL_SETTLED(2)（强完成）→ "adhoc"（纯自动直接折出）
    ② = SETTLED_SIGNAL_CANDIDATE(1)（弱候选）→ "kanban"（半自动 candidate，等裁判 + clarify）
    ③ = SETTLED_SIGNAL_NONE(0) / 无 P0 段 → "none"（不定案，全文回喂）

    宁漏勿错：错折面只剩上游 LLM 打 2 码一个入口；账本写侧零词表、零 re。
    """
    sig = _staged_signal(rendered)
    if sig == SETTLED_SIGNAL_SETTLED:
        return "adhoc"
    if sig == SETTLED_SIGNAL_CANDIDATE:
        return "kanban"
    return "none"


def _expand_small_code_placeholders(rendered: list) -> list:
    """R5（owner 批次 A #6 · 记账面小占位符展开，确定性零 LLM，fail-open）：

    逐段检测渲染块 text（`[code: ×N lines] …` 形态，sc.model_store 的 T1 折叠
    产物）为**小代码占位符**（正文行数 N < MODEL_LEDGER_FENCE_EXPAND_LINES，
    默认 20）→ 该段改用 `original_text`（围栏全文，L3 面 excerpt 同源）进账本，
    残句占位符不进账本（治诊断账本 #6：[code: ×8 lines] 残句混入）。
    大占位符（N ≥ 阈值）/ 非占位符段 / 异常 → 原样保留（零新风险；宁漏勿错
    = 展开失败保持残句，比错展开更安全）。

    纯函数：返回与输入同形的 [(ws, form, text)]（只改小占位符段的 text）；
    段 original_text 缺失（旧数据面）→ 原样（fail-open）。T1 折叠契约
    （入库字节 / 召回面 / 主 store 段）零扰动——只在记账渲染面展开。"""
    out: list = []
    for item in (rendered or []):
        ws, ff, tx = item
        try:
            if ff in ("L3", "L2") and sc_model_store._is_small_code_placeholder(tx):
                orig = getattr(getattr(ws, "segment", None), "original_text", None)
                if orig:
                    tx = orig
        except Exception:
            pass
        out.append((ws, ff, tx))
    return out


def _staged_by_tier(rendered: list, tier: str) -> tuple[str, Dict[str, int]]:
    """按 tier 拆 staged 块（owner 裁定 tier 分层：P0 定案块 / P1 推理可裁层）。

    返回 (该 tier 的块文本, 该 tier 的 forms 计数)。`rendered` = [(ws, form, text)]
    （choose_forms 产物，零 mutation）：只挑 `ws.segment.tier == tier` 的段渲染。
    R5（owner 批次 A #3/#5）：块内格式与 render_block 同源同化——行首 `#<rel>`
    = 块内相对号（逐段 1..N），L3/L2 全文行加 `prov=<user|model>` + `seq=<全局>`
    移入定位位；L1 行经 `sc.inject._l1_line(ws, rel)` 重算（同 render_block 面，
    prov/seq/相对号同源，账本 c#N 块号之下不再叠段级全局号）。该 tier 无段 →
    ("", {})。零 LLM、零持久态、纯函数（同 rendered → 同输出）。"""
    from sc.inject import _l1_line
    lines = []
    forms: Dict[str, int] = {}
    rel = 0
    for ws, ff, tx in (rendered or []):
        if getattr(getattr(ws, "segment", None), "tier", None) != tier:
            continue
        rel += 1
        prov = getattr(ws.segment, "provenance", "") or "?"
        if ff in ("L3", "L2"):
            lines.append(
                f"#{rel} {ws.segment.tier} prov={prov} seq={ws.segment.seq} [{ff}]\n{tx}")
        elif ff == "L1":
            lines.append(_l1_line(ws, rel))
        else:
            lines.append(tx)
        forms[ff] = forms.get(ff, 0) + 1
    return ("\n---\n".join(lines), forms)


def _s1_tail_deliver(out: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
    """S1 尾投递（PRD §0.4，确定性、零持久状态分支）：把块内容 `content`（recall / staged）
    投到消息列尾——末条 M.role==user → 并入 M.content（合并，不新增消息）；M.role!=user
    → 末位追加一条独立 user 消息（实际形态 mid-turn [tool→user]，OpenAI/Anthropic 两 wire
    均合法）。

    绝不再发第二条 in-list system（旧位置 2 注入面已废）：宿主 Anthropic 转换器对 in-list
    system 做「最后一条赢」覆盖赋值（anthropic_message_convert.py:718-723），旧面会让主
    prompt 静默丢失；S1 把块改投 user 面天然免疫。cache 面同理：块不再钉在位置 2 随
    query 重算前缀（forensic cache-bleed-2026-10-07：位置 2 字节一变 → 几十 K 历史全重算）。
    宿主 structural-clone 契约（#80498）不被写穿：不就地改输入 dict，只拷末条。
    """
    out = list(out)
    if not out:
        return [{"role": "user", "content": content}]
    last = out[-1]
    if last.get("role") == "user":
        _c = last.get("content")
        _m = dict(last)
        if isinstance(_c, list):
            _m["content"] = list(_c) + [{"type": "text", "text": content}]
        else:
            _m["content"] = content if _c in (None, "") else (_c + "\n\n" + content)
        out[-1] = _m
        return out
    return out + [{"role": "user", "content": content}]


# ── block-② recall cache 纯函数面（t_12f38672 · 零宿主、零 LLM、确定性）──────────────────
# R1（2026-10-10）：指纹面 CJK 单字剔除谓词——单字（你/我/修/代/码…）是日常语料高频重叠源，
# 是 Jaccard 误判「未实质变化」的主因。仅剔除**长度 1 的 CJK 单字**；CJK 双字（bigram，保双字词）
# + 拉丁词 + 路径 run 全保留。纯字面判定（零 LLM / 零 wall-clock / 确定性），与 sc.recall 同源
# CJK 区间（_CJK_RUN_RE 字符集），不引第二套 CJK 表。
_CJK_SINGLE_RE = re.compile(r"^[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]$")


def _fingerprint_tokens(query: str) -> list[str]:
    """指纹 token 面（R1）：tokenize（sc.recall 同源）后剔除 CJK 单字，
    只留 CJK 双字 + 拉丁词 + 路径 run。与召回面 tokenize 单源，仅指纹面窄化。"""
    return [t for t in tokenize(query or "") if not _CJK_SINGLE_RE.match(t)]


def _recall_token_fingerprint(query: str) -> frozenset:
    """query 归一化 token 集（R1：sc.recall.tokenize 同源，剔除 CJK 单字——只留双字+拉丁+路径 run）
    → frozenset（可哈希、集合代数安全）。

    口径钉死（docstring + tests/test_recall_cache.py）：Jaccard 相似度的分母 = 两集合
    并集——「未实质变化」= 本轮 token 集与上轮 ≥ recall_cache_threshold（默认 0.92）。
    空 query → 空集（空 vs 任意 = Jaccard 0 → 不判未实质变化 → 全量重算，fail-safe）。

    R1 动机：原指纹含 CJK 单字（日常高频重叠）→ 两条主题不同的中文 query 仅靠单字重叠
    即 Jaccard ≥ 旧阈值 0.85 → 误判「未实质变化」→ 上轮整块逐字节复放。剔单字后指纹面
    只剩双字+拉丁+路径 run（主题相关词），Jaccard 真正反映主题同异。确定性：纯字面过滤，
    同 query 两次全等。"""
    return frozenset(_fingerprint_tokens(query))


def _jaccard_ge(a: frozenset, b: frozenset, threshold: float) -> bool:
    """Jaccard(a, b) ≥ threshold？任一侧空 → False（fail-safe：不判未实质变化 → 全量重算）。

    纯函数、确定性（集合代数，零 LLM / 零外呼 / 零 wall-clock）；浮点阈值比较是
    本口径唯一非精确面——0.92（R1）默认值远离 0/1 两极，钉死面（全同 = 1.0 判复用 /
    全异 = 0.0 判重算）不受浮点误差影响。"""
    if not a or not b:
        return False
    denom = len(a | b)
    return denom > 0 and (len(a & b) / denom) >= threshold


class SupercompressContextEngine(sc_context_engine_base()):
    """Hermes ContextEngine 薄壳：全委托 sc 组件，算法零实现。

    壳薄度门禁（@reviewer grep 判据）：本文件内 distill/recall/budgets 只出现
    `sc.` 组件调用（Distiller.run / Recaller.search / 水位比较），无任何
    token-overlap / 逐出 / 预算分配算法体（re / _evict / _next_candidate 均不在壳内）。
    """

    name = "supercompress"

    def __init__(self, **kwargs: Any) -> None:
        cfg = _load_config()

        def _get(key, default):
            return kwargs.get(key, cfg.get(key, default))

        self.context_length: int = _get("context_length", DEFAULT_CONTEXT_LENGTH)
        self.compact_ratio: float = _get("compact_ratio", DEFAULT_COMPACT_RATIO)
        self.force_compact_ratio: float = _get("force_compact_ratio", DEFAULT_FORCE_COMPACT_RATIO)
        self.max_tail_turns: int = _get("max_tail_turns", DEFAULT_MAX_TAIL_TURNS)
        self.max_tail_tokens: int = _get("max_tail_tokens", DEFAULT_MAX_TAIL_TOKENS)
        # P1 extended-BM25 招③：召回时间衰减接线面（按参数控制启用面；默认 0 = 钉死面不扰动）
        self.recall_decay_lambda: float = _get("recall_decay_lambda", DEFAULT_RECALL_DECAY_LAMBDA)
        # cache-aware 决策面配置（owner 指令「1 要」）：阈值/权重全走 config 嵌套块
        # compression.supercompress.cache_aware_compress.{enabled,hit_min,delay_frac,recall_k}
        # （owner 指定键名；cfg = 本 __init__ 顶部 _load_config() 取到的 compression.supercompress 子表）。
        # 解析优先级：host/测试 kwarg（cache_aware_* 扁平键）> 嵌套块值 > 常量默认（缺失=安全默认，
        # enabled=False = 决策面恒走现状路径，零变化——fail-safe 红线）。单一路径，不引入第二套扁平
        # config 键面（避免与 owner 指定嵌套块漂移）。决策面只读 last_cache_hit_ratio 实例态。
        _caw = cfg.get("cache_aware_compress")
        if not isinstance(_caw, dict):
            _caw = {}
        self.cache_aware_enabled: bool = bool(kwargs.get("cache_aware_enabled",
                                                          _caw.get("enabled", DEFAULT_CACHE_AWARE_ENABLED)))
        self.cache_aware_hit_min: float = float(kwargs.get("cache_aware_hit_min",
                                                            _caw.get("hit_min", DEFAULT_CACHE_AWARE_HIT_MIN)))
        self.cache_aware_delay_frac: float = float(kwargs.get("cache_aware_delay_frac",
                                                               _caw.get("delay_frac", DEFAULT_CACHE_AWARE_DELAY_FRAC)))
        self.cache_aware_recall_k: int = int(kwargs.get("cache_aware_recall_k",
                                                         _caw.get("recall_k", DEFAULT_CACHE_AWARE_RECALL_K)))
        # block-② recall cache（t_12f38672）：config 嵌套键 recall_cache_threshold（缺失 =
        # DEFAULT = 0.85，fail-safe 零变化）+ 会话级实例态 _recall_cache（None = 冷启动 / 已清）；
        # 会话边界 on_session_start / on_session_reset 对称清零（防跨 session 泄漏）。
        self.recall_cache_threshold: float = float(_get("recall_cache_threshold",
                                                         DEFAULT_RECALL_CACHE_THRESHOLD))
        # A1 止血（2026-10-10）：每 turn 自动召回注入面 kill-switch。
        # 默认 False = select_context 整条面 no-op（return None，宿主 fail-open 字节不变）；
        # 显式 True 才恢复现状每 turn 自动注入（保留回滚路径）。
        # compress（compact 事件驱动面）不受本开关影响。
        self.auto_recall_enabled: bool = bool(_get("auto_recall", False))
        self._recall_cache: Optional[dict] = None
        # P3-block-② persistence: per-session sidecar next to <sid>.jsonl
        # (supercompress_store/<sid>.recall_cache.json) so the in-memory
        # _recall_cache survives a process restart / session cold-reopen.
        # The sidecar path is resolved on demand (rebind-safe, mirrors the
        # _digest_pool / _model_store path accessors) by
        # _recall_cache_sidecar_path() — see that method below.
        # 决策面翻转去抖哨兵（仅 log 决策态 0↔1↔2 翻转，防宿主同 turn 双调 / 同态多 turn 刷屏）：
        # 0 = 现状基线（未触发 / cache-aware 未抬高线）、1 = 抬高线后仍触发（维持）、
        # 2 = 抬高线推迟了基线会触发的压缩（推迟压缩 = cache-aware 介入的活体证据）。
        self._cache_aware_active: int = 0
        store_path = _get("external_store_path", _default_store_root())
        # 数据面：单一 active.jsonl（@data-engineer 契约：旧 chunk_* store 零携带零触碰，
        # 新 active.jsonl 是独立新文件，同住目录互不干扰）
        # legacy 目录（独立测试/bench 无 hermes_home kwarg 时回退；__init__ eager 建 JsonlStore）
        # 安装可移植（零宿主改动方案）：默认根 HERMES_HOME-aware——生产 gateway 进程自带
        # HERMES_HOME（per-profile 数据根），轮转 rebind 缺 kwarg 时 env 回退即落对 profile
        # 目录（Major#1 效果达成，宿主 conversation_compression.py 零改动）；任意安装环境
        # 设 HERMES_HOME 指数据根即可，无 /root、/usr/local 写死。
        self._legacy_store_dir = os.path.expanduser(store_path)
        self.store_path = os.path.join(self._legacy_store_dir, "active.jsonl")

        # S-Phase 1：per-session 独立 store —— on_session_start 按 host 注入的 hermes_home
        # rebind 到 <hermes_home>/supercompress_store/<sid>.jsonl（profile 隔离随
        # HERMES_HOME 按 profile 解析白捡：4 profile 各存各的）；无 hermes_home（独立测试/
        # bench）回退 legacy 目录。store 目录优先级：hermes_home > legacy。
        self.session_id: Optional[str] = None
        self._bound_session: Optional[str] = None   # 幂等哨兵：同实例同 sid 重入 → no-op
        # P0.4 单轨收口完成：宿主现成信号 `carry_over_context`（run_agent.py:402 经
        # on_session_start 下发；确认轮转事件才 True）为 carry 唯一源——插件自造
        # `carry_over` kwarg/配置键已删（reviewer S1 Minor#2）：严格 session 记忆默认关，事件驱动。

        self.store = JsonlStore(self.store_path)
        self._window: Optional[Window] = None
        self._tagger: Optional[Tagger] = None
        # 主模型运行时（由 update_model 注入，共享 agent 的 API 配置）
        self._main_runtime: Optional[dict] = None
        self._call_llm = _get_call_llm()
        # aux 短超时 → 超时/aux 未配置 → 主模型（两级路由参数，宿主 compression.supercompress 配置面）
        # 须先于 _build_aux 赋值（__init__ 建默认 _NoEndpoint 时消费）
        self.aux_timeout: float = _get("aux_timeout", 10.0)
        self.main_fallback_timeout: float = _get("main_fallback_timeout", 30.0)
        self.digest_max_tokens: int = int(_get("digest_max_tokens", 256))
        self.digest_fail_cooldown: float = _get("digest_fail_cooldown", 60.0)
        self._aux = self._build_aux(cfg)

        # B 卡（t_98809671）：model 旁路独立 store 配置面（compression.supercompress 块，
        # 逐键 _get 默认、缺失 = 安全默认；保留 kill-switch；不新增 provider/model/endpoint 配置）。
        self.model_store_enabled: bool = bool(_get("model_store_enabled", True))
        self.model_min_chars: int = int(_get("model_min_chars", 200))
        self._model_store = None            # 懒构造（on_session_start 三分支按 session 绑定；_model_store() 为 rebind-safe 访问器）
        self._model_fused: bool = False     # 保险丝态（fused → 召回并集 / T4 注入跳过该 store；内存 reset、文件保留）

        # token 账（宿主 run_agent.py 直接读这些字段）
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.compression_count: int = 0
        # per-turn cache 遥测（测量面）：宿主 turn_usage.py:110-119 的 usage_dict **真含**
        # cache_read_tokens/cache_write_tokens（measurement，非 estimate）。update_from_response
        # 读桶存态（沿用 last_prompt_tokens 的存法 + 真实读数在场才写、no-usage/{} 保留既有态的
        # 哨兵保护逻辑，不冲 -1）；本 turn select_context/compress 纯读 last_cache_hit_ratio
        # （不改宿主 select_context 接口签名——红线）。宿主 agent/ 零 diff。
        self.last_cache_read_tokens: int = 0
        self.last_cache_write_tokens: int = 0
        self.last_cache_hit_ratio: Optional[float] = None  # read/prompt（prompt>0 时；miss/无读数 → None）
        # 宿主契约 real-usage pairing（对齐内置 ContextCompressor，_reset_real_usage_pairing 口径，
        # context_compressor.py:2193-2196；宿主 0 改动原则：只补契约必需字段，壳不引入算法核）：
        # - last_real_prompt_tokens：provider 真实 prompt 读数账（0 = 尚无真实读数、meter 走本地粗估）。
        #   context_breakdown.context_display_source 靠「real 是数 且 shown!=real」判 local_estimate
        #   （打 ~ 粗估标记）；此前壳缺此属性 → getattr 得 None → 恒判 provider_usage，粗估态被伪装
        #   成 provider 真实读数（桌面 meter 永不出 ~，reviewer 坐实的口径反转 bug）。
        # - awaiting_real_usage_after_compression：compact 后一次性 latch。宿主 _compress_context 提交
        #   后置位（conversation_compression.py:3496-3501），本壳 update_from_response 消费；不清则
        #   turn_preflight / turn_context_compaction 的 `if awaiting: return` skip 分支每轮命中，
        #   自动/idle compact 永久锁死（只 manual /compress force=True 能绕过）。
        # - _verify_compaction_cleared_threshold：compact 有效性 verdict latch（宿主 record_completed_
        #   compaction 面，内置 context_compressor.py:2771 同源）；壳不 arm（无该宿主算法），仅在契约面
        #   消费/清零，保持与内置同形以便宿主 getattr 直接读。
        self.last_real_prompt_tokens: int = 0
        self.awaiting_real_usage_after_compression: bool = False
        self._verify_compaction_cleared_threshold: bool = False
        # 预检阈值 = context_length × DISTILL_WATER_FRAC（0.55 engine 水线 = 压缩/蒸馏触发线；壳不重算——
        # sc/distill.py 内部 80% evict-to 软水位是组件内独立常量、勿混同，见 :84-85 语义边界注释）
        self.threshold_tokens: int = int(self.context_length * DISTILL_WATER_FRAC)
        # 宿主实调位（tui_gateway/session_compression.py:169）：threshold_tokens 配置的上限。
        # 默认 None = 不加 cap（保持 context_length × 0.55 engine 水线）；宿主下发正整数 cap 且更小时，触发线收紧到 cap。
        # 纯配置规整（镜像内置 ContextCompressor._coerce_threshold_tokens_cap，context_compressor.py:2602-2611），
        # 壳不发明新算法——缺这个方法正是宿主 "Could not apply live compression config" WARNING 的根因。
        self.threshold_tokens_cap: Optional[int] = None
        # force_evict_mode: 禁用 digest 要求，直接逐出 resident P0/P1/P2
        self._force_evict_mode: bool = False
        # P2 telemetry（precision P2 · 三指标数据闸门）：会话内观测 sink（append-only，
        # 零控制流改动 G4；JSONL 跨会话攒数喂 P3；fail-open 永不炸宿主 turn）。
        # 数据根 HERMES_HOME-aware（0 硬编码路径，G2）。
        self.telemetry = TelemetryCollector(sink_path=self.telemetry_report_path())
        # P2（S-Phase 5）drift tracker：task epoch 状态机 + 锚点 + soft/hard 判定
        #（纯 sc 层，零 LLM/零网络/确定性；复用 P1 tokenize）+ P2 telemetry 留痕面
        #（sink 回调 = 仅 _fire 那一下记会话内 drift 事件，常驻路径零开销）
        self._drift = DriftTracker(sink=self._on_drift_fire)
        # P3（S-Phase 4）gate rerank：LLM 两步重排（独立 step，P1 打分核零动，同 P2 soft_rank 模式）。
        # 端点注入 = aux 主模型端点（宿主主模型共享，零新 provider / 零模型名 / 零 provider 块）；
        # gate 默认（low_score_threshold=3.0）：BM25 活体分数分布 2026-10-04 probe（463 段 store）：
        #   强匹配 14–16 / 中等 5–14 / 弱匹配 2.08–2.13 / 不相关 0（空档 2.13–5.22）。
        #   threshold=3.0 落在空档中位：top1 < 3.0 → 触发 LLM 重排（兜弱/空/歧义面），
        #   ≥ 3.0 → BM25 有把握、零 LLM（正常搜索面常态不放大）。非常态 LLM 调用不变。
        # 缓存键 (sha256(query), store.line_count) 会话内幂等（plain dict：CPython GIL 下
        # get/set 原子，并发最坏 = 一次良性的 LLM 调用重复，不炸不脏）；
        # 会话边界（on_session_start/on_session_reset）清零，防跨 session 泄漏。
        self.rerank_low_score_threshold: float = float(_get("rerank_low_score_threshold", 3.0))
        self._rerank_gate = RerankGate(low_score_threshold=self.rerank_low_score_threshold)
        self._rerank_cache: dict = {}
        # R4（2026-10-10 · 缺口 B）：rerank gate 触发时是否默认 drop model-provenance 段。
        # config compression.supercompress.drop_model_on_gate（缺失 = 安全默认 True——gate
        # 不确定性高面不放 agent 自述段；置 False 回滚原 keep-all 现状）。零宿主 diff，
        # 实例态走 engine 内部（同 auto_recall 的 A1 模式）。
        self.drop_model_on_gate: bool = bool(_get("drop_model_on_gate", True))
        # P4（precision P4）：边界宿主 LLM 语义扩展（「花最少」= drift 边界那一下花一次宿主 LLM
        # 干三件：扩词 + top-N 精排 + 重建 epoch 锚点；G3 端点 = 宿主主模型端点代理
        # （_CallLlmEndpoint 两级路由，宿主 update_model 注入 call_llm+main_runtime，0 外部 /
        # 0 ONNX / 0 模型名硬编码）+ per-epoch 缓存（同 epoch 不重算）+ 2 档冷却回退纯 BM25
        # （默认 30s→180s 封顶，短于 digest 60→300→1800s——边界 LLM 更低频；aux 两级路由
        # cooldown 阶梯模式复用；非确定性只存在于边界，常驻路径 0 LLM 确定性不破，G4）
        self._p4_fail_cooldown: float = _get("boundary_fail_cooldown", 30.0)
        self._p4_cache: dict = {}
        self._p4 = BoundaryExpander(
            _AuxMainEndpointProxy(self), cache=self._p4_cache,
            cooldown_ladder=self._p4_cooldown_ladder())
        # P0.4 rotation advisory：纯信号置位（零注入/零遥测/零 cache 影响，宿主契约 fail-safe）；
        # 会话边界（on_session_start/on_session_reset）对称清零，防泄漏进新 session。
        # P0.4b：契约化读面——底层私有存储 _rotation_advisory（rotation_advisory 升 property，
        # 对称 force_evict_mode；宿主 getattr(agent.context_compressor, "rotation_advisory") 可读）
        self._rotation_advisory: bool = False
        self._adv_consec: int = 0

        # 紧凑摘要账本（t_63bf3ba2 · append-only 累计布局）：compress() 每事件把本次 staged
        # 块 **append** 进累计段（旧块字节原样保留）→ 前缀单调 → provider 前缀缓存跨
        # compact #1/#2/… 命中（现状 = 恒 1 块新替旧 → 前缀动、全量回落）。侧车与 digest
        # pool / model store 同目录同 sid 前缀（G2 单源 splitext）；封顶 = 整段清 + 重新
        # 积累（seal），不做 mid-block 回改。懒绑定 _ledger 属性（store 会话边界 rebind 时重建）。
        # tier 分层后（owner 裁定）：P0 定案块 trimmable=False 永不裁（存根 22 token/块）、
        # P1 推理块 trimmable=True 封顶优先裁 → 上限主要护「未折出的 P0 定案全文 + 未定案 P0」
        # 别被整段 seal 清。65536 ≈ 32 个 L3 定案段全量，容得下长会话累积而不频繁 seal 断前缀。
        # 语义钉死：adhoc = 非看板常规主路（纯自动折出）；kanban = 看板加成（半自动 candidate）。
        self.compact_ledger_max_tokens: int = int(_get("compact_ledger_max_tokens", 65536))
        # tier 分层后每次 compact 最多 append 2 块（P0 定案块 + P1 可裁块），cap 提到 16
        # = 8 次 compact 的分层块；P1 可裁层封顶优先裁，实际活跃块 = P0 定案数（≤8）。
        self.compact_ledger_max_entries: int = int(_get("compact_ledger_max_entries", 16))
        self._ledger: Optional[CompactLedger] = None

        # R1 版本戳（报告#1 #1）：构造期算一个稳定 build_id —— 一眼判断"本进程跑的
        # 到底是哪一代"：git checkout → 短 sha（working tree 有改动带 -dirty）；env 冻结
        # 副本（无 .git）→ c-内容 hash（确定性可复现，同字节在任何位置同 build_id）。
        # 纯本地观测面：任何异常 fail-open 落 "unknown"，绝不阻断引擎构造（宿主 0 改动）。
        try:
            self.build_id = build_id()
        except Exception:
            self.build_id = "unknown"
        self._desync_checked = False
        self._check_version_desync()

        logger.info(
            "supercompress: shell v2 initialized — store=%s, context_length=%d, "
            "compact_ratio=%.2f, force_ratio=%.2f, segments_on_disk=%d, call_llm=%s, "
            "build=%s",
            self.store_path, self.context_length, self.compact_ratio,
            self.force_compact_ratio, self.store.line_count,
            "available" if self._call_llm else "not available",
            self.build_id,
        )

    def _check_version_desync(self) -> None:
        """R1 脱钩自检（报告#1 #1）：若可推导 repo 源（sc.version.repo_root：SC_REPO_ROOT
        override / 自身 git checkout / 仓根祖先），对比运行副本与 repo 源的内容 hash；
        不一致打一行 ``version desync: running=<..> repo=<..>`` WARNING（只警告不阻断，
        宿主 0 改动原则下不能崩车）；repo 不可推导 → 静默跳过。纯观测面：自检自身异常
        吞掉，不影响引擎启动；每实例只查一次。"""
        if self._desync_checked:
            return
        self._desync_checked = True
        try:
            desync_warning(log=logger)
        except Exception:
            pass

    # ───────────────────── 接线件（全部委托 sc）─────────────────────

    def _build_aux(self, cfg: Dict[str, Any]) -> AuxProvider:
        """aux 端点接线：优先使用注入的 call_llm（共享 agent API），否则 _NoEndpoint。

        两级路由（`_CallLlmEndpoint`）：
        tier-1 aux 通道（COMPRESSION_TASK，短 aux_timeout 显式盖过宿主默认 30s/floor）
        tier-2 主模型（task=None + main_runtime，宿主解析主 provider/model；主模型即
        agent 运行时本身，「超时/aux 未配置 → 主模型」= 同端点稍宽重试，非独立第二 provider）
        两级瞬态失败 → aux.transient → 冷却窗去抖 + force_evict（B/C 级）兜底。
        """
        # 优先：使用注入的 call_llm（与 compressor 相同路径）
        if self._call_llm and self._main_runtime:
            try:
                ep = _CallLlmEndpoint(
                    self._call_llm, self._main_runtime,
                    aux_timeout=self.aux_timeout,
                    main_fallback_timeout=self.main_fallback_timeout,
                    digest_max_tokens=self.digest_max_tokens,
                )
                logger.info(
                    "supercompress: aux endpoint = injected call_llm (model=%s, "
                    "aux_timeout=%.0fs main_fallback_timeout=%.0fs digest_max_tokens=%d)",
                    self._main_runtime.get("model"), self.aux_timeout,
                    self.main_fallback_timeout, self.digest_max_tokens)
                return AuxProvider(ep, fail_cooldown=self.digest_fail_cooldown)
            except Exception as e:
                logger.warning("supercompress: injected call_llm endpoint failed: %s", e)
                return AuxProvider(_NoEndpoint(), fail_cooldown=self.digest_fail_cooldown)

        # 没有运行时配置时直接返回 _NoEndpoint，不尝试 env 配置
        return AuxProvider(_NoEndpoint(), fail_cooldown=self.digest_fail_cooldown)

    def update_model(
        self, model: str, context_length: int, base_url: str = "", api_key: str = "",
        provider: str = "", api_mode: str = "",
    ) -> None:
        """接收主模型信息，重建 aux 端点（使用注入的 call_llm 共享 agent API）。"""
        # 保存主模型运行时配置（不复制，引用 agent 的运行时对象）
        self._main_runtime = {
            "model": model,
            "provider": provider,
            "base_url": base_url,
            "api_key": api_key,
            "api_mode": api_mode,
        }
        self.context_length = context_length
        # 尝试重新导入 call_llm（可能在 __init__ 时因依赖缺失而失败）
        if self._call_llm is None:
            self._call_llm = _get_call_llm()
        # 重建 aux（优先使用 call_llm）
        self._aux = self._build_aux({})
        # 重置 force_evict 标志
        self._force_evict_mode = False
        # 重算阈值
        self.threshold_tokens = int(context_length * DISTILL_WATER_FRAC)
        # P4：端点代理活读 self._aux.main_endpoint（update_model 重建 _aux 后自动骑上新端点）——
        # P4 状态（per-epoch 缓存 / 冷却阶梯）为端点健康 scoped 与会话 scoped 的交叉面：
        # 端点本身换了（新宿主运行时）→ 冷却连败归零（旧端点劣化不迁怒新端点）。
        self._p4.consec_fails = 0
        self._p4._cooldown_until = 0.0
        logger.info("supercompress: update_model model=%s ctx=%d aux=%s runtime=%s call_llm=%s",
                    model, context_length, type(self._aux.main_endpoint).__name__,
                    "injected" if self._main_runtime else "none",
                    "available" if self._call_llm else "unavailable")
        # B 卡：保险丝检查（2026-10-09 用户裁定后仅剩 ② 5MB 体积上限）：旁路 store 自动
        # 适配任何 model / 任何版本（③ 版本失配已撤），行数比 ① 亦撤（carry-over 基准错配
        # 误熔断）——check_fuse 仅按文件体积判（sc/model_store，仿 digest pool guard()）；
        # fused → 召回并集 / T4 注入跳过该 store（文件保留不删，下 session 重放再判）。
        # 传 main_line_count / current_model 仅为调用面签名兼容（check_fuse 不再消费二者）。
        if self.model_store_enabled and self._model_store is not None and not self._model_fused:
            if self._model_store.check_fuse(self.store.line_count, model):
                self._model_fused = True

    @property
    def force_evict_mode(self) -> bool:
        """是否启用强制逐出模式（跳过 digest 要求）。"""
        return getattr(self, '_force_evict_mode', False)

    @force_evict_mode.setter
    def force_evict_mode(self, value: bool) -> None:
        self._force_evict_mode = value

    # ── P0.4b rotation 信号契约面（纯 engine.py 内部新增；宿主 getattr 即读，sc/ 层零动）──

    @property
    def rotation_advisory(self) -> bool:
        """P0.4 rotation advisory 置位态（对称 force_evict_mode property；底层 _rotation_advisory）。

        宿主契约读面：`getattr(agent.context_compressor, "rotation_advisory")` 一行即读；
        内部赋值点（_check_rotation_advisory / on_session_start / on_session_reset 清零）
        自动经 setter 落底层私有存储，语义与原 plain attribute 完全等价。"""
        return getattr(self, '_rotation_advisory', False)

    @rotation_advisory.setter
    def rotation_advisory(self, value: bool) -> None:
        self._rotation_advisory = value

    def pressure(self) -> float:
        """当前窗口压力 = last_prompt_tokens / context_length（O(1) 纯读，除零安全：
        context_length <= 0 → 0.0，对称 _check_rotation_advisory 的既有防御）。

        宿主契约读面：宿主 / 未来 TUI 卡一行 getattr 即可算出窗口压力，无需重读 token 账。"""
        if self.context_length <= 0:
            return 0.0
        return self.last_prompt_tokens / self.context_length

    def rotation_reason(self) -> str:
        """rotation advisory 的宿主可读文案（一行）：已置位 → 原因串（含 pressure 与
        连续 turn 数）；未置位 → "none"。纯 O(1) 读，零 side-effect。"""
        if not self.rotation_advisory:
            return "none"
        return f"window pressure {self.pressure():.2f} for {self._adv_consec} consecutive turn(s)"

    def should_carry_over(self) -> bool:
        """P0.4b 纯 advisory 谓词：advisory 已置位 且 当前压力 ≥ ROTATION_PROMPT_RATIO
        （与 _check_rotation_advisory 置位线同源常量，零硬编码；pressure() 除零安全）。

        零 side-effect：不动 store / window / last_prompt_tokens / 任何持久态——纯读。
        边界：这是给宿主的读面，实际 carry 仍由宿主 `carry_over_context` 单轨门控
        （on_session_start，P0.4 设计不变）；宿主不调本方法即零行为、零注入、零 cache 破坏。"""
        return self.rotation_advisory and self.pressure() >= ROTATION_PROMPT_RATIO

    def _window_fresh(self) -> Window:
        """窗口 = 磁盘 replay 段（单一事实源；内存工作态崩溃清零 = session 重启已知边界）。"""
        segs = self.store.replay()
        # P4 钩子①（replay 查池，t_026a17f8）：池命中段 → 内存态 digest_ref=done + digest_text/
        # key_fact 回填，重启后该段免重蒸（distill 的 pending/digest_ready 门不再触达它）；
        # jsonl 源层字节零动（done 是内存态，不回写行）。未命中 → 现有 pending 流程不变。
        self._digest_pool_replay_hook(segs)
        self._window = Window(self.context_length, [WindowSeg(s) for s in segs])
        self._tagger = Tagger(self._aux, "")
        return self._window

    # ── P4 digest sidecar 水池（engine 薄接线，算法全在 sc/digest_pool.py，t_026a17f8）─────

    def _digest_pool(self):
        """当前 session 池 = 与源层 store 文件**同目录、同 sid 前缀**的派生池（`<sid>.jsonl`
        → `<sid>.digests.json`）；路径解析全在池模块（pool_path_for_jsonl 单源，零硬编码路径，
        生产 hermes_home 目录 / 测试 external_store_path 临时目录两种落点都对、GC 联动同目录
        语义自洽）。store 为 legacy active.jsonl 时池名 active.digests.json（同源同住）。"""
        return sc_digest_pool.DigestPool(sc_digest_pool.pool_path_for_jsonl(self.store.path))

    # ── 紧凑摘要账本（engine 薄接线，算法全在 sc/compact_ledger.py，t_63bf3ba2）─────────────

    def _compact_ledger(self) -> CompactLedger:
        """当前 session 账本访问器（rebind-safe，仿 _digest_pool()）：按 store.path 失配懒重建，
        锚定到 session 源层 jsonl 实际目录（`<sid>.jsonl` → `<sid>.compact.json`，splitext 单源，
        与 digest pool / model store 同目录同 sid 前缀，GC 联动语义自洽）。

        会话边界 rebind（轮转 materialize / 新 session）由 on_session_start 显式
        `self._ledger = None` 触发重建；本访问器只在 path 失配（跨 store 实例）时兜底重建。"""
        expected = compact_ledger_path_for_jsonl(self.store.path)
        lg = self._ledger
        if lg is not None and lg.path == expected:
            return lg
        lg = CompactLedger(
            expected,
            max_tokens=self.compact_ledger_max_tokens,
            max_entries=self.compact_ledger_max_entries,
            logger=logger,
        )
        self._ledger = lg
        return lg

    def _compact_ledger_reset(self) -> None:
        """会话边界（/new · 轮转 rebind）整段清账本：清空 + generation/seals +1 + 落盘。

        与 rerank/p4 缓存清零同口径（会话记忆 scope 不得跨 session 泄漏）；
        路径已切新 session 侧车时新文件天然空、本方法清的是**旧 session 内存账本态**，
        保证 rebind 后 compress() 从空账本重新积累（旧 session 累计块不跨 session 前缀）。"""
        lg = self._ledger
        if lg is not None:
            lg.reset()
            logger.info("supercompress: compact-ledger reset (session boundary) "
                        "generation=%d seals=%d path=%s", lg.generation, lg.seals, lg.path)

    # ── settled P0 半自动翻转面（owner 裁定 B 路线：引擎侧闭环，看板 task 重新活跃 → unsettle 复活）──
    def kanban_candidates(self) -> List[Dict[str, Any]]:
        """列出所有待拍板的 kanban candidate 块（settled=False + kind=kanban，
        由 compress() 写侧 `_staged_p0_decision()=="kanban"` 提名产生）。

        宿主 LLM 裁判读本面拿候选清单（O(n) 点查，账本 blocks 数小）→ 逐个提名
        → clarify 用户确认后才调 `settle_p0(seq, kind="kanban")` 真正折出。
        零新索引结构；fail-open（账本缺失/损坏 → 空清单，不炸宿主）。"""
        try:
            lg = self._compact_ledger()
            return [e for e in lg.entries
                    if (not e.get("settled")) and e.get("kind") == "kanban"]
        except Exception:
            return []

    def settle_p0(self, seq: int, kind: str = "kanban") -> Optional[bool]:
        """把 c#seq 块折出（settled=True，下轮渲染折单行存根；原文沉盘 <sid>.p0）。

        看板半自动面：宿主 LLM 裁判对 `kanban_candidates()` 提名 + clarify 用户确认后
        才调本面（不读 kanban 自动钩子，守「改动只在 supercompress」红线）。
        idempotent：已 settled / 找不到 seq → False（零写盘）。
        异常 fail-open 吞掉（宿主零炸）；只返回是否真翻转，零控制流旁路。
        """
        try:
            lg = self._compact_ledger()
            e = lg.settle(seq, kind=kind)
            if e is not None:
                _sink_dir = os.path.dirname(os.path.abspath(lg.path))
                _sid = os.path.splitext(os.path.basename(lg.path))[0].replace(".compact", "")
                if not scdata.sink_p0(_sink_dir, seq, e["text"], kind=kind, sid=_sid):
                    logger.warning("supercompress: settle_p0 c#%d flipped but p0 sink "
                                   "failed (in-memory only)", seq)
                logger.info("supercompress: compact-ledger c#%d settled (kind=%s)", seq, kind)
            return e is not None
        except Exception as e:
            logger.warning("supercompress: settle_p0(%s) failed open: %s", seq, e)
            return False

    def unsettle_p0(self, seq: int) -> Optional[bool]:
        """看板 task 重新活跃 → 复活 c#seq 块（settled=False，活渲染面恢复全文，可审计）。

        幂等：本就未 settled / 找不到 seq → False。原文仍存于账本 entry.text（沉盘件在，
        不丢数据）；find-fail 时若账本条目无 text（已被 seal 清）则尝试 restore_p0 捞回
        （复活面，fail-open 缺失 → False 静默）。
        """
        try:
            lg = self._compact_ledger()
            e = lg.unsettle(seq)
            if e is not None:
                logger.info("supercompress: compact-ledger c#%d unsettled (resumed)", seq)
                return True
            # 条目已不在内存账本（seal 清过）→ 尝试沉盘捞回（fail-open）
            _sink_dir = os.path.dirname(os.path.abspath(lg.path))
            _sid = os.path.splitext(os.path.basename(lg.path))[0].replace(".compact", "")
            got = scdata.restore_p0(_sink_dir, seq, sid=_sid)
            logger.info("supercompress: unsettle_p0 c#%d not in live ledger; "
                        "disk restore=%s (entry already sealed; nothing to revive in-memory)",
                        seq, "ok" if got else "miss")
            return False
        except Exception as e:
            logger.warning("supercompress: unsettle_p0(%s) failed open: %s", seq, e)
            return False

    def untrim_p1(self, seq: int) -> Optional[bool]:
        """R5（owner 批次 A #2）：复活被封顶折存根的 P1 可裁块 c#seq。

        先试活账本 untrim（折存根块仍在账本 = 常态，零 IO，entries 里 trimmed=True
        的块改回全权重回账本渲染面）；账本无此 seq（已 seal 清过 → entries=[] 全清）
        → 尝试 <seq>.p1 沉盘件读回**原文做审计参照**（fail-open 缺失 → miss，只 log
        不自动重插入账本——seal 是整段作废语义，复活要重新 accumulate，盘件供人工
        审计 / 后续再累积参照，非自动 untrim）。异常 fail-open 吞掉（宿主零炸）。

        诚实边界（reviewer P4）：docstring 旧措辞"捞回（同 restore_p0 复活面）"误导
        为"自动复活"，实际 seal 后 restore_p1 只读盘 log 不重插入 entries——本处已
        更正为"盘件供审计参照"。untrim_p1 返回值语义：True=活账本内 untrim 成功
        （块重回渲染面）；False=条目已 seal 清（沉盘件仅供审计，未自动复活）。"""
        try:
            lg = self._compact_ledger()
            e = lg.untrim(seq)
            if e is not None:
                logger.info("supercompress: compact-ledger c#%d P1 untrimmed (resumed)", seq)
                return True
            # 条目已不在活账本（seal 清过）→ 尝试 <seq>.p1 沉盘件捞回（fail-open 缺失 → miss）
            _sink_dir = os.path.dirname(os.path.abspath(lg.path))
            _sid = os.path.splitext(os.path.basename(lg.path))[0].replace(".compact", "")
            got = scdata.restore_p1(_sink_dir, seq, sid=_sid)
            logger.info("supercompress: untrim_p1 c#%d not in live ledger; "
                        "disk restore=%s (entry already sealed; nothing to revive in-memory)",
                        seq, "ok" if got else "miss")
            return False
        except Exception as ex:
            logger.warning("supercompress: untrim_p1(%s) failed open: %s", seq, ex)
            return False

    def _digest_pool_replay_hook(self, segs) -> None:
        """钩子①（replay 查池）：池命中且 model_version 匹配 → 段内存态 digest_ref=done，跳过重蒸。

        薄壳纪律：保险丝/版本失配整池作废（guard）、fail-open 读（load → {hash: entry}，
        缺失/损坏/结构不符一律空池回落现状）、逐段「命中且版本匹配」判定（load 结果 dict 查）
        全在 sc/digest_pool.py；壳只做 命中 → Segment 内存字段 的接线翻译，jsonl 行字节零动。
        marker 语义：spec 的「digest_ref=done」落在既有取值域 = `ready:<refid>`（digest_ready
        唯一判据即 startswith("ready:")，done = digest 已就绪免再生）——本壳钉 `ready:pool`
        （池水合标记，refid 不参与任何下游判定，跨进程稳定靠 content_hash 键）；digest_text/
        key_fact 回填使注入面 L2/L3 直接可用（DoD1：重启后注入未降级 head-80）。
        异常一律 fail-open（池=不可信派生层，零崩溃，行为=现状 pending + head-80，DoD2）。
        """
        if not segs:
            return
        try:
            pool = self._digest_pool()
            # 前置闸（读池前）：保险丝（>5MB / 条目>段数）+ model_version 失配 → 整池作废重蒸。
            # guard 纯判+作废一体（池模块）；命中 → 池已清空，直接回落现状 pending（= DoD2）。
            if pool.guard(model_version=self._current_model_version(),
                          segment_count=len(segs)):
                return
            entries = pool.load()                     # 单读全量（池模块 docstring 既定引擎读法）
            n_hit = 0
            for seg in segs:
                if seg.structural_p0:
                    continue                          # 结构地板（goal/ledger/用户指令）免蒸馏，不入池
                entry = entries.get(seg.content_hash)
                if entry is None:
                    continue                          # 未命中 → 现有 pending 流程不变
                seg.digest_ref = "ready:pool"         # done 语义：digest_ready=True，跳过重蒸
                seg.digest_text = str(entry.get("digest_text") or "")
                seg.key_fact = str(entry.get("key_fact") or "")
                n_hit += 1
            if n_hit:
                logger.info("supercompress: digest-pool replay hydrated %d segment(s) "
                            "digest_ref=done (re-distill skipped; jsonl bytes untouched)",
                            n_hit)
        except Exception as e:                        # 池=缓存，读面任何异常零崩（DoD2 回落现状）
            logger.warning("supercompress: digest-pool replay hook no-op (fail-open): %s", e)

    def _digest_pool_write_hook(self) -> None:
        """钩子②（digest 完成回写池）：新成功生成的 digest 即时原子 append（不批量，池模块
        upsert = load → 覆盖 → tmp+rename fsync 全在池模块）。

        调用点 = 「digest 刚可能新完成」的三处（on_turn_complete 蒸馏后 / compress 蒸馏后 /
        on_session_end aux.flush 汇合后）：此刻 digest_text 已填充的段 = 成功生成（端点真调用
        成功或钩子①池水合回填）；transient 跳过的段 digest_text 保持 ""（[no digest] marker 路径）
        不入池（下事件再生，池只兜成功产物）。

        去重 = **池即事实源**（壳无实例账，无生命周期可漂移）：shell 先内存薄过滤候选
        （digest_ready + 非结构地板 + digest_text 非空，零 IO）；有候选才 load() 池快照一次，
        只 upsert 池里尚无的 content_hash——幂等（已含条目跳过，不重复写盘）且 guard 整池作废
        后自愈（清空池 → 快照空 → 全量重写）。写失败（full-disk ENOSPC/EIO → PoolWriteError）
        fail-open：池=缓存，下事件再生，宿主收尾面零炸。"""
        try:
            cand = [seg for seg in self.store.by_hash.values()
                    if seg.digest_ready and not seg.structural_p0
                    and getattr(seg, "digest_text", "")]
            if not cand:
                return
            pool = self._digest_pool()
            pooled = pool.load()                      # 池即去重事实源（快照一次）
            mv = (self._main_runtime or {}).get("model", "")
            new = [seg for seg in cand if seg.content_hash not in pooled]
            if not new:
                return                                 # 全在池 → 零写盘（幂等）
            for seg in new:
                pool.upsert(content_hash=seg.content_hash, model_version=mv,
                            key_fact=str(seg.key_fact or ""),
                            digest_text=str(seg.digest_text or ""))
            logger.info("supercompress: digest-pool write-back appended %d entr(y/ies)", len(new))
        except sc_digest_pool.PoolWriteError as e:    # 池写 OSError 归一：缓存失败非致命
            logger.warning("supercompress: digest-pool write-back failed (fail-open, "
                           "re-generates next event): %s", e)
        except Exception as e:                        # 池=缓存，写面任何异常零崩（DoD2）
            logger.warning("supercompress: digest-pool write-back no-op (fail-open): %s", e)

    def _current_model_version(self):
        """池的「当前模型」锚点 = 宿主 update_model() 注入的主模型名（_main_runtime.model）；
        未注入 → None（guard 跳过版本判、只做保险丝；池 upsert 的 model_version 落 ""——
        段自身 model_version 非壳注入面，宿主零改动约束下取 当前运行时模型 为池条目版本键）。"""
        return (self._main_runtime or {}).get("model") or None

    # ── B 卡 model 旁路 store（engine 薄壳接线，算法全在 sc/model_store.py；仿 P4 digest pool 钩子形）──

    def _model_store_path(self) -> str:
        """model 旁路路径 = 主 store 路径 splitext + '.model.jsonl'（sc.model_store 单源，
        与 digest 池同目录同 sid 前缀；legacy active.jsonl → active.model.jsonl）。"""
        return sc_model_store.model_path_for_jsonl(self.store.path)

    def _model_model_version(self) -> str:
        """入库 stamp = 当前 model（宿主 update_model 注入 _main_runtime.model）；未注入 → ""
        （空 stamp 行 = 版本保险丝判定中性，沿用 digest pool guard 判定口径）。"""
        return (self._main_runtime or {}).get("model", "")

    def _model_tier_one(self, seg) -> bool:
        """tier 队列单段精化回调（宿主接线）：sc/tagger classify(text, "model") LLM 分支
        （宿主共享 call_llm，aux 两级路由 + fail cooldown 60→300→1800s + 连败 ≥3 degraded，
        全在既有 sc/aux_provider 语义内；壳零新 provider / 零模型名 / 零 provider 块）。

        True = 精化已应用（aux_ok，仅更新内存 tier 字段）；False = 保持 P0 地板（aux_floor：
        aux 失败/degraded/队列丢弃 → 磁盘 tier 恒 P0，冷重启全 P0 重放 = 保守地板，已文档化
        取舍；structural_p0 恒 False —— model 段可被 push cap 逐，P0/P1 注入门放行）。"""
        tagger = self._tagger or Tagger(self._aux, "")
        _res = tagger.classify(seg.original_text, "model")
        _tier, _structural, regime, _reason = _res[:4]
        _settled_signal = _res[4] if len(_res) > 4 else 0
        if regime == "transient_fail":
            return False                    # 瞬态失败/degraded → P0 地板（保守，永不当 P2 丢）
        seg.tier = _tier                    # 精化仅内存（never re-tier on replay，scdata 锁定）
        seg.settled_signal = _settled_signal  # 4a：LLM tagging 面打的定案信号码（纯内存）
        return True

    def _model_store_live(self):
        """当前 session model 旁路访问器（rebind-safe，仿 _digest_pool()；kill-switch 感知）：
        on_session_start 三分支先绑定（_build_model_store）；本方法按 store.path 失配懒重建，
        fused 态同步落壳标志。命名 = 与 self._model_store **实例属性**（当前旁路对象 / None）
        区分——属性优先，本访问器是路径失配重建面（同名方法会被 __init__ 赋的属性遮蔽，
        前轮遗留命名碰撞已消解）。"""
        if not self.model_store_enabled:
            return None
        expected = self._model_store_path()
        ms = self._model_store
        if ms is not None and ms.path == expected:
            return ms
        ms = sc_model_store.ModelStore(expected, tier_fn=self._model_tier_one)
        self._model_store = ms
        self._model_fused = ms.fused
        return ms

    def _model_replay_hook(self) -> None:
        """钩子①（on_session_start 三分支统一入口）：model store replay hydrate marker
        （`model-store replay hydrated N segment(s)`，N>0 才打；缺失 = 空池合法态不 marker；
        损坏/IO 不可读已在 ModelStore 构造 fail-open fused、marker 在 sc 层）+ 保险丝检查
        （2026-10-09 用户裁定后仅 ② 5MB 体积上限；① 行数比 / ③ model 版本失配已撤——旁路
        自动适配任何 model/版本，carry-over 池不再被新 session 主 store 行数误熔断；算法在
        sc/model_store.check_fuse，仿 digest pool guard()）。
        fused → 壳 _model_fused 标志（召回并集 + T4 注入跳过该 store；文件保留不删）。"""
        if not self.model_store_enabled:
            self._model_fused = False
            return
        ms = self._model_store
        if ms is None:
            return
        if ms.line_count > 0:
            logger.info("model-store replay hydrated %d segment(s)", ms.line_count)
        ms.check_fuse(self.store.line_count, self._model_model_version())
        self._model_fused = ms.fused

    def _model_ingest_hook(self, messages: List[Dict[str, Any]]) -> None:
        """钩子②（on_turn_complete）：model/assistant 段选择性入库 gate（确定性零 LLM，B 卡决策 3）。

        - 只 role=='assistant' 面（role=='tool' 永不入库；user 段主 store 路径零行为变化）；
        - 持久化 = len(text) ≥ model_min_chars（默认 200，可配）OR 命中有意义 marker
          （sc/tagger _P1/_P2 marker 族同口径；压缩摘要类无论长度恒持久化）；
        - text 空 strip → skip（tool-call-only）；nfc_hash 命中 model 旁路或主 store → dedup skip；
        - 命中 → 临时档 P0 非结构落盘 + 读回 + tier 队列（先持久化再精化，B 卡决策 4）；
        - marker：`model-store: ingested N assistant segment(s) (gate=…)`，N>0 才打（宿主侧 grep 验证）；
        - fail-open：fused 跳过 / sidecar 写失败由 on_turn_complete 外层 catch 吞掉，不炸宿主 turn。"""
        if not self.model_store_enabled or self._model_fused:
            return
        ms = self._model_store
        if ms is None:
            ms = self._model_store_live()
            if self._model_fused or ms is None:
                return
        counts: Dict[str, int] = {}
        ingested = 0
        for m in (messages or []):
            if m.get("role") != "assistant":
                continue
            ok, gate = ms.ingest_assistant(
                str(m.get("content") or ""), min_chars=self.model_min_chars,
                model_version=self._model_model_version(), main_by_hash=self.store.by_hash)
            if ok:
                ingested += 1
                counts[gate] = counts.get(gate, 0) + 1
        if ingested:
            gate_txt = ",".join("%s=%d" % (k, v) for k, v in sorted(counts.items()))
            logger.info("model-store: ingested %d assistant segment(s) (gate=%s)",
                        ingested, gate_txt)

    def _model_session_flush(self, action: str) -> None:
        """会话边界 flush 点（aux worker flush 后，model 旁路 P1/P2 lazy 批 flush；
        镜像 on_session_reset/end 主 store flush 块，OSError/StoreError 优雅降级 + 水位观测）。

        磁盘 tier 恒 P0 不变量 → lazy 批实际恒空（生命周期语义镜像；未来若 P1/P2 落盘即自动
        生效）；marker 复用既有 flush 族 + store=main|model 判别；fused 旁路 = no-op。"""
        ms = self._model_store
        if not self.model_store_enabled or ms is None:
            return
        try:
            n = ms.flush_lazy()
        except (StoreError, OSError) as e:
            logger.warning("supercompress: %s model-store flush failed (degraded, "
                           "pending P1/P2 kept in memory, store=model): %s", action, e)
            try:
                ms.log_water_warn(action + "-model-flush-failed", e)
            except Exception:
                pass
            return
        if n:
            logger.info("supercompress: %s model-store flushed %d segment(s) (store=model)",
                        action, n)
        # B 卡：tier worker 队列汇合点（会话边界 = 重启前权威汇合点，同 aux flush(timeout=5.0) 对称；
        # 先持久化再精化 → 超时/在途段保持 P0 地板（磁盘行恒 P0，下事件再生，已文档化取舍）；
        # sync 面（测试注入）= 立即 drain 返回 True，零线程、确定性。marker 复用既有 flush 族。
        try:
            if ms.flush(timeout=2.0):
                logger.info("supercompress: %s model-store tier worker joined (store=model)",
                            action)
        except Exception:
            pass

    def _recall_union_window(self, window: "Window") -> "Window":
        """召回语料并集（B 卡决策 5）：主 store 窗口 replay + model 旁路内存 live 段
        （fused / 关闭 / 空旁路 → 原样返回主窗口）；打分权重不变（P0×1.5 等按段自带 tier，
        P2 tag 的 model 段可命中排序但不注入 —— P0/P1-only 注入门对 model 段同语义生效，
        与主 P2 同语义）。仅召回 / P4 候选面；distill 窗口（主 store 独占）不变——
        push cap / 水位语义零扰动。"""
        if not self.model_store_enabled or self._model_fused:
            return window
        ms = self._model_store
        if ms is None:
            return window
        mseg = ms.live_segments()
        if not mseg:
            return window
        return Window(self.context_length, list(window.segs) + [WindowSeg(s) for s in mseg])

    def point_fetch(self, segment_id: str):
        """L0 指针 O(1) 点查（B 卡决策 6，文档化解析顺序）：主 store 优先、model 旁路次
        （`store.get(id) or model_store.get(id)`）——evicted model 段仍可 O(1) 点查。

        跨 store id 冲突安全性（id = content_hash[:16]+seq；不同内容 → 16-hex 前缀不同，
        同文已被入库 gate dedup 封死双写 → 两 store 间无重 id）由 tests/test_model_store.py 验证。"""
        seg = self.store.get(segment_id)
        if seg is None and self.model_store_enabled and not self._model_fused:
            ms = self._model_store
            if ms is not None:
                seg = ms.get(segment_id)
        return seg

    def _build_model_store(self, old_file: str = None, old_sid: str = None):
        """model 旁路构造点（B 卡 kill-switch 语义闭环，decision 8/9）：

        - `model_store_enabled=False` → None（零文件 / 零 tier 线程——开关即关，不留半态）；
        - carry 分支（old_file + old_sid 传入）：旧 model 文件存在 → 并行 materialize
          （字节稳定拷贝 + P1 stale 标记，与主 store JsonlStore.materialize 平行），
          否则全新空池（旧文件缺失 = 合法态，不 fuse）；
        - new / cold 分支（无参）：构造即 replay（与主 store 平行，同 sid 前缀）。

        旁路路径单源 = sc.model_store.model_path_for_jsonl（与主 store splitext + '.model.jsonl'，
        决策 1 同住同 sid 前缀 / G2 单源）。"""
        if not self.model_store_enabled:
            return None
        if old_file is not None:
            old_model_file = sc_model_store.model_path_for_jsonl(old_file)
            if os.path.exists(old_model_file):
                return sc_model_store.ModelStore.materialize(
                    old_model_file, self._model_store_path(), reopen_from=old_sid,
                    tier_fn=self._model_tier_one)
        return sc_model_store.ModelStore(self._model_store_path(),
                                         tier_fn=self._model_tier_one)

    # ── P2 telemetry（precision P2 · 三指标数据闸门）：宿主可读面 + drift 留痕回调 ──

    def telemetry_report_path(self) -> str:
        """P2 telemetry sink 路径（JSONL，跨会话 append-only 攒数喂 P3）。

        数据根 HERMES_HOME-aware（同 _default_store_root 单源语义；G2：0 硬编码路径）：
        - HERMES_HOME 设置 → <HERMES_HOME>/supercompress_store/telemetry.jsonl
          （per-profile 隔离白捡，与 on_session_start 生产 rebind 目录同住）；
        - 未设（独立测试/bench，external_store_path 显式指定数据根）→
          <_legacy_store_dir>/telemetry.jsonl（tmp 目录直落，测试面自洽）。
        """
        hh = os.environ.get("HERMES_HOME", "").strip()
        if hh:
            # 保留 expanduser(hh) 分支（env 值若含 ~ 亦展开，与本规则逐字口径）；
            # 仅目录名字面收敛到共享常量（G2 单源），不改变路径语义。
            return os.path.join(os.path.expanduser(hh), STORE_SUBDIR, "telemetry.jsonl")
        return os.path.join(self._legacy_store_dir, "telemetry.jsonl")

    def _on_drift_fire(self, evt: dict) -> None:
        """P2 telemetry（指标③ drift 留痕面）：仅 drift 事件 _fire 那一下回调
        （sc.drift.DriftTracker sink 缝；常驻 observe_user 路径零开销，零控制流改动 G4）。

        fail-open 契约：遥测记录异常不炸宿主 turn（telemetry 非关键面，同宽 catch 口径）。
        会话内 records 攒数，会话边界（on_session_reset / on_session_end）flush 落 JSONL。"""
        try:
            self.telemetry.record_drift(events=1, epoch=evt.get("epoch"), fired=True)
        except Exception:
            pass

    # ── 宿主调用面（abstract 三件 + 可选钩子）──

    def _record_cache_telemetry(self, usage: Dict[str, Any], log_hit: bool = True) -> None:
        """per-turn cache 遥测（测量面）：读宿主喂入 usage 的 cache 桶 → 存实例态（+ 命中行打日志）。

        对齐宿主 turn_usage.py:190-208 的 cache=X/Y (Z%) 口径：
        - cache_read_tokens/cache_write_tokens 缺键/None → 0（容错，不炸；旧宿主无 cache 键亦安全）；
        - 命中率 = read/prompt（prompt 优先取 usage 自带 prompt 键、缺失回退刚写的 last_prompt_tokens；
          prompt>0 时有效，否则 None 不除零）；
        - log_hit=True 且 read>0 且 prompt>0（真命中）→ 打一行 `supercompress: turn cache hit`；
          miss（read=0）→ 不打行（与宿主 cache=X/Y 口径一致：miss = 无行，非 0% 噪声）。
        宿主 agent/ 零 diff、不改 select_context 签名——纯读宿主已喂入的 usage + 存实例态。
        """
        _u = usage or {}
        read = int(_u.get("cache_read_tokens") or 0)
        write = int(_u.get("cache_write_tokens") or 0)
        prompt = int(_u.get("prompt_tokens") or _u.get("input_tokens") or 0)
        if prompt == 0:
            prompt = self.last_prompt_tokens  # usage 无 prompt 键时沿用刚写的 prompt 桶
        self.last_cache_read_tokens = read
        self.last_cache_write_tokens = write
        self.last_cache_hit_ratio = (read / prompt) if prompt > 0 else None
        if log_hit and read > 0 and prompt > 0:
            logger.info("supercompress: turn cache hit read=%d prompt=%d (%.0f%%) write=%d",
                        read, prompt, 100.0 * read / prompt, write)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """token 账更新（宿主每次 LLM 调用后）：直接存 canonical 桶，无算法。

        宿主契约对齐内置 ContextCompressor.update_from_response（context_compressor.py:2764-2771）
        —— 只补契约必需的 real-usage pairing + latch 生命周期，壳不引入压缩算法核（distill/recall/
        budgets 仍全住 sc/）：

        - **`-1` 哨兵保护**：宿主 _compress_context 提交 compact 时写 `last_prompt_tokens = -1` 并 arm
          `awaiting_real_usage_after_compression`（conversation_compression.py:3496-3501，#36718 哨兵
          语义：「压缩刚跑完、等下一次真实读数」）。provider 某响应无 usage 时，宿主调
          `update_from_response({})`（turn_usage.py:87-91）—— 若裸 `int(get(...,0))` 会把 `-1` 冲成 0，
          破坏哨兵（preflight 的 `last_prompt_tokens == -1` 分支、宿主 meter 粗估判据都会失真）。故真实
          prompt 读数在场（prompt_tokens/input_tokens 有值）才写 `last_prompt_tokens`；无 usage 时保留
          既有哨兵态（镜像内置 rollback_interrupted_preflight_display_tokens 对 `-1` 的保护，
          context_compressor.py:2816-2820）。
        - **`last_real_prompt_tokens`**（provider 真实读数账，acceptance #4）：真实 prompt_tokens>0 时
          记账。`context_breakdown.context_display_source` 靠「real 是数 且 shown != real」判 local_
          estimate（打 `~` 粗估标记）；此前壳缺此属性 → getattr 得 None → 恒判 provider_usage，粗估态
          被伪装成 provider 真实读数。
        - **latch 消费**：真实 usage 到达后清 `awaiting_real_usage_after_compression`（即使无 prompt_
          tokens 也消费，使 auto/idle compact 的 `if awaiting: return` skip 分支不再每轮命中、永久锁死）。
        """
        # 真实 usage 才写 token 桶；无 usage（宿主 {} no-usage 路径）→ 全保留既有桶（含 -1 哨兵），
        # 只消费 latch（镜像内置 rollback 对 -1 的保护 context_compressor.py:2816-2820 + turn_usage
        # no-usage 契约 turn_usage.py:87-91：裸 int(get(...,0)) 会把 -1 冲成 0，破坏 #36718 哨兵）。
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        if prompt != 0 or completion != 0:
            self.last_prompt_tokens = int(prompt)
            self.last_completion_tokens = int(completion)
            _total = usage.get("total_tokens")
            self.last_total_tokens = int(
                _total if _total else self.last_prompt_tokens + self.last_completion_tokens)
            # provider 真实 prompt 读数（>0）记账：context_breakdown.context_display_source 判
            # local_estimate（打 ~ 粗估）的依据（acceptance #4；此前壳缺此属性 → getattr 得 None）。
            if self.last_prompt_tokens > 0:
                self.last_real_prompt_tokens = self.last_prompt_tokens
            # per-turn cache 遥测（测量面）：仅在真实 usage 在场时记 cache 桶 + 命中率 + 命中行；
            # 宿主 no-usage（{}）路径不进此块 → 保留既有 cache 态（含 -1 哨兵下的上一次真实读数），
            # 不误清——与 prompt/completion 桶的哨兵保护同口径。
            self._record_cache_telemetry(usage)
        # 消费 real-usage latches（一次 usage 到达即清，防 auto/idle compact 永久 skip）：
        # 与内置 update_from_response 末行同源（context_compressor.py:2771）。
        self._verify_compaction_cleared_threshold = self.awaiting_real_usage_after_compression = False

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """engine 水线语义：当前 token 账 > 生效触发线（= min(宿主 cap, context_length × DISTILL_WATER_FRAC 0.55)；
        sc/distill.py 内部 80% evict-to 软水位是组件内独立常量、非本触发线）→ 本 turn 触发蒸馏逐出。
        cache-aware 决策面（owner 指令「1 要」）：生效线经 _cache_aware_trigger_threshold() 规整——
        上 turn 前缀 cache 命中高（last_cache_hit_ratio ≥ hit_min）→ 触发线抬高 base×(1+delay_frac)
        （推迟压缩、省 re-prime 成本）；命中低 / ratio None / config 关 → 恒等于现状 base（零变化）。
        活体 marker（仅 cache-aware **真正抬高了线**、且决策态 0↔1↔2 翻转时打，防宿主同 turn 双调
        / 同态多 turn 刷屏；未介入的基线路径零新增日志 = fail-safe 零变化可证）：
        - 态 2（推迟）：抬高了线、且基线会触发（tokens>base）但被抬高线拦住（tokens≤adjusted）
          → `cache-aware delay compress (hit=NN%)`（高命中推迟压缩的活体证据）
        - 态 1（维持）：抬高了线、且抬高线后仍过线触发（tokens>adjusted）
          → `cache-aware keep trigger (hit=NN%)`（维持触发的活体证据）
        - 态 0（未触发 / 线未抬高 / 抬高线但 tokens 连基线都没过、无推迟可言）：零变化，不打行。"""
        tokens = prompt_tokens if prompt_tokens else self.last_prompt_tokens
        base = self._effective_threshold()
        threshold = self._cache_aware_trigger_threshold()
        triggered = tokens > threshold
        # 决策态只在 cache-aware 真正抬高线（threshold>base）时可能取 1/2；否则基线路径恒 0（零日志）。
        # 抬高线后：过线触发 = 1（维持）；未触发且 tokens>base（基线本会触发、被推迟）= 2（推迟）；
        # 抬高线但 tokens≤base（连基线都没过、谈不上推迟）= 0（不打行）。
        if threshold > base:
            state = 1 if triggered else (2 if tokens > base else 0)
        else:
            state = 0
        if state != self._cache_aware_active:
            self._cache_aware_active = state
            if state == 1:
                logger.info("supercompress: cache-aware keep trigger (hit=%s)",
                            self._cache_hit_label())
            elif state == 2:
                logger.info("supercompress: cache-aware delay compress (hit=%s)",
                            self._cache_hit_label())
        return triggered

    def _cache_hit_label(self) -> str:
        """marker 行内的命中率标签（活体日志可读面）：ratio 在场 → `NN%`；None（无 usage /
        旧宿主无 cache 读数）→ `none`。纯格式化，零行为。"""
        ratio = self.last_cache_hit_ratio
        return ("none" if ratio is None
                else "%d%%" % int(round(100.0 * ratio)))

    def _effective_threshold(self) -> int:
        """生效触发线：context_length × DISTILL_WATER_FRAC（0.55 engine 水线；sc/distill.py 内部 80% cap 是
        evict-to 软水位、非本阈值），宿主下发 `threshold_tokens_cap`（session_compression.py:169）
        且更小则收紧到 cap；无 cap / cap 更大则保持 0.55 水线。纯配置规整，壳不重算算法。"""
        cap = self.threshold_tokens_cap
        return cap if (cap is not None and cap < self.threshold_tokens) else self.threshold_tokens

    def _cache_aware_trigger_threshold(self) -> int:
        """cache-aware 压缩触发线（决策面·压缩触发，owner 指令「1 要」）：

        只读上 turn 存好的 `last_cache_hit_ratio` 实例态（t_dc8e6b73 遥测面，本卡纯读——
        **不改** update_from_response / _record_cache_telemetry 存态逻辑）：
        - 决策介入（cache_aware_enabled 且 ratio 非 None 且 ≥ hit_min 且 base>0）→
          base×(1+delay_frac)（前缀越稳越推迟压缩、省 re-prime 成本；hit 高→推迟量同向）；
        - fail-safe（ratio None = 无 usage / 旧宿主无 cache 键、config 关、hit_min 未达、
          base≤0 哨兵路径）→ 恒等于现状 _effective_threshold()（行为零变化，红线）。
        阈值/权重全走实例配置属性（__init__ 从 compression.supercompress.cache_aware_compress.*
        逐键带默认规整），不硬编码进决策路径；-1 哨兵（base≤0）不进 cache-aware 路径。
        纯算术+日志，无 wall-clock / 无状态副作用；日志（含 `cache-aware delay compress
        (hit=NN%)` / `cache-aware keep trigger (hit=NN%)` 活体 marker）由 should_compress 的
        判定结果驱动（见 should_compress），本方法只负责返回生效触发线。"""
        base = self._effective_threshold()
        ratio = self.last_cache_hit_ratio
        if (self.cache_aware_enabled and ratio is not None
                and ratio >= self.cache_aware_hit_min and base > 0):
            return int(base * (1.0 + self.cache_aware_delay_frac))
        return base

    def _cache_aware_recall_k(self) -> int:
        """cache-aware 召回 K（决策面·召回，owner 指令「1 要」）：

        只读 `last_cache_hit_ratio`：高命中（enabled 且 ratio ≥ hit_min）→ 收窄到
        cache_aware_recall_k（优先 keep 已缓存段、少换 recall 内容 = 块面字节更稳）；
        fail-safe（config 关 / ratio None / 命中未达 hit_min）→ 恒 3（= 现状硬编码 K，零变化）。
        纯本地确定性（零 LLM / 零外呼）；select_context 宿主接口签名零改（cache 不作参数）。"""
        ratio = self.last_cache_hit_ratio
        if (self.cache_aware_enabled and ratio is not None
                and ratio >= self.cache_aware_hit_min):
            return max(1, self.cache_aware_recall_k)
        return 3

    # ── block-② recall cache 复用判定/存取（t_12f38672 · 实例态全在 engine 内部，不外溢宿主）──

    def _recall_cache_active(self, seg_ids: List[str]) -> bool:
        """段活性检查（fail-safe 复用红线）：上轮 K 段任一已不在当前 store
        （被 compact 淘汰/逐出、store rebind 后行消失）→ 返回 False → 不复用、
        回落全量重算（避免引用已淘汰段）。

        点查面 = store.get（segment_id → O(1)）；store rebind（on_session_start
        换 sid 文件）后新 store 天然查不到旧段 → 同判 False。异常 → False
        （fail-open 到全量重算，不 crash）。空段列表 → False（无段可复用）。"""
        if not seg_ids:
            return False
        try:
            return all(self.store.get(sid) is not None for sid in seg_ids)
        except Exception:
            return False

    def _recall_cache_try_reuse(self, query: str, p4_terms: List[str],
                                p4_verdicts: Dict[str, str]) -> Optional[List[str]]:
        """query 未实质变化判 + 段活性 → 复用上轮最终渲染文本；任一 fail-safe 不满足
        （冷启动 / 实例态结构损坏 / 指纹缺失 / Jaccard < 阈值 / 段被 compact 淘汰 /
        P4 verdicts 有 drop / 判定异常）→ None（全量重算，现状零变化）。

        复用 = 直接返回上轮**最终渲染文本列表**（P0 垫底序已在上轮写回时定死；
        全 keep 面下本轮回放与上轮逐字节相同 → provider 前缀不拆）。若 P4 verdicts
        把任一段判 drop，块的 P0 垫底序无法仅凭文本复原（drop 可能命中 P0 段）→
        回落全量重算（fail-safe，不强行拼序）。纯本地确定性，宿主零 diff。"""
        prev = self._recall_cache
        if not isinstance(prev, dict):
            return None  # 冷启动 / 未写回 → 全量重算（现状行为）
        try:
            fp = prev.get("fingerprint")
            if not isinstance(fp, frozenset):
                return None  # 指纹缺失/结构损坏（含全 None 损坏面）→ 全量重算
            cur = _recall_token_fingerprint(query + ((" " + " ".join(p4_terms)) if p4_terms else ""))
            if not _jaccard_ge(cur, fp, self.recall_cache_threshold):
                return None  # 实质变化 → 全量重算
            seg_ids = prev.get("seg_ids")
            texts = prev.get("recalled_texts")
            if not isinstance(seg_ids, list) or not seg_ids \
                    or not isinstance(texts, list) or not texts:
                return None  # 实例态结构异常 → 全量重算
            if len(seg_ids) != len(texts):
                return None  # 结构损坏（段/文本数不配）→ 全量重算
            if not self._recall_cache_active(seg_ids):
                return None  # 段被 compact 淘汰 → 回落全量重算（避免引用已淘汰段）
            # P4 verdicts 有 drop → P0 垫底序不可复原 → 回落全量重算（与全量重算
            # 路径同一过滤面；全 keep 面 = 默认 P4 无缓存 verdicts={} → 恒此通过）。
            if any(p4_verdicts.get(s, "keep") != "keep" for s in seg_ids):
                return None
            return list(texts)
        except Exception:
            return None  # 判定逻辑异常 → fail-open 全量重算（不 crash）

    def _recall_cache_store(self, query: str, p4_terms: List[str],
                            seg_ids: List[str], recalled_texts: List[str],
                            k: int) -> None:
        """全量重算后写回 recall 实例态（供下轮比对；实例态走 engine 内部）。

        存 = 上轮 query 指纹（sc.recall.tokenize 归一化 token 集）+ K 段 segment_id
        + 渲染后文本 + K 值。异常 fail-open（吞掉不炸宿主 turn；丢缓存机会无害——
        下轮只是再全量重算一次）。"""
        try:
            self._recall_cache = {
                "fingerprint": _recall_token_fingerprint(
                    query + ((" " + " ".join(p4_terms)) if p4_terms else "")),
                "seg_ids": list(seg_ids),
                "recalled_texts": list(recalled_texts),
                "k": int(k),
            }
            # P3-block-②: persist to the per-session sidecar so a later cold
            # start can rehydrate it. Fail-soft (no-op on any hiccup / pre-
            # rebind / cleared session).
            self._recall_cache_persist()
        except Exception:
            self._recall_cache = None

    def _recall_cache_sidecar_path(self) -> Optional[str]:
        """P3-block-② sidecar path: <store_dir>/<sid>.recall_cache.json, same
        dir + same sid prefix as the main <sid>.jsonl (naming aligned with
        the existing <sid>.digests.json sidecar). Returns None before the
        session rebind (legacy active.jsonl / pre on_session_start) so the
        cold-start hook never writes or reads there."""
        p = getattr(self.store, "path", None)
        if not p:
            return None
        d = os.path.dirname(p) or "."
        sid = os.path.splitext(os.path.basename(p))[0]
        return os.path.join(d, sid + ".recall_cache.json")

    def _recall_cache_persist(self) -> None:
        """P3-block-②: mirror the in-memory _recall_cache dict to the
        per-session sidecar so a later cold-start (process restart / session
        cold-reopen) can rehydrate it. Fail-soft: any hiccup -> no write,
        in-memory hot path unaffected. No-op when _recall_cache is None
        (fresh/cleared session) or the sidecar path is not yet bound
        (pre-rebind legacy store)."""
        prev = self._recall_cache
        if not isinstance(prev, dict):
            return
        path = self._recall_cache_sidecar_path()
        if not path:
            return
        try:
            payload = {
                "fingerprint": sorted(prev.get("fingerprint") or []),
                "seg_ids": list(prev.get("seg_ids") or []),
                "recalled_texts": list(prev.get("recalled_texts") or []),
                "k": int(prev.get("k") or 0),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass  # fail-soft: a persistence hiccup must never block recall

    def _recall_cache_rehydrate(self) -> None:
        """P3-block-②: on_session_start cold-reopen (existed branch) -> read
        the sidecar back into _recall_cache so the FIRST select_context after
        a process restart can hit the block-② reuse branch (when the query is
        ~unchanged AND the K segments are still alive in the re-rebound
        store). Missing/corrupt sidecar -> _recall_cache stays None
        (cold-start behavior, zero regression)."""
        path = self._recall_cache_sidecar_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            fp = payload.get("fingerprint")
            if not isinstance(fp, list):
                return
            self._recall_cache = {
                "fingerprint": frozenset(fp),
                "seg_ids": list(payload.get("seg_ids") or []),
                "recalled_texts": list(payload.get("recalled_texts") or []),
                "k": int(payload.get("k") or 0),
            }
        except Exception:
            self._recall_cache = None  # corrupt sidecar -> cold-start, fail-safe

    @staticmethod
    def _coerce_threshold_tokens_cap(value: Any) -> Optional[int]:
        """宿主实调位（tui_gateway/session_compression.py:169）：规整 threshold_tokens 配置。
        镜像内置 `ContextCompressor._coerce_threshold_tokens_cap`（= `_coerce_max_tokens`，
        context_compressor.py:2602-2611）：正 int 保留，None/非数/<=0 → None（不加 cap）。
        壳不发明新算法——缺这个方法正是宿主 "Could not apply live compression config" WARNING 根因。"""
        try:
            ivalue = int(value) if value is not None else 0
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """宿主 preflight（无真实 usage）：粗估 token（est_tokens 组件）过生效触发线即触发。
        cache-aware 决策面：触发线同 should_compress 经 _cache_aware_trigger_threshold() 规整
        （高命中推迟 / fail-safe 现状零变化）；preflight 纯读路不打 marker 日志（宿主每 turn
        双调 + preflight 粗估面高频调用，marker 归 should_compress 单一打点，同态去重口径）。"""
        rough = sum(est_tokens(str(m.get("content") or "")) for m in messages)
        return rough > self._cache_aware_trigger_threshold()

    def _check_rotation_advisory(self) -> None:
        """P0.4 rotation advisory 置位检查（turn 末调用；纯信号，不动 store 字节 / messages / window）。

        pressure = last_prompt_tokens / context_length（context_length 为 0 直接 return 防除零）：
        - pressure < ROTATION_RESET_RATIO → 静默重置（rotation_advisory=False; 计数归零，不 log 防刷屏）
        - pressure ≥ ROTATION_PROMPT_RATIO → 防抖计数 +1；连续 ADVISORY_DEBOUNCE_TURNS 且未置位
          → 置 True + logger.info（只 log False→True 翻转，文案含 pressure / 连续 turn 数）
        - 中间带 [RESET, PROMPT) → 不清已置位，仅重置防抖计数
        """
        if self.context_length <= 0:
            return
        pressure = self.last_prompt_tokens / self.context_length
        if pressure < ROTATION_RESET_RATIO:
            self.rotation_advisory = False
            self._adv_consec = 0
        elif pressure >= ROTATION_PROMPT_RATIO:
            self._adv_consec += 1
            if self._adv_consec >= ADVISORY_DEBOUNCE_TURNS and not self.rotation_advisory:
                self.rotation_advisory = True
                logger.info(
                    "supercompress: rotation advisory set (pressure=%.2f, "
                    "consecutive_turns=%d)", pressure, self._adv_consec)
        else:
            # 中间带：不清已置位，只重置防抖计数
            self._adv_consec = 0

    def _p4_cooldown_ladder(self) -> tuple:
        """P4 边界 LLM 2 档冷却阶梯（aux 两级路由 cooldown 模式复用：基线 × 乘数阶梯，封顶语义）。

        默认 boundary_fail_cooldown=30s → (30s, 180s)——边界 LLM 低频，短于 digest 60→300→1800s；
        =0 → 阶梯全 0 → 冷却窗恒关（fail-safe「单事件自愈」语义，与 sc 契约口径一致）。
        纯配置规整（镜像 aux_provider 阶梯派生），壳不发明算法。"""
        base = self._p4_fail_cooldown
        return tuple(base * m for m in (1.0, 6.0))

    def _fire_boundary_expand(self, text: str) -> None:
        """P4：drift 边界那一下花一次宿主 LLM 干三件（扩词 + top-N 精排 + 重建 epoch 锚点）。

        - 只在 `observe_user` 返回 True（drift 边界）触发——常驻路径 0 LLM（G4 确定性不破）；
        - per-epoch 缓存命中（同 epoch 不重算）→ 直接返回（0 LLM 调用）；
        - 种子词 = 当前 epoch 锚点词面（drift 触发 K 条重建后的「当前任务关键词集」）；
        - 候选 = extended-BM25 粗排 top-3（Recaller.search，P1 打分核零动）；
        - 成功 → 扩展词面喂回 P1 extended-BM25（select_context query 面）+ 精排 keep/drop
          （verdicts 破近义误召）+ 并入 epoch 锚点（merge_anchor，与 P0 合流拆 3.4% 填充墙）；
        - 失败/超时 → 纯 BM25 回退（degraded 保守面 + 冷却阶梯，sc/semantic 内闭环）；
        - 壳 fail-open 契约：任何异常不炸宿主 turn（宽 catch + 留痕 log，同 on_turn_complete 既有口径）。
        """
        try:
            epoch = self._drift.current_epoch
            if epoch in self._p4_cache:               # per-epoch 缓存命中 → 0 LLM（花最少）
                return
            seed = self._drift.anchor_terms()          # 当前任务关键词集（锚点词面，sorted 确定性）
            window = self._recall_union_window(self._window_fresh())
            recaller = Recaller(self.store, window)
            candidates = recaller.search(text, k=3)    # extended-BM25 粗排 top-3（扩词后喂回同面）
            res = self._p4.expand(epoch, seed, candidates, task_text=text)
            if not res.degraded:
                # ③ 重建 epoch 锚点：扩展词面并入 drift 锚点（内存面；session_topic 整数 epoch 标 /
                # jsonl 字节零动——与 P0 合流，epoch 锚点 = 扩展后任务关键词集）
                merged = self._drift.merge_anchor(res.expanded_terms)
                logger.info(
                    "supercompress: P4 boundary expand epoch=%d expanded=%d "
                    "anchor_merged=%d verdicts=%d cache=%d",
                    epoch, len(res.expanded_terms), merged, len(res.verdicts), len(self._p4_cache))
            else:
                logger.info("supercompress: P4 boundary expand epoch=%d degraded -> pure-BM25 "
                            "fallback (cooldown=%s)", epoch, self._p4.cooldown_status())
        except Exception as e:                          # 壳 fail-open：P4 面异常不炸宿主 turn
            logger.warning("supercompress: P4 boundary expand no-op (pure-BM25 fallback): %s", e)

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None,
                         **kwargs: Any) -> None:
        """turn 末接线：新 user 消息落盘（P0 即写即 fsync）+ aux turn 边界 + Distiller.run 逐出。

        全委托 sc：tagger.classify → store.append（tier-aware fsync）→ aux.end_turn → Distiller.run
        （sc.distill 内部 80% evict-to 软水位逐出 + push cap + 锚点发射；engine 侧触发线是 0.55 水线，见 :84-85）。壳内无任何逐出/预算算法体。
        """
        try:
            self._aux.start_turn()
            window = self._window_fresh()
            fresh = [m for m in (messages or [])
                     if m.get("role") == "user"
                     and nfc_hash(str(m.get("content") or "")) not in self.store.by_hash]
            # Ingress 拒存：群聊房间骨架（[Group chat 开首的 bot-mode 转写块）是
            # 瞬态会话结构，不是 user 指令/事实——存成 P0 user 段会被后续任意
            # 会话的召回注入（跨会话污染事故源）。与 egress 过滤共用 is_room_skeleton
            # 单源谓词（可观测性行：拒绝类必须留痕，不做静默 no-op drop）。
            _skel = [m for m in fresh if is_room_skeleton(str(m.get("content") or ""))]
            if _skel:
                logger.info("supercompress: ingress rejected %d room-skeleton user msg(s) "
                            "[Group chat skeleton is transient; never durable", len(_skel))
            for m in fresh:
                if is_room_skeleton(str(m.get("content") or "")):
                    continue
                seg = _make_segment(m, self._tagger, ts=time.time())
                # P2.1/P2.2：user 段 epoch 打标 + 漂移检测（零 LLM 确定性，复用 P1 tokenize）
                text = str(m.get("content") or "")
                triggered = self._drift.observe_user(text)
                seg.session_topic = str(self._drift.current_epoch)
                self.store.append(seg)
                # P4：drift 边界那一下花一次宿主 LLM（扩词+精排+重建锚点；常驻 0 LLM，G4）
                if triggered:
                    self._fire_boundary_expand(text)
            self._aux.end_turn()
            # B 卡：model/assistant 段选择性入库 gate（确定性零 LLM；user 段主 store 路径零行为变化）
            self._model_ingest_hook(messages)
            # R3：force_ratio 接 engine 同源配置 force_compact_ratio（kwarg = 接线非算法）
            result = Distiller(window, self.store, self._aux, force_evict=self._force_evict_mode,
                               force_ratio=self.force_compact_ratio).run()
            if result.status not in ("ok", "backpressure"):
                logger.warning("supercompress: distill status=%s events=%s",
                               result.status, result.events)
            # P4 钩子②（digest 完成回写池，t_026a17f8）：本事件蒸馏可能新完成 digest
            # （sync 即时 / async 经 end_turn 入队）→ 池落盘；幂等零 IO 快退，fail-open 不炸宿主。
            self._digest_pool_write_hook()
            # P0.4：turn 末 pressure 检查（防抖口径）+ 会话边界外不触碰任何持久态
            self._check_rotation_advisory()
            # per-turn cache 遥测（item 3）：宿主 turn 末喂入 usage=agent._last_turn_usage（含 cache 桶）
            # → 存上 turn 的 cache 命中率实例态，本 turn select_context()/compress() 纯读之（不改宿主
            # 接口签名——决策面另起卡）。log_hit=False：主路径 update_from_response 已打命中行，此处
            # 仅幂等存态、不重复打日志；usage 缺 cache 键/None → 容错 0（旧宿主亦安全）。
            self._record_cache_telemetry(usage, log_hit=False)
        except StoreError as e:
            logger.error("supercompress: store error (fail-safe, window untouched): %s", e)
        except Exception as e:  # 壳永不把宿主 turn 炸掉（fail-open，宿主契约）
            logger.error("supercompress: on_turn_complete degraded: %s", e)

    def select_context(
        self, request_messages: List[Dict[str, Any]],
        *, conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None,
        budget_tokens: int = 0,
        **kwargs: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """sc.recall 接线：Recaller.search（pull budget 内 P0/P1 hit）；无 hit → 返回 None（字节不变）。

        宿主契约：替换只在选中请求内，持久化 transcript 不动；malformed 永不 reach provider。

        A1 止血（2026-10-10）：auto_recall 默认关（config compression.supercompress.auto_recall，
        缺省 False）→ 本方法整条面直接 return None（宿主 fail-open 请求字节不变），
        杜绝每 turn 自动把 model P0（agent 自述）/recall 块投到末条 user 造成的
        "回放当用户拍板"归属错乱；仅 auto_recall 显式 True 时走下方全量召回路径。
        compress（compact 事件驱动注入面）不受本开关影响。
        """
        if not self.auto_recall_enabled:
            return None
        try:
            incoming = incoming_message
            if not incoming and conversation_messages:
                for m in reversed(conversation_messages):
                    if m.get("role") == "user":
                        incoming = m
                        break
            if not incoming:
                return None
            query = str(incoming.get("content") or "")
            if len(query.strip()) < 8:
                return None
            window = self._recall_union_window(self._window_fresh())
            # P4：边界扩展词面喂回 extended-BM25（词面扩词，不是向量；命中缓存 = 该 epoch
            # 边界 LLM 成功面 → 扩词并入 query 词面；无缓存/回退 → 原 query，常驻 100% 确定性 G4）
            p4_terms = self._p4.expanded_terms(self._drift.current_epoch)
            p4_query = query + ((" " + " ".join(p4_terms)) if p4_terms else "")
            # P4：top-N 精排 keep/drop 面（破近义误召；fail-safe keep-all——全 drop 回落原
            # 集合，不强制空召回；verdicts 缺失/回退 = {} → 零行为）。读取面提前到
            # block-② 复用判定之前（复用分支需要同一 verdicts 过滤复用文本；同 epoch 纯读
            # 缓存、零 LLM 调用，现状读序/行为零变化）。
            p4_verdicts = self._p4.verdicts(self._drift.current_epoch)
            # block-② recall cache 复用分支（t_12f38672）：query 未实质变化（归一化
            # token 集与上轮指纹 Jaccard ≥ recall_cache_threshold，默认 0.85）且上轮 K 段
            # 在当前 store 仍全部存活 → 直接复用上轮 K 段文本（不重打分/不重排，块字节
            # 与上轮逐字节相同 → provider 前缀不拆）。下游仍走同一 P4 verdicts 过滤 +
            # P0 垫底 + S1 尾投递面（复用分支只换文本来源，渲染本体零动）。
            # fail-safe 红线（任一不满足 → 回落下方全量重算路径，现状零变化）：冷启动 /
            # 实例态结构损坏 / 实质变化（Jaccard < 阈值）/ 段被 compact 淘汰 / 判定异常。
            _reused = self._recall_cache_try_reuse(query, p4_terms, p4_verdicts)
            if _reused is not None:
                recalled = _reused
                out = _s1_tail_deliver(
                    request_messages, "[supercompress recall]\n" + "\n\n".join(recalled))
                logger.info("supercompress: recall cache-hit (reused prev, K=%d)",
                            len(recalled))
                return out
            recaller = Recaller(self.store, window, decay_lambda=self.recall_decay_lambda)
            # 招③ 时间衰减接线面：decay_lambda = 实例默认（engine.recall_decay_lambda，
            # kwarg/配置 recall_decay_lambda 控制启用；默认 0 = 钉死面不扰动）
            # cache-aware 召回 K（owner 指令「1 要」·决策面·召回）：高命中（上 turn 前缀
            # cache 命中率 ≥ hit_min）→ 收窄 top-K 到 cache_aware_recall_k（优先 keep 已缓存段、
            # 少换 recall 内容 = 块面字节更稳）；fail-safe（config 关 / ratio None / 命中未达）
            # → 恒 3（= 现状硬编码 K，零变化）。纯读 last_cache_hit_ratio 实例态，宿主签名零改。
            hits = recaller.search(p4_query, k=self._cache_aware_recall_k())
            # P2.3 soft：旧 epoch 非结构 P1/P2 召回降权（P1 search 打分核零动，独立 rerank 步）
            hits = self._drift.soft_rank(hits)
            # P4：top-N 精排 keep/drop（破近义误召；fail-safe keep-all——全 drop 回落原集合，
            # 不强制空召回；verdicts 缺失/回退 = {} → 零行为）
            p4_verdicts = self._p4.verdicts(self._drift.current_epoch)
            if p4_verdicts:
                _kept = [h for h in hits if p4_verdicts.get(h.seg.segment_id, "keep") == "keep"]
                hits = _kept if _kept else hits
            # P3（S-Phase 4）gate LLM rerank（独立 step，P1 打分核零动）：
            # gate 三选一（召回空 / 显式搜索词 / top-1 < 低分阈值，默认保守）+ 缓存键
            # (sha256(query), store.line_count) + fail-safe（LLM 不可用/解析失败 → 原样返回）。
            # 端点 = aux 主模型端点（宿主主模型共享，零新 provider / 零模型名 / 零 provider 块）。
            # _NoEndpoint（未注入 call_llm）下 gate 触发也 fail-safe no-op（纯本地不炸）。
            # P4：rerank gate 面用 p4_query（= 实际召回 query，含边界扩词）——gate 判定 / 提示词 /
            # 缓存键与召回面同源；无 P4 缓存时 p4_query == query（P3 钉死面 0 扰动）。
            hits = rerank_hits(
                hits, p4_query, self._aux.main_endpoint,
                cache=self._rerank_cache, window_version=self.store.line_count,
                gate=self._rerank_gate, drop_model_on_gate=self.drop_model_on_gate)
            # S1 P0 垫底（PRD §0.9）：块内排序 = 背景 P1/P2/Pn 在前、被召回 P0 段垫底
            # （全请求最后字节 = recency 权重最强位；P0 范围 = 被召回段，非全量 structural
            # P0；A 块整块每 turn 变，P0 区内存位置不影响前缀匹配点，cache 代价中性）。
            _recalled_pairs = [(h.seg.tier, h.seg.original_text, h.seg.segment_id,
                                getattr(h.seg, "provenance", ""))
                               for h in hits
                               if h.seg.tier in ("P0", "P1") and h.score > 0.0]
            # Egress 过滤（存量止血）：磁盘上已存的 203 条 P0 group-skeleton 段
            # （pre-fix 入向）会按词重叠被召回注入到任意后续会话 = 跨会话污染。
            # 与 ingress 共用 is_room_skeleton 单源谓词（不删磁盘数据、可回滚；
            # 仅把房间骨架逐出自动注入面，观测行留痕）。
            _skel_out = [(t, x) for (t, x, _sid, _prov) in _recalled_pairs if is_room_skeleton(x)]
            if _skel_out:
                logger.info("supercompress: recall egress cut %d room-skeleton segment(s) "
                            "(legacy P0 group-chat residue)", len(_skel_out))
            _recalled_pairs = [(t, x, _sid, _prov) for (t, x, _sid, _prov) in _recalled_pairs
                               if not is_room_skeleton(x)]
            # R2-v2 egress（2026-10-10）：agent 自述段（"把前几轮复述当已确认回放"形态）
            # 逐出自动注入面——v2 判定式 = provenance=="model" 硬门 ∩ 自指结构/中英 marker
            # （user 段天然免疫误删；多语言自指结构词跨语言通用，不逐语言堆 marker）。
            # 与 is_room_skeleton 同模式（不删盘、可回滚、观测行留痕）。宁漏勿错。
            _selfrep_out = [(t, x) for (t, x, _sid, _prov) in _recalled_pairs
                            if is_agent_self_report(x, _prov)]
            if _selfrep_out:
                logger.info("supercompress: recall egress cut %d agent-self-report "
                            "segment(s) (P0 model provenance replay residue)",
                            len(_selfrep_out))
            _recalled_pairs = [(t, x, _sid, _prov) for (t, x, _sid, _prov) in _recalled_pairs
                               if not is_agent_self_report(x, _prov)]
            if not _recalled_pairs:
                return None
            _bg = [(x, sid) for (t, x, sid, _prov) in _recalled_pairs if t != "P0"]
            _p0 = [(x, sid) for (t, x, sid, _prov) in _recalled_pairs if t == "P0"]
            recalled = [x for (x, _sid) in _bg + _p0]
            # S1 尾投递（PRD §0.4）：recall 块不再注位置 2 的第二条 in-list system（cache
            # 断点 + Anthropic 覆盖赋值 bug 双免疫）——投到列尾：末条 user 并入 / 非 user
            # 追加独立 user。main system 保持原位（位置 0），recall 移到末条 user。
            out = _s1_tail_deliver(
                request_messages, "[supercompress recall]\n" + "\n\n".join(recalled))
            logger.info("supercompress: select_context recalled %d segment(s), breach=%d",
                        len(recalled), len(recaller.breach_log))
            # block-② recall cache 写回（t_12f38672）：全量重算结果写回实例态（P0 垫底序
            # 定死在 recalled 文本 + 对齐 seg_ids），供下轮比对。空召回（未走到此）/异常
            # → 不写（下轮冷启动再重算，fail-safe 零变化）。
            self._recall_cache_store(
                query, p4_terms,
                [sid for (_x, sid) in _bg + _p0],
                recalled,
                len(recalled))
            return out
        except Exception as e:
            logger.warning("supercompress: select_context no-op (recall degraded): %s", e)
            return None

    def compress(
        self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None, force: bool = False, memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """sc.distill 接线 + 宿主消息面输出：蒸馏后 = system head + P0/P1 常驻段 + 尾部。
        """
        self.compression_count += 1
        tokens = current_tokens or self.last_prompt_tokens
        if not force and tokens < self.threshold_tokens * self.force_compact_ratio:
            logger.info("supercompress: compress (light trim) tokens=%d threshold=%d",
                        tokens, self.threshold_tokens)
        else:
            logger.info("supercompress: compress (distill) tokens=%d force=%s", tokens, force)
        try:
            window = self._window_fresh()
            # 检查是否需要启用 force_evict_mode
            if not self._force_evict_mode and window.used_tokens > self.context_length:
                overflow = window.used_tokens - self.context_length
                p0_tokens = sum(ws.tokens() for ws in window.segs if ws.segment.tier == "P0")
                push_cap = int(self.context_length * 0.15)
                # 如果 overflow 超过 push cap，说明 P0 常驻集过大，需要强制逐出
                if p0_tokens > push_cap and overflow > push_cap * 0.5:
                    logger.warning("supercompress: overflow %d tokens > 50%% push_cap %d, "
                                   "enabling force_evict_mode", overflow, push_cap)
                    self._force_evict_mode = True
            # 执行 distill（R3：force_ratio 接同源配置 force_compact_ratio）
            result = Distiller(window, self.store, self._aux, force_evict=self._force_evict_mode,
                               force_ratio=self.force_compact_ratio).run()
            # 如果 normal distill 失败（transient_skip）且未启用 force_evict，自动降级
            if result.status == "backpressure" and not self._force_evict_mode:
                logger.warning("supercompress: distill backpressure, trying force_evict")
                self._force_evict_mode = True
                result = Distiller(window, self.store, self._aux, force_evict=True,
                                   force_ratio=self.force_compact_ratio).run()
                logger.info("supercompress: force_evict result status=%s events=%s",
                            result.status, result.events)
            if result.status not in ("ok", "backpressure"):
                logger.warning("supercompress: distill status=%s events=%s",
                               result.status, result.events)
            # P4 钩子②（digest 完成回写池，t_026a17f8）：本事件蒸馏结算的 digest（sync 即时；
            # async 在途部分由会话末 aux.flush 汇合点兜底）→ 池落盘；幂等零 IO 快退，
            # fail-open（钩子自含宽 catch，不炸宿主 compress 返回面）。
            self._digest_pool_write_hook()
        except (StoreError, OSError) as e:
            # R2：full-disk 时 store 交互抛 OSError（EIO/ENOSPC）——store 写路径已归一为
            # StoreError("io_error: ...")，此处双 catch 防御 replay/aux 面裸 OSError 逃逸。
            # 优雅降级：log warning + 磁盘水位观测（剩余空间 + 触发动作，ops 监控），
            # 本轮跳过持久化，不炸宿主（压缩失败 ≠ turn 失败；与 on_turn_complete 宽
            # catch 口径对齐——fail-open 返回原 messages，宿主本 turn 原样继续）
            logger.warning("supercompress: compress degraded (store error, "
                           "skipping this round's persistence, host untouched): %s", e)
            try:
                self.store.log_water_warn("compress-failed", e)
            except Exception:
                pass
            return messages  # fail-open：压缩失败 ≠ turn 失败
        # S-Phase 2（决定 B）：阶梯注入取代全文拼接——预算驱动选档（L3/L2/L1/L0），
        # 固定池无 pull 耦合：索引池 2K（L1+L2 共池）+ 细节池 4K（L3），总帽 6K，
        # 与模型窗口解耦（注入块不随窗口放大 = 决定 B 核心）；高压（used>cap）细节池
        # 归零只出 L1/L2/L0；structural P0 豁免（活账本永不裁）。壳只调 sc.inject，
        # 零算法体（壳薄度门禁不变）。
        kept = [ws for ws in window.segs if ws.segment.tier in ("P0", "P1")]
        # P2.3 hard：旧 epoch 非结构 P1/P2 → form=digest_only（可逆：jsonl 字节不动，
        # epoch 回切即恢复；structural P0 / goal / ledger 豁免）。纯内存翻转，确定性，零 LLM。
        _hard = {s.segment_id for s in self._drift.hard_evict([ws.segment for ws in kept])}
        for _ws in kept:
            if _ws.segment.segment_id in _hard:
                _ws.form = "digest_only"
        # Egress 过滤（同 select_context）：已存 group-skeleton P0 段逐出压缩
        # anchor-manifest 注入面——否则压缩时群聊骨架仍驻留进 durable 段块。
        _skel = [ws for ws in kept if is_room_skeleton(ws.segment.original_text)]
        if _skel:
            logger.info("supercompress: compress egress cut %d room-skeleton segment(s) "
                        "from anchor manifest", len(_skel))
        kept = [ws for ws in kept if not is_room_skeleton(ws.segment.original_text)]
        # R2-v2 egress（2026-10-10）：agent 自述段（provenance=model 的认账/复述/回放 text）
        # 逐出 compact 注入面——v2 判定式 = provenance=="model" 硬门 ∩ 自指结构/中英 marker
        # （user 段免疫误删；多语言自指结构词跨语言通用）。同 recall egress 模式
        # （不删盘、可回滚、观测行留痕）。宁漏勿错：漏 = 原样进 staged 块不误删。
        _selfrep = [ws for ws in kept if is_agent_self_report(
            ws.segment.original_text, getattr(ws.segment, "provenance", ""))]
        if _selfrep:
            logger.info("supercompress: compress egress cut %d agent-self-report "
                        "segment(s) from staged manifest (P0 model provenance)",
                        len(_selfrep))
        kept = [ws for ws in kept if not is_agent_self_report(
            ws.segment.original_text, getattr(ws.segment, "provenance", ""))]
        # B 卡 T4 注入（决策 6）：model P0/P1 段进 kept 集（主 store kept 面同形扩展），
        # 与主 store 段共用固定池 2K/4K + 既有确定性选档 + seq 稳定序（池预算不变；
        # 高压判定仍读主 window，细节池归零语义不变）；model 段不喂 DriftTracker
        # （user-only by design，决策 7，不受 P2.3 hard 逐出）；egress 房间骨架 +
        # R2 agent 自述过滤同语义生效。
        if self.model_store_enabled and not self._model_fused:
            _mstore = self._model_store
            if _mstore is not None:
                for _mseg in _mstore.live_segments():
                    # model store 段 provenance 恒 = "model"（model store by design 只存 agent 段）
                    if (_mseg.tier in ("P0", "P1")
                            and not is_room_skeleton(_mseg.original_text)
                            and not is_agent_self_report(
                                _mseg.original_text, getattr(_mseg, "provenance", "model"))):
                        kept.append(WindowSeg(_mseg))
        # P3 30% 零成本剔除档已于 2026-10-04 撤销（用户拍板 B）：活体 P1=0 → fill_share 恒 0.0，
        # 该档在活体上永远 no-op（纯 bench 保护逻辑）。注入路径回到 fill_prune 引入前逐字行为。
        # filler_form_share 仅保留给 telemetry record（撤档正确性的佐证观测面，纯审计、G4）。
        _fs = {}
        try:
            _fs = filler_form_share(window, query=focus_topic or "", segs=kept)
        except Exception:
            _fs = {}
        rendered = choose_forms(kept, window, query=focus_topic or "")
        block = render_block(rendered)
        _forms: Dict[str, int] = {}
        for _ws, _ff, _tx in rendered:
            _forms[_ff] = _forms.get(_ff, 0) + 1
        # P2 telemetry（指标① fill_share）：注入池填充占比观测面（镜像 choose_forms 同源
        # 排序/池记账；append-only 零控制流改动 G4；fail-open 永不炸宿主 turn）。
        if _fs:
            try:
                self.telemetry.record_compress(fill_share=_fs["fill_share"],
                                               pool_tokens=_fs["pool_tokens"],
                                               forms=_forms)
            except Exception:
                pass
        out: List[Dict[str, Any]] = list(messages[:self.protect_first_n])
        out.extend(messages[-self.protect_last_n:] if messages else [])
        # 紧凑摘要账本（t_63bf3ba2 · append-only 累计布局）：长会话（head+tail 之间仍有
        # 可压缩 middle）→ 本次 staged 块 **append** 进累计段（旧块 c#1.. 字节原样保留）
        # 并注入 head/tail 缝作独立 user 消息 = 前缀单调稳定位 → provider 前缀缓存跨
        # compact #1/#2/… 命中；唯一刷缓存位 = 滚动 tail。staged 块已进账本 → 不再 S1
        # 尾投递（避免同内容双发 + 尾位字节随块变）。短会话（无 middle）→ 零账本消息、
        # staged 照走既有 S1 尾投递（既有 test_staged_block_s1_tail 行为零变化）。
        # 幂等：同窗口重算同 staged 块 → should_append False → 零写盘（前缀缓存不抖动）。
        _has_middle = bool(messages) and len(messages) > (self.protect_first_n + self.protect_last_n)
        if block:
            # R5（owner 批次 A #6 · 记账面小占位符展开）：staged 块进账本前，
            # 逐段检测 [code: ×N lines] 小占位符（N < MODEL_LEDGER_FENCE_EXPAND_LINES，
            # 默认 20）→ 用该段 original_text（围栏原文）替换占位符进账本——小代码块
            # 全文比残句占位符信息量大（残句混入账本 = 诊断账本 #6 根因）；大占位符
            # （N≥20）维持残句（全文太贵）。T1 折叠契约（入库字节）零扰动——只在记账
            # 渲染面展开，段 original_text 已含展开所需的原文（L3 面 excerpt 同源）。
            # fail-open：检测异常 → 原样占位符进账本（零新风险）。
            _expanded = _expand_small_code_placeholders(rendered)
            _staged = (f"[supercompress] staged P0/P1 segments "
                       f"({len(_expanded)} segs, {block_tokens(_expanded)} tok, "
                       f"forms: {dict(sorted(_forms.items()))}):\n"
                       + render_block(_expanded))
            if _has_middle:
                # 长会话（真实压缩场景，head+tail 之间仍有可压缩 middle）：本次 staged 块
                # **append** 进累计账本（旧块 c#1.. 字节原样保留）→ 累计段渲染成一条**稳定位
                # user 消息**注入 head/tail 缝（唯一刷缓存位 = 滚动 tail，本消息字节在 c#N
                # 追加前不动）→ 前缀单调 → provider 前缀缓存跨 compact #1/#2/… 命中。
                # 块 c#N 沿用 staged 头文案 `[supercompress] staged …`（与 S1 尾投递同源标记）
                # → 既有 e2e 断言（恰一条 user 携带 staged 标记）在长会话门控面同样成立；
                # 短会话（无 middle）零账本消息、staged 照走既有 S1 尾投递（零行为变化）。
                #
                # 宿主 re-feed 去重（活体 compact #1→#2 面）：compact #1 已发行一条累计账本
                # 消息，宿主把它当作 live transcript 下一轮 re-feed 回 compress()。若旧账本消息
                # 恰好落在 head(3)/tail(6) 保护窗内，会与本次新累计段（c#1..c#N 全量，是旧段的
                # 严格超集）并存 → c#1 重复一份。此处薄剥除 stale 账本消息（只认**专属前缀**
                # `[supercompress] staged compact-ledger`，绝不误伤 S1 尾投递的
                # `[supercompress] staged P0/P1 segments` 消息）→ 输出恒 = 单条累计账本消息，
                # 前缀单调契约不被陈旧旧段破坏（append-only 只增不改，旧段已被超集覆盖）。
                out = [m for m in out if not str(m.get("content", "")).startswith(_COMPACT_LEDGER_PREFIX)]
                ledger = self._compact_ledger()
                # tier 分层（owner 裁定）：staged 块按 tier 拆 P0 定案块 + P1 推理可裁层，
                # 分别 append 进账本——定案账本不再被 P1 推理段撑爆，P1 可裁层封顶优先裁。
                #
                # P0 定案块（trimmable=False，永不裁）：三态判据 `_staged_p0_decision`：
                #   - "adhoc"  纯自动：强完成信号命中 → 本块 settled=True + kind=adhoc
                #     （下轮折单行存根，原文**同事务**沉盘 <sid>.p0，可复活可审计）；
                #   - "kanban" 半自动：弱候选信号命中 → 本块 settled=False + kind=kanban
                #     （candidate 提名，宿主 LLM 裁判 + clarify 确认后调 settle_p0 折出）；
                #   - "none"   不定案：双信号都未命中 → settled=False + kind=adhoc
                #     （未定案段照常全量回喂，治「重复问」只对真定案段生效）。
                _p0_block, _p0_forms = _staged_by_tier(_expanded, "P0")
                _p1_block, _p1_forms = _staged_by_tier(_expanded, "P1")
                _entry = None
                if _p0_block:
                    _decision = _staged_p0_decision(_p0_block, _p0_forms, rendered)
                    _settled_now = (_decision == "adhoc")
                    _kind = "kanban" if _decision == "kanban" else "adhoc"
                    _entry = ledger.append(
                        _p0_block, forms=_p0_forms or None,
                        settled=_settled_now, kind=_kind, trimmable=False)
                    if _settled_now and _entry is not None:
                        # 纯自动 settled → 原文同事务沉盘（幂等覆盖；写失败 fail-open 只告警）
                        _sink_dir = os.path.dirname(os.path.abspath(ledger.path))
                        _sid = os.path.splitext(os.path.basename(ledger.path))[0].replace(".compact", "")
                        if not scdata.sink_p0(_sink_dir, _entry["seq"], _entry["text"], kind=_kind, sid=_sid):
                            logger.warning("supercompress: compact-ledger c#%d settled but "
                                           "p0 sink failed (in-memory only)", _entry["seq"])
                    if _decision == "kanban" and _entry is not None:
                        logger.info("supercompress: compact-ledger c#%d kanban candidate "
                                    "(settled=False, awaiting LLM judge + clarify -> settle_p0)",
                                    _entry["seq"])
                # P1 推理可裁层（trimmable=True，封顶优先裁；原文 recall 层 segment_id 可捞回）：
                # 不折存根（还没定案，折了模型看不到推理过程），封顶时被 _trim_trimmable_oldest
                # 优先清掉 → 定案账本真正收缩到只剩 P0 存根 + 未定案 P0 全文。
                if _p1_block:
                    _p1_entry = ledger.append(_p1_block, forms=_p1_forms or None,
                                              settled=False, kind="adhoc", trimmable=True)
                    if _p1_entry is not None:
                        logger.info("supercompress: compact-ledger c#%d P1 trimmable layer "
                                    "(trimmable=True, prioritized at cap; original recallable)",
                                    _p1_entry["seq"])
                section = ledger.text()
                if _entry is not None:
                    logger.info("supercompress: compact-ledger appended P0 c#%d "
                                "(generation=%d seals=%d entries=%d section_tokens=%d "
                                "settled=%s kind=%s) tier-segregated, prefix stable across compacts",
                                _entry["seq"], ledger.generation, ledger.seals,
                                len(ledger.entries), est_tokens(section),
                                (_entry["settled"]), _entry["kind"])
                if section:
                    # 账本消息固定 head 侧高位——插入位 protect_first_n
                    # = head 保护段**末尾**（`out` 此时 = head(protect_first_n) + tail
                    # (protect_last_n)，缝位即尾段开头）→ 账本消息永不沉进滚动 tail
                    # 区、不受 max_tail_tokens 上限裁。
                    # L1 区裁定（owner）：撤 A1 前置重排 → section 按 c# 自然 append 序
                    # 原样渲染（ledger.text() 纯函数），P0 定案段「不被重读」改由
                    # settled 折出存根面承担（上方写侧接线），前缀才真单调稳定。
                    # 前缀单调契约（append-only 只增不回改，旧块 c#1.. 字节原样是
                    # section 严格前缀）→ provider 前缀缓存跨 compact #1/#2/… 命中；
                    # 唯一刷缓存位 = 滚动 tail（本消息恒在缝位）。
                    out.insert(self.protect_first_n,
                               {"role": "user",
                                # R5（owner 批次 A #4 头去重）：消息头不再双渲染
                                # generation/seals——单源 = section 首行 MARK
                                # （ledger.text() 的 `[supercompress compact-ledger]
                                # generation=… seals=…`），壳只做前缀标记 + 拼接，
                                # 消「第 1/2 行重复」，壳薄度原则。
                                "content": _COMPACT_LEDGER_PREFIX + "\n" + section})
            else:
                # 短会话：无 middle → 不发行账本前缀消息，staged 照走既有 S1 尾投递
                out = _s1_tail_deliver(out, _staged)
        return out

    # ── 会话边界 ──

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        """S-Phase 1：per-session 独立 store + carry_over 交接（P0.4 单轨收口）。

        宿主 kwargs（agent_init.py:2133）：hermes_home / platform / model / context_length/
        conversation_id；轮转事件额外带 old_session_id（conversation_compression.py:1703/2556）。
        - store 目录：hermes_home 存在 → <hermes_home>/supercompress_store（profile 隔离白捡）；
          缺省（独立测试/bench）→ legacy 目录
        - session 文件 = <dir>/<sid>.jsonl；同实例同 sid 重入 → 幂等 no-op
        - old_session_id + carry_over_context=True（宿主确认轮转信号，run_agent.py:402）
          → materialize(old→new, reopen_from=old)（激活 store.py:137 死代码，P1 自动 stale 标记）；
          无 carry_over_context 的自动路径 → 默认 OFF（P0.4 单轨收口：宿主信号 `carry_over_context` 唯一源，插件自造键已删）
        - 同 sid 冷重开（新实例、文件已存在）→ replayed P1 中间推理内存标 stale（不动字节）
        """
        home = kwargs.get("hermes_home")
        store_dir = (os.path.join(os.path.expanduser(home), STORE_SUBDIR)
                     if home else self._legacy_store_dir)
        target = os.path.join(store_dir, f"{_safe_session_name(session_id)}.jsonl")
        # P5 GC 面（owner 既裁定"保留 30 天"）：宿主 on_session_start 每开 session 顺手扫
        # 一次 >30 天旧 session 的侧车族全删（scdata.gc_stale_sidecars 纯算法面，插件内
        # stdlib，零宿主改动；天然分散——每 session 开清一次不集中爆扫）。30 天内新鲜
        # session 绝不碰；fail-open（目录缺失/单文件删失败 → 静默跳过，不炸宿主启动面）。
        # 扫 store_dir（与上面同一变量）而非 hermes_home 派生——legacy 独立测试路径下
        # store_dir = _legacy_store_dir，与 GC 面必须同住同目录才不误删别的 profile。
        try:
            _removed = scdata.gc_stale_sidecars_in_dir(store_dir)
            if _removed:
                logger.info("supercompress: gc_stale_sidecars removed %d file(s) "
                            "(>30d retention sweep)", len(_removed))
        except Exception as _e:
            logger.warning("supercompress: gc_stale_sidecars failed open (retention "
                           "sweep skipped, host startup unaffected): %s", _e)
        if self._bound_session == session_id and self.store.path == target:
            return  # 幂等重入：同实例同 session 已绑定。
            # 注：早退**不**重置 last_prompt_tokens / _force_evict_mode —— 它们是「本进程当前
            # 会话实态」（update_from_response 每次 LLM 调用后重写），重入 = 活会话继续，
            # 重置反而把活态打回 0。与 on_session_reset（会话边界，必须清）语义不同。
        old_sid = kwargs.get("old_session_id")
        # P0.4 单轨收口：宿主信号 `carry_over_context`（run_agent.py:402 下发）为唯一源，
        # 插件自造键 `carry_over` 已删——宿主面唯一事实源，零双轨歧义。
        carry = bool(kwargs.get("carry_over_context"))
        if old_sid and carry:
            old_file = os.path.join(store_dir, f"{_safe_session_name(old_sid)}.jsonl")
            if os.path.exists(old_file):
                self.store = JsonlStore.materialize(old_file, target, reopen_from=old_sid)
                logger.info("supercompress: carry_over handoff %s -> %s (%d segs)",
                            old_sid, session_id, self.store.line_count)
            else:
                self.store = JsonlStore(target)
                logger.warning("supercompress: carry_over requested but archive for %s not found (%s)",
                               old_sid, old_file)
            # B 卡：model 旁路平行交接（旧 model 文件存在 → materialize 字节稳定拷贝，否则新建；
            # 不新增自动 carry，仍只认宿主 carry_over_context kwarg，决策 8；kill-switch OFF → 不建）
            self._model_store = self._build_model_store(old_file=old_file, old_sid=old_sid)
            self._model_fused = self._model_store.fused if self._model_store is not None else False
        else:
            existed = os.path.exists(target) and os.path.getsize(target) > 0
            self.store = JsonlStore(target)
            # B 卡：model 旁路新建 / 冷重开（旧 model 文件存在 → 构造即 replay，与主 store 平行，决策 8）。
            # 定序：主 store 先 rebind 到 target，旁路路径才 = splitext(target) + '.model.jsonl' —
            # 先建 model 再 rebind 会把旁路错写到上一个 session 的 <prev>.model.jsonl（决策 1 同住同 sid 前缀、G2 单源）。
            # kill-switch OFF → 不建（零文件零线程，开关语义闭环）。
            self._model_store = self._build_model_store()
            self._model_fused = self._model_store.fused if self._model_store is not None else False
            if existed:
                # 同 session 冷重开（进程重启）：replayed P1 中间推理内存标 stale（文件字节不动）
                for seg in self.store.by_hash.values():
                    if seg.tier == "P1":
                        seg.stale = True
                logger.info("supercompress: session %s reopened, %d segs replayed (P1 stale-marked), build=%s",
                            session_id, self.store.line_count, self.build_id)
                # P3-block-②: 冷重开（同 sid，store 文件仍在）→ 把上一进程的
                # block-② recall cache 从 sidecar 回填，让重启后首个
                # select_context 命中复用分支（query 未变 + K 段仍存活时）。
                # 只挂 existed 分支：fresh / carry_over 的 sidecar 要么不存在
                # 要么属旧 session，回填正是注释「防跨 session 泄漏」要防的。
                self._recall_cache_rehydrate()
                # B 卡：冷重开 model store P1 行内存标 stale（与主 store 同语义 ×0.5，字节不动，决策 7）
                for mseg in (self._model_store.live_segments() if self._model_store else []):
                    if mseg.tier == "P1":
                        mseg.stale = True
            else:
                logger.info("supercompress: session %s start (fresh), store=%s, build=%s",
                            session_id, target, self.build_id)
        # P2：会话边界 drift 态恢复——有盘内容（carry_over/冷重开）rehydrate（取盘上
        # 最大整数 epoch + 重建锚点，避免冷重启 epoch 失配假阳性漂移）；fresh session reset。
        if self.store.line_count:
            self._drift.rehydrate(list(self.store.by_hash.values()))
        else:
            self._drift.reset()
        # B 卡：model 旁路 replay hydrate marker + 保险丝检查（三分支统一入口；N>0 才打；
        # fused → 召回并集 / T4 注入跳过该 store，文件保留不删）
        self._model_replay_hook()
        # 新 session 窗口重置：force_evict 是会话级升级态，fresh 窗口不应继承
        self._force_evict_mode = False
        # P0.4：rotation advisory 会话边界清零（对称 on_session_reset，防泄漏进新 session）
        self.rotation_advisory = False
        self._adv_consec = 0
        # P3：rerank 缓存会话边界清零（防跨 session 泄漏：新 store 的窗口指纹从 0 起步）
        self._rerank_cache = {}
        # P4：边界扩展缓存会话边界清零（对称 rerank 缓存；防跨 session 泄漏旧 epoch 扩词/精排）
        self._p4_cache.clear()
        # block-② recall cache（t_12f38672）：会话边界（轮转/新 session）清 recall 实例态
        # （session 记忆 scope，防旧 session 指纹/recall 结果泄漏；对称 on_session_reset）。
        self._recall_cache = None
        self.last_prompt_tokens = 0
        # 宿主契约 real-usage pairing 会话边界清零（对称 on_session_reset，对齐内置
        # _reset_real_usage_pairing context_compressor.py:2193-2196）：compact 后一次性 latch
        # 属旧 session 压缩后态，rebind/轮转 不得泄漏进新 session（否则新 session 首 turn 仍被
        # auto/idle compact 的 `if awaiting: return` 锁死）；置于幂等早退之后 → live 重入不清。
        self.last_real_prompt_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._verify_compaction_cleared_threshold = False
        # per-turn cache 遥测会话边界清零（对称 on_session_reset，防旧 session 的 cache 命中率 / 读数
        # 泄漏进 rebind/轮转 新 session 的读面；置于幂等早退之后 → live 重入不清）。
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_cache_hit_ratio = None
        self.session_id = session_id
        self._bound_session = session_id

    def on_session_reset(self) -> None:
        """/new / /reset：flush 在途 P1/P2 使当前 session 归档字节完整（carry-over 交接原子点）。

        宿主 `/new` 走 on_session_reset（非 on_session_end，base 契约）——不 flush 则 carry_over
        materialize 读到半成品归档。flush 后不改动 store 绑定（宿主随后 on_session_start 重绑）。

        重置集对齐宿主基类 ContextEngine.on_session_reset（agent/context_engine.py:187-205）：
        last_prompt/completion/total_tokens + compression_count 全 0（0 = "无真实 usage 用粗估"，
        非 -1 sentinel #36718）；另清 _force_evict_mode（会话级升级态，与 on_session_start 对称——
        旧 session 末 turn 的 backpressure 强制逐出态不得泄漏进新 session 首 turn）。
        super() 在测试环境 base=object 时不存在 → hasattr 守卫优雅降级，手动 4 字段仍确定性对齐。
        """
        try:
            self._aux.flush(timeout=5.0)
        except Exception:
            pass
        try:
            self.store.flush()  # P1/P2 lazy 批落盘（会话边界原子点）
        except (StoreError, OSError) as e:
            # R2：full-disk 时 flush 抛 OSError（EIO/ENOSPC，store 写路径归一为
            # StoreError("io_error: ...")，双 catch 防裸 OSError 逃逸到宿主 /new 面）。
            # 优雅降级：log warning + 水位观测，跳过本轮持久化（pending 批留内存可重试，
            # 断电丢批上界不变），重置流程继续走完——不炸宿主 /new。
            logger.warning("supercompress: session-reset flush failed (degraded, "
                           "pending P1/P2 kept in memory, reset proceeds, store=main): %s", e)
            try:
                self.store.log_water_warn("session-reset-flush-failed", e)
            except Exception:
                pass
        # B 卡：model 旁路 lazy 批 flush（会话边界原子点，镜像主 store flush 块）+ fused 态清零
        self._model_session_flush("session-reset")
        self._model_fused = False

        # 基类重置集（若宿主真基类在 MRO 中则委托 super；object 基类则手动 4 字段）
        try:
            if hasattr(super(), "on_session_reset"):
                super().on_session_reset()
        except Exception:
            pass
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        # 宿主契约 real-usage pairing 会话边界清零（对齐内置 _reset_session_compaction_state /
        # _reset_real_usage_pairing，context_compressor.py:2193-2220）：compact 后一次性 latch
        # （awaiting）属旧 session 的压缩后态，不得泄漏进 /new 新 session（否则新 session 首 turn
        # 仍被 turn_preflight/turn_context_compaction 的 `if awaiting: return` 锁死 auto/idle compact）；
        # 与 P0.4 rotation advisory / rerank / p4 缓存的会话边界清零同口径（同在本函数末尾对称清零）。
        self.last_real_prompt_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._verify_compaction_cleared_threshold = False
        self._force_evict_mode = False
        # per-turn cache 遥测会话边界清零（防旧 session 的 cache 命中率 / 读数泄漏进 /new 新 session 的读面）：
        # 与 last_prompt_tokens 等宿主契约字段的会话边界清零同口径（read/write 归 0、命中率归 None）。
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_cache_hit_ratio = None
        # P0.4：rotation advisory 会话边界清零（对称 on_session_start，防泄漏进新 session）
        self.rotation_advisory = False
        self._adv_consec = 0
        self._drift.reset()   # P2：会话边界 drift 态全清（session 记忆 scope）
        # P2 telemetry：会话边界清会话内 records（session 记忆 scope，同 _drift.reset）；
        # JSONL sink 文件 = 跨会话 append-only 观测面（本 reset 不动文件，文件在 end 面 flush）
        self.telemetry.reset()
        # P3：rerank 缓存会话边界清零（对称 on_session_start）
        self._rerank_cache = {}
        # P4：边界扩展缓存会话边界清零（对称 on_session_start）
        self._p4_cache.clear()
        # block-② recall cache（t_12f38672）：会话边界（/new）清 recall 实例态（session
        # 记忆 scope，防旧 session 指纹/recall 结果泄漏进新 session；与 rerank/p4 缓存清零同口径）。
        self._recall_cache = None
        # 紧凑摘要账本（t_63bf3ba2）：会话边界（/new）整段清——条目归 0、generation/seals +1
        # （同 /new 新 session 首 turn 不复用旧 session 累计前缀；与 rerank/p4 缓存清零同口径）。
        # 路径不变（宿主随后 on_session_start rebind 新 session 时 _compact_ledger 访问器按
        # store.path 失配重建新侧车）；本处清的是**旧 session 内存账本态**，防跨 /new 泄漏。
        self._compact_ledger_reset()
        logger.info("supercompress: session reset (flush), store lines=%d", self.store.line_count)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # 会话边界：先汇合异步 digest worker（在途段尽量 ready，缩短会话收尾等待），
        # 再 flush P1/P2 lazy 批落盘。flush 超时/无 worker → 立即返回，不阻塞宿主。
        try:
            self._aux.flush(timeout=5.0)
        except Exception:
            pass
        try:
            self.store.flush()  # P1/P2 lazy 批落盘（会话边界原子点）
        except (StoreError, OSError) as e:
            # R2：与 on_session_reset 同口径（full-disk OSError 优雅降级 + 水位观测）——
            # 会话收尾面同样不炸宿主（pending 批留内存，断电丢批上界不变）。
            logger.warning("supercompress: session-end flush failed (degraded, "
                           "pending P1/P2 kept in memory, store=main): %s", e)
            try:
                self.store.log_water_warn("session-end-flush-failed", e)
            except Exception:
                pass
        # B 卡：model 旁路 lazy 批 flush（aux worker flush 后；OSError 优雅降级，镜像主 store 块，marker + store=model 判别）
        self._model_session_flush("session-end")

        logger.info("supercompress: session %s end, store lines=%d",
                    session_id, self.store.line_count)
        # P4 钩子②（digest 完成回写池，t_026a17f8）：会话末 = 重启前权威汇合点
        # （上方 aux.flush 已汇合在途 digest worker）——此刻新 ready 段全部落池，
        # 重启后 replay 查池命中免重蒸（P4 核心目标）。幂等零 IO 快退，fail-open。
        self._digest_pool_write_hook()
        # P2 telemetry：会话末 flush 会话内 records 落 JSONL sink（跨会话 append-only
        # 攒数喂 P3；fail-open：写失败/无 sink → 清 records，不炸宿主收尾面）
        try:
            self.telemetry.flush()
        except Exception:
            pass

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return bool(self.store.by_hash)

    def is_available(self) -> bool:
        return True


def get_engine(**kwargs: Any) -> SupercompressContextEngine:
    """旧 `__init__.py` 导入面兼容（宿主 loader 走 register(ctx)；此入口留作直引 / bench 探针）。"""
    return SupercompressContextEngine(**kwargs)
