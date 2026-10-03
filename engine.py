"""Hermes host adapter shell — 薄接线层（PM 路①定稿：壳只做接线，零新业务逻辑）。

宿主契约（hermes-agent 宿主仓实码核清，@coder 直查 2026-09-27；宿主仓位置因安装而异，
本插件不引用宿主仓绝对路径——安装根从 __file__ 反推，数据根走 HERMES_HOME env）：
- loader: plugins/plugin_loader.py `load_plugin_module` → `instance_from_module`
  （`mod.register(collector)` 优先，collector._EngineCollector `register_context_engine(engine)` 收实例；
  register 失败则回退扫描模块属性找 ContextEngine 子类无参构造）。`probe_availability` 调
  实例 `is_available()`（若有）。
- 宿主调用面: select_context / update_from_response / should_compress / compress（宿主按签名过滤
  kwargs；旧 `tcmnet_hint` kwarg 保留）/ on_turn_complete / on_session_start|end / on_session_reset。
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
from sc.inject import choose_forms, render_block, block_tokens
from sc.recall import Recaller
from sc.rerank import RerankGate, rerank_hits
from sc.semantic import BoundaryExpander, BOUNDARY_TASK
from sc.tagger import Tagger
from sc.drift import DriftTracker
from sc.version import build_id, desync_warning


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
DISTILL_WATER_FRAC = 0.8  # sc.distill 软水位（组件内部定义；壳不重算算法）
# P1 extended-BM25 招③：召回时间衰减接线面（Recaller 构造默认 decay_lambda）。
# 默认 0.0 = 不启用（钉死面零扰动）；启用 = 同分下越早 seg.seq 越多被降（纯函数、无 wall-clock）。
DEFAULT_RECALL_DECAY_LAMBDA = 0.0
# P0.4 rotation advisory 阈值（引擎侧纯信号，零注入、零遥测、零 cache 影响；宿主契约 fail-safe）：
ROTATION_PROMPT_RATIO = 0.85   # T_prompt：连续压力达标置位线
ROTATION_RESET_RATIO = 0.75    # < 此值 = 已 compress 过，清 advisory（引擎侧等价「轮转前重读 pressure 二次确认防误轮转」）
ADVISORY_DEBOUNCE_TURNS = 2    # 连续 N turn 超阈值才置位（session-redesign S-Phase 6 防抖口径）


def _default_store_root() -> str:
    """默认 store 根（安装可移植，零宿主改动）：HERMES_HOME-aware 解析。

    优先级：显式 ``external_store_path`` 配置（__init__ 消费）> ``HERMES_HOME`` env
    （生产 gateway 进程 per-profile 自带，指向 profile 数据根）> ``~/.hermes``（平台默认）。
    生产轮转 rebind 缺 ``hermes_home`` kwarg 时，env 回退即落对 profile 目录
    （reviewer S1 Major#1 效果达成，宿主 conversation_compression.py 零改动）；
    任意安装环境设 ``HERMES_HOME`` 指数据根即可，无 /root、/usr/local 写死。
    与 sc/_bootstrap.hermes_home() 同源语义（同一 env 名，不 import 宿主）。
    """
    hh = os.environ.get("HERMES_HOME", "").strip()
    base = hh if hh else os.path.expanduser("~/.hermes")
    return os.path.join(base, "supercompress_store")


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
                    logger.info("tcm-supercompress: aux channel task=%s short-timeout/failed (%s) "
                                "-> falling back to main model route", task, e)
                continue
            # 兼容 dict 和 namespace 响应
            try:
                if isinstance(response, dict):
                    return response["choices"][0]["message"]["content"]
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
    """宿主 message → sc.Segment（接线：role→provenance；classify→tier/structural，全委托 sc.tagger）。"""
    provenance = "user" if msg.get("role") == "user" else "model"
    text = str(msg.get("content") or "")
    tier, structural, _regime, _reason = tagger.classify(text, provenance)
    return Segment(tier=tier, provenance=provenance, original_text=text,
                   ts=ts, structural_p0=structural)


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

        # token 账（宿主 run_agent.py 直接读这些字段）
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.compression_count: int = 0
        # 预检阈值读 sc.distill 软水位口径（80% 容量 = 蒸馏触发线；壳不重算）
        self.threshold_tokens: int = int(self.context_length * DISTILL_WATER_FRAC)
        # 宿主实调位（tui_gateway/session_compression.py:169）：threshold_tokens 配置的上限。
        # 默认 None = 不加 cap（保持 80% 软水位）；宿主下发正整数 cap 且更小时，蒸馏触发线收紧到 cap。
        # 纯配置规整（镜像内置 ContextCompressor._coerce_threshold_tokens_cap，context_compressor.py:2602-2611），
        # 壳不发明新算法——缺这个方法正是宿主 "Could not apply live compression config" WARNING 的根因。
        self.threshold_tokens_cap: Optional[int] = None
        # force_evict_mode: 禁用 digest 要求，直接逐出 resident P0/P1/P2
        self._force_evict_mode: bool = False
        # P2（S-Phase 5）drift tracker：task epoch 状态机 + 锚点 + soft/hard 判定
        #（纯 sc 层，零 LLM/零网络/确定性；复用 P1 tokenize）
        self._drift = DriftTracker()
        # P3（S-Phase 4）gate rerank：LLM 两步重排（独立 step，P1 打分核零动，同 P2 soft_rank 模式）。
        # 端点注入 = aux 主模型端点（宿主主模型共享，零新 provider / 零模型名 / 零 provider 块）；
        # gate 默认保守（low_score_threshold=0.0 → 只在「召回空 / 显式搜索词」触发，非常态 LLM 调用）；
        # 缓存键 (sha256(query), store.line_count) 会话内幂等（plain dict：CPython GIL 下
        # get/set 原子，并发最坏 = 一次良性的 LLM 调用重复，不炸不脏）；
        # 会话边界（on_session_start/on_session_reset）清零，防跨 session 泄漏。
        self.rerank_low_score_threshold: float = float(_get("rerank_low_score_threshold", 0.0))
        self._rerank_gate = RerankGate(low_score_threshold=self.rerank_low_score_threshold)
        self._rerank_cache: dict = {}
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
            "tcm-supercompress: shell v2 initialized — store=%s, context_length=%d, "
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
                    "tcm-supercompress: aux endpoint = injected call_llm (model=%s, "
                    "aux_timeout=%.0fs main_fallback_timeout=%.0fs digest_max_tokens=%d)",
                    self._main_runtime.get("model"), self.aux_timeout,
                    self.main_fallback_timeout, self.digest_max_tokens)
                return AuxProvider(ep, fail_cooldown=self.digest_fail_cooldown)
            except Exception as e:
                logger.warning("tcm-supercompress: injected call_llm endpoint failed: %s", e)
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
        logger.info("tcm-supercompress: update_model model=%s ctx=%d aux=%s runtime=%s call_llm=%s",
                    model, context_length, type(self._aux.main_endpoint).__name__,
                    "injected" if self._main_runtime else "none",
                    "available" if self._call_llm else "unavailable")

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
        self._window = Window(self.context_length, [WindowSeg(s) for s in segs])
        self._tagger = Tagger(self._aux, "")
        return self._window

    # ── 宿主调用面（abstract 三件 + 可选钩子）──

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """token 账更新（宿主每次 LLM 调用后）：直接存 canonical 桶，无算法。"""
        self.last_prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        self.last_completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
        self.last_total_tokens = int(usage.get("total_tokens", 0))

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """sc.distill 软水位语义：当前 token 账 > 生效触发线（80% 容量，宿主 cap 收紧）→ 本 turn 触发蒸馏逐出。"""
        tokens = prompt_tokens if prompt_tokens else self.last_prompt_tokens
        return tokens > self._effective_threshold()

    def _effective_threshold(self) -> int:
        """生效触发线：sc.distill 软水位（80% 容量），宿主下发 `threshold_tokens_cap`（session_compression.py:169）
        且更小则收紧到 cap；无 cap / cap 更大则保持软水位。纯配置规整，壳不重算算法。"""
        cap = self.threshold_tokens_cap
        return cap if (cap is not None and cap < self.threshold_tokens) else self.threshold_tokens

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
        """宿主 preflight（无真实 usage）：粗估 token（est_tokens 组件）过生效触发线即触发。"""
        rough = sum(est_tokens(str(m.get("content") or "")) for m in messages)
        return rough > self._effective_threshold()

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
                    "tcm-supercompress: rotation advisory set (pressure=%.2f, "
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
            window = self._window_fresh()
            recaller = Recaller(self.store, window)
            candidates = recaller.search(text, k=3)    # extended-BM25 粗排 top-3（扩词后喂回同面）
            res = self._p4.expand(epoch, seed, candidates, task_text=text)
            if not res.degraded:
                # ③ 重建 epoch 锚点：扩展词面并入 drift 锚点（内存面；session_topic 整数 epoch 标 /
                # jsonl 字节零动——与 P0 合流，epoch 锚点 = 扩展后任务关键词集）
                merged = self._drift.merge_anchor(res.expanded_terms)
                logger.info(
                    "tcm-supercompress: P4 boundary expand epoch=%d expanded=%d "
                    "anchor_merged=%d verdicts=%d cache=%d",
                    epoch, len(res.expanded_terms), merged, len(res.verdicts), len(self._p4_cache))
            else:
                logger.info("tcm-supercompress: P4 boundary expand epoch=%d degraded -> pure-BM25 "
                            "fallback (cooldown=%s)", epoch, self._p4.cooldown_status())
        except Exception as e:                          # 壳 fail-open：P4 面异常不炸宿主 turn
            logger.warning("tcm-supercompress: P4 boundary expand no-op (pure-BM25 fallback): %s", e)

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None,
                         **kwargs: Any) -> None:
        """turn 末接线：新 user 消息落盘（P0 即写即 fsync）+ aux turn 边界 + Distiller.run 逐出。

        全委托 sc：tagger.classify → store.append（tier-aware fsync）→ aux.end_turn → Distiller.run
        （80% 水位逐出 + push cap + 锚点发射）。壳内无任何逐出/预算算法体。
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
                logger.info("tcm-supercompress: ingress rejected %d room-skeleton user msg(s) "
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
            # R3：force_ratio 接 engine 同源配置 force_compact_ratio（kwarg = 接线非算法）
            result = Distiller(window, self.store, self._aux, force_evict=self._force_evict_mode,
                               force_ratio=self.force_compact_ratio).run()
            if result.status not in ("ok", "backpressure"):
                logger.warning("tcm-supercompress: distill status=%s events=%s",
                               result.status, result.events)
            # P0.4：turn 末 pressure 检查（防抖口径）+ 会话边界外不触碰任何持久态
            self._check_rotation_advisory()
        except StoreError as e:
            logger.error("tcm-supercompress: store error (fail-safe, window untouched): %s", e)
        except Exception as e:  # 壳永不把宿主 turn 炸掉（fail-open，宿主契约）
            logger.error("tcm-supercompress: on_turn_complete degraded: %s", e)

    def select_context(
        self, request_messages: List[Dict[str, Any]],
        *, conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None,
        budget_tokens: int = 0,
        **kwargs: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """sc.recall 接线：Recaller.search（pull budget 内 P0/P1 hit）；无 hit → 返回 None（字节不变）。

        宿主契约：替换只在选中请求内，持久化 transcript 不动；malformed 永不 reach provider。
        """
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
            window = self._window_fresh()
            recaller = Recaller(self.store, window, decay_lambda=self.recall_decay_lambda)
            # 招③ 时间衰减接线面：decay_lambda = 实例默认（engine.recall_decay_lambda，
            # kwarg/配置 recall_decay_lambda 控制启用；默认 0 = 钉死面不扰动）
            # P4：边界扩展词面喂回 extended-BM25（词面扩词，不是向量；命中缓存 = 该 epoch
            # 边界 LLM 成功面 → 扩词并入 query 词面；无缓存/回退 → 原 query，常驻 100% 确定性 G4）
            p4_terms = self._p4.expanded_terms(self._drift.current_epoch)
            p4_query = query + ((" " + " ".join(p4_terms)) if p4_terms else "")
            hits = recaller.search(p4_query, k=3)
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
                gate=self._rerank_gate)
            recalled = [h.seg.original_text for h in hits
                        if h.seg.tier in ("P0", "P1") and h.score > 0.0]
            # Egress 过滤（存量止血）：磁盘上已存的 203 条 P0 group-skeleton 段
            # （pre-fix 入向）会按词重叠被召回注入到任意后续会话 = 跨会话污染。
            # 与 ingress 共用 is_room_skeleton 单源谓词（不删磁盘数据、可回滚；
            # 仅把房间骨架逐出自动注入面，观测行留痕）。
            _skel_out = [t for t in recalled if is_room_skeleton(t)]
            if _skel_out:
                logger.info("tcm-supercompress: recall egress cut %d room-skeleton segment(s) "
                            "(legacy P0 group-chat residue)", len(_skel_out))
            recalled = [t for t in recalled if not is_room_skeleton(t)]
            if not recalled:
                return None
            head = [m for m in request_messages if m.get("role") == "system"][:1]
            out = list(head) + [{"role": "system",
                                 "content": "[supercompress recall]\n" + "\n\n".join(recalled)}]
            out += [m for m in request_messages if m.get("role") != "system"]
            logger.info("tcm-supercompress: select_context recalled %d segment(s), breach=%d",
                        len(recalled), len(recaller.breach_log))
            return out
        except Exception as e:
            logger.warning("tcm-supercompress: select_context no-op (recall degraded): %s", e)
            return None

    def compress(
        self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None, force: bool = False, memory_context: str = "",
        tcmnet_hint: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """sc.distill 接线 + 宿主消息面输出：蒸馏后 = system head + P0/P1 常驻段 + 尾部。

        tcmnet_hint kwarg 保留（旧引擎宿主调用面兼容；本壳不消费 TCM bridge——
        旧 integration/ 面随旧目录进备份，数据面零携带）。
        """
        self.compression_count += 1
        tokens = current_tokens or self.last_prompt_tokens
        if not force and tokens < self.threshold_tokens * self.force_compact_ratio:
            logger.info("tcm-supercompress: compress (light trim) tokens=%d threshold=%d",
                        tokens, self.threshold_tokens)
        else:
            logger.info("tcm-supercompress: compress (distill) tokens=%d force=%s", tokens, force)
        try:
            window = self._window_fresh()
            # 检查是否需要启用 force_evict_mode
            if not self._force_evict_mode and window.used_tokens > self.context_length:
                overflow = window.used_tokens - self.context_length
                p0_tokens = sum(ws.tokens() for ws in window.segs if ws.segment.tier == "P0")
                push_cap = int(self.context_length * 0.15)
                # 如果 overflow 超过 push cap，说明 P0 常驻集过大，需要强制逐出
                if p0_tokens > push_cap and overflow > push_cap * 0.5:
                    logger.warning("tcm-supercompress: overflow %d tokens > 50%% push_cap %d, "
                                   "enabling force_evict_mode", overflow, push_cap)
                    self._force_evict_mode = True
            # 执行 distill（R3：force_ratio 接同源配置 force_compact_ratio）
            result = Distiller(window, self.store, self._aux, force_evict=self._force_evict_mode,
                               force_ratio=self.force_compact_ratio).run()
            # 如果 normal distill 失败（transient_skip）且未启用 force_evict，自动降级
            if result.status == "backpressure" and not self._force_evict_mode:
                logger.warning("tcm-supercompress: distill backpressure, trying force_evict")
                self._force_evict_mode = True
                result = Distiller(window, self.store, self._aux, force_evict=True,
                                   force_ratio=self.force_compact_ratio).run()
                logger.info("tcm-supercompress: force_evict result status=%s events=%s",
                            result.status, result.events)
            if result.status not in ("ok", "backpressure"):
                logger.warning("tcm-supercompress: distill status=%s events=%s",
                               result.status, result.events)
        except (StoreError, OSError) as e:
            # R2：full-disk 时 store 交互抛 OSError（EIO/ENOSPC）——store 写路径已归一为
            # StoreError("io_error: ...")，此处双 catch 防御 replay/aux 面裸 OSError 逃逸。
            # 优雅降级：log warning + 磁盘水位观测（剩余空间 + 触发动作，ops 监控），
            # 本轮跳过持久化，不炸宿主（压缩失败 ≠ turn 失败；与 on_turn_complete 宽
            # catch 口径对齐——fail-open 返回原 messages，宿主本 turn 原样继续）
            logger.warning("tcm-supercompress: compress degraded (store error, "
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
            logger.info("tcm-supercompress: compress egress cut %d room-skeleton segment(s) "
                        "from anchor manifest", len(_skel))
        kept = [ws for ws in kept if not is_room_skeleton(ws.segment.original_text)]
        rendered = choose_forms(kept, window, query=focus_topic or "")
        block = render_block(rendered)
        _forms: Dict[str, int] = {}
        for _ws, _ff, _tx in rendered:
            _forms[_ff] = _forms.get(_ff, 0) + 1
        out: List[Dict[str, Any]] = list(messages[:self.protect_first_n])
        if block:
            out.append({"role": "system",
                        "content": f"[supercompress] staged P0/P1 segments "
                                    f"({len(rendered)} segs, {block_tokens(rendered)} tok, "
                                    f"forms: {dict(sorted(_forms.items()))}):\n"
                                    + block})
        out.extend(messages[-self.protect_last_n:] if messages else [])
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
        store_dir = (os.path.join(os.path.expanduser(home), "supercompress_store")
                     if home else self._legacy_store_dir)
        target = os.path.join(store_dir, f"{_safe_session_name(session_id)}.jsonl")
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
                logger.info("tcm-supercompress: carry_over handoff %s -> %s (%d segs)",
                            old_sid, session_id, self.store.line_count)
            else:
                self.store = JsonlStore(target)
                logger.warning("tcm-supercompress: carry_over requested but archive for %s not found (%s)",
                               old_sid, old_file)
        else:
            existed = os.path.exists(target) and os.path.getsize(target) > 0
            self.store = JsonlStore(target)
            if existed:
                # 同 session 冷重开（进程重启）：replayed P1 中间推理内存标 stale（文件字节不动）
                for seg in self.store.by_hash.values():
                    if seg.tier == "P1":
                        seg.stale = True
                logger.info("tcm-supercompress: session %s reopened, %d segs replayed (P1 stale-marked), build=%s",
                            session_id, self.store.line_count, self.build_id)
            else:
                logger.info("tcm-supercompress: session %s start (fresh), store=%s, build=%s",
                            session_id, target, self.build_id)
        # P2：会话边界 drift 态恢复——有盘内容（carry_over/冷重开）rehydrate（取盘上
        # 最大整数 epoch + 重建锚点，避免冷重启 epoch 失配假阳性漂移）；fresh session reset。
        if self.store.line_count:
            self._drift.rehydrate(list(self.store.by_hash.values()))
        else:
            self._drift.reset()
        # 新 session 窗口重置：force_evict 是会话级升级态，fresh 窗口不应继承
        self._force_evict_mode = False
        # P0.4：rotation advisory 会话边界清零（对称 on_session_reset，防泄漏进新 session）
        self.rotation_advisory = False
        self._adv_consec = 0
        # P3：rerank 缓存会话边界清零（防跨 session 泄漏：新 store 的窗口指纹从 0 起步）
        self._rerank_cache = {}
        # P4：边界扩展缓存会话边界清零（对称 rerank 缓存；防跨 session 泄漏旧 epoch 扩词/精排）
        self._p4_cache.clear()
        self.last_prompt_tokens = 0
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
            logger.warning("tcm-supercompress: session-reset flush failed (degraded, "
                           "pending P1/P2 kept in memory, reset proceeds): %s", e)
            try:
                self.store.log_water_warn("session-reset-flush-failed", e)
            except Exception:
                pass
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
        self._force_evict_mode = False
        # P0.4：rotation advisory 会话边界清零（对称 on_session_start，防泄漏进新 session）
        self.rotation_advisory = False
        self._adv_consec = 0
        self._drift.reset()   # P2：会话边界 drift 态全清（session 记忆 scope）
        # P3：rerank 缓存会话边界清零（对称 on_session_start）
        self._rerank_cache = {}
        # P4：边界扩展缓存会话边界清零（对称 on_session_start）
        self._p4_cache.clear()
        logger.info("tcm-supercompress: session reset (flush), store lines=%d", self.store.line_count)

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
            logger.warning("tcm-supercompress: session-end flush failed (degraded, "
                           "pending P1/P2 kept in memory): %s", e)
            try:
                self.store.log_water_warn("session-end-flush-failed", e)
            except Exception:
                pass
        logger.info("tcm-supercompress: session %s end, store lines=%d",
                    session_id, self.store.line_count)

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return bool(self.store.by_hash)

    def is_available(self) -> bool:
        return True


def get_engine(**kwargs: Any) -> SupercompressContextEngine:
    """旧 `__init__.py` 导入面兼容（宿主 loader 走 register(ctx)；此入口留作直引 / bench 探针）。"""
    return SupercompressContextEngine(**kwargs)
