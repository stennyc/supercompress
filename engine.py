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
旧 Ollama `No route to host`（旧 engine.py:1795-1802 调内网 Ollama embedding 端点）整类消失。
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
from sc.tagger import Tagger
from sc.drift import DriftTracker


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
        # P0.4 rotation advisory：纯信号置位（零注入/零遥测/零 cache 影响，宿主契约 fail-safe）；
        # 会话边界（on_session_start/on_session_reset）对称清零，防泄漏进新 session
        self.rotation_advisory: bool = False
        self._adv_consec: int = 0

        logger.info(
            "tcm-supercompress: shell v2 initialized — store=%s, context_length=%d, "
            "compact_ratio=%.2f, force_ratio=%.2f, segments_on_disk=%d, call_llm=%s",
            self.store_path, self.context_length, self.compact_ratio,
            self.force_compact_ratio, self.store.line_count,
            "available" if self._call_llm else "not available",
        )

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
                self._drift.observe_user(str(m.get("content") or ""))
                seg.session_topic = str(self._drift.current_epoch)
                self.store.append(seg)
            self._aux.end_turn()
            result = Distiller(window, self.store, self._aux, force_evict=self._force_evict_mode).run()
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
            recaller = Recaller(self.store, window)
            hits = recaller.search(query, k=3)
            # P2.3 soft：旧 epoch 非结构 P1/P2 召回降权（P1 search 打分核零动，独立 rerank 步）
            hits = self._drift.soft_rank(hits)
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
            # 执行 distill
            result = Distiller(window, self.store, self._aux, force_evict=self._force_evict_mode).run()
            # 如果 normal distill 失败（transient_skip）且未启用 force_evict，自动降级
            if result.status == "backpressure" and not self._force_evict_mode:
                logger.warning("tcm-supercompress: distill backpressure, trying force_evict")
                self._force_evict_mode = True
                result = Distiller(window, self.store, self._aux, force_evict=True).run()
                logger.info("tcm-supercompress: force_evict result status=%s events=%s",
                            result.status, result.events)
            if result.status not in ("ok", "backpressure"):
                logger.warning("tcm-supercompress: distill status=%s events=%s",
                               result.status, result.events)
        except StoreError as e:
            logger.error("tcm-supercompress: compress degraded (store error): %s", e)
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
                logger.info("tcm-supercompress: session %s reopened, %d segs replayed (P1 stale-marked)",
                            session_id, self.store.line_count)
            else:
                logger.info("tcm-supercompress: session %s start (fresh), store=%s", session_id, target)
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
        except StoreError as e:
            logger.error("tcm-supercompress: session-reset flush failed: %s", e)
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
        except StoreError as e:
            logger.error("tcm-supercompress: session-end flush failed: %s", e)
        logger.info("tcm-supercompress: session %s end, store lines=%d",
                    session_id, self.store.line_count)

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return bool(self.store.by_hash)

    def is_available(self) -> bool:
        return True


def get_engine(**kwargs: Any) -> SupercompressContextEngine:
    """旧 `__init__.py` 导入面兼容（宿主 loader 走 register(ctx)；此入口留作直引 / bench 探针）。"""
    return SupercompressContextEngine(**kwargs)
