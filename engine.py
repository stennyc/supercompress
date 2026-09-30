"""Hermes host adapter shell — 薄接线层（PM 路①定稿：壳只做接线，零新业务逻辑）。

宿主契约（远端 117 `/usr/local/lib/hermes-agent` 实码核清，@coder 117 直查 2026-09-27）：
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
import sys
import time
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
from sc.recall import Recaller
from sc.tagger import Tagger


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
DEFAULT_EXTERNAL_STORE_PATH = "~/.hermes/supercompress_store"
DISTILL_WATER_FRAC = 0.8  # sc.distill 软水位（组件内部定义；壳不重算算法）


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
        store_path = _get("external_store_path", DEFAULT_EXTERNAL_STORE_PATH)
        # 数据面：单一 active.jsonl（@data-engineer 契约：旧 chunk_* store 零携带零触碰，
        # 新 active.jsonl 是独立新文件，同住目录互不干扰）
        self.store_path = os.path.join(os.path.expanduser(store_path), "active.jsonl")

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
                self.store.append(_make_segment(m, self._tagger, ts=time.time()))
            self._aux.end_turn()
            result = Distiller(window, self.store, self._aux, force_evict=self._force_evict_mode).run()
            if result.status not in ("ok", "backpressure"):
                logger.warning("tcm-supercompress: distill status=%s events=%s",
                               result.status, result.events)
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
        kept_texts = [ws.segment.original_text for ws in window.segs
                      if ws.segment.tier in ("P0", "P1")]
        # Egress 过滤（同 select_context）：已存 group-skeleton P0 段逐出压缩
        # anchor-manifest 注入面——否则压缩时群聊骨架仍驻留进 durable 段块。
        _skel_c = [t for t in kept_texts if is_room_skeleton(t)]
        if _skel_c:
            logger.info("tcm-supercompress: compress egress cut %d room-skeleton segment(s) "
                        "from anchor manifest", len(_skel_c))
        kept_texts = [t for t in kept_texts if not is_room_skeleton(t)]
        out: List[Dict[str, Any]] = list(messages[:self.protect_first_n])
        if kept_texts:
            out.append({"role": "system",
                        "content": "[supercompress] durable P0/P1 segments (anchor manifest):\n"
                                   + "\n---\n".join(kept_texts)})
        out.extend(messages[-self.protect_last_n:] if messages else [])
        return out

    # ── 会话边界 ──

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        self.store = JsonlStore(self.store_path)  # 重放磁盘态（active.jsonl 缺失 = 空窗口，正常首启）
        self.last_prompt_tokens = 0
        logger.info("tcm-supercompress: session %s start, store replayed %d segments",
                    session_id, self.store.line_count)

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
