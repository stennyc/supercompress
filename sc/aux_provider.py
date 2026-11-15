"""Block ④ aux provider（gate v4 (A)：零新 provider——aux ≡ 主模型端点）。

(A) 锁死语义：
- aux = 主模型端点（Hermes 原生 `auxiliary.compression` 配置，不单独部署本地 aux 实例，
  用户零配置）；`fallback_endpoint` = 主端点（失败域同源，无第二 provider），
  `fallback_chain` 链尾 = main（原生兜底，0 行自包）；
- digest 生成 turn 内排队（不与主模型解码撞车）、turn 末单槽执行（`max_concurrency:1`）；
- `transient_fail` = 主 LLM 瞬态失败（限流/超时）→ 单事件 skip、下事件自愈（`aux.transient` 标志）；
- digest 幂等再生键 = (content_hash, model_version, session_topic) 三元组（scdata.digest_regen_key 单源）；
- gate⑤ 故障注入 marker：`marker_enabled()` off-by-default（生产恒 inert），
  `SC_INJECT_FAULT` 环境变量双向可控（不用 `__debug__`：CPython 生产态恒 True）。

测试替身 `FakeEndpoint` 住 tests/fakes.py——生产 import 面不含测试脚手架。
模块名 `aux_provider`（非 `aux`）：`aux` 是 Windows 保留设备名，git/OS 路径面无法索引。
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time

import scdata

from .schema import Segment

logger = logging.getLogger(__name__)

# R3 健康探测/降级（任务 t_d94cc07e 报告#3）：不加新 endpoint / 新 provider（LLM 约束），
# 探测 = 现有 digest 调用（本端点同通道）的成功/失败统计。连败计数 _consec_fails +
# 退避阶梯（基线 = fail_cooldown，乘数 1×→5×→30× 封顶；engine 默认 60s 配置 →
# 60s→300s→1800s 契约曲线）→ 冷却窗自适应拉长，防「每 N 秒一波 pending 段重发端点」
# 的放大风暴；端点恢复（1 次成功）→ 计数归零、冷却窗关闭回落基线。状态变化打一行 INFO。
COOLDOWN_BACKOFF_MULTIPLIERS = (1.0, 5.0, 30.0)


def effective_cooldown(aux: "AuxProvider", n_fails: int) -> float:
    """第 n_fails 次连败（1 起）对应的冷却窗时长 = ladder[min(n-1, len(ladder)-1)]。

    纯阶梯索引（尾档封顶 = 指数退避封顶语义）；阶梯全 0 → 返回 0.0（冷却窗恒关，
    `if cooldown > 0` 守卫下 `_cooldown_until` 保持 0 —— fail_cooldown=0 的 sc 契约
    替身「transient 单事件自愈」口径 0 行为变化）。冷却窗是否启用由 fail_cooldown
    总开关语义决定（=0 → 阶梯全 0 → 恒关）。
    """
    ladder = getattr(aux, "cooldown_backoff", ()) or ()
    if not ladder:
        return 0.0
    return ladder[min(max(n_fails, 1) - 1, len(ladder) - 1)]

# digest 提示词（S2a/修 S5：之前 LLM 响应正文被丢弃、digest_text 只填 head80 占位——
# 同一处 LLM 调用、零新增调用次数，现在捕获正文并结构化解析出 key_fact/summary）。
DIGEST_INSTRUCTION = (
    "Compress the conversation segment below into strict JSON (no code fence, no extra text):\n"
    '{"key_fact": "<=60 chars: the single most important fact/decision", '
    '"summary": "<=200 chars: 2-4 sentence digest"}\n'
    "--- segment ---\n"
)


def _balanced_json_candidates(t: str) -> list:
    """括号平衡法：从 t 里依次提取所有「首个 { 起、配平 } 结束」的候选片段。

    正确处理字符串内的 { } 和转义引号，避免简单正则误抓散文里自带的片段。
    """
    out = []
    i = 0
    n = len(t)
    while i < n:
        if t[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escaped = False
        j = i
        while j < n:
            ch = t[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(t[i:j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            break  # 配不平（截断）：剩余部分不作为候选
    return out


def parse_digest_response(text: str | None) -> tuple[str, str]:
    """digest LLM 响应 → (key_fact, summary)。容忍裸 JSON / code fence / 纯文本 / None；
    空响应 → ("", "")（调用方回退 head 摘录占位，bench 空响应不变量保持）。"""
    if not text or not str(text).strip():
        return "", ""
    t = str(text).strip()
    # Strip thinking/reasoning blocks BEFORE any JSON extraction.
    # agnes-2.5-flash (and other reasoning models) embed <thinking>...</thinking>
    # or <reasoning>...</reasoning> tags in response content. If these tags
    # contain { or } characters, the greedy r"\{.*\}" regex below would swallow
    # the thinking block + the real JSON into one giant invalid string.
    t = re.sub(r"<thinking[^>]*>.*?</thinking>", "", t, flags=re.DOTALL).strip()
    t = re.sub(r"<reasoning[^>]*>.*?</reasoning>", "", t, flags=re.DOTALL).strip()
    candidates = [t]
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        candidates.insert(0, m.group(1).strip())
    # Use balanced-brace matching instead of greedy \{.*\} to avoid
    # swallowing prose text that contains { or } characters.
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", t):
        candidates.insert(1, m.group(0))
        break
    candidates.extend(_balanced_json_candidates(t))
    for c in candidates:
        try:
            d = json.loads(c)
        except Exception:
            continue
        if isinstance(d, dict):
            # 限长兜底（F5）：LLM 自报字段进 system 通道 L1/L2 行前截断，防无上限正文
            kf = str(d.get("key_fact", "")).strip()[:80]
            sm = str(d.get("summary", "")).strip()[:400]
            if kf or sm:
                return kf, sm
    # 纯文本回退：整体当 summary，key_fact 取头部（仍远优于无）
    return t[:60], t[:400]

# digest 全局并发帽（跨会话同端点去抖，第 2 层）：模块级 BoundedSemaphore，
# 同进程内多会话的 digest 调用共享上限（默认 2；SC_DIGEST_CONCURRENCY 进程级覆盖）。
# BoundedSemaphore 不可 resize → 导入时定死；测试可经 AuxProvider(digest_sem=...) 注入独立实例。
_SC_CONCURRENCY_RAW = os.environ.get("SC_DIGEST_CONCURRENCY", "2").strip()
try:
    _SC_DIGEST_CONCURRENCY = max(1, int(_SC_CONCURRENCY_RAW))
except ValueError:
    _SC_DIGEST_CONCURRENCY = 2
_DIGEST_SEM = threading.BoundedSemaphore(_SC_DIGEST_CONCURRENCY)


class _FlushSignal:
    """flush(timeout) 汇合点信号：worker 处理到即 set event（不中断队列）。"""
    __slots__ = ("event",)

    def __init__(self, event: threading.Event) -> None:
        self.event = event


def marker_enabled() -> bool:
    """gate⑤ 故障注入 marker：off-by-default（生产恒 inert，inject_fault_after 参数被忽略）。"""
    return os.environ.get("SC_INJECT_FAULT") not in (None, "", "0", "false", "False")


class AuxProvider:
    """(A) aux ≡ 主模型端点。endpoint 注入：生产 = Hermes 原生网关，测试 = FakeEndpoint。

    R3 健康探测读面（零新 endpoint / 零新 provider，LLM 约束）：探测 = 本端点 digest
    调用同通道的成功/失败统计——`consec_fails`（连败计数，property）+ `endpoint_degraded`
    （端点持续劣化判定，property：连败 ≥ ESCAPE_STREAK_THRESHOLD，引擎/蒸馏器逃生门
    据此放行 P0 pending 兜底逐出）+ `cooldown_status()`（一行可观测：连败/降级/冷却窗/阶梯）。
    """

    COMPRESSION_TASK = "auxiliary.compression"   # digest 映射原生 summarizer 任务
    # R3 逃生门连败阈值：端点连续失败达此数才判「持续劣化」（单一事件瞬态 →
    # 既有 transient_skip 自愈口径不变；2 次失败可能是抖动，3 次起才降级）。
    ESCAPE_STREAK_THRESHOLD: int = 3

    def __init__(self, main_endpoint, max_concurrency: int = 1, fail_cooldown: float = 60.0,
                 digest_sem: "threading.BoundedSemaphore | None" = None,
                 async_digest: bool = True,
                 cooldown_backoff: "tuple[float, ...] | None" = None) -> None:
        self.main_endpoint = main_endpoint
        self.fallback_endpoint = main_endpoint   # (A) 失败域与主模型同源
        self.fallback_chain: list[str] = ["main"]
        self.max_concurrency = max_concurrency    # 单槽保护（shipped default 1）
        self._in_turn = False
        self._queue: list[Segment] = []
        self._queued: set[str] = set()
        self._results: dict[str, dict] = {}       # segment_id -> digest 结果（会话内幂等）
        self._digest_cache: dict[tuple, str] = {}  # 再生三元组 -> refid（跨 turn 幂等）
        self.transient = False                    # 本事件主 LLM 瞬态失败标志
        # 失败冷却窗（去抖）：端点一次瞬态失败（超时/限流）后，冷却窗内 pending 段
        # 直接跳过、不再逐个重发端点 —— 多会话 × pending 段 × 重试的放大风暴封死；
        # 跨 turn 有效（turn 边界只重置 queue/transient，冷却到点才放行下一波）。
        self.fail_cooldown = fail_cooldown
        self._cooldown_until: float = 0.0
        # R3 健康探测/降级口径：不加新 endpoint / 新 provider（LLM 约束），探测 = 现有
        # digest 调用（本端点同通道）的成功/失败统计。连败计数 _consec_fails + 退避
        # 阶梯 cooldown_backoff（默认 = fail_cooldown × COOLDOWN_BACKOFF_MULTIPLIERS
        # 派生：60s 配置 → 60s→300s→1800s 契约曲线，第 1 连败 = 1× 基线（legacy 行为
        # 完全一致），2/3 连败起 5×/30× 拉长封顶；fail_cooldown=0 → 阶梯全 0 → 冷却窗
        # 恒关（sc 契约测试「transient 单事件自愈」0 行为变化）；可显式注入绝对阶梯）。
        # 1 次成功 → 计数归零、冷却窗关闭回落基线。状态变化打一行 INFO（可观测）。
        self.cooldown_backoff: tuple = tuple(cooldown_backoff) if cooldown_backoff is not None \
            else tuple(fail_cooldown * m for m in COOLDOWN_BACKOFF_MULTIPLIERS)
        self._consec_fails: int = 0
        # digest 全局并发帽（去抖第 2 层）：默认共享模块级 _DIGEST_SEM（跨会话同端点
        # 不互相放大）；测试可注入独立实例。
        self.digest_sem = digest_sem if digest_sem is not None else _DIGEST_SEM
        # 异步 digest worker：有界 work queue + 单 daemon 线程。turn 末批次/出 turn 单发
        # 只入队立即返回（宿主 turn 线程零阻塞，慢端点不再卡 preflight）；冷却窗在
        # worker 内检查（冷却窗内跳过端点，非入队后逐段重试）。digest 非关键路径——
        # 未 ready 的后果仅是本事件不逐出（over-store 可接受；force_evict 门不过 digest）。
        # sync 模式（测试/sc 契约确定性）= 无 worker，原同步路径不变。
        self.async_digest = async_digest
        self._work_q: "queue.Queue" = queue.Queue(maxsize=256)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._dropped: int = 0               # 队满丢弃计数（可观测；段保持 pending → 下事件再生）
        if self.async_digest:
            self._spawn_worker()

    # ── 异步 worker 生命周期 ──
    def _spawn_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._worker_loop, name="sc-digest", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._work_q.get()
            try:
                if isinstance(item, _FlushSignal):
                    item.event.set()
                    continue
                self._generate(item)
            except Exception:
                pass   # worker 线程永不炸（端点异常已在 _generate 内吞成 transient）
            finally:
                self._work_q.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """等待 worker 处理完当前队列（session-end 汇合点；测试确定性屏障）。
        返回 True = 队列已清空；False = 超时仍有在途（段保持 pending → 下 turn 再生）。"""
        if not self.async_digest or self._worker is None:
            return True
        sig_event = threading.Event()
        try:
            self._work_q.put(_FlushSignal(sig_event))
        except queue.Full:
            return False
        return sig_event.wait(timeout)

    @property
    def in_turn(self) -> bool:
        return self._in_turn

    # ── R3 健康探测读面（零新 endpoint / 零新 provider，LLM 约束）──
    # 探测 = 本端点 digest 调用同通道的成功/失败统计，供引擎/蒸馏器逃生门决策：
    # `consec_fails`（连败计数）+ `endpoint_degraded`（连败 ≥ 阈值 → 持续劣化判定，
    # 逃生门据此放行 P0 pending 兜底逐出；单一瞬态事件 consec=1 < 阈值 → 不降级，
    # 既有 transient_skip 单事件自愈口径 0 行为变化）+ `cooldown_status`（一行可观测）。
    @property
    def consec_fails(self) -> int:
        """真实端点调用连败计数（冷却窗内跳段 ≠ 端点失败，不累加——退避不自我放大）。"""
        return self._consec_fails

    @property
    def endpoint_degraded(self) -> bool:
        """端点持续劣化判定：连败 ≥ ESCAPE_STREAK_THRESHOLD（3）。单一/两次瞬态抖动
        不触发（保持 sc 契约「transient 单事件自愈」）；3 连败起才判降级放行逃生门。"""
        return self._consec_fails >= self.ESCAPE_STREAK_THRESHOLD

    def cooldown_status(self) -> dict:
        """R3 观测面（一行可 grep）：连败 / 是否降级 / 冷却窗剩余 / 退避阶梯快照。"""
        remaining = max(0.0, self._cooldown_until - time.monotonic())
        return {
            "consec_fails": self._consec_fails,
            "degraded": self.endpoint_degraded,
            "cooldown_remaining_s": round(remaining, 3),
            "ladder_s": [f"{c:g}" for c in self.cooldown_backoff],
        }

    # ── turn 边界：digest 生成只发生在 turn 末（不与主模型解码撞车）──
    def start_turn(self) -> None:
        self._in_turn = True
        self._queue = []
        self._queued = set()
        self.transient = False

    def end_turn(self) -> None:
        self._in_turn = False
        queued, self._queue = self._queue, []
        if self.async_digest:
            # 异步：批次入队立即返回（宿主 turn 线程零阻塞，慢端点不再卡 preflight）；
            # 实际生成在 worker 线程（冷却窗/并发帽均在 _generate 内生效）。
            for seg in queued:
                self._enqueue(seg)
        else:
            for seg in queued:
                self._generate(seg)

    def _enqueue(self, seg: Segment) -> None:
        """入队；队满（256）丢弃——段保持 pending → distill 本事件跳过该段（over-store
        可接受；force_evict 兜底），绝不为消化溢出反堵调用线程。丢弃计数可观测。"""
        try:
            self._work_q.put_nowait(seg)
        except queue.Full:
            self._dropped += 1

    def ensure_digest(self, seg: Segment) -> dict | None:
        """幂等：已生成 → 返回缓存（不重算、不发起端点调用）；turn 内 → 排队 None；
        出 turn → async 入队立即返回 None（worker 后台生成），sync 立即生成。"""
        if seg.segment_id in self._results:
            return self._results[seg.segment_id]
        if self._in_turn:
            if seg.segment_id not in self._queued:
                self._queue.append(seg)
                self._queued.add(seg.segment_id)
            return None
        if self.async_digest:
            self._enqueue(seg)
            return None
        return self._generate(seg)

    def _generate(self, seg: Segment) -> dict | None:
        if self._cooldown_until and time.monotonic() < self._cooldown_until:
            # 冷却窗内：不调端点（瞬态失败风暴去抖）；transient 标志保持 → 事件走 transient regime。
            # R3：跳段 ≠ 端点失败——连败计数不动（退避不自我放大，计数只统计真实端点调用失败）。
            self.transient = True
            return None
        try:
            self.digest_sem.acquire()
            try:
                response = self.main_endpoint.call(
                    self.COMPRESSION_TASK, DIGEST_INSTRUCTION + seg.original_text[:4096])
            finally:
                self.digest_sem.release()
        except Exception:
            self.transient = True
            # R3 连败退避：计数 +1 → 冷却窗按阶梯 effective_cooldown(连败数) 自适应拉长
            #（默认 60→300→1800s 封顶；cooldown_backoff 可注入）。fail_cooldown 保留
            # 原「冷却总开关」语义（>0 才落冷却窗；=0 → 计数照常但窗口恒 0，sc 契约
            # 测试「transient 单事件自愈」口径不变，零行为变化）。退避跳段（冷却窗内）
            # 不累加计数（不自我放大）。状态变化打一行 INFO（端点劣化观测面）。
            self._consec_fails += 1
            if self.fail_cooldown > 0:
                cooldown = effective_cooldown(self, self._consec_fails)
                if cooldown > 0:
                    self._cooldown_until = time.monotonic() + cooldown
                    logger.info(
                        "sc-aux: endpoint fail streak=%d cooldown=%.0fs "
                        "(backoff ladder=%s)",
                        self._consec_fails, cooldown,
                        ",".join(f"{c:g}s" for c in self.cooldown_backoff))
            return None
        # R3 成功复位：1 次成功 → 连败计数归零、冷却窗关闭回落基线（端点恢复观测面）。
        if self._consec_fails or self._cooldown_until:
            logger.info("sc-aux: endpoint recovered (fail streak %d -> 0, cooldown cleared)",
                        self._consec_fails)
            self._consec_fails = 0
            self._cooldown_until = 0.0
        key = scdata.digest_regen_key(
            {"content_hash": seg.content_hash}, seg.model_version, seg.session_topic)
        refid = self._digest_cache.get(key)
        if refid is None:
            refid = scdata.sha256_nfc("digest|" + "/".join(key))[:16]
            self._digest_cache[key] = refid
        seg.digest_ref = "ready:%s" % refid
        # S2a/修 S5：捕获 LLM 正文并解析 key_fact/summary（零新增调用点——之前正文被丢弃、
        # digest_text 只填 head80 占位）。空响应 / 无法解析 → 回退 head80 占位（bench 空响应
        # 不变量保持）；transient（call 抛异常）走上方 except，digest_text 保持 ""（[no digest]
        # marker 路径，与本路径分离）。
        key_fact, summary = parse_digest_response(response)
        seg.key_fact = key_fact
        seg.digest_text = summary or seg.original_text[:80]
        out = {"digest_ref_id": refid, "content_hash": seg.content_hash}
        self._results[seg.segment_id] = out
        return out
