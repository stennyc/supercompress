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
import os
import queue
import re
import threading
import time

import scdata

from .schema import Segment

# digest 提示词（S2a/修 S5：之前 LLM 响应正文被丢弃、digest_text 只填 head80 占位——
# 同一处 LLM 调用、零新增调用次数，现在捕获正文并结构化解析出 key_fact/summary）。
DIGEST_INSTRUCTION = (
    "Compress the conversation segment below into strict JSON (no code fence, no extra text):\n"
    '{"key_fact": "<=60 chars: the single most important fact/decision", '
    '"summary": "<=200 chars: 2-4 sentence digest"}\n'
    "--- segment ---\n"
)


def parse_digest_response(text: str | None) -> tuple[str, str]:
    """digest LLM 响应 → (key_fact, summary)。容忍裸 JSON / code fence / 纯文本 / None；
    空响应 → ("", "")（调用方回退 head 摘录占位，bench 空响应不变量保持）。"""
    if not text or not str(text).strip():
        return "", ""
    t = str(text).strip()
    candidates = [t]
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        candidates.insert(0, m.group(1).strip())
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        candidates.append(m.group(0))
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
    """(A) aux ≡ 主模型端点。endpoint 注入：生产 = Hermes 原生网关，测试 = FakeEndpoint。"""

    COMPRESSION_TASK = "auxiliary.compression"   # digest 映射原生 summarizer 任务

    def __init__(self, main_endpoint, max_concurrency: int = 1, fail_cooldown: float = 60.0,
                 digest_sem: "threading.BoundedSemaphore | None" = None,
                 async_digest: bool = True) -> None:
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
            # 冷却窗内：不调端点（瞬态失败风暴去抖）；transient 标志保持 → 事件走 transient regime
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
            if self.fail_cooldown > 0:
                self._cooldown_until = time.monotonic() + self.fail_cooldown
            return None
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
