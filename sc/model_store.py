"""Model 段独立旁路 store `<sid>.model.jsonl`（B 卡，用户裁定 2026-10-06「折中 C + 独立 jsonl」）。

主 store 字节契约零改动：model/assistant 段进旁路 `<sid>.model.jsonl`（与 digest 池
`<sid>.digests.json` 同住同 sid 前缀，legacy active.jsonl → active.model.jsonl）。
旁路 = 不可信信任边界：fail-open + 保险丝（构造/replay 对缺失/损坏 fail-open 空池；
保险丝三选一命中 → 内存 reset（保留文件，不删）+ fused 标志（召回并集跳过该 store）+
日志 marker `model-store sidecar fused (reason=…)`）。

入库 gate（确定性零 LLM，宿主 on_turn_complete 面，sc 层契约面）：
- role=='assistant'：text 空 strip → skip（tool-call-only）；nfc_hash 在 model store 或
  主 store 命中 → dedup skip（同文双写封死）；role=='tool' 永不入库（调用方 gate，
  本入口只收 assistant text）；
- 持久化 = len(text) ≥ model_min_chars（默认 200，宿主可配）OR 命中有意义 marker
  （复用 sc/tagger 的 _P1/_P2 marker 族同口径子串判定；压缩摘要类无论长度恒持久化）。

Tiering = 先持久化再异步精化：gate 命中段先以**临时档 P0 非结构**
（structural_p0=False，可被 push cap 逐，P0/P1 注入门放行）落盘 + 进有界队列
（maxsize 256，满即丢 + _dropped 计数，仿 digest worker 单槽 daemon 线程 +
模块级并发帽 2）；worker 用宿主接线的 sc/tagger classify(text, "model") LLM 分支
（宿主共享 call_llm，aux 两级路由 + fail cooldown 60→300→1800s + 连败 ≥3 degraded
全复用既有 sc/aux_provider 语义，本模块零新 provider / 零模型名 / 零 provider 块）。
replay 去重保首行（scdata 锁定不变量 "never re-tier on replay"）→ **磁盘 tier 永远 =
临时档 P0；精化仅内存**（Segment.tier 字段 in-memory update），冷重启重放 = 全 P0
（保守地板，已文档化取舍）；aux 失败/degraded/队列丢弃 → 保持 P0 地板。
model_version stamp = 当前模型（digest 再生三元组用；空 stamp 行 = 版本判定中性）。

本文件为 B 卡新文件，LF 行结束（宿主 CRLF 文件 engine.py/sc/store.py 按文件保留，
禁整文件 universal-newline 重写；本模块对宿主零改动、对 sc/store.py 原样复用零改动）。
"""
from __future__ import annotations

import logging
import os
import queue
import threading

import scdata

from .schema import Segment, nfc_hash
from .store import JsonlStore

logger = logging.getLogger(__name__)

# ── 旁路常量（单一事实源：改名字/上限只改这一处，G2）────────────────────
MODEL_MAX_BYTES = 5 * 1024 * 1024     # 保险丝②：model 文件 > 5MB → 不可信
MODEL_LINE_RATIO = 2                 # 保险丝①：model 行数 > 2× 主 store 行数 → 不可信
MODEL_TIER_QUEUE_MAX = 256           # tier 有界队列：满即丢（段保持 P0 地板，丢弃上界 = 一批）
MODEL_TIER_CONCURRENCY = 2           # 模块级 tier 并发帽（跨实例共享，仿 aux_provider._DIGEST_SEM）

# 跨实例 tier 并发帽（导入时定死；测试可经 ModelStore(sem=…) 注入独立实例）。
_MODEL_TIER_SEM = threading.BoundedSemaphore(MODEL_TIER_CONCURRENCY)


def model_path_for_jsonl(jsonl_path: str) -> str:
    """model 旁路路径 = 主 store 路径 splitext + '.model.jsonl'（单源，零硬编码）。

    `<sid>.jsonl` → `<sid>.model.jsonl`；legacy `active.jsonl` → `active.model.jsonl`。
    与 digest 池同目录同 sid 前缀（GC 联动语义自洽）。"""
    return os.path.splitext(jsonl_path)[0] + ".model.jsonl"


def model_marker_hit(text: str) -> bool:
    """有意义 marker 命中（复用 sc/tagger 的 _P1/_P2 marker 正则族同口径子串判定）：
    压缩摘要 / 系统噪声类**无论长度恒持久化**（gate 的 OR 支；B 卡决策 3）。"""
    from .tagger import _P1_MARKERS, _P2_MARKERS
    t = (text or "")
    return any(m in t for m in _P1_MARKERS) or any(m in t for m in _P2_MARKERS)


class _FlushSignal:
    """flush(timeout) 汇合点信号：worker 处理到即 set event（不中断队列；仿 aux_provider 同形）。"""
    __slots__ = ("event",)

    def __init__(self, event: threading.Event) -> None:
        self.event = event


def _empty_jsonl_store(path: str) -> JsonlStore:
    """fail-open 空池：按 __init__ 同形构造 JsonlStore 实例态（绕过 replay 路径；
    损坏文件字节保留不动——不可信旁路层不自动删）。sc/store.py 原样复用零改动。"""
    st = JsonlStore.__new__(JsonlStore)
    st.path = path
    st.lazy_batch_cap = 256 * 1024
    st.by_hash = {}
    st.index = {}
    st.durable = set()
    st._pending = []
    st._pending_bytes = 0
    st._line_count = 0
    st._next_seq = 0
    st.fsynced_p0_count = 0
    st.kpi_log = []
    st._write_fail = False
    return st


class ModelStore:
    """Model 旁路 store：JsonlStore（原样复用）+ fail-open 构造 + 保险丝 + 入库 gate + 有界 tier 队列。

    fail-open（旁路 = 不可信信任边界，零崩溃）：
    - 缺失 → 空池（合法态，不 fuse）；
    - 损坏 JSON 行 → scdata.replay torn-tail 静默丢弃 → 空池（数据面重放契约，不 fuse）；
    - 不可读（OSError / 非 UTF-8 → ValueError 族）→ 空池 + fuse（保留文件）+ marker。

    保险丝（三选一，命中即 `fuse()`：内存 reset 保留文件 + fused 标志 + marker）：
    ① model 行数 > 2× 主 store 行数；② 文件 > 5MB；③ 当前 model 已知（非空）且任一
    replayed 行 model_version 非空且 ≠ 当前 model（空 stamp 行 = 中性，沿用 digest
    pool guard 判定口径）。

    磁盘 tier 永远 P0（"never re-tier on replay"，scdata 锁定）：精化仅更新内存 tier
    字段；aux 失败/degraded/队列丢弃 → P0 地板（保守，永不当 P2 丢——同 tagger 契约）。
    """

    def __init__(self, path: str, *, tier_fn=None, sync: bool = False, sem=None) -> None:
        self.path = path
        self._tier_fn = tier_fn
        self.sync = sync
        self._sem = sem if sem is not None else _MODEL_TIER_SEM
        self._fused = False
        self.fuse_reasons: list[str] = []
        self._dropped: int = 0
        self._q: "queue.Queue" = queue.Queue(maxsize=MODEL_TIER_QUEUE_MAX)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        try:
            self.store = JsonlStore(path)          # replay 数据面（scdata 权威单路径）
        except (OSError, ValueError) as e:
            # 不可读 sidecar（IO 故障 / 非 UTF-8）→ fail-open 空池 + fuse（保留文件，不删）
            self.store = _empty_jsonl_store(path)
            self._fused = True
            self.fuse_reasons.append(f"replay_fail_open: {e}")
            logger.warning("model-store sidecar fused (reason=replay_fail_open: %s)", e)
        if not sync:
            self._spawn_worker()

    # ── 路径 / 状态读面 ──
    @property
    def fused(self) -> bool:
        return self._fused

    @property
    def dropped(self) -> int:
        """有界队列满即丢计数（可观测；段保持 P0 地板 → 冷重启全 P0 重放，丢弃 = 已文档化上界）。"""
        return self._dropped

    @property
    def line_count(self) -> int:
        return self.store.line_count

    def get(self, segment_id: str):
        """点查 O(1)（透传 JsonlStore.index；L0 指针面由壳 `point_fetch` 解析序 main→model）。"""
        return self.store.get(segment_id)

    def live_segments(self) -> list[Segment]:
        """内存事实源（replay + 本进程 ingest + 内存精化后 tier）= 召回并集 / T4 注入集物料。
        fused 态 = []（内存已 reset；召回并集跳过该 store）。"""
        return list(self.store.by_hash.values())

    # ── 保险丝（薄壳判+作废一体，算法面收口本模块；mirror digest pool guard()）──
    def check_fuse(self, main_line_count: int, current_model: str | None) -> list[str]:
        """保险丝三选一检查 + 命中即 fuse（mirror `DigestPool.guard()` 判+作废一体）。

        命中面（B 卡决策 2）：① 行数比 > 2×；② 文件 > 5MB；③ model 版本失配
        （当前 model 已知非空 + 任一 replayed 行 stamp 非空且 ≠ 当前；空 stamp 中性）。
        纯检查：未命中返 []；命中已 fuse（幂等）返既有 fuse_reasons。"""
        if self._fused:
            return self.fuse_reasons
        reasons: list[str] = []
        mc = max(0, int(main_line_count))
        # fuse① 行数比 = model > 2×main（锁定语义）：主 store 空（main=0）而 model 有行 =
        # 旁路孤儿/主 store 被篡改的异常态，保险丝**应当**命中（fail-safe 优先），故不加
        # 主 store 非空前置——测试需按真实会话喂主 store user 段，而非放宽保险丝。
        if self.store.line_count > MODEL_LINE_RATIO * mc:
            reasons.append(
                f"line_count_over_2x_main: {self.store.line_count} > {MODEL_LINE_RATIO}*{mc}")
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0
        if size > MODEL_MAX_BYTES:
            reasons.append(f"size_over_cap: {size} > {MODEL_MAX_BYTES}")
        if current_model:
            bad = sorted({s.model_version for s in self.store.by_hash.values()
                         if s.model_version and s.model_version != current_model})
            if bad:
                reasons.append(
                    f"model_version_mismatch: stamped={bad} current={current_model}")
        if reasons:
            self.fuse("; ".join(reasons))
        return reasons

    def fuse(self, reason: str) -> None:
        """保险丝命中：内存 reset（**保留文件，不删**——不可信旁路层不自动删）+ fused 标志
        （召回并集跳过该 store）+ 日志 marker。幂等（已 fused 再调 = no-op）。"""
        if self._fused:
            return
        self._fused = True
        self.fuse_reasons.append(reason)
        st = self.store
        st.by_hash, st.index, st.durable = {}, {}, set()
        st._pending, st._pending_bytes = [], 0
        st._line_count = 0
        st._next_seq = 0
        st.fsynced_p0_count = 0
        st.kpi_log = []
        logger.warning("model-store sidecar fused (reason=%s)", reason)

    # ── 入库 gate（确定性零 LLM；B 卡决策 3）──
    def ingest_assistant(self, text: str, *, min_chars: int, model_version: str,
                         main_by_hash: dict | None) -> tuple[bool, str]:
        """gate（调用方只喂 role=='assistant' text —— role=='tool' 永不入库）：
        - text 空 strip → skip（tool-call-only）；
        - nfc_hash 在 model store.by_hash 或主 store.by_hash → dedup skip（同文双写封死）；
        - 持久化 = len(text) ≥ min_chars OR 有意义 marker 命中（压缩摘要类无论长度恒持久化）。
        命中 → 临时档 P0 非结构（structural_p0=False，可被 push cap 逐，P0/P1 注入门放行）
        P0 即写 + fsync + 读回 + tier 队列（先持久化再精化）。model_version stamp = 当前模型
        （digest 再生三元组用）。返回 (ingested, gate_reason)；写失败抛 StoreError（壳 fail-safe）。"""
        if self._fused:
            return (False, "fused")
        text = str(text or "")
        if not text.strip():
            return (False, "blank")
        h = nfc_hash(text)
        if h in self.store.by_hash or h in (main_by_hash or {}):
            return (False, "dedup")
        if len(text) >= min_chars:
            gate = "chars"
        elif model_marker_hit(text):
            gate = "marker"
        else:
            return (False, "below_min_no_marker")
        seg = Segment(tier="P0", provenance="model", original_text=text,
                      model_version=str(model_version or ""), structural_p0=False)
        self.store.append(seg)          # 临时档 P0 = 即写 + fsync + 读回（磁盘 tier 永远 P0）
        self.queue_tier(seg)
        return (True, gate)

    # ── tier 队列（先持久化再异步精化；单槽 worker + 模块级并发帽）──
    def queue_tier(self, seg: Segment) -> None:
        """tier 队列入队（有界 MODEL_TIER_QUEUE_MAX=256；满即丢 + _dropped 计数——仿
        digest worker：段保持 P0 地板，丢弃上界 = 一批（已文档化取舍））。sync 面
        （sc 契约测试确定性）无 worker 线程，显式 drain() 消化。"""
        try:
            self._q.put_nowait(seg)
        except queue.Full:
            self._dropped += 1
            return
        if not self.sync:
            self._spawn_worker()

    def _spawn_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._worker_loop,
                                             name="sc-model-tier", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if isinstance(item, _FlushSignal):
                    item.event.set()
                    continue
                self._tier_one(item)
                self.drain()            # 队列余量同批聚合（单线程序，每批一行 marker）
            except Exception:
                pass                   # worker 线程永不炸（精化异常 = P0 地板）
            finally:
                self._q.task_done()

    def _tier_one(self, seg: Segment) -> bool:
        """单段 LLM 精化（模块级并发帽 2 跨实例共享，仿 aux_provider._DIGEST_SEM 用法）。

        tier_fn = 宿主接线（sc/tagger classify(text, "model") LLM 分支；aux 两级路由 +
        fail cooldown + degraded 全在 sc/aux_provider 语义内）：True = 精化已应用
        （aux_ok，内存 tier 字段更新）；False = 保持 P0 地板（aux_floor）。
        精化异常 → fail-open P0 地板（保守，永不当 P2 丢）。**磁盘行永远 P0**
        （"never re-tier on replay"——只更内存 tier 字段）。"""
        if self._tier_fn is None:
            return False
        try:
            self._sem.acquire()
            try:
                return bool(self._tier_fn(seg))
            finally:
                self._sem.release()
        except Exception:
            return False

    def drain(self) -> dict:
        """消化当前队列 + 聚合统计（单线程序；每批一行 marker）：
        `model-store: tiered N (aux_ok=… aux_floor=…)`（N=0 静默；_dropped 不计入本行）。
        sync 面 = 测试/sc 契约显式调用点；async 面 = worker 汇合点
        （engine 层 _model_session_flush 在会话边界调 flush(timeout=2.0) 确定性汇合；
        超时/在途段保持 P0 地板、下事件再生——已文档化取舍）。"""
        n = ok = floor = 0
        flushed: list[_FlushSignal] = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _FlushSignal):
                flushed.append(item)
                break
            n += 1
            if self._tier_one(item):
                ok += 1
            else:
                floor += 1
        if n:
            logger.info("model-store: tiered %d (aux_ok=%d aux_floor=%d)", n, ok, floor)
        for sig in flushed:
            sig.event.set()
        return {"n": n, "aux_ok": ok, "aux_floor": floor}

    def flush(self, timeout: float | None = None) -> bool:
        """worker 汇合点（会话边界；仿 AuxProvider.flush）：队列清空至本信号才 set。
        返回 True = 队列已清空；False = 超时/队满仍有在途（段保持 P0 地板 → 下事件再生）。
        sync 面 = 立即 drain，返 True。"""
        if self.sync:
            self.drain()
            return True
        if self._worker is None:
            return True
        sig_event = threading.Event()
        try:
            self._q.put_nowait(_FlushSignal(sig_event))
        except queue.Full:
            return False
        return sig_event.wait(timeout)

    # ── 生命周期 flush 点（engine 会话边界镜像调用）──
    def flush_lazy(self) -> int:
        """P1/P2 lazy 批 flush（会话边界原子点，镜像主 store engine flush 块）。

        磁盘 tier 永远 P0 不变量 → lazy 批实际恒空（本调用 = 生命周期语义镜像 +
        未来若 P1/P2 落盘即自动生效）；OSError/StoreError 由调用方优雅降级
        （flush marker + store=model 判别）。"""
        return self.store.flush()

    def log_water_warn(self, action: str, exc: BaseException) -> None:
        """写盘失败水位 WARNING 观测面（委托 JsonlStore；自身 fail-safe 绝不反向上抛）。"""
        self.store.log_water_warn(action, exc)

    # ── 跨期重开交接（carry 分支；与主 store JsonlStore.materialize 平行）──
    @staticmethod
    def materialize(archive_path: str, new_path: str, *, reopen_from: str | None = None,
                    tier_fn=None, sync: bool = False, sem=None) -> "ModelStore":
        """跨期重开：归档 model 文件 → 新 session model 文件（scdata.materialize 字节稳定
        拷贝 + P1 stale 标记，与主 store JsonlStore.materialize 同语义）。

        归档缺失/不可读 → fail-open 全新空池（零崩溃；主 store 分支的 warning 语义不变，
        model 分支本行留痕）。"""
        try:
            scdata.materialize(archive_path, new_path)
        except (OSError, ValueError) as e:
            logger.warning("model-store: carry materialize fail-open (fresh empty sidecar): %s", e)
        ms = ModelStore(new_path, tier_fn=tier_fn, sync=sync, sem=sem)
        if reopen_from:
            for seg in ms.live_segments():
                seg.reopened_from = reopen_from
                if seg.tier == "P1":
                    seg.stale = True
        return ms
