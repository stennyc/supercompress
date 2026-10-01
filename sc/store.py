"""单一事实源 jsonl 存储（session 级，filememory 式）。

数据面委托单一权威实现 scdata（@reviewer 单一实现门 + @data-engineer 权威层）：
行字节 = scdata.to_line、重放 = scdata.replay（torn-tail 单 replay 路径）、
物化 = scdata.materialize（字节稳定拷贝）、锚点闭合 = scdata.verify_anchor_closure
（全树恰好 1 个 live 定义，本模块只做委托，不另造第二套）。
写路径 = 本模块 own 的 tier-aware fsync 策略：
- P0（含全部 ledger / 顶层 goal）：写入即强制 fsync + 读回校验（write-before-compress 硬 gate 的一半）；
- P1/P2：lazy 批（size trigger ≤256KB 先到即 flush；不逐行 fsync）；
- 锚点只发射「已 durable（fsync+读回）」的段——pending 字节物理上无指针。
崩溃语义：断电 = 丢一个 P1/P2 pending 批（≤ lazy cap），P0 + ledger 全 durable；
重开 = materialize(archive, active, reopen_from) 字节稳定拷贝 + 单 replay 路径。
"""
from __future__ import annotations

import logging
import os

import scdata

from .schema import Segment

logger = logging.getLogger(__name__)

# R2：磁盘水位观测阈值（写盘前例行检查用；≥ 此值打一行 INFO 供 ops 监控 grep）。
# 纯观测、零自动清理（#5 retention 是独立任务，本行不触发任何清理动作）。
DISK_WATER_HIGH_PCT = 0.90


class StoreError(Exception):
    pass


class JsonlStore:
    def __init__(self, path: str, lazy_batch_cap: int = 256 * 1024):
        self.path = path
        self.lazy_batch_cap = lazy_batch_cap
        self.by_hash: dict[str, Segment] = {}      # content_hash -> Segment（replay 后 = 磁盘态）
        self.index: dict[str, Segment] = {}        # segment_id -> Segment（点查 O(1)）
        self.durable: set[str] = set()             # 已 fsync 确认的 content_hash
        self._pending: list[Segment] = []          # P1/P2 lazy 批（内存）
        self._pending_bytes = 0
        self._line_count = 0
        self._next_seq = 0
        self.fsynced_p0_count = 0                  # 预检/断言用
        self.kpi_log: list[dict] = []              # KPI #1 全量日志挂点
        self._write_fail = False                   # fail-safe 注入（测试）

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self.replay()

    # ───────────── 磁盘水位观测（R2：full-disk 优雅降级配套，一行 ops 监控面）─────────────
    def disk_water_level(self) -> tuple[int, float]:
        """store 文件所在文件系统水位 → (剩余字节, 已用百分比 0.0-1.0)。

        statvfs 自身失败（只读挂载 / 盘消失 / 容器异常）→ (0, -1.0)：
        调用方按「水位未知」处理，观测面绝不反向上抛（不能成为新的宿主炸点）。"""
        try:
            st = os.statvfs(os.path.dirname(os.path.abspath(self.path)) or "/")
            free_b = st.f_bavail * st.f_frsize
            total_b = st.f_blocks * st.f_frsize
            pct = (1.0 - st.f_bavail / st.f_blocks) if st.f_blocks else 0.0
            return int(free_b), float(pct)
        except OSError:
            return 0, -1.0

    def log_water_check(self, action: str) -> None:
        """写盘前例行水位观测：used ≥ DISK_WATER_HIGH_PCT 才打一行 INFO（剩余空间 + 触发动作）；
        健康态（< 阈值 / 水位未知）静默——零日志污染。自身 fail-safe：绝不抛出。"""
        try:
            free_b, pct = self.disk_water_level()
            if pct >= DISK_WATER_HIGH_PCT:
                logger.info("tcm-supercompress: disk water-level HIGH: free=%.1fGiB used=%.1f%% "
                            "action=%s", free_b / (1024 ** 3), pct * 100.0, action)
        except Exception:
            pass   # 观测面永不炸宿主

    def log_water_warn(self, action: str, exc: BaseException) -> None:
        """写盘失败时水位 WARNING（恒打一行：剩余空间 + 触发动作 + 错误），供 ops 监控定位
        full-disk 事故（tui_gateway EIO/EIO-unhandledRejection 同因面）。自身 fail-safe。"""
        try:
            free_b, pct = self.disk_water_level()
            used_txt = f"{pct * 100.0:.1f}%" if pct >= 0 else "unknown"
            logger.warning("tcm-supercompress: disk water-level: free=%.1fGiB used=%s "
                          "action=%s failed: %s", free_b / (1024 ** 3), used_txt, action, exc)
        except Exception:
            pass

    # ───────────── 写路径（tier-aware fsync）─────────────
    def append(self, seg: Segment) -> Segment:
        """P0 即写即 fsync + 读回；P1/P2 进 lazy 批。写失败 → StoreError（调用方 fail-safe 回滚）。

        R2：底层 OSError（full-disk ENOSPC / EIO）在此归一为 StoreError("io_error: ...")
        统一语义——调用方（distill gate① / engine 壳）只需 catch StoreError 一族，
        裸 OSError 不再逃逸到宿主。"""
        if self._write_fail:
            raise StoreError("source_unavailable")
        seg.seq = self._next_seq
        self._next_seq += 1
        line = scdata.to_line(seg.to_dict())        # 行字节单点产出：scdata 权威
        if seg.tier == "P0":
            self.log_water_check("p0-append")       # 写盘前例行水位观测（≥90% 才出 INFO）
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                self.fsynced_p0_count += 1
                # 读回校验（write-before-compress gate 的源存储侧半）：单一解析 = Segment.from_line
                with open(self.path, "r", encoding="utf-8") as f:
                    last = [l for l in f.readlines() if l.strip()][-1].strip()
            except OSError as e:
                raise StoreError(f"io_error: {e}") from e
            if Segment.from_line(last) is None:
                raise StoreError("readback_verify_failed")
            self._record(seg, durable=True)
        else:
            self._pending.append(seg)
            self._pending_bytes += len(line.encode("utf-8")) + 1
            self._record(seg, durable=False)
        self._line_count += 1
        return seg

    def _record(self, seg: Segment, durable: bool) -> None:
        self.by_hash[seg.content_hash] = seg
        self.index[seg.segment_id] = seg
        if durable:
            self.durable.add(seg.content_hash)

    def flush(self) -> int:
        """P1/P2 lazy 批 flush + fsync（批间隔 = 断电丢失上界：一个批）。

        R2：写块 OSError 归一为 StoreError("io_error: ...")；失败时 **不清** _pending
        （批留在内存可重试；重放 by_hash 天然去重；断电丢批上界语义不变 = 一个批）。"""
        if self._write_fail:
            raise StoreError("source_unavailable")
        n = 0
        if self._pending:
            self.log_water_check("flush")
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    for seg in self._pending:
                        f.write(scdata.to_line(seg.to_dict()) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                raise StoreError(f"io_error: {e}") from e
            for seg in self._pending:
                self._record(seg, durable=True)
            n = len(self._pending)
            self._pending.clear()
            self._pending_bytes = 0
        return n

    # ───────────── 状态查询 ─────────────
    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_p1p2_bytes(self) -> int:
        return self._pending_bytes

    @property
    def line_count(self) -> int:
        return self._line_count

    def get(self, segment_id: str) -> Segment | None:
        """点查 O(1)：segment_id（行级唯一身份）→ Segment。"""
        return self.index.get(segment_id)

    def pending_digest_segments(self, tier_order: tuple = ("P0", "P1", "P2")) -> list[Segment]:
        """resume 重入队：digest_ref=pending 的段按 P0>P1>P2 顺序补生成（幂等三元组）。"""
        order = {t: i for i, t in enumerate(tier_order)}
        segs = [s for s in self.by_hash.values() if s.digest_ref == "pending"]
        return sorted(segs, key=lambda s: (order[s.tier], s.seq))

    # ───────────── 崩溃 / 重开路径（数据面委托 scdata 单一实现）─────────────
    def replay(self) -> list[Segment]:
        """torn-tail 重放：权威数据面 = scdata.replay（逐行验 hash、torn 尾静默丢弃、
        从有效头部整体重建）；本模块只把已验 dict 重建为 Segment 并刷新内存索引。"""
        self.by_hash, self.index, self.durable = {}, {}, set()
        self._pending, self._pending_bytes = [], 0
        self._line_count, self.fsynced_p0_count = 0, 0
        idx = scdata.replay(self.path)              # 单一重放路径
        segs: list[Segment] = []
        for line in idx.by_hash.values():
            seg = Segment.from_dict(line)
            segs.append(seg)
            self._record(seg, durable=True)
        self._next_seq = max((s.seq for s in segs), default=-1) + 1
        # 重放后磁盘态行数 = 已验段数（_line_count 反映 on-disk 事实，非「本次进程 append 数」）
        self._line_count = len(segs)
        return segs

    @staticmethod
    def materialize(archive_path: str, new_active_path: str, reopen_from: str | None = None) -> "JsonlStore":
        """跨期重开：归档 → 新活跃文件（scdata.materialize 字节稳定拷贝，不重归一化/重编码）。

        重放走新活跃文件（单 replay 路径）；重开后按 reopened_from 打 stale 语义标记
        （P1 中间推理 → stale，召回降权、行动前 verify）。
        """
        scdata.materialize(archive_path, new_active_path)   # 权威物化：字节稳定
        store = JsonlStore(new_active_path)
        if reopen_from:
            for seg in store.by_hash.values():
                seg.reopened_from = reopen_from
                if seg.tier == "P1":
                    seg.stale = True
        return store

    def dangling_anchors(self, anchor_manifest) -> list[str]:
        """锚点闭合校验：委托 scdata.verify_anchor_closure（全树唯一 live 实现，不自造第二套）。

        anchor_manifest 每项带 content_hash（= replay 索引键）；「闭合输入非空」前置由调用侧
        断言（@reviewer 非平凡前置）+ 权威实现空 replay_index 语义 = raise 而非静默 []。
        """
        idx = scdata.replay(self.path)
        return scdata.verify_anchor_closure(
            [{"segment_id": m.content_hash} for m in anchor_manifest], idx)

    # ───────────── fail-safe 注入（测试）─────────────
    def arm_write_failure(self) -> None:
        self._write_fail = True

    def disarm_write_failure(self) -> None:
        self._write_fail = False
