"""P4 digest sidecar 水池——核心池逻辑（plan doc P4 节，用户 2026-10-05 拍板开工）。

背景：`digest_text`/`key_fact` 是 Segment 运行期字段，不落盘（schema 定稿「可经再生三元组重生成」）；
gateway 重启后 reopened 会话全量重蒸馏（实测 13:05 重启 ~50 次 aux 调用）。本池 = 旁路派生文件
`<hermes_home>/supercompress_store/<sid>.digests.json`，兜住蒸馏产物，源层 jsonl **零改动**
（append-only 字节稳定不变式不动）。

信任模型（plan P4 落点#6）：池 = 不可信派生层（LLM 生成文本）。池缺失 / 损坏 → 整池当不存在，
回落现状（pending + head-80），零崩溃。故所有读路径 fail-open（异常 / 结构不符一律回空池，绝不抛出）。

原子写（plan P4 落点#1）：tmp + os.replace。rename 在同卷内原子——断电 / 崩溃永不产出半写池文件。

卡片范围：
- 前卡 t_49148aa2 = 核心池逻辑：条目 schema + 原子写 + 读前查池 + LF 行结束 + 缺失/损坏零崩。
- 本卡 t_567e3b29 = GC 联动（会话归档/删除跟删池 + 孤儿清扫）+ 保险丝（池 >5MB 或条目数 > 段数
  → 作废重建）+ 版本失配整池作废（model_version 换代 → 全池重蒸）。
- 后继：engine.py 两钩子（replay 查池 + digest 完成回写）= t_026a17f8；
  pytest 用例 = t_b5bf5201；双解释器回归 + 宿主零 diff 验收 = t_ba65468c。

约束（全数适用）：宿主文件零改动；零硬编码路径（池路径走 HERMES_HOME 解析链，不出 /root、profile 名）；
本文件为新文件用 LF（与 sc/drift.py、sc/inject.py 同批新文件惯例一致）。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── 池文件常量（单一事实源；改名字/上限只改这一处，勿各写各的字面——G2 可维护性铁律）──
POOL_VERSION = 1                            # 池文件格式版本（前向兼容面；下卡保险丝判「版本不符→作废」）
POOL_MAX_BYTES = 5 * 1024 * 1024            # 保险丝：池 >5MB → 作废重建（本卡仅暴露常量，判定逻辑在下卡）
# 条目 schema 冻结五字段（plan P4 落点#1 逐字；读路径校验只认这五键，多余键静默丢弃）
POOL_ENTRY_FIELDS: tuple = (
    "content_hash", "model_version", "key_fact", "digest_text", "ts",
)


def pool_path_for_session(sid: str, hermes_home: Optional[str] = None) -> str:
    """池文件路径 = `<hermes_home>/supercompress_store/<sid>.digests.json`（G2 单源，零硬编码）。

    SUBDIR 名字面从 `sc._bootstrap` 引用（全树只在该处定义一次）。hermes_home 显式传（宿主会话钩子
    下发的 per-profile 数据根）优先；缺省走 `_bootstrap.store_dir()`（HERMES_HOME env 优先、缺省
    ~/.hermes——同 engine/store 数据面解析链）。与源层 `<sid>.jsonl` 同住，GC 联动时跟着删。
    """
    from ._bootstrap import STORE_SUBDIR
    if hermes_home:
        return os.path.join(hermes_home, STORE_SUBDIR, f"{sid}.digests.json")
    from ._bootstrap import store_dir
    return os.path.join(store_dir(), f"{sid}.digests.json")


def pool_path_for_jsonl(jsonl_path: str, hermes_home: Optional[str] = None) -> str:
    """池文件路径 = 与给定 session ``.jsonl`` 源层**同目录、同 sid 前缀**的派生池
    （``<sid>.jsonl`` → ``<sid>.digests.json``，spec 落点#1 的 GC 对应关系同目录实现）。

    与 ``pool_path_for_session`` 的分工：
    - ``pool_path_for_session(sid, hermes_home)``：生产 gateway（HERMES_HOME / host 注入
      hermes_home 数据根）→ ``<hermes_home>/supercompress_store/<sid>.digests.json``。
    - 本函数：engine 侧把池文件**锚定到 session 源层 jsonl 的实际所在目录**（宿主重绑后的
      ``store.path`` 的父目录）——池与源层永远同住（测试 ``external_store_path`` 临时目录、
      生产 ``hermes_home`` 目录两种落点都对），GC 联动（按同目录同 sid 前缀删）语义自洽。

    ``hermes_home`` 参数保留只为签名对称（引擎侧总是显式给出源层目录，无需再解析数据根）；
    零硬编码路径（sid 取自文件名，目录取自 jsonl 实际位置，不出 /root、profile 名）。
    """
    sid = os.path.basename(jsonl_path)
    if sid.endswith(".jsonl"):
        sid = sid[: -len(".jsonl")]
    root = os.path.dirname(os.path.abspath(jsonl_path)) or "."
    return os.path.join(root, f"{sid}.digests.json")


class DigestPool:
    """P4 池：JSON 派生层（可删可重建，非事实源）。

    文件格式（`<sid>.digests.json`）：
        {"version": 1, "entries": [{content_hash, model_version, key_fact, digest_text, ts}, ...]}
    读返回 `{content_hash: entry}`（同 content_hash 后写覆盖前写 = upsert 语义）。

    失败语义（池=缓存，绝不炸宿主 turn）：
    - 读缺失 / 读 IO 错 / JSON 解不开 / 顶层非 list/dict → `{}`（空池回落）；
    - 写 OSError（full-disk ENOSPC / EIO）→ 归一为 `PoolWriteError`（调用方 catch 一族即够）。
    """

    def __init__(self, path: str) -> None:
        self.path = path

    # ───────────── 读路径（fail-open：缺失/损坏一律空池，零崩溃）─────────────
    def load(self) -> dict:
        """读整池 → `{content_hash: entry}`。缺失 / 损坏 / 结构不符 → `{}`（池=缓存，回落现状）。

        缓存无关：每次调用重新读盘（replay 一次 `load()` 后按 dict 逐段查即可，单读全量）；
        本对象是长生命周期（engine 侧），自缓存会撞「append-on-complete」写路径的陈旧面。
        """
        try:
            with open(self.path, "r", encoding="utf-8", newline="") as f:
                raw = f.read()
        except FileNotFoundError:
            return {}                                   # 池从未写过（= 现状），合法态
        except OSError as e:
            logger.warning("digest-pool: read failed (%s) -> empty pool", e)
            return {}                                    # IO 故障（盘错/被删）当空池，零崩溃
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict:
        """池文本 → `{content_hash: entry}`。JSON 失败 / 结构不符 → `{}`（损坏回落，DoD2 零崩面）。"""
        if not raw.strip():
            return {}
        try:
            doc = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}                                    # 半写残留 / 人为损坏 → 整池当不存在
        # 兼容两种顶层形状：{version, entries:[...]}（本模块写的）与裸 [...]（手工造池/旧档）
        if isinstance(doc, dict):
            entries = doc.get("entries", [])
        elif isinstance(doc, list):
            entries = doc
        else:
            return {}
        by_hash: dict = {}
        for e in entries:
            if not isinstance(e, dict):
                continue                                  # 坏条目静默丢，不连累整池
            h = e.get("content_hash")
            if not h or not isinstance(h, str):
                continue                                  # 无 key = 无法索引 → 丢
            entry = {k: e.get(k) for k in POOL_ENTRY_FIELDS}   # 只认 schema 五字段，多余键丢
            by_hash[h] = entry
        return by_hash

    def get(self, content_hash: str, model_version: Optional[str] = None) -> dict | None:
        """点查 content_hash → entry。单查便捷入口（整段查走 `load()`）。

        传 `model_version` 且池内条目版本不匹配 → 返 None（整池作废重蒸的判定入口：池条目是可删
        缓存，模型换代后旧条目全部失效；「作废」的重建动作在下卡，本卡只给「版本不匹配=未命中」契约）。
        """
        entry = self.load().get(content_hash)
        if entry is None:
            return None
        if model_version is not None and entry.get("model_version") != model_version:
            return None
        return entry

    def has(self, content_hash: str, model_version: Optional[str] = None) -> bool:
        """池命中判定（engine 读路径钩子的布尔面：True → digest_ref=done、跳过蒸馏）。"""
        return self.get(content_hash, model_version) is not None

    # ───────────── 写路径（原子：tmp + os.replace；每个 digest 完成即写一次，不做批量）─────────────
    def upsert(self, content_hash: str, model_version: str, key_fact: str, digest_text: str,
               ts: Optional[float] = None) -> None:
        """追加/覆盖单条目（按 content_hash upsert）→ 原子写全池。

        语义（plan P4 落点#3）：即时、原子、逐条（不批量）。同 content_hash 已存在 → 覆盖
        （更新 model_version/ts/digest 文本）。`ts` 缺省 = 当前 epoch（与 Segment.ts 同型 float）。
        写 OSError（ENOSPC/EIO）→ `PoolWriteError`（调用方 catch 一族；池=缓存，写失败非致命）。
        """
        if ts is None:
            ts = time.time()
        by_hash = self.load()
        by_hash[content_hash] = {
            "content_hash": content_hash,
            "model_version": model_version,
            "key_fact": key_fact,
            "digest_text": digest_text,
            "ts": ts,
        }
        self._atomic_write(by_hash)

    def _atomic_write(self, by_hash: dict) -> None:
        """tmp + os.replace 原子写（同卷 rename 原子；崩溃永不产出半写池文件）。LF 行结束。

        步骤：mkstemp(同目录) → 写 JSON(indent=2, LF, 尾换行) → flush+fsync → os.replace 覆盖目标
        → 尽力 fsync 目录（rename 本身落盘需目录项 fsync；best-effort，失败仅告警不反向上抛）。
        任一步炸 → unlink tmp 清场再原样抛出（不留 .digests-*.tmp 垃圾）。
        """
        doc = {"version": POOL_VERSION, "entries": [by_hash[h] for h in by_hash]}
        payload = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
        target_dir = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".digests-", suffix=".tmp", dir=target_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:   # newline="\n"：LF 行结束（禁 CRLF）
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)                                      # 同卷原子 rename
            self._fsync_dir(target_dir)                                     # 尽力 fsync 目录项（best-effort）
        except OSError as e:
            self._safe_unlink(tmp)
            raise PoolWriteError(f"io_error: {e}") from e
        except BaseException:
            self._safe_unlink(tmp)
            raise

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    @staticmethod
    def _fsync_dir(dirpath: str) -> None:
        """fsync 目录（rename 持久化需目录项落盘）；平台不支持 / 只读挂载 → 静默（best-effort）。"""
        try:
            dfd = os.open(dirpath, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dfd)
        except OSError:
            pass
        finally:
            os.close(dfd)

    # ───────────── GC 原语 + 保险丝/版本作废判定（本卡 t_567e3b29 收口）─────────────
    def clear(self) -> bool:
        """删池文件（会话归档/删除时 GC 联动调用）。存在→删返 True；不存在/IO 错→False（幂等，零崩）。"""
        try:
            os.unlink(self.path)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def check_safety(self, segment_count: Optional[int] = None) -> list:
        """保险丝（plan P4 item 4）：池 >5MB 或 条目数 > 会话段数 → 整池作废重建。

        纯判（无副作用）：返回违规原因列表，空列表 = 池健康。任一命中即应 `invalidate()`
        （全量重建）。调用方（engine replay 钩子）可单独 triage，或用 `guard()` 一次判 + 作废。
        - `size_over_cap`：池文件字节数 > POOL_MAX_BYTES（5MB）。os.path.getsize 量原始文件
          （不依赖可解析——半写/损坏的大文件同样须作废）。文件不存在 → 0（不判）。
        - `entry_count_over_segments`：池内条目数 > 会话段数。段数 None → 不判该项（只按 size）。
          池损坏时 load() fail-open 回 {}，条目数 = 0（不误判）。
        """
        reasons = []
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0
        if size > POOL_MAX_BYTES:
            reasons.append(
                f"size_over_cap: {size} bytes > POOL_MAX_BYTES ({POOL_MAX_BYTES})")
        if segment_count is not None:
            n_entries = len(self.load())
            if n_entries > segment_count:
                reasons.append(
                    f"entry_count_over_segments: {n_entries} entries > {segment_count} segments")
        return reasons

    def pool_model_version_mismatch(self, model_version: str) -> bool:
        """版本作废判（plan P4 item 3）：池内任一条目由「别的模型」生成 → 整池陈旧。

        纯判（无副作用）：空池 = 无失配（False，无可作废）；条目 model_version 全等于当前
        模型 → False；任一不同（或缺 model_version）→ True。池 = 可删缓存，模型换代后旧蒸
        馏全失效 → 整池作废重蒸（而非逐条筛——那会留下陈旧条目污染 replay）。
        """
        by_hash = self.load()
        if not by_hash:
            return False
        return any(e.get("model_version") != model_version for e in by_hash.values())

    def invalidate(self) -> bool:
        """作废整池 → 重置为「空」以全量重建（plan P4 item 3/4 动作面）。

        池 = 可删缓存非事实源：作废 = 删池文件，下条 digest 完成时重新写回。幂等：池
        不存在/IO 错 → False（零崩，绝不炸宿主 turn）；实际删了一个文件 → True。
        纯动作不打日志——日志由知道原因的调用方 `guard()` 负责（避免重复面）。
        """
        return self.clear()

    def guard(self, model_version: Optional[str] = None,
              segment_count: Optional[int] = None) -> bool:
        """一次性前置安全闸（engine replay 钩子，读池前调）：任一命中即作废整池返 True，
        池健康返 False（可安全读）。命中面：
        - model_version 失配（`pool_model_version_mismatch`）
        - 保险丝（`check_safety`：size>5MB 或 条目数>段数）
        判 + 作废一体，把「何时该重建池」的算法全收在本模块（薄壳纪律）。命中即打一行
        WARNING（含原因，ops grep 面「为何全量重蒸」）。
        """
        tripped = []
        if model_version is not None and self.pool_model_version_mismatch(model_version):
            tripped.append("model_version_mismatch")
        tripped.extend(self.check_safety(segment_count))
        if not tripped:
            return False
        self.invalidate()
        logger.warning("digest-pool: invalidated for full rebuild "
                      "(reasons=%s; path=%s)", "; ".join(tripped), self.path)
        return True


def gc_session_pool(sid: str, hermes_home: Optional[str] = None) -> bool:
    """GC 联动（plan P4 item 4）：会话归档/删除时，跟随删除该会话的池文件。

    事件驱动原语——engine 在 session 归档/删除钩子调用（薄壳：算法全在此）。幂等：池
    不存在/IO 错 → False（零崩）。返回 True 仅当实际删了一个存在的池文件。路径走
    hermes_home 解析链（零硬编码，不出 /root、profile 名）。
    """
    return DigestPool(pool_path_for_session(sid, hermes_home)).clear()


def gc_orphan_pools(hermes_home: Optional[str] = None) -> list:
    """GC 清扫（plan P4 item 4 兜底）：遍历池目录，删「源会话 jsonl 已不在」的孤儿池。

    对应关系：池 `<sid>.digests.json` ↔ 会话 `<sid>.jsonl`（同住 store_dir、同 sid 前缀）。
    会话文件缺失（带外删除/归档、gateway 没跑 per-session clear）→ 该池孤儿 → 删。
    返回实际删除的路径列表（空 = 无孤儿 / 目录不存在）。存活池（jsonl 仍在）绝不触碰；
    单个删除失败（权限/并发）静默跳过、下轮 GC 再试（零崩）。
    """
    from ._bootstrap import STORE_SUBDIR, store_dir
    root = os.path.join(hermes_home, STORE_SUBDIR) if hermes_home else store_dir()
    if not os.path.isdir(root):
        return []
    removed = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".digests.json"):
            continue
        sid = name.removesuffix(".digests.json")
        if os.path.exists(os.path.join(root, sid + ".jsonl")):
            continue                       # 会话文件仍在 → 池非孤儿，不动
        try:
            os.unlink(os.path.join(root, name))
            removed.append(os.path.join(root, name))
        except OSError:
            pass                          # 零崩：删不掉的孤儿留下轮 GC
    return removed


class PoolWriteError(Exception):
    """池写路径 OSError 归一（full-disk ENOSPC / EIO）：调用方 catch 一族即够，裸 OSError 不逃逸。

    池=缓存非事实源——写失败调用方静默忽略即可（下事件再生），故独立异常族而非复用 store 的
    StoreError（池是独立派生层，不并入 jsonl 单一事实源的异常面）。
    """
    pass
