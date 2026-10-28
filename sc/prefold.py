"""预折（prefold）数据面 —— 6 槽存根 list schema + 旁路 temp 文件。

owner 定稿（PRD §2 D1-D2，2026-10-28 · 全 owner-locked）：
- 存根 list = 6 槽：files_changed / done / open / user_decisions / why / next；
- 填充来源二分（D2）：3 确定性槽（files_changed=path_terms 路径词面、open=未闭环清单、user_decisions=定案码+决策句）
  + 3 LLM 槽（why/next/done，仅 P1/P2 model 段开；P0 段全确定性，3 LLM 槽标 "确定性面"）；
- 预折产物是【一次 compact 事件的临时态】（compact 执行后失效 / 水位回落白蒸作废 / 会话边界清零），
  独立旁路落盘面（本模块），**绝不进 sc.digest_pool**（durable LLM 缓存，5MB guard +
  content_hash dedup 键面 + 语义红线）。

数据面契约（本模块只负责 schema + 序列化 + 文件读写，零算法/零 LLM/零宿主耦合）：
- `StubList`：单段 6 槽结构（确定性槽与 LLM 槽都装这里；P0 段 LLM 3 槽值为 None，
  渲染面按 "确定性面" 省略）；
- `StubListFile`：`<sid>.prefold.tmp` 旁路文件读写（JSON 派生层，可删可重建、非事实源；
  fail-open 口径同 sc.digest_pool —— 缺失/损坏 → 空面，零崩溃，下事件再生）。
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

SLOT_FILES_CHANGED = "files_changed"
SLOT_DONE = "done"
SLOT_OPEN = "open"
SLOT_USER_DECISIONS = "user_decisions"
SLOT_WHY = "why"
SLOT_NEXT = "next"

STUB_LIST_SLOTS: tuple[str, ...] = (
    SLOT_FILES_CHANGED, SLOT_DONE, SLOT_OPEN,
    SLOT_USER_DECISIONS, SLOT_WHY, SLOT_NEXT,
)
DETERMINISTIC_SLOTS: frozenset[str] = frozenset(
    {SLOT_FILES_CHANGED, SLOT_OPEN, SLOT_USER_DECISIONS})
LLM_SLOTS: frozenset[str] = frozenset({SLOT_WHY, SLOT_NEXT, SLOT_DONE})

SLOT_VALUE_MAX = 120          # 单串槽值上限（LLM 自报字段限长兜底，+ '…' 截断标记）
SLOT_LIST_MAX_ITEMS = 8      # list 槽项数上限（防爆无界）


@dataclass
class StubList:
    """单段 6 槽存根 list（预折产物，临时态）。

    - `files_changed` / `open` / `user_decisions`：确定性面（list[str]，元素已限长）；
    - `why` / `next` / `done`：LLM 面（单串，P0 段 None = 渲染省略 "确定性面"）；
    - 全空（6 槽皆无值）= 预折未成文 → 渲染面回退现状纯编号锚点（fail-open，零误面）。
    """
    segment_id: str = ""
    files_changed: list[str] = field(default_factory=list)
    done: Optional[str] = None
    open: list[str] = field(default_factory=list)
    user_decisions: list[str] = field(default_factory=list)
    why: Optional[str] = None
    next: Optional[str] = None
    p0_deterministic: bool = False

    @staticmethod
    def extract_files_changed(original_text: str) -> list[str]:
        """files_changed 槽：path_terms 路径词面（sc.recall.path_terms 同源，独立小正则
        避免数据面 import sc.recall 的宿主耦合——路径 run 语义逐字同 recall.py:64 面）。"""
        from sc.recall import path_terms
        seen: list[str] = []
        for p in path_terms(original_text or ""):
            if p not in seen:
                seen.append(p)
                if len(seen) >= SLOT_LIST_MAX_ITEMS:
                    break
        return seen

    @staticmethod
    def _truncate_value(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            return None
        if len(v) > SLOT_VALUE_MAX:
            return v[:SLOT_VALUE_MAX] + '…'
        return v

    @staticmethod
    def _truncate_list(items) -> list[str]:
        out: list[str] = []
        for it in (items or []):
            it = str(it).strip()
            if not it:
                continue
            out.append(StubList._truncate_value(it))
            if len(out) >= SLOT_LIST_MAX_ITEMS:
                break
        return out

    def normalize(self) -> "StubList":
        """槽值限长/项数兜底（LLM 自报字段可能超上限，同 parse_digest_response F5 限长口径）。"""
        self.files_changed = self._truncate_list(self.files_changed)
        self.open = self._truncate_list(self.open)
        self.user_decisions = self._truncate_list(self.user_decisions)
        self.done = self._truncate_value(self.done)
        self.why = self._truncate_value(self.why)
        self.next = self._truncate_value(self.next)
        if self.p0_deterministic:
            # P0 段：LLM 3 槽恒 None（"确定性面"，D3 免 LLM 蒸）
            self.done = None
            self.why = None
            self.next = None
        return self

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "files_changed": list(self.files_changed),
            "done": self.done,
            "open": list(self.open),
            "user_decisions": list(self.user_decisions),
            "why": self.why,
            "next": self.next,
            "p0_deterministic": bool(self.p0_deterministic),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StubList":
        sl = cls(
            segment_id=str(d.get("segment_id", "")),
            files_changed=list(d.get("files_changed") or []),
            done=d.get("done"),
            open=list(d.get("open") or []),
            user_decisions=list(d.get("user_decisions") or []),
            why=d.get("why"),
            next=d.get("next"),
            p0_deterministic=bool(d.get("p0_deterministic", False)),
        )
        return sl.normalize()

    def is_empty(self) -> bool:
        """全 6 槽无值 = 预折未成文（渲染面回退纯编号锚点）。"""
        return not (self.files_changed or self.done
                    or self.open or self.user_decisions
                    or self.why or self.next)

    def render_line(self, anchor: str) -> str:
        """`<anchor> · files=[…] done=… open=[…] decisions=[…] why=… next=…`。

        空槽省略（不渲染 `slot=` 空位）；P0 段 LLM 3 槽 None → 整槽省略（= "确定性面"）；
        全空 → 原样返回 anchor（零膨胀，fail-open 零误面）。单行、确定性（同输入同输出）。"""
        if self.is_empty():
            return anchor
        parts: list[str] = []
        if self.files_changed:
            parts.append('files=[' + ' '.join(self.files_changed) + ']')
        if self.done:
            parts.append('done=' + str(self.done).replace(' ', '_')[:SLOT_VALUE_MAX])
        if self.open:
            parts.append('open=[' + ' '.join(self.open) + ']')
        if self.user_decisions:
            parts.append('decisions=[' + ' '.join(self.user_decisions) + ']')
        if self.why:
            parts.append('why=' + str(self.why).replace(' ', '_')[:SLOT_VALUE_MAX])
        if self.next:
            parts.append('next=' + str(self.next).replace(' ', '_')[:SLOT_VALUE_MAX])
        return anchor + ' · ' + ' '.join(parts)


def prefold_tmp_path_for_jsonl(jsonl_path: str, sid: Optional[str] = None,
                              root: Optional[str] = None) -> str:
    """`<sid>.jsonl` → 同目录 `<sid>.prefold.tmp`（与 source jsonl 同 sid 前缀同住）。

    GC 联动口径 = sc.digest_pool.pool_path_for_jsonl 同源（同目录同 sid 前缀删）；
    但本文件是 **compact 边界前的独立工作区**，清零时机独立（compact 后 / 水位回落 /
    会话边界），不进 digest_pool 的 5MB guard / content_hash dedup 键面。
    """
    sid = os.path.basename(jsonl_path)
    if sid.endswith('.jsonl'):
        sid = sid[: -len('.jsonl')]
    root = os.path.dirname(os.path.abspath(jsonl_path)) or '.'
    return os.path.join(root, f"{sid}.prefold.tmp")


class StubListFile:
    """`<sid>.prefold.tmp` 旁路：`{segment_id: StubList.to_dict()}` 的原子 JSON 文件。

    语义（owner-locked D7/D8）：
    - 预折产物【临时态】：compact 执行后 / 水位回落 / 会话边界三处清零（clear()）；
    - 写 = 全量原子（load 快照 + upsert + tmp+rename fsync，同 digest_pool _atomic_write
      口径；单文件小（预折区 ≤ 数十段 × 6 槽），无需流式 append）；
    - 读缺失 / 损坏 → `{}`（fail-open，下事件再生，宿主收尾面零炸）；
    - 绝不被 digest_pool 的 guard/dedup/GC 面触碰（独立文件，物理正交）。
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> dict:
        """读整文件 → `{segment_id: StubList}`。缺失 / IO 错 / 结构不符 → {}（零崩溃）。"""
        try:
            with open(self.path, 'r', encoding='utf-8', newline='') as f:
                raw = f.read()
        except (FileNotFoundError, OSError):
            return {}
        if not raw or not raw.strip():
            return {}
        try:
            doc = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            import logging
            logging.getLogger(__name__).warning(
                "prefold-tmp: read failed (%s) -> empty", self.path)
            return {}
        if not isinstance(doc, dict):
            return {}
        out: dict = {}
        for k, v in doc.items():
            if not isinstance(v, dict) or not str(v.get("segment_id", "")):
                continue
            sl = StubList.from_dict(v)
            if sl.segment_id:
                out[sl.segment_id] = sl
        return out

    def _atomic_write(self, doc: dict) -> None:
        """全量原子写（tmp + rename + fsync，同 digest_pool._atomic_write 口径；自有实现
        零跨模块依赖 —— 数据面不 import store 私有件）。"""
        import tempfile
        d = os.path.dirname(self.path) or '.'
        fd, tmp = tempfile.mkstemp(prefix='.prefold-', suffix='.tmp', dir=d)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
                json.dump(doc, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def upsert(self, seg_id: str, stub: StubList) -> None:
        """单段 upsert（幂等：已存同段 → 覆盖 = 预折再成文取最新；新段追加）。原子写。"""
        with self._lock:
            doc = {k: v.to_dict() for k, v in self.load().items()}
            doc[seg_id] = stub.to_dict()
            self._atomic_write(doc)

    def upsert_many(self, entries: dict) -> int:
        """批量 upsert（预折区一次写整区；返回新写入段数）。原子写一次。"""
        if not entries:
            return 0
        with self._lock:
            doc = {k: v.to_dict() for k, v in self.load().items()}
            new = 0
            for seg_id, stub in entries.items():
                if seg_id not in doc:
                    new += 1
                doc[seg_id] = stub.to_dict()
            self._atomic_write(doc)
            return new

    def get(self, seg_id: str) -> Optional[StubList]:
        """点查（compact 提取替换时按 segment_id 取预折 list；miss = None → 回退纯编号锚点）。"""
        return self.load().get(seg_id)

    def clear(self) -> bool:
        """清零（三处清零点：compact 后 / 水位回落 / 会话边界）。缺失 → no-op 合法态。"""
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
                return True
            return False
        except OSError:
            return False


def prefold_evict_region(window, water_line: int,
                         tiers: tuple = ("P2", "P1", "P0")) -> list:
    """蒸区划定：按 distill 选段语义（P2→P1→P0 逐 tier、tier 内 seq 升序、
    structural P0 门）选出 0.55 compact 会逐出成存根的段。

    口径与 sc.distill._evict_until 同源（复用选段顺序 + structural P0 永不逐语义），
    但**纯读、零写、零 LLM**：不逐出、不折存根、不挂 ledger cb——只吐 seq 列表，
    供壳侧对这批段预蒸 6 槽 + 落 prefold.tmp。

    `water_line` = 0.45 预折带水位对应的 token 线（shell 侧算好传入：
    context_length × prefold_water_frac）。逐段累加 `used_tokens`，压到 water_line 下停
    （与 _evict_until 的 `while used > water: evict next` 同停点）。

    structural P0（goal/ledger 尾，结构地板）永不进蒸区（对齐 distill `_next_candidate`
    结构门）；kind=segment 的 user 指令 P0 可进蒸区。

    档位口径说明（reviewer 备查 #2 · 与 distill tier 表逐档对齐，非缺陷）：
    蒸区选段**超集覆盖**——纳入 user 指令 P0（对齐 90% 超压 p0_force 口径）。但默认
    0.45 预折带下 distill 实际走 tier75（p0_force=False，仅 P2/P1 档，P0 不触达）→
    该 user-P0 段的预折 StubList 在默认面不会被 0.55 compact 逐出消费，只能被后续
    90% 超压档（p0_force=True）或 L0 降档面消费。即：预折区比默认 compact 档"多蒸"
    user-P0 段是**提前备料**（一旦水位真冲到 90% 或段被降 L0，6 槽 list 已就绪），
    不造成当前档的误蒸/误替——蒸区只吐 seq 列表供壳侧预蒸，实际替换由 compact
    逐出点 / choose_forms L0 面按各档真实逐出面消费，纯读零写零 LLM 的蒸区划定本身
    对默认 tier75 零行为影响（fail-open 超集）。

    # 返回 = 该预折段 seq 升序列表（空窗 / 未超水线 → 空列表，fail-open 零误面）。
    # 注：蒸区 seq 升序（跨 tier 合并，供 compact 精准替换按 seq 区间对齐，D9）；
    # 「P2 先蒸」是 LLM 预生成的顺序（壳侧 `_prefold_one_segment` 按 seq 逆序即
    # P2→P1→P0 消费），本纯函数只吐区间、不吐消费序。
    """
    if water_line is None or water_line <= 0:
        return []
    segs = [ws for ws in window.segs if ws.form == "resident"]
    if not segs:
        return []
    used = int(getattr(window, "used_tokens", 0) or 0)
    picked: list[int] = []
    if used <= water_line:
        return []                                   # 未超预折带 → 无蒸区（零预折，fail-open）
    from .schema import est_tokens
    for tier in tiers:                              # P2 先（最弱），P1 次，P0 末（最强）
        cands = sorted((ws for ws in segs if ws.segment.tier == tier),
                       key=lambda ws: ws.segment.seq)
        for ws in cands:
            if used <= water_line:
                break
            seg = ws.segment
            # structural P0（结构地板 goal/ledger）永不进蒸区（对齐 distill 结构门）
            if seg.structural_p0 and seg.kind in ("goal", "ledger"):
                continue
            # 预折区已 durable 的段（digest_only 已逐出）不进蒸区（只有 resident 待折的进）
            # —— 上方 segs 已过滤 form==resident，这里双保险
            picked.append(seg.seq)
            # 累加回收量（对齐 distill `_evict_until` 的 used 递减语义：逐一段省一段）
            _full = est_tokens(seg.original_text)
            _stub = est_tokens(seg.digest_text or "[no digest]")
            used -= max(0, _full - _stub)
    return sorted(picked)


PREFOLD_LLM_TASK = "prefold.stub_list"


def build_prefold_llm_prompt(seg, prev_key_fact: str, next_key_fact: str) -> str:
    """P1/P2 model 段 3 LLM 槽（why/next/done）的预蒸 prompt（确定性拼串，零 LLM）。

    邻段 key_fact（既有 80 字单串）作相邻上下文喂进去（滑窗前 1 后 1）→ 大纲在 +
    单次输入规模不变（段原文 ≤4096 截断 + 2×80 字 key_fact，aux 短超时有保证）。
    LLM 回 JSON（三键 why/next/done，壳侧 parse；解析失败 → 三槽 None 回退确定性面）。
    纯字符串拼装，确定性（同输入同输出），测试面可钉。
    """
    text = (seg.original_text or "")[:4096]
    prev = (prev_key_fact or "").strip() or "(无前段)"
    nxt = (next_key_fact or "").strip() or "(无后段)"
    return (
        "[prefold stub-list]\n"
        f"prev_segment_key_fact: {prev}\n"
        "segment_text:\n"
        f"{text}\n"
        f"next_segment_key_fact: {nxt}\n"
        "Emit JSON with keys why/next/done (single-string each, ≤120 chars; "
        "why=root-cause/basis, next=current-progress-anchor+next-step, "
        "done=completed-items narrative). No other keys."
    )


def parse_prefold_llm_response(text) -> dict:
    """LLM 响应 → {why, next, done}（三键单串）。容忍裸 JSON / code fence / 纯文本；
    解析失败 → {}（壳侧三槽 None 回退确定性面，fail-open 零崩溃）。纯函数零网络。"""
    import re as _re
    if not text or not str(text).strip():
        return {}
    t = str(text).strip()
    candidates = [t]
    m = _re.search(r'```(?:json)?\s*(.*?)```', t, _re.DOTALL)
    if m:
        candidates.insert(1, m.group(1).strip())
    m = _re.search(r'\{.*\}', t, _re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            d = json.loads(c)
        except Exception:
            continue
        if isinstance(d, dict):
            return {k: str(d.get(k, "")).strip() for k in ("why", "next", "done")}
    return {}
