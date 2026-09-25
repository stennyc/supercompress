"""Schema：Segment 信封 + NFC content_hash + 确定性 token 估计。

数据层定稿（@data-engineer 锁定，gate v4 (A)）：
- 单 jsonl、append-only、行内同 envelope；
- content_hash = sha256(NFC(utf8(original_text))) —— 归一化进 preimage，跨语言/编码稳定，重放去重幂等；
- segment_id = content_hash[:16] + "-" + seq —— 行级唯一身份（同文不同期 = 不同 id），
  随行进 jsonl，物化/重放时原样保留（stale 臂靠它区分同文的旧/新段）；
- digest 幂等再生键 = (content_hash, model_version, session_topic) 三元组（scdata.digest_regen_key，纯函数）；
- digest_text 为运行期字段，不落盘（可经三元组再生，不污染行字节）。

行字节由 scdata.to_line 单点产出（权威数据面，@reviewer 单一实现门）；
本模块只定义信封与解析，不另造第二套行格式。
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass


def nfc_hash(text: str) -> str:
    """内容哈希：NFC 归一化进 preimage（跨语言/编码稳定，物化字节不变、重放去重幂等）。"""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def est_tokens(text: str) -> int:
    """确定性 token 估计（无外部依赖）：ceil(len/4)，1 段下限 1。"""
    return max(1, math.ceil(len(text) / 4))


# 行内已知字段（from_dict 过滤：scdata 权威行可能带额外键，未知键静默丢弃）
_KNOWN = ("tier", "provenance", "original_text", "seq", "kind", "ts", "session_topic",
          "digest_ref", "model_version", "structural_p0", "reopened_from", "stale",
          "goal", "phase_id", "state", "decisions", "open_tasks", "guardrail")


@dataclass
class Segment:
    """jsonl 行信封。kind=segment（普通段）| kind=ledger（活账本段）。

    字段顺序：无默认值（tier/provenance/original_text）在前，带默认值在后（dataclass 语法要求）。
    """
    tier: str                       # "P0" | "P1" | "P2"
    provenance: str                 # "user" | "model"（信任级：user-sourced = 权威）
    original_text: str
    # ── 以下带默认值 ──
    seq: int = 0                    # 由 JsonlStore.append 赋值（行级唯一）
    kind: str = "segment"           # "segment" | "ledger"
    ts: float = 0.0
    session_topic: str = ""
    digest_ref: str = "pending"     # "pending" | "ready:<refid>" | "none"
    model_version: str = ""
    structural_p0: bool = False     # 命中确定性结构地板（user msg / TODO 标记 / 决策词 / 唯一错误）
    reopened_from: str | None = None  # 跨期重开物化来源（归档 id）；None = 原 session
    stale: bool = False             # 隔期重开的 P1 中间推理：降召回优先级，行动前 verify
    digest_text: str = ""           # 运行期：digest 生成后的摘要文本（不落盘，三元组可再生）
    # ── ledger 字段（kind=ledger 时启用）──
    goal: str | None = None         # 顶层目标：仅 user-sourced，永久 pin
    phase_id: str | None = None
    state: str | None = None        # done | in_progress | blocked
    decisions: list = None
    open_tasks: list = None
    guardrail: str | None = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.decisions is None:
            self.decisions = []
        if self.open_tasks is None:
            self.open_tasks = []
        if not self.content_hash:
            self.content_hash = nfc_hash(self.original_text)

    @property
    def segment_id(self) -> str:
        """行级唯一身份 = content_hash 前 16 + seq（同文不同期 = 不同 id；物化/重放原样保留）。"""
        return f"{self.content_hash[:16]}-{self.seq}"

    @property
    def digest_ready(self) -> bool:
        return self.digest_ref.startswith("ready:")

    def to_dict(self) -> dict:
        """行信封 dict（scdata.to_line 的入参——行字节单点产出在 scdata，权威数据面）。

        digest_text 为运行期字段，不序列化（可经再生三元组重生成，不污染行字节）。
        """
        return {
            "tier": self.tier, "provenance": self.provenance,
            "original_text": self.original_text, "seq": self.seq, "kind": self.kind,
            "ts": self.ts, "session_topic": self.session_topic,
            "digest_ref": self.digest_ref, "model_version": self.model_version,
            "structural_p0": self.structural_p0, "reopened_from": self.reopened_from,
            "stale": self.stale, "goal": self.goal, "phase_id": self.phase_id,
            "state": self.state, "decisions": self.decisions,
            "open_tasks": self.open_tasks, "guardrail": self.guardrail,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        """从已解析的 jsonl 行（scdata.replay 产出的 dict，已验 content_hash）重建。不重跑 tiering。"""
        known = {k: d[k] for k in d if k in _KNOWN}
        return cls(**known)

    @classmethod
    def from_line(cls, raw: str) -> "Segment | None":
        """单行解析 + content_hash 校验（scdata 同语义的薄解析，非第二套闭合实现）：
        JSON 解析失败 / hash 验不过 → None（torn 尾、坏行静默丢）。"""
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            d = json.loads(raw)
        except Exception:
            return None
        text = d.get("original_text", "")
        if nfc_hash(text) != d.get("content_hash"):
            return None
        try:
            return cls.from_dict(d)
        except Exception:
            return None
