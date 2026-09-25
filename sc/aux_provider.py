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

import os

import scdata

from .schema import Segment


def marker_enabled() -> bool:
    """gate⑤ 故障注入 marker：off-by-default（生产恒 inert，inject_fault_after 参数被忽略）。"""
    return os.environ.get("SC_INJECT_FAULT") not in (None, "", "0", "false", "False")


class AuxProvider:
    """(A) aux ≡ 主模型端点。endpoint 注入：生产 = Hermes 原生网关，测试 = FakeEndpoint。"""

    COMPRESSION_TASK = "auxiliary.compression"   # digest 映射原生 summarizer 任务

    def __init__(self, main_endpoint, max_concurrency: int = 1) -> None:
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
        for seg in self._queue:
            self._generate(seg)
        self._queue = []

    def ensure_digest(self, seg: Segment) -> dict | None:
        """幂等：已生成 → 返回缓存（不重算、不发起端点调用）；turn 内 → 排队 None；否则立即生成。"""
        if seg.segment_id in self._results:
            return self._results[seg.segment_id]
        if self._in_turn:
            if seg.segment_id not in self._queued:
                self._queue.append(seg)
                self._queued.add(seg.segment_id)
            return None
        return self._generate(seg)

    def _generate(self, seg: Segment) -> dict | None:
        try:
            self.main_endpoint.call(self.COMPRESSION_TASK, seg.original_text[:4096])
        except Exception:
            self.transient = True
            return None
        key = scdata.digest_regen_key(
            {"content_hash": seg.content_hash}, seg.model_version, seg.session_topic)
        refid = self._digest_cache.get(key)
        if refid is None:
            refid = scdata.sha256_nfc("digest|" + "/".join(key))[:16]
            self._digest_cache[key] = refid
        seg.digest_ref = "ready:%s" % refid
        if not seg.digest_text:
            seg.digest_text = seg.original_text[:80]   # head 摘录（marker 态占位小）
        out = {"digest_ref_id": refid, "content_hash": seg.content_hash}
        self._results[seg.segment_id] = out
        return out
