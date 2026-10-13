"""Model 段独立旁路 store `<sid>.model.jsonl`（B 卡，用户裁定 2026-10-06「折中 C + 独立 jsonl」）。

主 store 字节契约零改动：model/assistant 段进旁路 `<sid>.model.jsonl`（与 digest 池
`<sid>.digests.json` 同住同 sid 前缀，legacy active.jsonl → active.model.jsonl）。
旁路 = 不可信信任边界：fail-open + 保险丝（构造/replay 对缺失/损坏 fail-open 空池；
保险丝命中 = 仅 ② 文件 > 5MB → 内存 reset（保留文件，不删）+ fused 标志（召回并集/T4 注入
跳过该 store）+ 日志 marker `model-store sidecar fused (reason=…)`）。① 行数比 2×、
③ model 版本失配已撤（carry-over 基准错配误熔断 + 旁路须适配任何 model/版本，见 check_fuse）。

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

import json
import logging
import os
import queue
import threading
import time

import scdata

from .schema import Segment, nfc_hash
from .store import JsonlStore, StoreError

logger = logging.getLogger(__name__)

# ── 旁路常量（单一事实源：改名字/上限只改这一处，G2）────────────────────
MODEL_MAX_BYTES = 5 * 1024 * 1024     # 保险丝②：model 文件 > 5MB → 不可信（唯一保留的保险丝）
# （保险丝①「model 行数 > 2× 主 store」已撤：carry-over 旁路池跨 session 累积，新 session
#  主 store 刚起步行数近 0，用当前 session 主 store 行数卡跨 session 累积池的 2× 阈值基准
#  错配，每个新 session 都误熔断砍召回；真实安全网 = ② 5MB 体积上限。③ model 版本失配亦已
#  撤：旁路 store 自动适配任何 model/版本，model_version 仅作行级 audit，不作 fuse 判定。）
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


# ── T1 脏水归一（确定性零 LLM；纯 stdlib；入库 gate 在既有 dedup/chars/marker 判定之后、
#    写盘之前）：把 audit 定罪的两类 model 侧脏水形状折叠为**单行自描述引用**——
#    (a) 顶层可解析裸 JSON 回复（dict/list）→ key 名索引 + 第一段可读 verdict；
#    (b) ``` 围栏代码 → [code:<lang> ×N lines] 首行引用。
#    归一后字节即入库字节（存量段不回写，只影响新入库段）。标记走 original_text 自带
#    前缀（[model-json]/[code:]）= 零 schema 变更（_KNOWN 不动）；fail-open：解析失败 /
#    围栏不配对 / 无脏水形状 → 原文原样入库，不阻塞。确定性：同输入两次逐字节相同。──
_NORM_LINE_CAP = 200         # 归一单行上限（verdict 行 ≤200 字符，截断加 ...）
_NORM_JSON_PREFIX = "[model-json]"
_NORM_CODE_PREFIX = "[code"
# R5（owner 批次 A #6 · 记账面阈值）：staged 块进账本前的占位符展开阈值。
# T1 折叠契约（_fold_code_fences）不变——任何围栏都折单行引用（治脏水、封 token）；
# 阈值只作用在**记账面**（engine.compress 的 staged 块生成处）：块内某段是
# [code: ×N lines] 占位符且 N < 阈值 → 该段展开原文进账本（小代码块全文比
# 残句占位符信息量大，残句混入账本 = 诊断账本 #6 根因）；N ≥ 阈值维持
# 占位符（大代码块全文太贵，残句够索引用）。0 = 关阈值（全占位符，旧行为）。
MODEL_LEDGER_FENCE_EXPAND_LINES = 20


def _is_small_code_placeholder(text: str, min_lines: int = MODEL_LEDGER_FENCE_EXPAND_LINES) -> bool:
    """R5 #6：text 是 [code: ×N lines] 占位符且 N < min_lines → True（小占位符，
    记账面应展开原文）。非占位符 / 大占位符（N≥阈值）/ min_lines=0 → False。
    确定性纯 stdlib，同 T1 前缀形态识别（_NORM_CODE_PREFIX）。"""
    t = str(text or "")
    if not t.startswith(_NORM_CODE_PREFIX):
        return False
    if min_lines <= 0:
        return False
    import re
    m = re.search(r"×(\d+) lines", t)
    if not m:
        return False
    return int(m.group(1)) < min_lines


def _json_scalar(v) -> str | None:
    """确定性标量渲染（bool/int/float/str → 稳定单行文本）；嵌套 dict/list = 不可读 verdict → None。
    字符串值：压平折行（\\n/\\t/\\r → 空格）+ 中和围栏（``去掉反引号，保 ``` 脏水特征不穿透
    verdict 单行入库）+ 120 字符截断（...）。verdict 恒单行且无 ``` 标记——JSON 值内嵌的代码
    围栏若原样入库会与「归一后单行自描述」契约冲突（审计 4 段 json∩fence 重叠形态即此形状）。"""
    if isinstance(v, str):
        s = " ".join(v.replace("`", " ").split())
        if not s:
            return None
        return s if len(s) <= 120 else s[:120] + "..."
    if isinstance(v, (bool, int, float)):
        return json.dumps(v)
    return None


def _first_readable_verdict(obj) -> str | None:
    """按插入顺序取第一段可读 verdict（dict: 首个标量值 → 'k=v'；list: 首个标量值）。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            sv = _json_scalar(v)
            if sv is not None:
                return f"{k}={sv}"
        return None
    if isinstance(obj, list):
        for v in obj:
            sv = _json_scalar(v)
            if sv is not None:
                return sv
        return None
    return None


def _norm_line(body: str) -> str:
    """拼 [prefix] body 并整行截断到 _NORM_LINE_CAP（确定性；截断加 ...）。单行不变式硬校验：
    归一产物必须无 \\n/\\r（脏水特征不得穿透单行契约）；含换行 = 内部形状 bug → 抛 ValueError
    （调用面 fail-open 捕获 → 原文入库，绝不落盘多行归一字节）。"""
    body = " ".join(body.split())
    line = f"{_NORM_JSON_PREFIX} {body}".strip()
    if len(line) > _NORM_LINE_CAP:
        line = line[:_NORM_LINE_CAP].rstrip() + "..."
    if "\n" in line or "\r" in line:
        raise ValueError("normalized line violates single-line invariant")
    return line


def _normalize_json_reply(text: str) -> str | None:
    """(a) 顶层裸 JSON 回复（strip 后可 json.loads 且顶层 dict/list）→ 单行 key 索引 + verdict。
    非 JSON / 顶层标量（42/'hi'）= 非脏水形状 → None（fail-open 走围栏/原文）。"""
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if isinstance(obj, dict):
        body = "keys=" + ",".join(str(k) for k in obj.keys())
        verdict = _first_readable_verdict(obj)
        if verdict:
            body += f" · {verdict}"
        return _norm_line(body)
    if isinstance(obj, list):
        body = f"{len(obj)} items"
        verdict = _first_readable_verdict(obj)
        if verdict:
            body += f" · {verdict}"
        return _norm_line(body)
    return None          # 顶层标量 JSON 非脏水形状


def _fold_code_fences(text: str) -> str | None:
    """(b) ``` 围栏折叠为单行引用（保留语言标注 + 行数 + 首行截断引用），围栏内容不全文入库。
    无围栏 / 围栏不配对（奇数个 ``` marker）→ None（fail-open 原文）。

    R5 修订（owner 批次 A #6 · 零回归裁定）：**T1 折叠契约不变**——任何配对的
    ``` 围栏（1 行起）都折单行引用，既有 7 条 normalize 测试锁死的「小围栏也折」
    语义原样保留。小 N 占位符混入账本的治理改在**记账面**（CompactLedger 渲染
    层对 [code: ×N lines] 占位符的记账/展示），不动本函数行为。"""
    lines = (text or "").splitlines()
    markers = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("```")]
    if not markers:
        return None
    if len(markers) % 2 != 0:
        return None          # 不配对 → fail-open
    refs: list[str] = []
    for m in range(0, len(markers), 2):
        open_j, close_j = markers[m], markers[m + 1]
        tag = lines[open_j].lstrip().lstrip("`").strip()
        lang = tag.split()[0] if (tag and not tag[0].isdigit()) else ""
        body = [ln.rstrip() for ln in lines[open_j + 1:close_j] if ln.strip()]
        n = len(body)
        first = (body[0].strip()[:40] + "...") if len(body[0].strip()) > 40 else (body[0].strip() if body else "")
        refs.append(f"{_NORM_CODE_PREFIX}:{lang} ×{n} lines] {first}")
    return " ".join(refs)


def normalize_dirty_text(text: str) -> str:
    """T1 入库归一入口（确定性、纯 stdlib、零 LLM）。

    顺序：(a) 顶层裸 JSON 回复 → key 索引单行；否则 (b) ``` 围栏 → [code:...] 单行；
    都无脏水形状 / 解析失败 / 围栏不配对 → 原文原样返回（fail-open，不阻塞）。
    返回值即入库 original_text 字节（nfc_hash / content_hash 均按此计算）。"""
    t = str(text or "")
    js = _normalize_json_reply(t)
    if js is not None:
        return js
    cf = _fold_code_fences(t)
    if cf is not None:
        return cf
    return t


def detect_norm_kind(text: str) -> str | None:
    """确定性脏水形状判别（零 LLM）：'json'（顶层 dict/list 可解析）/ 'fence'（成对 ``` 围栏）/
    None（无脏水形状）。归一优先级与 normalize_dirty_text 一致：JSON 判在前，JSON 命中即 'json'
    （即便内嵌围栏也归 json，不 fallback fence）；fail-open 不阻塞。"""
    if _normalize_json_reply(str(text or "")) is not None:
        return "json"
    if _fold_code_fences(str(text or "")) is not None:
        return "fence"
    return None


def normalize_model_response(text: str) -> tuple[str, str | None]:
    """T1 归一 API：返回 (归一后字节, 归一形状 kind)。kind ∈ {'json','fence',None}；
    kind=None = 原文原样（fail-open）。入库 gate 的轻量标记（key_fact 前缀，零 schema 变更）
    按 'norm:<kind>' 打点。确定性：同输入两次逐字节相同。"""
    t = str(text or "")
    kind = detect_norm_kind(t)
    if kind is None:
        return (t, None)
    return (normalize_dirty_text(t), kind)


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

    保险丝（命中即 `fuse()`：内存 reset 保留文件 + fused 标志 + marker；2026-10-09 用户裁定后
    仅保留 ②）：② 文件 > 5MB（唯一保留保险丝）。
    已撤：① 行数比 > 2× 主 store（carry-over 旁路池跨 session 累积 vs 新 session 主 store
    刚起步，基准错配，每个新 session 误熔断砍召回——真实安全网 = ② 体积上限）；
    ③ model 版本失配（旁路 store 自动适配任何 model / 任何版本；model_version 仅留作行级
    audit + digest 再生三元组数据，不作 fuse 判定）。

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
        self._superseded: set[str] = set()   # T3 清洗面：原段 segment_id 索引（召回池过滤）
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
        self._rebuild_superseded_index()   # T3 清洗面：原段过滤索引从行字节重建（fused/空池零成本）
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
        fused 态 = []（内存已 reset；召回并集跳过该 store）。
        T3 清洗面：被 supersede 的原段**出召回池**（原段行保留在盘、L0 点查仍可命中——
        append-only 不变式：0 删除、0 原段改写；过滤面 = 两个 engine 注入口的 model 物料源）。"""
        if self._superseded:
            return [s for s in self.store.by_hash.values()
                   if s.segment_id not in self._superseded]
        return list(self.store.by_hash.values())

    def _rebuild_superseded_index(self) -> None:
        """T3：清洗行（行内 `supersedes=<原段 segment_id>`，行级键）重建原段过滤索引。
        行字节事实源（重放幂等）；未知行键经 schema 契约静默丢弃 = 对读面零风险。"""
        self._superseded = set()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        d = json.loads(raw)
                    except Exception:
                        break                     # 与 scdata.replay 同口径：坏行起停
                    sup = d.get("supersedes")
                    if isinstance(sup, str) and sup:
                        self._superseded.add(sup)
        except (OSError, ValueError):
            pass                                  # fused 态（内存已 reset）= 空索引合法


    def supersede(self, segment_id: str, normalized_text: str, *,
                  model_version: str = "") -> Segment | None:
        """T3 存量清洗（append-only supersede，卡片钉死不变式）：

        - 新段 = **单物理行**：envelope + 行级审计键 `supersedes=<原段 segment_id>`
          （原段行**绝不改写**；重放未知键经 schema 契约静默丢弃，零读面风险；
          单行 = 避免双行 content_hash 重复导致重放 dedup 丢 supersedes 键）；
        - P0 写契约镜像 JsonlStore.append：seq 自增 + fsync + 读回校验 + 内存索引登记
          （行字节自洽 = nfc_hash 校验通过才读回）；
        - 幂等：归一段已在池（nfc_hash 命中，含重跑场景）→ 返回既有段、0 新增行、
          仅登记 superseded 过滤（不重复 append）；
        - 原段出召回池（live_segments 过滤）但保留：磁盘行字节不变、L0 点查仍可命中；
        - 原段缺失（未入库 id）/ fused / 空归一文本 → None no-op（绝不阻塞、绝不删行）。
        """
        if self._fused:
            return None
        orig = self.get(segment_id)
        if orig is None:
            return None
        norm_text = str(normalized_text or "")
        if not norm_text.strip():
            return None
        h = nfc_hash(norm_text)
        if h in self.store.by_hash:              # 归一段已在池（重跑/并发）→ 幂等不重 append
            self._superseded.add(segment_id)
            return self.store.by_hash[h]
        st = self.store
        seg = Segment(tier="P0", provenance="model", original_text=norm_text,
                      model_version=str(model_version or ""), structural_p0=False,
                      ts=time.time())
        seg.seq = st._next_seq
        st._next_seq += 1
        row = seg.to_dict()
        row["supersedes"] = segment_id           # 行级审计键（单物理行，重放静默丢弃）
        line = scdata.to_line(row)               # 行字节单点产出（scdata 权威面）
        st.log_water_check("supersede-append")
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            st.fsynced_p0_count += 1
            with open(self.path, "r", encoding="utf-8") as f:
                last = [l for l in f.readlines() if l.strip()][-1].strip()
        except OSError as e:
            raise StoreError(f"io_error: {e}") from e
        if Segment.from_line(last) is None:
            raise StoreError("readback_verify_failed")
        st._record(seg, durable=True)            # by_hash/index/durable 登记（读面一致）
        st._line_count += 1
        self._superseded.add(segment_id)         # 原段即刻出召回池（行键在盘，冷重开可重建）
        logger.info("model-store: superseded %s -> %s (append-only)",
                    segment_id, seg.segment_id)
        return seg

    # ── 保险丝（薄壳判+作废一体，算法面收口本模块；mirror digest pool guard()）──
    def check_fuse(self, main_line_count: int = 0, current_model: str | None = None) -> list[str]:
        """保险丝检查 + 命中即 fuse（mirror `DigestPool.guard()` 判+作废一体）。

        命中面（2026-10-09 用户裁定）：仅 ② 文件 > 5MB。
        - ① 行数比 > 2× 已撤：carry-over 旁路池跨 session 累积，新 session 主 store 刚起步
          行数近 0，用「当前 session 主 store 行数」卡「跨 session 累积池」的 2× 阈值基准错配，
          每个新 session 都误熔断砍掉召回（251>2×124 / 69>2×3 / 7>2×0）。真实安全网 = ② 5MB
          体积上限（外部篡改/失控膨胀由 ② 捕获；行数本身无「健康 carry-over」vs「篡改」的可判阈值）。
        - ③ model 版本失配已撤：旁路 store 须自动适配任何 model / 任何版本，不再因版本 stamp
          失配熔断（model_version 仍作行级 audit 数据保留，仅作 digest 再生三元组用，不作 fuse 判定）。
        纯检查：未命中返 []；命中已 fuse（幂等）返既有 fuse_reasons。
        注：main_line_count / current_model 保留参数仅为调用面（engine.py / 测试）签名兼容，
        本函数不再消费二者——①③ 已撤，判面单点 = ②。"""
        if self._fused:
            return self.fuse_reasons
        reasons: list[str] = []
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0
        if size > MODEL_MAX_BYTES:
            reasons.append(f"size_over_cap: {size} > {MODEL_MAX_BYTES}")
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
        # T1 脏水归一（确定性零 LLM）：入库字节 = 归一后字节（顶层裸 JSON → key 索引单行；
        # ``` 围栏 → [code:…] 单行引用；fail-open：无脏水形状/解析失败 → 原文原样）。
        # gate 判定（blank/dedup/chars/marker）全按原文不变——归一只改写盘字节，不改 gate。
        norm_text, norm_kind = normalize_model_response(text)
        h_norm = nfc_hash(norm_text)
        if h_norm in self.store.by_hash:        # 归一后同文双写封死（store 以归一 hash 为键；
            return (False, "dedup")             # clean 文本 h_norm==h，与上面 dedup 等价零扰动）
        seg = Segment(tier="P0", provenance="model", original_text=norm_text,
                      model_version=str(model_version or ""), structural_p0=False,
                      ts=time.time())
        # 轻量标记字段（卡片预案 B：_KNOWN 不动 = 零 schema 变更风险；key_fact 为运行期字段
        # 不落盘——归一事实本身已编码在入库字节前缀（[model-json]/[code]），可重放再现）：
        if norm_kind is not None:
            seg.key_fact = f"norm:{norm_kind}"
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
