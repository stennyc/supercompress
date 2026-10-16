"""紧凑摘要账本（卡片 t_63bf3ba2：append-only 累计布局，前缀单调，跨 compact 命中
provider 前缀缓存）。

现状（PRD §0.9 钉死）：engine.compress() 出块 = head(protect_first_n=3) + 滚动
tail(protect_last_n=6) + staged 块（S1 尾投递到 tail 末条 user）。staged 块恒 1 个、
每次 compact 新替旧 → compact #2 重算块 → 位置 2/tail 字节全变 → provider 前缀缓存
全量回落（根因矛盾，owner 指令"1 要"）。

目标布局（owner 裁定 · v6.1 页化 §2.4）：
    head(3) + [页化账本 c#1..c#N（每页只装自己段新 entries + 页头 notes 区前页 settled
    快照）] + 滚动 tail
- 每次 compact 只 **append** 新页 c#N（N 跨压缩单调）；**旧页字节永不回改** →
  无 settled 状态翻转时除最后一页外全部旧页字节不变 → provider 前缀缓存命中
  除最后一页外的全部旧页（v6.1 新锚；旧「累计超集严格前缀」契约已随页化迁移，
  见 tests/test_ledger_paging.py）。
- 某页 later settle 只在它**之后**新页的 notes 区体现（快照冻结铁律 3），旧页字节冻结。
- 唯一刷缓存位 = 滚动 tail（本就每 turn 变，便宜位）+ 状态翻转那次 compact 的
  新页 notes 区（规则 4 一次性 flush）。
- 封顶 / 轮转 = **整段清 + 重新积累**（seal：entries 清空、generation +1、seq 连续），
  不做 mid-block 回改（前缀已交付 provider 的旧字节永不被改写）。

侧车落点（G2 单源 + 与 digest pool / model store 同形同住）：
    <sid>.jsonl → <sid>.compact.json   （splitext 单源，零硬编码路径；零 .db、零宿主）

失败语义（账本 = 不可信派生层，与 digest pool 同口径，绝不炸宿主 compress 返回面）：
- 缺失 / 损坏 / 结构不符 → 空账本（generation 单调 +1 留痕），下次 compress 重新积累；
- 写 OSError（ENOSPC/EIO）→ 只告警，内存态照旧（下事件再写），宿主零感知。

格式（`<sid>.compact.json`）：
    {"version": 1, "generation": 0, "seals": 0, "seq": 0,
     "entries": [{"seq": 1, "text": "<staged 块原文>", "tokens": 123}, ...]}
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from .schema import est_tokens

MARK = "[supercompress compact-ledger]"
SECTION_MARKER = f"{MARK} "
_FORMAT_VERSION = 1

# ── settled 折出 + 单行存根 + 落盘旧 P0（owner 裁定：L1 区 + 纯+半 + 看板走半）──
# 不变量：存根 #seq == 盘文件名 <seq>.p0（一一对应，重开按 seq 捞原文）。
# 存根 16 hex（session 内 seq 无碰撞，审计面够）；kind = kanban(半自动) / adhoc(纯自动)。
_SETTLED_STUB_HASH_LEN = 16


def _settled_stub_line(seq: int, text: str, kind: str, sid: str = "") -> str:
    """settled P0 块在活渲染面里的单行存根（固定短字节，不随原文长度变）。

    P5 跨 session 隔离：盘形 `<sid>.p0.<seq>`（sid 空退化为旧形 `<seq>.p0`），存根
    定位提示同步渲染真实盘形——restore 按提示找文件不 miss。"""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:_SETTLED_STUB_HASH_LEN]
    disk = f"{sid}.p0.{seq}" if sid else f"{seq}.p0"
    return f"#seq={seq} P0 settled sha256={h} kind={kind} (full text on disk: {disk})"


def _trimmed_stub_line(seq: int, text: str, kind: str = "P1") -> str:
    """R5（owner 批次 A #2 编号空洞）：trimmable P1 块封顶被裁时折的单行存根。

    与 settled 存根同形（sha256 可审计 + 定位提示），但定位面不同：P1 可裁块
    原文**在 recall 层（segment_id 点查 / 主 store 行）**，不在 <seq>.p0 沉盘面
    （那是 settled P0 专属）。折存根而非硬删 = 账本行不产生编号空洞（seq 连续），
    被裁块仍经存根行可被人工定位原文（hash 与 store 行逐字可核）。

    参数名 `kind`（第三轮 owner 修正）：调用面传 `e.get("kind", "P1")`（'adhoc' /
    'kanban'，P1 块的 kind 审计值），存根渲染成 `#seq=N <kind> trimmed...`。
    原形参名 `tier` 与实传值（kind）错位，本轮对齐。
    """
    import hashlib
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:_SETTLED_STUB_HASH_LEN]
    return f"#seq={seq} {kind} trimmed sha256={h} (full text in recall/store)"


def evict_stub_line(seq: int, text: str, tier: str, anchor_ids: str) -> str:
    """v6 §2.1（存根挂逐出事件）：80%/90% 逐出点（sc.distill 逐出翻 digest_only）
    在 compact_ledger 活账本渲染面折的单行存根。

    与 `_trimmed_stub_line` 同形（复用其 sha256 审计 + 定位提示渲染），但定位面
    不同：被逐出段原文锚在 **durable 锚点**（store fsync+读回的 content_hash，
    复用现有锚点发射机制，owner 选 a 零新增沉盘），`anchor_ids` = 该段 durable
    锚点 segment_id 串（多个用 ", " 连接；无 durable 锚点的段不逐出存根——
    锚点未发射 = 逐出未发生，见 sc.distill 逐出接线面）。
    存根与 §2.4 页头 notes 区正交（T10）：本行写在逐出段自己那页的 entries 面，
    不进 notes 区快照；召回段（§2.2 source=recall）恒不产生本行。"""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:_SETTLED_STUB_HASH_LEN]
    return f"#seq={seq} {tier} evicted sha256={h} (durable anchors: {anchor_ids})"


def compact_ledger_path_for_jsonl(jsonl_path: str) -> str:
    """账本侧车路径 = 主 store 路径 splitext + '.compact.json'（G2 单源，零硬编码）。

    `<sid>.jsonl` → `<sid>.compact.json`；legacy `active.jsonl` → `active.compact.json`。
    与 digest pool（`<sid>.digests.json`）/ model 旁路（`<sid>.model.jsonl`）同目录、
    同 sid 前缀——GC 联动（按同目录同前缀删）语义自洽。
    """
    return os.path.splitext(jsonl_path)[0] + ".compact.json"


class CompactLedger:
    """append-only 紧凑摘要账本（整段 seal / reset；幂等；fail-open）。

    引擎读法：`_ledger` 懒构造一次（每 store 绑定）；`compress()` 每事件调
    `append(staged_text)`；会话边界（/new、轮转 rebind）调 `reset()`。
    跨进程：同 session 冷重开 → 新实例 load 既有侧车（seq/generation/seals 单调延续）。
    """

    def __init__(self, path: str,
                 max_tokens: int = 24576,
                 max_entries: int = 8,
                 logger: Any = None) -> None:
        self.path = path
        # P5+F4：sid = 盘形前缀（self.path 基名去 .compact.json），存根定位提示 +
        # token 预检按真实 sid 形（<sid>.p0.<seq>）口径，与 text() 渲染面对齐——
        # 预检用 sid="" 旧形（<seq>.p0）会让 _token_sum 比实际渲染面每块低 ~9 token
        # （F4），封帽账实二次错位；统一缓存一次，避免每块重算 basename。
        self._sid = os.path.splitext(os.path.basename(path))[0].replace(".compact", "")
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        self._log = logger
        # v6.1 页化（§2.4）：每 entry = 一页 c#N。页面只装自己段的新 entries
        # （`text` 全文恒在、不在 notes 区折存根）；`notes` = 该页写入那一刻
        # 前 N-1 页各自 settled 状态的一行快照（list of (seq, settled_flag)），
        # 写定后永不回改（快照冻结铁律 1-4，见 text() 渲染面）。
        self.entries: List[Dict[str, Any]] = []   # [{"seq","text","tokens","notes","settled","kind","trimmable","evicted"}]
        self.generation: int = 0
        self.seals: int = 0
        self.seq: int = 0
        self.load()

    # ───────────── 读路径（fail-open：缺失/损坏一律空账本 + generation 留痕）─────────────
    def load(self) -> None:
        """读侧车 → entries/generation/seals/seq。缺失（从未写过，合法态）→ 全 0 保持；
        损坏 / 结构不符 → 整段作废（generation +1、seals +1、seq 单调 +1），fail-open。

        P1（reviewer 二次复盘 · 跨进程竞态补全）：读面**同样**进专属 flock 临界区
        （`_flock_ledger` 的 LOCK_EX 短临界区：拿锁→读盘+解析→释锁），杜绝"双宿主
        冷重开并发 load→内存 append→flock persist"里 load 读到 stale 盘状态、各自
        append 到旧 seq 上、flock 写盘时互相覆盖丢块。flock 只护 `_persist` 写不护
        `load` 读 = 读侧 stale read 面（第一轮 F1 修复不完整，reviewer P1 实测"磁盘
        只剩 B1、A1 丢失"）。fail-open 不变：锁 open/flock 失败 → 退化裸读（pre-P1
        行为，零宿主回归）；锁临界区内的 FileNotFoundError/损坏仍走既有 fail-open
        语义（全 0 / _scrap），锁本身不改变读结果的错误处理。

        实现：把"读盘+解析+entries 填充"整体包在 `_do_read()` 里，在 flock 临界区
        （正常）或裸读兜底（锁面异常 fail-open）内执行——`raw` 与作用域自洽，锁面
        异常不引入 NameError。

        **合法部署形态边界（防误读）**：flock 治的是双宿主 / 双 provider **先后**
        （非同时）compress 同一 session 的竞态——B 的 load 读到 A 写好的完整盘状态、
        续 seq、不丢块。跨进程**真并发同时** compress 同一 session 是**非法部署
        形态**（宿主契约：一个 session 单 provider 在 compress，engine 会话面保证），
        不在 ledger 层为它做跨调用持锁——同 session 单 provider 由宿主层保证。"""
        _lock_cm = self._flock_ledger(os.path.dirname(os.path.abspath(self.path)) or ".")

        def _do_read():
            try:
                with open(self.path, "r", encoding="utf-8", newline="") as f:
                    raw = f.read()
            except FileNotFoundError:
                return
            except OSError as e:
                if self._log is not None:
                    self._log.warning("compact-ledger: read failed (%s) -> empty ledger", e)
                self._scrap("io_error")
                return
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:
                self._scrap("corrupt_json")
                return
            if not isinstance(data, dict):
                self._scrap("bad_structure")
                return
            if data.get("version") not in (None, _FORMAT_VERSION):
                self._scrap("version_mismatch")
                return
            self.generation = max(0, int(data.get("generation") or 0))
            self.seals = max(0, int(data.get("seals") or 0))
            self.seq = max(0, int(data.get("seq") or 0))
            entries = data.get("entries")
            self.entries = []
            if isinstance(entries, list):
                for e in entries:
                    if (isinstance(e, dict) and isinstance(e.get("text"), str)
                            and isinstance(e.get("seq"), int)):
                        _notes_raw = e.get("notes")
                        # v6.1 页化 notes 区读面：旧盘（无 notes 键）→ 空快照
                        # （前 N-1 页状态未知，fail-open 不回填 = 不回改旧页字节，
                        # 只对新写页生效——旧页 notes 缺失按其写入时快照缺失渲染）。
                        _notes = []
                        if isinstance(_notes_raw, list):
                            _notes = [(int(_p[0]), bool(_p[1])) for _p in _notes_raw
                                      if isinstance(_p, (list, tuple)) and len(_p) == 2]
                        self.entries.append({
                            "seq": e["seq"], "text": e["text"],
                            "tokens": int(e.get("tokens") or est_tokens(e["text"])),
                            # settled/kind 读面容忍旧条目无此键（→False/"adhoc"），
                            # 零破坏既有 compact-ledger 消费面 + 零 mutation（只增字段）。
                            # ⑤ 迁移缝：旧盘 settled 条目的 tokens 保留 3-A 前全文权重
                            # （load() 原样读回，不回改历史字节）→ 升级当次封帽过计、
                            # 一次 seal 自愈；保守方向（过计=seal 早到，绝不少计溢出），
                            # 故不修逻辑只留痕。
                            "settled": bool(e.get("settled", False)),
                            "kind": str(e.get("kind", "adhoc")),
                            "trimmable": bool(e.get("trimmable", False)),
                            "notes": _notes,
                            # v6 §2.1：旧盘无 evicted 键 → 空串（未逐出，渲染面
                            # 走原文/trimmed/settled 既有分支，fail-open 零扰动）
                            "evicted": str(e.get("evicted", "")),
                        })

        try:
            with _lock_cm():
                _do_read()
        except Exception:
            # 锁面意外（_flock_ledger 返回的 cm 本身抛，理论上不抛）→ fail-open 裸读
            _do_read()

    def _scrap(self, reason: str) -> None:
        """整段作废（不可信派生层语义）：entries 清空、generation/seals/seq 单调 +1。"""
        self.entries = []
        self.generation += 1
        self.seals += 1
        self.seq += 1
        if self._log is not None:
            self._log.info("compact-ledger: %s -> whole-section scrap "
                           "(generation=%d seals=%d seq=%d)",
                           reason, self.generation, self.seals, self.seq)

    # ───────────── 写路径（原子：tmp + os.replace；幂等零写快退；F1 跨进程 flock）────
    def _flock_ledger(self, target_dir: str):
        """F1（owner 既定决议 #4）：跨进程共享同一 <sid>.compact.json 时（双宿主 / 双
        provider 并发 compress），裸 tmp+os.replace 的 read-RMW-write 会让后写者
        覆盖先写者（seq 回退、新块丢失）——本仓零 flock（reviewer F1 实测：grep
        'flock' 零命中），多进程并发 = 数据丢失。

        修法（owner 裁定）：给 ledger 一个**专属 flock 文件** `<target_dir>/.ledger.flock`
        （不进账本 JSON、不随账本 GC），fcntl.flock(LOCK_EX) 阻塞拿锁、LOCK_UN
        释放；**fail-open**——锁文件 open 失败 / flock 抛 OSError → 静默返回
        _LockedNullContext（原 fail-soft 语义不破坏，宿主 worker 不被阻塞），
        竞争退化为无锁 RMW（丢行风险 = 既有行为，零回归），绝不炸宿主。
        threading.Lock（model_store 面）只管单进程内线程，跨进程面 = 本 flock。
        与 scdata 沉盘面解耦（flock 只护账本 JSON 的 read-RMW-write 临界区，
        <seq>.p0/.p1 沉盘走各自 O_APPEND，不在本锁面内）。
        """
        # 锁的 open/flock 失败 = fail-open 退化（null_cm 走裸 RMW，同 pre-F1）
        import fcntl
        import contextlib

        @contextlib.contextmanager
        def _null_cm():
            yield

        lock_path = os.path.join(target_dir, ".ledger.flock")
        try:
            _lf = open(lock_path, "a+")   # a+ 不截断、不创建内容竞争（flock 只看 inode）
        except OSError:
            if self._log is not None:
                self._log.warning("compact-ledger: flock open failed -> no-lock RMW "
                                  "(fail-open, same as pre-F1 behavior)")
            return _null_cm
        try:
            fcntl.flock(_lf.fileno(), fcntl.LOCK_EX)
        except OSError:
            try:
                _lf.close()
            except OSError:
                pass
            if self._log is not None:
                self._log.warning("compact-ledger: flock LOCK_EX failed -> no-lock RMW "
                                  "(fail-open)")
            return _null_cm

        @contextlib.contextmanager
        def _locked_cm():
            try:
                yield
            finally:
                try:
                    fcntl.flock(_lf.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    _lf.close()
                except OSError:
                    pass
        return _locked_cm

    def _persist(self) -> None:
        data = {"version": _FORMAT_VERSION, "generation": self.generation,
                "seals": self.seals, "seq": self.seq, "entries": self.entries}
        target_dir = os.path.dirname(os.path.abspath(self.path)) or "."
        # F1：整个 read-RMW-write（tmp 写 + os.replace + 目录 fsync）在专属 flock
        # 临界区内，跨进程并发 append 串行化（后写者等先写者释放）；fail-open
        # 退化为既有裸 RMW（零行为回归，锁面只是竞争时加锁）。
        _lock_cm = self._flock_ledger(target_dir)
        try:
            fd, tmp = tempfile.mkstemp(prefix=".compact-", suffix=".tmp", dir=target_dir)
        except OSError as e:
            if self._log is not None:
                self._log.warning("compact-ledger: persist open failed (%s), "
                                  "in-memory only this round", e)
            return
        try:
            with _lock_cm():
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
                # 目录项 fsync（rename 持久化 best-effort，平台不支持/只读静默）
                try:
                    _fsync_dir(target_dir)
                except Exception:
                    pass
        except OSError as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            if self._log is not None:
                self._log.warning("compact-ledger: persist failed (%s) -> "
                                  "in-memory only this round", e)

    # ───────────── append-only 核心 ─────────────
    def _token_sum(self) -> int:
        return sum(e["tokens"] for e in self.entries)

    def _full_block_count(self) -> int:
        """R5 #2：未折存根的全文块计数（条数帽单口径）。

        条数帽语义 = 最多同时保留 N 个**全文块**；settled / trimmed 存根块
        只剩单行（token 面已按存根实权重计，见 3-A），不占条数帽位——否则
        折存根后 entry 仍在账本、len 不降 → 条数帽永远超限 → 裁量层形同
        虚设、总被 ③ seal 兜底（= 编号空洞原样保留的根因）。"""
        return sum(1 for e in self.entries
                   if not e.get("settled") and not e.get("trimmed")
                   and not e.get("evicted"))

    def append(self, text: str, forms: Optional[Dict[str, int]] = None,
               settled: bool = False, kind: str = "adhoc",
               trimmable: bool = False) -> Optional[Dict[str, Any]]:
        """append 一个新累计块 c#N（幂等）；封顶 → **trimmable 层优先裁 + settled 层次之 + 整段 seal 保底**：

        封顶语义（owner 裁定 L1 区 + tier 分层 + settled 折出）：
        - `max_entries` = 累计段最多同时保留 N 块；
        - `max_tokens` = 累计段 token 上限；
        - 超帽裁切顺序（最省活渲染面字节、前缀最稳）：
            ① 折账本内**全部 trimmable 块**（P1 推理段，原文 recall 可捞回，丢弃零数据
               损失）——`_trim_trimmable_oldest` 实现为「扫账本内**所有** trimmable 且
               未 settled/未 trimmed 的块逐一折 trimmed 存根」（全量折，无位置门槛），
               **P0 块（trimmable=False）永不被 P1 层折**；P1-only 账本全部折存根，
               混合 P0+P1 账本折掉所有 P1、P0 全文/存根照留（seal 不抢跑，编号连续）；
            ② 仍超 → 裁量层第二面 `_trim_settled_oldest`（R5 #2 语义 X 下退化为 no-op
               保底——settled 块已折单行存根、不硬删 = 不产生编号空洞，见该方法 doc）；
            ③ 还超 → 回退整段 seal（保底，清空账本，新块重入账）。
          未 trimmable 且未 settled 的 P0 定案块（含 kanban candidate）**永不被 ①② 裁**
          （只可能被 ③ seal 全清）——定案账本只剩 P0 存根 + 未定案 P0 全文，P1 推理
          可裁层被优先全折。
        settled 块在 `text()` 里折成单行存根（`_settled_stub_line`），原文沉盘
        （engine 侧调 scdata.sink_p0）；trimmable 块 `text()` 照常渲染全文（P1 可裁层
        不折存根——还没定案，折了模型看不到推理过程），封顶时被优先裁掉。
        3-A 账实一致：settled 块 `tokens` 按存根实渲染权重记账（append(settled=True) /
        settle() 翻转时同步改计；封帽预检与实存同口径），封帽检查用实值——seal 不再
        被存根全权重误触发。

        幂等：text 与末条相同 → None（零写盘）。返回新条目 dict 或 None（幂等快退）。
        """
        t = str(text or "")
        if not t:
            return None
        # R5（owner 批次 A #8 语义去重门·零 LLM 近重复版）：新块**开首 40 字符**与
        # 账本内任一**未 settled/未 trimmed** 块的开首前缀碰撞（一方是另一方前缀 /
        # 相同）→ 跳过不 append（旧块信息量更大照留，零写盘）。诚实边界（owner 已知）：
        # 只拦"近重复"（同 compact 事件重生成 / 同模板清点），"同任务详版+精简版"
        # 开首不同字面 → 拦不住（需 LLM 裁判面 = 批次 B，不在本批）。
        # R5（owner 批次 A #8 语义去重门·零 LLM 强判据版）：新块与账本内任一**未
        # settled/未 trimmed** 块的 **sha256 全文哈希前 16 字节**碰撞 → 跳过不 append
        # （整块字面相同 = 真近重复，旧块信息量更大照留，零写盘）。
        # 诚实边界（owner 已知）：只拦"整块字面重生成"（同 compact 事件重算 staging
        # 块、同模板清点），"同任务详版+精简版"字面不同 → 拦不住（需 LLM 裁判面 =
        # 批次 B，不在本批）。
        # 为何用全文 sha256 前 16 字节而非开首 40 字符前缀碰撞：staging 窗口滚动前
        # 起点未动时，重算 P0 块开首 40 字与既有块完全相同字面（同一批 user 段），
        # 前缀碰撞判据误杀合法滚动重生成块 → c#2 永不 append（批次 A #8 实现缺陷）。
        # 改全文哈希前缀碰撞：staging 重生成块整体字面不变 = 真近重复（幂等快退已
        # 拦），整体不同 = 窗口滚动了 = 合法新块（不拦），误杀面归零。
        import hashlib as _hl
        _hash_new = _hl.sha256(t.encode("utf-8")).digest()[:16]
        for _e in self.entries:
            if _e.get("settled") or _e.get("trimmed"):
                continue
            _hash_old = _hl.sha256(_e["text"].encode("utf-8")).digest()[:16]
            if _hash_new == _hash_old:
                if self._log is not None:
                    self._log.info("compact-ledger: near-dup skip (sha256 prefix collision with c#%s)",
                                   _e["seq"])
                return None
        if self.entries and self.entries[-1]["text"] == t:
            return None                      # 幂等快退：同窗口重算同块
        # 3-A 账实单口径：封帽预检的 incoming 权重必须与实存口径一致——
        # settled=True 时该块实渲染面是存根（_settled_stub_line），按存根权重预检；
        # 未 settled 按全文 est_tokens(t)。否则「预检用全文、实存取根」= 账实二次错位
        # （settled 块封帽过计，seal 比该来得早）。
        next_seq = self.seq + 1
        if settled:
            tokens = est_tokens(_settled_stub_line(next_seq, t, str(kind), self._sid))
        else:
            tokens = est_tokens(t)
        cumulative = self._token_sum() + tokens
        # 封顶（R5 #2 语义 X·owner 裁定）：
        #  ① 条数帽只数**未折存根全文块**（`_full_block_count` = 未 trimmed/未 settled）
        #     + 本次新块（必为全文块）——trimmed/settled 存根块仅单行、不占帽位。
        #     这是语义 X 的真治理点：被裁 P1 折存根后 entry 常驻（编号零空洞），len(entries)
        #     不下降，若仍数 len(entries) → 条数帽永远超限 → 总走 ③ seal 全清 = 硬删编号空洞
        #     原样保留（#2 没修）。改数全文块数 → 折存根真腾帽位，裁量层实压生效、seal 不抢跑。
        #  ② token 帽仍数全账本 `_token_sum`（trimmed/settled 存根块按存根实权重计，
        #     账实单口径同 settled，见 append entry["tokens"] 改计面）。
        # 判定式与旧 `len(entries)+1 > max` 在 trimmable 硬删语义下逐字等价（回归锚定
        # test_compact_ledger 既有行为零扰动）；语义 X 下 trimmed 存根块不再参与判定 = 行为变化。
        if (self._full_block_count() + 1 > self.max_entries) or \
                cumulative > self.max_tokens:
            # 语义 X：P1 层持续折存根直到 cap 真正满足（trim 循环）——
            # 每折 1 块 tokens 骤降（原文 ~40 → 存根 ~6），重算 _token_sum 再判，
            # 不满足就再折 1 块，直到「全文块数 ≤ max 且 全账本 token ≤ max」才停。
            # 条数帽 + token 帽双条件都满足才跳出；③ seal 仅在 P1 层彻底无块可折
            # （账本全是 P0 全文块，token 帽仍超）时兜底全清。
            _guard = 0
            while ((self._full_block_count() + 1 > self.max_entries)
                   or (self._token_sum() + tokens > self.max_tokens)) and _guard < 512:
                _before = self._trim_trimmable_oldest()
                if not _before:
                    break      # P1 层无块可折（全 P0）→ 跳出，③ seal 兜底（防死循环）
                _guard += 1
            self._trim_settled_oldest()
            # recheck 与预检同口径（全文块数 + 全账本 token）；seal 仍是 ③ 保底全清
            if self._full_block_count() > self.max_entries or \
                    self._token_sum() + tokens > self.max_tokens:
                self.entries = []
                self.generation += 1
                self.seals += 1
        self.seq += 1
        entry = {"seq": self.seq, "text": t, "tokens": tokens,
                 "settled": bool(settled), "kind": str(kind),
                 "trimmable": bool(trimmable)}
        # v6.1 页化（§2.4）：每页 notes 区 = 该页写入那一刻前 N-1 页各自
        # settled 状态的逐页一行快照（冻结）——写入后永不回改（快照冻结铁律 2/3：
        # 某页 later settle 只在它之后的新页 notes 区体现，绝不回头改旧页，否则
        # 旧页字节变 → 前缀缓存断）。零新状态面：快照 = 当前 entries 既有各页实态
        # 的投影，append-only，不回读不回写旧页。
        entry["notes"] = [[e["seq"], bool(e.get("settled"))]
                          for e in self.entries]
        # 3-A 账实一致：直接 append(settled=True) 的块按存根实渲染权重记账（活渲染面
        # 折存根），与 settle() 翻转后的 tokens 口径对称；未 settled 块按原文全权重。
        if entry["settled"]:
            entry["tokens"] = est_tokens(_settled_stub_line(self.seq, t, entry["kind"], self._sid))
        # C.1a：forms 非空才写入（安全：空/None 不占 ledger 体积）；存为排序后的
        # (form,count) 列表（JSON 可序列化；dict 键序不稳定，排序保证读面确定性）
        if forms:
            entry["forms"] = sorted((str(k), int(v)) for k, v in forms.items())
        self.entries.append(entry)
        self._persist()
        return entry

    def _trim_trimmable_oldest(self) -> bool:
        """封顶裁量第一层（owner 裁定 tier 分层 + R5 #2 语义 X·折存根不硬删）：
        把账本内**所有 trimmable 且未 settled/未 trimmed 的块**折成单行 trimmed 存根，
        返回本次是否至少折了 1 块（调用方据此判断是否继续循环）。

        语义 X（owner 裁定，真治诊断账本 #2 编号空洞）：被裁块 entry **常驻 `entries`**
        （trimmed=True 标记 + tokens 改计存根实渲染权重，封帽账实单口径同 settled），
        `text()` 把该块渲染面收缩成存根行（= 一次可控缓存刷位，同 settled 折存根面），
        **编号 seq 恒连续、零空洞**。原文仍在 recall 层（segment_id 点查可捞回），零数据
        损失。关键不变量：trimmed 存根块不占条数帽位（封帽判定改数
        `_full_block_count()`，见 append()），故折存根真能腾帽位、裁量层实压生效、
        seal 不抢跑；被裁块 entry 保留 → `len(entries)` 不下降（既有测试对"裁后条目数
        减少"的断言须随语义 X 更新为"trimmed 存根在场 + seals==0"）。
        全量折（reviewer P7 对齐实现）：本层折账本内**所有** trimmable 且未
        settled/未 trimmed 的块（无位置门槛——P0 挡在最旧端不影响中间/后端的 P1
        块被折；P1-only 账本全折，混合 P0+P1 账本折掉所有 P1、P0 照留）。
        P0 块（trimmable=False）永不被本层折；P1 真正被清光靠 ③ seal 保底（全 P0
        账本无 trimmable 块可折时本层 no-op → append() while 循环 return False 跳出）。
        """
        _trimmed_before = sum(1 for e in self.entries if e.get("trimmed"))
        for e in list(self.entries):
            if e.get("trimmable") and not e.get("settled") and not e.get("trimmed"):
                e["trimmed"] = True
                e["tokens"] = est_tokens(
                    _trimmed_stub_line(e["seq"], e["text"], e.get("kind", "P1")))
        did = sum(1 for e in self.entries if e.get("trimmed")) > _trimmed_before
        if did:
            # R5 #2 同事务沉盘（fail-open 只告警不炸宿主，同 engine settle_p0 sink 口径）：
            # 被折存根的 trimmable P1 块原文同事务沉 <seq>.p1（与 <seq>.p0 同目录对称
            # 不撞名）——seal 清账后仍可复活（engine.untrim_p1 → restore 面）；写失败
            # 账本仍折存根（原文 recall 层 segment_id 点查兜底，不阻塞）。
            # 折存根本身与 logger 解耦（封帽核心行为不因 logger 缺失失效）；
            # 沉盘 IO 面 logger 在场才告警（fail-open）。
            try:
                import scdata
                _sink_dir = os.path.dirname(os.path.abspath(self.path))
                _sid = os.path.splitext(os.path.basename(self.path))[0].replace(".compact", "")
                for _e in self.entries:
                    if _e.get("trimmed"):
                        if not scdata.sink_p1(_sink_dir, _e["seq"], _e["text"], sid=_sid):
                            if self._log is not None:
                                self._log.warning(
                                    "compact-ledger: c#%d p1 sink failed "
                                    "(in-memory trimmed only, recall layer fallback)",
                                    _e["seq"])
            except Exception:
                pass
            self._persist()
        return did

    def _trim_settled_oldest(self) -> None:
        """封顶裁量第二层（R5 #2 修订）：settled 块已是存根形态（text() 折单行存根，
        tokens 面按存根实权重计）——裁量层**不再删 settled entry**（硬删 = 编号空洞，
        诊断账本 #2 根因之一）；settled 存根行常驻账本（每块 ~22 token，封帽 token 面
        已按实值计、不会误触发 seal），腾帽位工作全由 ① trimmable 层（全文→存根，
        token 骤降）完成。本层退化为 no-op 保底（保留方法 = 封帽三态调用面契约不变，
        未来若引入「settled 存根再折」面在此扩展）。"""

    def reset(self) -> None:
        """会话边界（/new、轮转 rebind 新 session 侧车）整段清：条目 0、generation +1、
        seals +1（seq 不动——同 session 文件的单调账；新 session = 新侧车文件，
        旧 session 账本随 GC 联动删）。落盘一次。"""
        self.entries = []
        self.generation += 1
        self.seals += 1
        self._persist()

    # ───────────── 渲染（engine 只调本函数，零算法体）─────────────
    def text(self, new_text: str = "") -> str:
        """累计段完整渲染（v6.1 页化 + 页头 notes 区快照冻结，§2.4）：

        布局 = MARK 行 + 逐页「── 页 c#N ──」段（段间 \\n---\\n 分隔，既有消费面
        零扰动）：
          页 c#N = 页头 notes 区（该页写入时前 N-1 页各自 settled 一行快照）
                 + 本页 entries（自己段的新 message 全文，恒在、不在 notes 区折存根）
        快照冻结铁律（T8/T9 锁死）：
          - 每页 c#N 只装自己段的新 entries（规则 1）；
          - 页头 notes 区 = 写入那一刻的前 N-1 页 settled 快照（规则 2），写定后永不回改；
          - 某页 later settle 只在它之后的新页 notes 区体现（规则 3），旧页字节冻结；
          - 无 settled 翻转时除最后一页（新页）外全部旧页字节不变（规则 4）→
            provider 前缀缓存命中除最后一页外的全部旧页（v6.1 新锚；旧「累计超集
            严格前缀」契约已随页化迁移，见 tests/test_ledger_paging.py）。
        存根与 notes 区正交（T10）：被逐出段存根行（v6 §2.1 evict_stub_line）写在
        逐出段自己那页的 entries 面；召回段（§2.2 source=recall）恒不进 notes 区。
        未传 new_text = 只渲染已 append 的既有条目（幂等读面，compress() 常见调用）；
        传入 new_text = 预览「若再 append 一块」的前缀形态（new_text 与末条相同 → 不重复追加）。
        header 面（2026-10-16 owner 裁定 · 前缀缓存面收窄）：首行 MARK **不再渲染
        generation/seals 变体数字**——seal/scrap/reset 才 +1 的计数进活渲染面首行
        = 前缀断裂点从「块头」扩散到「整个 section 前部旧页」（旧页字节不变也救不
        回，前缀匹配从块头起算，首行变体 → 全落）。审计面单源保留：计数仍写在
        <sid>.compact.json（generation/seals 键，_persist 零改动）+ logger.info
        （append/scrap/reset 各观测位零改动）——活渲染面只读不渲染变体数字。
        前缀缓存面收窄后 = 旧页字节不变（铁律 4）+ 尾行滚动 tail，header 变体
        不再是已知刷位。
        settled 块折出：settled=True 的 P0 块渲染为**单行存根**（_settled_stub_line），
        原文不吐——模型只看到「#seq P0 settled <hash> <kind>」，不重读全文（治重复问）；
        trimmed 块（R5 #2 封顶折存根的 P1 可裁层）同折单行存根（_trimmed_stub_line，
        原文 recall/store 可定位、sha256 可核）；evicted 块（v6 §2.1 逐出存根）按
        `evicted` 存根字段渲染；未折出块照常渲染全文。
        旧盘兼容（fail-open）：无 `notes` 键的旧页（旧格式账本 reload）notes 区渲染
        为「notes: -」占位一行（快照缺失不回填 = 不回改旧页字节，§2.4 铁律 3 精神），
        旧页正文渲染面逐字节不变。
        空账本 → ""（engine 零发行账本消息）。"""
        all_entries = self.entries
        if new_text and (not all_entries or all_entries[-1]["text"] != new_text):
            _preview = self.entries + [{"seq": self.seq + 1, "text": str(new_text),
                                       "tokens": est_tokens(new_text)}]
            _preview[-1].setdefault("notes",
                                    [[e["seq"], bool(e.get("settled"))]
                                     for e in self.entries])
            all_entries = _preview
        if not all_entries:
            return ""
        # P5：存根定位提示渲染真实盘形 <sid>.p0.<seq>（sid 从 self.path 基名取）
        _sid = os.path.splitext(os.path.basename(self.path))[0].replace(".compact", "")
        # header 面收窄（2026-10-16）：首行只留 MARK（零变体数字）——前缀缓存命中面
        # = 旧页字节不变 + 尾行滚动 tail；generation/seals 计数退审计面（JSON 侧车
        # + logger，_persist 零改动），不进活渲染面。
        lines = [MARK]
        for i, e in enumerate(all_entries):
            # 页头 notes 区：本页写入时前 N-1 页的 settled 快照（冻结）。
            # 旧盘（无 notes 键）→ 空快照 → "notes: -" 占位（不回填旧页）。
            _notes = e.get("notes")
            if _notes:
                lines.append(f"── 页 c#{e['seq']} ──")
                lines.append("notes " + " ".join(
                    f"c#{_n}={ 'settled' if _s else 'open' }"
                    for _n, _s in _notes))
            else:
                lines.append(f"── 页 c#{e['seq']} ──")
                lines.append("notes: -")
            if e.get("evicted"):
                # v6 §2.1 逐出存根：原文退场（durable 锚点定位，evict_stub_line 渲染）
                lines.append(e["evicted"])
            elif e.get("trimmed"):
                # R5 #2：trimmable P1 块被封顶裁 → 单行存根（不硬删 = 无编号空洞；
                # 原文 recall/store 可定位，sha256 可核）
                lines.append(f"c#{e['seq']}: " + _trimmed_stub_line(e["seq"], e["text"], e.get("kind", "P1")))
            elif e.get("settled"):
                # settled P0 块 → 单行存根（原文沉盘 <sid>.p0.<seq>，重开按 seq 捞回，见 sink/restore）
                lines.append(f"c#{e['seq']}: " + _settled_stub_line(e["seq"], e["text"], e.get("kind", "adhoc"), _sid))
            else:
                lines.append(f"c#{e['seq']}: {e['text']}")
        return "\n---\n".join(lines)

    # ───────────── v6 §2.1 存根挂逐出事件（engine 逐出点接线，§2.1 唯一新增环节）────
    def evict_page(self, seq: int, tier: str, anchor_ids: str) -> Optional[Dict[str, Any]]:
        """被逐出段（80%/90% 水线翻 digest_only）在自己那页折单行逐出存根。

        复用 `_trimmed_stub_line` 渲染形态（sha256 可审计 + 定位提示），定位面 =
        durable 锚点 segment_id 串（owner 选 a：复用现有 fsync+读回锚点，零新增沉盘）。
        幂等：该页已折 evicted 存根 → None（零写盘）；找不到 seq → None（fail-open，
        账本与窗口漂移时不炸宿主）。与 §2.4 notes 区正交（T10）：存根行写在本页
        entries 面（`e["evicted"]`），不影响页头 notes 快照（notes 只记 settled）；
        与 §2.2 召回段正交：source=recall 段恒不进账本（engine 结算点先清），
        本方法只对真逐出段调用，恒不产生召回存根（T4 防递归）。

        LOW-4 去重门控（owner 拍板）：同一 segment_id 已在活账本内折过 evicted 存根
        （跨多次 compact 重 staged → 新账本页逐出 cb 折新页存根，sha256 相同）时，
        本方法直接返回 None（幂等零写盘）——去重判据 = 扫描 `self.entries` 中
        既有任何 `evicted` 字段是否已含 `(durable anchors: <anchor_ids>)` 子串，
        纯内存读、零 LLM、零宿主、fail-open（扫描异常 → 继续走正常折存根面，
        不炸宿主）。不同 segment_id 的存根不受影响（不误并）。"""
        _anchor_key = f"(durable anchors: {str(anchor_ids)})"
        for _e in self.entries:
            if _e.get("evicted") and _anchor_key in _e["evicted"]:
                if self._log is not None:
                    self._log.info("compact-ledger: evict_page dedup skip "
                                    "(anchor %s already stubbed in c#%s)", anchor_ids, _e["seq"])
                return None
        for e in self.entries:
            if e["seq"] == seq:
                if e.get("evicted"):
                    return None                       # 幂等快退
                e["evicted"] = evict_stub_line(seq, e["text"], str(tier), str(anchor_ids))
                e["trimmed"] = True                   # 存根块不占条数帽位（同 trimmed 口径）
                e["tokens"] = est_tokens(e["evicted"]) # 3-A 账实单口径：封帽用存根实权重
                self._persist()
                return e
        return None

    # ───────────── v6 §2.2 召回暂存先清（owner 选 i 直接丢，不进存根账）─────────────
    def discard_entries(self, seqs) -> int:
        """直接丢弃指定 seq 的账本页（owner 选 i：不留占位、不进存根账）。

        召回暂存段（source=recall，compact 结算点先清）是临时捞回、用完即弃——
        不进活会话 P0 存根账（防递归多存根，T4 关键：召回段恒不进存根账）。
        durable 锚点仍在 store 可审计（丢弃 ≠ 删数据）。`seqs` = 被丢页 seq 集；
        幂等（重复丢弃 = 只丢第一次找到的）；找不到任何 seq → 0（零写盘，fail-open）。
        返回实丢页数（engine 观测面日志用）。丢弃后 seq 不回填——编号空洞只可能
        出现在「账本活面」（丢弃面），封帽判定按剩余页实值计（`_full_block_count`
        / `_token_sum` 天然只数现存 entries），seal 兜底语义零变化。"""
        _seqs = set(seqs)
        _before = len(self.entries)
        self.entries = [e for e in self.entries if e["seq"] not in _seqs]
        _n = _before - len(self.entries)
        if _n:
            self._persist()
        return _n

    # ───────────── settled 状态翻转面（append-only：只置位/清位，不回改旧字节）─────────────
    def settle(self, seq: int, kind: str = "adhoc") -> Optional[Dict[str, Any]]:
        """把 seq 对应块置 settled=True（折出活渲染面 → 存根；原文保留在 entry.text 供 sink 沉盘）。

        幂等：已 settled → None（零写盘）。找不到 seq → None（fail-open，不炸宿主）。
        kind = kanban(半自动：LLM 裁判 + clarify 确认) / adhoc(纯自动)。
        3-A 账实一致：折出存根的同时把 `e["tokens"]` 从原文 est_tokens 改计为**存根实渲染权重**
        （`est_tokens(_settled_stub_line(...))`）——封帽检查（_token_sum）不再被存根全权重
        误触发（seal 不再比 engine 注释假设的「存根 ~22 token/块」来得早）；账实对齐。
        unsettle 回翻时复原为原文 est_tokens（活渲染面恢复全文，cap 检查用回全权重）。
        前缀影响：settled 块的活渲染面从全文变存根 = 该块字节缩短 → 其【之后】块前缀位移
        （= 一次可控的缓存刷位）；旧 settled 块不再增长，前缀在 settled 稳定后单调。"""
        for e in self.entries:
            if e["seq"] == seq:
                if e.get("settled"):
                    return None                       # 幂等快退
                e["settled"] = True
                e["kind"] = str(kind)
                # 3-A：tokens 改计存根实渲染权重（活渲染面字节收缩，封帽用实值）
                e["tokens"] = est_tokens(_settled_stub_line(e["seq"], e["text"], e["kind"], self._sid))
                self._persist()
                return e
        return None

    def unsettle(self, seq: int) -> Optional[Dict[str, Any]]:
        """重开（复活）：把 settled 块清回 settled=False（活渲染面恢复全文，重新回喂）。

        可审计：清位只置 False，不删 entry（原文仍在，重开不丢数据）。
        幂等：本就未 settled → None。找不到 seq → None（fail-open）。
        3-A 账实一致：复原 `e["tokens"]` 为原文 est_tokens（活渲染面恢复全文，
        封帽检查回到全权重——与 settle 的存根权重对称翻转）。"""
        for e in self.entries:
            if e["seq"] == seq:
                if not e.get("settled"):
                    return None
                e["settled"] = False
                e["tokens"] = est_tokens(e["text"])   # 3-A：复原原文全权重
                self._persist()
                return e
        return None

    def untrim(self, seq: int) -> Optional[Dict[str, Any]]:
        """R5（owner 批次 A #2）：把 trimmed P1 存根块复活（账本活渲染面恢复全文）。

        与 unsettle 对称但作用面不同：unsettle 只对 settled（P0 定案沉盘面）有效；
        trimmed = P1 可裁层被封顶折存根（原文从未沉盘、在 recall/store 层），复活
        后 tokens 复原为原文全权重。幂等：本就未 trimmed → None（零写盘）；找不到
        seq → None（fail-open）。账实单口径：复活后封帽检查用回全文权重（同 unsettle
        的 3-A 对称翻转）。"""
        for e in self.entries:
            if e["seq"] == seq:
                if not e.get("trimmed"):
                    return None
                e["trimmed"] = False
                e["tokens"] = est_tokens(e["text"])   # 3-A：复原原文全权重
                self._persist()
                return e
        return None

    def entry_by_seq(self, seq: int) -> Optional[Dict[str, Any]]:
        """O(n) 点查（账本 blocks 数小，O(n) 足够；零新索引结构 = YAGNI）。"""
        for e in self.entries:
            if e["seq"] == seq:
                return e
        return None


def _fsync_dir(dir_path: str) -> None:
    """目录 fsync（best-effort，镜像 digest pool 口径）。"""
    dfd = os.open(dir_path, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
