"""Block ⑤ drift：任务漂移发现（topic epoch，S-Phase 5 / plan P2，2026-10-01 实装）。

零 LLM、零网络、纯确定性（宿主侧话题判定，开放问题#3 收口；LLM-clarify 路线已弃用，
会破 prompt-cache 前缀）。P1 是前置——本模块**复用 P1 的 `sc.recall.tokenize`**
（拉丁词 + CJK unigram+bigram），无空格 CJK 话题也能建锚/判分，100+ 语言覆盖。

三层（引擎侧、可逆；goal/ledger/structural P0 永不触碰）：
- P2.1 epoch 打标：复用 `Segment.session_topic`（生产 464 行恒 ""）= task epoch id；
  新 user 段按当前 epoch 打标（string 化存，兼容 _KNOWN 持久化 + replay 原样保留）。
- P2.2 检测：session 头 N_ANCHOR 条 user 段建话题锚点（显著 token 并集）；后续 user 段
  对锚点算**显著 token 命中率**（|q∩anchor|/|q|，确定性、自标定、无绝对魔数），
  连续 K_CONSEC 条 < 阈值 → drift 事件 + epoch 前进 + 锚点用触发 K 条重建。
- P2.3 双档响应（引擎侧调用，本模块只给判定/重排，不逐出——逐出是 distill.form 语义）：
  soft = 旧 epoch 非结构 P1/P2 召回 rerank ×SOFT_FACTOR（P1 的 search() 打分核零动，
  独立 rerank 步）；hard = 旧 epoch 非结构 P1/P2 → 窗口 form=digest_only（可逆：
  jsonl 单一事实源字节不动，epoch 再断言/回切即恢复；goal/ledger/structural P0 豁免）。

DoD（plan P2）：3 task 切换注入，检测延迟 ≤ 2 条（故 K_CONSEC 默认 2——spec 原文 K=3
与 DoD≤2 冲突，取 2 满足 DoD，可配）；同主题 session 不误报（FPR 面）；hard 可逆性。
确定性：锚点/查询 token 按 sorted 序、无 wall-clock/随机 → 同 user-seg 序列两次全等。
fence：纯 sc 层算法 + engine 接线；不新增数据列（复用 session_topic）、不触碰 P1 recall
排序核、不碰宿主原代码（用户硬约束①②贯穿）。
"""
from __future__ import annotations

from .recall import tokenize

# ── P2 定稿参数（plan P2.x；引擎配置面可覆写）──
N_ANCHOR = 5            # 头 N 条 user 段建话题锚点（评分开始前）
K_CONSEC = 2            # 连续 K 条低于阈值 → drift（DoD≤2 消息；spec K=3 与之冲突，取 2）
DRIFT_THRESH = 0.30     # 命中率阈值：|q∩anchor|/|q| < 0.30 判为「低相关」（自标定，非绝对分）
SOFT_FACTOR = 0.30      # P2.3 soft：旧 epoch 非结构 P1/P2 召回 rerank 权重


def _significant(text: str) -> set[str]:
    """显著 token 集合（话题信号）：CJK 表意单字 或 长度≥2 的拉丁/数字 token。

    纯标点/单字母拉丁词不算话题信号（避免 CJK 单字被误当噪声）；确定性（纯字面）。
    """
    out: set[str] = set()
    for t in tokenize(text):
        is_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" for c in t)
        if is_cjk or len(t) >= 2:
            out.add(t)
    return out


def hit_ratio(query_text: str, anchor: set[str]) -> float:
    """命中率 = 查询的显著 token 落在锚点里的比例（0..1，确定性）。空查询 → 0。"""
    q = _significant(query_text)
    if not q:
        return 0.0
    return len(q & anchor) / len(q)


class DriftTracker:
    """任务 epoch 状态机 + 锚点 + soft/hard 判定。纯函数面：无宿主、无 LLM、无网络、确定性。

    语义边界（session 记忆 scope）：reset() 只在会话边界调（on_session_start / on_session_reset）；
    epoch 单调前进（不回退）——「回切旧任务」= 旧 epoch 段重新可被召回（soft 因子自动归 1，
    因 is_old_epoch 比较 current_epoch），字节从未丢失（jsonl 单一事实源），即 hard 可逆。
    """

    def __init__(self) -> None:
        self.current_epoch: int = 0
        self.anchor: set[str] = set()        # 当前 epoch 话题锚点（显著 token 并集）
        self.anchor_user_count: int = 0       # 当前 epoch 已入锚 user 段数
        self.consec_low: int = 0              # 连续低于阈值计数
        self.user_seq: int = 0               # 本 session user 段计数（drift 事件可观测）
        self.drift_events: list[dict] = []    # 宿主日志/审计面（{epoch, user_seq, ratio}）
        self._recent: list[set[str]] = []     # 最近 K 条 user 显著 token（drift 时重建锚点）

    def reset(self) -> None:
        """会话边界：session 记忆 scope → 全清（fresh session 回到 epoch 0 建锚）。"""
        self.current_epoch = 0
        self.anchor = set()
        self.anchor_user_count = 0
        self.consec_low = 0
        self.user_seq = 0
        self.drift_events = []
        self._recent = []

    # ── P2.2 检测（喂一条 user 段 → 是否触发 drift）──
    def observe_user(self, text: str) -> bool:
        """喂一条新 user 段；建锚期累计头 N 条，评分期连续 K 条低相关 → drift。确定性。"""
        q = _significant(text)
        self.user_seq += 1
        if not q:
            return False                      # 无显著 token（纯标点/空）不判
        self._recent.append(q)
        if len(self._recent) > K_CONSEC:
            self._recent.pop(0)
        if self.anchor_user_count < N_ANCHOR:
            # 建锚期：头 N 条 user 段进锚点（不判漂移）
            self.anchor |= q
            self.anchor_user_count += 1
            self.consec_low = 0
            return False
        ratio = hit_ratio(text, self.anchor)
        if ratio < DRIFT_THRESH:
            self.consec_low += 1
        else:
            self.consec_low = 0
        if self.consec_low >= K_CONSEC:
            self._fire(q, ratio)
            return True
        return False

    def _fire(self, last_q: set[str], last_ratio: float) -> None:
        """drift 事件：epoch 前进；锚点用最近 K 条（触发本次的 user 段）重建。"""
        self.current_epoch += 1
        self.anchor = set().union(*self._recent) if self._recent else set(last_q)
        self.anchor_user_count = N_ANCHOR      # 新锚已建（无需再走建锚期）
        self.consec_low = 0
        self.drift_events.append({"epoch": self.current_epoch,
                                   "user_seq": self.user_seq, "ratio": round(last_ratio, 4)})

    # ── P2.3 判定/重排（引擎侧调用；本模块不逐出，逐出是 distill.form 语义）──
    def is_old_epoch(self, seg) -> bool:
        """旧 epoch 段：打了**整数字符串** epoch 标且 ≠ 当前 epoch。

        仅整数串算 epoch 标（epoch 是单调 int，引擎按 str(current_epoch) 打标）；
        非整数 session_topic（如 digest 幂等键用的 topic 串 "supercompress-dev"、
        生产 464 行恒 ""）一律**中性**——不降权、不逐出。这是 133 钉死不变量 + 生产
        数据安全的双重护城河：只有 P2 引擎打标的段（"0","1",…）才参与 soft/hard。
        """
        ep = getattr(seg, "session_topic", "") or ""
        if not ep.isdigit():                  # "" / "supercompress-dev" / 任意非整数 → 中性
            return False
        return ep != str(self.current_epoch)

    def soft_rank(self, hits: list) -> list:
        """P2.3 soft：旧 epoch 非结构 P1/P2 召回权重 ×SOFT_FACTOR，重排（P1 search 零动）。

        确定性：(-score, tier 序) 排序；P0/goal/ledger 不动（is_old_epoch 对 "" 返 False 且 P0 豁免）。
        """
        tidx = {"P0": 0, "P1": 1, "P2": 2}
        for h in hits:
            seg = h.seg
            if self.is_old_epoch(seg) and not seg.structural_p0 and seg.tier in ("P1", "P2"):
                h.score = h.score * SOFT_FACTOR
        return sorted(hits, key=lambda h: (-h.score, tidx.get(h.seg.tier, 9)))

    def hard_evict(self, segs: list) -> list:
        """P2.3 hard 候选：旧 epoch 非结构 P1/P2（→ 调用方置 form=digest_only）。

        可逆：只筛段不改字节（jsonl 单一事实源，point_fetch 仍可全量取回）；
        goal/ledger（kind=ledger）/ structural P0 永不逐。
        """
        return [s for s in segs
                if self.is_old_epoch(s) and not s.structural_p0
                and s.tier in ("P1", "P2") and s.kind != "ledger"]

    # ── DoD#2 hard 可逆：回切旧 epoch 即恢复段召回资格（软因子自动归 1）──
    def restore_epoch(self, epoch_id: str) -> None:
        """显式回切：current_epoch 对齐旧 epoch → is_old_epoch(旧段)=False → soft/hard 自动撤销。

        字节从未逐出（jsonl 不动），回切即恢复；单调前进的常规路径不调此（drift 只前进）。
        """
        self.current_epoch = int(epoch_id or 0)
        self.consec_low = 0

    # ── 冷重开恢复（网关重启）：避免 epoch 失配假阳性漂移 ──
    def rehydrate(self, segs: list) -> None:
        """会话冷重开（replay 了旧盘）：从现存段恢复 current_epoch + 锚点。

        缺陷护城河（session 记忆 scope，非 reset 归零）：裸 reset() 把 epoch 打回 0，
        但 replay 盘上已有 "1"/"2" 段（重启前写的）→ 全被 is_old_epoch 判旧 → 冷重启
        即假阳性 soft 降权 / hard 逐出。故 rehydrate 取**盘上最大整数 epoch** 为 current，
        并用该 epoch 的头 N 条有显著 token 的段近似重建锚点（seq 时序序）。
        - 无整数 epoch 段（全 pre-P2 "" / 非整数 topic）→ 等价 reset（epoch 0、中性）。
        - 非 user 段 / 非整数 epoch 段不重建锚点（中性，不参与话题判定）。
        确定性：纯字面，无 wall-clock/随机 → 同 replay 序列两次全等。
        """
        self.reset()
        ints = [s for s in segs if (getattr(s, "session_topic", "") or "").isdigit()]
        if not ints:
            return                                   # 全中性 → reset 态（epoch 0）
        self.current_epoch = max(int(s.session_topic) for s in ints)
        cur = [s for s in ints if s.session_topic == str(self.current_epoch)]
        cur.sort(key=lambda s: getattr(s, "seq", 0))  # seq 时序序（锚点用头 N 条）
        for s in cur:
            if self.anchor_user_count >= N_ANCHOR:
                break
            q = _significant(s.original_text)
            if q:
                self.anchor |= q
                self.anchor_user_count += 1
