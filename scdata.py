"""scdata — supercompress session-memory data layer (reference implementation, gate v4 (A)).

Locked design points (design-review final):
- Single jsonl, append-only; envelope content_hash = sha256(NFC(utf8(original_text)))
- Two-level idempotent keys:
    * structural dedup / replay skip key = content_hash (single key, never re-tier on replay)
    * digest regeneration key = (content_hash, model_version, session_topic) (pure function)
- replay(): per-line JSON parse + content_hash verify; torn tail silently dropped;
  index rebuilt from valid head only (never incrementally patched).
- materialize(): archive -> new active file = byte-stable copy (verbatim lines, no
  re-normalization / re-encoding; any byte drift breaks dedup and double-writes).
- verify_anchor_closure(): the SINGLE implementation of the zero-dangling-anchor invariant.
  Anchors are only emitted after fsync + readback (write-before-compress gate); pending
  bytes physically have no pointer. Two implementations = dead-pointer bug farm.
"""
import hashlib
import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional


def sha256_nfc(text: str) -> str:
    """NFC-normalized sha256 over utf-8 bytes — the only hash primitive in the layer."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def make_segment(seq: int, original_text: str, *, kind: str = "segment",
                 tier: str = "P2", provenance: str = "model",
                 structural_p0: bool = False, digest_ref: str = "none",
                 model_version: str = "none", session_topic: str = "",
                 **extra) -> dict:
    return {
        "seq": seq,
        "kind": kind,
        "tier": tier,
        "provenance": provenance,
        "structural_p0": structural_p0,
        "original_text": original_text,
        "content_hash": sha256_nfc(original_text),
        "digest_ref": digest_ref,
        "model_version": model_version,
        "session_topic": session_topic,
        **extra,
    }


def to_line(seg: dict) -> str:
    return json.dumps(seg, ensure_ascii=False)


def append_line(path: str, seg: dict) -> None:
    """Append one line. Byte-stable: materialization copies these raw lines verbatim."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(to_line(seg) + "\n")


class ReplayIndex:
    def __init__(self):
        self.by_hash: dict[str, dict] = {}
        self.dropped: list[tuple[int, str]] = []  # (line_no, reason)

    def lookup(self, content_hash: str) -> Optional[dict]:
        """O(1) point lookup by handle (= content_hash)."""
        return self.by_hash.get(content_hash)

    def __len__(self) -> int:
        return len(self.by_hash)


def replay(path: str) -> ReplayIndex:
    """Rebuild the in-memory index from the valid head of a jsonl file.

    Torn-tail rule: at the first line that fails JSON parse OR content_hash verify,
    stop trusting and drop that line and everything after it.
    Idempotent dedup: a content_hash already in the index is skipped (structural key);
    replay NEVER re-runs tiering.
    """
    idx = ReplayIndex()
    p = Path(path)
    if not p.exists():
        return idx
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            line = json.loads(raw)
        except json.JSONDecodeError:
            idx.dropped.append((lineno, "json_parse_fail"))
            break
        stored = line.get("content_hash", "")
        if stored != sha256_nfc(line.get("original_text", "")):
            idx.dropped.append((lineno, "content_hash_mismatch"))
            break
        if stored in idx.by_hash:
            idx.dropped.append((lineno, "dup_skip_idempotent"))
            continue
        line["_line_no"] = lineno
        idx.by_hash[stored] = line
    return idx


def materialize(archive_path: str, new_active_path: str) -> ReplayIndex:
    """Archive -> new active file = byte-stable copy (verbatum lines, no re-encoding).

    Regeneration only happens when original_text actually changes.
    Returns the replay of the new active file (single replay path — never a second one).
    """
    src_raw = Path(archive_path).read_text(encoding="utf-8").splitlines()
    with open(new_active_path, "w", encoding="utf-8") as f:
        for raw in src_raw:
            if raw.strip():
                f.write(raw + "\n")
    return replay(new_active_path)


def verify_anchor_closure(anchor_manifest: Iterable, replay_index: ReplayIndex) -> list[str]:
    """SINGLE implementation of the zero-dangling-anchor invariant.

    anchor_manifest: iterable of segment_ids (content_hash) or {"segment_id": ...} dicts.
    A segment is dangling iff its anchor is not backed by a content_hash-verified jsonl
    line present in replay_index.

    Empty-input semantics (locked):
      - replay_index empty   -> raise ValueError (check did not run; explicit failure,
                                never a silent green []
      - anchor_manifest empty -> return [] (no anchors; legal)
    """
    if len(replay_index) == 0:
        raise ValueError(
            f"verify_anchor_closure: replay_index is empty ({len(replay_index)} segments) — "
            "closure check did not run; expected non-empty after replay/materialization"
        )
    dangling: list[str] = []
    for ref in anchor_manifest:
        sid = ref["segment_id"] if isinstance(ref, dict) else ref
        if replay_index.lookup(sid) is None:
            dangling.append(sid)
    return sorted(set(dangling))


def digest_regen_key(seg: dict, model_version: str, session_topic: str) -> tuple:
    """Two-level idempotent key, level 2: digest is a pure function of this triple.
    Same triple -> no recompute; model_version bump = intentional regeneration."""
    return (seg["content_hash"], model_version, session_topic)


# ── settled P0 原文沉盘（owner 裁定：settled 折出存根 + 旧 P0 落盘可复活可审计）────
# 不变量：文件名 <seq>.p0 与 compact-ledger 存根 #seq 一一对应（重开按 seq 捞原文）。
# sink = 原文写盘（幂等覆盖，字节稳定）；restore = 按 seq 读回（缺失 → None，fail-open）。
# 零 .db、零宿主面：只写 supercompress 侧车同目录的 <seq>.p0 文件（GC 联动同前缀删）。

def p0_sink_path(base_dir: str, seq: int, sid: str = "") -> str:
    """settled P0 原文落盘点 = <base_dir>/<sid>.p0.<seq>（P5 跨 session 隔离）。

    sid = session 前缀（如 `20261013_092849_4af071`），使不同 session 的沉盘文件
    同名 seq 不冲突——共享 store_dir 下多个 session 各自 seq 从 0 起，无 sid 前缀
    会跨 session 同名覆盖（restore 捞回别的 session 的内容 = 数据损坏）。sid 空
    （旧调用面/测试无 sid）→ 退化为 <seq>.p0（兼容既有文件，fail-open）。"""
    if sid:
        return os.path.join(base_dir, f"{sid}.p0.{seq}")
    return os.path.join(base_dir, f"{seq}.p0")


def sink_p0(base_dir: str, seq: int, text: str, kind: str = "adhoc",
            sid: str = "") -> bool:
    """把 settled P0 块原文沉盘（<sid>.p0.<seq>，P5 跨 session 隔离）。幂等：
    同 sid+seq 重复写覆盖（字节稳定）。目录缺失 / 写失败 → 只告警不炸宿主
    （fail-open，与账本侧车同口径）；返回是否落盘成功。"""
    target = p0_sink_path(base_dir, seq, sid)
    try:
        os.makedirs(base_dir, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(str(kind) + "\n")          # 首行 = kind（审计面：kanban 半 / adhoc 纯）
            f.write(str(text or ""))
            f.flush()
            os.fsync(f.fileno())
        return True
    except OSError:
        return False


def restore_p0(base_dir: str, seq: int, sid: str = "") -> Optional[tuple]:
    """按 sid+seq 捞回 settled P0 原文（复活面）。返回 (kind, text)；缺失 → None
    （fail-open）。sid 空 → 兼容旧 <seq>.p0 文件。"""
    target = p0_sink_path(base_dir, seq, sid)
    try:
        with open(target, "r", encoding="utf-8", newline="") as f:
            body = f.read()
    except (FileNotFoundError, OSError):
        return None
    lines = body.split("\n", 1)
    kind = lines[0] if lines and lines[0] else "adhoc"
    text = lines[1] if len(lines) > 1 else ""
    return (kind, text)


# ── R5（owner 批次 A #2）：trimmed P1 块原文沉盘 / 复活（与 P0 面同形对称）──
def p1_trim_path(base_dir: str, seq: int, sid: str = "") -> str:
    """trimmed P1 块原文落盘点 = <base_dir>/<sid>.p1.<seq>（P5 跨 session 隔离，
    与 <sid>.p0.<seq> 同前缀对称不撞名）。sid 空 → 兼容旧 <seq>.p1。"""
    if sid:
        return os.path.join(base_dir, f"{sid}.p1.{seq}")
    return os.path.join(base_dir, f"{seq}.p1")


def sink_p1(base_dir: str, seq: int, text: str, sid: str = "") -> bool:
    """把 trimmed P1 块原文沉盘（<sid>.p1.<seq>，P5 跨 session 隔离）。幂等：
    同 sid+seq 重复写覆盖（字节稳定）。目录缺失 / 写失败 → fail-open
    （调用方只告警不炸宿主）；返回是否落盘成功。P1 块无 kind 审计行（可裁层
    非定案，首行直接原文）——与 sink_p0 的 kind 首行面不同，restore_p1 相应
    只返回 text。"""
    target = p1_trim_path(base_dir, seq, sid)
    try:
        os.makedirs(base_dir, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(str(text or ""))
            f.flush()
            os.fsync(f.fileno())
        return True
    except OSError:
        return False


def restore_p1(base_dir: str, seq: int, sid: str = "") -> Optional[str]:
    """按 sid+seq 捞回 trimmed P1 原文（复活面）。返回 text；缺失 → None
    （fail-open）。sid 空 → 兼容旧 <seq>.p1 文件。"""
    target = p1_trim_path(base_dir, seq, sid)
    try:
        with open(target, "r", encoding="utf-8", newline="") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return None


# ── P5 GC 面（owner 既裁定"保留 30 天"）：过期 session 侧车族清扫 ──────────────
# 宿主不替 supercompress 删侧车（宿主 gateway 自己的 90 天 retention 只管宿主数据），
# 且红线"不改宿主" → 30 天 retention 落在插件侧：宿主 on_session_start 每开 session
# 顺手扫一次 >30 天旧 session，纯 stdlib、零宿主改动、天然分散（每 session 开清一次，
# 不集中爆扫）。与 gc_orphan_pools（带外删除/归档语义，删"源 jsonl 已不在的孤儿池"）
# 是两个独立维度：本面按源 jsonl mtime 判 30 天 TTL，不清带外删的孤儿（那归 gc_orphan_pools）。
RETENTION_DAYS = 30

# 一个 session 的完整侧车族后缀（同住 store_dir、同 sid 前缀，删一个 sid 全族）。
# .audit / .flock 是 ledger 锁 + 审计侧车；.p0.<sid>.<seq> / .p1.<sid>.<seq> 是 P5 隔离沉盘。
_SIDECAR_SUFFIXES: tuple = (
    ".jsonl",                    # 源 store（mtime 判定基准）
    ".compact.json",             # compact-ledger 账本
    ".model.jsonl",             # B 卡 model 旁路
    ".digests.json",            # digest 池
    ".recall_cache.json",       # recall 缓存
    ".flock",                   # ledger flock 锁
    ".audit",                   # 审计侧车
)


def gc_stale_sidecars_in_dir(store_dir: str, now: Optional[float] = None,
                             retention_days: int = RETENTION_DAYS) -> List[str]:
    """P5 GC 面核心算法（直接吃已解析的 store_dir 路径）。

    `gc_stale_sidecars`（宿主接线面）薄壳到这里；engine 按自己已解析的 store_dir
    （hermes_home > legacy 同一变量）调本面，不重新从 hermes_home 派生——legacy
    独立测试路径下 store_dir = _legacy_store_dir，与 GC 面必须同住同目录才不误删。
    判据：源 jsonl `<sid>.jsonl` mtime 超 retention_days → 删该 sid 全侧车族。
    返回实际删除的文件路径列表（空 = 无过期 / 目录不存在）。fail-open 零崩。
    """
    # 派生面后缀白名单（非源 jsonl 的侧车文件，不参与 TTL 判定，跟源 sid 走全族删）。
    # P0.7 碰撞加固 sid（`my_app_sid.bd345c0436.jsonl`，sid 本体含点）的源 jsonl 形仍
    # 以 `.jsonl` 结尾且**不**以 `.model.jsonl` 结尾 → 正确识别为源（旧判据 `if "." in sid`
    # 会把它当派生面误跳 = retention gap 无限增长，第五轮 reviewer minor 实修）。
    _DERIVED_SUFFIXES: tuple = (
        ".model.jsonl", ".compact.json", ".digests.json",
        ".recall_cache.json", ".flock", ".audit",
    )
    if now is None:
        now = time.time()
    root = store_dir
    if not os.path.isdir(root):
        return []
    cutoff = now - retention_days * 86400.0
    removed: List[str] = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    for name in names:
        # 只以源 jsonl `<sid>.jsonl` 为判定基准（mtime 锚点）。源 jsonl = ends .jsonl
        # 且非派生面（`.model.jsonl` 等白名单后缀）。排除 `.p0.<sid>.<seq>`/`.p1.<sid>.<seq>`
        # 沉盘（seq 在 sid 之后，非源判定基准，跟源 sid 走全族删）。
        if not name.endswith(".jsonl"):
            continue
        if name.endswith(".model.jsonl"):
            continue                      # B 卡旁路派生面，非源判定基准
        sid = name[:-len(".jsonl")]      # P0.7 sid 含点号也照算（源 jsonl 形正确）
        jsonl_path = os.path.join(root, name)
        try:
            mtime = os.path.getmtime(jsonl_path)
        except OSError:
            continue                      # stat 失败（并发删/权限）→ 跳过，下轮再试
        if mtime > cutoff:
            continue                      # 30 天内新鲜 → 绝不碰
        # 过期 → 删该 sid 全族（源 jsonl + 所有同 sid 前缀侧车，F1 例外）
        for candidate in [name] + [n for n in names
                                    if n.startswith(sid + ".") and n != name]:
            # F1：排除 ops 审计文件（<sid>.jsonl.bak_* / <sid>.jsonl_cleaned_* 等）——
            # 它们是人工 ops 产出的审计留痕，非 session 侧车，30 天 TTL 不该连带删。
            # 判据：候选 = 源 jsonl（name 本身）放行；侧车候选若含 ".jsonl.bak" 或
            # ".jsonl_cleaned"（即派生自源 jsonl 的 ops 备份命名）→ 跳过。
            if candidate != name and (".jsonl.bak" in candidate or ".jsonl_cleaned" in candidate):
                continue
            target = os.path.join(root, candidate)
            try:
                os.unlink(target)
                removed.append(target)
            except OSError:
                pass                     # 零崩：删不掉的留下轮（fail-open）
    return removed


def gc_stale_sidecars(hermes_home: Optional[str] = None, now: Optional[float] = None,
                      retention_days: int = RETENTION_DAYS) -> List[str]:
    """P5 GC 面宿主接线面（薄壳）：解析 store_dir 后委托 `gc_stale_sidecars_in_dir`。

    hermes_home 给定 → `<hermes_home>/supercompress_store`；缺省 → `_bootstrap.store_dir()`
    （HERMES_HOME env 优先，per-profile 隔离；独立测试回退 legacy）。engine 通常直接调
    `gc_stale_sidecars_in_dir`（喂它已解析的同一 store_dir 变量），本面供宿主/ops 按
    hermes_home 直接触发的接线点保留。
    """
    if hermes_home:
        root = os.path.join(hermes_home, "supercompress_store")
    else:
        from sc._bootstrap import store_dir
        root = store_dir()
    return gc_stale_sidecars_in_dir(root, now=now, retention_days=retention_days)
