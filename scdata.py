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
import unicodedata
from pathlib import Path
from typing import Iterable, Optional


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
