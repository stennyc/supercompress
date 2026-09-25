"""共享测试 helpers（按 block 拆分的测试文件公用；fixtures 在 conftest.py）。

pytest prepend import mode：tests/ 目录自动进 sys.path → `helpers`/`fakes` 可平级导入。
"""
from fakes import FakeEndpoint
from sc.aux_provider import AuxProvider
from sc.schema import Segment
from sc.store import JsonlStore
from sc.distill import Window, WindowSeg

TOPIC = "supercompress-dev"
MV = "m1"


def mkseg(store, text, tier="P1", prov="model", structural=False, kind="segment",
          digest_ref="pending", goal=None, phase_id=None, state=None, open_tasks=None):
    s = Segment(kind=kind, tier=tier, provenance=prov, original_text=text,
                session_topic=TOPIC, model_version=MV, digest_ref=digest_ref,
                structural_p0=structural, goal=goal, phase_id=phase_id, state=state,
                open_tasks=open_tasks or [])
    store.append(s)
    return s


def complete_digest(aux, seg):
    aux.end_turn()
    d = aux.ensure_digest(seg)
    assert d is not None, "digest 应已生成"
    return d


def mk_window(cap, segs):
    return Window(cap, [WindowSeg(s) for s in segs])
