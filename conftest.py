"""Root conftest: ensure the repo root is on sys.path so `sc` and `scdata` are importable.

pytest's prepend import mode already puts the test file's dir (tests/) on the path for
`fakes`, but `sc`/`scdata` live at the repo root — pin that explicitly so the single
authoritative data layer (scdata.py) is reachable from the sc package and tests.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
