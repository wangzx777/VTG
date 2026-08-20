"""Launch HieraMamba with a narrow NumPy/Python randint compatibility shim."""

from __future__ import annotations

import operator
import random
import runpy
import sys
from pathlib import Path


def _integer_index(value: object) -> int:
    """Return an integer index without silently truncating fractional values."""

    try:
        return operator.index(value)
    except TypeError:
        converted = int(value)
        if converted != value:
            raise TypeError(f"randint bound is not integral: {value!r}")
        return converted


_original_randint = random.randint


def _compatible_randint(start: object, end: object) -> int:
    return _original_randint(_integer_index(start), _integer_index(end))


random.randint = _compatible_randint

workspace = Path(__file__).resolve().parents[4]
upstream_repo = workspace / "repos" / "HieraMamba"
upstream_train = upstream_repo / "train.py"
sys.path.insert(0, str(upstream_repo))
print("Applied integral NumPy scalar compatibility for random.randint.")
runpy.run_path(str(upstream_train), run_name="__main__")
