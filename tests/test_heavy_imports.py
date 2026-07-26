"""The transformers import guard, and the rule that every call site uses it.

Background — this is a real failure, not a hypothetical one. `main.py` starts
the asr-loader thread and the translation thread back to back. Both reach into
`transformers`, which is a lazy package: the name is resolved by importing the
submodule that defines it. Both wanted a name out of
`transformers.models.auto.modeling_auto`, and the ASR side deterministically lost:

    ImportError: cannot import name 'AutoModelForRNNT' from 'transformers'

Reproduced 6 times out of 6 before the guard and 0 out of 6 after. The damage
was silent — nemotron "failed to load", the app fell back to whisper, and the
only sign was a status line that scrolls away in seconds.

The expensive half of that (actually importing transformers twice in two
threads) is not re-run here: it costs ~4.6 s and would more than double the
suite. What is checked instead is the invariant that keeps it fixed — the lock
serialises, and no call site under `src/` imports transformers outside it. A
fourth unguarded call site is exactly how this bug comes back.
"""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from typing import List, Tuple

import pytest

import heavy_imports

SRC = Path(__file__).resolve().parent.parent / "src"


# ----------------------------------------------------------------------
# The guard itself
# ----------------------------------------------------------------------


def test_guard_serialises_two_threads():
    order: List[str] = []
    inside = threading.Event()
    second_started = threading.Event()

    def slow():
        with heavy_imports.transformers():
            inside.set()
            order.append("first-in")
            # Give the other thread every chance to barge in.
            second_started.wait(timeout=1.0)
            time.sleep(0.05)
            order.append("first-out")

    def fast():
        inside.wait(timeout=1.0)
        second_started.set()
        with heavy_imports.transformers():
            order.append("second-in")

    a = threading.Thread(target=slow)
    b = threading.Thread(target=fast)
    a.start(); b.start()
    a.join(timeout=5); b.join(timeout=5)

    assert order == ["first-in", "first-out", "second-in"]


def test_guard_is_reentrant():
    """A guarded import nested inside another must not deadlock."""
    with heavy_imports.transformers():
        with heavy_imports.transformers():
            pass


def test_guard_releases_on_exception():
    with pytest.raises(RuntimeError):
        with heavy_imports.transformers():
            raise RuntimeError("boom")
    # Still acquirable from another thread — i.e. the lock was released.
    done = threading.Event()

    def worker():
        with heavy_imports.transformers():
            done.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    assert done.is_set()


# ----------------------------------------------------------------------
# The rule: nobody imports transformers outside the guard
# ----------------------------------------------------------------------


def _is_guard(node: ast.With) -> bool:
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "transformers"
            and isinstance(func.value, ast.Name)
            and func.value.id == "heavy_imports"
        ):
            return True
    return False


def _transformers_imports(tree: ast.AST) -> List[Tuple[int, bool]]:
    """(line, guarded) for every `from transformers[...] import …` in the tree."""
    found: List[Tuple[int, bool]] = []

    def walk(node: ast.AST, guarded: bool) -> None:
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root == "transformers":
                found.append((node.lineno, guarded))
            return
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "transformers" for a in node.names):
                found.append((node.lineno, guarded))
            return
        inner = guarded or (isinstance(node, ast.With) and _is_guard(node))
        for child in ast.iter_child_nodes(node):
            walk(child, inner)

    walk(tree, False)
    return found


def _all_sites() -> List[Tuple[Path, int, bool]]:
    sites: List[Tuple[Path, int, bool]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, guarded in _transformers_imports(tree):
            sites.append((path, line, guarded))
    return sites


def test_every_transformers_import_is_guarded():
    unguarded = [
        f"{path.relative_to(SRC)}:{line}"
        for path, line, guarded in _all_sites()
        if not guarded
    ]
    assert not unguarded, (
        "transformers imported outside heavy_imports.transformers(): "
        + ", ".join(unguarded)
        + " — two threads racing on the same lazy submodule make the loser fail "
        "with a bogus ImportError; see src/heavy_imports.py"
    )


def test_the_rule_has_something_to_check():
    """Guards against the previous test passing because the call sites moved."""
    assert len(_all_sites()) >= 3
