"""Repo-relative path resolution, so every tool can be called with bare names.

Expected layout::

    SNAIL_Solver/                   <- REPO_ROOT (holds pyproject.toml)
    |-- src/snail_solver/*.py       <- code (this module lives here)
    |-- devices/                    <- device JSONs
    |-- results/                    <- sweep / calibration outputs
    +-- slurm/                      <- SLURM scripts

``--device dev.json`` resolves to ``devices/dev.json``; default outputs land under
``results/``. The root is the first ancestor of this file holding ``pyproject.toml``,
else one holding ``devices/`` or ``slurm/``, else the current working directory (a
non-editable install). ``SNAIL_SOLVER_ROOT`` overrides the search, e.g. to put
outputs on scratch in a batch job (``devices/`` is then expected there too)::

    SNAIL_SOLVER_ROOT=/scratch/$USER/snail python -m snail_solver.run_sweep_zhou ...
"""

from __future__ import annotations

import os
from pathlib import Path

#: Directory holding the package's ``.py`` files.
CODE_DIR: Path = Path(__file__).resolve().parent

_ROOT_ENV_VAR = "SNAIL_SOLVER_ROOT"


def _find_root(start: Path) -> Path:
    """Repository root at or above `start` (see the module docstring for the order)."""
    override = os.environ.get(_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    candidates = (start, *start.parents)
    for d in candidates:
        if (d / "pyproject.toml").is_file():
            return d
    for d in candidates:
        if (d / "devices").is_dir() or (d / "slurm").is_dir():
            return d
    return Path.cwd().resolve()


REPO_ROOT: Path = _find_root(CODE_DIR)
DEVICES_DIR: Path = REPO_ROOT / "devices"
RESULTS_DIR: Path = REPO_ROOT / "results"
SLURM_DIR: Path = REPO_ROOT / "slurm"


def resolve_device(name: str) -> str:
    """`name` if it exists, else ``devices/<name>`` if that exists, else `name`
    unchanged (so the caller's own error surfaces)."""
    p = Path(name)
    if p.exists():
        return str(p)
    cand = DEVICES_DIR / name
    return str(cand) if cand.exists() else str(p)


def in_results(name: str) -> str:
    """Place a relative output name under results/ (absolute paths pass through),
    creating its parent directory."""
    p = Path(name)
    if p.is_absolute():
        out = p
    elif p.parts and p.parts[0] == RESULTS_DIR.name:      # user already wrote results/...
        out = REPO_ROOT / p
    else:
        out = RESULTS_DIR / p
    os.makedirs(out.parent, exist_ok=True)
    return str(out)
