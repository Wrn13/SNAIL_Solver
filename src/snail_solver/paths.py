"""Repo-relative path resolution, so every tool can be called with bare names.

The expected layout is::

    SNAIL_Solver/                   <- REPO_ROOT (holds pyproject.toml)
    |-- src/snail_solver/*.py       <- code (this module lives here)
    |-- devices/                    <- device JSONs
    |-- results/                    <- sweep / calibration outputs
    +-- slurm/                      <- SLURM scripts

``--device dev.json`` resolves to ``devices/dev.json`` and default outputs land
under ``results/``.

The root is found by walking up from this file to the first ancestor holding a
``pyproject.toml``, so it works from any working directory. Two fallbacks cover
the cases where that fails:

* An ancestor holding ``devices/`` or ``slurm/`` -- for a source tree without
  packaging metadata.
* The current working directory -- for a non-editable install, where the code
  sits in ``site-packages`` and has no repo above it.

Set ``SNAIL_SOLVER_ROOT`` to override the search entirely. That is the usual way
to put outputs on fast scratch storage in a batch job::

    SNAIL_SOLVER_ROOT=/scratch/$USER/snail python -m snail_solver.run_sweep_zhou ...

Note that ``devices/`` is then expected under that root too, so pass an explicit
path to ``--device`` (or copy the JSONs across) when overriding.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Directory holding the package's ``.py`` files.
CODE_DIR: Path = Path(__file__).resolve().parent

_ROOT_ENV_VAR = "SNAIL_SOLVER_ROOT"


def _find_root(start: Path) -> Path:
    """Locate the repository root above `start`.

    Parameters
    ----------
    start : Path
        Directory to start the upward walk from (inclusive).

    Returns
    -------
    Path
        The repository root, or the current working directory if no marker is
        found (which is the sane default for an installed-to-site-packages copy).
    """
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
    """Resolve a device path: use it if it exists, else look under devices/.

    Parameters
    ----------
    name : str
        A path or a bare filename.

    Returns
    -------
    str
        The resolved path (unchanged if already valid or if not found, so the
        caller's own error surfaces).
    """
    p = Path(name)
    if p.exists():
        return str(p)
    cand = DEVICES_DIR / name
    return str(cand) if cand.exists() else str(p)


def in_results(name: str) -> str:
    """Place a relative output name under results/ (absolute paths pass through).

    Ensures the parent directory exists.

    Parameters
    ----------
    name : str
        Output path or bare name.

    Returns
    -------
    str
        The resolved output path.
    """
    p = Path(name)
    if p.is_absolute():
        out = p
    elif p.parts and p.parts[0] == RESULTS_DIR.name:      # user already wrote results/...
        out = REPO_ROOT / p
    else:
        out = RESULTS_DIR / p
    os.makedirs(out.parent, exist_ok=True)
    return str(out)
