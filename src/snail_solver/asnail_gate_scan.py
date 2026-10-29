r"""``subharmonic_gate_scan`` with the pump walked across the A-SNAIL exchange instead.

Same tune-up, caching, sharding and HDF5 schema; only the column axis differs::

    w_b = w_b0 + delta;   w_p = w_a - w_b = (w_s - w_a) - delta   (branch "below")
    w_s - w_a - w_p = delta                       <- the 1-pump a s^dag beat
    w_s - w_b - 2 w_p = delta                     <- the 2-pump b s^dag beat, tied to it

with ``w_b0 = 2 w_a - w_s`` (3.3 GHz for ``devices/4.0Gate4.7SNAIL_asnail.json``), so
``delta = 0`` puts the gate pump exactly on the A-SNAIL conversion. The B-SNAIL 2-pump
exchange crosses at the same point for every device: ``w_s - w_b = (w_s - w_a) + w_p``.

A separate module rather than an option on ``subharmonic_gate_scan``, so the running-scan
code is untouched: it swaps that module's ``columns_for`` (only ever called in the
parent, before the columns -- which carry ``w_p_GHz`` -- go to the workers) and runs its
``main``. Every CLI flag is the same; ``--branch`` must be ``below``. The axis printed by
``--dry-run`` and the per-column ``A-subharm beat`` are still the A-subharmonic's
(``w_a - 2 w_p``), which is far off on this axis.

    python -m snail_solver.asnail_gate_scan --device devices/4.0Gate4.7SNAIL_asnail.json \
        --branch below --offsets=-0.2:0.2:41 --target-etas 1.0:1.5:6 --keep-origin ...
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Sequence

from snail_solver import subharmonic_gate_scan as sgs


def columns_for(config: Dict[str, Any], offsets_GHz: Sequence[float], *,
                drop_origin: bool = True) -> List[Dict[str, Any]]:
    """``[{delta_GHz, w_p_GHz, ...}]`` with ``w_p = (w_s - w_a) - delta``."""
    from snail_solver.subharmonic_convergence import _freqs, detuning_for_wp
    wa, _wb, ws = _freqs(config)
    out: List[Dict[str, Any]] = []
    for d in offsets_GHz:
        d = float(d)
        if drop_origin and abs(d) < sgs.ORIGIN_EPS_GHz:
            continue
        w_p = (ws - wa) - d
        out.append({"delta_GHz": d, "w_p_GHz": float(w_p),
                    "subharm_beat_MHz": 1.0e3 * (wa - 2.0 * w_p),
                    "delta_sub_GHz": float(detuning_for_wp(config, w_p))})
    return out


def main() -> None:
    argv = sys.argv[1:]
    if "--branch" not in argv or argv[argv.index("--branch") + 1] != "below":
        sys.exit("asnail_gate_scan: pass --branch below (w_b = w_a - w_p is this axis)")
    sgs.columns_for = columns_for
    sgs.main()


if __name__ == "__main__":
    main()
