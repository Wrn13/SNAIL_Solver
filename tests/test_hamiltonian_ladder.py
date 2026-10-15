"""Hamiltonian ladder: the reduced engine must reduce to the full model, and the loader
must reproduce figs_ridge_tracked/table.txt."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytical_exploration"))

from ladder import engine  # noqa: E402
from ladder.levels import LEVELS, included, term_class  # noqa: E402


def _coupler(levels=(3, 3, 4), delta=-0.14, chirp=(0.002, -0.001, 0.0005)):
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, make_chirp
    from snail_solver.envelope import ENVELOPE_KINDS
    wa = 3.5
    wp = wa / 2 + delta
    cpl = ZhouCoupler([wa, wa + wp, 4.7], 2, {0: 0.1, 1: 0.1}, {3: 0.06}, list(levels),
                      anharmonicities_GHz={0: -0.12, 1: -0.12})
    env = ENVELOPE_KINDS["sine_power"](amp=1.0, t_g=120.0, m=3, t_rise=60.0)
    cpl.set_pump(PumpTone(w_p_GHz=wp + 0.0013, envelope=env, is_eta=True,
                          chirp=make_chirp(list(chirp), 120.0)), normalize_iswap=(0, 1))
    return cpl


def test_order1_all_slow_is_the_full_model():
    """Order 1 with every term slow is H_I itself, truncation, chirp and offset included."""
    cpl = _coupler()
    H = engine.reduced_qobjevo(cpl, "L5", order=1, window_GHz=1e3)
    for t in (7.3, 41.0, 88.8):
        assert np.allclose(H(t).full(), cpl.hamiltonian_matrix(t), atol=1e-10)


def test_levels_are_cumulative():
    combos = [(c, n) for c in ("diag", "exchange", "spectator") for n in range(1, 6)]
    for lo, hi in zip(LEVELS, LEVELS[1:]):
        assert all(included(hi, c, n) for c, n in combos if included(lo, c, n))
    assert all(included("L5", c, n) for c, n in combos)


def test_term_classes():
    assert term_class((0, (0, 0, 0))) == "diag"
    assert term_class((1, (1, -1, 0))) == "exchange"
    assert term_class((-2, (-2, 2, 0))) == "exchange"
    assert term_class((2, (-1, 0, 0))) == "spectator"
    assert term_class((3, (0, 0, -1))) == "spectator"


@pytest.mark.skipif(not os.path.exists(os.path.join(
    ROOT, "results", "drag_curve_5MHz_2026-09-22", "figs_ridge_tracked", "table.txt")),
    reason="stored 5 MHz run not present")
def test_loader_matches_table():
    from ladder import stored
    cols = stored.columns(check_table=True)          # raises on any mismatch
    assert len(cols) == 71
