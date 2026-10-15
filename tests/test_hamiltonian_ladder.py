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


def _device():
    return (3.5, 3.5 + 1.61, 4.7), (0.1, 0.1, 1.0), (-0.12, -0.12, 0.0), {3: 0.06}, (3, 3, 9)


def test_undisplaced_frame_is_the_plain_first_order_operator():
    """With nothing displaced, H' is levels' first-order V (up to the eta^3 c-number)."""
    import scipy.sparse as sps
    from ladder import displaced as D
    from ladder.levels import matrix_expansion
    f, lam, al, g, dims = _device()
    V0, b = D.displaced_hamiltonian(f, lam, al, g, dims, 1.3, disp_window_GHz=None)
    ref = matrix_expansion(f, lam, al, g, dims, 1.3, 1, 1e6)[1]
    assert b == {}
    Z = sps.csr_matrix((81, 81), dtype=complex)
    for k in set(V0) | set(ref):
        d = (V0.get(k, Z) - ref.get(k, Z)).toarray()
        assert np.allclose(d - np.eye(81) * d.diagonal().mean(), 0, atol=1e-12)


def test_displacement_cancels_the_fast_linear_coupler_drive():
    """Harmonic balance: the pump-driven classical drive on the coupler is cancelled.

    What remains of <1_s|H'|0_s> on the qubit vacuum is the pump-independent
    normal-ordering term of X_q^3 (3 g3 sum lam^2), the same at eta = 0."""
    from ladder import displaced as D
    f, lam, al, g, dims = _device()
    V, b = D.displaced_hamiltonian(f, lam, al, g, dims, 1.3)
    V0, _ = D.displaced_hamiltonian(f, lam, al, g, dims, 0.0)
    Vb, _ = D.displaced_hamiltonian(f, lam, al, g, dims, 1.3, disp_window_GHz=None)
    assert abs(b[2]) > 0.05                    # the SNAIL subharmonic response is displaced
    assert 3 not in b                          # 3 w_p = w_s stays slow
    el = lambda bins, k: complex(bins[(k, (0, 0, -1))][1, 0]) if (k, (0, 0, -1)) in bins else 0j
    for k in b:
        assert abs(el(V, k) - el(V0, k)) < 1e-9
    assert abs(el(Vb, 2) - el(V0, 2)) > 0.1    # it was there before displacing
