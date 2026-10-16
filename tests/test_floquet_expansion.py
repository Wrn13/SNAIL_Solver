"""The FVV expansion (Bock, Ernst & Uhrig, arXiv:2609.17121) as implemented in
analytical_exploration/floquet: the recursion against the paper's closed forms and a
numerically exact propagator, the effective model against the full Hamiltonian, the
displaced frame, and the stored target."""
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytical_exploration"))

from floquet import expansion as E, fourier as Fo  # noqa: E402

DEVICE = ((3.5, 3.5 + 1.61, 4.7), (0.1, 0.1, 1.0), (-0.12, -0.12, 0.0), {3: 0.06})


def _toy(eps=0.05, seed=1, two_freqs=False):
    """A random 4x4 H(t) with one frequency (or two incommensurate ones)."""
    rng = np.random.default_rng(seed)
    m = lambda: rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H0 = m()
    H0 = (H0 + H0.conj().T) / 2
    H = {(0, (0, 0)): eps * H0}
    for sig in ((1, (0, 0)), (2, (0, 0))) + (((0, (1, 0)),) if two_freqs else ()):
        A = eps * m() / sig[0] if sig[0] else eps * m()
        H[sig] = A
        H[Fo.conj(sig)] = A.conj().T
    freqs = (1.0, np.sqrt(2.0))                    # carrier = s * 1 + Delta_0 * sqrt 2
    car = lambda s: s[0] * freqs[0] + s[1][0] * freqs[1]
    return {k: sps.csr_matrix(v) for k, v in H.items()}, car


def _close(A, B, tol):
    keys = set(A) | set(B)
    z = sps.csr_matrix(next(iter(A.values())).shape, dtype=complex)
    return max(abs((A.get(k, z) - B.get(k, z)).toarray()).max() for k in keys) < tol


@pytest.mark.parametrize("two_freqs", [False, True])
def test_recursion_reproduces_the_closed_forms(two_freqs):
    """Eq. (27) solved order by order equals (28a), (30b), (29b), (30c) and (29c)."""
    H, car = _toy(two_freqs=two_freqs)
    ex = E.floquet_magnus(H, car, 3, window_GHz=1e-9, gauge="fvv")
    assert _close(ex.F[1], E.F1(H, car), 1e-14)
    assert _close(ex.Lambda[1], E.Lambda1(H, car), 1e-14)
    assert _close(ex.F[2], E.F2_fvv(H, car), 1e-14)
    assert _close(ex.Lambda[2], E.Lambda2_fvv(H, car), 1e-13)
    assert _close(ex.F[3], E.F3_fvv(H, car), 1e-13)


def test_stroboscopic_propagator_converges_like_the_paper():
    """Eq. (3) over one period: the error falls with order, FVV below AHT (Fig. 2)."""
    from scipy.linalg import expm
    H, car = _toy(eps=0.05)
    N = 4000
    U = np.eye(4, dtype=complex)
    for j in range(N):
        U = expm(-2j * np.pi * Fo.dense(H, (1.0, 0.0), (j + 0.5) / N) / N) @ U
    errs = {}
    for gauge in ("fvv", "aht"):
        ex = E.floquet_magnus(H, car, 4, window_GHz=1e-9, gauge=gauge)
        errs[gauge] = []
        for n in range(1, 5):
            F = ex.F_upto(n).get((0, (0, 0)))
            L = ex.Lambda_upto(n - 1)
            Lt = lambda t: (Fo.dense(L, (1.0, 0.0), t) if L else np.zeros((4, 4)))
            Uf = expm(-1j * Lt(1.0)) @ expm(-2j * np.pi * F.toarray()) @ expm(1j * Lt(0.0))
            errs[gauge].append(np.abs(Uf - U).max())
    for gauge, e in errs.items():
        assert all(b < a for a, b in zip(e, e[1:])), (gauge, e)
    assert errs["fvv"][-1] < 1e-3 and errs["fvv"][-1] < errs["aht"][-1]


def test_displacement_cancels_the_classical_coupler_drive():
    from floquet.classical import harmonic_balance
    (f, lam, al, g) = DEVICE
    dims = (3, 3, 9)
    b = harmonic_balance(f, lam[2], g, 1.3)
    assert abs(b[2]) > 0.05 and 3 not in b          # subharmonic displaced, 3 w_p = w_s kept
    H = Fo.coefficients(f, lam, al, g, dims, 1.3, displacement=b)
    H0 = Fo.coefficients(f, lam, al, g, dims, 0.0)
    Hb = Fo.coefficients(f, lam, al, g, dims, 1.3)
    el = lambda C, k: complex(C[(k, (0, 0, -1))][1, 0]) if (k, (0, 0, -1)) in C else 0j
    for k in b:                                     # <1_s|H'|0_s>: only the pump-free part
        assert abs(el(H, k) - el(H0, k)) < 1e-9
    assert abs(el(Hb, 2) - el(H0, 2)) > 0.1


def test_order_one_model_with_everything_secular_is_the_full_model():
    """F at order 1 with an infinite window is H_I itself, chirp and offset included."""
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, make_chirp
    from snail_solver.envelope import ENVELOPE_KINDS
    from floquet import model as M, propagator as P
    f = [3.5, 3.5 + 1.61, 4.7]
    cpl = ZhouCoupler(f, 2, {0: 0.1, 1: 0.1}, {3: 0.06}, [3, 3, 4],
                      anharmonicities_GHz={0: -0.12, 1: -0.12})
    env = ENVELOPE_KINDS["sine_power"](amp=1.0, t_g=120.0, m=3, t_rise=60.0)
    cpl.set_pump(PumpTone(w_p_GHz=1.61 + 0.0013, envelope=env, is_eta=True,
                          chirp=make_chirp([0.002, -0.001, 0.0005], 120.0)), normalize_iswap=(0, 1))
    model = M.build(*M.device_of(cpl), order=1, window_GHz=1e3, workers=2)
    H = P.qobjevo(cpl, model)
    for t in (7.3, 41.0, 88.8):
        d = H(t).full() - cpl.hamiltonian_matrix(t)
        assert np.allclose(d - np.eye(len(d)) * d.diagonal().mean(), 0, atol=1e-9)


@pytest.mark.skipif(not os.path.exists(os.path.join(
    ROOT, "results", "drag_curve_5MHz_2026-09-22", "figs_ridge_tracked", "table.txt")),
    reason="stored 5 MHz run not present")
def test_target_loader_matches_table():
    from floquet import target
    assert len(target.columns(check_table=True)) == 71


def test_term_classes_and_names():
    assert Fo.term_class((0, (0, 0, 0))) == "diag"
    assert Fo.term_class((1, (1, -1, 0))) == "exchange"
    assert Fo.term_class((2, (-1, 0, 0))) == "spectator"
    assert Fo.describe((1, (1, 0, -1))) == "eta a s†"
    assert Fo.describe((3, (0, 0, -1))) == "eta^3 s†"
