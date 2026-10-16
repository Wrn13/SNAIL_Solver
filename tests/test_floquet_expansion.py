"""The FVV expansion of Bock, Ernst & Uhrig (arXiv:2609.17121) as implemented in
analytical_exploration/floquet:

* the charge frame turns the gate Hamiltonian into the paper's single-frequency series;
* the recursion (44)-(45) reproduces every closed form of the paper, through App. B;
* eq. (3) converges on a periodic toy model, with FVV below AHT (the paper's Fig. 2);
* eq. (3) in closed form reproduces the full model at weak constant drive;
* the |eta| fit is exact; the stored target loads."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytical_exploration"))

from floquet import expansion as E, fourier as Fo, frame as Fr  # noqa: E402

F_GHZ = [3.5, 3.5 + 1.61, 4.7]
LAM, ALPHA, G = (0.1, 0.1, 1.0), (-0.12, -0.12, 0.0), {3: 0.06}


def _coupler(dims=(3, 3, 4), eta=0.8, dw=0.0013, t_g=500.0):
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone
    from snail_solver.envelope import ConstantPulse
    cpl = ZhouCoupler(F_GHZ, 2, {0: 0.1, 1: 0.1}, {3: 0.06}, list(dims),
                      anharmonicities_GHz={0: -0.12, 1: -0.12})
    tone = PumpTone(w_p_GHz=1.61 + dw, envelope=ConstantPulse(amp=eta, t_g=t_g), is_eta=True)
    cpl.set_pump(tone)
    return cpl, tone


def _toy(eps=0.05, seed=1):
    rng = np.random.default_rng(seed)
    m = lambda: rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H0 = m()
    H = {0: eps * (H0 + H0.conj().T) / 2}
    for k in (1, 2):
        A = eps * m() / k
        H[k], H[-k] = A, A.conj().T
    return H, 2 * np.pi


def _dev(dims=(3, 3, 4), eta=0.8):
    bins = Fo.coefficients(F_GHZ, LAM, ALPHA, G, dims, eta)
    return Fr.periodic_coefficients(bins, F_GHZ, dims)


def _diff(A, B):
    if isinstance(A, dict):
        return max(abs(A.get(k, 0) - B.get(k, 0)).max() for k in set(A) | set(B))
    return abs(A - B).max()


def test_charge_frame_is_the_full_hamiltonian():
    """sum_m H_m e^{i m chi} + q.n theta' = R^dag H_I R + phi'.n, up to the c-number."""
    dims = (3, 3, 4)
    cpl, tone = _coupler(dims)
    H, w = _dev(dims)
    occ = Fr.number_diagonal(dims)
    q = np.array(Fr.CHARGE)
    for t in (3.3, 47.1, 133.0):
        chi, thd = Fr.pump_phase(cpl, tone, t, w)
        Hrot = sum(M * np.exp(1j * m * chi) for m, M in H.items()) + np.diag(occ @ (q * thd))
        r = Fr.rotation(dims, 2 * np.pi * np.array(F_GHZ) * t - q * chi)
        ref = r.conj()[:, None] * cpl.hamiltonian_matrix(t) * r[None, :] \
            + np.diag(occ @ (2 * np.pi * np.array(F_GHZ) - q * (w - thd)))
        d = Hrot - ref
        assert np.allclose(d - np.eye(len(d)) * d.diagonal().mean(), 0, atol=1e-9)


@pytest.mark.parametrize("which", ["toy", "device"])
def test_recursion_reproduces_every_closed_form(which):
    """(44)-(45) against (28a-b), (29b-c), (30c), (47) and (48)."""
    H, w = _toy() if which == "toy" else _dev()
    ex = E.floquet_magnus(H, w, 4)
    tol = 1e-12 * max(1.0, max(abs(M).max() for M in H.values()))
    assert _diff(ex.F[1], E.F1(H, w)) < tol
    assert _diff(ex.Lambda[1], E.Lambda1(H, w)) < tol
    assert _diff(ex.F[2], E.F2(H, w)) < tol
    assert _diff(ex.Lambda[2], E.Lambda2(H, w)) < tol
    assert _diff(ex.F[3], E.F3(H, w)) < tol
    assert _diff(ex.F[4], E.F4(H, w)) < tol
    assert _diff(ex.Lambda[3], E.Lambda3(H, w)) < tol


def test_propagator_converges_like_the_paper():
    """Eq. (3) over one period of the toy: the error falls with order, FVV below AHT."""
    from scipy.linalg import expm
    H, w = _toy()
    T, N = 2 * np.pi / w, 4000
    U = np.eye(4, dtype=complex)
    for j in range(N):
        t = (j + 0.5) * T / N
        U = expm(-1j * sum(M * np.exp(1j * m * w * t) for m, M in H.items()) * T / N) @ U
    errs = {}
    for gauge in E.GAUGES:
        ex = E.floquet_magnus(H, w, 4, gauge=gauge)
        errs[gauge] = [np.abs(ex.propagator(T, 0.0, n) - U).max() for n in range(1, 5)]
    for gauge, e in errs.items():
        assert all(b < a for a, b in zip(e, e[1:])), (gauge, e)
    assert errs["fvv"][-1] < 1e-3 and errs["fvv"][-1] < errs["aht"][-1]


def test_closed_form_propagator_matches_the_full_model_at_weak_drive():
    """Eq. (3) in closed form (no time stepping) against sesolve of the full H_I."""
    from floquet import propagator as P
    dims = (3, 3, 4)
    cpl, tone = _coupler(dims, eta=0.1)
    times = [40.0, 80.0]
    full = P.propagate_columns(cpl, cpl.to_qutip_hamiltonian(), times)
    idx = cpl._subspace_indices(0, 1)
    Us, _ = P.constant_drive(F_GHZ, LAM, ALPHA, G, dims, 0.1, float(tone.w_p_GHz), times, 4)
    for U, S in zip(Us, full):
        assert P.gate_distance(U[np.ix_(idx, idx)], S[idx, :]) < 1e-4


def test_fit_in_eta_is_exact():
    from floquet import model as M
    dims = (3, 3, 3)
    model = M.build(F_GHZ, LAM, ALPHA, G, dims, order=3)
    assert model.fit_residual < 1e-9
    ex, _ = M.expansion_at(F_GHZ, LAM, ALPHA, G, dims, 0.77, 3)
    assert np.allclose(model.F_at(0.77), ex.F_upto(3), atol=1e-9)


def test_process_names():
    from floquet import model as M
    assert M.describe((1, -1, 0)) == "eta a b†"          # the exchange
    assert M.describe((1, 0, -1)) == "eta a s†"          # pump-assisted a<->s conversion
    assert M.describe((0, 0, -1)) == "eta^3 s†"          # 3 w_p = w_s
    assert Fr.charge((1, (1, -1, 0))) == 0 and Fr.charge((2, (0, 0, -1))) == -1


@pytest.mark.skipif(not os.path.exists(os.path.join(
    ROOT, "results", "drag_curve_5MHz_2026-09-22", "figs_ridge_tracked", "table.txt")),
    reason="stored 5 MHz run not present")
def test_target_loader_matches_table():
    from floquet import target
    assert len(target.columns(check_table=True)) == 71


def test_shaped_pulse_replay_converges_and_ignores_the_signed_zero_at_the_ends():
    """A sine^3 pulse evaluates to -0 or -1e-18 at its ends, where arg eta flips by pi.
    The frame phase must not follow that flip: the FVV gate converges to the full one."""
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone
    from snail_solver.envelope import ENVELOPE_KINDS
    from floquet import model as M, propagator as P
    dims = (3, 3, 4)
    cpl = ZhouCoupler(F_GHZ, 2, {0: 0.1, 1: 0.1}, {3: 0.06}, list(dims),
                      anharmonicities_GHz={0: -0.12, 1: -0.12})
    env = ENVELOPE_KINDS["sine_power"](amp=0.1, t_g=100.0, m=3, t_rise=50.0)
    tone = PumpTone(w_p_GHz=1.61, envelope=env, is_eta=True)
    cpl.set_pump(tone)
    chi0, _ = Fr.PumpPhase(cpl, tone, 2 * np.pi * 1.61)(0.0)
    assert abs(chi0) < 1e-9
    full = P.replay(cpl, 100.0)
    d = [P.gate_distance(P.replay(cpl, 100.0, M.build(F_GHZ, LAM, ALPHA, G, dims, order=o),
                                  kicks=False)["U"], full["U"]) for o in (2, 3)]
    assert d[1] < d[0] < 2e-3
