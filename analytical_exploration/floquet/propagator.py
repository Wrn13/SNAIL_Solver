"""Eq. (3) for the gate, in the charge frame.

In the charge frame (`frame`) the paper's propagator is

    U_rot(t, t0) = e^{-i Lambda(chi(t))} W(t, t0) e^{+i Lambda(chi(t0))}

and back in the interaction picture

    U_I = R(t) U_rot R(t0)^dag,   R(t) = exp(i sum_k phi_k(t) n_k).

At a constant pump F is static, so W = e^{-i F (t - t0)} and the whole propagator is closed
form (`constant_drive`): eq. (3) exactly, with no time stepping. For the shaped gate,
F(t) = F(|eta(t)|) + sum_k q_k theta'(t) n_k varies slowly (`model`), so W is integrated
numerically (`replay`). The pulse vanishes at both ends, so the kicks are Lambda at eta = 0.

`install` hands the effective Hamiltonian to the unchanged pipeline (tune-up, chevrons,
scoring) through ``ZhouCoupler.to_qutip_hamiltonian``. The pipeline integrates bare
interaction-picture kets, so it gets R (F - phi' . n) R^dag and no kicks.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import scipy.sparse as sps

from floquet.frame import CHARGE, TWO_PI, PumpPhase, number_diagonal, rotation
from floquet.model import EffectiveModel, Term, for_coupler


def _tone(cpl):
    if len(cpl._pump_tones) != 1:
        raise ValueError("a single pump tone is supported")
    return cpl._pump_tones[0]


def _check_range(cpl, model):
    tone = _tone(cpl)
    t_end = getattr(tone.envelope, "t_g", None)
    if t_end:
        peak = max(abs(cpl._eta(tone, t)) for t in np.linspace(0.0, t_end, 401))
        if peak > model.x_max * (1 + 1e-6):
            raise ValueError(f"pulse peak |eta| = {peak:.3f} is beyond the fitted range {model.x_max}")


class _Assembler:
    """sum_j c_j(t) M_j as one sparse mat-vec on the union sparsity pattern."""

    def __init__(self, mats):
        union = sum(abs(m) for m in mats).tocsr()
        union.sort_indices()
        self.pattern = union.nonzero()
        pos = {(r, c): k for k, (r, c) in enumerate(zip(*self.pattern))}
        rows, cols, vals = [], [], []
        for j, m in enumerate(mats):
            m = sps.coo_matrix(m)
            for r, c, v in zip(m.row, m.col, m.data):
                rows.append(pos[(r, c)])
                cols.append(j)
                vals.append(v)
        self.A = sps.csr_matrix((vals, (rows, cols)), shape=(len(self.pattern[0]), len(mats)))


def qobjevo(cpl, model: EffectiveModel, keep: Optional[Callable[[Term], bool]] = None,
            scale: Optional[Dict[Any, float]] = None, frame: str = "rotating"):
    """The effective Hamiltonian as a QuTiP QobjEvo (rad/ns).

    frame="rotating": F(t) = F(|eta(t)|) + sum_k q_k theta'(t) n_k, in the charge frame.
    frame="interaction": R (F - phi' . n) R^dag, for integrating bare kets (the pipeline).
    `scale` multiplies the terms of the given (order, delta) groups (sensitivity runs)."""
    import qutip as qt
    tone = _tone(cpl)
    _check_range(cpl, model)
    dims = [list(cpl.dims), list(cpl.dims)]
    dim = int(np.prod(cpl.dims))
    scale = scale or {}
    terms = model.select(keep)
    mats = [t.matrix * scale.get((t.order, t.delta), 1.0) for t in terms]
    powers = np.array([t.power for t in terms], float)
    occ = number_diagonal(cpl.dims)
    qn = occ @ np.array(CHARGE, float)                   # sum_k q_k n_k per basis state
    extra = [sps.diags(model.frame_diagonal, 0, format="csr"), sps.diags(qn, 0, format="csr")]
    asm = _Assembler(mats + extra + [sps.identity(dim, format="csr")])
    r_idx, c_idx = asm.pattern
    dn = occ[r_idx] - occ[c_idx]                         # n_i - n_j per stored element
    f, w, q = np.array(model.freqs_GHz), model.omega, np.array(CHARGE)
    nt = len(terms)
    phase = PumpPhase(cpl, tone, w)

    def H(t: float, *args, **kw):
        x = abs(cpl._eta(tone, t))
        chi, thd = phase(t)
        c = np.zeros(asm.A.shape[1], complex)
        c[:nt] = x ** powers
        if frame == "rotating":
            c[nt], c[nt + 1] = 1.0, thd
            data = asm.A @ c
        else:
            data = (asm.A @ c) * np.exp(1j * (dn @ (TWO_PI * f * t - q * chi)))
        return qt.Qobj(sps.csr_matrix((data, asm.pattern), shape=(dim, dim)), dims=dims)
    return qt.QobjEvo(H)


def replay(cpl, t_g: float, model: Optional[EffectiveModel] = None, *, kicks: bool = True,
           keep=None, scale=None, atol: float = 1e-10, rtol: float = 1e-8,
           nsteps: int = 500000) -> Dict[str, Any]:
    """The pipeline's 4x4 projected propagator and leakage-aware iSWAP fidelity.

    With `model` None this is the full model. Otherwise the FVV model is integrated in the
    charge frame, U_I = R(t_g) e^{-i Lambda(t_g)} W e^{+i Lambda(0)} R(0)^dag, with the kicks
    to order - 1 when `kicks` is set."""
    import qutip as qt
    from scipy.linalg import expm
    options = cpl._qutip_options(atol, rtol, nsteps)
    idx = cpl._subspace_indices(0, 1)
    qdims = [list(cpl.dims), [1] * cpl.n_modes]
    tlist = np.linspace(0.0, t_g, int(np.clip(np.ceil(t_g / 5.0), 1, 4000)) + 1)
    U = np.zeros((4, 4), complex)
    if model is None:
        H, pre, post = cpl.to_qutip_hamiltonian(), None, None
    else:
        tone = _tone(cpl)
        H = qobjevo(cpl, model, keep=keep, scale=scale)
        f, w, q = np.array(model.freqs_GHz), model.omega, np.array(CHARGE)
        phase = PumpPhase(cpl, tone, w)
        chi0, _ = phase(0.0)
        chig, _ = phase(t_g)
        R0 = rotation(cpl.dims, -q * chi0)
        Rg = rotation(cpl.dims, TWO_PI * f * t_g - q * chig)
        n = model.order - 1 if kicks else 0
        K0 = expm(1j * model.kick_at(chi0, n)) if n > 0 else np.eye(cpl.dim)
        K1 = expm(-1j * model.kick_at(chig, n)) if n > 0 else np.eye(cpl.dim)
        pre = lambda v: K0 @ (R0.conj() * v)
        post = lambda v: Rg * (K1 @ v)
    for col, start in enumerate(idx):
        v = np.zeros(cpl.dim, complex)
        v[start] = 1.0
        if pre:
            v = pre(v)
        psi = qt.sesolve(H, qt.Qobj(v.reshape(-1, 1), dims=qdims), tlist,
                         options=options).states[-1].full().ravel()
        if post:
            psi = post(psi)
        U[:, col] = psi[idx]
    F, leak = cpl._iswap_fidelity_from_U(U, True)
    return {"F_avg": F, "leakage": leak, "U": U}


def constant_drive(freqs_GHz, lam, alpha, g, dims, eta: float, w_p_GHz: float, times,
                   order: int, *, kicks: bool = True, gauge: str = "fvv"):
    """Eq. (3) in closed form for a constant real pump eta at frequency w_p (GHz):

        U_I(t) = R(t) e^{-i Lambda(w t)} e^{-i F t} e^{+i Lambda(0)},   chi = w t, w = 2pi w_p

    The carrier offset dw = w_p - w_nom is exact: chi = w t, and H_0 gains
    -2pi dw sum_k q_k n_k. Matrix exponentials only. Returns ([U_I(t)], expansion)."""
    from scipy.linalg import expm
    from floquet import fourier as Fo
    from floquet.expansion import floquet_magnus
    from floquet.frame import periodic_coefficients
    f = np.asarray(freqs_GHz, float)
    H, _ = periodic_coefficients(Fo.coefficients(freqs_GHz, lam, alpha, g, dims, eta), f, dims)
    dw = float(w_p_GHz) - Fo.pump_frequency(f)
    occ = number_diagonal(dims)
    q = np.array(CHARGE)
    H[0] = H[0] + np.diag((-TWO_PI * dw * (occ @ q)).astype(complex))
    w = TWO_PI * float(w_p_GHz)
    ex = floquet_magnus(H, w, order, gauge=gauge)
    F = ex.F_upto(order)
    n = order - 1 if kicks else 0
    K0 = expm(1j * ex.Lambda_at(0.0, n)) if n > 0 else np.eye(len(F))
    out = []
    for t in times:
        K1 = expm(-1j * ex.Lambda_at(t, n)) if n > 0 else np.eye(len(F))
        R = rotation(dims, TWO_PI * f * t - q * w * t)
        out.append(R[:, None] * (K1 @ expm(-1j * F * t) @ K0))
    return out, ex


def gate_distance(U: np.ndarray, U_ref: np.ndarray) -> float:
    """||U - e^{i phi} U_ref||_F^2 / (d + 1) on the 4x4 block, with the best global phase.
    It is zero only for identical blocks, leakage included, and equals the average gate
    infidelity between them for a small unitary difference."""
    ov = np.trace(U_ref.conj().T @ U)
    ph = ov / abs(ov) if abs(ov) > 0 else 1.0
    return float(np.linalg.norm(U - ph * U_ref) ** 2 / 5.0)


def susceptibility(U_plus: np.ndarray, U_minus: np.ndarray, eps: float) -> float:
    """Leading-order average infidelity per (d ln c)^2 between gates with one term scaled
    by (1 +/- eps): (|G|^2 - |Tr G|^2 / d) / (d + 1), with G = U0^dag dU/d ln c."""
    dU = (U_plus - U_minus) / (2 * eps)
    u, _, vh = np.linalg.svd((U_plus + U_minus) / 2)
    G = (u @ vh).conj().T @ dU
    return float((np.linalg.norm(G) ** 2 - abs(np.trace(G)) ** 2 / 4) / 5)


def propagate_columns(cpl, H, times, init: Optional[np.ndarray] = None, atol=1e-10,
                      rtol=1e-8, nsteps=500000):
    """Full states (dim x 4) at each time, from the computational kets (or `init`'s columns)."""
    import qutip as qt
    options = cpl._qutip_options(atol, rtol, nsteps)
    idx = cpl._subspace_indices(0, 1)
    dims = [list(cpl.dims), [1] * cpl.n_modes]
    tl = np.concatenate([[0.0], np.asarray(times, float)])
    out = [np.zeros((cpl.dim, 4), complex) for _ in times]
    for col, start in enumerate(idx):
        if init is None:
            v = np.zeros(cpl.dim, complex)
            v[start] = 1.0
        else:
            v = np.asarray(init[:, col], complex)
        res = qt.sesolve(H, qt.Qobj(v.reshape(-1, 1), dims=dims), tl, options=options)
        for j, st in enumerate(res.states[1:]):
            out[j][:, col] = st.full().ravel()
    return out


# --------------------------------------------------------------------------- pipeline hook
_STATE: Dict[str, Any] = {}


def install(*, cache_dir: Optional[str] = None, keep=None, **model_kw) -> None:
    """Make every ZhouCoupler in this process (and its forks) solve the FVV effective
    Hamiltonian (interaction picture, no kicks) instead of the full H_I."""
    from snail_solver import zhou_coupler as Z
    if "orig" not in _STATE:
        _STATE["orig"] = Z.ZhouCoupler.to_qutip_hamiltonian
    _STATE.update(cache_dir=cache_dir, keep=keep, kw=model_kw)
    os.environ["FVV_MODEL"] = repr(model_kw)

    def to_qutip_hamiltonian(self, cutoff_GHz: float = np.inf, sparse: bool = True):
        model = for_coupler(self, cache_dir=_STATE["cache_dir"], **_STATE["kw"])
        return qobjevo(self, model, keep=_STATE["keep"], frame="interaction")
    Z.ZhouCoupler.to_qutip_hamiltonian = to_qutip_hamiltonian


def uninstall() -> None:
    from snail_solver import zhou_coupler as Z
    if "orig" in _STATE:
        Z.ZhouCoupler.to_qutip_hamiltonian = _STATE.pop("orig")
