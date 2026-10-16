"""Floquet-Magnus / Floquet-Van Vleck expansion of the gate Hamiltonian.

Bock, Ernst & Uhrig, "Floquet Theory and Average Hamiltonian Theory Revisited"
(arXiv:2609.17121), write the propagator of a periodic H(t) = sum_m H_m e^{i m w t} as

    U(t, t0) = e^{-i Lambda(t)} e^{-i F (t - t0)} e^{+i Lambda(t0)}            (3)

and solve the defining equation (8b)

    dLambda/dt = sum_k B_k/k! ( [-i Lambda, H]_k - [i Lambda, F]_k )

in Fourier space, order by order (27, 28a-f), without time integrals. B_k are the
Bernoulli numbers.

Two generalisations make it apply here:

* Several incommensurate frequencies (w_a, w_s, w_p). The Fourier index m becomes the
  term signature sigma = (s, Delta), and m w becomes -Omega_sigma (our bins carry
  e^{-i Omega t}). The recursion is otherwise unchanged:
  -i Omega_sigma Lambda_sigma = RHS_sigma.
* Near-resonant terms. Pure FVV puts every m != 0 term into Lambda, dividing by
  m w. Here 3 w_p = w_s (130 MHz, matrix element / carrier ~ 0.9) and the qubit
  subharmonic (280 MHz) would make the first order diverge. Every signature with
  |Omega| < `window_GHz` is therefore "secular": it goes into F with its residual
  carrier, F(t) = sum_slow F_sigma e^{-i Omega t}. (8b) holds pointwise in t, so it
  holds for this F(t) too. F(t) is then evolved exactly.

Gauges (the paper's Lambda_0 freedom, 28c-f):

* ``"fvv"``: Lambda_sigma = 0 on every slow sigma. This is Floquet-Van Vleck (29, 30),
  the same gauge as `displaced.sw_recursion`.
* ``"aht"``: Lambda(t0 = 0) = 0 at constant |eta|, i.e.
  Lambda_0 = -sum_fast Lambda_sigma. This is average Hamiltonian theory (31).

At constant |eta| the expansion is evaluated on a grid of |eta| and fitted, as in
`levels` and `displaced`, and the pulse supplies eta(t) (the adiabatic step). The
pulse ends have eta = 0, so the kick there is the pump-free Lambda. `replay`
optionally applies e^{+i Lambda(0)} to the initial states and e^{-i Lambda(t_g)} to the
final states, which is (3) for the whole gate.
"""
from __future__ import annotations

from math import factorial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sps

from ladder.levels import MatBins, _add, _comm, _lin, _prune, term_class
from ladder.levels import Entry
from ladder import displaced as D

#: signed Bernoulli numbers B_k (B_1 = -1/2), enough for order 9
BERNOULLI = {0: 1.0, 1: -0.5, 2: 1 / 6, 4: -1 / 30, 6: 1 / 42, 8: -1 / 30}
GAUGES = ("fvv", "aht")


def _scale(bins: MatBins, c: complex) -> MatBins:
    return {s: c * M for s, M in bins.items()}


def fme_expansion(V: MatBins, carrier, window_GHz: float, order: int, gauge: str = "fvv",
                  ) -> Tuple[Dict[int, MatBins], Dict[int, MatBins]]:
    """Eq. (27) solved order by order: ({n: F^(n) bins}, {n: Lambda^(n) bins}), GHz / 1.

    V is H(t) as signature bins (order 1). T[k][q] collects the k-fold nested
    commutators [Lambda, [..., [Lambda, H]]] of total order q, and U[k][q] the same with
    F at the centre; they are built recursively, as in the paper's integration-free
    scheme (App. A)."""
    if gauge not in GAUGES:
        raise ValueError(f"gauge {gauge!r}: expected one of {GAUGES}")
    zero = (0, (0,) * len(next(iter(V))[1]))
    slow = lambda s: abs(carrier(s)) < window_GHz
    F: Dict[int, MatBins] = {}
    L: Dict[int, MatBins] = {}
    T: Dict[int, Dict[int, MatBins]] = {0: {1: V}}
    U: Dict[int, Dict[int, MatBins]] = {0: {}}
    for n in range(1, order + 1):
        parts = [(1.0, V)] if n == 1 else []
        for k in range(1, n):
            Tk = [(1.0, _comm(L[m], T[k - 1][n - m])) for m in range(1, n)
                  if n - m in T[k - 1]]
            Uk = [(1.0, _comm(L[m], U[k - 1][n - m])) for m in range(1, n)
                  if n - m in U[k - 1]]
            T.setdefault(k, {})[n] = _lin(*Tk)
            U.setdefault(k, {})[n] = _lin(*Uk)
            Bk = BERNOULLI.get(k, 0.0)
            if Bk:
                f = Bk / factorial(k)
                parts.append((f * (-1j) ** k, T[k][n]))
                parts.append((-f * (1j) ** k, U[k][n]))
        R = _lin(*parts)
        Fn: MatBins = {}
        Ln: MatBins = {}
        for sig, M in R.items():
            if slow(sig):
                Fn[sig] = M
            else:
                Ln[sig] = (1j / carrier(sig)) * M     # -i Omega Lambda = RHS
        if gauge == "aht" and Ln:
            # Lambda(0) = 0: the zero-signature part cancels the fast parts at t = 0.
            # It is fixed after F^(n) (it first enters at order n + 1).
            L0 = None
            for M in Ln.values():
                L0 = -M if L0 is None else L0 - M
            _add(Ln, zero, L0)
            Ln = _prune(Ln)
        F[n], L[n] = Fn, Ln
        U[0][n] = Fn
    return F, L


# --------------------------------------------------------------------------- models
def _node(job):
    V, _ = D.displaced_hamiltonian(job["freqs_GHz"], job["lam"], job["alpha"], job["g"],
                                   job["dims"], job["x"], disp_window_GHz=None)
    F, L = fme_expansion(V, D.carrier_of(job["freqs_GHz"]), job["window_GHz"],
                         job["order"], job["gauge"])
    return F, (L if job["x"] == 0.0 else None)


def build_fvv_model(freqs_GHz, lam, alpha, g, dims, *, order: int = 3,
                    window_GHz: float = 0.6, gauge: str = "fvv", n_nodes: int = 13,
                    x_max: float = 1.6, workers: int = 0):
    """The expansion at `n_nodes` values of |eta| (in parallel), fitted in |eta|.

    The coefficients of order n are exact polynomials of degree <= 3n in |eta| (g3 only),
    so the fit is exact up to round-off. Returns (model, kick), where kick =
    {n: Lambda^(n) bins} at eta = 0, the pulse ends."""
    from concurrent.futures import ProcessPoolExecutor
    dims = tuple(int(d) for d in dims)
    xs = D._nodes(n_nodes, x_max)
    carrier = D.carrier_of(freqs_GHz)
    job = dict(freqs_GHz=tuple(freqs_GHz), lam=tuple(lam), alpha=tuple(alpha), g=dict(g),
               dims=dims, order=order, window_GHz=window_GHz, gauge=gauge)
    with ProcessPoolExecutor(max_workers=workers or len(xs)) as pool:
        results = list(pool.map(_node, [dict(job, x=float(x)) for x in xs]))
    kick = results[0][1]
    vals: Dict[Tuple[int, Tuple], List] = {}
    for j, (F, _) in enumerate(results):
        for n, bins in F.items():
            for sig, M in bins.items():
                vals.setdefault((n, sig), [None] * len(xs))[j] = M
    ref = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))
    model = D.DisplacedModel(tuple(float(f) for f in freqs_GHz), ref, dims, order,
                             float(window_GHz), harmonics=0, disp_window_GHz=None)
    model.x_max = float(x_max)
    dim = int(np.prod(dims))
    worst = 0.0
    degree = 3 * order + (3 * order) % 2
    for (n, sig), Ms in vals.items():
        s = sig[0]
        Y = np.stack([np.zeros((dim, dim), complex) if M is None else M.toarray() for M in Ms])
        Y = Y.reshape(len(xs), -1)
        mask = np.any(np.abs(Y) > 1e-15, axis=0)
        if not mask.any():
            continue
        powers, ck, r = D._fit(xs, Y[:, mask], s, min(degree, 3 * n + 1))
        worst = max(worst, r)
        for k, row in zip(powers, ck):
            if np.max(np.abs(row)) < 1e-15:
                continue
            full = np.zeros(dim * dim, complex)
            full[mask] = row
            M = sps.csr_matrix(full.reshape(dim, dim))
            M.eliminate_zeros()
            model.entries.append(Entry(n, term_class(sig), sig, s, int(k) - abs(s),
                                       float(carrier(sig)), M))
    model.fit_residual = worst
    return model, kick


def kick_matrix(kick: Dict[int, MatBins], freqs_GHz, t: float, upto: int) -> np.ndarray:
    """Lambda(t) at eta = 0 (pump-free bins only), summed to order `upto`, dense."""
    carrier = D.carrier_of(freqs_GHz)
    out = None
    for n in range(1, upto + 1):
        for sig, M in kick.get(n, {}).items():
            term = M.toarray() * np.exp(-2j * np.pi * carrier(sig) * t)
            out = term if out is None else out + term
    return out


# --------------------------------------------------------------------------- scoring
def replay(cpl, H, t_g: float, kick0: Optional[np.ndarray] = None,
           kick1: Optional[np.ndarray] = None, atol: float = 1e-10, rtol: float = 1e-8,
           nsteps: int = 500000) -> Dict[str, Any]:
    """The pipeline's projected propagator and leakage-aware iSWAP fidelity for the
    QobjEvo `H`, with the frame change (3) applied at both ends when kicks are given:
    initial kets e^{+i Lambda(0)}|k>, final kets e^{-i Lambda(t_g)}|psi>."""
    import qutip as qt
    from scipy.linalg import expm
    options = cpl._qutip_options(atol, rtol, nsteps)
    idx = cpl._subspace_indices(0, 1)
    dims = [list(cpl.dims), [1] * cpl.n_modes]
    K0 = expm(1j * kick0) if kick0 is not None else None
    K1 = expm(-1j * kick1) if kick1 is not None else None
    n_out = int(np.clip(np.ceil(t_g / 5.0), 1, 4000)) + 1
    tlist = np.linspace(0.0, t_g, n_out)
    Umat = np.zeros((4, 4), complex)
    for col, start in enumerate(idx):
        v = np.zeros(cpl.dim, complex)
        v[start] = 1.0
        if K0 is not None:
            v = K0 @ v
        res = qt.sesolve(H, qt.Qobj(v.reshape(-1, 1), dims=dims), tlist, options=options)
        psi = res.states[-1].full().ravel()
        if K1 is not None:
            psi = K1 @ psi
        Umat[:, col] = psi[idx]
    Fz, leak = cpl._iswap_fidelity_from_U(Umat, True)
    return {"F_avg": Fz, "leakage": leak, "U": Umat}


def model_distance(U_red: np.ndarray, U_full: np.ndarray) -> float:
    """1 - leakage-aware average fidelity of the reduced 4x4 block against the full one
    (the full block normalised to a unitary on its support): how differently the two
    models make the same pulse act, independent of how good the gate is."""
    u, _, vh = np.linalg.svd(U_full)
    W = u @ vh
    d = 4
    ov = abs(np.trace(W.conj().T @ U_red)) ** 2
    return 1.0 - float((ov + np.real(np.trace(U_red.conj().T @ U_red))) / (d * (d + 1)))
