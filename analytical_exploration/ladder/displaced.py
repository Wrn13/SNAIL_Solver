"""The ladder in the frame displaced by the coupler's classical pump response.

The plain expansion (`levels.matrix_expansion`) treats every pump-driven term as a
perturbation of the undriven vacuum. One of them is not small: the pump drives the coupler
linearly, n g_n lam_s [x_cl^(n-1)]_k s^dag (the SNAIL subharmonic eta^2 s^dag at
k = 2, about 0.3 GHz at a 1.5 GHz detuning for |eta| = 1.3). That is a classical
displacement of the coupler, and expanding it in powers climbs the coupler ladder, where
sqrt(n) makes the ratio grow. Orders 2-5 at delta = -140 MHz do not settle.

Here that response is removed exactly. With D = exp(beta s^dag - beta* s),

    H' = D^dag H_I D - i D^dag dD/dt
       = sum_n g_n (X_q + x_cl(t))^n + H_anh - i (beta' s^dag - beta'* s)

where X_q is the quantum flux (truncated ladder matrices, as ZhouCoupler builds it), and
x_cl(t) = eta e^{-i w_p t} + lam_s beta e^{-i w_s t} + c.c. is the classical flux. The
displacement is a sum of pump harmonics, beta e^{-i w_s t} = sum_k b_k e^{-i k w_p t},
fixed by harmonic balance at constant |eta|:

    b_k = F_k / Omega_k,   F_k = sum_n n g_n lam_s [x_cl^(n-1)]_k,   Omega_k = k w_p - w_s

so the adiabatic part of -i beta' s^dag, -Omega_k b_k s^dag, cancels the linear drive at
every displaced harmonic. What the cancellation leaves out, the envelope and chirp
derivative of b_k(eta(t)) (the non-adiabatic part), is evolved exactly in `qobjevo`.
Harmonics with |Omega_k| < `disp_window_GHz` are NOT displaced: they stay in H' as slow
linear drives. The default equals the SW slow window, so 3 w_p = w_s (k = 3, carrier
0.55 + 3 delta GHz) is never displaced: near its resonance the classical response is large
and the harmonic balance loses its solution.

H' is identical to H_I in the untruncated space. In the truncated space it is the same
Hamiltonian truncated after the displacement instead of before, so `identity_check`
(no SW, every term evolved) measures what that costs against the stored fidelity.

Since eta(0) = eta(t_g) = 0, every b_k is zero at the pulse ends and the displaced and
bare frames coincide there: U' = U, and the pipeline's scoring is unchanged.

SW then runs on H' exactly as in `levels.matrix_expansion`, with the whole of H' as the
first-order perturbation. Its coefficients are no longer finite polynomials in |eta| (b_k
is a power series), so they are fitted with a high-degree polynomial in |eta| and the fit
residual is reported.
"""
from __future__ import annotations

import cmath
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from math import comb, factorial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps

from ladder.levels import (LEVELS, Entry, MatBins, ReducedModel, _add, _comm, _letters,
                           _lin, _prune, term_class)

TWO_PI = 2.0 * np.pi
COUPLER = 2
#: highest pump harmonic carried by the classical flux
DEFAULT_HARMONICS = 6
#: displace only what the SW treats as fast; slower harmonics stay in H' exactly
DEFAULT_DISP_WINDOW_GHZ = 0.6
DEFAULT_ORDER = 3
DEFAULT_WINDOW_GHZ = 0.6
#: largest |eta| the fit is anchored on. It must stay below the fold of the classical
#: response (|eta| ~ 1.475 for the g3-only 6Gate4.7SNAIL device), where harmonic balance
#: loses its solution and b_k(|eta|) has a square-root singularity.
X_MAX = 1.4
#: bins whose largest element is under this (GHz) are dropped from H' and from every SW
#: order: 10 kHz over a 130 ns gate is a phase of ~1e-2 rad, an infidelity of ~1e-5
DEFAULT_TOL_GHZ = 0.0       # dropping makes a bin flicker between nodes and spoils the fit


# --------------------------------------------------------------------------- classical
def _conv(p: Dict[int, complex], q: Dict[int, complex], K: int) -> Dict[int, complex]:
    out: Dict[int, complex] = {}
    for i, u in p.items():
        for j, v in q.items():
            if abs(i + j) <= 2 * K:
                out[i + j] = out.get(i + j, 0.0) + u * v
    return out


def _power(x: Dict[int, complex], m: int, K: int) -> Dict[int, complex]:
    out: Dict[int, complex] = {0: 1.0}
    for _ in range(m):
        out = _conv(out, x, K)
    return out


def classical_flux(b: Dict[int, complex], eta: float, lam_s: float) -> Dict[int, complex]:
    """x_cl's pump harmonics {k: c_k}, x_cl = sum_k c_k e^{-i k w_p t} (real field)."""
    c: Dict[int, complex] = {1: complex(eta), -1: complex(eta)}
    for k, v in b.items():
        c[k] = c.get(k, 0.0) + lam_s * v
        c[-k] = c.get(-k, 0.0) + lam_s * np.conj(v)
    return {k: v for k, v in c.items() if v != 0}


def linear_drive(c: Dict[int, complex], g: Dict[int, float], lam_s: float,
                 K: int) -> Dict[int, complex]:
    """F_k: the coefficient of s^dag e^{-i (k w_p - w_s) t} in sum_n g_n x_cl^n (GHz)."""
    F: Dict[int, complex] = {}
    for n, gn in g.items():
        for k, v in _power(c, int(n) - 1, K).items():
            F[k] = F.get(k, 0.0) + int(n) * gn * lam_s * v
    return F


def harmonic_balance(freqs_GHz, lam_s: float, g: Dict[int, float], eta: float, *,
                     harmonics: int = DEFAULT_HARMONICS,
                     disp_window_GHz: float = DEFAULT_DISP_WINDOW_GHZ,
                     tol: float = 1e-14, max_iter: int = 2000,
                     damping: float = 0.5) -> Dict[int, complex]:
    """{k: b_k} for the displaced harmonics |k| <= `harmonics`, at real pump amplitude."""
    fs = float(freqs_GHz[COUPLER])
    wp = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))
    ks = [k for k in range(-harmonics, harmonics + 1) if abs(k * wp - fs) >= disp_window_GHz]
    b: Dict[int, complex] = {k: 0.0 for k in ks}
    for _ in range(max_iter):
        F = linear_drive(classical_flux(b, eta, lam_s), g, lam_s, harmonics)
        new = {k: F.get(k, 0.0) / (k * wp - fs) for k in ks}
        err = max((abs(new[k] - b[k]) for k in ks), default=0.0)
        b = {k: (1 - damping) * b[k] + damping * new[k] for k in ks}
        if err < tol:
            return b
    raise RuntimeError(f"harmonic balance did not converge at eta={eta} (step {err:.2e})")


# --------------------------------------------------------------------------- operators
def _mul(A: MatBins, B: MatBins) -> MatBins:
    out: MatBins = {}
    for sa, Ma in A.items():
        for sb, Mb in B.items():
            _add(out, (sa[0] + sb[0], tuple(x + y for x, y in zip(sa[1], sb[1]))), Ma @ Mb)
    return _prune(out)


def _quantum_powers(lam, dims, n_max: int) -> Dict[int, MatBins]:
    """{j: bins of X_q^j}: products of the truncated letters, as ZhouCoupler builds X^n."""
    P1: MatBins = {}
    for M, sig, la, pump in _letters(lam, dims):
        if not pump:
            _add(P1, sig, la * M)
    eye = sps.identity(int(np.prod(dims)), dtype=complex, format="csr")
    P = {0: {(0, (0,) * len(dims)): eye}, 1: _prune(P1)}
    for j in range(2, n_max + 1):
        P[j] = _mul(P[j - 1], P[1])
    return P


def _anharmonicity(alpha, dims) -> Optional[sps.csr_matrix]:
    anh = None
    for i, dd in enumerate(dims):
        if alpha[i]:
            fac = [sps.identity(e, format="csr") for e in dims]
            fac[i] = sps.diags([0.5 * float(alpha[i]) * k * (k - 1) for k in range(dd)], 0,
                               format="csr")
            A = fac[0]
            for f in fac[1:]:
                A = sps.kron(A, f, format="csr")
            anh = A if anh is None else anh + A
    return None if anh is None else anh.astype(complex)


def displaced_hamiltonian(freqs_GHz, lam, alpha, g, dims, eta: float, *,
                          harmonics: int = DEFAULT_HARMONICS,
                          disp_window_GHz: Optional[float] = DEFAULT_DISP_WINDOW_GHZ,
                          _powers=None) -> Tuple[MatBins, Dict[int, complex]]:
    """H' at constant real |eta| as signature bins (GHz), and the displacement {k: b_k}.

    ``disp_window_GHz=None`` displaces nothing: the bins are then H_I itself, the same as
    `levels.matrix_expansion`'s first-order operator.
    """
    dims = tuple(int(d) for d in dims)
    n_modes = len(dims)
    lam_s = float(lam[COUPLER])
    nmax = max(int(n) for n in g)
    P = _powers or _quantum_powers(lam, dims, nmax)
    b = ({} if disp_window_GHz is None else
         harmonic_balance(freqs_GHz, lam_s, g, eta, harmonics=harmonics,
                          disp_window_GHz=disp_window_GHz))
    c = classical_flux(b, eta, lam_s)
    zero = (0,) * n_modes
    V: MatBins = {}
    for n, gn in g.items():
        n = int(n)
        for m in range(0, n):                   # m = n is a c-number: a global phase
            xm = _power(c, m, 4 * harmonics)
            for sq, M in P[n - m].items():
                for k, v in xm.items():
                    amp = gn * comb(n, m) * v
                    if amp != 0:
                        _add(V, (sq[0] + k, sq[1]), amp * M)
    anh = _anharmonicity(alpha, dims)
    if anh is not None:
        _add(V, (0, zero), anh)
    # the adiabatic part of -i beta' s^dag + h.c.: -Omega_k b_k s^dag - Omega_k b_k* s
    wp = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))
    sdag = tuple(-1 if i == COUPLER else 0 for i in range(n_modes))
    s_lo = tuple(+1 if i == COUPLER else 0 for i in range(n_modes))
    S = P[1][(0, s_lo)] / lam_s                # the truncated s (lam_s folded out)
    for k, v in b.items():
        om = k * wp - float(freqs_GHz[COUPLER])
        _add(V, (k, sdag), -om * v * S.T.conj().tocsr())
        _add(V, (-k, s_lo), -om * np.conj(v) * S)
    return _prune(V), b


def carrier_of(freqs_GHz):
    ref = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))

    def carrier(sig):
        return sum(d * f for d, f in zip(sig[1], freqs_GHz)) + sig[0] * ref
    return carrier


def _drop(bins: MatBins, tol: float) -> MatBins:
    return {s: M for s, M in bins.items() if M.nnz and np.max(np.abs(M.data)) >= tol}


def sw_recursion(V: MatBins, carrier, window_GHz: float, order: int,
                 tol_GHz: float = 0.0) -> Dict[int, MatBins]:
    """tdsw's HighOrderExpansion recursion on matrix bins (as `levels.matrix_expansion`),
    dropping every intermediate bin under `tol_GHz`."""
    V = _drop(V, tol_GHz)
    H: Dict[int, MatBins] = {}
    S: Dict[int, MatBins] = {}
    T: Dict[int, Dict[int, MatBins]] = {0: {1: V}}
    U: Dict[int, Dict[int, MatBins]] = {0: {}}
    for n in range(1, order + 1):
        parts = [(1.0, V)] if n == 1 else []
        for k in range(1, n):
            Tk, Uk = [], []
            for m in range(1, n):
                if n - m in T[k - 1]:
                    Tk.append((1.0, _comm(S[m], T[k - 1][n - m])))
                if n - m in U[k - 1]:
                    Uk.append((1.0, _comm(S[m], U[k - 1][n - m])))
            T.setdefault(k, {})[n] = _lin(*Tk)
            U.setdefault(k, {})[n] = _lin(*Uk)
            parts.append((1.0 / factorial(k), T[k][n]))
            parts.append((1.0 / factorial(k + 1), U[k][n]))
        R = _drop(_lin(*parts), tol_GHz)
        Hn, Sn, Wn = {}, {}, {}
        for sig, M in R.items():
            if abs(carrier(sig)) < window_GHz:
                Hn[sig] = M
            else:
                Sn[sig] = (-1.0 / carrier(sig)) * M
                Wn[sig] = -M
        H[n], S[n] = Hn, Sn
        U[0][n] = Wn
    return H


# --------------------------------------------------------------------------- diagnostics
def fast_ratios(V: MatBins, freqs_GHz, dims, window_GHz: float = DEFAULT_WINDOW_GHZ,
                max_quanta: Optional[int] = None) -> List[Tuple[float, Any, float, float]]:
    """Every FAST bin's strength over its carrier, largest first:
    (ratio, signature, strength GHz, carrier GHz). `max_quanta` restricts it to columns
    (initial states) with at most that many coupler quanta.

    An off-diagonal bin's strength is its largest matrix element. A diagonal bin
    (Delta = 0) only acts through commutators, [D, O]_ij = (D_ii - D_jj) O_ij, and its
    part proportional to the identity is a global phase, so its strength is the largest
    difference D_ii - D_jj between states one quantum apart in any mode."""
    carrier = carrier_of(freqs_GHz)
    dims = tuple(int(d) for d in dims)
    idx = np.arange(int(np.prod(dims)))
    occ = np.stack(np.unravel_index(idx, dims), axis=1)
    keep = idx if max_quanta is None else idx[occ[:, COUPLER] <= max_quanta]
    out = []
    for sig, M in V.items():
        om = carrier(sig)
        if abs(om) < window_GHz:
            continue
        if not any(sig[1]):
            d = M.diagonal()
            el = 0.0
            for m, dm in enumerate(dims):
                step = int(np.prod(dims[m + 1:]))
                for i in keep:
                    for sgn in (+1, -1):
                        o = occ[i, m] + sgn
                        if 0 <= o < dm:
                            el = max(el, abs(d[i] - d[i + sgn * step]))
            if el > 0:
                out.append((el / abs(om), sig, el, om))
            continue
        A = abs(M.tocsc())[:, keep]
        el = float(A.max()) if A.nnz else 0.0
        if el > 0:
            out.append((el / abs(om), sig, el, om))
    return sorted(out, key=lambda r: -r[0])


# --------------------------------------------------------------------------- the model
@dataclass
class DisplacedModel(ReducedModel):
    harmonics: int = DEFAULT_HARMONICS
    disp_window_GHz: float = DEFAULT_DISP_WINDOW_GHZ
    #: {k: polynomial coefficients of b_k(x) in powers x^|k|, x^(|k|+2), ...}
    disp_poly: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    x_max: float = X_MAX
    #: largest fit error of any b_k over the nodes (dimensionless); `fit_residual` is the
    #: largest error of any H_eff matrix element, in GHz
    disp_fit_residual: float = 0.0

    def b_at(self, k: int, x: float) -> float:
        powers, coef = self.disp_poly[k]
        return float(np.dot(coef, x ** powers))


def _nodes(n: int, x_max: float) -> np.ndarray:
    return np.concatenate([[0.0], 0.5 * x_max * (1 - np.cos(np.pi * (np.arange(n - 1) + 0.5)
                                                            / (n - 1)))])


def _fit(xs, Y, s, degree):
    """Least squares in powers x^|s|, x^(|s|+2), ... <= degree, on the scaled x / max(xs)."""
    Y2 = Y if Y.ndim > 1 else Y[:, None]
    X_MAX = float(np.max(xs))
    powers = np.arange(abs(s), degree + 1, 2)
    Vm = (xs / X_MAX)[:, None] ** powers[None, :]
    ck, *_ = np.linalg.lstsq(Vm, Y2, rcond=None)
    resid = float(np.max(np.abs(Vm @ ck - Y2)))           # absolute, in Y's units
    ck = ck / (X_MAX ** powers)[:, None]
    return powers, (ck if Y.ndim > 1 else ck[:, 0]), resid


def build_displaced_model(freqs_GHz, lam, alpha, g, dims, *, order: int = DEFAULT_ORDER,
                          window_GHz: float = DEFAULT_WINDOW_GHZ,
                          harmonics: int = DEFAULT_HARMONICS,
                          disp_window_GHz: float = DEFAULT_DISP_WINDOW_GHZ,
                          degree: int = 24, n_nodes: int = 21,
                          tol_GHz: float = DEFAULT_TOL_GHZ, workers: int = 0,
                          x_max: float = X_MAX) -> DisplacedModel:
    """Expand H' at `n_nodes` values of |eta| (in parallel over `workers` processes; 0 = one
    per node) and fit every coefficient in |eta|."""
    from concurrent.futures import ProcessPoolExecutor
    dims = tuple(int(d) for d in dims)
    xs = _nodes(n_nodes, x_max)
    carrier = carrier_of(freqs_GHz)
    job = dict(freqs_GHz=tuple(freqs_GHz), lam=tuple(lam), alpha=tuple(alpha), g=dict(g),
               dims=dims, order=order, window_GHz=window_GHz, harmonics=harmonics,
               disp_window_GHz=disp_window_GHz, tol_GHz=tol_GHz)
    with ProcessPoolExecutor(max_workers=workers or len(xs)) as pool:
        results = list(pool.map(_node, [dict(job, x=float(x)) for x in xs]))
    vals: Dict[Tuple[int, Tuple], List] = {}
    bs: Dict[int, List] = {}
    for j, (b, H) in enumerate(results):
        for k, v in b.items():
            bs.setdefault(k, [0.0] * len(xs))[j] = v
        for n, bins in H.items():
            for sig, M in bins.items():
                vals.setdefault((n, sig), [None] * len(xs))[j] = M
    ref = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))
    model = DisplacedModel(tuple(float(f) for f in freqs_GHz), ref, dims, order,
                           float(window_GHz), harmonics=harmonics,
                           disp_window_GHz=(None if disp_window_GHz is None else float(disp_window_GHz)))
    model.x_max = float(x_max)
    worst = 0.0
    for k, v in bs.items():
        powers, ck, r = _fit(xs, np.real(np.asarray(v)), k, degree)
        model.disp_poly[k] = (powers, ck)
        model.disp_fit_residual = max(model.disp_fit_residual, r)
    dim = int(np.prod(dims))
    for (n, sig), Ms in vals.items():
        s = sig[0]
        Y = np.stack([np.zeros((dim, dim), complex) if M is None else M.toarray() for M in Ms])
        Y = Y.reshape(len(xs), -1)
        mask = np.any(np.abs(Y) > 1e-15, axis=0)
        if not mask.any():
            continue
        powers, ck, r = _fit(xs, Y[:, mask], s, degree)
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
    return model


def _node(job):
    V, b = displaced_hamiltonian(job["freqs_GHz"], job["lam"], job["alpha"], job["g"],
                                 job["dims"], job["x"], harmonics=job["harmonics"],
                                 disp_window_GHz=job["disp_window_GHz"])
    return b, sw_recursion(V, carrier_of(job["freqs_GHz"]), job["window_GHz"], job["order"],
                           job["tol_GHz"])


_MEMO: Dict[str, DisplacedModel] = {}


def cached_displaced_model(freqs_GHz, lam, alpha, g, dims, *, cache_dir: Optional[str] = None,
                           **kw) -> DisplacedModel:
    blob = json.dumps({"f": [round(float(v), 9) for v in freqs_GHz],
                       "lam": [round(float(v), 9) for v in lam],
                       "alpha": [round(float(v), 9) for v in alpha],
                       "g": {str(k): round(float(v), 12) for k, v in sorted(g.items())},
                       "dims": [int(d) for d in dims], "kw": kw, "xmax": X_MAX,
                       "kind": "displaced", "v": 2}, sort_keys=True, default=str)
    key = hashlib.sha1(blob.encode()).hexdigest()[:16]
    if key in _MEMO:
        return _MEMO[key]
    path = os.path.join(cache_dir, f"disp_{key}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            _MEMO[key] = pickle.load(f)
            return _MEMO[key]
    model = build_displaced_model(freqs_GHz, lam, alpha, g, dims, **kw)
    _MEMO[key] = model
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(model, f)
        os.replace(tmp, path)
    return model


# --------------------------------------------------------------------------- the engine
_STATE: Dict[str, Any] = {}


def model_for(cpl, **kw) -> DisplacedModel:
    if cpl.n_modes != 3 or cpl.coupler_index != COUPLER:
        raise ValueError("the displaced ladder handles the (a, b, coupler) system only")
    opts = dict(_STATE.get("kw", {}))
    opts.update(kw)
    return cached_displaced_model(
        tuple(float(w) / TWO_PI for w in cpl.omega), tuple(float(v) for v in cpl.participation),
        tuple(float(v) / TWO_PI for v in cpl.anharm),
        {n: float(v) / TWO_PI for n, v in cpl.g_n.items()}, tuple(cpl.dims),
        cache_dir=_STATE.get("cache_dir"), **opts)


def qobjevo(cpl, level: str, model: Optional[DisplacedModel] = None, h_ns: float = 1e-3):
    """H'(t) in rad/ns: the level's slow terms plus the non-adiabatic displacement term

        -i G_k'(t) e^{-i 2pi Omega_k t} s^dag + h.c.,   G_k = b_k(|eta|) u^k e^{-i 2pi k dw t}

    with u = eta/|eta| (chirp) and dw the carrier offset, exactly what the adiabatic
    cancellation leaves of -i beta' s^dag."""
    import qutip as qt
    if len(cpl._pump_tones) != 1:
        raise ValueError("the displaced ladder handles a single pump tone")
    model = model or model_for(cpl)
    tone = cpl._pump_tones[0]
    dw = float(tone.w_p_GHz) - model.ref_wp_GHz
    dims = [list(cpl.dims), list(cpl.dims)]
    dim = int(np.prod(cpl.dims))

    # every term is matrix * x^K u^S e^{-i om t}: one sparse map from the coefficient
    # vector to the data of H(t) on the union sparsity pattern, so a call is one matvec
    mats, K, Sx, OM = [], [], [], []
    for e in model.select(level):
        mats.append(e.matrix)
        K.append(e.power + abs(e.s))
        Sx.append(e.s)
        OM.append(TWO_PI * (e.carrier_GHz + e.s * dw))
    n_terms = len(mats)
    s_mat = sps.kron(sps.identity(cpl.dims[0] * cpl.dims[1]),
                     sps.diags(np.sqrt(np.arange(1, cpl.dims[2])), 1), format="csr")
    kicks = sorted(model.disp_poly)
    for k in kicks:                          # coefficients filled in by `kick`
        mats.append(s_mat.T.conj().tocsr())
        mats.append(s_mat)
    union = sum(abs(m) for m in mats).tocsr()
    union.sort_indices()
    pos = {(r, c): q for q, (r, c) in enumerate(zip(*union.nonzero()))}
    rows, cols, vals = [], [], []
    for j, m in enumerate(mats):
        m = m.tocoo()
        for r, c, v in zip(m.row, m.col, m.data):
            rows.append(pos[(r, c)])
            cols.append(j)
            vals.append(v)
    A = sps.csr_matrix((vals, (rows, cols)), shape=(union.nnz, len(mats)))
    pattern = union.nonzero()
    K, Sx, OM = np.array(K, float), np.array(Sx, float), np.array(OM, float)
    fs = model.freqs_GHz[COUPLER]
    om_k = np.array([TWO_PI * (k * model.ref_wp_GHz - fs) for k in kicks])

    def G(t: float) -> np.ndarray:
        e = cpl._eta(tone, t)
        x = abs(e)
        u = e / x if x > 0 else 1.0
        return np.array([model.b_at(k, x) * u ** k * cmath.exp(-1j * TWO_PI * k * dw * t)
                         for k in kicks], complex)

    peak = max(abs(cpl._eta(tone, t)) for t in np.linspace(0, tone.envelope.t_g, 401)) \
        if hasattr(tone.envelope, "t_g") else 0.0
    if peak > model.x_max * (1 + 1e-6):
        raise ValueError(f"pulse peak |eta| = {peak:.3f} is beyond the fitted range "
                         f"{model.x_max} of the displaced model")

    def H(t: float, *args, **kw):
        e = cpl._eta(tone, t)
        x = abs(e)
        u = e / x if x > 0 else 1.0 + 0j
        c = np.empty(len(mats), complex)
        c[:n_terms] = TWO_PI * x ** K * u ** Sx * np.exp(-1j * OM * t)
        if kicks:
            dG = (G(t + h_ns) - G(t - h_ns)) / (2 * h_ns)          # 1/ns
            kk = -1j * dG * np.exp(-1j * om_k * t)
            c[n_terms::2] = kk
            c[n_terms + 1::2] = np.conj(kk)
        data = A @ c
        return qt.Qobj(sps.csr_matrix((data, pattern), shape=(dim, dim)), dims=dims)
    return qt.QobjEvo(H)


def install(level: str, *, cache_dir: Optional[str] = None, **kw) -> None:
    """Make every ZhouCoupler in this process (and its forks) solve the displaced model."""
    from snail_solver import zhou_coupler as Z
    if level not in LEVELS:
        raise ValueError(f"level {level!r}: expected one of {LEVELS}")
    if "orig" not in _STATE:
        _STATE["orig"] = Z.ZhouCoupler.to_qutip_hamiltonian
    _STATE.update(level=level, cache_dir=cache_dir, kw=kw)
    os.environ["LADDER_LEVEL"] = f"displaced:{level}"

    def to_qutip_hamiltonian(self, cutoff_GHz: float = np.inf, sparse: bool = True):
        return qobjevo(self, _STATE["level"])
    Z.ZhouCoupler.to_qutip_hamiltonian = to_qutip_hamiltonian


def uninstall() -> None:
    from snail_solver import zhou_coupler as Z
    if "orig" in _STATE:
        Z.ZhouCoupler.to_qutip_hamiltonian = _STATE.pop("orig")
