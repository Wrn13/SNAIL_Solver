"""The effective Hamiltonian of one device, sorted into the ladder's term classes.

`tdsw.HighOrderExpansion` gives the slow Hamiltonian at a CONSTANT pump amplitude. Every
coefficient of a term with net pump number s is a polynomial in (eta, eta*), so at
eta = x e^{i theta} it is c(x) e^{i s theta}, where c(x) is the value at real x:

    c(x) = sum_k c_k x^k,   k = |s|, |s| + 2, ..., <= 2 n   (order n)

That makes ONE expansion per device usable for any pulse. The expansion is evaluated at
a few real x, each coefficient is fitted exactly on that basis, and the pulse supplies
eta(t). The chirp and any carrier offset enter as the pump phase, the same way
``ZhouCoupler._eta_at`` puts them in, so the SW denominators are taken at the nominal
pump w_p = |w_b - w_a|. That neglects a shift s*delta/Omega of order MHz/GHz.

Term classes (signature sigma = (s, Delta), order n)
----------------------------------------------------
* ``diag``       s = 0, Delta = 0: anharmonicity (n=1), Stark / Kerr / Lamb shifts (n>=2)
* ``exchange``   s = k, Delta = (k, -k, 0): the iSWAP coupling and its multi-quantum
                 copies (a^2 b^dag2 eta^2 is the |20> <-> |02> exchange)
* ``spectator``  every other slow signature: the qubit-A subharmonic (s=2; a:-1),
                 including its 1->2 rung, the 3w_p = w_b / w_s processes, ...

Ladder (cumulative)::

    L0  diag n=1 (H_anh)       + exchange n=1          bare RWA exchange
    L1  + diag n=2                                     second-order Stark / Kerr
    L2  + diag n>=3, exchange n>=2                     higher-order shifts and rate
    L3  + spectator n=1                                near-resonant first-order channels
    L4  + spectator n=2                                second-order slow processes
    L5  + spectator n>=3                               everything kept by the expansion
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps

from ladder import _AE  # noqa: F401  (puts tdsw on sys.path)
from tdsw import Device, HighOrderExpansion, LinearFrameSW

LEVELS: Tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5")
MODES: Tuple[str, ...] = ("a", "b", "s")
DEFAULT_ORDER = 4
#: Signatures whose linear-frame carrier is under this (GHz) are kept in H_eff with their
#: residual carrier. It is wide enough that every feature of the 5 MHz curve is slow at
#: every column (A subharmonic 2*delta, its 1->2 rung 2*delta + |alpha|, and
#: 3 w_p - w_s = 0.55 + 3*delta GHz over delta = -0.19..+0.16).
DEFAULT_WINDOW_GHZ = 0.6
#: Largest |eta| the coefficient fit is anchored on; the polynomial is exact beyond it.
X_MAX = 1.6


def term_class(sig) -> str:
    s, d = sig
    if s == 0 and not any(d):
        return "diag"
    if d[2] == 0 and d[0] == -d[1] and s == d[0] and s != 0:
        return "exchange"
    return "spectator"


def included(level: str, cls: str, order: int) -> bool:
    """Whether a term of this class and SW order belongs to `level` (cumulative)."""
    i = LEVELS.index(level)
    if cls == "diag":
        return order == 1 or (order == 2 and i >= 1) or i >= 2
    if cls == "exchange":
        return order == 1 or i >= 2
    return (order == 1 and i >= 3) or (order == 2 and i >= 4) or i >= 5


@dataclass
class Entry:
    """One operator of H_eff: ``matrix * x^power * (eta/|eta|)^s * e^{-i 2pi carrier t}``.

    `matrix` is in GHz (cycles). The pump phase is carried as ``eta^|s|`` (``eta*`` for
    s < 0) with ``power = k - |s|`` left over, so x = 0 is never divided by.
    """
    order: int
    cls: str
    sig: Tuple[int, Tuple[int, ...]]
    s: int
    power: int
    carrier_GHz: float
    matrix: sps.csr_matrix


@dataclass
class ReducedModel:
    freqs_GHz: Tuple[float, float, float]
    ref_wp_GHz: float
    dims: Tuple[int, int, int]
    order: int
    window_GHz: float
    entries: List[Entry] = field(default_factory=list)
    fit_residual: float = 0.0

    def select(self, level: str) -> List[Entry]:
        return [e for e in self.entries if included(level, e.cls, e.order)]

    def matrix_at(self, level: str, eta: complex, t: float = 0.0) -> np.ndarray:
        """Dense H_eff(t) in GHz at pump amplitude `eta` (for tests and diagnostics)."""
        x = abs(eta)
        dim = int(np.prod(self.dims))
        H = np.zeros((dim, dim), dtype=complex)
        for e in self.select(level):
            ph = (eta if e.s > 0 else np.conj(eta)) ** abs(e.s)
            H += e.matrix.toarray() * (x ** e.power) * ph * np.exp(-2j * np.pi * e.carrier_GHz * t)
        return H


def _op_matrix(ops, dims, cache) -> sps.csr_matrix:
    key = (ops, dims)
    if key not in cache:
        out = sps.identity(1, dtype=complex, format="csr")
        for (p, q), d in zip(ops, dims):
            a = sps.diags(np.sqrt(np.arange(1, d, dtype=float)), 1, format="csr").astype(complex)
            ad = a.T.conj()
            m = sps.identity(d, dtype=complex, format="csr")
            for _ in range(p):
                m = m @ ad
            for _ in range(q):
                m = m @ a
            out = sps.kron(out, m, format="csr")
        cache[key] = out.tocsr()
    return cache[key]


def device(freqs_GHz, lam, alpha, g, eta: float) -> Device:
    fa, fb, fs = (float(f) for f in freqs_GHz)
    return Device(modes=list(MODES), omega={"a": fa, "b": fb, "s": fs},
                  lam=dict(zip(MODES, (float(v) for v in lam))),
                  alpha=dict(zip(MODES, (float(v) for v in alpha))),
                  g={int(k): float(v) for k, v in g.items()},
                  omega_p=abs(fb - fa), eta=complex(eta))


def build_model(freqs_GHz: Sequence[float], lam: Sequence[float], alpha: Sequence[float],
                g: Dict[int, float], dims: Sequence[int], *, order: int = DEFAULT_ORDER,
                window_GHz: float = DEFAULT_WINDOW_GHZ, q_final: Optional[int] = None
                ) -> ReducedModel:
    """Expand once per node of |eta|, fit every coefficient, build the sparse operators.

    `q_final` (tdsw's truncation) makes matrix elements exact only out of states with at
    most that many quanta in total; None keeps them exact on the whole Fock space.
    """
    dims = tuple(int(d) for d in dims)
    n_nodes = 2 * order + 3
    # Chebyshev nodes on [0, X_MAX], plus x = 0 itself (the pump-free terms)
    xs = np.concatenate([[0.0], 0.5 * X_MAX * (1 - np.cos(np.pi * (np.arange(n_nodes - 1) + 0.5)
                                                          / (n_nodes - 1)))])
    coef: Dict[Tuple[int, Tuple], np.ndarray] = {}
    for j, x in enumerate(xs):
        L = LinearFrameSW(device(freqs_GHz, lam, alpha, g, x), slow_cutoff=window_GHz)
        E = HighOrderExpansion(L, max_order=order, q_final=q_final, resonance_ratio=None)
        for n in range(1, order + 1):
            for (s, ops), c in E.H[n].terms.items():
                coef.setdefault((n, (s, ops)), np.zeros(len(xs), dtype=complex))[j] += complex(c)
    ref = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))
    model = ReducedModel(tuple(float(f) for f in freqs_GHz), ref, dims, order, float(window_GHz))
    grouped: Dict[Tuple, sps.csr_matrix] = {}
    cache: Dict = {}
    worst = 0.0
    for (n, (s, ops)), vals in coef.items():
        sig = (s, tuple(q - p for p, q in ops))
        powers = np.arange(abs(s), 2 * n + 1, 2)
        V = xs[:, None] ** powers[None, :]
        ck, *_ = np.linalg.lstsq(V, vals, rcond=None)
        scale = max(np.max(np.abs(vals)), 1e-300)
        worst = max(worst, float(np.max(np.abs(V @ ck - vals)) / scale))
        M = _op_matrix(ops, dims, cache)
        if M.nnz == 0:
            continue                           # the truncation kills this monomial
        for k, c in zip(powers, ck):
            if abs(c) < 1e-15:
                continue
            key = (n, sig, int(k))
            grouped[key] = grouped.get(key, 0) + c * M
    for (n, sig, k), M in grouped.items():
        M = sps.csr_matrix(M)
        M.eliminate_zeros()
        if M.nnz == 0:
            continue
        s = sig[0]
        carrier = sum(dk * f for dk, f in zip(sig[1], freqs_GHz)) + s * ref
        model.entries.append(Entry(n, term_class(sig), sig, s, k - abs(s), float(carrier), M))
    model.fit_residual = worst
    return model


MatBins = Dict[Tuple[int, Tuple[int, ...]], sps.csr_matrix]


def _letters(lam, dims):
    """X's letters on the TRUNCATED space, as ZhouCoupler builds them: (op, signature, amp, pump)."""
    n = len(dims)
    eye = [sps.identity(d, dtype=complex, format="csr") for d in dims]
    out = []
    for i, d in enumerate(dims):
        a = sps.diags(np.sqrt(np.arange(1, d, dtype=float)), 1, format="csr").astype(complex)
        fac = list(eye)
        fac[i] = a
        A = fac[0]
        for f in fac[1:]:
            A = sps.kron(A, f, format="csr")
        unit = tuple(1 if k == i else 0 for k in range(n))
        out.append((A, (0, unit), float(lam[i]), 0))
        out.append((A.T.conj().tocsr(), (0, tuple(-u for u in unit)), float(lam[i]), 0))
    one = sps.identity(int(np.prod(dims)), dtype=complex, format="csr")
    out.append((one, (1, (0,) * n), 1.0, +1))
    out.append((one, (-1, (0,) * n), 1.0, -1))
    return out


def _add(bins: MatBins, sig, M) -> None:
    bins[sig] = bins[sig] + M if sig in bins else M


def _prune(bins: MatBins, tol: float = 1e-14) -> MatBins:
    out = {}
    for sig, M in bins.items():
        M = sps.csr_matrix(M)
        if M.nnz and np.max(np.abs(M.data)) > tol:
            out[sig] = M
    return out


def _comm(A: MatBins, B: MatBins) -> MatBins:
    out: MatBins = {}
    for sa, Ma in A.items():
        for sb, Mb in B.items():
            sig = (sa[0] + sb[0], tuple(x + y for x, y in zip(sa[1], sb[1])))
            _add(out, sig, Ma @ Mb - Mb @ Ma)
    return _prune(out)


def _lin(*terms) -> MatBins:
    out: MatBins = {}
    for c, bins in terms:
        for sig, M in bins.items():
            _add(out, sig, c * M)
    return _prune(out)


def matrix_expansion(freqs_GHz, lam, alpha, g, dims, x: float, order: int,
                     window_GHz: float) -> Dict[int, MatBins]:
    """tdsw's HighOrderExpansion recursion (resonance_ratio=None) on TRUNCATED matrices.

    The order-1 operator is built exactly as ``ZhouCoupler.expand_terms`` builds it,
    products of truncated ladder matrices, so the expansion is of the full model's own
    Hamiltonian, top-level truncation included. Returns {n: slow bins of order n} (GHz).
    """
    from math import factorial
    from itertools import product as iproduct
    ref = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))
    n_modes = len(dims)

    def carrier(sig):
        return sum(d * f for d, f in zip(sig[1], freqs_GHz)) + sig[0] * ref

    def slow(sig):
        return abs(carrier(sig)) < window_GHz

    letters = _letters(lam, dims)
    V: MatBins = {}
    for nn, gn in g.items():
        for combo in iproduct(letters, repeat=int(nn)):
            amp = float(gn)
            op = None
            s, d = 0, [0] * n_modes
            for (M, (ls, ld), la, pump) in combo:
                amp *= la * (x if pump else 1.0)
                op = M if op is None else op @ M
                s += ls
                d = [u + v for u, v in zip(d, ld)]
            _add(V, (s, tuple(d)), amp * op)
    anh = None
    for i, dd in enumerate(dims):
        if alpha[i]:
            loc = sps.diags([0.5 * float(alpha[i]) * k * (k - 1) for k in range(dd)], 0, format="csr")
            fac = [sps.identity(e, format="csr") for e in dims]
            fac[i] = loc
            A = fac[0]
            for f in fac[1:]:
                A = sps.kron(A, f, format="csr")
            anh = A if anh is None else anh + A
    if anh is not None:
        _add(V, (0, (0,) * n_modes), anh.astype(complex))
    V = _prune(V)

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
        R = _lin(*parts)
        Hn, Sn, Wn = {}, {}, {}
        for sig, M in R.items():
            if slow(sig):
                Hn[sig] = M
            else:
                Sn[sig] = (-1.0 / carrier(sig)) * M
                Wn[sig] = -M
        H[n], S[n] = Hn, Sn
        U[0][n] = Wn
    return H


def build_model_matrix(freqs_GHz: Sequence[float], lam: Sequence[float],
                       alpha: Sequence[float], g: Dict[int, float], dims: Sequence[int], *,
                       order: int = DEFAULT_ORDER, window_GHz: float = DEFAULT_WINDOW_GHZ
                       ) -> ReducedModel:
    """`build_model` on the truncated-matrix expansion (the default: matches the full model)."""
    dims = tuple(int(d) for d in dims)
    n_nodes = 2 * order + 3
    xs = np.concatenate([[0.0], 0.5 * X_MAX * (1 - np.cos(np.pi * (np.arange(n_nodes - 1) + 0.5)
                                                          / (n_nodes - 1)))])
    vals: Dict[Tuple[int, Tuple], List] = {}
    for j, x in enumerate(xs):
        H = matrix_expansion(freqs_GHz, lam, alpha, g, dims, float(x), order, window_GHz)
        for n, bins in H.items():
            for sig, M in bins.items():
                vals.setdefault((n, sig), [None] * len(xs))[j] = M
    ref = abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))
    model = ReducedModel(tuple(float(f) for f in freqs_GHz), ref, dims, order, float(window_GHz))
    dim = int(np.prod(dims))
    worst = 0.0
    for (n, sig), Ms in vals.items():
        s = sig[0]
        Y = np.stack([np.zeros((dim, dim), complex) if M is None else M.toarray() for M in Ms])
        Y = Y.reshape(len(xs), -1)
        mask = np.any(np.abs(Y) > 1e-15, axis=0)
        if not mask.any():
            continue
        powers = np.arange(abs(s), 2 * n * max(max(g), 2) // 2 + 1, 2)
        Vm = xs[:, None] ** powers[None, :]
        ck, *_ = np.linalg.lstsq(Vm, Y[:, mask], rcond=None)
        scale = max(float(np.max(np.abs(Y[:, mask]))), 1e-300)
        worst = max(worst, float(np.max(np.abs(Vm @ ck - Y[:, mask]))) / scale)
        carrier = sum(dk * f for dk, f in zip(sig[1], freqs_GHz)) + s * ref
        for k, row in zip(powers, ck):
            if np.max(np.abs(row)) < 1e-15:
                continue
            full = np.zeros(dim * dim, complex)
            full[mask] = row
            M = sps.csr_matrix(full.reshape(dim, dim))
            M.eliminate_zeros()
            model.entries.append(Entry(n, term_class(sig), sig, s, int(k) - abs(s), float(carrier), M))
    model.fit_residual = worst
    return model


def model_key(freqs_GHz, lam, alpha, g, dims, order, window_GHz, q_final=None,
              method="matrix") -> str:
    blob = json.dumps({"f": [round(float(v), 9) for v in freqs_GHz],
                       "lam": [round(float(v), 9) for v in lam],
                       "alpha": [round(float(v), 9) for v in alpha],
                       "g": {str(k): round(float(v), 12) for k, v in sorted(g.items())},
                       "dims": [int(d) for d in dims], "order": int(order),
                       "W": round(float(window_GHz), 9), "xmax": X_MAX, "q": q_final, "method": method,
                       "v": 2}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


_MEMO: Dict[str, ReducedModel] = {}


def cached_model(freqs_GHz, lam, alpha, g, dims, *, order: int = DEFAULT_ORDER,
                 window_GHz: float = DEFAULT_WINDOW_GHZ, q_final: Optional[int] = None,
                 method: str = "matrix", cache_dir: Optional[str] = None) -> ReducedModel:
    """The model memoized in-process and (if `cache_dir`) on disk, atomically.

    ``method="matrix"`` (default) expands the full model's truncated operators
    (`build_model_matrix`); ``"nopoly"`` is tdsw's untruncated algebra truncated
    afterwards (`build_model`), which differs at each mode's top Fock level.
    """
    key = model_key(freqs_GHz, lam, alpha, g, dims, order, window_GHz, q_final, method)
    if key in _MEMO:
        return _MEMO[key]
    path = os.path.join(cache_dir, f"sw_{key}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                _MEMO[key] = pickle.load(f)
                return _MEMO[key]
        except Exception:
            pass                               # a torn file: rebuild it
    if method == "matrix":
        model = build_model_matrix(freqs_GHz, lam, alpha, g, dims, order=order,
                                   window_GHz=window_GHz)
    else:
        model = build_model(freqs_GHz, lam, alpha, g, dims, order=order, window_GHz=window_GHz,
                            q_final=q_final)
    _MEMO[key] = model
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(model, f)
        os.replace(tmp, path)
    return model
