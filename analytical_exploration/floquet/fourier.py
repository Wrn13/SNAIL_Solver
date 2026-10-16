"""Fourier coefficients of the gate Hamiltonian, eqs. (2) and (6a), and their algebra.

    H_I(t) = sum_n g_n X(t)^n + sum_i (alpha_i / 2) n_i (n_i - 1)
    X(t)   = sum_k lam_k (c_k e^{-i w_k t} + h.c.) + (eta e^{-i w_p t} + c.c.)

At a constant pump amplitude eta, H_I(t) = sum_sigma H_sigma e^{-i Omega_sigma t}. Each
coefficient is stored as a sparse matrix in GHz, keyed by its signature sigma = (s, Delta).
X is built from TRUNCATED ladder matrices and raised to the power n, exactly as ZhouCoupler
builds it, so the coefficients are those of the full model, top-level truncation included.

Conventions: Delta_k = +1 for an annihilator of mode k, -1 for a creator (so c_k e^{-i w_k t}
has carrier +w_k), and s counts eta (+1) and eta* (-1). The carrier uses the nominal pump
|w_b - w_a|; the true pump's offset and chirp enter later as the phase of eta(t).
"""
from __future__ import annotations

from math import comb
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps

Signature = Tuple[int, Tuple[int, ...]]
Bins = Dict[Signature, sps.csr_matrix]
COUPLER = 2
MODES = ("a", "b", "s")


# --------------------------------------------------------------------------- signatures
def pump_frequency(freqs_GHz: Sequence[float]) -> float:
    """The nominal pump |w_b - w_a| (GHz) the carriers are taken at."""
    return abs(float(freqs_GHz[1]) - float(freqs_GHz[0]))


def carrier(sig: Signature, freqs_GHz: Sequence[float]) -> float:
    """Omega_sigma = sum_k Delta_k w_k + s w_p (GHz). It plays the paper's -m w."""
    return sum(d * f for d, f in zip(sig[1], freqs_GHz)) + sig[0] * pump_frequency(freqs_GHz)


def carrier_of(freqs_GHz: Sequence[float]):
    return lambda sig: carrier(sig, freqs_GHz)


def conj(sig: Signature) -> Signature:
    """The signature of H_sigma^dagger (the paper's -m)."""
    return (-sig[0], tuple(-d for d in sig[1]))


def zero(n_modes: int = 3) -> Signature:
    return (0, (0,) * n_modes)


def term_class(sig: Signature) -> str:
    """diag (s = 0, Delta = 0), exchange (s = k, Delta = (k, -k, 0)), or spectator."""
    s, d = sig
    if s == 0 and not any(d):
        return "diag"
    if d[2] == 0 and d[0] == -d[1] and s == d[0] and s != 0:
        return "exchange"
    return "spectator"


def describe(sig: Signature) -> str:
    """A readable name for the net process, e.g. (1, (1, 0, -1)) -> 'eta a s†'."""
    s, delta = sig

    def op(m, d):
        if d == 0:
            return ""
        if d > 0:
            return m + (f"^{d}" if d > 1 else "")
        return m + "†" + (f"^{-d}" if d < -1 else "")
    pump = "" if s == 0 else ("eta" if s > 0 else "eta*") + (f"^{abs(s)}" if abs(s) > 1 else "")
    ops = " ".join(p for p in (op(m, d) for m, d in zip(MODES, delta)) if p)
    return (pump + " " + ops).strip() or "diag"


# --------------------------------------------------------------------------- algebra
def add(bins: Bins, sig: Signature, M) -> None:
    bins[sig] = bins[sig] + M if sig in bins else M


def prune(bins: Bins, tol: float = 1e-14) -> Bins:
    out = {}
    for sig, M in bins.items():
        M = sps.csr_matrix(M)
        if M.nnz and np.max(np.abs(M.data)) > tol:
            out[sig] = M
    return out


def lin(*terms) -> Bins:
    """sum_i c_i A_i for (c_i, A_i) pairs of coefficient sets."""
    out: Bins = {}
    for c, bins in terms:
        for sig, M in bins.items():
            add(out, sig, c * M)
    return prune(out)


def _shift(a: Signature, b: Signature) -> Signature:
    return (a[0] + b[0], tuple(x + y for x, y in zip(a[1], b[1])))


def product(A: Bins, B: Bins) -> Bins:
    """Fourier coefficients of A(t) B(t): signatures add (a convolution)."""
    out: Bins = {}
    for sa, Ma in A.items():
        for sb, Mb in B.items():
            add(out, _shift(sa, sb), Ma @ Mb)
    return prune(out)


def comm(A: Bins, B: Bins) -> Bins:
    """Fourier coefficients of [A(t), B(t)]; for single coefficients this is eq. (26)'s
    [n, m] := [H_n, H_m], placed at signature n + m."""
    out: Bins = {}
    for sa, Ma in A.items():
        for sb, Mb in B.items():
            add(out, _shift(sa, sb), Ma @ Mb - Mb @ Ma)
    return prune(out)


def bracket(H: Bins, n: Signature, m: Signature) -> Optional[sps.csr_matrix]:
    """Eq. (26): [n, m] = [H_n, H_m] (None if either coefficient vanishes)."""
    if n not in H or m not in H:
        return None
    return H[n] @ H[m] - H[m] @ H[n]


def nested(X: Bins, Y: Bins, k: int) -> Bins:
    """Eq. (7): [X, Y]_k, the k-fold nested commutator [X, [X, ... [X, Y]]]."""
    out = Y
    for _ in range(k):
        out = comm(X, out)
    return out


def dense(bins: Bins, freqs_GHz, t: float) -> np.ndarray:
    """sum_sigma M_sigma e^{-i 2 pi Omega_sigma t} at time t (ns), dense, GHz."""
    out = None
    for sig, M in bins.items():
        term = M.toarray() * np.exp(-2j * np.pi * carrier(sig, freqs_GHz) * t)
        out = term if out is None else out + term
    return out


# --------------------------------------------------------------------------- operators
def annihilators(dims: Sequence[int]):
    """The truncated annihilation operator of each mode on the full Fock space."""
    eye = [sps.identity(d, dtype=complex, format="csr") for d in dims]
    out = []
    for i, d in enumerate(dims):
        fac = list(eye)
        fac[i] = sps.diags(np.sqrt(np.arange(1, d, dtype=float)), 1, format="csr").astype(complex)
        A = fac[0]
        for f in fac[1:]:
            A = sps.kron(A, f, format="csr")
        out.append(A.tocsr())
    return out


def flux_powers(lam: Sequence[float], dims: Sequence[int], n_max: int) -> Dict[int, Bins]:
    """{j: coefficients of X_q^j}, X_q the quantum flux sum_k lam_k (c_k e^{-i w_k t} + h.c.),
    as products of truncated matrices (ZhouCoupler's X^n)."""
    n_modes = len(dims)
    X: Bins = {}
    for i, A in enumerate(annihilators(dims)):
        unit = tuple(1 if k == i else 0 for k in range(n_modes))
        add(X, (0, unit), float(lam[i]) * A)
        add(X, (0, tuple(-u for u in unit)), float(lam[i]) * A.T.conj().tocsr())
    eye = sps.identity(int(np.prod(dims)), dtype=complex, format="csr")
    P = {0: {zero(n_modes): eye}, 1: prune(X)}
    for j in range(2, n_max + 1):
        P[j] = product(P[j - 1], P[1])
    return P


def anharmonicity(alpha: Sequence[float], dims: Sequence[int]) -> Optional[sps.csr_matrix]:
    """sum_i (alpha_i / 2) n_i (n_i - 1), diagonal, GHz."""
    out = None
    for i, d in enumerate(dims):
        if alpha[i]:
            fac = [sps.identity(e, format="csr") for e in dims]
            fac[i] = sps.diags([0.5 * float(alpha[i]) * k * (k - 1) for k in range(d)], 0,
                               format="csr")
            A = fac[0]
            for f in fac[1:]:
                A = sps.kron(A, f, format="csr")
            out = A if out is None else out + A
    return None if out is None else out.astype(complex).tocsr()


def _cpower(c: Dict[int, complex], m: int) -> Dict[int, complex]:
    out: Dict[int, complex] = {0: 1.0}
    for _ in range(m):
        nxt: Dict[int, complex] = {}
        for i, u in out.items():
            for j, v in c.items():
                nxt[i + j] = nxt.get(i + j, 0.0) + u * v
        out = nxt
    return out


def coefficients(freqs_GHz, lam, alpha, g: Dict[int, float], dims, eta: float, *,
                 displacement: Optional[Dict[int, complex]] = None, _powers=None) -> Bins:
    """The Fourier coefficients H_sigma (GHz) of H_I at constant real pump amplitude eta.

    With `displacement` = {k: b_k} (see `classical.harmonic_balance`), they are those of
    the frame displaced by the coupler's classical response, s -> s + beta(t):

        H' = sum_n g_n (X_q + x_cl(t))^n + H_anh - (Omega_k b_k s^dag + h.c.)

    Here x_cl is the classical flux, and the last term is the adiabatic part of
    -i D^dag dD/dt. The non-adiabatic remainder is evolved in `propagator`. The
    c-number x_cl^n is dropped (a global phase).
    """
    from floquet.classical import classical_flux
    dims = tuple(int(d) for d in dims)
    n_modes = len(dims)
    lam_s = float(lam[COUPLER])
    P = _powers or flux_powers(lam, dims, max(int(n) for n in g))
    b = displacement or {}
    c = classical_flux(b, eta, lam_s)
    H: Bins = {}
    for n, gn in g.items():
        n = int(n)
        for m in range(n):                      # m = n is the c-number
            for sq, M in P[n - m].items():
                for k, v in _cpower(c, m).items():
                    amp = gn * comb(n, m) * v
                    if amp != 0:
                        add(H, (sq[0] + k, sq[1]), amp * M)
    anh = anharmonicity(alpha, dims)
    if anh is not None:
        add(H, zero(n_modes), anh)
    if b:
        wp, fs = pump_frequency(freqs_GHz), float(freqs_GHz[COUPLER])
        s_lo = tuple(+1 if i == COUPLER else 0 for i in range(n_modes))
        S = annihilators(dims)[COUPLER]
        for k, v in b.items():
            om = k * wp - fs
            add(H, (k, conj((0, s_lo))[1]), -om * v * S.T.conj().tocsr())
            add(H, (-k, s_lo), -om * np.conj(v) * S)
    return prune(H)
