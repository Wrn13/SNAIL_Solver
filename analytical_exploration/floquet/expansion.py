"""The Floquet-Magnus expansion of Bock, Ernst & Uhrig (arXiv:2609.17121), as written.

A periodic Hamiltonian (hbar = 1, angular units)

    H(t) = sum_m H_m e^{i m w t}                                                   (2)

has the propagator

    U(t, t0) = e^{-i Lambda(t)} e^{-i F (t - t0)} e^{+i Lambda(t0)}                 (3)

with F static and Lambda(t) = sum_m Lambda_m e^{i m w t} periodic (5). Inserting (3) into
the Schrodinger equation gives (8b), whose Fourier coefficients are (27) = (43):

    i m w Lambda_m = -F delta_{m,0} + H_m + (i/2) { sum_n1 [Lambda_n1, H_{m-n1}] + [Lambda_m, F] }
        + sum_{j>=2} (-i)^j B_j / j! { sum [Lambda_n1, ..., Lambda_nj, H_{m-n1-...-nj}]
                                       + (-1)^{j+1} sum [Lambda_n1, ..., Lambda_{m-n1-...}, F] }

Taking m = 0 and m != 0 separately gives F (44) and Lambda_{m != 0} (45). Equating orders
gives (46a-f) = (28a-f). The gauge is Lambda_0:

    "fvv"   Lambda_0^(n) = 0                                  (24), (29), (30), App. B.1
    "aht"   Lambda_0^(n) = -sum_{m != 0} Lambda_m^(n) e^{i m w t0}         (31), App. A

Here H and every coefficient are dense matrices keyed by the integer Fourier index m.
`floquet_magnus` solves (44)-(45) order by order. The explicit formulas of the paper
(28a-f, 29, 30, 47, 48) are implemented separately below, and the tests check that the
recursion reproduces each of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Dict, Optional

import numpy as np

Coeffs = Dict[int, np.ndarray]           # m -> the m-th Fourier coefficient

#: signed Bernoulli numbers B_k (eq. 9: x / (e^x - 1) = sum B_n x^n / n!), B_1 = -1/2
BERNOULLI = {0: 1.0, 1: -0.5, 2: 1 / 6, 4: -1 / 30, 6: 1 / 42, 8: -1 / 30, 10: 5 / 66}
GAUGES = ("fvv", "aht")
_TOL = 1e-15


def _clean(c: Coeffs) -> Coeffs:
    return {m: M for m, M in c.items() if np.max(np.abs(M)) > _TOL}


def _comm(A: Coeffs, B: Coeffs) -> Coeffs:
    """Fourier coefficients of [A(t), B(t)]: indices add (a convolution)."""
    out: Coeffs = {}
    for ma, Ma in A.items():
        for mb, Mb in B.items():
            C = Ma @ Mb - Mb @ Ma
            out[ma + mb] = out[ma + mb] + C if ma + mb in out else C
    return _clean(out)


def _lin(*terms) -> Coeffs:
    out: Coeffs = {}
    for c, A in terms:
        for m, M in A.items():
            out[m] = out[m] + c * M if m in out else c * M
    return _clean(out)


def bracket(H: Coeffs, n: int, m: int) -> Optional[np.ndarray]:
    """Eq. (26): [n, m] := [H_n, H_m]."""
    if n not in H or m not in H:
        return None
    return H[n] @ H[m] - H[m] @ H[n]


@dataclass
class Expansion:
    """F^(n) (static, angular units) and Lambda^(n)_m by order n, for drive frequency w."""
    omega: float
    F: Dict[int, np.ndarray] = field(default_factory=dict)
    Lambda: Dict[int, Coeffs] = field(default_factory=dict)
    gauge: str = "fvv"

    @property
    def order(self) -> int:
        return max(self.F) if self.F else 0

    def F_upto(self, n: Optional[int] = None) -> np.ndarray:
        """F^[n] = sum_{k <= n} F^(k), eq. (12b)."""
        n = self.order if n is None else n
        return sum(self.F[k] for k in range(1, n + 1))

    def Lambda_upto(self, n: int) -> Coeffs:
        """Lambda^[n] = sum_{k <= n} Lambda^(k), eq. (12a)."""
        return _lin(*[(1.0, self.Lambda[k]) for k in range(1, n + 1) if k in self.Lambda])

    def Lambda_at(self, t: float, upto: int) -> np.ndarray:
        """Lambda^[upto](t) = sum_m Lambda_m e^{i m w t}, eq. (5)."""
        L = self.Lambda_upto(upto)
        dim = self.F[1].shape[0]
        return sum((M * np.exp(1j * m * self.omega * t) for m, M in L.items()),
                   np.zeros((dim, dim), complex))

    def propagator(self, t: float, t0: float = 0.0, order: Optional[int] = None) -> np.ndarray:
        """Eq. (3) with F to `order` and Lambda to `order - 1` (the n-th order evolution
        operator needs n orders of F and n - 1 of Lambda)."""
        from scipy.linalg import expm
        n = self.order if order is None else order
        F = self.F_upto(n)
        L = (lambda s: self.Lambda_at(s, n - 1)) if n > 1 else (lambda s: 0 * F)
        return expm(-1j * L(t)) @ expm(-1j * F * (t - t0)) @ expm(1j * L(t0))

    def stroboscopic(self, t0: float = 0.0, order: Optional[int] = None) -> np.ndarray:
        """F~ = e^{-i Lambda(t0)} F e^{+i Lambda(t0)}, eq. (4): U(t0 + T, t0) = e^{-i F~ T}."""
        from scipy.linalg import expm
        n = self.order if order is None else order
        K = expm(-1j * self.Lambda_at(t0, n - 1)) if n > 1 else np.eye(self.F[1].shape[0])
        return K @ self.F_upto(n) @ K.conj().T


def floquet_magnus(H: Coeffs, omega: float, order: int, *, gauge: str = "fvv",
                   t0: float = 0.0) -> Expansion:
    """Eqs. (44) and (45), solved order by order (the integration-free scheme, App. A).

    H is first order. A[j][q] collects the j-fold nested commutators
    [Lambda_n1, ..., Lambda_nj, H_.] of total order q, and B[j][q] the same with F at the
    centre. Both are built from Lambda^(k < n), so each order needs only lower ones."""
    if gauge not in GAUGES:
        raise ValueError(f"gauge {gauge!r}: expected one of {GAUGES}")
    H = _clean({int(m): np.asarray(M, complex) for m, M in H.items()})
    dim = next(iter(H.values())).shape[0]
    zero = np.zeros((dim, dim), complex)
    ex = Expansion(omega=float(omega), gauge=gauge)
    A: Dict[int, Dict[int, Coeffs]] = {0: {1: H}}
    B: Dict[int, Dict[int, Coeffs]] = {0: {}}
    for n in range(1, order + 1):
        rhs = [(1.0, H)] if n == 1 else []
        for j in range(1, n):
            A.setdefault(j, {})[n] = _lin(*[(1.0, _comm(ex.Lambda[k], A[j - 1][n - k]))
                                            for k in range(1, n) if n - k in A[j - 1]])
            B.setdefault(j, {})[n] = _lin(*[(1.0, _comm(ex.Lambda[k], B[j - 1][n - k]))
                                            for k in range(1, n) if n - k in B[j - 1]])
            Bj = BERNOULLI.get(j, 0.0)
            if Bj:
                c = (-1j) ** j * Bj / factorial(j)
                rhs.append((c, A[j][n]))
                rhs.append((c * (-1) ** (j + 1), B[j][n]))
        R = _lin(*rhs)
        ex.F[n] = R.get(0, zero.copy())                          # (44): m = 0
        L = {m: M / (1j * m * omega) for m, M in R.items() if m != 0}   # (45): m != 0
        if gauge == "aht" and L:
            L[0] = -sum(M * np.exp(1j * m * omega * t0) for m, M in L.items())
        ex.Lambda[n] = _clean(L)
        B[0][n] = {0: ex.F[n]}
    return ex


# --------------------------------------------------------------------------- closed forms
# The paper's explicit orders in the FVV gauge (Lambda_0 = 0), with [n, m] = [H_n, H_m].

def F1(H: Coeffs, w: float) -> np.ndarray:
    """(28a), (29a): F^(1) = H_0."""
    return H[0]


def Lambda1(H: Coeffs, w: float) -> Coeffs:
    """(28b), (30b): Lambda^(1)_m = H_m / (i m w)."""
    return {m: M / (1j * m * w) for m, M in H.items() if m != 0}


def F2(H: Coeffs, w: float) -> np.ndarray:
    """(29b): F^(2) = sum_{n != 0} [n, -n] / (2 n w)."""
    return sum((bracket(H, n, -n) / (2 * n * w) for n in H if n != 0 and -n in H),
               0 * H[0])


def Lambda2(H: Coeffs, w: float) -> Coeffs:
    """(30c): Lambda^(2)_m = sum_{n != 0} [n, m-n] / (2i n m w^2) + [m, 0] / (2i m^2 w^2).
    The sum includes n = m (so m - n = 0); that literal reading is what (27) gives."""
    out: Coeffs = {}
    for n in H:
        for k in H:
            m = n + k
            if n == 0 or m == 0:
                continue
            out[m] = out.get(m, 0) + bracket(H, n, k) / (2j * n * m * w ** 2)
    for m in H:
        if m != 0 and 0 in H:
            out[m] = out.get(m, 0) + bracket(H, m, 0) / (2j * m ** 2 * w ** 2)
    return _clean(out)


def F3(H: Coeffs, w: float) -> np.ndarray:
    """(29c): F^(3) = sum_{n,m != 0} [n, [m, -m-n]] / (3 n m w^2)
                    + sum_{n != 0} [n, [0, -n]] / (6 n^2 w^2)."""
    out = 0 * H[0]
    c = lambda X, Y: X @ Y - Y @ X
    for n in H:
        if n == 0:
            continue
        for m in H:
            if m == 0 or -m - n not in H:
                continue
            out = out + c(H[n], bracket(H, m, -m - n)) / (3 * n * m * w ** 2)
        if -n in H:
            out = out + c(H[n], bracket(H, 0, -n)) / (6 * n ** 2 * w ** 2)
    return out


def F4(H: Coeffs, w: float) -> np.ndarray:
    """(47a-e), the fourth-order FVV effective Hamiltonian (App. B.1)."""
    c = lambda X, Y: X @ Y - Y @ X
    h = lambda k: H.get(k)
    out = 0 * H[0]
    idx = [k for k in H if k != 0]
    for n in idx:
        for k in idx:
            if n + k == 0:
                continue
            for l in idx:
                r = -n - k - l
                if n + k + l == 0 or h(r) is None:
                    continue
                nk = c(H[n], H[k])
                out = out + c(nk, c(H[l], H[r])) / (24 * (n + k + l) * (n + k) * n * w ** 3)   # (47a)
                out = out + c(H[n], c(H[k], c(H[l], H[r]))) / (24 * n * k * l * w ** 3)       # (47b)
                out = out + c(H[r], c(H[l], nk)) / (6 * n * (n + k) * (n + k + l) * w ** 3)    # (47b)
            r = -n - k
            if h(r) is not None and 0 in H:
                nk = c(H[n], H[k])
                out = out + c(c(H[r], H[0]), nk) / (12 * n * (n + k) ** 2 * w ** 3)         # (47c)
                out = out + c(H[r], c(H[k], c(H[n], H[0]))) / (3 * n ** 2 * (n + k) * w ** 3)  # (47d)
                out = out + c(H[r], c(H[0], nk)) / (4 * n * (n + k) ** 2 * w ** 3)          # (47d)
    for n in idx:
        if -n not in H:
            continue
        for k in idx:
            if -k in H:
                out = out + c(H[n], c(H[-n], c(H[-k], H[k]))) / (8 * n ** 2 * k * w ** 3)   # (47e)
        if 0 in H:
            out = out + c(H[-n], c(H[0], c(H[n], H[0]))) / (2 * n ** 3 * w ** 3)           # (47e)
    return out


def Lambda3(H: Coeffs, w: float) -> Coeffs:
    """(48a-c), the third-order FVV kick operator (App. B.1)."""
    c = lambda X, Y: X @ Y - Y @ X
    idx = [k for k in H if k != 0]
    ms = {a + b + d for a in H for b in H for d in H} - {0}
    out: Coeffs = {}
    for m in ms:
        acc = None
        def add(X):
            nonlocal acc
            acc = X if acc is None else acc + X
        if 0 in H and m in H:
            add(c(H[0], c(H[0], H[m])) / (1j * m ** 3 * w ** 3))                              # (48a)
        for n in idx:
            if m in H and -n in H:
                add((6 * n + m) / (12j * m ** 2 * n ** 2 * w ** 3) * c(H[n], c(H[m], H[-n])))  # (48a)
            if n != m and 0 in H and (m - n) in H:
                add((3 * m ** 2 - m * n - 3 * n ** 2) / (6j * m ** 2 * n ** 2 * (m - n) * w ** 3)
                    * c(H[m - n], c(H[0], H[n])))                                             # (48b)
            for k in idx:
                if n + k != 0 and (m - n - k) in H:
                    add((2 * k - n) / (6j * n * k * m * (n + k) * w ** 3)
                        * c(H[n], c(H[k], H[m - n - k])))                                     # (48c)
        if acc is not None:
            out[m] = acc
    return _clean(out)
