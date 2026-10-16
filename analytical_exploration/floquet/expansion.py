"""The Floquet-Magnus expansion, solved order by order in Fourier space (eqs. 8b, 27-31).

Inserting U = e^{-i Lambda(t)} e^{-i F (t - t0)} e^{+i Lambda(t0)} (eq. 3) into the
Schrodinger equation gives

    dLambda/dt = sum_k B_k/k! ( [-i Lambda, H]_k - [i Lambda, F]_k )                (8b)

with B_k the Bernoulli numbers (B_1 = -1/2). In Fourier space, with our coefficients
carrying e^{-i Omega t} (the paper's m w is -Omega), this becomes (27)

    -i Omega_sigma Lambda_sigma = -F_sigma + H_sigma
        + sum_{k>=1} B_k/k! (-i)^k  sum [Lambda, [..., [Lambda, H]]]_sigma
        - sum_{k>=1} B_k/k! (+i)^k  sum [Lambda, [..., [Lambda, F]]]_sigma

Every Lambda in the k-fold commutators is summed over its Fourier index, and the
signatures add up to sigma. Collecting terms of equal order gives F^(n) and Lambda^(n)
from lower orders only (28a-f).

F_sigma is nonzero only on secular signatures. In the paper that is m = 0. Here it is
every |Omega| < `window_GHz` (see the package docstring). Lambda_sigma on the secular
signatures is the gauge freedom (28c-f):

    "fvv"  Lambda_sigma = 0 on every secular sigma                    (24), (29), (30)
    "aht"  Lambda(t0 = 0) = 0: Lambda_0 = -sum_fast Lambda_sigma      (31)

AHT's F then carries the t0-dependent non-secular terms of (31b). With incommensurate
frequencies and a chirped pump, that F is not a function of |eta| alone, so only FVV
survives `model`'s adiabatic step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Callable, Dict, Optional

import numpy as np

from floquet.fourier import Bins, Signature, add, bracket, comm, conj, lin, prune, zero

#: signed Bernoulli numbers B_k (B_1 = -1/2), eq. (9); odd k >= 3 vanish
BERNOULLI = {0: 1.0, 1: -0.5, 2: 1 / 6, 4: -1 / 30, 6: 1 / 42, 8: -1 / 30, 10: 5 / 66}
GAUGES = ("fvv", "aht")


@dataclass
class Expansion:
    """F^(n) (secular coefficients, GHz) and Lambda^(n) (dimensionless) by order n."""
    F: Dict[int, Bins] = field(default_factory=dict)
    Lambda: Dict[int, Bins] = field(default_factory=dict)
    gauge: str = "fvv"

    @property
    def order(self) -> int:
        return max(self.F) if self.F else 0

    def F_upto(self, n: Optional[int] = None) -> Bins:
        """F^[n] = sum_{m <= n} F^(m) (12b)."""
        n = self.order if n is None else n
        return lin(*[(1.0, self.F[m]) for m in range(1, n + 1)])

    def Lambda_upto(self, n: int) -> Bins:
        """Lambda^[n] = sum_{m <= n} Lambda^(m) (12a)."""
        return lin(*[(1.0, self.Lambda[m]) for m in range(1, n + 1) if m in self.Lambda])


def floquet_magnus(H: Bins, carrier: Callable[[Signature], float], order: int, *,
                   window_GHz: float = 0.6, gauge: str = "fvv") -> Expansion:
    """Eq. (27), order by order: F^(n) and Lambda^(n) for n = 1..order.

    H is counted as first order. T[k][q] holds the k-fold nested commutators
    [Lambda, [..., [Lambda, H]]] of total order q, and U[k][q] the same with F at the
    centre. Both are built recursively from Lambda^(m < n), which is the paper's
    integration-free scheme (App. A)."""
    if gauge not in GAUGES:
        raise ValueError(f"gauge {gauge!r}: expected one of {GAUGES}")
    secular = lambda sig: abs(carrier(sig)) < window_GHz
    n_modes = len(next(iter(H))[1])
    ex = Expansion(gauge=gauge)
    T: Dict[int, Dict[int, Bins]] = {0: {1: H}}
    U: Dict[int, Dict[int, Bins]] = {0: {}}
    for n in range(1, order + 1):
        rhs = [(1.0, H)] if n == 1 else []
        for k in range(1, n):
            T.setdefault(k, {})[n] = lin(*[(1.0, comm(ex.Lambda[m], T[k - 1][n - m]))
                                           for m in range(1, n) if n - m in T[k - 1]])
            U.setdefault(k, {})[n] = lin(*[(1.0, comm(ex.Lambda[m], U[k - 1][n - m]))
                                           for m in range(1, n) if n - m in U[k - 1]])
            Bk = BERNOULLI.get(k, 0.0)
            if Bk:
                rhs.append((Bk / factorial(k) * (-1j) ** k, T[k][n]))
                rhs.append((-Bk / factorial(k) * (1j) ** k, U[k][n]))
        R = lin(*rhs)
        Fn: Bins = {}
        Ln: Bins = {}
        for sig, M in R.items():
            if secular(sig):
                Fn[sig] = M                              # F_sigma = RHS_sigma
            else:
                Ln[sig] = (1j / carrier(sig)) * M        # -i Omega Lambda_sigma = RHS_sigma
        if gauge == "aht" and Ln:
            L0 = None
            for M in Ln.values():
                L0 = -M if L0 is None else L0 - M
            add(Ln, zero(n_modes), L0)
            Ln = prune(Ln)
        ex.F[n], ex.Lambda[n] = Fn, Ln
        U[0][n] = Fn
    return ex


# --------------------------------------------------------------------------- closed forms
# The paper's explicit low orders, written with its [n, m] notation (26) and translated
# by m w -> -Omega_m. They hold for a strictly periodic secular set (only Omega = 0 is
# secular) in the FVV gauge. `floquet_magnus` must reproduce them, and the tests check it.

def F1(H: Bins, carrier) -> Bins:
    """(28a), (29a): F^(1) = H_0."""
    return {s: M for s, M in H.items() if abs(carrier(s)) < 1e-12}


def Lambda1(H: Bins, carrier) -> Bins:
    """(28b), (30b): Lambda^(1)_m = H_m / (i m w)  ->  i H_sigma / Omega_sigma."""
    return {s: (1j / carrier(s)) * M for s, M in H.items() if abs(carrier(s)) >= 1e-12}


def F2_fvv(H: Bins, carrier) -> Bins:
    """(29b): F^(2) = sum_{n != 0} [n, -n] / (2 n w)  ->  sum [n, -n] / (-2 Omega_n)."""
    out: Bins = {}
    for n in H:
        if abs(carrier(n)) < 1e-12:
            continue
        b = bracket(H, n, conj(n))
        if b is not None:
            add(out, (0, (0,) * len(n[1])), b / (-2 * carrier(n)))
    return prune(out)


def Lambda2_fvv(H: Bins, carrier) -> Bins:
    """(30c): Lambda^(2)_m = sum_{n != 0} [n, m-n] / (2i n m w^2) + [m, 0] / (2i m^2 w^2)
    -> [n, m-n] / (2i Omega_n Omega_m) + [m, 0] / (2i Omega_m^2).

    The sum runs over every n != 0, n = m (m - n = 0) included, and the separate [m, 0]
    term comes on top. That literal reading is the one eq. (27) reproduces."""
    out: Bins = {}
    z = (0, (0,) * len(next(iter(H))[1]))
    for n in H:
        for k in H:                                   # k = m - n, any index
            m = (n[0] + k[0], tuple(a + b for a, b in zip(n[1], k[1])))
            om_n, om_m = carrier(n), carrier(m)
            if abs(om_n) < 1e-12 or abs(om_m) < 1e-12:
                continue
            add(out, m, bracket(H, n, k) / (2j * om_n * om_m))
    for m in H:
        om = carrier(m)
        if abs(om) >= 1e-12 and z in H:
            add(out, m, bracket(H, m, z) / (2j * om ** 2))
    return prune(out)


def F3_fvv(H: Bins, carrier) -> Bins:
    """(29c): F^(3) = sum_{n,m != 0} [n, [m, -m-n]] / (3 n m w^2)
                    + sum_{n != 0} [n, [0, -n]] / (6 n^2 w^2)."""
    out: Bins = {}
    z = (0, (0,) * len(next(iter(H))[1]))
    fast = [s for s in H if abs(carrier(s)) >= 1e-12]
    for n in fast:
        for m in fast:
            r = (-m[0] - n[0], tuple(-a - b for a, b in zip(m[1], n[1])))
            if r not in H:
                continue
            inner = H[m] @ H[r] - H[r] @ H[m]
            add(out, z, (H[n] @ inner - inner @ H[n]) / (3 * carrier(n) * carrier(m)))
        if z in H and conj(n) in H:
            inner = H[z] @ H[conj(n)] - H[conj(n)] @ H[z]
            add(out, z, (H[n] @ inner - inner @ H[n]) / (6 * carrier(n) ** 2))
    return prune(out)
