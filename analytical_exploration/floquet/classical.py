"""The coupler's classical response to the pump (not in the paper; used by the displaced frame).

The pump drives the coupler linearly through n g_n lam_s [x_cl^(n-1)]_k s^dag. The SNAIL
subharmonic eta^2 s^dag (k = 2) is about 0.3 GHz at a 1.5 GHz detuning for |eta| = 1.3.
Treating that response as a perturbation of the undriven vacuum climbs the coupler ladder.
Instead, the response is found classically and removed by a displacement.

With beta e^{-i w_s t} = sum_k b_k e^{-i k w_p t}, the classical flux is

    x_cl(t) = eta e^{-i w_p t} + lam_s beta e^{-i w_s t} + c.c. = sum_k c_k e^{-i k w_p t}

and harmonic balance at constant |eta| reads

    b_k = F_k / Omega_k,   F_k = sum_n n g_n lam_s [x_cl^(n-1)]_k,   Omega_k = k w_p - w_s.

Harmonics with |Omega_k| < `window_GHz` are not displaced. They stay quantum and are kept
in F, like any secular term. 3 w_p = w_s is the near-resonant one: solving it classically
too moves the fold of the response from |eta| ~ 1.475 to ~1.96 (delta = -140 MHz).
"""
from __future__ import annotations

from typing import Dict

from floquet.fourier import COUPLER, pump_frequency

DEFAULT_HARMONICS = 6


def _conv(p: Dict[int, complex], q: Dict[int, complex], K: int) -> Dict[int, complex]:
    out: Dict[int, complex] = {}
    for i, u in p.items():
        for j, v in q.items():
            if abs(i + j) <= 2 * K:
                out[i + j] = out.get(i + j, 0.0) + u * v
    return out


def classical_flux(b: Dict[int, complex], eta: float, lam_s: float) -> Dict[int, complex]:
    """x_cl's pump harmonics {k: c_k} (a real field: c_-k = c_k*)."""
    c: Dict[int, complex] = {1: complex(eta), -1: complex(eta)}
    for k, v in b.items():
        c[k] = c.get(k, 0.0) + lam_s * v
        c[-k] = c.get(-k, 0.0) + lam_s * complex(v).conjugate()
    return {k: v for k, v in c.items() if v != 0}


def linear_drive(c: Dict[int, complex], g: Dict[int, float], lam_s: float,
                 K: int) -> Dict[int, complex]:
    """F_k: the coefficient of s^dag e^{-i (k w_p - w_s) t} in sum_n g_n x_cl^n (GHz)."""
    F: Dict[int, complex] = {}
    for n, gn in g.items():
        p: Dict[int, complex] = {0: 1.0}
        for _ in range(int(n) - 1):
            p = _conv(p, c, K)
        for k, v in p.items():
            F[k] = F.get(k, 0.0) + int(n) * gn * lam_s * v
    return F


def harmonic_balance(freqs_GHz, lam_s: float, g: Dict[int, float], eta: float, *,
                     harmonics: int = DEFAULT_HARMONICS, window_GHz: float = 0.6,
                     tol: float = 1e-14, max_iter: int = 2000,
                     damping: float = 0.5) -> Dict[int, complex]:
    """{k: b_k} for the displaced harmonics |k| <= `harmonics` at real pump amplitude eta,
    by damped fixed-point iteration. It raises past the fold, where the solution ends."""
    fs = float(freqs_GHz[COUPLER])
    wp = pump_frequency(freqs_GHz)
    ks = [k for k in range(-harmonics, harmonics + 1) if abs(k * wp - fs) >= window_GHz]
    b: Dict[int, complex] = {k: 0.0 for k in ks}
    err = 0.0
    for _ in range(max_iter):
        F = linear_drive(classical_flux(b, eta, lam_s), g, lam_s, harmonics)
        new = {k: F.get(k, 0.0) / (k * wp - fs) for k in ks}
        err = max((abs(new[k] - b[k]) for k in ks), default=0.0)
        b = {k: (1 - damping) * b[k] + damping * new[k] for k in ks}
        if err < tol:
            return b
    raise RuntimeError(f"harmonic balance did not converge at eta={eta} (step {err:.2e}); "
                       "past the fold of the classical response?")
