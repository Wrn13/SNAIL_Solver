"""The charge frame: the gate as the paper's single-frequency periodic problem.

In the interaction picture a term of H_I with signature sigma = (s, Delta) carries

    e^{-i 2pi (sum_k Delta_k f_k) t} * eta^s,   eta e^{-i w_p t} = |eta| e^{-i chi(t)}

so the pump phase chi(t) = w_nom t - theta(t) collects the carrier, the offset
dw = w_p - w_nom, and the chirp: theta(t) = arg eta(t) - 2pi dw t.

Give each mode a pump charge q_k: a^dag carries 2 pump quanta (2 w_p ~ w_a), b^dag 3
(w_b = w_a + w_p), s^dag 3 (3 w_p ~ w_s). Then rotate each mode by

    R(t) = exp(+i sum_k phi_k(t) n_k),   phi_k(t) = 2pi f_k t - q_k chi(t).

In that frame a term's phase is e^{-i Q_sigma chi(t)}, with charge Q_sigma = s + sum_k Delta_k q_k,
and the frame adds sum_k phi_k' n_k. At a constant pump (chi = w t) the Hamiltonian is
exactly periodic at the pump frequency, in the paper's notation (2):

    H(t) = sum_m H_m e^{i m w t},   H_m = sum_{Q_sigma = -m} H_sigma,
    H_0 += sum_k (2pi f_k - q_k w) n_k

So the paper applies as written: one frequency w, an integer Fourier index m, static F.
Every near-resonant process conserves the charge and so sits in H_0: the exchange, the
a<->s and b<->s conversions, both subharmonics, and 3 w_p = w_s. A non-zero index means
a frequency of at least w_p = 1.6 GHz.

With a chirp, chi' = w - theta'(t) and H_0 gains sum_k q_k theta'(t) n_k. That term is a
slow (MHz) diagonal shift. `model` treats it exactly in F^(1) and neglects it in
F^(n >= 2), an error of order theta'/w.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from floquet.fourier import Bins, Signature, pump_frequency

TWO_PI = 2.0 * np.pi
#: pump charge of one quantum of (a, b, s)
CHARGE = (2, 3, 3)


def charge(sig: Signature, q: Sequence[int] = CHARGE) -> int:
    """Q_sigma = s + sum_k Delta_k q_k. Its Fourier index in the charge frame is m = -Q."""
    return int(sig[0] + sum(d * c for d, c in zip(sig[1], q)))


def detunings_GHz(freqs_GHz: Sequence[float], q: Sequence[int] = CHARGE) -> np.ndarray:
    """nu_k = f_k - q_k w_nom: each mode's frequency in the charge frame (GHz)."""
    wp = pump_frequency(freqs_GHz)
    return np.array([float(f) - c * wp for f, c in zip(freqs_GHz, q)])


def number_diagonal(dims: Sequence[int]) -> np.ndarray:
    """occ[i, k] = n_k of basis state i."""
    dims = tuple(int(d) for d in dims)
    return np.stack(np.unravel_index(np.arange(int(np.prod(dims))), dims), axis=1)


def periodic_coefficients(bins: Bins, freqs_GHz: Sequence[float], dims: Sequence[int],
                          q: Sequence[int] = CHARGE) -> Tuple[Dict[int, np.ndarray], float]:
    """The paper's H_m (dense, rad/ns) at a constant pump in the charge frame, and w (rad/ns).

    `bins` are the interaction-picture coefficients (GHz) at real eta (`fourier.coefficients`)."""
    w = TWO_PI * pump_frequency(freqs_GHz)
    dim = int(np.prod(dims))
    H: Dict[int, np.ndarray] = {}
    for sig, M in bins.items():
        m = -charge(sig, q)
        H[m] = H[m] + TWO_PI * M.toarray() if m in H else TWO_PI * M.toarray()
    occ = number_diagonal(dims)
    H0 = np.diag((occ @ (TWO_PI * detunings_GHz(freqs_GHz, q))).astype(complex))
    H[0] = H.get(0, np.zeros((dim, dim), complex)) + H0
    return H, w


def rotation(dims: Sequence[int], phi: np.ndarray) -> np.ndarray:
    """R = exp(+i sum_k phi_k n_k), diagonal, as a vector of phases."""
    return np.exp(1j * (number_diagonal(dims) @ np.asarray(phi, float)))


class PumpPhase:
    """chi(t) and theta'(t) for a coupler's pulse, robust where the pulse vanishes.

    theta(t) = arg eta(t) - 2pi dw t, and chi = w_nom t - theta. arg eta is undefined
    where eta = 0, and at the pulse ends the envelope rounds to -0 or -1e-18, which flips
    arg eta by pi. So the phase is taken at the time clamped to [t_a, t_b], the span where
    |eta| exceeds `rel` times its peak. Any smooth chi gives an exact frame, provided R and
    theta' use the same chi. Outside [t_a, t_b] the terms carry |eta|^s < rel^s, so the
    true phase there does not matter."""

    def __init__(self, cpl, tone, w_nom: float, rel: float = 1e-8, h: float = 1e-4):
        self.cpl, self.tone, self.w = cpl, tone, float(w_nom)
        self.dw = TWO_PI * (float(tone.w_p_GHz) - self.w / TWO_PI)
        self.h = h
        T = float(getattr(tone.envelope, "t_g", 0.0) or 0.0)
        if T > 0:
            ts = np.linspace(0.0, T, 4001)
            mag = np.array([abs(cpl._eta(tone, t)) for t in ts])
            on = np.flatnonzero(mag > rel * max(mag.max(), 1e-300))
            self.t_a, self.t_b = (ts[on[0]], ts[on[-1]]) if len(on) else (0.0, T)
        else:
            self.t_a, self.t_b = -np.inf, np.inf

    def _arg(self, t: float) -> float:
        e = self.cpl._eta(self.tone, min(max(t, self.t_a), self.t_b))
        return float(np.angle(e)) if abs(e) > 0 else 0.0

    def __call__(self, t: float) -> Tuple[float, float]:
        """(chi(t), theta'(t)), rad and rad/ns."""
        a, b = self._arg(t + self.h), self._arg(t - self.h)
        dth = np.angle(np.exp(1j * (a - b))) / (2 * self.h)
        return self.w * t - (self._arg(t) - self.dw * t), dth - self.dw


def pump_phase(cpl, tone, t: float, w_nom: float) -> Tuple[float, float]:
    """(chi(t), theta'(t)) without caching (see `PumpPhase`)."""
    return PumpPhase(cpl, tone, w_nom)(t)
