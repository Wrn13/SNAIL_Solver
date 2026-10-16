"""The adiabatic step: the expansion at constant |eta|, fitted in |eta| (not in the paper).

The paper treats a strictly periodic H(t). The gate's pump amplitude eta(t) has a sine^3
envelope, which is slow next to every non-secular carrier (all >= 0.6 GHz away). So the
expansion runs at a constant amplitude, on a grid of |eta|, and the pulse supplies eta(t)
afterwards. This neglects terms of order eta'/Omega.

Only |eta| needs a grid. A coefficient with s pump quanta is c(|eta|) (eta/|eta|)^s, so the
pump phase (chirp, carrier offset) is a known factor. In the plain frame, c(|eta|) is an exact
polynomial of degree <= 3n at order n (g3 only): the carriers do not depend on eta, and each
of the n factors of H_I carries at most eta^3. In the displaced frame the classical response
b_k(|eta|) is a power series, so the fit is high-degree and its residual is reported.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps

from floquet.fourier import Bins, Signature, carrier, coefficients, flux_powers, pump_frequency, term_class
from floquet.expansion import floquet_magnus

DEFAULT_ORDER = 4
DEFAULT_WINDOW_GHZ = 0.6
X_MAX_PLAIN = 1.6
#: the displaced frame's fit must stay below the fold of the classical response
#: (|eta| ~ 1.475 at delta = -140 MHz with 3 w_p = w_s kept quantum)
X_MAX_DISPLACED = 1.4


@dataclass
class Term:
    """One operator of F: ``matrix * |eta|^power * eta^|s| (eta* for s < 0) * e^{-i 2pi Omega t}``,
    in GHz, from F^(order) at signature `sig`."""
    order: int
    sig: Signature
    s: int
    power: int
    carrier_GHz: float
    matrix: sps.csr_matrix

    @property
    def kind(self) -> str:
        return term_class(self.sig)


@dataclass
class EffectiveModel:
    freqs_GHz: Tuple[float, ...]
    dims: Tuple[int, ...]
    order: int
    window_GHz: float
    gauge: str
    terms: List[Term] = field(default_factory=list)
    #: Lambda^(n) at eta = 0, the pulse ends: {n: coefficients}
    kick: Dict[int, Bins] = field(default_factory=dict)
    #: displaced frame only: {k: (powers, coefficients)} of b_k(|eta|)
    displacement: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    x_max: float = X_MAX_PLAIN
    fit_residual_GHz: float = 0.0
    displacement_fit_residual: float = 0.0

    @property
    def pump_GHz(self) -> float:
        return pump_frequency(self.freqs_GHz)

    def b_at(self, k: int, x: float) -> float:
        powers, coef = self.displacement[k]
        return float(np.dot(coef, x ** powers))

    def select(self, keep: Optional[Callable[[Term], bool]] = None) -> List[Term]:
        return [t for t in self.terms if keep is None or keep(t)]

    def F_at(self, eta: complex, t: float = 0.0, keep=None) -> np.ndarray:
        """Dense F(t) in GHz at a constant pump amplitude (diagnostics)."""
        dim = int(np.prod(self.dims))
        out = np.zeros((dim, dim), complex)
        x = abs(eta)
        for tm in self.select(keep):
            ph = (eta if tm.s > 0 else np.conj(eta)) ** abs(tm.s)
            out += tm.matrix.toarray() * x ** tm.power * ph * np.exp(-2j * np.pi * tm.carrier_GHz * t)
        return out


def _nodes(n: int, x_max: float) -> np.ndarray:
    """x = 0 (the pump-free terms and the end kick) plus Chebyshev nodes on [0, x_max]."""
    return np.concatenate([[0.0], 0.5 * x_max * (1 - np.cos(np.pi * (np.arange(n - 1) + 0.5)
                                                            / (n - 1)))])


def _fit(xs, Y, s, degree):
    """Least squares in x^|s|, x^(|s|+2), ... <= degree (on x / max x); absolute residual."""
    Y2 = Y if Y.ndim > 1 else Y[:, None]
    xm = float(np.max(xs))
    powers = np.arange(abs(s), degree + 1, 2)
    Vm = (xs / xm)[:, None] ** powers[None, :]
    ck, *_ = np.linalg.lstsq(Vm, Y2, rcond=None)
    resid = float(np.max(np.abs(Vm @ ck - Y2)))
    ck = ck / (xm ** powers)[:, None]
    return powers, (ck if Y.ndim > 1 else ck[:, 0]), resid


def _node(job):
    from floquet.classical import harmonic_balance
    b = None
    if job["displaced"]:
        b = harmonic_balance(job["freqs_GHz"], job["lam"][2], job["g"], job["x"],
                             harmonics=job["harmonics"], window_GHz=job["window_GHz"])
    H = coefficients(job["freqs_GHz"], job["lam"], job["alpha"], job["g"], job["dims"],
                     job["x"], displacement=b)
    ex = floquet_magnus(H, lambda sig: carrier(sig, job["freqs_GHz"]), job["order"],
                        window_GHz=job["window_GHz"], gauge=job["gauge"])
    return b or {}, ex.F, (ex.Lambda if job["x"] == 0.0 else None)


def build(freqs_GHz: Sequence[float], lam: Sequence[float], alpha: Sequence[float],
          g: Dict[int, float], dims: Sequence[int], *, order: int = DEFAULT_ORDER,
          window_GHz: float = DEFAULT_WINDOW_GHZ, gauge: str = "fvv", displaced: bool = False,
          harmonics: int = 6, n_nodes: Optional[int] = None, degree: Optional[int] = None,
          x_max: Optional[float] = None, workers: int = 0) -> EffectiveModel:
    """Run `floquet_magnus` at each |eta| node (in parallel) and fit every coefficient."""
    from concurrent.futures import ProcessPoolExecutor
    dims = tuple(int(d) for d in dims)
    x_max = x_max or (X_MAX_DISPLACED if displaced else X_MAX_PLAIN)
    degree = degree or (24 if displaced else 3 * order + (3 * order) % 2)
    n_nodes = n_nodes or (21 if displaced else 3 * order // 2 + 5)
    xs = _nodes(n_nodes, x_max)
    job = dict(freqs_GHz=tuple(float(f) for f in freqs_GHz), lam=tuple(float(v) for v in lam),
               alpha=tuple(float(v) for v in alpha), g={int(k): float(v) for k, v in g.items()},
               dims=dims, order=order, window_GHz=window_GHz, gauge=gauge, displaced=displaced,
               harmonics=harmonics)
    with ProcessPoolExecutor(max_workers=workers or len(xs)) as pool:
        results = list(pool.map(_node, [dict(job, x=float(x)) for x in xs]))
    model = EffectiveModel(job["freqs_GHz"], dims, order, float(window_GHz), gauge,
                           kick=results[0][2] or {}, x_max=float(x_max))
    bvals: Dict[int, List] = {}
    vals: Dict[Tuple[int, Signature], List] = {}
    for j, (b, F, _) in enumerate(results):
        for k, v in b.items():
            bvals.setdefault(k, [0.0] * len(xs))[j] = v
        for n, coeffs in F.items():
            for sig, M in coeffs.items():
                vals.setdefault((n, sig), [None] * len(xs))[j] = M
    for k, v in bvals.items():
        powers, ck, r = _fit(xs, np.real(np.asarray(v)), k, degree)
        model.displacement[k] = (powers, ck)
        model.displacement_fit_residual = max(model.displacement_fit_residual, r)
    dim = int(np.prod(dims))
    for (n, sig), Ms in vals.items():
        Y = np.stack([np.zeros((dim, dim), complex) if M is None else M.toarray() for M in Ms])
        Y = Y.reshape(len(xs), -1)
        mask = np.any(np.abs(Y) > 1e-15, axis=0)
        if not mask.any():
            continue
        deg = degree if displaced else min(degree, 3 * n + 1)
        powers, ck, r = _fit(xs, Y[:, mask], sig[0], deg)
        model.fit_residual_GHz = max(model.fit_residual_GHz, r)
        for k, row in zip(powers, ck):
            if np.max(np.abs(row)) < 1e-15:
                continue
            full = np.zeros(dim * dim, complex)
            full[mask] = row
            M = sps.csr_matrix(full.reshape(dim, dim))
            M.eliminate_zeros()
            model.terms.append(Term(n, sig, sig[0], int(k) - abs(sig[0]),
                                    float(carrier(sig, freqs_GHz)), M))
    return model


def device_of(cpl) -> Tuple:
    """(freqs, lam, alpha, g, dims) of a ZhouCoupler in the units `build` takes (GHz)."""
    if cpl.n_modes != 3 or cpl.coupler_index != 2:
        raise ValueError("the FVV models handle the (a, b, coupler) system only")
    tp = 2 * np.pi
    return (tuple(float(w) / tp for w in cpl.omega), tuple(float(v) for v in cpl.participation),
            tuple(float(v) / tp for v in cpl.anharm),
            {int(n): float(v) / tp for n, v in cpl.g_n.items()}, tuple(cpl.dims))


_MEMO: Dict[str, EffectiveModel] = {}


def for_coupler(cpl, *, cache_dir: Optional[str] = None, **kw) -> EffectiveModel:
    """`build` for this coupler's device, memoized in-process and (if `cache_dir`) on disk."""
    dev = device_of(cpl)
    blob = json.dumps({"dev": [list(dev[0]), list(dev[1]), list(dev[2]),
                               sorted(dev[3].items()), list(dev[4])],
                       "kw": kw, "v": 1}, sort_keys=True, default=str)
    key = hashlib.sha1(blob.encode()).hexdigest()[:16]
    if key in _MEMO:
        return _MEMO[key]
    path = os.path.join(cache_dir, f"fvv_{key}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            _MEMO[key] = pickle.load(f)
        return _MEMO[key]
    model = build(*dev, **kw)
    _MEMO[key] = model
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(model, f)
        os.replace(tmp, path)
    return model
