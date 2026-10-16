"""The adiabatic step for a shaped pulse (not in the paper).

The paper's expansion assumes a constant drive. The gate's pump amplitude |eta(t)| has a
sine^3 envelope, which is slow next to the drive frequency w_p. So `floquet_magnus` runs at
constant |eta| on a grid of |eta|, and the pulse supplies |eta(t)|. Terms of order
|eta|'/w are neglected.

In the charge frame (`frame`) the pump phase is absorbed, so F depends on |eta| alone. Each
F^(n) is a polynomial of degree <= 3n in |eta|, because each of the n factors of H carries
at most eta^3. The fit on the grid is therefore exact. Each F^(n) is stored split by
process: the change Delta of the mode occupations between the states it connects. In the
charge frame Delta fixes the pump number s = -sum_k Delta_k q_k, so Delta names the
process (e.g. (1, -1, 0) is the exchange eta a b^dag).
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

from floquet import fourier as Fo
from floquet.expansion import Coeffs, floquet_magnus
from floquet.frame import CHARGE, TWO_PI, number_diagonal, periodic_coefficients

DEFAULT_ORDER = 4
X_MAX = 1.6
Delta = Tuple[int, ...]


def process(delta: Delta, q=CHARGE) -> Fo.Signature:
    """The signature (s, Delta) of a charge-conserving process: s = -sum_k Delta_k q_k."""
    return (-int(sum(d * c for d, c in zip(delta, q))), tuple(int(d) for d in delta))


def describe(delta: Delta) -> str:
    return Fo.describe(process(delta))


def split_by_process(M: np.ndarray, dims: Sequence[int]) -> Dict[Delta, np.ndarray]:
    """M = sum_Delta M_Delta, with M_Delta connecting states whose occupations differ by
    Delta (annihilators count +1, as in `fourier`)."""
    occ = number_diagonal(dims)
    rows, cols = np.nonzero(np.abs(M) > 1e-15)
    out: Dict[Delta, np.ndarray] = {}
    for r, c in zip(rows, cols):
        d = tuple(int(x) for x in occ[c] - occ[r])
        out.setdefault(d, np.zeros_like(M))[r, c] = M[r, c]
    return out


@dataclass
class Term:
    """One piece of F: ``matrix * |eta|^power`` (rad/ns), from F^(order), process `delta`."""
    order: int
    delta: Delta
    power: int
    matrix: sps.csr_matrix

    @property
    def kind(self) -> str:
        return Fo.term_class(process(self.delta))

    @property
    def name(self) -> str:
        return describe(self.delta)


@dataclass
class EffectiveModel:
    freqs_GHz: Tuple[float, ...]
    dims: Tuple[int, ...]
    order: int
    gauge: str
    omega: float                                   # w = 2pi w_nom (rad/ns)
    terms: List[Term] = field(default_factory=list)
    #: the frame's sum_k (2pi f_k - q_k w) n_k, part of F^(1) = H_0 (rad/ns)
    frame_diagonal: np.ndarray = None
    #: Lambda^(n)_m at eta = 0 (the pulse ends): {n: {m: dense}}
    kick: Dict[int, Coeffs] = field(default_factory=dict)
    x_max: float = X_MAX
    fit_residual: float = 0.0

    def select(self, keep: Optional[Callable[[Term], bool]] = None) -> List[Term]:
        return [t for t in self.terms if keep is None or keep(t)]

    def F_at(self, x: float, keep=None, frame: bool = True) -> np.ndarray:
        """F(|eta| = x), dense, rad/ns (with or without the frame diagonal)."""
        dim = int(np.prod(self.dims))
        out = np.diag(self.frame_diagonal).astype(complex) if frame else np.zeros((dim, dim), complex)
        for t in self.select(keep):
            out = out + t.matrix.toarray() * x ** t.power
        return out

    def kick_at(self, chi: float, upto: int) -> np.ndarray:
        """Lambda^[upto] at eta = 0 and pump phase chi: sum_m Lambda_m e^{i m chi} (eq. 5)."""
        dim = int(np.prod(self.dims))
        out = np.zeros((dim, dim), complex)
        for n in range(1, upto + 1):
            for m, M in self.kick.get(n, {}).items():
                out = out + M * np.exp(1j * m * chi)
        return out


def expansion_at(freqs_GHz, lam, alpha, g, dims, x: float, order: int, gauge: str = "fvv"):
    """`floquet_magnus` for the device at constant |eta| = x, in the charge frame."""
    bins = Fo.coefficients(freqs_GHz, lam, alpha, g, dims, x)
    H, w = periodic_coefficients(bins, freqs_GHz, dims)
    return floquet_magnus(H, w, order, gauge=gauge), H


def build(freqs_GHz: Sequence[float], lam: Sequence[float], alpha: Sequence[float],
          g: Dict[int, float], dims: Sequence[int], *, order: int = DEFAULT_ORDER,
          gauge: str = "fvv", x_max: float = X_MAX, n_nodes: Optional[int] = None
          ) -> EffectiveModel:
    """Run the expansion on a grid of |eta| and fit each F^(n) exactly in |eta|."""
    dims = tuple(int(d) for d in dims)
    n_nodes = n_nodes or 3 * order + 4
    xs = np.concatenate([[0.0], 0.5 * x_max * (1 - np.cos(np.pi * (np.arange(n_nodes - 1) + 0.5)
                                                          / (n_nodes - 1)))])
    Fs, kick = [], {}
    for x in xs:
        ex, _ = expansion_at(freqs_GHz, lam, alpha, g, dims, float(x), order, gauge)
        Fs.append(ex.F)
        if x == 0.0:
            kick = ex.Lambda
    occ = number_diagonal(dims)
    from floquet.frame import detunings_GHz
    frame_diag = (occ @ (TWO_PI * detunings_GHz(freqs_GHz))).astype(float)
    model = EffectiveModel(tuple(float(f) for f in freqs_GHz), dims, order, gauge,
                           TWO_PI * Fo.pump_frequency(freqs_GHz), frame_diagonal=frame_diag,
                           kick=kick, x_max=float(x_max))
    dim = int(np.prod(dims))
    for n in range(1, order + 1):
        Y = np.stack([F[n] - (np.diag(frame_diag) if n == 1 else 0) for F in Fs]).reshape(len(xs), -1)
        mask = np.any(np.abs(Y) > 1e-15, axis=0)
        if not mask.any():
            continue
        powers = np.arange(0, 3 * n + 1)
        V = (xs / x_max)[:, None] ** powers[None, :]
        ck, *_ = np.linalg.lstsq(V, Y[:, mask], rcond=None)
        model.fit_residual = max(model.fit_residual, float(np.max(np.abs(V @ ck - Y[:, mask]))))
        ck = ck / (x_max ** powers)[:, None]
        for p, row in zip(powers, ck):
            if np.max(np.abs(row)) < 1e-14:
                continue
            full = np.zeros(dim * dim, complex)
            full[mask] = row
            for d, Md in split_by_process(full.reshape(dim, dim), dims).items():
                model.terms.append(Term(n, d, int(p), sps.csr_matrix(Md)))
    return model


def device_of(cpl) -> Tuple:
    """(freqs, lam, alpha, g, dims) of a ZhouCoupler in GHz, as `build` takes them."""
    if cpl.n_modes != 3 or cpl.coupler_index != 2:
        raise ValueError("the (a, b, coupler) system only")
    return (tuple(float(w) / TWO_PI for w in cpl.omega), tuple(float(v) for v in cpl.participation),
            tuple(float(v) / TWO_PI for v in cpl.anharm),
            {int(n): float(v) / TWO_PI for n, v in cpl.g_n.items()}, tuple(cpl.dims))


_MEMO: Dict[str, EffectiveModel] = {}


def for_coupler(cpl, *, cache_dir: Optional[str] = None, **kw) -> EffectiveModel:
    """`build` for this coupler's device, memoized in-process and (if `cache_dir`) on disk."""
    dev = device_of(cpl)
    blob = json.dumps({"dev": [list(dev[0]), list(dev[1]), list(dev[2]), sorted(dev[3].items()),
                               list(dev[4])], "kw": kw, "v": 2}, sort_keys=True, default=str)
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
