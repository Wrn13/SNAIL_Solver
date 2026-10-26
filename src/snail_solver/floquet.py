"""Floquet quasienergies of the constantly pumped device, and branch tracking in |eta|.

A pump of constant displaced amplitude |eta| on one column's device, LAB frame, no RWA:

    H(t) = sum_i w_i n_i + H_anh + sum_n g_n [ X_s + 2 |eta| cos(w_p t) ]^n ,
    X_s  = sum_i lambda_i (a_i + a_i^dag)

built from the same ``ZhouCoupler`` operators the gate solver uses. H has period
T = 1/f_p, so the one-period propagator's eigenphases are the quasienergies
``eps = -arg(lambda) / T`` (defined mod w_p) and its eigenvectors the Floquet modes.

Branch tracking (:func:`track`). Every Floquet mode is labelled by the Fock state it
is at |eta| = 0 and carried up in |eta| by the assignment of modes to labels that
maximises the total overlap with the previous step (Hungarian algorithm), so no two
branches ever claim one mode; each branch's quasienergy is unwrapped onto the zone
nearest its last value, starting from its bare energy. Per-state character is NOT a
usable label at strong drive -- by |eta| ~ 1 a coupler Fock state is spread over
several modes -- but continuity is. With fine steps a narrow avoided crossing (gap
small against the beat's change per step) is passed straight through, the branches
keeping their character; a broad one is followed round, the two labels swapping
character at its centre.

Diabatic energies (``diabatic_of``). A labelled branch keeps its continuity, not its
character: at a broad avoided crossing with a third level it turns round and leaves
its Fock state behind. The quantity that keeps the character is the diagonal of the
effective Floquet Hamiltonian in the Fock basis,

    E_s(|eta|) = sum_m |<s|m>|^2 eps_m / sum_m |<s|m>|^2 ,   |eps_m - E_s| < window,

each eps_m taken on the zone nearest E_s's previous value: the weighted mean
quasienergy of the modes that carry |s> near its own energy. The window matters:
the static g_3 dressing gives |s> small admixtures of levels whole GHz away, which
fold into the zone a few hundred MHz off and would bias a plain sum; the near-resonant
hybrids that make a crossing lie inside it. E_s goes smoothly through crossings with
other levels, and E_f - E_i - k w_p changes sign where the k-pump process i -> f is
resonant. ``spread`` (the weighted standard deviation inside the window) says how far
|s> has been split over several nearby modes, ``captured`` how much of its weight the
window holds.

The computational states |100> and |010> are degenerate by design (w_b - w_a = w_p)
and split into the iSWAP doublet; ``DOUBLET`` is the mean of their two branches on the
|100> zone (|010> sits exactly w_p above), so a k-pump process from |010> is a
(k+1)-pump one from it.

Dressed beat of a k-pump process i -> f:

    Delta_k(|eta|) = eps_f(|eta|) - eps_i(|eta|) - k w_p

At |eta| = 0 it is the statically dressed beat (g_3 dresses the levels without any
pump: the coupler's self-Kerr puts |002> ~150 MHz below 2 w_s). On the diabatic
energies it reaches zero where the drive has Stark-shifted the process into resonance
(:func:`resonance`); a pulse whose peak exceeds that |eta| crosses the resonance on
its rise and again on its fall.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm, schur
from scipy.optimize import linear_sum_assignment

TWO_PI = 2.0 * np.pi
DOUBLET = "doublet"
_PAIR = ((1, 0, 0), (0, 1, 0))


@dataclass
class FloquetModel:
    """Lab-frame pieces of H(t): ``H0`` (bare + anharmonic, rad/ns), the coupler
    flux ``Xs``, the nonlinearity ``g_n`` (rad/ns), the pump ``w_p`` (rad/ns)."""
    H0: np.ndarray
    Xs: np.ndarray
    g_n: Dict[int, float]
    w_p: float
    dims: Tuple[int, ...]
    fock_index: Callable[[Sequence[int]], int]

    @property
    def period(self) -> float:
        return TWO_PI / self.w_p

    def bare_energy(self, occ: Sequence[int]) -> float:
        i = self.fock_index(list(occ))
        return float(np.real(self.H0[i, i]))


def model_from_coupler(cpl, w_p_GHz: float) -> FloquetModel:
    """:class:`FloquetModel` of a built ``ZhouCoupler`` pumped at `w_p_GHz`."""
    H0 = sum(cpl.omega[i] * (cpl.ad_ops[i] @ cpl.a_ops[i]) for i in range(cpl.n_modes))
    H0 = np.asarray(H0, dtype=complex) + np.asarray(cpl._anharm_op, dtype=complex)
    Xs = sum(amp * np.asarray(op, dtype=complex) for op, _w, amp, _k in cpl._mode_letters)
    return FloquetModel(H0=H0, Xs=np.asarray(Xs, dtype=complex), g_n=dict(cpl.g_n),
                        w_p=float(w_p_GHz) * TWO_PI, dims=tuple(cpl.dims),
                        fock_index=cpl.fock_index)


def model_at_detuning(device: Dict[str, Any], delta_GHz: float, *, branch: str = "above",
                      levels: Optional[int] = None) -> FloquetModel:
    """The column at ``w_p = w_a/2 + delta`` on the nominal carrier (no chirp, no
    offset), exactly as the gate scan builds it."""
    from snail_solver.device_utils import build_coupler
    from snail_solver.subharmonic_convergence import _freqs, config_at_wp
    wa, _wb, _ws = _freqs(device)
    cfg = config_at_wp(device, 0.5 * wa + float(delta_GHz), branch=branch, levels=levels)
    cpl, w_p, _ = build_coupler(cfg, 100.0, 1.0, 0.0, None, None, chirp_coeffs_GHz=[])
    return model_from_coupler(cpl, w_p)


def period_propagator(model: FloquetModel, eta: float, n_steps: int = 1000) -> np.ndarray:
    """U(T) by exponential-midpoint steps; ``H(t) = A_0 + sum_k c(t)^k A_k`` with
    ``c = 2 |eta| cos(w_p t)``. 1000 steps per period are converged (checked to
    1e-12 against ``hamiltonian_matrix``)."""
    dim = model.H0.shape[0]
    nmax = max(model.g_n)
    P = [np.eye(dim, dtype=complex)]
    for _ in range(nmax):
        P.append(P[-1] @ model.Xs)
    A = [np.zeros((dim, dim), dtype=complex) for _ in range(nmax + 1)]
    for n, g in model.g_n.items():
        for k in range(n + 1):
            A[k] = A[k] + g * comb(n, k) * P[n - k]
    A[0] = A[0] + model.H0
    dt = model.period / n_steps
    U = np.eye(dim, dtype=complex)
    for j in range(n_steps):
        c = 2.0 * eta * np.cos(model.w_p * (j + 0.5) * dt)
        H = A[0] + sum(c ** k * A[k] for k in range(1, nmax + 1))
        U = expm(-1j * H * dt) @ U
    return U


def quasienergies(model: FloquetModel, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(eps rad/ns in (-w_p/2, w_p/2], modes as columns)`` of a period propagator.

    By complex Schur decomposition: U is normal, so its Schur form is diagonal and the
    Schur vectors are orthonormal eigenvectors even where eigenvalues (nearly) coincide.
    """
    T, Z = schur(U, output="complex")
    return -np.angle(np.diag(T)) / model.period, Z


def _unwrap(prev, eps, w_p: float):
    """`eps` shifted by a multiple of w_p onto the zone nearest `prev`."""
    return eps + w_p * np.round((prev - eps) / w_p)


def key(occ: Sequence[int]) -> str:
    """``(0, 0, 2) -> "002"``."""
    return "".join(map(str, occ))


def track(model: FloquetModel, etas: Sequence[float], n_steps: int = 1000,
          weights_on: Sequence[Sequence[int]] = (),
          diabatic_of: Sequence[Sequence[int]] = (),
          window_GHz: float = 0.15) -> Dict[str, Any]:
    """Label every Floquet branch by its |eta| = 0 Fock state and follow all of them,
    and the diabatic energy of each Fock state in `diabatic_of`.

    Returns ``{"eta": (n,), "labels": [key, ...] (dim,), "eps_GHz": (n, dim) unwrapped,
    "own_weight": (n, dim) each branch's population on its own label,
    "step_overlap": (n, dim) overlap with its previous-step mode,
    "weights": {key: (n, dim)} each branch's population on the Fock states in
    `weights_on`, "diabatic_GHz": {key: (n,)}, "spread_GHz": {key: (n,)},
    "captured": {key: (n,)}, "w_p_GHz"}`` (diabatic averages over `window_GHz`). A ``step_overlap`` well below 1 away from a crossing means the
    |eta| step was too coarse.
    """
    etas = np.asarray(etas, dtype=float)
    if etas[0] != 0.0:
        raise ValueError("tracking starts from |eta| = 0, where the modes are Fock states")
    dim = model.H0.shape[0]
    labels = []
    for i in range(dim):
        occ, r = [], i
        for d in reversed(model.dims):
            occ.append(r % d)
            r //= d
        labels.append(key(reversed(occ)))
    bare = np.real(np.diag(model.H0))
    w_idx = {key(o): model.fock_index(list(o)) for o in weights_on}
    eps_out = np.empty((len(etas), dim))
    own = np.empty((len(etas), dim))
    step = np.empty((len(etas), dim))
    wout = {k: np.empty((len(etas), dim)) for k in w_idx}
    d_idx = {key(o): model.fock_index(list(o)) for o in diabatic_of}
    dia = {k: np.empty(len(etas)) for k in d_idx}
    spread = {k: np.empty(len(etas)) for k in d_idx}
    captured = {k: np.empty(len(etas)) for k in d_idx}
    win = float(window_GHz) * TWO_PI
    d_prev = {k: bare[i] for k, i in d_idx.items()}
    prev_vec = np.eye(dim, dtype=complex)                  # label l <-> Fock state l
    prev_eps = bare.copy()
    for n, eta in enumerate(etas):
        eps, V = quasienergies(model, period_propagator(model, eta, n_steps))
        ov = np.abs(prev_vec.conj().T @ V) ** 2           # [label, mode]
        rows, cols = linear_sum_assignment(-ov)
        mode = np.empty(dim, dtype=int)
        mode[rows] = cols
        Vl = V[:, mode]
        prev_eps = _unwrap(prev_eps, eps[mode], model.w_p)
        prev_vec = Vl
        eps_out[n] = prev_eps / TWO_PI
        own[n] = np.abs(Vl[np.arange(dim), np.arange(dim)]) ** 2
        step[n] = ov[np.arange(dim), mode]
        for k, i in w_idx.items():
            wout[k][n] = np.abs(Vl[i, :]) ** 2
        for k, i in d_idx.items():
            w = np.abs(V[i, :]) ** 2
            E = d_prev[k]
            for _ in range(3):                             # re-centre the window
                e = _unwrap(E, eps, model.w_p)
                m = np.abs(e - E) < win
                if not m.any():
                    break
                E = float(w[m] @ e[m] / w[m].sum())
            d_prev[k] = E
            dia[k][n] = E / TWO_PI
            spread[k][n] = float(np.sqrt(w[m] @ (e[m] - E) ** 2 / w[m].sum())) / TWO_PI
            captured[k][n] = float(w[m].sum())
    return {"eta": etas, "labels": labels, "eps_GHz": eps_out, "own_weight": own,
            "step_overlap": step, "weights": wout, "diabatic_GHz": dia,
            "spread_GHz": spread, "captured": captured, "w_p_GHz": model.w_p / TWO_PI}


def branch(tr: Dict[str, Any], label: str, diabatic: bool = True) -> np.ndarray:
    """A state's quasienergy (GHz) vs |eta|: its diabatic energy (default; it must be
    in ``diabatic_of``) or its labelled branch. :data:`DOUBLET` is the mean of |100>
    and |010> on the |100> zone."""
    w_p = float(tr["w_p_GHz"])

    def one(lab):
        if diabatic:
            return np.asarray(tr["diabatic_GHz"][lab])
        return np.asarray(tr["eps_GHz"])[:, list(tr["labels"]).index(lab)]

    if label != DOUBLET:
        return one(label)
    e1, e2 = one(key(_PAIR[0])), one(key(_PAIR[1]))
    return 0.5 * (e1 + e2 - w_p * np.round((e2 - e1) / w_p))


def dressed_beat(tr: Dict[str, Any], initial: str, final: str, k: int,
                 diabatic: bool = True) -> np.ndarray:
    """``Delta_k(|eta|) = E_f - E_i - k w_p`` (GHz) along a :func:`track` result, on the
    diabatic energies by default. `initial`/`final` are keys such as ``"002"`` or
    :data:`DOUBLET`."""
    return (branch(tr, final, diabatic) - branch(tr, initial, diabatic)
            - k * float(tr["w_p_GHz"]))


def resonance(etas: Sequence[float], beat: Sequence[float],
              gap_max: float) -> Tuple[Optional[float], Optional[float], str]:
    """``(|eta|*, min |beat|, kind)`` of the first resonance of a dressed beat.

    ``kind`` is ``"cross"`` where the beat changes sign (a crossing passed straight
    through), ``"avoided"`` where |beat| has a local minimum below `gap_max` without
    changing sign (the branches turned round: the minimum is the gap), else ``"none"``.
    """
    etas = np.asarray(etas, dtype=float)
    beat = np.asarray(beat, dtype=float)
    a = np.abs(beat)
    s = np.sign(beat)
    hit = np.nonzero((s[1:] * s[:-1] < 0) & (a[1:] < gap_max) & (a[:-1] < gap_max))[0]
    loc = np.nonzero((a[1:-1] <= a[:-2]) & (a[1:-1] <= a[2:]) & (a[1:-1] < gap_max))[0] + 1
    first_cross = int(hit[0]) if hit.size else None
    first_min = int(loc[0]) if loc.size else None
    if first_cross is not None and (first_min is None or first_cross <= first_min):
        i = first_cross
        e = etas[i] - beat[i] * (etas[i + 1] - etas[i]) / (beat[i + 1] - beat[i])
        return float(e), float(min(a[i], a[i + 1])), "cross"
    if first_min is not None:
        return float(etas[first_min]), float(a[first_min]), "avoided"
    return None, None, "none"
