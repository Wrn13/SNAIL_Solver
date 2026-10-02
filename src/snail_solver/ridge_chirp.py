"""The Stark law read off the measured ridge itself, not a truncated power series.

``tune_up`` fits the Rabi ridge with ``delta0 + k2|eta|^2 + k4|eta|^4`` and builds
the chirp from that. Two failures follow from the TRUNCATION, not the measurement:

* where the series does not converge (``|k4 eta*^4 / k2 eta*^2| > 0.25``; 38/61
  columns of the 2026-09-22 5 MHz grid) the chirp rests on a law that is not a law;
* where an avoided crossing moves through the drive sweep, the ridge steeps sharply
  (+60 MHz, eta* = 1.3: 2 MHz inside one 0.026 step of |eta|, against a 4.7 MHz
  half-linewidth) and no low-order polynomial follows it, so the column was declared
  a ``StarkCrossingInSweep`` and calibrated with NO chirp -- although the ridge is
  the peak of P(|10>), which is exactly what the gate transfers into.

Here the instantaneous shift ``delta(x)``, ``x = |eta|``, is a piecewise-linear
function of ``x^2`` on the measured drives (so ``delta ~ x^2`` near zero, as a Stark
shift must be), with ``delta(0) = 0`` and a static ``delta0``. A SHAPED probe reports
the envelope average ``R(eta) = delta0 + int w(u) delta(eta s(u)) du`` (the same
operator whose polynomial special case is :func:`stark_chirp.stark_moments`'s
``M2``/``M4``), so the nodes are found by regularized least squares through that
operator; for a constant probe the operator is plain interpolation.

The inversion is ill-posed (the operator averages), so it is regularized by a
second-difference penalty and the weight is set by the DISCREPANCY principle: the
smoothest law that still reproduces the ridge to ``2 sigma``, with ``sigma`` the
ridge's own row-to-row noise (robust, from its second differences, so a crossing's
few large steps do not inflate it; floor 0.01 MHz). The 2026-09-22 ridges carry
``sigma ~ 0.005 MHz``. A fixed 0.1 MHz tolerance over-smooths them (0.13 MHz law error
on a clean column); cross-validation under-smooths a near-noiseless ridge and rings.

Past the last row that measured (rejected rows at the top of a sweep), the shift is
HELD at its last value, the pipeline's held-row convention: it under-states the shift
rather than extrapolate a slope the data never showed.

Two modes in :func:`law_from_table`:

* ``direct`` (default): the ridge IS the law, smoothed at its own noise level, with
  the drive-dependent part divided by the shaped probe's ``M2``. Exact for the
  ``|eta|^2`` term, ~10% low on ``|eta|^4`` (``M4/M2 = 0.90`` for the m=3 sine-power
  envelope), and it cannot ring: it follows a step as measured.
* ``deconvolve``: the regularized inversion above. Unbiased on a smooth ridge, but it
  oscillates by ~1 MHz through a crossing (+60 MHz: 4.5 -> 3.6 -> 5.4 MHz).

No solve happens here: everything is linear algebra on a stored Rabi table.
"""
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

#: Quadrature points over the pulse for the shaped-probe operator.
N_PULSE_QUAD = 801

#: Search range for the second-difference penalty, relative to ||A^T W A||.
LAMBDA_GRID = np.logspace(-6, 3, 37)

#: Discrepancy safety factor on the estimated ridge noise, and its floor (MHz).
NOISE_FACTOR = 2.0
NOISE_FLOOR_MHZ = 0.01


def ridge_noise_MHz(ridge_MHz: Sequence[float]) -> float:
    """Robust row-to-row noise of a ridge: MAD of second differences / sqrt(6).

    For white noise of rms ``sigma`` the second difference has rms ``sqrt(6) sigma``;
    the median ignores the handful of steps a crossing contributes.
    """
    r = np.asarray(ridge_MHz, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float("nan")
    d2 = np.diff(r, 2)
    return float(1.4826 * np.median(np.abs(d2)) / np.sqrt(6.0))


def probe_weight(shape_fn: Callable, weighting: str, u: np.ndarray) -> np.ndarray:
    """The chevron's averaging weight over the pulse, normalized to unit trapezoid area.

    Same three conventions as :func:`stark_chirp.stark_moments` (``rabi``:
    ``sin theta(t)``; ``coupling``: ``|eta(t)|``; ``uniform``: 1). `u` ascending on
    ``[-1, 1]``.
    """
    from snail_solver.stark_chirp import MOMENT_WEIGHTINGS, rabi_angle
    weighting = str(weighting)
    if weighting not in MOMENT_WEIGHTINGS:
        raise ValueError(f"weighting={weighting!r}: expected one of "
                         f"{list(MOMENT_WEIGHTINGS)}")
    f = np.clip(np.asarray(shape_fn(u), dtype=float), 0.0, None)
    if weighting == "uniform":
        w = np.ones_like(u)
    elif weighting == "coupling":
        w = np.sqrt(f)
    else:
        w = np.sin(rabi_angle(shape_fn, u))
    norm = np.trapz(w, u)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{weighting} weighting has no positive normalization")
    return w / norm


def _hat_matrix(x2_eval: np.ndarray, nodes2: np.ndarray) -> np.ndarray:
    """``H[k, j]``: weight of node j in the piecewise-linear interpolant (in x^2) at
    each evaluation point; the node at 0 is EXCLUDED (``delta(0) = 0`` is pinned)."""
    x2 = np.clip(np.asarray(x2_eval, dtype=float), nodes2[0], nodes2[-1])
    n = nodes2.size
    idx = np.clip(np.searchsorted(nodes2, x2, side="right") - 1, 0, n - 2)
    t = (x2 - nodes2[idx]) / (nodes2[idx + 1] - nodes2[idx])
    H = np.zeros((x2.size, n))
    rows = np.arange(x2.size)
    H[rows, idx] += 1.0 - t
    H[rows, idx + 1] += t
    return H[:, 1:]


def ridge_law(eta: Sequence[float], ridge_MHz: Sequence[float], *,
              weights: Optional[Sequence[float]] = None,
              shape_fn: Optional[Callable] = None, weighting: str = "rabi",
              noise_MHz: Optional[float] = None,
              lam: Optional[float] = None) -> Dict[str, Any]:
    """The pointwise shift ``delta(|eta|)`` (MHz) that reproduces a measured ridge.

    Parameters
    ----------
    eta, ridge_MHz : sequences
        The Rabi table's drive ladder and ridge (``delta_MHz``). Non-finite rows are
        skipped (dropped chevrons).
    weights : sequence, optional
        Per-row fit weights (the table's ``quality``); default 1.
    shape_fn : callable, optional
        ``f(u) = |eta(u)|^2 / eta_peak^2`` on ``u in [-1, 1]`` for a SHAPED probe;
        None for a constant probe (the ridge is then pointwise already).
    weighting : str
        The shaped probe's averaging convention (see :func:`probe_weight`).
    noise_MHz : float, optional
        The rms ridge misfit allowed; the LARGEST smoothing weight within it wins.
        Default ``max(NOISE_FACTOR * ridge_noise_MHz, NOISE_FLOOR_MHZ)``.
    lam : float, optional
        Fixed smoothing weight (relative), overriding `noise_MHz`.

    Returns
    -------
    dict
        ``nodes_eta`` (0 then the measured drives), ``delta_MHz`` at the nodes
        (``delta[0] = 0``), ``delta0`` (static, MHz), ``resid_MHz`` (rms of the
        reproduced ridge), ``lam``, ``noise_MHz``, ``n_used``, ``shaped``,
        ``weighting``.
    """
    eta = np.asarray(eta, dtype=float)
    R = np.asarray(ridge_MHz, dtype=float)
    q = np.ones_like(R) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(eta) & np.isfinite(R) & np.isfinite(q) & (q > 0) & (eta > 0)
    if ok.sum() < 4:
        raise ValueError(f"ridge_law needs >= 4 usable rows, got {int(ok.sum())}")
    order = np.argsort(eta[ok])
    x, R, q = eta[ok][order], R[ok][order], q[ok][order]
    if noise_MHz is None:
        sig = ridge_noise_MHz(R)
        noise_MHz = max(NOISE_FACTOR * (sig if np.isfinite(sig) else 0.0),
                        NOISE_FLOOR_MHZ)
    nodes = np.concatenate([[0.0], x])
    nodes2 = nodes ** 2
    if np.any(np.diff(nodes2) <= 0):
        raise ValueError("ridge_law needs distinct drive values")

    if shape_fn is None:
        A = _hat_matrix(x ** 2, nodes2)
    else:
        u = np.linspace(-1.0, 1.0, N_PULSE_QUAD)
        w = probe_weight(shape_fn, weighting, u)
        f = np.clip(np.asarray(shape_fn(u), dtype=float), 0.0, None)
        tw = np.full(u.size, u[1] - u[0])
        tw[[0, -1]] *= 0.5                                    # trapezoid weights
        A = np.empty((x.size, nodes.size - 1))
        for i, xi in enumerate(x):
            A[i] = (tw * w) @ _hat_matrix(xi ** 2 * f, nodes2)
    # unknowns: delta at nodes[1:], then delta0
    A = np.hstack([A, np.ones((x.size, 1))])
    n = nodes.size - 1
    # second difference in x^2 on the shift nodes (delta0 unpenalized); the pinned
    # delta(0) = 0 enters the first row as a known zero
    h = np.diff(nodes2)
    L = np.zeros((max(n - 1, 0), n + 1))
    for k in range(n - 1):
        # nodes k, k+1, k+2 (shifted: node index k is unknown k-1, k=0 is the pin)
        a, b = h[k], h[k + 1]
        coef = (2.0 / (a * (a + b)), -2.0 / (a * b), 2.0 / (b * (a + b)))
        for c, j in zip(coef, (k - 1, k, k + 1)):
            if j >= 0:
                L[k, j] += c
    Wd = q / np.mean(q)
    AtWA = A.T @ (Wd[:, None] * A)
    LtL = L.T @ L
    scale = np.trace(AtWA) / max(np.trace(LtL), 1e-300)
    AtWR = A.T @ (Wd * R)

    def solve(lr):
        M = AtWA + lr * scale * LtL
        return np.linalg.solve(M, AtWR), M

    if lam is None:
        # smoothest weight whose rms misfit stays within noise_MHz; else the closest
        fits = []
        for lr in LAMBDA_GRID:
            try:
                sol, _M = solve(lr)
            except np.linalg.LinAlgError:
                continue
            fits.append((lr, float(np.sqrt(np.mean((R - A @ sol) ** 2)))))
        if not fits:
            raise ValueError("ridge_law: no well-posed smoothing weight")
        within = [lr for lr, r in fits if r <= float(noise_MHz)]
        lam = max(within) if within else min(fits, key=lambda t: t[1])[0]
    sol, _M = solve(float(lam))
    res = R - A @ sol
    return {"nodes_eta": nodes.tolist(),
            "delta_MHz": [0.0] + [float(v) for v in sol[:n]],
            "delta0": float(sol[n]),
            "resid_MHz": float(np.sqrt(np.mean(res ** 2))),
            "lam": float(lam), "noise_MHz": float(noise_MHz), "n_used": int(x.size),
            "shaped": shape_fn is not None, "weighting": str(weighting)}


def shift_MHz(law: Dict[str, Any], a) -> np.ndarray:
    """The law's drive-dependent shift (MHz) at amplitudes `a` (no ``delta0``).

    Piecewise linear in ``|eta|^2`` between nodes; HELD at the last node past the last
    measured drive (``np.interp``'s edge rule).
    """
    nodes2 = np.asarray(law["nodes_eta"], dtype=float) ** 2
    d = np.asarray(law["delta_MHz"], dtype=float)
    a2 = np.abs(np.asarray(a, dtype=float)) ** 2
    return np.interp(a2, nodes2, d)


#: Tracking: a row may continue the ridge only inside this many half-widths (or grid
#: steps, or MHz, whichever is widest) of where the rows below predict it.
TRACK_HWHM = 2.5
TRACK_STEPS = 3
TRACK_MIN_MHZ = 3.0
#: ...and its windowed centre must land within this many half-widths (or grid steps)
#: of the prediction: a window wide enough to FIT a Lorentzian is too wide to judge
#: continuity by (+20 MHz: the top row jumped 4.4 MHz inside a 7.7 MHz window).
TRACK_ACCEPT_HWHM = 1.0
TRACK_ACCEPT_STEPS = 1.5
#: ...and only while its windowed peak stands this far above the window's floor.
TRACK_CONTRAST_MIN = 0.25
#: Tracked rows count for this fraction of their quality weight in the law fit.
TRACK_WEIGHT = 0.5


def track_ridge(chevrons: Sequence[Dict[str, Any]], *, n_seed: int = 3,
                max_misses: int = 2, **quality_kw):
    """The ridge, CONTINUED through rows the whole-row gate rejected.

    ``ridge_refit.gate_rows`` judges each row on its own: a second transition entering
    the window (``multi_peak``) or skewing the fit (``poor_fit``) discards the row even
    when the main peak is clean (+20 MHz, eta* = 1.3: the peak at +4 -> +8 MHz holds
    contrast 0.98 -> 0.75 while a competitor at -14 MHz kills every row above
    |eta| = 0.88). Here each row is fitted only in a window around where the rows
    below put the ridge (linear in |eta|^2 through the last `n_seed` rows), so a
    competing peak elsewhere in the row cannot capture it.

    A row the gate ACCEPTED is kept as measured. A rejected row is recovered when its
    windowed peak clears :data:`TRACK_CONTRAST_MIN` and the windowed Lorentzian lands
    inside the window; it then carries :data:`TRACK_WEIGHT` of its quality weight.
    Tracking stops after `max_misses` consecutive unrecoverable rows: a peak that has
    genuinely faded (low contrast all the way) is NOT followed.

    Returns ``(ridge_MHz, weights, status)`` with ``status`` per row: ``"gated"``,
    ``"tracked"``, or the original reject reason.
    """
    from snail_solver.ridge_refit import gate_rows
    from snail_solver.tune_up import chevron_quality, fit_chevron_center

    ridge, wts, rej = gate_rows(chevrons, **quality_kw)
    ridge = np.array(ridge, dtype=float)
    wts = np.array(wts, dtype=float)
    status = ["gated" if r is None else str(r) for r in rej]
    eta = np.asarray([float(c["eta"]) for c in chevrons])
    order = np.argsort(eta)
    done: list = []                       # (eta, centre_MHz, hwhm_MHz) along the ridge
    misses = 0
    qkw = {k: v for k, v in quality_kw.items() if k != "contrast_min"}
    for i in order:
        c = chevrons[i]
        if np.isfinite(ridge[i]):
            hw = float((c.get("fit") or {}).get("hwhm_GHz", np.nan)) * 1e3
            done.append((eta[i], ridge[i], hw))
            misses = 0
            continue
        if len(done) < 2 or misses >= int(max_misses):
            continue
        tail = done[-int(n_seed):]
        e2 = np.array([t[0] for t in tail]) ** 2
        cen = np.array([t[1] for t in tail])
        slope, icpt = np.polyfit(e2, cen, 1) if len(tail) >= 2 else (0.0, cen[-1])
        pred = float(slope * eta[i] ** 2 + icpt)
        off = np.asarray(c["offsets_GHz"], dtype=float) * 1e3
        met = np.asarray(c["metric"], dtype=float)
        step = float(np.median(np.diff(off))) if off.size > 1 else 1.0
        hws = [t[2] for t in tail if np.isfinite(t[2])]
        W = max(TRACK_HWHM * (np.median(hws) if hws else 0.0),
                TRACK_STEPS * step, TRACK_MIN_MHZ)
        sl = (off >= pred - W) & (off <= pred + W)
        ok = False
        if sl.sum() >= 5:
            o, m = off[sl], met[sl]
            j = int(np.nanargmax(m))
            interior = 0 < j < m.size - 1
            if interior and float(m[j] - np.nanmin(m)) >= TRACK_CONTRAST_MIN:
                cen_fit = fit_chevron_center(o / 1e3, m)
                q = chevron_quality(cen_fit, o / 1e3, m, 2.0 * W,
                                    contrast_min=TRACK_CONTRAST_MIN, **qkw)
                x = float(cen_fit["center_GHz"]) * 1e3
                tol = max(TRACK_ACCEPT_HWHM * (np.median(hws) if hws else 0.0),
                          TRACK_ACCEPT_STEPS * step)
                # the rail rule: a centre within a grid step of the ROW's edge is the
                # window's boundary, not a measurement (-5 MHz, |eta| = 0.81)
                half = float(c.get("span_MHz", np.nan)) / 2.0
                on_edge = np.isfinite(half) and abs(abs(x) - half) <= step
                if (cen_fit.get("ok") and q["reject"] in (None, "high_leakage")
                        and abs(x - pred) <= tol and not on_edge):
                    ridge[i] = x
                    wts[i] = TRACK_WEIGHT * float(q["weight"])
                    status[i] = "tracked"
                    done.append((eta[i], x, float(cen_fit["hwhm_GHz"]) * 1e3))
                    ok = True
        misses = 0 if ok else misses + 1
    return ridge, wts, status


def probe_shape_fn(config: Dict[str, Any]) -> Callable:
    """``f(u) = |eta(u)|^2 / eta_peak^2`` for this config's envelope (the shaped probe
    plays the gate's own shape; see :func:`tune_up.probe_moments`)."""
    from snail_solver.tune_up import _shape_envelope, shape_config
    shape, shape_kw = shape_config(config)
    env = _shape_envelope(shape, shape_kw)

    def f(u):
        return np.abs(np.asarray(env.value_at(np.asarray(u, dtype=float) + 1.0, np),
                                 dtype=complex)) ** 2
    return f


def law_from_table(table: Dict[str, Any], config: Dict[str, Any], target_eta: float,
                   *, probe_shape: str = "gate", moment_weighting: str = "rabi",
                   noise_MHz: Optional[float] = None, t_g_ns: Optional[float] = None,
                   mode: str = "direct", **quality_kw) -> Dict[str, Any]:
    """:func:`ridge_law` for a Rabi table, live or stored, plus the chirp verdict.

    The ridge is re-derived from the chevrons and continued through rows the
    whole-row gate rejected (:func:`track_ridge`): a stored ``delta_MHz`` is post-hold. Adds ``railed`` (some row's centre sits on its scan
    edge -- not a measurement, so no chirp can be built), ``excursion_MHz`` (the law's
    swing at `target_eta`) and ``excursion_frac_linewidth`` (against ``1/(2 t_g)``,
    the same discriminator ``tune_up`` uses for a genuinely chirp-free column).
    `mode` is ``direct`` or ``deconvolve`` (module docstring). `quality_kw` goes to
    ``tune_up.chevron_quality`` (``contrast_min``, ...).
    """
    from snail_solver.tune_up import nominal_t_g

    chev = table.get("chevrons") or []
    if not chev:
        raise ValueError("law_from_table needs the table's chevrons")
    ridge, wts, status = track_ridge(chev, **quality_kw)
    eta = np.asarray([float(c["eta"]) for c in chev])
    spans = np.asarray([float(c.get("span_MHz", np.nan)) for c in chev])
    n_off = np.asarray([len(c["offsets_GHz"]) for c in chev], dtype=float)
    step = spans / np.maximum(n_off - 1.0, 1.0)
    railed = np.isfinite(ridge) & (np.abs(np.abs(ridge) - spans / 2.0) <= step)
    shaped = str(probe_shape) != "constant"
    wq = np.where(wts > 0, wts, np.nan)
    if str(mode) == "deconvolve":
        law = ridge_law(eta, ridge, weights=wq,
                        shape_fn=probe_shape_fn(config) if shaped else None,
                        weighting=moment_weighting, noise_MHz=noise_MHz)
        law["M2"] = None
    elif str(mode) == "direct":
        law = ridge_law(eta, ridge, weights=wq, shape_fn=None, noise_MHz=noise_MHz)
        m2 = 1.0
        if shaped:
            from snail_solver.tune_up import probe_moments
            m2 = float(probe_moments(config, moment_weighting)[0])
        law["delta_MHz"] = [float(v) / m2 for v in law["delta_MHz"]]
        law["M2"] = m2
        law["shaped"] = shaped
    else:
        raise ValueError(f"mode={mode!r}: expected 'direct' or 'deconvolve'")
    law["mode"] = str(mode)
    t_g = float(t_g_ns or table.get("t_g_ref_ns") or nominal_t_g(config, target_eta))
    exc = float(abs(shift_MHz(law, float(target_eta))))
    law.update({"railed": bool(railed.any()), "n_railed": int(railed.sum()),
                "target_eta": float(target_eta),
                "measured_eta_max": float(np.nanmax(eta[np.isfinite(ridge)])),
                "excursion_MHz": exc,
                "excursion_frac_linewidth": exc / (1e3 / (2.0 * t_g)),
                # the ridge the law was built from, for drawing it over the chevrons
                "ridge_eta": [float(v) for v in eta],
                "ridge_MHz": [float(v) for v in ridge],
                "ridge_status": list(status),
                "n_tracked": int(sum(1 for v in status if v == "tracked"))})
    return law
