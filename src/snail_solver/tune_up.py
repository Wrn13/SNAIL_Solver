"""
tune_up.py
==========

Hardware-style gate tune-up: Rabi -> chirp -> fix the amplitude -> fit the length.

This mirrors the calibration procedure used for subharmonic single-transmon gates,
rather than the (offset, amp_scale) grid in ``calibration_map``:

1. **Rabi**: measure the resonant drive frequency AS A FUNCTION of drive strength.
2. **Chirp**: build delta(t) tracking that shift through the pulse.
3. **Fix the amplitude** at a chosen peak |eta*|.
4. **Length is then the only free parameter**, fitted from a time-Rabi.
5. **DRAG** shifts the detuning, so calibrate it separately -- and iterate, because
   the chirp and DRAG are mutually coupled (see `run_tune_up`).

Two things about step 1 that are easy to get wrong
----------------------------------------------------
**It has to be a chevron, not a calibration-map slice.** At fixed gate length,
raising the amplitude over-rotates the gate, and an over-rotated pulse transfers most
population slightly OFF resonance (detuning is what removes the excess rotation). So
a fixed-length map's per-row argmax tracks rotation-angle error, not the Stark shift
-- on evan_device this cost peak transfer 0.76 -> 0.23 across the amplitude window, a
railed ridge, and r2 = 0.58, even with the exact solver. A chevron avoids this because
it scans TIME too: full contrast is reached on resonance at any amplitude. Its window
must scale as 1/|eta| so every row gets the same number of swaps, not nanoseconds.

**Not all of the measured offset is a Stark shift.** The ridge splits into a static
part that survives at zero drive (from the static Hamiltonian -- a pure carrier
retune) and a drive-dependent part. Only the drive-dependent part has a shape along
the pulse, so only it belongs in the chirp (:func:`fit_shift_curve`). On evan_device
the static part is the bulk of the signal.

Why fix the amplitude (the reason this ordering exists)
---------------------------------------------------------
At fixed peak |eta| the envelope in normalized gate time u = 2t/t_g - 1 is
``|eta(u)| = eta* cos^2(pi u / 2)`` -- **independent of t_g**. So the Stark shift
delta(u) and the chirp coefficients are t_g-independent too, which decouples the
frequency calibration from the length calibration and makes "length is the only
remaining free parameter" literally true.

The alternative ordering -- fixed t_g, scan amp_scale -- doesn't have this property:
changing amp_scale changes |eta|, which changes the Stark shift, which changes the
required offset. That's the feedback ``calibration_map``'s docstring blames for
making alternating 1-D scans rail.

The amplitude/length algebra
----------------------------
``set_pump(normalize_iswap=...)`` holds the pulse AREA at A = (pi/2)/(6 g3 la lb),
independent of t_g, so a Hann pulse has ``peak_eta = 2A/t_g``. Hence::

    amp_scale(t_g) = eta* t_g / (2A)      # holds |eta| fixed as t_g varies
    t_g0           = 2A / eta*            # = device_utils.auto_t_g; amp_scale == 1 here

Rotation angle goes as area = amp x t_g, so at fixed |eta| it's proportional to t_g,
and a full iSWAP sits near t_g0. The fitted length is the empirical correction to
``auto_t_g``, just as ``amp_scale`` is the empirical correction to the analytic
amplitude in the other ordering.

Watch the direction: a LONGER t_g needs a LARGER amp_scale (the normalizer shrinks
amplitude as 1/t_g to hold the area, so holding the peak means scaling back up).
Inverting this still produces a smooth curve with a maximum, so it's unit-tested.

CLI
---
    python -m snail_solver.tune_up --device evan_device.json --target-eta 1.8 \
        --save-point tuneup
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

from snail_solver.device_utils import auto_t_g, target_eta_area

TWO_PI = 2.0 * np.pi


class RabiFitError(ValueError):
    """A Rabi sweep that measured fine but cannot be turned into a chirp.

    Carries the partial table on ``.table`` so the caller can still plot and save
    what was measured. Every one of these errors tells the user to inspect the
    chevrons; without the data attached there would be nothing to inspect.
    """

    def __init__(self, message: str, table: Dict[str, Any]):
        super().__init__(message)
        self.table = table


# ===========================================================================
# Fixed-amplitude algebra
# ===========================================================================
def _area(config: Dict[str, Any]) -> float:
    """A = (pi/2)/(6 g3 lam_a lam_b) in ns -- the t_g-INDEPENDENT full-iSWAP area."""
    return target_eta_area(float(config["g3_GHz"]), float(config["lam_a"]),
                           float(config["lam_b"]))


def fixed_eta_amp_scale(config: Dict[str, Any], t_g: float,
                        target_eta: float) -> float:
    """``amp_scale`` that holds the physical peak |eta| at `target_eta` at this `t_g`.

    Equals exactly 1.0 at ``t_g = auto_t_g(..., target_eta)``.

    Note the direction: LONGER t_g -> LARGER amp_scale. `normalize_iswap` shrinks the
    amplitude as 1/t_g to hold the pulse area, so holding the PEAK requires scaling
    back up. Getting this backwards still yields a plausible length-Rabi curve.
    """
    return float(target_eta) * float(t_g) / (2.0 * _area(config))


def peak_eta_of(config: Dict[str, Any], t_g: float, amp_scale: float) -> float:
    """Physical peak |eta| for a normalized Hann pulse at (t_g, amp_scale).

    The inverse of :func:`fixed_eta_amp_scale`; used to convert a calibration map's
    amp_scale axis into physical drive. Unaffected by DRAG (Hann has deta/dt = 0 at
    the peak) or by a chirp (a pure phase).
    """
    return float(amp_scale) * 2.0 * _area(config) / float(t_g)


def nominal_t_g(config: Dict[str, Any], target_eta: float) -> float:
    """t_g0 = 2A/eta*, the analytic full-iSWAP length at this drive strength."""
    return auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                    float(config["lam_b"]), float(target_eta))


# ===========================================================================
# Step 1 -- the Rabi experiment: resonance vs drive strength
# ===========================================================================
def fit_shift_curve(eta: np.ndarray, delta_MHz: np.ndarray,
                    weights: Optional[np.ndarray] = None,
                    fit_static: bool = True) -> Dict[str, Any]:
    """Fit delta(|eta|) = delta0 + k2 |eta|^2 + k4 |eta|^4 and split it in two.

    * ``delta0`` -- the offset that survives at ZERO drive. It comes from the static
      Hamiltonian (qutrit anharmonicity, coupler dressing), not the pump, so it's a
      correction to the bare resonance |w_b - w_a| and belongs entirely in
      ``wp_offset_GHz``. A chirp must not track it -- a constant chirp is just a
      retuned carrier, so putting it there would double-count.
    * ``k2``, ``k4`` -- the drive-dependent AC-Stark shift, the part that varies
      along the pulse and so the only part the chirp can correct.

    On evan_device the static part dominates (ridge near -0.7 MHz, barely moving
    with drive). Forcing the curve through the origin would charge that constant to
    the Stark terms, inflating them into a confident, wrong chirp -- caught by the
    r2 guard, since the fit is then poor. Set ``fit_static=False`` only when the
    probe is known to have no static offset.

    Even powers only: the shift depends on drive intensity, not sign, and the chirp
    needs delta down to |eta| = 0 (extrapolation), where an unconstrained interpolant
    would invent structure.

    Parameters
    ----------
    eta : ndarray
        Physical peak |eta| per row.
    delta_MHz : ndarray
        Measured resonance offset per row (MHz). NaNs are dropped.
    weights : ndarray, optional
        Per-row weight; pass the chevron contrast -- low-drive rows have the
        broadest peaks and hence the noisiest ridge.
    fit_static : bool, default True
        Include the drive-independent term.

    Returns
    -------
    dict
        ``delta0`` (MHz), ``k2``, ``k4`` (MHz per |eta|^2 / ^4), ``r2``, ``n_used``,
        ``resid_MHz``, ``stark_span_MHz`` (how much the DRIVE-DEPENDENT part moves
        across the measured window -- at the resolution limit there's no chirp to
        build).
    """
    eta = np.asarray(eta, dtype=float)
    y = np.asarray(delta_MHz, dtype=float)
    ok = np.isfinite(eta) & np.isfinite(y)
    n_min = 4 if fit_static else 3
    if ok.sum() < n_min:
        raise ValueError(f"need >= {n_min} usable rows to fit the shift curve, "
                         f"got {ok.sum()}")
    x, y = eta[ok], y[ok]
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)[ok]
    w = np.sqrt(np.clip(w, 0.0, None))

    cols = ([np.ones_like(x)] if fit_static else []) + [x ** 2, x ** 4]
    M = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(M * w[:, None], y * w, rcond=None)
    delta0 = float(coef[0]) if fit_static else 0.0
    k2, k4 = float(coef[-2]), float(coef[-1])
    resid = y - (delta0 + k2 * x ** 2 + k4 * x ** 4)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    stark = k2 * x ** 2 + k4 * x ** 4
    return {"delta0": delta0, "k2": k2, "k4": k4, "r2": r2, "n_used": int(ok.sum()),
            "resid_MHz": float(np.max(np.abs(resid))),
            "stark_span_MHz": float(np.max(stark) - np.min(stark))}


def fit_chevron_center(offsets_GHz: np.ndarray, metric: np.ndarray) -> Dict[str, Any]:
    """Resonance offset of a constant-drive chevron, by fitting its ANALYTIC envelope.

    For a two-level exchange at Rabi rate Omega and detuning d, peak transfer over
    time is the Lorentzian ``Omega^2 / (Omega^2 + d^2)``. Fitting that whole shape --
    rather than a parabola through the three points around the argmax -- is what
    makes a SUB-MHz shift measurable on a grid whose step is larger than the shift
    itself: every offset constrains the centre, so precision comes from the fit, not
    the grid.

    ``find_stark_resonance.locate_resonance`` keeps the parabolic estimator, the
    right tool for the broad, strongly-peaked chevrons it's used on; returned here
    as ``vertex_GHz`` so the two can be compared.

    Returns
    -------
    dict
        ``center_GHz``, ``hwhm_GHz``, ``depth``, ``base`` (the fitted floor),
        ``rmse``, ``vertex_GHz``, ``ok``.
    """
    from scipy.optimize import curve_fit
    from snail_solver.find_stark_resonance import locate_resonance

    x = np.asarray(offsets_GHz, dtype=float)
    y = np.asarray(metric, dtype=float)
    ok_pts = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok_pts], y[ok_pts]
    vertex = float(locate_resonance(x, y)) if x.size >= 3 else float("nan")
    if x.size < 5:
        return {"center_GHz": vertex, "hwhm_GHz": float("nan"), "depth": float("nan"),
                "base": float("nan"), "rmse": float("nan"), "vertex_GHz": vertex,
                "ok": False}

    def model(xx, amp, x0, w, c):
        return amp * w ** 2 / (w ** 2 + (xx - x0) ** 2) + c

    span = float(x.max() - x.min())
    p0 = [float(y.max() - y.min()), vertex, max(span / 6.0, 1e-6), float(y.min())]
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=40000,
                            bounds=([0.0, float(x.min()), 1e-7, -1.0],
                                    [2.0, float(x.max()), 10.0 * span, 1.0]))
        amp, x0, w, c = (float(v) for v in popt)
        rmse = float(np.sqrt(np.mean((model(x, amp, x0, w, c) - y) ** 2)))
        # a centre outside the scan, or a width comparable to it, is not a measurement
        good = bool(rmse < 0.1 * max(float(y.max() - y.min()), 1e-9)
                    and abs(w) < span and x.min() < x0 < x.max())
    except Exception:                                        # pragma: no cover
        amp, x0, w, c, rmse, good = (float("nan"),) * 5 + (False,)
    return {"center_GHz": x0 if good else vertex, "hwhm_GHz": abs(w), "depth": amp,
            "base": c, "rmse": rmse, "vertex_GHz": vertex, "ok": good}


def rabi_shift_table(config: Dict[str, Any], target_eta: float, *,
                     eta_lo: float = 0.3, eta_hi: float = 1.0,
                     amp_points: int = 9, wp_span_MHz: Optional[float] = None,
                     wp_points: int = 25, span_linewidths: float = 4.0,
                     chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                     drag_beat_GHz: Optional[float] = None, drag_n_pump: int = 1,
                     spec_abs_GHz: Optional[float] = None,
                     wp_offset_GHz: float = 0.0, r2_min: float = 0.9,
                     contrast_min: float = 0.35,
                     window_tg: float = 2.0, n_time: int = 161, jobs: int = 0,
                     solver: Optional[Dict[str, Any]] = None,
                     logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Step 1: measure the resonant pump offset as a function of drive strength.

    One CONSTANT-amplitude chevron per drive strength: at each |eta| the pump
    frequency is scanned and the population read out over TIME, and the resonance
    is the offset of maximum Rabi contrast. See the module docstring for why this
    has to be a chevron rather than a fixed-length calibration-map slice.

    The constant probe measures the INSTANTANEOUS shift delta(|eta|), exactly what
    the chirp needs -- a shaped-pulse ridge would give the pulse average instead,
    which would need deconvolving. It's also blind to DRAG (d eta/dt = 0), so
    `drag_beat_GHz` here only reaches the shaped cross-check in
    `calibrate_drag_offset`.

    Every point is an exact ``sesolve`` trajectory covering all times, so the whole
    table costs ``amp_points * wp_points`` solves with no reduced-model
    approximation to validate.

    The offset span and the time window are both ADAPTIVE per row: a chevron's
    linewidth is the exchange rate, ``Omega ~ 1 / (2 t_g(|eta|))``, which grows with
    drive. A span sized for the weakest row would leave the strongest row's wings
    unsampled; a window fixed in ns would give each row a different number of swaps.
    Each row starts at ``span_linewidths`` estimated linewidths and re-measures wider
    if the fitted width comes back too large for its own window (measured widths ran
    ~2-4x above the leading-order estimate).

    Parameters
    ----------
    eta_lo, eta_hi : float
        Amplitude window as a FRACTION of `target_eta`. `eta_hi` defaults to 1.0
        because the pulse never exceeds its own peak, and a constant probe held
        above the operating drive is exactly where the chevron stops being a
        two-level feature.
    wp_span_MHz : float, optional
        Fixed offset span for every row. Leave as None to size each row from its
        own linewidth, which is almost always what you want.
    span_linewidths : float
        Half-span in estimated linewidths when `wp_span_MHz` is None.
    window_tg : float
        Chevron time window in SWAPS, not ns: each row runs for
        ``window_tg * nominal_t_g(|eta|)``, so every drive strength gets the same
        number of exchange periods. ~2 gives a clean contrast peak without holding
        a strong drive long enough to leak.
    r2_min : float
        Refuse to return a curve whose fit is worse than this -- a railed or
        artefact-tracking ridge yields a plausible-looking wrong chirp.
    contrast_min : float
        Drop rows whose chevron contrast falls below this. At strong drive leakage
        can outpace the exchange, and the surviving feature is not a resonance.

    Returns
    -------
    dict
        ``eta``, ``delta_MHz`` (the ridge), ``fit`` (from :func:`fit_shift_curve`),
        ``t_g_ref_ns``, ``target_eta``, ``contrast``, ``chevrons``.
    """
    from snail_solver import find_stark_resonance as FSR

    t_g0 = nominal_t_g(config, target_eta)
    eta = np.linspace(float(eta_lo), float(eta_hi), int(amp_points)) * float(target_eta)

    def linewidth_MHz(e: float) -> float:
        """Leading-order chevron HWHM: a full swap in T means Omega = 1/(2T)."""
        return 1e3 / (2.0 * nominal_t_g(config, float(e)))

    ridge = np.full(eta.size, np.nan)
    contrast = np.full(eta.size, np.nan)
    windows = np.full(eta.size, np.nan)
    spans = np.full(eta.size, np.nan)
    chevrons = []
    for i, e in enumerate(eta):
        # Window scales with drive: fixed in ns, the weakest row would never complete
        # a swap and the strongest would sit in leakage. window_tg swaps per row.
        window_ns = float(window_tg) * nominal_t_g(config, float(e))
        span = (float(wp_span_MHz) if wp_span_MHz is not None
                else 2.0 * float(span_linewidths) * linewidth_MHz(e))

        for attempt in range(3):
            offsets_GHz = (np.linspace(-span / 2.0, span / 2.0, int(wp_points)) * 1e-3
                           + float(wp_offset_GHz))
            chev = FSR.scan(config, t_g0, 1.0, offsets_GHz, window_ns, int(n_time),
                            solver=solver, n_jobs=jobs, spec_abs_GHz=spec_abs_GHz,
                            shape="constant", chirp_coeffs_GHz=None,
                            drag_n_pump=drag_n_pump, eta_op=float(e))
            m = np.asarray(chev["resonance_metric"], dtype=float)
            cen = fit_chevron_center(chev["offsets_GHz"], m)
            # Widen only when the width itself is unconstrained (wings never sampled).
            # A bad-rmse rejection instead means leakage broke the two-level chevron
            # -- the contrast floor below catches that -- so widening won't help.
            if (cen["hwhm_GHz"] * 1e3 <= 0.4 * span or wp_span_MHz is not None
                    or attempt == 2):
                break
            span *= 3.0
            if logger:
                logger.info(f"    row {i + 1}: hwhm {cen['hwhm_GHz'] * 1e3:.2f} MHz too "
                            f"wide for +/-{span / 6:.1f} MHz -- retrying at "
                            f"+/-{span / 2:.1f} MHz")

        windows[i] = window_ns
        spans[i] = span
        contrast[i] = float(np.nanmax(m) - np.nanmin(m))
        # No contrast means no resonance: at strong drive leakage can outpace the
        # exchange, and one such row would otherwise set k2, k4 for the whole chirp.
        # fit_shift_curve ignores NaNs, so dropping it here is enough.
        if contrast[i] < contrast_min:
            ridge[i] = np.nan
            if logger:
                logger.info(f"  rabi row {i + 1}/{eta.size}: |eta|={e:.4f} DROPPED -- "
                            f"contrast {contrast[i]:.3f} < {contrast_min}; the swap is "
                            f"leaking, so this chevron does not locate a resonance")
            chevrons.append({"eta": float(e), "offsets_GHz": chev["offsets_GHz"],
                             "metric": m, "fit": cen, "window_ns": window_ns,
                             "span_MHz": span, "dropped": "low_contrast",
                             "times_ns": chev["times_ns"], "P10": chev["P10"]})
            continue
        # report the ridge RELATIVE to the offset the probe already carries, so the
        # caller accumulates a residual rather than re-adding the current setting
        ridge[i] = (cen["center_GHz"] - float(wp_offset_GHz)) * 1e3
        chevrons.append({"eta": float(e), "offsets_GHz": chev["offsets_GHz"],
                         "metric": m, "fit": cen, "window_ns": window_ns,
                         "span_MHz": span, "times_ns": chev["times_ns"],
                         "P10": chev["P10"]})
        if logger:
            logger.info(f"  rabi row {i + 1}/{eta.size}: |eta|={e:.4f} -> "
                        f"{ridge[i]:+.4f} MHz (span +/-{span / 2:.1f} MHz, contrast "
                        f"{contrast[i]:.3f}, hwhm {cen['hwhm_GHz'] * 1e3:.2f} MHz, "
                        f"{'lorentzian' if cen['ok'] else 'PARABOLIC FALLBACK'}, "
                        f"vertex {(cen['vertex_GHz'] - wp_offset_GHz) * 1e3:+.4f} MHz)")

    partial = {"eta": eta, "delta_MHz": ridge, "t_g_ref_ns": t_g0,
               "target_eta": float(target_eta), "contrast": contrast,
               "windows_ns": windows, "spans_MHz": spans, "chevrons": chevrons}

    # a ridge sitting on the scan edge is not a measurement
    step = spans / max(int(wp_points) - 1, 1)
    railed = np.isfinite(ridge) & (np.abs(np.abs(ridge) - spans / 2.0) <= step)
    if railed.any():
        raise RabiFitError(
            f"{int(railed.sum())}/{ridge.size} ridge rows rail against their scan "
            f"window (spans {np.nanmin(spans) / 2:.1f}-{np.nanmax(spans) / 2:.1f} MHz "
            f"half-width) -- raise --span-linewidths. A railed ridge produces a "
            f"confident, wrong chirp.", partial)

    fit = fit_shift_curve(eta, ridge, weights=contrast)
    partial["fit"] = fit
    if not (fit["r2"] >= r2_min):
        raise RabiFitError(
            f"the ridge is not well described by delta0 + k2|eta|^2 + k4|eta|^4 "
            f"(r2 = {fit['r2']:.3f} < {r2_min}, residual {fit['resid_MHz']:.4f} MHz). "
            f"The ridge may be tracking leakage rather than the Stark shift, or the "
            f"offset grid may be too coarse to resolve it. Inspect the chevrons "
            f"before trusting a chirp built from them.", partial)
    # A drive-dependent span below the fit's own scatter is not a measured shift, and
    # a chirp built from it would be fitted noise dressed as physics. The STATIC part
    # is still trustworthy -- it is the bulk of the signal -- so this is a chirp
    # problem, not an offset problem.
    if fit["stark_span_MHz"] <= fit["resid_MHz"]:
        raise RabiFitError(
            f"the DRIVE-DEPENDENT shift ({fit['stark_span_MHz']:.4f} MHz across "
            f"|eta| in [{eta[0]:.2f}, {eta[-1]:.2f}]) is smaller than the fit residual "
            f"({fit['resid_MHz']:.4f} MHz), so there is no resolved Stark shift to "
            f"build a chirp from. The static offset delta0 = {fit['delta0']:+.4f} MHz "
            f"is still meaningful -- calibrate wp_offset and run without a chirp, or "
            f"widen --eta-lo/--eta-hi and refine --wp-points until the drive "
            f"dependence clears the noise.", partial)
    if logger:
        logger.info(f"  rabi: delta0={fit['delta0']:+.4f} MHz (static), "
                    f"k2={fit['k2']:+.4f} MHz/|eta|^2, k4={fit['k4']:+.4f} "
                    f"MHz/|eta|^4, r2={fit['r2']:.4f} over {fit['n_used']} rows; "
                    f"drive-dependent span {fit['stark_span_MHz']:.4f} MHz vs "
                    f"residual {fit['resid_MHz']:.4f} MHz")
    return {"eta": eta, "delta_MHz": ridge, "fit": fit, "t_g_ref_ns": t_g0,
            "target_eta": float(target_eta), "contrast": contrast,
            "windows_ns": windows, "spans_MHz": spans, "chevrons": chevrons}


# ===========================================================================
# Step 2 -- chirp by projecting the MEASURED curve
# ===========================================================================
def chirp_from_measured_shift(table: Dict[str, Any], target_eta: Optional[float] = None,
                              degree: int = 4, pin_c0: bool = True,
                              drag_beat_GHz: Optional[float] = None,
                              drag_n_pump: int = 1, t_g: Optional[float] = None,
                              max_iters: int = 12,
                              tol_GHz: float = 1e-12) -> Dict[str, Any]:
    """Step 2: project the measured shift onto the Legendre chirp basis.

    Evaluates the fitted shift along the pulse -- ``|eta(u)| = eta* cos^2(pi u / 2)``
    for a Hann envelope -- and projects delta(u) onto Legendre polynomials by
    Gauss-Legendre quadrature. Exact, not approximate: delta is a degree-8
    polynomial in cos(pi u / 2), so a modest node count integrates it to machine
    precision. Pointwise evaluation needs no deconvolution because the measured
    curve is the INSTANTANEOUS shift -- the whole reason :func:`rabi_shift_table`
    uses a constant probe.

    Generalizes ``stark_chirp.stark_chirp_seed``, which assumes shift proportional
    to |eta|^2 and so only reproduces the tabulated cos^4 shape; with a measured k4
    the cos^8 content is captured too (``rel_diff`` reports how much that's worth).

    DRAG is a fixed point, not a formula
    -------------------------------------
    DRAG adds a quadrature ``q(t) = (d eta/dt) / Delta(t)``, so the drive the qubit
    sees is ``|eta_tot|^2 = |eta|^2 + q^2`` and the Stark shift follows the TOTAL
    drive. But ``Delta(t) = Delta_0 - k delta(t)`` moves with the chirp being
    computed, so each depends on the other -- solved by iterating to a fixed point.

    * The quadrature is measured, not assumed: k2, k4 come from the constant-probe
      sweep, evaluated here at the larger total amplitude. The probe's blindness to
      DRAG (d eta/dt = 0) isn't a gap -- it measures delta vs drive, and DRAG only
      adds drive.
    * ``q`` scales as 1/t_g, so with DRAG on the chirp is no longer length-
      independent. The length/chirp decoupling is a DRAG-off statement; with DRAG
      on, the two are re-solved together by ``run_tune_up``'s outer loop.

    Returns
    -------
    dict
        ``coeffs_GHz`` (length degree+1, c_0 = 0 when pinned), ``mean_shift_GHz``,
        ``rel_diff`` vs the pure-|eta|^2 analytic seed, ``quartic_fraction``,
        ``measured_eta_max`` and ``extrapolation_ratio`` (target_eta over the
        largest |eta| the Rabi sweep actually measured -- 1 means pure
        interpolation; well above 1 means the chirp's peak is built from a
        polynomial extrapolated past where it was fitted), and with DRAG on
        ``drag_iters``, ``drag_delta_frac`` (how much of the shift the quadrature
        adds) and ``min_abs_detuning_GHz``.
    """
    from numpy.polynomial import legendre as L
    from snail_solver import stark_chirp as SC

    fit = table["fit"]
    eta_star = float(target_eta if target_eta is not None else table["target_eta"])
    k2, k4 = float(fit["k2"]), float(fit["k4"])
    # eta_star can exceed the eta the Rabi sweep actually measured (see eta_hi in
    # rabi_shift_table): the chevron diagnostic must stay in the weakly-nonlinear
    # regime, but the chirp built from its fit is free to be evaluated at a higher
    # drive. Past the measured window that's an EXTRAPOLATION of a low-order
    # polynomial into a regime the fit never saw -- worth flagging, not silently
    # trusting, since it's exactly where the perturbative eta^2+eta^4 model is also
    # least likely to hold.
    measured_eta = table.get("eta")
    measured_eta_max = (float(np.nanmax(np.asarray(measured_eta, dtype=float)))
                        if measured_eta is not None else float("nan"))
    extrapolation_ratio = (eta_star / measured_eta_max
                           if measured_eta_max > 0 else float("nan"))

    n_quad = max(2 * int(degree) + 8, 32)
    u, w = np.polynomial.legendre.leggauss(n_quad)
    s = np.cos(np.pi * u / 2.0) ** 2                       # |eta(u)| / eta*
    amp = eta_star * s

    def shift_GHz(a):                                      # MHz law -> GHz
        return (k2 * a ** 2 + k4 * a ** 4) * 1e-3

    delta_GHz = shift_GHz(amp)
    base_norm = float(np.linalg.norm(delta_GHz))
    extra: Dict[str, Any] = {}

    if drag_beat_GHz:
        if t_g is None:
            raise ValueError("chirp_from_measured_shift needs t_g when DRAG is on: "
                             "the quadrature is (d eta/dt)/Delta(t) and so scales as "
                             "1/t_g, which breaks the length-independence of the chirp")
        # Hann: eta(t) = eta* cos^2(pi u / 2) with u = 2t/t_g - 1
        #   -> d eta/dt = -eta* (pi / t_g) sin(pi u)                       [1/ns]
        deta_dt = -eta_star * (np.pi / float(t_g)) * np.sin(np.pi * u)
        detuning0 = float(drag_beat_GHz) * TWO_PI                          # rad/ns
        k_pump = int(drag_n_pump)
        min_abs = float("inf")
        for it in range(int(max_iters)):
            detuning = detuning0 - k_pump * (delta_GHz * TWO_PI)   # Delta(t) = D0 - k d(t)
            min_abs = float(np.min(np.abs(detuning)) / TWO_PI)
            q = deta_dt / detuning
            new = shift_GHz(np.sqrt(amp ** 2 + q ** 2))
            step = float(np.max(np.abs(new - delta_GHz)))
            delta_GHz = new
            if step < tol_GHz:
                break
        else:
            raise RuntimeError(
                f"the chirp<->DRAG fixed point did not settle in {max_iters} passes "
                f"(last step {step:.2e} GHz, min|Delta(t)| = {min_abs * 1e3:.3f} MHz). "
                f"Near a collision the quadrature and the chirp can chase each other; "
                f"pick a further-detuned beat or a weaker drive.")
        drag_norm = float(np.linalg.norm(delta_GHz))
        extra = {"drag_iters": it + 1, "min_abs_detuning_GHz": min_abs,
                 "drag_delta_frac": ((drag_norm - base_norm) / base_norm
                                     if base_norm else float("nan"))}

    coeffs = np.array([(2 * k + 1) / 2.0 * np.sum(w * delta_GHz
                                                  * L.legval(u, np.eye(k + 1)[k]))
                       for k in range(int(degree) + 1)])
    coeffs[1::2] = 0.0                       # odd terms vanish by parity; keep it exact
    stark_mean_GHz = float(coeffs[0])
    # The static part of the measured ridge is drive-INDEPENDENT, so it has no shape
    # for a chirp to track -- it is purely a carrier retune, and it is added to the
    # offset only. Folding it into the chirp instead would be exactly the c_0
    # double-counting that pinning c_0 exists to prevent.
    mean_shift_GHz = stark_mean_GHz + float(fit.get("delta0", 0.0)) * 1e-3
    if pin_c0:
        coeffs[0] = 0.0

    analytic = SC.stark_chirp_seed(stark_mean_GHz, degree=degree, pin_c0=pin_c0)
    denom = float(np.linalg.norm(analytic))
    rel_diff = float(np.linalg.norm(coeffs - analytic) / denom) if denom else float("nan")
    quartic = (abs(k4 * eta_star ** 4) / abs(k2 * eta_star ** 2)
               if k2 else float("inf"))
    return {"coeffs_GHz": coeffs, "mean_shift_GHz": mean_shift_GHz,
            "stark_mean_GHz": stark_mean_GHz,
            "static_GHz": float(fit.get("delta0", 0.0)) * 1e-3,
            "rel_diff": rel_diff, "quartic_fraction": float(quartic), **extra,
            "degree": int(degree), "target_eta": eta_star,
            "measured_eta_max": measured_eta_max,
            "extrapolation_ratio": extrapolation_ratio}


# ===========================================================================
# Seeing what was fitted
# ===========================================================================
#: Categorical slots 1 and 2 of the reference data-viz palette, which is validated
#: for CVD separation and contrast. Fixed by ROLE, never by rank: the Lorentzian is
#: always blue and the parabolic vertex always orange, in every panel, so a reader
#: who learns the pairing once keeps it.
_C_LORENTZ = "#2a78d6"
_C_VERTEX = "#eb6834"
_C_INK = "#52514e"


def plot_rabi_table(table: Dict[str, Any], out: str = "figs/rabi_chevrons.png",
                    title: Optional[str] = None) -> str:
    """Render the Rabi sweep: every chevron, its envelope fit, and the shift curve.

    One row per drive strength, left to right: the raw chevron; the Rabi oscillation
    on resonance and one linewidth off it; and the max-over-time envelope with its
    fitted Lorentzian. The bottom panel is the result: located resonance vs |eta|,
    with the ``delta0 + k2|eta|^2 + k4|eta|^4`` curve through it.

    The middle column is the measurement in its rawest form -- detuning speeds the
    oscillation up and shrinks it (``Omega_eff = sqrt(Omega^2 + d^2)``, peak
    ``Omega^2/Omega_eff^2``), and the right column fits the peak of that envelope
    over all offsets. If the on-resonance trace doesn't reach 1 and come back, the
    chevron has no well-defined centre no matter how good the Lorentzian looks.

    The Lorentzian centre and parabolic vertex are drawn together on every envelope
    because their disagreement is the diagnostic: on a clean two-level chevron they
    coincide, and when they separate by more than the shift being measured, the
    lineshape isn't ``Omega^2/(Omega^2 + delta^2)`` and no fit recovers a resonance
    -- which is exactly what happens at strong drive on these devices.

    Accepts a full table or the partial one carried by :class:`RabiFitError`, so a
    sweep that failed its guards can still be inspected.

    Returns
    -------
    str
        The path written.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:                                        # style is a nicety
        pass

    chevrons = list(table.get("chevrons", []))
    if not chevrons:
        raise ValueError("no chevrons to plot")
    eta = np.asarray(table["eta"], dtype=float)
    ridge = np.asarray(table["delta_MHz"], dtype=float)
    fit = table.get("fit")
    n = len(chevrons)

    fig = plt.figure(figsize=(16.2, 2.9 * n + 3.4), layout="constrained")
    gs = fig.add_gridspec(n + 1, 3, width_ratios=[1.0, 0.95, 1.1],
                          height_ratios=[2.9] * n + [3.4])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)]
                     for r in range(n)])
    for i, ch in enumerate(chevrons):
        ax0, axt, ax1 = axes[i, 0], axes[i, 1], axes[i, 2]
        off = np.asarray(ch["offsets_GHz"], dtype=float) * 1e3
        m = np.asarray(ch["metric"], dtype=float)
        cen, vtx = ch["fit"]["center_GHz"] * 1e3, ch["fit"]["vertex_GHz"] * 1e3
        dropped = ch.get("dropped")

        if "P10" in ch:
            # sequential magnitude -> one perceptually uniform ramp, pinned to [0, 1]
            # so every row is directly comparable to every other
            mesh = ax0.pcolormesh(off, np.asarray(ch["times_ns"], dtype=float),
                                  np.asarray(ch["P10"], dtype=float).T,
                                  shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            fig.colorbar(mesh, ax=ax0, label=r"$P(|10\rangle)$", pad=0.02)
        ax0.axvline(cen, color=_C_LORENTZ, ls="--", lw=1.6)
        ax0.set_ylabel("time (ns)")
        ax0.set_title(rf"$|\eta|$ = {ch['eta']:.3f}   "
                      rf"({ch['window_ns']:.0f} ns window)", fontsize=10)

        # -- the oscillation itself, on resonance and one linewidth away ---------
        if "P10" in ch:
            P = np.asarray(ch["P10"], dtype=float)
            ts = np.asarray(ch["times_ns"], dtype=float)
            hw = ch["fit"].get("hwhm_GHz", np.nan) * 1e3
            j_on = int(np.argmin(np.abs(off - cen)))
            axt.plot(ts, P[j_on], "-", lw=2.0, color=_C_LORENTZ,
                     label=rf"on resonance ({off[j_on]:+.2f} MHz)")
            if np.isfinite(hw):
                j_off = int(np.argmin(np.abs(off - (cen + hw))))
                if j_off != j_on:
                    axt.plot(ts, P[j_off], "-", lw=1.6, color=_C_VERTEX, alpha=0.85,
                             label=rf"+1 HWHM ({off[j_off]:+.2f} MHz)")
            axt.axhline(1.0, color=_C_INK, ls=":", lw=1.0)
            axt.set_ylim(-0.03, 1.22)      # headroom so the legend clears the trace
            axt.set_ylabel(r"$P(|10\rangle)$")
            axt.set_title(rf"Rabi oscillation, peak {P[j_on].max():.3f}", fontsize=9)
            axt.legend(fontsize=7.5, framealpha=0.95, loc="upper right",
                       ncol=2, borderaxespad=0.3)
            axt.grid(alpha=0.25)

        ax1.plot(off, m, "o", ms=4.5, color=_C_INK, label="max-over-time $P(|10\\rangle)$")
        f = ch["fit"]
        if np.isfinite(f.get("hwhm_GHz", np.nan)) and f.get("ok"):
            xs = np.linspace(off.min(), off.max(), 400)
            w, d, b = f["hwhm_GHz"] * 1e3, f["depth"], f.get("base", 0.0)
            ax1.plot(xs, d * w ** 2 / (w ** 2 + (xs - cen) ** 2) + b,
                     "-", lw=2.0, color=_C_LORENTZ,
                     label=rf"Lorentzian, HWHM {w:.1f} MHz")
        ax1.axvline(cen, color=_C_LORENTZ, ls="--", lw=1.6,
                    label=f"centre {cen:+.2f} MHz")
        ax1.axvline(vtx, color=_C_VERTEX, ls=":", lw=1.8,
                    label=f"parabolic vertex {vtx:+.2f} MHz")
        gap = abs(cen - vtx)
        note = f"contrast {np.nanmax(m) - np.nanmin(m):.2f}   estimators differ {gap:.2f} MHz"
        if dropped:
            note += f"   DROPPED ({dropped})"
        ax1.set_title(note, fontsize=9,
                      color=("#b3261e" if dropped else _C_INK))
        ax1.legend(fontsize=7.5, framealpha=0.95, loc="upper left",
                   borderaxespad=0.4)
        ax1.grid(alpha=0.25)
        if i == n - 1:
            for ax in (ax0, ax1):
                ax.set_xlabel(r"pump offset from $|\omega_b-\omega_a|$ (MHz)")
            axt.set_xlabel("time (ns)")

    # -- the result: resonance vs drive, and the curve fitted through it --------
    axr = fig.add_subplot(gs[n, :])
    ok = np.isfinite(ridge)
    axr.plot(eta[ok], ridge[ok], "o", ms=7, color=_C_LORENTZ, label="located resonance")
    if (~ok).any():
        axr.plot(eta[~ok], np.zeros((~ok).sum()), "x", ms=9, color=_C_VERTEX,
                 label="dropped (low contrast)")
    if fit:
        xs = np.linspace(0.0, float(eta.max()) * 1.05, 300)
        axr.plot(xs, fit["delta0"] + fit["k2"] * xs ** 2 + fit["k4"] * xs ** 4,
                 "-", lw=2.0, color=_C_INK,
                 label=(rf"$\delta =$ {fit['delta0']:+.3f} "
                        rf"{fit['k2']:+.3f}$\,|\eta|^2$ {fit['k4']:+.3f}$\,|\eta|^4$"
                        rf"   ($r^2$ = {fit['r2']:.4f})"))
        axr.axhline(fit["delta0"], color=_C_INK, ls=":", lw=1.2)
    axr.set_xlabel(r"drive strength $|\eta|$")
    axr.set_ylabel("resonance offset (MHz)")
    axr.set_title("the shift curve the chirp is built from", fontsize=10)
    axr.legend(fontsize=8, framealpha=0.9)
    axr.grid(alpha=0.25)

    fig.suptitle(title or (rf"Rabi sweep: resonance vs drive, "
                           rf"$\eta^*$ = {table['target_eta']:.2f}"), fontsize=12)
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Step 4 -- length: the only free parameter once the amplitude is fixed
# ===========================================================================
def fit_swap_period(times_ns: np.ndarray, P: np.ndarray) -> Dict[str, Any]:
    """Fit ``P(t) = A sin^2(pi t / (2 T_swap)) + C`` and return the full-swap time.

    Seeded from the FFT peak of the mean-removed trace, which makes the fit robust to
    a bad initial guess -- a plain least squares on a sinusoid is otherwise happy to
    land on a harmonic.

    Returns
    -------
    dict
        ``T_swap_ns`` (time of the FIRST full swap), ``amplitude``, ``offset``,
        ``rmse``, and ``ok`` (False when the fit is untrustworthy).
    """
    from scipy.optimize import curve_fit

    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(P, dtype=float)
    if t.size < 8:
        raise ValueError("need at least 8 samples to fit a time-Rabi trace")

    def model(tt, A, T, C):
        return A * np.sin(np.pi * tt / (2.0 * T)) ** 2 + C

    # FFT seed: P ~ sin^2 oscillates at 1/(2 T_swap) in P-space -> peak frequency f
    # gives T_swap = 1/(2f).
    dt = float(np.mean(np.diff(t)))
    spec = np.abs(np.fft.rfft(y - y.mean()))
    freqs = np.fft.rfftfreq(y.size, dt)
    f0 = float(freqs[int(np.argmax(spec[1:])) + 1]) if y.size > 2 else 0.0
    T0 = 1.0 / (2.0 * f0) if f0 > 0 else float(t[-1]) / 2.0

    try:
        popt, _ = curve_fit(model, t, y, p0=[max(y.max() - y.min(), 1e-3), T0,
                                             float(y.min())],
                            bounds=([0.0, 1e-3, -0.2], [1.5, 10.0 * float(t[-1]), 0.5]),
                            maxfev=20000)
        A, T, C = (float(v) for v in popt)
        rmse = float(np.sqrt(np.mean((model(t, A, T, C) - y) ** 2)))
        ok = bool(rmse < 0.1 and A > 0.1 and T < float(t[-1]) * 5.0)
    except Exception:                                        # pragma: no cover
        A, T, C, rmse, ok = float("nan"), float("nan"), float("nan"), float("inf"), False
    return {"T_swap_ns": T, "amplitude": A, "offset": C, "rmse": rmse, "ok": ok,
            "T_seed_ns": T0}


def time_rabi(config: Dict[str, Any], eta_op: float, *, wp_offset_GHz: float = 0.0,
              window_ns: Optional[float] = None, n_time: int = 400,
              spec_abs_GHz: Optional[float] = None,
              solver: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Hardware-style time-Rabi: a CONSTANT drive, population read out vs time.

    One ``evolve_trajectory`` call gives the whole trace, so this costs a single
    solve regardless of ``n_time``. It is what the lab does with a square pulse, and
    it provides an independent seed for the shaped length scan -- but
    :func:`length_rabi` is what actually sets ``t_g_ns``, because only that evaluates
    the real Hann-shaped gate.

    A constant pump has ``deta/dt = 0``, so this probe is blind to DRAG (and cannot
    carry a chirp, which is defined on the gate's normalized time). It measures the
    bare exchange rate at this drive.
    """
    from snail_solver.find_stark_resonance import build_chevron_coupler

    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    if window_ns is None:
        window_ns = 3.0 * nominal_t_g(config, eta_op)
    times = np.linspace(0.0, float(window_ns), int(n_time))
    cpl, w_p = build_chevron_coupler(config, float(eta_op), float(wp_offset_GHz),
                                     float(window_ns), spec_abs_GHz=spec_abs_GHz,
                                     shape="constant")
    init = [0] * cpl.n_modes; init[1] = 1                     # |01...>
    tgt = [0] * cpl.n_modes; tgt[0] = 1                       # |10...>
    states = cpl.evolve_trajectory(init, times, **solver)
    P = np.abs(states[:, cpl.fock_index(tgt)]) ** 2
    fit = fit_swap_period(times, P)
    return {"times_ns": times, "P10": P, "fit": fit, "eta_op": float(eta_op),
            "w_p_GHz": float(w_p), "window_ns": float(window_ns)}


def length_rabi(config: Dict[str, Any], target_eta: float,
                t_g_grid: Optional[Sequence[float]] = None, *,
                wp_offset_GHz: float = 0.0,
                chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                drag_beat_GHz: Optional[float] = None, drag_n_pump: int = 1,
                spec_abs_GHz: Optional[float] = None,
                solver: Optional[Dict[str, Any]] = None,
                refine: bool = True,
                logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Step 4: sweep the gate LENGTH at fixed peak |eta| and find the full swap.

    This is the authoritative length calibration. Unlike :func:`time_rabi` it
    evaluates the actual gate at each candidate length -- Hann ramp, chirp, DRAG, the
    full Hilbert space -- and reads the swap out at exactly the time the gate is used,
    so no ramp-shape conversion factor is involved.

    At each `t_g` the amplitude is re-derived by :func:`fixed_eta_amp_scale`, which is
    what holds the physical drive constant while the length varies. The grid defaults
    to +/-30% about ``t_g0``, where a full iSWAP sits by construction.
    """
    from snail_solver.device_utils import maximize_1d, transfer_probability

    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    t_g0 = nominal_t_g(config, target_eta)
    grid = (np.asarray(t_g_grid, dtype=float) if t_g_grid is not None
            else t_g0 * np.linspace(0.7, 1.3, 13))

    def score(t_g: float) -> float:
        return transfer_probability(
            config, float(t_g), fixed_eta_amp_scale(config, float(t_g), target_eta),
            wp_offset_GHz, solver, spec_abs_GHz=spec_abs_GHz,
            drag_beat_GHz=drag_beat_GHz, chirp_coeffs_GHz=chirp_coeffs_GHz,
            drag_n_pump=drag_n_pump)

    P = np.array([score(t) for t in grid])
    k = int(np.argmax(P))
    if logger:
        logger.info(f"  length: coarse best t_g={grid[k]:.3f} ns  P={P[k]:.5f} "
                    f"(t_g0={t_g0:.3f} ns)")

    best_t, best_P, nfev = float(grid[k]), float(P[k]), int(grid.size)
    if refine and 0 < k < grid.size - 1:
        lo, hi = float(grid[k - 1]), float(grid[k + 1])
        best_t, best_P, n = maximize_1d(score, lo, hi, n_points=7, n_refine=2)
        nfev += int(n)
    return {"t_g_grid": grid, "P": P, "t_g_ns": best_t, "transfer": best_P,
            "t_g0_ns": t_g0, "amp_scale": fixed_eta_amp_scale(config, best_t,
                                                              target_eta),
            "nfev": nfev, "railed": bool(k in (0, grid.size - 1))}


# ===========================================================================
# Step 5 -- DRAG shifts the detuning
# ===========================================================================
def calibrate_drag_offset(config: Dict[str, Any], t_g: float, target_eta: float,
                          drag_beat_GHz: float, *, drag_n_pump: int = 1,
                          chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                          span_MHz: float = 40.0, points: int = 31,
                          time_points: int = 120,
                          spec_abs_GHz: Optional[float] = None,
                          solver: Optional[Dict[str, Any]] = None,
                          jobs: int = 0,
                          predicted_GHz: Optional[float] = None) -> Dict[str, Any]:
    """MEASURE the resonance shift DRAG introduces: shaped chevron, DRAG off vs on.

    This is the empirical counterpart to the analytic quadrature model inside
    :func:`chirp_from_measured_shift`, and the only place a DRAG-induced Stark shift
    is *observed* rather than predicted. The model assumes the shift follows the
    total drive through the SAME law the constant probe measured,
    ``|eta_tot|^2 = |eta|^2 + [(d eta/dt)/Delta(t)]^2``. That is an assumption. If
    DRAG shifts the resonance through some other route -- a channel the quadrature
    opens that the bare drive does not, or a beat-dependence the law cannot see --
    the difference shows up here and nowhere else.

    Uses the SHAPED chevron in both cases, which is essential: a constant probe has
    ``d eta/dt = 0`` and so has no DRAG quadrature at all, and it cannot carry a
    chirp, which is defined on the gate's normalized time.

    Compare ``drag_shift_GHz`` against ``predicted_drag_shift_GHz`` (pass
    `predicted_GHz` to have the gap computed here). They are pulse-averaged
    resonances, so they are directly comparable.
    """
    from snail_solver import find_stark_resonance as FS

    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    amp_scale = fixed_eta_amp_scale(config, t_g, target_eta)
    offsets = np.linspace(-span_MHz / 2e3, span_MHz / 2e3, int(points))

    def locate(beat):
        return FS.scan(config, float(t_g), amp_scale, offsets,
                       1.05 * float(t_g), int(time_points), solver, n_jobs=jobs,
                       spec_abs_GHz=spec_abs_GHz, shape="raised_cosine",
                       drag_beat_GHz=beat, chirp_coeffs_GHz=chirp_coeffs_GHz,
                       drag_n_pump=drag_n_pump)

    off = locate(None)
    on = locate(float(drag_beat_GHz))
    d0 = float(off["resonance_offset_GHz"])
    d1 = float(on["resonance_offset_GHz"])
    out = {"wp_offset_nodrag_GHz": d0, "wp_offset_drag_GHz": d1,
           "drag_shift_GHz": d1 - d0, "drag_beat_GHz": float(drag_beat_GHz),
           "drag_n_pump": int(drag_n_pump), "t_g_ns": float(t_g),
           "target_eta": float(target_eta), "span_MHz": float(span_MHz),
           "chevron_nodrag": off, "chevron_drag": on}
    if predicted_GHz is not None:
        out["predicted_drag_shift_GHz"] = float(predicted_GHz)
        out["excess_GHz"] = (d1 - d0) - float(predicted_GHz)
    return out


def drag_shift_table(config: Dict[str, Any], target_eta: float, drag_beat_GHz: float, *,
                     drag_n_pump: int = 1, eta_lo: float = 0.6, eta_hi: float = 1.2,
                     amp_points: int = 4,
                     chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                     span_MHz: float = 40.0, points: int = 25,
                     time_points: int = 120,
                     spec_abs_GHz: Optional[float] = None,
                     solver: Optional[Dict[str, Any]] = None, jobs: int = 0,
                     logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """DRAG-induced resonance shift as a function of DRIVE STRENGTH.

    The drive-resolved version of :func:`calibrate_drag_offset`: a shaped
    DRAG-off/DRAG-on chevron pair at each of several peak |eta|, so a DRAG-induced
    Stark shift can be identified by its SCALING rather than by a single number.

    Each row is a full-swap gate at its own drive, ``t_g = nominal_t_g(|eta|)`` with
    ``amp_scale = 1``, which is what keeps the rotation angle fixed while the drive
    varies (the same reason the Rabi step uses a chevron at all).

    The expected scaling is a strong prediction, and that is what makes this a test.
    For a Hann pulse ``d eta/dt ~ eta / t_g`` and ``t_g ~ 1/eta``, so the quadrature
    ``q ~ eta^2`` and a shift following the measured ``k2 |eta|^2`` law goes as
    ``q^2 ~ eta^4``. A measured exponent near 4 means DRAG is shifting the resonance
    simply by adding drive, which the chirp already accounts for. An exponent that is
    NOT 4 -- or a shift that depends on the beat at fixed |eta| -- means a mechanism
    the analytic model does not contain, and the chirp built from the constant-probe
    law will be wrong by that much.

    Returns
    -------
    dict
        ``eta``, ``t_g_ns``, ``shift_MHz`` (DRAG-on minus DRAG-off), ``exponent``
        (fitted power of |eta|), ``rows``.
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    eta = np.linspace(float(eta_lo), float(eta_hi), int(amp_points)) * float(target_eta)
    shift = np.full(eta.size, np.nan)
    t_gs = np.full(eta.size, np.nan)
    rows = []
    for i, e in enumerate(eta):
        t_g = nominal_t_g(config, float(e))
        t_gs[i] = t_g
        row = calibrate_drag_offset(
            config, t_g, float(e), float(drag_beat_GHz), drag_n_pump=drag_n_pump,
            chirp_coeffs_GHz=chirp_coeffs_GHz, span_MHz=span_MHz, points=points,
            time_points=time_points, spec_abs_GHz=spec_abs_GHz, solver=solver,
            jobs=jobs)
        shift[i] = row["drag_shift_GHz"] * 1e3
        rows.append(row)
        if logger:
            logger.info(f"  drag row {i + 1}/{eta.size}: |eta|={e:.4f} "
                        f"(t_g={t_g:.2f} ns) -> DRAG shifts the resonance by "
                        f"{shift[i]:+.4f} MHz")

    # power-law exponent, fitted in logs on the rows that carry a resolvable shift
    ok = np.isfinite(shift) & (np.abs(shift) > 1e-6)
    exponent = float("nan")
    # a sign change across the window is not a power law, so do not report one
    if ok.sum() >= 2 and (np.all(shift[ok] > 0) or np.all(shift[ok] < 0)):
        exponent = float(np.polyfit(np.log(eta[ok]),
                                    np.log(np.abs(shift[ok])), 1)[0])
    if logger:
        logger.info(f"  drag shift scales as |eta|^{exponent:.2f} "
                    f"(4 = 'DRAG only adds drive', anything else is a mechanism the "
                    f"quadrature model does not contain)")
    return {"eta": eta, "t_g_ns": t_gs, "shift_MHz": shift, "exponent": exponent,
            "drag_beat_GHz": float(drag_beat_GHz), "drag_n_pump": int(drag_n_pump),
            "rows": rows}


def project_nodrag_mean(table: Dict[str, Any], target_eta: float,
                        degree: int = 4) -> float:
    """Pulse-averaged shift with the DRAG quadrature switched off (GHz).

    Subtracting this from the DRAG-on mean isolates what the quadrature MODEL
    predicts DRAG contributes, which is the number the shaped chevron measurement is
    checked against.
    """
    return float(chirp_from_measured_shift(table, target_eta,
                                           degree=degree)["mean_shift_GHz"])


# ===========================================================================
# Orchestrator
# ===========================================================================
def run_tune_up(config: Dict[str, Any], target_eta: float, *,
                drag_beat_GHz: Optional[float] = None, drag_n_pump: int = 1,
                spec_abs_GHz: Optional[float] = None, chirp_degree: int = 4,
                eta_lo: float = 0.3, eta_hi: float = 1.0,
                amp_points: int = 9, wp_span_MHz: Optional[float] = None,
                wp_points: int = 25, tg_points: int = 13, max_drag_iters: int = 4,
                window_tg: float = 2.0, n_time: int = 161,
                span_linewidths: float = 4.0, drag_shift_points: int = 0,
                contrast_min: float = 0.35,
                chirp_tol_GHz: float = 1e-4, offset_tol_MHz: float = 0.2,
                do_time_rabi: bool = True, jobs: int = 0,
                solver: Optional[Dict[str, Any]] = None,
                logger: Optional[logging.Logger] = None,
                **map_kw) -> Dict[str, Any]:
    """Run the full tune-up and return an operating-point record.

    Order, and what is held fixed at each step::

        0.  t_g0 = auto_t_g(eta*);  amp_scale := fixed_eta_amp_scale(t_g)  [never free]
        1.  Rabi, constant probe, chirp OFF, DRAG OFF   -> the shift LAW k2, k4
        2.  chirp by projecting that law along the pulse -> c_k, wp_offset
        3.  residual offset from a SHAPED chevron with the chirp on
        4.  length via the shaped scan                  -> t_g_ns
        --- with DRAG, iterate 2-4 to self-consistency ---

    The Rabi sweep runs ONCE: it measures a property of the device, not of the pulse,
    so the chirp, DRAG's quadrature and the length are all derived from it without
    re-measuring.

    Why steps 2-4 iterate when DRAG is on. DRAG's quadrature raises the total drive
    and hence the Stark shift; the chirp built from that shift then moves
    ``Delta(t)`` itself. That inner fixed point is solved analytically inside
    :func:`chirp_from_measured_shift`. But the quadrature also scales as ``1/t_g``,
    so with DRAG on the chirp depends on the gate length too -- the outer loop here
    closes that remaining coupling (chirp -> length -> chirp), converging in 2-3
    passes since the quadrature is a small fraction of the drive.
    """
    log = logger or logging.getLogger("tune_up")
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    t_g0 = nominal_t_g(config, target_eta)
    log.info(f"tune-up: target_eta={target_eta} -> t_g0={t_g0:.3f} ns "
             f"(amp_scale=1 there)")

    common = dict(eta_lo=eta_lo, eta_hi=eta_hi, amp_points=amp_points,
                  wp_span_MHz=wp_span_MHz, wp_points=wp_points,
                  spec_abs_GHz=spec_abs_GHz, drag_n_pump=drag_n_pump,
                  window_tg=window_tg, n_time=n_time, jobs=jobs, solver=solver,
                  span_linewidths=span_linewidths, contrast_min=contrast_min,
                  logger=log, **map_kw)

    # -- 1: the Rabi sweep, measured once ------------------------------------
    log.info("step 1: Rabi (constant-probe chevron per drive strength), DRAG OFF")
    table = rabi_shift_table(config, target_eta, **common)

    def project(t_g: float) -> Dict[str, Any]:
        """The chirp implied by the measured law at this gate length."""
        return chirp_from_measured_shift(
            table, target_eta, degree=chirp_degree, drag_beat_GHz=drag_beat_GHz,
            drag_n_pump=drag_n_pump, t_g=t_g)

    def shaped_residual(t_g: float, chirp, wp_offset: float) -> float:
        """Residual offset of the ASSEMBLED gate: shaped pulse, chirp and DRAG on.

        The chirp was derived from a constant-probe law, so this is the one place the
        real pulse gets to disagree -- and with DRAG on it is also the only direct
        measurement of the quadrature's effect on the resonance.
        """
        from snail_solver import find_stark_resonance as FSR
        # Same linewidth sizing as the Rabi rows: the shaped gate's chevron is just
        # as wide as the constant probe's at the same drive, so a span fixed in MHz
        # would be as wrong here as it was there.
        span = (float(wp_span_MHz) if wp_span_MHz is not None
                else 2.0 * float(span_linewidths) * 1e3 / (2.0 * float(t_g)))
        offs = (np.linspace(-span / 2e3, span / 2e3, int(wp_points))
                + float(wp_offset))
        chev = FSR.scan(config, float(t_g),
                        fixed_eta_amp_scale(config, float(t_g), target_eta), offs,
                        1.05 * float(t_g), int(n_time), solver, n_jobs=jobs,
                        spec_abs_GHz=spec_abs_GHz, shape="raised_cosine",
                        drag_beat_GHz=drag_beat_GHz, chirp_coeffs_GHz=list(chirp),
                        drag_n_pump=drag_n_pump)
        return float(chev["resonance_offset_GHz"]) - float(wp_offset)

    stages: Dict[str, Any] = {"rabi": table}
    history = []
    t_g = t_g0
    chirp, wp_offset, length = None, 0.0, None
    # With DRAG off nothing below depends on t_g, so one pass IS the fixed point.
    n_outer = int(max_drag_iters) if drag_beat_GHz is not None else 1

    for it in range(n_outer):
        prev_chirp = None if chirp is None else np.array(chirp)
        prev_t_g = t_g

        # -- 2: the chirp implied by the law at the current length ------------
        proj = project(t_g)
        chirp = [float(c) for c in proj["coeffs_GHz"]]
        wp_offset = float(proj["mean_shift_GHz"])
        msg = (f"  chirp {['%+.6f' % c for c in chirp]} GHz, "
               f"wp_offset={wp_offset * 1e3:+.3f} MHz "
               f"(quartic {proj['quartic_fraction']:.2f} of quadratic, "
               f"rel_diff vs pure-|eta|^2 seed {proj['rel_diff']:.3f}")
        if "drag_delta_frac" in proj:
            msg += (f"; DRAG adds {100 * proj['drag_delta_frac']:+.2f}% of the shift, "
                    f"min|Delta(t)|={proj['min_abs_detuning_GHz'] * 1e3:.2f} MHz, "
                    f"{proj['drag_iters']} inner pass(es)")
        log.info(f"step 2 (pass {it + 1}/{n_outer}) at t_g={t_g:.3f} ns:\n{msg})")
        xr = proj["extrapolation_ratio"]
        if xr > 1.15:
            log.info(f"  WARNING: target_eta={target_eta:.2f} is {xr:.2f}x the "
                     f"largest |eta|={proj['measured_eta_max']:.2f} the Rabi sweep "
                     f"measured -- the chirp near the pulse peak is a quartic "
                     f"EXTRAPOLATION past the fitted window, in the same "
                     f"majority-nonlinear regime where the diagnostic chevron "
                     f"itself stops being trustworthy. Treat it as a candidate, "
                     f"not a calibration, until checked against the gate's own "
                     f"performance (chirp_ablation below, or calibration_map).")

        # -- 3: what the assembled gate still wants ---------------------------
        residual_GHz = shaped_residual(t_g, chirp, wp_offset)
        wp_offset += residual_GHz
        log.info(f"step 3: shaped-chevron residual {residual_GHz * 1e3:+.3f} MHz "
                 f"-> wp_offset={wp_offset * 1e3:+.3f} MHz")

        # -- 4: the length, everything else frozen ----------------------------
        log.info("step 4: length scan at fixed |eta| (the only free parameter)")
        length = length_rabi(config, target_eta, wp_offset_GHz=wp_offset,
                             chirp_coeffs_GHz=chirp, drag_beat_GHz=drag_beat_GHz,
                             drag_n_pump=drag_n_pump, spec_abs_GHz=spec_abs_GHz,
                             solver=solver, logger=log,
                             t_g_grid=t_g0 * np.linspace(0.7, 1.3, int(tg_points)))
        t_g = float(length["t_g_ns"])
        if length["railed"]:
            log.info("  WARNING: the length optimum railed against the scan window")

        dc = (float("inf") if prev_chirp is None
              else float(np.max(np.abs(np.array(chirp) - prev_chirp))))
        dt = abs(t_g - prev_t_g)
        history.append({"iter": it, "t_g_ns": t_g, "max_dc_GHz": dc,
                        "d_t_g_ns": dt, "wp_offset_GHz": wp_offset,
                        "residual_GHz": residual_GHz, "chirp_GHz": list(chirp)})
        if drag_beat_GHz is None:
            break
        log.info(f"  pass {it + 1}: max|dc|={dc:.2e} GHz, |d t_g|={dt:.4f} ns")
        if dc < chirp_tol_GHz and dt < 1e-3 * t_g0:
            break
    else:
        raise RuntimeError(
            f"the chirp<->length loop did not converge in {n_outer} passes (last "
            f"max|dc|={history[-1]['max_dc_GHz']:.2e} GHz, "
            f"|d t_g|={history[-1]['d_t_g_ns']:.4f} ns). With DRAG on the quadrature "
            f"scales as 1/t_g, so the two are genuinely coupled; returning the last "
            f"iterate would be a guess. Inspect the beat and the drive.")

    stages["chirp"] = proj
    stages["length"] = length
    stages["residual_GHz"] = residual_GHz
    drag_info = ({"history": history, "iters": len(history)}
                 if drag_beat_GHz is not None else None)

    # Everything above assumes DRAG shifts the resonance only by adding drive,
    # through the law the (DRAG-blind) constant probe measured. These shaped
    # DRAG-off/DRAG-on chevrons are the direct test of that assumption.
    if drag_beat_GHz is not None:
        predicted = (float(proj["mean_shift_GHz"])
                     - float(project_nodrag_mean(table, target_eta, chirp_degree)))
        meas = calibrate_drag_offset(
            config, t_g, target_eta, float(drag_beat_GHz), drag_n_pump=drag_n_pump,
            chirp_coeffs_GHz=chirp, span_MHz=(wp_span_MHz or 40.0),
            points=int(wp_points), time_points=int(n_time),
            spec_abs_GHz=spec_abs_GHz, solver=solver, jobs=jobs,
            predicted_GHz=predicted)
        stages["drag_measured"] = meas
        log.info(f"DRAG check: measured shift {meas['drag_shift_GHz'] * 1e3:+.4f} MHz "
                 f"vs quadrature-model prediction {predicted * 1e3:+.4f} MHz "
                 f"-> excess {meas['excess_GHz'] * 1e3:+.4f} MHz")
        scale = max(abs(predicted), abs(meas["drag_shift_GHz"]))
        if scale > 0 and abs(meas["excess_GHz"]) > 0.5 * scale:
            log.info("  WARNING: the measured DRAG shift disagrees with the "
                     "quadrature model by more than 50%. DRAG is moving this "
                     "resonance by some route other than simply adding drive, so the "
                     "chirp -- which is built from the DRAG-blind constant probe -- "
                     "does not account for it. Run drag_shift_table to get the "
                     "scaling with |eta|; an exponent far from 4 identifies it.")
        if drag_shift_points:
            stages["drag_shift_table"] = drag_shift_table(
                config, target_eta, float(drag_beat_GHz), drag_n_pump=drag_n_pump,
                amp_points=int(drag_shift_points), chirp_coeffs_GHz=chirp,
                span_MHz=(wp_span_MHz or 40.0), points=int(wp_points),
                time_points=int(n_time), spec_abs_GHz=spec_abs_GHz, solver=solver,
                jobs=jobs, logger=log)
    if drag_info:
        stages["drag_loop"] = drag_info

    if do_time_rabi:
        try:
            tr = time_rabi(config, target_eta, wp_offset_GHz=wp_offset,
                           spec_abs_GHz=spec_abs_GHz, solver=solver)
            stages["time_rabi"] = tr
            if tr["fit"]["ok"]:
                # constant-drive swap time vs the shaped gate; a large disagreement
                # means the ramp-shape conversion is not the whole story
                rel = abs(2.0 * tr["fit"]["T_swap_ns"] - length["t_g_ns"]) / \
                    max(length["t_g_ns"], 1e-9)
                log.info(f"  time-Rabi T_swap={tr['fit']['T_swap_ns']:.3f} ns "
                         f"-> 2T={2 * tr['fit']['T_swap_ns']:.3f} ns vs shaped "
                         f"t_g={length['t_g_ns']:.3f} ns ({100 * rel:.1f}% apart)")
        except Exception as exc:                             # never fatal: it is a check
            log.info(f"  time-Rabi skipped: {exc}")

    t_g = float(length["t_g_ns"])
    amp_scale = fixed_eta_amp_scale(config, t_g, target_eta)

    # -- does the chirp's SHAPE actually help, over a plain retuned carrier? -----
    # wp_offset already carries the chirp's mean component, so zeroing the chirp
    # here isolates exactly what tracking the shift THROUGH the pulse buys, at the
    # same length and drive. This is the direct answer to "is the chirp effective".
    try:
        from snail_solver.device_utils import transfer_probability
        flat_transfer = transfer_probability(
            config, t_g, amp_scale, wp_offset, solver, spec_abs_GHz=spec_abs_GHz,
            drag_beat_GHz=drag_beat_GHz, chirp_coeffs_GHz=[], drag_n_pump=drag_n_pump)
        stages["chirp_ablation"] = {
            "transfer_with_chirp": float(length["transfer"]),
            "transfer_flat_carrier": float(flat_transfer),
            "leakage_with_chirp": 1.0 - float(length["transfer"]),
            "leakage_flat_carrier": 1.0 - float(flat_transfer)}
        log.info(f"chirp ablation: transfer {length['transfer']:.6f} with the chirp "
                 f"vs {flat_transfer:.6f} with a flat retuned carrier at the same "
                 f"t_g/amp_scale/wp_offset -- leakage {1 - length['transfer']:.2e} "
                 f"vs {1 - flat_transfer:.2e}")
    except Exception as exc:                                  # never fatal: it is a check
        log.info(f"  chirp ablation skipped: {exc}")

    pair = list(np.asarray(config["qubit_freqs_GHz"], dtype=float))
    record = {
        "amp_scale": amp_scale,
        "wp_offset_GHz": float(wp_offset),
        "t_g_ns": t_g,
        "chirp_coeffs_GHz": [float(c) for c in chirp],
        "wa_GHz": pair[0], "wb_GHz": pair[1],
        "spec_abs_GHz": spec_abs_GHz,
        "drag_beat_GHz": drag_beat_GHz,
        "drag_n_pump": int(drag_n_pump),
        "target_eta": float(target_eta),
        "metric": "transfer", "score": float(length["transfer"]),
        "source": "tune_up",
    }
    log.info(f"done: t_g={t_g:.3f} ns  amp_scale={record['amp_scale']:.5f}  "
             f"wp_offset={wp_offset * 1e3:+.3f} MHz  transfer={length['transfer']:.5f}")
    return {"operating_point": record, "stages": stages, "t_g0_ns": t_g0,
            "drag": drag_info}


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    """CLI entry point."""
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.tune_up", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--target-eta", type=float, required=True,
                    help="peak |eta| to FIX for the whole tune-up. This is the one "
                         "number you choose: it sets the drive and hence the nominal "
                         "length t_g0 = 2A/eta*. Larger -> shorter gate, more leakage")
    ap.add_argument("--drag-beat-GHz", type=float, default=None,
                    help="calibrate with DRAG on at this beat; enables the "
                         "chirp<->DRAG iteration")
    ap.add_argument("--drag-n-pump", type=int, default=1,
                    help="pump quanta of the suppressed process (1 one-pump, "
                         "2 subharmonic, 0 static/pump-independent)")
    ap.add_argument("--spec-abs-GHz", type=float, default=None)
    ap.add_argument("--chirp-degree", type=int, default=4,
                    help="Legendre truncation for the chirp (even terms only matter)")
    ap.add_argument("--window-tg", type=float, default=2.0,
                    help="chevron time window in units of t_g0; ~2 captures a full "
                         "exchange even for the weakest (slowest) drive row")
    ap.add_argument("--n-time", type=int, default=161,
                    help="chevron readout times (one solve covers all of them)")
    ap.add_argument("--eta-lo", type=float, default=0.3,
                    help="Rabi amplitude window, as a fraction of --target-eta")
    ap.add_argument("--eta-hi", type=float, default=1.0,
                    help="the pulse never exceeds its peak, so sampling above 1.0 is "
                         "extrapolation into where a constant probe misbehaves")
    ap.add_argument("--contrast-min", type=float, default=0.35,
                    help="drop chevrons with less contrast than this -- at strong "
                         "drive leakage can outpace the exchange and the surviving "
                         "feature is not a resonance")
    ap.add_argument("--amp-points", type=int, default=9,
                    help="drive-strength rows; each is one exact chevron")
    ap.add_argument("--wp-span-MHz", type=float, default=None,
                    help="fixed chevron offset span for EVERY row. Default: size each "
                         "row from its own linewidth (see --span-linewidths), since "
                         "the linewidth is the exchange rate and so grows with drive")
    ap.add_argument("--span-linewidths", type=float, default=4.0,
                    help="chevron half-span in estimated linewidths; rows whose "
                         "fitted width is too wide for their window are re-measured "
                         "wider automatically")
    ap.add_argument("--wp-points", type=int, default=25)
    ap.add_argument("--drag-shift-points", type=int, default=0,
                    help="with --drag-beat-GHz, also measure the DRAG-induced shift "
                         "at this many drive strengths and fit its power law. 4 = "
                         "'DRAG only adds drive' (already in the chirp); anything "
                         "else is a mechanism the quadrature model does not contain")
    ap.add_argument("--tg-points", type=int, default=13)
    ap.add_argument("--max-drag-iters", type=int, default=4)
    ap.add_argument("--skip-time-rabi", action="store_true")
    ap.add_argument("--coupler-levels", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--out", default=None, help="write the record + stages to JSON")
    ap.add_argument("--plot", nargs="?", const="figs/rabi_chevrons.png", default=None,
                    help="render every chevron, its envelope fit and the shift curve. "
                         "Written even when the run FAILS its guards -- those errors "
                         "ask you to inspect the chevrons, so they have to be visible")
    ap.add_argument("--save-point", default=None,
                    help="save the result into the device JSON under this name")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from snail_solver.device_utils import load_device
    from snail_solver.log_utils import setup_run_logger
    from snail_solver.paths import resolve_device

    logger = setup_run_logger(None, "tune_up")
    device_path = resolve_device(args.device)
    config = load_device(device_path)
    if args.coupler_levels is not None:
        config = {**config, "coupler_levels": int(args.coupler_levels)}

    try:
        out = run_tune_up(
            config, args.target_eta, drag_beat_GHz=args.drag_beat_GHz,
            drag_n_pump=args.drag_n_pump, spec_abs_GHz=args.spec_abs_GHz,
            chirp_degree=args.chirp_degree,
            window_tg=args.window_tg, n_time=args.n_time,
            span_linewidths=args.span_linewidths,
            drag_shift_points=args.drag_shift_points,
            eta_lo=args.eta_lo, eta_hi=args.eta_hi, amp_points=args.amp_points,
            contrast_min=args.contrast_min,
            wp_span_MHz=args.wp_span_MHz, wp_points=args.wp_points,
            tg_points=args.tg_points, max_drag_iters=args.max_drag_iters,
            do_time_rabi=not args.skip_time_rabi, jobs=args.jobs,
            solver={"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps},
            logger=logger)
    except RabiFitError as exc:
        # The measurement succeeded; only the interpretation failed. Save and draw it
        # before dying, so the "inspect the chevrons" instruction is actionable.
        if args.plot:
            print(f"  wrote {plot_rabi_table(exc.table, args.plot)} "
                  f"(the sweep that failed)")
        raise

    rec = out["operating_point"]
    print("\n=== tune-up result ===")
    print(f"  target_eta   = {rec['target_eta']}  (t_g0 = {out['t_g0_ns']:.3f} ns)")
    print(f"  t_g_ns       = {rec['t_g_ns']:.4f}   <- the fitted length")
    print(f"  amp_scale    = {rec['amp_scale']:.6f}  (holds |eta| at the target)")
    print(f"  wp_offset    = {rec['wp_offset_GHz'] * 1e3:+.4f} MHz")
    print(f"  chirp        = [{', '.join(f'{c:+.6f}' for c in rec['chirp_coeffs_GHz'])}] GHz")
    if rec["drag_beat_GHz"] is not None:
        print(f"  drag         = beat {rec['drag_beat_GHz'] * 1e3:.2f} MHz, "
              f"k={rec['drag_n_pump']}, converged in {out['drag']['iters']} pass(es)")
    print(f"  transfer     = {rec['score']:.6f}")
    abl = out["stages"].get("chirp_ablation")
    if abl:
        print(f"  chirp helps  = {abl['transfer_with_chirp']:.6f} (chirped) vs "
              f"{abl['transfer_flat_carrier']:.6f} (flat carrier, same t_g/amp/offset)"
              f" -- leakage {abl['leakage_with_chirp']:.2e} vs "
              f"{abl['leakage_flat_carrier']:.2e}")

    if args.out:
        from snail_solver.paths import in_results
        path = in_results(args.out)

        def _plain(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return str(o)
        with open(path, "w") as fh:
            json.dump({"operating_point": rec, "t_g0_ns": out["t_g0_ns"],
                       "stages": out["stages"]}, fh, indent=2, default=_plain)
        print(f"  written {path}")

    if args.plot:
        print(f"  wrote {plot_rabi_table(out['stages']['rabi'], args.plot)}")

    if args.save_point:
        from snail_solver.operating_points import save_point
        save_point(device_path, args.save_point, rec, overwrite=args.overwrite)
        print(f"  saved operating point {args.save_point!r} to {device_path}")


if __name__ == "__main__":
    main()
