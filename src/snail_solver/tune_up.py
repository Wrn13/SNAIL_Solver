"""
tune_up.py
==========

Hardware-style gate tune-up: Rabi -> chirp -> fix the amplitude -> fit the length.

1. **Rabi**: measure the resonant drive frequency AS A FUNCTION of drive strength.
2. **Chirp**: build delta(t) tracking that shift through the pulse.
3. **Fix the amplitude** at a chosen peak |eta*|.
4. **Length is then the only free parameter**, fitted from a time-Rabi.
5. **DRAG** shifts the detuning, so calibrate it separately and iterate, because
   the chirp and DRAG are mutually coupled (see `run_tune_up`).

Pitfalls in step 1
------------------
**It must be a chevron, not a calibration-map slice.** At fixed length, raising the
amplitude over-rotates the gate, and an over-rotated pulse transfers most population
slightly OFF resonance, so a fixed-length map's per-row argmax tracks rotation error,
not the Stark shift (on evan_device: transfer 0.76 -> 0.23, railed ridge, r2 = 0.58).
A chevron also scans TIME, so full contrast is reached on resonance at any amplitude.
Its window scales as 1/|eta| so every row gets the same number of swaps.

**Not all of the measured offset is a Stark shift.** The ridge splits into a static
part surviving at zero drive (a pure carrier retune) and a drive-dependent part; only
the latter has a shape along the pulse and belongs in the chirp (:func:`fit_shift_curve`).

Why fix the amplitude
---------------------
At fixed peak |eta| the envelope in normalized time u = 2t/t_g - 1 is
``|eta(u)| = eta* cos^2(pi u / 2)`` -- independent of t_g. So the Stark shift and the
chirp coefficients are t_g-independent too, decoupling frequency from length
calibration. Fixed t_g with scanned amp_scale lacks this: amp_scale changes |eta|,
hence the Stark shift, hence the required offset (why ``calibration_map`` 1-D scans rail).

The amplitude/length algebra
----------------------------
``set_pump(normalize_iswap=...)`` holds the pulse AREA at A = (pi/2)/(6 g3 la lb),
independent of t_g, so a Hann pulse has ``peak_eta = 2A/t_g``. Hence::

    amp_scale(t_g) = eta* t_g / (2A)      # holds |eta| fixed as t_g varies
    t_g0           = 2A / eta*            # = device_utils.auto_t_g; amp_scale == 1 here

At fixed |eta| the rotation angle is proportional to t_g, and a full iSWAP sits near
t_g0; the fitted length is the empirical correction to ``auto_t_g``. Note a LONGER t_g
needs a LARGER amp_scale; the inverted relation still gives a plausible curve, so it
is unit-tested.

CLI
---
    python -m snail_solver.tune_up --device evan_device.json --target-eta 1.8 \
        --out tuneup.h5 --save-point tuneup

``--out`` writes the operating point and every chevron's axes and traces as one HDF5
file (:mod:`h5_io`); ``--out name.json`` gives JSON. ``--replot`` reads it back with no
device and no solves (format is sniffed). A run inside a larger file (``tune_up_sweep``)
is addressed as ``FILE:/runs/eta1p8``::

    python -m snail_solver.tune_up --replot results/etasweep/eta_sweep.h5:/runs/eta1p8 \
        --plot-ridge figs/eta1p8_ridge.png

The merged device configuration the solves ran with (after ``--coupler-levels``) is
copied into the file as ``device`` (root attr ``device_path`` records its origin); it is
what ``post_chirp --from-tuneup`` reads. Every figure from ``--plot``/``--plot-ridge``/
``--plot-post-chirp`` is also embedded in the file; ``--replot`` with no ``--out``
refreshes the embedded copies. Extract them with::

    python -m snail_solver.h5_io run.h5 --extract figs/
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from snail_solver.device_utils import auto_t_g, target_eta_area

TWO_PI = 2.0 * np.pi
_DEFAULT_SOLVER = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}   # QuTiP tolerances


class RabiFitError(ValueError):
    """A Rabi sweep that measured fine but cannot be turned into a chirp.

    Carries the partial table on ``.table`` so the caller can still plot and save
    the chevrons the message tells the user to inspect.
    """

    def __init__(self, message: str, table: Dict[str, Any]):
        super().__init__(message)
        self.table = table


class DragFixedPointDiverged(RuntimeError):
    """The chirp<->DRAG fixed point ran away instead of settling.

    ``q = (d eta/dt) / Delta_j(t)`` and ``Delta_j(t) = 2 pi beat - n_pump delta(t)``,
    so the chirp being solved for sits in its own denominator: a larger shift shrinks
    |Delta_j|, which grows the quadrature, which grows the shift. Near a collision the
    two chase each other. ``min_abs_detuning_GHz`` (how close the loop drove its own
    denominator to zero) is the diagnosis.
    """

    def __init__(self, message: str, min_abs_detuning_GHz: float = float("nan")):
        super().__init__(message)
        self.min_abs_detuning_GHz = float(min_abs_detuning_GHz)


# ===========================================================================
# Fixed-amplitude algebra
# ===========================================================================
def _area(config: Dict[str, Any]) -> float:
    """A = (pi/2)/(6 g3 lam_a lam_b) in ns -- the t_g-INDEPENDENT full-iSWAP area."""
    return target_eta_area(float(config["g3_GHz"]), float(config["lam_a"]),
                           float(config["lam_b"]))


def area_factor(config: Dict[str, Any]) -> float:
    """``f = area / (amp * t_g)`` for this device's envelope shape.

    Relates the AREA ``normalize_iswap`` pins to the PEAK this module holds fixed.
    Exactly 0.5 for the raised cosine (Hann); for :class:`envelope.SinePowerRamp`
    (the Li/Calarco/Motzoi Eq. 13 ramp recursive DRAG needs) it depends on ``m`` and
    the rise fraction.
    """
    kind = config.get("envelope", "raised_cosine")
    if kind == "raised_cosine":
        return 0.5
    from snail_solver.envelope import ENVELOPE_KINDS
    cls = ENVELOPE_KINDS.get(kind)
    if cls is None or kind == "constant":
        return 1.0
    # Built at t_g = 1 with the rise as a FRACTION of the gate: an absolute t_rise
    # would make this factor (hence the Stark shift and chirp) t_g-dependent and
    # break the length/frequency decoupling.
    kw = ({"m": int(config.get("envelope_m", 3)),
           "t_rise": float(config.get("envelope_rise_frac", 0.5))}
          if cls.__name__ == "SinePowerRamp" else {})
    env = cls(amp=1.0, t_g=1.0, **kw)
    return float(env.area())              # amp = t_g = 1, so area IS the factor


def fixed_eta_amp_scale(config: Dict[str, Any], t_g: float,
                        target_eta: float) -> float:
    """``amp_scale`` that holds the physical peak |eta| at `target_eta` at this `t_g`.

    Equals 1.0 at ``t_g = auto_t_g(..., target_eta)`` for the raised cosine
    (``2 * area_factor`` in general). LONGER t_g -> LARGER
    amp_scale: `normalize_iswap` shrinks the amplitude as 1/t_g to hold the area.
    """
    return (float(target_eta) * float(t_g) * area_factor(config)) / _area(config)


def peak_eta_of(config: Dict[str, Any], t_g: float, amp_scale: float) -> float:
    """Physical peak |eta| for a normalized pulse at (t_g, amp_scale).

    The inverse of :func:`fixed_eta_amp_scale`; used to convert a calibration map's
    amp_scale axis into physical drive. Unaffected by a chirp (a pure phase).

    .. warning::
       Unaffected by FIRST-ORDER DRAG (a Hann window has ``deta/dt = 0`` at its peak),
       but not by RECURSIVE DRAG: two nested corrections contribute
       ``-eta''/(Delta_a Delta_b)`` and ``eta''`` at the peak is nonzero. That
       unmodelled term propagates into :func:`fixed_eta_amp_scale`; see
       :func:`device_utils.drag_correction_ratio` for its size.
    """
    return float(amp_scale) * _area(config) / (float(t_g) * area_factor(config))


def nominal_t_g(config: Dict[str, Any], target_eta: float) -> float:
    """t_g0 = 2A/eta*, the analytic full-iSWAP length at this drive strength."""
    return auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                    float(config["lam_b"]), float(target_eta))


def fixed_span_MHz(config: Dict[str, Any], target_eta: float, eta_hi: float = 1.0,
                   span_linewidths: float = 4.0) -> float:
    """A single ``wp_span_MHz`` wide enough for every row in a Rabi sweep.

    Linewidth is the exchange rate ``Omega ~ 1/(2 t_g(|eta|))`` and t_g shrinks with
    drive, so the STRONGEST row (``eta_hi * target_eta``) sets the span. Passed as
    ``wp_span_MHz`` it puts every row on one offset grid, which :func:`plot_chirp_ridge`
    requires (it does not interpolate).
    """
    e_max = float(eta_hi) * float(target_eta)
    linewidth_MHz = 1e3 / (2.0 * nominal_t_g(config, e_max))
    return 2.0 * float(span_linewidths) * linewidth_MHz


def ridge_span_MHz(config: Dict[str, Any], target_eta: float, *, eta_lo: float = 0.3,
                   eta_hi: float = 1.0, span_linewidths: float = 4.0,
                   wp_points: int = 25, pts_per_hwhm: float = 3.0,
                   logger=None) -> tuple:
    """The fixed ``wp_span_MHz`` :func:`plot_chirp_ridge` needs, AND the ``wp_points``
    that span demands -- returned together because one without the other is a trap.

    A span sized for the strongest row UNDERSAMPLES the weak rows, whose HWHM shrinks
    with drive::

        pts per HWHM (adaptive, any row) = (wp_points - 1) / (2 span_linewidths)
        pts per HWHM (fixed, weakest row) = that x (eta_lo / eta_hi)

    At the defaults that is 3.0 vs 0.9 (eta_lo=0.3): the Lorentzian fit returns junk,
    rows drop as ``poor_fit``, or the ridge fails the r2 floor -- a confidently WRONG
    chirp. Hence

        wp_points >= 1 + 2 span_linewidths pts_per_hwhm (eta_hi / eta_lo)

    (61 at eta_lo=0.4, 81 at 0.3). Cheap: :func:`find_stark_resonance.scan` fans out
    over offsets, while the rows are serial.

    ``pts_per_hwhm`` counts against the MODEL linewidth ``1e3 / (2 nominal_t_g)``, not
    the fitted HWHM, which runs ~1.6x broader (1Gate4.2SNAIL, target_eta=1.2: 1.73 vs
    2.88 MHz). The factor cancels in the ratio that sets ``want``; only the absolute
    number is pessimistic, in the safe direction. The 1.6 is an observation, so don't
    lower the default without re-measuring.

    Returns
    -------
    (float, int)
        ``(span_MHz, want_points)``. A warning is logged when ``wp_points`` is below
        ``want_points``; nothing is raised, because the caller may knowingly accept a
        coarser weak row.
    """
    span = fixed_span_MHz(config, target_eta, eta_hi=eta_hi,
                          span_linewidths=span_linewidths)
    want = int(np.ceil(1.0 + 2.0 * float(span_linewidths) * float(pts_per_hwhm)
                       * (float(eta_hi) / float(eta_lo))))
    if logger is not None and int(wp_points) < want:
        got = (int(wp_points) - 1) * (float(eta_lo) / float(eta_hi)) \
            / (2.0 * float(span_linewidths))
        logger.warning(
            f"  fixed span {span:.2f} MHz leaves the WEAKEST Rabi row "
            f"(|eta|={eta_lo * target_eta:.2f}) only {got:.2f} points per MODEL "
            f"linewidth at wp_points={int(wp_points)}; want >= {want} for "
            f"{pts_per_hwhm:.1f}. Fitted HWHMs run ~1.6x broader, so this is a "
            f"conservative floor, not the sampling you will read off the rows. "
            f"Expect dropped rows or a failed r2. Offsets fan out over the process "
            f"pool, so raising --wp-points to {want} is close to free.")
    return float(span), want


# ===========================================================================
# Part 0 -- does the nominal eta still mean sqrt(n_s)?
# ===========================================================================
def verify_eta_matches_ns(config: Dict[str, Any], target_eta: float, *,
                          eta_fracs: Sequence[float] = (0.3, 0.6, 0.85, 1.0, 1.2),
                          shape: str = "raised_cosine", n_time: int = 161,
                          window_tg: float = 2.0,
                          spec_abs_GHz: Optional[float] = None,
                          solver: Optional[Dict[str, Any]] = None,
                          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Check whether the nominal ``eta`` still equals the coupler mode's ``sqrt(<n_s>)``.

    Pump tones are built with ``is_eta=True``, which DECLARES the envelope amplitude
    to be ``eta``; it is meant to be ``sqrt(n_s)`` (McKinney et al., arXiv:2409.18262,
    Eq. 9), with ``g_eff = 6 g3 lam_a lam_b eta`` (their Eq. 10). This checks the
    label against the actual occupation once multi-photon channels populate the
    coupler.

    CAVEAT (unresolved): in ``ZhouCoupler.dressed_flux`` the coupler is a dynamical
    mode at ``w_s`` and ``eta_p(t)`` a SEPARATE classical term at ``w_p``, so whether
    the coupler's own Fock occupation is McKinney's ``sqrt(n_s)`` is open. Treat
    ``ratio`` as "how much the coupler is incidentally populated", not a validated
    test of a labelling bug.

    Each row evolves from bare VACUUM (growth is due to the pump alone) at fixed
    ``t_g = nominal_t_g(target_eta)``, with ``amp_scale`` from
    :func:`fixed_eta_amp_scale` -- the pulse `run_tune_up` builds, at other drives.

    `eta_fracs` are fractions of `target_eta` to probe; `shape` is the shaped gate
    or "constant" (the Rabi-sweep probe, run for ``window_tg * nominal_t_g(eta)``).

    Returns
    -------
    dict
        ``eta_nominal``, ``sqrt_ns_peak``, ``ratio = sqrt_ns_peak / eta_nominal``,
        ``target_eta``, ``shape``, and ``rows`` (per-row ``times_ns``, ``n_s_t``,
        ``eta_nominal``, ``t_g_ns``) for :func:`plot_eta_vs_ns`.
    """
    from snail_solver import find_stark_resonance as FSR

    solver = solver or dict(_DEFAULT_SOLVER)
    t_g = nominal_t_g(config, target_eta)
    eta = np.asarray(eta_fracs, dtype=float) * float(target_eta)

    eta_nominal = np.full(eta.size, np.nan)
    sqrt_ns_peak = np.full(eta.size, np.nan)
    rows = []
    for i, e in enumerate(eta):
        if shape == "raised_cosine":
            window_ns, t_g_row = 1.05 * t_g, t_g
            kw = {"shape": "raised_cosine", "t_g_ns": t_g,
                  "amp_scale": fixed_eta_amp_scale(config, t_g, float(e))}
        else:
            window_ns = t_g_row = float(window_tg) * nominal_t_g(config, float(e))
            kw = {"shape": "constant"}
        cpl, _w_p = FSR.build_chevron_coupler(
            config, float(e), 0.0, window_ns, spec_abs_GHz=spec_abs_GHz, **kw)

        times = np.linspace(0.0, window_ns, int(n_time))
        init = [0] * cpl.n_modes                              # bare vacuum
        states = cpl.evolve_trajectory(init, times, **solver)
        n_s = FSR.coupler_number_trace(cpl, states)

        eta_nominal[i] = float(e)
        sqrt_ns_peak[i] = float(np.sqrt(max(np.max(n_s), 0.0)))
        rows.append({"eta_nominal": float(e), "times_ns": times, "n_s_t": n_s,
                    "t_g_ns": float(t_g_row)})
        if logger:
            logger.info(f"  verify_eta_matches_ns: eta_nominal={e:.4f} -> "
                        f"sqrt(max n_s)={sqrt_ns_peak[i]:.4f}  "
                        f"(ratio {sqrt_ns_peak[i] / e:.3f})")

    ratio = sqrt_ns_peak / eta_nominal
    return {"eta_nominal": eta_nominal, "sqrt_ns_peak": sqrt_ns_peak, "ratio": ratio,
            "target_eta": float(target_eta), "shape": shape, "rows": rows}


def plot_eta_vs_ns(result: Dict[str, Any], out: str = "figs/eta_vs_ns.png",
                   title: Optional[str] = None) -> str:
    """Render :func:`verify_eta_matches_ns`; returns the path written.

    Top row: measured ``sqrt(<n_s(t)>)`` against the nominal ``|eta(t)|`` per drive.
    Bottom: ``sqrt(max n_s)`` vs nominal ``eta`` with a ``y = x`` reference; a
    departure that GROWS with drive means the eta axis is mislabelled there.
    """
    from snail_solver.envelope import RaisedCosine
    plt = _pyplot()

    rows = result["rows"]
    n = len(rows)
    fig = plt.figure(figsize=(3.4 * n, 7.2), layout="constrained")
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.3])

    for i, row in enumerate(rows):
        ax = fig.add_subplot(gs[0, i])
        ts = np.asarray(row["times_ns"], dtype=float)
        ax.plot(ts, np.sqrt(np.clip(row["n_s_t"], 0.0, None)), "-", lw=2.0,
               color=_C_LORENTZ, label=r"measured $\sqrt{\langle n_s(t)\rangle}$")
        if result["shape"] == "raised_cosine":
            env = np.asarray(RaisedCosine(row["eta_nominal"], row["t_g_ns"]).value_at(ts))
        else:
            env = np.full_like(ts, row["eta_nominal"])
        ax.plot(ts, env, "--", lw=1.6, color=_C_INK, label=r"nominal $|\eta(t)|$")
        ax.set_title(rf"$\eta_{{nom}}$ = {row['eta_nominal']:.3f}", fontsize=9)
        ax.set_xlabel("time (ns)")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel(r"$|\eta|$")
            ax.legend(fontsize=7, framealpha=0.9, loc="upper right")

    axr = fig.add_subplot(gs[1, :])
    eta_nom = np.asarray(result["eta_nominal"], dtype=float)
    sqrt_ns = np.asarray(result["sqrt_ns_peak"], dtype=float)
    axr.plot(eta_nom, sqrt_ns, "o-", ms=8, lw=1.4, color=_C_LORENTZ,
             label=r"measured $\sqrt{\max_t\langle n_s(t)\rangle}$")
    hi = float(max(np.nanmax(eta_nom), np.nanmax(sqrt_ns)) * 1.08)
    axr.plot([0.0, hi], [0.0, hi], "--", lw=1.6, color=_C_INK,
             label=r"$y=x$ (label matches $n_s$)")
    axr.axvline(result["target_eta"], color=_C_VERTEX, ls=":", lw=1.6,
                label=f"target_eta={result['target_eta']:.2f}")
    axr.set_xlabel(r"nominal $\eta$ (what the pulse was built to have)")
    axr.set_ylabel(r"measured $\sqrt{\langle n_s\rangle}$ (coupler occupation)")
    axr.set_title("does eta still mean sqrt(n_s)?", fontsize=11)
    axr.legend(fontsize=8, framealpha=0.9)
    axr.grid(alpha=0.25)

    fig.suptitle(title or f"eta vs sqrt(n_s), shape={result['shape']}", fontsize=12)
    return _savefig(fig, out)


# ===========================================================================
# Step 1 -- the Rabi experiment: resonance vs drive strength
# ===========================================================================
def fit_shift_curve(eta: np.ndarray, delta_MHz: np.ndarray,
                    weights: Optional[np.ndarray] = None,
                    fit_static: bool = True) -> Dict[str, Any]:
    """Fit delta(|eta|) = delta0 + k2 |eta|^2 + k4 |eta|^4 and split it in two.

    * ``delta0`` -- the offset surviving at ZERO drive, from the static Hamiltonian
      (qutrit anharmonicity, coupler dressing). It belongs in ``wp_offset_GHz``; a
      chirp tracking it would double-count.
    * ``k2``, ``k4`` -- the drive-dependent AC-Stark shift, the only part a chirp
      can correct.

    On evan_device the static part dominates (~-0.7 MHz); forcing the curve through
    the origin would inflate k2/k4 into a wrong chirp. Use ``fit_static=False`` only
    when the probe has no static offset. Even powers only: the shift depends on
    intensity, and the chirp extrapolates down to |eta| = 0.

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
        ``resid_MHz``, ``stark_span_MHz`` (how much the drive-dependent part moves
        across the measured window).
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


def shift_curve_stability(eta: np.ndarray, delta_MHz: np.ndarray,
                          weights: Optional[np.ndarray] = None,
                          target_eta: Optional[float] = None, *,
                          cutoffs: Sequence[float] = (1.0, 0.9, 0.8, 0.7),
                          fit_static: bool = True) -> Dict[str, Any]:
    """Refit ``delta(|eta|)`` on nested weakest-drive subsets; report how much
    the extrapolated ``delta(target_eta)`` moves.

    A good ``r2`` over a mix of clean and contaminated rows says nothing about the few
    highest-drive rows a chirp at ``target_eta`` depends on most. If dropping the top
    10-30% of rows swings the shift at ``target_eta`` by a large fraction of itself,
    the fit has not determined it. Unlike ``extrapolation_ratio`` this also catches
    contamination of an INTERPOLATING fit.

    Parameters
    ----------
    eta, delta_MHz, weights : ndarray
        As passed to :func:`fit_shift_curve`.
    target_eta : float, optional
        Where to evaluate the extrapolated shift; defaults to ``max(eta)``.
    cutoffs : sequence of float
        Fractions of the max USED ``|eta|`` to refit on (nested, weakest-drive
        subsets).
    fit_static : bool
        As in :func:`fit_shift_curve`.

    Returns
    -------
    dict
        ``cutoffs``, ``delta_by_cutoff`` (MHz, NaN where too few rows survive
        a cutoff), ``n_used_by_cutoff``, ``target_eta``, and ``delta_spread``
        (``(max - min) / max(|median|, eps)`` over the finite entries of
        ``delta_by_cutoff``; NaN if fewer than 2 cutoffs produced a fit).
    """
    eta = np.asarray(eta, dtype=float)
    y = np.asarray(delta_MHz, dtype=float)
    ok = np.isfinite(eta) & np.isfinite(y)
    w_all = np.ones_like(eta) if weights is None else np.asarray(weights, dtype=float)
    cutoffs = list(cutoffs)
    if not ok.any():
        return {"cutoffs": cutoffs, "delta_by_cutoff": [float("nan")] * len(cutoffs),
                "n_used_by_cutoff": [0] * len(cutoffs), "delta_spread": float("nan"),
                "target_eta": float(target_eta) if target_eta is not None else float("nan")}
    eta_max = float(np.max(eta[ok]))
    eta_star = float(target_eta) if target_eta is not None else eta_max
    n_min = 4 if fit_static else 3

    delta_by_cutoff = []
    n_used_by_cutoff = []
    for c in cutoffs:
        keep = ok & (eta <= float(c) * eta_max + 1e-12)
        n_used_by_cutoff.append(int(keep.sum()))
        if keep.sum() < n_min:
            delta_by_cutoff.append(float("nan"))
            continue
        try:
            fit = fit_shift_curve(eta[keep], y[keep], weights=w_all[keep],
                                  fit_static=fit_static)
        except ValueError:
            delta_by_cutoff.append(float("nan"))
            continue
        delta_by_cutoff.append(float(fit["k2"] * eta_star ** 2 + fit["k4"] * eta_star ** 4))

    finite = [d for d in delta_by_cutoff if np.isfinite(d)]
    if len(finite) >= 2:
        med = float(np.median(finite))
        spread = (max(finite) - min(finite)) / max(abs(med), 1e-9)
    else:
        spread = float("nan")

    return {"cutoffs": cutoffs, "delta_by_cutoff": delta_by_cutoff,
            "n_used_by_cutoff": n_used_by_cutoff, "delta_spread": float(spread),
            "target_eta": eta_star}


def fit_chevron_center(offsets_GHz: np.ndarray, metric: np.ndarray) -> Dict[str, Any]:
    """Resonance offset of a constant-drive chevron, by fitting its ANALYTIC envelope.

    For a two-level exchange at Rabi rate Omega and detuning d, peak transfer over
    time is the Lorentzian ``Omega^2 / (Omega^2 + d^2)``. Fitting the whole shape
    (rather than a 3-point parabola at the argmax) makes a SUB-MHz shift measurable
    on a coarser grid. The parabolic estimate
    (``find_stark_resonance.locate_resonance``) is returned as ``vertex_GHz``.

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


def chevron_quality(cen: Dict[str, Any], offsets_GHz: np.ndarray, metric: np.ndarray,
                    span_MHz: float, *, leak: Optional[float] = None,
                    contrast_min: float = 0.35, gap_frac_max: float = 0.5,
                    nrmse_max: float = 0.1, secondary_max: float = 0.4,
                    leak_max: float = 0.35) -> Dict[str, Any]:
    """How much of a two-level Lorentzian this chevron actually is.

    Cheap checks from arrays already in hand:

    * ``contrast`` -- max - min of the envelope.
    * ``gap_frac`` -- ``|center - vertex|`` over the linewidth (not the shift, which
      is what is being measured). Weak alone: a failed fit falls back to the vertex
      for both, giving ``gap_frac == 0``.
    * ``nrmse`` -- fit rmse over contrast (the ratio :func:`fit_chevron_center`'s
      ``ok`` test thresholds at 0.1).
    * ``secondary`` -- second-highest local maximum (above 30% of the range), its
      height above the minimum as a fraction of the range; 0 for a unimodal
      chevron. This is what catches a bimodal chevron when both estimators
      collapse onto the same wrong peak.
    * ``hwhm_frac`` -- ``hwhm_MHz / span_MHz``; reported, not used to reject.
    * ``leak`` -- caller-supplied leakage at the resonance, if given.

    `cen` is :func:`fit_chevron_center`'s output for the envelope `metric` on
    `offsets_GHz`; `span_MHz` is the row's scan span. Returns the checks above plus a
    composite ``weight`` in ``[0, 1]`` (plain ``contrast`` on a clean row with
    ``leak=0``) and ``reject``: the first failing of low_contrast, multi_peak,
    poor_fit, estimators_disagree, high_leakage (or ``None``).
    """
    x = np.asarray(offsets_GHz, dtype=float)
    y = np.asarray(metric, dtype=float)
    ok_pts = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok_pts], y[ok_pts]
    contrast = float(np.nanmax(y) - np.nanmin(y)) if y.size else float("nan")

    hwhm_MHz = float(cen.get("hwhm_GHz", np.nan)) * 1e3
    grid_step_MHz = (float(np.min(np.diff(np.sort(x)))) * 1e3
                     if x.size >= 2 else float("nan"))
    denom = max(v for v in (hwhm_MHz, grid_step_MHz, 1e-9) if np.isfinite(v))
    gap_MHz = abs(float(cen.get("center_GHz", np.nan))
                 - float(cen.get("vertex_GHz", np.nan))) * 1e3
    gap_frac = gap_MHz / denom if np.isfinite(gap_MHz) and denom > 0 else float("nan")

    rmse = float(cen.get("rmse", np.nan))
    nrmse = rmse / max(contrast, 1e-9) if np.isfinite(rmse) else float("nan")

    # secondary peak: local maxima of y more than 30% of the way from min to max,
    # excluding the global max itself.
    secondary = 0.0
    if y.size >= 5:
        y_max, y_min = float(np.nanmax(y)), float(np.nanmin(y))
        thresh = y_min + 0.3 * (y_max - y_min)
        is_peak = np.zeros(y.size, dtype=bool)
        is_peak[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > thresh)
        is_peak[int(np.nanargmax(y))] = False
        if is_peak.any():
            secondary = float((np.nanmax(y[is_peak]) - y_min) / max(y_max - y_min, 1e-9))

    hwhm_frac = (hwhm_MHz / float(span_MHz)
                if np.isfinite(hwhm_MHz) and span_MHz else float("nan"))
    leak_val = float(leak) if leak is not None and np.isfinite(leak) else 0.0

    def _score(value: float, limit: float) -> float:
        return 1.0 if not np.isfinite(value) else float(np.exp(-(value / limit) ** 2))

    weight = (max(contrast, 0.0) * _score(gap_frac, gap_frac_max)
             * _score(nrmse, nrmse_max) * max(1.0 - secondary, 0.0)
             * max(1.0 - leak_val, 0.0))

    reject = None
    if not np.isfinite(contrast) or contrast < contrast_min:
        reject = "low_contrast"
    elif np.isfinite(secondary) and secondary > secondary_max:
        reject = "multi_peak"
    elif np.isfinite(nrmse) and nrmse > nrmse_max:
        reject = "poor_fit"
    elif np.isfinite(gap_frac) and gap_frac > gap_frac_max:
        reject = "estimators_disagree"
    elif leak_val > leak_max:
        reject = "high_leakage"

    return {"contrast": contrast, "gap_frac": gap_frac, "nrmse": nrmse,
            "secondary": secondary, "hwhm_frac": hwhm_frac, "leak": leak_val,
            "weight": float(np.clip(weight, 0.0, 1.0)), "reject": reject}


class StarkCrossingInSweep(RabiFitError):
    """The ridge changed transition part-way up the drive sweep.

    Unlike other `RabiFitError` failures this is not a bad fit: the measurement is
    of two different things. `crossing_eta` (where the transitions swapped) locates the
    collision without a separate sweep.
    """

    def __init__(self, message, table=None, crossing_eta=None):
        super().__init__(message, table)
        self.crossing_eta = crossing_eta


#: Weight of a ridge row HELD across a contrast/leakage drop, relative to the median
#: surviving weight: too small to set k2/k4 alone, large enough that the unweighted
#: r2 in `fit_shift_curve` does not reject the column over it.
HELD_ROW_WEIGHT = 0.25

#: A ridge step this many times the median adjacent step (and the typical step on
#: either side) means the centre-finder CHANGED TRANSITION: the Stark shift is
#: continuous in drive and cannot step.
CONTINUITY_STEPS = 4.0


def rabi_shift_table(config: Dict[str, Any], target_eta: float, *,
                     eta_lo: float = 0.3, eta_hi: float = 1.0,
                     amp_points: int = 9, wp_span_MHz: Optional[float] = None,
                     wp_points: int = 25, span_linewidths: float = 4.0,
                     chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                     drag_beat_GHz: Optional[float] = None, drag_n_pump: int = 1,
                     spec_abs_GHz: Optional[float] = None,
                     wp_offset_GHz: float = 0.0, r2_min: float = 0.9,
                     contrast_min: float = 0.35, gap_frac_max: float = 0.5,
                     nrmse_max: float = 0.1, secondary_max: float = 0.4,
                     leak_max: float = 0.35, stability_max: float = 0.3,
                     stability_cutoffs: Sequence[float] = (1.0, 0.9, 0.8, 0.7),
                     window_tg: float = 2.0, n_time: int = 161, jobs: int = 0,
                     zero_chirp_frac: float = 0.0,
                     max_span_growths: int = 3, max_wp_points: int = 121,
                     span_growth: str = "railed", max_compound_growths: int = 1,
                     probe_shape: str = "constant",
                     moment_weighting: str = "rabi",
                     solver: Optional[Dict[str, Any]] = None,
                     logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Step 1: measure the resonant pump offset as a function of drive strength.

    One CONSTANT-amplitude chevron per drive strength: the pump frequency is scanned
    and the population read out over TIME; the resonance is the offset of maximum
    Rabi contrast. The constant probe measures the INSTANTANEOUS shift delta(|eta|),
    exactly what the chirp needs, and is blind to DRAG (d eta/dt = 0), so
    `drag_beat_GHz` only reaches the shaped cross-check in `calibrate_drag_offset`.

    ``probe_shape="gate"`` exists because the constant probe fails at strong drive
    (held at ``|eta| = 1.2`` on ``4Gate4.5SNAIL`` it leaks 0.245, transiently 0.93; the
    shaped pulse at the same peak leaks 4e-3). A shaped rung reports a moment-weighted
    average; the law being an even polynomial, the average is diagonal in
    ``{eta^2, eta^4}`` (:func:`stark_chirp.stark_moments`)::

        <delta>(eta*) = k2 M2 eta*^2 + k4 M4 eta*^4   ->   k2 = K2/M2, k4 = K4/M4

    ``delta0`` is NOT rescaled, and downstream still receives a POINTWISE law. The
    shift is smaller by ``M2`` (1.4x on a Hann with ``"rabi"``), but higher peak
    drives become reachable and ``extrapolation_ratio -> 1``.

    ``moment_weighting`` scales the chirp directly (the weightings span 2x). The
    default ``"rabi"`` is derived (the chevron centre averages the shift against
    ``sin theta(t)``) and confirmed by :func:`cross_check_probe_moments`; the others
    are for comparison: their smaller ``M2`` OVERESTIMATES k2, hence the chirp.

    Every point is an exact ``sesolve`` trajectory: ``amp_points * wp_points`` solves.
    The span and window are ADAPTIVE per row, since the linewidth
    ``Omega ~ 1 / (2 t_g(|eta|))`` grows with drive; a row re-measures wider if its
    fitted centre rails at the window edge (``span_growth="both"`` also widens a row
    whose fitted width is too large for its window).

    Parameters
    ----------
    eta_lo, eta_hi : float
        Amplitude window as a FRACTION of `target_eta`. A constant probe held above
        the operating drive stops being a two-level feature, hence ``eta_hi=1``.
    wp_span_MHz : float, optional
        Fixed offset span for every row; None sizes each row from its linewidth.
    span_linewidths : float
        Half-span in estimated linewidths when `wp_span_MHz` is None.
    window_tg : float
        Chevron window of a constant-probe row, in units of that row's
        ``nominal_t_g(|eta|)`` (2 swaps each at constant drive); a shaped rung runs
        one pulse.
    r2_min : float
        Refuse a fit worse than this (a railed ridge yields a plausible wrong chirp).
    contrast_min, gap_frac_max, nrmse_max, secondary_max, leak_max : float
        :func:`chevron_quality` thresholds; a row failing any is dropped.
    stability_max : float
        Warning-only threshold on :func:`shift_curve_stability`'s ``delta_spread``;
        logged and recorded, never raises.
    stability_cutoffs : sequence of float
        Passed through to :func:`shift_curve_stability`.

    Returns
    -------
    dict
        ``eta``, ``delta_MHz`` (the ridge), ``fit`` (from :func:`fit_shift_curve`),
        ``stability`` (from :func:`shift_curve_stability`), ``t_g_ref_ns``,
        ``target_eta``, ``contrast``, ``quality`` (the composite weight each row
        was fit with), ``leakage`` (leak at each row's resonance), ``chevrons``.
    """
    from snail_solver import find_stark_resonance as FSR

    t_g0 = nominal_t_g(config, target_eta)
    eta = np.linspace(float(eta_lo), float(eta_hi), int(amp_points)) * float(target_eta)

    def linewidth_MHz(e: float) -> float:
        """Leading-order chevron HWHM: a full swap in T means Omega = 1/(2T)."""
        return 1e3 / (2.0 * nominal_t_g(config, float(e)))

    ridge = np.full(eta.size, np.nan)
    contrast = np.full(eta.size, np.nan)
    quality = np.full(eta.size, np.nan)
    leakage = np.full(eta.size, np.nan)
    windows = np.full(eta.size, np.nan)
    spans = np.full(eta.size, np.nan)
    n_offsets = np.full(eta.size, float(wp_points))
    chevrons = []
    shaped = str(probe_shape) != "constant"
    for i, e in enumerate(eta):
        # window_tg x nominal_t_g(e) per constant row. A SHAPED rung is a full iSWAP
        # over nominal_t_g(e) (amp_scale = 1.0 makes the peak exactly e only on a
        # Hann: 2 * area_factor in general), so its window is the pulse.
        t_g_rung = nominal_t_g(config, float(e))
        window_ns = t_g_rung if shaped else float(window_tg) * t_g_rung
        span = (float(wp_span_MHz) if wp_span_MHz is not None
                else 2.0 * float(span_linewidths) * linewidth_MHz(e))

        n_off = int(wp_points)
        for attempt in range(int(max_span_growths) + 1):
            offsets_GHz = (np.linspace(-span / 2.0, span / 2.0, n_off) * 1e-3
                           + float(wp_offset_GHz))
            chev = FSR.scan(config,
                            t_g_rung if shaped else t_g0, 1.0,
                            offsets_GHz, window_ns, int(n_time),
                            solver=solver, n_jobs=jobs, spec_abs_GHz=spec_abs_GHz,
                            shape=("gate" if shaped else "constant"),
                            chirp_coeffs_GHz=None,
                            drag_n_pump=drag_n_pump, eta_op=float(e),
                            keep_full_channels=True)
            m = np.asarray(chev["resonance_metric"], dtype=float)
            cen = fit_chevron_center(chev["offsets_GHz"], m)
            # Two triggers to widen, selected by `span_growth`:
            # (a) too-wide (3x, only with "both"): the fitted WIDTH exceeds 0.4 span.
            #     Default OFF -- measured over the 2026-09-25 grid (a 3x growth keeps
            #     the step, so the central third is an exact counterfactual) it LOST
            #     rows (505 kept vs 601), moved centres by p99 11.4 MHz, cost 28.8% of
            #     offset-solves, and mostly admitted the neighbouring transition. A
            #     broad line is not an unmeasured one.
            # (b) railed (2x): the CENTRE sits at the window edge; without growth
            #     99.4% of such rows rail and kill whole columns in the post-loop check.
            # Compounding is capped (max_compound_growths): a second growth kept 4.1%
            # of its rows. The grid GROWS WITH THE SPAN: widening at fixed wp_points
            # coarsens the step, degrading the fit and slackening the rail test
            # (whose tolerance is the step).
            _step = span / max(n_off - 1, 1)
            _railed_row = (abs((cen["center_GHz"] - float(wp_offset_GHz)) * 1e3)
                           >= 0.5 * span - _step)
            _too_wide = (cen["hwhm_GHz"] * 1e3 > 0.4 * span
                         and str(span_growth) == "both")
            if str(span_growth) == "off":
                _railed_row = False
            if (not (_too_wide or _railed_row) or wp_span_MHz is not None
                    or attempt == int(max_span_growths)
                    or attempt >= int(max_compound_growths)):
                break
            why = "ridge railed at the window edge" if _railed_row else (
                f"hwhm {cen['hwhm_GHz'] * 1e3:.2f} MHz too wide")
            grow = 3.0 if _too_wide else 2.0
            span *= grow
            n_off = min(int(round((n_off - 1) * grow)) + 1, int(max_wp_points))
            if logger:
                logger.info(f"    row {i + 1}: {why} for +/-{span / (2 * grow):.1f} "
                            f"MHz -- retrying at +/-{span / 2:.1f} MHz over "
                            f"{n_off} offsets")

        windows[i] = window_ns
        spans[i] = span
        n_offsets[i] = n_off
        contrast[i] = float(np.nanmax(m) - np.nanmin(m))
        leakage[i] = float(chev["leak_on_resonance"])
        j_res = int(np.argmin(np.abs(np.asarray(chev["offsets_GHz"], dtype=float)
                                     - cen["center_GHz"])))
        leak_breakdown = {k: float(chev[f"leak_{k}"][j_res])
                          for k in ("f_a", "f_b", "coupler", "double", "spectator")}
        q = chevron_quality(cen, chev["offsets_GHz"], m, span, leak=leakage[i],
                            contrast_min=contrast_min, gap_frac_max=gap_frac_max,
                            nrmse_max=nrmse_max, secondary_max=secondary_max,
                            leak_max=leak_max)
        quality[i] = q["weight"]
        row_common = {"eta": float(e), "offsets_GHz": chev["offsets_GHz"],
                     "metric": m, "fit": cen, "window_ns": window_ns,
                     "span_MHz": span, "times_ns": chev["times_ns"],
                     "P10": chev["P10"], "P01": chev["P01"], "P_leak": chev["P_leak"],
                     "leak_at_metric": chev["leak_at_metric"],
                     "leak_breakdown": leak_breakdown,
                     "norm_defect_max": chev["norm_defect_max"], "quality": q}
        # A rejected chevron is not a resonance; one such row would otherwise set
        # k2, k4 for the whole chirp. fit_shift_curve ignores NaNs.
        leak_txt = (f"leak {leakage[i]:.3f} (f_a {leak_breakdown['f_a']:.3f} "
                    f"f_b {leak_breakdown['f_b']:.3f} "
                    f"coupler {leak_breakdown['coupler']:.3f} "
                    f"|11> {leak_breakdown['double']:.3f})")
        if q["reject"]:
            ridge[i] = np.nan
            if logger:
                logger.info(f"  rabi row {i + 1}/{eta.size}: |eta|={e:.4f} DROPPED "
                            f"({q['reject']}) -- contrast {contrast[i]:.3f}, {leak_txt}")
            chevrons.append({**row_common, "dropped": q["reject"]})
            continue
        # report the ridge RELATIVE to the offset the probe already carries, so the
        # caller accumulates a residual rather than re-adding the current setting
        ridge[i] = (cen["center_GHz"] - float(wp_offset_GHz)) * 1e3
        chevrons.append(row_common)
        if logger:
            logger.info(f"  rabi row {i + 1}/{eta.size}: |eta|={e:.4f} -> "
                        f"{ridge[i]:+.4f} MHz (span +/-{span / 2:.1f} MHz, contrast "
                        f"{contrast[i]:.3f}, quality {quality[i]:.3f}, "
                        f"hwhm {cen['hwhm_GHz'] * 1e3:.2f} MHz, "
                        f"{'lorentzian' if cen['ok'] else 'PARABOLIC FALLBACK'}, "
                        f"vertex {(cen['vertex_GHz'] - wp_offset_GHz) * 1e3:+.4f} MHz, "
                        f"{leak_txt})")

    partial = {"eta": eta, "delta_MHz": ridge, "t_g_ref_ns": t_g0,
               "target_eta": float(target_eta), "contrast": contrast,
               "quality": quality, "leakage": leakage,
               "windows_ns": windows, "spans_MHz": spans, "chevrons": chevrons}

    # A ridge on the scan edge is not a measurement; anything still railing here
    # outran the growth budget (or sits on a hand-pinned --wp-span-MHz).
    step = spans / np.maximum(n_offsets - 1.0, 1.0)
    railed = np.isfinite(ridge) & (np.abs(np.abs(ridge) - spans / 2.0) <= step)
    if railed.any():
        raise RabiFitError(
            f"{int(railed.sum())}/{ridge.size} ridge rows still rail against their "
            f"scan window after growing it "
            f"{'(pinned by --wp-span-MHz)' if wp_span_MHz is not None else
               f'up to {max_span_growths}x'} "
            f"(spans {np.nanmin(spans) / 2:.1f}-{np.nanmax(spans) / 2:.1f} MHz "
            f"half-width) -- raise --span-linewidths or --max-span-growths. A railed "
            f"ridge produces a confident, wrong chirp.", partial)

    # The Stark shift is CONTINUOUS in drive, so a ridge that holds one level, jumps,
    # and holds another means the centre-finder locked onto a DIFFERENT TRANSITION
    # from some drive up. The branch continuous with the BOTTOM of the sweep (where
    # the shift -> 0) is the real one. E.g. delta = +85 MHz, eta* = 1.3 (2026-09-22):
    # a +1.43 MHz step (~50x the median) where the hwhm spikes, then smooth on a new
    # level; all 25 r2 failures on that grid looked like this. No chirp is built
    # through it; the column is reported as a CROSSING instead.
    _fin = np.flatnonzero(np.isfinite(ridge))
    partial["stark_crossing_eta"] = None
    if _fin.size >= 8:
        _d = np.abs(np.diff(ridge[_fin]))
        _med = float(np.median(_d))
        _j = int(np.argmax(_d))
        # A step only counts if what follows SETTLES (a single wild row is noise, a
        # monotone steepening is k4): compare against the typical step on each side.
        _lo, _hi = _d[:_j], _d[_j + 1:]
        _local = max(float(np.median(_lo)) if _lo.size else 0.0,
                     float(np.median(_hi)) if _hi.size else 0.0)
        if (_med > 0 and _lo.size >= 3 and _hi.size >= 3
                and _d[_j] > CONTINUITY_STEPS * max(_med, _local)):
            _eta_c = float(eta[_fin[_j]])
            if logger:
                logger.info(
                    f"  rabi: the ridge STEPS {_d[_j]:.3f} MHz between |eta| = "
                    f"{_eta_c:.4f} and {eta[_fin[_j + 1]]:.4f}, against a "
                    f"{_med:.3f} MHz median step, and then settles on the new level. "
                    f"A Stark shift is continuous in drive, so this is the "
                    f"centre-finder tracking a DIFFERENT transition above "
                    f"|eta| = {_eta_c:.3f} -- an avoided crossing inside the sweep.")
            partial["stark_crossing_eta"] = _eta_c
            partial["fit"] = partial.get("fit")
            raise StarkCrossingInSweep(
                f"the measured ridge steps {_d[_j]:.3f} MHz at |eta| = {_eta_c:.4f} "
                f"({_d[_j] / max(_med, 1e-12):.0f}x the median step) and then holds "
                f"the new level. The Stark shift is continuous in drive, so above "
                f"that drive the chevron centre is a DIFFERENT transition -- the "
                f"correct one has left the window or is buried under a competing "
                f"peak. eta* = {target_eta:g} sits above the crossing, so a chirp "
                f"here would be built from mis-tracked rows, and the law fitted "
                f"below would have to be extrapolated across the crossing to reach "
                f"the operating point. Widen --span-linewidths so the correct peak "
                f"is inside the window, or lower --eta-hi / the target drive to stay "
                f"below |eta| = {_eta_c:.3f}.", partial, crossing_eta=_eta_c)

    # Rescue when too few rows survive: rows drop for leakage/low contrast at the TOP
    # of the drive range, so survivors are a prefix in |eta|. Hold the last good value
    # across the dropped tail (and the first across any dropped head) -- a floor that
    # UNDER-states a monotone law, so the chirp is conservative. Held rows get
    # HELD_ROW_WEIGHT, not zero: r2 is computed unweighted, so a zero-weight flat tail
    # would still fail r2_min. Recorded in `n_held` (on the RabiFitError table only;
    # the success return drops it): such a law is weaker evidence.
    held = np.zeros(eta.size, dtype=bool)
    if int(np.isfinite(ridge).sum()) < 4 and np.isfinite(ridge).any():
        _good = np.flatnonzero(np.isfinite(ridge))
        _w = float(np.nanmedian(quality[_good])) if np.isfinite(
            quality[_good]).any() else 1.0
        _w = HELD_ROW_WEIGHT * (_w if np.isfinite(_w) else 1.0)
        _last = float(ridge[_good[-1]])
        for _i in range(_good[-1] + 1, eta.size):
            if not np.isfinite(ridge[_i]):
                ridge[_i], held[_i], quality[_i] = _last, True, _w
        _first = float(ridge[_good[0]])
        for _i in range(_good[0] - 1, -1, -1):
            if not np.isfinite(ridge[_i]):
                ridge[_i], held[_i], quality[_i] = _first, True, _w
        if held.any() and logger:
            logger.info(
                f"  rabi: only {_good.size} row(s) survived the contrast/leakage "
                f"floor -- HOLDING the ridge at {_last:+.4f} MHz across "
                f"{int(held.sum())} dropped row(s) so the law stays determined. The "
                f"held rows carry {HELD_ROW_WEIGHT:g}x the surviving rows' weight "
                f"and the shift is assumed flat beyond the last row that swapped, "
                f"which under-states it.")
    partial["held"] = held
    partial["n_held"] = int(held.sum())

    # Re-raised as RabiFitError with the table attached, so main() can still save and
    # draw the chevrons ("which rows dropped, and why" is the diagnosis).
    try:
        fit = fit_shift_curve(eta, ridge, weights=quality)
    except ValueError as exc:
        dropped = [f"|eta|={c['eta']:.3f} {c['dropped']}"
                   for c in partial["chevrons"] if c.get("dropped")]
        raise RabiFitError(
            f"{exc}. Dropped rows: {', '.join(dropped) if dropped else 'none'}. "
            f"Inspect the chevrons: 'multi_peak' means a competing resonance sits "
            f"in the window, 'low_contrast' means the probe never completed a swap "
            f"there. Widen --wp-span-MHz if the ridge is leaving the window, or "
            f"lower --eta-hi to stay inside the drive range that still swaps.",
            partial) from exc
    if shaped:
        # A shaped rung reported <delta>: undo the envelope moments ONCE so consumers
        # receive a pointwise k2/k4. delta0 is drive-independent and not rescaled;
        # stark_span_MHz stays in measured units, like the residual it is compared to.
        M2, M4 = probe_moments(config, moment_weighting)
        fit = dict(fit, k2=float(fit["k2"]) / M2, k4=float(fit["k4"]) / M4,
                   K2_measured=float(fit["k2"]), K4_measured=float(fit["k4"]),
                   M2=float(M2), M4=float(M4),
                   moment_weighting=str(moment_weighting))
    fit["probe_shape"] = str(probe_shape)
    partial["fit"] = fit
    stability = shift_curve_stability(eta, ridge, weights=quality, target_eta=target_eta,
                                      cutoffs=stability_cutoffs)
    partial["stability"] = stability

    # r2 is RELATIVE, so a column with a near-zero Stark shift fails it on a small
    # denominator even with pristine chevrons (delta = -100 MHz, 2026-09-17: contrast
    # >= 0.874, leak <= 0.001, r2 = 0.585 on a 0.405 MHz signal). What matters is
    # whether a chirp would DO anything: the excursion |k2 eta^2 + k4 eta^4| against
    # the half-width 1/(2 t_g). Keep --zero-chirp-frac conservative (a 48% excursion
    # was worth 2.81x; 10-15% columns are the recoverable ones). Off by default here
    # (it changes the pulse), but run_tune_up passes chirp_free_max_frac when unset.
    _exc = abs(fit["k2"] * target_eta ** 2 + fit["k4"] * target_eta ** 4)
    _half_lw = 1e3 / (2.0 * float(partial.get("t_g_ref_ns") or nominal_t_g(
        config, target_eta)))
    _frac = _exc / _half_lw if _half_lw else float("inf")
    fit["chirp_excursion_MHz"] = float(_exc)
    fit["chirp_excursion_frac_linewidth"] = float(_frac)
    fit["chirp_zeroed"] = False
    fit["stark_absorbed_MHz"] = 0.0
    if (not (fit["r2"] >= r2_min)) and zero_chirp_frac > 0.0 \
            and _frac <= zero_chirp_frac:
        # The shift at eta* is static, not absent: ABSORB it into delta0 (a retuned
        # carrier) instead of discarding it. Both are resonance offsets in the same
        # frame, so they add.
        _stark = float(fit["k2"]) * target_eta ** 2 + float(fit["k4"]) * target_eta ** 4
        fit["k2"], fit["k4"] = 0.0, 0.0
        fit["chirp_zeroed"] = True
        fit["stark_absorbed_MHz"] = float(_stark)
        fit["delta0_before_absorb_MHz"] = float(fit["delta0"])
        fit["delta0"] = float(fit["delta0"]) + _stark
        fit["stark_span_MHz"] = 0.0
        if logger:
            logger.info(
                f"  rabi: r2 = {fit['r2']:.3f} < {r2_min}, but the chirp would sweep "
                f"only {_exc:.3f} MHz = {_frac:.0%} of the {_half_lw:.2f} MHz "
                f"half-linewidth (<= --zero-chirp-frac {zero_chirp_frac:g}). The "
                f"shift is static rather than absent, so it is ABSORBED INTO THE "
                f"CARRIER: delta0 {fit['delta0_before_absorb_MHz']:+.4f} "
                f"-> {fit['delta0']:+.4f} MHz ({_stark:+.4f} MHz of Stark shift at "
                f"eta* = {target_eta:g}), and the chirp is ZERO.")
    if not (fit["r2"] >= r2_min) and not fit["chirp_zeroed"]:
        raise RabiFitError(
            f"the ridge is not well described by delta0 + k2|eta|^2 + k4|eta|^4 "
            f"(r2 = {fit['r2']:.3f} < {r2_min}, residual {fit['resid_MHz']:.4f} MHz). "
            f"The ridge may be tracking leakage rather than the Stark shift, or the "
            f"offset grid may be too coarse to resolve it. Inspect the chevrons "
            f"before trusting a chirp built from them.", partial)
    # A drive-dependent span below the fit's scatter is not a measured shift (the
    # static part is still trustworthy).
    if fit["stark_span_MHz"] <= fit["resid_MHz"] and not fit["chirp_zeroed"]:
        raise RabiFitError(
            f"the DRIVE-DEPENDENT shift ({fit['stark_span_MHz']:.4f} MHz across "
            f"|eta| in [{eta[0]:.2f}, {eta[-1]:.2f}]) is smaller than the fit residual "
            f"({fit['resid_MHz']:.4f} MHz), so there is no resolved Stark shift to "
            f"build a chirp from. The static offset delta0 = {fit['delta0']:+.4f} MHz "
            f"is still meaningful -- calibrate wp_offset and run without a chirp, or "
            f"widen --eta-lo/--eta-hi and refine --wp-points until the drive "
            f"dependence clears the noise.", partial)
    if logger:
        if shaped:
            logger.info(f"  rabi: SHAPED probe ({config.get('envelope')}), moments "
                        f"M2={fit['M2']:.4f} M4={fit['M4']:.4f} "
                        f"({fit['moment_weighting']}): measured K2={fit['K2_measured']:+.4f} "
                        f"-> pointwise k2={fit['k2']:+.4f}")
        logger.info(f"  rabi: delta0={fit['delta0']:+.4f} MHz (static), "
                    f"k2={fit['k2']:+.4f} MHz/|eta|^2, k4={fit['k4']:+.4f} "
                    f"MHz/|eta|^4, r2={fit['r2']:.4f} over {fit['n_used']} rows; "
                    f"drive-dependent span {fit['stark_span_MHz']:.4f} MHz vs "
                    f"residual {fit['resid_MHz']:.4f} MHz")
        # Warn-only: the shift at target_eta may be underdetermined even at good r2.
        if np.isfinite(stability["delta_spread"]) and stability["delta_spread"] > stability_max:
            logger.info(f"  WARNING: delta(target_eta) is UNSTABLE under row cutoffs "
                        f"(spread {stability['delta_spread']:.2f} > {stability_max}, "
                        f"values {['%.2f' % d for d in stability['delta_by_cutoff']]} MHz "
                        f"over cutoffs {stability['cutoffs']}) -- the fit has not "
                        f"determined the shift at this drive; it has found a curve "
                        f"through noisy/contaminated rows. Treat any chirp built from "
                        f"it as a candidate, not a calibration.")
    return {"eta": eta, "delta_MHz": ridge, "fit": fit, "stability": stability,
            "t_g_ref_ns": t_g0, "target_eta": float(target_eta), "contrast": contrast,
            "quality": quality, "leakage": leakage,
            "windows_ns": windows, "spans_MHz": spans, "chevrons": chevrons}


# ===========================================================================
# Step 2 -- chirp by projecting the MEASURED curve
# ===========================================================================
def parse_drag_channels(specs: Optional[Sequence[str]]) -> Optional[list]:
    """Parse repeated ``--drag-channel BEAT[:K[:N]]`` into :class:`DragChannel` list.

    ``K`` is the pump-quanta count (how the beat moves under a chirp) and ``N`` the
    photon count in ``F^(n)``; ``N`` defaults to ``K`` (equal for every channel this
    device produces, but they enter in different places). Returns None for no
    channels, so the caller falls through to ``--drag-beat-GHz``.
    """
    if not specs:
        return None
    from snail_solver.envelope import DragChannel
    out = []
    for text in specs:
        parts = str(text).split(":")
        if not 1 <= len(parts) <= 3:
            raise ValueError(f"--drag-channel wants BEAT[:K[:N]], got {text!r}")
        beat = float(parts[0])
        k = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        n = int(parts[2]) if len(parts) > 2 and parts[2] else max(k, 1)
        out.append(DragChannel(beat, n_pump=k, n_photon=n, quotient_rule=True))
    return out


def _shape_envelope(shape: str = "raised_cosine",
                    shape_kw: Optional[Dict[str, Any]] = None,
                    t_g: float = 2.0):
    """Unit-amplitude envelope on `t_g`, 2 by default so ``t = u + 1`` maps to [0,2].

    Reads the shape in NORMALIZED gate time (every envelope is t_g-independent in u).
    Pass the REAL gate length when derivatives IN PHYSICAL TIME are wanted (recursive
    DRAG): ``SinePowerRamp`` precomputes ``pi/t_rise`` in ``__init__``, so reassigning
    ``.t_g`` afterwards does NOT rescale the ramp.
    """
    from snail_solver.envelope import ENVELOPE_KINDS
    cls = ENVELOPE_KINDS.get(str(shape))
    if cls is None:
        raise ValueError(f"unknown envelope shape {shape!r}; "
                         f"known: {sorted(ENVELOPE_KINDS)}")
    kw = dict(shape_kw or {})
    t_g = float(t_g)
    if cls.__name__ == "SinePowerRamp":
        # rise given as a FRACTION of the gate
        kw = {"m": int(kw.get("m", 3)),
              "t_rise": t_g * float(kw.get("rise_frac", 0.5))}
    return cls(amp=1.0, t_g=t_g, **kw)


def shape_config(config: Dict[str, Any]) -> tuple:
    """``(shape, shape_kw)`` for this device, to pass to the chirp projection."""
    kind = str(config.get("envelope", "raised_cosine"))
    kw = ({"m": int(config.get("envelope_m", 3)),
           "rise_frac": float(config.get("envelope_rise_frac", 0.5))}
          if kind == "sine_power" else {})
    return kind, kw


def cross_check_probe_moments(config: Dict[str, Any], target_eta: float, *,
                              weightings: Sequence[str] = ("rabi", "uniform",
                                                           "coupling"),
                              logger: Optional[logging.Logger] = None,
                              **kw: Any) -> Dict[str, Any]:
    """Pin the moment weighting by measuring the law BOTH ways at the same drive.

    The weightings (``"rabi"`` = ``sin theta(t)``, ``"uniform"``, ``"coupling"``)
    span 2x in ``k2``. At a drive weak enough (``target_eta <~ 0.5``) that the
    CONSTANT probe is clean, its pointwise ``k2/k4`` are ground truth; the right
    convention is the one whose deconvolved shaped fit reproduces them. As run
    (4Gate4.5-like, delta = -100 MHz, target_eta = 0.45, 7 rows): ``"rabi"`` within
    ~4% on ``k2``, ``"coupling"`` 19%, ``"uniform"`` 2.1x. ``M4`` is NOT pinned by so
    short a ladder (k2/k4 ~96% anticorrelated), so read the ``k2`` column.

    ``**kw`` goes to :func:`rabi_shift_table`. Returns ``constant`` (the reference
    fit), ``shaped`` (per weighting: ``k2``, ``k4`` and relative errors) and ``best``
    (smallest ``k2`` error).
    """
    log = logger or logging.getLogger("tune_up")
    ref = rabi_shift_table(config, target_eta, probe_shape="constant",
                           logger=log, **kw)["fit"]
    log.info(f"cross-check: constant probe gives k2={ref['k2']:+.4f} "
             f"k4={ref['k4']:+.4f} (r2={ref['r2']:.4f})")
    # ONE shaped measurement; the weightings differ only in the deconvolution.
    shaped_raw = rabi_shift_table(config, target_eta, probe_shape="gate",
                                  moment_weighting=weightings[0], logger=log,
                                  **kw)["fit"]
    K2, K4 = shaped_raw["K2_measured"], shaped_raw["K4_measured"]
    out: Dict[str, Any] = {}
    for w in weightings:
        M2, M4 = probe_moments(config, w)
        k2, k4 = K2 / M2, K4 / M4
        err2 = abs(k2 - ref["k2"]) / max(abs(ref["k2"]), 1e-12)
        out[str(w)] = {"M2": float(M2), "M4": float(M4), "k2": float(k2),
                       "k4": float(k4), "k2_rel_err": float(err2),
                       "k4_rel_err": float(abs(k4 - ref["k4"])
                                           / max(abs(ref["k4"]), 1e-12))}
        log.info(f"cross-check: {w:9s} M2={M2:.4f} -> k2={k2:+.4f} "
                 f"({100 * err2:+.1f}% vs constant)")
    best = min(out, key=lambda w: out[w]["k2_rel_err"])
    log.info(f"cross-check: best weighting = {best!r} "
             f"({100 * out[best]['k2_rel_err']:.1f}% k2 error)")
    return {"constant": ref, "shaped": out, "best": best,
            "K2_measured": float(K2), "K4_measured": float(K4),
            "target_eta": float(target_eta)}


def probe_moments(config: Dict[str, Any], weighting: str = "rabi") -> tuple:
    """``(M2, M4)`` for THIS device's envelope -- see :func:`stark_chirp.stark_moments`.

    Read off the configured shape, so a shaped probe and the chirp built from its fit
    cannot disagree about which pulse was played.
    """
    from snail_solver.stark_chirp import stark_moments
    shape, shape_kw = shape_config(config)
    env = _shape_envelope(shape, shape_kw)

    def shape_fn(u):
        s = np.abs(np.asarray(env.value_at(np.asarray(u, dtype=float) + 1.0, np),
                              dtype=complex))
        return s ** 2                       # |eta(u)|^2 / eta_peak^2

    return stark_moments(shape_fn, weighting)


def _resolve_drag_channels(beat_GHz: Optional[float], n_pump: int = 1,
                           channels: Optional[Sequence[Any]] = None) -> tuple:
    """Normalize this module's DRAG arguments to a channel tuple, innermost-first.

    Same rule as :meth:`envelope.PumpTone.drag_channels_resolved`: an explicit
    `channels` list wins, else the `beat_GHz`/`n_pump` shorthand, else ``()`` (off).
    """
    from snail_solver.drag import order_channels
    from snail_solver.envelope import DragChannel
    if channels:
        return order_channels(tuple(channels))
    if beat_GHz:
        return (DragChannel(float(beat_GHz), n_pump=int(n_pump)),)
    return ()


def _project(values: np.ndarray, u: np.ndarray, w: np.ndarray,
             degree: int) -> np.ndarray:
    """Gauss-Legendre projection of `values` onto P_0..P_degree, odd terms zeroed.

    Odd coefficients vanish by parity (the envelope is symmetric about mid-gate);
    zeroing them keeps that exact instead of leaving quadrature dust.
    """
    from numpy.polynomial import legendre as L
    c = np.array([(2 * k + 1) / 2.0 * np.sum(w * values * L.legval(u, np.eye(k + 1)[k]))
                  for k in range(int(degree) + 1)])
    c[1::2] = 0.0
    return c


def chirp_from_measured_shift(table: Dict[str, Any], target_eta: Optional[float] = None,
                              degree: int = 8, pin_c0: bool = True,
                              drag_beat_GHz: Optional[float] = None,
                              drag_n_pump: int = 1, t_g: Optional[float] = None,
                              drag_channels: Optional[Sequence[Any]] = None,
                              shape: str = "raised_cosine",
                              shape_kw: Optional[Dict[str, Any]] = None,
                              max_iters: int = 12,
                              tol_GHz: float = 1e-12,
                              quartic_warn: float = 0.25,
                              couple_drag: bool = True,
                              ridge_law: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Step 2: project the measured shift onto the Legendre chirp basis.

    With `ridge_law` (:func:`ridge_chirp.law_from_table`) the shift along the pulse is
    the law read off the measured ridge instead of ``k2|eta|^2 + k4|eta|^4``, and its
    ``delta0`` is the static part; ``quartic_fraction`` is then NaN (no truncated
    series) and ``chirp_source`` is ``"ridge"``.

    Evaluates the fitted shift ``k2 |eta|^2 + k4 |eta|^4`` along the pulse (``|eta(u)|
    = eta* cos^2(pi u / 2)`` for a Hann) and projects delta(u) onto Legendre
    polynomials by Gauss-Legendre quadrature. delta(u) is a polynomial in cos(pi u / 2),
    NOT in u, so the Legendre series never terminates: every `degree` truncates. It
    converges fast (Hann, k2 term alone: peak error ~16% at degree 4, ~0.7% at 8,
    ~4e-5 at 12). The table's ``k2``/``k4`` are already the instantaneous law (a
    constant probe, or a shaped probe deconvolved by :func:`rabi_shift_table`), so no
    deconvolution happens here. ``rel_diff`` compares against
    ``stark_chirp.stark_chirp_seed``, a pure-|eta|^2 HANN shape with the same mean, so
    it also picks up k4, a non-Hann `shape` and DRAG.

    DRAG is a fixed point, not a formula
    -------------------------------------
    DRAG turns the drive into ``eta - i (d eta/dt) / Delta(t)`` (per channel, nested,
    see :func:`drag.apply_drag`), and the Stark shift follows the magnitude of that
    TOTAL drive. But ``Delta(t) = 2 pi beat + (stark_scale - n_pump) delta(t)`` moves
    with the chirp being computed, so with `couple_drag` the two are iterated to a
    fixed point. k2/k4 from the DRAG-free probe still apply (DRAG only adds drive).
    The correction scales as 1/t_g, so the coupled chirp is length-dependent and
    ``run_tune_up``'s outer loop re-solves chirp and length together.
    ``couple_drag=False`` keeps the DRAG-off chirp (see the comment in the loop).

    Returns
    -------
    dict
        ``coeffs_GHz`` (length degree+1, c_0 = 0 when pinned); ``stark_mean_GHz``
        (the unpinned c_0, the pulse-mean Stark shift); ``static_GHz`` (the fit's
        ``delta0``); ``mean_shift_GHz`` (their sum, the carrier offset);
        ``rel_diff``; ``quartic_fraction``; ``perturbative_ok``; ``degree``;
        ``target_eta``; ``measured_eta_max`` and ``extrapolation_ratio`` (target_eta
        over the largest measured |eta|; well above 1 means the chirp's peak is
        extrapolated). With DRAG on also ``drag_iters``, ``drag_coupled``,
        ``min_abs_detuning_GHz`` (and ``_per_channel_GHz``), ``n_drag_channels``,
        ``drag_correction_ratio`` (peak change of |drive| over peak |drive|),
        ``drag_delta_frac`` (relative change in the norm of delta(t) from DRAG; 0 when
        decoupled) and ``neglected_shift_frac`` (the DRAG shift a decoupled chirp
        leaves out, over the peak shift; 0 when coupled).

    ``quartic_fraction = |k4 eta*^4 / k2 eta*^2|`` sizes the last kept term of the
    truncated POWER series in |eta| (not the Legendre series); when not small, the
    unmeasured eta^6 term is plausibly as large. ``perturbative_ok`` flags it at
    `quartic_warn` (0.25 -> next term ~6%). It inherits the fit's instability;
    :func:`shift_curve_stability` is independent.
    """
    from snail_solver import stark_chirp as SC

    fit = (dict(delta0=ridge_law["delta0"]) if ridge_law is not None
           else table["fit"])
    eta_star = float(target_eta if target_eta is not None else table["target_eta"])
    k2, k4 = ((float("nan"), float("nan")) if ridge_law is not None
              else (float(fit["k2"]), float(fit["k4"])))
    # eta_star may exceed the measured window: flag the extrapolation.
    measured_eta = table.get("eta")
    measured_eta_max = (float(np.nanmax(np.asarray(measured_eta, dtype=float)))
                        if measured_eta is not None else float("nan"))
    extrapolation_ratio = (eta_star / measured_eta_max
                           if measured_eta_max > 0 else float("nan"))

    n_quad = max(2 * int(degree) + 8, 32)
    u, w = np.polynomial.legendre.leggauss(n_quad)
    # |eta(u)| / eta*, read off the actual envelope (cos^2(pi u / 2) only for Hann),
    # at t_g = 2 so u = t - 1.
    shape_env = _shape_envelope(shape, shape_kw)
    s = np.abs(np.asarray(shape_env.value_at(u + 1.0, np), dtype=complex))
    amp = eta_star * s

    def shift_GHz(a):                                      # MHz law -> GHz
        if ridge_law is not None:
            from snail_solver.ridge_chirp import shift_MHz
            return np.asarray(shift_MHz(ridge_law, a), dtype=float) * 1e-3
        return (k2 * a ** 2 + k4 * a ** 4) * 1e-3

    delta_GHz = shift_GHz(amp)
    base_norm = float(np.linalg.norm(delta_GHz))
    extra: Dict[str, Any] = {}

    channels = _resolve_drag_channels(drag_beat_GHz, drag_n_pump, drag_channels)
    if channels:
        if t_g is None:
            raise ValueError("chirp_from_measured_shift needs t_g when DRAG is on: "
                             "the quadrature is (d eta/dt)/Delta(t) and so scales as "
                             "1/t_g, which breaks the length-independence of the chirp")
        from snail_solver import drag as _drag
        from snail_solver.envelope import Chirp, PumpTone
        t_g = float(t_g)
        # The base envelope built AT the real t_g, so its jet (eta, eta', ...; bound
        # to `shape` below) is in physical time (see `_shape_envelope`). abs() of the
        # DRAG'd drive is the general form: recursive DRAG's correction has a real
        # part (-eta''/(Da Db)), so sqrt(amp^2 + q^2) holds only for one channel.
        env = _shape_envelope(shape, shape_kw, t_g)
        env.amp = eta_star
        order = _drag.required_order(channels)
        shape = env.jet_at(t_g * (u + 1.0) / 2.0, order, np)
        min_abs = float("inf")
        # couple_drag=False is ONE-SHOT mode, the first Picard iterate: the chirp is
        # built from the bare envelope and DRAG (whose Delta(t) sees that chirp) is
        # applied on top without feeding back. It cannot diverge (no chirp chasing its
        # own denominator), the chirp equals the DRAG-off chirp (a clean ablation), and
        # it stays length-independent. The Stark shift of the added correction, which
        # the chirp then leaves un-cancelled, is reported as `neglected_shift_frac`.
        iters = int(max_iters) if couple_drag else 1
        for it in range(iters):
            # The iterate is delta(t) on the quadrature nodes; each pass re-projects
            # it to a Legendre chirp (the detuning jet needs d/dt of the chirp) at
            # degree >= 24, above the emitted `degree`: truncation ripple near
            # |u| = 1 lands in a physical denominator and can flip the sign of
            # Delta - Delta_0 on a 50 MHz beat.
            chirp = Chirp(_project(delta_GHz, u, w, max(int(degree), 24)), t_g)
            tone = PumpTone(w_p_GHz=0.0, envelope=env, chirp=chirp,
                            drag_channels=list(channels))
            jets = [tone.channel_detuning_jet(c, t_g * (u + 1.0) / 2.0, order, np)
                    for c in channels]
            floors = [float(np.min(np.abs(j[0])) / TWO_PI) for j in jets]
            min_abs = min(floors)
            eta_tot = np.abs(_drag.apply_drag(shape, jets, channels, np))
            new = shift_GHz(eta_tot)
            step = float(np.max(np.abs(new - delta_GHz)))
            if not couple_drag:          # keep the bare chirp; `new` only reports
                break
            delta_GHz = new
            if step < tol_GHz:
                break
        else:
            raise DragFixedPointDiverged(
                f"the chirp<->DRAG fixed point did not settle in {max_iters} passes "
                f"(last step {step:.2e} GHz, min|Delta(t)| = {min_abs * 1e3:.3f} MHz, "
                f"{len(channels)} channel(s)). Near a collision the quadrature and the "
                f"chirp can chase each other; pick a further-detuned beat or a weaker "
                f"drive. Note the d-th nested correction scales as 1/t_g^d, so a deeper "
                f"recursion couples the chirp and the length more tightly and may need "
                f"more passes.", min_abs)
        drag_norm = float(np.linalg.norm(delta_GHz))
        extra = {"drag_iters": it + 1, "min_abs_detuning_GHz": min_abs,
                 "drag_coupled": bool(couple_drag),
                 "neglected_shift_frac": (
                     float(np.max(np.abs(new - delta_GHz))
                           / max(float(np.max(np.abs(delta_GHz))), 1e-30))
                     if not couple_drag else 0.0),
                 "min_abs_detuning_per_channel_GHz": floors,
                 "n_drag_channels": len(channels),
                 "drag_correction_ratio": float(
                     np.max(np.abs(eta_tot - amp)) / max(float(np.max(amp)), 1e-30)),
                 "drag_delta_frac": ((drag_norm - base_norm) / base_norm
                                     if base_norm else float("nan"))}

    coeffs = _project(delta_GHz, u, w, degree)
    stark_mean_GHz = float(coeffs[0])
    # The fit's static delta0 is a carrier retune: added to the offset, never the
    # chirp. The pulse-mean c_0 goes to the offset too, and stays in the chirp as
    # well unless `pin_c0`.
    mean_shift_GHz = stark_mean_GHz + float(fit.get("delta0", 0.0)) * 1e-3
    if pin_c0:
        coeffs[0] = 0.0

    analytic = SC.stark_chirp_seed(stark_mean_GHz, degree=degree, pin_c0=pin_c0)
    denom = float(np.linalg.norm(analytic))
    rel_diff = float(np.linalg.norm(coeffs - analytic) / denom) if denom else float("nan")
    if ridge_law is not None:
        quartic = float("nan")                  # no truncated series to judge
    else:
        quartic = (abs(k4 * eta_star ** 4) / abs(k2 * eta_star ** 2)
                   if k2 else float("inf"))
    return {"coeffs_GHz": coeffs, "mean_shift_GHz": mean_shift_GHz,
            "chirp_source": "ridge" if ridge_law is not None else "law",
            "stark_mean_GHz": stark_mean_GHz,
            "static_GHz": float(fit.get("delta0", 0.0)) * 1e-3,
            "rel_diff": rel_diff, "quartic_fraction": float(quartic),
            "perturbative_ok": bool(ridge_law is not None
                                    or quartic < quartic_warn), **extra,
            "degree": int(degree), "target_eta": eta_star,
            "measured_eta_max": measured_eta_max,
            "extrapolation_ratio": extrapolation_ratio}


# ===========================================================================
# Seeing what was fitted
# ===========================================================================
#: Palette colours fixed by ROLE in every panel: Lorentzian blue, parabolic vertex
#: orange, ink grey, leakage aqua.
_C_LORENTZ = "#2a78d6"
_C_VERTEX = "#eb6834"
_C_INK = "#52514e"
_C_LEAK = "#1baf7a"


def _pyplot():
    """Headless pyplot with the literature style applied when available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:                                        # style is a nicety
        pass
    return plt


def _savefig(fig, out: str) -> str:
    """Write `fig` to `out` (creating its directory), close it, return `out`."""
    import matplotlib.pyplot as plt
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rabi_table(table: Dict[str, Any], out: str = "figs/rabi_chevrons.png",
                    title: Optional[str] = None) -> str:
    """Render the Rabi sweep: every chevron, its envelope fit, and the shift curve.

    One row per drive strength: the raw chevron; the leakage raster
    (``P_leak = norm - P01 - P10``) on the same grid; the Rabi oscillation on resonance
    and one HWHM off (if the on-resonance trace doesn't reach 1 and come back, there
    is no well-defined centre); and the max-over-time envelope with its Lorentzian.
    Centre and parabolic vertex are drawn together because their disagreement is the
    diagnostic of a non-two-level lineshape. Bottom band: resonance vs |eta| with the
    fitted law, and leakage vs |eta|.

    Accepts the partial table carried by :class:`RabiFitError`. Returns the path.
    """
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    chevrons = list(table.get("chevrons", []))
    if not chevrons:
        raise ValueError("no chevrons to plot")
    eta = np.asarray(table["eta"], dtype=float)
    ridge = np.asarray(table["delta_MHz"], dtype=float)
    leakage = np.asarray(table.get("leakage", np.full(eta.size, np.nan)), dtype=float)
    fit = table.get("fit")
    n = len(chevrons)
    # Single-hue ramp for leakage rasters: color follows the entity (leakage is
    # always _C_LEAK), rather than a second multi-hue map competing with viridis.
    leak_cmap = LinearSegmentedColormap.from_list("leak", ["#ffffff", _C_LEAK])

    fig = plt.figure(figsize=(21.0, 2.9 * n + 3.4), layout="constrained")
    gs = fig.add_gridspec(n + 1, 4, width_ratios=[1.0, 1.0, 0.95, 1.1],
                          height_ratios=[2.9] * n + [3.4])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)]
                     for r in range(n)])
    for i, ch in enumerate(chevrons):
        ax0, axL, axt, ax1 = axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 3]
        off = np.asarray(ch["offsets_GHz"], dtype=float) * 1e3
        m = np.asarray(ch["metric"], dtype=float)
        cen, vtx = ch["fit"]["center_GHz"] * 1e3, ch["fit"]["vertex_GHz"] * 1e3
        dropped = ch.get("dropped")
        leak_env = np.asarray(ch.get("leak_at_metric", np.full_like(off, np.nan)),
                              dtype=float)

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

        # -- where the population that ISN'T P10 actually is ---------------------
        if "P_leak" in ch:
            meshL = axL.pcolormesh(off, np.asarray(ch["times_ns"], dtype=float),
                                   np.asarray(ch["P_leak"], dtype=float).T,
                                   shading="auto", cmap=leak_cmap, vmin=0.0, vmax=1.0)
            fig.colorbar(meshL, ax=axL, label=r"$P_{leak}$", pad=0.02)
        axL.axvline(cen, color=_C_LORENTZ, ls="--", lw=1.6)
        lb = ch.get("leak_breakdown", {})
        leak_row = leakage[i] if i < leakage.size else float("nan")
        axL.set_title(rf"leak {leak_row:.2f} at centre "
                      rf"(coupler {lb.get('coupler', float('nan')):.2f}, "
                      rf"$|f\rangle$ {lb.get('f_a', 0.0) + lb.get('f_b', 0.0):.2f}, "
                      rf"$|11\rangle$ {lb.get('double', float('nan')):.2f})", fontsize=8)

        # -- the oscillation itself, on resonance and one linewidth away ---------
        if "P10" in ch:
            P = np.asarray(ch["P10"], dtype=float)
            P01 = np.asarray(ch.get("P01", np.full_like(P, np.nan)), dtype=float)
            P_leak_full = np.asarray(ch.get("P_leak", np.full_like(P, np.nan)), dtype=float)
            ts = np.asarray(ch["times_ns"], dtype=float)
            hw = ch["fit"].get("hwhm_GHz", np.nan) * 1e3
            j_on = int(np.argmin(np.abs(off - cen)))
            axt.plot(ts, P[j_on], "-", lw=2.0, color=_C_LORENTZ,
                     label=rf"on resonance ({off[j_on]:+.2f} MHz)")
            axt.plot(ts, P01[j_on], "-", lw=1.4, color=_C_INK, alpha=0.6,
                     label=r"$P(|01\rangle)$")
            axt.plot(ts, P_leak_full[j_on], "--", lw=1.4, color=_C_LEAK,
                     label=r"$P_{leak}$")
            if np.isfinite(hw):
                j_off = int(np.argmin(np.abs(off - (cen + hw))))
                if j_off != j_on:
                    axt.plot(ts, P[j_off], "-", lw=1.6, color=_C_VERTEX, alpha=0.85,
                             label=rf"+1 HWHM ({off[j_off]:+.2f} MHz)")
            axt.axhline(1.0, color=_C_INK, ls=":", lw=1.0)
            axt.set_ylim(-0.03, 1.22)      # headroom so the legend clears the trace
            axt.set_ylabel(r"$P(|10\rangle)$")
            axt.set_title(rf"Rabi oscillation, peak {P[j_on].max():.3f}", fontsize=9)
            axt.legend(fontsize=6.5, framealpha=0.95, loc="upper right",
                       ncol=2, borderaxespad=0.3)
            axt.grid(alpha=0.25)

        ax1.plot(off, m, "o", ms=4.5, color=_C_INK, label="max-over-time $P(|10\\rangle)$")
        if np.any(np.isfinite(leak_env)):
            ax1.plot(off, leak_env, "--", lw=1.4, color=_C_LEAK, label=r"leak at metric")
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
        q = ch.get("quality", {})
        note = (f"contrast {np.nanmax(m) - np.nanmin(m):.2f}   estimators differ "
               f"{gap:.2f} MHz   leak {leakage[i] if i < leakage.size else float('nan'):.2f}"
               f"   quality {q.get('weight', float('nan')):.2f}")
        if dropped:
            note += f"   DROPPED ({dropped})"
        ax1.set_title(note, fontsize=8.5,
                      color=("#b3261e" if dropped else _C_INK))
        ax1.legend(fontsize=6.5, framealpha=0.95, loc="upper left",
                   borderaxespad=0.4)
        ax1.grid(alpha=0.25)
        if i == n - 1:
            for ax in (ax0, axL, ax1):
                ax.set_xlabel(r"pump offset from $|\omega_b-\omega_a|$ (MHz)")
            axt.set_xlabel("time (ns)")

    # -- the result: resonance vs drive (left) and leakage vs drive (right) -----
    axr = fig.add_subplot(gs[n, :2])
    axLr = fig.add_subplot(gs[n, 2:])
    ok = np.isfinite(ridge)
    axr.plot(eta[ok], ridge[ok], "o", ms=7, color=_C_LORENTZ, label="located resonance")
    if (~ok).any():
        reasons = sorted({ch.get("dropped") for ch in chevrons if ch.get("dropped")})
        axr.plot(eta[~ok], np.zeros((~ok).sum()), "x", ms=9, color=_C_VERTEX,
                 label=f"dropped ({', '.join(reasons)})" if reasons else "dropped")
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

    axLr.plot(eta, leakage, "o-", ms=7, lw=1.4, color=_C_LEAK, label="leak at resonance")
    stability = table.get("stability")
    if stability and np.isfinite(stability.get("delta_spread", np.nan)):
        axLr.set_title(f"leakage vs drive   (fit stability spread "
                       f"{stability['delta_spread']:.2f})", fontsize=10)
    else:
        axLr.set_title("leakage vs drive", fontsize=10)
    axLr.axvline(float(table["target_eta"]), color=_C_INK, ls=":", lw=1.4,
                label=f"target_eta={float(table['target_eta']):.2f}")
    axLr.set_xlabel(r"drive strength $|\eta|$")
    axLr.set_ylabel(r"$P_{leak}$ at the located resonance")
    axLr.set_ylim(-0.03, 1.03)
    axLr.legend(fontsize=8, framealpha=0.9)
    axLr.grid(alpha=0.25)

    fig.suptitle(title or (rf"Rabi sweep: resonance vs drive, "
                           rf"$\eta^*$ = {table['target_eta']:.2f}"), fontsize=12)
    return _savefig(fig, out)


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Quad edges bracketing `centers`, for a `pcolormesh` with `shading="flat"`.

    Midpoints inside, half a step beyond each end; the ridge map stacks rows on
    DIFFERENT offset grids one quad row at a time. A single point gets a unit cell.
    """
    c = np.asarray(centers, dtype=float)
    if c.size == 0:
        return np.zeros(0, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid,
                           [c[-1] + (c[-1] - mid[-1])]])


def _law_at(fit: Dict[str, Any], eta: np.ndarray) -> Optional[np.ndarray]:
    """``delta(|eta|)`` from a stored fit, or None when there is no law.

    Sums whatever ``k<n>`` powers the fit carries. None rather than raising: a
    FAILED column has no law but its ridge is still worth drawing.
    """
    if not fit:
        return None
    terms = [(int(k[1:]), float(v)) for k, v in fit.items()
             if isinstance(k, str) and len(k) > 1 and k[0] == "k"
             and k[1:].isdigit() and v is not None]
    if not terms:
        return None
    out = np.full_like(np.asarray(eta, dtype=float),
                       float(fit.get("delta0") or 0.0))
    for power, coeff in terms:
        out = out + coeff * np.asarray(eta, dtype=float) ** power
    return out


def plot_chirp_ridge(table: Dict[str, Any], proj: Dict[str, Any], wp_offset_GHz: float,
                     t_g: float, out: str = "figs/chirp_ridge.png",
                     title: Optional[str] = None,
                     shape: Optional[str] = None,
                     shape_kw: Optional[Dict[str, Any]] = None,
                     ridge_law: Optional[Dict[str, Any]] = None) -> str:
    """Overlay the chirp's pump trajectory on the Rabi map itself, not just its fit.

    `shape`/`shape_kw` (``shape_config(config)``) map time to drive along the pulse
    actually played; omitted, a raised cosine is assumed, which misplaces the chirp
    vertically on any other envelope. `ridge_law` (``stages.ridge_law``) supplies the
    ridge the chirp was built from, including rows the tracker recovered (drawn red);
    a replayed table carries no ``delta_MHz`` of its own.

    Qiu et al. 2023 (arXiv:2306.10162) Fig. 4b: pump frequency on x, drive on y, the
    Rabi metric as colour, with the chirp's instantaneous frequency riding the
    Stark-shifted ridge and a flat carrier at the same mean offset for contrast.

    NO INTERPOLATION: each row is drawn at its own measured offsets, so per-row
    adaptive spans render fine; resolution comes from more ``amp_points``/
    ``wp_points``. A table with no fitted law (a failed column) still draws, and
    rejected rows are marked. Past `measured_eta_max` (horizontal line) the ridge is
    extrapolated. This is a DEFINITIONAL self-consistency check; the independent one
    is the shaped-chevron residual in `run_tune_up` or ``chirp_ablation``.

    Returns the path written.
    """
    from snail_solver.envelope import Chirp, RaisedCosine
    plt = _pyplot()

    chevrons = list(table.get("chevrons", []))
    if not chevrons:
        raise ValueError("no chevrons to plot")
    eta = np.asarray(table["eta"], dtype=float)
    ridge = np.asarray(table.get("delta_MHz", np.full(eta.shape, np.nan)),
                       dtype=float)
    status = None
    if ridge_law is not None and ridge_law.get("ridge_MHz") is not None:
        eta = np.asarray(ridge_law["ridge_eta"], dtype=float)
        ridge = np.asarray(ridge_law["ridge_MHz"], dtype=float)
        status = list(ridge_law.get("ridge_status") or [])
    fit = table.get("fit") or {}
    target_eta = float(proj["target_eta"])
    measured_eta_max = float(proj.get("measured_eta_max", np.nanmax(eta)))

    row_eta = np.array([float(c["eta"]) for c in chevrons])
    row_off = [np.asarray(c["offsets_GHz"], dtype=float) * 1e3 for c in chevrons]
    row_met = [np.asarray(c["metric"], dtype=float) for c in chevrons]
    dropped = [c.get("dropped") for c in chevrons]
    if status is not None and len(status) == len(chevrons):
        dropped = [None if st in ("gated", "tracked") else st for st in status]

    ts = np.linspace(0.0, float(t_g), 400)
    if shape is not None:
        _env = _shape_envelope(shape, shape_kw, float(t_g))
        _env.amp = target_eta
        eta_t = np.abs(np.asarray(_env.value_at(ts, np), dtype=complex))
    else:
        eta_t = np.asarray(RaisedCosine(target_eta, float(t_g)).value_at(ts))
    chirp_MHz = 1e3 * (float(wp_offset_GHz)
                       + np.asarray(Chirp(proj["coeffs_GHz"], float(t_g)).detuning(ts))
                       / TWO_PI)
    flat_MHz = np.full_like(ts, 1e3 * float(wp_offset_GHz))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    # One quad row per measured row on its OWN offset axis (edges, not centres).
    y_edges = _cell_edges(row_eta)
    mesh = None
    for i, (off, met) in enumerate(zip(row_off, row_met)):
        mesh = ax.pcolormesh(_cell_edges(off), y_edges[i:i + 2],
                            met[None, :], shading="flat", cmap="viridis",
                            vmin=0.0, vmax=1.0)
    fig.colorbar(mesh, ax=ax, label=r"$P(|10\rangle)$ (max over time)", pad=0.02)

    ys = np.linspace(float(row_eta.min()), float(max(row_eta.max(), target_eta)) * 1.02,
                     300)
    if target_eta > measured_eta_max:
        ax.axhline(measured_eta_max, color=_C_VERTEX, ls=":", lw=1.6,
                  label=f"measured up to |eta|={measured_eta_max:.2f}")
    law = _law_at(fit, ys)
    if law is not None:
        ax.plot(law, ys, "-", lw=1.4, color="white", alpha=0.85,
               label="fitted ridge")
    _trk = (np.array([st == "tracked" for st in status]) if status is not None
            and len(status) == len(ridge) else np.zeros(ridge.shape, bool))
    ax.plot(ridge[~_trk], eta[~_trk], "o", ms=6, color="white", mec=_C_INK, mew=1.0,
           label="measured ridge")
    if _trk.any():
        ax.plot(ridge[_trk], eta[_trk], "o", ms=6, color="#e34948", mec=_C_INK,
               mew=1.0, label="ridge, tracked through a rejected row")
    drop_eta = [e for e, d in zip(row_eta, dropped) if d]
    if drop_eta:
        ax.plot([0.0] * len(drop_eta), drop_eta, "x", ms=7, mew=1.6,
               color=_C_VERTEX, transform=ax.get_yaxis_transform(),
               clip_on=False,
               label=f"dropped ({', '.join(sorted({str(d) for d in dropped if d}))})")
    ax.plot(chirp_MHz, eta_t, "-", lw=2.6, color=_C_LORENTZ,
           label="chirped pump (rides the ridge)")
    ax.plot(flat_MHz, eta_t, "--", lw=2.0, color=_C_VERTEX,
           label="flat carrier (same mean offset)")
    ax.set_ylim(y_edges.min(), ys.max())
    ax.set_xlim(min(o.min() for o in row_off), max(o.max() for o in row_off))
    ax.set_ylabel(r"drive strength $|\eta|$ (the amplitude/voltage axis)")
    ax.set_xlabel("pump frequency offset (MHz)")
    ax.set_title(title or (rf"Rabi map with the chirp riding the ridge, "
                          rf"$\eta^*$ = {target_eta:.2f}"), fontsize=11)
    ax.legend(fontsize=7.5, framealpha=0.9, loc="best")
    return _savefig(fig, out)


# ===========================================================================
# Post-chirp validation: does the calibrated chirp actually help, on the real
# shaped gate, and up to what drive?
# ===========================================================================
def post_chirp_table(config: Dict[str, Any], record: Dict[str, Any], *,
                     eta_lo: float = 0.5, eta_hi: float = 1.0, amp_points: int = 7,
                     wp_span_MHz: Optional[float] = None, wp_points: int = 25,
                     span_linewidths: float = 4.0, n_time: int = 161,
                     compare_flat: bool = True, reproject_chirp: bool = False,
                     rabi_table: Optional[Dict[str, Any]] = None,
                     chirp_degree: int = 8, jobs: int = 0,
                     solver: Optional[Dict[str, Any]] = None,
                     logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Re-run the chevron with the SHAPED, CHIRPED gate across drive strengths.

    :func:`rabi_shift_table` characterises the device; this measures the pulse. One
    chevron per ``|eta|`` with the real raised-cosine gate and calibrated chirp, and
    (with `compare_flat`) the same gate unchirped -- ``chirp_ablation`` generalised
    across drive and pump offset.

    Each row's length is RESCALED to a full swap at its own drive,
    ``t_g_i = nominal_t_g(e) * (record["t_g_ns"] / nominal_t_g(target_eta))``, so a
    weaker row is not a partial rotation by construction (the fixed-length confound).

    Parameters
    ----------
    config : dict
        Merged device configuration.
    record : dict
        A tune-up ``operating_point`` (``target_eta``, ``t_g_ns``, ``wp_offset_GHz``,
        ``chirp_coeffs_GHz``, ``spec_abs_GHz``, ``drag_beat_GHz``, ``drag_n_pump``).
    eta_lo, eta_hi, amp_points : float, float, int
        Drive-strength row sweep, as fractions of ``record["target_eta"]``.
    wp_span_MHz, span_linewidths, wp_points : as in :func:`rabi_shift_table`.
    compare_flat : bool, default True
        Also run the identical gate with ``chirp_coeffs_GHz=[]``.
    reproject_chirp : bool, default False
        False: the recorded chirp on every row (the gate that will run). True:
        re-derive it at each row's drive (needs `rabi_table`) -- "would the
        procedure work at this drive" rather than "does the calibrated gate".
    rabi_table : dict, optional
        Required when `reproject_chirp` is True; the :func:`rabi_shift_table`
        result the chirp was originally built from.
    jobs, solver : as in :func:`rabi_shift_table`.

    Returns
    -------
    dict
        ``eta``, ``residual_MHz`` (located resonance minus ``wp_offset`` --
        should be ~0 where the chirp is valid), ``transfer_chirped``,
        ``transfer_flat``, ``leak_chirped``, ``leak_flat`` (all ``[n_rows]``),
        ``record``, ``compare_flat``, ``reproject_chirp``, ``rows`` (per-row
        detail for :func:`plot_post_chirp_table`).
    """
    from snail_solver import find_stark_resonance as FSR

    target_eta = float(record["target_eta"])
    t_g_star = float(record["t_g_ns"])
    t_g0_at_target = nominal_t_g(config, target_eta)
    wp_offset = float(record["wp_offset_GHz"])
    chirp_star = [float(c) for c in record["chirp_coeffs_GHz"]]
    spec_abs_GHz = record.get("spec_abs_GHz")
    drag_beat_GHz = record.get("drag_beat_GHz")
    drag_n_pump = int(record.get("drag_n_pump") or 1)

    if reproject_chirp and rabi_table is None:
        raise ValueError("reproject_chirp=True needs rabi_table (the "
                         "rabi_shift_table result the chirp was built from)")

    eta = np.linspace(float(eta_lo), float(eta_hi), int(amp_points)) * target_eta
    transfer_chirped = np.full(eta.size, np.nan)
    transfer_flat = np.full(eta.size, np.nan)
    leak_chirped = np.full(eta.size, np.nan)
    leak_flat = np.full(eta.size, np.nan)
    residual_MHz = np.full(eta.size, np.nan)
    rows = []

    for i, e in enumerate(eta):
        t_g_i = float(nominal_t_g(config, float(e)) * (t_g_star / t_g0_at_target))
        amp_scale_i = fixed_eta_amp_scale(config, t_g_i, float(e))

        if reproject_chirp:
            chirp_i = [float(c) for c in chirp_from_measured_shift(
                rabi_table, float(e), degree=chirp_degree,
                drag_beat_GHz=drag_beat_GHz, drag_n_pump=drag_n_pump,
                t_g=t_g_i)["coeffs_GHz"]]
        else:
            chirp_i = chirp_star

        span = (float(wp_span_MHz) if wp_span_MHz is not None
                else 2.0 * float(span_linewidths) * 1e3 / (2.0 * t_g_i))
        offs = np.linspace(-span / 2e3, span / 2e3, int(wp_points)) + wp_offset
        j_wp = int(np.argmin(np.abs(offs - wp_offset)))

        common = dict(solver=solver, n_jobs=jobs, spec_abs_GHz=spec_abs_GHz,
                     shape="raised_cosine", drag_beat_GHz=drag_beat_GHz,
                     drag_n_pump=drag_n_pump, keep_full_channels=True)

        def _measure(chirp_coeffs):
            chev = FSR.scan(config, t_g_i, amp_scale_i, offs, 1.05 * t_g_i,
                            int(n_time), chirp_coeffs_GHz=chirp_coeffs, **common)
            m = np.asarray(chev["resonance_metric"], dtype=float)
            cen = fit_chevron_center(chev["offsets_GHz"], m)
            q = chevron_quality(cen, chev["offsets_GHz"], m, span,
                                leak=chev["leak_on_resonance"])
            return {"metric": m, "P10": chev["P10"], "P_leak": chev["P_leak"],
                    "leak_at_metric": chev["leak_at_metric"], "fit": cen, "quality": q,
                    "transfer_at_wp_offset": float(m[j_wp]),
                    "leak_at_wp_offset": float(chev["leak_at_metric"][j_wp])}

        chirped = _measure(chirp_i)
        transfer_chirped[i] = chirped["transfer_at_wp_offset"]
        leak_chirped[i] = chirped["leak_at_wp_offset"]
        residual_MHz[i] = (chirped["fit"]["center_GHz"] - wp_offset) * 1e3

        row = {"eta": float(e), "t_g_ns": t_g_i, "amp_scale": float(amp_scale_i),
              "chirp_GHz": chirp_i, "offsets_GHz": offs, "chirped": chirped}
        if compare_flat:
            flat = _measure([])
            transfer_flat[i] = flat["transfer_at_wp_offset"]
            leak_flat[i] = flat["leak_at_wp_offset"]
            row["flat"] = flat

        rows.append(row)
        if logger:
            msg = (f"  post-chirp row {i + 1}/{eta.size}: |eta|={e:.4f} "
                  f"(t_g={t_g_i:.2f} ns) -> transfer(chirped)="
                  f"{transfer_chirped[i]:.4f}  leak={leak_chirped[i]:.4f}  "
                  f"residual={residual_MHz[i]:+.3f} MHz")
            if compare_flat:
                msg += (f"  |  transfer(flat)={transfer_flat[i]:.4f}  "
                       f"leak={leak_flat[i]:.4f}")
            logger.info(msg)

    return {"eta": eta, "residual_MHz": residual_MHz,
            "transfer_chirped": transfer_chirped, "transfer_flat": transfer_flat,
            "leak_chirped": leak_chirped, "leak_flat": leak_flat,
            "record": record, "compare_flat": compare_flat,
            "reproject_chirp": reproject_chirp, "rows": rows}


def plot_post_chirp_table(post: Dict[str, Any], out: str = "figs/post_chirp_chevrons.png",
                          title: Optional[str] = None,
                          rabi_table: Optional[Dict[str, Any]] = None) -> str:
    """Render :func:`post_chirp_table`: does the calibrated chirp actually help?

    Per drive row: chirped raster; flat raster (same scale, A/B); envelopes with
    leakage; time trace at ``wp_offset``. Bottom band: transfer, leakage and
    resonance residual vs ``|eta|`` (residual ~0 where the chirp is valid), with the
    constant-probe ridge overlaid if `rabi_table` is given. Returns the path.
    """
    plt = _pyplot()

    rows = list(post.get("rows", []))
    if not rows:
        raise ValueError("no rows to plot")
    eta = np.asarray(post["eta"], dtype=float)
    compare_flat = bool(post.get("compare_flat", False))
    n = len(rows)
    wp_MHz = float(post["record"]["wp_offset_GHz"]) * 1e3

    ncols = 4 if compare_flat else 3
    fig = plt.figure(figsize=(5.2 * ncols, 2.9 * n + 3.4), layout="constrained")
    gs = fig.add_gridspec(n + 1, ncols, height_ratios=[2.9] * n + [3.4])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)]
                     for r in range(n)])

    for i, row in enumerate(rows):
        off = np.asarray(row["offsets_GHz"], dtype=float) * 1e3
        chirped, flat = row["chirped"], row.get("flat")
        show_flat = compare_flat and flat is not None
        col = 0
        ax_c = axes[i, col]; col += 1
        mesh = ax_c.pcolormesh(off, np.arange(chirped["P10"].shape[1]),
                               np.asarray(chirped["P10"], dtype=float).T,
                               shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        fig.colorbar(mesh, ax=ax_c, label=r"$P(|10\rangle)$ (chirped)", pad=0.02)
        ax_c.axvline(wp_MHz, color=_C_LORENTZ,
                    ls="--", lw=1.6)
        ax_c.set_title(rf"$|\eta|$ = {row['eta']:.3f}  chirped", fontsize=9)
        ax_c.set_ylabel("time index")

        if show_flat:
            ax_f = axes[i, col]; col += 1
            meshf = ax_f.pcolormesh(off, np.arange(flat["P10"].shape[1]),
                                    np.asarray(flat["P10"], dtype=float).T,
                                    shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            fig.colorbar(meshf, ax=ax_f, label=r"$P(|10\rangle)$ (flat)", pad=0.02)
            ax_f.axvline(wp_MHz, color=_C_VERTEX,
                        ls="--", lw=1.6)
            ax_f.set_title(rf"$|\eta|$ = {row['eta']:.3f}  flat carrier", fontsize=9)

        ax_e = axes[i, col]; col += 1
        ax_e.plot(off, chirped["metric"], "-", lw=2.0, color=_C_LORENTZ,
                 label="chirped")
        ax_e.plot(off, chirped["leak_at_metric"], "--", lw=1.2, color=_C_LEAK, alpha=0.8,
                 label="leak (chirped)")
        if show_flat:
            ax_e.plot(off, flat["metric"], "-", lw=1.6, color=_C_VERTEX, label="flat")
            ax_e.plot(off, flat["leak_at_metric"], ":", lw=1.2, color=_C_LEAK, alpha=0.5,
                     label="leak (flat)")
        ax_e.axvline(wp_MHz, color=_C_INK,
                    ls=":", lw=1.4, label="wp_offset")
        ax_e.set_ylim(-0.03, 1.05)
        ax_e.set_title(rf"P($t_g$) at wp_offset: {chirped['transfer_at_wp_offset']:.3f}"
                       + (rf" vs {flat['transfer_at_wp_offset']:.3f}"
                          if show_flat else ""), fontsize=8.5)
        ax_e.legend(fontsize=6.5, framealpha=0.9, loc="upper left")
        ax_e.grid(alpha=0.25)

        ax_t = axes[i, col]
        j_wp = int(np.argmin(np.abs(off - wp_MHz)))
        ax_t.plot(np.asarray(chirped["P10"])[j_wp], "-", lw=2.0, color=_C_LORENTZ,
                 label="chirped")
        ax_t.plot(np.asarray(chirped["P_leak"])[j_wp], "--", lw=1.2, color=_C_LEAK,
                 alpha=0.8, label="leak (chirped)")
        if show_flat:
            ax_t.plot(np.asarray(flat["P10"])[j_wp], "-", lw=1.6, color=_C_VERTEX,
                     label="flat")
            ax_t.plot(np.asarray(flat["P_leak"])[j_wp], ":", lw=1.2, color=_C_LEAK,
                     alpha=0.5, label="leak (flat)")
        ax_t.set_ylim(-0.03, 1.05)
        ax_t.set_title("time trace at wp_offset", fontsize=8.5)
        ax_t.legend(fontsize=6.5, framealpha=0.9, loc="upper left")
        ax_t.grid(alpha=0.25)
        if i == n - 1:
            ax_c.set_xlabel(r"pump offset from $\omega_b-\omega_a$ (MHz)")
            if show_flat:
                axes[i, 1].set_xlabel(r"pump offset from $\omega_b-\omega_a$ (MHz)")
            ax_e.set_xlabel(r"pump offset (MHz)")
            ax_t.set_xlabel("time index")

    # -- the money plots: transfer / leakage / residual vs drive -----------------
    gsb = gs[n, :].subgridspec(1, 3)
    ax_tr = fig.add_subplot(gsb[0, 0])
    ax_lk = fig.add_subplot(gsb[0, 1])
    ax_rs = fig.add_subplot(gsb[0, 2])

    transfer_chirped = np.asarray(post["transfer_chirped"], dtype=float)
    transfer_flat = np.asarray(post["transfer_flat"], dtype=float)
    leak_chirped = np.asarray(post["leak_chirped"], dtype=float)
    leak_flat = np.asarray(post["leak_flat"], dtype=float)
    residual_MHz = np.asarray(post["residual_MHz"], dtype=float)
    target_eta = float(post["record"]["target_eta"])

    ax_tr.plot(eta, transfer_chirped, "o-", ms=6, lw=1.4, color=_C_LORENTZ,
              label="chirped")
    if compare_flat:
        ax_tr.plot(eta, transfer_flat, "s--", ms=5, lw=1.4, color=_C_VERTEX,
                  label="flat carrier")
    ax_tr.axvline(target_eta, color=_C_INK, ls=":", lw=1.4)
    ax_tr.set_xlabel(r"drive strength $|\eta|$")
    ax_tr.set_ylabel(r"$P(|10\rangle)$ at wp_offset")
    ax_tr.set_title("does the chirp help?", fontsize=10)
    ax_tr.legend(fontsize=8, framealpha=0.9)
    ax_tr.grid(alpha=0.25)

    ax_lk.plot(eta, leak_chirped, "o-", ms=6, lw=1.4, color=_C_LEAK, label="chirped")
    if compare_flat:
        ax_lk.plot(eta, leak_flat, "s--", ms=5, lw=1.4, color=_C_INK, alpha=0.7,
                  label="flat carrier")
    ax_lk.axvline(target_eta, color=_C_INK, ls=":", lw=1.4)
    ax_lk.set_xlabel(r"drive strength $|\eta|$")
    ax_lk.set_ylabel(r"$P_{leak}$ at wp_offset")
    ax_lk.set_title("leakage vs drive", fontsize=10)
    ax_lk.legend(fontsize=8, framealpha=0.9)
    ax_lk.grid(alpha=0.25)

    ax_rs.plot(eta, residual_MHz, "o-", ms=6, lw=1.4, color=_C_LORENTZ,
              label="chirped residual")
    if rabi_table is not None:
        rt_eta = np.asarray(rabi_table["eta"], dtype=float)
        rt_ridge = np.asarray(rabi_table["delta_MHz"], dtype=float)
        ok = np.isfinite(rt_ridge)
        ax_rs.plot(rt_eta[ok], rt_ridge[ok], "x--", ms=6, lw=1.0, color=_C_INK,
                  alpha=0.7, label="constant-probe ridge")
    ax_rs.axhline(0.0, color=_C_INK, ls=":", lw=1.2)
    ax_rs.axvline(target_eta, color=_C_INK, ls=":", lw=1.4)
    ax_rs.set_xlabel(r"drive strength $|\eta|$")
    ax_rs.set_ylabel("located resonance - wp_offset (MHz)")
    ax_rs.set_title("chirp residual vs drive", fontsize=10)
    ax_rs.legend(fontsize=8, framealpha=0.9)
    ax_rs.grid(alpha=0.25)

    fig.suptitle(title or (rf"Post-chirp validation, "
                           rf"$\eta^*$ = {target_eta:.2f}"), fontsize=12)
    return _savefig(fig, out)


# ===========================================================================
# Step 4 -- length: the only free parameter once the amplitude is fixed
# ===========================================================================
def fit_swap_period(times_ns: np.ndarray, P: np.ndarray) -> Dict[str, Any]:
    """Fit ``P(t) = A sin^2(pi t / (2 T_swap)) + C`` and return the full-swap time.

    Seeded from the FFT peak of the mean-removed trace, so the fit does not land on
    a harmonic.

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

    One ``evolve_trajectory`` solve for the whole trace. An independent seed only:
    :func:`length_rabi` sets ``t_g_ns`` on the real shaped gate. Blind to DRAG and
    chirp; it measures the bare exchange rate at this drive.
    """
    from snail_solver.find_stark_resonance import build_chevron_coupler

    solver = solver or dict(_DEFAULT_SOLVER)
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
                drag_channels=None,
                spec_abs_GHz: Optional[float] = None,
                solver: Optional[Dict[str, Any]] = None,
                refine: bool = True, extend: bool = True,
                max_t_g_factor: float = 2.0, min_t_g_factor: float = 0.5,
                extend_rounds: int = 6,
                logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Step 4: sweep the gate LENGTH at fixed peak |eta| and find the full swap.

    The authoritative length calibration: evaluates the actual gate (shape, chirp,
    DRAG, full Hilbert space) at each candidate length, with the amplitude re-derived
    by :func:`fixed_eta_amp_scale` so the peak drive stays fixed. The grid defaults
    to +/-30% about ``t_g0``.

    ``t_g0`` assumes the BARE exchange rate; a rate reduced by ``r`` (chirp, coupler
    occupation, parasite dressing near a subharmonic) needs ``1/r`` more time, and
    +/-30% tolerates only ``r >= 0.77``. An endpoint maximum was never BRACKETED (the
    gate is left under-rotated; 18/73 columns of the 2026-09-16 grid sat on
    ``1.3 t_g0``), so with `extend` the grid grows by its own step, up to
    ``max_t_g_factor * t_g0``, until the maximum is interior; ``railed`` then means
    "no interior optimum out to the cap", a RESULT. Stopping as soon as the argmax
    is interior returns the FIRST swap, not a later Rabi cycle.

    ``max_t_g_factor = 2``: the transfer score charges nothing for duration, but at
    eta = 1.3 and T_eff = 12.5 us, 2 ``t_g0`` already costs ~1.7e-2 incoherently.
    That cost is charged downstream (``subharmonic_gate_scan.total_infidelity``).
    """
    from snail_solver.device_utils import maximize_1d, transfer_probability

    solver = solver or dict(_DEFAULT_SOLVER)
    t_g0 = nominal_t_g(config, target_eta)
    grid = (np.asarray(t_g_grid, dtype=float) if t_g_grid is not None
            else t_g0 * np.linspace(0.7, 1.3, 13))

    def score(t_g: float) -> float:
        return transfer_probability(
            config, float(t_g), fixed_eta_amp_scale(config, float(t_g), target_eta),
            wp_offset_GHz, solver, spec_abs_GHz=spec_abs_GHz,
            drag_beat_GHz=drag_beat_GHz, chirp_coeffs_GHz=chirp_coeffs_GHz,
            drag_n_pump=drag_n_pump, drag_channels=drag_channels)

    grid = np.sort(np.asarray(grid, dtype=float))
    P = np.array([score(t) for t in grid])
    k = int(np.argmax(P))
    nfev = int(grid.size)
    if logger:
        logger.info(f"  length: coarse best t_g={grid[k]:.3f} ns  P={P[k]:.5f} "
                    f"(t_g0={t_g0:.3f} ns)")

    n_extend = 0                  # bracket the maximum; never report a boundary
    if extend and grid.size > 1:
        step = float(np.median(np.diff(grid)))
        hi_cap, lo_cap = float(max_t_g_factor) * t_g0, float(min_t_g_factor) * t_g0
        while n_extend < int(extend_rounds) and not 0 < k < grid.size - 1:
            if k == grid.size - 1:
                new_t = grid[-1] + step * np.arange(1.0, 4.0)
                new_t = new_t[new_t <= hi_cap + 1e-9]
                if new_t.size == 0:
                    break
                grid = np.concatenate([grid, new_t])
                P = np.concatenate([P, [score(t) for t in new_t]])
                where = "upper"
            else:
                new_t = grid[0] - step * np.arange(1.0, 4.0)
                new_t = np.sort(new_t[new_t >= lo_cap - 1e-9])
                if new_t.size == 0:
                    break
                grid = np.concatenate([new_t, grid])
                P = np.concatenate([[score(t) for t in new_t], P])
                where = "lower"
            nfev += int(new_t.size)
            k = int(np.argmax(P))
            n_extend += 1
            if logger:
                logger.info(f"  length: maximum was on the {where} edge; extended to "
                            f"[{grid[0]:.1f}, {grid[-1]:.1f}] ns "
                            f"({grid[0]/t_g0:.2f}-{grid[-1]/t_g0:.2f} t_g0), best now "
                            f"t_g={grid[k]:.3f} ns P={P[k]:.5f}")

    railed = not 0 < k < grid.size - 1
    best_t, best_P = float(grid[k]), float(P[k])
    if n_extend and logger:        # the transfer score charges nothing for duration
        logger.info(f"  length: extended fit sits at {best_t / t_g0:.2f} t_g0 "
                    f"({best_t:.1f} ns), i.e. {100 * (best_t / t_g0 - 1):+.0f}% gate "
                    f"time against t_g0 -- the incoherent error scales with it and is "
                    f"charged by total_infidelity, not by this transfer score.")
    if refine and not railed:
        lo, hi = float(grid[k - 1]), float(grid[k + 1])
        best_t, best_P, n = maximize_1d(score, lo, hi, n_points=7, n_refine=2)
        nfev += int(n)
    if railed and logger:
        logger.info(f"  length: STILL railed at t_g={best_t:.3f} ns "
                    f"({best_t/t_g0:.2f} t_g0, P={best_P:.5f}) after {n_extend} "
                    f"extension(s) -- no interior optimum out to "
                    f"[{min_t_g_factor:g}, {max_t_g_factor:g}] t_g0. That is a RESULT "
                    f"about this operating point, not a fitted length.")
    return {"t_g_grid": grid, "P": P, "t_g_ns": best_t, "transfer": best_P,
            "t_g0_ns": t_g0, "amp_scale": fixed_eta_amp_scale(config, best_t,
                                                              target_eta),
            "nfev": nfev, "railed": bool(railed),
            "t_g_over_t_g0": float(best_t / t_g0),
            "n_extensions": int(n_extend),
            "grid_span_t_g0": [float(grid[0] / t_g0), float(grid[-1] / t_g0)]}


# ===========================================================================
# Step 5 -- DRAG shifts the detuning
# ===========================================================================
def calibrate_drag_offset(config: Dict[str, Any], t_g: float, target_eta: float,
                          drag_beat_GHz: float, *, drag_n_pump: int = 1,
                          drag_channels=None,
                          chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                          span_MHz: float = 40.0, points: int = 31,
                          time_points: int = 120,
                          spec_abs_GHz: Optional[float] = None,
                          solver: Optional[Dict[str, Any]] = None,
                          jobs: int = 0,
                          predicted_GHz: Optional[float] = None) -> Dict[str, Any]:
    """MEASURE the resonance shift DRAG introduces: shaped chevron, DRAG off vs on.

    The empirical counterpart to the quadrature model in
    :func:`chirp_from_measured_shift`, which ASSUMES the shift follows the total drive
    through the constant-probe law; any other DRAG mechanism shows up only here.
    Both legs use the SHAPED chevron (a constant probe has no quadrature). Pass
    `predicted_GHz` to get ``excess_GHz``; both are pulse-averaged, so comparable.
    """
    from snail_solver import find_stark_resonance as FS

    solver = solver or dict(_DEFAULT_SOLVER)
    amp_scale = fixed_eta_amp_scale(config, t_g, target_eta)
    offsets = np.linspace(-span_MHz / 2e3, span_MHz / 2e3, int(points))

    def locate(beat):
        return FS.scan(config, float(t_g), amp_scale, offsets,
                       1.05 * float(t_g), int(time_points), solver, n_jobs=jobs,
                       spec_abs_GHz=spec_abs_GHz, shape="raised_cosine",
                       drag_beat_GHz=beat, chirp_coeffs_GHz=chirp_coeffs_GHz,
                       drag_n_pump=drag_n_pump,
                       drag_channels=(drag_channels if beat is not None else None))

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
                     drag_n_pump: int = 1, drag_channels=None, eta_lo: float = 0.6, eta_hi: float = 1.2,
                     amp_points: int = 4,
                     chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                     span_MHz: float = 40.0, points: int = 25,
                     time_points: int = 120,
                     spec_abs_GHz: Optional[float] = None,
                     solver: Optional[Dict[str, Any]] = None, jobs: int = 0,
                     logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """DRAG-induced resonance shift as a function of DRIVE STRENGTH.

    :func:`calibrate_drag_offset` at several peak |eta|, each a full-swap gate at its
    own drive (``t_g = nominal_t_g(|eta|)``), so the shift is identified by its
    SCALING. With ``q ~ eta / t_g ~ eta^2`` a shift that only adds drive goes as
    ``eta^4``; any other exponent is a mechanism the quadrature model lacks.

    Returns
    -------
    dict
        ``eta``, ``t_g_ns``, ``shift_MHz`` (DRAG-on minus DRAG-off), ``exponent``
        (fitted power of |eta|), ``rows``.
    """
    solver = solver or dict(_DEFAULT_SOLVER)
    eta = np.linspace(float(eta_lo), float(eta_hi), int(amp_points)) * float(target_eta)
    shift = np.full(eta.size, np.nan)
    t_gs = np.full(eta.size, np.nan)
    rows = []
    for i, e in enumerate(eta):
        t_g = nominal_t_g(config, float(e))
        t_gs[i] = t_g
        row = calibrate_drag_offset(
            config, t_g, float(e), float(drag_beat_GHz), drag_n_pump=drag_n_pump,
            drag_channels=drag_channels,
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
                        degree: int = 8) -> float:
    """Pulse-averaged shift with the DRAG quadrature switched off (GHz).

    Subtracted from the DRAG-on mean, it gives the model's predicted DRAG shift.
    """
    return float(chirp_from_measured_shift(table, target_eta,
                                           degree=degree)["mean_shift_GHz"])


# ===========================================================================
# Orchestrator
# ===========================================================================
def run_tune_up(config: Dict[str, Any], target_eta: float, *,
                drag_beat_GHz: Optional[float] = None, drag_n_pump: int = 1,
                drag_channels=None,
                spec_abs_GHz: Optional[float] = None, chirp_degree: int = 8,
                quartic_warn: float = 0.25, chirp_max_passes: int = 12,
                eta_lo: float = 0.3, eta_hi: float = 1.0,
                amp_points: int = 9, wp_span_MHz: Optional[float] = None,
                wp_points: int = 25, tg_points: int = 13,
                tg_lo: float = 0.7, tg_hi: float = 1.3, max_drag_iters: int = 4,
                window_tg: float = 2.0, n_time: int = 161,
                span_linewidths: float = 4.0, drag_shift_points: int = 0,
                contrast_min: float = 0.35,
                chirp_tol_GHz: float = 1e-4, offset_tol_MHz: float = 0.2,
                do_time_rabi: bool = True, jobs: int = 0,
                zero_chirp_frac: float = 0.0,
                chirp_free_fallback: bool = False,
                chirp_free_max_frac: float = 0.10,
                max_span_growths: int = 3, max_wp_points: int = 121,
                span_growth: str = "railed", max_compound_growths: int = 1,
                couple_drag: bool = True,
                drag_decouple_fallback: bool = False,
                probe_shape: str = "constant", moment_weighting: str = "rabi",
                post_chirp_points: int = 0,
                chirp_source: str = "law",
                ridge_noise_MHz: Optional[float] = None,
                rabi_table: Optional[Dict[str, Any]] = None,
                drop_swept_channels: bool = False,
                solver: Optional[Dict[str, Any]] = None,
                logger: Optional[logging.Logger] = None,
                **map_kw) -> Dict[str, Any]:
    """Run the full tune-up and return an operating-point record.

    ``chirp_source="ridge"`` builds the chirp from the measured ridge itself
    (:mod:`snail_solver.ridge_chirp`) rather than the ``k2/k4`` truncation: a crossing
    or a non-converging series no longer costs the chirp. Only a railed ridge, a ridge
    with no usable row, or an excursion within ``chirp_free_max_frac`` of a
    half-linewidth leaves the column chirp-free. ``rabi_table`` replays a STORED step
    1 (ridge mode only), so steps 2-4 can be redone without re-measuring.
    ``drop_swept_channels`` drops a DRAG channel whose beat the calibrated chirp
    sweeps inside the skip window, instead of raising when the gate is built; the
    drops are in ``stages.drag_swept_dropped`` and the survivors in the returned
    ``drag_channels``.

    Order, and what is held fixed at each step::

        0.  t_g0 = auto_t_g(eta*);  amp_scale := fixed_eta_amp_scale(t_g)  [never free]
        1.  Rabi, constant probe, chirp OFF, DRAG OFF   -> the shift LAW k2, k4
        2.  chirp by projecting that law along the pulse -> c_k, wp_offset
        3.  residual offset from a SHAPED chevron with the chirp on
        4.  length via the shaped scan                  -> t_g_ns
        --- with DRAG, iterate 2-4 to self-consistency ---

    The Rabi sweep runs ONCE (it measures the device, not the pulse). With DRAG on,
    the chirp<->Delta(t) fixed point is solved inside :func:`chirp_from_measured_shift`,
    but the quadrature also scales as ``1/t_g``, so the outer loop here closes the
    chirp -> length -> chirp coupling (2-3 passes).

    ``post_chirp_points``, if nonzero, runs :func:`post_chirp_table` on the result
    (never fatal) and stores it under ``stages.post_chirp``.

    Failed shift laws (``chirp_free_fallback``): a failed law kills the chirp, not
    the gate -- the bare pulse needs only a length and a carrier (``delta0`` is
    measured fine, and step 3 measures the rest). ``chirp_free_reason`` separates a
    real 1.00x (``no_measurable_shift``: excursion <= ``chirp_free_max_frac`` of a
    half-linewidth) from EXCLUSIONS (``stark_crossing``, ``chirp_not_converged``),
    which must not be averaged into chirp-gain ratios.
    """
    log = logger or logging.getLogger("tune_up")
    solver = solver or dict(_DEFAULT_SOLVER)
    t_g0 = nominal_t_g(config, target_eta)
    log.info(f"tune-up: target_eta={target_eta} -> t_g0={t_g0:.3f} ns "
             f"(amp_scale=1 there)")

    common = dict(eta_lo=eta_lo, eta_hi=eta_hi, amp_points=amp_points,
                  wp_span_MHz=wp_span_MHz, wp_points=wp_points,
                  spec_abs_GHz=spec_abs_GHz, drag_n_pump=drag_n_pump,
                  window_tg=window_tg, n_time=n_time, jobs=jobs, solver=solver,
                  span_linewidths=span_linewidths, contrast_min=contrast_min,
                  probe_shape=probe_shape, moment_weighting=moment_weighting,
                  max_span_growths=int(max_span_growths),
                  span_growth=str(span_growth),
                  max_compound_growths=int(max_compound_growths),
                  max_wp_points=int(max_wp_points),
                  # A chirp that does not MOVE is a static detuning: absorb it into
                  # the carrier rather than routing the column to the fallback.
                  zero_chirp_frac=(float(zero_chirp_frac) if zero_chirp_frac > 0.0
                                   else float(chirp_free_max_frac)),
                  logger=log, **map_kw)

    # -- 1: the Rabi sweep, measured once ------------------------------------
    log.info(f"step 1: Rabi ({'SHAPED gate-pulse' if probe_shape != 'constant' else 'constant-probe'}"
             f" chevron per drive strength), DRAG OFF")
    # See the docstring for the chirp-free fallback. The excursion |k2 eta^2 + k4
    # eta^4| vs the half-width separates cleanly (2026-09-22 grid): <= 10% -> residual
    # 1.0-16.7x the excursion (no signal); > 20% -> 0.21-0.58x (a real law, so an
    # exclusion, not a 1.00x). A railed ridge has no fit, so its excursion is unknown.
    chirp_free = False
    chirp_free_reason = None
    stark_crossing_eta = None
    ridge = str(chirp_source) == "ridge"
    if str(chirp_source) not in ("law", "ridge"):
        raise ValueError(f"chirp_source={chirp_source!r}: expected 'law' or 'ridge'")
    if rabi_table is not None and not ridge:
        raise ValueError("rabi_table replays step 1 for chirp_source='ridge' only: "
                         "the law path re-derives its verdicts while measuring")
    try:
        if rabi_table is not None:
            log.info("step 1: REPLAYING a stored Rabi table (no chevrons solved)")
            table = dict(rabi_table)
        else:
            table = rabi_shift_table(config, target_eta, **common)
    except StarkCrossingInSweep as exc:
        # No trustworthy ridge at eta*: calibrate bare (and DRAG) and EXCLUDE the chirp.
        if ridge:                    # the ridge law follows a crossing; judged below
            table = exc.table
            stark_crossing_eta = exc.crossing_eta
        elif not chirp_free_fallback:
            raise
        else:
            table = exc.table
            chirp_free = True
            chirp_free_reason = "stark_crossing"
            stark_crossing_eta = exc.crossing_eta
            log.info(f"step 1: {exc} Calibrating the column WITHOUT a chirp so the "
                     f"bare and DRAG series survive; the chirp series is EXCLUDED "
                     f"here, not reported as no-gain.")
    except RabiFitError as exc:
        if ridge:                    # no law needed; the ridge itself is judged below
            table = exc.table
        elif not chirp_free_fallback:
            raise
        _fit = ((exc.table or {}).get("fit") or {})
        _frac = _fit.get("chirp_excursion_frac_linewidth")
        table = exc.table
        chirp_free = not ridge
        if ridge:
            pass
        elif _frac is not None and float(_frac) <= float(chirp_free_max_frac):
            chirp_free_reason = "no_measurable_shift"
            log.info(
                f"step 1: no usable shift law ({exc}). The chirp would sweep only "
                f"{float(_frac):.1%} of a half-linewidth, so there is nothing to "
                f"chirp: falling back to a CHIRP-FREE calibration -- delta0 sets "
                f"the carrier, step 3 measures the rest, and the length scan never "
                f"needed the law.")
        else:
            # A real shift the law does not describe: an EXCLUSION, like a crossing,
            # so the bare and DRAG gates survive. (Offline refits with k6, k8, odd
            # powers or shorter ranges rescued none of 64 such columns.)
            chirp_free_reason = "chirp_not_converged"
            _why = ("the ridge railed, so no shift law was fitted and the excursion "
                    "is unknown" if _frac is None else
                    f"the chirp would sweep {float(_frac):.0%} of a half-linewidth "
                    f"(> --chirp-free-max-frac {chirp_free_max_frac:g}), so there "
                    f"IS a shift here")
            log.warning(
                f"step 1: {exc} Not chirp-free: {_why}. Calibrating the column "
                f"WITHOUT a chirp so the bare and DRAG series survive; the chirp "
                f"series is EXCLUDED here, NOT reported as no-gain. To measure a "
                f"chirp at this column the MEASUREMENT or the LAW has to change "
                f"(--span-linewidths for a railed ridge, --wp-points/--amp-points "
                f"for a noisy one, a shorter --eta-hi for a truncated series).")

    # -- 1b: the chirp's law read off the ridge itself (chirp_source="ridge") ----
    law = None
    if ridge:
        from snail_solver.ridge_chirp import law_from_table
        _qkw = {"contrast_min": float(contrast_min)}
        if map_kw.get("leak_max") is not None:
            _qkw["leak_max"] = float(map_kw["leak_max"])
        try:
            law = law_from_table(table, config, target_eta,
                                 probe_shape=probe_shape,
                                 moment_weighting=moment_weighting,
                                 noise_MHz=ridge_noise_MHz, t_g_ns=t_g0, **_qkw)
        except ValueError as exc:
            chirp_free, chirp_free_reason = True, "no_usable_ridge"
            log.warning(f"step 1b: no ridge law ({exc}); calibrating CHIRP-FREE "
                        f"(an exclusion: nothing measured to chirp along)")
        else:
            if law["railed"]:
                chirp_free, chirp_free_reason = True, "railed_ridge"
                log.warning(f"step 1b: {law['n_railed']} ridge row(s) sit on their "
                            f"scan edge -- not a measurement; calibrating CHIRP-FREE")
            elif law["excursion_frac_linewidth"] <= float(chirp_free_max_frac):
                chirp_free, chirp_free_reason = True, "no_measurable_shift"
                log.info(f"step 1b: the ridge law swings "
                         f"{law['excursion_MHz']:.3f} MHz at eta*, "
                         f"{law['excursion_frac_linewidth']:.1%} of a half-linewidth: "
                         f"nothing to chirp (an honest 1.00x)")
            else:
                chirp_free, chirp_free_reason = False, None
                log.info(f"step 1b: RIDGE law over {law['n_used']} rows, misfit "
                         f"{law['resid_MHz']:.4f} MHz (tolerance "
                         f"{law['noise_MHz']:.4f}), swing {law['excursion_MHz']:.3f} "
                         f"MHz at eta* = {law['excursion_frac_linewidth']:.0%} of a "
                         f"half-linewidth"
                         + (f"; through the crossing at |eta| = "
                            f"{stark_crossing_eta:.3f}" if stark_crossing_eta
                            else ""))

    def project(t_g: float) -> Dict[str, Any]:
        """The chirp implied by the measured law at this gate length."""
        if chirp_free:
            # No law, no chirp; delta0 is still the static carrier retune. The
            # neutral keys let every downstream reader work unchanged.
            static = (float(law["delta0"]) if law is not None else
                      float((table.get("fit") or {}).get("delta0", 0.0))) * 1e-3
            return {"coeffs_GHz": [], "chirp_free": True, "mean_shift_GHz": static,
                    "quartic_fraction": 0.0, "rel_diff": 0.0,
                    "perturbative_ok": True, "extrapolation_ratio": 1.0,
                    "measured_eta_max": float(target_eta),
                    "target_eta": float(target_eta), "degree": int(chirp_degree),
                    "stark_mean_GHz": 0.0, "static_GHz": static}
        kw = dict(degree=chirp_degree, drag_beat_GHz=drag_beat_GHz,
                  drag_n_pump=drag_n_pump, drag_channels=drag_channels, t_g=t_g,
                  shape=_shape_kind, shape_kw=_shape_kw, quartic_warn=quartic_warn,
                  max_iters=int(chirp_max_passes), ridge_law=law)
        try:
            return chirp_from_measured_shift(
                table, target_eta, couple_drag=bool(couple_drag), **kw)
        except DragFixedPointDiverged as exc:
            if not drag_decouple_fallback or not couple_drag:
                raise
            # Fall back to the first Picard iterate (decoupled DRAG), only on this
            # failure, so every column that CAN be solved coupled still is.
            drag_decoupled.append(float(exc.min_abs_detuning_GHz))
            log.info(
                f"step 2: the chirp<->DRAG fixed point diverged "
                f"(min|Delta| = {exc.min_abs_detuning_GHz * 1e3:.3f} MHz). Falling "
                f"back to DECOUPLED DRAG at this column -- the chirp is the "
                f"bare-envelope one and the quadrature is applied on top of it. "
                f"neglected_shift_frac reports the term that is then dropped.")
            out = chirp_from_measured_shift(
                table, target_eta, couple_drag=False, **kw)
            out["drag_decoupled_fallback"] = True
            return out

    def shaped_residual(t_g: float, chirp, wp_offset: float) -> float:
        """Residual offset of the ASSEMBLED gate: shaped pulse, chirp and DRAG on.

        The one place the real pulse can disagree with the constant-probe law. The
        span is sized in linewidths, like the Rabi rows.
        """
        from snail_solver import find_stark_resonance as FSR
        span = (float(wp_span_MHz) if wp_span_MHz is not None
                else 2.0 * float(span_linewidths) * 1e3 / (2.0 * float(t_g)))
        offs = (np.linspace(-span / 2e3, span / 2e3, int(wp_points))
                + float(wp_offset))
        chev = FSR.scan(config, float(t_g),
                        fixed_eta_amp_scale(config, float(t_g), target_eta), offs,
                        1.05 * float(t_g), int(n_time), solver, n_jobs=jobs,
                        spec_abs_GHz=spec_abs_GHz, shape="raised_cosine",
                        drag_beat_GHz=drag_beat_GHz, chirp_coeffs_GHz=list(chirp),
                        drag_n_pump=drag_n_pump, drag_channels=drag_channels)
        return float(chev["resonance_offset_GHz"]) - float(wp_offset)

    stages: Dict[str, Any] = {"rabi": table}
    if law is not None:
        stages["ridge_law"] = law
    swept_dropped: List[Dict[str, Any]] = []
    history = []
    t_g = t_g0
    drag_decoupled: List[float] = []      # min|Delta| at each fallback, if any
    chirp, wp_offset, length = None, 0.0, None
    # With DRAG off (or decoupled) the chirp is length-independent: one pass. The
    # d-th nested correction scales as 1/t_g^d, so K channels get more passes.
    _shape_kind, _shape_kw = shape_config(config)
    _drag_on = drag_beat_GHz is not None or bool(drag_channels)
    _n_ch = len(_resolve_drag_channels(drag_beat_GHz, drag_n_pump, drag_channels))
    n_outer = (max(int(max_drag_iters), 2 * _n_ch)
               if (_drag_on and couple_drag) else 1)

    for it in range(n_outer):
        prev_chirp = None if chirp is None else np.array(chirp)
        prev_t_g = t_g

        # -- 2: the chirp implied by the law at the current length ------------
        proj = project(t_g)
        # A chirp can sweep a DRAG beat into the skip window mid-pulse even when the
        # unchirped beat is clear; the quadrature then diverges. Drop exactly that
        # channel (the skip window's own rule, applied to the CHIRPED beat) and
        # re-project, rather than lose the column when the gate is built.
        while drop_swept_channels and drag_channels and proj.get(
                "min_abs_detuning_per_channel_GHz"):
            from snail_solver.sweep_common import _drag_skip_GHz
            _skip = _drag_skip_GHz(config)
            _res = _resolve_drag_channels(None, drag_n_pump, drag_channels)
            _fl = proj["min_abs_detuning_per_channel_GHz"]
            _keep = [c for c, f in zip(_res, _fl) if float(f) >= _skip]
            if len(_keep) == len(_res):
                break
            for c, f in zip(_res, _fl):
                if float(f) < _skip:
                    swept_dropped.append({"beat_GHz": float(c.beat_GHz),
                                          "n_pump": int(c.n_pump),
                                          "min_abs_detuning_GHz": float(f),
                                          "pass": int(it)})
                    log.warning(f"step 2: the chirp sweeps the DRAG beat "
                                f"{c.beat_GHz * 1e3:+.1f} MHz (k={c.n_pump}) to "
                                f"{float(f) * 1e3:.3f} MHz, inside the "
                                f"{_skip * 1e3:g} MHz skip window: DROPPING that "
                                f"channel and re-projecting")
            drag_channels = list(_keep)
            _drag_on = drag_beat_GHz is not None or bool(drag_channels)
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
        if not proj["perturbative_ok"]:
            log.info(f"  WARNING: quartic_fraction={proj['quartic_fraction']:.2f} "
                     f">= {quartic_warn} at target_eta={target_eta:.2f} -- the "
                     f"eta^2 + eta^4 Stark law is not converging (the 'correction' "
                     f"term is not small next to the leading one), so the unmeasured "
                     f"eta^6 term is plausibly of the same order and this law carries "
                     f"little information about delta(target_eta). Not raised (this "
                     f"pipeline reports, it doesn't refuse), but treat the chirp as a "
                     f"candidate -- check stages.rabi's stability warning too, since "
                     f"this number is computed FROM the same fit.")

        # -- 3: what the assembled gate still wants ---------------------------
        residual_GHz = shaped_residual(t_g, chirp, wp_offset)
        wp_offset += residual_GHz
        log.info(f"step 3: shaped-chevron residual {residual_GHz * 1e3:+.3f} MHz "
                 f"-> wp_offset={wp_offset * 1e3:+.3f} MHz")

        # -- 4: the length, everything else frozen ----------------------------
        log.info("step 4: length scan at fixed |eta| (the only free parameter)")
        length = length_rabi(config, target_eta, wp_offset_GHz=wp_offset,
                             chirp_coeffs_GHz=chirp, drag_beat_GHz=drag_beat_GHz,
                             drag_n_pump=drag_n_pump, drag_channels=drag_channels,
                             spec_abs_GHz=spec_abs_GHz,
                             solver=solver, logger=log,
                             t_g_grid=t_g0 * np.linspace(float(tg_lo), float(tg_hi),
                                                         int(tg_points)))
        t_g = float(length["t_g_ns"])
        if length["railed"]:
            edge = "lower" if t_g <= float(length["t_g_grid"][0]) else "upper"
            log.warning(
                f"  the length optimum railed against the {edge} edge of the scan "
                f"window [{tg_lo:g}, {tg_hi:g}] x t_g0 = [{tg_lo * t_g0:.1f}, "
                f"{tg_hi * t_g0:.1f}] ns. A maximum on the grid EDGE is not a "
                f"maximum: the true full swap lies outside it, so this t_g_ns is a "
                f"bound, not a calibration. t_g0 = 2A/eta* assumes the leading-order "
                f"rate 6 g3 la lb |eta|; a device whose dressed rate differs needs a "
                f"wider window -- rerun with --tg-{'lo' if edge == 'lower' else 'hi'} "
                f"past {tg_lo if edge == 'lower' else tg_hi:g}.")

        # A chirp-free column has no coefficients (np.max([]) raises): dc is 0.
        if prev_chirp is None:
            dc = float("inf")
        elif len(chirp) == 0 and len(prev_chirp) == 0:
            dc = 0.0
        else:
            dc = float(np.max(np.abs(np.array(chirp) - prev_chirp)))
        dt = abs(t_g - prev_t_g)
        history.append({"iter": it, "t_g_ns": t_g, "max_dc_GHz": dc,
                        "d_t_g_ns": dt, "wp_offset_GHz": wp_offset,
                        "residual_GHz": residual_GHz, "chirp_GHz": list(chirp)})
        if not _drag_on or not couple_drag:
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
                 if _drag_on else None)

    # Direct test of "DRAG shifts the resonance only by adding drive".
    if _drag_on and drag_beat_GHz is not None:
        predicted = (float(proj["mean_shift_GHz"])
                     - float(project_nodrag_mean(table, target_eta, chirp_degree)))
        meas = calibrate_drag_offset(
            config, t_g, target_eta, float(drag_beat_GHz), drag_n_pump=drag_n_pump,
            drag_channels=drag_channels,
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
                drag_channels=drag_channels,
                amp_points=int(drag_shift_points), chirp_coeffs_GHz=chirp,
                span_MHz=(wp_span_MHz or 40.0), points=int(wp_points),
                time_points=int(n_time), spec_abs_GHz=spec_abs_GHz, solver=solver,
                jobs=jobs, logger=log)
    if drag_info:
        stages["drag_loop"] = drag_info
    if swept_dropped:
        stages["drag_swept_dropped"] = swept_dropped

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

    # -- does the chirp's SHAPE help over a retuned carrier? wp_offset already carries
    # the chirp's mean, so zeroing the chirp isolates what tracking the shift buys.
    try:
        from snail_solver.device_utils import transfer_probability
        flat_transfer = transfer_probability(
            config, t_g, amp_scale, wp_offset, solver, spec_abs_GHz=spec_abs_GHz,
            drag_beat_GHz=drag_beat_GHz, chirp_coeffs_GHz=[], drag_n_pump=drag_n_pump,
            drag_channels=drag_channels)
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
        # Name the TYPE, so a signature drift (TypeError) is not hidden.
        log.info(f"  chirp ablation skipped: {type(exc).__name__}: {exc}")

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
        # chirp_free distinguishes a fallback from a chirp that fitted to zero;
        # the reason decides real 1.00x vs exclusion (see the docstring).
        "chirp_free": bool(chirp_free),
        "chirp_free_reason": chirp_free_reason,
        "stark_crossing_eta": stark_crossing_eta,
        "chirp_source": str(chirp_source),
        # True when the coupled fixed point diverged and DRAG fell back to decoupled.
        "drag_decoupled": bool(drag_decoupled),
        "drag_decoupled_min_abs_detuning_GHz": (min(drag_decoupled)
                                                if drag_decoupled else None),
        # The length fit's own verdict: a railed t_g is a window bound, not a length.
        "t_g_railed": bool(length["railed"]),
        "t_g_over_t_g0": float(length.get("t_g_over_t_g0", t_g / t_g0)),
        "t_g_grid_span_t_g0": length.get("grid_span_t_g0"),
        "t_g_extensions": int(length.get("n_extensions", 0)),
    }

    if post_chirp_points:
        try:
            stages["post_chirp"] = post_chirp_table(
                config, record, amp_points=int(post_chirp_points),
                rabi_table=table, chirp_degree=chirp_degree, jobs=jobs,
                solver=solver, logger=log)
        except Exception as exc:                              # never fatal: it is a check
            log.info(f"  post-chirp validation skipped: {exc}")

    log.info(f"done: t_g={t_g:.3f} ns  amp_scale={record['amp_scale']:.5f}  "
             f"wp_offset={wp_offset * 1e3:+.3f} MHz  transfer={length['transfer']:.5f}")
    return {"operating_point": record, "stages": stages, "t_g0_ns": t_g0,
            "drag": drag_info,
            # the channels the calibrated pulse plays (after any swept drop)
            "drag_channels": (list(drag_channels) if drag_channels is not None
                              else None)}


# ===========================================================================
# CLI
# ===========================================================================
def _git_describe() -> str:
    """``<sha>`` or ``<sha>-dirty`` for the repo this module lives in, else ""."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return ""
        out = sha.stdout.strip()
        st = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                            capture_output=True, text=True, timeout=5)
        return out + ("-dirty" if st.stdout.strip() else "")
    except Exception:                      # provenance must never fail a run
        return ""


def _run_attrs(device_path: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """Provenance for the root of an ``--out`` file.

    Stored as HDF5 ROOT ATTRIBUTES, outside the result tree, so it is never mistaken
    for data while ``h5ls -v run.h5`` shows which command, device and commit (with a
    ``-dirty`` flag) produced the file.
    """
    import platform
    import shlex
    import sys
    import time

    # Rebuild the `-m` spelling: argv[0] is the resolved .py path under `-m`.
    argv = " ".join(shlex.quote(a) for a in sys.argv[1:])
    return {"tool": "snail_solver.tune_up",
            "git_commit": _git_describe(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": f"python -m snail_solver.tune_up {argv}".strip(),
            "device_path": str(device_path) if device_path else "",
            "host": platform.node(),
            **extra}


def main() -> None:
    """CLI entry point."""
    import argparse

    from snail_solver.h5_io import (attach_figures, is_hdf5, load_doc, save_doc,
                                    split_address)
    from snail_solver.stark_chirp import MOMENT_WEIGHTINGS

    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.tune_up", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="required unless --replot is given (or --save-point, which "
                         "needs it to know which device JSON to write into)")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="peak |eta| to FIX for the whole tune-up; sets the nominal "
                         "length t_g0 = 2A/eta*. Larger -> shorter gate, more leakage. "
                         "Required unless --replot is given")
    ap.add_argument("--replot", metavar="FILE", default=None,
                    help="skip the run and regenerate --plot/--plot-ridge from a "
                         "previous --out file (no solves). HDF5 or JSON, detected by "
                         "content; FILE:/runs/eta1p8 reads one run of a "
                         "tune_up_sweep file")
    ap.add_argument("--drag-beat-GHz", type=float, default=None,
                    help="calibrate with DRAG on at this beat; enables the "
                         "chirp<->DRAG iteration")
    ap.add_argument("--drag-channel", action="append", default=None,
                    metavar="BEAT[:K[:N]]",
                    help="RECURSIVE multi-derivative DRAG (Li/Calarco/Motzoi, npj QI "
                         "10, 66 (2024)), one correction per process. Repeatable. "
                         "BEAT in GHz, K = pump quanta (default 1), N = photons in "
                         "F^(n) (default K), e.g. --drag-channel 0.30 "
                         "--drag-channel 0.55:2:2 . Exclusive with --drag-beat-GHz. "
                         "More than one channel needs envelope=sine_power with "
                         "envelope_m >= the channel count.")
    ap.add_argument("--drag-n-pump", type=int, default=1,
                    help="pump quanta of the suppressed process (1 one-pump, "
                         "2 subharmonic, 0 static/pump-independent)")
    ap.add_argument("--drag-auto", action="store_true",
                    help="DERIVE the recursive-DRAG channels from the device "
                         "(spectator_audit.select_drag_channels: leakage, coupler and "
                         "mode-subharmonic processes, then the strongest correctable "
                         "parasite). The audit is printed before solving. Exclusive "
                         "with --drag-channel / --drag-beat-GHz.")
    ap.add_argument("--max-drag-channels", type=int, default=4,
                    help="cap on --drag-auto's recursion depth [4]; the device's "
                         "envelope_m must be >= the channels selected.")
    ap.add_argument("--force", action="store_true",
                    help="with --drag-auto, solve even when a parasite is NOT "
                         "PERTURBATIVE (g/|det| >= 1), which no pulse shape fixes.")
    ap.add_argument("--spec-abs-GHz", type=float, default=None)
    ap.add_argument("--chirp-degree", type=int, default=8,
                    help="Legendre truncation for the chirp (in u). delta(u) is a "
                         "polynomial in cos(pi u/2), not in u, so every degree "
                         "truncates; 8 leaves ~0.7%% peak error on a Hann")
    ap.add_argument("--quartic-warn", type=float, default=0.25,
                    help="warn (never raise) when the quartic term is this fraction "
                         "of the quadratic at target_eta")
    ap.add_argument("--window-tg", type=float, default=2.0,
                    help="chevron time window in units of each row's own "
                         "nominal_t_g(|eta|) (2 swaps per unit at constant drive)")
    ap.add_argument("--n-time", type=int, default=161,
                    help="chevron readout times (one solve covers all of them)")
    ap.add_argument("--eta-lo", type=float, default=0.3,
                    help="Rabi amplitude window, as a fraction of --target-eta")
    ap.add_argument("--eta-hi", type=float, default=1.0,
                    help="the pulse never exceeds its peak, so sampling above 1.0 is "
                         "extrapolation into where a constant probe misbehaves")
    ap.add_argument("--contrast-min", type=float, default=0.35,
                    help="drop chevrons with less contrast than this (leakage "
                         "outpacing the exchange leaves no resonance)")
    ap.add_argument("--amp-points", type=int, default=9,
                    help="drive-strength rows; each is one exact chevron")
    ap.add_argument("--wp-span-MHz", type=float, default=None,
                    help="fixed chevron offset span for EVERY row. Default: size each "
                         "row from its own linewidth (see --span-linewidths)")
    ap.add_argument("--span-linewidths", type=float, default=4.0,
                    help="chevron half-span in estimated linewidths; railed rows are "
                         "re-measured wider automatically")
    ap.add_argument("--wp-points", type=int, default=25)
    ap.add_argument("--drag-shift-points", type=int, default=0,
                    help="with --drag-beat-GHz, also measure the DRAG-induced shift "
                         "at this many drive strengths and fit its power law "
                         "(4 = 'DRAG only adds drive')")
    ap.add_argument("--tg-points", type=int, default=13)
    ap.add_argument("--tg-lo", type=float, default=0.7,
                    help="length-scan window low edge, as a fraction of t_g0 [0.7]")
    ap.add_argument("--tg-hi", type=float, default=1.3,
                    help="length-scan window high edge, as a fraction of t_g0; widen "
                         "when the scan rails [1.3]")
    ap.add_argument("--max-drag-iters", type=int, default=4)
    ap.add_argument("--probe-shape", choices=("constant", "gate"), default="constant",
                    help="step 1's probe. 'constant' measures the law POINTWISE with a "
                         "flat pump. 'gate' plays the gate envelope at each PEAK |eta|: "
                         "~50x less leakage and no extrapolation; its moment-weighted "
                         "rungs are deconvolved to a pointwise k2/k4 "
                         "(stark_chirp.stark_moments)")
    ap.add_argument("--chirp-max-passes", type=int, default=12,
                    help="passes for the chirp<->DRAG FIXED POINT [12] (not the outer "
                         "--max-drag-iters loop). Tolerance is 1e-12 GHz, so strong "
                         "drive can need 20-80 passes")
    ap.add_argument("--moment-weighting", choices=MOMENT_WEIGHTINGS,
                    default="rabi",
                    help="what a shaped rung's chevron centre averages. 'rabi' "
                         "[default] weights by sin(theta(t)), the accumulated Rabi "
                         "angle (derived). 'coupling' and 'uniform' are kept for "
                         "comparison; they overestimate k2 by 19%% and 2.1x")
    ap.add_argument("--cross-check-moments", action="store_true",
                    help="measure the law with BOTH probes at --target-eta (keep it "
                         "weak, <~ 0.5) and report which moment weighting reproduces "
                         "the constant probe's k2. Solves and exits")
    ap.add_argument("--skip-time-rabi", action="store_true")
    ap.add_argument("--coupler-levels", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--gpu", action="store_true",
                    help="run via qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--out", default=None,
                    help="write the record + every stage to this file, under results/ "
                         "unless absolute. HDF5 (a bare name gains .h5); a .json "
                         "suffix gives JSON. Also written when the Rabi guards FAIL")
    ap.add_argument("--plot", nargs="?", const="figs/rabi_chevrons.png", default=None,
                    help="render every chevron, its fit and the shift curve; written "
                         "even when the guards FAIL, and embedded in the --out file")
    ap.add_argument("--plot-ridge", nargs="?", const="figs/chirp_ridge.png", default=None,
                    help="overlay the chirp's pump trajectory on the Rabi map "
                         "(arXiv:2306.10162 Fig. 4). Needs a completed run")
    ap.add_argument("--post-chirp-points", type=int, default=0,
                    help="validate the chirp on the REAL shaped gate across this many "
                         "drive strengths (see post_chirp.py); 0 (default) skips it")
    ap.add_argument("--plot-post-chirp", nargs="?",
                    const="figs/post_chirp_chevrons.png", default=None,
                    help="render the post-chirp validation sweep (needs "
                         "--post-chirp-points > 0)")
    ap.add_argument("--save-point", default=None,
                    help="save the result into the device JSON under this name")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.drag_channel and args.drag_beat_GHz is not None:
        ap.error("--drag-channel and --drag-beat-GHz are two spellings of the same "
                 "setting (--drag-beat-GHz is the one-channel shorthand); pass only one")
    if args.drag_auto and (args.drag_channel or args.drag_beat_GHz is not None):
        ap.error("--drag-auto derives the channels itself; pass it OR "
                 "--drag-channel/--drag-beat-GHz, not both")
    _cli_channels = parse_drag_channels(args.drag_channel)
    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1
    if args.replot is None and (args.device is None or args.target_eta is None):
        ap.error("--device and --target-eta are required unless --replot is given")
    if args.save_point and args.device is None:
        ap.error("--save-point needs --device (to know which device JSON to write)")

    from snail_solver.paths import in_results, resolve_device

    device_path = resolve_device(args.device) if args.device else None

    if args.replot:
        # Everything the plots need is in a previous --out file: no device, no solves.
        saved = load_doc(args.replot)
        device_config = saved.get("device")               # the copy, if it has one
        out = {"operating_point": saved["operating_point"], "stages": saved["stages"],
              "t_g0_ns": saved["t_g0_ns"],
              "drag": saved["stages"].get("drag_loop")}
    else:
        from snail_solver.device_utils import load_device
        from snail_solver.log_utils import setup_run_logger

        logger = setup_run_logger(None, "tune_up")
        config = load_device(device_path)
        if args.coupler_levels is not None:
            config = {**config, "coupler_levels": int(args.coupler_levels)}
        # Stored in the run file: the configuration the solves actually ran with.
        device_config = config

        # With --plot-ridge (and no explicit --wp-span-MHz), put every row on one
        # fixed offset axis.
        if args.plot_ridge and args.wp_span_MHz is None:
            args.wp_span_MHz, _want = ridge_span_MHz(
                config, args.target_eta, eta_lo=args.eta_lo, eta_hi=args.eta_hi,
                span_linewidths=args.span_linewidths, wp_points=args.wp_points,
                logger=logger)
            logger.info(f"  --plot-ridge: fixing wp_span_MHz="
                        f"{args.wp_span_MHz:.2f} so every row shares one offset axis")

        from snail_solver import find_stark_resonance as FSR
        print(f"device={args.device}  target_eta={args.target_eta}  "
              f"jobs={FSR._resolve_jobs(args.jobs)}{' GPU' if args.gpu else ''}")

        if args.cross_check_moments:
            out = cross_check_probe_moments(
                config, args.target_eta, eta_lo=args.eta_lo, eta_hi=args.eta_hi,
                amp_points=args.amp_points, wp_points=args.wp_points,
                wp_span_MHz=args.wp_span_MHz, span_linewidths=args.span_linewidths,
                n_time=args.n_time, window_tg=args.window_tg,
                contrast_min=args.contrast_min, jobs=args.jobs,
                spec_abs_GHz=args.spec_abs_GHz, logger=logger)
            print(f"\nbest moment weighting: {out['best']!r}")
            for w, v in out["shaped"].items():
                print(f"  {w:9s} M2={v['M2']:.4f} M4={v['M4']:.4f}  "
                      f"k2={v['k2']:+.4f} ({100 * v['k2_rel_err']:+.1f}%)  "
                      f"k4={v['k4']:+.4f} ({100 * v['k4_rel_err']:+.1f}%)")
            print(f"  constant  k2={out['constant']['k2']:+.4f} "
                  f"k4={out['constant']['k4']:+.4f}  <- reference")
            return

        if args.drag_auto:
            # Audited at the NOMINAL length; verdicts move with t_g, the ranking does not.
            from snail_solver.spectator_audit import (print_channel_audit,
                                                      select_drag_channels)
            _cli_channels, _audit = select_drag_channels(
                config, nominal_t_g(config, args.target_eta),
                max_channels=args.max_drag_channels,
                spec_abs_GHz=args.spec_abs_GHz)
            print_channel_audit(_audit)
            if _audit["blocking"] and not args.force:
                ap.error(
                    f"{len(_audit['blocking'])} channel(s) are NOT PERTURBATIVE at this "
                    f"operating point (g/|det| >= 1), so DRAG has no leading term to "
                    f"cancel: "
                    + "; ".join(f"{b['name']} (g={b['g_MHz']:.2f} MHz, "
                                f"det={b['detuning_MHz']:.2f} MHz)"
                                for b in _audit["blocking"][:3])
                    + ". Move the operating point, or pass --force to solve anyway.")

        try:
            out = run_tune_up(
                config, args.target_eta, drag_beat_GHz=args.drag_beat_GHz,
                drag_n_pump=args.drag_n_pump,
                drag_channels=_cli_channels,
                spec_abs_GHz=args.spec_abs_GHz,
                chirp_degree=args.chirp_degree, quartic_warn=args.quartic_warn,
                window_tg=args.window_tg, n_time=args.n_time,
                span_linewidths=args.span_linewidths,
                drag_shift_points=args.drag_shift_points,
                eta_lo=args.eta_lo, eta_hi=args.eta_hi, amp_points=args.amp_points,
                contrast_min=args.contrast_min,
                probe_shape=args.probe_shape,
                moment_weighting=args.moment_weighting,
                chirp_max_passes=args.chirp_max_passes,
                wp_span_MHz=args.wp_span_MHz, wp_points=args.wp_points,
                tg_points=args.tg_points, tg_lo=args.tg_lo, tg_hi=args.tg_hi,
                max_drag_iters=args.max_drag_iters,
                do_time_rabi=not args.skip_time_rabi, jobs=args.jobs,
                post_chirp_points=args.post_chirp_points,
                solver={"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps},
                logger=logger)
        except RabiFitError as exc:
            # Only the interpretation failed: save and draw the sweep before dying,
            # so "inspect the chevrons" is actionable.
            written = (save_doc(in_results(args.out),
                                {"stages": {"rabi": exc.table},
                                 "device": device_config},
                                attrs=_run_attrs(device_path,
                                                 status="rabi_fit_failed",
                                                 error=str(exc)))
                       if args.out else None)
            if written:
                print(f"  wrote {written} (the sweep that failed)")
            if args.plot:
                fig = plot_rabi_table(exc.table, args.plot)
                print(f"  wrote {fig} (the sweep that failed)")
                if written and is_hdf5(written):
                    attach_figures(written, {"rabi": fig})
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

    written = None
    if args.out:
        doc = {"operating_point": rec, "t_g0_ns": out["t_g0_ns"],
               "stages": out["stages"]}
        if device_config is not None:
            doc["device"] = device_config
        written = save_doc(in_results(args.out), doc, attrs=_run_attrs(device_path))
        print(f"  written {written}")

    # Every figure below is also embedded in the run file (one run, one artefact).
    figs: Dict[str, str] = {}

    if args.plot:
        figs["rabi"] = plot_rabi_table(out["stages"]["rabi"], args.plot)
        print(f"  wrote {figs['rabi']}")

    if args.plot_ridge:
        figs["chirp_ridge"] = plot_chirp_ridge(
            out["stages"]["rabi"], out["stages"]["chirp"], rec["wp_offset_GHz"],
            rec["t_g_ns"], args.plot_ridge)
        print(f"  wrote {figs['chirp_ridge']}")

    if args.plot_post_chirp:
        post = out["stages"].get("post_chirp")
        if post:
            figs["post_chirp"] = plot_post_chirp_table(
                post, out=args.plot_post_chirp, rabi_table=out["stages"]["rabi"])
            print(f"  wrote {figs['post_chirp']}")
        else:
            print("  --plot-post-chirp given but no post_chirp stage ran "
                  "(pass --post-chirp-points > 0)")

    # With no --out, a --replot re-embeds into the document it drew FROM.
    dest = written or args.replot
    if figs and dest:
        if is_hdf5(split_address(dest)[0]):
            n = attach_figures(dest, figs)
            print(f"  embedded {n} figure(s) in {dest} "
                  f"(python -m snail_solver.h5_io {dest} --extract DIR)")
        elif written:
            print("  (--out is JSON, which cannot hold the figures; use an HDF5 "
                  "--out to keep them with the run)")

    if args.save_point:
        from snail_solver.operating_points import save_point
        save_point(device_path, args.save_point, rec, overwrite=args.overwrite)
        print(f"  saved operating point {args.save_point!r} to {device_path}")


if __name__ == "__main__":
    main()
