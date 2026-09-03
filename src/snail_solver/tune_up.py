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


def area_factor(config: Dict[str, Any]) -> float:
    """``f = area / (amp * t_g)`` for this device's envelope shape.

    The whole fixed-amplitude algebra below rests on the relation between a pulse's
    AREA (which ``normalize_iswap`` pins) and its PEAK (which this module holds
    fixed). For a Hann window that ratio is exactly 1/2, which is why the two
    functions here have read ``2 A / t_g`` since they were written.

    That constant is a property of the SHAPE, not of the physics, so it has to move
    when the shape does -- e.g. to the Li/Calarco/Motzoi Eq. (13) ramp that recursive
    DRAG requires (:class:`envelope.SinePowerRamp`), whose factor depends on ``m`` and
    the rise time. Returns exactly 0.5 for the raised cosine, so every existing
    call site is bit-for-bit unchanged.
    """
    kind = config.get("envelope", "raised_cosine")
    if kind == "raised_cosine":
        return 0.5
    from snail_solver.envelope import ENVELOPE_KINDS
    cls = ENVELOPE_KINDS.get(kind)
    if cls is None or kind == "constant":
        return 1.0
    # Built at t_g = 1 with the rise as a FRACTION of the gate. That is exactly why
    # the config carries a fraction rather than a rise time in ns: a fixed absolute
    # t_rise would make this factor t_g-dependent, and with it the Stark shift and
    # the chirp -- destroying the length/frequency decoupling the whole module rests
    # on (see the "Why fix the amplitude" section of the module docstring).
    kw = ({"m": int(config.get("envelope_m", 3)),
           "t_rise": float(config.get("envelope_rise_frac", 0.5))}
          if cls.__name__ == "SinePowerRamp" else {})
    env = cls(amp=1.0, t_g=1.0, **kw)
    return float(env.area())              # amp = t_g = 1, so area IS the factor


def fixed_eta_amp_scale(config: Dict[str, Any], t_g: float,
                        target_eta: float) -> float:
    """``amp_scale`` that holds the physical peak |eta| at `target_eta` at this `t_g`.

    Equals exactly 1.0 at ``t_g = auto_t_g(..., target_eta)``.

    Note the direction: LONGER t_g -> LARGER amp_scale. `normalize_iswap` shrinks the
    amplitude as 1/t_g to hold the pulse area, so holding the PEAK requires scaling
    back up. Getting this backwards still yields a plausible length-Rabi curve.
    """
    return (float(target_eta) * float(t_g) * area_factor(config)) / _area(config)


def peak_eta_of(config: Dict[str, Any], t_g: float, amp_scale: float) -> float:
    """Physical peak |eta| for a normalized pulse at (t_g, amp_scale).

    The inverse of :func:`fixed_eta_amp_scale`; used to convert a calibration map's
    amp_scale axis into physical drive. Unaffected by a chirp (a pure phase).

    .. warning::
       It is also unaffected by FIRST-ORDER DRAG, because a Hann window has
       ``deta/dt = 0`` at its peak -- which is what this docstring used to claim
       outright. That no longer holds under RECURSIVE DRAG: two nested corrections
       contribute ``-eta''/(Delta_a Delta_b)``, and ``eta''`` at the peak is not zero.
       So with several channels the true peak |eta| carries a DRAG-dependent term
       this function does not model, and it propagates into
       :func:`fixed_eta_amp_scale` and hence the fixed-|eta| premise of the whole
       tune-up. Use :func:`device_utils.drag_correction_ratio` to see how large it is.
    """
    return float(amp_scale) * _area(config) / (float(t_g) * area_factor(config))


def nominal_t_g(config: Dict[str, Any], target_eta: float) -> float:
    """t_g0 = 2A/eta*, the analytic full-iSWAP length at this drive strength."""
    return auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                    float(config["lam_b"]), float(target_eta))


def fixed_span_MHz(config: Dict[str, Any], target_eta: float, eta_hi: float = 1.0,
                   span_linewidths: float = 4.0) -> float:
    """A single ``wp_span_MHz`` wide enough for every row in a Rabi sweep.

    Linewidth is the exchange rate, ``Omega ~ 1/(2 t_g(|eta|))``, and ``t_g`` SHRINKS
    as the drive strengthens -- so the STRONGEST row (``eta_hi * target_eta``) needs
    the widest span, and a span picked for any weaker row would rail it. Pass this
    to ``rabi_shift_table``/``run_tune_up`` as ``wp_span_MHz`` to put every row on
    the SAME frequency grid -- no per-row resizing, no resampling needed to plot
    them together -- at the cost of oversampling the weaker rows' much narrower
    peaks. This is what :func:`plot_chirp_ridge` requires: it does not interpolate,
    so every chevron in the table must already share one offset axis.
    """
    e_max = float(eta_hi) * float(target_eta)
    linewidth_MHz = 1e3 / (2.0 * nominal_t_g(config, e_max))
    return 2.0 * float(span_linewidths) * linewidth_MHz


def ridge_span_MHz(config: Dict[str, Any], target_eta: float, *, eta_lo: float = 0.3,
                   eta_hi: float = 1.0, span_linewidths: float = 4.0,
                   wp_points: int = 25, pts_per_hwhm: float = 3.0,
                   logger=None) -> tuple:
    """The fixed ``wp_span_MHz`` :func:`plot_chirp_ridge` needs, AND the ``wp_points``
    that span demands -- returned together because using one without the other is a
    trap.

    :func:`fixed_span_MHz` alone is not enough. Going from per-row adaptive spans to
    one fixed span does not just oversample the weak rows, it UNDERSAMPLES them
    relative to their own linewidth, because the span is sized for the strongest row
    and the HWHM shrinks with the drive:

        pts per HWHM (adaptive, any row) = (wp_points - 1) / (2 span_linewidths)
        pts per HWHM (fixed, weakest row) = that x (eta_lo / eta_hi)

    At the defaults (wp_points=25, span_linewidths=4) the adaptive scheme gives every
    row 3.0 points per HWHM; the fixed span gives the weakest row 3.0 eta_lo/eta_hi
    -- only 0.9 at eta_lo=0.3. A Lorentzian fitted to ~1 point on the feature is not
    a fit: :func:`fit_chevron_center` returns junk, :func:`chevron_quality` drops the
    row as ``poor_fit``, and if enough rows survive to be fitted anyway the ridge
    fails the r2 floor in :func:`rabi_shift_table`. That trades a loud ValueError
    from the plotter for a confidently WRONG chirp, which is strictly worse.

    So the compensating point count is

        wp_points >= 1 + 2 span_linewidths pts_per_hwhm (eta_hi / eta_lo)

    = 61 at eta_lo=0.4 and 81 at eta_lo=0.3, versus 25 for the adaptive default. That
    is affordable: :func:`find_stark_resonance.scan` fans out over pump offsets, so
    on a many-core box extra ``wp_points`` cost no wall-clock at all, while
    ``amp_points`` (the rows) are serial and do.

    What "HWHM" means here, and why this is deliberately conservative
    ----------------------------------------------------------------
    ``pts_per_hwhm`` is counted against the MODEL linewidth :func:`fixed_span_MHz`
    uses -- ``1e3 / (2 nominal_t_g(|eta|))`` -- not against the Lorentzian HWHM
    :func:`fit_chevron_center` actually fits. The two differ by roughly a constant
    factor: on 1Gate4.2SNAIL at target_eta=1.2 the weakest row (|eta|=0.48) has a
    model linewidth of 1.73 MHz against a fitted HWHM of 2.88 MHz, i.e. ~1.6x
    broader, so its measured sampling was 2.0 points per true HWHM where the
    warning below reported 1.20.

    That factor cancels out of everything this function decides. It is common to
    both rows, so the adaptive scheme still lands every row on the SAME sampling
    whatever the drive, and the fixed span still costs the weakest row exactly
    ``eta_lo / eta_hi`` of it -- the ratio is what sets ``want``. Only the absolute
    number is pessimistic, and in the safe direction: the default
    ``pts_per_hwhm=3.0`` buys about 4.8 points across the true HWHM. Do not "correct"
    it by lowering the default without re-measuring fitted HWHMs on the device in
    hand; the 1.6 is an observation, not a derivation.

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
    """Check whether the nominal ``eta`` this whole pipeline is built on still
    equals the coupler mode's real ``sqrt(<n_s>)``.

    Every pump tone here is built with ``is_eta=True`` (see
    :func:`find_stark_resonance.build_chevron_coupler`), which DECLARES the
    envelope amplitude to be ``eta`` and never derives it from a photon number.
    ``eta`` is supposed to mean ``sqrt(n_s)``, the coherent photon-number
    amplitude of the driven SNAIL/coupler mode (McKinney et al., "Spectator-Aware
    Frequency Allocation in Tunable-Coupler Quantum Architectures",
    arXiv:2409.18262, Eq. 9) -- and the two-qubit exchange rate this module's
    whole amplitude/length algebra is built on, ``g_eff = 6 g3 lam_a lam_b eta``,
    matches that paper's Eq. 10 exactly. But nothing here has ever checked that
    the LABEL and the coupler's actual occupation agree once multi-photon
    channels start populating the coupler -- which is exactly the regime where
    the Rabi chevrons stop being two-level Lorentzians (see the module docstring
    and ``chevron_quality``).

    CAVEAT (unresolved): in this module's actual multi-mode Hamiltonian
    (``ZhouCoupler.dressed_flux``), the coupler is just one more dynamical mode
    oscillating at its own bare frequency ``w_s``, and ``eta_p(t)`` is a SEPARATE
    classical term at the pump frequency ``w_p`` -- they are not obviously the
    same object. Whether the coupler's OWN Fock occupation (what this function
    measures) is the quantity McKinney et al.'s ``eta = sqrt(n_s)`` refers to, or
    a different induced-leakage channel entirely, was not resolved by comparing
    against Zhou's own paper (arXiv:2306.10162, a single-qubit subharmonic-drive
    process, not a two-qubit coupler gate) -- treat the ``ratio`` this returns as
    "how much the coupler mode itself gets incidentally populated", a real and
    useful number on its own terms, not as a validated test of a labelling bug.

    Each row evolves the coupler from bare VACUUM (no qubit excitation), so any
    coupler photon-number growth is attributable to the pump alone, at a fixed
    gate length ``t_g = nominal_t_g(target_eta)`` with ``amp_scale`` dialed via
    :func:`fixed_eta_amp_scale` to reach each row's peak ``eta`` -- i.e. the same
    pulse `run_tune_up` would actually build, just probed at different drives.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    target_eta : float
        The operating peak |eta| to probe around.
    eta_fracs : sequence of float
        Fractions of `target_eta` to probe; default spans below, at, and just
        past the operating point.
    shape : str, default "raised_cosine"
        Probe pulse: the actual shaped gate, or "constant" for the Rabi-sweep
        probe shape.
    n_time : int
        Output times per row.
    window_tg : float
        For ``shape="constant"`` only: window in swaps (see :func:`rabi_shift_table`).
    spec_abs_GHz : float, optional
        Spectator frequency, if the device configuration includes one.
    solver : dict, optional
        QuTiP tolerances.

    Returns
    -------
    dict
        ``eta_nominal``, ``sqrt_ns_peak``, ``ratio = sqrt_ns_peak / eta_nominal``,
        ``target_eta``, ``shape``, and ``rows`` (per-row ``times_ns``, ``n_s_t``,
        ``eta_nominal``, ``t_g_ns``) for :func:`plot_eta_vs_ns`.
    """
    from snail_solver import find_stark_resonance as FSR

    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    t_g = nominal_t_g(config, target_eta)
    eta = np.asarray(eta_fracs, dtype=float) * float(target_eta)

    eta_nominal = np.full(eta.size, np.nan)
    sqrt_ns_peak = np.full(eta.size, np.nan)
    rows = []
    for i, e in enumerate(eta):
        if shape == "raised_cosine":
            amp_scale = fixed_eta_amp_scale(config, t_g, float(e))
            window_ns = 1.05 * t_g
            cpl, _w_p = FSR.build_chevron_coupler(
                config, float(e), 0.0, window_ns, spec_abs_GHz=spec_abs_GHz,
                shape="raised_cosine", t_g_ns=t_g, amp_scale=amp_scale)
            t_g_row = t_g
        else:
            window_ns = float(window_tg) * nominal_t_g(config, float(e))
            cpl, _w_p = FSR.build_chevron_coupler(
                config, float(e), 0.0, window_ns, spec_abs_GHz=spec_abs_GHz,
                shape="constant")
            t_g_row = window_ns

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
    """Render :func:`verify_eta_matches_ns`: does the nominal ``eta`` still mean
    ``sqrt(n_s)``?

    Top row: per-drive-strength overlay of the measured coupler-mode
    ``sqrt(<n_s(t)>)`` against the nominal ``|eta(t)|`` envelope the pulse was
    built to have. Bottom panel: measured ``sqrt(max n_s)`` vs nominal ``eta``
    with a ``y = x`` reference -- departure from that line, especially one that
    GROWS with drive, means the eta axis this whole pipeline is built on is not
    measuring what it claims to at that drive.

    Returns
    -------
    str
        The path written.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from snail_solver.envelope import RaisedCosine
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:                                        # style is a nicety
        pass

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
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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


def shift_curve_stability(eta: np.ndarray, delta_MHz: np.ndarray,
                          weights: Optional[np.ndarray] = None,
                          target_eta: Optional[float] = None, *,
                          cutoffs: Sequence[float] = (1.0, 0.9, 0.8, 0.7),
                          fit_static: bool = True) -> Dict[str, Any]:
    """Refit ``delta(|eta|)`` on nested weakest-drive subsets; report how much
    the extrapolated ``delta(target_eta)`` moves.

    :func:`fit_shift_curve`'s ``r2``/residual guards can pass a fit that is, in
    fact, underdetermined: a good ``r2`` over a MIX of clean and contaminated
    rows says nothing about whether the FEW highest-drive rows -- exactly the
    ones a chirp at ``target_eta`` depends on most -- are trustworthy. This is
    the direct test: if dropping the top 10-30% of measured rows swings the
    fitted shift at ``target_eta`` by a large fraction of its own value, the
    fit has not determined that value; it has merely found *a* curve through
    noisy data.

    This catches contamination of an INTERPOLATING fit, which
    ``extrapolation_ratio`` (which compares ``target_eta`` to the largest
    MEASURED ``|eta|``) structurally cannot: a sweep that measured up to or
    past ``target_eta`` reports ``extrapolation_ratio <= 1`` and looks fine by
    that guard alone, even when the top rows it measured are themselves not
    resonances.

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


def chevron_quality(cen: Dict[str, Any], offsets_GHz: np.ndarray, metric: np.ndarray,
                    span_MHz: float, *, leak: Optional[float] = None,
                    contrast_min: float = 0.35, gap_frac_max: float = 0.5,
                    nrmse_max: float = 0.1, secondary_max: float = 0.4,
                    leak_max: float = 0.35) -> Dict[str, Any]:
    """How much of a two-level Lorentzian this chevron actually is.

    :func:`fit_chevron_center` already computes everything needed to answer
    that -- the Lorentzian centre, the parabolic vertex, the fit rmse -- and
    today spends it on a plot annotation (see the module docstring: "when they
    separate ... the lineshape isn't Omega^2/(Omega^2+delta^2)"). Five
    independent, cheap checks, each from arrays already in hand:

    * ``contrast`` -- unchanged from the existing floor (max - min of the
      envelope).
    * ``gap_frac`` -- ``|center - vertex|``, normalised by the linewidth (NOT by
      the shift itself, since the shift is the thing being measured and can't
      calibrate its own error bar). Weak ALONE: when the Lorentzian fit fails
      outright, :func:`fit_chevron_center` falls back to ``vertex_GHz`` for
      BOTH fields, so a failed fit can show ``gap_frac == 0``.
    * ``nrmse`` -- the fit residual (``cen["rmse"]``) normalised by contrast;
      the same ratio :func:`fit_chevron_center`'s internal ``ok`` test already
      thresholds at 0.1, promoted here to a returned, tunable number.
    * ``secondary`` -- the second-highest local maximum of ``metric`` (above
      30% of the global max), as a fraction of the global max; 0 for a clean
      unimodal chevron. This is what actually catches a bimodal/multi-branch
      chevron -- exactly the failure mode a raw centre/vertex gap misses when
      the fit fails and both estimators collapse onto the same wrong peak.
    * ``hwhm_frac`` -- ``hwhm_MHz / span_MHz``: a width comparable to the scan
      window is not really constrained. Reported, not currently used to reject
      (folded into ``nrmse``/``gap_frac`` in practice).
    * ``leak`` -- the caller-supplied leakage read-out at the resonance (e.g.
      ``scan``'s ``leak_on_resonance``), if given.

    Returns a composite ``weight`` in ``[0, 1]`` (reduces to plain ``contrast``
    on a clean row with ``leak=0``) and a ``reject`` reason string (or
    ``None``). ``contrast_min`` stays one of the checks here, so today's drop
    behaviour is a strict subset of this function's.

    Parameters
    ----------
    cen : dict
        Output of :func:`fit_chevron_center`.
    offsets_GHz, metric : ndarray
        The chevron's offset axis and envelope (max-over-time, or the
        at-``t_g`` metric for a shaped probe).
    span_MHz : float
        This row's scan span, for ``hwhm_frac``.
    leak : float, optional
        Leakage at the resonance.
    contrast_min, gap_frac_max, nrmse_max, secondary_max, leak_max : float
        Thresholds; a row failing any one is rejected (first failing reason
        wins, checked in the order above).

    Returns
    -------
    dict
        ``contrast``, ``gap_frac``, ``nrmse``, ``secondary``, ``hwhm_frac``,
        ``leak``, ``weight``, ``reject``.
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

    # secondary peak: local maxima of y above 30% of the global max, excluding
    # the global max itself.
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
    gap_frac_max, nrmse_max, secondary_max, leak_max : float
        Additional :func:`chevron_quality` drop thresholds, checked alongside
        ``contrast_min`` -- a row failing ANY of them is dropped the same way a
        low-contrast row always was. See :func:`chevron_quality` for what each
        one catches (bimodal chevrons, failed Lorentzian fits, disagreeing
        centre/vertex estimators, high leakage at the resonance).
    stability_max : float
        Warning-only threshold on :func:`shift_curve_stability`'s
        ``delta_spread`` -- how much the extrapolated shift at `target_eta`
        moves when the fit is refit on nested weakest-drive subsets. Never
        raises (only ``r2_min`` and the drive-span guard below do); this is
        purely diagnostic, logged and recorded in the returned dict.
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
                            drag_n_pump=drag_n_pump, eta_op=float(e),
                            keep_full_channels=True)
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
        leakage[i] = float(chev["leak_on_resonance"])
        j_res = int(np.argmin(np.abs(np.asarray(chev["offsets_GHz"], dtype=float)
                                     - cen["center_GHz"])))
        leak_breakdown = {"f_a": float(chev["leak_f_a"][j_res]),
                          "f_b": float(chev["leak_f_b"][j_res]),
                          "coupler": float(chev["leak_coupler"][j_res]),
                          "double": float(chev["leak_double"][j_res]),
                          "spectator": float(chev["leak_spectator"][j_res])}
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
        # No contrast (or a bimodal/poorly-fit/high-leakage chevron) means no
        # resonance: at strong drive leakage can outpace the exchange, and one such
        # row would otherwise set k2, k4 for the whole chirp. fit_shift_curve ignores
        # NaNs, so dropping it here is enough.
        if q["reject"]:
            ridge[i] = np.nan
            if logger:
                logger.info(f"  rabi row {i + 1}/{eta.size}: |eta|={e:.4f} DROPPED "
                            f"({q['reject']}) -- contrast {contrast[i]:.3f}, "
                            f"leak {leakage[i]:.3f} (f_a {leak_breakdown['f_a']:.3f} "
                            f"f_b {leak_breakdown['f_b']:.3f} "
                            f"coupler {leak_breakdown['coupler']:.3f} "
                            f"|11> {leak_breakdown['double']:.3f})")
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
                        f"leak {leakage[i]:.3f} (f_a {leak_breakdown['f_a']:.3f} "
                        f"f_b {leak_breakdown['f_b']:.3f} "
                        f"coupler {leak_breakdown['coupler']:.3f} "
                        f"|11> {leak_breakdown['double']:.3f}))")

    partial = {"eta": eta, "delta_MHz": ridge, "t_g_ref_ns": t_g0,
               "target_eta": float(target_eta), "contrast": contrast,
               "quality": quality, "leakage": leakage,
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

    # Too few surviving rows is a RabiFitError like every other one here: the
    # measurement ran, only the interpretation failed. Re-raised with the table
    # attached so main() can still save and DRAW the chevrons -- which is the whole
    # point, since "which rows were dropped, and why" is the diagnosis. A bare
    # ValueError escaping here left the one device that hit it un-inspectable.
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
    partial["fit"] = fit
    stability = shift_curve_stability(eta, ridge, weights=quality, target_eta=target_eta,
                                      cutoffs=stability_cutoffs)
    partial["stability"] = stability
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
        # Warn-only: unlike r2_min/the span guard above, this never raises -- it
        # flags that the extrapolated shift at target_eta is itself underdetermined
        # even when r2 looks fine, which r2 alone cannot see (see
        # shift_curve_stability's docstring).
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
    photon count in ``F^(n)``; ``N`` defaults to ``K`` because for every channel this
    device produces they are the same integer -- but they stay separately settable,
    since they enter the substitution in different places.

    Returns None for no channels, so the caller falls through to the scalar
    ``--drag-beat-GHz`` shorthand.
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
                    shape_kw: Optional[Dict[str, Any]] = None):
    """Unit-amplitude envelope on ``t_g = 2``, so ``t = u + 1`` maps [-1,1] -> [0,2].

    Used to read the pulse shape in NORMALIZED gate time. t_g = 2 is arbitrary and
    harmless precisely because every envelope here is t_g-independent in u -- the
    property the module docstring's "Why fix the amplitude" section depends on.
    """
    from snail_solver.envelope import ENVELOPE_KINDS
    cls = ENVELOPE_KINDS.get(str(shape))
    if cls is None:
        raise ValueError(f"unknown envelope shape {shape!r}; "
                         f"known: {sorted(ENVELOPE_KINDS)}")
    kw = dict(shape_kw or {})
    if cls.__name__ == "SinePowerRamp":
        # rise given as a FRACTION of the gate; t_g = 2 here
        kw = {"m": int(kw.get("m", 3)),
              "t_rise": 2.0 * float(kw.get("rise_frac", 0.5))}
    return cls(amp=1.0, t_g=2.0, **kw)


def shape_config(config: Dict[str, Any]) -> tuple:
    """``(shape, shape_kw)`` for this device, to pass to the chirp projection."""
    kind = str(config.get("envelope", "raised_cosine"))
    kw = ({"m": int(config.get("envelope_m", 3)),
           "rise_frac": float(config.get("envelope_rise_frac", 0.5))}
          if kind == "sine_power" else {})
    return kind, kw


def _resolve_drag_channels(beat_GHz: Optional[float], n_pump: int = 1,
                           channels: Optional[Sequence[Any]] = None) -> tuple:
    """Normalize this module's DRAG arguments to a channel tuple, innermost-first.

    Same resolution rule as :meth:`envelope.PumpTone.drag_channels_resolved`, so the
    calibration and the pulse cannot disagree about which processes are suppressed:
    an explicit `channels` list wins, else the scalar `beat_GHz`/`n_pump` shorthand,
    else DRAG is off (empty tuple).
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
                              quartic_warn: float = 0.25) -> Dict[str, Any]:
    """Step 2: project the measured shift onto the Legendre chirp basis.

    Evaluates the fitted shift along the pulse -- ``|eta(u)| = eta* cos^2(pi u / 2)``
    for a Hann envelope -- and projects delta(u) onto Legendre polynomials by
    Gauss-Legendre quadrature. The QUADRATURE is exact regardless of `degree` (a
    modest node count integrates any of this to machine precision), but delta(u)
    as a function of u is itself a degree-8 polynomial in cos(pi u / 2) -- so the
    Legendre series in u does not terminate below ``degree=8``. The module default
    IS 8 (an exact reconstruction of the already-fit k2/k4 curve -- algebra on
    k2/k4, not an added fit, so it costs nothing); pass a lower `degree` only to
    deliberately truncate (:func:`plot_chirp_ridge` shows the resulting small
    overshoot of the chirp above the measured ridge near |eta| -> 0). Pointwise
    evaluation needs no deconvolution because the measured curve is the
    INSTANTANEOUS shift -- the whole reason :func:`rabi_shift_table` uses a
    constant probe.

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
        polynomial extrapolated past where it was fitted), ``perturbative_ok``
        (``quartic_fraction < quartic_warn`` -- see below), and with DRAG on
        ``drag_iters``, ``drag_delta_frac`` (how much of the shift the quadrature
        adds) and ``min_abs_detuning_GHz``.

    Convergence of the eta^2 + eta^4 law itself
    --------------------------------------------
    ``delta = k2 eta^2 (1 + (k4/k2) eta^2)`` is a TRUNCATED series in the drive.
    ``quartic_fraction = |k4 eta*^4 / k2 eta*^2|`` is how large the last kept term
    is relative to the first -- when it is not small, the unmeasured eta^6 term is
    plausibly of the same order, and the law carries little information about
    delta(eta*) beyond what a wild guess would. ``perturbative_ok`` flags this at
    `quartic_warn` (default 0.25: the correction is a quarter of the leading term,
    so the next unmeasured term is plausibly ~6%, tolerable). This is computed
    FROM the fit, so it inherits the fit's own instability -- see
    ``shift_curve_stability`` for a check that is independent of this one; trust
    ``perturbative_ok`` only once the Rabi rows themselves have been through
    ``chevron_quality``.
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
    # |eta(u)| / eta*. The closed form cos^2(pi u / 2) is the HANN case; read it off
    # the actual envelope so selecting a different base shape (which recursive DRAG
    # requires past one channel -- see envelope.SinePowerRamp) cannot leave the chirp
    # tracking a pulse the solver is not playing. Evaluated at t_g = 2 so u = t - 1:
    # every envelope here is t_g-independent in normalized gate time, which is the
    # property this whole module's amplitude/length decoupling rests on.
    shape_env = _shape_envelope(shape, shape_kw)
    s = np.abs(np.asarray(shape_env.value_at(u + 1.0, np), dtype=complex))
    amp = eta_star * s

    def shift_GHz(a):                                      # MHz law -> GHz
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
        # The pulse the SOLVER will play, built the same way it builds it, rather
        # than a hand-derived model of it. Previously this line differentiated the
        # Hann by hand and formed sqrt(amp^2 + q^2) -- which is |amp - i q| ONLY
        # because a first-order correction is purely imaginary. Recursive DRAG's
        # correction has a real part (-eta''/(Da Db)), so abs() is the general form
        # and the old expression is simply wrong beyond one channel.
        env = _shape_envelope(shape, shape_kw)
        env.amp, env.t_g = eta_star, t_g
        order = _drag.required_order(channels)
        shape = env.jet_at(t_g * (u + 1.0) / 2.0, order, np)
        min_abs = float("inf")
        for it in range(int(max_iters)):
            # Iterate on the LEGENDRE COEFFICIENTS, not on sampled values: the
            # detuning JET needs d/dt of the current chirp iterate, and only a
            # coefficient representation has an analytic derivative.
            #
            # Projected at a HIGHER degree than the chirp we ultimately emit.
            # delta(u) is built from cos^4/cos^8(pi u / 2), whose Legendre series does
            # not actually terminate, so truncating at `degree` leaves ripple near
            # |u| = 1 -- and that ripple lands in a PHYSICAL denominator. At degree 8
            # it is enough to flip the sign of Delta - Delta_0 on a 50 MHz beat
            # (by ~6 kHz), which would misreport the singularity guard and invert the
            # k-scaling the beat is supposed to show. The output chirp is still
            # truncated to `degree`; only Delta's internal representation is refined.
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
            delta_GHz = new
            if step < tol_GHz:
                break
        else:
            raise RuntimeError(
                f"the chirp<->DRAG fixed point did not settle in {max_iters} passes "
                f"(last step {step:.2e} GHz, min|Delta(t)| = {min_abs * 1e3:.3f} MHz, "
                f"{len(channels)} channel(s)). Near a collision the quadrature and the "
                f"chirp can chase each other; pick a further-detuned beat or a weaker "
                f"drive. Note the d-th nested correction scales as 1/t_g^d, so a deeper "
                f"recursion couples the chirp and the length more tightly and may need "
                f"more passes.")
        drag_norm = float(np.linalg.norm(delta_GHz))
        extra = {"drag_iters": it + 1, "min_abs_detuning_GHz": min_abs,
                 "min_abs_detuning_per_channel_GHz": floors,
                 "n_drag_channels": len(channels),
                 "drag_correction_ratio": float(
                     np.max(np.abs(eta_tot - amp)) / max(float(np.max(amp)), 1e-30)),
                 "drag_delta_frac": ((drag_norm - base_norm) / base_norm
                                     if base_norm else float("nan"))}

    coeffs = _project(delta_GHz, u, w, degree)
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
            "rel_diff": rel_diff, "quartic_fraction": float(quartic),
            "perturbative_ok": bool(quartic < quartic_warn), **extra,
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
#: Categorical slot 3 (aqua). Leakage is ALWAYS this colour, in every panel of
#: every figure -- fixed by role, same discipline as the three constants above.
_C_LEAK = "#1baf7a"


def plot_rabi_table(table: Dict[str, Any], out: str = "figs/rabi_chevrons.png",
                    title: Optional[str] = None) -> str:
    """Render the Rabi sweep: every chevron, its envelope fit, and the shift curve.

    One row per drive strength, left to right: the raw chevron; the leakage raster
    (``P_leak = norm - P01 - P10``, see :func:`find_stark_resonance.population_channels`)
    on the SAME offset/time grid; the Rabi oscillation on resonance and one linewidth
    off it; and the max-over-time envelope with its fitted Lorentzian. The bottom
    band is the result: located resonance vs |eta| (left) and leakage vs |eta|
    (right), with the ``delta0 + k2|eta|^2 + k4|eta|^4`` curve through the former.

    The third column is the measurement in its rawest form -- detuning speeds the
    oscillation up and shrinks it (``Omega_eff = sqrt(Omega^2 + d^2)``, peak
    ``Omega^2/Omega_eff^2``), and the fourth column fits the peak of that envelope
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
    from matplotlib.colors import LinearSegmentedColormap
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
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_chirp_ridge(table: Dict[str, Any], proj: Dict[str, Any], wp_offset_GHz: float,
                     t_g: float, out: str = "figs/chirp_ridge.png",
                     title: Optional[str] = None) -> str:
    """Overlay the chirp's pump trajectory on the Rabi map itself, not just its fit.

    This is the Fig. 4b picture from Qiu et al. 2023 (arXiv:2306.10162): pump
    frequency on x, drive strength on y, and the Rabi signal itself as the color --
    the SAME per-row ``offsets_GHz``/``metric`` data :func:`plot_rabi_table` draws
    as separate per-amplitude panels, here stitched into one map. The ac-Stark-
    shifted ridge is the bright band curving through it. A frequency-modulated
    drive tracks that band through the pulse instead of crossing it at one fixed
    frequency: as the Hann/raised-cosine envelope |eta(t)| rises from 0 to the peak
    and back down, the chirp's instantaneous frequency traces along the ridge
    almost exactly -- by construction, since that's the curve the chirp was
    Legendre-projected from (:func:`chirp_from_measured_shift`). A flat, unchirped
    carrier at the same mean offset is drawn for contrast: the gap that opens up
    between it and the ridge as |eta(t)| rises is exactly the resonance error a
    chirp removes.

    NO INTERPOLATION: every row is stacked at its own real, measured offsets, with
    no resampling in either direction. That means every chevron in `table` MUST
    already share the identical offset axis -- which only happens when the sweep
    was run with a FIXED ``wp_span_MHz`` (:func:`fixed_span_MHz` picks one wide
    enough for the whole drive range), not the per-row adaptive default. A "high
    definition" map is a resolution problem, solved by running the sweep with more
    ``amp_points``/``wp_points`` on a fixed span, not by rendering trickery.

    This is a DEFINITIONAL check, not an independent measurement -- it shows the
    calibration is self-consistent, not that the assembled gate actually achieves
    it. For that, see the shaped-chevron residual in `run_tune_up` step 3, or the
    ``chirp_ablation`` transfer-probability comparison.

    Past `measured_eta_max` (marked with a horizontal line) the ridge itself is
    extrapolated past where the Rabi sweep measured it -- the chirp there is
    tracking a fit, not data.

    Returns
    -------
    str
        The path written.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from snail_solver.envelope import Chirp, RaisedCosine
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
    fit = table["fit"]
    target_eta = float(proj["target_eta"])
    measured_eta_max = float(proj.get("measured_eta_max", np.nanmax(eta)))

    row_eta = np.array([float(c["eta"]) for c in chevrons])
    off_MHz = np.asarray(chevrons[0]["offsets_GHz"], dtype=float) * 1e3
    for c in chevrons[1:]:
        other = np.asarray(c["offsets_GHz"], dtype=float) * 1e3
        if other.shape != off_MHz.shape or not np.allclose(other, off_MHz):
            raise ValueError(
                "plot_chirp_ridge does not interpolate, so every row needs the "
                "SAME offsets -- this table was swept with a per-row adaptive "
                "span. Rerun rabi_shift_table/run_tune_up with "
                "wp_span_MHz, wp_points = ridge_span_MHz(config, target_eta, "
                "eta_lo=..., eta_hi=...) so every row shares one grid, then "
                "replot. The CLI does this for you: `--plot-ridge` sizes the span "
                "automatically unless you pass --wp-span-MHz yourself. Use "
                "ridge_span_MHz, not fixed_span_MHz -- the fixed span also needs "
                "MORE wp_points, or the weakest row is undersampled.")
    Z = np.stack([np.asarray(c["metric"], dtype=float) for c in chevrons], axis=0)

    ts = np.linspace(0.0, float(t_g), 400)
    eta_t = np.asarray(RaisedCosine(target_eta, float(t_g)).value_at(ts))
    chirp_MHz = 1e3 * (float(wp_offset_GHz)
                       + np.asarray(Chirp(proj["coeffs_GHz"], float(t_g)).detuning(ts))
                       / TWO_PI)
    flat_MHz = np.full_like(ts, 1e3 * float(wp_offset_GHz))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    mesh = ax.pcolormesh(off_MHz, row_eta, Z, shading="nearest", cmap="viridis",
                        vmin=0.0, vmax=1.0)
    fig.colorbar(mesh, ax=ax, label=r"$P(|10\rangle)$ (max over time)", pad=0.02)

    ys = np.linspace(float(row_eta.min()), float(max(row_eta.max(), target_eta)) * 1.02,
                     300)
    if target_eta > measured_eta_max:
        ax.axhline(measured_eta_max, color=_C_VERTEX, ls=":", lw=1.6,
                  label=f"measured up to |eta|={measured_eta_max:.2f}")
    ax.plot(fit["delta0"] + fit["k2"] * ys ** 2 + fit["k4"] * ys ** 4, ys,
           "-", lw=1.4, color="white", alpha=0.85, label="fitted ridge")
    ax.plot(ridge, eta, "o", ms=6, color="white", mec=_C_INK, mew=1.0,
           label="measured ridge")
    ax.plot(chirp_MHz, eta_t, "-", lw=2.6, color=_C_LORENTZ,
           label="chirped pump (rides the ridge)")
    ax.plot(flat_MHz, eta_t, "--", lw=2.0, color=_C_VERTEX,
           label="flat carrier (same mean offset)")
    ax.set_ylim(row_eta.min(), ys.max())
    ax.set_xlim(off_MHz.min(), off_MHz.max())
    ax.set_ylabel(r"drive strength $|\eta|$ (the amplitude/voltage axis)")
    ax.set_xlabel("pump frequency offset (MHz)")
    ax.set_title(title or (rf"Rabi map with the chirp riding the ridge, "
                          rf"$\eta^*$ = {target_eta:.2f}"), fontsize=11)
    ax.legend(fontsize=7.5, framealpha=0.9, loc="best")

    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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

    :func:`rabi_shift_table` measures a CONSTANT probe: it characterises the
    device, not the pulse. This measures the pulse. Same row structure -- one
    chevron per ``|eta|`` -- but each row runs the real raised-cosine gate with
    the calibrated chirp, and (with `compare_flat`) the identical gate with
    ``chirp_coeffs_GHz=[]`` at the same ``t_g``/``amp_scale``/``wp_offset``. That
    is exactly the ``chirp_ablation`` comparison :func:`run_tune_up` already
    does at ONE point, generalised across drive strength AND across the
    pump-offset axis.

    Each row's gate length is RESCALED so every row is a genuine full swap at
    its own drive: ``t_g_i = nominal_t_g(e) * (record["t_g_ns"] /
    nominal_t_g(target_eta))``, carrying the same empirical length correction
    the calibration found. This avoids the fixed-length rotation-angle confound
    the module docstring warns about elsewhere (see "WHY A CHEVRON AND NOT A
    CALIBRATION-MAP SLICE") -- at fixed ``t_g`` a row at less than the
    calibrated drive is a partial rotation by construction, and its contrast
    would collapse from that alone, not from leakage.

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
        Also run the identical gate with the chirp switched off (``chirp_coeffs_GHz
        =[]``), at the same ``t_g``/``amp_scale``/``wp_offset`` -- the direct
        "does the chirp's SHAPE help" comparison.
    reproject_chirp : bool, default False
        False (default): use ``record["chirp_coeffs_GHz"]`` verbatim on every row --
        the one chirp the gate will actually run, the honest test. True: re-derive
        the chirp AT EACH ROW's drive via :func:`chirp_from_measured_shift` (needs
        `rabi_table`) -- answers the different question "would the procedure work
        at this drive", not "does the calibrated gate work here".
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

    One row per drive strength, left to right: the chirped shaped chevron
    raster; the flat-carrier raster (same grid, same colour scale -- direct
    visual A/B); the envelope comparison (chirped vs flat, both P(|10>) in
    [0, 1], no twin axis, with their leakage traces); and the time trace at
    ``wp_offset``. Bottom band, three panels: transfer at ``wp_offset`` vs
    ``|eta|`` (the generalised ``chirp_ablation`` -- "does the chirp help, and
    up to what drive"), leakage vs ``|eta|``, and the located-resonance
    residual vs ``|eta|`` (should sit near zero where the chirp is valid),
    with the original constant-probe ridge overlaid if `rabi_table` is given.

    Returns
    -------
    str
        The path written.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:                                        # style is a nicety
        pass

    rows = list(post.get("rows", []))
    if not rows:
        raise ValueError("no rows to plot")
    eta = np.asarray(post["eta"], dtype=float)
    compare_flat = bool(post.get("compare_flat", False))
    n = len(rows)
    leak_cmap = LinearSegmentedColormap.from_list("leak", ["#ffffff", _C_LEAK])

    ncols = 4 if compare_flat else 3
    fig = plt.figure(figsize=(5.2 * ncols, 2.9 * n + 3.4), layout="constrained")
    gs = fig.add_gridspec(n + 1, ncols, height_ratios=[2.9] * n + [3.4])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)]
                     for r in range(n)])

    for i, row in enumerate(rows):
        off = np.asarray(row["offsets_GHz"], dtype=float) * 1e3
        chirped, flat = row["chirped"], row.get("flat")
        col = 0
        ax_c = axes[i, col]; col += 1
        mesh = ax_c.pcolormesh(off, np.arange(chirped["P10"].shape[1]),
                               np.asarray(chirped["P10"], dtype=float).T,
                               shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        fig.colorbar(mesh, ax=ax_c, label=r"$P(|10\rangle)$ (chirped)", pad=0.02)
        ax_c.axvline(float(post["record"]["wp_offset_GHz"]) * 1e3, color=_C_LORENTZ,
                    ls="--", lw=1.6)
        ax_c.set_title(rf"$|\eta|$ = {row['eta']:.3f}  chirped", fontsize=9)
        ax_c.set_ylabel("time index")

        if compare_flat and flat is not None:
            ax_f = axes[i, col]; col += 1
            meshf = ax_f.pcolormesh(off, np.arange(flat["P10"].shape[1]),
                                    np.asarray(flat["P10"], dtype=float).T,
                                    shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            fig.colorbar(meshf, ax=ax_f, label=r"$P(|10\rangle)$ (flat)", pad=0.02)
            ax_f.axvline(float(post["record"]["wp_offset_GHz"]) * 1e3, color=_C_VERTEX,
                        ls="--", lw=1.6)
            ax_f.set_title(rf"$|\eta|$ = {row['eta']:.3f}  flat carrier", fontsize=9)

        ax_e = axes[i, col]; col += 1
        ax_e.plot(off, chirped["metric"], "-", lw=2.0, color=_C_LORENTZ,
                 label="chirped")
        ax_e.plot(off, chirped["leak_at_metric"], "--", lw=1.2, color=_C_LEAK, alpha=0.8,
                 label="leak (chirped)")
        if compare_flat and flat is not None:
            ax_e.plot(off, flat["metric"], "-", lw=1.6, color=_C_VERTEX, label="flat")
            ax_e.plot(off, flat["leak_at_metric"], ":", lw=1.2, color=_C_LEAK, alpha=0.5,
                     label="leak (flat)")
        ax_e.axvline(float(post["record"]["wp_offset_GHz"]) * 1e3, color=_C_INK,
                    ls=":", lw=1.4, label="wp_offset")
        ax_e.set_ylim(-0.03, 1.05)
        ax_e.set_title(rf"P($t_g$) at wp_offset: {chirped['transfer_at_wp_offset']:.3f}"
                       + (rf" vs {flat['transfer_at_wp_offset']:.3f}"
                          if compare_flat and flat is not None else ""), fontsize=8.5)
        ax_e.legend(fontsize=6.5, framealpha=0.9, loc="upper left")
        ax_e.grid(alpha=0.25)

        ax_t = axes[i, col]
        j_wp = int(np.argmin(np.abs(off - float(post["record"]["wp_offset_GHz"]) * 1e3)))
        ax_t.plot(np.asarray(chirped["P10"])[j_wp], "-", lw=2.0, color=_C_LORENTZ,
                 label="chirped")
        ax_t.plot(np.asarray(chirped["P_leak"])[j_wp], "--", lw=1.2, color=_C_LEAK,
                 alpha=0.8, label="leak (chirped)")
        if compare_flat and flat is not None:
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
            if compare_flat and flat is not None:
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
                drag_channels=None,
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
            drag_n_pump=drag_n_pump, drag_channels=drag_channels)

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
                          drag_channels=None,
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
                drag_channels=None,
                spec_abs_GHz: Optional[float] = None, chirp_degree: int = 8,
                quartic_warn: float = 0.25,
                eta_lo: float = 0.3, eta_hi: float = 1.0,
                amp_points: int = 9, wp_span_MHz: Optional[float] = None,
                wp_points: int = 25, tg_points: int = 13,
                tg_lo: float = 0.7, tg_hi: float = 1.3, max_drag_iters: int = 4,
                window_tg: float = 2.0, n_time: int = 161,
                span_linewidths: float = 4.0, drag_shift_points: int = 0,
                contrast_min: float = 0.35,
                chirp_tol_GHz: float = 1e-4, offset_tol_MHz: float = 0.2,
                do_time_rabi: bool = True, jobs: int = 0,
                post_chirp_points: int = 0,
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

    ``post_chirp_points``, if nonzero, runs :func:`post_chirp_table` on the
    resulting operating point right after the ``chirp_ablation`` check (same
    never-fatal try/except style -- a plotting/validation step should not fail
    the calibration) and stores it under ``stages.post_chirp``. This is the
    ``chirp_ablation`` comparison generalised across drive strength, on the
    real shaped gate, using this same Rabi table.
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
            drag_n_pump=drag_n_pump, drag_channels=drag_channels, t_g=t_g,
            shape=_shape_kind, shape_kw=_shape_kw, quartic_warn=quartic_warn)

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
                        drag_n_pump=drag_n_pump, drag_channels=drag_channels)
        return float(chev["resonance_offset_GHz"]) - float(wp_offset)

    stages: Dict[str, Any] = {"rabi": table}
    history = []
    t_g = t_g0
    chirp, wp_offset, length = None, 0.0, None
    # With DRAG off nothing below depends on t_g, so one pass IS the fixed point.
    # A recursive tone couples the chirp and the length harder than a first-order
    # one: the d-th nested correction scales as 1/t_g^d, so with K channels the
    # residual coupling is ~1/t_g^K rather than 1/t_g. Give the loop more passes.
    _shape_kind, _shape_kw = shape_config(config)
    _drag_on = drag_beat_GHz is not None or bool(drag_channels)
    _n_ch = len(_resolve_drag_channels(drag_beat_GHz, drag_n_pump, drag_channels))
    n_outer = (max(int(max_drag_iters), 2 * _n_ch) if _drag_on else 1)

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

        dc = (float("inf") if prev_chirp is None
              else float(np.max(np.abs(np.array(chirp) - prev_chirp))))
        dt = abs(t_g - prev_t_g)
        history.append({"iter": it, "t_g_ns": t_g, "max_dc_GHz": dc,
                        "d_t_g_ns": dt, "wp_offset_GHz": wp_offset,
                        "residual_GHz": residual_GHz, "chirp_GHz": list(chirp)})
        if not _drag_on:
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

    # Everything above assumes DRAG shifts the resonance only by adding drive,
    # through the law the (DRAG-blind) constant probe measured. These shaped
    # DRAG-off/DRAG-on chevrons are the direct test of that assumption.
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
        # Name the TYPE: this except once hid a TypeError from a transfer_probability
        # signature drift, so the ablation silently reported nothing for every run.
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
    ap.add_argument("--device", default=None,
                    help="required unless --replot is given (or --save-point, which "
                         "needs it to know which device JSON to write into)")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="peak |eta| to FIX for the whole tune-up. This is the one "
                         "number you choose: it sets the drive and hence the nominal "
                         "length t_g0 = 2A/eta*. Larger -> shorter gate, more leakage. "
                         "Required unless --replot is given")
    ap.add_argument("--replot", metavar="JSON", default=None,
                    help="skip the run entirely and regenerate --plot/--plot-ridge "
                         "from a previously written --out JSON -- no new solves, so "
                         "this is the way to re-render a plot (e.g. after a "
                         "plot_chirp_ridge change) without repeating an expensive "
                         "cluster run")
    ap.add_argument("--drag-beat-GHz", type=float, default=None,
                    help="calibrate with DRAG on at this beat; enables the "
                         "chirp<->DRAG iteration")
    ap.add_argument("--drag-channel", action="append", default=None,
                    metavar="BEAT[:K[:N]]",
                    help="RECURSIVE multi-derivative DRAG (Li/Calarco/Motzoi, npj QI "
                         "10, 66 (2024)): suppress several off-resonant processes at "
                         "once by composing one derivative correction per process. "
                         "Repeatable. BEAT in GHz, K = pump quanta (chirp tracking, "
                         "default 1), N = photons in F^(n) (default = K). e.g. "
                         "--drag-channel 0.30 --drag-channel -0.22:1 "
                         "--drag-channel 0.55:2:2 . Mutually exclusive with "
                         "--drag-beat-GHz. NOTE: more than one channel needs a base "
                         "shape with enough vanishing end derivatives -- set "
                         "envelope=sine_power with envelope_m >= the channel count, "
                         "or the pulse diverges at the gate edges.")
    ap.add_argument("--drag-n-pump", type=int, default=1,
                    help="pump quanta of the suppressed process (1 one-pump, "
                         "2 subharmonic, 0 static/pump-independent)")
    ap.add_argument("--spec-abs-GHz", type=float, default=None)
    ap.add_argument("--chirp-degree", type=int, default=8,
                    help="Legendre truncation for the chirp (even terms only matter). "
                         "delta(u) is an EXACT degree-8 polynomial in cos(pi u/2), so "
                         "the default of 8 reconstructs it exactly with no added noise "
                         "sensitivity -- the extra terms are algebra on the already-fit "
                         "k2/k4, not new fit parameters. Below 8, plot_chirp_ridge will "
                         "show the chirp overshoot the measured ridge near |eta| -> 0")
    ap.add_argument("--quartic-warn", type=float, default=0.25,
                    help="warn (never raise) when the fitted quartic term is this "
                         "fraction of the quadratic term at target_eta -- the eta^2 + "
                         "eta^4 Stark law is not converging past this point")
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
    ap.add_argument("--tg-lo", type=float, default=0.7,
                    help="length-scan window low edge, as a fraction of t_g0 [0.7]")
    ap.add_argument("--tg-hi", type=float, default=1.3,
                    help="length-scan window high edge, as a fraction of t_g0. "
                         "Widen when the scan rails: t_g0 = 2A/eta* assumes the "
                         "leading-order rate, so a device whose dressed rate differs "
                         "puts the real full swap outside the default +/-30%% [1.3]")
    ap.add_argument("--max-drag-iters", type=int, default=4)
    ap.add_argument("--skip-time-rabi", action="store_true")
    ap.add_argument("--coupler-levels", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--gpu", action="store_true",
                    help="run via qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--out", default=None, help="write the record + stages to JSON")
    ap.add_argument("--plot", nargs="?", const="figs/rabi_chevrons.png", default=None,
                    help="render every chevron, its envelope fit and the shift curve. "
                         "Written even when the run FAILS its guards -- those errors "
                         "ask you to inspect the chevrons, so they have to be visible")
    ap.add_argument("--plot-ridge", nargs="?", const="figs/chirp_ridge.png", default=None,
                    help="overlay the chirp's pump trajectory on the fitted ridge "
                         "(the Fig. 4 / arXiv:2306.10162 picture -- riding the ridge "
                         "instead of a flat carrier). Needs a completed run, so this "
                         "is skipped when the Rabi guards fail")
    ap.add_argument("--post-chirp-points", type=int, default=0,
                    help="validate the calibrated chirp on the REAL shaped gate "
                         "across this many drive strengths (see post_chirp.py); "
                         "0 (default) skips this -- it costs its own "
                         "amp_points x wp_points x 2 solves")
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
    _cli_channels = parse_drag_channels(args.drag_channel)
    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1
    if args.replot is None and (args.device is None or args.target_eta is None):
        ap.error("--device and --target-eta are required unless --replot is given")
    if args.save_point and args.device is None:
        ap.error("--save-point needs --device (to know which device JSON to write)")

    from snail_solver.paths import resolve_device

    device_path = resolve_device(args.device) if args.device else None

    if args.replot:
        # Everything plot_rabi_table/plot_chirp_ridge need is already in a JSON
        # written by a previous --out: no config, no device, no new solves. This
        # is the fast path back to a figure after e.g. a plotting-code change.
        with open(args.replot) as fh:
            saved = json.load(fh)
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

        # plot_chirp_ridge does not interpolate, so it needs every chevron row on ONE
        # offset axis -- which only happens with a fixed span. Size it here rather
        # than letting the run finish and then refuse to draw: the plot call is the
        # very last thing main() does, so the alternative is throwing away the whole
        # sweep over a missing flag. Only when the ridge was actually asked for; an
        # explicit --wp-span-MHz still wins, and without --plot-ridge the per-row
        # adaptive spans (better sampling, no shared axis) are left alone.
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
                wp_span_MHz=args.wp_span_MHz, wp_points=args.wp_points,
                tg_points=args.tg_points, tg_lo=args.tg_lo, tg_hi=args.tg_hi,
                max_drag_iters=args.max_drag_iters,
                do_time_rabi=not args.skip_time_rabi, jobs=args.jobs,
                post_chirp_points=args.post_chirp_points,
                solver={"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps},
                logger=logger)
        except RabiFitError as exc:
            # The measurement succeeded; only the interpretation failed. Save and draw
            # it before dying, so the "inspect the chevrons" instruction is actionable.
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

    if args.plot_ridge:
        ridge_path = plot_chirp_ridge(out["stages"]["rabi"], out["stages"]["chirp"],
                                      rec["wp_offset_GHz"], rec["t_g_ns"], args.plot_ridge)
        print(f"  wrote {ridge_path}")

    if args.plot_post_chirp:
        post = out["stages"].get("post_chirp")
        if post:
            post_path = plot_post_chirp_table(post, out=args.plot_post_chirp,
                                              rabi_table=out["stages"]["rabi"])
            print(f"  wrote {post_path}")
        else:
            print("  --plot-post-chirp given but no post_chirp stage ran "
                  "(pass --post-chirp-points > 0)")

    if args.save_point:
        from snail_solver.operating_points import save_point
        save_point(device_path, args.save_point, rec, overwrite=args.overwrite)
        print(f"  saved operating point {args.save_point!r} to {device_path}")


if __name__ == "__main__":
    main()
