"""
stark_chirp.py
==============

Chirp seeds that TRACK the AC-Stark shift instead of averaging it.

The |01> <-> |10> Stark shift scales as |eta(t)|^2. For the Hann envelope,
S(t) = cos^2(pi u / 2) with u = 2t/t_g - 1, so |eta|^2 / eta_pk^2 = cos^4(pi u / 2),
sweeping 0 -> 1 -> 0. A constant pump offset (``wp_offset_GHz``) cancels only its
time average; a chirp delta(t) ~ -|eta(t)|^2 follows the resonance.

``cos^4(pi u / 2) = 3/8 + (1/2) cos(pi u) + (1/8) cos(2 pi u)`` is EVEN in u, so
only even Legendre terms appear:

    ====  ===========  ==========  ==================
    k     a_k          a_k / a_0   closed form
    ====  ===========  ==========  ==================
    0      0.375000     1.000000   3/8
    2     -0.712415    -1.899772   -225 / (32 pi^2)
    4      0.500397     1.334393
    6     -0.207850    -0.554267
    8      0.051776     0.138069
    ====  ===========  ==========  ==================

* ``a_0 = 3/8`` reproduces the Hann <eta^2> = 0.375 eta_pk^2 of
  ``find_stark_resonance.operating_eta`` (a normalization cross-check).
* ``c_0`` is exactly a constant pump offset (see ``envelope.Chirp`` and
  ``test_constant_chirp_equals_offset_rad``), so the first new degree of freedom is
  ``c_2``; a linear (odd) chirp is the wrong first guess.

Since ``0.375 s`` is the pulse-averaged shift a calibration map measures as its
optimum ``wp_offset_GHz``, the seed is a single scalar::

    c_k = (delta_stark_GHz / 0.375) * a_k   for k >= 2,      c_0 := 0

i.e. ``c_2 = -1.900 * delta_stark``, ``c_4 = +1.334 * delta_stark``.

Relative L2 truncation error (pinned by
``test_hann_stark_legendre_reconstructs_the_shape``):

    degree 2: 33.8%   degree 4: 11.3%   degree 6: 2.4%   degree 8: 0.32%

Degree 4 (two free coefficients) is the default: the seed only needs the right
basin for the optimizer. Use 6 open-loop.

No matplotlib or solver imports, so ``grape`` can use this cheaply.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

#: Legendre coefficients a_k of cos^4(pi u / 2), k = 0 .. 8 (odd terms vanish by
#: parity). Pinned by a test against a_0 = 3/8 and a_2 = -225/(32 pi^2).
HANN_STARK_LEGENDRE: np.ndarray = np.array([
    0.375000000000000,      # a_0 = 3/8, exactly
    0.0,                    # a_1
    -0.712414572485201,     # a_2 = -225 / (32 pi^2), exactly
    0.0,                    # a_3
    0.500397358312571,      # a_4
    0.0,                    # a_5
    -0.207850229644700,     # a_6
    0.0,                    # a_7
    0.051775701660015,      # a_8
])

#: a_0: the pulse-averaged value of cos^4(pi u / 2). Also the Hann <eta^2> factor.
HANN_MEAN_FACTOR: float = 0.375


def shape_stark_legendre(shape_fn, degree: int = 8) -> np.ndarray:
    """Legendre coefficients ``a_0 .. a_degree`` of an arbitrary Stark-tracking shape.

    Generalizes :data:`HANN_STARK_LEGENDRE` (``shape_fn(u) = cos^4(pi u / 2)``) to
    shapes such as :class:`envelope.SinePowerRamp` used by recursive DRAG.
    `shape_fn` maps ``u -> |eta(u)|^2 / eta_peak^2`` on ``[-1, 1]`` (vectorized). Odd
    terms are zeroed exactly (symmetric envelope), not left as quadrature dust.
    """
    from numpy.polynomial import legendre as L
    n_quad = max(2 * int(degree) + 8, 32)
    u, w = np.polynomial.legendre.leggauss(n_quad)
    y = np.asarray(shape_fn(u), dtype=float)
    a = np.array([(2 * k + 1) / 2.0 * np.sum(w * y * L.legval(u, np.eye(k + 1)[k]))
                  for k in range(int(degree) + 1)])
    a[1::2] = 0.0
    return a


def shape_mean_factor(shape_fn) -> float:
    """Pulse-averaged ``shape_fn``, i.e. its ``a_0``. 0.375 for a Hann."""
    return float(shape_stark_legendre(shape_fn, degree=0)[0])


#: Weightings for :func:`stark_moments`: ``"rabi"`` (derived, default), ``"uniform"``
#: (time average) and ``"coupling"`` (weight ``|eta|``). The latter two bracket it and
#: are reported by :func:`tune_up.cross_check_probe_moments`; they span 2x on a Hann.
MOMENT_WEIGHTINGS = ("rabi", "uniform", "coupling")


def rabi_angle(shape_fn, u) -> np.ndarray:
    r"""Accumulated Rabi angle ``theta(u)``, running ``0 -> pi`` across the pulse.

    ``theta = 2 \int g dt'`` for a full-iSWAP (area ``pi``) pulse. With
    ``g ~ |eta| = eta_peak sqrt(f(u))`` the normalization cancels ``eta_peak`` and
    ``t_g``::

        theta(u) = pi * int_{-1}^{u} sqrt(f) du' / int_{-1}^{1} sqrt(f) du'

    `u` must be ASCENDING and span ``[-1, 1]`` (the cumulative trapezoid runs on this
    grid; a partial span gives a partial angle).
    """
    u = np.asarray(u, dtype=float)
    s = np.sqrt(np.clip(np.asarray(shape_fn(u), dtype=float), 0.0, None))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(u))])
    if not np.isfinite(cum[-1]) or cum[-1] <= 0.0:
        raise ValueError("rabi weighting needs a positive pulse area")
    return np.pi * cum / cum[-1]


def stark_moments(shape_fn, weighting: str = "rabi", n_quad: int = 4001) -> tuple:
    r"""``(M2, M4)``: how a SHAPED probe's chevron reports the Stark law.

    A constant probe at ``|eta|`` measures the law pointwise,
    ``delta = k2 |eta|^2 + k4 |eta|^4``. A shaped probe of PEAK ``eta*`` instead
    reports an average over its own envelope, and because the law is an even
    polynomial that average is DIAGONAL in ``{eta^2, eta^4}``::

        <delta>(eta*) = k2 M2 eta*^2 + k4 M4 eta*^4
        M2 = <f>,  M4 = <f^2>        f(u) = |eta(u)|^2 / eta*^2

    so ``k2 = K2/M2``, ``k4 = K4/M4`` with no mixing between orders -- which makes a
    shaped ladder usable where a constant probe leaks too much (see
    ``tune_up.rabi_shift_table``). An odd or non-polynomial term would need a real
    deconvolution.

    The weight is derived: in the two-level subspace (coupling area ``pi``, detuning
    ``D(t) = Delta + delta_stark(t)``), the frame following the ideal rotation gives a
    first-order error amplitude ``-1/2 int D(t) sin theta(t) dt`` (:func:`rabi_angle`),
    so the chevron peaks at::

        Delta* = -<delta_stark>_w        w(t) = sin theta(t),  theta: 0 -> pi

    ``w`` vanishes at the pulse ends (state at a Bloch pole, where ``z`` rotation does
    nothing) and peaks mid-gate, so ``M2`` is near 1, far above the time average:

    ============== ========== ========== =========================================
    Hann           M2         M4         weight
    ============== ========== ========== =========================================
    rabi           0.7115     0.5836     ``sin theta(t)``     <- derived, default
    coupling       0.6250     0.4922     ``|eta(t)|``
    uniform        0.3750     0.2734     ``1``
    ============== ========== ========== =========================================

    A two-level integration reproduces ``"rabi"`` to 5 decimals; against a constant
    probe on a 3-mode device (``tune_up.cross_check_probe_moments``) ``M2`` is within
    ~4% of ``"rabi"`` vs 19% for ``"coupling"`` and 2.1x for ``"uniform"``.

    `n_quad` is the trapezoid grid for ``"rabi"`` (a cumulative integral, so not
    Gauss-Legendre); converged to <1e-6 well below the default. Returns
    ``(M2, M4)``, both in ``(0, 1]``.
    """
    if str(weighting) not in MOMENT_WEIGHTINGS:
        raise ValueError(f"weighting={weighting!r}: expected one of "
                         f"{list(MOMENT_WEIGHTINGS)}")
    def mean_pow(p):
        return shape_mean_factor(lambda u: np.asarray(shape_fn(u), float) ** p)

    if str(weighting) == "uniform":
        return (shape_mean_factor(shape_fn), mean_pow(2))
    if str(weighting) == "coupling":
        norm = shape_mean_factor(lambda u: np.sqrt(np.asarray(shape_fn(u), float)))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("coupling weighting needs a positive <|eta|>")
        return (mean_pow(1.5) / norm, mean_pow(2.5) / norm)
    u = np.linspace(-1.0, 1.0, max(int(n_quad), 101))
    f = np.clip(np.asarray(shape_fn(u), dtype=float), 0.0, None)
    w = np.sin(rabi_angle(shape_fn, u))
    norm = np.trapz(w, u)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rabi weighting needs a positive sin(theta) normalization")
    return (float(np.trapz(w * f, u) / norm), float(np.trapz(w * f * f, u) / norm))


def hann_stark_legendre(degree: int = 4) -> np.ndarray:
    """``a_0 .. a_degree`` of the Hann Stark shape (a copy); ValueError out of range."""
    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if degree >= HANN_STARK_LEGENDRE.size:
        raise ValueError(
            f"degree {degree} exceeds the tabulated range "
            f"(max {HANN_STARK_LEGENDRE.size - 1}); the higher coefficients fall off "
            f"fast enough that this is almost certainly not what you want")
    return HANN_STARK_LEGENDRE[:degree + 1].copy()


def stark_chirp_seed(delta_stark_GHz: float, degree: int = 4,
                     pin_c0: bool = True) -> np.ndarray:
    """Chirp coefficients ``c_0 .. c_degree`` (GHz) tracking a Stark shift.

    Scales the Hann Stark shape so its MEAN is `delta_stark_GHz` (the pulse-averaged
    shift, e.g. the calibrated ``wp_offset_GHz`` or
    ``stark_slope_from_map(...)['delta_stark_GHz']``). `pin_c0` zeroes c_0, which is
    exactly degenerate with ``wp_offset_GHz``: leaving both free gives an optimizer a
    flat direction and an ambiguous saved point.

    Examples
    --------
    >>> np.round(stark_chirp_seed(0.002, degree=4), 6)
    array([ 0.      ,  0.      , -0.0038  ,  0.      ,  0.002669])
    """
    c = float(delta_stark_GHz) / HANN_MEAN_FACTOR * hann_stark_legendre(degree)
    if pin_c0:
        c[0] = 0.0
    return c


def map_ridge(Z: np.ndarray, offsets_MHz: np.ndarray,
              amps: np.ndarray) -> np.ndarray:
    """Best pump offset (MHz) per amplitude row of a calibration map's ``Z``.

    The ridge is the Stark shift vs drive. Each row's argmax is refined by a parabola
    through its neighbours; NaN for an all-NaN row. `amps` is used only for its length.
    """
    Z = np.asarray(Z, dtype=float)
    offsets_MHz = np.asarray(offsets_MHz, dtype=float)
    if Z.shape != (len(amps), offsets_MHz.size):
        raise ValueError(f"Z has shape {Z.shape}, expected "
                         f"{(len(amps), offsets_MHz.size)}")

    ridge = np.full(Z.shape[0], np.nan)
    for i, row in enumerate(Z):
        if np.all(np.isnan(row)):
            continue
        j = int(np.nanargmax(row))
        if 0 < j < row.size - 1 and np.all(np.isfinite(row[j - 1:j + 2])):
            y0, y1, y2 = row[j - 1], row[j], row[j + 1]
            denom = y0 - 2.0 * y1 + y2
            # denom == 0 is a flat top; denom > 0 means the "peak" is a local minimum
            # (a ragged row), and in both cases the vertex formula is meaningless.
            shift = 0.0 if denom >= 0 else 0.5 * (y0 - y2) / denom
            shift = float(np.clip(shift, -1.0, 1.0))
            step = offsets_MHz[j + 1] - offsets_MHz[j]
            ridge[i] = offsets_MHz[j] + shift * step
        else:
            ridge[i] = offsets_MHz[j]                    # railed at an edge
    return ridge


def stark_slope_from_map(result: Dict[str, Any],
                         amp_scale: Optional[float] = None) -> Dict[str, Any]:
    """Fit a ``calibration_map`` result's ridge as ``offset_MHz = m * amp^2 + b``.

    The shift ~ |eta|^2 ~ ``amp_scale**2``. The intercept is kept as a diagnostic: a
    large ``b`` means the nominal offset already absorbed part of the shift, or the
    ridge railed. `amp_scale` defaults to ``result['best']['amp_scale']``.

    Returns ``ridge_MHz``, ``slope_MHz_per_amp2``, ``intercept_MHz``, ``amp_scale``,
    ``delta_stark_MHz`` / ``delta_stark_GHz`` (shift at `amp_scale`) and ``r2``; a
    poor ``r2`` means the ridge is not Stark-dominated.
    """
    Z = np.asarray(result["Z"], dtype=float)
    offsets = np.asarray(result["offsets_MHz"], dtype=float)
    amps = np.asarray(result["amps"], dtype=float)
    ridge = map_ridge(Z, offsets, amps)

    ok = np.isfinite(ridge)
    if ok.sum() < 3:
        raise ValueError("need at least 3 usable amplitude rows to fit the ridge")
    x = amps[ok] ** 2
    y = ridge[ok]
    m, b = np.polyfit(x, y, 1)
    resid = y - (m * x + b)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    if amp_scale is None:
        amp_scale = float(result.get("best", {}).get("amp_scale", 1.0))
    delta_MHz = float(m * amp_scale ** 2 + b)
    return {"ridge_MHz": ridge, "slope_MHz_per_amp2": float(m),
            "intercept_MHz": float(b), "amp_scale": float(amp_scale),
            "delta_stark_MHz": delta_MHz, "delta_stark_GHz": delta_MHz * 1e-3,
            "r2": r2}


def seed_from_calibration_map(result: Dict[str, Any], degree: int = 4,
                              amp_scale: Optional[float] = None,
                              pin_c0: bool = True) -> np.ndarray:
    """Chirp seed (GHz) straight from a calibration map (fit ridge, then seed)."""
    fit = stark_slope_from_map(result, amp_scale=amp_scale)
    return stark_chirp_seed(fit["delta_stark_GHz"], degree=degree, pin_c0=pin_c0)


def describe_seed(coeffs: Sequence[float], delta_stark_GHz: float) -> str:
    """One-line summary of a seed and the shift it was built to track."""
    nz = ", ".join(f"c{k}={c:+.5f}" for k, c in enumerate(coeffs) if c)
    return (f"stark-chirp seed for delta_stark={delta_stark_GHz * 1e3:+.3f} MHz: "
            f"[{nz or 'all zero'}] GHz")
