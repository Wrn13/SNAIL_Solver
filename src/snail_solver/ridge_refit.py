"""Re-interpret a measured Stark ridge without re-measuring it.

A column's cost is its Rabi sweep: ``amp_points x wp_points`` independent chevron
solves, measured at 91% of a pass-A column's wall time and 69% of a pass-B
column's on the 2026-09-25 grid. Everything after it -- gating the rows, fitting
``delta(|eta|)``, and deciding whether the law is good enough to chirp with -- is
pure numpy over arrays the run already wrote to its HDF5 file under
``columns/<tag>/stages/rabi``.

That asymmetry is the whole point of this module. A question about the FIT
policy costs seconds and answers itself from 2.3 GB already on disk; a question
about the MEASUREMENT costs days. So every fit-policy change is settled here,
against every column of a real grid, before anything is edited in the pipeline.

What is stored, for all 189 columns of the 2 GHz grid INCLUDING the 68 that
failed (a failed column attaches its partial table to the exception, and
``_write_run`` stores it anyway -- that is why the chevron figures rendered for
189 of 189 while only 121 produced a fidelity):

    stages/rabi/
      eta delta_MHz contrast quality leakage windows_ns spans_MHz held
      fit/       delta0 k2 k4 r2 n_used resid_MHz stark_span_MHz
                 K2_measured K4_measured M2 M4 chirp_excursion_MHz
                 chirp_excursion_frac_linewidth chirp_zeroed ...
      stability/ delta_spread target_eta
      chevrons/  i00000..i00040, each with
                   offsets_GHz metric times_ns P10 P01 P_leak leak_at_metric
                   fit/{center_GHz hwhm_GHz depth base rmse vertex_GHz ok}
                   quality/{contrast gap_frac nrmse secondary hwhm_frac leak
                            weight reject}
                   leak_breakdown/{f_a f_b coupler double spectator}
                 attrs: eta, window_ns, span_MHz, norm_defect_max, and
                        `dropped` only on a rejected row

THREE THINGS A RE-FIT MUST NOT GET WRONG, each found by reading the real file
rather than the code:

* ``delta_MHz`` and ``quality`` are stored POST-HOLD. The held-row rescue at
  ``tune_up.py:1230-1251`` mutates them in place before the table is written, so
  a failed column's flat held tail is already baked into them. Re-derive the
  ridge from ``chevrons[i]["fit"]["center_GHz"]`` and a fresh gate; reuse those
  arrays and a held tail gets held twice.
* ``n_offsets`` is not stored, and the rail check needs
  ``step = spans / (n_offsets - 1)``. Re-derive it per row as
  ``len(chevrons[i]["offsets_GHz"])`` -- which is exactly why the per-row
  adaptive span, and its 15/29/43 point counts, matter here.
* The stored ``k2`` is ALREADY ``K2_measured / M2``. The shaped-probe moment
  de-convolution at ``tune_up.py:1277-1285`` has been applied. Discard the
  stored ``fit`` and rebuild from the rows; refining it applies the moments
  twice.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Powers of ``|eta|`` in the shift law the pipeline fits today. ``0`` is the
#: static ``delta0`` that a chirp must NOT track (see `tune_up.fit_shift_curve`).
DEFAULT_POWERS: Tuple[int, ...] = (0, 2, 4)

#: The peak-detection floor `chevron_quality` uses for its `secondary` measure
#: (`tune_up.py:801`). Repeated rather than imported so a reader can see what a
#: "peak" means here without chasing it, and asserted against the source in the
#: test suite.
PEAK_FRAC = 0.3


# --------------------------------------------------------------------------
# reading a stored column
# --------------------------------------------------------------------------

def load_column_rabi(path: str, tag: str) -> Dict[str, Any]:
    """One column's stored Rabi table, in the shape `rabi_shift_table` returns.

    Deliberately thin. :func:`h5_io.load_tree` already reverses the entire
    encoding -- the ``_container: "list"`` convention, ``h5py.Empty`` back to
    ``None``, dtypes -- and reads ONLY the addressed group, so pulling one
    column out of a 2.3 GB file costs about a quarter of a second rather than
    the whole file. Writing a second deserializer here would be a second
    encoding to keep in step with `h5_io`.

    What is left is the two schema differences a re-fit must not trip over:

    * ``held``/``n_held``/``stark_crossing_eta`` exist only on a column whose
      table was stored from a ``RabiFitError`` partial (``tune_up.py:1117``); a
      column that succeeded stored the ``:1387`` return, which has none of them.
    * ``delta_MHz``/``quality``/``fit`` are post-hold or post-moment (see the
      module docstring), so they are DROPPED rather than handed on as if they
      were raw measurements.

    The dropped keys are returned separately by :func:`load_column_stored_fit`
    for anyone who wants to compare against them.
    """
    from snail_solver.h5_io import load_tree, split_address

    file_path, _ = split_address(path)
    table = load_tree(file_path, group=f"columns/{tag}/stages/rabi")
    if not table.get("chevrons"):
        raise KeyError(f"{file_path}:/columns/{tag} has no chevrons to re-fit")
    for key in ("fit", "stability", "held", "n_held", "stark_crossing_eta",
                "delta_MHz", "quality"):
        table.pop(key, None)
    return table


def load_column_stored_fit(path: str, tag: str) -> Dict[str, Any]:
    """The ``fit`` dict the run itself recorded, for before/after comparison.

    Returns ``{}`` when the column stored none -- a ridge that railed never
    reached `fit_shift_curve`, which is the case `_stale_chirp_free` also has to
    treat as "the excursion is unknown" rather than "the excursion is small".
    """
    from snail_solver.h5_io import load_tree, split_address

    file_path, _ = split_address(path)
    try:
        return load_tree(file_path, group=f"columns/{tag}/stages/rabi/fit")
    except (KeyError, OSError):
        return {}


def iter_columns(path: str) -> Iterator[Tuple[str, str, float]]:
    """``(tag, status, delta_GHz)`` for every column in a scan file.

    ``status`` is the attribute `_write_run` stamps on the column group
    (``subharmonic_gate_scan.py:872``): ``"ok"`` when the column produced a
    fidelity, ``"failed"`` when it did not. Both kinds carry their chevrons,
    which is what makes a failed column re-analysable at all.
    """
    import h5py

    from snail_solver.h5_io import split_address

    file_path, _ = split_address(path)
    with h5py.File(file_path, "r") as h:
        cols = h.get("columns")
        if cols is None:
            return
        for tag in cols:
            g = cols[tag]
            yield (str(tag), str(g.attrs.get("status", "?")),
                   float(g.attrs.get("delta_GHz", float("nan"))))


# --------------------------------------------------------------------------
# the row gate
# --------------------------------------------------------------------------

def narrow_slice(offsets_GHz: np.ndarray, n_narrow: int) -> slice:
    """The central `n_narrow` points -- a widened row's ORIGINAL grid.

    A span growth re-samples at the same step: ``rabi_shift_table`` grows
    ``n_off`` by the same factor as the span (``tune_up.py:1056-1059``), so a
    3x-widened 43-point row is the original 15-point grid with 14 points added
    on each side, at the same offsets. That makes the "what if this row had
    never been widened" counterfactual an exact re-read of stored data rather
    than a model -- which is the only reason the widening policy could be
    settled without re-solving anything.

    Raises when the arithmetic does not line up, because a silent off-by-one
    here would quietly compare two different measurements.
    """
    n = len(offsets_GHz)
    if n_narrow > n or (n - n_narrow) % 2:
        raise ValueError(
            f"{n} offsets cannot be an even growth of {n_narrow}; this row was "
            f"not produced by the symmetric span growth the slice assumes")
    pad = (n - n_narrow) // 2
    return slice(pad, n - pad)


def gate_rows(chevrons: Sequence[Dict[str, Any]], *,
              wp_offset_GHz: float = 0.0,
              n_narrow: Optional[int] = None,
              refit_centers: bool = False,
              **quality_kw) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
    """Re-run the per-row quality gate over stored chevrons.

    Returns ``(ridge_MHz, weights, rejects)``, with ``nan`` in the ridge
    wherever the row was rejected -- the same convention `rabi_shift_table` uses
    (``tune_up.py:991``, ``:1090-1103``).

    By default this calls `chevron_quality` on each row's STORED Lorentzian fit,
    so re-gating under the original thresholds returns the ridge the run already
    recorded. Pass ``refit_centers=True`` to re-run `fit_chevron_center` as
    well, which is what a policy that changes the FITTING (rather than the
    accept/reject decision) needs.

    `n_narrow` emulates a run made without the ``_too_wide`` span growth: each
    row is cropped to its original central grid before being fitted and gated.
    Measured over the 2 GHz grid, cropping the 834 ``_too_wide`` rows this way
    KEEPS MORE of them (62.1% vs 54.0%) -- the widening was losing the rows it
    was meant to save.
    """
    from snail_solver.tune_up import chevron_quality, fit_chevron_center

    n = len(chevrons)
    ridge = np.full(n, np.nan, dtype=float)
    weights = np.zeros(n, dtype=float)
    rejects: List[Optional[str]] = [None] * n

    for i, ch in enumerate(chevrons):
        off = np.asarray(ch["offsets_GHz"], dtype=float)
        met = np.asarray(ch["metric"], dtype=float)
        span = float(ch.get("span_MHz", np.nan))
        if n_narrow is not None and len(off) > n_narrow:
            sl = narrow_slice(off, n_narrow)
            # The span shrinks with the grid; the gate compares hwhm against it.
            span *= (n_narrow - 1) / (len(off) - 1)
            off, met = off[sl], met[sl]
            cen = fit_chevron_center(off, met)
        elif refit_centers:
            cen = fit_chevron_center(off, met)
        else:
            cen = dict(ch.get("fit") or {})

        leak = (ch.get("quality") or {}).get("leak")
        q = chevron_quality(cen, off, met, span,
                            leak=(None if leak is None else float(leak)),
                            **quality_kw)
        rejects[i] = q["reject"]
        weights[i] = float(q["weight"])
        if q["reject"]:
            continue
        ridge[i] = (float(cen["center_GHz"]) - float(wp_offset_GHz)) * 1e3
    return ridge, weights, rejects


# --------------------------------------------------------------------------
# the law
# --------------------------------------------------------------------------

def fit_law(eta: np.ndarray, delta_MHz: np.ndarray,
            weights: Optional[np.ndarray] = None, *,
            powers: Sequence[int] = DEFAULT_POWERS,
            eta_max: Optional[float] = None) -> Dict[str, Any]:
    """Weighted least squares of ``delta = sum_p c_p |eta|^p``.

    A generalization of `tune_up.fit_shift_curve`, which hardwires
    ``(0, 2, 4)``. Two knobs, and they answer different questions:

    * `powers` -- keep another order of the Stark expansion. Costs nothing in
      extrapolation, since the fit still spans every measured row.
    * `eta_max` -- fit only the rows below a drive and EXTRAPOLATE to eta*.
      Measured over the 64 fittable failed columns of the 2 GHz grid, capping
      at ``0.6 eta*`` lifts median r2 from 0.732 to 0.999 -- but a law fitted to
      eta <= 0.78 and evaluated at 1.3 is a bigger leap than one fitted
      throughout, and r2 cannot see that. `extrapolation_ratio` can, and is
      returned here so the trade is visible rather than implied.

    r2 is computed WEIGHTED, unlike `fit_shift_curve`, which solves the least
    squares weighted and then scores it unweighted (``tune_up.py:572``). That
    inconsistency is in the pipeline and is not this function's to change
    silently, so `r2_unweighted` is returned alongside for direct comparison
    against a stored `fit`.
    """
    eta = np.asarray(eta, dtype=float)
    y = np.asarray(delta_MHz, dtype=float)
    w = (np.ones_like(eta) if weights is None
         else np.asarray(weights, dtype=float))

    ok = np.isfinite(eta) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if eta_max is not None:
        ok &= eta <= float(eta_max)
    powers = tuple(int(p) for p in powers)
    if int(ok.sum()) < len(powers) + 1:
        raise ValueError(
            f"need >= {len(powers) + 1} usable rows to fit a {len(powers)}-term "
            f"law, have {int(ok.sum())}")

    x, y, w = eta[ok], y[ok], w[ok]
    A = np.column_stack([x ** p for p in powers])
    s = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * s[:, None], y * s, rcond=None)
    pred = A @ coef

    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - np.average(y, weights=w)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ur = float(((y - pred) ** 2).sum())
    ut = float(((y - y.mean()) ** 2).sum())

    out: Dict[str, Any] = {
        "powers": powers,
        "coeffs": {f"k{p}": float(c) for p, c in zip(powers, coef)},
        "r2": float(r2),
        "r2_unweighted": float(1.0 - ur / ut if ut > 0 else 0.0),
        "n_used": int(ok.sum()),
        "resid_MHz": float(np.sqrt(ss_res / w.sum())),
        "eta_max_fitted": float(x.max()),
    }
    out["delta0"] = out["coeffs"].get("k0", 0.0)
    return out


def probe_moments_general(config: Dict[str, Any], powers: Sequence[int], *,
                          weighting: str = "rabi", n_quad: int = 4001) -> Dict[int, float]:
    """``{power: M_power}`` for this device's envelope, at ANY even order.

    `stark_chirp.stark_moments` returns exactly ``(M2, M4)`` because those are
    the two orders the pipeline's law has. Its derivation is not limited to
    them: a shaped probe of peak ``eta*`` reports
    ``<delta>(eta*) = sum_n k_2n M_2n eta*^2n`` with ``M_2n = <f^n>_w``,
    ``f(u) = |eta(u)|^2 / eta*^2``, diagonal in the even powers because the law
    is an even polynomial. So ``M6 = <f^3>_w`` and the rest follows.

    Written here rather than in `stark_chirp` because extending the law is a
    decision this module exists to inform, not one it should presume. The
    orders it shares with `stark_moments` must agree to round-off, which the
    test suite asserts -- if they ever diverge, this is the copy that is wrong.

    ``M0 = 1`` by construction: the static term is not drive-dependent and no
    envelope averages it away.
    """
    import numpy as _np

    from snail_solver.stark_chirp import rabi_angle, shape_mean_factor
    from snail_solver.tune_up import _shape_envelope, shape_config

    shape, shape_kw = shape_config(config)
    env = _shape_envelope(shape, shape_kw)

    def shape_fn(u):
        s = _np.abs(_np.asarray(env.value_at(_np.asarray(u, dtype=float) + 1.0, _np),
                                dtype=complex))
        return s ** 2

    orders = sorted({int(p) for p in powers if int(p) != 0})
    if any(p % 2 for p in orders):
        raise ValueError(
            f"odd powers {[p for p in orders if p % 2]} mix orders under a shaped "
            f"probe and need a real deconvolution, not a moment division")

    out: Dict[int, float] = {0: 1.0}
    if str(weighting) == "uniform":
        for p in orders:
            out[p] = shape_mean_factor(
                lambda u, n=p // 2: _np.asarray(shape_fn(u), float) ** n)
        return out
    if str(weighting) == "coupling":
        norm = shape_mean_factor(lambda u: _np.sqrt(_np.asarray(shape_fn(u), float)))
        if not _np.isfinite(norm) or norm <= 0.0:
            raise ValueError("coupling weighting needs a positive <|eta|>")
        for p in orders:
            out[p] = shape_mean_factor(
                lambda u, n=p // 2: _np.asarray(shape_fn(u), float) ** (n + 0.5)) / norm
        return out

    u = _np.linspace(-1.0, 1.0, max(int(n_quad), 101))
    f = _np.clip(_np.asarray(shape_fn(u), dtype=float), 0.0, None)
    w = _np.sin(rabi_angle(shape_fn, u))
    norm = _np.trapz(w, u)
    if not _np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rabi weighting needs a positive sin(theta) normalization")
    for p in orders:
        out[p] = float(_np.trapz(w * f ** (p // 2), u) / norm)
    return out


def deconvolve_moments(fit: Dict[str, Any],
                       moments: Dict[int, float]) -> Dict[str, Any]:
    """Divide a SHAPED-probe law by its moments to get the pointwise one.

    ``k_2n = K_2n / M_2n`` (`stark_chirp.stark_moments`). A constant probe needs
    none of this; a shaped one measures ``K`` and the chirp needs ``k``.

    This is the step that makes a stored ``fit`` unusable as an input: the run
    already applied it, so the stored ``k2`` IS ``K2_measured / M2``. Applying
    it to a stored law would divide twice. Only ever call this on a law fitted
    here, from the rows.
    """
    out = dict(fit)
    out["coeffs"] = {}
    out["coeffs_measured"] = dict(fit["coeffs"])
    out["moments"] = {f"M{p}": float(moments[p]) for p in fit["powers"]}
    for p in fit["powers"]:
        m = float(moments[p])
        if m <= 0.0:
            raise ValueError(f"moment M{p} = {m} is not positive; cannot deconvolve")
        out["coeffs"][f"k{p}"] = float(fit["coeffs"][f"k{p}"]) / m
    out["delta0"] = out["coeffs"].get("k0", 0.0)
    return out


def evaluate_law(fit: Dict[str, Any], eta: float, *,
                 drop_static: bool = True) -> float:
    """The law's drive-dependent shift at `eta`, in MHz.

    ``delta0`` is excluded by default: it survives at zero drive, comes from the
    static Hamiltonian rather than the pump, and belongs in ``wp_offset_GHz``
    rather than in the chirp (``tune_up.fit_shift_curve``, ``:522``). Every
    excursion and quartic-fraction number downstream means the drive-dependent
    part, so getting this wrong would flatter or damn a law by a constant.
    """
    total = 0.0
    for p, c in zip(fit["powers"], [fit["coeffs"][f"k{p}"] for p in fit["powers"]]):
        if drop_static and p == 0:
            continue
        total += c * float(eta) ** p
    return float(total)


def law_diagnostics(fit: Dict[str, Any], target_eta: float,
                    t_g_ns: float) -> Dict[str, Any]:
    """Whether this law may be chirped with -- the three questions that matter.

    * ``excursion_frac_linewidth`` -- would a chirp DO anything? The excursion
      against the resonance half-width ``1/(2 t_g)``. This is the discriminator
      `tune_up.py:1298-1303` uses to tell "no shift to chirp" from "a shift we
      failed to measure", and the two want opposite treatment.
    * ``last_term_fraction`` -- is the series converged? The magnitude of the
      HIGHEST kept term against the lowest drive-dependent one, at eta*. The
      generalization of ``quartic_fraction`` (``:1808``), whose warn threshold
      is 0.25. A law whose last term is not small is a truncation that has not
      settled, whatever its r2.
    * ``extrapolation_ratio`` -- is eta* inside the measurement? Above 1 the law
      is being read outside the rows that produced it, which no r2 can detect.
    """
    eta = float(target_eta)
    drive = [p for p in fit["powers"] if p != 0]
    exc = abs(evaluate_law(fit, eta))
    half_lw = 1e3 / (2.0 * float(t_g_ns)) if t_g_ns else float("nan")

    if len(drive) >= 2:
        lo, hi = min(drive), max(drive)
        base = abs(fit["coeffs"][f"k{lo}"] * eta ** lo)
        last = abs(fit["coeffs"][f"k{hi}"] * eta ** hi)
        frac = last / base if base else float("inf")
    else:
        frac = 0.0

    return {
        "excursion_MHz": float(exc),
        "excursion_frac_linewidth": float(exc / half_lw) if half_lw else float("inf"),
        "last_term_fraction": float(frac),
        "extrapolation_ratio": float(eta / fit["eta_max_fitted"])
        if fit.get("eta_max_fitted") else float("nan"),
    }


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------

def refit_column(table: Dict[str, Any], target_eta: float, *,
                 powers: Sequence[int] = DEFAULT_POWERS,
                 eta_max: Optional[float] = None,
                 n_narrow: Optional[int] = None,
                 refit_centers: bool = False,
                 wp_offset_GHz: float = 0.0,
                 config: Optional[Dict[str, Any]] = None,
                 probe_shape: str = "constant",
                 moment_weighting: str = "rabi",
                 **quality_kw) -> Dict[str, Any]:
    """Gate, fit and diagnose one stored column. No solves.

    Returns the law, its diagnostics and the row census. Raises `ValueError`
    when too few rows survive to fit the requested law -- the offline mirror of
    the ``"need >= 4 usable rows"`` failure, which is 2 of the 2 GHz grid's 68.

    `config` and `probe_shape` are needed only for a SHAPED probe, where the
    fitted coefficients are the law's scaled by the envelope moments and have
    to be divided back out (``tune_up.py:1272-1283``). Without them a shaped
    column's excursion is reported in measured rather than pointwise units,
    which understates it by ~1/M2 -- so the report says so rather than
    guessing.
    """
    chevrons = list(table["chevrons"])
    eta = np.asarray(table["eta"], dtype=float)
    t_g = float(table.get("t_g_ref_ns") or np.nan)

    ridge, weights, rejects = gate_rows(
        chevrons, wp_offset_GHz=wp_offset_GHz, n_narrow=n_narrow,
        refit_centers=refit_centers, **quality_kw)

    fit = fit_law(eta, ridge, weights, powers=powers, eta_max=eta_max)
    shaped = str(probe_shape) != "constant"
    fit["probe_shape"] = str(probe_shape)
    if shaped and config is not None:
        fit = deconvolve_moments(
            fit, probe_moments_general(config, fit["powers"],
                                       weighting=moment_weighting))
        fit["moment_weighting"] = str(moment_weighting)
    elif shaped:
        fit["moments_unavailable"] = True
    diag = law_diagnostics(fit, target_eta, t_g)

    census: Dict[str, int] = {}
    for r in rejects:
        census[r or "kept"] = census.get(r or "kept", 0) + 1

    return {"fit": fit, "diagnostics": diag, "rows": census,
            "n_rows": len(chevrons), "ridge_MHz": ridge, "weights": weights,
            "rejects": rejects, "eta": eta, "t_g_ref_ns": t_g}


#: Named policies, so a policy travels through a report (or, later, a worker
#: payload) as a string. A closure cannot be pickled; a name can.
POLICIES: Dict[str, Dict[str, Any]] = {
    # Exactly what the pipeline does today, and the baseline every other policy
    # is measured against.
    "stored": {},
    # One more order of the Stark expansion, fitted over every measured row --
    # no extrapolation penalty.
    "k6": {"powers": (0, 2, 4, 6)},
    "k8": {"powers": (0, 2, 4, 6, 8)},
    # Better conditioned, at the cost of reading the law outside the rows that
    # produced it. `extrapolation_ratio` is what makes that cost visible.
    "narrow_fit_0.8": {"eta_max_frac": 0.8},
    "narrow_fit_0.6": {"eta_max_frac": 0.6},
    # A run made without the `_too_wide` span growth, reconstructed exactly from
    # the widened rows' central grids.
    "no_widen": {"n_narrow": 15, "refit_centers": True},
    "no_widen_k6": {"n_narrow": 15, "refit_centers": True,
                    "powers": (0, 2, 4, 6)},
}


def policy_kwargs(name: str, target_eta: float) -> Dict[str, Any]:
    """Resolve a named policy against a column's eta*.

    ``eta_max_frac`` is stored as a FRACTION of eta* rather than an absolute
    drive so one policy name means the same thing at eta* = 1.3 and 1.5.
    """
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    kw = dict(POLICIES[name])
    frac = kw.pop("eta_max_frac", None)
    if frac is not None:
        kw["eta_max"] = float(frac) * float(target_eta)
    return kw
