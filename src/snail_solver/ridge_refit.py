"""Re-interpret a measured Stark ridge without re-measuring it.

A column's cost is its Rabi sweep (``amp_points x wp_points`` chevron solves, most of
its wall time). Everything after it -- gating rows, fitting ``delta(|eta|)``, judging
whether the law is good enough to chirp with -- is pure numpy over arrays the run
already wrote under ``columns/<tag>/stages/rabi``. So fit-policy questions are
settled here, in seconds, against every column of a real grid.

Stored for every column, INCLUDING failed ones (a failed column attaches its partial
table to the exception and ``_write_run`` stores it anyway):

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

Three pitfalls a re-fit must avoid:

* ``delta_MHz`` and ``quality`` are stored POST-HOLD (tune_up's held-row rescue
  mutates them before writing). Re-derive the ridge from
  ``chevrons[i]["fit"]["center_GHz"]`` and a fresh gate, or a held tail is held twice.
* ``n_offsets`` is not stored; the rail check's ``step = spans / (n_offsets - 1)``
  needs ``len(chevrons[i]["offsets_GHz"])`` per row (the adaptive span varies it).
* The stored ``k2`` is ALREADY ``K2_measured / M2``. Rebuild the fit from the rows;
  refining the stored one applies the moments twice.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Powers of ``|eta|`` in the pipeline's shift law. ``0`` is the static ``delta0``
#: that a chirp must NOT track (see `tune_up.fit_shift_curve`).
DEFAULT_POWERS: Tuple[int, ...] = (0, 2, 4)


# --------------------------------------------------------------------------
# reading a stored column
# --------------------------------------------------------------------------

def load_column_rabi(path: str, tag: str) -> Dict[str, Any]:
    """One column's stored Rabi table, in the shape `rabi_shift_table` returns.

    :func:`h5_io.load_tree` reverses the whole encoding and reads only the addressed
    group, so this stays thin. Two schema quirks are handled: ``held``/``n_held``/
    ``stark_crossing_eta`` exist only on a column stored from a ``RabiFitError``
    partial, and ``delta_MHz``/``quality``/``fit`` are post-hold or post-moment (see
    the module docstring). All are DROPPED; :func:`load_column_stored_fit` returns
    the stored fit for comparison.
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


def load_npz_rabi(path: str) -> Dict[str, Any]:
    """A Rabi table from the ``*_rabi.npz`` a column worker writes beside its cache.

    Inverts :func:`subharmonic_gate_scan.save_column_rabi` (flat keys like
    ``chevrons/00037/metric``) into the nested shape :func:`load_column_rabi`
    returns. Prefer it when present: the worker writes it before returning, so it
    survives a killed run and exists for cache-served columns the parent never stores.
    """
    with np.load(path, allow_pickle=False) as z:
        table: Dict[str, Any] = {}
        rows: Dict[int, Dict[str, Any]] = {}
        for k in list(z.keys()):
            v = z[k]
            val = v.item() if v.ndim == 0 else v
            if isinstance(val, bytes):
                val = val.decode()
            parts = k.split("/")
            if parts[0] == "chevrons" and len(parts) >= 3:
                node, parts = rows.setdefault(int(parts[1]), {}), parts[2:]
            else:
                node = table
            for seg in parts[:-1]:
                node = node.setdefault(seg, {})
            node[parts[-1]] = val
    table["chevrons"] = [rows[i] for i in sorted(rows)]
    for key in ("fit", "stability", "held", "n_held", "stark_crossing_eta",
                "delta_MHz", "quality"):
        table.pop(key, None)
    return table


def load_column_stored_fit(path: str, tag: str) -> Dict[str, Any]:
    """The ``fit`` dict the run itself recorded, for before/after comparison.

    ``{}`` when the column stored none (a railed ridge never reached
    `fit_shift_curve`): the excursion is unknown, not small.
    """
    from snail_solver.h5_io import load_tree, split_address

    file_path, _ = split_address(path)
    try:
        return load_tree(file_path, group=f"columns/{tag}/stages/rabi/fit")
    except (KeyError, OSError):
        return {}


def iter_columns(path: str) -> Iterator[Tuple[str, str, float]]:
    """``(tag, status, delta_GHz)`` for every column in a scan file.

    ``status`` is the group attribute `_write_run` stamps: ``"ok"`` or ``"failed"``.
    Both kinds carry their chevrons, so failed columns are re-analysable too.
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

    ``rabi_shift_table`` grows ``n_off`` by the same factor as the span, so a
    widened row is the original grid plus equal padding on each side at the same
    step; the "never widened" counterfactual is an exact re-read. Raises ValueError
    when the counts do not line up, rather than silently comparing different data.
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

    Returns ``(ridge_MHz, weights, rejects)`` with ``nan`` in the ridge for rejected
    rows (as `rabi_shift_table` does). By default `chevron_quality` is applied to each
    row's STORED Lorentzian fit, so the original thresholds reproduce the recorded
    ridge; ``refit_centers=True`` re-runs `fit_chevron_center` too. `n_narrow` emulates
    a run without the ``_too_wide`` span growth by cropping each row to its original
    central grid (and span) before fitting and gating.
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
            # the span shrinks with the grid; the gate compares hwhm against it
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

    Generalizes `tune_up.fit_shift_curve` (hardwired to ``(0, 2, 4)``):
    `powers` keeps more orders of the Stark expansion; `eta_max` fits only rows below
    a drive and EXTRAPOLATES to eta* -- better r2, but read outside the measured rows,
    which `law_diagnostics`' ``extrapolation_ratio`` exposes.

    ``r2`` is WEIGHTED; `fit_shift_curve` scores its weighted fit unweighted, so
    ``r2_unweighted`` is returned too for comparison against a stored `fit`.
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

    `stark_chirp.stark_moments` returns only ``(M2, M4)``, but its derivation is
    general: a shaped probe of peak ``eta*`` reports ``<delta>(eta*) = sum_n k_2n
    M_2n eta*^2n`` with ``M_2n = <f^n>_w``, ``f(u) = |eta(u)|^2 / eta*^2`` (diagonal
    because the law is even). Shared orders must agree with `stark_moments` to
    round-off (tested). ``M0 = 1``: the static term is not drive-dependent.
    """
    from snail_solver.stark_chirp import rabi_angle, shape_mean_factor
    from snail_solver.tune_up import _shape_envelope, shape_config

    shape, shape_kw = shape_config(config)
    env = _shape_envelope(shape, shape_kw)

    def shape_fn(u):
        s = np.abs(np.asarray(env.value_at(np.asarray(u, dtype=float) + 1.0, np),
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
                lambda u, n=p // 2: np.asarray(shape_fn(u), float) ** n)
        return out
    if str(weighting) == "coupling":
        norm = shape_mean_factor(lambda u: np.sqrt(np.asarray(shape_fn(u), float)))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("coupling weighting needs a positive <|eta|>")
        for p in orders:
            out[p] = shape_mean_factor(
                lambda u, n=p // 2: np.asarray(shape_fn(u), float) ** (n + 0.5)) / norm
        return out

    u = np.linspace(-1.0, 1.0, max(int(n_quad), 101))
    f = np.clip(np.asarray(shape_fn(u), dtype=float), 0.0, None)
    w = np.sin(rabi_angle(shape_fn, u))
    norm = np.trapz(w, u)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rabi weighting needs a positive sin(theta) normalization")
    for p in orders:
        out[p] = float(np.trapz(w * f ** (p // 2), u) / norm)
    return out


def deconvolve_moments(fit: Dict[str, Any],
                       moments: Dict[int, float]) -> Dict[str, Any]:
    """Divide a SHAPED-probe law by its moments: ``k_2n = K_2n / M_2n``.

    Only call this on a law fitted here from the rows: a stored ``fit`` already had
    it applied, and dividing twice is wrong.
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

    ``delta0`` is excluded by default: it is static, belongs in ``wp_offset_GHz``
    rather than the chirp, and every downstream excursion number means the
    drive-dependent part only.
    """
    coeffs = [fit["coeffs"][f"k{p}"] for p in fit["powers"]]
    total = 0.0
    for p, c in zip(fit["powers"], coeffs):
        if drop_static and p == 0:
            continue
        total += c * float(eta) ** p
    return float(total)


def law_diagnostics(fit: Dict[str, Any], target_eta: float,
                    t_g_ns: float) -> Dict[str, Any]:
    """Whether this law may be chirped with.

    * ``excursion_frac_linewidth`` -- would a chirp do anything? The excursion over
      the resonance half-width ``1/(2 t_g)``.
    * ``last_term_fraction`` -- is the series converged? |highest kept term| over
      |lowest drive-dependent term| at eta* (generalizes ``quartic_fraction``).
    * ``extrapolation_ratio`` -- eta* over the largest fitted eta; above 1 the law is
      read outside its rows, which no r2 can detect.
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

    Returns the law, its diagnostics and the row census; raises ValueError when too
    few rows survive (see `fit_law`). A SHAPED probe needs `config` to divide the
    envelope moments back out; without it the fit is flagged ``moments_unavailable``
    and stays in measured (not pointwise) units.
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


#: Named policies, so a policy travels through a report or worker payload as a
#: picklable string.
POLICIES: Dict[str, Dict[str, Any]] = {
    "stored": {},                                   # the pipeline today (baseline)
    # more orders of the Stark expansion, fitted over every row (no extrapolation)
    "k6": {"powers": (0, 2, 4, 6)},
    "k8": {"powers": (0, 2, 4, 6, 8)},
    # better conditioned, but read outside the fitted rows (see extrapolation_ratio)
    "narrow_fit_0.8": {"eta_max_frac": 0.8},
    "narrow_fit_0.6": {"eta_max_frac": 0.6},
    # a run without the `_too_wide` span growth, reconstructed from central grids
    "no_widen": {"n_narrow": 15, "refit_centers": True},
    "no_widen_k6": {"n_narrow": 15, "refit_centers": True,
                    "powers": (0, 2, 4, 6)},
}


def policy_kwargs(name: str, target_eta: float) -> Dict[str, Any]:
    """Resolve a named policy against a column's eta*; ``eta_max_frac`` is a
    FRACTION of eta* so one name means the same thing at every eta*."""
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    kw = dict(POLICIES[name])
    frac = kw.pop("eta_max_frac", None)
    if frac is not None:
        kw["eta_max"] = float(frac) * float(target_eta)
    return kw
