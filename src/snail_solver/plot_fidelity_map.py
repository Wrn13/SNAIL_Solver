"""Side-by-side DRAG / no-DRAG fidelity heatmaps over the target allocation grid.

Pivots a ``target`` sweep's ``summary.csv`` (partner frequency :math:`\\omega_b` =
``wb_GHz`` vs spectator :math:`\\omega_c` = ``spec_GHz``) onto a grid and renders
no-DRAG and DRAG heatmaps on one shared colour scale, optionally with a dotted
``nearest_beat = 0`` collision contour.

Two DRAG layouts are handled:

* separate ``--drags true,false`` rows, routed by ``drag_applied``;
* in-row ``--drag-compare``: ``F_avg`` is no-DRAG and ``F_avg_drag`` is DRAG. Here
  ``drag_applied`` is unreliable, so the presence of ``F_avg_drag`` takes precedence.

Where both exist for a cell, the dedicated DRAG-on row wins.

CLI
---
``python -m snail_solver.plot_fidelity_map --outdir results/<run> [--metric fidelity|infidelity]
[--log/--no-log] [--cmap NAME] [--no-collision-line] [--diff] [--out fig.png]``
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def _to_float(x: object) -> Optional[float]:
    """Parse a CSV cell into a finite float, or ``None``."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _to_bool(x: object) -> bool:
    """Parse a CSV cell into a bool (``True`` for ``true``/``1``/``yes``)."""
    return str(x).strip().lower() in ("true", "1", "yes")


def load_summary_rows(path: str) -> List[Dict[str, str]]:
    """Read ``summary.csv`` (from a sweep directory or a direct path) into row dicts.

    Raises ``FileNotFoundError`` if the CSV does not exist (``collect`` not run).
    """
    csv_path = os.path.join(path, "summary.csv") if os.path.isdir(path) else path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found; run `collect` first.")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_fidelity_grids(
    rows: List[Dict[str, str]],
    metric: str = "fidelity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot target-sweep rows onto the :math:`(\\omega_b, \\omega_c)` grid.

    ``metric="fidelity"`` maps each cell to ``F_avg``; ``"infidelity"`` to ``1 - F_avg``.

    Returns
    -------
    wb_vals, spec_vals : ndarray
        Sorted unique ``wb_GHz`` (x) and ``spec_GHz`` (y) values.
    z_nodrag, z_drag, beat_grid : ndarray, shape (n_spec, n_wb)
        Metric without / with DRAG and ``nearest_beat_GHz``; ``nan`` where missing.

    Raises
    ------
    KeyError
        If the rows are not from a target sweep (no ``wb_GHz``/``spec_GHz``).
    """
    if not rows or "wb_GHz" not in rows[0] or "spec_GHz" not in rows[0]:
        raise KeyError(
            "these rows are not a target sweep (need 'wb_GHz' and 'spec_GHz'); "
            "this tool plots the (omega_b, omega_c) allocation grid."
        )
    if metric not in ("fidelity", "infidelity"):
        raise ValueError("metric must be 'fidelity' or 'infidelity'")

    Key = Tuple[float, float]
    nodrag: Dict[Key, float] = {}
    drag_ded: Dict[Key, float] = {}   # dedicated DRAG-on rows
    drag_cmp: Dict[Key, float] = {}   # in-row F_avg_drag compare
    beat: Dict[Key, float] = {}

    for r in rows:
        wb, sp = _to_float(r.get("wb_GHz")), _to_float(r.get("spec_GHz"))
        if wb is None or sp is None:
            continue
        key = (round(wb, 9), round(sp, 9))
        f_avg = _to_float(r.get("F_avg"))
        f_drag = _to_float(r.get("F_avg_drag"))
        b = _to_float(r.get("nearest_beat_GHz"))
        if b is not None:
            beat[key] = b
        if f_drag is not None:
            # in-row compare: drag_applied is True here even though F_avg is DRAG-off
            drag_cmp[key] = f_drag
            dst = nodrag
        else:
            dst = drag_ded if _to_bool(r.get("drag_applied")) else nodrag
        if f_avg is not None:
            dst[key] = f_avg

    drag = {**drag_cmp, **drag_ded}  # dedicated DRAG-on wins where both exist

    keys = set(nodrag) | set(drag) | set(beat)
    wb_vals = np.array(sorted({k[0] for k in keys}))
    spec_vals = np.array(sorted({k[1] for k in keys}))
    wb_ix = {w: i for i, w in enumerate(wb_vals)}
    sp_ix = {s: i for i, s in enumerate(spec_vals)}

    def _grid(src: Dict[Key, float], transform: bool) -> np.ndarray:
        dst = np.full((spec_vals.size, wb_vals.size), np.nan)
        for (w, s), v in src.items():
            dst[sp_ix[s], wb_ix[w]] = (1.0 - v) if (transform and metric ==
                                                    "infidelity") else v
        return dst

    return (wb_vals, spec_vals, _grid(nodrag, True), _grid(drag, True),
            _grid(beat, False))


def _load_grids(source: str, metric: str):
    """``build_fidelity_grids`` on ``source``; raise if the grid is empty."""
    grids = build_fidelity_grids(load_summary_rows(source), metric=metric)
    if grids[0].size == 0 or grids[1].size == 0:
        raise ValueError("no (wb_GHz, spec_GHz) grid found in the summary.")
    return grids


def _has_beat_crossing(beat: np.ndarray) -> bool:
    """True if ``beat`` changes sign somewhere (so a zero contour exists)."""
    return bool(np.isfinite(beat).any() and (np.nanmin(beat) < 0 < np.nanmax(beat)))


def _masked_cmap(name: str):
    """Copy of colormap ``name`` with missing cells drawn light grey."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("0.85")
    return cmap


def _save(fig, source: str, out: Optional[str], default_name: str) -> str:
    """Save ``fig`` to ``out`` (or ``<source dir>/figs/<default_name>``) and close it."""
    if out is None:
        base = source if os.path.isdir(source) else os.path.dirname(source) or "."
        os.makedirs(os.path.join(base, "figs"), exist_ok=True)
        out = os.path.join(base, "figs", default_name)
    else:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fidelity_map(
    source: str,
    *,
    metric: str = "fidelity",
    log: Optional[bool] = None,
    cmap: Optional[str] = None,
    collision_line: bool = True,
    out: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Render the DRAG vs no-DRAG fidelity heatmaps and return the PNG path.

    Parameters
    ----------
    source : str
        Sweep output directory (or a direct ``summary.csv`` path).
    metric : {"fidelity", "infidelity"}, optional
        Colour quantity; ``"infidelity"`` with a log scale shows collision ridges best.
    log : bool or None, optional
        Log colour scale; ``None`` = log for infidelity, linear for fidelity.
    cmap : str or None, optional
        ``None`` = ``"viridis"`` for fidelity, ``"inferno"`` for infidelity.
    collision_line : bool, optional
        Overlay the dotted ``nearest_beat_GHz = 0`` (one-pump spectator resonance)
        contour.
    out : str or None, optional
        ``None`` writes ``<dir>/figs/fidelity_map_<metric>.png``.
    title : str or None, optional
        Figure suptitle.

    Missing cells render light grey.
    """
    if log is None:
        log = (metric == "infidelity")
    if cmap is None:
        cmap = "inferno" if metric == "infidelity" else "viridis"

    wb, spec, z_off, z_on, beat = _load_grids(source, metric)

    finite = np.concatenate([z_off[np.isfinite(z_off)].ravel(),
                             z_on[np.isfinite(z_on)].ravel()])
    if finite.size == 0:
        raise ValueError("summary has no finite fidelities to plot.")
    if log:
        pos = finite[finite > 0]
        vmin = float(pos.min()) if pos.size else 1e-4
        vmax = float(finite.max())
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.0001))
    else:
        norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

    cmap_obj = _masked_cmap(cmap)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             constrained_layout=True, sharey=True)
    wb_mesh, sp_mesh = np.meshgrid(wb, spec)
    draw_beat = collision_line and _has_beat_crossing(beat)

    im = None
    for ax, z, lab in ((axes[0], z_off, "no DRAG"), (axes[1], z_on, "with DRAG")):
        im = ax.pcolormesh(wb, spec, np.ma.masked_invalid(z), shading="nearest",
                           cmap=cmap_obj, norm=norm)
        if draw_beat:
            ax.contour(wb_mesh, sp_mesh, beat, levels=[0.0],
                       colors="white", linestyles=":", linewidths=1.1, alpha=0.75)
        ax.set_xlabel(r"$\omega_b$ (GHz)")
        ax.set_title(lab + ("" if np.isfinite(z).any() else "  (no data)"), fontsize=11)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel(r"$\omega_c$ spectator (GHz)")

    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label(("infidelity $1-F$" if metric == "infidelity"
                  else r"fidelity $F$") + ("  (log)" if log else ""))
    if title is None:
        title = "iSWAP allocation: DRAG vs no DRAG"
        if draw_beat:
            title += "   (dotted: spectator resonance, beat $=0$)"
    fig.suptitle(title, fontsize=12)
    return _save(fig, source, out, f"fidelity_map_{metric}.png")


def _peak_2d(Z: np.ndarray, wb: np.ndarray, spec: np.ndarray):
    """``(F, wb, spec)`` at the max of ``Z`` (rows = spec, cols = wb), or
    ``(nan, None, None)`` if ``Z`` is entirely non-finite."""
    if not np.isfinite(Z).any():
        return (float("nan"), None, None)
    i, j = np.unravel_index(int(np.nanargmax(Z)), Z.shape)
    return (float(Z[i, j]), float(wb[j]), float(spec[i]))


def _fidelity_summary(wb: np.ndarray, spec: np.ndarray,
                      z_off: np.ndarray, z_on: np.ndarray) -> Dict[str, object]:
    """Best-fidelity summary; ``best_of_both`` is ``np.fmax`` (a cell present in
    only one condition keeps that value)."""
    best = np.fmax(z_off, z_on)
    return {
        "wb": wb, "spec": spec,
        "F_nodrag": z_off, "F_drag": z_on, "best_of_both": best,
        "nodrag": _peak_2d(z_off, wb, spec),
        "drag": _peak_2d(z_on, wb, spec),
        "overall": _peak_2d(best, wb, spec),
    }


def best_fidelities(source: str) -> Dict[str, object]:
    """Best average iSWAP fidelity with and without DRAG.

    Returns a dict with grid axes ``wb``, ``spec``; 2-D grids ``F_nodrag``,
    ``F_drag``, ``best_of_both`` (shape ``(n_spec, n_wb)``, ``nan`` = missing); and
    ``nodrag`` / ``drag`` / ``overall`` peaks as ``(F, wb, spec)`` tuples
    (``(nan, None, None)`` if that condition has no data).
    """
    rows = load_summary_rows(source)
    wb, spec, z_off, z_on, _beat = build_fidelity_grids(rows, metric="fidelity")
    return _fidelity_summary(wb, spec, z_off, z_on)


def plot_diff_and_best(
    source: str,
    *,
    collision_line: bool = True,
    diff_pct: float = 95.0,
    out: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[str, Dict[str, object]]:
    """Render ``F_DRAG - F_noDRAG`` beside the best-of-both map.

    Left: :math:`\\Delta F` on a diverging scale (blue = DRAG better, red = worse).
    Right: :math:`\\max(F_{\\mathrm{DRAG}}, F_{\\mathrm{no\\,DRAG}})`.

    ``diff_pct`` is the percentile of ``|ΔF|`` setting the symmetric diff range;
    larger cells saturate so collision bands (where DRAG calibration mislocates)
    don't wash out small genuine differences. ``out=None`` writes
    ``<dir>/figs/fidelity_diff_best.png``.

    Returns the PNG path and the :func:`best_fidelities` summary dict.
    """
    wb, spec, z_off, z_on, beat = _load_grids(source, "fidelity")
    info = _fidelity_summary(wb, spec, z_off, z_on)

    diff = z_on - z_off                                   # +ve => DRAG improves F
    finite_d = np.abs(diff[np.isfinite(diff)])
    m = float(np.percentile(finite_d, diff_pct)) if finite_d.size else 1.0
    m = max(m, 1e-6)

    wb_mesh, sp_mesh = np.meshgrid(wb, spec)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             constrained_layout=True, sharey=True)

    im0 = axes[0].pcolormesh(wb, spec, np.ma.masked_invalid(diff), shading="nearest",
                             cmap=_masked_cmap("RdBu"), norm=Normalize(vmin=-m, vmax=+m))
    axes[0].set_title(r"$\Delta F = F_{\mathrm{DRAG}} - F_{\mathrm{no\,DRAG}}$"
                      "\n(blue: DRAG better, red: worse)", fontsize=10)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.9, pad=0.02)
    cb0.set_label(rf"$\Delta F$  (saturates at $\pm{m:.3f}$)")

    fb = info["best_of_both"][np.isfinite(info["best_of_both"])]
    norm_b = Normalize(vmin=float(fb.min()), vmax=float(fb.max())) if fb.size else None
    im1 = axes[1].pcolormesh(wb, spec, np.ma.masked_invalid(info["best_of_both"]),
                             shading="nearest", cmap=_masked_cmap("viridis"), norm=norm_b)
    axes[1].set_title(r"best of both: $\max(F_{\mathrm{DRAG}},\,F_{\mathrm{no\,DRAG}})$",
                      fontsize=10)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.9, pad=0.02)
    cb1.set_label(r"fidelity $F$")

    draw_beat = collision_line and _has_beat_crossing(beat)
    for ax in axes:
        if draw_beat:
            ax.contour(wb_mesh, sp_mesh, beat, levels=[0.0], colors="0.3",
                       linestyles=":", linewidths=1.0, alpha=0.7)
        ax.set_xlabel(r"$\omega_b$ (GHz)")
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel(r"$\omega_c$ spectator (GHz)")

    fig.suptitle(title or "DRAG vs no-DRAG: difference and best-of-both", fontsize=12)
    return _save(fig, source, out, "fidelity_diff_best.png"), info


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.plot_fidelity_map", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", "--csv", dest="source", required=True,
                    help="sweep directory (with summary.csv) or a summary.csv path")
    ap.add_argument("--metric", choices=["fidelity", "infidelity"], default="fidelity",
                    help="colour quantity (default: fidelity)")
    ap.add_argument("--log", dest="log", action="store_true", default=None,
                    help="log colour scale (default: auto -- log for infidelity)")
    ap.add_argument("--no-log", dest="log", action="store_false",
                    help="force a linear colour scale")
    ap.add_argument("--cmap", default=None,
                    help="matplotlib colormap (default: viridis / inferno)")
    ap.add_argument("--no-collision-line", dest="collision_line", action="store_false",
                    help="do not overlay the beat=0 spectator-resonance contour")
    ap.add_argument("--diff", action="store_true",
                    help="render the DRAG-minus-no-DRAG difference and best-of-both "
                         "panels, and print the best fidelity in each condition")
    ap.add_argument("--diff-pct", type=float, default=95.0,
                    help="percentile of |ΔF| for the diff colour range (default 95)")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--title", default=None, help="figure suptitle")
    args = ap.parse_args()

    if args.diff:
        path, info = plot_diff_and_best(args.source, collision_line=args.collision_line,
                                        diff_pct=args.diff_pct, out=args.out,
                                        title=args.title)

        def _fmt(peak) -> str:
            F, w, s = peak
            return ("n/a (no data)" if w is None
                    else f"F = {F:.4f}  (1-F = {1 - F:.3e})  at  "
                         f"omega_b = {w:.3f} GHz, omega_c = {s:.3f} GHz")

        print(f"best without DRAG : {_fmt(info['nodrag'])}")
        print(f"best with DRAG    : {_fmt(info['drag'])}")
        print(f"best of both      : {_fmt(info['overall'])}")
        d = info["F_drag"] - info["F_nodrag"]
        both = np.isfinite(d)
        if both.any():
            print(f"DRAG improves F in {float((d[both] > 0).mean()) * 100:.0f}% of the "
                  f"{int(both.sum())} cells where both conditions ran "
                  f"(median dF = {float(np.median(d[both])):+.4f}).")
        print(f"wrote {path}")
        return

    path = plot_fidelity_map(args.source, metric=args.metric, log=args.log,
                             cmap=args.cmap, collision_line=args.collision_line,
                             out=args.out, title=args.title)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
