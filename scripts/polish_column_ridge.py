"""Re-render one column's chirp-ridge figure with presentation labels. No solves.

Draws through ``tune_up.plot_chirp_ridge`` unchanged (the running pipeline imports
it), then relabels the captured figure: bare axis labels, a symbolic title, and a
legend without entries that drew nothing.

    .venv/bin/python scripts/polish_column_ridge.py \
        results/drag_curve_5MHz_2026-09-22/passC_ridge/eta1p3_full.h5 d0p035_eta1p3
"""
import argparse
import os

import numpy as np

from snail_solver import tune_up
from snail_solver.h5_io import load_doc
from snail_solver.tune_up import plot_chirp_ridge, shape_config


def _title(col):
    bits = []
    if "delta_GHz" in col:
        bits.append(rf"$\delta$ = {float(col['delta_GHz']) * 1e3:+.0f} MHz")
    if "w_p_GHz" in col:
        bits.append(rf"$\omega_p/2\pi$ = {float(col['w_p_GHz']):.3f} GHz")
    eta = col.get("target_eta", (col.get("operating_point") or {}).get("target_eta"))
    if eta is not None:
        bits.append(rf"$|\eta^*|$ = {float(eta):g}")
    return "Chirped pump on the Rabi map:  " + ",  ".join(bits)


def _polish(fig, title):
    ax = fig.axes[0]
    ax.set_ylabel(r"drive strength $|\eta|$")
    ax.set_title(title, fontsize=11)
    # The flat carrier retraces itself on the ramp-down, which fills in its dashes.
    for ln in ax.get_lines():
        if ln.get_label().startswith("flat carrier"):
            y = np.asarray(ln.get_ydata(), float)
            k = int(np.argmax(y)) + 1
            ln.set_data(np.asarray(ln.get_xdata(), float)[:k], y[:k])
    # An all-NaN series (e.g. no stored ridge points) still gets a legend entry.
    keep = [ln for ln in ax.get_lines()
            if not ln.get_label().startswith("_")
            and np.isfinite(np.asarray(ln.get_xdata(), float)).any()
            and np.isfinite(np.asarray(ln.get_ydata(), float)).any()]
    ax.legend(handles=keep, fontsize=8, framealpha=0.95, loc="lower left")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scan", help="scan .h5 holding /columns/<tag>")
    ap.add_argument("tag", help="column tag, e.g. d0p035_eta1p3")
    ap.add_argument("--out", default=None,
                    help="output stem (default: <scan dir>/column_figs/<tag>_ridge_polished)")
    args = ap.parse_args()

    doc = load_doc(args.scan)
    col = dict((doc.get("columns") or {})[args.tag])
    for r in ((doc.get("scan") or doc).get("rows") or []):
        if r.get("tag") == args.tag:
            for k in ("delta_GHz", "w_p_GHz", "target_eta"):
                col.setdefault(k, r.get(k))
    st, op = col["stages"], col["operating_point"]
    shape, shape_kw = shape_config(col.get("device") or {})
    stem = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scan)),
                                    "column_figs", f"{args.tag}_ridge_polished")

    captured = {}
    tune_up._savefig = lambda fig, out: captured.setdefault("fig", fig) and out
    plot_chirp_ridge(st["rabi"], st["chirp"], float(op.get("wp_offset_GHz") or 0.0),
                     float(op["t_g_ns"]), "unused.png", shape=shape, shape_kw=shape_kw,
                     ridge_law=st.get("ridge_law"))
    fig = captured["fig"]
    _polish(fig, _title(col))
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        print(f"{stem}.{ext}")


if __name__ == "__main__":
    main()
