"""Power spectra of the pulses, |FT{eta(t)}|^2 vs drive frequency.

Three pulses per column (bare, chirp, chirp + DRAG with the fixed sign; see
`drag_sign_common`), rebuilt from the stored operating points.

Usage:  plot_pulse_fourier.py OUTDIR [DELTA_MHz,...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drag_sign_common as D  # noqa: E402

SPAN_MHz = 60.0
OFF = np.linspace(-SPAN_MHz, SPAN_MHz, 1201) * 1e-3          # GHz
VARIANTS = (
    ("bare",  "#eb6834", (0, (1, 1.5)), "bare"),
    ("chirp", "#1baf7a", (0, (5, 2)),   "chirp"),
    ("fixed", "#4a3aa7", (0, (6, 1.5, 1.5, 1.5)), "chirp + DRAG, sign fixed"),
)


def main():
    outdir = sys.argv[1]
    deltas = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [-185, -180, -175, -170, -165, -160])
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
    for ax, d in zip(axes.ravel(), deltas):
        c = D.column(d)
        f_GHz = c["w_p_GHz"] + OFF
        for key, col, ls, lab in VARIANTS:
            t, eta = D.eta_nominal(c["variants"][key])
            ax.plot(f_GHz, np.abs(D.spectrum(t, eta, OFF)) ** 2, color=col, ls=ls, lw=1.6,
                    label=lab)
        ax.set_title(rf"$\delta$ = {d:+.0f} MHz", fontsize=10)
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel(r"power $|\mathrm{FT}\{\eta(t)\}|^2$ (ns$^2$)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pulse_fourier_spectra.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
