"""Fit 1-F(delta) to a sum of resonance tails at frequency-determined positions.

A parasitic channel excites with ``P ~ 2 (g/Delta)^2`` and every ``Delta`` is linear
in ``delta``, so each contributes a squared-Lorentzian tail whose POSITION is fixed by
a frequency condition, not fitted:

    delta = 0                   2 w_p = w_a            A subharmonic
    delta = alpha/2             2 w_p = w_a + alpha    A |1>->|2> ladder
    delta = (w_s - w_a)/2       2 w_p = w_s            SNAIL subharmonic
    delta = w_s - 1.5 w_a         w_p = w_s - w_a      A-SNAIL spectator
    delta = w_s/3 - w_a/2       3 w_p = w_s            third harmonic on the SNAIL

Only amplitudes and widths are free. The last one is SECOND ORDER in g3 (3 pump + 1
coupler is four letters; ``expand_terms`` expands ``g3 X^3``), so `spectator_audit` is
blind to it -- yet it sits at -183 MHz on the 4.7 GHz device, on a band of poor columns
the first-order audit cannot explain.

Independent errors compose, so the total saturates; fitted in log space (the data
spans three decades):

    1 - F = 1 - exp(-[f0 + sum_j A_j / ((delta - delta_j)^2 + w_j^2)])

LIMITS (measured). Over the full axis R^2(log) ~ 0.29-0.35 (2026-09-17 grid), because
the dominant columns break the premise: at ``delta = +200`` the SNAIL subharmonic has
``g/|Delta| = 0.38`` and the coupler holds 0.85 photons, so a 14% ratio change gives a
35x infidelity change -- a THRESHOLD no fixed-pole model reproduces (11x under). On the
perturbative subset (``--max-infidelity 0.05``) the bare series fits at R^2 ~ 0.68 and
the SNAIL subharmonic comes out ~12x the A subharmonic. So: a DIAGNOSTIC on perturbative
columns, not a presentation curve, and never across drives -- the 3 w_p = w_s feature
moves with eta (-170 MHz at 1.2, -180 at 1.3, -190 at 1.5).

Usage:
    fit_resonance_model.py CURVES.json [--baseline BASE.json] [--device D.json]
                           [--eta 1.3] [--max-infidelity 0.05] [--out FIT.json]
"""
import argparse
import json

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:                                     # pragma: no cover
    raise SystemExit("fit_resonance_model needs scipy")


def resonances(device_path):
    """[(delta_MHz, name, order)] -- positions are frequency conditions, not fits."""
    dev = json.load(open(device_path))
    w_a = float(dev["qubit_freqs_GHz"][0])
    w_s = float(dev["coupler_freq_GHz"])
    alpha = float(dev["anharm_qubit_GHz"])
    return [(0.0, "A subharmonic  2wp=wa", 1),
            (alpha * 1e3 / 2.0, "A |1>->|2>     2wp=wa+a", 1),
            ((w_s - w_a) * 1e3 / 2.0, "SNAIL subharm  2wp=ws", 1),
            ((w_s - 1.5 * w_a) * 1e3, "A-SNAIL spect   wp=ws-wa", 1),
            ((w_s / 3.0 - w_a / 2.0) * 1e3, "3wp=ws (2nd order in g3)", 2)]


def _series(rows, base, eta, name):
    d, y = [], []
    for r in rows:
        if round(float(r["target_eta"]), 3) != round(eta, 3) or not r.get("ok"):
            continue
        dm = round(r["delta_GHz"] * 1e3)
        src = base.get((dm, round(eta, 3))) if (base and name == "bare") else r
        t = ((src or {}).get("traces") or {}).get(name)
        if t:
            d.append(dm)
            y.append(1.0 - t["F_avg"])
    return np.array(d, float), np.array(y, float)


def _amp_width(p, j, n, shared_width):
    """(A_j, w_j) from the log-parameter vector of an `n`-channel model."""
    if shared_width:
        return np.exp(p[1 + j]), np.exp(p[1 + n])
    return np.exp(p[1 + 2 * j]), np.exp(p[2 + 2 * j])


def model(p, d, pos, shared_width=False):
    S = np.full_like(d, np.exp(p[0]))
    for j, x0 in enumerate(pos):
        A, w = _amp_width(p, j, len(pos), shared_width)
        S = S + A / ((d - x0) ** 2 + w ** 2)
    return 1.0 - np.exp(-S)


def fit(d, y, pos, min_width=5.0, max_width=400.0, shared_width=False):
    """Widths are bounded below by ~the pulse bandwidth: unbounded, a width runs to
    zero and the fit degenerates.

    `shared_width` ties every channel to ONE width (``n+2`` parameters, not ``2n+1``):
    the perturbative subset has only ~11 columns, where a width each would leave the
    fit exactly determined and its R^2 meaningless.
    """
    if shared_width:
        lo = [np.log(1e-8)] + [np.log(1e-3)] * len(pos) + [np.log(min_width)]
        hi = [np.log(1e-1)] + [np.log(1e7)] * len(pos) + [np.log(max_width)]
        p0 = [np.log(2e-3)] + [np.log(50.0)] * len(pos) + [np.log(30.0)]
    else:
        lo = [np.log(1e-8)] + sum(([np.log(1e-3), np.log(min_width)] for _ in pos), [])
        hi = [np.log(1e-1)] + sum(([np.log(1e7), np.log(max_width)] for _ in pos), [])
        p0 = [np.log(2e-3)] + sum(([np.log(50.0), np.log(30.0)] for _ in pos), [])
    r = least_squares(lambda p: np.log(model(p, d, pos, shared_width)) - np.log(y),
                      p0, bounds=(lo, hi), max_nfev=40000)
    pred = model(r.x, d, pos, shared_width)
    resid = np.log(y) - np.log(pred)
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)
    return r.x, pred, float(r2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("curves")
    ap.add_argument("--baseline", default=None,
                    help="curves.json from a --max-drag-channels 0 run; its `bare` "
                         "supersedes the main run's, which is calibrated DRAG-aware "
                         "and so reads ~2.4x too good")
    ap.add_argument("--device", default="devices/6Gate4.7SNAIL.json")
    ap.add_argument("--eta", type=float, default=1.3)
    ap.add_argument("--max-infidelity", type=float, default=1.0,
                    help="drop columns above this. The model assumes P ~ 2(g/D)^2; "
                         "above ~0.05 the coupler is non-perturbative and the "
                         "response is a threshold, which no fixed-pole model fits.")
    ap.add_argument("--shared-width", action="store_true",
                    help="tie all channels to one width: n+2 parameters instead of "
                         "2n+1. Needed on the perturbative subset, which has too few "
                         "columns to support a width each.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.load(open(args.curves))
    base = {}
    if args.baseline:
        base = {(round(r["delta_GHz"] * 1e3), round(float(r["target_eta"]), 3)): r
                for r in json.load(open(args.baseline))}
    res = resonances(args.device)
    pos = [x for x, _n, _o in res]
    print("resonances (MHz), positions FIXED by frequency conditions:")
    for x, nm, order in res:
        print(f"  {x:+8.1f}   {nm}" + ("   [invisible to the audit]" if order > 1
                                       else ""))
    out = {}
    for name, lab in (("bare", "no chirp, no DRAG"), ("chirp+DRAG", "chirp + DRAG")):
        d, y = _series(rows, base, args.eta, name)
        keep = y < args.max_infidelity
        d, y = d[keep], y[keep]
        npar = (len(pos) + 2) if args.shared_width else (2 * len(pos) + 1)
        if d.size <= npar:
            print(f"\n=== {lab}: {d.size} columns against {npar} free parameters -- "
                  f"skipped. A fit with no degrees of freedom has a meaningless R^2; "
                  f"use --shared-width to drop to {len(pos) + 2}.")
            continue
        p, pred, r2 = fit(d, y, pos, shared_width=args.shared_width)
        print(f"\n=== {lab}:  n = {d.size},  R^2(log) = {r2:.4f} ===")
        print(f"    floor 1-F = {np.exp(p[0]):.3e}")
        amps = {}
        for j, (x0, nm, _o) in enumerate(res):
            A, w = map(float, _amp_width(p, j, len(pos), args.shared_width))
            amps[nm] = {"delta_MHz": x0, "A_MHz2": A, "width_MHz": w}
            print(f"    {nm:<26} A = {A:>10.1f} MHz^2   w = {w:>6.1f} MHz")
        k = np.argsort(-np.abs(np.log(pred) - np.log(y)))[:3]
        print("    worst: " + ", ".join(
            f"{d[i]:+.0f} MHz obs {y[i]:.2e} fit {pred[i]:.2e}" for i in k))
        if r2 < 0.5:
            hint = ("" if args.max_infidelity <= 0.05 else
                    "; try --max-infidelity 0.05")
            print(f"    NOTE R^2 < 0.5 -- the perturbative premise does not hold "
                  f"here{hint}")
        out_dof = d.size - npar
        print(f"    degrees of freedom: {out_dof}")
        out[name] = {"r2_log": r2, "n": int(d.size),
                     "dof": int(out_dof), "shared_width": bool(args.shared_width),
                     "floor": float(np.exp(p[0])), "channels": amps,
                     "delta_MHz": d.tolist(), "observed": y.tolist(),
                     "fitted": pred.tolist()}
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
