"""Clean pump windows from the wide-band Floquet map (floquet_wide_map.py), process-agnostic.

The named processes of analyze_floquet_map.py only exist near w_p = w_a/2. Across
10 MHz .. 2.5 GHz the leak partners change, so this reads the map without a list.

Hybridisation at the pulse peak is reversible: a ramp that passes an anticrossing fully
adiabatically or fully diabatically returns the state on the way down. What leaks is a
crossing passed at an intermediate rate, so the metric is Landau-Zener:

  crossings   for each computational state s in {000, 100, 010, 110}: at each |eta| the
              branch j1 holding most of s, and the most-s-carrying branch j2 that is NOT
              a computational-block branch (block {000} | {100, 010} | {110}; mixing
              100 <-> 010 is the gate). For a two-level avoided crossing the coupling
              g = Omega sqrt(p (1 - p)) is the same at every point along it (Omega the
              quasienergy gap, p the weight of s on j2), so g is read even where the gap
              falls between grid points. A crossing is a local minimum of the diabatic
              distance d = sqrt(Omega^2 - 4 g^2) against one partner that lies within a
              grid step of zero; its sweep slope s = d d/d|eta| comes from the neighbours.
  leak        for a pulse peaking at |eta|_pk over t_g (the scan's SinePowerRamp, m = 3,
              no plateau): every crossing below the peak is passed twice, at rate
              v = s d|eta|/dt, P = exp(-2 pi g^2 / v) each, average leak 2 P (1 - P);
              summed over crossings, averaged over the four computational states.
  split_MHz   the gap between the two branches with the most |100> + |010> weight:
              2 g_exchange where the gate is resonant. A gate speed proxy.
  L_static    the old hybridisation measure, out-of-block weight above its |eta| = 0
              value (running max): reversible, kept as a diagnostic only.

Writes FDIR/wide_analysis_levels<L>.npz and FDIR/figs/, prints the clean windows per
peak |eta| and checks the predicted leak against the 1 MHz-wide scan's pass A leakage on
the shared 1.5 .. 2.0 GHz.

Usage:  analyze_floquet_wide.py FDIR [--levels 9] [--t-g 120]
                                [--scan results/drag_curve_1MHz_2026-10-23]
"""
import argparse
import glob
import json
import os

import numpy as np

COMP = ("000", "100", "010", "110")
BLOCK = {"000": ("000",), "100": ("100", "010"), "010": ("100", "010"), "110": ("110",)}
CLEAN = (1e-3, 1e-2)
MIN_WINDOW_MHZ = 20.0
ETA_REPORT = (0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.2)
P_FLOOR = 1e-6                 # partner weight below this is numerical noise
SLOPE_REACH = 10               # |eta| steps either side used for the sweep slope
SLOPE_MAX = 5000.0             # MHz per unit |eta|: beyond this a "crossing" is a fold jump
TWO_PI = 2.0 * np.pi


def fold(x, z):
    return (x + z / 2) % z - z / 2


def ramp():
    """(u, du/dtau) of the rise of the scan's pulse, tau = t/t_g, u = |eta| / |eta|_pk."""
    from snail_solver.envelope import SinePowerRamp
    e = SinePowerRamp(amp=1.0, t_g=1.0, m=3, t_rise=0.5)
    tau = np.linspace(0.0, 0.5, 4001)
    return np.asarray(e.value_at(tau)), np.asarray(e.deriv_at(tau))


U, DU = ramp()


def eta_dot(eta_c, eta_pk, t_g):
    """d|eta|/dt (1/ns) where the rise passes eta_c."""
    u = np.clip(np.asarray(eta_c) / eta_pk, 0.0, 1.0)
    return eta_pk * np.interp(u, U, DU) / t_g


def crossings(eta, eps, W, labels, s, wp):
    """[(eta_c, g_MHz, slope_MHz_per_eta, partner)] of state s's avoided crossings."""
    n, dim = eps.shape
    blk = sum(W[b] for b in BLOCK[s])                    # (n, dim) block weight per branch
    rows = []
    for i in range(n):
        j1 = int(np.argmax(W[s][i]))
        w = np.where(blk[i] < 0.5, W[s][i], -1.0)
        w[j1] = -1.0
        j2 = int(np.argmax(w))
        p = float(w[j2])
        if p < P_FLOOR:
            rows.append((j2, np.nan, np.nan))
            continue
        om = abs(fold(eps[i, j1] - eps[i, j2], wp))
        # two-level: g = Omega sqrt(p (1-p)), diabatic distance d = Omega |1 - 2p|
        # (the latter stays well conditioned at p ~ 1/2, unlike sqrt(Omega^2 - 4 g^2))
        rows.append((j2, om * np.sqrt(p * (1.0 - p)), om * abs(1.0 - 2.0 * p)))
    de = eta[1] - eta[0]
    out = []
    for i in range(1, n - 1):
        j, g, d = rows[i]
        if not np.isfinite(d):
            continue
        jl, _gl, dl = rows[i - 1]
        jr, _gr, dr = rows[i + 1]
        if jl != j or jr != j or not (np.isfinite(dl) and np.isfinite(dr)):
            continue
        if not (d <= dl and d <= dr):
            continue
        # sweep slope from the same partner up to SLOPE_REACH steps either side; a
        # resolved (broad) crossing needs the reach, a narrow one gets it from +/-1
        est = []
        for m in range(1, SLOPE_REACH + 1):
            if i - m < 0 or i + m >= n:
                break
            (a, _, da), (b, _, db) = rows[i - m], rows[i + m]
            if a != j or b != j or not (np.isfinite(da) and np.isfinite(db)):
                break
            est.append(0.5 * (da + db) / (m * de))
        slope = float(np.median(est))
        if slope <= 0 or slope > SLOPE_MAX or d > slope * de:
            continue                                     # no zero within a grid step
        out.append((float(eta[i]), float(g), slope, labels[j]))
    return out


def leak_for(cross, eta_pk, t_g):
    """Average leak of one state for a pulse peaking at eta_pk: sum of 2 P (1 - P)."""
    tot = 0.0
    for eta_c, g, slope, _ in cross:
        if eta_c >= eta_pk:
            continue
        v = slope * eta_dot(eta_c, eta_pk, t_g)          # MHz / ns
        if v <= 0:
            continue
        # P = exp(-2 pi g^2 / v), g and v in rad/ns and rad/ns^2
        x = TWO_PI * (TWO_PI * 1e-3 * g) ** 2 / (TWO_PI * 1e-3 * v)
        P = np.exp(-x)
        tot += 2.0 * P * (1.0 - P)
    return tot


def one(args):
    path, t_g = args
    z = np.load(path)
    keys = [str(k) for k in z["watch"]]
    W = {k: z["weights"][i] for i, k in enumerate(keys)}
    labels = [str(x) for x in z["labels"]]
    eps = z["eps_GHz"] * 1e3
    eta = z["eta"]
    wp = float(z["w_p_GHz"]) * 1e3
    n = len(eta)
    cross = {s: crossings(eta, eps, W, labels, s, wp) for s in COMP}
    leak = np.array([[leak_for(cross[s], e, t_g) for e in eta] for s in COMP])
    stat = []
    for s in COMP:
        j = np.argmax(W[s], axis=1)
        inb = sum(W[b][np.arange(n), j] for b in BLOCK[s])
        out_b = np.clip(1.0 - inb, 0.0, 1.0)
        stat.append(np.maximum.accumulate(np.clip(out_b - out_b[0], 0.0, 1.0)))
    d = W["100"] + W["010"]
    top = np.argsort(d, axis=1)[:, -2:]
    e = np.take_along_axis(eps, top, axis=1)
    split = np.abs(fold(e[:, 1] - e[:, 0], wp))
    # the worst crossing below each peak, for naming the process
    flat = [(c, s) for s in COMP for c in cross[s]]
    return dict(w_p=wp, delta=float(z["delta_MHz"]), eta=eta, leak_s=leak,
                leak=leak.mean(axis=0), L_static=np.max(stat, axis=0), split=split,
                cross=flat)


def worst_process(r, eta_pk, t_g):
    best, tag = 0.0, ""
    for (eta_c, g, slope, partner), s in r["cross"]:
        x = leak_for([(eta_c, g, slope, partner)], eta_pk, t_g)
        if x > best:
            k = ""
            best, tag = x, f"{s}->{partner} @|eta| {eta_c:.2f}, g {g:.2g} MHz"
    return best, tag


def windows(wp, ok):
    """Contiguous runs of True as (lo, hi) MHz, at least MIN_WINDOW_MHZ wide."""
    out, start, prev = [], None, None
    for x, o in zip(list(wp) + [np.inf], list(ok) + [False]):
        if o and start is None:
            start = x
        if not o and start is not None:
            if prev - start >= MIN_WINDOW_MHZ:
                out.append((start, prev))
            start = None
        prev = x
    return out


def validate(res, scan, t_g_default):
    """Pass A leakage vs the predicted leak at each column's own t_g, |eta|_pk = 1.3."""
    path = os.path.join(scan, "curves_nodrag_fine.json")
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))
    wp = np.array([r["w_p"] for r in res])
    out = {}
    for var in ("bare", "chirp+DRAG"):
        pts = []
        for row in rows:
            t = row.get("traces", {}).get(var)
            if not row.get("ok") or not t:
                continue
            j = int(np.argmin(np.abs(wp - row["w_p_GHz"] * 1e3)))
            if abs(wp[j] - row["w_p_GHz"] * 1e3) > 1.01:
                continue
            r = res[j]
            pred = np.mean([leak_for([c for c, s in r["cross"] if s == st], row["target_eta"],
                                     t["t_g_ns"]) for st in COMP])
            pts.append((pred, t["leakage"], t["infidelity_coherent"]))
        if len(pts) < 10:
            continue
        p = np.array(pts)
        rk = lambda x: np.argsort(np.argsort(x))
        out[var] = dict(n=len(p), rho_leak=float(np.corrcoef(rk(p[:, 0]), rk(p[:, 1]))[0, 1]),
                        rho_inf=float(np.corrcoef(rk(p[:, 0]), rk(p[:, 2]))[0, 1]),
                        pts=p)
    return out


def figures(fdir, res, wp, eta, leak, split, val):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    os.makedirs(os.path.join(fdir, "figs"), exist_ok=True)
    x = wp * 1e-3
    fig, ax = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    m = ax[0].pcolormesh(x, eta, np.clip(leak.T, 1e-6, 1), norm=LogNorm(1e-6, 1),
                         cmap="magma_r", shading="nearest", rasterized=True)
    ax[0].contour(x, eta, leak.T, levels=list(CLEAN), colors=["#1f77b4", "#2ca02c"],
                  linewidths=0.8)
    fig.colorbar(m, ax=ax[0], label="predicted LZ leak")
    ax[0].set_ylabel("peak |η|")
    ax[0].set_title("Landau–Zener leak of the computational states; contours 1e-3 (blue), 1e-2 (green)")
    m = ax[1].pcolormesh(x, eta, np.clip(split.T, 0.1, 300), norm=LogNorm(0.1, 300),
                         cmap="viridis", shading="nearest", rasterized=True)
    fig.colorbar(m, ax=ax[1], label="|100>/|010> splitting (MHz)")
    ax[1].set_ylabel("|η|")
    ax[1].set_xlabel("pump frequency ω_p / 2π (GHz)")
    for a in ax:
        for xv in (1.2, 4.7 / 2, 4.7 / 3, 1.75):
            a.axvline(xv, color="k", lw=0.5, ls=":")
    for xv, lab in ((1.2, "ω_b=ω_s"), (4.7 / 2, "2ω_p=ω_s"), (4.7 / 3, "3ω_p=ω_s"),
                    (1.75, "ω_a/2")):
        ax[0].text(xv, eta[-1] * 1.01, lab, fontsize=7, ha="center", va="bottom")
    fig.tight_layout()
    p1 = os.path.join(fdir, "figs", "wide_leak_map.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    if not val:
        return [p1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (var, d) in zip(ax, val.items()):
        p = d["pts"]
        a.loglog(np.clip(p[:, 0], 1e-7, 1), np.clip(p[:, 1], 1e-5, 1), ".", ms=3)
        a.plot([1e-7, 1], [1e-7, 1], "k:", lw=0.6)
        a.set_xlabel("Floquet LZ leak (predicted)")
        a.set_ylabel("pass A leakage")
        a.set_title(f"{var}: Spearman ρ = {d['rho_leak']:.2f}")
    fig.tight_layout()
    p2 = os.path.join(fdir, "figs", "wide_leak_vs_passA.png")
    fig.savefig(p2, dpi=150)
    return [p1, p2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fdir")
    ap.add_argument("--levels", type=int, default=9)
    ap.add_argument("--t-g", type=float, default=120.0, help="pulse length for the map (ns)")
    ap.add_argument("--scan", default="results/drag_curve_1MHz_2026-10-23")
    a = ap.parse_args()
    paths = glob.glob(os.path.join(a.fdir, f"levels{a.levels}", "d*.npz"))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(32, os.cpu_count())) as ex:
        res = sorted(ex.map(one, [(p, a.t_g) for p in paths], chunksize=8),
                     key=lambda r: r["w_p"])
    wp = np.array([r["w_p"] for r in res])
    eta = res[0]["eta"]
    leak = np.array([r["leak"] for r in res])
    split = np.array([r["split"] for r in res])
    val = validate(res, a.scan, a.t_g)
    out = os.path.join(a.fdir, f"wide_analysis_levels{a.levels}.npz")
    np.savez_compressed(out, w_p_MHz=wp, eta=eta, comp=np.array(COMP), t_g_ns=a.t_g,
                        leak=leak, leak_s=np.array([r["leak_s"] for r in res]),
                        L_static=np.array([r["L_static"] for r in res]), split_MHz=split,
                        n_cross=np.array([len(r["cross"]) for r in res]))
    print(f"{len(res)} pumps, {wp[0]:.0f}..{wp[-1]:.0f} MHz, t_g = {a.t_g:g} ns -> {out}")
    for p in figures(a.fdir, res, wp, eta, leak, split, val):
        print(f"figure -> {p}")
    if val:
        print("\ncheck vs pass A (1.5..2.0 GHz, each column's own t_g, |eta|_pk = 1.3):")
        for var, d in val.items():
            print(f"  {var:10s} n={d['n']}: Spearman rho vs leakage {d['rho_leak']:.2f}, "
                  f"vs 1-F {d['rho_inf']:.2f}")
    for e in ETA_REPORT:
        i = int(np.argmin(np.abs(eta - e)))
        print(f"\npeak |eta| = {eta[i]:.2f}:")
        for thr in CLEAN:
            ws = windows(wp, leak[:, i] < thr)
            tot = sum(h - l for l, h in ws)
            desc = ", ".join(f"{l / 1e3:.3f}-{h / 1e3:.3f} (2g~{np.median(split[(wp >= l) & (wp <= h), i]):.0f} MHz)"
                             for l, h in ws[:12]) or "none"
            more = f" (+{len(ws) - 12} more)" if len(ws) > 12 else ""
            print(f"  leak < {thr:g}: {tot:.0f} MHz in {len(ws)} windows: {desc}{more}")
    print("\nworst crossing at peak |eta| = 1.3 and 2.2, by band (median leak):")
    edges = [10, 100, 300, 600, 900, 1200, 1400, 1600, 1800, 2000, 2200, 2501]
    for lo, hi in zip(edges, edges[1:]):
        sel = [k for k, r in enumerate(res) if lo <= r["w_p"] < hi]
        parts = []
        for e in (1.3, 2.2):
            i = int(np.argmin(np.abs(eta - e)))
            tags = [worst_process(res[k], eta[i], a.t_g)[1].split(" @")[0] for k in sel]
            tags = [t for t in tags if t]
            c = max(set(tags), key=tags.count) if tags else "-"
            parts.append(f"{e}: {np.median(leak[sel, i]):.1e} {c} [{tags.count(c)}/{len(sel)}]")
        print(f"  {lo / 1e3:.2f}-{hi / 1e3:.2f} GHz  " + " | ".join(parts))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
