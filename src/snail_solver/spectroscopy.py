"""Power-vs-frequency spectroscopy of a transmon transition, with stored data.

Maps drive amplitude x drive frequency, coloured by the population of a target level
(|e> from |g> by default): the direct line bends with power (AC-Stark pull), and
multiphoton features sit at fractions of the transition, e.g. ge/2. The grid is saved
to ``.npz`` so ``--replot`` can restyle a figure without recomputing.

Evolution uses the same ``ZhouCoupler`` as the sweeps, with QuTiP ``sesolve`` on the
exact time-dependent Hamiltonian (no rotating frame, no carrier cutoff), so every
multiphoton process in the model appears. QuTiP is REQUIRED.

MULTIPHOTON ORDERS
------------------
At first order in g3 only ge and ge/2 appear, but higher orders in g3 generate ge/3,
ge/4, ... (weaker). Verified by an independent lab-frame integration at g4 = 0
(w_a = 4.6 GHz, eta = 1.2, 85 ns): peak |e> 0.99 at ge/2, 0.51 at ge/3, 0.05 at ge/4.
Do NOT assume a subharmonic is absent because g4 = 0.

ALLOCATION WARNING
------------------
A subharmonic landing on the gate pump (w_a/n = w_b - w_a) drives a mode out of |g>,
i.e. w_b = w_a (1 + 1/n): 1.5, 1.333, 1.25 w_a for n = 2, 3, 4. The example device
(4.6, 5.7) is 50 MHz from n=4, so its ge/4 line (1.150 GHz) is 50 MHz from the
1.100 GHz pump.

CLI
---
    python -m snail_solver.spectroscopy --device evan_device.json \\
        --f-lo 1.5 --f-hi 3.8 --f-points 121 --amp-hi 3.0 --amp-points 41 \\
        --probe-ns 200 --reduce max --nproc 8 \\
        --save-data results/ge_map.npz --out figs/ge_map.png

    python -m snail_solver.spectroscopy --replot results/ge_map.npz --out figs/ge_restyled.png
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# population helpers
# --------------------------------------------------------------------------- #
def marginal_population(probs: np.ndarray, dims: Sequence[int], mode: int,
                        level: int) -> float:
    """Population of ``level`` (1 = |e>) in ``mode``, marginalised over all other modes.

    ``probs`` is over the full product basis ordered as ``dims``. A single Fock
    amplitude would undercount whenever the drive also excites another mode (the
    coupler is dynamical), so the marginal is the honest analogue of a measured |e>.
    NaN if ``level`` is beyond the mode's truncation.
    """
    t = np.asarray(probs, dtype=float).reshape(tuple(dims))
    if level >= t.shape[mode]:
        return float("nan")
    return float(np.take(t, level, axis=mode).sum())


def expected_number(probs: np.ndarray, dims: Sequence[int], mode: int) -> float:
    """``<n>`` of ``mode``, marginalised over all others.

    The real occupation of a mode (e.g. the coupler), used to check whether a pump
    label ``eta`` (nominally ``sqrt(n_s)``) still matches it once multi-photon
    channels populate the mode.
    """
    return float(sum(level * marginal_population(probs, dims, mode, level)
                     for level in range(1, int(dims[mode]))))


def _reduce_populations(pops: np.ndarray, reduce: str) -> float:
    """Reduce a population time series to one number (see ``reduce`` in scan_ge)."""
    if reduce == "max":
        return float(np.max(pops))
    if reduce == "mean":
        return float(np.mean(pops))
    return float(pops[-1])


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def transition_lines(config: Dict[str, Any], mode: int = 0,
                     orders: Sequence[int] = (1, 2, 3, 4),
                     exchange: bool = True) -> Dict[str, float]:
    """Expected drive frequencies of ge / ef lines and their subharmonics (w / n).

    ``ef`` is one anharmonicity below ``ge``; orders to 4 by default since higher
    orders in g3 populate them even at g4 = 0.
    """
    freqs = list(np.asarray(config["qubit_freqs_GHz"], dtype=float))
    w_ge = float(freqs[mode])
    alpha = float(config.get("anharm_qubit_GHz", 0.0))
    out: Dict[str, float] = {}
    for n in orders:
        out[f"ge/{n}" if n > 1 else "ge"] = w_ge / n
        out[f"ef/{n}" if n > 1 else "ef"] = (w_ge + alpha) / n
    if exchange and len(freqs) > 1:
        # pump-driven exchange (iSWAP) resonances: |w_i - w_j|. Only visible when the
        # initial state carries an excitation -- an exchange annihilates |gg>.
        for j, w_j in enumerate(freqs):
            if j != mode:
                out[f"swap({mode}-{j})"] = abs(w_ge - float(w_j))
        w_c = config.get("coupler_freq_GHz")
        if w_c is not None:
            out[f"swap({mode}-c)"] = abs(w_ge - float(w_c))
    return out


def _scan_column(task: Tuple[int, float, Dict[str, Any], Dict[str, Any]]
                 ) -> Tuple[int, np.ndarray]:
    """Population column over the amplitude axis at one drive frequency.

    ``task = (column_index, drive_frequency_GHz, config, params)``; top-level so it
    pickles for multiprocessing. Modes come from ``device_utils.build_coupler`` (as in
    the sweeps); its pump is replaced by one tone at this column's frequency, so the
    requested amplitude is exactly ``|eta|_peak``. Returns ``(column_index, column)``.
    """
    jf, f_d, config, prm = task
    from snail_solver.device_utils import build_coupler
    from snail_solver.zhou_coupler import PumpTone, RaisedCosine, ConstantPulse

    amps = np.asarray(prm["amps"], dtype=float)
    probe_ns = float(prm["probe_ns"])
    reduce = prm["reduce"]
    EnvCls = RaisedCosine if prm["envelope"] == "raised_cosine" else ConstantPulse
    solver = dict(atol=prm["atol"], rtol=prm["rtol"], nsteps=prm["nsteps"])

    cpl, _wp, _eta = build_coupler(config, t_g=probe_ns, amp_scale=1.0,
                                   wp_offset_GHz=0.0,
                                   spec_abs_GHz=prm["spec_abs_GHz"])
    occ0 = prm.get("init_occ") or [0] * len(cpl.dims)
    if len(occ0) != len(cpl.dims):
        raise SystemExit(f"--init has {len(occ0)} entries but the system has "
                         f"{len(cpl.dims)} modes {list(cpl.dims)}")
    times = (np.linspace(0.0, probe_ns, int(prm["n_times"]))
             if reduce in ("max", "mean") else None)

    def pop(probs):
        return marginal_population(probs, cpl.dims, prm["target_mode"], prm["level"])

    # zero-amplitude baseline: the observable in the undriven initial state
    base = np.zeros(cpl.dim)
    base[cpl.fock_index(occ0)] = 1.0
    base_pop = pop(base)

    col = np.empty(len(amps))
    for ia, amp in enumerate(amps):
        if amp == 0.0:
            col[ia] = base_pop
            continue
        # Deliberately UN-chirped: this is a swept probe at f_d, and the device's
        # `chirp_coeffs_GHz` is defined about the gate's own carrier |w_b - w_a|.
        cpl.set_pump(PumpTone(w_p_GHz=f_d, envelope=EnvCls(amp=float(amp),
                                                          t_g=probe_ns),
                              is_eta=True))
        if times is None:
            psi = cpl.evolve_state(occ0, probe_ns, **solver)
            pops = np.array([pop(np.abs(psi) ** 2)])
        else:
            traj = cpl.evolve_trajectory(occ0, times, **solver)
            pops = np.array([pop(np.abs(psi_t) ** 2) for psi_t in traj])
        col[ia] = _reduce_populations(pops, reduce)
    return jf, col


def scan_ge(config: Dict[str, Any], *, f_lo: float, f_hi: float, f_points: int = 81,
            amp_lo: float = 0.0, amp_hi: float = 2.0, amp_points: int = 31,
            probe_ns: float = 200.0, target_mode: int = 0, level: int = 1,
            envelope: str = "constant", reduce: str = "final",
            n_times: int = 41, rtol: float = 1e-8, atol: float = 1e-10,
            nsteps: int = 500000,
            spec_abs_GHz: Optional[float] = None,
            init_occ: Optional[Sequence[int]] = None,
            nproc: int = 1, columns: Optional[Sequence[int]] = None,
            verbose: bool = True) -> Dict[str, Any]:
    """Scan drive frequency (GHz) x drive amplitude (|eta|), recording a level population.

    The coupler is rebuilt per frequency column, so columns are independent.

    Parameters
    ----------
    probe_ns : float
        Drive duration (ns); linewidth ~ 1/T.
    target_mode, level : int
        Observable: population of ``level`` (1 = |e>) in ``target_mode`` (0 = qubit a).
    envelope : {'constant', 'raised_cosine'}
        ``constant`` is the usual, sharpest probe; ``raised_cosine`` matches the gate.
    reduce : {'final', 'max', 'mean'}
        Reduction over the probe: ``final`` is a fixed-duration measurement (Rabi
        fringes); ``max`` cleanly maps where transitions live. 'max'/'mean' sample
        ``n_times`` output times.
    rtol, atol, nsteps
        QuTiP sesolve settings.
    spec_abs_GHz : float, optional
        Include a spectator at this absolute frequency.
    init_occ : sequence of int, optional
        Initial Fock occupations per mode (default |g...>). This decides which
        resonances can show: from |g> only excitation-creating processes appear; an
        exchange such as the iSWAP conserves excitation number and is invisible. To
        map the iSWAP start from e.g. ``[0, 1, 0]`` and record qubit a's |e>: the
        line then appears at |w_b - w_a|.
    nproc : int
        Worker processes over frequency columns (1 = serial).
    columns : sequence of int, optional
        Compute only these columns (array sharding); the rest stay NaN for
        :func:`merge_shards`.

    Returns
    -------
    dict
        freqs_GHz, amps, Z (shape [amp_points, f_points]), lines, meta.
    """
    freqs = np.linspace(f_lo, f_hi, f_points)
    amps = np.linspace(amp_lo, amp_hi, amp_points)
    Z = np.full((amp_points, f_points), np.nan)
    todo = list(range(f_points)) if columns is None else sorted(set(columns))

    params = dict(amps=amps, probe_ns=probe_ns, target_mode=target_mode, level=level,
                  envelope=envelope, reduce=reduce, n_times=n_times,
                  rtol=rtol, atol=atol, nsteps=nsteps, spec_abs_GHz=spec_abs_GHz,
                  init_occ=(list(init_occ) if init_occ is not None else None))
    tasks = [(jf, float(freqs[jf]), config, params) for jf in todo]

    def collect(results):
        for k, (jf, col) in enumerate(results):
            Z[:, jf] = col
            if verbose:
                print(f"    [{k + 1}/{len(tasks)}] f={freqs[jf]:6.3f} GHz  "
                      f"max pop = {np.nanmax(col):.3f}", flush=True)

    if nproc and nproc > 1 and len(tasks) > 1:
        import multiprocessing as mp
        # 'fork' where available: 'spawn' re-imports __main__ in every worker, which
        # hangs from a notebook/heredoc or an unguarded module. Workers are pure
        # compute and SLURM pins BLAS to one thread, so fork is safe. imap_unordered
        # keeps workers fed since per-column cost grows with drive frequency.
        try:
            ctx = mp.get_context("fork")
        except ValueError:                              # non-POSIX
            ctx = mp.get_context()
        with ctx.Pool(processes=int(nproc)) as pool:
            collect(pool.imap_unordered(_scan_column, tasks))
    else:
        collect(map(_scan_column, tasks))

    return dict(freqs_GHz=freqs, amps=amps, Z=Z,
                lines=transition_lines(config, target_mode),
                meta=dict(probe_ns=probe_ns, target_mode=target_mode, level=level,
                          envelope=envelope, reduce=reduce, n_times=n_times,
                          solver="qutip_sesolve_exact", rtol=rtol, atol=atol,
                          spec_abs_GHz=spec_abs_GHz,
                          init_occ=(list(init_occ) if init_occ is not None else None),
                          columns=list(todo), f_points=int(f_points),
                          qubit_freqs_GHz=list(np.asarray(
                              config["qubit_freqs_GHz"], dtype=float)),
                          coupler_freq_GHz=config.get("coupler_freq_GHz"),
                          anharm_qubit_GHz=config.get("anharm_qubit_GHz"),
                          g3_GHz=config.get("g3_GHz"),
                          g4_GHz=config.get("g4_GHz", 0.0)))


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_scan(result: Dict[str, Any], path: str) -> str:
    """Write the scan grid + metadata to an ``.npz`` so it can be re-plotted."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, freqs_GHz=result["freqs_GHz"], amps=result["amps"],
                        Z=result["Z"], lines_json=json.dumps(result["lines"]),
                        meta_json=json.dumps(result["meta"]))
    print("wrote", path)
    return path


def load_scan(path: str) -> Dict[str, Any]:
    """Load a scan written by :func:`save_scan`."""
    with np.load(path, allow_pickle=False) as d:
        return dict(freqs_GHz=d["freqs_GHz"], amps=d["amps"], Z=d["Z"],
                    lines=json.loads(str(d["lines_json"])),
                    meta=json.loads(str(d["meta_json"])))


def export_csv(result: Dict[str, Any], path: str) -> str:
    """Write the grid as long-form CSV (freq, amp, population) for external tools."""
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drive_freq_GHz", "amp_eta", "population"])
        for ia, amp in enumerate(result["amps"]):
            for jf, f in enumerate(result["freqs_GHz"]):
                w.writerow([f"{f:.6f}", f"{amp:.6f}", f"{result['Z'][ia, jf]:.6f}"])
    print("wrote", path)
    return path


def merge_shards(prefix: str, out_npz: Optional[str] = None) -> Dict[str, Any]:
    """Merge ``<prefix>_shard*.npz`` into one grid (optionally written to ``out_npz``).

    Shards share axes but fill only their own columns (the rest NaN), so this is a
    NaN-aware overlay. Columns no shard filled (failed / running tasks) are reported
    rather than silently plotted as stripes. Returns the merged scan.
    """
    import glob
    paths = sorted(glob.glob(f"{prefix}_shard*.npz"))
    if not paths:
        raise SystemExit(f"no shards matching {prefix}_shard*.npz")
    merged: Optional[Dict[str, Any]] = None
    filled: set = set()
    for path in paths:
        part = load_scan(path)
        if merged is None:
            merged = dict(freqs_GHz=part["freqs_GHz"], amps=part["amps"],
                          Z=np.full_like(part["Z"], np.nan),
                          lines=part["lines"], meta=dict(part["meta"]))
        else:
            # Validate axis VALUES, not just lengths: same point count but different
            # F_LO/F_HI would otherwise merge silently at the wrong frequencies.
            for name in ("freqs_GHz", "amps"):
                if len(part[name]) != len(merged[name]):
                    raise SystemExit(f"{path}: {name} has {len(part[name])} points but "
                                     f"the other shards have {len(merged[name])}")
                if not np.allclose(part[name], merged[name], rtol=0, atol=1e-9):
                    raise SystemExit(
                        f"{path}: {name} values differ from the other shards "
                        f"([{part[name][0]:.6g}..{part[name][-1]:.6g}] vs "
                        f"[{merged[name][0]:.6g}..{merged[name][-1]:.6g}]). These "
                        f"shards are from DIFFERENT scans -- merging them would put "
                        f"data at the wrong frequencies. Re-run one scan with a "
                        f"single set of axis settings.")
            # physics settings must match too, or columns are not comparable
            for key in ("probe_ns", "envelope", "reduce", "level", "target_mode",
                        "init_occ", "qubit_freqs_GHz", "coupler_freq_GHz",
                        "anharm_qubit_GHz", "g3_GHz", "g4_GHz", "spec_abs_GHz"):
                a, b = merged["meta"].get(key), part["meta"].get(key)
                if a != b:
                    raise SystemExit(f"{path}: meta[{key!r}] = {b!r} but the other "
                                     f"shards have {a!r}; these are different scans")
        cols = part["meta"].get("columns")
        cols = range(part["Z"].shape[1]) if cols is None else cols
        for jf in cols:
            if np.isfinite(part["Z"][:, jf]).any():
                merged["Z"][:, jf] = part["Z"][:, jf]
                filled.add(int(jf))
    n_tot = merged["Z"].shape[1]
    missing = sorted(set(range(n_tot)) - filled)
    merged["meta"]["columns"] = sorted(filled)
    merged["meta"]["shards_merged"] = len(paths)
    print(f"merged {len(paths)} shard(s): {len(filled)}/{n_tot} columns filled")
    if missing:
        print(f"WARNING: {len(missing)} column(s) missing (failed/incomplete tasks): "
              f"{missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if out_npz:
        save_scan(merged, out_npz)
    return merged


def missing_columns(result: Dict[str, Any]) -> List[int]:
    """Indices of frequency columns with no finite data (the white stripes)."""
    Z = np.asarray(result["Z"])
    return [int(j) for j in range(Z.shape[1]) if not np.isfinite(Z[:, j]).any()]


def report_missing(prefix: str, nshards: Optional[int] = None) -> Dict[str, Any]:
    """Diagnose gaps (all-NaN columns) in a sharded scan and print how to fill them.

    Sharding is strided, so column j belongs to shard ``j % nshards``; with
    ``nshards`` the gaps map back to the few shard indices to resubmit. Returns
    ``missing`` (columns), ``shards`` (to resubmit) and ``array_spec``.
    """
    merged = merge_shards(prefix)
    missing = missing_columns(merged)
    out: Dict[str, Any] = dict(missing=missing, shards=[], array_spec="")
    n_tot = int(merged["Z"].shape[1])
    if not missing:
        print(f"no gaps: all {n_tot} columns present")
        return out
    print(f"{len(missing)}/{n_tot} column(s) missing")
    if nshards:
        shards = sorted({j % int(nshards) for j in missing})
        out["shards"] = shards
        try:
            from snail_solver.sweep_common import _compress_ranges
            spec = _compress_ranges(shards)
        except Exception:
            spec = ",".join(str(i) for i in shards)
        out["array_spec"] = spec
        print(f"belongs to {len(shards)} shard(s) of {nshards}: {shards}")
        print(f"resubmit just those:\n"
              f"  PREFIX={prefix} NSHARDS={nshards} <other env...> \\\n"
              f"      sbatch --array={spec} slurm/snail_spectroscopy.slurm")
        if max(shards) >= int(nshards):
            print("NOTE: a missing shard index is >= NSHARDS -- the array was smaller "
                  "than NSHARDS, so those shards were never submitted.")
    else:
        print("pass --nshards to map these columns back to shard indices")
    return out


def inspect_scan(path: str) -> Dict[str, Any]:
    """Print a scan's stored axes, probe, observable and device metadata, plus its peak.

    These determine where features are EXPECTED, so a shifted line can be checked
    rather than guessed at.
    """
    r = load_scan(path)
    f, a, m = r["freqs_GHz"], r["amps"], r["meta"]
    df = (f[1] - f[0]) * 1e3 if len(f) > 1 else float("nan")
    print(f"{path}")
    print(f"  freq axis : {f[0]:.6f} .. {f[-1]:.6f} GHz, {len(f)} pts, df = {df:.3f} MHz")
    print(f"  amp axis  : {a[0]:.4f} .. {a[-1]:.4f}, {len(a)} pts")
    print(f"  probe     : {m.get('probe_ns')} ns  -> linewidth ~ "
          f"{1e3 / float(m.get('probe_ns', np.nan)):.2f} MHz")
    print(f"  envelope  : {m.get('envelope')}   reduce: {m.get('reduce')}")
    print(f"  observable: mode {m.get('target_mode')} level {m.get('level')}, "
          f"init = {m.get('init_occ') or 'ground'}")
    print(f"  device    : qubits {m.get('qubit_freqs_GHz')} coupler "
          f"{m.get('coupler_freq_GHz')} alpha {m.get('anharm_qubit_GHz')} "
          f"g3 {m.get('g3_GHz')} g4 {m.get('g4_GHz')}")
    gaps = missing_columns(r)
    if gaps:
        print(f"  GAPS      : {len(gaps)} empty column(s)")
    wq = m.get("qubit_freqs_GHz")
    if wq and len(wq) > 1:
        print(f"  expected swap |w_b - w_a| = {abs(float(wq[1]) - float(wq[0])):.6f} GHz")
    Z = r["Z"]
    if np.isfinite(Z).any():
        ia = int(np.nanargmax(np.nanmax(Z, axis=1)))
        jf = int(np.nanargmax(Z[ia]))
        print(f"  observed peak: {f[jf]:.6f} GHz at amp {a[ia]:.3f} "
              f"(value {Z[ia, jf]:.4f})")
    return r


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def plot_spectroscopy(result: Dict[str, Any], out: str = "figs/ge_map.png",
                      title: Optional[str] = None, annotate: bool = True,
                      cmap: str = "viridis") -> None:
    """Amplitude x frequency map coloured by the recorded level population."""
    freqs, amps, Z = result["freqs_GHz"], result["amps"], result["Z"]
    meta = result.get("meta", {})
    lvl = int(meta.get("level", 1))
    lname = {1: r"|e\rangle", 2: r"|f\rangle"}.get(lvl, rf"|{lvl}\rangle")

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    pcm = ax.pcolormesh(freqs, amps, Z, shading="nearest", cmap=cmap,
                        vmin=0.0, vmax=1.0)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(rf"${lname}$ population")

    if annotate:
        lines = result.get("lines", {})
        lo, hi = float(np.min(freqs)), float(np.max(freqs))
        for label, f0 in sorted(lines.items(), key=lambda kv: kv[1]):
            if not (lo <= f0 <= hi):
                continue
            ax.axvline(f0, color="w", lw=0.7, ls=":", alpha=0.8, zorder=3)
            ax.text(f0, amps[-1], f" {label}", color="w", fontsize=8.5,
                    rotation=90, va="top", ha="left", zorder=4)

    gaps = missing_columns(result)
    if gaps:
        ax.text(0.02, 0.02, f"{len(gaps)} of {Z.shape[1]} columns missing "
                            f"(incomplete shards)", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.5, color="w",
                bbox=dict(boxstyle="round", fc="#B03A2EAA", ec="none"))
        print(f"WARNING: {len(gaps)} all-NaN column(s) render as blank stripes; "
              f"fill them with:  python -m snail_solver.spectroscopy --missing <prefix> "
              f"--nshards <N>")
    ax.set_xlabel("drive frequency (GHz)")
    ax.set_ylabel(r"drive amplitude  $|\eta|$")
    ax.set_title(title or rf"${lname}$ population vs drive frequency and power",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pdf = out.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("wrote", out, "and", pdf)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.spectroscopy", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", default=None,
                    help="render from a stored .npz instead of scanning")
    ap.add_argument("--device", default=None)
    ap.add_argument("--f-lo", type=float, default=1.5)
    ap.add_argument("--f-hi", type=float, default=3.8)
    ap.add_argument("--f-points", type=int, default=81)
    ap.add_argument("--amp-lo", type=float, default=0.0)
    ap.add_argument("--amp-hi", type=float, default=2.0)
    ap.add_argument("--amp-points", type=int, default=31)
    ap.add_argument("--probe-ns", type=float, default=200.0)
    ap.add_argument("--target-mode", type=int, default=0)
    ap.add_argument("--level", type=int, default=1, help="1 = |e>, 2 = |f>")
    ap.add_argument("--n-times", type=int, default=41,
                    help="output times for --reduce max/mean (ignored for 'final')")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--envelope", choices=["constant", "raised_cosine"],
                    default="constant")
    ap.add_argument("--reduce", choices=["final", "max", "mean"], default="final",
                    help="population reduction over the probe: 'final' matches a "
                         "fixed-duration measurement (shows Rabi fringes, and "
                         "speckles on a coarse grid); 'max' maps where transitions "
                         "live without Rabi-phase aliasing")
    ap.add_argument("--init", default=None,
                    help="initial Fock occupations, comma-separated per mode (e.g. "
                         "'0,1,0'). Default all-zero (|g>). NOTE an exchange "
                         "resonance like the iSWAP is invisible from |g> -- it "
                         "conserves excitation number -- so use e.g. --init 0,1,0 "
                         "and --target-mode 0 to map the a<->b transfer.")
    ap.add_argument("--spec-abs-GHz", type=float, default=None)
    ap.add_argument("--nproc", type=int, default=1,
                    help="worker processes over frequency columns (within one node)")
    ap.add_argument("--shard", type=int, default=None,
                    help="array sharding: this task's index (0-based). Columns are "
                         "taken strided (shard::nshards) so per-column cost, which "
                         "grows with drive frequency, balances across tasks.")
    ap.add_argument("--nshards", type=int, default=None,
                    help="array sharding: total number of shards")
    ap.add_argument("--inspect", default=None,
                    help="print the axes, probe, observable and device metadata "
                         "stored in a .npz, plus where its peak actually is")
    ap.add_argument("--missing", default=None,
                    help="diagnose blank stripes: report which columns (and with "
                         "--nshards, which shard indices) are absent, and print the "
                         "resubmit command")
    ap.add_argument("--merge", default=None,
                    help="merge <PREFIX>_shard*.npz into one grid and plot it")
    ap.add_argument("--save-data", default=None, help="write the grid to this .npz")
    ap.add_argument("--save-csv", default=None, help="also write long-form CSV")
    ap.add_argument("--out", default="figs/ge_map.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        inspect_scan(args.inspect)
        return
    if args.missing:
        report_missing(args.missing, args.nshards)
        return
    if args.merge:
        result = merge_shards(args.merge, out_npz=args.save_data)
    elif args.replot:
        result = load_scan(args.replot)
        print(f"loaded {args.replot}: {result['Z'].shape} grid, "
              f"meta = {result['meta']}")
    else:
        if not args.device:
            ap.error("--device is required unless --replot is given")
        try:
            import qutip  # noqa: F401
        except ModuleNotFoundError:
            raise SystemExit(
                "spectroscopy scans integrate the exact Hamiltonian with QuTiP "
                "sesolve (via ZhouCoupler), so qutip must be installed:\n"
                "    uv pip install qutip\n"
                "Plot-only modes (--replot / --merge) do not need it.")
        from snail_solver.paths import resolve_device
        from snail_solver.device_utils import load_device
        cfg = load_device(resolve_device(args.device))
        if float(cfg.get("g4_GHz", 0.0)) == 0.0:
            print("NOTE: g4_GHz = 0, so the expansion is cubic: the ge line and the "
                  "ge/2 subharmonic are reachable, ge/3 is NOT (it needs g4).")
        columns = None
        if args.shard is not None:
            if not args.nshards:
                ap.error("--shard requires --nshards")
            columns = list(range(args.shard, args.f_points, args.nshards))
            if not columns:
                print(f"shard {args.shard} has no columns of {args.f_points} "
                      f"across {args.nshards} shards -- nothing to do")
                return
            print(f"shard {args.shard}/{args.nshards}: {len(columns)} column(s)")
        result = scan_ge(cfg, f_lo=args.f_lo, f_hi=args.f_hi, f_points=args.f_points,
                         amp_lo=args.amp_lo, amp_hi=args.amp_hi,
                         amp_points=args.amp_points, probe_ns=args.probe_ns,
                         target_mode=args.target_mode, level=args.level,
                         envelope=args.envelope, reduce=args.reduce,
                         n_times=args.n_times, rtol=args.rtol, atol=args.atol,
                         nsteps=args.nsteps,
                         init_occ=([int(v) for v in args.init.split(',') if v.strip()]
                                   if args.init else None),
                         spec_abs_GHz=args.spec_abs_GHz,
                         nproc=args.nproc, columns=columns)
        path = args.save_data or "scan"
        prefix = path[:-4] if path.endswith(".npz") else path
        if args.save_data:
            if args.shard is not None:                # one file per array task
                path = f"{prefix}_shard{args.shard:03d}.npz"
            save_scan(result, path)
        if args.save_csv:
            export_csv(result, args.save_csv)
        if args.shard is not None:
            print(f"shard complete; merge with:  --merge {prefix}")
            return                                    # a partial grid is not plottable

    plot_spectroscopy(result, out=args.out, title=args.title,
                      annotate=not args.no_annotate, cmap=args.cmap)
    print("  expected features:", ", ".join(
        f"{k}={v:.3f}GHz" for k, v in sorted(result["lines"].items(),
                                             key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()