# SNAIL Solver

Simulation, calibration and frequency-allocation tooling for a **charge-pumped
SNAIL parametric coupler**, built in the dressed-mode framework of Chao Zhou,
*Quantum Operations with Charge-pumped Parametric Interactions*, PhD thesis,
University of Pittsburgh (2023), Ch. 2.

The Hamiltonian is assembled directly from experimental inputs — the coupler
non-linearity `g3` (plus optional `g4`) and the measured participations
`lambda_is = g_is / Delta_is` — rather than from an adiabatically eliminated
effective model. The SNAIL is an explicit mode, not an effective edge.

The question the tooling exists to answer: **where do you place qubit and
spectator frequencies so a pumped iSWAP stays high-fidelity, and how much does
DRAG buy you near a collision?**

## Install

```bash
uv sync                     # or: uv pip install -e ".[test]"
```

Everything is importable as the `snail_solver` package and every tool runs as a
module:

```bash
uv run python -m snail_solver.run_sweep_zhou --help
```

Bare device names resolve under `devices/` and bare output names land under
`results/`, from any working directory — see
[paths.py](src/snail_solver/paths.py). Set `SNAIL_SOLVER_ROOT` to relocate that
root (useful for putting batch output on scratch storage).

## Layout

```
devices/                 device JSONs (the tracked inputs)
results/                 sweep + calibration output (gitignored)
slurm/                   SLURM batch drivers
src/snail_solver/        the package
tests/                   QuTiP-free regression suite
```

## Quick start

A 1-D spectator sweep, analytic only (no QuTiP, runs in seconds) — this is the
DRAG-vs-detuning figure:

```bash
uv run python -m snail_solver.run_sweep_zhou prepare \
    --sweep spectator --device 2Gate4.9SNAIL.json \
    --drags false,true --specfreqs=-1.60,-1.65,-1.70,-1.75,-1.80 \
    --no-integrate --outdir demo
uv run python -m snail_solver.run_sweep_zhou local --outdir demo --nproc 4
uv run python -m snail_solver.plot_results --outdir results/demo
```

Drop `--no-integrate` to evolve the exact Hamiltonian through QuTiP instead of
reporting only the analytic collision map.

> The frequency list is device-specific. For `2Gate4.9SNAIL` (`w_a = 3.8`,
> `w_b = 5.5`) the pump is `w_p = 1.7`, so the one-pump collision sits at
> `|Delta| = 1.7`. The negative (above-`w_b`) branch is used because on the
> `+w_p` branch `w_spec` lands exactly on `w_a`, turning an isolated
> pump-assisted collision into a direct one. Note `--specfreqs=`: argparse reads
> a leading `-` as a flag without the `=`.

## The sweep CLI

`run_sweep_zhou` is the batch driver, with a prepare/run/collect split so a grid
can be spread over a SLURM array.

| Mode | Does | Needs QuTiP |
| --- | --- | --- |
| `prepare` | Write `grid.json`; all physics and grid choices are baked in here | no |
| `point` | Evaluate one `--index` (one array task) | if integrating |
| `local` | Evaluate the whole grid in a process pool on one node | if integrating |
| `collect` | Gather points into `summary.csv` + `combined.npz` | no |
| `missing` | List gaps (a point file is written only on success) | no |
| `chevrons` | Gather the per-point AC-Stark chevrons | no |

Two sweep kinds:

- `--sweep spectator` — fix the target pair, walk one spectator in frequency.
- `--sweep target` — fix `w_a` and the SNAIL, scan a 2-D grid over the partner
  frequency `w_b` and the spectator's absolute frequency. `--no-spectator`
  reduces this to the bare 3-mode gate, where `2 w_p` scans the SNAIL
  subharmonic.

Because `prepare` bakes its choices into `grid.json`, flags divide into
prepare-time and run-time; `slurm/snail_sweep.slurm` documents which is which.

## Tools

**Physics core**

| Module | Purpose |
| --- | --- |
| [zhou_coupler](src/snail_solver/zhou_coupler.py) | The coupler Hamiltonian and propagators |
| [envelope](src/snail_solver/envelope.py) | Pulse envelopes, chirps, `PumpTone` (single source of truth) |
| [jax_engine](src/snail_solver/jax_engine.py) | Batched JAX/diffrax propagator behind `--engine jax` |
| [device_utils](src/snail_solver/device_utils.py) | Device JSON I/O, coupler construction, 1-D maximizer |
| [operating_points](src/snail_solver/operating_points.py) | Named calibrated operating points inside a device JSON |
| [paths](src/snail_solver/paths.py) | `devices/` and `results/` resolution |
| [h5_io](src/snail_solver/h5_io.py) | Run files: nested results ↔ HDF5, with the run's figures stored inside them |

**Calibration and optimal control**

| Module | Purpose |
| --- | --- |
| [tune_up](src/snail_solver/tune_up.py) | Hardware-order tune-up: Rabi → chirp → **fix the amplitude** → fit the length |
| [tune_up_sweep](src/snail_solver/tune_up_sweep.py) | Re-runs the whole tune-up at each `target_eta` and scores the gate: the speed/leakage trade-off curve |
| [stark_chirp](src/snail_solver/stark_chirp.py) | Legendre chirp that tracks the AC-Stark shift through the pulse |
| [calibrate_gate](src/snail_solver/calibrate_gate.py) | End-to-end tune-up mirroring the experiment (frequency chevron, then Rabi) |
| [find_stark_resonance](src/snail_solver/find_stark_resonance.py) | Locate the AC-Stark-shifted iSWAP resonance |
| [calibration_map](src/snail_solver/calibration_map.py) | 2-D (pump offset × pump strength) landscape; can save an operating point |
| [grape](src/snail_solver/grape.py) | Optimal control on the pump envelope vs. the DRAG baseline |

`tune_up` and `calibration_map` answer the same question in opposite orders.
`calibration_map` scans (offset, amp_scale) at a fixed length; `tune_up` *fixes* the
peak drive |η| instead, which makes the pulse shape in normalized gate time
independent of `t_g` and so decouples the frequency calibration from the length
calibration — leaving the length as the only free parameter, as in the lab.

A run's outputs — the operating point, every chevron's traces, the figures drawn
from them, and a copy of the device configuration the solves actually ran with —
go into ONE HDF5 file (`--out run.h5`), so a run is a single self-describing
artefact: it cannot be separated from its own pictures, and it does not have to be
re-interpreted against a device JSON that has been edited since (which is why
`post_chirp --from-tuneup` no longer needs `--device`):

```
python -m snail_solver.h5_io results/run.h5                  # what is in there
python -m snail_solver.h5_io results/run.h5 --extract figs/  # the PNGs back out
```

A sweep keeps every η's complete tune-up in the same file, each addressable on its
own as `eta_sweep.h5:/runs/eta1p8` wherever a path is taken (`tune_up --replot`,
`post_chirp --from-tuneup`). `--out name.json` still writes the old text format.

`tune_up_sweep` sits one level above both: it re-runs the entire tune-up at every
`target_eta` — new Rabi sweep, new chirp, new offset, new length — and scores the
resulting gate, because the chirp and the carrier offset are themselves functions
of the drive, so a range of drives cannot be scored against one calibration.

**The Rabi ladder can use the gate's own envelope** (`--probe-shape gate`). The
default ladder holds the amplitude CONSTANT, which measures the Stark law
`δ = k2|η|² + k4|η|⁴` pointwise — but a constant probe at `|η| = 1.2` leaks 0.245
(0.93 transiently), so the two-level chevron its centre fit assumes is gone, while the
shaped pulse at the same peak leaks 4e-3. A shaped rung instead reports an average over
its own envelope, and because the law is an even polynomial that average is *diagonal*
in `{η², η⁴}` — so recovering the pointwise law is two divisions, `k2 = K2/M2`,
`k4 = K4/M4`, and nothing downstream changes.

Which average is **derived, not chosen**: a chevron's centre weights the shift by
`sin θ(t)`, the accumulated Rabi angle, which vanishes at both pulse ends (a `z`
rotation does nothing at a pole of the Bloch sphere) and peaks mid-gate. That is
`--moment-weighting rabi`, the default; `coupling` and `uniform` are the earlier guesses
and both underestimate `k2`, by 19% and 2.1×. Re-pin it against ground truth whenever an
envelope changes — `tune_up --cross-check-moments` runs a clean constant leg at weak
drive and reports which convention reproduces it:

```
uv run python -m snail_solver.tune_up --device 4Gate4.5SNAIL.json \
    --target-eta 0.45 --cross-check-moments --amp-points 7 --jobs 24
```

The derivation, the two-level confirmation, and why the same ladder cannot pin `M4` are
in [docs/chirped-recursive-drag.md](docs/chirped-recursive-drag.md) (Addendum 5).

**Analysis and figures**

| Module | Purpose |
| --- | --- |
| [plot_results](src/snail_solver/plot_results.py) | Spectator-sweep figures (infidelity, leakage, heatmap vs. Delta) |
| [plot_bare_sweep](src/snail_solver/plot_bare_sweep.py) | 1-D bare-gate sweep (infidelity + coupler occupation) |
| [plot_fidelity_map](src/snail_solver/plot_fidelity_map.py) | 2-D allocation heatmaps, DRAG on vs. off |
| [plot_allocation](src/snail_solver/plot_allocation.py) | Frequency-allocation diagram: *where* the collisions are |
| [spectator_audit](src/snail_solver/spectator_audit.py) | Channel audit: *how strong* each parasitic process is |
| [spectroscopy](src/snail_solver/spectroscopy.py) | Power-vs-frequency spectroscopy map |
| [stark_vs_detuning](src/snail_solver/stark_vs_detuning.py) | How the Stark-shifted resonance moves as a spectator is walked |
| [subharmonic_convergence](src/snail_solver/subharmonic_convergence.py) | Coupler levels vs distance from the SNAIL subharmonic: *how far must the gate be detuned before a truncated model is trustworthy* |
| [subharmonic_gate_scan](src/snail_solver/subharmonic_gate_scan.py) | Gate quality vs pump offset from a qubit's OWN subharmonic (`w_p = w_a/2 + delta`) and vs drive strength: *what does the calibration do as `2 w_p` lands on a qubit, and how hard can the pump be driven* |
| [open_system](src/snail_solver/open_system.py) | Leakage-aware iSWAP fidelity with T1/T2 (`mesolve` + collapse operators): *what the gate achieves*, for confirming a winner the closed-system scan picked out |
| [calibration_plots](src/snail_solver/calibration_plots.py) | The chevron figure that shows the Stark shift explicitly |
| [validate_engines](src/snail_solver/validate_engines.py) | Measure the JAX engine against the QuTiP reference |

Every one takes `--help`.

## Batch (SLURM)

Submit from the repo root, so `devices/`, `results/` and `slurm/` resolve.

The whole pipeline — prepare on the login node, then a point array, then collect,
then plot, chained with `--dependency=afterok`:

```bash
./slurm/submit_sweep.sh target alloc_v1 -- \
    --sweep target --device evan_device.json --drags false,true \
    --wb-GHz 3.8,3.9,4.0 --spec-min-GHz 3.1 --spec-max-GHz 6.1 --spec-step-GHz 0.1
```

Or drive the array directly:

```bash
uv run python -m snail_solver.run_sweep_zhou prepare --sweep target ... --outdir alloc
OUTDIR=results/alloc sbatch --array=0-269 slurm/snail_sweep.slurm   # prepare prints this line
uv run python -m snail_solver.run_sweep_zhou missing --outdir results/alloc
uv run python -m snail_solver.run_sweep_zhou collect --outdir results/alloc
```

`RUNNER` names a **module**, not a file: `RUNNER=snail_solver.run_sweep_zhou`.
`CODE_DIR` overrides the working directory and should point at the repo root.
Each script's header documents its own environment knobs; the examples in them
are tuned to the device they name and run as written.

| Script | Purpose |
| --- | --- |
| `snail_sweep.slurm` | The point array (also `MODE=local` to pool on one node) |
| `snail_collect.slurm` / `snail_plot.slurm` | Post-processing stages |
| `submit_sweep.sh` | Chain all of the above |
| `snail_calibrate_gate.slurm` | Per-device gate tune-up |
| `snail_tune_up.slurm` / `submit_tune_up.sh` | Hardware-order tune-up (fast explore → exact confirm) |
| `snail_calibration_map.slurm` / `snail_gpu_scan.slurm` | Calibration landscape (CPU / GPU) |
| `snail_subharmonic.slurm` | Truncation convergence vs subharmonic detuning (resumable; run `DRY=1` first) |
| `snail_stark.slurm` / `snail_stark_detuning.slurm` | Stark resonance and its detuning dependence |
| `snail_grape.slurm` | Single-point optimal control |
| `snail_spectroscopy.slurm` | Sharded spectroscopy map |
| `snail_eta_scan.slurm` | DRAG gain vs. pump strength |

### Without SLURM (one box)

`subharmonic_gate_scan` has no SLURM script: it runs detached on a single machine,
which is where the GH200 work happens. `scripts/run_wp_scan.sh` forwards its
arguments to the module under `setsid nohup`, pins BLAS to one thread per process
(the tune-up already fans its chevron out over a process pool, so unpinned BLAS
thrashes), and prints a PID and a log path:

```bash
# always first: solves nothing, prints the drive-feasibility table and the
# per-column DRAG channel audit
uv run python -m snail_solver.subharmonic_gate_scan \
    --device 4Gate4.5SNAIL.json --offsets=-0.1:0.1:21 \
    --target-etas 0.6,0.8,1.0,1.2 --dry-run

scripts/run_wp_scan.sh --device 4Gate4.5SNAIL.json --offsets=-0.1:0.1:21 \
    --target-etas 0.6,0.8,1.0,1.2 --amp-points 41 --coupler-levels 7 \
    --t1-us 50 --t2-us 50 --open-system 3 \
    --column-workers 8 --jobs 8 \
    --out wpscan_band.h5 --plot figs/wpscan_band.png
```

`--target-etas` is the drive/speed axis (`t_g = 2A/eta`). `--amp-points` /
`--eta-lo` / `--eta-hi` are a different thing: the Rabi amplitude ladder that
*builds* each column's chirp, taken as a fraction of that column's own
`target_eta` so it never probes above the pulse's own peak.

**The drive axis has a ceiling, and it is not the truncation.** The subharmonic
coupling is a two-pump process, so `g ∝ η²` while its detuning is set by `δ` —
past `η ≈ 1` a ±100 MHz scan goes *non-perturbative* (`g/|det| ≥ 1`) and
recursive DRAG has no leading term to cancel. The dry run prints the feasibility
table; columns carrying such a channel are refused unless `--force`.

**The solve is closed-system, so gate length is otherwise free.** Without
`--t1-us` / `--t2-us` the ranking always prefers the weakest drive and ignores
that its gate is five times longer. Those flags add a first-order incoherent term
(transparent, and adjustable with `--decoh-prefactor`) and rank on the combined
infidelity; `--open-system TOP` then re-scores the best few points with a real
`mesolve`.

**Spend the cores across columns, not only inside them.** `tune_up` step 4
(`length_rabi`) takes no `jobs` — it is an optimizer over gate length and runs
single-threaded, for minutes per column once the pulse carries a multi-channel
recursion, and `run_tune_up` repeats it `2 × n_channels` times. A `--jobs 72` run
leaves ~70 cores idle throughout, so pair a moderate `--jobs` with
`--column-workers`.

Resumable: each column is cached per `(δ, η)` the moment it succeeds, so
relaunching the identical command re-reads what is done and solves only what is
missing — and the eta band can be extended one value at a time. `--offsets=`
needs the `=`, since a bare leading `-` reads as a flag. Don't use `--gpu`: this
Hilbert space is ~10² states, well below the CPU/GPU crossover, and `--gpu`
forces `--jobs 1`.

## Tests

```bash
uv run pytest -q
```

366 tests, deliberately QuTiP-free and ~1 min, so they can gate a cluster
submission. Each one encodes an invariant that a real bug once violated (beat
sign conventions, collision labelling, the subharmonic factor of 2, blank
spectator plumbing) — a failure means a specific known-bad behaviour is back.

## Notes on the physics

- The pump sits at `w_p = |w_b - w_a|`, amplitude-normalized to a full iSWAP at
  `t_g`. With `--target-eta`, `t_g` follows from the gate area
  (`t_g = 2 * eta_area / eta`) — don't also pass `--t-g-ns`.
- DRAG is the first-order Motzoi quadrature (PRL **103**, 110501 (2009)) tuned to
  the spectator's beat detuning. It is *skipped* within 0.5 MHz of a collision:
  on resonance the answer is frequency allocation, not DRAG. It helps in the
  off-resonant-but-close window, roughly the inner 100 MHz.
- `g3 X^3` necessarily contains `3 g3 eta^2 s^dag` at the **subharmonic
  detuning** `Delta_sub = w_s - 2 w_p` — a linear drive on the SNAIL with no
  participation suppression, where the gate itself is down by `lam^2 ~ 0.01`. It
  displaces the coupler by `|alpha| ~ 3 g3 eta^2 / Delta_sub`, and a coherent
  state has weight on every Fock level, so close to the subharmonic the answer is
  set by `coupler_levels` rather than by the physics. `subharmonic_convergence`
  measures how far the gate has to be detuned before a given truncation converges;
  the diagnosis is in [docs/chirped-recursive-drag.md](docs/chirped-recursive-drag.md).
- At realistic pump strengths (`|eta| ~ 1–1.5`) the open-loop normalization is
  not accurate: the pulse over-rotates and the AC-Stark shift moves the
  resonance, and the two couple because the shift grows as `|eta|^2`. Calibrate
  **frequency first**, then amplitude, and iterate — that is what
  `calibrate_gate` does.
