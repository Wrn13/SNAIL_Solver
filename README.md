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

**Calibration and optimal control**

| Module | Purpose |
| --- | --- |
| [tune_up](src/snail_solver/tune_up.py) | Hardware-order tune-up: Rabi → chirp → **fix the amplitude** → fit the length |
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
| `snail_stark.slurm` / `snail_stark_detuning.slurm` | Stark resonance and its detuning dependence |
| `snail_grape.slurm` | Single-point optimal control |
| `snail_spectroscopy.slurm` | Sharded spectroscopy map |
| `snail_eta_scan.slurm` | DRAG gain vs. pump strength |

## Tests

```bash
uv run pytest -q
```

51 tests, deliberately QuTiP-free and ~10 s, so they can gate a cluster
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
- At realistic pump strengths (`|eta| ~ 1–1.5`) the open-loop normalization is
  not accurate: the pulse over-rotates and the AC-Stark shift moves the
  resonance, and the two couple because the shift grows as `|eta|^2`. Calibrate
  **frequency first**, then amplitude, and iterate — that is what
  `calibrate_gate` does.
