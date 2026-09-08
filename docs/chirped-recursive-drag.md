# Chirped recursive DRAG

## Progress (as of 2026-08-31)

**Suite: 190 passed, 97 subtests — all 125 pre-existing tests still pass unmodified.**
Baseline before any change was 125 passed / 29 subtests.

**Phases 1–7 complete. Phase 8 (Givens) not started.** Phase 7's harness is built and
runs, but it has not yet produced a physics verdict — see "Phase 7" below, which is
the single most important open item.

### Performance — resolved

The scalar-callback slowdown recorded earlier is fixed, and the fix was much larger
than the jet micro-optimizations:

| | before | after |
|---|---|---|
| `_eta` per call, recursive x3 | 359 us | **224 us** (shared chirp jet) |
| 12 ns toy gate, no DRAG | 54 s | **8.0 s** |
| 12 ns toy gate, first-order | 124 s | **3.1 s** (40x) |
| 12 ns toy gate, recursive x3 | never finished (>9 min) | **7.7 s** |

The dominant cost was never the jet algebra. `to_qutip_hamiltonian` builds one
coefficient closure per Hamiltonian term, and each was independently calling `_eta`
at the *same* `t` — **12 identical evaluations per timestep** on a bare 3-mode pair.
`zhou_coupler.to_qutip_hamiltonian` now shares one evaluation per `(tone, t)` via a
one-slot-per-tone cache scoped to the QobjEvo, so it cannot go stale across the
in-place tone mutation `grape` does between solves. Fidelities are unchanged
bit-for-bit for every case that previously completed. This speeds up **every** path,
DRAG or not.

### Open items — both resolved

1. **Eq. 13 transcription — CONFIRMED as a slip in the paper, decisively.** Tested both
   readings against the paper's own m=1 claim:

   | reading | `Ω'(t_r)` (should be 0) | `max abs(ramp − Hann)` |
   |---|---|---|
   | as printed, `sin^m(pi t'/2 t_r)` | **1.000** | 2.5e-01 |
   | corrected, `sin^m(pi t'/t_r)` | **0.000** | **5.6e-16** |

   The printed form has the integrand at its *maximum* where the derivative is supposed
   to vanish. The corrected form reproduces the Hann window to 5.6e-16, exactly as the
   paper states. `SinePowerRamp` implements the corrected reading and documents why.
2. **`--max-drag-iters`** — resolved in code rather than left to measurement:
   `n_outer = max(max_drag_iters, 2 * n_channels)`, since the d-th nested correction
   scales as `1/t_g^d`.

### Done

**Phase 1 — derivative primitives.** Complete.
- `src/snail_solver/jet.py` (new). Truncated Taylor-coefficient arithmetic:
  `add/sub/neg/scale/mul/div/deriv/powi/powf/constant/from_derivs/derivs/truncate`.
  `powf(1.0)` short-circuits to the identity object (`assertIs`-checked), which is what
  keeps every single-photon caller free of the safe-divide guard. `_safe_div` uses the
  double-`where`; there is a test that `jax.grad` through a zero denominator is finite.
- `Envelope.jet_at(t, order, xp)` on the base class (orders 0/1 delegate to
  `value_at`/`deriv_at`; higher raises naming the class), with closed-form overrides on
  `ConstantPulse`, `RaisedCosine` (`-amp/2 w^n cos(wt + n pi/2)`) and
  `IQFourierEnvelope` (Leibniz over `_shape_derivs` x `_mod_derivs`).
- `Chirp.detuning_jet(t, order, xp)` + `_legendre_deriv_stack`, using the value
  recurrence differentiated in place. Order 0 is *exactly* `detuning()`; orders >= 1
  carry an explicit support mask rather than leaning on `xp.clip`'s subgradient.

**Phase 2 — the recursion.** Complete.
- `src/snail_solver/drag.py` (new): `apply_drag`, `order_channels`, `required_order`.
  The sort is enforced inside `apply_drag` (zipped with the detuning jets, so a caller
  that mis-orders is corrected, not mis-paired). `mode="givens"` raises.
- `DragChannel` (frozen dataclass) and `PumpTone.drag_channels` /
  `drag_channels_resolved()` / `is_legacy_drag` / `channel_detuning` /
  `channel_detuning_jet` in `envelope.py`. `drag_detuning_floor` now minimises over all
  channels; `drag_detuning_floors()` added for the per-channel breakdown.

**Phase 3 — wiring.** Mirrors 1 and 2 done; see Remaining for 3 and 4.
- `zhou_coupler._eta_at`: `is_legacy_drag` fast path (bit-identical) + general
  `apply_drag` branch.
- `jax_engine`: `_hann_derivs`, `_iq_mod_derivs`, `_shape_jet_at` with a genuine
  `_SHAPE_JETS` kind table (unknown kind raises rather than silently defaulting to
  Hann), `_chirp_detuning_jet`, `_channel_detuning_jet`, `pulse_spec` gains static
  `drag_channels` + `legacy_drag`, `eta_at` gains the general branch.
  **`_shape_at` was deliberately left untouched** so the first-order path stays
  bit-identical; the jet forms are used only on the recursive path.

**Phase 3, mirrors 3 and 4 + guards.** Complete.
- `grape`: `_tone_drag_channels` (third companion to `_tone_chirp`/`_tone_n_pump`),
  `_raised_cosine_eta(..., channels=None)` routed through `apply_drag` when channels
  are present and byte-identical otherwise, threaded into the three BASELINE call
  sites (`:462`, `:765`, `:1207`) but deliberately NOT the warmstart seed.
  `_optimize_crab` now clears and restores `tone.drag_channels` — `drag=False` alone
  does not disable an explicit list, so the recursion would otherwise have fired on
  top of the CRAB ansatz.
- `device_utils.check_drag_detuning` names the offending channel (keeps the
  `"min|Delta(t)|"` substring the old test pins); `drag_correction_ratio` added;
  `build_coupler(drag_channels=..., correction_warn=, logger=)` warns (never raises).
- `sweep_common._drag_channels_filtered` drops only the offending channel;
  `_drag_ok_with_chirp` keeps its exact signature and delegates.

**Phase 5 — `SinePowerRamp`.** Complete.
- The Eq. (13) class, implemented via the finite complex-exponential expansion of
  `sin^m` (one code path for odd and even m; antiderivative and all derivatives are
  one-liners). Closed-form `area()`, matching quadrature to 2e-16.
- `ENVELOPE_KINDS` registry; `jax_engine._SHAPE_JETS` entry (delegates to the class,
  which is safe *because* the shape has no free params, so nothing needs to come from
  `params` to stay a grad axis); `build_coupler` routing.
- `tune_up.area_factor` replaces the hardcoded Hann 1/2 in `fixed_eta_amp_scale` /
  `peak_eta_of`. Returns exactly 0.5 for a raised cosine, and both rewrites are
  bit-identical there (the rescalings are powers of two).
- The config key is `envelope_rise_frac`, a FRACTION of t_g, not a rise time in ns.
  An absolute rise would make the area factor — and hence the Stark shift and the
  chirp — t_g-dependent, destroying the length/frequency decoupling the module rests
  on. Caught while writing `area_factor`.
- Corrected a real bug found by the m=1/Hann test: the region masks left `t == t_r`
  falling through to the plateau branch, zeroing a derivative that is genuinely
  nonzero when there is no plateau.

**Phase 6 — `tune_up` calibration.** Complete except `stark_chirp` (see Remaining).
- `chirp_from_measured_shift` now builds the pulse with the same `apply_drag` the
  solver uses, and takes `|.|` of it rather than `sqrt(amp^2 + q^2)` — which is the
  same number only while the correction is purely imaginary, i.e. only at one channel.
- The fixed point iterates on Legendre COEFFICIENTS (the `Delta` jet needs an analytic
  `d/dt` of the current iterate). Internally projected at degree >= 24, not at the
  output `degree`: `delta(u)` comes from `cos^4`/`cos^8`, whose Legendre series does
  not terminate, and at degree 8 the leftover ripple was enough to flip the sign of
  `Delta - Delta_0` on a 50 MHz beat by ~6 kHz — misreporting the singularity guard.
  Found by an existing test, `test_k_scales_the_detuning_pull`.
- `_resolve_drag_channels` and `_project` factored out; `drag_channels` threaded
  through `length_rabi`, `calibrate_drag_offset`, `drag_shift_table`, `run_tune_up`
  (9 call sites) and `find_stark_resonance.build_chevron_coupler` / `scan` /
  `build_kw` (picklable, so it survives the multiprocessing fan-out).
- `run_tune_up`: `_drag_on` now covers channels, and
  `n_outer = max(max_drag_iters, 2 * n_channels)`.
- CLI: repeatable `--drag-channel BEAT[:K[:N]]` via `parse_drag_channels`, mutually
  exclusive with `--drag-beat-GHz`. (Negative beats need `=` syntax:
  `--drag-channel=-0.22:1`, since argparse reads a leading `-` as a flag.)
- `peak_eta_of`'s docstring corrected: its "unaffected by DRAG" claim holds only at
  first order, since two nested corrections contribute `-eta''/(Da Db)` and `eta''`
  at the peak is not zero.

**Phase 4 — tests.** New classes in `tests/test_physics.py`: `TestJetArithmetic`,
`TestEnvelopeJets`, `TestJetImplementationsAgree`, `TestRecursiveDrag`,
`TestRecursiveDragPlumbing`, `TestSinePowerRamp`, `TestTuneUpRecursiveDrag`.

### Measured — every quantitative prediction in this plan held

| Claim | Predicted | Measured |
|---|---|---|
| Order-1 general path == legacy expression | exact | **5.9e-17** (chirped and unchirped) |
| Quotient-rule term at 300 MHz beat | 1–3% | **3.2%** of the correction |
| Quotient-rule term at 20 MHz beat | O(1) | **72%** |
| Hann + full 3-channel recursion at the edge | `dt^(-1/2)` | ratios 1.331 → 1.391 → **1.408** (→ √2) |
| Envelope/chirp `jet_at` vs finite differences | O(h²) | h-ratios **4.00** at every order |
| `SinePowerRamp(m=1)` vs `RaisedCosine` | identical | **2.4e-17** at every derivative order; area equal |
| `SinePowerRamp(m=3)` under the same 3-channel recursion | well-posed | edge \|eta\| **decays** 2.6e-5 → 1.1e-6 (vs Hann's 9.2e-3 → 2.4e-2) |
| Rewritten `chirp_from_measured_shift` at one channel | pure refactor | **4.6e-16 relative** vs the old code |

The Hann divergence is now a regression test
(`TestRecursiveDrag.test_hann_diverges_under_the_full_recursion`), so **Phase 5 is
confirmed as a hard prerequisite, not a refinement** — as argued below.

Derivatives are verified by *h-refinement*, not by a fixed tolerance: `_fd_order`
asserts the residual falls as O(h²) across three step sizes. A wrong closed form leaves
a constant offset and the ratio collapses to 1, which no tolerance-based check catches.

**Shape-aware chirp seed.** Done, and it mattered:
`stark_chirp.shape_stark_legendre(shape_fn, degree)` + `shape_mean_factor` generalize
the Hann-only table (they reproduce `HANN_STARK_LEGENDRE` to **1.3e-14**, which is now
the regression fixture). `chirp_from_measured_shift` gained `shape` / `shape_kw` and
reads `|eta(u)|` off the actual envelope instead of the hardcoded `cos^2(pi u/2)`;
`tune_up.shape_config(config)` feeds it from the device and `run_tune_up` passes it
through. Measured: with `sine_power(m=3)` the chirp coefficients differ from the Hann
assumption by **17.5%**, so this was a real inconsistency, not a tidy-up.

**Phase 7 — the physics check.** Harness complete, verdict still open.
`src/snail_solver/validate_recursive_drag.py`: sweeps the spectator frequency (the
axis that controls how *crowded* the spectrum is, which is what the paper's claim is
about) and scores {none, F1, F1∘F1, F1∘F1∘F2} at each placement, auto-filling channels
from `sweep_common.collision_drag_channels`. Defaults to `sine_power(m=3)`, since on a
Hann the deepest scheme would measure the `t^(-1/2)` divergence rather than physics.

`sweep_common` gained `_collision_candidates` (factored out of `_nearest_collision`,
which is now `min` over it) and `collision_drag_channels`, giving the "explicit list,
auto as a default" behaviour chosen at planning time.

**It does not yet yield a verdict, because no available config has a working gate to
measure on.** Measured, DRAG entirely absent and no spectator:

| config | reference gate |
|---|---|
| `evan_device` at its own recorded operating point (t_g 77.2 ns, peak η≈1.8) | **F = 0.063, leak = 0.879** |
| `DEFAULT_CONFIG`, η*=1.0 | peak transfer 0.24 |
| `DEFAULT_CONFIG`, η*=0.5 | peak transfer 0.74 |

At η≈1.8 these devices dump ~88% of the population into the coupler *before* any
spectator exists — the same regime that made the tune-up drop Rabi rows 3–4 for
`coupler` leakage. A leakage-suppression scheme cannot be seen underneath that.

So the sweep now solves a **reference gate** (no spectator, no DRAG) first and
**refuses to report a verdict** when it is already bad. This is not defensive
decoration: on `evan_device` the scheme ranking came out `F1oF1oF2` 0.890 < `F1` 0.919,
which reads exactly like "recursion wins" and is pure noise on a gate at dF = 0.95.
Without the guard that number would have been reported as a result.

**To finish Phase 7:** find an operating point where the reference gate is good
(dF < ~0.1) — a weaker drive, or the output of a completed `tune_up` run — and re-run.
The harness, the guard and the channel auto-fill are all in place.

### Performance finding (superseded — kept for the reasoning)

Measured cost per `ZhouCoupler._eta` call (3-mode system, chirped):

| | before the fix below | after |
|---|---|---|
| no DRAG | 19.8 us | 21.1 us |
| first-order (unchanged path) | 44.1 us | 42.6 us |
| **recursive, 3 channels** | **359 us** | **224 us** |

QuTiP calls this once per Hamiltonian term per substep, so a full
`propagator_columns` scales with it: on a 12 ns toy gate, no-DRAG took 54 s and
first-order 124 s, and the 3-channel case did not finish inside a 9-minute budget.
**Unit-test correctness is thorough (184 tests), but no full QuTiP solve of a
3-channel pulse has actually been completed yet.**

A profile showed the dominant cost was duplicated work, not the jet algebra: every
channel rebuilt the same chirp's Legendre recurrence, and `detuning_jet` recomputed
order 0 via `detuning()` a second time — 15000 recurrence evaluations per 1500 `_eta`
calls. Fixed both (`PumpTone.channel_detuning_jets` shares one chirp evaluation across
channels; `detuning_jet` reuses its own stack), giving the 1.6x above.

Remaining options if it is still too slow, roughly in order of value:
1. Give the QuTiP path an ARRAY-valued coefficient (`_eta_at` is already vectorized
   and array evaluation amortizes nearly all the Python overhead) instead of the
   scalar `_eta` callback.
2. `Jet.div` guards every division with the double-`where`, but the DRAG denominators
   are guaranteed far from zero by `check_drag_detuning`; only `powf`'s divide by the
   vanishing envelope actually needs it. A `safe=False` opt-out on `div` would remove
   ~16500 `xp.where` calls per 1500 evaluations.
3. Use the `jax_engine` path for recursive tones, where everything traces and compiles.

---

## Why the fidelities are poor — investigated 2026-08-31

Two separate questions came up; both now have measured answers, and the second one
matters more than the recursion work itself.

### 1. Would DRAG on the leakage channel help? **No.**

The dominant error is leakage into the COUPLER, and none of the sweeps were targeting
it: `collision_drag_channels` only enumerates the coupler when `drag_subharmonic` is
set. On `evan_device` that channel is real and well-defined —
`w_p = 2.198`, `2 w_p = 4.396` vs `w_s = 4.7`, a **+304 MHz two-pump beat**. Adding it
explicitly:

| η* | scheme | F | leak | corr_ratio |
|---|---|---|---|---|
| 1.0 | none | 0.87215 | 0.08153 | — |
| 1.0 | coupler `F^(2)` | 0.87037 | 0.08137 | 0.012 |
| 1.4 | none | 0.87152 | 0.09761 | — |
| 1.4 | coupler `F^(2)` | 0.86671 | 0.10061 | 0.017 |
| 1.8 | none | 0.07122 | 0.87118 | — |
| 1.8 | coupler `F^(2)` | 0.07839 | 0.86617 | 0.021 |

Nothing moves — under 0.3% relative. `corr_ratio` says why: the correction is only
**1–2%** of the pulse, because `eta_dot/Delta ~ 1/(t_g Delta)` and with `t_g` ~ 100 ns
against a 304 MHz beat the pulse is already deeply adiabatic. There is essentially no
diabatic excitation for a derivative correction to cancel.

### 2. So what IS the leakage? Drive strength, not turn-on speed.

At **fixed** `t_g = 99.2 ns` (adiabaticity held constant), scanning `amp_scale`:

| peak η | F | leak |
|---|---|---|
| 0.84 | 0.910 | 0.003 |
| 1.12 | **0.976** | 0.014 |
| 1.40 | 0.870 | 0.099 |
| 1.61 | 0.635 | 0.344 |
| 1.82 | 0.068 | 0.893 |

Leakage is a function of drive AMPLITUDE at constant gate length. Diabatic error would
be flat here. **This is the decisive result: DRAG — recursive or not — is structurally
the wrong tool for this device's dominant error.** The recursion is correct and will
help where the error IS diabatic (short gates, near collisions); it cannot help here.

The device JSON's operating point (`t_g_ns: 77.2`, peak η ≈ 1.8) sits well past the
cliff. This is also exactly what `tune_up.verify_eta_matches_ns` was written to worry
about, and why the tune-up dropped Rabi rows 3–4 for `coupler` leakage.

### 3. ⚠ The model does not converge in `coupler_levels` — at ANY drive tested

This is the finding that blocks everything downstream, and it is **pre-existing**, not
caused by this work (`coupler_levels` feeds `ZhouCoupler` construction, upstream of
every line changed here; the `levels=5` number reproduces bit-for-bit with the new
eta cache stashed out).

| levels | F at η=1.12 | F at η=1.82 |
|---|---|---|
| 5 | 0.97601 | 0.06766 |
| 7 | 0.94322 | 0.26388 |
| 9 | 0.74385 | 0.08333 |
| 12 | 0.62532 | 0.08538 |

At the *good* operating point F falls monotonically 0.976 → 0.625 as the coupler
Hilbert space grows. That is not convergence — the truncation is acting as the
regularizer.

### 4. ✅ RESOLVED — it is the SNAIL subharmonic at 2·w_p, not an unbounded cubic

My "unbounded cubic" guess above was **wrong**. The cause is a specific, identifiable
near-resonance, and the model is fine once it is detuned.

`evan_device` has `w_a = 3.5`, `w_b = 5.7`, `w_s = 4.7`, so the gate pump sits at
`w_p = |w_b − w_a| = 2.198` and its **second harmonic `2 w_p = 4.396` lands 304 MHz
below the SNAIL**. Since `H = g3 X³` with `X ∋ s e^{-i w_s t} + eta(t) e^{-i w_p t} + h.c.`,
the expansion necessarily contains

    3 g3 eta(t)^2 s^dag e^{i(w_s − 2 w_p) t}

— a **linear drive on the SNAIL**, detuned by only 304 MHz. Enumerating every term of
`expand_terms()` at peak eta = 1.12 and sorting by detuning:

| detuning | pump quanta | Omega (rad/ns) | \|Omega/Delta\| | process |
|---|---|---|---|---|
| 0.002 GHz | +1 | 0.051 | — | the intended iSWAP |
| **0.304 GHz** | **+2** | **4.013** | **2.10** | **2 w_p → SNAIL** |
| 0.998 GHz | +1 | 1.013 | 0.16 | a ↔ SNAIL conversion |
| 1.198 GHz | +1 | 1.013 | 0.14 | b ↔ SNAIL conversion |

The spurious drive's 0→1 matrix element is **1.42 rad/ns against a gate of
0.025 rad/ns — 56× stronger than the gate itself.** That is structural, not a tuning
accident: the gate is a two-participation process (`6 g3 lam_a lam_b eta`, suppressed
by `lam² = 0.01`) while the subharmonic drives the coupler directly at `lam = 1`. Only
the 304 MHz detuning holds it off, and it does not hold it off nearly well enough.

It produces a **coherent displacement** `|alpha| ≈ 3 g3 eta²/Delta`, which is why all
three earlier observations fit at once:
- **Leakage tracks drive amplitude, not gate length** (finding 2) — `alpha ∝ eta²`.
- **DRAG does not help** (finding 1) — the displacement is adiabatic (`corr_ratio` 1–2%),
  so there is no diabatic transition for a derivative correction to cancel.
- **No truncation convergence** (finding 3) — a coherent state of `|alpha| ~ 0.75`
  has support on every Fock level, and the coupler ladder is harmonic (a SNAIL is
  defined by `g3`/`g4` alone, so there is no level-dependent detuning to stop the climb
  while the matrix element grows as √n). Truncation *is* the regularizer.

**Confirmed causally** by sweeping `w_s` at fixed eta = 1.12, t_g = 99.2 ns:

| w_s | Delta_sub | \|alpha\| | F(5 lvl) | F(9 lvl) | leak(9) | **spread** |
|---|---|---|---|---|---|---|
| 4.70 | 0.304 | 0.743 | 0.97607 | 0.74379 | 0.23557 | **0.23227** |
| 5.05 | 0.654 | 0.345 | 0.91969 | 0.96519 | 0.00769 | **0.04550** |
| 6.50 | 2.104 | 0.107 | 0.88325 | 0.88153 | 0.00160 | **0.00173** |
| 7.50 | 3.104 | 0.073 | 0.80713 | 0.80736 | 0.00060 | **0.00023** |

The truncation spread collapses by **three orders of magnitude** as `|alpha|` falls, and
leakage with it. **The model converges perfectly well; the device placement does not.**
Nothing is wrong with the truncated cubic, `propagator_columns`, or the solver.

(The falling `F` in the last two rows is unrelated and benign: leakage there is ~1e-3,
so the residual infidelity is *coherent* — the amplitude and length are still calibrated
for `w_s = 4.7`, and the Stark shift moves with `w_s`. Those points need a re-tune, not
a fix.)

**Consequences.**
1. `w_s = 4.7` with `w_p = 2.198` is a **bad frequency placement** — `w_s` sits 304 MHz
   from `2 w_p`. This is a device design issue, not a code defect, and it is entirely
   independent of the DRAG work.
2. The three constraints on `w_s` are clean: keep it away from `2 w_p = 4.396`
   (subharmonic, unsuppressed), from `w_b` (= `w_a + w_p`, the a↔s conversion) and from
   `w_a` (the b↔s conversion, since `w_b − w_p = w_a`). The band between 4.4 and 5.7 is
   narrow; `w_s` well above `w_b` or well below `w_a` is roomier.
3. Fidelities computed at `w_s = 4.7` and 5 coupler levels — **including the
   F = 0.976 "good" operating point** — are truncation artifacts and should not be
   quoted. At a well-placed `w_s`, 5 levels is adequate.

### 5. Where DRAG does and does not act — the `beat x t_g` criterion

DRAG removes only the **spectral** (residual) response — the excitation left at the end,
which is set by the envelope's Fourier content at the beat. It cannot touch the
**forced** (adiabatic) response `alpha(t) ~ Omega(t)/Delta` during the pulse, because
that is the drive's instantaneous response, not a transition.

Measured split at `w_s = 4.7`, `t_g = 99.2 ns`, 9 levels (coupler `<n>` from `|01>`):

| eta | \|alpha\| pred | `<n>` at t_g/2 | `<n>` at t_g |
|---|---|---|---|
| 0.40 | 0.095 | 0.0315 | 0.0073 |
| 1.12 | 0.743 | **2.357** | **0.947** |

At the operating point the coupler holds **2.36 quanta mid-gate**. That is forced
response; no derivative correction addresses it.

**The controlling parameter is `|beat| x t_g`** — beat cycles inside the gate. Scanning
`t_g` at eta = 0.4 (weak enough to stay converged), with and without `F^(2)` on the
+304 MHz channel:

| t_g (ns) | beat·t_g | leak, none | leak, DRAG | ratio |
|---|---|---|---|---|
| 3.0 | 0.91 | 3.76e-02 | 4.24e-03 | **0.113** |
| 6.0 | 1.82 | 3.21e-02 | 3.23e-03 | **0.101** |
| 12.0 | 3.65 | 7.49e-03 | 6.47e-03 | 0.864 |
| 25.0 | 7.60 | 9.76e-04 | 9.66e-04 | 0.989 |
| 50.0 | 15.20 | 1.48e-03 | 1.47e-03 | 0.993 |

**DRAG buys ~10x below ~2 beat cycles and exactly nothing above ~7.** (The rise from
9.8e-4 to 1.5e-3 in the last row is the forced-response floor taking over from the
spectral residual — DRAG tracks the latter and stops mattering once the former wins.)

This is also the **first end-to-end physics validation of the DRAG implementation**: a
10x leakage suppression on a real solve, not a unit test.

**Consequences.**
1. The device operating point is at `0.304 x 99.2 = 30` beat cycles — a factor 30 past
   where DRAG stops doing anything. That is the quantitative reason the subharmonic-DRAG
   test in section 1 moved fidelity by <0.3%.
2. **DRAG's useful window is `drag_skip (5 MHz) <~ |beat| <~ 3/t_g`.** At `t_g = 100 ns`
   that is 5–30 MHz — narrow. At `t_g = 20 ns` it is 5–150 MHz — roomy. So this work
   pays off at **short gates and close collisions**, which is the paper's regime; it
   cannot rescue a long gate with a 304 MHz spurious drive.
3. For the **recursion** specifically to matter, two or more channels must sit inside
   that window simultaneously. At `t_g ~ 100 ns` that needs two collisions under 30 MHz;
   at `t_g ~ 20 ns` it is much more likely. Phase 7 should be run at a short `t_g`.
4. On a deeply-adiabatic channel DRAG is very slightly **harmful**: it adds amplitude,
   and the spurious drive scales as `eta^2`, so the correction feeds the thing it is
   meant to suppress (measured in section 1: F 0.8722 → 0.8704).

`F^(1)` vs `F^(2)` on this two-pump channel is **not resolved** by the data: at
beat·t_g = 0.91 `F^(1)` wins (1.41e-3 vs 4.24e-3), at 1.82 `F^(2)` wins
(3.23e-3 vs 4.73e-3). Both beat no-DRAG by 7–25x. At these beat·t_g the perturbative
expansion is marginal, so this is not a clean test of the exponent; a proper one needs a
channel where `Omega/Delta` is small but `beat*t_g` is still ~1.

### 6. `devices/1Gate4.2SNAIL.json` — same non-convergence, different cause, and a FIX

Checked 2026-09-01 at the user's request. `w_a = 4.7, w_b = 5.7, w_s = 4.2` gives
`w_p = 1.0`, so `2 w_p = 2.0` is **2.2 GHz** from the SNAIL — the subharmonic that ruins
`evan_device` is well placed here. It still does not converge at its own operating point:

| levels | F | leak |
|---|---|---|
| 5 | 0.37821 | 0.43684 |
| 7 | 0.13587 | 0.65122 |
| 9 | 0.07672 | 0.83472 |
| 12 | 0.10288 | 0.79042 |

The cause is **drive strength**, not placement. At `t_g = 77.2 ns` the normalisation puts
peak eta at **1.799**, and the coupler-exciting terms (diagonal Stark terms excluded) are:

| \|det\| GHz | pump quanta | dn_s | Omega rad/ns | \|Omega/det\| |
|---|---|---|---|---|
| 4.2 | 0 | 1 | 21.13 | 0.801 |
| 4.2 | 2 | 1 | 20.71 | 0.785 |
| 2.2 | 2 | 1 | 10.35 | 0.749 |
| 0.5 | 1 | 1 | 1.63 | 0.518 |

The 4.2 GHz / 0-pump entry is the SNAIL's **own cubic self-term** `g3 s^dag s^dag s`,
detuned by `w_s`. It carries no `lam` suppression and its matrix element grows as
`n^1.5`, so its ratio to the fixed `w_s` **grows with the truncation** — which is why
adding levels always makes things worse *once the ladder is populated*. There is also a
diagonal `6 g3 eta s^dag s` term at detuning `w_p = 1.0 GHz` with ratio 4.9: a ~0.65 GHz
frequency modulation of the coupler, which puts sidebands on everything else.

**The fix — a real, converged operating point.** `amp_scale = 1.0` renormalises eta so
the gate always completes, so a longer gate is automatically a weaker drive:

| t_g (ns) | peak eta | F(5 lvl) | F(9 lvl) | leak(9) | **spread** |
|---|---|---|---|---|---|
| 77.2 | 1.799 | 0.37821 | 0.07672 | 0.83472 | 0.30148 |
| 154.4 | 0.900 | 0.85683 | 0.28081 | 0.65067 | 0.57602 |
| **308.8** | **0.450** | **0.96164** | **0.96244** | **0.00492** | **0.00080** |

**`1Gate4.2SNAIL` at `t_g = 308.8 ns` is a working, truncation-converged operating
point: F = 0.962, leak = 0.005, and 5 levels agrees with 9 to 8e-4.** This is the first
trustworthy absolute fidelity in this whole investigation, and it is the point Phase 7
should run at.

**General conclusion across both devices.** The truncated cubic is not broken. Both
devices are simply being driven far too hard: at `eta ~ 1.8` the coupler ladder is
populated, and once it is, the `n^1.5` cubic self-term makes the answer truncation-
controlled. The model's validity domain is roughly `eta <~ 0.5`, which costs gate speed
(4x longer here). `evan_device` additionally has the `2 w_p` placement defect on top of
this, which is why it is the worse of the two.

---

### Remaining

Ordered by what actually blocks a result.

0. ~~Resolve the `coupler_levels` non-convergence.~~ **Done — see section 4 above.**
   It was the `2 w_p` SNAIL subharmonic, and the model converges once `w_s` is placed
   away from it.

1. **A working operating point — needs a device decision first.** `evan_device`'s
   `w_s = 4.7` is 304 MHz from `2 w_p = 4.396`, which costs more fidelity than
   anything DRAG can recover. Either move `w_s` (e.g. ~6.5 GHz, where 5 coupler levels
   converge to 2e-3) and re-run `tune_up` there, or accept `w_s = 4.7` and run at
   `eta` low enough that `|alpha| = 3 g3 eta²/0.304 << 1` — which means eta ≲ 0.4 and a
   correspondingly long gate. **This is a physics question about the device, not a code
   defect** — it is visible with DRAG entirely absent.
2. **Re-run Phase 7** at that point. No code changes expected. Note the reference-gate
   guard will (correctly) refuse a verdict until (0) and (1) are settled.
3. **A completed `tune_up --drag-channel` run.** The one I attempted aborted for two
   reasons unrelated to the recursion: I was editing modules while it ran (Windows
   `ProcessPoolExecutor` workers re-import from disk, hence the traceback whose line
   numbers landed on docstrings), and at `target_eta=1.8` two of five Rabi rows drop
   for leakage, leaving fewer than the four `fit_shift_curve` needs. Re-run at a
   weaker drive with the tree quiescent.
4. **`IQFourierEnvelope(shape=...)`** — the CRAB ansatz still hardcodes its Hann shape
   function, so a `sine_power` baseline and a CRAB run would disagree about the
   envelope. Only matters when combining GRAPE with the new shape.
5. **Phase 8** — the Givens variant, blocked on extracting `kappa` per process.

### Gotchas worth knowing before picking this up

- `--drag-channel` with a NEGATIVE beat needs `=` syntax (`--drag-channel=-0.22:1`);
  argparse reads a leading `-` as a flag.
- Any ad-hoc script that calls `find_stark_resonance.scan` with `n_jobs != 1` needs an
  `if __name__ == "__main__":` guard, or Windows spawn re-imports it and the pool
  breaks. A heredoc piped to `python` cannot use multiprocessing at all.
- Do not edit the modules while a multiprocess run is live; the workers import from
  disk mid-run and the resulting traceback is misleading.

---

## Context

`tune_up.py` calibrates a subharmonic two-qubit gate: Rabi → chirp → fix amplitude →
fit length, with DRAG calibrated separately and iterated because chirp and DRAG are
mutually coupled. The DRAG it applies is **first-order Motzoi only** —
`eta → eta − i·(deta/dt)/Delta(t)` — implemented in four mirrored places.

We want the **recursive multi-derivative DRAG** of Li, Calarco & Motzoi,
*"Experimental error suppression in Cross-Resonance gates via multi-derivative pulse
shaping"*, npj QI 10, 66 (2024) / [arXiv:2303.01427](https://arxiv.org/abs/2303.01427).
Their central experimental finding is that single-derivative DRAG is *insufficient*
whenever more than one off-resonant transition matters — it can only compromise between
them — whereas recursively composing one derivative correction per transition suppresses
all of them, analytically, **with no free parameters and no calibration**. That is
directly the situation here: the pump sits near several collisions at once, each already
tagged by `sweep_common._nearest_collision` with a pump-quanta count.

Intended outcome: the solver plays a recursively-corrected pulse, and `tune_up` calibrates
*that* pulse rather than a first-order model of it.

### The paper's math (ground truth for this work)

Perturbative substitution operator, **Eq. (4)**:

    Omega = F^(n)_Delta(Om) := ( Om^n − i·d/dt[ Om^n / Delta ] )^(1/n)

`n` = drive photons in the suppressed transition; `n=1` is familiar DRAG.

Recursive pulse, **Eq. (8)**, composed **right to left**:

    Omega_CR = F^(1)_{D21} ∘ F^(1)_{D10} ∘ F^(2)_{D20} (Omega)

Explicit form, **Eqs. (10)–(12)**:

    Omega_CR = Om1 − i·Om1' / D10
    Om1      = Om2 − i·Om2' / D21
    Om2      = sqrt( Om3² − i·2·Om3·Om3' / D20 )

Base shape, **Eq. (13)**, "chosen such that the obtained pulse is continuous and starts
and ends in zero", m-times differentiable with **m vanishing derivatives at both ends**:

    Omega^(m)(t) = Omega_max · I0 · ∫₀ᵗ sin^m( pi·t' / t_r ) dt'

> **Transcription note.** The paper prints `sin^m(pi t'/2 t_r)`, which makes the integrand
> equal 1 at `t = t_r` and so `Omega'(t_r) != 0` — contradicting the property the equation
> is introduced to provide. The `pi t'/t_r` reading above is confirmed by the paper's own
> remark that *"for m = 1 and with zero holding time, the pulse is the same as the Hann
> window"*: `∫₀ᵗ sin(pi t'/t_r) → ½(1 − cos(pi t/t_r))`, which at `t_r = t_g/2` is exactly
> `RaisedCosine`. Paper uses **m = 3**, and warns against larger m (high-frequency content
> → non-adiabatic error). **Re-confirm against the PDF before Phase 5.**

The paper's "chirp" is the time-dependent detuning cancelling the residual IZ term; they
approximate it by a constant ramp. This repo's `envelope.Chirp` is already the *general*
Legendre `delta(t)` — so per your answer, **the existing chirp is the chirp**. What is
genuinely new on the chirp side is that verbatim Eq. (4) with time-dependent `Delta` needs
`delta'(t)` and `delta''(t)`, which do not exist anywhere in the codebase.

### Key correspondence

The repo's `drag_n_pump` k (`sweep_common._PUMP_QUANTA = {"onepump":1,"static":0,
"subharm":2,"none":1}`) is physically the same integer as the paper's photon number n.
But it enters a **different place** — today it only appears in the *denominator*
(`Delta(t) = Delta_0 − k·delta(t)`), whereas the paper's n is a *numerator* exponent
(`Om^n`). Overloading one field for both would silently switch every existing
`drag_n_pump=2` call site (`sweep_target.py:142`, `sweep_spectator.py:101/128/179`,
`tune_up --drag-n-pump 2`) from first- to second-order DRAG. **Keep them separate.**

---

## Design decisions (confirmed with you)

| | |
|---|---|
| Recursion | Li/Calarco/Motzoi Eqs. (4), (8), (10)–(12) |
| Chirp | the existing `envelope.Chirp`; no new detuning term |
| Scope | core pulse (all solve paths) **and** `tune_up` |
| Base shape | add the Eq. (13) `sin^m` ramp so the recursion is well-posed |
| Channels | explicit list is the API; auto-fill from `_nearest_collision` as the default |

---

## Why the new base shape is a hard prerequisite, not a refinement

Hann has `Om(0) = Om'(0) = 0` but `Om''(0) = amp/2·(2pi/t_g)² != 0`. Two consequences,
both quantitative and both turned into regression tests below:

- **Two nested `F^(1)`** give `x − i·x'/Da − i·x'/Db − x''/(Da·Db)`, so the pulse turns on
  with a **finite real amplitude step** `−x''(0)/(Da·Db)` — broadband excitation, and it
  invalidates the frame transformation the construction rests on.
- **`F^(2)` innermost** is worse. For `Om3 ~ c·t^p`, the imaginary term dominates as
  `t→0` (their ratio `2p/(t·Delta)` diverges), so `Om2 ~ t^(p−1/2)`. Hann has `p = 2` →
  `t^1.5`; two further derivatives give `t^(−1/2)` — **the recursive pulse diverges at
  both gate edges**, with the first grid sample scaling as `dt^(−1/2)`. The Eq. 13 shape
  at `m = 3` has `p = 4` → `t^3.5`, which survives three derivatives with room to spare.

So nothing beyond a single `F^(1)` is physically meaningful on Hann. The math machinery
(Phases 1–4) can still be built and unit-tested against finite differences without the new
shape, which is why the phases separate cleanly — but no physics claim holds until Phase 5.

---

## Phase 1 — derivative primitives (no behaviour change)

**New `src/snail_solver/jet.py`.** Truncated Taylor-coefficient arithmetic:
`a_k = f^(k)(t)/k!`, so multiplication is a convolution and `d/dt` is `b_k = (k+1)a_{k+1}`.
Ops: `add/sub/neg/scale/mul/div/deriv/powi/powf`, plus `conj/abs2/absj/arctanj` (Givens
only, Phase 8). `Jet.from_derivs` / `.derivs()` convert at the module boundary.

Chosen over closed-form hand expansion (unauditable with three different time-dependent
`Delta_j` under a square root, and would need hand-mirroring into `jax_engine`), over
`jax.jacfwd` (breaks the numpy/QuTiP callback — the repo's core invariant is that
`_eta_at` and `jax_engine.eta_at` are the same function twice), and over finite
differences (roundoff floor `~eps/h³` at 3rd order, and it destroys the exact order-1
reduction the byte-identity regression depends on). `jax.experimental.jet` is this same
algorithm, JAX-only — perfect as a **test oracle**, unusable as the implementation.

Two invariants, both easy to get silently wrong:
- **Static order.** `order` is a Python `int` so the loops unroll under trace. A chain of
  K operators needs the base jet at order K (each `F` consumes exactly one).
- **Safe divide.** `div`/`powf` divide by `a_0`, which vanishes at the gate endpoints. Use
  the **double-`where`** guard (`den_safe = where(mask, den, 1.0)`; `where(mask, num/den_safe, 0.0)`)
  — a single `where` still produces NaN in the *backward* pass. This is the most likely
  silent breakage in the feature; it needs an explicit comment.

**`Envelope.jet_at(t, order, xp)`** on the base class (order ≤ 1 delegates to existing
`value_at`/`deriv_at`; ≥ 2 raises `NotImplementedError` naming the class), with closed-form
overrides on `ConstantPulse`, `RaisedCosine`
(`d^n/dt^n[−amp/2·cos(wt)] = −amp/2·w^n·cos(wt + n·pi/2)`) and `IQFourierEnvelope`
(Leibniz over `_shape`/`_mod`). Every envelope here is a finite trigonometric polynomial,
so arbitrary-order analytic derivatives are cheap.

**`Chirp.detuning_jet`** in [envelope.py](src/snail_solver/envelope.py) — differentiate the
existing `_legendre_stack` recurrence in place:
`dP_{k+1} = ((2k+1)(P_k + u·dP_k) − k·dP_{k−1})/(k+1)`, chain rule `du/dt = 2/t_g`.
`_u` **clips** to `[−1,1]`, so apply the support mask explicitly rather than relying on
`xp.clip`'s subgradient. Mirror as `jax_engine._chirp_detuning_jet` reading
`params["chirp"][p]` (must stay a grad axis).

Zero call sites change in this phase.

## Phase 2 — the recursion, unwired

**New `src/snail_solver/drag.py`** — one function, the single source of truth, knowing
nothing about `PumpTone`/`Chirp`/`Envelope`/`spec`:

```python
def drag_apply(shape_jet, delta_jets, channels, xp):
    """Compose F^(n_j)_{Delta_j} over `channels`, innermost first."""
    g = shape_jet
    for ch, dj in zip(channels, delta_jets):
        p = g.powi(ch.n_photon)
        q = (p.div(dj).deriv() if ch.quotient_rule    # Eq. (4) verbatim
             else p.deriv().div(dj))                  # legacy: Delta held constant
        g = p.sub(q.scale(1j)).powf(1.0 / ch.n_photon)
    return g
```

**Ordering is enforced, not requested**: `drag_apply` sorts by `n_photon` descending
(stable, so ties keep caller order). Reason to record in the docstring: `F^(n≥2)` forms
`Om^n` and takes an n-th root, whose branch is unambiguous only while `Om` is real (as the
raw base shape is); applying it to an already-complex `F^(1)`-corrected amplitude flips
branches and invalidates the perturbative bookkeeping. `F^(1)` factors commute among
themselves only to the order kept — the difference is `O(1/D³)`, exactly the dropped residual.

**`DragChannel`** (frozen dataclass, in `envelope.py` so it can be hashed as a jit static):

```python
beat_GHz: float;  n_pump: int = 1;  n_photon: int = 1
mode: str = "perturbative";  quotient_rule: bool = False;  kappa: Optional[float] = None

@classmethod
def from_collision(cls, beat_GHz, kind, **kw):   # auto-fill path
    k = _PUMP_QUANTA.get(kind, 1)
    return cls(beat_GHz, n_pump=k, n_photon=max(k, 1), quotient_rule=True, **kw)
```

`quotient_rule` exists because **today's code is not Eq. (4) verbatim on a chirped tone**:
[zhou_coupler.py:609](src/snail_solver/zhou_coupler.py#L609) computes `eta − i·eta'/Delta`,
while Eq. (4) at n=1 expands to `eta − i·eta'/Delta + i·eta·Delta'/Delta²`. On an unchirped
tone `Delta' = 0` and they agree exactly; chirped, the missing term is ~1–2% of the
correction at a 300 MHz beat and O(1) near a collision. Default `False` keeps the legacy
path byte-identical; the recursive path sets `True`. A test *measures* the discrepancy so
the choice is documented rather than assumed.

**`PumpTone`** gains one field (all existing fields have defaults, so positional
construction is preserved) and one resolver:

```python
drag_channels: Optional[Sequence[DragChannel]] = None

def drag_channels_resolved(self) -> Tuple[DragChannel, ...]:
    """Normalized, innermost-first; () when DRAG is off — mirroring `make_chirp`
    returning None, which is what keeps the DRAG-off solver path byte-identical."""
```
Explicit list wins; else the scalar `delta_drag_GHz`/`drag_n_pump` shorthand; else `()`.
Plus `channel_detuning(ch, t, xp)` and `channel_detuning_jet(ch, t, order, xp)`.

## Phase 3 — wire the four mirrored paths

`drag_apply` is shared, so the *recursion* is never duplicated; only jet construction is
mirrored, which is smaller and already has a pattern (`TestChirpPhaseImplementationsAgree`).

**Byte-identity fast path**, in both `_eta_at` and `jax_engine.eta_at`. Jet arithmetic
differs from `a − 1j*b/d` in the last ULP, so keep an explicit branch when there is exactly
one perturbative `n_photon == 1`, `quotient_rule == False` channel. This is not a
micro-optimization dodge: it keeps `test_chirp_gradient_flows_through_drag`'s JAX graph and
the two `test_the_correction_is_observable` numbers bit-identical, **and** it makes the
general path's correctness an independently testable claim
(`assert_allclose(fast, general, atol=1e-13)`) rather than an assumption.

1. **[zhou_coupler.py:577-622](src/snail_solver/zhou_coupler.py#L577-L622)** `_eta_at` —
   build `shape_jet` + `delta_jets`, call `drag_apply`, then apply the chirp phase and
   `phi_p` exactly as now. The existing ordering commentary (chirp phase applied *after*,
   never differentiated) stays valid and should be restated for the recursive case.
2. **[jax_engine.py:265-357](src/snail_solver/jax_engine.py#L265-L357)** —
   `_shape_at(..., deriv: bool)` → `_shape_jet_at(..., order: int)`; replace the `kind`
   if/else with a table `_SHAPE_DERIV = {"ConstantPulse":…, "RaisedCosine":…,
   "IQFourierEnvelope":…, "SinePowerRamp":…}` so a new envelope is one entry in each of two
   files. Keep `deriv: bool` as a thin `order=1` wrapper. `pulse_spec` gains
   `"drag_channels"` as **static** data (frozen dataclass of floats/ints/strs/bools) beside
   the existing static `drag_rad`. **Load-bearing:** chirp coeffs stay in `params`, never
   `spec` — the recursive path makes this worse than today, since the chirp now enters via
   `Delta`, `Delta'` *and* `Delta''`, so three gradient routes would silently die.
3. **[grape.py:220-243](src/snail_solver/grape.py#L220-L243)** `_raised_cosine_eta` →
   `_baseline_eta(env, n_ctrl, channels, chirp)` routed through `drag_apply`; callers at
   `grape.py:905,997`. **Also**: `optimize_pulse` saves/restores `tone.drag`,
   `delta_drag_GHz`, `chirp` in place (`:859`, `:895-897`, `:906`, `:1048`) —
   `drag_channels` must join that set, or an optimizer run leaves a channel list installed
   on a tone whose `drag` flag was reset, and "explicit list wins" makes that stick silently.
4. **`tune_up.chirp_from_measured_shift`** — eliminated as a mirror entirely (Phase 6).

**Guards.** `device_utils.check_drag_detuning`
([device_utils.py:132](src/snail_solver/device_utils.py#L132)) loops over channels and
names the offender; **keep the exact substring `"min|Delta(t)|"`** (asserted at
`test_physics.py:1104`) and keep the single-channel message verbatim.
`drag_detuning_floor` keeps returning one float (min over channels) since
`test_physics.py:1101/1107` assert on it; add `drag_detuning_floors()` for the per-channel
list. `sweep_common._drag_ok_with_chirp` keeps its exact signature (five tests pin it) and
delegates to a new `_drag_channels_ok`; add `_drag_channels_filtered` so the sweeps **drop
the offending channel and keep the rest** rather than disabling DRAG wholesale.

**New guard the recursion needs.** `min|Delta(t)|` is necessary but not sufficient:
perturbative `F^(n)` is valid only while `|Om'/(Om·Delta)| << 1` at every level, and with
three nestings the product can exceed 1 — the "corrected" pulse ends up *larger* than the
base. Add `device_utils.drag_correction_ratio(tone, n=257)` =
`max_t|eta_corr − eta_base| / max_t|eta_base|`. **Warn, never raise** (matching the
pipeline's report-don't-refuse style at `tune_up.py:2143`) at ~0.3. Nothing else in the
codebase would surface this.

## Phase 4 — tests for Phases 1–3

See **Testing** below. Land before Phase 5 so the math is nailed down independently of the
shape change.

## Phase 5 — `SinePowerRamp` (Eq. 13)

```python
class SinePowerRamp(Envelope):
    def __init__(self, amp, t_g, m: int = 3, t_rise: Optional[float] = None)
```
`t_rise=None` → `t_g/2` (a bell, the direct Hann analogue); otherwise rise / plateau /
mirrored fall, assembled with `xp.where` masks, never a Python branch on `t`. Implement by
expanding `sin^m(pi·t'/t_r)` in its finite Fourier series (`sin³x = (3sin x − sin3x)/4`;
general m from the binomial expansion of `((e^ix − e^−ix)/2i)^m`), integrating termwise for
`value_at` and differentiating termwise for `jet_at`. `I0` and `area()` are closed forms,
not quadrature, so both stay exact and trace-safe. Smoothness: `Om ~ t^(m+1)` at the edges;
`m = 3` covers `K = 3` with no headroom at the plateau junctions — worth a comment, and a
reason not to push K above m.

**The mitigation that makes the Hann-specific algebra cheap:** `SinePowerRamp(m=1,
t_rise=t_g/2)` **is** `RaisedCosine`. Assert to 1e-15 and every Hann constant below is
protected by construction rather than by re-derivation.

Affected, in order of cost:
1. `stark_chirp.HANN_STARK_LEGENDRE` (:78-88) / `HANN_MEAN_FACTOR` (:91) — the Legendre
   projection of `cos⁴(pi·u/2)`. Add `stark_chirp.shape_stark_legendre(shape_fn, degree)`
   doing the Gauss-Legendre projection for an arbitrary shape; keep the Hann array as a
   cached constant **and** as the regression fixture for the general projector.
2. `tune_up._area` / `fixed_eta_amp_scale` / `peak_eta_of`
   ([tune_up.py:97-123](src/snail_solver/tune_up.py#L97-L123)) hardcode `area = amp·t_g/2`.
   Parameterize by an area factor `f = area/(amp·t_g)` (Hann: exactly `0.5`):
   `fixed_eta_amp_scale = eta*·t_g·f/A`, `peak_eta_of = amp_scale·A/(t_g·f)`. Both reduce
   to today's expressions at `f = 1/2` with the same float ops.
3. `chirp_from_measured_shift`'s `s = cos(pi·u/2)**2`
   ([tune_up.py:1009](src/snail_solver/tune_up.py#L1009)) → `|env.value_at(t(u))|/amp`.
4. `jax_engine._shape_jet_at` — one table entry (Phase 3).
5. `IQFourierEnvelope` hardcodes the Hann CRAB shape function (`_shape`/`_dshape`, :296-302).
   Add optional `shape: Optional[Envelope] = None` (None → today's inline Hann,
   byte-identical); keep `test_zero_coefficient_iq_reduces_to_raised_cosine` as the
   `shape=None` case and add a parallel `SinePowerRamp` one.
6. Registry: add `envelope.ENVELOPE_KINDS = {"raised_cosine":…, "constant":…,
   "sine_power":…}` and route `build_coupler`'s `EnvCls` (:234) and
   `find_stark_resonance.scan(shape=…)` through it. **Never change the default** —
   `SinePowerRamp` is opt-in via `config["envelope"]`.
7. `grape._drag_seed_params` (:275-291) assumes the Hann derivative. Seed only, and every
   seed is scored before use — leave it, add a note.

## Phase 6 — `tune_up` calibration

**`chirp_from_measured_shift`** ([tune_up.py:1019-1048](src/snail_solver/tune_up.py#L1019-L1048)).
Today it hand-derives the Hann quadrature (`deta_dt = −eta*·(pi/t_g)·sin(pi·u)`) and forms
`sqrt(amp² + q²)`. Replace with the same `drag_apply` the solver uses, evaluated on the
Gauss-Legendre `u` grid mapped to `t = t_g(u+1)/2`, taking `|drag_apply(...).coeffs[0]|`.

- `sqrt(amp² + q²) == |amp − i·q|` **only** because at order 1 the correction is purely
  imaginary. At order ≥ 2 it has a real part `−Om''/(Da·Db)`, so `abs(...)` is the correct
  generalization and the old formula is simply wrong there. At order 1 they are
  algebraically identical — assert to 1e-15 as the "this refactor changed nothing" test.
- The calibration now models the pulse the solver actually plays **by construction**,
  rather than by hand across a comment boundary.
- **Reorder the fixed point** to iterate on Legendre *coefficients* rather than sampled
  values: the `Delta` jet needs `d/dt` of the current `delta_GHz` iterate, which is a
  sampled array. The projection at `:1050-1053` is already written — run it inside the
  loop, build a `Chirp`, take its analytic `detuning_jet`. Also gives a better convergence
  criterion (`max|dc|` in GHz, directly comparable to `run_tune_up`'s `chirp_tol_GHz`).
- Signature gains `drag_channels=None`, resolved by the same rule as `PumpTone`. All five
  `TestTuneUpDragCoupledChirp` tests use the scalar form and must pass unchanged. Returns
  gain `min_abs_detuning_per_channel_GHz` and `drag_correction_ratio`.

**`calibrate_drag_offset`** (:1869) — add `drag_channels=None`, thread to `FS.scan`; the
DRAG-off leg is untouched. Its docstring premise (`|eta_tot|² = |eta|² + [(deta/dt)/Delta]²`)
becomes `|eta_tot| = |drag_apply(...)|`, and its warning becomes *more* load-bearing: with
nested corrections there are more routes by which DRAG could shift the resonance other than
by adding drive, and this measurement is the only check.

**`drag_shift_table`** (:1925) — add `drag_channels=None` passthrough. **Keep the expected
`|eta|` exponent at 4.** (The design pass proposed `2K+2`; that is wrong. A K-fold nesting
generates the *whole* series d=1…K, so the first-order term `−i·Om'·Σⱼ(1/Delta_j)` still
dominates and the shift stays `∝ eta⁴` for any K. What K changes is the coefficient and the
subleading eta⁶/eta⁸ content.) Worth logging the fitted exponent's *drift* from 4 as the
new diagnostic.

**`run_tune_up`** (:2010) — `drag_channels` kwarg overriding `drag_beat_GHz`/`drag_n_pump`
wherever they thread (`:2062` `common`, `project` `:2073`, `shaped_residual` `:2079`,
`length_rabi` `:2155`, `calibrate_drag_offset` `:2195`). The outer chirp↔length loop
(`n_outer`, `:2107`) must trigger on `drag_channels` too. Its rationale gets stronger: the
d-th nested correction scales as `1/t_g^d`, so with K=3 the chirp↔length coupling is
`~1/t_g³` rather than `1/t_g` — expect slower convergence, so the `--max-drag-iters`
default of 4 may need raising, and the `RuntimeError` at `:2176` should mention K.

**CLI** (:2318+) — `--drag-channels beat[:n_pump[:n_photon[:mode]]]`, repeatable (or a JSON
list); `--drag-recursive` as sugar for "auto-fill the K nearest collisions via
`DragChannel.from_collision`"; `--drag-quotient-rule` to opt into Eq. (4) verbatim on the
legacy single-channel path. Keep `--drag-beat-GHz`/`--drag-n-pump` exactly as they are;
error if both forms are given. `drag_info` (:2186) gains the channel list so a saved
operating point round-trips — check `operating_points.py` and `calibration_map.py`
serializers accept it.

**Also correct a now-false docstring**: `peak_eta_of`
([tune_up.py:116-123](src/snail_solver/tune_up.py#L116-L123)) claims it is "unaffected by
DRAG (Hann has deta/dt = 0 at the peak)". Under recursion the correction at `t_g/2`
contains `−Om''/(Da·Db)`, which is nonzero — so `peak_eta()` acquires a DRAG-dependent
term that feeds `fixed_eta_amp_scale` and hence the whole fixed-|eta| premise. Surface it.

## Phase 7 — physics check (the Fig. 2b analogue)

`src/snail_solver/validate_recursive_drag.py` (or a mode of `validate_engines.py`): sweep
the spectator beat, compare gate infidelity for {no DRAG, single `F^(1)`, `F^(1)∘F^(1)`,
full `F^(1)∘F^(1)∘F^(2)`} on `SinePowerRamp(m=3)`. Claim to reproduce: recursive beats
single-derivative across the sweep, gap widening as the beats close in.

## Phase 8 — Givens variant (optional, highest risk, defer)

Eq. (7) needs `kappa` (`g = kappa·Omega`), the coupling-per-unit-drive of the *specific*
suppressed process — for the iSWAP channel that is `6·g3·lam_a·lam_b`
([zhou_coupler.py:552](src/snail_solver/zhou_coupler.py#L552)), but for an arbitrary
spectator collision it must be extracted per-term from `expand_terms_symbolic`.
`_nearest_collision` identifies *which* process but never its amplitude, so this is a real
sub-project with matching-ambiguity risk. `DragChannel.kappa` defaults `None` and
`mode="givens"` raises a clear error if unset. Endpoint behaviour is also worse than the
perturbative form (`phi' = Im(Om'/Om)` is `0/0` where the envelope vanishes).

---

## Reuse — do not rebuild

- `envelope.Chirp` (`detuning`, closed-form `phase`, `_legendre_stack`) and `make_chirp`'s
  return-None-when-inert instinct — mirror it for `drag_channels_resolved() -> ()`.
- `sweep_common._PUMP_QUANTA` / `_pump_quanta_of` / `_nearest_collision` — the auto-fill source.
- `PumpTone.drag_detuning` / `drag_detuning_floor`, `device_utils.check_drag_detuning`,
  `DRAG_FLOOR_GHz`, `sweep_common._drag_skip_GHz` / `_drag_ok_with_chirp`.
- `stark_chirp.stark_chirp_seed` / `HANN_STARK_LEGENDRE` — keep as the regression fixture
  for the generalized projector.
- The Gauss-Legendre projection already written at `tune_up.py:1050-1053`.
- `device_utils.build_coupler` — the one central tone builder; thread through it.

---

## Testing / verification

**Must pass unchanged** (the contract — if one needs editing, that is a design-review
trigger, not a test to update): `TestTimeDependentDrag` (test_physics.py:1018, especially
`test_chirp_gradient_flows_through_drag` at :1119), `TestEnvelopeArrayAPI` (:385),
`TestChirpPhaseImplementationsAgree` (:911), `TestStarkChirp` (:833),
`TestTuneUpChirpProjection` (:1421), `TestTuneUpDragCoupledChirp` (:1517),
`TestPumpQuantaMapping` (:1160).

**New, per phase:**

- *P1 `TestJetArithmetic`* — each op vs hand-derived closed forms (1e-13);
  `RaisedCosine.jet_at(order=3)` vs central FD on a strictly **interior** grid
  (`[2, t_g−2]`, matching the existing FD test's discipline — the support mask makes the
  true derivative distributional at the edges); `Chirp.detuning_jet` vs FD and vs
  `jax_engine._chirp_detuning_jet` (1e-12); where jax is available, vs
  `jax.experimental.jet` as an independent oracle; `jax.grad` through the jet at exactly
  `t = 0` and `t = t_g` produces **no NaN** (the double-`where` guard).
- *P2/P3 `TestRecursiveDrag`* — **order-1 reduction is exact** (`atol=0` on the fast path,
  1e-13 on the general path, chirped and unchirped); `drag_channels=None` is inert;
  **composition order matters with the right scaling** (swapping two `F^(1)` → difference
  `∝ D^−3`, fit the exponent and assert `2.7 < p < 3.3`; moving `F^(2)` outermost →
  `∝ D^−2` and numerically much larger; `drag_apply` reorders a wrong caller order back);
  `TestDragImplementationsAgree` over `{1,2,3} channels × {n_photon 1,2} × {chirped,
  unchirped}` at 1e-12; **gradient survives the recursion** (`jax.grad` w.r.t. chirp coeffs
  through a 3-channel `eta_at` vs central differences — strictly harder than the existing
  :1119 test); the quotient-rule term is measured (~1–3% at 300 MHz, O(1) at 20 MHz);
  N-channel guard raises naming the offender and `_drag_channels_filtered` keeps the rest.
- *P5 `TestSinePowerRamp`* — **`SinePowerRamp(m=1, t_rise=t_g/2) == RaisedCosine`** to
  1e-15 in value, deriv, `jet_at` and `area()` (the keystone); `jet_at` orders `0..m`
  vanish at both ends and `m+1` does not; add it to `TestEnvelopeArrayAPI._envelopes` to
  inherit vectorized==scalar / FD / zero-outside; **the Hann-breaks-the-recursion claim as
  a regression** — full 3-channel recursion on `RaisedCosine` has `|eta(dt)|` growing as
  `dt` shrinks (exponent near `−1/2` under grid refinement) while on `SinePowerRamp(m=3)`
  it goes to zero; `shape_stark_legendre(hann, 8)` reproduces `HANN_STARK_LEGENDRE` (1e-13).
- *P6 `TestTuneUpRecursiveDrag`* — one order-1 channel reproduces today's
  `chirp_from_measured_shift` to 1e-15; all five `TestTuneUpDragCoupledChirp` properties
  re-asserted for a 2-channel recursion; `drag_shift_table`'s fitted exponent stays ≈4.
- *P7* — slow, marked/env-gated: recursion beats single-derivative at every swept beat.

**Also test**: with a zero-ended shape and constant `Delta`, every pure-derivative term
integrates to zero over the gate, so `normalize_iswap`'s division by `envelope.area()`
([zhou_coupler.py:553](src/snail_solver/zhou_coupler.py#L553)) stays correct — but with a
**chirped** `Delta` the quotient-rule terms do **not**, so the pi/2 calibration drifts.
Bound that drift; if it exceeds ~1e-3, normalization must switch to integrating the
corrected `|eta|`.

**End to end:**
```
pytest tests/test_physics.py -q                       # after every phase
python -m snail_solver.tune_up --device devices/evan_device.json \
    --target-eta 1.8 --drag-recursive --save-point tuneup    # after P6
python -m snail_solver.validate_recursive_drag        # P7
```

---

## Open items

1. **Re-confirm the Eq. 13 transcription** against the PDF before Phase 5 (see the note in
   Context). The whole `SinePowerRamp` design assumes the `pi·t'/t_r` reading.
2. `--max-drag-iters` default may need raising once the K-dependence of the chirp↔length
   convergence is measured (Phase 6).

---

## Addendum, 2026-09-03 — how far the gate must sit from the subharmonic, measured

Section 4 above showed the truncation spread collapsing as `w_s` was walked away
from `2 w_p`, on four hand-picked points. `snail_solver.subharmonic_convergence`
turns that into a calibrated map: `coupler_levels` against
`Delta_sub = w_s - 2 w_p`, coloured by fidelity, with each column re-calibrated
(offset + length, at fixed peak `|eta|`) so the answer is truncation error rather
than a stale calibration. Two runs on `4Gate4.5SNAIL` (`w_a = 3.5`, `w_s = 4.5`,
`g3 = 0.06`, `lam = 0.1`), 13 coupler levels as the reference, tolerance 2e-3.

The axis moves the PARTNER QUBIT — `w_b = w_a + (w_s - Delta_sub)/2` — not the
SNAIL. The iSWAP rate `6 g3 lam_a lam_b eta` carries no `w_p`, so at fixed
`target_eta` every column runs at the same nominal length and drive, and only the
spurious detuning moves.

### `eta = 0.6` (inside the validity domain), 8 columns x 6 truncations, 31 min

| `Delta_sub` | \|alpha\| | transfer | F(3) | F(5) | F(7) | F(9) | F(11) | F(13) | max spread |
|---|---|---|---|---|---|---|---|---|---|
| 0.050 | 1.296 | 0.743 ! | 0.859 | 0.308 | 0.498 | 0.857 | 0.882 | 0.880 | 5.7e-01 |
| 0.074 | 0.872 | 0.985 | 0.984 | 0.108 | 0.729 | 0.800 | 0.970 | 0.976 | 8.7e-01 |
| 0.110 | 0.587 | 0.985 | 0.971 | 0.963 | 0.930 | 0.838 | 0.948 | 0.969 | 1.3e-01 |
| 0.164 | 0.395 | 0.895 ! | 0.976 | 0.912 | 0.815 | 0.715 | 0.841 | 0.927 | 2.1e-01 |
| 0.244 | 0.266 | 0.772 ! | 0.951 | 0.963 | 0.975 | 0.954 | 0.559 | 0.756 | 2.2e-01 |
| 0.362 | 0.179 | 0.999 | 0.988 | 0.994 | 0.995 | 0.997 | 0.997 | 0.996 | 8.0e-03 |
| **0.538** | **0.120** | **0.999** | **0.997** | **0.996** | **0.996** | **0.997** | **0.997** | **0.997** | **5.2e-04** |
| 0.800 | 0.081 | 0.999 | 0.991 | 0.995 | 0.971 | 0.976 | 0.982 | 0.983 | 1.2e-02 |

`!` = calibrated transfer under 0.9, i.e. that column's cells all score a badly
calibrated pulse and its spread is not evidence about the truncation.

**`Delta_sub = 0.538 GHz` is a converged operating point: every truncation from 3
to 13 levels agrees to 5.2e-04 at `F = 0.997`.** Three coupler levels is enough
there — the cheapest trustworthy simulation of this device found so far.

### `eta = 1.2` (past the validity domain), 13 columns x 6 truncations, 17 min

Ten of the thirteen columns could not be calibrated at all (transfer 0.14–0.88):
inside `Delta_sub < 0.7 GHz` the coupler holds 1.8–2.9 photons at `t_g` and the
"iSWAP" is not an iSWAP. Of the three healthy columns (0.800, 1.350, 1.900), only
`Delta_sub = 1.35 GHz` converges — 5, 9, 11 and 13 levels agree to 5.6e-04 at
`F = 0.9956`.

### The rule both runs agree on

Convergence tracks `|alpha|`, not `Delta_sub`:

| run | converged column | \|alpha\| there | nearest non-converged | \|alpha\| there |
|---|---|---|---|---|
| `eta = 0.6` | 0.538 (and 0.362) | 0.120 (0.179) | 0.244 | 0.266 |
| `eta = 1.2` | 1.350 | 0.192 | 1.150 | 0.225 |

> **SUPERSEDED — see the 2026-09-03 two-sided addendum below.** Both runs above
> sampled only `Delta_sub > 0` (`2 w_p < w_s`). Sampling the other side shows the
> effect is NOT a function of `|alpha|`: at `2 w_p > w_s` the coupler stays cold
> and 7 levels converge even at `|alpha| = 1.08`. The `|alpha|` threshold below
> is real but is a property of the POSITIVE branch alone.

Both put the threshold at **`|alpha| ~ 0.2`**, i.e.

    Delta_sub  >~  15 g3 eta^2         (|alpha| = 3 g3 eta^2 / Delta_sub  <~  0.2)

which is 0.32 GHz at `eta = 0.6` and 1.30 GHz at `eta = 1.2` for `g3 = 0.06` —
both borne out. The predicted level count `|alpha|^2 + 4|alpha| + 1` is then under
2, consistent with 3 levels sufficing at `Delta_sub = 0.538`.

**The rule is necessary, not sufficient.** Convergence is NOT monotone in
`Delta_sub`: at `eta = 0.6` the outermost column (0.800, `|alpha| = 0.081`)
regresses to a 1.2e-02 spread at 7 and 9 levels with the coupler cold
(`<n_s> = 0.001`), so something other than the displacement is at work there; at
`eta = 1.2`, 1.600 and 2.200 are both worse than 1.350. Check the specific
operating point — that is what the map is for.

### Consequences for this device

1. `4Gate4.5SNAIL` as shipped sits at `Delta_sub = +3.90 GHz`, which satisfies the
   rule at any sane drive — but it is 67 MHz from the 2-pump `b -> s` conversion
   at 3.833 GHz, a different collision entirely. Worth its own check.
2. Phase 7 work wanting a cheap converged model should use `eta = 0.6` at
   `Delta_sub = 0.538 GHz` (`w_b = 5.481`, `t_g = 209.5 ns`, 3 coupler levels).
3. `eta = 1.2` is not usable on this device at any detuning sampled inside
   2.2 GHz. That is the same conclusion section 6 reached from drive strength
   alone, now with the placement axis held responsible separately.

Reproduce (both runs are cached under `results/subharm_*`, so a re-run only fills
gaps, and `--replot` re-reads the boundary at a different `--tol` for free):

```bash
uv run python -m snail_solver.subharmonic_convergence \
    --device 4Gate4.5SNAIL.json --target-eta 0.6 \
    --detunings 0.05:0.8:8:log --levels 3,5,7,9,11,13 --tg-lo 0.5 \
    --wp-points 31 --jobs 30 --outdir results/subharm_4Gate4.5SNAIL_eta0p6 \
    --plot figs/subharm_4Gate4.5SNAIL_eta0p6/convergence_map.png
```

---

## Addendum 2, 2026-09-03 — isolating the subharmonic: it is sign-asymmetric

The runs above walked the pump one way only (`2 w_p < w_s`, `Delta_sub > 0`) and
concluded that convergence tracks `|alpha| = 3 g3 eta^2 / |Delta_sub|`. **That
conclusion is wrong as stated.** Sampling both sides of the resonance shows the
damage is confined to `2 w_p < w_s`; on the other branch the coupler stays cold
and the model converges at 7 levels even at `|alpha| = 1.08`.

### The design

`w_a = 3.5`, `w_s = 4.5` puts the nearest other resonances at `Delta_sub = +1.0`
(`2 w_p = w_a`) and `-2.5` (`w_p = w_a`), leaving a clean 3.5 GHz window. Inside
it, MIRRORED PAIRS at `|Delta_sub| = 0.06, 0.10, 0.16, 0.25, 0.40, 0.63` plus
negative-only columns at `-1.0, -1.6, -2.0`; `eta = 0.6`; 3-18 coupler levels
with **18 as the reference**; every column re-calibrated (offset + length) at 18
levels. 15 columns x 6 truncations = 90 cells, 47 min.

A mirrored pair shares `|alpha|` EXACTLY, while every other pump-activated
channel sits at a different detuning on the two sides because
`w_p = (w_s -/+ |Delta_sub|)/2` differs. So the pair comparison is the isolation
experiment, and it needs no modelling.

### The mirror table (`F` at 18 levels)

| `\|Delta_sub\|` | `\|alpha\|` | `F(+)` | `F(-)` | `\|diff\|` | `n_s(+)` | `n_s(-)` |
|---|---|---|---|---|---|---|
| 0.060 | 1.080 | 0.95226 | 0.99668 | 4.4e-02 | 0.0283 | 0.0014 |
| 0.100 | 0.648 | 0.92992 | 0.99669 | 6.7e-02 | 0.1471 | 0.0017 |
| 0.160 | 0.405 | 0.63908 | 0.99636 | 3.6e-01 | 0.9693 | 0.0011 |
| 0.250 | 0.259 | 0.62934 | 0.99685 | 3.7e-01 | 1.1661 | 0.0008 |
| 0.400 | 0.162 | 0.99651 | 0.99669 | 1.7e-04 | 0.0018 | 0.0011 |
| 0.630 | 0.103 | 0.99657 | 0.99499 | 1.6e-03 | 0.0017 | 0.0035 |

Median `|F(+) - F(-)| = 5.6e-02` against a 2e-3 tolerance. The two sides do NOT
agree, so `|alpha|` is not the controlling parameter.

### It really is the subharmonic, and it really is the sign

Two checks, because "the sides differ" is not yet "the subharmonic differs".

**1. Nothing else differs.** `spectator_audit.interaction_channels` at `+/-0.25`,
window 0.9 GHz:

| | `Delta_sub = +0.25` | `Delta_sub = -0.25` |
|---|---|---|
| 2p SNAIL subharmonic | det **-249.7** MHz, g 64.80, ratio 0.26 | det **+249.2** MHz, g 64.80, ratio 0.26 |
| 1p `\|2>` leakage (via A/B) | det -119.8 / -120.2 MHz, g 3.05 | det -119.6 / -120.4 MHz, g 3.05 |

The subharmonic's strength and `|Omega/Delta|` are IDENTICAL; only the sign of its
detuning flips. Every other channel is unchanged -- the `|2>` leakage sits at the
anharmonicity on both sides, as it must (`w_p - (w_b - w_a + alpha) = -alpha`,
independent of `w_p`), and so does the `b <-> s` conversion
(`w_s - w_a` identically, for `w_p > w_s - w_a`).

**2. It is not the calibration.** The `+` side's length fit failed
(`t_g = 185` vs `219 ns`), so the pulses differed too. Scoring BOTH signs with
BOTH pulses removes that:

| `Delta_sub` | pulse from | `t_g` | `F(5)` | `F(9)` | `F(13)` | `F(18)` | `n_s(18)` |
|---|---|---|---|---|---|---|---|
| +0.25 | +0.25 | 185 ns | 0.978 | 0.946 | 0.922 | **0.629** | 1.17 |
| +0.25 | -0.25 | 219 ns | 0.982 | 0.918 | 0.932 | **0.451** | 2.23 |
| -0.25 | +0.25 | 185 ns | 0.963 | 0.964 | 0.964 | **0.964** | 0.004 |
| -0.25 | -0.25 | 219 ns | 0.994 | 0.997 | 0.997 | **0.997** | 0.001 |

With either pulse the `+` side runs away and the `-` side is flat in `N`. The
failed calibration on the `+` side is a SYMPTOM of the hot coupler, not its cause.

Note the direction of the `+`-side error: `F` falls monotonically as levels are
ADDED (0.978 -> 0.946 -> 0.922 -> 0.629) while `n_s` grows. More room lets the
ladder climb further -- the `n^1.5` cubic self-term of section 6, now localised to
one branch.

### The result

| branch | 3 levels | 5 | 7 | 9 | 13 |
|---|---|---|---|---|---|
| `2 w_p > w_s` (`Delta_sub < 0`) | none of 9 | `>= 2.00` | **`>= 0.06` (9/9)** | **`>= 0.06` (9/9)** | **`>= 0.06` (9/9)** |
| `2 w_p < w_s` (`Delta_sub > 0`) | none of 6 | `>= 0.40` | `>= 0.40` (2/6) | `>= 0.40` (2/6) | `>= 0.40` (2/6) |

**Place the gate so that `2 w_p > w_s` and 7 coupler levels are enough anywhere
down to 60 MHz from the subharmonic** (`F = 0.9966`, `n_s ~ 1e-3`, spread
`~1e-3`). On the other branch nothing under 18 levels converges inside
`|Delta_sub| = 0.40 GHz`, and the fidelity there is 0.63-0.95 regardless.

### Open: the `+`-branch damage is not monotone in `Delta_sub`

`n_s` on the positive branch peaks at `Delta_sub = 0.16-0.25` (0.97, 1.17
photons) and is SMALLER closer in (0.028 at 0.06, 0.147 at 0.10), where
`|alpha|^2` is largest. That is resonance-like, not `1/Delta^2`-like, and it is
not explained by the displacement picture. A cubic-nonlinearity cascade fits the
shape -- climbing the ladder shifts successive transitions, so one sign of
detuning sweeps INTO resonance as `n` grows and the other sweeps out, with the
cascade condition met at a particular detuning rather than at `Delta -> 0` -- but
that is a hypothesis, not a measurement. The test is a fine `Delta_sub` scan on
the positive branch with `<n_s>(t)` recorded THROUGH the pulse rather than at
`t_g`.

### Two tooling findings from this run

**The GPU is 6x slower here.** `--gpu-levels-min N` routes the big truncations
through qutip-jax / diffrax. Measured on an NVIDIA GH200, `t_g = 209 ns`,
`Delta_sub = 0.538`: a 13-level cell took **761 s on the GPU against 128 s on one
CPU core** (18 levels: 176 s on CPU), at 95% device utilisation, with `F`
agreeing to six digits (0.996610). dim ~ 162 with a time-dependent coefficient is
far below the JAX crossover, and the CPU alternative is `jobs` independent cells
at once, not one core. Use it to cross-check the CPU path, not to go faster.

**Pin the BLAS threads.** At 18 levels (dim 162) numpy goes threaded where 13 did
not: 15 pool workers took **127 threads each**, ~1900 threads on 72 cores, load
average 107, each worker burning 310% CPU to do one core's work. The CLI now sets
`OMP_NUM_THREADS=1` and friends before numpy loads (and only as the CLI). Load
dropped 107 -> 17 and the run finished in 47 min.

Reproduce:

```bash
uv run python -m snail_solver.subharmonic_convergence \
    --device 4Gate4.5SNAIL.json --target-eta 0.6 \
    --detunings=0.06,0.1,0.16,0.25,0.4,0.63,-0.06,-0.1,-0.16,-0.25,-0.4,-0.63,-1.0,-1.6,-2.0 \
    --levels 3,5,7,9,13,18 --ref-levels 18 --calib-levels 18 \
    --tg-lo 0.5 --tg-hi 1.4 --wp-points 31 --jobs 30 \
    --outdir results/subharm_4Gate_mirror_eta0p6 \
    --plot figs/subharm_4Gate_mirror_eta0p6/convergence_map.png
```
