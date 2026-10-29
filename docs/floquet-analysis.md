# Floquet analysis of the 1 MHz-wide η = 1.3 scan

What sets the gate infidelity across δ = −250 … +250 MHz, read off the Floquet
quasienergies of the constantly pumped device.

- **Scan:** `results/drag_curve_1MHz_2026-10-23` (branch `scan-1mhz-wide`).
- **Code:** `src/snail_solver/floquet.py`.
- **Drivers:** `scripts/floquet_branch_map.py`, `scripts/analyze_floquet_map.py`,
  `scripts/plot_floquet_tracking.py`, `scripts/replay_leak_trajectories.py` and
  `scripts/plot_wide_curve.py`.
- **Tests:** `tests/test_floquet.py`.

The figures live in `docs/floquet-analysis/`. The repo ignores `*.png`, so they are not
tracked; [Reproducing](#reproducing) regenerates them.

## Summary

Every large error in the scan is **leakage**, not a calibration miss, and each band
belongs to one multiphoton process. Two facts of the frame explain where the processes
sit.

**1. Qubit B moves with the pump.** The conversion gate holds ω_b = ω_a + ω_p, so
ω_b/2π = 5.25 GHz + δ. Two consequences follow:

- Processes that involve B shift with δ.
- Different processes can share one beat. A |1⟩→|2⟩ with 2 pumps and B |1⟩→|2⟩ with
  3 pumps both have the beat −120 − 2δ MHz.

**2. Coupler levels Stark-shift strongly; qubit levels barely do.** The coupler 0→1
shift is about −100 |η|² MHz.

- A coupler process is therefore swept across a band of detunings during the pulse,
  100–170 MHz wide.
- A qubit-only process stays pinned near its zero-drive resonance.

Beat values below are in MHz.

| δ band (MHz) | Process (pump quanta) | Beat at zero drive | What the pulse does |
|---|---|---|---|
| −250 … −205 | 3ω_p = ω_s: \|000⟩→\|001⟩ (3) | −550 − 3δ bare (zero at −183), dressed zero near −200 | Swept, strongly coupled: at −240 a broad anticrossing at \|η\| ≈ 0.66; the coupler climbs to ⟨n⟩ ≈ 2–4, beyond the 9-level cutoff |
| −224 … −144 | \|100⟩→\|201⟩ (5) | −670 − 5δ bare, dressed zero near −144 | Swept: crosses at \|η\| ≈ 1.1 at −170 (the DRAG-sign study's 5-photon process) |
| −80 … −40 | A/B \|1⟩→\|2⟩ (2 / 3) | −120 − 2δ (zero at −60) | Pinned and strongly coupled (g ≈ 43 MHz ≳ \|beat\|) |
| −15 … +15 | A subharmonic \|000⟩→\|100⟩ (2) | −2δ | Pinned and strongly coupled (g ≈ 30 MHz) |
| +35 … +167 | \|010⟩→\|002⟩ (2) | 650 − 3δ bare, ≈ 500 − 3δ with static dressing (zero at +167) | Swept: crosses on the rise and on the fall, so the error ripples (Stückelberg) |
| +166 … +250 | \|100⟩→\|201⟩ (4) | 1080 − 4δ bare, ≈ 1035 − 4δ dressed | Swept: at +200 a broad anticrossing at the pulse peak (\|η\| ≈ 1.2–1.3) |

![Gate infidelity versus pump detuning](floquet-analysis/infidelity_vs_detuning.png)

*Figure 1. Coherent gate infidelity of the bare pulse, the ridge chirp, and chirp + the
fixed DRAG set. The top axis gives the absolute pump frequency. Grey bands: the coupler
holds more than its 9-level cutoff represents, so those values read too good.*

## The Floquet model

The model is a pump of constant displaced amplitude |η| on one column's device, in the
lab frame, with no rotating-wave approximation:

    H(t) = Σ_i ω_i n_i + H_anh + Σ_n g_n [ X_s + 2|η| cos(ω_p t) ]^n ,
    X_s  = Σ_i λ_i (a_i + a_i†)

It is built from the same `ZhouCoupler` operators the gate solver uses
(`floquet.model_at_detuning`), i.e. the dressed-mode model of the charge-pumped SNAIL
coupler [1, 2]. Dropping the RWA matters here: counter-rotating pump terms are what
make several of the multiphoton processes below possible [11, 12].

- **Quasienergies.** H(t) has period T = 2π/ω_p, so Floquet's theorem applies
  [3, 4, 5]. The one-period propagator U(T) uses exponential-midpoint steps (the
  second-order Magnus integrator [7]), 1000 per period (converged).
- **Floquet modes.** U(T)'s eigenphases give the quasienergies ε = −arg λ / T, defined
  modulo ω_p. Its Schur vectors give the Floquet modes, and they stay orthonormal where
  eigenvalues coincide [9].
- **Validation.** The module reproduces the stored DRAG-sign-study coupler shift,
  −16.56 MHz at |η| = 0.4 for δ = −170, to 0.01 MHz.

A constant-drive Floquet spectrum is the adiabatic picture of the gate. The calibrated
pulses are smooth (sine-power m = 3), about 100 ns long, and peak at |η| ≈ 1.3. So at
each instant the state follows the Floquet modes at the instantaneous |η(t)|. The
pulse leaks only where that following breaks down:

- at a resonance the pulse sweeps through;
- where a coupling is strong enough that the dressing does not return adiabatically.

### Tracking branches

Following "the |002⟩ level" up to strong drive is the hard part. Three approaches were
tried, and the first two fail at the drives that matter. The order is kept because the
failures are informative.

1. **Maximum overlap, state by state.** This follows the eigenvector continuously. In
   the dense coupler ladder it wanders off the state: by |η| ≈ 1 the branch tracked
   from |002⟩ keeps 0.4% of its |002⟩ weight.
2. **Largest weight on the Fock state.** By |η| ≈ 1 a coupler Fock state is spread
   over several modes, the best holding only ~25% of its weight. The choice hops
   between modes, and the beat jumps by hundreds of MHz.
3. **All branches at once, labelled by continuity** (`floquet.track`). Every mode is
   labelled by its |η| = 0 Fock state. Between steps the labels are carried by the
   assignment that maximises the total overlap (Hungarian algorithm [8], via SciPy
   [10]), so no two
   branches claim one mode. On a fine grid (201 points to |η| = 1.5):
   - a narrow avoided crossing is passed straight through;
   - a broad one is followed round, and its two labels swap character.

   These labelled branches are the ones drawn in Figures 3, 4, 6, 7, 8 and 9.

The module also computes a windowed **diabatic energy**,
E_s = Σ_m |⟨s|m⟩|² ε_m. Only modes within 150 MHz of the running estimate enter the
sum. The window matters:

- Static g₃ dressing mixes |s⟩ slightly with levels whole GHz away. Folded into one
  zone, those admixtures land a few hundred MHz off and would bias a plain sum.
- The near-resonant hybrids that make a crossing lie inside the window.

The diabatic energy is exact at weak drive. It leaves out level repulsion from
partners inside the window, about 4 MHz on the coupler shift at |η| = 0.4 (bounded by
the `spread` it reports). It loses meaning once a state is spread over a band wider
than the window. For |002⟩ that happens above |η| ≈ 1: less than half its weight lies
within 150 MHz. So the dressed-beat figure masks it there, and the resonance positions
come from the branches instead.

### Two observables

- **Hybridisation** (`hyb`): how much of the leak state the computational branch
  carries. That is the |100⟩/|010⟩ branches' mean weight on |f⟩, or the |000⟩ branch's
  weight for a process that starts from |000⟩. Plotted over (δ, |η|), each process
  shows up as a fan from its zero-drive resonance (Figure 2).
- **Resonance drive |η|*** (`eta_meet`): the first |η| at which some branch carrying at
  least 5% of |f⟩ comes within 15 MHz of the computational branch. It doesn't depend on
  any one label keeping its character.

  A pulse whose peak exceeds |η|* crosses the resonance twice, once on the rise and once
  on the fall. The 15 MHz threshold misses anticrossings wider than 30 MHz, and the
  strongly coupled qubit processes have those. Their reach is read from g/|beat|
  instead (Figure 10).

![Hybridisation maps](floquet-analysis/floquet_hybridization_maps.png)

*Figure 2. Weight of the computational branch on each leak state, over pump detuning
and constant drive. The dashed line is the gates' peak drive. The coupler processes
fan out sideways as the drive grows: |010⟩→|002⟩ from +167, the two |100⟩→|201⟩
processes from about −144 and +260, and 3ω_p = ω_s from −200. The qubit-only
processes, A |1⟩→|2⟩ at −60 and the A subharmonic at 0, stay put and broaden. The
vertical stripes are label hand-offs between neighbouring detunings, not physics.*

## How each feature arises

### Positive side, +17 … +167 MHz: two crossings that interfere

The leaked population goes to **|002⟩**, two coupler photons with both qubits empty.
From |110⟩ it goes to |102⟩. The replay of the calibrated pulses shows this directly
(Figure 11, left).

The process is |010⟩ → |002⟩: B's photon plus two pump photons become two coupler
photons. Its beat is bare 650 − 3δ MHz. The coupler's static self-Kerr lowers |002⟩ by
about 150 MHz, so the dressed zero-drive resonance sits at +167. With drive the
coupler Stark shift pulls it to smaller δ.

The |002⟩-carrying branch reaches the doublet (|η|*) at:

| δ (MHz) | +40 | +60 | +80 | +100 | +120 | +140 | +160 |
|---|---|---|---|---|---|---|---|
| \|η\|* | 1.18 | 1.06 | 0.78 | 0.86 | 0.72 | 0.53 | 0.16 |

From about +35 up to +167 the calibrated pulse (peak 1.3) therefore passes the resonance
twice (Figures 3 and 4).

Each passage transfers some amplitude into |002⟩. This is Landau–Zener–Stückelberg
interference [13–17]: the two amplitudes interfere with a
phase that grows as 3 (t₂ − t₁) per MHz of δ, because the beat carries −3δ. So the
infidelity ripples in δ with period **1 / (3T)**, where T is the time the pulse spends
above |η|*.

Figure 5 computes T on each column's own calibrated envelope; no parameter is fitted:

- **At +40 MHz:** T = 28 ns, predicting 11.8 MHz against 11.6 measured.
- **Above +75 MHz:** |η|* is lower, so T is longer (43–51 ns) and the period shorter.
  The prediction is 6.5–7.7 MHz against about 5.4 measured. That's the right trend, but
  30–40% high; a second leak into |004⟩ appears there and may account for part of it.

Below about +35 the resonance is approached but not crossed: |η|* ≈ 1.4–1.45 at +20 …
+28. There the A-subharmonic wing (below) dominates.

![Branches at +40 MHz](floquet-analysis/floquet_branches_d+40.png)
![Branches at +120 MHz](floquet-analysis/floquet_branches_d+120.png)

*Figures 3–4. Labelled Floquet branches relative to the computational doublet (black),
coloured by their weight on |002⟩.*
- *At +40 the |002⟩ branch comes down and meets the doublet at |η| ≈ 1.18.*
- *At +120 it meets at 0.72. Beyond that the doublet splits by ±100 MHz: it has
  hybridised with |002⟩.*

![Stückelberg period](floquet-analysis/floquet_stueckelberg.png)

*Figure 5. Top: infidelity on the positive side. Bottom: the ripple period, measured as
the spacing between neighbouring maxima, against the parameter-free prediction
1/(3T).*

### +200 MHz: a 4-pump resonance sitting at the peak

The leak goes to |201⟩ and |021⟩: one qubit in |2⟩ plus one coupler photon. Two
processes contribute, |100⟩ → |201⟩ with 4 pumps and |010⟩ → |021⟩ with 5. Because
ω_b = ω_a + ω_p, they share one beat, 1080 − 4δ (about 1035 − 4δ dressed).

At +200 the branches carrying |201⟩ come within ~37 MHz of the doublet by |η| ≈ 1.19,
and from |η| ≈ 1.3 the doublet itself splits (49 MHz at 1.4): a broad anticrossing
centred at the pulse peak (Figure 6). The pulse dwells next to it through its whole
flat top. Only 35% of the population ends in the target state, and the calibration
fell back to its shortest gate, 60 ns.

This is the 4-pump sibling of the 5-pump |100⟩ → |201⟩ process that caused the
−185…−160 MHz errors (Figure 7, δ = −170: crossing at |η| ≈ 1.06–1.12 depending on
the thresholds, against 1.13 in the DRAG-sign study). The intermediate step is the
coupler's two-pump subharmonic, with g ≈ 304 MHz.

![Branches at +200 MHz](floquet-analysis/floquet_branches_d+200.png)
![Branches at -170 MHz](floquet-analysis/floquet_branches_d-170.png)

*Figures 6–7. The |201⟩ branch reaching the doublet: at +200 MHz (4 pumps) a broad
anticrossing around |η| ≈ 1.2–1.3; at −170 MHz (5 pumps) a crossing at |η| ≈ 1.1.*

*Reading Figure 6 (+200 MHz). The two orange curves are one state, |201⟩, handed
between branches. They are not two leak processes.*
- *|η| = 0 … 0.95: the upper dark curve is the |201⟩ branch (98% |201⟩ at zero drive),
  starting 230 MHz above the doublet. The coupler Stark shift pulls it down to about
  +125 MHz.*
- *|η| ≈ 0.96: a steep coupler-ladder level crosses it in a narrow avoided crossing,
  and the two branches swap character. The faint line turning upward is the original
  |201⟩-labelled branch, now mostly |001⟩/|002⟩. The orange line continuing downward
  has taken over the |201⟩ weight (0.37 at |η| = 1.0).*
- *|η| ≈ 1.0 … 1.3: the descending line pales as |201⟩ spreads over several branches
  (10–30% each) and mixes with |200⟩, one coupler photon away. It comes within about
  25–35 MHz of the doublet at the peak drive (dotted) without crossing.*
- *|η| > 1.3: the doublet is pushed apart to about ±55 MHz, and faint |201⟩ weight
  reappears below it: the doublet has hybridised with the |201⟩/|200⟩ cluster.*

*So the resonance is reached only at the pulse peak, as a strong, broad anticrossing.
Population moves during the flat top, not in a fast passage (Figure 11, middle). Breaks
in the lines are drawing artefacts: segments where a branch folds by ω_p are dropped,
and segments with little |201⟩ weight fade to grey.*

### −250 … −205 MHz: the coupler is driven up its ladder

This band is 3ω_p = ω_s, where three pump photons make one coupler photon
(|000⟩ → |001⟩). It does not involve the qubits, so every computational state leaks.
Its dressed zero-drive resonance is near −200. At −240 the branches carrying |001⟩ come
within ~53 MHz of the |000⟩ branch at |η| ≈ 0.66 and are repelled, a broad, strongly
coupled anticrossing (Figure 8).

Past that point the coupler is driven resonantly and climbs, to ⟨n⟩ ≈ 2–4 in the
replay (Figure 11, right). A pump-driven climb up a Josephson
mode's ladder of this kind is described in [11, 12]. The population spreads over
|101⟩, |103⟩, |006⟩, |204⟩ and more, and 1 − F saturates near 0.7. Nine coupler
levels cannot represent this, so the band locates the resonance but gives no reliable fidelity.

![Branches at -240 MHz](floquet-analysis/floquet_branches_d-240.png)

*Figure 8. At −240 MHz the |001⟩-carrying branches reach within ~53 MHz of the |000⟩
branch (black) at |η| ≈ 0.66 and turn round; above that the coupler ladder is a dense
mixture.*

### Why the left edge is a cliff and the right side a slow tail

Figure 1 shows an asymmetry. Going left from the −60…0 plateau, 1 − F falls from 0.45 at
−68 MHz to 0.004 at −88: a factor 100 in 20 MHz. Going right from +5, it falls only
5× by +25 and then declines slowly:

- **Bare:** about δ^−1.8, close to 1/δ².
- **Chirped:** about δ^−1.

The cause is **which kind of process lies beyond each edge** (Figures 9 and 10).

- **Left edge.** The plateau is the A/B |1⟩→|2⟩ process: qubit-only, pinned near
  −60, strongly coupled. Its first-order coupling at the target drive is g ≈ 43 MHz,
  against a beat of |2δ + 120|. While g/|beat| ≳ 1 (−80 ≲ δ ≲ −40), the replay leaks
  into |200⟩ and |020⟩:

  | δ (MHz) | −60 | −66 | −72 | −76 | −80 | −88 |
  |---|---|---|---|---|---|---|
  | Leaked population | 75% | 32% | 14% | 3% | 1% | < 0.1% |

  Past −80 the ratio drops below 1. Because the process does not Stark-shift, the pulse
  never brings it back into resonance. With a smooth envelope the off-resonant dressing
  returns adiabatically, so the leak collapses rather than tailing off [6]. The next
  crossed process on this side is the 5-pump |201⟩ band, which starts at −144.

  Figure 9 shows the pinning. At −70 the |200⟩ branch starts 20 MHz above the doublet
  and only moves away as the drive grows, carrying a large |200⟩ admixture with it.
  Nothing is crossed; the doublet is simply strongly dressed.

  Between the two lies a clean window, −96 … −88 MHz, at the ~3 × 10⁻³ floor. The
  DRAG-sign study found that floor is mostly static dressing. This is where the scan's
  best gate sits: 2.4 × 10⁻³ at −97.
- **Right side.** The same kind of fall starts from the A subharmonic (g/|beat| = 1 at
  +15). It is cut short by the swept coupler process |010⟩→|002⟩. From +35 to +167 the
  pulse crosses that resonance on every column, so there is no clean window. As δ
  grows, the crossing happens at weaker drive (|η|* falls from 1.18 to 0.16). The
  process takes two pump quanta, so its coupling grows as |η|², and it is weaker there.
  The crossing leak therefore shrinks gradually with δ, as a power law with the
  Stückelberg ripple on top, instead of collapsing.
- **Bare vs. chirped on the right.** The bare trace's steeper, roughly 1/δ² slope also
  includes the uncorrected Stark detuning of the doublet. The chirp removes that part,
  which leaves the ~δ^−1 leak envelope.

![Branches at -70 MHz](floquet-analysis/floquet_branches_d-70.png)

*Figure 9. At −70 MHz the |200⟩ branch sits 20 MHz above the doublet at zero drive and
is pushed away as the drive grows: a pinned, strongly coupled qubit process, not a
swept crossing.*

![Leak mechanisms](floquet-analysis/floquet_leak_mechanisms.png)

*Figure 10. Top: infidelity. Middle: g/|beat| of the two pinned qubit processes; both
edges of the high-error plateau sit where these cross 1. Bottom: the resonance drive
|η|\* of the swept coupler process |010⟩→|002⟩. Wherever it lies below the peak drive
(shaded), the pulse crosses the resonance. The dip at +76 … +80 is where a second
|002⟩-carrying branch meets first; the thresholds move |η|\* most there (0.70–0.97).*

![Replay dynamics](floquet-analysis/replay_leak_dynamics.png)

*Figure 11. Exact replays of the calibrated pulses.*
- *Left: |002⟩ fills on the rise (~0.3 t_g) and takes a final step on the fall
  (~0.7 t_g). That is the two-passage interference; the final value ripples with δ.*
- *Middle: at +200 MHz, transfer into |201⟩/|021⟩ during the flat top.*
- *Right: coupler occupation. It exceeds two photons at −240.*

### The dressed beats

The dressed beat E_f − E_i − kω_p on the diabatic energies is masked wherever either
state keeps less than half its weight inside the window. `analyze_floquet_map.py`
stores it as `beat_<process>`. Where it is valid it agrees with the branch picture:

- At +40 the beat falls from 380 MHz at zero drive to ~125 MHz by |η| ≈ 1.05. There
  |002⟩ dissolves into a band ~100 MHz wide, which overlaps the doublet.
- At −170 and +200 it crosses zero at |η| ≈ 1.03 and 1.05. That is somewhat earlier
  than the branch picture (≈ 1.1, and the +200 anticrossing at 1.2–1.3), because the
  diabatic energy omits the level repulsion.

Treat the branch-meeting |η|* as the resonance position. The diabatic beat says which
way the resonance moves.

## Convergence and caveats

- **Coupler levels.** See below for 9 vs. 12 levels.
- **Constant drive versus a pulse.** |η|* is a constant-drive property. A pulse crossing
  it is the right picture where the crossing is slow on the scale of 1/gap. The
  Stückelberg period at +40 agrees to 2%, which supports that.
- **Thresholds.** |η|* depends on its thresholds (5% weight, 15 MHz). Over 3–10% and
  10–25 MHz it moves by about ±0.03 at most detunings (+40: 1.16–1.18; +120:
  0.69–0.73; −170: 1.06–1.12), but by up to ±0.15 around +80, where several
  |002⟩-carrying branches approach at once. It also misses broad anticrossings
  (gap > 30 MHz) such as +200 and −240; those are read from the branch diagrams.
- **Isolated |η|\* hits.** On the negative side the |002⟩ meeting test also fires at
  scattered detunings, where ladder branches carrying ~5% |002⟩ pass the doublet at
  high drive. The figures use it only for δ ≥ +30, where the process is the
  identified one.

### 9 vs. 12 coupler levels

The scan used 9 coupler levels, so the 9-level map is the like-for-like explanation of
its numbers. A 12-level map on every 10 MHz (`levels12/`) tests whether the physics
holds for the device itself. Each entry is |η|* at 9 / 12 levels.

| Process | Detuning (MHz) and \|η\|* | Verdict |
|---|---|---|
| \|100⟩→\|201⟩, 5 pumps | −190: 0.83 / 0.82, −180: 0.58 / 0.58, −170: 1.09 / 1.09, −160: 0.88 / 0.88, −150: 0.47 / 0.47 | converged |
| \|010⟩→\|002⟩ | +60: 1.06 / 1.02, +100: 0.85 / 0.83, +120: 0.72 / 0.70, +140: 0.53 / 0.53, +160: 0.16 / 0.16 | converged |
| \|010⟩→\|002⟩ | **+40: 1.18 / 0.91**, **+80: 0.78 / 0.93** | not converged |
| A/B \|1⟩→\|2⟩ | hybridisation at the peak: −80: 0.07 / 0.07, −70: 0.10 / 0.10, −60: 0.13 / 0.14 | converged |

At +40 MHz a higher coupler level that carries |002⟩ meets the doublet earlier with 12
levels. That would lengthen the time above |η|* and shorten the predicted ripple period
there. The scan's own 11.6 MHz period matches the 9-level prediction, as it should,
since both use the same truncation. So:

- the positive-side mechanism (a swept |002⟩ crossing) is robust;
- its detailed crossing drive at +40…+80 MHz is truncation-limited.

This fits the earlier finding that 9 coupler levels read the η = 1.3 fidelities about
35–40% optimistic. The A-subharmonic and 3ω_p = ω_s hybridisation at the peak drive
changes between truncations at single detunings. That comes from label hand-offs in a
dense spectrum, not a systematic shift.

## Reproducing

```bash
# Floquet branch map: 251 detunings x 201 drives, ~20 min on 70 cores
scripts/floquet_branch_map.py results/drag_curve_1MHz_2026-10-23/floquet --levels 9
scripts/floquet_branch_map.py results/drag_curve_1MHz_2026-10-23/floquet --levels 12 --deltas=-250:250:51
# hybridisation, |eta|*, Stueckelberg period (reads the scan's calibrated pulses)
scripts/analyze_floquet_map.py results/drag_curve_1MHz_2026-10-23/floquet results/drag_curve_1MHz_2026-10-23 9 12
# exact replays of the calibrated pulses
scripts/replay_leak_trajectories.py results/drag_curve_1MHz_2026-10-23/analysis/replay_features.json 36,41,48,52,200,-240,-130
# figures
scripts/plot_floquet_tracking.py results/drag_curve_1MHz_2026-10-23/floquet results/drag_curve_1MHz_2026-10-23 docs/floquet-analysis
scripts/plot_wide_curve.py results/drag_curve_1MHz_2026-10-23 docs/floquet-analysis/infidelity_vs_detuning
```

Run each with `.venv/bin/python` and `OMP_NUM_THREADS=1`.

## References

The methods are standard; these are the background sources for them. The device
model, operators and frame come from [1]; the rest supports the Floquet and
interference arguments.

**Device and model**

1. C. Zhou, *Quantum Operations with Charge-pumped Parametric Interactions*, PhD
   thesis, University of Pittsburgh (2023), Ch. 2. The dressed-mode SNAIL coupler
   model implemented in `zhou_coupler.py`.
2. N. E. Frattini, U. Vool, S. Shankar, A. Narla, K. M. Sliwa and M. H. Devoret,
   "3-wave mixing Josephson dipole element," Appl. Phys. Lett. **110**, 222603 (2017).
   The SNAIL and its g₃ nonlinearity.

**Floquet theory**

3. J. H. Shirley, "Solution of the Schrödinger equation with a Hamiltonian periodic in
   time," Phys. Rev. **138**, B979 (1965). Quasienergies and the Floquet picture.
4. H. Sambe, "Steady states and quasienergies of a quantum-mechanical system in an
   oscillating field," Phys. Rev. A **7**, 2203 (1973). The extended (Sambe) space in
   which multiphoton resonances are level crossings.
5. M. Grifoni and P. Hänggi, "Driven quantum tunneling," Phys. Rep. **304**, 229
   (1998). Review of Floquet methods, quasienergy crossings and avoided crossings.
6. D. W. Hone, R. Ketzmerick and W. Kohn, "Time-dependent Floquet theory and
   absence of an adiabatic limit," Phys. Rev. A **56**, 4045 (1997). Adiabatic
   following of Floquet states under a slowly varying drive, and where it fails.

**Numerics**

7. S. Blanes, F. Casas, J. A. Oteo and J. Ros, "The Magnus expansion and some of its
   applications," Phys. Rep. **470**, 151 (2009). The exponential-midpoint step used
   for U(T).
8. H. W. Kuhn, "The Hungarian method for the assignment problem," Naval Res. Logist.
   Q. **2**, 83 (1955); J. Munkres, "Algorithms for the assignment and transportation
   problems," J. Soc. Ind. Appl. Math. **5**, 32 (1957). Branch labelling by
   maximum-overlap assignment.
9. G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed. (Johns Hopkins
   University Press, 2013). The Schur decomposition of U(T).
10. P. Virtanen *et al.*, "SciPy 1.0: fundamental algorithms for scientific computing
    in Python," Nat. Methods **17**, 261 (2020). `scipy.linalg.schur` and
    `scipy.optimize.linear_sum_assignment`.

**Driven Josephson circuits beyond the RWA**

11. R. Lescanne, L. Verney, Q. Ficheux, M. H. Devoret, B. Huard, M. Mirrahimi and
    Z. Leghtas, "Escape of a driven quantum Josephson circuit into unconfined states,"
    Phys. Rev. Applied **11**, 014030 (2019).
12. L. Verney, R. Lescanne, M. H. Devoret, Z. Leghtas and M. Mirrahimi, "Structural
    instability of driven Josephson circuits prompted by counter-rotating terms,"
    Phys. Rev. Applied **11**, 024003 (2019).

**Landau–Zener–Stückelberg interference**

13. L. D. Landau, "Zur Theorie der Energieübertragung. II," Phys. Z. Sowjetunion **2**,
    46 (1932).
14. C. Zener, "Non-adiabatic crossing of energy levels," Proc. R. Soc. Lond. A **137**,
    696 (1932).
15. E. C. G. Stückelberg, "Theorie der unelastischen Stösse zwischen Atomen," Helv.
    Phys. Acta **5**, 369 (1932).
16. E. Majorana, "Atomi orientati in campo magnetico variabile," Nuovo Cimento **9**,
    43 (1932).
17. S. N. Shevchenko, S. Ashhab and F. Nori, "Landau–Zener–Stückelberg
    interferometry," Phys. Rep. **492**, 1 (2010). Two-passage interference and its
    phase, the basis of the 1/(3T) period.
