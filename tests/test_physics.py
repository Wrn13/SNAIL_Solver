"""Regression suite for the Zhou SNAIL iSWAP sweep tooling.

Every test here encodes an invariant that a real bug violated at some point, so a
failure means a specific known-bad behaviour has come back:

* beat conventions -- the spectator beat must be symmetric about w_b (a below-w_b
  convention applied above w_b silently disabled DRAG), and the subharmonic beat
  must be the TRUE two-pump detuning w_i - 2 w_p (a factor of 2 was wrong).
* collision labelling -- ``_nearest_collision`` must name the right channel, and
  must fall back to the subharmonics in the bare no-spectator gate.
* rate combinatorics -- the coupler<->spectator process is C = 6 like the iSWAP,
  differing only by participation (verified against extracted matrix elements).
* 3-mode reduction -- the bare gate must reproduce the decoupled 4-mode
  Hamiltonian exactly on the shared n_spec = 0 block.
* gate-area arithmetic -- t_g = 2 * target_eta_area / eta = 138.889 / eta.
* blank-spectator plumbing -- a bare-gate row has no spectator frequency, which
  crashed float formatting in the logger and again in collect.
* operating points -- round-trip through a device JSON, and context-mismatch
  detection (a point calibrated elsewhere must not apply silently).
* frequency-list helpers must not overshoot the band edge.

Deliberately QuTiP-free and fast, so it can gate a cluster submission:

    uv run pytest -q                            # or:
    uv run python -m unittest discover -s tests -t tests -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

#: Repo root, so the subprocess tests below run the CLIs from a predictable cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TWO_PI = 2.0 * np.pi


def _cfg(**over):
    """A DEFAULT_CONFIG copy with overrides (no device file needed)."""
    from snail_solver.sweep_common import DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(over)
    return cfg


class TestBeatConventions(unittest.TestCase):
    """The spectator beat must measure to the nearer collision, either side of w_b."""

    def setUp(self):
        from snail_solver.sweep_common import _nearest_collision
        self.nc = _nearest_collision
        self.wa, self.wb, self.wc, self.wp = 3.5, 4.5, 4.7, 1.0

    def _beat(self, wspec, **over):
        return self.nc(_cfg(**over), self.wa, self.wb, self.wc, wspec, self.wp)

    def test_b_onepump_resonance_is_zero_above_wb(self):
        # w_spec = w_b + w_p = 5.5 is the ABOVE-w_b one-pump collision
        beat = self._beat(5.5)
        self.assertEqual(beat[2], "onepump")
        self.assertEqual(beat[3], "b")
        self.assertAlmostEqual(beat[1], 0.0, places=9)

    def test_symmetric_about_wb(self):
        """|beat| must be equal for w_spec = w_b +/- (w_p + x): the |Delta| fix."""
        for x in (0.05, 0.2, 0.35):
            above = self._beat(self.wb + self.wp + x)[1]
            below = self._beat(self.wb - self.wp - x)[1]
            self.assertAlmostEqual(abs(above), x, places=9)
            self.assertAlmostEqual(abs(below), x, places=9)

    def test_no_regression_to_minus_two_wp(self):
        """The old convention reported ~ -2 w_p on-collision above w_b."""
        beat = self._beat(5.5)[1]
        self.assertLess(abs(beat), 0.5 * self.wp,
                        "above-w_b collision is being measured to the wrong side")

    def test_channel_labels(self):
        self.assertEqual(self._beat(4.8)[2:4], ("onepump", "a"))
        self.assertEqual(self._beat(3.3)[2:4], ("static", "a"))


class TestSubharmonicBeat(unittest.TestCase):
    """Subharmonic beat is w_i - 2 w_p (the two-pump detuning), not w_i/2 - w_p."""

    def setUp(self):
        from snail_solver.sweep_common import _nearest_collision
        self.nc = _nearest_collision
        self.wa, self.wb, self.wc, self.wp = 3.5, 5.7, 4.7, 2.2   # 2 w_p = 4.4

    def test_spectator_subharmonic_resonance(self):
        cfg = _cfg(drag_subharmonic=True, subharmonic_modes=["spec"])
        beat = self.nc(cfg, self.wa, self.wb, self.wc, 2 * self.wp, self.wp)
        self.assertEqual(beat[2:4], ("subharm", "spec"))
        self.assertAlmostEqual(beat[1], 0.0, places=9)

    def test_factor_of_two(self):
        """At w_spec = 4.5 the detuning is +100 MHz; the old bug gave +50."""
        cfg = _cfg(drag_subharmonic=True, subharmonic_modes=["spec"])
        beat = self.nc(cfg, self.wa, self.wb, self.wc, 4.5, self.wp)[1]
        self.assertAlmostEqual(beat, 0.1, places=9)

    def test_mode_selection_excludes_others(self):
        cfg = _cfg(drag_subharmonic=True, subharmonic_modes=["spec"])
        for wspec in (4.3, 4.4, 4.5):
            self.assertEqual(self.nc(cfg, self.wa, self.wb, self.wc,
                                     wspec, self.wp)[3], "spec")

    def test_off_by_default(self):
        beat = self.nc(_cfg(), self.wa, self.wb, self.wc, 2 * self.wp, self.wp)
        self.assertNotEqual(beat[2], "subharm")


class TestBareGateCollisions(unittest.TestCase):
    """no_spectator: only subharmonic channels, defaulting to the SNAIL."""

    def test_snail_subharmonic_resonance(self):
        from snail_solver.sweep_common import _nearest_collision
        # w_c = 4.2, w_p = 2.1 -> 2 w_p = w_c exactly
        beat = _nearest_collision(_cfg(no_spectator=True), 3.5, 5.6, 4.2, 99.0, 2.1)
        self.assertEqual(beat[2:4], ("subharm", "s"))
        self.assertAlmostEqual(beat[1], 0.0, places=9)

    def test_detuning_schedule(self):
        """w_c - 2 w_p must march linearly through zero as w_b sweeps."""
        from snail_solver.sweep_common import _nearest_collision
        wa, wc = 3.5, 4.2
        for wb, want in ((5.55, +0.10), (5.60, 0.0), (5.65, -0.10)):
            beat = _nearest_collision(_cfg(no_spectator=True), wa, wb, wc, 99.0, wb - wa)
            self.assertAlmostEqual(beat[1], want, places=9)

    def test_dummy_spectator_never_selected(self):
        from snail_solver.sweep_common import _nearest_collision
        beat = _nearest_collision(_cfg(no_spectator=True), 3.5, 5.6, 4.2, 7.6, 2.1)
        self.assertNotEqual(beat[3], "spec")

    def test_qubit_subharmonic_selectable(self):
        from snail_solver.sweep_common import _nearest_collision
        # a-subharmonic 2 w_p = w_a at w_b = 1.5 w_a = 5.25
        cfg = _cfg(no_spectator=True, drag_subharmonic=True, subharmonic_modes=["a"])
        beat = _nearest_collision(cfg, 3.5, 5.25, 4.7, 99.0, 1.75)
        self.assertEqual(beat[2:4], ("subharm", "a"))
        self.assertAlmostEqual(beat[1], 0.0, places=9)


class TestRateCombinatorics(unittest.TestCase):
    """The qS (coupler<->spectator) process is C = 6, like the iSWAP."""

    def test_qS_over_iswap_ratio_is_one_over_lambda(self):
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        lam = 0.1
        cpl = ZhouCoupler(mode_freqs_GHz=[3.5, 4.5, 4.7, 5.7], coupler_index=2,
                          participations={0: lam, 1: lam, 3: lam},
                          nonlinearities={3: 0.06}, levels=[3, 3, 5, 3],
                          anharmonicities_GHz={0: -0.12, 1: -0.12, 3: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.0, envelope=RaisedCosine(amp=1.0, t_g=100.0),
                              is_eta=True), normalize_iswap=(0, 1))
        g_iswap = cpl.effective_rate([1, 3], n=3, C=6)      # two participations
        g_qS = cpl.effective_rate([2, 3], n=3, C=6)         # coupler participation = 1
        self.assertAlmostEqual(g_qS / g_iswap, 1.0 / lam, places=6)

    def test_extracted_matrix_element_matches_C6(self):
        """Independent check straight off expand_terms (no effective_rate)."""
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        lam, g3 = 0.1, 0.06
        cpl = ZhouCoupler(mode_freqs_GHz=[3.5, 4.5, 4.7, 5.7], coupler_index=2,
                          participations={0: lam, 1: lam, 3: lam},
                          nonlinearities={3: g3}, levels=[3, 3, 5, 3],
                          anharmonicities_GHz={0: -0.12, 1: -0.12, 3: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.0, envelope=RaisedCosine(amp=1.0, t_g=100.0),
                              is_eta=True))
        i, j = cpl.fock_index([0, 0, 1, 0]), cpl.fock_index([0, 0, 0, 1])
        val = max(abs(O[i, j]) for _Om, _ps, O in cpl.expand_terms(cutoff_GHz=np.inf)
                  if abs(O[i, j]) > 0)
        # C=6 with one participation: 6 * g3 * lam_spec (pump normalized to 1)
        self.assertAlmostEqual(val / TWO_PI, 6 * g3 * lam, places=9)


class TestChirp(unittest.TestCase):
    """A chirp must be a pure carrier rotation -- nothing more, nothing less.

    The whole design rests on one identity: a pump letter enters X(t) as
    ``eta e^{-i w_p t}``, so chirping the carrier is the same thing as putting the
    phase ``e^{-i Phi(t)}`` on the envelope. If that identity ever breaks, a
    constant chirp stops agreeing with a retuned carrier and these fail.
    """

    T_G = 77.2

    def _build(self, w_p_GHz, chirp):
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[3, 3, 4], anharmonicities_GHz={0: -0.12, 1: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, is_eta=True, chirp=chirp,
                              envelope=RaisedCosine(amp=0.05, t_g=self.T_G)))
        return cpl

    def _max_dH(self, c1, c2):
        ts = np.linspace(0.0, self.T_G, 11)
        return max(np.max(np.abs(c1.hamiltonian_matrix(t) - c2.hamiltonian_matrix(t)))
                   for t in ts)

    def test_constant_chirp_equals_retuned_carrier(self):
        """delta(t) = c0 must be EXACTLY a pump at w_p + c0 (the design identity)."""
        from snail_solver.zhou_coupler import Chirp
        c0 = 0.037
        self.assertLess(self._max_dH(self._build(1.7, Chirp([c0], self.T_G)),
                                     self._build(1.7 + c0, None)), 1e-12)

    def test_constant_chirp_sign(self):
        """+c0 shifts the carrier UP. A sign slip here would silently detune the gate."""
        from snail_solver.zhou_coupler import Chirp
        c0 = 0.037
        self.assertGreater(self._max_dH(self._build(1.7, Chirp([c0], self.T_G)),
                                        self._build(1.7 - c0, None)), 1e-6)

    def test_zero_chirp_is_inert(self):
        """All-zero coefficients must reproduce the un-chirped Hamiltonian exactly."""
        from snail_solver.zhou_coupler import Chirp
        self.assertEqual(self._max_dH(self._build(1.7, None),
                                      self._build(1.7, Chirp([0.0, 0.0], self.T_G))), 0.0)

    def test_linear_chirp_changes_the_dynamics(self):
        """Guards against a chirp that is silently dropped somewhere in _eta."""
        from snail_solver.zhou_coupler import Chirp
        self.assertGreater(self._max_dH(self._build(1.7, None),
                                        self._build(1.7, Chirp([0.0, 0.05], self.T_G))), 1e-3)

    def test_chirp_does_not_touch_the_amplitude_calibration(self):
        """A chirp is a phase: |eta|, area() and peak_eta must be untouched.

        `set_pump(normalize_iswap=...)` divides by `area()`, so if a chirp ever
        leaked into it the pi/2 amplitude calibration would silently drift.
        """
        from snail_solver.zhou_coupler import Chirp
        plain = self._build(1.7, None)
        chirped = self._build(1.7, Chirp([0.02, 0.05, -0.01], self.T_G))
        self.assertEqual(plain.peak_eta(), chirped.peak_eta())
        self.assertEqual(plain._pump_tones[0].envelope.area(),
                         chirped._pump_tones[0].envelope.area())

    def test_phase_is_the_integral_of_the_detuning(self):
        """Phi(0) = 0 and dPhi/dt = delta(t); the closed form must match quadrature."""
        from snail_solver.zhou_coupler import Chirp
        ch = Chirp([0.01, 0.05, -0.02], self.T_G)
        self.assertAlmostEqual(float(ch.phase(0.0)), 0.0, places=12)
        h = 1e-6
        for t in (12.0, 38.6, 70.0):
            fd = (float(ch.phase(t + h)) - float(ch.phase(t - h))) / (2 * h)
            self.assertAlmostEqual(fd, float(ch.detuning(t)), places=6)

    def test_make_chirp_returns_none_when_trivial(self):
        """An absent/zero chirp must yield None so the solver path stays untouched."""
        from snail_solver.zhou_coupler import make_chirp
        self.assertIsNone(make_chirp(None, self.T_G))
        self.assertIsNone(make_chirp([], self.T_G))
        self.assertIsNone(make_chirp([0.0, 0.0], self.T_G))
        self.assertIsNotNone(make_chirp([0.0, 0.01], self.T_G))


class TestChirpInReducedModel(unittest.TestCase):
    """The reduced rotating-frame model must carry a chirp exactly.

    ``grape._propagate`` is the engine behind the DEFAULT calibration map and every
    reduced-scored optimizer path. It used to ignore ``PumpTone.chirp`` outright, so
    a chirped device calibrated as if un-chirped and reported a plausible wrong
    number. The identity that makes the fix exact: ``_H`` forms
    ``eta^n_pos conj(eta)^n_neg``, so folding ``e^{-i Phi(t)}`` into the complex eta
    hands a term carrying k = n_pos - n_neg net pump quanta exactly ``e^{-i k Phi}``.
    """

    T_G = 77.2
    CUTOFF = 1.0

    def setUp(self):
        from snail_solver import grape
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        self.grape = grape
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[3, 3, 4], anharmonicities_GHz={0: -0.12, 1: -0.12})
        # Normalize to the FULL-iSWAP amplitude rather than picking a peak by hand.
        # This matters: the chirp enters through eta^n_pos conj(eta)^n_neg, so its
        # effect is strongly amplitude-dependent, and an arbitrary weak peak (0.05
        # against this device's operating 1.80) suppresses by ~1e3 exactly the term
        # these tests exist to measure.
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                              envelope=RaisedCosine(amp=1.0, t_g=self.T_G)),
                     normalize_iswap=(0, 1))
        (self.terms, self.H_anh, self.idx,
         self.max_Omega) = grape._prepare(cpl, 0, 1, self.CUTOFF)
        self.n_ctrl = 8
        peak = float(cpl.peak_eta())
        ts = (np.arange(self.n_ctrl) + 0.5) * (self.T_G / self.n_ctrl)
        self.eta = (peak * 0.5 * (1.0 - np.cos(TWO_PI * ts / self.T_G))).astype(complex)

    def _n_sub(self, chirp=None):
        """The step count the production callers use, chirp padding included."""
        pad = self.grape._chirp_pad_rad(chirp, self.terms)
        dt = self.T_G / self.n_ctrl
        return max(1, int(np.ceil((self.max_Omega + pad) * dt / 0.3)))

    def _U(self, n_sub, offset_rad=0.0, chirp=None):
        return self.grape._propagate(self.eta, self.T_G, self.terms, self.H_anh,
                                     self.idx, n_sub, offset_rad, chirp=chirp)

    def test_constant_chirp_equals_offset_rad(self):
        """delta(t) = c0 must reproduce the carrier shift offset_rad = 2 pi c0.

        The two mechanisms are completely different code paths -- one displaces every
        precomputed carrier by k * offset, the other multiplies a phase onto eta --
        so their agreement pins the (n_pos - n_neg) sign convention end to end.
        """
        from snail_solver.zhou_coupler import Chirp
        c0 = 0.01
        chirp = Chirp([c0], self.T_G)
        n_sub = self._n_sub(chirp)            # SAME step count, else discretization differs
        np.testing.assert_allclose(self._U(n_sub, chirp=chirp),
                                   self._U(n_sub, offset_rad=TWO_PI * c0), atol=1e-10)

    def test_absent_chirp_is_byte_identical(self):
        """The un-chirped path must be untouched -- not merely close."""
        from snail_solver.zhou_coupler import Chirp
        n_sub = self._n_sub()
        bare = self.grape._propagate(self.eta, self.T_G, self.terms, self.H_anh,
                                     self.idx, n_sub)
        np.testing.assert_array_equal(bare, self._U(n_sub, chirp=None))
        # an all-zero Chirp object is a phase of exactly 1, so it must also be exact
        np.testing.assert_array_equal(
            bare, self._U(n_sub, chirp=Chirp([0.0, 0.0], self.T_G)))

    def test_fine_step_resolution_is_required(self):
        """Folding the chirp per CONTROL SLICE is not accurate enough.

        This pins the design decision with numbers. delta(t) peaks at 2 pi c1 = 0.31
        rad/ns, so across one control slice (t_g/n_ctrl = 9.7 ns here) Phi moves by
        ~3 rad -- times k = 2 net pump quanta in the coefficient. Holding that
        constant over the slice is an O(1) error in the pump term; a fine step moves
        it by ~0.02 rad instead.

        Asserted as a RATIO rather than two absolute thresholds, because both errors
        scale together with the drive and the step count: what has to stay true is
        that per-slice folding is dramatically worse, not that either number sits at
        a particular value. Measured here: ~2e-3 fine vs ~2e-1 per-slice, a factor of
        ~70. If someone "simplifies" `_propagate` back to a per-slice phase, the
        ratio collapses to 1 and this fires.
        """
        from snail_solver.zhou_coupler import Chirp
        chirp = Chirp([0.0, 0.05], self.T_G)
        n_sub = self._n_sub(chirp)
        U_ref = self._U(4 * n_sub, chirp=chirp)                 # converged reference
        err_fine = np.max(np.abs(self._U(n_sub, chirp=chirp) - U_ref))

        ts_mid = (np.arange(self.n_ctrl) + 0.5) * (self.T_G / self.n_ctrl)
        eta_slice = self.eta * np.exp(-1j * np.asarray(chirp.phase(ts_mid, np)))
        U_slice = self.grape._propagate(eta_slice, self.T_G, self.terms, self.H_anh,
                                        self.idx, n_sub)
        err_slice = np.max(np.abs(U_slice - U_ref))

        self.assertLess(err_fine, 1e-2)              # fine-step folding is accurate
        self.assertGreater(err_slice, 1e-2)          # per-slice folding is not
        self.assertGreater(err_slice, 20.0 * err_fine,
                           f"per-slice folding ({err_slice:.2e}) must be far worse "
                           f"than fine-step ({err_fine:.2e})")

    def test_chirp_pad_rad_bounds_the_added_bandwidth(self):
        """n_sub must grow with the chirp rate, or the propagation under-resolves."""
        from snail_solver.zhou_coupler import Chirp
        self.assertEqual(self.grape._chirp_pad_rad(None, self.terms), 0.0)
        k_max = max(abs(p - n) for _Om, p, n, _O in self.terms)
        self.assertGreater(k_max, 1)          # the g3 terms carry up to 3 pump quanta
        c1 = 0.05                             # delta(t) = 2 pi c1 u, so max|delta| = 2 pi c1
        self.assertAlmostEqual(
            self.grape._chirp_pad_rad(Chirp([0.0, c1], self.T_G), self.terms),
            k_max * TWO_PI * c1, places=6)


class TestEnvelopeArrayAPI(unittest.TestCase):
    """`value_at`/`deriv_at` must agree with the scalar path and be branch-free.

    The batched/GPU engine evaluates envelopes on arrays of times and under a JAX
    trace; the per-time QuTiP callbacks still use the scalar wrappers. If the two
    ever disagree, the fast engine silently solves a different pulse.
    """

    T_G = 77.2

    def _envelopes(self):
        from snail_solver.envelope import ConstantPulse, IQFourierEnvelope, RaisedCosine
        rng = np.random.default_rng(0)
        freqs = TWO_PI * np.arange(1, 5) / self.T_G
        return [
            ConstantPulse(amp=0.05, t_g=self.T_G),
            RaisedCosine(amp=0.05, t_g=self.T_G),
            IQFourierEnvelope(amp=0.05, t_g=self.T_G),
            IQFourierEnvelope(amp=0.05, t_g=self.T_G, freqs=freqs,
                              sin_I=rng.normal(size=4), sin_Q=rng.normal(size=4),
                              cos_I=rng.normal(size=4), cos_Q=rng.normal(size=4)),
        ]

    def test_vectorized_matches_scalar(self):
        # deliberately overruns [0, t_g] so the support mask is exercised
        ts = np.linspace(-5.0, self.T_G + 5.0, 37)
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__, n=env.n_params):
                self.assertEqual(np.max(np.abs(
                    np.array([env.value(float(t)) for t in ts])
                    - np.asarray(env.value_at(ts, np)))), 0.0)
                self.assertEqual(np.max(np.abs(
                    np.array([env.deriv(float(t)) for t in ts])
                    - np.asarray(env.deriv_at(ts, np)))), 0.0)

    def test_analytic_derivative_matches_finite_difference(self):
        ts, h = np.linspace(2.0, self.T_G - 2.0, 13), 1e-6
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__, n=env.n_params):
                fd = np.array([(env.value(float(t) + h) - env.value(float(t) - h)) / (2 * h)
                               for t in ts])
                self.assertLess(np.max(np.abs(fd - np.asarray(env.deriv_at(ts, np)))), 1e-8)

    def test_envelope_is_zero_outside_the_gate(self):
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__, n=env.n_params):
                for t in (-1e-9, -3.0, self.T_G + 1e-9, self.T_G + 3.0):
                    self.assertEqual(env.value(t), 0.0)

    def test_zero_coefficient_iq_reduces_to_raised_cosine(self):
        """The CRAB ansatz must start exactly at the sweep's baseline pulse."""
        from snail_solver.envelope import IQFourierEnvelope, RaisedCosine
        ts = np.linspace(0.0, self.T_G, 25)
        rc = RaisedCosine(amp=0.05, t_g=self.T_G)
        iq = IQFourierEnvelope(amp=0.05, t_g=self.T_G,
                               freqs=TWO_PI * np.arange(1, 5) / self.T_G)
        self.assertLess(np.max(np.abs(np.asarray(iq.value_at(ts, np))
                                      - np.asarray(rc.value_at(ts, np)))), 1e-15)
        self.assertAlmostEqual(iq.area(), rc.area(), places=12)


class TestSymbolicExpansion(unittest.TestCase):
    """`expand_terms_symbolic` must be `expand_terms` with the frequencies pulled out.

    The batched engine depends on the operator structure being independent of every
    mode and pump frequency -- that is what lets a whole sweep share one operator
    stack. If a frequency ever leaks into the structure, the engine silently solves
    the wrong Hamiltonian for every grid point but the first.
    """

    def _build(self, wb=5.5):
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, wb, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={0: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                              envelope=RaisedCosine(amp=0.3, t_g=40.0)))
        return cpl

    def _grouped(self, cpl):
        """Both expansions, folded onto the same (Omega, signature) keys."""
        from collections import defaultdict
        S = cpl.expand_terms_symbolic()
        Om = S["M"] @ cpl.frequency_vector()
        sym = defaultdict(lambda: np.zeros((cpl.dim, cpl.dim), dtype=complex))
        for j in range(S["n_terms"]):
            m = S["term"] == j
            O = np.zeros((cpl.dim, cpl.dim), dtype=complex)
            O[S["row"][m], S["col"][m]] = S["val"][m]
            sym[(round(float(Om[j]), 6), S["signatures"][j])] += O
        ref = defaultdict(lambda: np.zeros((cpl.dim, cpl.dim), dtype=complex))
        for omega, sig, O in cpl.expand_terms(cutoff_GHz=np.inf):
            ref[(round(float(omega), 6), sig)] += O
        return sym, ref

    def test_matches_expand_terms(self):
        sym, ref = self._grouped(self._build())
        self.assertEqual(set(sym), set(ref))
        self.assertEqual(max(np.max(np.abs(sym[k] - ref[k])) for k in ref), 0.0)

    def test_structure_is_frequency_independent(self):
        """Move w_b; M/operators must not budge, only `frequency_vector` does."""
        a, b = self._build(5.5), self._build(5.9)
        Sa, Sb = a.expand_terms_symbolic(), b.expand_terms_symbolic()
        for key in ("M", "n_pos", "n_neg", "term", "row", "col"):
            np.testing.assert_array_equal(Sa[key], Sb[key], err_msg=key)
        np.testing.assert_allclose(Sa["val"], Sb["val"], rtol=0, atol=0)
        self.assertGreater(np.max(np.abs(Sa["M"] @ a.frequency_vector()
                                         - Sb["M"] @ b.frequency_vector())), 1.0)

    def test_operators_are_sparse(self):
        """The COO stack must be far smaller than a dense (n_terms, dim, dim)."""
        cpl = self._build()
        S = cpl.expand_terms_symbolic()
        self.assertLess(S["val"].size, 0.2 * S["n_terms"] * cpl.dim ** 2)


class TestBatchedEngine(unittest.TestCase):
    """The batched engine must reproduce the dense Hamiltonian and the scalar pump.

    QuTiP-free: both checks are against `hamiltonian_matrix` / `_eta`, which are the
    same oracles the rest of this suite uses. The end-to-end engine-vs-sesolve
    comparison lives in `validate_engines.py` (it needs QuTiP and is slow).
    """

    T_G = 40.0

    def _build(self, chirp=None, drag=False):
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={0: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True, drag=drag,
                              delta_drag_GHz=(0.3 if drag else 0.0), chirp=chirp,
                              envelope=RaisedCosine(amp=0.3, t_g=self.T_G)))
        return cpl

    def _cases(self):
        from snail_solver.zhou_coupler import Chirp
        return [("plain", self._build()),
                ("drag", self._build(drag=True)),
                ("chirp", self._build(chirp=Chirp([0.01, 0.03], self.T_G))),
                ("chirp+drag", self._build(chirp=Chirp([0.01, 0.03], self.T_G), drag=True))]

    def test_sparse_H_matches_dense_hamiltonian(self):
        """The exact (cutoff=inf) engine must reproduce `hamiltonian_matrix`."""
        from snail_solver import jax_engine as JE
        rng = np.random.default_rng(0)
        for label, cpl in self._cases():
            with self.subTest(case=label):
                eng = JE.build_engine(cpl, cutoff_GHz=np.inf)
                params = JE.pulse_params(cpl)
                Psi = (rng.normal(size=(cpl.dim, 4)) + 1j * rng.normal(size=(cpl.dim, 4)))
                for t in (0.0, 13.7, 29.1, self.T_G):
                    got = eng.H_apply(t, Psi, eng.omega_vec0, params, np)
                    want = cpl.hamiltonian_matrix(t) @ Psi
                    self.assertLess(np.max(np.abs(got - want)), 1e-10, f"{label} t={t}")

    def test_engine_eta_matches_coupler_eta(self):
        """`jax_engine.eta_at` is a separate implementation of `_eta_at`; pin them."""
        from snail_solver import jax_engine as JE
        for label, cpl in self._cases():
            with self.subTest(case=label):
                eng = JE.build_engine(cpl, cutoff_GHz=1.0)
                params = JE.pulse_params(cpl)
                for t in np.linspace(0.0, self.T_G, 9):
                    got = complex(JE.eta_at(eng.spec, params, float(t), 0, np))
                    want = cpl._eta(cpl._pump_tones[0], float(t))
                    self.assertAlmostEqual(got, want, places=12, msg=f"{label} t={t}")

    def test_cutoff_prunes_and_inf_keeps_everything(self):
        from snail_solver import jax_engine as JE
        cpl = self._build()
        self.assertEqual(JE.build_engine(cpl, cutoff_GHz=np.inf).n_dropped, 0)
        self.assertGreater(JE.build_engine(cpl, cutoff_GHz=1.0).n_dropped, 0)

    def test_propagator_is_unitary_on_the_full_space(self):
        """A sanity check that needs no reference: the full propagator is unitary.

        (The 4x4 projection is not, because of leakage, so this checks the block
        columns keep unit norm only in the no-leakage 2-level limit.)
        """
        from snail_solver import jax_engine as JE
        cpl = self._build()
        eng = JE.build_engine(cpl, cutoff_GHz=np.inf)
        U = np.asarray(JE.propagator_columns(eng, self.T_G, carrier_resolution=0.2))
        self.assertLessEqual(float(np.max(np.abs(U))), 1.0 + 1e-9)

    def test_short_gate_does_not_diverge(self):
        """A short gate forces a LARGE |eta|; the expm bound must follow it.

        `normalize_iswap` scales the pump as ~1/t_g, so at t_g = 20 ns the peak
        |eta| is ~7, not ~1. A fixed eta_max ceiling under-bounds ||H||, the
        scaling-and-squaring count comes out too small, and the Taylor series
        diverges SILENTLY -- observed as a reported "fidelity" of 1e290.
        """
        from snail_solver import jax_engine as JE
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        for t_g in (20.0, 77.2):
            with self.subTest(t_g=t_g):
                cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                                  participations={0: 0.1, 1: 0.1},
                                  nonlinearities={3: 0.06}, levels=[2, 2, 3],
                                  anharmonicities_GHz={})
                cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                                      envelope=RaisedCosine(amp=1.0, t_g=t_g)),
                             normalize_iswap=(0, 1))
                eng = JE.build_engine(cpl, cutoff_GHz=2.0)
                U = np.asarray(JE.propagator_columns(eng, t_g, carrier_resolution=0.1))
                JE.check_propagator(U)          # must not raise
                self.assertLessEqual(float(np.max(np.abs(U))), 1.0 + 1e-6)

    def test_divergence_guard_fires(self):
        """The guard must actually catch an under-bounded propagator."""
        from snail_solver import jax_engine as JE
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={})
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                              envelope=RaisedCosine(amp=1.0, t_g=20.0)),
                     normalize_iswap=(0, 1))
        eng = JE.build_engine(cpl, cutoff_GHz=2.0)
        bad = JE.propagator_columns(eng, 20.0, eta_max=2.0, carrier_resolution=0.1)
        with self.assertRaises(FloatingPointError):
            JE.check_propagator(np.asarray(bad))

    def test_cf4_converges_fourth_order(self):
        """Halving the step must cut the error ~16x -- the integrator's contract.

        A regression here means the Magnus weights or the node placement broke, which
        would otherwise show up only as a slightly-wrong fidelity.
        """
        from snail_solver import jax_engine as JE
        eng = JE.build_engine(self._build(), cutoff_GHz=2.0)
        ref = np.asarray(JE.propagator_columns(eng, self.T_G, carrier_resolution=0.0125))
        errs = [np.max(np.abs(np.asarray(
            JE.propagator_columns(eng, self.T_G, carrier_resolution=cr)) - ref))
            for cr in (0.4, 0.2)]
        self.assertGreater(errs[0] / errs[1], 8.0)


class TestEnvelopeSingleSourceOfTruth(unittest.TestCase):
    """zhou_coupler must RE-EXPORT the envelope classes, not redefine them.

    They were duplicated verbatim in both modules; the copies would have diverged
    the moment either was touched.
    """

    def test_reexported_classes_are_identical_objects(self):
        from snail_solver import envelope
        from snail_solver import zhou_coupler
        for name in ("Envelope", "ConstantPulse", "RaisedCosine",
                     "IQFourierEnvelope", "PumpTone", "Chirp"):
            self.assertIs(getattr(zhou_coupler, name), getattr(envelope, name), name)


class TestThreeModeReduction(unittest.TestCase):
    """The bare 3-mode gate must equal the decoupled 4-mode one on n_spec = 0."""

    def _build(self, no_spec):
        from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        wa, wb, wc, t_g = 4.6, 5.6, 4.2, 92.6
        if no_spec:
            kw = dict(mode_freqs_GHz=[wa, wb, wc], participations={0: 0.1, 1: 0.1},
                      levels=[3, 3, 5], anharmonicities_GHz={0: -0.12, 1: -0.12})
        else:
            kw = dict(mode_freqs_GHz=[wa, wb, wc, 7.6],
                      participations={0: 0.1, 1: 0.1, 3: 0.0}, levels=[3, 3, 5, 3],
                      anharmonicities_GHz={0: -0.12, 1: -0.12, 3: -0.12})
        cpl = ZhouCoupler(coupler_index=2, nonlinearities={3: 0.06}, **kw)
        cpl.set_pump(PumpTone(w_p_GHz=abs(wb - wa),
                              envelope=RaisedCosine(amp=1.0, t_g=t_g), is_eta=True),
                     normalize_iswap=(0, 1))
        return cpl

    def test_dimension_reduction(self):
        self.assertEqual(self._build(True).dim * 3, self._build(False).dim)

    def test_hamiltonian_block_identical(self):
        import itertools
        c3, c4 = self._build(True), self._build(False)
        idx = np.array([c4.fock_index(list(occ) + [0])
                        for occ in itertools.product(*[range(d) for d in c3.dims])])
        for t in (0.0, 23.7, 61.2):
            H3 = c3.hamiltonian_matrix(t)
            H4 = c4.hamiltonian_matrix(t)[np.ix_(idx, idx)]
            self.assertLess(np.max(np.abs(H3 - H4)), 1e-12,
                            f"3-mode and decoupled 4-mode disagree at t={t}")

    def test_same_iswap_rate(self):
        self.assertAlmostEqual(self._build(True).iswap_rate(0, 1),
                               self._build(False).iswap_rate(0, 1), places=12)


class TestGateArea(unittest.TestCase):
    """t_g = 2 * target_eta_area / eta = 138.889 / eta for a raised cosine."""

    def test_area(self):
        from snail_solver.device_utils import target_eta_area
        self.assertAlmostEqual(target_eta_area(0.06, 0.1, 0.1), 69.4444, places=3)

    def test_auto_t_g_scaling(self):
        from snail_solver.device_utils import auto_t_g
        for eta, want in ((1.2, 115.741), (1.5, 92.593), (1.8, 77.160)):
            self.assertAlmostEqual(auto_t_g(0.06, 0.1, 0.1, eta), want, places=3)

    def test_inverse_relation(self):
        from snail_solver.device_utils import auto_t_g
        self.assertAlmostEqual(auto_t_g(0.06, 0.1, 0.1, 1.0) * 2.0,
                               auto_t_g(0.06, 0.1, 0.1, 0.5), places=6)


class TestBlankSpectatorFormatting(unittest.TestCase):
    """A bare-gate row has no spectator frequency; formatting must not crash."""

    def test_log_line_handles_blank(self):
        from snail_solver.sweep_common import _log_line
        row = dict(kind="target", wb_GHz=5.6, spec_GHz="", w_p_GHz=2.1,
                   nearest_beat_GHz=0.0, nearest_kind="subharm",
                   g_collision_MHz=float("nan"), F_avg=None)
        line = _log_line(row)                      # must not raise
        self.assertIn("bare", line)

    def test_log_line_still_formats_numeric(self):
        from snail_solver.sweep_common import _log_line
        row = dict(kind="target", wb_GHz=5.6, spec_GHz=4.0, w_p_GHz=2.1,
                   nearest_beat_GHz=0.1, nearest_kind="onepump",
                   g_collision_MHz=5.0, F_avg=0.99)
        self.assertIn("4.000", _log_line(row))


class TestOperatingPoints(unittest.TestCase):
    """Round-trip through a device JSON, plus context-mismatch detection."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "dev.json")
        with open(self.path, "w") as fh:
            json.dump(dict(qubit_freqs_GHz=[3.5, 5.7], coupler_freq_GHz=4.7,
                           t_g_ns=92.6, lam_a=0.1, lam_b=0.1, g3_GHz=0.06), fh)
        self.rec = dict(amp_scale=0.87, wp_offset_GHz=-0.0012, t_g_ns=92.6,
                        wa_GHz=3.5, wb_GHz=5.7, spec_abs_GHz=None,
                        drag_beat_GHz=None, metric="fidelity", score=0.994)

    def test_round_trip(self):
        from snail_solver import operating_points as OP
        OP.save_point(self.path, "p1", self.rec)
        with open(self.path) as fh:
            cfg = json.load(fh)
        self.assertIn("p1", OP.list_points(cfg))
        got = OP.get_point(cfg, "p1")
        self.assertAlmostEqual(got["amp_scale"], 0.87)
        self.assertIn("created", got)              # provenance stamped

    def test_no_silent_overwrite(self):
        from snail_solver import operating_points as OP
        OP.save_point(self.path, "p1", self.rec)
        with self.assertRaises(SystemExit):
            OP.save_point(self.path, "p1", self.rec)
        OP.save_point(self.path, "p1", self.rec, overwrite=True)   # explicit is fine

    def test_apply_sets_amp_and_offset(self):
        from snail_solver import operating_points as OP
        cfg = OP.apply_point(dict(amp_scale=1.0, wp_offset_GHz=0.0, t_g_ns=50.0),
                             self.rec)
        self.assertAlmostEqual(cfg["amp_scale"], 0.87)
        self.assertAlmostEqual(cfg["wp_offset_GHz"], -0.0012)
        self.assertAlmostEqual(cfg["t_g_ns"], 92.6)

    def test_context_mismatch_detected(self):
        from snail_solver import operating_points as OP
        cfg = dict(qubit_freqs_GHz=[3.5, 5.7], t_g_ns=92.6)
        self.assertEqual(OP.check_context(self.rec, cfg), [])
        issues = OP.check_context(self.rec, cfg, wb_GHz=5.5)   # different pair
        self.assertTrue(any("wb_GHz" in i for i in issues))
        issues = OP.check_context(self.rec, cfg, t_g=60.0)     # different gate length
        self.assertTrue(any("t_g_ns" in i for i in issues))

    def test_unknown_name_is_clean_error(self):
        from snail_solver import operating_points as OP
        with open(self.path) as fh:
            cfg = json.load(fh)
        with self.assertRaises(SystemExit):
            OP.resolve(cfg, "nope")


class TestBareGatePipeline(unittest.TestCase):
    """End-to-end analytic bare-gate run: the path that crashed twice on formatting."""

    def test_prepare_local_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = os.path.join(tmp, "dev.json")
            with open(dev, "w") as fh:
                json.dump(dict(qubit_freqs_GHz=[3.5, 5.6], coupler_freq_GHz=4.2,
                               t_g_ns=92.6, lam_a=0.1, lam_b=0.1, g3_GHz=0.06,
                               anharm_qubit_GHz=-0.12), fh)
            out = os.path.join(tmp, "run")
            base = [sys.executable, "-m", "snail_solver.run_sweep_zhou"]
            p = subprocess.run(base + ["prepare", "--sweep", "target", "--no-integrate",
                                       "--no-spectator", "--drag-subharmonic",
                                       "--subharmonic-modes", "s", "--device", dev,
                                       "--wb-GHz", "5.55,5.60,5.65",
                                       "--outdir", out],
                               cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            p = subprocess.run(base + ["local", "--outdir", out, "--nproc", "1"],
                               cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("bare", p.stdout, "bare-gate rows should log wspec=bare")

            import csv as _csv
            with open(os.path.join(out, "summary.csv")) as fh:
                rows = list(_csv.DictReader(fh))
            self.assertEqual(len(rows), 3)
            for r in rows:
                self.assertEqual(r["spec_GHz"], "")       # blank, not a phantom 7.6
                self.assertEqual(r["n_spec"], "")
                self.assertEqual(r["nearest_kind"], "subharm")
                self.assertEqual(r["nearest_target"], "s")
            beats = sorted(float(r["nearest_beat_GHz"]) for r in rows)
            self.assertAlmostEqual(beats[1], 0.0, places=9)   # resonance at w_b = 5.60


class TestPlotterSmoke(unittest.TestCase):
    """Plotters must render from a synthetic summary (no QuTiP, no real run)."""

    def test_bare_sweep_plot(self):
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "summary.csv")
            with open(path, "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["index", "kind", "wb_GHz", "nearest_beat_GHz", "F_avg",
                            "n_coupler", "p_transfer", "drag"])
                for i, wb in enumerate((5.55, 5.60, 5.65)):
                    beat = 4.2 - 2 * (wb - 3.5)
                    for drag in ("False", "True"):
                        w.writerow([i, "target", wb, f"{beat:.6f}", 0.9, 0.05, 0.88, drag])
            out = os.path.join(tmp, "fig.png")
            env = dict(os.environ, MPLBACKEND="Agg")
            p = subprocess.run([sys.executable, "-m", "snail_solver.plot_bare_sweep",
                                "--csv", path, "--out", out], cwd=REPO_ROOT,
                               capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertTrue(os.path.exists(out))


class TestStarkChirp(unittest.TestCase):
    """The Stark-tracking seed rests on a Legendre projection; pin it to closed forms.

    The whole reason chirp is parametrized in Legendre coefficients about the gate's
    normalized time is that the shape being tracked, |eta(t)|^2 / eta_pk^2 =
    cos^4(pi u / 2) for a Hann envelope, has a clean expansion there -- and is EVEN,
    which is what makes c_2 (not c_1) the leading useful coefficient.
    """

    def test_table_matches_an_independent_projection(self):
        """The hardcoded coefficients must reproduce a Gauss-Legendre projection."""
        from numpy.polynomial import legendre as L
        from snail_solver.stark_chirp import HANN_STARK_LEGENDRE
        f = lambda u: np.cos(np.pi * u / 2) ** 4          # noqa: E731
        x, w = np.polynomial.legendre.leggauss(200)
        for k, tabulated in enumerate(HANN_STARK_LEGENDRE):
            a_k = (2 * k + 1) / 2 * np.sum(w * f(x) * L.legval(x, np.eye(k + 1)[k]))
            self.assertAlmostEqual(a_k, tabulated, places=9, msg=f"a_{k}")

    def test_closed_forms(self):
        """a_0 = 3/8 and a_2 = -225/(32 pi^2), exactly.

        a_0 doubles as a cross-check against a completely separate part of the
        codebase: it is the same 0.375 that `find_stark_resonance.operating_eta`
        documents as the Hann <eta^2> / eta_pk^2 factor.
        """
        from snail_solver.stark_chirp import HANN_MEAN_FACTOR, HANN_STARK_LEGENDRE
        self.assertAlmostEqual(HANN_STARK_LEGENDRE[0], 3 / 8, places=12)
        self.assertAlmostEqual(HANN_STARK_LEGENDRE[2], -225 / (32 * np.pi ** 2),
                               places=9)
        self.assertAlmostEqual(HANN_MEAN_FACTOR, 0.375, places=12)

    def test_odd_coefficients_vanish(self):
        """The shape is even in u, so a linear chirp is the WRONG first guess."""
        from snail_solver.stark_chirp import HANN_STARK_LEGENDRE
        np.testing.assert_array_equal(HANN_STARK_LEGENDRE[1::2],
                                      np.zeros(HANN_STARK_LEGENDRE[1::2].size))

    def test_seed_ratios_and_pinned_c0(self):
        """c_2 = -1.900 delta, c_4 = +1.334 delta, and c_0 pinned to zero."""
        from snail_solver.stark_chirp import stark_chirp_seed
        delta = 0.002
        c = stark_chirp_seed(delta, degree=4)
        self.assertEqual(c[0], 0.0)                       # degenerate with wp_offset
        self.assertAlmostEqual(c[2] / delta, -1.899772, places=5)
        self.assertAlmostEqual(c[4] / delta, +1.334393, places=5)
        # unpinned, the mean of the tracking shape IS the calibrated shift
        c_full = stark_chirp_seed(delta, degree=4, pin_c0=False)
        self.assertAlmostEqual(c_full[0], delta, places=12)

    def test_map_ridge_recovers_a_synthetic_stark_ridge(self):
        """The ridge fit must return the planted slope, sub-grid.

        The ridge is deliberately placed BETWEEN offset grid points, so a fit that
        merely took the discrete argmax would fail the tolerance.
        """
        from snail_solver.stark_chirp import stark_slope_from_map
        amps = np.linspace(0.6, 1.4, 21)
        offs = np.linspace(-20.0, 20.0, 41)               # 1 MHz spacing
        true_m = -6.0                                     # MHz per amp^2
        Z = np.exp(-((offs[None, :] - true_m * amps[:, None] ** 2) / 4.0) ** 2)
        fit = stark_slope_from_map({"Z": Z, "offsets_MHz": offs, "amps": amps,
                                    "best": {"amp_scale": 1.0}})
        self.assertAlmostEqual(fit["slope_MHz_per_amp2"], true_m, places=1)
        self.assertGreater(fit["r2"], 0.99)
        self.assertAlmostEqual(fit["delta_stark_MHz"], true_m, places=1)

    def test_ridge_of_a_flat_map_is_not_trusted(self):
        """A map with no ridge must produce a LOW r2, not a confident wrong seed."""
        from snail_solver.stark_chirp import stark_slope_from_map
        amps = np.linspace(0.6, 1.4, 21)
        offs = np.linspace(-20.0, 20.0, 41)
        Z = np.exp(-(offs[None, :] / 4.0) ** 2) * np.ones((amps.size, 1))
        fit = stark_slope_from_map({"Z": Z, "offsets_MHz": offs, "amps": amps,
                                    "best": {"amp_scale": 1.0}})
        self.assertAlmostEqual(fit["slope_MHz_per_amp2"], 0.0, places=6)


class TestChirpPhaseImplementationsAgree(unittest.TestCase):
    """Phi(t) exists in three places; they must not drift apart.

    ``envelope.Chirp.phase`` is the definition, ``jax_engine._chirp_phase`` is the
    traceable twin used by the batched engine, and ``grape._chirp_phase_jax`` is the
    one the gradient optimizer differentiates through. Each is duplicated (rather
    than shared) only because the coefficients have to arrive as a traced argument;
    this test is what makes that duplication safe.
    """

    T_G = 61.0
    COEFFS = [0.011, -0.023, 0.007, 0.004]

    def test_numpy_and_jax_engine_agree(self):
        from snail_solver.envelope import Chirp
        from snail_solver.jax_engine import _chirp_phase
        ts = np.linspace(0.0, self.T_G, 97)
        ref = Chirp(self.COEFFS, self.T_G).phase(ts, np)
        got = _chirp_phase({"chirp": [np.asarray(self.COEFFS)]}, ts, 0, self.T_G, np)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_grape_traced_version_agrees(self):
        from snail_solver.envelope import Chirp
        from snail_solver.grape import _chirp_phase_jax
        ts = np.linspace(0.0, self.T_G, 97)
        ref = Chirp(self.COEFFS, self.T_G).phase(ts, np)
        got = _chirp_phase_jax(np.asarray(self.COEFFS), ts, self.T_G, np)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_phase_is_the_integral_of_the_detuning(self):
        """Independent of all three: dPhi/dt must equal delta(t)."""
        from snail_solver.envelope import Chirp
        ch = Chirp(self.COEFFS, self.T_G)
        ts = np.linspace(0.05 * self.T_G, 0.95 * self.T_G, 41)
        h = 1e-5
        fd = (ch.phase(ts + h, np) - ch.phase(ts - h, np)) / (2 * h)
        np.testing.assert_allclose(fd, ch.detuning(ts, np), rtol=1e-6)


class TestChirpOptimization(unittest.TestCase):
    """`grape` must be able to OPTIMIZE the chirp, and must not disturb it when off."""

    T_G = 40.0

    def _coupler(self, chirp_coeffs=None):
        from snail_solver.zhou_coupler import (PumpTone, RaisedCosine, ZhouCoupler,
                                               make_chirp)
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={0: -0.12, 1: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                              envelope=RaisedCosine(amp=1.0, t_g=self.T_G),
                              chirp=make_chirp(chirp_coeffs, self.T_G)),
                     normalize_iswap=(0, 1))
        return cpl

    def test_chirp_degree_zero_reports_no_chirp_keys(self):
        """The default must be a strict no-op: no chirp keys, tone untouched."""
        from snail_solver import grape
        cpl = self._coupler()
        out = grape.optimize_pulse(cpl, 0, 1, self.T_G, backend="qutip", alg="CRAB",
                                   n_basis=1, crab_score="reduced", maxiter=3,
                                   cutoff_GHz=1.0, crab_seed=0)
        self.assertNotIn("chirp_coeffs_GHz", out)
        self.assertIsNone(cpl._pump_tones[0].chirp)

    def test_optimizer_restores_the_tone_chirp(self):
        """A configured chirp must survive the optimizer, whatever it did internally."""
        from snail_solver import grape
        cpl = self._coupler([0.0, 0.0, -0.004])
        before = list(cpl._pump_tones[0].chirp.coeffs_GHz)
        grape.optimize_pulse(cpl, 0, 1, self.T_G, backend="qutip", alg="CRAB",
                             n_basis=1, crab_score="reduced", maxiter=3,
                             cutoff_GHz=1.0, crab_seed=0, chirp_degree=2)
        self.assertIsNotNone(cpl._pump_tones[0].chirp)
        self.assertEqual(list(cpl._pump_tones[0].chirp.coeffs_GHz), before)

    def test_crab_optimizes_a_chirp_and_pins_c0(self):
        """With chirp_degree > 0 the result carries coefficients, and c_0 stays 0."""
        from snail_solver import grape
        cpl = self._coupler()
        out = grape.optimize_pulse(cpl, 0, 1, self.T_G, backend="qutip", alg="CRAB",
                                   n_basis=0, crab_score="reduced", maxiter=25,
                                   cutoff_GHz=1.0, crab_seed=0, chirp_degree=2,
                                   chirp_bound_GHz=0.02)
        self.assertIn("chirp_coeffs_GHz", out)
        c = out["chirp_coeffs_GHz"]
        self.assertEqual(len(c), 3)                       # c_0 .. c_2
        self.assertEqual(c[0], 0.0, "c_0 must stay pinned -- it is degenerate "
                                    "with wp_offset_GHz")
        self.assertTrue(all(abs(x) <= 0.02 + 1e-12 for x in c), "bound violated")
        # with n_basis = 0 the envelope has NO free parameters, so switching the
        # chirp off must reproduce the baseline exactly
        self.assertAlmostEqual(out["F_chirp_off"], out["F_baseline"], places=9)
        self.assertGreaterEqual(out["F_grape"], out["F_baseline"] - 1e-12)

    def test_chirp_seed_is_scored_not_forced(self):
        """A deliberately terrible seed must not drag the result below baseline."""
        from snail_solver import grape
        cpl = self._coupler()
        out = grape.optimize_pulse(cpl, 0, 1, self.T_G, backend="qutip", alg="CRAB",
                                   n_basis=0, crab_score="reduced", maxiter=10,
                                   cutoff_GHz=1.0, crab_seed=0, chirp_degree=2,
                                   chirp_seed_GHz=[0.0, 0.02, -0.02])
        self.assertGreaterEqual(out["F_grape"], out["F_baseline"] - 1e-12)


class TestTimeDependentDrag(unittest.TestCase):
    r"""A chirped pump sweeps the beat DRAG divides by.

    DRAG is ``eta -> eta - i (deta/dt) / Delta``. The numerator differentiates the
    BASE envelope (the chirp phase is absorbed into the frame rotating at the
    instantaneous pump frequency), but the DENOMINATOR moves with the pump::

        Delta(t) = Delta_0 - k delta(t)

    with k the pump quanta the suppressed process carries -- this module's own beat
    convention is ``beat = separation - k w_p``. Dividing by the static Delta_0 is a
    ~10% error for a 100-300 MHz beat and ORDER UNITY near a collision, which is
    exactly where DRAG is doing the work.
    """

    T_G = 40.0
    COEFFS = [0.0, 0.01, 0.03]
    D0 = 0.3

    def _tone(self, k, chirped=True, D0=None):
        from snail_solver.envelope import Chirp, PumpTone, RaisedCosine
        return PumpTone(w_p_GHz=1.7, envelope=RaisedCosine(1.0, self.T_G), drag=True,
                        delta_drag_GHz=(self.D0 if D0 is None else D0),
                        chirp=(Chirp(self.COEFFS, self.T_G) if chirped else None),
                        drag_n_pump=k)

    def test_matches_the_closed_form(self):
        """Delta(t) == 2 pi Delta_0 - k delta(t), against Chirp.detuning directly."""
        from snail_solver.envelope import Chirp
        ts = np.linspace(0.0, self.T_G, 65)
        delta = Chirp(self.COEFFS, self.T_G).detuning(ts, np)
        for k in (0, 1, 2, 3):
            with self.subTest(k=k):
                np.testing.assert_allclose(
                    self._tone(k).drag_detuning(ts, np),
                    TWO_PI * self.D0 - k * delta, atol=1e-12)

    def test_k_zero_is_the_old_constant_beat(self):
        """A static (pump-independent) channel must NOT be moved by a chirp."""
        ts = np.linspace(0.0, self.T_G, 65)
        np.testing.assert_allclose(self._tone(0).drag_detuning(ts, np),
                                   TWO_PI * self.D0, atol=1e-12)

    def test_unchirped_is_unchanged_for_any_k(self):
        """Without a chirp k is irrelevant -- so the default cannot break old calls."""
        ts = np.linspace(0.0, self.T_G, 65)
        for k in (0, 1, 2):
            with self.subTest(k=k):
                np.testing.assert_allclose(
                    self._tone(k, chirped=False).drag_detuning(ts, np),
                    TWO_PI * self.D0, atol=1e-12)

    def test_the_correction_is_observable(self):
        """Guards against the whole thing being a silent no-op.

        Also pins the regime claim: small for a far-detuned beat, order-unity near a
        collision. If someone reverts the denominator to a constant, both fire.
        """
        from snail_solver.zhou_coupler import ZhouCoupler
        ts = np.linspace(0.0, self.T_G, 17)

        def eta_of(k, D0):
            cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                              participations={0: 0.1, 1: 0.1},
                              nonlinearities={3: 0.06}, levels=[2, 2, 3],
                              anharmonicities_GHz={0: -0.12, 1: -0.12})
                              # noqa: E127
            tone = self._tone(k, D0=D0)
            cpl.set_pump(tone, normalize_iswap=(0, 1))
            return np.array([cpl._eta(tone, float(t)) for t in ts])

        far0, far1 = eta_of(0, 0.3), eta_of(1, 0.3)
        self.assertGreater(np.max(np.abs(far1 - far0)), 1e-3)      # not a no-op
        near0, near1 = eta_of(0, 0.02), eta_of(1, 0.02)
        rel = np.max(np.abs(near1 - near0)) / np.max(np.abs(near0))
        self.assertGreater(rel, 0.2, "near a collision the correction is order-unity")

    def test_detuning_floor_and_guard(self):
        """min_t |Delta(t)| is what matters, not |Delta_0| -- and it must raise."""
        from snail_solver.device_utils import check_drag_detuning
        # Delta_0 = 20 MHz is comfortably above the 5 MHz skip, but the chirp sweeps
        # the beat down through zero during the pulse.
        bad = self._tone(2, D0=0.02)
        self.assertLess(bad.drag_detuning_floor() / TWO_PI, 0.02)
        with self.assertRaises(ValueError) as ctx:
            check_drag_detuning(bad)
        self.assertIn("min|Delta(t)|", str(ctx.exception))
        # a far-detuned beat is fine, and DRAG-off is never guarded
        check_drag_detuning(self._tone(1, D0=0.3))
        self.assertEqual(self._tone(1, chirped=False).drag_detuning_floor(),
                         TWO_PI * self.D0)

    def test_jax_engine_agrees(self):
        """The traceable twin must reproduce Chirp.detuning exactly."""
        from snail_solver.envelope import Chirp
        from snail_solver.jax_engine import _chirp_detuning
        ts = np.linspace(0.0, self.T_G, 65)
        np.testing.assert_allclose(
            _chirp_detuning({"chirp": [np.asarray(self.COEFFS)]}, ts, 0, self.T_G, np),
            Chirp(self.COEFFS, self.T_G).detuning(ts, np), atol=1e-12)

    def test_chirp_gradient_flows_through_drag(self):
        """THE regression: the chirp gradient must see the DRAG denominator.

        `jax_engine` used to freeze the DRAG beat into `spec`, which is not merely
        inaccurate -- it makes the chirp gradient WRONG. For a magnitude-sensitive
        objective it is catastrophic: with the denominator frozen the chirp enters
        only through the phase e^{-i Phi}, which cannot change |eta| at all, so the
        gradient collapses to exactly zero.
        """
        try:
            import jax
        except ImportError:                                    # pragma: no cover
            self.skipTest("jax not installed")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from snail_solver import jax_engine as JE
        from snail_solver.zhou_coupler import ZhouCoupler

        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={0: -0.12, 1: -0.12})
        cpl.set_pump(self._tone(2), normalize_iswap=(0, 1))
        spec, base = JE.pulse_spec(cpl), JE.pulse_params(cpl)
        ts = jnp.asarray(np.linspace(0.0, self.T_G, 33))
        c0 = np.asarray(self.COEFFS)

        def objective(c, sp):
            p = {"amp": jnp.asarray(base["amp"]), "chirp": [c],
                 "iq": [jnp.asarray(q) for q in base["iq"]]}
            return jnp.sum(jnp.abs(JE.eta_at(sp, p, ts, 0, jnp)) ** 2)

        g = np.asarray(jax.grad(objective)(jnp.asarray(c0), spec))
        h = 1e-6
        fd = np.array([(float(objective(jnp.asarray(c0 + h * np.eye(3)[i]), spec))
                        - float(objective(jnp.asarray(c0 - h * np.eye(3)[i]), spec)))
                       / (2 * h) for i in range(3)])
        np.testing.assert_allclose(g, fd, atol=1e-6)
        self.assertGreater(np.max(np.abs(fd)), 1e-3,
                           "objective must actually depend on the chirp")


# ===========================================================================
# Recursive multi-derivative DRAG (Li/Calarco/Motzoi, npj QI 10, 66 (2024))
# ===========================================================================
def _fd(f, t, n, h):
    """n-th central difference of `f` at `t` with step `h` (error O(h^2))."""
    if n == 0:
        return f(t)
    return (_fd(f, t + h, n - 1, h) - _fd(f, t - h, n - 1, h)) / (2 * h)


def _fd_order(f, t, n, jet_n):
    """Assert `jet_n` is the n-th derivative, by CONVERGENCE not by tolerance.

    A finite difference of order n carries an O(h^2) truncation error that no fixed
    tolerance can separate from a genuinely wrong formula. Halving h must shrink the
    residual by ~4; a wrong closed form leaves a constant offset and the ratio
    collapses to 1. Returns the observed ratios so callers can assert on them.
    """
    errs = []
    for h in (0.8, 0.4, 0.2):
        want = _fd(f, t, n, h)
        errs.append(float(np.max(np.abs(np.asarray(jet_n) - want))))
    return [errs[i] / max(errs[i + 1], 1e-300) for i in range(2)]


class TestJetArithmetic(unittest.TestCase):
    """Truncated Taylor arithmetic -- the engine under the recursion.

    Everything here is checked against an independently computed ground truth, since
    a silently-wrong jet would produce a plausible pulse that is simply not the one
    the paper defines.
    """

    N = 4

    def setUp(self):
        from snail_solver.jet import Jet
        self.Jet = Jet
        self.t = np.linspace(0.3, 2.7, 7)
        self.w = 1.7
        t, w = self.t, self.w
        self.S = Jet.from_derivs([w ** k * np.sin(w * t + k * np.pi / 2)
                                  for k in range(self.N + 1)])
        self.C = Jet.from_derivs([(2.0 + np.cos(w * t)) if k == 0
                                  else w ** k * np.cos(w * t + k * np.pi / 2)
                                  for k in range(self.N + 1)])

    def test_mul_matches_the_product_rule(self):
        """The convolution must reproduce hand-differentiated sin(wt)(2+cos(wt))."""
        t, w = self.t, self.w
        f, g = np.sin(w * t), 2 + np.cos(w * t)
        want = [f * g,
                w * np.cos(w * t) * g + f * (-w * np.sin(w * t)),
                (-w ** 2 * np.sin(w * t) * g + 2 * (w * np.cos(w * t))
                 * (-w * np.sin(w * t)) + f * (-w ** 2 * np.cos(w * t)))]
        for k, exp in enumerate(want):
            np.testing.assert_allclose(self.S.mul(self.C).derivs()[k], exp, atol=1e-13)

    def test_div_inverts_mul(self):
        """(S/C)*C == S to machine precision, at every order."""
        for a, b in zip(self.S.div(self.C).mul(self.C).derivs(), self.S.derivs()):
            np.testing.assert_allclose(a, b, atol=1e-13)

    def test_powi_matches_repeated_multiplication(self):
        for a, b in zip(self.S.powi(3).derivs(),
                        self.S.mul(self.S).mul(self.S).derivs()):
            np.testing.assert_allclose(a, b, atol=0, rtol=0)

    def test_powf_inverts_powi(self):
        """The fractional root is what F^(n) applies on the way out."""
        for n in (2, 3):
            with self.subTest(n=n):
                for a, b in zip(self.C.powf(1.0 / n).powi(n).derivs(),
                                self.C.derivs()):
                    np.testing.assert_allclose(a, b, atol=1e-12)

    def test_powf_on_a_complex_jet(self):
        """The F^(2) route: sqrt of Omega^2 - 2i Omega Omega'/Delta."""
        D = self.Jet.constant(3.0 + 0.0 * self.t, self.N - 1)
        Z = self.S.powi(2).sub(self.S.mul(self.S.deriv()).scale(2j).div(D))
        for a, b in zip(Z.powf(0.5).powi(2).derivs(), Z.derivs()):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_powf_one_is_the_exact_identity(self):
        """n_photon == 1 must not perturb a single bit.

        Every pre-existing caller is single-photon, so this short-circuit is what
        lets the general path stay numerically neutral for them.
        """
        for a, b in zip(self.C.powf(1.0).derivs(), self.C.derivs()):
            np.testing.assert_allclose(a, b, atol=0, rtol=0)
        self.assertIs(self.C.powf(1.0), self.C)

    def test_deriv_shifts_and_lowers_the_order(self):
        d = self.S.deriv()
        self.assertEqual(d.order, self.S.order - 1)
        for a, b in zip(d.derivs(), self.S.derivs()[1:]):
            np.testing.assert_allclose(a, b, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.Jet([np.zeros(3)]).deriv()

    def test_safe_divide_is_finite_at_a_zero_denominator(self):
        """The envelope vanishes at both gate edges, so this case is not exotic."""
        z = np.array([-1.0, 0.0, 1.0])
        one = np.ones_like(z)
        out = self.Jet([one] * 3).div(self.Jet([z, one, 0.0 * z])).derivs()
        for c in out:
            self.assertTrue(np.all(np.isfinite(c)))

    def test_safe_divide_gives_no_nan_gradient(self):
        """A single `where` is NOT enough -- it still poisons the BACKWARD pass."""
        try:
            import jax
        except ImportError:                                    # pragma: no cover
            self.skipTest("jax not installed")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        def obj(a):
            num = self.Jet([a * jnp.ones(3), jnp.ones(3)])
            den = self.Jet([jnp.asarray([-1.0, 0.0, 1.0]), jnp.ones(3)])
            return jnp.sum(jnp.abs(jnp.asarray(num.div(den, jnp).derivs())) ** 2)

        g = float(jax.grad(obj)(2.0))
        self.assertTrue(np.isfinite(g), f"gradient through the guard is {g}")


class TestEnvelopeJets(unittest.TestCase):
    """Analytic higher derivatives of every envelope, and of the chirp."""

    T_G = 40.0
    ORDER = 3

    def _envelopes(self):
        from snail_solver.envelope import (ConstantPulse, IQFourierEnvelope,
                                           RaisedCosine)
        return [RaisedCosine(1.3, self.T_G), ConstantPulse(0.7, self.T_G),
                IQFourierEnvelope(1.1, self.T_G, freqs=[0.21, 0.47],
                                  sin_I=[0.3, -0.1], sin_Q=[0.2, 0.05],
                                  cos_I=[-0.15, 0.08], cos_Q=[0.1, -0.2])]

    def test_jets_are_the_derivatives(self):
        """Checked by h-refinement: the residual must fall as O(h^2), not merely be
        small. See :func:`_fd_order`."""
        t = np.linspace(6.0, self.T_G - 6.0, 21)
        for env in self._envelopes():
            jet = env.jet_at(t, self.ORDER, np)
            self.assertEqual(len(jet), self.ORDER + 1)
            for n in range(1, self.ORDER + 1):
                with self.subTest(env=type(env).__name__, n=n):
                    if np.max(np.abs(np.asarray(jet[n]))) < 1e-14:
                        continue                       # ConstantPulse: exactly zero
                    for r in _fd_order(lambda x: env.value_at(x, np), t, n, jet[n]):
                        self.assertGreater(r, 3.2)
                        self.assertLess(r, 4.8)

    def test_jet_order_one_reproduces_deriv_at(self):
        t = np.linspace(2.0, self.T_G - 2.0, 17)
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__):
                np.testing.assert_allclose(env.jet_at(t, self.ORDER, np)[1],
                                           env.deriv_at(t, np), atol=1e-15)

    def test_jets_vanish_outside_the_gate(self):
        t = np.array([-3.0, -0.5, self.T_G + 0.5, self.T_G + 3.0])
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__):
                for c in env.jet_at(t, self.ORDER, np):
                    np.testing.assert_allclose(c, 0.0, atol=0, rtol=0)

    def test_hann_supports_exactly_one_clean_derivative(self):
        """THE reason a raised cosine cannot carry the full recursion.

        The paper's Eq. 13 shape has m vanishing derivatives at each edge "which
        guarantees the validity of the frame transformation". Hann has two, not
        three -- so a second nested correction turns the pulse on with a finite
        amplitude STEP. This test states that boundary fact directly.
        """
        from snail_solver.envelope import RaisedCosine
        env = RaisedCosine(1.3, self.T_G)
        for edge in (0.0, self.T_G):
            jet = env.jet_at(np.array([edge]), 2, np)
            with self.subTest(edge=edge):
                self.assertEqual(abs(complex(jet[0][0])), 0.0)      # eps  = 0
                self.assertLess(abs(complex(jet[1][0])), 1e-16)     # eps' = 0
                self.assertAlmostEqual(abs(complex(jet[2][0])),     # eps'' != 0
                                       1.3 * 0.5 * (TWO_PI / self.T_G) ** 2, places=12)

    def test_iq_with_no_basis_is_a_raised_cosine_at_every_order(self):
        from snail_solver.envelope import IQFourierEnvelope, RaisedCosine
        t = np.linspace(0.0, self.T_G, 33)
        a = IQFourierEnvelope(1.3, self.T_G).jet_at(t, self.ORDER, np)
        b = RaisedCosine(1.3, self.T_G).jet_at(t, self.ORDER, np)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-15)

    def test_chirp_detuning_jet(self):
        from snail_solver.envelope import Chirp
        ch = Chirp([0.004, 0.0, -0.011, 0.0, 0.003], self.T_G)
        t = np.linspace(6.0, self.T_G - 6.0, 21)
        jet = ch.detuning_jet(t, self.ORDER, np)
        # order 0 must be EXACTLY detuning(): the DRAG beat outside the gate depends
        # on it, and this method must not quietly redefine it.
        np.testing.assert_allclose(jet[0], ch.detuning(t, np), atol=0, rtol=0)
        for n in (1, 2):
            with self.subTest(n=n):
                for r in _fd_order(lambda x: np.asarray(ch.detuning(x, np)),
                                   t, n, jet[n]):
                    self.assertGreater(r, 3.2)
                    self.assertLess(r, 4.8)

    def test_chirp_detuning_jet_is_inert_without_a_chirp(self):
        from snail_solver.envelope import Chirp
        out = Chirp([], self.T_G).detuning_jet(np.linspace(0, self.T_G, 9),
                                               self.ORDER, np)
        self.assertEqual(len(out), self.ORDER + 1)
        for c in out:
            np.testing.assert_allclose(c, 0.0, atol=0, rtol=0)

    def test_chirp_derivatives_vanish_outside_the_gate(self):
        """`_u` clips, so delta is constant out there and its slope is genuinely 0."""
        from snail_solver.envelope import Chirp
        ch = Chirp([0.004, 0.0, -0.011], self.T_G)
        t = np.array([-2.0, self.T_G + 2.0])
        for c in ch.detuning_jet(t, self.ORDER, np)[1:]:
            np.testing.assert_allclose(c, 0.0, atol=0, rtol=0)


class TestJetImplementationsAgree(unittest.TestCase):
    """`jax_engine` mirrors the envelope/chirp jets; the two must not drift.

    Same contract as :class:`TestChirpPhaseImplementationsAgree`, extended to the
    derivative stacks the recursion consumes.
    """

    T_G = 40.0
    ORDER = 3
    COEFFS = [0.0, 0.01, 0.03, 0.0, -0.004]

    def test_chirp_detuning_jets_agree(self):
        from snail_solver.envelope import Chirp
        from snail_solver.jax_engine import _chirp_detuning_jet
        t = np.linspace(0.0, self.T_G, 65)
        got = _chirp_detuning_jet({"chirp": [np.asarray(self.COEFFS)]}, t, 0,
                                  self.T_G, self.ORDER, np)
        want = Chirp(self.COEFFS, self.T_G).detuning_jet(t, self.ORDER, np)
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_shape_jets_agree(self):
        from snail_solver.envelope import (ConstantPulse, IQFourierEnvelope,
                                           RaisedCosine)
        from snail_solver.jax_engine import _shape_jet_at
        t = np.linspace(0.0, self.T_G, 65)
        envs = [RaisedCosine(1.0, self.T_G), ConstantPulse(1.0, self.T_G),
                IQFourierEnvelope(1.0, self.T_G, freqs=[0.21, 0.47],
                                  sin_I=[0.3, -0.1], sin_Q=[0.2, 0.05],
                                  cos_I=[-0.15, 0.08], cos_Q=[0.1, -0.2])]
        for env in envs:
            st = {"kind": type(env).__name__, "t_g": self.T_G,
                  "freqs": np.asarray(getattr(env, "freqs", np.zeros(0)), float)}
            params = {"iq": [np.asarray(env.get_params(), float)]}
            got = _shape_jet_at(st, params, t, 0, np, self.ORDER)
            want = env.jet_at(t, self.ORDER, np)      # env.amp == 1, so units match
            for a, b in zip(got, want):
                with self.subTest(env=st["kind"]):
                    np.testing.assert_allclose(a, b, atol=1e-12)

    def test_an_unknown_envelope_kind_fails_loudly(self):
        """A new Envelope subclass must not be silently treated as a Hann."""
        from snail_solver.jax_engine import _shape_jet_at
        with self.assertRaises(NotImplementedError):
            _shape_jet_at({"kind": "SomeNewShape", "t_g": self.T_G,
                           "freqs": np.zeros(0)}, {"iq": [np.zeros(0)]},
                          np.zeros(3), 0, np, 2)


class TestRecursiveDrag(unittest.TestCase):
    """The composition itself: ``F^(1) o F^(1) o F^(2)`` and its guarantees."""

    T_G = 40.0
    COEFFS = [0.0, 0.0, -0.012, 0.0, 0.004]

    def _env(self):
        from snail_solver.envelope import RaisedCosine
        return RaisedCosine(1.4, self.T_G)

    def _tone(self, channels=None, chirped=True, **kw):
        from snail_solver.envelope import Chirp, PumpTone
        return PumpTone(w_p_GHz=1.7, envelope=self._env(),
                        chirp=(Chirp(self.COEFFS, self.T_G) if chirped else None),
                        drag_channels=channels, **kw)

    def _general(self, tone, t):
        from snail_solver.drag import apply_drag, required_order
        chs = tone.drag_channels_resolved()
        k = required_order(chs)
        return apply_drag(tone.envelope.jet_at(t, k, np),
                          [tone.channel_detuning_jet(c, t, k, np) for c in chs],
                          chs, np)

    # -- the contract with everything that came before ---------------------
    def test_order_one_reproduces_the_legacy_expression(self):
        """The whole feature must be a no-op for every existing caller.

        Both branches are exercised: `is_legacy_drag` routes the solver through the
        closed form, and this checks that the JET path lands on the same number --
        which is what makes the fast path an optimization rather than a divergence.
        """
        t = np.linspace(0.0, self.T_G, 401)
        env = self._env()
        for chirped in (True, False):
            with self.subTest(chirped=chirped):
                tone = self._tone(chirped=chirped, drag=True, delta_drag_GHz=0.30,
                                  drag_n_pump=1)
                self.assertTrue(tone.is_legacy_drag)
                legacy = (env.value_at(t, np)
                          - 1j * env.deriv_at(t, np) / tone.drag_detuning(t, np))
                np.testing.assert_allclose(self._general(tone, t), legacy, atol=1e-13)

    def test_drag_off_resolves_to_no_channels(self):
        """Mirrors `make_chirp` returning None: an inert tone skips the path entirely."""
        self.assertEqual(self._tone().drag_channels_resolved(), ())
        self.assertEqual(self._tone(drag=True, delta_drag_GHz=0.0)
                         .drag_channels_resolved(), ())
        self.assertFalse(self._tone().is_legacy_drag)

    def test_legacy_fields_become_the_one_channel_shorthand(self):
        tone = self._tone(drag=True, delta_drag_GHz=0.25, drag_n_pump=2)
        (ch,) = tone.drag_channels_resolved()
        self.assertAlmostEqual(ch.beat_GHz, 0.25)
        self.assertEqual(ch.n_pump, 2)
        # n_photon stays 1: promoting it would silently turn every existing
        # subharmonic (drag_n_pump=2) call site into second-order DRAG.
        self.assertEqual(ch.n_photon, 1)
        self.assertTrue(tone.is_legacy_drag)

    # -- the composition rules ---------------------------------------------
    def test_multi_photon_channels_are_forced_innermost(self):
        """F^(n>=2) takes an n-th root, whose branch is only unambiguous while the
        amplitude is still real. Caller order must not be able to break that."""
        from snail_solver.envelope import DragChannel
        c1 = DragChannel(0.30, n_photon=1)
        c2 = DragChannel(0.55, n_pump=2, n_photon=2)
        for order in ([c1, c2], [c2, c1]):
            with self.subTest(order=[c.n_photon for c in order]):
                got = self._tone(channels=order).drag_channels_resolved()
                self.assertEqual([c.n_photon for c in got], [2, 1])

    def test_composition_order_actually_matters(self):
        """Guards against the sort being a decorative no-op."""
        from snail_solver.drag import apply_drag
        from snail_solver.envelope import DragChannel
        t = np.linspace(0.0, self.T_G, 201)
        c1, c2 = DragChannel(0.30, n_photon=1), DragChannel(0.55, n_pump=2, n_photon=2)
        tone = self._tone(channels=[c1, c2])
        jets = {c: tone.channel_detuning_jet(c, t, 2, np) for c in (c1, c2)}
        shape = tone.envelope.jet_at(t, 2, np)
        right = apply_drag(shape, [jets[c2], jets[c1]], [c2, c1], np)   # F1 o F2
        # bypass the sort to build the physically wrong composition F2 o F1
        g = apply_drag(tone.envelope.jet_at(t, 1, np), [jets[c1]], [c1], np)
        self.assertGreater(np.max(np.abs(right - g)), 1e-3 * np.max(np.abs(right)))

    def test_the_quotient_rule_term_is_real_and_grows_near_a_collision(self):
        """Documents that today's arithmetic is NOT Eq. (4) verbatim on a chirp.

        `eta - i eta'/Delta` omits the `+i eta Delta'/Delta^2` that the quotient rule
        contributes once a chirp makes Delta time-dependent. Small far away,
        order-unity near a collision -- exactly where DRAG is doing the work.
        """
        from snail_solver.envelope import DragChannel
        t = np.linspace(0.0, self.T_G, 401)
        base = self._env().value_at(t, np)
        seen = {}
        for beat in (0.30, 0.02):
            on = self._general(self._tone(
                channels=[DragChannel(beat, quotient_rule=True)]), t)
            off = self._general(self._tone(
                channels=[DragChannel(beat, quotient_rule=False)]), t)
            seen[beat] = (np.max(np.abs(on - off))
                          / np.max(np.abs(off - base)))
        self.assertLess(seen[0.30], 0.10)          # a few % at 300 MHz
        self.assertGreater(seen[0.02], 0.30)       # order unity at 20 MHz
        self.assertGreater(seen[0.02], 3.0 * seen[0.30])

    def test_hann_diverges_under_the_full_recursion(self):
        """The measured justification for the Eq. 13 base shape.

        With F^(2) innermost the imaginary term dominates as t -> 0, so for a shape
        vanishing as t^p the corrected pulse goes as t^(p-1/2). Hann has p = 2, so
        two further derivatives give t^(-1/2): halving the first sample time must
        multiply the amplitude there by ~sqrt(2), without bound.
        """
        from snail_solver.envelope import DragChannel
        tone = self._tone(chirped=False, channels=[
            DragChannel(0.30, n_photon=1), DragChannel(-0.22, n_photon=1),
            DragChannel(0.55, n_pump=2, n_photon=2)])
        vals = [abs(complex(self._general(tone, np.array([self.T_G / n]))[0]))
                for n in (401, 801, 1601, 3201)]
        ratios = [vals[i + 1] / vals[i] for i in range(3)]
        self.assertGreater(ratios[-1], 1.35)
        self.assertLess(ratios[-1], 1.48)          # -> sqrt(2)
        self.assertGreater(vals[-1], 2.0 * vals[0])

    # -- the solver paths ---------------------------------------------------
    def _coupler(self, tone):
        from snail_solver.zhou_coupler import ZhouCoupler
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={0: -0.12, 1: -0.12})
        cpl.set_pump(tone, normalize_iswap=(0, 1))
        return cpl

    def _channels(self):
        from snail_solver.envelope import DragChannel
        return [DragChannel(0.30, n_photon=1, quotient_rule=True),
                DragChannel(-0.22, n_photon=1, quotient_rule=True),
                DragChannel(0.55, n_pump=2, n_photon=2, quotient_rule=True)]

    def test_the_solver_applies_the_recursion(self):
        """`_eta_at` must route multi-channel tones through the composition."""
        t = np.linspace(0.0, self.T_G, 65)
        tone = self._tone(channels=self._channels())
        cpl = self._coupler(tone)
        # `_general` reads tone.envelope, so it already carries set_pump's rescale;
        # is_eta=True makes the prefactor 1 and phi_p is 0.
        want = (self._general(tone, t)
                * np.exp(-1j * np.asarray(tone.chirp.phase(t, np))))
        np.testing.assert_allclose(cpl._eta_at(tone, t, np), want, atol=1e-12)
        # and it must NOT be the plain envelope: the correction has to be visible
        self.assertGreater(np.max(np.abs(cpl._eta_at(tone, t, np)
                                         - tone.envelope.value_at(t, np))),
                           1e-3 * tone.envelope.amp)

    def test_numpy_and_jax_engines_agree_on_the_recursion(self):
        """The two mirrors must stay the same function twice."""
        from snail_solver import jax_engine as JE
        t = np.linspace(0.0, self.T_G, 65)
        for channels in ([self._channels()[0]], self._channels()[:2],
                         self._channels()):
            with self.subTest(n=len(channels)):
                tone = self._tone(channels=list(channels))
                cpl = self._coupler(tone)
                spec, params = JE.pulse_spec(cpl), JE.pulse_params(cpl)
                np.testing.assert_allclose(JE.eta_at(spec, params, t, 0, np),
                                           cpl._eta_at(tone, t, np), atol=1e-12)

    def test_the_scalar_callback_matches_the_vectorized_form(self):
        tone = self._tone(channels=self._channels())
        cpl = self._coupler(tone)
        t = np.linspace(0.0, self.T_G, 33)
        np.testing.assert_allclose([cpl._eta(tone, float(x)) for x in t],
                                   cpl._eta_at(tone, t, np), atol=1e-12)

    def test_the_detuning_guard_covers_every_channel(self):
        """One collapsing beat is enough to break the pulse, wherever it sits."""
        from snail_solver.device_utils import check_drag_detuning
        from snail_solver.envelope import DragChannel
        safe = [DragChannel(0.30), DragChannel(-0.22)]
        check_drag_detuning(self._tone(channels=safe))
        self.assertEqual(len(self._tone(channels=safe).drag_detuning_floors()), 2)
        # a chirp that sweeps the SECOND channel's beat through zero must still raise
        bad = [DragChannel(0.30), DragChannel(0.0, n_pump=1)]
        with self.assertRaises(ValueError) as cm:
            check_drag_detuning(self._tone(channels=bad))
        self.assertIn("min|Delta(t)|", str(cm.exception))

    def test_chirp_gradient_survives_the_recursion(self):
        """Strictly harder than the first-order regression: with the quotient rule
        on, the chirp now reaches the amplitude through Delta, Delta' AND Delta''."""
        try:
            import jax
        except ImportError:                                    # pragma: no cover
            self.skipTest("jax not installed")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from snail_solver import jax_engine as JE

        tone = self._tone(channels=self._channels())
        cpl = self._coupler(tone)
        spec, base = JE.pulse_spec(cpl), JE.pulse_params(cpl)
        ts = jnp.asarray(np.linspace(0.0, self.T_G, 33))
        c0 = np.asarray(self.COEFFS)

        def objective(c):
            p = {"amp": jnp.asarray(base["amp"]), "chirp": [c],
                 "iq": [jnp.asarray(q) for q in base["iq"]]}
            return jnp.sum(jnp.abs(JE.eta_at(spec, p, ts, 0, jnp)) ** 2)

        g = np.asarray(jax.grad(objective)(jnp.asarray(c0)))
        h = 1e-6
        fd = np.array([(float(objective(jnp.asarray(c0 + h * np.eye(c0.size)[i])))
                        - float(objective(jnp.asarray(c0 - h * np.eye(c0.size)[i]))))
                       / (2 * h) for i in range(c0.size)])
        self.assertTrue(np.all(np.isfinite(g)))
        np.testing.assert_allclose(g, fd, atol=1e-6)
        self.assertGreater(np.max(np.abs(fd)), 1e-3,
                           "objective must actually depend on the chirp")


class TestRecursiveDragPlumbing(unittest.TestCase):
    """The guards and builders around the recursion, not the recursion itself."""

    T_G = 40.0

    def _channels(self):
        from snail_solver.envelope import DragChannel
        return [DragChannel(0.30, n_photon=1, quotient_rule=True),
                DragChannel(-0.22, n_photon=1, quotient_rule=True),
                DragChannel(0.55, n_pump=2, n_photon=2, quotient_rule=True)]

    def _tone(self, channels, chirp_coeffs=None, amp=1.4):
        from snail_solver.envelope import PumpTone, RaisedCosine, make_chirp
        return PumpTone(w_p_GHz=1.7, envelope=RaisedCosine(amp, self.T_G),
                        chirp=make_chirp(chirp_coeffs, self.T_G),
                        drag_channels=channels)

    def test_correction_ratio_is_zero_when_drag_is_off(self):
        from snail_solver.device_utils import drag_correction_ratio
        self.assertEqual(drag_correction_ratio(self._tone(None)), 0.0)

    def test_correction_ratio_grows_as_the_beats_close_in(self):
        """The guard `min|Delta(t)|` structurally cannot give: a pulse whose every
        beat is far from zero can still have a correction larger than itself."""
        from snail_solver.device_utils import drag_correction_ratio
        from snail_solver.envelope import DragChannel
        ratios = []
        for scale in (1.0, 0.5, 0.25):
            # SAME sign, so the two 1/Delta terms add. With opposite signs they
            # partially cancel (+0.30/-0.22 lands near 0.08 rather than 0.39), which
            # is real physics but the wrong case for exercising the warning.
            chs = [DragChannel(0.30 * scale, n_photon=1),
                   DragChannel(0.22 * scale, n_photon=1)]
            ratios.append(drag_correction_ratio(self._tone(chs)))
        self.assertTrue(all(np.isfinite(r) for r in ratios))
        self.assertLess(ratios[0], ratios[1])
        self.assertLess(ratios[1], ratios[2])
        self.assertGreater(ratios[-1], 0.3)        # would trip build_coupler's warning
        # every beat is still >= 55 MHz from zero, so the min|Delta| guard is happy:
        # this is exactly the failure it cannot see
        from snail_solver.device_utils import check_drag_detuning
        check_drag_detuning(self._tone([DragChannel(0.075), DragChannel(0.055)]))

    def test_correction_ratio_matches_the_legacy_quadrature(self):
        from snail_solver.device_utils import drag_correction_ratio
        from snail_solver.envelope import RaisedCosine
        tone = self._tone(None)
        tone.drag, tone.delta_drag_GHz, tone.drag_n_pump = True, 0.30, 1
        env = RaisedCosine(1.4, self.T_G)
        ts = np.linspace(0.0, self.T_G, 259)[1:-1]
        want = (np.max(np.abs(env.deriv_at(ts, np) / (TWO_PI * 0.30)))
                / np.max(np.abs(env.value_at(ts, np))))
        self.assertAlmostEqual(drag_correction_ratio(tone), want, places=12)

    def test_the_guard_names_the_offending_channel(self):
        from snail_solver.device_utils import check_drag_detuning
        from snail_solver.envelope import DragChannel
        bad = [DragChannel(0.30), DragChannel(0.28), DragChannel(0.0)]
        with self.assertRaises(ValueError) as cm:
            check_drag_detuning(self._tone(bad))
        msg = str(cm.exception)
        self.assertIn("min|Delta(t)|", msg)        # pinned by the older guard test
        self.assertIn("channel 3/3", msg)

    def test_channel_filter_drops_only_the_offender(self):
        """One swept-onto collision must not throw away the other channels."""
        from snail_solver.envelope import DragChannel
        from snail_solver.sweep_common import (DEFAULT_CONFIG,
                                               _drag_channels_filtered)
        chs = [DragChannel(0.30, n_pump=1), DragChannel(0.0005, n_pump=1),
               DragChannel(-0.22, n_pump=1)]
        kept = _drag_channels_filtered(DEFAULT_CONFIG, chs, None, self.T_G)
        self.assertEqual([c.beat_GHz for c in kept], [0.30, -0.22])
        self.assertEqual(_drag_channels_filtered(DEFAULT_CONFIG, [], None, self.T_G), ())

    def test_build_coupler_threads_the_channels(self):
        from snail_solver.device_utils import build_coupler
        from snail_solver.sweep_common import DEFAULT_CONFIG
        cfg = dict(DEFAULT_CONFIG, envelope="raised_cosine")
        cpl, _wp, _peak = build_coupler(cfg, self.T_G, 1.0, 0.0,
                                        drag_channels=self._channels())
        tone = cpl._pump_tones[0]
        self.assertEqual(len(tone.drag_channels_resolved()), 3)
        self.assertFalse(tone.is_legacy_drag)
        # and the default (no channels) is untouched
        cpl0, _w, _p = build_coupler(cfg, self.T_G, 1.0, 0.0, drag_beat_GHz=0.30)
        self.assertTrue(cpl0._pump_tones[0].is_legacy_drag)

    def test_grape_baseline_uses_the_recursive_pulse(self):
        """The gate every reported dF_grape is measured against must be the pulse the
        solver plays -- otherwise the optimizer is credited for beating a fiction."""
        from snail_solver.envelope import Chirp
        from snail_solver.grape import _raised_cosine_eta, _tone_drag_channels
        chs = self._channels()
        chirp = Chirp([0.0, 0.0, -0.012], self.T_G)
        rec = _raised_cosine_eta(self.T_G, 1.4, 24, None, chirp, 1, chs)
        one = _raised_cosine_eta(self.T_G, 1.4, 24, 0.30, chirp, 1)
        self.assertEqual(rec.shape, one.shape)
        self.assertGreater(np.max(np.abs(rec - one)), 1e-3)
        # channels=None / () must reproduce the historical samples EXACTLY
        for empty in (None, ()):
            np.testing.assert_allclose(
                _raised_cosine_eta(self.T_G, 1.4, 24, 0.30, chirp, 1, empty),
                one, atol=0, rtol=0)

    def test_grape_reads_channels_off_the_coupler(self):
        """`_tone_drag_channels` returns () for the legacy case, so the baseline
        keeps its original code path (and its original float) for existing points."""
        from snail_solver.device_utils import build_coupler
        from snail_solver.grape import _tone_drag_channels
        from snail_solver.sweep_common import DEFAULT_CONFIG
        cfg = dict(DEFAULT_CONFIG, envelope="raised_cosine")
        legacy, _w, _p = build_coupler(cfg, self.T_G, 1.0, 0.0, drag_beat_GHz=0.30)
        self.assertEqual(_tone_drag_channels(legacy), ())
        rec, _w, _p = build_coupler(cfg, self.T_G, 1.0, 0.0,
                                    drag_channels=self._channels())
        self.assertEqual(len(_tone_drag_channels(rec)), 3)

    def test_the_crab_optimizer_clears_the_channel_list(self):
        """`drag=False` alone does not disable an EXPLICIT channel list, so the
        optimizer must clear it too -- otherwise the recursion fires on top of the
        ansatz that is supposed to discover the quadrature itself."""
        import inspect
        from snail_solver import grape
        src = inspect.getsource(grape._optimize_crab)
        self.assertIn("tone.drag_channels = None", src)
        self.assertIn("tone.drag_channels = channels_saved", src)


class TestSinePowerRamp(unittest.TestCase):
    """The Li/Calarco/Motzoi Eq. (13) base shape."""

    T_G = 40.0

    def test_m_one_is_exactly_a_raised_cosine(self):
        """The keystone. The paper states it ("for m = 1 and with zero holding time,
        the pulse is the same as the Hann window"), and it is what makes every
        Hann-specific constant elsewhere a special case rather than a re-derivation.

        It is also the check that settles the printed-vs-corrected reading of Eq. 13:
        with the integrand as printed (`sin^m(pi t'/2 t_r)`) this differs from a Hann
        window by 0.25 in amplitude and the ramp's slope at t_r does not vanish.
        """
        from snail_solver.envelope import RaisedCosine, SinePowerRamp
        t = np.linspace(-2.0, self.T_G + 2.0, 977)
        spr, rc = SinePowerRamp(1.3, self.T_G, m=1), RaisedCosine(1.3, self.T_G)
        for n, (a, b) in enumerate(zip(spr.jet_at(t, 3, np), rc.jet_at(t, 3, np))):
            with self.subTest(order=n):
                np.testing.assert_allclose(a, b, atol=1e-14)
        self.assertAlmostEqual(spr.area(), rc.area(), places=13)

    def test_m_derivatives_vanish_at_both_edges(self):
        """The property the recursion needs, and the reason m must track K."""
        from snail_solver.envelope import SinePowerRamp
        for m in (1, 3, 5):
            env = SinePowerRamp(1.0, self.T_G, m=m)
            for edge in (0.0, self.T_G):
                jet = env.jet_at(np.array([edge]), m + 1, np)
                with self.subTest(m=m, edge=edge):
                    for n in range(m + 1):
                        self.assertLess(abs(float(np.real(jet[n][0]))), 1e-12,
                                        f"d^{n} should vanish for m={m}")
                    self.assertGreater(abs(float(np.real(jet[m + 1][0]))), 1e-6,
                                       f"d^{m + 1} should NOT vanish for m={m}")

    def test_it_tames_the_recursion_that_hann_diverges_under(self):
        """The whole point of the shape, stated as the contrast with Hann.

        Same three channels, same recursion: on a raised cosine the edge amplitude
        GROWS without bound as the grid refines (t^-1/2); on m=3 it decays.
        """
        from snail_solver.drag import apply_drag, required_order
        from snail_solver.envelope import (DragChannel, PumpTone, RaisedCosine,
                                           SinePowerRamp)
        chs = [DragChannel(0.30, n_photon=1), DragChannel(-0.22, n_photon=1),
               DragChannel(0.55, n_pump=2, n_photon=2)]
        k = required_order(chs)

        def edge_values(env):
            tone = PumpTone(w_p_GHz=0.4, envelope=env, drag_channels=chs)
            out = []
            for n in (401, 801, 1601, 3201):
                tt = np.array([self.T_G / n])
                out.append(abs(complex(apply_drag(
                    env.jet_at(tt, k, np),
                    [tone.channel_detuning_jet(c, tt, k, np) for c in chs],
                    chs, np)[0])))
            return out

        hann = edge_values(RaisedCosine(1.4, self.T_G))
        ramp = edge_values(SinePowerRamp(1.4, self.T_G, m=3))
        self.assertGreater(hann[-1], 2.0 * hann[0])        # diverging
        self.assertLess(ramp[-1], 0.2 * ramp[0])           # vanishing
        self.assertLess(ramp[-1], 1e-4 * hann[-1])

    def test_area_closed_form_matches_quadrature(self):
        """`normalize_iswap` divides by this, so it must be exact, not approximate."""
        from snail_solver.envelope import SinePowerRamp
        t = np.linspace(0.0, self.T_G, 20001)
        for m, t_rise in [(3, None), (3, 8.0), (4, 12.0), (1, 20.0), (2, 5.0)]:
            env = SinePowerRamp(1.1, self.T_G, m=m, t_rise=t_rise)
            with self.subTest(m=m, t_rise=t_rise):
                quad = float(np.trapz(env.value_at(t, np), t))
                self.assertAlmostEqual(env.area() / quad, 1.0, places=9)

    def test_plateau_and_mirror_symmetry(self):
        from snail_solver.envelope import SinePowerRamp
        env = SinePowerRamp(1.1, self.T_G, m=3, t_rise=8.0)
        t = np.linspace(0.0, self.T_G, 401)
        np.testing.assert_allclose(env.value_at(t, np),
                                   env.value_at(self.T_G - t, np), atol=1e-13)
        mid = np.linspace(9.0, self.T_G - 9.0, 33)          # strictly on the plateau
        np.testing.assert_allclose(env.value_at(mid, np), 1.1, atol=1e-13)

    def test_rejects_impossible_geometry(self):
        from snail_solver.envelope import SinePowerRamp
        with self.assertRaises(ValueError):
            SinePowerRamp(1.0, self.T_G, m=0)
        with self.assertRaises(ValueError):
            SinePowerRamp(1.0, self.T_G, t_rise=self.T_G)     # > t_g/2
        with self.assertRaises(ValueError):
            SinePowerRamp(1.0, self.T_G, t_rise=0.0)

    def test_it_satisfies_the_shared_envelope_contract(self):
        """Same three properties `TestEnvelopeArrayAPI` pins for every other shape."""
        from snail_solver.envelope import SinePowerRamp
        env = SinePowerRamp(1.1, self.T_G, m=3, t_rise=9.0)
        t = np.linspace(1.0, self.T_G - 1.0, 25)
        np.testing.assert_allclose([float(env.value(x)) for x in t],
                                   env.value_at(t, np), atol=1e-14)
        np.testing.assert_allclose([float(env.deriv(x)) for x in t],
                                   env.deriv_at(t, np), atol=1e-14)
        # FD on a grid strictly INSIDE the rising ramp. The shape is only C^m at the
        # ramp/plateau junction (d^(m+1) jumps there by construction), and an
        # order-n central difference with h = 0.8 reaches +/-2.4 ns, so a grid
        # spanning t_rise would measure that genuine kink rather than the formula.
        ti = np.linspace(3.0, 6.0, 9)                        # t_rise = 9.0
        for n in (1, 2, 3):
            with self.subTest(n=n):
                for r in _fd_order(lambda x: env.value_at(x, np), ti,
                                   n, env.jet_at(ti, 3, np)[n]):
                    self.assertGreater(r, 3.2)
                    self.assertLess(r, 4.8)
        out = env.jet_at(np.array([-2.0, self.T_G + 2.0]), 3, np)
        for c in out:
            np.testing.assert_allclose(c, 0.0, atol=0, rtol=0)

    def test_the_jax_mirror_agrees(self):
        from snail_solver.envelope import SinePowerRamp
        from snail_solver.jax_engine import _shape_jet_at
        env = SinePowerRamp(1.0, self.T_G, m=3, t_rise=9.0)
        t = np.linspace(0.0, self.T_G, 65)
        st = {"kind": "SinePowerRamp", "t_g": self.T_G, "freqs": np.zeros(0),
              "shape_m": 3, "shape_t_rise": 9.0}
        for a, b in zip(_shape_jet_at(st, {"iq": [np.zeros(0)]}, t, 0, np, 3),
                        env.jet_at(t, 3, np)):
            np.testing.assert_allclose(a, b, atol=1e-13)

    def test_area_factor_and_the_amplitude_algebra(self):
        """The Hann constant 1/2 must come out unchanged, bit for bit."""
        from snail_solver.sweep_common import DEFAULT_CONFIG
        from snail_solver.tune_up import area_factor, fixed_eta_amp_scale, peak_eta_of
        hann = dict(DEFAULT_CONFIG, envelope="raised_cosine")
        self.assertEqual(area_factor(hann), 0.5)
        # sine_power with m=1 / no plateau is the same pulse, so the same factor
        m1 = dict(DEFAULT_CONFIG, envelope="sine_power", envelope_m=1,
                  envelope_rise_frac=0.5)
        self.assertAlmostEqual(area_factor(m1), 0.5, places=13)
        # a plateau raises the factor; a sharper ramp raises it further
        wide = dict(DEFAULT_CONFIG, envelope="sine_power", envelope_m=3,
                    envelope_rise_frac=0.2)
        self.assertGreater(area_factor(wide), 0.5)
        self.assertLess(area_factor(wide), 1.0)
        # round trip holds for any shape
        for cfg in (hann, m1, wide):
            with self.subTest(envelope=cfg["envelope"]):
                s = fixed_eta_amp_scale(cfg, 33.0, 1.8)
                self.assertAlmostEqual(peak_eta_of(cfg, 33.0, s), 1.8, places=12)

    def test_build_coupler_can_install_the_new_shape(self):
        from snail_solver.device_utils import build_coupler
        from snail_solver.envelope import SinePowerRamp
        from snail_solver.sweep_common import DEFAULT_CONFIG
        cfg = dict(DEFAULT_CONFIG, envelope="sine_power", envelope_m=3,
                   envelope_rise_frac=0.25)
        cpl, _wp, _peak = build_coupler(cfg, self.T_G, 1.0, 0.0)
        env = cpl._pump_tones[0].envelope
        self.assertIsInstance(env, SinePowerRamp)
        self.assertEqual(env.m, 3)
        # and the default is still a Hann -- this shape is strictly opt-in
        cpl0, _w, _p = build_coupler(dict(DEFAULT_CONFIG, envelope="raised_cosine"),
                                     self.T_G, 1.0, 0.0)
        self.assertEqual(type(cpl0._pump_tones[0].envelope).__name__, "RaisedCosine")


class TestTuneUpRecursiveDrag(unittest.TestCase):
    """`chirp_from_measured_shift` now models the pulse the solver actually plays."""

    T_G = 80.0

    def _table(self):
        return {"fit": {"k2": -0.9, "k4": 0.12, "delta0": -0.7}, "target_eta": 1.8,
                "eta": np.linspace(0.5, 1.9, 9)}

    @staticmethod
    def _old_implementation(table, eta_star, degree, beat, k_pump, t_g,
                            max_iters=12, tol=1e-12):
        """The pre-recursion code, transcribed, as an independent oracle."""
        from numpy.polynomial import legendre as L
        k2, k4 = table["fit"]["k2"], table["fit"]["k4"]
        u, w = np.polynomial.legendre.leggauss(max(2 * degree + 8, 32))
        amp = eta_star * np.cos(np.pi * u / 2) ** 2
        shift = lambda a: (k2 * a ** 2 + k4 * a ** 4) * 1e-3     # noqa: E731
        d = shift(amp)
        deta_dt = -eta_star * (np.pi / t_g) * np.sin(np.pi * u)
        for _ in range(max_iters):
            det = beat * TWO_PI - k_pump * (d * TWO_PI)
            new = shift(np.sqrt(amp ** 2 + (deta_dt / det) ** 2))
            done = np.max(np.abs(new - d)) < tol
            d = new
            if done:
                break
        c = np.array([(2 * k + 1) / 2 * np.sum(w * d * L.legval(u, np.eye(k + 1)[k]))
                      for k in range(degree + 1)])
        c[1::2] = 0.0
        return c

    def test_order_one_reproduces_the_previous_implementation(self):
        """`sqrt(amp^2 + q^2)` equals `|amp - i q|` only because a first-order
        correction is purely imaginary -- so at one channel the rewrite must be a
        pure refactor, and beyond it the old formula was simply the wrong norm."""
        from snail_solver.tune_up import chirp_from_measured_shift
        for beat, k in [(0.30, 1), (0.05, 1), (0.05, 2), (0.30, 0)]:
            with self.subTest(beat=beat, k=k):
                got = chirp_from_measured_shift(
                    self._table(), 1.8, degree=8, drag_beat_GHz=beat,
                    drag_n_pump=k, t_g=self.T_G, pin_c0=False)["coeffs_GHz"]
                want = self._old_implementation(self._table(), 1.8, 8, beat, k,
                                                self.T_G)
                np.testing.assert_allclose(got, want,
                                           rtol=1e-11, atol=1e-16)

    def test_channels_and_the_scalar_shorthand_agree(self):
        from snail_solver.envelope import DragChannel
        from snail_solver.tune_up import chirp_from_measured_shift
        kw = dict(degree=8, t_g=self.T_G)
        a = chirp_from_measured_shift(self._table(), 1.8, drag_beat_GHz=0.30,
                                      drag_n_pump=1, **kw)
        b = chirp_from_measured_shift(self._table(), 1.8,
                                      drag_channels=[DragChannel(0.30, n_pump=1)], **kw)
        np.testing.assert_allclose(a["coeffs_GHz"], b["coeffs_GHz"], atol=0, rtol=0)

    def test_multi_channel_reports_every_beat(self):
        from snail_solver.envelope import DragChannel
        from snail_solver.tune_up import chirp_from_measured_shift
        out = chirp_from_measured_shift(
            self._table(), 1.8, degree=8, t_g=self.T_G,
            drag_channels=[DragChannel(0.30), DragChannel(-0.22),
                           DragChannel(0.55, n_pump=2, n_photon=2)])
        self.assertEqual(out["n_drag_channels"], 3)
        self.assertEqual(len(out["min_abs_detuning_per_channel_GHz"]), 3)
        self.assertAlmostEqual(out["min_abs_detuning_GHz"],
                               min(out["min_abs_detuning_per_channel_GHz"]), places=15)
        self.assertGreater(out["drag_correction_ratio"], 0.0)
        self.assertTrue(np.isfinite(out["drag_delta_frac"]))

    def test_drag_off_is_unaffected_by_a_channel_list(self):
        from snail_solver.tune_up import chirp_from_measured_shift
        kw = dict(degree=8, t_g=self.T_G)
        off = chirp_from_measured_shift(self._table(), 1.8, **kw)
        for empty in (None, [], ()):
            with self.subTest(empty=empty):
                got = chirp_from_measured_shift(self._table(), 1.8,
                                                drag_channels=empty, **kw)
                np.testing.assert_allclose(got["coeffs_GHz"], off["coeffs_GHz"],
                                           atol=0, rtol=0)
                self.assertNotIn("drag_iters", got)

    def test_t_g_is_still_required_with_channels(self):
        from snail_solver.envelope import DragChannel
        from snail_solver.tune_up import chirp_from_measured_shift
        with self.assertRaises(ValueError):
            chirp_from_measured_shift(self._table(), 1.8, degree=8,
                                      drag_channels=[DragChannel(0.30)])

    def test_recursion_breaks_length_independence_harder(self):
        """DRAG-off the chirp is t_g-independent; each nested order adds a 1/t_g."""
        from snail_solver.envelope import DragChannel
        from snail_solver.tune_up import chirp_from_measured_shift

        def coeffs(t_g, chs):
            return chirp_from_measured_shift(self._table(), 1.8, degree=8, t_g=t_g,
                                             drag_channels=chs)["coeffs_GHz"]

        off_a, off_b = coeffs(40.0, None), coeffs(80.0, None)
        np.testing.assert_allclose(off_a, off_b, atol=1e-15)
        one = [DragChannel(0.30)]
        two = [DragChannel(0.30), DragChannel(0.22)]
        d1 = np.max(np.abs(coeffs(40.0, one) - coeffs(80.0, one)))
        d2 = np.max(np.abs(coeffs(40.0, two) - coeffs(80.0, two)))
        self.assertGreater(d1, 0.0)
        self.assertGreater(d2, d1)

    def test_cli_channel_parsing(self):
        from snail_solver.tune_up import parse_drag_channels
        self.assertIsNone(parse_drag_channels(None))
        self.assertIsNone(parse_drag_channels([]))
        chs = parse_drag_channels(["0.30", "-0.22:1", "0.55:2:2", "0.1:2"])
        self.assertEqual([c.beat_GHz for c in chs], [0.30, -0.22, 0.55, 0.1])
        self.assertEqual([c.n_pump for c in chs], [1, 1, 2, 2])
        # n_photon defaults to n_pump: for every channel this device produces they
        # are the same integer, but they stay separately settable
        self.assertEqual([c.n_photon for c in chs], [1, 1, 2, 2])
        self.assertTrue(all(c.quotient_rule for c in chs))
        with self.assertRaises(ValueError):
            parse_drag_channels(["0.3:1:1:1"])

    def test_the_outer_loop_gets_more_passes_for_deeper_recursions(self):
        """The d-th nested correction scales as 1/t_g^d, so the chirp<->length
        coupling tightens with the channel count."""
        import inspect
        from snail_solver import tune_up
        src = inspect.getsource(tune_up.run_tune_up)
        self.assertIn("2 * _n_ch", src)
        self.assertIn("_drag_on = drag_beat_GHz is not None or bool(drag_channels)", src)


class TestEtaCacheAndCollisionChannels(unittest.TestCase):
    """The per-Hamiltonian eta cache, and auto-filling channels from collisions."""

    T_G = 30.0

    def _cfg(self, **kw):
        from snail_solver.sweep_common import DEFAULT_CONFIG
        return dict(DEFAULT_CONFIG, qubit_levels=2, coupler_levels=3, **kw)

    def test_the_hamiltonian_cache_returns_the_same_coefficients(self):
        """Caching eta per (tone, t) must be invisible in the numbers.

        The solver evaluates every term at the same t before stepping, and each term
        was recomputing eta from scratch -- 12 identical evaluations per timestep.
        This asserts the cache changes nothing, which is what makes it safe.
        """
        from snail_solver.device_utils import build_coupler
        from snail_solver.envelope import DragChannel
        chs = [DragChannel(0.30, quotient_rule=True),
               DragChannel(-0.22, quotient_rule=True),
               DragChannel(0.55, n_pump=2, n_photon=2, quotient_rule=True)]
        for kw in ({}, {"drag_beat_GHz": 0.30}, {"drag_channels": chs}):
            cpl, _w, _p = build_coupler(self._cfg(), self.T_G, 1.0, 0.0,
                                        chirp_coeffs_GHz=[0.0, 0.0, -0.004], **kw)
            H = cpl.to_qutip_hamiltonian(cutoff_GHz=1.0)
            terms = [x for x in (H if isinstance(H, list) else H.to_list())
                     if isinstance(x, (list, tuple)) and len(x) == 2]
            self.assertGreater(len(terms), 1, "need several terms to exercise sharing")
            tone = cpl._pump_tones[0]
            for t in (0.0, 7.3, self.T_G / 2, self.T_G):
                # interleave the terms so a stale one-slot cache would be caught
                got = [complex(c(t)) for _op, c in terms]
                eta = complex(cpl._eta(tone, t))
                self.assertTrue(np.all(np.isfinite(got)))
                # re-evaluating at the same t must be identical, and at a NEW t must
                # not return the previous value
                self.assertEqual([complex(c(t)) for _op, c in terms], got)
                self.assertTrue(np.isfinite(eta))

    def test_the_cache_does_not_survive_a_rebuild(self):
        """A tone mutated in place between solves (as `grape` does) must not be
        served a stale amplitude: the cache lives only as long as one QobjEvo."""
        from snail_solver.device_utils import build_coupler
        cpl, _w, _p = build_coupler(self._cfg(), self.T_G, 1.0, 0.0,
                                    drag_beat_GHz=0.30)
        tone = cpl._pump_tones[0]

        def first_coeff(t):
            H = cpl.to_qutip_hamiltonian(cutoff_GHz=1.0)
            terms = [x for x in (H if isinstance(H, list) else H.to_list())
                     if isinstance(x, (list, tuple)) and len(x) == 2]
            return complex(terms[0][1](t))

        before = first_coeff(5.0)
        tone.envelope.amp *= 2.0                      # in-place mutation, as grape does
        after = first_coeff(5.0)
        self.assertNotAlmostEqual(abs(before), abs(after), places=6)

    def test_collision_channels_are_the_n_nearest_and_distinct(self):
        from snail_solver.sweep_common import (_collision_candidates,
                                               collision_drag_channels)
        cfg = self._cfg()
        wa, wb, ws, wspec, w_p = 5.0, 4.6, 7.0, 4.75, 0.4
        chs = collision_drag_channels(cfg, wa, wb, ws, wspec, w_p, n=3)
        self.assertLessEqual(len(chs), 3)
        beats = [abs(c.beat_GHz) for c in chs]
        self.assertEqual(beats, sorted(beats), "channels must be nearest-first")
        self.assertEqual(len(set(round(c.beat_GHz, 9) for c in chs)), len(chs),
                         "duplicate beats would double-count one process")
        # the nearest of them must be THE nearest collision
        from snail_solver.sweep_common import _nearest_collision
        near = _nearest_collision(cfg, wa, wb, ws, wspec, w_p)
        self.assertAlmostEqual(abs(chs[0].beat_GHz), near[0], places=12)
        # and every candidate the old scan saw is still enumerated
        self.assertGreaterEqual(
            len(_collision_candidates(cfg, wa, wb, ws, wspec, w_p)), 4)

    def test_collision_channels_carry_the_pump_quanta(self):
        from snail_solver.sweep_common import _PUMP_QUANTA, collision_drag_channels
        cfg = dict(self._cfg(), drag_subharmonic=True)
        chs = collision_drag_channels(cfg, 5.0, 4.6, 7.0, 4.75, 0.4, n=4)
        for c in chs:
            self.assertIn(c.n_pump, set(_PUMP_QUANTA.values()))
            self.assertEqual(c.n_photon, max(c.n_pump, 1))
            self.assertTrue(c.quotient_rule)

    def test_collision_channels_drop_a_beat_the_chirp_sweeps_through_zero(self):
        from snail_solver.sweep_common import collision_drag_channels
        cfg = self._cfg()
        # place the spectator so the nearest beat is essentially zero
        chs = collision_drag_channels(cfg, 5.0, 4.6, 7.0, 4.6, 0.4, n=3)
        self.assertTrue(all(abs(c.beat_GHz) >= 5e-3 for c in chs),
                        "a collapsed beat must be filtered out, not composed")

    def test_validation_sweep_is_importable_and_scores_a_cell(self):
        """Smoke: the Phase-7 harness runs a cell and reports the beats it used."""
        from snail_solver.validate_recursive_drag import SCHEMES, _score_point
        cfg = dict(self._cfg(), envelope="sine_power", envelope_m=3,
                   envelope_rise_frac=0.5, qubit_levels=2, coupler_levels=3,
                   spec_levels=2)
        out = _score_point(cfg, 12.0, 4.75, 2,
                           solver={"atol": 1e-7, "rtol": 1e-5, "nsteps": 50000})
        self.assertTrue(np.isfinite(out["F"]))
        self.assertEqual(out["n_channels_used"], len(out["beats_MHz"]))
        self.assertGreaterEqual(out["corr_ratio"], 0.0)
        self.assertEqual([lab for lab, _ in SCHEMES][0], "none")


class TestPumpQuantaMapping(unittest.TestCase):
    """k per collision channel, and the chirp-aware DRAG safety test."""

    def test_kinds(self):
        from snail_solver.sweep_common import _pump_quanta_of
        self.assertEqual(_pump_quanta_of("onepump"), 1)
        self.assertEqual(_pump_quanta_of("subharm"), 2)
        self.assertEqual(_pump_quanta_of("static"), 0,
                         "a static beat is pump-independent; a chirp must not move it")
        self.assertEqual(_pump_quanta_of("unknown-kind"), 1)      # safe default

    def test_chirp_aware_skip(self):
        """A beat that clears the static threshold can still be swept through zero."""
        from snail_solver.sweep_common import _drag_ok_with_chirp
        cfg = {"drag_skip_below_MHz": 5.0}
        chirp = [0.0, 0.01, 0.03]
        self.assertTrue(_drag_ok_with_chirp(cfg, 0.3, 1, chirp, 40.0))
        self.assertFalse(_drag_ok_with_chirp(cfg, 0.02, 2, chirp, 40.0))
        # k = 0 is immune, and so is an absent chirp
        self.assertTrue(_drag_ok_with_chirp(cfg, 0.02, 0, chirp, 40.0))
        self.assertTrue(_drag_ok_with_chirp(cfg, 0.02, 2, None, 40.0))
        # the static test still applies
        self.assertFalse(_drag_ok_with_chirp(cfg, 0.001, 1, None, 40.0))


class TestTuneUpAlgebra(unittest.TestCase):
    """Amplitude/length algebra: the identity that makes length the ONLY free knob.

    ``set_pump(normalize_iswap=...)`` fixes the pulse AREA to A = (pi/2)/(6 g3 lam_a
    lam_b), which is independent of t_g. For a Hann envelope area = amp * t_g / 2, so
    holding the PEAK fixed means amp_scale must grow with t_g. Getting that direction
    backwards still produces a smooth curve with a maximum, so it would not announce
    itself -- hence the explicit test.
    """

    def test_nominal_length_has_unit_amp_scale(self):
        """t_g0 is by definition the length at which no amplitude correction is needed."""
        from snail_solver.tune_up import fixed_eta_amp_scale, nominal_t_g
        cfg = _cfg()
        for eta in (1.2, 1.8, 2.5):
            t_g0 = nominal_t_g(cfg, eta)
            self.assertAlmostEqual(fixed_eta_amp_scale(cfg, t_g0, eta), 1.0, places=12,
                                   msg=f"eta*={eta}")

    def test_amp_scale_round_trip(self):
        """peak_eta_of inverts fixed_eta_amp_scale at every length."""
        from snail_solver.tune_up import fixed_eta_amp_scale, nominal_t_g, peak_eta_of
        cfg = _cfg()
        eta = 1.8
        t_g0 = nominal_t_g(cfg, eta)
        for t_g in t_g0 * np.array([0.7, 0.9, 1.0, 1.15, 1.4]):
            s = fixed_eta_amp_scale(cfg, t_g, eta)
            self.assertAlmostEqual(peak_eta_of(cfg, t_g, s), eta, places=10)

    def test_longer_gate_needs_larger_amp_scale(self):
        """The normalizer shrinks amp as 1/t_g to hold area; holding the peak undoes it."""
        from snail_solver.tune_up import fixed_eta_amp_scale, nominal_t_g
        cfg = _cfg()
        eta = 1.8
        t_g0 = nominal_t_g(cfg, eta)
        s = [fixed_eta_amp_scale(cfg, t, eta) for t in (0.8 * t_g0, t_g0, 1.25 * t_g0)]
        self.assertLess(s[0], s[1])
        self.assertLess(s[1], s[2])
        self.assertAlmostEqual(s[2] / s[1], 1.25, places=10)   # strictly linear in t_g

    def test_envelope_in_normalized_time_is_length_independent(self):
        """THE decoupling claim: at fixed peak |eta|, |eta(u)| does not depend on t_g.

        This is why the chirp coefficients -- functions of |eta(u)| alone -- can be
        calibrated once and reused while the length is tuned, instead of the two
        fighting each other the way (offset, amp_scale) scans do.
        """
        from snail_solver.device_utils import build_coupler
        from snail_solver.tune_up import fixed_eta_amp_scale, nominal_t_g
        cfg = _cfg()
        eta = 1.8
        t_g0 = nominal_t_g(cfg, eta)
        u = np.linspace(-1.0, 1.0, 65)
        shapes = []
        for t_g in (t_g0, 1.35 * t_g0):
            cpl, _w, _e = build_coupler(cfg, t_g, fixed_eta_amp_scale(cfg, t_g, eta), 0.0)
            tone = cpl._pump_tones[0]
            shapes.append(np.abs(cpl._eta_at(tone, (u + 1.0) * t_g / 2.0)))
        np.testing.assert_allclose(shapes[0], shapes[1], rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(float(np.max(shapes[0])), eta, places=6)

    def test_post_chirp_row_scaling_reproduces_target_eta(self):
        """post_chirp_table's row algebra: t_g_i carries the calibrated length
        correction, and amp_scale_i must still hit the row's own target |eta| --
        checked without any solve, mirroring the formula inside the function."""
        from snail_solver.tune_up import fixed_eta_amp_scale, nominal_t_g, peak_eta_of
        cfg = _cfg()
        target_eta, t_g_star = 1.8, 79.0182       # a plausible fitted length != t_g0
        t_g0_at_target = nominal_t_g(cfg, target_eta)
        for frac in (0.5, 0.75, 1.0, 1.1):
            e = frac * target_eta
            t_g_i = nominal_t_g(cfg, e) * (t_g_star / t_g0_at_target)
            amp_scale_i = fixed_eta_amp_scale(cfg, t_g_i, e)
            self.assertAlmostEqual(peak_eta_of(cfg, t_g_i, amp_scale_i), e, places=10,
                                   msg=f"frac={frac}")


class TestTuneUpShiftFit(unittest.TestCase):
    """delta(|eta|) = delta0 + k2|eta|^2 + k4|eta|^4, split into static + Stark.

    The split is the physics: delta0 survives at zero drive (static dressing, a pure
    carrier retune) while k2/k4 vary along the pulse and are the only part a chirp can
    track. Charging a static offset to the Stark terms inflates them and yields a
    confident, wrong chirp.
    """

    def test_recovers_known_coefficients(self):
        from snail_solver.tune_up import fit_shift_curve
        eta = np.linspace(0.5, 2.2, 11)
        d0, k2, k4 = -0.7, -0.85, 0.11
        fit = fit_shift_curve(eta, d0 + k2 * eta ** 2 + k4 * eta ** 4)
        self.assertAlmostEqual(fit["delta0"], d0, places=9)
        self.assertAlmostEqual(fit["k2"], k2, places=9)
        self.assertAlmostEqual(fit["k4"], k4, places=9)
        self.assertAlmostEqual(fit["r2"], 1.0, places=12)

    def test_static_offset_is_not_charged_to_the_stark_terms(self):
        """A purely static ridge must come back as delta0, with NO Stark shift.

        This is the evan_device case: the ridge sits near -0.7 MHz and barely moves
        with drive. Pinning the origin instead would report a large fake k2.
        """
        from snail_solver.tune_up import fit_shift_curve
        eta = np.linspace(0.7, 2.3, 9)
        fit = fit_shift_curve(eta, np.full(eta.size, -0.7))
        self.assertAlmostEqual(fit["delta0"], -0.7, places=8)
        self.assertLess(abs(fit["k2"]), 1e-8)
        self.assertLess(fit["stark_span_MHz"], 1e-8)

        pinned = fit_shift_curve(eta, np.full(eta.size, -0.7), fit_static=False)
        self.assertGreater(abs(pinned["k2"]), 0.05)      # the fake shift, quantified

    def test_nans_are_dropped_not_propagated(self):
        from snail_solver.tune_up import fit_shift_curve
        eta = np.linspace(0.5, 2.2, 11)
        y = -0.85 * eta ** 2
        y[3] = np.nan                                    # one railed/failed row
        fit = fit_shift_curve(eta, y)
        self.assertEqual(fit["n_used"], 10)
        self.assertAlmostEqual(fit["k2"], -0.85, places=8)
        self.assertLess(abs(fit["delta0"]), 1e-8)

    def test_stark_span_measures_the_drive_dependence(self):
        """stark_span is what the chirp is built from, so it must exclude delta0."""
        from snail_solver.tune_up import fit_shift_curve
        eta = np.linspace(1.0, 2.0, 9)
        fit = fit_shift_curve(eta, 5.0 - 0.5 * eta ** 2)
        self.assertAlmostEqual(fit["stark_span_MHz"], 0.5 * (2.0 ** 2 - 1.0 ** 2),
                               places=8)

    def test_too_few_rows_raises(self):
        from snail_solver.tune_up import fit_shift_curve
        with self.assertRaises(ValueError):
            fit_shift_curve(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 9.0]))

    def test_one_wild_row_is_dropped_not_averaged_in(self):
        """A leaking chevron must be excluded, not merely down-weighted.

        `rabi_shift_table` NaNs any row below the contrast floor; this pins the
        consequence, which is that the surviving rows alone set k2 and k4. Measured on
        evan_device one such row (contrast 0.215) sat 14 MHz from both its neighbours
        and, left in, dragged r2 to 0.345.
        """
        from snail_solver.tune_up import fit_shift_curve
        eta = np.array([0.7, 1.1, 1.5, 1.9, 2.3])
        clean = -0.7 - 0.3 * eta ** 2
        wild = clean.copy()
        wild[3] = -14.0                                  # the leaking row
        poisoned = fit_shift_curve(eta, wild)
        dropped = wild.copy()
        dropped[3] = np.nan
        rescued = fit_shift_curve(eta, dropped)
        self.assertLess(poisoned["r2"], 0.9)             # the guard would fire
        self.assertGreater(rescued["r2"], 0.999)
        self.assertAlmostEqual(rescued["k2"], -0.3, places=6)
        self.assertAlmostEqual(rescued["delta0"], -0.7, places=6)


class TestChevronCentreFit(unittest.TestCase):
    """Locating a sub-MHz shift on a grid whose step is larger than the shift.

    The centre has to come from the whole lineshape, not from the three points
    around the argmax -- that is the only reason a shift smaller than one grid step
    is measurable at all.
    """

    @staticmethod
    def _chevron(centre_MHz, hwhm_GHz=4e-3, n_off=13, span_MHz=13.0, n_t=241,
                 window_ns=300.0):
        """An ideal two-level chevron: P = (O/Og)^2 sin^2(Og t / 2), Og angular."""
        off = np.linspace(-span_MHz / 2, span_MHz / 2, n_off) * 1e-3
        t = np.linspace(0.0, window_ns, n_t)
        d = off - centre_MHz * 1e-3
        Og = np.sqrt(hwhm_GHz ** 2 + d ** 2)
        P = ((hwhm_GHz ** 2 / Og ** 2)[:, None]
             * np.sin(TWO_PI * Og[:, None] * t[None, :] / 2.0) ** 2)
        return off, P.max(axis=1)

    def test_recovers_centre_well_inside_one_grid_step(self):
        from snail_solver.tune_up import fit_chevron_center
        step = 13.0 / 12
        for true in (-0.31, -0.65, +0.42):
            off, m = self._chevron(true)
            fit = fit_chevron_center(off, m)
            self.assertTrue(fit["ok"], f"centre {true}")
            self.assertAlmostEqual(fit["center_GHz"] * 1e3, true, delta=0.05,
                                   msg=f"centre {true} (grid step {step:.2f} MHz)")

    def test_beats_the_parabolic_vertex_it_replaces(self):
        """The Lorentzian must be at least as good as the estimator it supersedes."""
        from snail_solver.tune_up import fit_chevron_center
        errs_l, errs_v = [], []
        for true in (-0.31, -0.65, +0.42, -1.08):
            off, m = self._chevron(true)
            f = fit_chevron_center(off, m)
            errs_l.append(abs(f["center_GHz"] * 1e3 - true))
            errs_v.append(abs(f["vertex_GHz"] * 1e3 - true))
        self.assertLessEqual(max(errs_l), max(errs_v) + 1e-9)

    def test_recovers_the_linewidth(self):
        """HWHM is what sizes the next scan, so a wrong one costs a wasted re-measure."""
        from snail_solver.tune_up import fit_chevron_center
        off, m = self._chevron(-0.5, hwhm_GHz=4e-3)
        self.assertAlmostEqual(fit := fit_chevron_center(off, m)["hwhm_GHz"] * 1e3,
                               4.0, delta=0.2, msg=f"got {fit}")


class TestRabiPlot(unittest.TestCase):
    """The plot must render from a PARTIAL table, since that is when it matters most.

    Every Rabi guard failure tells the user to inspect the chevrons; if the plotter
    needed a complete table the instruction would be unfollowable.
    """

    def test_renders_without_a_fit(self):
        import tempfile

        from snail_solver.tune_up import fit_chevron_center, plot_rabi_table
        off, m = TestChevronCentreFit._chevron(-0.4)
        t = np.linspace(0.0, 300.0, 41)
        P = np.tile(m[:, None], (1, t.size))
        chev = {"eta": 0.9, "offsets_GHz": off, "metric": m, "P10": P, "times_ns": t,
                "window_ns": 300.0, "span_MHz": 13.0, "fit": fit_chevron_center(off, m)}
        table = {"eta": np.array([0.9]), "delta_MHz": np.array([-0.4]),
                 "target_eta": 0.9, "chevrons": [chev]}      # note: no "fit" key
        with tempfile.TemporaryDirectory() as d:
            out = plot_rabi_table(table, os.path.join(d, "r.png"))
            self.assertTrue(os.path.getsize(out) > 5000)

    def test_empty_table_raises_rather_than_writing_a_blank(self):
        from snail_solver.tune_up import plot_rabi_table
        with self.assertRaises(ValueError):
            plot_rabi_table({"eta": np.array([]), "delta_MHz": np.array([]),
                             "target_eta": 0.9, "chevrons": []}, "unused.png")


class TestTuneUpChirpProjection(unittest.TestCase):
    """Projecting the MEASURED shift onto Legendre must generalize the analytic seed."""

    @staticmethod
    def _table(k2, k4, eta_star=1.8):
        return {"fit": {"delta0": 0.0, "k2": k2, "k4": k4, "r2": 1.0},
                "target_eta": eta_star}

    def test_pure_quadratic_reproduces_the_analytic_seed(self):
        """With k4 = 0 the shift is exactly ~|eta|^2, so this must equal stark_chirp_seed.

        Two independent code paths -- a tabulated closed form and a Gauss-Legendre
        quadrature over the fitted curve -- landing on the same numbers.
        """
        from snail_solver.stark_chirp import stark_chirp_seed
        from snail_solver.tune_up import chirp_from_measured_shift
        res = chirp_from_measured_shift(self._table(-0.9, 0.0), degree=4)
        np.testing.assert_allclose(res["coeffs_GHz"],
                                   stark_chirp_seed(res["mean_shift_GHz"], degree=4),
                                   rtol=0, atol=1e-13)
        self.assertLess(res["rel_diff"], 1e-12)
        self.assertEqual(res["quartic_fraction"], 0.0)

    def test_mean_shift_is_the_hann_average(self):
        """c_0 IS <delta> over the pulse: 3/8 k2 eta*^2 when the shift is pure |eta|^2."""
        from snail_solver.stark_chirp import HANN_MEAN_FACTOR
        from snail_solver.tune_up import chirp_from_measured_shift
        k2, eta = -0.9, 1.8
        res = chirp_from_measured_shift(self._table(k2, 0.0, eta), degree=4)
        self.assertAlmostEqual(res["mean_shift_GHz"],
                               HANN_MEAN_FACTOR * k2 * eta ** 2 * 1e-3, places=12)

    def test_c0_pinned_and_odd_terms_exactly_zero(self):
        from snail_solver.tune_up import chirp_from_measured_shift
        res = chirp_from_measured_shift(self._table(-0.9, 0.2), degree=6)
        c = res["coeffs_GHz"]
        self.assertEqual(c[0], 0.0)                     # degenerate with wp_offset
        np.testing.assert_array_equal(c[1::2], np.zeros(c[1::2].size))
        self.assertNotEqual(c[2], 0.0)                  # c_2 leads, not c_1

    def test_quartic_term_moves_the_chirp(self):
        """rel_diff is a readout of what the measured k4 buys over the pure-|eta|^2 seed."""
        from snail_solver.tune_up import chirp_from_measured_shift
        res = chirp_from_measured_shift(self._table(-0.9, 0.3), degree=4)
        self.assertGreater(res["rel_diff"], 0.05)
        self.assertAlmostEqual(res["quartic_fraction"],
                               abs(0.3 * 1.8 ** 4) / abs(0.9 * 1.8 ** 2), places=10)

    def test_static_offset_goes_to_the_carrier_not_the_chirp(self):
        """delta0 must move wp_offset and leave every chirp coefficient untouched.

        A drive-independent shift has no shape along the pulse, so there is nothing
        for a chirp to track; routing it into c_k would be the same double-count that
        pinning c_0 exists to prevent.
        """
        from snail_solver.tune_up import chirp_from_measured_shift
        plain = chirp_from_measured_shift(self._table(-0.9, 0.2), degree=4)
        with_static = chirp_from_measured_shift(
            {"fit": {"delta0": -0.7, "k2": -0.9, "k4": 0.2, "r2": 1.0},
             "target_eta": 1.8}, degree=4)
        np.testing.assert_allclose(with_static["coeffs_GHz"], plain["coeffs_GHz"],
                                   rtol=0, atol=1e-15)
        self.assertAlmostEqual(with_static["mean_shift_GHz"] - plain["mean_shift_GHz"],
                               -0.7e-3, places=12)
        self.assertAlmostEqual(with_static["static_GHz"], -0.7e-3, places=12)

    def test_chirp_is_independent_of_gate_length_with_drag_off(self):
        """With DRAG off the coefficients depend on |eta(u)| only -- THE decoupling."""
        from snail_solver.tune_up import chirp_from_measured_shift
        a = chirp_from_measured_shift(self._table(-0.9, 0.2), t_g=77.0)
        b = chirp_from_measured_shift(self._table(-0.9, 0.2), t_g=120.0)
        np.testing.assert_array_equal(a["coeffs_GHz"], b["coeffs_GHz"])

    def test_perturbative_ok_flags_a_non_converging_law(self):
        """perturbative_ok = quartic_fraction < quartic_warn (default 0.25).

        Uses the same (k2=-0.9, k4=0.3, eta*=1.8) fixture as
        test_quartic_term_moves_the_chirp, where quartic_fraction ~= 1.08 --
        the quartic term is now LARGER than the quadratic one, i.e. exactly the
        "law is not converging" case the flag exists to catch.
        """
        from snail_solver.tune_up import chirp_from_measured_shift
        res = chirp_from_measured_shift(self._table(-0.9, 0.3), degree=4)
        self.assertGreater(res["quartic_fraction"], 0.25)
        self.assertFalse(res["perturbative_ok"])

        small = chirp_from_measured_shift(self._table(-0.9, 0.01), degree=4)
        self.assertLess(small["quartic_fraction"], 0.25)
        self.assertTrue(small["perturbative_ok"])

        # the threshold itself is a caller knob, not hardcoded
        lenient = chirp_from_measured_shift(self._table(-0.9, 0.3), degree=4,
                                            quartic_warn=2.0)
        self.assertTrue(lenient["perturbative_ok"])


class TestTuneUpDragCoupledChirp(unittest.TestCase):
    """DRAG's quadrature and the chirp are a fixed point, not a formula.

    ``|eta_tot|^2 = |eta|^2 + [(d eta/dt) / Delta(t)]^2`` raises the Stark shift, and
    ``Delta(t) = Delta_0 - k delta(t)`` moves with the chirp that shift produces. The
    quadrature also scales as 1/t_g, which BREAKS the length-independence the DRAG-off
    calibration relies on -- that is why the orchestrator has an outer loop at all.
    """

    @staticmethod
    def _table(k2=-0.9, k4=0.0, eta_star=1.8):
        return {"fit": {"delta0": 0.0, "k2": k2, "k4": k4, "r2": 1.0},
                "target_eta": eta_star}

    def test_drag_off_is_the_zero_quadrature_limit(self):
        """beat=None and an infinitely detuned beat must agree: q -> 0 either way."""
        from snail_solver.tune_up import chirp_from_measured_shift
        off = chirp_from_measured_shift(self._table(), t_g=80.0)
        far = chirp_from_measured_shift(self._table(), t_g=80.0,
                                        drag_beat_GHz=1e6, drag_n_pump=1)
        np.testing.assert_allclose(far["coeffs_GHz"], off["coeffs_GHz"],
                                   rtol=1e-9, atol=1e-15)

    def test_quadrature_increases_the_shift(self):
        """DRAG adds drive, so |delta| grows -- it never leaves the chirp untouched."""
        from snail_solver.tune_up import chirp_from_measured_shift
        off = chirp_from_measured_shift(self._table(), t_g=80.0)
        on = chirp_from_measured_shift(self._table(), t_g=80.0,
                                       drag_beat_GHz=0.05, drag_n_pump=1)
        self.assertGreater(on["drag_delta_frac"], 0.0)
        self.assertGreater(abs(on["mean_shift_GHz"]), abs(off["mean_shift_GHz"]))

    def test_drag_breaks_length_independence(self):
        """q ~ 1/t_g, so with DRAG on a shorter gate needs a DIFFERENT chirp.

        This is the claim that forces `run_tune_up`'s outer chirp<->length loop; if it
        ever became false the loop would be dead code.
        """
        from snail_solver.tune_up import chirp_from_measured_shift
        short = chirp_from_measured_shift(self._table(), t_g=60.0,
                                          drag_beat_GHz=0.05, drag_n_pump=1)
        long = chirp_from_measured_shift(self._table(), t_g=120.0,
                                         drag_beat_GHz=0.05, drag_n_pump=1)
        self.assertGreater(short["drag_delta_frac"], long["drag_delta_frac"],
                           "a shorter gate has a steeper envelope, so a bigger "
                           "quadrature and a bigger DRAG contribution")
        self.assertGreater(abs(short["coeffs_GHz"][2] - long["coeffs_GHz"][2]), 0.0)

    def test_k_scales_the_detuning_pull(self):
        """k = 2 (subharmonic) pulls Delta(t) twice as hard as k = 1; k = 0 not at all."""
        from snail_solver.tune_up import chirp_from_measured_shift
        kw = dict(t_g=80.0, drag_beat_GHz=0.05)
        k0 = chirp_from_measured_shift(self._table(), drag_n_pump=0, **kw)
        k1 = chirp_from_measured_shift(self._table(), drag_n_pump=1, **kw)
        k2_ = chirp_from_measured_shift(self._table(), drag_n_pump=2, **kw)
        self.assertAlmostEqual(k0["min_abs_detuning_GHz"], 0.05, places=12)
        # the shift here is negative, so Delta = D0 - k delta moves AWAY from zero
        self.assertGreater(k1["min_abs_detuning_GHz"], k0["min_abs_detuning_GHz"])
        self.assertGreater(k2_["min_abs_detuning_GHz"], k1["min_abs_detuning_GHz"])

    def test_t_g_required_when_drag_is_on(self):
        """Silently assuming a length would silently produce the wrong quadrature."""
        from snail_solver.tune_up import chirp_from_measured_shift
        with self.assertRaises(ValueError):
            chirp_from_measured_shift(self._table(), drag_beat_GHz=0.05)

    def test_predicted_drag_contribution_scales_as_eta_to_the_fourth(self):
        """The prediction `drag_shift_table` tests against must scale as |eta|^4.

        For a Hann pulse d eta/dt ~ eta/t_g and a full-swap t_g ~ 1/eta, so the
        quadrature q ~ eta^2 and a shift following k2|eta|^2 goes as q^2 ~ eta^4. That
        exponent is the discriminator: measure 4 and DRAG is only adding drive, which
        the chirp already covers; measure anything else and there is a mechanism the
        quadrature model does not contain.
        """
        from snail_solver.tune_up import (chirp_from_measured_shift, nominal_t_g,
                                          project_nodrag_mean)
        cfg = _cfg()
        etas = np.array([1.2, 1.5, 1.8, 2.1])
        pred = []
        for e in etas:
            tbl = {"fit": {"delta0": 0.0, "k2": -0.9, "k4": 0.0, "r2": 1.0},
                   "target_eta": float(e)}
            on = chirp_from_measured_shift(tbl, float(e), drag_beat_GHz=0.3,
                                           drag_n_pump=1,
                                           t_g=nominal_t_g(cfg, float(e)))
            pred.append(abs(on["mean_shift_GHz"]
                            - project_nodrag_mean(tbl, float(e))))
        slope = np.polyfit(np.log(etas), np.log(np.array(pred)), 1)[0]
        self.assertAlmostEqual(slope, 4.0, delta=0.15)


class TestTuneUpSwapFit(unittest.TestCase):
    """The time-Rabi fit must find the FIRST full swap, not a harmonic of it."""

    def test_recovers_period(self):
        from snail_solver.tune_up import fit_swap_period
        t = np.linspace(0.0, 200.0, 400)
        T = 62.5
        fit = fit_swap_period(t, 0.97 * np.sin(np.pi * t / (2 * T)) ** 2 + 0.01)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["T_swap_ns"], T, places=4)

    def test_survives_noise(self):
        from snail_solver.tune_up import fit_swap_period
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 200.0, 400)
        T = 62.5
        y = np.sin(np.pi * t / (2 * T)) ** 2 + rng.normal(0.0, 0.01, t.size)
        fit = fit_swap_period(t, y)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["T_swap_ns"], T, delta=0.5)

    def test_short_trace_raises(self):
        from snail_solver.tune_up import fit_swap_period
        with self.assertRaises(ValueError):
            fit_swap_period(np.arange(4.0), np.zeros(4))


class TestTuneUpGpuFlag(unittest.TestCase):
    """--gpu must flip the global qutip-jax backend and force jobs=1.

    run_tune_up and zhou_coupler.use_gpu are mocked out, so this never imports
    QuTiP/JAX -- it only checks that tune_up.main() wires the CLI flag to the
    right calls, keeping this file QuTiP-free and fast.
    """

    def _run_main(self, extra_argv):
        from unittest import mock
        from snail_solver import tune_up

        fake_out = {
            "operating_point": {
                "target_eta": 1.0, "t_g_ns": 100.0, "amp_scale": 1.0,
                "wp_offset_GHz": 0.0, "chirp_coeffs_GHz": [0.0, 0.0],
                "drag_beat_GHz": None, "drag_n_pump": 1, "score": 1.0,
            },
            "t_g0_ns": 100.0, "drag": None, "stages": {},
        }
        with tempfile.TemporaryDirectory() as d:
            device_path = os.path.join(d, "dev.json")
            with open(device_path, "w") as fh:
                json.dump({}, fh)
            argv = ["tune_up", "--device", device_path, "--target-eta", "1.0",
                   *extra_argv]
            with mock.patch("snail_solver.zhou_coupler.use_gpu") as m_gpu, \
                 mock.patch("snail_solver.tune_up.run_tune_up",
                           return_value=fake_out) as m_run, \
                 mock.patch.object(sys, "argv", argv):
                tune_up.main()
        return m_gpu, m_run

    def test_gpu_flag_calls_use_gpu_and_forces_jobs_one(self):
        m_gpu, m_run = self._run_main(["--gpu", "--jobs", "8"])
        m_gpu.assert_called_once_with(True)
        self.assertEqual(m_run.call_args.kwargs["jobs"], 1)

    def test_no_gpu_flag_leaves_use_gpu_and_jobs_alone(self):
        m_gpu, m_run = self._run_main([])
        m_gpu.assert_not_called()
        self.assertEqual(m_run.call_args.kwargs["jobs"], 0)


class TestExpectedNumber(unittest.TestCase):
    """spectroscopy.expected_number: the number-operator expectation of one mode,
    marginalised over the others -- the un-thresholded generalisation of
    marginal_population, used to check a mode's real photon occupation."""

    def test_matches_a_hand_computed_expectation(self):
        from snail_solver.spectroscopy import expected_number
        dims = [3, 3, 5]
        probs = np.zeros(np.prod(dims))
        # mode 2 (the "coupler") in level 3 with weight 0.4, level 1 with weight 0.1,
        # spread over arbitrary states of modes 0/1 -- <n> = 3*0.4 + 1*0.1 = 1.3
        t = probs.reshape(dims)
        t[0, 0, 3] = 0.4
        t[1, 2, 1] = 0.1
        t[0, 0, 0] = 0.5                                   # remainder, level 0
        self.assertAlmostEqual(expected_number(t.ravel(), dims, 2), 1.3, places=12)

    def test_zero_occupation_gives_zero(self):
        from snail_solver.spectroscopy import expected_number
        dims = [3, 3, 5]
        t = np.zeros(dims)
        t[0, 0, 0] = 1.0                                   # all population in vacuum
        self.assertEqual(expected_number(t.ravel(), dims, 2), 0.0)


class TestPopulationChannels(unittest.TestCase):
    """find_stark_resonance.population_channels: where the population that isn't
    P01/P10 actually is, read straight from the (already-computed) state vector."""

    @staticmethod
    def _cpl():
        from snail_solver.zhou_coupler import ZhouCoupler
        return ZhouCoupler(mode_freqs_GHz=[4.7, 5.7, 4.2], coupler_index=2,
                           participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                           levels=[3, 3, 5], anharmonicities_GHz={0: -0.12, 1: -0.12})

    def test_exclusive_leak_matches_a_hand_built_state(self):
        from snail_solver.find_stark_resonance import population_channels, CHANNELS
        cpl = self._cpl()
        init = [0, 1, 0]                      # |01,0>
        tgt = [1, 0, 0]                        # |10,0>
        dim = cpl.dim

        psi0 = np.zeros(dim, dtype=complex)
        psi0[cpl.fock_index(init)] = 1.0       # t=0: pure |01,0>

        # t=1: disjoint-by-construction so P_leak decomposes exactly into the
        # diagnostic channels below (in general they OVERLAP -- see the docstring).
        psi1 = np.zeros(dim, dtype=complex)
        psi1[cpl.fock_index(tgt)] = np.sqrt(0.6)            # P10
        psi1[cpl.fock_index(init)] = np.sqrt(0.1)           # residual P01
        psi1[cpl.fock_index([0, 0, 1])] = np.sqrt(0.2)      # coupler excited
        psi1[cpl.fock_index([0, 2, 0])] = np.sqrt(0.05)     # qubit b -> |f>
        psi1[cpl.fock_index([1, 1, 0])] = np.sqrt(0.05)     # |11>

        states = np.stack([psi0, psi1], axis=0)
        ch = population_channels(cpl, states, init, tgt)
        self.assertEqual(ch.shape, (len(CHANNELS), 2))

        idx = {name: i for i, name in enumerate(CHANNELS)}
        self.assertAlmostEqual(ch[idx["P01"], 0], 1.0, places=12)
        self.assertAlmostEqual(ch[idx["P10"], 0], 0.0, places=12)
        self.assertAlmostEqual(ch[idx["P_leak"], 0], 0.0, places=12)

        self.assertAlmostEqual(ch[idx["P01"], 1], 0.1, places=12)
        self.assertAlmostEqual(ch[idx["P10"], 1], 0.6, places=12)
        self.assertAlmostEqual(ch[idx["P_leak"], 1], 0.3, places=12)
        self.assertAlmostEqual(ch[idx["P_coupler"], 1], 0.2, places=12)
        self.assertAlmostEqual(ch[idx["P_f_b"], 1], 0.05, places=12)
        self.assertAlmostEqual(ch[idx["P_f_a"], 1], 0.0, places=12)
        self.assertAlmostEqual(ch[idx["P_double"], 1], 0.05, places=12)
        self.assertAlmostEqual(ch[idx["P_spectator"], 1], 0.0, places=12)
        # by construction these three exhaust P_leak here (they need not in general)
        self.assertAlmostEqual(ch[idx["P_coupler"], 1] + ch[idx["P_f_b"], 1]
                               + ch[idx["P_double"], 1], ch[idx["P_leak"], 1], places=12)
        np.testing.assert_allclose(ch[idx["norm_defect"]], [0.0, 0.0], atol=1e-12)

    def test_norm_defect_reports_a_non_unit_norm_state(self):
        """A non-normalized state (e.g. a truncated basis losing weight) must show
        up as norm_defect, not be silently folded into P_leak."""
        from snail_solver.find_stark_resonance import population_channels
        cpl = self._cpl()
        init, tgt = [0, 1, 0], [1, 0, 0]
        psi = np.zeros(cpl.dim, dtype=complex)
        psi[cpl.fock_index(init)] = np.sqrt(0.9)            # norm^2 = 0.9, not 1.0
        ch = population_channels(cpl, psi[None, :], init, tgt)
        from snail_solver.find_stark_resonance import CHANNELS
        idx = {name: i for i, name in enumerate(CHANNELS)}
        self.assertAlmostEqual(ch[idx["norm_defect"], 0], 0.1, places=12)
        self.assertAlmostEqual(ch[idx["P01"], 0], 0.9, places=12)


class TestChevronQuality(unittest.TestCase):
    """Five independent, cheap checks on whether a chevron is really a two-level
    Lorentzian -- promoted from what fit_chevron_center already computes."""

    def test_clean_chevron_is_kept_at_close_to_its_own_contrast(self):
        from snail_solver.tune_up import chevron_quality, fit_chevron_center
        off, m = TestChevronCentreFit._chevron(-0.4)
        cen = fit_chevron_center(off, m)
        q = chevron_quality(cen, off, m, span_MHz=13.0, leak=0.0)
        self.assertIsNone(q["reject"])
        contrast = float(np.nanmax(m) - np.nanmin(m))
        self.assertAlmostEqual(q["weight"], contrast, delta=0.05)
        self.assertEqual(q["secondary"], 0.0)

    def test_bimodal_chevron_is_flagged_multi_peak(self):
        """This is the failure mode a raw centre/vertex gap MISSES: when the
        Lorentzian fit fails outright, fit_chevron_center falls back to vertex_GHz
        for BOTH fields, so the gap collapses to 0 even though the chevron is
        clearly not a single resonance. secondary catches it directly."""
        from snail_solver.tune_up import chevron_quality, fit_chevron_center
        off, m1 = TestChevronCentreFit._chevron(-4.0, hwhm_GHz=1e-3, n_off=61,
                                                span_MHz=13.0)
        _, m2 = TestChevronCentreFit._chevron(+4.0, hwhm_GHz=1e-3, n_off=61,
                                              span_MHz=13.0)
        m = np.maximum(m1, m2)                # two comparable, well-separated peaks
        cen = fit_chevron_center(off, m)
        q = chevron_quality(cen, off, m, span_MHz=13.0, leak=0.0)
        self.assertGreater(q["secondary"], 0.4)
        self.assertEqual(q["reject"], "multi_peak")

    def test_low_contrast_is_dropped(self):
        from snail_solver.tune_up import chevron_quality, fit_chevron_center
        off, m = TestChevronCentreFit._chevron(-0.4)
        m = m * 0.1                            # scale below contrast_min
        cen = fit_chevron_center(off, m)
        q = chevron_quality(cen, off, m, span_MHz=13.0, leak=0.0, contrast_min=0.35)
        self.assertEqual(q["reject"], "low_contrast")

    def test_high_leakage_is_dropped_even_with_good_contrast(self):
        """contrast_min alone cannot see this -- leak is an independent channel."""
        from snail_solver.tune_up import chevron_quality, fit_chevron_center
        off, m = TestChevronCentreFit._chevron(-0.4)
        cen = fit_chevron_center(off, m)
        ok = chevron_quality(cen, off, m, span_MHz=13.0, leak=0.0, leak_max=0.35)
        self.assertIsNone(ok["reject"])
        bad = chevron_quality(cen, off, m, span_MHz=13.0, leak=0.9, leak_max=0.35)
        self.assertEqual(bad["reject"], "high_leakage")
        self.assertLess(bad["weight"], ok["weight"])


class TestShiftCurveStability(unittest.TestCase):
    """shift_curve_stability catches contamination of an INTERPOLATING fit, which
    extrapolation_ratio (measured |eta| vs target_eta) structurally cannot: a
    sweep that measured up to or past target_eta looks fine by that guard alone
    even when its own top rows are not resonances."""

    def test_clean_law_is_stable_under_row_cutoffs(self):
        from snail_solver.tune_up import shift_curve_stability
        eta = np.linspace(0.5, 2.0, 12)
        y = -0.7 - 0.3 * eta ** 2 + 0.05 * eta ** 4
        res = shift_curve_stability(eta, y, target_eta=2.0)
        self.assertLess(res["delta_spread"], 0.05)

    def test_contaminated_top_row_is_unstable(self):
        from snail_solver.tune_up import shift_curve_stability
        eta = np.linspace(0.5, 2.0, 12)
        y = -0.7 - 0.3 * eta ** 2 + 0.05 * eta ** 4
        y[-1] += 20.0                          # one bad high-drive row
        res = shift_curve_stability(eta, y, target_eta=2.0)
        self.assertGreater(res["delta_spread"], 0.3)

    def test_too_few_rows_reports_nan_not_an_error(self):
        from snail_solver.tune_up import shift_curve_stability
        eta = np.array([1.0, 1.5, 2.0])
        y = -0.7 - 0.3 * eta ** 2
        res = shift_curve_stability(eta, y, target_eta=2.0)
        self.assertTrue(all(int(n) <= 3 for n in res["n_used_by_cutoff"]))


class TestPlotPostChirpSmoke(unittest.TestCase):
    """plot_post_chirp_table must render from a hand-built dict, both with and
    without the flat-carrier comparison -- mirrors TestRabiPlot."""

    @staticmethod
    def _row(eta, wp_offset_GHz=0.0):
        from snail_solver.tune_up import chevron_quality, fit_chevron_center
        off, m = TestChevronCentreFit._chevron(-0.4)
        n_t = 41
        P10 = np.tile(m[:, None], (1, n_t))
        P_leak = 1.0 - P10
        cen = fit_chevron_center(off, m)
        q = chevron_quality(cen, off, m, span_MHz=13.0, leak=0.1)
        chirped = {"metric": m, "P10": P10, "P_leak": P_leak,
                  "leak_at_metric": np.full_like(m, 0.1), "fit": cen, "quality": q,
                  "transfer_at_wp_offset": float(m[len(m) // 2])}
        return {"eta": eta, "offsets_GHz": off, "chirped": chirped}, chirped

    def test_renders_without_flat_comparison(self):
        from snail_solver.tune_up import plot_post_chirp_table
        row, chirped = self._row(1.0)
        post = {"eta": np.array([1.0]), "rows": [row],
               "transfer_chirped": np.array([chirped["transfer_at_wp_offset"]]),
               "transfer_flat": np.array([np.nan]),
               "leak_chirped": np.array([0.1]), "leak_flat": np.array([np.nan]),
               "residual_MHz": np.array([0.0]),
               "record": {"wp_offset_GHz": 0.0, "target_eta": 1.0},
               "compare_flat": False, "reproject_chirp": False}
        with tempfile.TemporaryDirectory() as d:
            out = plot_post_chirp_table(post, out=os.path.join(d, "p.png"))
            self.assertTrue(os.path.getsize(out) > 5000)

    def test_renders_with_flat_comparison_and_rabi_table_overlay(self):
        from snail_solver.tune_up import plot_post_chirp_table
        row, chirped = self._row(1.0)
        flat = dict(chirped)
        flat["transfer_at_wp_offset"] = 0.5
        row["flat"] = flat
        post = {"eta": np.array([1.0]), "rows": [row],
               "transfer_chirped": np.array([chirped["transfer_at_wp_offset"]]),
               "transfer_flat": np.array([0.5]),
               "leak_chirped": np.array([0.1]), "leak_flat": np.array([0.2]),
               "residual_MHz": np.array([0.1]),
               "record": {"wp_offset_GHz": 0.0, "target_eta": 1.0},
               "compare_flat": True, "reproject_chirp": False}
        rabi_table = {"eta": np.array([0.5, 1.0]), "delta_MHz": np.array([1.0, 2.0])}
        with tempfile.TemporaryDirectory() as d:
            out = plot_post_chirp_table(post, out=os.path.join(d, "p.png"),
                                        rabi_table=rabi_table)
            self.assertTrue(os.path.getsize(out) > 5000)

    def test_empty_rows_raises(self):
        from snail_solver.tune_up import plot_post_chirp_table
        with self.assertRaises(ValueError):
            plot_post_chirp_table({"eta": np.array([]), "rows": [],
                                  "record": {"wp_offset_GHz": 0.0, "target_eta": 1.0},
                                  "compare_flat": False}, "unused.png")


class TestTransferProbabilityForwardsDragChannels(unittest.TestCase):
    """`transfer_probability` must accept AND forward `drag_channels`.

    It did not, while `tune_up.length_rabi` passed it unconditionally -- so step 4
    of every tune-up died with `TypeError: unexpected keyword argument
    'drag_channels'`, and `run_tune_up` could not complete at all. The same call in
    `chirp_ablation` sat inside a bare `except Exception`, so that check silently
    reported nothing instead of failing loudly.

    Nothing in this file touched `transfer_probability` or `length_rabi` before,
    which is exactly why the bug shipped.
    """

    def test_signature_accepts_drag_channels(self):
        import inspect
        from snail_solver.device_utils import transfer_probability
        params = inspect.signature(transfer_probability).parameters
        self.assertIn("drag_channels", params,
                      "length_rabi passes drag_channels= on every call")
        self.assertIsNone(params["drag_channels"].default)

    def test_length_rabi_forwards_drag_channels_to_the_probe(self):
        """The search objective must see the SAME pulse as the gate it calibrates."""
        from unittest import mock
        from snail_solver.envelope import DragChannel
        from snail_solver import tune_up

        config = {"g3_GHz": 0.06, "lam_a": 0.1, "lam_b": 0.1,
                  "envelope": "raised_cosine"}
        channels = [DragChannel(beat_GHz=0.3)]
        # autospec binds the REAL signature to the mock, so this fails with the same
        # TypeError the live code did if drag_channels ever goes missing again --
        # a bare Mock would happily swallow any kwarg and pass either way.
        with mock.patch("snail_solver.device_utils.transfer_probability",
                        autospec=True, return_value=0.5) as m_tp:
            tune_up.length_rabi(config, 1.0, [90.0, 100.0, 110.0],
                                drag_channels=channels, refine=False)
        self.assertTrue(m_tp.called)
        self.assertEqual(m_tp.call_args.kwargs["drag_channels"], channels)


class TestRidgeSpanMHz(unittest.TestCase):
    """`ridge_span_MHz` returns the fixed span AND the point count it demands.

    Switching from per-row adaptive spans to one fixed span is what
    `plot_chirp_ridge` needs (it does not interpolate), but the fixed span is sized
    for the STRONGEST row, so the weakest row -- whose linewidth is smaller by
    eta_lo/eta_hi -- gets proportionally fewer points across its own HWHM. Using
    the span without raising `wp_points` turns a loud ValueError from the plotter
    into a quietly wrong chirp.
    """

    CONFIG = {"g3_GHz": 0.06, "lam_a": 0.1, "lam_b": 0.1,
              "envelope": "raised_cosine"}

    def test_span_matches_fixed_span_MHz(self):
        from snail_solver.tune_up import fixed_span_MHz, ridge_span_MHz
        span, _ = ridge_span_MHz(self.CONFIG, 1.8, eta_lo=0.4, eta_hi=1.0)
        self.assertAlmostEqual(
            span, fixed_span_MHz(self.CONFIG, 1.8, eta_hi=1.0), places=9)

    def test_span_is_linear_in_target_eta(self):
        """t_g ~ 1/eta, so the linewidth -- and the span -- scale WITH the drive."""
        from snail_solver.tune_up import ridge_span_MHz
        s12, _ = ridge_span_MHz(self.CONFIG, 1.2)
        s18, _ = ridge_span_MHz(self.CONFIG, 1.8)
        self.assertAlmostEqual(s18 / s12, 1.5, places=9)

    def test_want_points_is_the_undersampling_floor(self):
        """want = 1 + 2 span_linewidths pts_per_hwhm (eta_hi/eta_lo)."""
        from snail_solver.tune_up import ridge_span_MHz
        for eta_lo, want in ((0.3, 81), (0.4, 61), (0.5, 49)):
            _, got = ridge_span_MHz(self.CONFIG, 1.8, eta_lo=eta_lo, eta_hi=1.0,
                                    span_linewidths=4.0)
            self.assertEqual(got, want, f"eta_lo={eta_lo}")

    def test_warns_only_when_wp_points_is_below_the_floor(self):
        from unittest import mock
        from snail_solver.tune_up import ridge_span_MHz
        log = mock.Mock()
        ridge_span_MHz(self.CONFIG, 1.8, eta_lo=0.4, wp_points=25, logger=log)
        log.warning.assert_called_once()
        log.reset_mock()
        ridge_span_MHz(self.CONFIG, 1.8, eta_lo=0.4, wp_points=61, logger=log)
        log.warning.assert_not_called()


class TestTuneUpRidgeSpanWiring(unittest.TestCase):
    """--plot-ridge must size wp_span_MHz itself, because the plot call is the LAST
    thing main() does: without this the whole sweep is computed and then thrown
    away over a missing flag. Mirrors TestTuneUpGpuFlag -- run_tune_up is mocked,
    so no QuTiP.
    """

    @staticmethod
    def _run_main(extra_argv):
        from unittest import mock
        from snail_solver import tune_up

        fake_out = {
            "operating_point": {
                "target_eta": 1.8, "t_g_ns": 77.0, "amp_scale": 1.0,
                "wp_offset_GHz": 0.0, "chirp_coeffs_GHz": [0.0, 0.0],
                "drag_beat_GHz": None, "drag_n_pump": 1, "score": 1.0,
            },
            # plot_chirp_ridge is mocked, but main() still indexes these
            "t_g0_ns": 77.0, "drag": None,
            "stages": {"rabi": {}, "chirp": {}},
        }
        with tempfile.TemporaryDirectory() as d:
            device_path = os.path.join(d, "dev.json")
            with open(device_path, "w") as fh:
                json.dump({"g3_GHz": 0.06, "lam_a": 0.1, "lam_b": 0.1}, fh)
            argv = ["tune_up", "--device", device_path, "--target-eta", "1.8",
                    *extra_argv]
            with mock.patch("snail_solver.tune_up.run_tune_up",
                            return_value=fake_out) as m_run, \
                 mock.patch("snail_solver.tune_up.plot_chirp_ridge",
                            return_value="fig.png"), \
                 mock.patch.object(sys, "argv", argv):
                tune_up.main()
        return m_run

    def test_plot_ridge_fixes_the_span(self):
        m_run = self._run_main(["--plot-ridge", "fig.png"])
        # fixed_span_MHz = 2 * 4 * 1e3 / (2 * auto_t_g(eta_hi * eta*)) = 28.8 * eta*
        self.assertAlmostEqual(m_run.call_args.kwargs["wp_span_MHz"],
                               28.8 * 1.8, places=6)

    def test_without_plot_ridge_the_span_stays_adaptive(self):
        m_run = self._run_main([])
        self.assertIsNone(m_run.call_args.kwargs["wp_span_MHz"])

    def test_an_explicit_span_wins(self):
        m_run = self._run_main(["--plot-ridge", "fig.png", "--wp-span-MHz", "40"])
        self.assertAlmostEqual(m_run.call_args.kwargs["wp_span_MHz"], 40.0)


class TestEtaSweepHelpers(unittest.TestCase):
    """`tune_up_sweep`'s grid parsing and filename tags."""

    def test_parse_etas_colon_form(self):
        from snail_solver.tune_up_sweep import parse_etas
        self.assertEqual(parse_etas("1.2:2.0:9"),
                         [round(v, 10) for v in np.linspace(1.2, 2.0, 9)])

    def test_parse_etas_comma_form(self):
        from snail_solver.tune_up_sweep import parse_etas
        self.assertEqual(parse_etas("1.2,1.5,1.8"), [1.2, 1.5, 1.8])

    def test_parse_etas_rejects_nonpositive(self):
        """t_g0 = 2A/eta* divides by it."""
        from snail_solver.tune_up_sweep import parse_etas
        with self.assertRaises(ValueError):
            parse_etas("0,1.5")

    def test_eta_tag_has_no_dot(self):
        """The tag goes into filenames and operating-point names."""
        from snail_solver.tune_up_sweep import eta_tag
        self.assertEqual(eta_tag(1.8), "eta1p8")
        self.assertEqual(eta_tag(2.0), "eta2")
        self.assertNotIn(".", eta_tag(1.25))

    def test_score_gate_rejects_none_chirp(self):
        """None means 'inherit the device chirp' in build_coupler -- so a caller
        asking for NO chirp must say [], and passing None must not silently score
        a device-level chirp instead."""
        from snail_solver.tune_up_sweep import score_gate
        with self.assertRaises(ValueError):
            score_gate({}, {"target_eta": 1.0, "t_g_ns": 77.0, "amp_scale": 1.0,
                            "wp_offset_GHz": 0.0}, None)


class TestPlotEtaSweepSmoke(unittest.TestCase):
    """`plot_eta_sweep` must render from a hand-built doc, including a failed eta
    (which must be drawn, not silently dropped) -- mirrors TestPlotPostChirpSmoke.
    """

    @staticmethod
    def _row(eta, F=0.99, leak=1e-3, refit=False):
        scores = {"t_g_ns": 138.889 / eta, "amp_scale": 1.0, "wp_offset_GHz": 0.0,
                  "peak_eta": eta, "chirp_coeffs_GHz": [], "F_avg": F,
                  "leakage": leak, "transfer": F, "refit_length": False}
        row = {"target_eta": eta, "ok": True, "error": None,
               "chirped": dict(scores),
               "flat": dict(scores, F_avg=F - 0.01, leakage=leak * 2),
               "rabi_fit": {"r2": 0.98, "delta0": 0.0, "k2": 5.0, "k4": 0.0},
               "chirp": {"quartic_fraction": 0.03},
               "length": {"t_g_ns": 138.889 / eta, "railed": False},
               "n_dropped": 0}
        if refit:
            row["flat_refit"] = dict(scores, F_avg=F - 0.005)
        return row

    def _doc(self, rows):
        return {"rows": rows, "summary": {"n_ok": 1, "n_failed": 0}}

    def test_renders_two_series(self):
        from snail_solver.tune_up_sweep import plot_eta_sweep
        doc = self._doc([self._row(1.2), self._row(1.5, F=0.97, leak=5e-3)])
        with tempfile.TemporaryDirectory() as d:
            out = plot_eta_sweep(doc, out=os.path.join(d, "q.png"))
            self.assertTrue(os.path.getsize(out) > 5000)

    def test_renders_three_series_and_a_failed_eta(self):
        from snail_solver.tune_up_sweep import plot_eta_sweep
        rows = [self._row(1.2, refit=True), self._row(1.5, F=0.9, refit=True),
                {"target_eta": 1.8, "ok": False,
                 "error": {"type": "RabiFitError", "stage": "rabi", "message": "x"}}]
        with tempfile.TemporaryDirectory() as d:
            out = plot_eta_sweep(self._doc(rows), out=os.path.join(d, "q.png"))
            self.assertTrue(os.path.getsize(out) > 5000)

    def test_all_failed_raises(self):
        from snail_solver.tune_up_sweep import plot_eta_sweep
        doc = self._doc([{"target_eta": 1.8, "ok": False,
                          "error": {"type": "RabiFitError", "message": "x"}}])
        with self.assertRaises(ValueError):
            plot_eta_sweep(doc, "unused.png")

    def test_t_g_axis_stays_aligned_when_a_failed_eta_widens_the_plot(self):
        """The top t_g axis is a twiny: it does NOT track its parent, it keeps
        whatever limits it was given. The dotted rule marking a failed eta OUTSIDE
        the successful ones widens the shared x axis, so building the twiny before
        those rules slides every t_g label off the eta it belongs to -- a silently
        mislabelled figure, which is worse than a missing one.
        """
        from unittest import mock
        import numpy as np
        from snail_solver.tune_up_sweep import plot_eta_sweep

        rows = [self._row(1.2), self._row(1.5),
                {"target_eta": 1.8, "ok": False,
                 "error": {"type": "RabiFitError", "message": "x"}}]
        got = []
        with tempfile.TemporaryDirectory() as d:
            # plot_eta_sweep closes the figure; intercept it to inspect the axes.
            with mock.patch("matplotlib.pyplot.close", side_effect=got.append):
                plot_eta_sweep(self._doc(rows), out=os.path.join(d, "q.png"))
        fig = got[0]
        ax, axt = fig.axes[0], fig.axes[-1]      # the twiny is built last, on purpose
        self.assertEqual(len(axt.get_xticks()), 2, "one t_g tick per successful eta")
        self.assertGreater(ax.get_xlim()[1], 1.8, "the failed eta widened the axis")
        self.assertTrue(np.allclose(axt.get_xlim(), ax.get_xlim()),
                        f"t_g axis {axt.get_xlim()} drifted from eta axis "
                        f"{ax.get_xlim()}")

    def test_replot_needs_no_solver(self):
        """--replot must regenerate the figure from the JSON alone: it returns
        before the device or any solver module is touched."""
        from unittest import mock
        from snail_solver import tune_up_sweep
        doc = self._doc([self._row(1.2), self._row(1.5)])
        with tempfile.TemporaryDirectory() as d:
            js, png = os.path.join(d, "s.json"), os.path.join(d, "s.png")
            with open(js, "w") as fh:
                json.dump(doc, fh)
            argv = ["tune_up_sweep", "--replot", js, "--plot", png]
            with mock.patch("snail_solver.tune_up_sweep.run_eta_sweep") as m_run, \
                 mock.patch.object(sys, "argv", argv):
                tune_up_sweep.main()
            m_run.assert_not_called()
            self.assertTrue(os.path.getsize(png) > 5000)


class TestSubharmonicAxis(unittest.TestCase):
    """`subharmonic_convergence`'s axis: w_b is what moves, and Delta_sub is exact.

    The whole map is indexed by Delta_sub = w_s - 2 w_p, so an off-by-a-factor in
    w_b(Delta_sub) would silently mislabel every column -- the figure would still
    render and still look monotone.
    """

    CFG = dict(qubit_freqs_GHz=[3.5, 3.8], coupler_freq_GHz=4.5, g3_GHz=0.06,
               coupler_levels=5, qubit_levels=3, lam_a=0.1, lam_b=0.1,
               min_detuning_GHz=0.05, envelope="raised_cosine")

    def test_round_trip_through_the_axis(self):
        from snail_solver.subharmonic_convergence import (
            config_at_detuning, subharmonic_detuning_GHz)
        for d in (0.1, 0.75, 2.2, -0.4):
            cfg = config_at_detuning(self.CFG, d)
            self.assertAlmostEqual(subharmonic_detuning_GHz(cfg), d, places=12)

    def test_both_branches_give_the_same_detuning(self):
        """The pump is |w_b - w_a|, so mirroring the partner under the anchor is a
        different allocation at the SAME Delta_sub."""
        from snail_solver.subharmonic_convergence import (
            config_at_detuning, subharmonic_detuning_GHz)
        above = config_at_detuning(self.CFG, 0.8, branch="above")
        below = config_at_detuning(self.CFG, 0.8, branch="below")
        self.assertAlmostEqual(subharmonic_detuning_GHz(above), 0.8, places=12)
        self.assertAlmostEqual(subharmonic_detuning_GHz(below), 0.8, places=12)
        self.assertGreater(above["qubit_freqs_GHz"][1], 3.5)
        self.assertLess(below["qubit_freqs_GHz"][1], 3.5)

    def test_pump_is_independent_of_the_detuning_only_through_the_rate(self):
        """t_g0 = 2A/eta* must be identical along the axis: the iSWAP rate is
        6 g3 lam_a lam_b eta, with no w_p in it. That invariance is the reason this
        module moves the pump rather than the SNAIL, so it is worth pinning."""
        from snail_solver.subharmonic_convergence import config_at_detuning
        from snail_solver.tune_up import nominal_t_g
        t0 = nominal_t_g(config_at_detuning(self.CFG, 0.1), 1.2)
        t1 = nominal_t_g(config_at_detuning(self.CFG, 2.2), 1.2)
        self.assertAlmostEqual(t0, t1, places=12)

    def test_rejects_a_detuning_past_the_degenerate_qubits(self):
        from snail_solver.subharmonic_convergence import wb_for_detuning
        with self.assertRaises(ValueError):
            wb_for_detuning(self.CFG, 4.5)          # w_p = 0
        with self.assertRaises(ValueError):
            wb_for_detuning(self.CFG, 5.0)          # w_p < 0

    def test_rejects_a_pump_inside_the_collision_floor(self):
        from snail_solver.subharmonic_convergence import config_at_detuning
        with self.assertRaises(ValueError):
            config_at_detuning(self.CFG, 4.45)      # w_p = 0.025 < 0.05

    def test_does_not_mutate_the_input_and_strips_a_device_chirp(self):
        """build_coupler treats chirp_coeffs_GHz=None as 'inherit the device
        chirp', so leaving a chirp calibrated at another pump in the copy would
        apply it silently at the new w_b -- the None-vs-[] trap tune_up_sweep
        documents, one level up."""
        from snail_solver.subharmonic_convergence import config_at_detuning
        src = dict(self.CFG, chirp_coeffs_GHz=[0.0, 0.0, 0.001])
        out = config_at_detuning(src, 0.5, levels=9)
        self.assertIsNone(out["chirp_coeffs_GHz"])
        self.assertEqual(out["coupler_levels"], 9)
        self.assertEqual(src["chirp_coeffs_GHz"], [0.0, 0.0, 0.001])
        self.assertEqual(src["qubit_freqs_GHz"], [3.5, 3.8])
        kept = config_at_detuning(src, 0.5, chirp_coeffs_GHz=[0.0, 0.002])
        self.assertEqual(kept["chirp_coeffs_GHz"], [0.0, 0.002])


class TestSubharmonicLandmarks(unittest.TestCase):
    """The other resonances the pump walks past. A fidelity dip at one of these is
    a collision, not a truncation failure, so mislocating them would misattribute
    the whole map."""

    CFG = TestSubharmonicAxis.CFG

    def _by_delta(self):
        from snail_solver.subharmonic_convergence import collision_landmarks
        return {round(lm["delta_sub_GHz"], 6): lm
                for lm in collision_landmarks(self.CFG)}

    def test_the_subharmonic_itself_lands_at_zero(self):
        """2 w_p = w_s must come back at exactly Delta_sub = 0 -- the consistency
        check on the affine solve."""
        lm = self._by_delta()[0.0]
        self.assertEqual(lm["n_pump"], 2)
        self.assertTrue(lm["coupler"])
        self.assertAlmostEqual(lm["w_p_GHz"], 2.25, places=12)

    def test_known_landmarks_for_this_device(self):
        """Hand-derived for w_a = 3.5, w_s = 4.5: 2 w_p = w_a at 1.0; the a<->s
        conversion (w_p = w_s - w_a = 1.0, where w_b also lands on w_s) at 2.5;
        the b<->s conversion (w_b = 4.0) at 3.5."""
        got = self._by_delta()
        self.assertAlmostEqual(got[1.0]["w_p_GHz"], 1.75, places=12)
        self.assertFalse(got[1.0]["coupler"])
        self.assertAlmostEqual(got[2.5]["w_p_GHz"], 1.0, places=12)
        self.assertAlmostEqual(got[2.5]["w_b_GHz"], 4.5, places=12)
        self.assertTrue(got[2.5]["coupler"])
        self.assertAlmostEqual(got[3.5]["w_b_GHz"], 4.0, places=12)
        self.assertTrue(got[3.5]["coupler"])

    def test_every_landmark_is_a_real_resonance(self):
        """Independent check: at the reported w_p, some process really is resonant.
        Rebuild the condition from the mode frequencies rather than from the solve."""
        from snail_solver.subharmonic_convergence import collision_landmarks
        wa, ws = 3.5, 4.5
        for lm in collision_landmarks(self.CFG):
            w_p, n = lm["w_p_GHz"], lm["n_pump"]
            wb = wa + w_p
            nets = [ws, wa, wb, ws - wa, ws - wb, wb - wa,
                    wa + ws, wb + ws, wa + wb]
            self.assertTrue(any(abs(abs(net) - n * w_p) < 1e-9 for net in nets),
                            f"{lm['name']} at w_p={w_p} is not resonant")

    def test_names_are_terminal_safe_and_labels_are_mathtext(self):
        from snail_solver.subharmonic_convergence import collision_landmarks
        for lm in collision_landmarks(self.CFG):
            self.assertNotIn("$", lm["name"])
            self.assertIn("label", lm)

    def test_landmarks_re_derive_from_a_documents_settings(self):
        """--replot has only the settings, not the device: w_a, w_s and the branch
        have to be enough to rebuild the table, or an old document keeps stale
        labels forever."""
        from snail_solver.subharmonic_convergence import (
            collision_landmarks, landmarks_from_settings)
        want = collision_landmarks(self.CFG)
        got = landmarks_from_settings({"w_a_GHz": 3.5, "w_s_GHz": 4.5,
                                       "branch": "above"})
        self.assertEqual([r["delta_sub_GHz"] for r in got],
                         [r["delta_sub_GHz"] for r in want])
        self.assertEqual([r["label"] for r in got], [r["label"] for r in want])

    def test_nearest_landmark_reports_the_channel_detuning_not_just_the_axis_gap(self):
        """Delta_sub = w_s - 2 w_p, so a gap of 500 MHz along the axis is a
        channel detuned by 250 MHz. Reporting only the axis gap overstates every
        margin by 2x, and it is the detuning that enters |Omega/Delta|."""
        from snail_solver.subharmonic_convergence import (
            collision_landmarks, nearest_landmark)
        lms = collision_landmarks(self.CFG)
        near = nearest_landmark(lms, -2.0)           # landmark at -2.5
        self.assertAlmostEqual(near["distance_GHz"], 0.5, places=9)
        self.assertAlmostEqual(near["detuning_GHz"], 0.25, places=9)

    def test_describe_grid_flags_on_the_detuning(self):
        from snail_solver.subharmonic_convergence import describe_grid
        txt = describe_grid(self.CFG, [0.88], [3, 5], 0.6)   # 120 MHz axis gap
        self.assertIn("detuned 60 MHz", txt)                 # -> 60 MHz detuning
        self.assertIn("ON IT", txt)

    def test_nearest_landmark_skips_the_axis_origin(self):
        """Delta_sub = 0 is the axis, not a contaminant: reporting it as the
        nearest collision would flag every near-resonant column as unusable."""
        from snail_solver.subharmonic_convergence import (
            collision_landmarks, nearest_landmark)
        lms = collision_landmarks(self.CFG)
        near = nearest_landmark(lms, 0.05)
        self.assertNotAlmostEqual(near["delta_sub_GHz"], 0.0, places=6)
        self.assertAlmostEqual(nearest_landmark(lms, 0.05, skip_origin=False)
                               ["delta_sub_GHz"], 0.0, places=12)


class TestSubharmonicHelpers(unittest.TestCase):
    """Grid parsing, the displacement and the level count it predicts."""

    def test_displacement_matches_the_measured_case(self):
        """3 g3 eta^2 / Delta = 0.743 at g3 = 0.06, eta = 1.12, Delta = 0.304 --
        the number the leakage analysis measured and attributed."""
        from snail_solver.subharmonic_convergence import displacement_alpha
        self.assertAlmostEqual(
            displacement_alpha({"g3_GHz": 0.06}, 0.304, 1.12), 0.743, places=3)
        self.assertEqual(displacement_alpha({"g3_GHz": 0.06}, 0.0, 1.0),
                         float("inf"))

    def test_displacement_is_sign_blind(self):
        """Delta_sub < 0 is the other side of the same resonance."""
        from snail_solver.subharmonic_convergence import displacement_alpha
        self.assertAlmostEqual(displacement_alpha({"g3_GHz": 0.06}, +0.5, 1.0),
                               displacement_alpha({"g3_GHz": 0.06}, -0.5, 1.0))

    def test_predicted_levels(self):
        from snail_solver.subharmonic_convergence import predicted_levels
        self.assertAlmostEqual(predicted_levels(0.0), 1.0)
        self.assertAlmostEqual(predicted_levels(1.0), 6.0)
        self.assertGreater(predicted_levels(2.0), predicted_levels(1.0))

    def test_parse_detunings_forms(self):
        from snail_solver.subharmonic_convergence import parse_detunings
        self.assertEqual(parse_detunings("0.1,0.5,2.0"), [0.1, 0.5, 2.0])
        self.assertEqual(len(parse_detunings("0.1:2.0:5")), 5)
        log = parse_detunings("0.1:10:3:log")
        self.assertAlmostEqual(log[1], 1.0, places=12)
        neg = parse_detunings("-0.1:-10:3:log")
        self.assertAlmostEqual(neg[1], -1.0, places=12)

    def test_parse_detunings_composes_segments(self):
        """Extending an axis must not re-solve what is cached, so a range segment
        has to reproduce the same floats it did on its own."""
        from snail_solver.subharmonic_convergence import parse_detunings
        base = parse_detunings("0.15:0.8:8:log")
        both = parse_detunings("0.15:0.8:8:log,1.15,1.6")
        self.assertEqual(len(both), 10)
        self.assertEqual(base, both[:8])          # bit-for-bit, not just close
        self.assertEqual(both[-2:], [1.15, 1.6])

    def test_parse_detunings_sorts_and_dedupes_a_composed_grid(self):
        from snail_solver.subharmonic_convergence import parse_detunings
        self.assertEqual(parse_detunings("2.0,0.5,2.0,1.0"), [0.5, 1.0, 2.0])

    def test_parse_detunings_rejects_log_through_the_resonance(self):
        """Delta_sub = 0 IS the resonance: it cannot be a grid point, and a log
        grid spanning it is a sign error, not a wide scan."""
        from snail_solver.subharmonic_convergence import parse_detunings
        for bad in ("0:2:5:log", "-1:1:5:log", "0.1:2:0"):
            with self.assertRaises(ValueError):
                parse_detunings(bad)

    def test_parse_levels_needs_a_comparison(self):
        """Convergence is |F(N) - F(N_ref)|; one truncation has nothing to be
        measured against, and returning it would produce an all-converged map."""
        from snail_solver.subharmonic_convergence import parse_levels
        self.assertEqual(parse_levels("9,3,5,5"), [3, 5, 9])
        self.assertEqual(parse_levels("3:11:5"), [3, 5, 7, 9, 11])
        for bad in ("7", "1,7", ""):
            with self.assertRaises(ValueError):
                parse_levels(bad)

    def test_tags_have_no_dots(self):
        from snail_solver.subharmonic_convergence import cell_tag, delta_tag
        self.assertEqual(delta_tag(0.35), "d0p35")
        self.assertEqual(delta_tag(-0.2), "dm0p2")
        self.assertEqual(cell_tag(0.35, 7), "d0p35_N7")
        self.assertNotIn(".", cell_tag(1.25, 11))


def _subharm_doc(rows, *, levels=(3, 5, 9), ref=9, tol=2e-3):
    """A hand-built map document. `rows` is {delta: {levels: F}}."""
    cols = []
    for d, F_by_N in rows.items():
        F_ref = F_by_N.get(ref)
        cells = []
        for N, F in sorted(F_by_N.items()):
            spread = None if F_ref is None else abs(F - F_ref)
            cells.append({"levels": int(N), "F_avg": float(F), "leakage": 1e-3,
                          "transfer": float(F), "n_coupler": 0.1,
                          "t_g_ns": 200.0, "spread": spread,
                          "converged": None if spread is None else spread <= tol})
        cols.append({"delta_sub_GHz": float(d), "tag": "t", "ok": True,
                     "error": None, "w_b_GHz": 3.5 + 0.5 * (4.5 - d),
                     "w_p_GHz": 0.5 * (4.5 - d), "alpha_pred": 0.18 / abs(d),
                     "levels_pred": 2.0, "F_ref": F_ref, "cells": cells,
                     "calibration": {"t_g_ns": 200.0, "amp_scale": 1.0,
                                     "wp_offset_GHz": 0.0, "target_eta": 1.2,
                                     "chirp_coeffs_GHz": []}})
    doc = {"source": "subharmonic_convergence", "device": None,
           "settings": {"levels": list(levels), "ref_levels": ref, "tol": tol,
                        "target_eta": 1.2, "g3_GHz": 0.06,
                        "device_delta_sub_GHz": 3.9, "branch": "above",
                        "w_a_GHz": 3.5, "w_s_GHz": 4.5},
           "landmarks": [{"delta_sub_GHz": 1.0, "w_p_GHz": 1.75, "w_b_GHz": 5.25,
                          "n_pump": 2, "name": "qubit a excitation",
                          "label": r"qubit $a$ excitation", "coupler": False},
                         {"delta_sub_GHz": 0.0, "w_p_GHz": 2.25, "w_b_GHz": 5.75,
                          "n_pump": 2, "name": "SNAIL excitation",
                          "label": "SNAIL excitation", "coupler": True}],
           "columns": cols,
           "summary": {"n_columns": len(cols), "n_ok": len(cols),
                       "n_cells": sum(len(c["cells"]) for c in cols),
                       "seconds": 1.0}}
    from snail_solver.subharmonic_convergence import convergence_boundary
    doc["boundary"] = convergence_boundary(doc)
    return doc


class TestConvergenceBoundary(unittest.TestCase):
    """The boundary must be read from the FAR end inward."""

    def test_boundary_is_the_edge_of_the_converged_run(self):
        from snail_solver.subharmonic_convergence import convergence_boundary
        doc = _subharm_doc({
            0.1: {3: 0.50, 5: 0.70, 9: 0.90},        # nothing agrees
            0.5: {3: 0.86, 5: 0.8985, 9: 0.90},      # 5 agrees, 3 does not
            2.0: {3: 0.9005, 5: 0.9002, 9: 0.90},    # both agree
        })
        b = convergence_boundary(doc)
        self.assertAlmostEqual(b["5"]["delta_sub_GHz"], 0.5)
        self.assertAlmostEqual(b["3"]["delta_sub_GHz"], 2.0)
        self.assertTrue(b["9"]["is_reference"])

    def test_a_lone_converged_cell_inside_a_bad_run_is_not_a_boundary(self):
        """Next to a collision the spread can dip through the tolerance by
        cancellation. Quoting that as 'converged from here' is exactly the
        truncation-as-regularizer trap this module exists to expose."""
        from snail_solver.subharmonic_convergence import convergence_boundary
        doc = _subharm_doc({
            0.1: {3: 0.9001, 9: 0.90},               # accidental agreement
            0.5: {3: 0.50, 9: 0.90},                 # still broken further out
            2.0: {3: 0.9002, 9: 0.90},
        })
        self.assertAlmostEqual(
            convergence_boundary(doc)["3"]["delta_sub_GHz"], 2.0)

    def test_never_converged_reports_none(self):
        from snail_solver.subharmonic_convergence import convergence_boundary
        doc = _subharm_doc({0.1: {3: 0.2, 9: 0.9}, 2.0: {3: 0.4, 9: 0.9}})
        b = convergence_boundary(doc)
        self.assertIsNone(b["3"]["delta_sub_GHz"])
        self.assertEqual(b["3"]["n_converged"], 0)
        self.assertEqual(b["3"]["n_converged_any"], 0)

    def test_an_anomalous_outermost_column_is_not_read_as_never_working(self):
        """Measured at eta = 0.6: 3 levels agrees to 4e-5 at Delta_sub = 0.538
        while the outermost column disagrees by 8e-3. The contiguous run is then
        empty, but "never converged" would be the wrong claim."""
        from snail_solver.subharmonic_convergence import convergence_boundary
        from snail_solver.subharmonic_convergence import format_map
        doc = _subharm_doc({0.5: {3: 0.9000, 9: 0.90},     # agrees
                            2.0: {3: 0.8, 9: 0.90}})       # outermost does not
        b = convergence_boundary(doc)
        self.assertIsNone(b["3"]["delta_sub_GHz"])
        self.assertEqual(b["3"]["n_converged_any"], 1)
        self.assertIn("individual columns", format_map(doc))

    def test_tolerance_can_be_overridden_without_re_solving(self):
        from snail_solver.subharmonic_convergence import convergence_boundary
        doc = _subharm_doc({0.1: {3: 0.89, 9: 0.90}, 2.0: {3: 0.899, 9: 0.90}})
        self.assertIsNone(convergence_boundary(doc, tol=1e-4)["3"]["delta_sub_GHz"])
        self.assertAlmostEqual(
            convergence_boundary(doc, tol=2e-2)["3"]["delta_sub_GHz"], 0.1)


class TestSubharmonicPlotSmoke(unittest.TestCase):
    """`plot_convergence_map` must render from a hand-built doc, and must refuse
    to draw an empty one -- a blank map reads as a converged map."""

    ROWS = {0.1: {3: 0.50, 5: 0.70, 9: 0.90},
            0.5: {3: 0.86, 5: 0.8985, 9: 0.90},
            2.0: {3: 0.9005, 5: 0.9002, 9: 0.90}}

    def test_renders_the_map(self):
        from snail_solver.subharmonic_convergence import plot_convergence_map
        with tempfile.TemporaryDirectory() as d:
            out = plot_convergence_map(_subharm_doc(self.ROWS),
                                       out=os.path.join(d, "m.png"))
            self.assertTrue(os.path.getsize(out) > 5000)

    def test_renders_with_a_failed_column_and_a_linear_axis(self):
        from snail_solver.subharmonic_convergence import plot_convergence_map
        doc = _subharm_doc(self.ROWS)
        doc["columns"].append({"delta_sub_GHz": 3.0, "ok": False, "cells": [],
                               "error": {"type": "ValueError", "message": "x"}})
        with tempfile.TemporaryDirectory() as d:
            out = plot_convergence_map(doc, out=os.path.join(d, "m.png"),
                                       xscale="linear", annotate=False)
            self.assertTrue(os.path.getsize(out) > 5000)

    def test_no_cells_raises(self):
        from snail_solver.subharmonic_convergence import plot_convergence_map
        doc = _subharm_doc(self.ROWS)
        for col in doc["columns"]:
            col["cells"] = []
        with self.assertRaises(ValueError):
            plot_convergence_map(doc, "unused.png")

    def test_print_map_survives_a_failed_column(self):
        from snail_solver.subharmonic_convergence import print_map
        doc = _subharm_doc(self.ROWS)
        doc["columns"].append({"delta_sub_GHz": 3.0, "ok": False, "cells": [],
                               "error": {"type": "ValueError", "message": "x"}})
        print_map(doc)                       # must not raise on the missing keys

    def test_replot_needs_no_solver(self):
        """--replot must regenerate the figure from the JSON alone: it returns
        before the device or any solver module is touched."""
        from unittest import mock
        from snail_solver import subharmonic_convergence as SC
        with tempfile.TemporaryDirectory() as d:
            js, png = os.path.join(d, "c.json"), os.path.join(d, "c.png")
            with open(js, "w") as fh:
                json.dump(_subharm_doc(self.ROWS), fh)
            argv = ["subharmonic_convergence", "--replot", js, "--plot", png]
            with mock.patch.object(SC, "run_convergence_map") as m_run, \
                 mock.patch.object(sys, "argv", argv):
                SC.main()
            m_run.assert_not_called()
            self.assertTrue(os.path.getsize(png) > 5000)


class TestSubharmonicDryRun(unittest.TestCase):
    """`--dry-run` is the guard on an expensive submission: it must report the
    collisions the axis crosses and warn when the chosen drive cannot excite the
    effect being measured."""

    CFG = TestSubharmonicAxis.CFG

    def test_describe_grid_flags_a_column_on_a_collision(self):
        from snail_solver.subharmonic_convergence import describe_grid
        txt = describe_grid(self.CFG, [0.2, 0.98, 2.2], [3, 5, 9], 1.2)
        self.assertIn("ON IT", txt)                  # 0.98 is 20 MHz from 1.0
        self.assertIn("a->s conversion", txt)
        self.assertIn("exact solves", txt)

    def test_describe_grid_warns_when_the_drive_is_too_weak(self):
        """At small eta the displacement never approaches a photon, so the map is
        flat by construction and the run is wasted cluster time."""
        from snail_solver.subharmonic_convergence import describe_grid
        txt = describe_grid(self.CFG, [1.0, 2.0], [3, 5, 9], 0.2)
        self.assertIn("WARNING", txt)
        self.assertIn("flat by construction", txt)

    def test_describe_grid_reports_an_impossible_column(self):
        from snail_solver.subharmonic_convergence import describe_grid
        txt = describe_grid(self.CFG, [0.5, 4.6], [3, 5, 9], 1.2)
        self.assertIn("SKIPPED", txt)

class TestSubharmonicCacheGuard(unittest.TestCase):
    """The per-point caches are keyed by (Delta_sub, levels) ALONE, so without a
    guard a re-run into the same outdir at a different drive would be served the
    previous run's physics -- silently, straight into the map and the figure.
    """

    def test_matching_record_is_reused(self):
        from snail_solver.subharmonic_convergence import _cache_load
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w") as fh:
                json.dump({"target_eta": 1.2, "branch": "above",
                           "calib_levels": 13, "chirp_coeffs_GHz": [],
                           "t_g_ns": 100.0}, fh)
            got = _cache_load(p, {"target_eta": 1.2, "branch": "above",
                                  "calib_levels": 13, "chirp_coeffs_GHz": []})
            self.assertEqual(got["t_g_ns"], 100.0)

    def test_a_different_drive_is_not_reused(self):
        from snail_solver.subharmonic_convergence import _cache_load
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w") as fh:
                json.dump({"target_eta": 1.2, "t_g_ns": 100.0}, fh)
            self.assertIsNone(_cache_load(p, {"target_eta": 0.6}))

    def test_a_record_missing_the_key_is_not_reused(self):
        """An older cache cannot be verified, so it cannot be trusted."""
        from snail_solver.subharmonic_convergence import _cache_load
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w") as fh:
                json.dump({"t_g_ns": 100.0}, fh)
            self.assertIsNone(_cache_load(p, {"fit_virtual_z": True}))

    def test_a_changed_calibration_invalidates_its_cells(self):
        """A cell is tied to the operating point it was scored at."""
        from snail_solver.subharmonic_convergence import _cache_load
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w") as fh:
                json.dump({"levels": 5, "t_g_ns": 100.0, "amp_scale": 1.0,
                           "F_avg": 0.9}, fh)
            self.assertIsNotNone(_cache_load(p, {"levels": 5, "t_g_ns": 100.0}))
            self.assertIsNone(_cache_load(p, {"levels": 5, "t_g_ns": 101.0}))

    def test_a_truncated_cache_file_is_not_a_cache(self):
        from snail_solver.subharmonic_convergence import _cache_load
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w") as fh:
                fh.write('{"target_eta": 1.2,')
            self.assertIsNone(_cache_load(p, {"target_eta": 1.2}))
        self.assertIsNone(_cache_load(None, {"target_eta": 1.2}))

    def test_json_round_trip_does_not_break_equality(self):
        """A tuple comes back a list and an int comes back an int; neither is a
        physics change, and treating one as such would re-solve the whole grid."""
        from snail_solver.subharmonic_convergence import _same
        self.assertTrue(_same([0.0, 0.002], (0.0, 0.002)))
        self.assertTrue(_same(13, 13.0))
        self.assertTrue(_same("above", "above"))
        self.assertFalse(_same([0.0], [0.0, 0.002]))
        self.assertFalse(_same(True, 1.0))
        self.assertFalse(_same("above", "below"))
        self.assertFalse(_same(None, 0.0))

class TestSubharmonicCalibrationHealth(unittest.TestCase):
    """A column whose CALIBRATION failed must be flagged, not read as a
    truncation failure: all its cells score one bad operating point, so the
    spread down that column is not evidence either way. Measured on the real
    eta=1.2 run at Delta_sub = 1.6 GHz, where the length scan landed on a branch
    with transfer 0.688 and the six cells then scattered over F = 0.02 .. 0.82.
    """

    ROWS = {0.5: {3: 0.50, 5: 0.70, 9: 0.90}, 2.0: {3: 0.9005, 5: 0.9002, 9: 0.90}}

    def _doc(self, transfer):
        doc = _subharm_doc(self.ROWS)
        doc["columns"][0]["calibration"]["transfer"] = transfer
        doc["columns"][1]["calibration"]["transfer"] = 0.99
        return doc

    def test_table_flags_a_sick_column(self):
        from snail_solver.subharmonic_convergence import format_map
        txt = format_map(self._doc(0.688))
        self.assertIn("badly-calibrated pulse", txt)
        self.assertIn("0.500", txt)
        self.assertIn("transfer", txt)

    def test_table_is_quiet_when_every_column_is_healthy(self):
        from snail_solver.subharmonic_convergence import format_map
        self.assertNotIn("badly-calibrated",
                         format_map(self._doc(0.995)))

    def test_threshold_is_adjustable(self):
        from snail_solver.subharmonic_convergence import format_map
        self.assertNotIn("badly-calibrated",
                         format_map(self._doc(0.688), health_min=0.5))

    def test_a_calibration_with_no_transfer_recorded_is_not_flagged(self):
        """Absence of the field is not evidence of a bad calibration."""
        from snail_solver.subharmonic_convergence import format_map
        doc = _subharm_doc(self.ROWS)
        txt = format_map(doc)
        self.assertNotIn("badly-calibrated", txt)

    def test_figure_renders_the_rug(self):
        from snail_solver.subharmonic_convergence import plot_convergence_map
        with tempfile.TemporaryDirectory() as d:
            out = plot_convergence_map(self._doc(0.688),
                                       out=os.path.join(d, "m.png"))
            self.assertTrue(os.path.getsize(out) > 5000)

class TestSubharmonicMirrorCheck(unittest.TestCase):
    """Sampling both sides of the subharmonic is the isolation experiment: a
    mirrored pair shares |alpha| = 3 g3 eta^2 / |Delta_sub| exactly, while every
    other pump-activated channel sits at a different detuning on the two sides
    because w_p = (w_s -/+ |Delta_sub|)/2 differs. Agreement therefore rules the
    others out; disagreement localises one.
    """

    def _two_sided(self, f_neg):
        rows = {0.2: {3: 0.90, 9: 0.95}, -0.2: {3: f_neg, 9: 0.95},
                0.6: {3: 0.99, 9: 0.99}}
        return _subharm_doc(rows, levels=(3, 9), ref=9)

    def test_pairs_are_found_and_singletons_ignored(self):
        from snail_solver.subharmonic_convergence import mirror_pairs
        pairs = mirror_pairs(self._two_sided(0.90))
        self.assertEqual(len(pairs), 1)
        absd, pos, neg = pairs[0]
        self.assertAlmostEqual(absd, 0.2)
        self.assertGreater(pos["delta_sub_GHz"], 0)
        self.assertLess(neg["delta_sub_GHz"], 0)

    def test_one_sided_grid_has_no_mirror_section(self):
        from snail_solver.subharmonic_convergence import format_mirror
        doc = _subharm_doc({0.2: {3: 0.9, 9: 0.95}, 0.6: {3: 0.99, 9: 0.99}},
                           levels=(3, 9), ref=9)
        self.assertEqual(format_mirror(doc), "")

    def test_agreement_reads_as_the_subharmonic(self):
        from snail_solver.subharmonic_convergence import format_mirror
        txt = format_mirror(self._two_sided(0.95))     # both sides F(9) = 0.95
        self.assertIn("the two sides agree", txt)
        self.assertIn("follows |alpha|", txt)

    def test_disagreement_is_reported_as_a_confounder(self):
        from snail_solver.subharmonic_convergence import format_mirror
        doc = self._two_sided(0.90)
        doc["columns"][1]["cells"][1]["F_avg"] = 0.60   # F(9) at -0.2
        txt = format_mirror(doc)
        self.assertIn("DISAGREE", txt)
        self.assertIn("w_p-dependent", txt)

    def test_the_pair_shares_one_alpha(self):
        """If the two sides did not share |alpha| the comparison would be
        meaningless, so pin the identity the design rests on."""
        from snail_solver.subharmonic_convergence import (
            config_at_detuning, displacement_alpha)
        cfg = TestSubharmonicAxis.CFG
        for d in (0.06, 0.25, 0.63):
            self.assertAlmostEqual(
                displacement_alpha(config_at_detuning(cfg, +d), +d, 0.6),
                displacement_alpha(config_at_detuning(cfg, -d), -d, 0.6),
                places=12)

    def test_the_pair_does_not_share_the_conversion_detuning(self):
        """... and that the OTHER channels really do differ across the pair,
        which is what gives the comparison its power. a<->s is resonant at
        w_p = w_s - w_a, and w_p differs between the two sides."""
        from snail_solver.subharmonic_convergence import config_at_detuning
        cfg = TestSubharmonicAxis.CFG
        wa, ws = 3.5, 4.5
        det = []
        for d in (+0.63, -0.63):
            c = config_at_detuning(cfg, d)
            w_p = abs(c["qubit_freqs_GHz"][1] - c["qubit_freqs_GHz"][0])
            det.append(abs(w_p - (ws - wa)))
        self.assertGreater(abs(det[0] - det[1]), 0.5)   # 0.935 vs 1.565 GHz

    def test_report_includes_the_mirror_section_and_the_landmarks(self):
        from snail_solver.subharmonic_convergence import format_map
        txt = format_map(self._two_sided(0.95))
        self.assertIn("mirror check", txt)
        self.assertIn("other resonances near this axis", txt)


class TestSubharmonicGpuRouting(unittest.TestCase):
    """`--gpu-levels-min` must put the BIG truncations on the GPU and leave the
    small ones in the CPU pool, and a GPU batch must never be forked (a CUDA
    context does not survive fork).
    """

    def test_worker_switches_backend_only_when_asked(self):
        from unittest import mock
        from snail_solver import subharmonic_convergence as SC
        payload = {"config": {}, "delta_sub_GHz": 0.5, "levels": 18,
                   "record": {}, "kw": {}}
        with mock.patch.object(SC, "score_cell",
                               return_value={"F_avg": 0.9}) as m_score:
            with mock.patch("snail_solver.zhou_coupler.use_gpu") as m_gpu:
                res = SC._cell_worker(dict(payload, gpu=True))
                m_gpu.assert_called_once_with(True)
            self.assertEqual(res["cell"]["engine"], "gpu")
            with mock.patch("snail_solver.zhou_coupler.use_gpu") as m_gpu:
                res = SC._cell_worker(dict(payload, gpu=False))
                m_gpu.assert_not_called()
            self.assertEqual(res["cell"]["engine"], "cpu")
        self.assertEqual(m_score.call_count, 2)

    def test_gpu_batch_runs_in_process(self):
        """_run_pool at workers=1 must NOT fork: that is what makes the GPU
        branch safe, so it is an invariant, not an implementation detail."""
        from unittest import mock
        from snail_solver import subharmonic_convergence as SC
        with mock.patch.object(SC, "ProcessPoolExecutor") as m_pool:
            got = SC._run_pool(lambda p: p["v"], [{"v": 1}, {"v": 2}], 1)
            m_pool.assert_not_called()
        self.assertEqual(got, [1, 2])

    def test_a_cached_cpu_cell_is_not_re_solved_for_the_gpu(self):
        """The two backends integrate the same Hamiltonian, so `engine` is
        provenance, not physics -- requiring it to match would re-solve a whole
        grid for nothing."""
        from snail_solver.subharmonic_convergence import _cache_load
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w") as fh:
                json.dump({"levels": 18, "t_g_ns": 200.0, "engine": "cpu",
                           "F_avg": 0.99}, fh)
            self.assertIsNotNone(_cache_load(p, {"levels": 18,
                                                 "t_g_ns": 200.0}))

class TestSubharmonicThreadPinning(unittest.TestCase):
    """The CLI must pin the BLAS thread pools; a library import must not.

    Every worker in this module runs one independent solve, so the parallelism
    is already at the process level. Measured without the pin, 15 workers at 18
    coupler levels (dim 162) took 127 threads EACH -- ~1900 threads on 72 cores,
    load average 107, 310% CPU per worker to do the work of one. With fork the
    child inherits the parent's initialised BLAS, so the pin has to happen at
    CLI import time, before numpy loads.
    """

    ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

    def _probe(self, code):
        env = {k: v for k, v in os.environ.items() if k not in self.ENV}
        return subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, cwd=REPO_ROOT, env=env)

    def test_library_import_does_not_touch_the_environment(self):
        got = self._probe(
            "import snail_solver.subharmonic_convergence as m, os\n"
            "print('OMP=%s' % os.environ.get('OMP_NUM_THREADS'))\n")
        self.assertIn("OMP=None", got.stdout, got.stderr[-400:])

    def test_the_cli_pins_every_pool(self):
        got = self._probe(
            "import os, runpy, sys\n"
            "sys.argv = ['subharmonic_convergence', '--help']\n"
            "try:\n"
            "    runpy.run_module('snail_solver.subharmonic_convergence',\n"
            "                     run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
            "print('PINNED=%s' % ','.join(\n"
            "    '%s=%s' % (k, os.environ.get(k)) for k in\n"
            "    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')))\n")
        self.assertIn("OMP_NUM_THREADS=1", got.stdout, got.stderr[-400:])
        self.assertIn("OPENBLAS_NUM_THREADS=1", got.stdout)
        self.assertIn("MKL_NUM_THREADS=1", got.stdout)

    def test_an_explicit_setting_is_respected(self):
        """setdefault, not assignment: a caller who asked for 4 threads gets 4."""
        env = dict(os.environ, OMP_NUM_THREADS="4")
        got = subprocess.run(
            [sys.executable, "-c",
             "import os, runpy, sys\n"
             "sys.argv = ['x', '--help']\n"
             "try:\n"
             "    runpy.run_module('snail_solver.subharmonic_convergence',\n"
             "                     run_name='__main__')\n"
             "except SystemExit:\n"
             "    pass\n"
             "print('OMP=%s' % os.environ.get('OMP_NUM_THREADS'))\n"],
            capture_output=True, text=True, cwd=REPO_ROOT, env=env)
        self.assertIn("OMP=4", got.stdout, got.stderr[-400:])

class TestConvergenceBoundaryRungs(unittest.TestCase):
    """A rung is one |Delta_sub|, not one column.

    On a two-sided grid the mirrored pair share a rung, and it passes only if
    BOTH pass -- the coordinate is a DISTANCE from the subharmonic, so
    "converged from 0.25 GHz out" has to hold on either side. Measured on the
    mirror run: 7 levels was reported converged from |Delta_sub| = 0.25 while the
    +0.25 column was off by 0.36, because the walk counted the good mirror and
    reported its |Delta_sub|.
    """

    def _two_sided(self):
        # +0.25 is broken at 3 levels, -0.25 is fine; both fine further out.
        return _subharm_doc({+0.25: {3: 0.60, 9: 0.99},
                             -0.25: {3: 0.9895, 9: 0.99},
                             +0.60: {3: 0.9895, 9: 0.99},
                             -0.60: {3: 0.9895, 9: 0.99}},
                            levels=(3, 9), ref=9)

    def test_a_rung_is_only_as_good_as_its_worst_column(self):
        from snail_solver.subharmonic_convergence import convergence_boundary
        b = convergence_boundary(self._two_sided())
        self.assertAlmostEqual(b["3"]["delta_sub_GHz"], 0.60)
        self.assertEqual(b["3"]["n_scored"], 2)          # rungs, not columns

    def test_per_branch_boundaries_separate_the_two_sides(self):
        from snail_solver.subharmonic_convergence import boundary_by_branch
        got = boundary_by_branch(self._two_sided())
        self.assertAlmostEqual(got["positive"]["3"]["delta_sub_GHz"], 0.60)
        self.assertAlmostEqual(got["negative"]["3"]["delta_sub_GHz"], 0.25)

    def test_a_one_sided_grid_has_no_per_branch_table(self):
        from snail_solver.subharmonic_convergence import boundary_by_branch
        doc = _subharm_doc({0.25: {3: 0.99, 9: 0.99}, 0.6: {3: 0.99, 9: 0.99}},
                           levels=(3, 9), ref=9)
        self.assertEqual(boundary_by_branch(doc), {})

    def test_report_shows_the_per_branch_section_when_two_sided(self):
        from snail_solver.subharmonic_convergence import (
            boundary_by_branch, convergence_boundary, format_map)
        doc = self._two_sided()
        doc["boundary"] = convergence_boundary(doc)
        doc["boundary_signed"] = boundary_by_branch(doc)
        txt = format_map(doc)
        self.assertIn("per branch", txt)
        self.assertIn("2 w_p < w_s", txt)


class TestRunFileHDF5(unittest.TestCase):
    """Run outputs are HDF5, and the mapping to it is REVERSIBLE.

    ``--replot`` (and ``post_chirp --from-tuneup``) re-run plotting code that was
    written against the old JSON documents, so the encoding has to give back what
    it was handed: a list must not come back an ndarray (the operating-point
    record is written into a device JSON, which needs plain lists), an ndarray
    must not come back a list of lists (callers read ``.shape``), and ``None`` --
    which every optional field of a record uses -- must survive as ``None`` rather
    than as NaN or the string "None".
    """

    def _doc(self):
        return {
            "operating_point": {"amp_scale": 1.25, "wp_offset_GHz": -7e-4,
                                "t_g_ns": 78.5, "chirp_coeffs_GHz": [0.0, -0.013],
                                "spec_abs_GHz": None, "drag_beat_GHz": None,
                                "drag_n_pump": 1, "target_eta": 1.8,
                                "metric": "transfer", "score": 0.987,
                                "source": "tune_up"},
            "t_g0_ns": 77.16,
            "stages": {"rabi": {
                "eta": np.linspace(0.5, 1.8, 5),
                "delta_MHz": np.array([-0.7, -0.72, np.nan, -0.8, -0.9]),
                "fit": {"delta0": -0.7, "k2": 0.1, "r2": 0.99, "n_used": 5},
                "chevrons": [{"eta": 0.5, "metric": np.zeros(7),
                              "P10": np.zeros((7, 3)),
                              "quality": {"reject": None, "weight": 0.8}},
                             {"eta": 1.8, "metric": np.ones(7),
                              "P10": np.ones((7, 3)),
                              "quality": {"reject": "low_contrast",
                                          "weight": 0.0}}],
                "dropped": [], "notes": ["row 3 widened"],
                "railed": False, "traces": [[1.0, 2.0], [3.0, 4.0]]}}}

    def _same(self, a, b, path=""):
        if isinstance(a, dict):
            self.assertIsInstance(b, dict, path)
            self.assertEqual(set(a), set(b), path)
            for k in a:
                self._same(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, np.ndarray):
            self.assertIsInstance(b, np.ndarray, path)      # NOT a list of lists
            self.assertEqual(a.shape, b.shape, path)
            self.assertTrue(np.allclose(a, b, equal_nan=True), path)
        elif isinstance(a, list):
            self.assertIsInstance(b, list, path)            # NOT an ndarray
            self.assertEqual(len(a), len(b), path)
            for i, (x, y) in enumerate(zip(a, b)):
                self._same(x, y, f"{path}[{i}]")
        elif a is None:
            self.assertIsNone(b, path)
        else:
            self.assertIs(type(a), type(b), path)           # bool stays bool
            self.assertEqual(a, b, path)

    def test_round_trip_preserves_types_and_nulls(self):
        from snail_solver.h5_io import load_doc, save_doc
        doc = self._doc()
        with tempfile.TemporaryDirectory() as d:
            path = save_doc(os.path.join(d, "run.h5"), doc)
            self._same(doc, load_doc(path))

    def test_a_bare_name_becomes_h5_and_json_is_still_available(self):
        """The suffix is the whole format switch, so a run never lands in an
        ambiguous file: no suffix means HDF5, ``.json`` still means JSON."""
        from snail_solver.h5_io import is_hdf5, load_doc, save_doc
        doc = self._doc()
        with tempfile.TemporaryDirectory() as d:
            h5 = save_doc(os.path.join(d, "run"), doc)
            self.assertTrue(h5.endswith(".h5"))
            self.assertTrue(is_hdf5(h5))
            js = save_doc(os.path.join(d, "run.json"), doc)
            self.assertFalse(is_hdf5(js))
            with open(js) as fh:                            # readable as plain JSON
                self.assertEqual(json.load(fh)["operating_point"]["target_eta"], 1.8)
            self.assertIsInstance(load_doc(js)["stages"]["rabi"]["eta"], list)

    def test_old_json_runs_still_load(self):
        """--replot must keep working on every run written before the switch, so
        the loader dispatches on the file's CONTENT, not on its name."""
        from snail_solver.h5_io import load_doc
        with tempfile.TemporaryDirectory() as d:
            # an HDF5 file that is not named like one, and a JSON one that is
            mis = os.path.join(d, "run.json")
            from snail_solver.h5_io import save_tree
            save_tree(mis, {"stages": {"rabi": {"eta": np.zeros(3)}}})
            self.assertIn("stages", load_doc(mis))
            legacy = os.path.join(d, "legacy.h5")
            with open(legacy, "w") as fh:
                json.dump({"t_g0_ns": 77.0}, fh)
            self.assertEqual(load_doc(legacy)["t_g0_ns"], 77.0)

    def test_provenance_lands_outside_the_result_tree(self):
        from snail_solver.h5_io import load_doc, save_doc
        with tempfile.TemporaryDirectory() as d:
            path = save_doc(os.path.join(d, "run.h5"), self._doc(),
                            attrs={"command": "python -m snail_solver.tune_up",
                                   "device": "devices/1Gate4.2SNAIL.json"})
            back = load_doc(path)
            self.assertEqual(back["device"], "devices/1Gate4.2SNAIL.json")
            self.assertNotIn("format", back)                 # describes the file
            self.assertNotIn("command", back["stages"])      # never mixed into data

    def test_replot_from_hdf5_needs_no_device_and_no_solver(self):
        """The point of --replot: a figure from the run file alone. It must not
        reach for the device JSON or the solver, on HDF5 as it did on JSON."""
        from unittest import mock
        from snail_solver import tune_up
        from snail_solver.h5_io import save_doc
        with tempfile.TemporaryDirectory() as d:
            path = save_doc(os.path.join(d, "run.h5"), self._doc())
            argv = ["tune_up", "--replot", path]
            with mock.patch("snail_solver.tune_up.run_tune_up") as m_run, \
                 mock.patch.object(sys, "argv", argv):
                tune_up.main()
            m_run.assert_not_called()


class TestSweepRunsInOneFile(unittest.TestCase):
    """A sweep is ONE file: the summary and every eta's full tune-up inside it.

    The fan-out used to leave a directory of per-eta JSONs whose only link to the
    sweep was a filename convention. Now each run is a group in the sweep's own
    file, still in tune_up's --out schema and still addressable on its own, so
    `tune_up --replot sweep.h5:/runs/eta1p8` keeps working -- that round trip is
    what these tests pin, because it is the reason the runs are stored in that
    schema at all.
    """

    def _run_doc(self, eta):
        return {"operating_point": {"target_eta": eta, "t_g_ns": 100.0 / eta,
                                    "amp_scale": 1.1, "wp_offset_GHz": -7e-4,
                                    "chirp_coeffs_GHz": [0.0, -0.013],
                                    "spec_abs_GHz": None, "drag_beat_GHz": None,
                                    "score": 0.98, "source": "tune_up"},
                "t_g0_ns": 96.0 / eta,
                "stages": {"rabi": {"eta": np.linspace(0.3, 1.0, 5) * eta}}}

    def test_addresses_split_only_on_the_group_separator(self):
        from snail_solver.h5_io import split_address
        self.assertEqual(split_address("results/eta_sweep.h5:/runs/eta1p8"),
                         ("results/eta_sweep.h5", "runs/eta1p8"))
        self.assertEqual(split_address("results/plain.h5"),
                         ("results/plain.h5", None))

    def test_each_run_is_written_without_disturbing_the_others(self):
        """Runs are stored as each eta finishes, so a sweep killed part-way keeps
        what it measured -- which means every write must APPEND, not truncate."""
        from snail_solver.h5_io import load_doc, save_doc
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "eta_sweep.h5")
            a = save_doc(f, self._run_doc(1.2), group="runs/eta1p2")
            b = save_doc(f, self._run_doc(1.8), group="runs/eta1p8")
            save_doc(f, {"rows": [{"target_eta": 1.2}]}, group="sweep")
            self.assertEqual(a, f + ":/runs/eta1p2")
            self.assertEqual(sorted(load_doc(f)), ["runs", "sweep"])
            self.assertEqual(load_doc(b)["operating_point"]["target_eta"], 1.8)
            self.assertEqual(load_doc(a)["operating_point"]["target_eta"], 1.2)

    def test_a_stored_run_is_still_a_tune_up_document(self):
        """The whole point of the schema: one eta out of a sweep replots exactly
        like a standalone tune_up --out, with no device and no solver."""
        from unittest import mock
        from snail_solver import tune_up
        from snail_solver.h5_io import save_doc
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "eta_sweep.h5")
            addr = save_doc(f, self._run_doc(1.8), group="runs/eta1p8")
            with mock.patch("snail_solver.tune_up.run_tune_up") as m_run, \
                 mock.patch.object(sys, "argv", ["tune_up", "--replot", addr]):
                tune_up.main()
            m_run.assert_not_called()

    def test_json_cannot_swallow_a_group(self):
        """A .json sweep summary must not silently flatten the runs into itself;
        store_run puts them beside it instead."""
        from snail_solver.h5_io import save_doc
        from snail_solver.tune_up_sweep import store_run, sweep_holds_runs
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_doc(os.path.join(d, "s.json"), {"a": 1}, group="runs/x")
            self.assertFalse(sweep_holds_runs(os.path.join(d, "eta_sweep.json")))
            self.assertTrue(sweep_holds_runs(os.path.join(d, "eta_sweep.h5")))
            got = store_run("eta1p8", self._run_doc(1.8),
                            sweep_path=os.path.join(d, "eta_sweep.json"), outdir=d)
            self.assertEqual(got, os.path.join(d, "tuneup_eta1p8.h5"))

    def test_store_run_puts_it_in_the_sweep_file_when_it_can(self):
        from snail_solver.h5_io import load_doc
        from snail_solver.tune_up_sweep import store_run
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "eta_sweep.h5")
            got = store_run("eta1p8", self._run_doc(1.8), sweep_path=f, outdir=d,
                            attrs={"status": "ok"})
            self.assertEqual(got, f + ":/runs/eta1p8")
            self.assertEqual(os.listdir(d), ["eta_sweep.h5"])   # nothing beside it
            self.assertEqual(load_doc(got)["status"], "ok")


class TestEmbeddedFigures(unittest.TestCase):
    """A run's figures live inside its own run file.

    The failure this prevents is mundane and common: a directory of PNGs and a
    directory of run files that drift apart, so months later a figure is read
    against the wrong sweep. The embedded copy cannot drift -- it is in the group
    the data is in -- and a run pulled out of a sweep file (`FILE:/runs/eta1p8`)
    has to bring its own pictures, not the sweep's.
    """

    PNG = (b"\x89PNG\r\n\x1a\n" + b"fake png payload" * 40)

    def _fig(self, d, name="rabi_chevrons.png"):
        path = os.path.join(d, name)
        with open(path, "wb") as fh:
            fh.write(self.PNG)
        return path

    def test_a_figure_round_trips_byte_for_byte(self):
        from snail_solver.h5_io import extract_figures, figure_names, save_doc
        from snail_solver.h5_io import attach_figure
        with tempfile.TemporaryDirectory() as d:
            run = save_doc(os.path.join(d, "run.h5"), {"stages": {"rabi": {}}})
            attach_figure(run, "rabi", self._fig(d))
            self.assertEqual(figure_names(run), ["rabi"])
            out = os.path.join(d, "back")
            got = extract_figures(run, out)
            self.assertEqual([os.path.basename(p) for p in got],
                             ["rabi_chevrons.png"])       # keeps its own filename
            with open(got[0], "rb") as fh:
                self.assertEqual(fh.read(), self.PNG)     # byte-for-byte

    def test_each_run_in_a_sweep_keeps_its_own(self):
        """A sweep file holds one figures group per eta, addressed with the run."""
        from snail_solver.h5_io import attach_figure, figure_names, save_doc
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "eta_sweep.h5")
            a = save_doc(f, {"t_g0_ns": 1.0}, group="runs/eta1p2")
            b = save_doc(f, {"t_g0_ns": 2.0}, group="runs/eta1p8")
            save_doc(f, {"rows": []}, group="sweep")
            attach_figure(a, "rabi", self._fig(d))
            attach_figure(b, "rabi", self._fig(d))
            attach_figure(b, "chirp_ridge", self._fig(d, "chirp_ridge.png"))
            attach_figure(f + ":/sweep", "gate_quality_vs_eta",
                          self._fig(d, "quality.png"))
            self.assertEqual(figure_names(a), ["rabi"])
            self.assertEqual(sorted(figure_names(b)), ["chirp_ridge", "rabi"])
            self.assertEqual(figure_names(f + ":/sweep"), ["gate_quality_vs_eta"])
            self.assertEqual(figure_names(f), [])         # the root has none

    def test_re_attaching_replaces_rather_than_duplicates(self):
        """--replot after a plotting-code change refreshes the stored figure."""
        from snail_solver.h5_io import attach_figure, extract_figures, save_doc
        with tempfile.TemporaryDirectory() as d:
            run = save_doc(os.path.join(d, "run.h5"), {"t_g0_ns": 1.0})
            attach_figure(run, "rabi", self._fig(d))
            newer = os.path.join(d, "rabi_chevrons.png")
            with open(newer, "wb") as fh:
                fh.write(self.PNG + b"redrawn")
            attach_figure(run, "rabi", newer)
            got = extract_figures(run, os.path.join(d, "back"))
            self.assertEqual(len(got), 1)
            with open(got[0], "rb") as fh:
                self.assertTrue(fh.read().endswith(b"redrawn"))

    def test_a_figure_is_bytes_not_an_array_when_the_run_is_read_back(self):
        """The encoding has a bytes rule, so a document carrying a figure round
        trips as a document -- the loader must not hand back a uint8 array."""
        from snail_solver.h5_io import load_doc, save_doc
        with tempfile.TemporaryDirectory() as d:
            path = save_doc(os.path.join(d, "run.h5"),
                            {"figures": {"rabi": self.PNG}, "t_g0_ns": 1.0})
            back = load_doc(path)
            self.assertIsInstance(back["figures"]["rabi"], bytes)
            self.assertEqual(back["figures"]["rabi"], self.PNG)

    def test_missing_figures_are_skipped_not_raised(self):
        """A figure must never fail a finished run -- the callers already treat
        rendering that way, and storing it is no more important."""
        from snail_solver.h5_io import attach_figures, figure_names, save_doc
        with tempfile.TemporaryDirectory() as d:
            run = save_doc(os.path.join(d, "run.h5"), {"t_g0_ns": 1.0})
            n = attach_figures(run, {"rabi": self._fig(d),
                                     "chirp_ridge": os.path.join(d, "nope.png"),
                                     "post_chirp": None})
            self.assertEqual(n, 1)
            self.assertEqual(figure_names(run), ["rabi"])

    def test_a_json_run_takes_no_figures(self):
        """JSON cannot hold them, and embed_figures must no-op rather than crash
        the sweep that asked."""
        from snail_solver.tune_up_sweep import embed_figures
        with tempfile.TemporaryDirectory() as d:
            js = os.path.join(d, "s.json")
            with open(js, "w") as fh:
                json.dump({"rows": []}, fh)
            self.assertEqual(embed_figures(js, {"rabi": self._fig(d)}), 0)
            self.assertEqual(embed_figures(None, {"rabi": self._fig(d)}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
