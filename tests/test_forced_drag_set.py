"""The subharmonic scan's fixed DRAG set, and calibrating what the audit calls hopeless.

``--drag-set subharm-leak`` must play the A |0>-|1> subharmonic, the A |1>->|2>
subharmonic and the |2> leakage at EVERY column -- including where g/|det| is past the
ratio limits -- and nothing else. Only the skip window may drop one, because DRAG's
``1/beat`` is undefined at the collision itself.
"""
import unittest

import numpy as np

FORCED = {"a1->0 (2 pump)", "a1->2 (2 pump)"}
LEAK = {"a1->0 b1->2 (1 pump)", "a1->2 b1->0 (1 pump)"}


def _column(delta_GHz, levels=5):
    from snail_solver import subharmonic_gate_scan as GS
    from snail_solver.device_utils import load_device
    from snail_solver.paths import resolve_device
    base = GS.scan_config(load_device(resolve_device("6Gate4.7SNAIL.json")),
                          max_drag_channels=3, envelope_m=3)
    col = GS.columns_for(base, [delta_GHz], drop_origin=False)[0]
    settings = {"target_eta": 1.3, "branch": "above", "coupler_levels": levels,
                "max_drag_channels": 3, "min_ratio": 0.02, "max_ratio": 0.3,
                "drag_set": "subharm-leak"}
    return base, col, settings


def _selected(audit):
    return [r for r in audit["rows"] if r["selected"]]


class TestForcedDragSet(unittest.TestCase):

    def test_exactly_the_three_channels_even_past_the_ratio_limits(self):
        from snail_solver import subharmonic_gate_scan as GS
        base, col, settings = _column(0.015)      # A subharmonic g/|det| ~ 1
        chans, audit = GS.audit_column(base, col, settings)
        sel = _selected(audit)
        self.assertEqual(len(chans), 3)
        self.assertEqual(len(sel), 3)
        trans = {r["transition"] for r in sel}
        self.assertTrue(FORCED <= trans, trans)
        self.assertEqual(len(trans & LEAK), 1)
        self.assertEqual(trans - FORCED - LEAK, set())
        self.assertTrue(all(r["reason"] == "forced" for r in sel))
        self.assertFalse(audit["forced_missing"])
        # The A-SNAIL spectator the ranked selector always played is NOT played.
        self.assertFalse(any(r["category"] == "coupler" for r in sel))

    def test_only_the_skip_window_drops_a_forced_channel(self):
        from snail_solver import subharmonic_gate_scan as GS
        for delta, gone in ((0.0, "a1->0 (2 pump)"), (-0.06, "a1->2 (2 pump)")):
            base, col, settings = _column(delta)
            chans, audit = GS.audit_column(base, col, settings)
            self.assertEqual(len(chans), 2, delta)
            row = next(r for r in audit["rows"] if r["transition"] == gone)
            self.assertFalse(row["selected"])
            self.assertIn("skip window", row["reason"])

    def test_a_coincident_beat_is_covered_not_missing(self):
        """At delta = -30 MHz both A subharmonics beat at +60 MHz: one substitution."""
        from snail_solver import subharmonic_gate_scan as GS
        base, col, settings = _column(-0.03)
        chans, audit = GS.audit_column(base, col, settings)
        self.assertEqual(len(chans), 2)
        self.assertFalse(audit["forced_missing"])
        row = next(r for r in audit["rows"] if r["transition"] == "a1->2 (2 pump)")
        self.assertIn("same substitution", row["reason"])

    def test_ranked_is_still_the_default(self):
        from snail_solver import subharmonic_gate_scan as GS
        base, col, settings = _column(0.015)
        settings = {k: v for k, v in settings.items() if k != "drag_set"}
        _chans, audit = GS.audit_column(base, col, settings)
        self.assertIsNone(audit["forced"])
        self.assertNotIn("drag_set", GS._column_expect(
            {**col, "target_eta": 1.3}, _expect_settings()))
        self.assertEqual(GS._column_expect(
            {**col, "target_eta": 1.3},
            {**_expect_settings(), "drag_set": "subharm-leak"})["drag_set"],
            "subharm-leak")


def _expect_settings():
    return {"branch": "above", "coupler_levels": 5, "amp_points": 41,
            "eta_lo": 0.2, "eta_hi": 1.0, "max_drag_channels": 3, "min_ratio": 0.02,
            "max_ratio": 0.3, "contrast_min": 0.35, "probe_shape": "gate",
            "moment_weighting": "uniform", "envelope_m": 3}


class TestNonPerturbativeColumnIsCalibrated(unittest.TestCase):
    """The audit's verdict is a prediction; by default the gate is solved anyway."""

    SETTINGS = {"target_eta": 2.5, "branch": "below", "coupler_levels": 5,
                "eta_lo": 0.2, "eta_hi": 1.0, "amp_points": 41,
                "max_drag_channels": 3, "min_ratio": 0.02, "max_ratio": 0.3,
                "envelope_m": 3, "drag_retries": 0,
                "wp_points": 25, "wp_span_MHz": None, "span_linewidths": 4.0,
                "n_time": 61, "window_tg": 2.0, "tg_points": 7,
                "tg_lo": 0.7, "tg_hi": 1.3, "chirp_degree": 8, "max_drag_iters": 2,
                "contrast_min": 0.35, "quartic_warn": 0.25, "leak_max": None,
                "map_kw": {}, "probe_shape": "constant",
                "moment_weighting": "uniform"}

    def test_a_blocking_column_is_solved_and_flagged(self):
        from unittest import mock
        from snail_solver import subharmonic_gate_scan as GS
        from snail_solver.sweep_common import DEFAULT_CONFIG
        rec = {"target_eta": 2.5, "t_g_ns": 150.0, "amp_scale": 1.0,
               "wp_offset_GHz": 2e-3, "chirp_coeffs_GHz": [0.0, -0.004],
               "wa_GHz": 3.5, "wb_GHz": 1.7, "spec_abs_GHz": None,
               "drag_beat_GHz": None, "drag_n_pump": 1, "score": 0.99,
               "source": "tune_up"}

        def fake_run_tune_up(cfg, target_eta, **kw):
            return {"operating_point": rec, "t_g0_ns": 140.0,
                    "stages": {"rabi": {"eta": np.linspace(0.5, 2.5, 41),
                                        "fit": {"k2": 1.0, "k4": 0.1, "r2": 0.999}},
                               "chirp": {"quartic_fraction": 0.1}}}

        def fake_score_gate(cfg, record, chirp, **kw):
            return {"F_avg": 0.6, "leakage": 1e-2, "transfer": 0.6,
                    "t_g_ns": 150.0, "n_drag_channels": 0}

        cfg = {**DEFAULT_CONFIG, "qubit_freqs_GHz": [3.5, 3.8],
               "coupler_freq_GHz": 4.5, "envelope": "sine_power", "envelope_m": 3}
        col = GS.columns_for(cfg, [0.05])[0]
        block = [{"name": "qubit a subharmonic", "g_MHz": 6.5,
                  "detuning_MHz": 0.0, "verdict": "fails: not perturbative"}]
        with mock.patch("snail_solver.tune_up.run_tune_up", fake_run_tune_up), \
             mock.patch("snail_solver.tune_up_sweep.score_gate", fake_score_gate), \
             mock.patch("snail_solver.subharmonic_convergence.coupler_occupation",
                        lambda *a, **k: 1e-3), \
             mock.patch("snail_solver.subharmonic_gate_scan.audit_column",
                        lambda *a, **k: ((), {"blocking": block, "total_error": 1.0,
                                              "t_g0_ns": 140.0})):
            row = GS.solve_column(cfg, col, self.SETTINGS)
        self.assertTrue(row["ok"])
        self.assertTrue(row["audit_nonperturbative"])
        self.assertAlmostEqual(row["infidelity_coherent"], 0.4)


if __name__ == "__main__":
    unittest.main()
