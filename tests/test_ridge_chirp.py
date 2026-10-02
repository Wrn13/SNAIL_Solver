"""The Stark law read off the measured ridge (snail_solver.ridge_chirp).

A shaped probe reports the envelope average of the instantaneous shift. For an even
polynomial that average is the M2/M4 moment rule, so the ridge law must reproduce the
moment-deconvolved k2/k4 -- and, unlike them, follow a ridge no low-order polynomial
fits.
"""
import unittest

import numpy as np


def _cfg():
    from snail_solver import subharmonic_gate_scan as GS
    from snail_solver.device_utils import load_device
    from snail_solver.paths import resolve_device
    return GS.scan_config(load_device(resolve_device("6Gate4.7SNAIL.json")),
                          max_drag_channels=3, envelope_m=3)


class TestRidgeLaw(unittest.TestCase):

    def test_a_shaped_polynomial_ridge_deconvolves_to_the_moment_rule(self):
        from snail_solver.ridge_chirp import probe_shape_fn, ridge_law, shift_MHz
        from snail_solver.tune_up import probe_moments
        cfg = _cfg()
        M2, M4 = probe_moments(cfg, "rabi")
        eta = np.linspace(0.26, 1.3, 41)
        k2, k4, d0 = -2.7, 3.65, 0.4
        R = d0 + k2 * M2 * eta ** 2 + k4 * M4 * eta ** 4
        law = ridge_law(eta, R, shape_fn=probe_shape_fn(cfg), weighting="rabi")
        # Only the SHAPE reaches the pulse: a uniform offset trades against delta0
        # (the pulse barely dwells where |eta| < 0.26), and the chirp's constant
        # term is pinned into the carrier, which step 3 re-measures anyway.
        x = np.array([0.6, 0.9, 1.1, 1.3])
        true = k2 * x ** 2 + k4 * x ** 4
        got = shift_MHz(law, x)
        # The top node is set by the last few rows alone: ~1.6% there is the
        # inversion's real edge error, not noise.
        np.testing.assert_allclose(got - got[0], true - true[0], atol=0.03, rtol=0.02)
        self.assertLessEqual(law["resid_MHz"], law["noise_MHz"])

    def test_a_constant_probe_ridge_is_already_pointwise(self):
        from snail_solver.ridge_chirp import ridge_law, shift_MHz
        eta = np.linspace(0.2, 1.2, 21)
        R = 0.1 - 1.5 * eta ** 2
        law = ridge_law(eta, R, shape_fn=None, lam=1e-6)
        np.testing.assert_allclose(shift_MHz(law, eta), -1.5 * eta ** 2, atol=1e-3)

    def test_a_steep_section_is_followed_not_refused(self):
        """The +60 MHz shape: flat, a fast rise of ~2 MHz, flat again."""
        from snail_solver.ridge_chirp import probe_shape_fn, ridge_law, shift_MHz
        cfg = _cfg()
        eta = np.linspace(0.26, 1.3, 41)
        inst = lambda x: -0.3 * x ** 2 + 2.0 / (1.0 + np.exp(-(x - 1.07) / 0.01))
        from snail_solver.ridge_chirp import probe_weight
        u = np.linspace(-1, 1, 2001)
        f = probe_shape_fn(cfg)(u)
        w = probe_weight(probe_shape_fn(cfg), "rabi", u)
        R = np.array([np.trapz(w * inst(e * np.sqrt(f)), u) for e in eta])
        law = ridge_law(eta, R, shape_fn=probe_shape_fn(cfg))
        lo, hi = shift_MHz(law, [0.9, 1.25])
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 1.0)          # the step is in the law


class TestDirectMode(unittest.TestCase):
    """The ridge as the law, divided by M2: exact on |eta|^2, ~M4/M2 on |eta|^4."""

    def test_direct_mode_bias_is_the_moment_ratio(self):
        from snail_solver.ridge_chirp import ridge_law, shift_MHz
        from snail_solver.tune_up import probe_moments
        M2, M4 = probe_moments(_cfg(), "rabi")
        eta = np.linspace(0.26, 1.3, 41)
        for k2, k4 in ((-2.0, 0.0), (0.0, 1.0)):
            R = k2 * M2 * eta ** 2 + k4 * M4 * eta ** 4
            law = ridge_law(eta, R, shape_fn=None, lam=1e-6)
            got = shift_MHz(law, 1.2) / M2
            want = k2 * 1.2 ** 2 + k4 * (M4 / M2) * 1.2 ** 4
            self.assertAlmostEqual(got, want, delta=0.02)


class TestLawFromTable(unittest.TestCase):

    def _table(self, centres_MHz, span_MHz=10.0, n=15):
        off = np.linspace(-span_MHz / 2e3, span_MHz / 2e3, n)
        chev = []
        for i, c in enumerate(centres_MHz):
            met = 1.0 / (1.0 + ((off - c * 1e-3) / 1.5e-3) ** 2)
            chev.append({"eta": 0.3 + 0.05 * i, "offsets_GHz": off, "metric": met,
                         "span_MHz": span_MHz,
                         "fit": {"center_GHz": c * 1e-3, "hwhm_GHz": 1.5e-3,
                                 "depth": 1.0, "base": 0.0, "rmse": 1e-4,
                                 "vertex_GHz": c * 1e-3, "ok": True},
                         "quality": {"leak": 1e-4}})
        return {"chevrons": chev, "t_g_ref_ns": 107.0}

    def test_a_railed_row_is_reported(self):
        from snail_solver.ridge_chirp import law_from_table
        cen = list(-0.2 * np.arange(12) ** 1.2)
        cen[-1] = 4.9                           # on the +5 MHz edge
        law = law_from_table(self._table(cen), _cfg(), 0.85)
        self.assertTrue(law["railed"])

    def test_a_clean_table_gives_an_excursion(self):
        from snail_solver.ridge_chirp import law_from_table
        cen = list(-0.1 * np.arange(12))
        law = law_from_table(self._table(cen), _cfg(), 0.85)
        self.assertFalse(law["railed"])
        self.assertGreater(law["excursion_MHz"], 0.3)


if __name__ == "__main__":
    unittest.main()
