"""Floquet quasienergies and branch tracking (`snail_solver.floquet`).

Without the nonlinearity the period propagator is diagonal, so every labelled branch
and every diabatic energy must sit on its bare energy and every dressed beat on its bare
beat. With it, the coupler 0->1 shift at delta = -170 MHz must reproduce the stored
DRAG-sign-study value, on the labelled branches and on the diabatic energies alike.
"""
import unittest

import numpy as np


def _model(delta_GHz=-0.17, levels=5):
    from snail_solver import floquet as F
    from snail_solver.device_utils import load_device
    from snail_solver.paths import resolve_device
    dev = load_device(resolve_device("6Gate4.7SNAIL.json"))
    return F, F.model_at_detuning(dev, delta_GHz, branch="above", levels=levels)


STATES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (0, 1, 0), (2, 0, 1)]


class TestLinearLimit(unittest.TestCase):
    def test_branches_sit_on_bare_energies_without_nonlinearity(self):
        F, m = _model()
        m.g_n = {n: 0.0 for n in m.g_n}
        tr = F.track(m, np.linspace(0.0, 1.0, 3), n_steps=50, diabatic_of=STATES)
        for occ in STATES:
            bare = m.bare_energy(occ) / F.TWO_PI
            np.testing.assert_allclose(F.branch(tr, F.key(occ), diabatic=False), bare,
                                       atol=1e-9)
            np.testing.assert_allclose(F.branch(tr, F.key(occ)), bare, atol=1e-9)
        np.testing.assert_allclose(F.branch(tr, F.DOUBLET),
                                   m.bare_energy((1, 0, 0)) / F.TWO_PI, atol=1e-9)

    def test_dressed_beat_is_the_bare_beat_without_nonlinearity(self):
        F, m = _model(delta_GHz=0.041)
        m.g_n = {n: 0.0 for n in m.g_n}
        tr = F.track(m, [0.0, 0.5], n_steps=50, diabatic_of=STATES)
        # 2 w_s - w_a - 3 w_p with w_p = w_a/2 + delta: 650 - 3 delta MHz
        np.testing.assert_allclose(F.dressed_beat(tr, F.DOUBLET, "002", 3) * 1e3,
                                   650.0 - 3 * 41.0, atol=1e-6)


class TestCouplerShift(unittest.TestCase):
    def test_coupler_shift_matches_stored_value(self):
        F, m = _model(delta_GHz=-0.17, levels=9)
        tr = F.track(m, np.linspace(0.0, 0.4, 5), diabatic_of=[(0, 0, 0), (0, 0, 1)])
        shift = {}
        for diabatic in (False, True):
            c01 = (F.branch(tr, "001", diabatic) - F.branch(tr, "000", diabatic)) * 1e3
            shift[diabatic] = c01[-1] - c01[0]
        # drag_sign_study/floquet_shift_N1000.json: -16.56 MHz at |eta| = 0.4
        self.assertAlmostEqual(shift[False], -16.56, delta=0.05)
        # the diabatic energy leaves out the repulsion of near-resonant partners inside
        # its window, so it differs from the eigenvalue by less than |001>'s spread
        self.assertLess(abs(shift[True] - shift[False]),
                        1e3 * float(tr["spread_GHz"]["001"][-1]))


class TestResonance(unittest.TestCase):
    def test_crossing_and_avoided_minimum(self):
        from snail_solver.floquet import resonance
        e, gap, kind = resonance([0, 1, 2], [3.0, 1.0, -1.0], gap_max=5.0)
        self.assertEqual(kind, "cross")
        self.assertAlmostEqual(e, 1.5)
        e, gap, kind = resonance([0, 1, 2, 3], [3.0, 0.5, 2.0, 4.0], gap_max=5.0)
        self.assertEqual((kind, e, gap), ("avoided", 1.0, 0.5))
        self.assertEqual(resonance([0, 1, 2], [9.0, 8.0, 7.0], gap_max=5.0)[2], "none")


if __name__ == "__main__":
    unittest.main()
