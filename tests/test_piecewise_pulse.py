"""Invariants of the ramp / plateau / ramp study (``snail_solver.piecewise_pulse``).

* the three-window hand-off IS the single-shot solve (absolute-time carriers);
* kets and density matrices give the same channel populations;
* the plateau gate holds |eta| = eta_flat and the iSWAP area;
* the plateau phase modulation is zero on the ramps and its jets are derivatives;
* the matrix pencil recovers known lines.

    uv run python -m unittest tests.test_piecewise_pulse -v
"""
from __future__ import annotations

import json
import os
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _small_config(levels: int = 4):
    from snail_solver.subharmonic_convergence import config_at_wp
    with open(os.path.join(REPO_ROOT, "devices", "6Gate4.7SNAIL.json")) as fh:
        dev = json.load(fh)
    return config_at_wp(dev, 1.75 + 0.04, branch="above", levels=levels)


class TestPlateauGate(unittest.TestCase):
    def test_peak_and_area(self):
        from snail_solver.piecewise_pulse import build_plateau_gate, segment_bounds
        from snail_solver.tune_up import _area
        cfg = _small_config()
        cpl, t_g, _ = build_plateau_gate(cfg, 1.1, 15.0, 0.0)
        env = cpl._pump_tones[0].envelope
        self.assertAlmostEqual(cpl.peak_eta(), 1.1, places=9)
        self.assertAlmostEqual(env.area(), _area(cfg), places=7)
        b = segment_bounds(env)
        self.assertEqual(b[0], (0.0, 15.0))
        self.assertAlmostEqual(b[2][1], t_g)
        # the plateau really is flat
        tt = np.linspace(b[1][0], b[1][1], 11)
        np.testing.assert_allclose(env.value_at(tt), 1.1, rtol=1e-12)

    def test_no_plateau_raises(self):
        from snail_solver.piecewise_pulse import plateau_t_g
        with self.assertRaises(ValueError):
            plateau_t_g(_small_config(), 5.0, 60.0)


class TestPiecewiseEvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from snail_solver.piecewise_pulse import build_plateau_gate
        cls.cfg = _small_config()
        cls.cpl, cls.t_g, _ = build_plateau_gate(cls.cfg, 1.2, 15.0, 0.005)

    def test_handoff_equals_single_solve(self):
        from snail_solver.piecewise_pulse import evolve_piecewise, segment_bounds
        cpl = self.cpl
        init = [0, 1, 0]
        psi0 = np.zeros(cpl.dim, complex)
        psi0[cpl.fock_index(init)] = 1.0
        run = evolve_piecewise(cpl, psi0, segment_bounds(cpl._pump_tones[0].envelope))
        ref = cpl.evolve_state(init, self.t_g)
        overlap = abs(np.vdot(ref, run["boundary"][-1])) ** 2
        self.assertGreater(overlap, 1.0 - 1e-7)

    def test_ket_and_rho_agree(self):
        from snail_solver.piecewise_pulse import (channel_populations, evolve_piecewise,
                                                  segment_bounds)
        cpl = self.cpl
        init, tgt = [0, 1, 0], [1, 0, 0]
        psi0 = np.zeros(cpl.dim, complex)
        psi0[cpl.fock_index(init)] = 1.0
        bounds = segment_bounds(cpl._pump_tones[0].envelope)[:1]    # ramp only: cheap
        ket = evolve_piecewise(cpl, psi0, bounds)["boundary"][-1]
        rho = np.outer(ket, ket.conj())
        a = channel_populations(cpl, ket, init, tgt)
        b = channel_populations(cpl, rho, init, tgt, is_rho=True)
        c = channel_populations(cpl, rho[None], init, tgt)
        for k in a:
            np.testing.assert_allclose(a[k], b[k], atol=1e-12)
            np.testing.assert_allclose(a[k], c[k], atol=1e-12)
        self.assertLess(abs(a["norm_defect"][0]), 1e-7)


class TestPlateauPhaseMod(unittest.TestCase):
    def setUp(self):
        from snail_solver.piecewise_pulse import PlateauPhaseMod
        self.pm = PlateauPhaseMod(0.8, 0.05, 15.0, 100.0, edge_ns=6.0)

    def test_zero_on_ramps(self):
        t = np.concatenate([np.linspace(0, 15.0, 20), np.linspace(85.0, 100.0, 20)])
        np.testing.assert_allclose(self.pm.phase(t), 0.0, atol=1e-14)
        np.testing.assert_allclose(self.pm.detuning(t), 0.0, atol=1e-12)

    def test_jets_are_derivatives(self):
        t = np.linspace(16.0, 84.0, 301)
        h = 1e-4
        jet = self.pm.detuning_jet(t, 2)
        fd_phase = (self.pm.phase(t + h) - self.pm.phase(t - h)) / (2 * h)
        np.testing.assert_allclose(jet[0], fd_phase, atol=1e-6)
        fd1 = (self.pm.detuning(t + h) - self.pm.detuning(t - h)) / (2 * h)
        np.testing.assert_allclose(jet[1], fd1, atol=1e-5)

    def test_continuous_at_window_edges(self):
        for t0 in (15.0, 21.0, 79.0, 85.0):
            a = self.pm.detuning_jet(np.array([t0 - 1e-9, t0 + 1e-9]), 1)
            for d in a:
                self.assertLess(abs(d[1] - d[0]), 1e-6)


class TestMatrixPencil(unittest.TestCase):
    def test_two_close_lines_on_short_record(self):
        from snail_solver.piecewise_pulse import matrix_pencil
        t = np.linspace(0.0, 60.0, 601)                     # 60 ns: FFT bin ~17 MHz
        y = (1e-3 * np.cos(2 * np.pi * 0.030 * t)
             + 4e-4 * np.cos(2 * np.pi * 0.042 * t + 0.3) + 0.01 * t / 60.0)
        lines = matrix_pencil(t, y)
        f = sorted(round(ln["f_MHz"], 2) for ln in lines if ln["amplitude"] > 1e-4)
        self.assertIn(30.0, f)
        self.assertIn(42.0, f)


if __name__ == "__main__":
    unittest.main()
