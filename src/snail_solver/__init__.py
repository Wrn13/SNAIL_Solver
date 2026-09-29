"""SNAIL-coupler gate simulation and calibration toolkit.

Independently runnable tools sharing one physics core. Nothing heavy is imported
here (``qutip``, ``jax`` and ``matplotlib`` load only in the submodules that need
them), so ``import snail_solver`` stays cheap.

Layout
------
Physics core
    :mod:`~snail_solver.zhou_coupler` (Hamiltonian, propagators),
    :mod:`~snail_solver.envelope` (envelopes, chirps, :class:`PumpTone`),
    :mod:`~snail_solver.jax_engine` (JAX/diffrax backend, cross-checked vs QuTiP),
    :mod:`~snail_solver.device_utils`, :mod:`~snail_solver.operating_points`.
Calibration and optimal control
    :mod:`~snail_solver.calibrate_gate`, :mod:`~snail_solver.calibration_map`,
    :mod:`~snail_solver.find_stark_resonance`, :mod:`~snail_solver.grape`.
Sweeps
    :mod:`~snail_solver.run_sweep_zhou` (``prepare`` / ``point`` / ``local`` /
    ``collect``), with :mod:`~snail_solver.sweep_common`,
    :mod:`~snail_solver.sweep_spectator`, :mod:`~snail_solver.sweep_target`.
Analysis and figures
    :mod:`~snail_solver.spectroscopy`, :mod:`~snail_solver.spectator_audit`,
    :mod:`~snail_solver.stark_vs_detuning`, :mod:`~snail_solver.validate_engines`,
    and the ``plot_*`` / ``calibration_plots`` modules.

Every tool runs as a module, e.g.::

    python -m snail_solver.run_sweep_zhou prepare --device evan_device.json ...

Bare device names resolve against ``devices/`` and bare output names land under
``results/`` (see :mod:`~snail_solver.paths`).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
