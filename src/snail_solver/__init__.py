"""SNAIL-coupler gate simulation and calibration toolkit.

The package is a collection of independently runnable tools that share one
physics core. Nothing heavy is imported here: ``qutip``, ``jax`` and
``matplotlib`` are pulled in only by the submodule that needs them, so
``import snail_solver`` stays cheap.

Layout
------
Physics core
    :mod:`~snail_solver.zhou_coupler`
        The three-mode SNAIL coupler Hamiltonian and propagators.
    :mod:`~snail_solver.envelope`
        Pulse envelopes, chirps and the :class:`PumpTone` container (the single
        source of truth; re-exported by :mod:`~snail_solver.zhou_coupler`).
    :mod:`~snail_solver.jax_engine`
        JAX/diffrax propagation backend, cross-checked against QuTiP.
    :mod:`~snail_solver.device_utils`, :mod:`~snail_solver.operating_points`
        Device JSON loading, coupler construction and named operating points.

Calibration and optimal control
    :mod:`~snail_solver.calibrate_gate`, :mod:`~snail_solver.calibration_map`,
    :mod:`~snail_solver.find_stark_resonance`, :mod:`~snail_solver.grape`

Sweeps
    :mod:`~snail_solver.run_sweep_zhou`
        The sweep CLI (``prepare`` / ``point`` / ``local`` / ``collect``).
    :mod:`~snail_solver.sweep_common`, :mod:`~snail_solver.sweep_spectator`,
    :mod:`~snail_solver.sweep_target`
        Shared grid machinery and the two sweep implementations.

Analysis and figures
    :mod:`~snail_solver.spectroscopy`, :mod:`~snail_solver.spectator_audit`,
    :mod:`~snail_solver.stark_vs_detuning`, :mod:`~snail_solver.validate_engines`,
    and the ``plot_*`` / ``calibration_plots`` modules.

Every tool is a module with a ``__main__`` block, so it runs as::

    python -m snail_solver.run_sweep_zhou prepare --device evan_device.json ...

Bare device names resolve against ``devices/`` and bare output names land under
``results/`` -- see :mod:`~snail_solver.paths`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
