r"""Open-system scoring: what the gate achieves once T1 and T2 are switched on.

Everything else in this package solves a CLOSED system (``qt.sesolve``), which is the
right default -- it isolates the coherent error that pulse shaping can actually fix.
But it also means gate LENGTH is free, and since ``t_g = 2A/eta`` that biases every
comparison toward weak drive: a low ``eta`` buys a converged model and low leakage
while its five-times-longer gate costs nothing at all.
:func:`subharmonic_gate_scan.coherence_penalty` supplies a first-order estimate for
ranking. This module supplies the actual number for the handful of points worth
confirming.

The metric is the SAME leakage-aware average gate fidelity the closed-system path
uses (Pedersen PLA 367, 47 (2007); Wood & Gambetta PRA 97, 032306 (2018)), extended
from one Kraus operator to many. For a completely positive map ``E`` restricted to the
computational subspace, with Kraus operators ``K_i`` and error matrices
``M_i = U_ideal^dag K_i``::

    F = ( sum_i |Tr M_i|^2 + sum_i Tr[M_i M_i^dag] ) / ( d (d+1) )
    leakage = 1 - sum_i Tr[K_i^dag K_i] / d                          d = 4

With a single (unitary) Kraus operator both sums collapse and this IS
``ZhouCoupler._iswap_fidelity_from_U``. That is not a coincidence to be admired but a
TEST: :func:`open_iswap_fidelity` with no collapse operators must reproduce
``ZhouCoupler.iswap_fidelity`` to solver tolerance, which is what pins the
normalisation. See ``tests/test_physics.py::TestOpenSystemScoring``.

Neither sum needs the Kraus operators themselves. Since
``E(|k><m|) = sum_i K_i |k><m| K_i^dag``::

    <j| E(|k><m|) |l> = sum_i (K_i)_jk conj((K_i)_lm)

so propagating the 16 basis operators ``|k><m|`` of the computational subspace gives
both ingredients directly. That is 16 ``mesolve`` runs per point, on a density matrix
of ``dim^2`` -- which is why this is for confirming a winner, not for scanning.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Computational subspace dimension for a two-qubit gate.
_D = 4


def dephasing_rate_MHz(t1_us: Optional[float], t2_us: Optional[float]) -> float:
    r"""Pure-dephasing rate ``1/T_phi`` from ``T1`` and ``T2``, in 1/ns.

    ``1/T2 = 1/(2 T1) + 1/T_phi``, so the pure-dephasing part is what is LEFT after
    relaxation's contribution is removed. A ``T2`` longer than ``2 T1`` is
    unphysical; it is clamped to zero rather than silently producing a negative rate
    (which would be gain, not noise).
    """
    inv_t2 = 0.0 if not t2_us else 1.0 / (float(t2_us) * 1e3)      # 1/ns
    inv_t1 = 0.0 if not t1_us else 1.0 / (float(t1_us) * 1e3)
    return max(inv_t2 - 0.5 * inv_t1, 0.0)


def collapse_ops(cpl, *, t1_us: Optional[float] = None,
                 t2_us: Optional[float] = None,
                 qubits: Sequence[int] = (0, 1),
                 coupler_t1_us: Optional[float] = None,
                 coupler_index: Optional[int] = None) -> List[Any]:
    r"""Collapse operators for the qubits (and optionally the coupler).

    Built from the coupler's own embedded ladder operators, exactly as
    ``ZhouCoupler``'s docstring prescribes: ``sqrt(1/T1) a_i`` for relaxation and
    ``sqrt(2/T_phi) a_i^dag a_i`` for pure dephasing.

    Times are in microseconds; rates come out in 1/ns to match ``t_g``.

    Parameters
    ----------
    cpl : ZhouCoupler
        The built gate.
    t1_us, t2_us : float, optional
        Applied to every mode in `qubits`. Either may be omitted.
    qubits : sequence of int, default (0, 1)
        Mode indices to decohere.
    coupler_t1_us : float, optional
        Coupler/SNAIL loss. Worth including near a subharmonic, where the coupler is
        driven directly and carries real population.
    coupler_index : int, optional
        Which mode that is; defaults to ``cpl.coupler_index`` when present.

    Returns
    -------
    list of qutip.Qobj
        Possibly empty, which makes the scoring reduce to the closed-system result.
    """
    import qutip as qt

    dims = [[int(d) for d in cpl.dims], [int(d) for d in cpl.dims]]
    c_ops: List[Any] = []

    def _add(idx: int, t1: Optional[float], t2: Optional[float]) -> None:
        a = qt.Qobj(np.asarray(cpl.a_ops[idx]), dims=dims)
        if t1:
            c_ops.append(np.sqrt(1.0 / (float(t1) * 1e3)) * a)
        g_phi = dephasing_rate_MHz(t1, t2)
        if g_phi > 0.0:
            c_ops.append(np.sqrt(2.0 * g_phi) * (a.dag() * a))

    for i in qubits:
        _add(int(i), t1_us, t2_us)
    if coupler_t1_us:
        idx = (coupler_index if coupler_index is not None
               else getattr(cpl, "coupler_index", None))
        if idx is not None:
            _add(int(idx), coupler_t1_us, None)
    return c_ops


def _ideal_iswap() -> np.ndarray:
    """The target, in the same |00>, |01>, |10>, |11> ordering the projection uses."""
    from snail_solver.zhou_coupler import _ideal_iswap as ideal
    return np.asarray(ideal(), dtype=complex)


def _fidelity_from_map(E: np.ndarray, U_ideal: np.ndarray) -> Tuple[float, float]:
    """Pedersen fidelity and leakage from ``E[k, m] = P E(|k><m|) P`` (4x4 blocks)."""
    # sum_i Tr[K_i^dag K_i] = sum_k Tr[P E(|k><k|) P]
    surviving = float(np.real(sum(np.trace(E[k, k]) for k in range(_D))))
    # sum_i |Tr M_i|^2 = sum_jklm conj(A_jk) A_lm <j| E(|k><m|) |l>,  A = U_ideal
    A = U_ideal
    overlap = 0.0 + 0.0j
    for k in range(_D):
        for m in range(_D):
            blk = E[k, m]
            # sum_jl conj(A_jk) A_lm blk[j, l]
            overlap += np.conj(A[:, k]) @ blk @ A[:, m]
    return (float((np.real(overlap) + surviving) / (_D * (_D + 1))),
            float(1.0 - surviving / _D))


def _fit_virtual_z(E: np.ndarray, U_ideal: np.ndarray) -> Tuple[float, np.ndarray]:
    """Maximise the fidelity over the two single-qubit virtual-Z phases.

    Free in software, so scoring without them charges for a phase nobody would leave
    uncorrected. Optimised on the already-built map, so it costs no extra solves --
    a coarse grid to avoid the local maxima a bare local search falls into, then
    Nelder-Mead from the best cell.
    """
    from scipy.optimize import minimize

    def target(phases: Sequence[float]) -> np.ndarray:
        pa, pb = float(phases[0]), float(phases[1])
        diag = np.array([1.0, np.exp(1j * pb), np.exp(1j * pa),
                         np.exp(1j * (pa + pb))], dtype=complex)
        return U_ideal @ np.diag(diag)

    def neg_f(phases: Sequence[float]) -> float:
        return -_fidelity_from_map(E, target(phases))[0]

    grid = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    best = min(((neg_f((a, b)), (a, b)) for a in grid for b in grid),
               key=lambda t: t[0])
    res = minimize(neg_f, np.array(best[1]), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 800})
    phases = res.x if res.fun <= best[0] else np.array(best[1])
    return float(-min(res.fun, best[0])), target(phases)


def open_iswap_fidelity(cpl, a: int, b: int, t_g: float, *,
                        c_ops: Optional[Sequence[Any]] = None,
                        fit_virtual_z: bool = True,
                        atol: float = 1e-10, rtol: float = 1e-8,
                        nsteps: int = 500000) -> Dict[str, Any]:
    r"""Leakage-aware average iSWAP fidelity of the OPEN-system gate.

    Propagates the 16 basis operators ``|k><m|`` of the ``(a, b)`` computational
    subspace with ``qt.mesolve`` and scores the resulting map with the multi-Kraus
    Pedersen formula (see the module docstring).

    With ``c_ops`` empty or None this reduces to
    ``ZhouCoupler.iswap_fidelity(a, b, t_g)`` to solver tolerance.

    Parameters
    ----------
    cpl : ZhouCoupler
        The built gate, pump already set.
    a, b : int
        Target-qubit mode indices.
    t_g : float
        Gate duration (ns).
    c_ops : sequence, optional
        Collapse operators, e.g. from :func:`collapse_ops`.
    fit_virtual_z : bool, default True
        Divide out the optimal single-qubit Z phases.
    atol, rtol, nsteps : float, float, int
        ODE tolerances, matching the closed-system path's defaults.

    Returns
    -------
    dict
        ``F_avg``, ``leakage``, ``n_solves``, ``open`` (whether any collapse operator
        was applied).
    """
    import qutip as qt

    H = cpl.to_qutip_hamiltonian()
    options = cpl._qutip_options(atol, rtol, nsteps)
    idx = cpl._subspace_indices(a, b)
    dims = [[int(d) for d in cpl.dims], [int(d) for d in cpl.dims]]
    ops = list(c_ops or ())

    E = np.zeros((_D, _D, _D, _D), dtype=complex)
    n_solves = 0
    for k in range(_D):
        for m in range(_D):
            rho0 = np.zeros((cpl.dim, cpl.dim), dtype=complex)
            rho0[idx[k], idx[m]] = 1.0
            # mesolve is LINEAR, so an off-diagonal |k><m| is a legitimate input even
            # though it is not a state; the 16 of them span the map.
            res = qt.mesolve(H, qt.Qobj(rho0, dims=dims), [0.0, float(t_g)],
                             c_ops=ops, options=options)
            n_solves += 1
            rho = np.asarray(res.states[-1].full())
            E[k, m] = rho[np.ix_(idx, idx)]

    U_ideal = _ideal_iswap()
    if fit_virtual_z:
        F, target = _fit_virtual_z(E, U_ideal)
        leak = _fidelity_from_map(E, target)[1]
    else:
        F, leak = _fidelity_from_map(E, U_ideal)
    return {"F_avg": float(F), "leakage": float(leak), "n_solves": int(n_solves),
            "open": bool(ops)}
