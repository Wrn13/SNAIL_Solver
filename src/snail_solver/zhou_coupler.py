"""
zhou_coupler.py
===============

Charge-pumped parametric coupler in the dressed-mode framework of

    Chao Zhou, "Quantum Operations with Charge-pumped Parametric Interactions,"
    PhD thesis, University of Pittsburgh (2023), Chapter 2.

The Hamiltonian is built directly from EXPERIMENTAL inputs -- the coupler's
intrinsic non-linearity g_n (g3 for a SNAIL three-wave mixer) and the measured
participations

        lambda_is = g_is / Delta_is                                   (Eq. 57)

of each mode in the dressed coupler. No effective transmon-transmon edge is
assumed: the star connectivity lives entirely in the participations, because the
dressed coupler operator is (Eq. 56)

        s' ~= s + sum_i lambda_is a_i .

Master Hamiltonian (Zhou Eqs. 54 / 60; two-level/qubit form Eq. 72)
-------------------------------------------------------------------
        H_I(t) / hbar = sum_n  g_n  X(t)^n ,

        X(t) = s e^{-i w_s t}
             + sum_i lambda_is a_i e^{-i w_i t}
             + sum_p eta_p(t) e^{-i w_p t}
             + h.c. ,

with the displaced pump amplitude (Eq. 50)

        eta_p(t) = 2 w_p / (w_p^2 - w_s^2) * eps_p(t) .

Expanding X(t)^n generates every multi-wave-mixing process at once (iSWAP,
bSWAP, sub-harmonic drive, cross-Kerr, and all spectators / higher orders).
Pumping at  k w_p = w_rot  (Eq. 61) makes one process static under the RWA; its
strength is (Eq. 62)

        g_eff = C g_n eta^k prod_i lambda_is ,

with C a multinomial coefficient. The canonical two-qubit example -- pumping a
three-wave coupler at w_p = w_b - w_a -- gives the iSWAP family with (Eqs. 55/73)

        g_eff = 6 g3 lambda_as lambda_bs |eta| .          (verified in __main__)

g_n is the OVERALL prefactor of the dynamics: with g3 = 0 there is no gate.

Conventions
-----------
* Frequencies and non-linearities are entered in GHz and converted to angular
  units (rad/ns) internally: omega = 2 pi f. Time is in ns.
* A mode with 2 levels reproduces Zhou's sigma_- (qubit) form (Eq. 72); use
  >= 3 levels to keep it an oscillator/transmon so that leakage is captured.
* The coupler S is kept as a dynamical mode, so the qubit-coupler spectator
  channel is automatic.

Chirped pumps
-------------
Chirping the carrier, w_p -> w_p + delta(t), is ALGEBRAICALLY IDENTICAL to the phase
e^{-i Phi(t)} (Phi = integral delta) on the complex envelope, so w_p stays the
expansion reference and a term carrying k pump quanta picks up e^{-i k Phi(t)} on
its own. Attach a `Chirp` to the `PumpTone` (envelope.py).

Time evolution
--------------
`to_qutip_hamiltonian()` exports the EXACT Hamiltonian (no terms pruned) as a sparse
list-format QobjEvo that `evolve_state`, `propagator_columns` and `iswap_fidelity`
drive through `qt.sesolve`. QuTiP is imported lazily. The dense `hamiltonian_matrix`
is only a correctness oracle. `expand_terms_symbolic` factors the frequencies out
(Omega = M @ frequency_vector()) so a whole sweep shares one sparse operator stack
-- the structure `jax_engine.py` batches (validated in `validate_engines.py`).
"""

from __future__ import annotations

import cmath
import itertools
from math import factorial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# envelope.py names are re-exported for `from zhou_coupler import RaisedCosine, ...`
from snail_solver import drag
from snail_solver.envelope import (  # noqa: F401
    Chirp,
    ConstantPulse,
    DragChannel,
    Envelope,
    IQFourierEnvelope,
    PumpTone,
    RaisedCosine,
    make_chirp,
)

TWO_PI: float = 2.0 * np.pi

# --- solver backend (CPU QuTiP by default; GPU via qutip-jax / diffrax) -------
# Operators as `jaxdia`, states as dense `jax`, diffrax integrator (QuTiP 5,
# Lambert et al., arXiv:2412.04705).
_SOLVER_BACKEND: Dict[str, Any] = {"gpu": False, "op_dtype": "jaxdia", "state_dtype": "jax",
                                   "method": "diffrax"}


def use_gpu(enable: bool = True, x64: bool = True) -> None:
    """Route the QuTiP solver through the qutip-jax / diffrax GPU backend.

    When enabled, constant operators use the `jaxdia` data layer, initial states
    dense `jax`, the pump coefficients are `jax.jit`-wrapped builds of the SAME
    `_eta_at(..., xp)` the CPU path uses, and the solvers integrate with diffrax.
    Requires `qutip-jax` (ImportError otherwise).

    The coefficients must be ALREADY `jax.jit`-wrapped: qutip_jax maps a
    `PjitFunction` to the hashable `JaxJitCoeff`, whereas a plain closure becomes an
    unhashable `FunctionCoefficient` that diffrax's `equinox.filter_jit` rejects.
    Verified against CPU QuTiP on a pumped, shaped, chirped coupler to ~1e-7--1e-8
    (unitarity ~1e-16). diffrax warns that complex dtypes are "a work in progress";
    re-check if the physics changes qualitatively. On a 150 ns chirped gate (dim 45)
    diffrax ran ~10x faster than CPU QuTiP even on CPU-backed JAX -- time your case.

    Parameters
    ----------
    enable : bool, default True
        Turn the GPU backend on (True) or back to CPU QuTiP (False).
    x64 : bool, default True
        Enable JAX double precision (recommended for gate fidelities).
    """
    if enable:
        import jax
        import qutip_jax  # noqa: F401  (registers the 'jax'/'jaxdia' data layers)
        if x64:
            jax.config.update("jax_enable_x64", True)
    _SOLVER_BACKEND["gpu"] = bool(enable)


def gpu_enabled() -> bool:
    """True if the qutip-jax / diffrax GPU backend is active."""
    return bool(_SOLVER_BACKEND["gpu"])


# --- type aliases (documented shapes used throughout) ----------------------
# A "letter" of X(t): (operator, signed_frequency_rad, constant_amplitude,
# pump_key). The term's time factor is exp(-i * signed_frequency * t). pump_key
# is None for a mode operator, or (tone_index, is_conjugate) for a pump factor
# whose (time-dependent) amplitude eta_p(t) is supplied at evaluation time.
FluxLetter = Tuple[np.ndarray, float, float, Optional[Tuple[int, bool]]]

# A grouped term of the expanded Hamiltonian: (Omega_rad, pump_signature,
# operator). pump_signature is the sorted tuple of (tone_index, is_conjugate)
# pump factors carried by the term.
HamiltonianTerm = Tuple[float, Tuple[Tuple[int, bool], ...], np.ndarray]


# ===========================================================================
# Gate-metric helpers (pure numpy)
# ===========================================================================
def _ideal_iswap() -> np.ndarray:
    """Target iSWAP on the two-qubit subspace (basis |00>, |01>, |10>, |11>)."""
    U = np.eye(4, dtype=complex)
    U[1, 1] = U[2, 2] = 0.0
    U[1, 2] = U[2, 1] = 1j
    return U


def _fit_virtual_z(U: np.ndarray, U_ideal: np.ndarray) -> np.ndarray:
    r"""Apply the single-qubit virtual-Z rotation that best aligns `U` with the
    target (virtual-Z is free in software: McKay et al., PRA 96, 022330 (2017)).

    With :math:`Z = \mathrm{diag}(1, e^{i\varphi_b}, e^{i\varphi_a},
    e^{i(\varphi_a+\varphi_b)})` and :math:`c_k = (U U_{\rm ideal}^\dagger)_{kk}`,

    .. math::
        \mathrm{Tr}(U_{\rm ideal}^\dagger Z U)
          = \underbrace{(c_0 + e^{i\varphi_a} c_2)}_{A(\varphi_a)}
          + e^{i\varphi_b}\underbrace{(c_1 + e^{i\varphi_a} c_3)}_{B(\varphi_a)} ,

    maximised over :math:`\varphi_b` EXACTLY by rotating B onto A
    (:math:`|A| + |B|`, :math:`\varphi_b^\star = \arg A - \arg B`). The remaining 1-D
    maximisation over :math:`\varphi_a` is a vectorised sweep plus ternary search.

    Parameters
    ----------
    U : ndarray, shape (4, 4)
        Realised propagator projected onto the computational subspace.
    U_ideal : ndarray, shape (4, 4)
        Target unitary (here the ideal iSWAP).

    Returns
    -------
    ndarray, shape (4, 4)
        Z(pa*, pb*) U, with the phases maximising |Tr(U_ideal^dag Z U)|^2.
    """
    def z_rotation(phase_a: float, phase_b: float) -> np.ndarray:
        return np.diag([1.0,
                        np.exp(1j * phase_b),
                        np.exp(1j * phase_a),
                        np.exp(1j * (phase_a + phase_b))])

    c = np.diag(U @ U_ideal.conj().T)

    def merit(phase_a):
        """|A| + |B| -- the overlap already maximised over phase_b analytically."""
        e = np.exp(1j * np.asarray(phase_a))
        return np.abs(c[0] + e * c[2]) + np.abs(c[1] + e * c[3])

    n_coarse = 1024
    coarse = np.linspace(0.0, TWO_PI, n_coarse, endpoint=False)
    pa = float(coarse[int(np.argmax(merit(coarse)))])       # vectorised, one pass

    lo, hi = pa - TWO_PI / n_coarse, pa + TWO_PI / n_coarse
    for _ in range(60):                     # ternary search; (2/3)^60 -> ~1e-13 rad
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if merit(m1) < merit(m2):
            lo = m1
        else:
            hi = m2
    pa = 0.5 * (lo + hi)

    e = np.exp(1j * pa)
    A, B = c[0] + e * c[2], c[1] + e * c[3]
    # phase_b that rotates B onto A; B == 0 leaves the overlap phase_b-independent
    pb = float(np.angle(A) - np.angle(B)) if abs(B) > 0.0 else 0.0
    return z_rotation(pa, pb) @ U


# ===========================================================================
# The coupler
# ===========================================================================
class ZhouCoupler:
    r"""Parametric coupler built from experimental g_n and participations.

    Parameters
    ----------
    mode_freqs_GHz : sequence of float
        Frequencies of ALL modes, INCLUDING the coupler S (GHz). After the
        Bogoliubov dressing these are the dressed (i.e. measured) frequencies.
    coupler_index : int
        Index of the coupler mode S within `mode_freqs_GHz`.
    participations : dict[int, float]
        lambda_is = g_is/Delta_is for each non-coupler mode that participates in
        the dressed coupler (Eq. 57; for chained modes use Eq. 58 and pass the
        product). Omitted modes do not appear in X(t). The coupler's own
        participation is 1 by definition and must not be listed.
    nonlinearities : dict[int, float]
        {n: g_n} in GHz. {3: g3} for a pure three-wave SNAIL; add {4: g4} for
        four-wave mixing.
    levels : int or sequence of int, default 3
        Per-mode Fock truncation. 2 -> qubit (sigma_-, Eq. 72); >= 3 -> oscillator
        (captures leakage). A scalar applies to every mode.
    anharmonicities_GHz : dict[int, float], optional
        Per-mode transmon anharmonicity alpha_i (GHz); omitted modes are harmonic.

    Attributes: ``omega`` and ``g_n`` (rad/ns), ``n_modes``, ``dim``, ``dims``,
    ``participation`` (1 on the coupler), ``a_ops`` / ``ad_ops`` (embedded ladder
    operators), ``identity``.
    """

    def __init__(
        self,
        mode_freqs_GHz: Sequence[float],
        coupler_index: int,
        participations: Dict[int, float],
        nonlinearities: Dict[int, float],
        levels: Union[int, Sequence[int]] = 3,
        anharmonicities_GHz: Optional[Dict[int, float]] = None,
    ) -> None:
        self.omega: np.ndarray = np.asarray(mode_freqs_GHz, dtype=float) * TWO_PI  # rad/ns
        self.n_modes: int = self.omega.size

        self.coupler_index: int = int(coupler_index)
        if not (0 <= self.coupler_index < self.n_modes):
            raise ValueError("coupler_index out of range.")

        # per-mode truncation
        if isinstance(levels, int):
            self.dims: List[int] = [levels] * self.n_modes
        else:
            self.dims = [int(d) for d in levels]
            if len(self.dims) != self.n_modes:
                raise ValueError("len(levels) must equal the number of modes.")
        self.dim: int = int(np.prod(self.dims))   # total Hilbert-space dimension

        # participation vector: 1 on the coupler, lambda_is on coupled modes, 0 else
        self.participation: np.ndarray = np.zeros(self.n_modes, dtype=float)
        self.participation[self.coupler_index] = 1.0
        for index, value in participations.items():
            if index == self.coupler_index:
                raise ValueError("Do not list the coupler in participations (it is 1).")
            if not (0 <= index < self.n_modes):
                raise ValueError(f"participation index {index} out of range.")
            self.participation[index] = float(value)

        # non-linear coefficients, GHz -> rad/ns
        self.g_n: Dict[int, float] = {int(n): float(g) * TWO_PI
                                      for n, g in nonlinearities.items()}
        if not self.g_n:
            raise ValueError("Provide at least one non-linearity, e.g. {3: g3_GHz}.")

        # per-mode anharmonicity alpha_i (rad/ns); the coupler's own non-linearity
        # lives in g_n, so alpha is meant for the transmon/spectator modes.
        self.anharm: np.ndarray = np.zeros(self.n_modes, dtype=float)
        if anharmonicities_GHz:
            for index, value in anharmonicities_GHz.items():
                if not (0 <= int(index) < self.n_modes):
                    raise ValueError(f"anharmonicity index {index} out of range.")
                self.anharm[int(index)] = float(value) * TWO_PI

        # embedded ladder operators (dense, mixed-radix), built once
        self.a_ops: List[np.ndarray] = [self._embed(self._annihilation(self.dims[i]), i)
                                        for i in range(self.n_modes)]
        self.ad_ops: List[np.ndarray] = [op.conj().T for op in self.a_ops]
        self.identity: np.ndarray = np.eye(self.dim, dtype=complex)

        # static anharmonicity  sum_i (alpha_i/2) n_i (n_i - 1): number-diagonal, so
        # it enters the interaction picture unchanged (Omega = 0).
        self._anharm_op: np.ndarray = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.n_modes):
            if self.anharm[i] != 0.0:
                d = self.dims[i]
                local = np.diag([float(k * (k - 1)) for k in range(d)]).astype(complex)
                self._anharm_op = self._anharm_op + 0.5 * self.anharm[i] * self._embed(local, i)

        # time-independent (mode) letters of X(t); _flux_letters() adds the pumps
        self._mode_letters: List[FluxLetter] = []
        for i in range(self.n_modes):
            lam = self.participation[i]
            if lam == 0.0:
                continue
            # a_i ~ e^{-i w_i t} (signed_freq +w_i); a_i^dag ~ e^{+i w_i t} (-w_i)
            self._mode_letters.append((self.a_ops[i], +self.omega[i], lam, None))
            self._mode_letters.append((self.ad_ops[i], -self.omega[i], lam, None))

        self._pump_tones: List[PumpTone] = []

    # -- mode-space construction -------------------------------------------
    @staticmethod
    def _annihilation(d: int) -> np.ndarray:
        """Annihilation operator on a single d-level mode."""
        return np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)

    def _embed(self, op: np.ndarray, mode: int) -> np.ndarray:
        """Embed a single-mode operator `op` acting on `mode` into the full
        tensor-product space."""
        factors = [np.eye(d, dtype=complex) for d in self.dims]
        factors[mode] = op
        embedded = factors[0]
        for factor in factors[1:]:
            embedded = np.kron(embedded, factor)
        return embedded

    # -- Fock-space indexing ------------------------------------------------
    def fock_index(self, occupations: Sequence[int]) -> int:
        """Mixed-radix flat index of the Fock state |n_0, n_1, ...> (mode 0 most
        significant)."""
        index = 0
        for occ, d in zip(occupations, self.dims):
            if not (0 <= occ < d):
                raise ValueError("occupation out of range.")
            index = index * d + occ
        return index

    def decode_index(self, index: int) -> List[int]:
        """Inverse of `fock_index`: per-mode occupations of a flat index."""
        occupations: List[int] = []
        for d in reversed(self.dims):
            index, occ = divmod(index, d)
            occupations.append(occ)
        return occupations[::-1]

    def basis_state(self, occupations: Sequence[int]) -> np.ndarray:
        """Unit ket, shape (dim,), for the Fock state with these occupations."""
        psi = np.zeros(self.dim, dtype=complex)
        psi[self.fock_index(occupations)] = 1.0
        return psi

    def mean_occupation(self, probabilities: np.ndarray, mode: int) -> float:
        """<n_mode> = sum_k p_k occ_mode(k) for a Fock-basis probability vector."""
        return float(sum(probabilities[k] * self.decode_index(k)[mode]
                         for k in range(self.dim)))

    def delta(self, i: int, j: int) -> float:
        """Angular detuning w_i - w_j (rad/ns) between two modes."""
        return float(self.omega[i] - self.omega[j])

    # -- pump ---------------------------------------------------------------
    def set_pump(self, tones: Union[PumpTone, Sequence[PumpTone]],
                 normalize_iswap: Optional[Tuple[int, int]] = None) -> None:
        """Attach one or more pump tones to the coupler.

        Parameters
        ----------
        tones : PumpTone or sequence of PumpTone
            The pump tone(s). A single tone may be passed directly.
        normalize_iswap : tuple(int, int), optional
            If given as (a, b), the FIRST tone's amplitude is rescaled so that,
            pumped at w_b - w_a, the time-integrated rate realises a full iSWAP on
            the (a, b) pair (rotation angle pi/2). The condition is
            integral 6 g3 lambda_as lambda_bs |eta(t)| dt = pi/2 (Eqs. 55/73).
        """
        self._pump_tones = [tones] if isinstance(tones, PumpTone) else list(tones)
        if normalize_iswap is None:
            return

        a, b = normalize_iswap
        g3 = self.g_n.get(3, 0.0)
        if g3 == 0.0:
            raise ValueError("normalize_iswap needs a three-wave g3.")
        lam_a, lam_b = self.participation[a], self.participation[b]
        if lam_a == 0.0 or lam_b == 0.0:
            raise ValueError("Both qubits must have nonzero participation.")

        tone = self._pump_tones[0]
        omega_p = tone.w_p_GHz * TWO_PI
        omega_s = self.omega[self.coupler_index]
        # |eta(t)| = eta_prefactor * eps(t); see _eta and Eq. 50.
        eta_prefactor = 1.0 if tone.is_eta else abs(2 * omega_p / (omega_p ** 2 - omega_s ** 2))

        target_eta_area = (np.pi / 2) / (6 * g3 * lam_a * lam_b)   # required integral |eta| dt
        current_eta_area = eta_prefactor * tone.envelope.area()
        if current_eta_area == 0.0:
            raise ValueError("Pump envelope has zero area.")
        tone.envelope.amp *= target_eta_area / current_eta_area

    def scale_pump_amplitude(self, scale: float, tone_index: int = 0) -> None:
        """Multiply tone `tone_index`'s envelope amplitude by `scale`, AFTER any
        normalization -- a calibrated correction to the open-loop pi/2 amplitude
        (the leading-order rate under-estimates the dressed rate at finite eta)."""
        self._pump_tones[tone_index].envelope.amp *= float(scale)

    def _eta_at(self, tone: PumpTone, t: Any, xp: Any = np) -> Any:
        """Displaced pump amplitude eta_p(t) for scalar OR array `t`, built from
        `xp` primitives (Eq. 50; Motzoi PRL 103, 110501 (2009)).

        The SINGLE definition of the pump amplitude: the QuTiP callback (`_eta`) and
        the batched engine both route through it.

        DRAG acts on the BASE envelope and the chirp phase multiplies the result: the
        chirp rotates the pump frame, and differentiating it would inject a spurious
        -i delta(t) eta / Delta term. The DENOMINATOR does move with the chirp,
        ``Delta(t) = Delta_0 - k delta(t)`` with k = ``tone.drag_n_pump``
        (`PumpTone.drag_detuning`); a static Delta_0 would mis-weight the quadrature
        most exactly near a collision. Several channels apply recursive DRAG
        (:mod:`snail_solver.drag`) under the same rules.

        The single-channel first-order case keeps its closed-form branch: jet
        arithmetic reaches the same value by a different floating-point sequence, so
        routing it through jets would perturb recorded results in the last ulp (the
        two are asserted equal to 1e-13 in the tests).
        """
        omega_p = tone.w_p_GHz * TWO_PI
        omega_s = self.omega[self.coupler_index]
        prefactor = 1.0 if tone.is_eta else (2 * omega_p / (omega_p ** 2 - omega_s ** 2))
        amplitude = tone.envelope.value_at(t, xp)
        channels = tone.drag_channels_resolved()
        if tone.is_legacy_drag:
            detuning = tone.drag_detuning(t, xp)               # rad/ns, time-dependent
            amplitude = amplitude - 1j * tone.envelope.deriv_at(t, xp) / detuning
        elif channels:
            order = drag.required_order(channels)
            amplitude = drag.apply_drag(
                tone.envelope.jet_at(t, order, xp),
                tone.channel_detuning_jets(channels, t, order, xp),
                channels, xp)
        if tone.chirp is not None:
            # A chirped carrier w_p + delta(t) is exactly the fixed carrier w_p
            # times e^{-i Phi(t)} on the amplitude; the sign matches the
            # e^{-i w_p t} convention of the pump letter in _flux_letters.
            amplitude = amplitude * xp.exp(-1j * tone.chirp.phase(t, xp))
        return prefactor * amplitude * xp.exp(1j * tone.phi_p)

    def _eta(self, tone: PumpTone, t: float) -> complex:
        """Scalar pump amplitude eta_p(t) -- the QuTiP coefficient callback.

        Thin wrapper over :meth:`_eta_at`; see there for the physics.
        """
        return complex(self._eta_at(tone, float(t), np))

    def _flux_letters(self) -> List[FluxLetter]:
        """All letters of X(t): the static mode operators plus the current pump
        tones. A tone contributes eta_p e^{-i w_p t} (key (i, False)) and its
        conjugate eta_p* e^{+i w_p t} (key (i, True))."""
        letters = list(self._mode_letters)
        for tone_index, tone in enumerate(self._pump_tones):
            omega_p = tone.w_p_GHz * TWO_PI
            letters.append((self.identity, +omega_p, 1.0, (tone_index, False)))
            letters.append((self.identity, -omega_p, 1.0, (tone_index, True)))
        return letters

    # -- the dressed flux and the exact Hamiltonian (reference builders) ----
    def dressed_flux(self, t: float) -> np.ndarray:
        r"""Dressed flux operator X(t) (Hermitian).

        X(t) = sum_i lambda_is a_i e^{-i w_i t} + sum_p eta_p(t) e^{-i w_p t} + h.c.,
        dense, at time `t` (ns); the coupler (lambda = 1) is one of the modes.
        """
        etas = [self._eta(tone, t) for tone in self._pump_tones]
        X = np.zeros((self.dim, self.dim), dtype=complex)
        for op, signed_freq, amp, pump_key in self._flux_letters():
            coeff: complex = amp * np.exp(-1j * signed_freq * t)
            if pump_key is not None:
                tone_index, is_conjugate = pump_key
                eta = etas[tone_index]
                coeff *= np.conj(eta) if is_conjugate else eta
            X = X + coeff * op
        return X

    def hamiltonian_matrix(self, t: float) -> np.ndarray:
        """Exact interaction-picture Hamiltonian
        H_I(t) = sum_n g_n X(t)^n + sum_i (alpha_i/2) n_i(n_i-1), dense, rad/ns.

        A reference oracle only; not used in the production solve path.
        """
        X = self.dressed_flux(t)
        highest_order = max(self.g_n)
        powers: Dict[int, np.ndarray] = {1: X}
        for n in range(2, highest_order + 1):       # X^2, X^3, ... accumulated once
            powers[n] = powers[n - 1] @ X
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for n, g in self.g_n.items():
            H = H + g * powers[n]
        H = H + self._anharm_op          # static transmon anharmonicity (diagonal)
        return H

    # -- exact operator decomposition (engine of the QuTiP export) ----------
    def expand_terms(self, cutoff_GHz: float = np.inf) -> List[HamiltonianTerm]:
        r"""Expand sum_n g_n X(t)^n into constant operators grouped by net carrier
        frequency Omega = sum (+/- w_i) + sum (+/- w_p).

        Each returned (Omega, pump_signature, O) satisfies

            H(t) = sum_terms  exp(-i Omega t) * prod_p eta_p(t)^(...) * O ,

        where O folds in g_n and the participation products and the pump factors
        are supplied at evaluation time (so envelope and DRAG stay exact).

        Parameters
        ----------
        cutoff_GHz : float, default inf
            Keep only terms with |Omega| <= 2 pi cutoff. inf prunes NOTHING (the sum
            reproduces `hamiltonian_matrix(t)` exactly); a finite cutoff gives a
            rotating-wave reduction.

        Returns
        -------
        list of (float, tuple, ndarray)
            (exact Omega in rad/ns, pump signature, constant operator) per group.
        """
        cutoff_rad = abs(cutoff_GHz) * TWO_PI
        letters = self._flux_letters()

        # group by (rounded Omega, pump signature); accumulate the operator and
        # remember the exact Omega for phase-accurate evaluation.
        groups: Dict[Tuple[float, Tuple[Tuple[int, bool], ...]], List] = {}
        for order, g in self.g_n.items():
            for combo in itertools.product(letters, repeat=order):
                net_freq = sum(letter[1] for letter in combo)
                if abs(net_freq) > cutoff_rad:
                    continue
                amplitude, operator, signature = self._letter_product(combo)
                key = (round(net_freq, 6), signature)
                contribution = (g * amplitude) * operator
                if key in groups:
                    groups[key][0] += contribution
                else:
                    groups[key] = [contribution, net_freq]   # [operator_sum, exact Omega]

        return [(exact_omega, key[1], operator_sum)
                for key, (operator_sum, exact_omega) in groups.items()]

    @staticmethod
    def _letter_product(combo: Sequence[Any]) -> Tuple[float, Any, Tuple[Tuple[int, bool], ...]]:
        """``(amplitude, operator, sorted pump signature)`` of an ordered letter product."""
        amplitude = 1.0
        operator = None
        pump_signature: List[Tuple[int, bool]] = []
        for op, _carrier, amp, pump_key in combo:
            amplitude *= amp
            operator = op if operator is None else operator @ op
            if pump_key is not None:
                pump_signature.append(pump_key)
        return amplitude, operator, tuple(sorted(pump_signature))

    # -- frequency-INDEPENDENT expansion (engine of the batched solver) ------
    def _symbolic_letters(self) -> List[Tuple[np.ndarray, np.ndarray, float,
                                              Optional[Tuple[int, bool]]]]:
        """Letters of X(t) carrying an INTEGER carrier row instead of a float.

        Identical to `_flux_letters` except that the carrier ``+/- w`` is recorded
        as a unit row in Z^(n_modes + n_tones), so the frequencies can be supplied
        later: ``Omega = m . [w_0 ... w_{n-1}, w_p0 ...]``.
        """
        n_freq = self.n_modes + len(self._pump_tones)
        letters = []
        for i in range(self.n_modes):
            lam = self.participation[i]
            if lam == 0.0:
                continue
            row = np.zeros(n_freq, dtype=np.int64)
            row[i] = 1
            # a_i ~ e^{-i w_i t} (row +e_i); a_i^dag ~ e^{+i w_i t} (row -e_i)
            letters.append((self.a_ops[i], row, lam, None))
            letters.append((self.ad_ops[i], -row, lam, None))
        for tone_index in range(len(self._pump_tones)):
            row = np.zeros(n_freq, dtype=np.int64)
            row[self.n_modes + tone_index] = 1
            letters.append((self.identity, row, 1.0, (tone_index, False)))
            letters.append((self.identity, -row, 1.0, (tone_index, True)))
        return letters

    def frequency_vector(self) -> np.ndarray:
        """The carrier vector [w_0 .. w_{n-1}, w_p0 ..] (rad/ns) that pairs with
        the integer matrix `M` of :meth:`expand_terms_symbolic`."""
        return np.concatenate([self.omega,
                               [tone.w_p_GHz * TWO_PI for tone in self._pump_tones]])

    def expand_terms_symbolic(self) -> Dict[str, Any]:
        r"""Expand sum_n g_n X(t)^n into a FREQUENCY-INDEPENDENT term structure.

        Same algebra as :meth:`expand_terms`, grouped by the integer carrier row
        instead of the numerical Omega, so it depends only on
        ``(dims, participation, g_n, n_tones)`` and a frequency sweep shares one
        operator stack::

            Omega = M @ cpl.frequency_vector()          # (n_terms,)
            H(t)  = sum_j e^{-i Omega_j t} prod_p eta_p^{n_pos} conj(eta_p)^{n_neg} O_j

        Operators are returned as COO triplets: every letter product is a generalized
        permutation matrix, so each grouped operator has ~O(dim) non-zeros.

        Returns
        -------
        dict
            ``M`` (n_terms, n_modes + n_tones) int64 -- carrier rows;
            ``n_pos`` / ``n_neg`` (n_terms, n_tones) int64 -- eta exponents, i.e.
            the term carries ``prod_p eta_p^{n_pos} conj(eta_p)^{n_neg}``;
            ``term`` / ``row`` / ``col`` (nnz,) int64 and ``val`` (nnz,) complex --
            the stacked COO triplets, sorted by term;
            ``dim``, ``n_terms``, ``n_tones``, and ``signatures`` (the
            `expand_terms`-style pump signature per term, for cross-checking).

        See Also
        --------
        expand_terms : the frequency-SPECIFIC form consumed by the QuTiP export.
        """
        import scipy.sparse as sp

        letters = self._symbolic_letters()
        n_tones = len(self._pump_tones)

        # CSR products of generalized permutation matrices cost O(dim), not the
        # O(dim^3) of a dense matmul (~50 s -> sub-second at dim = 135)
        sletters = [(sp.csr_matrix(op), row, amp, key) for op, row, amp, key in letters]

        groups: Dict[Tuple[Tuple[int, ...], Tuple[Tuple[int, bool], ...]], List] = {}
        for order, g in self.g_n.items():
            for combo in itertools.product(sletters, repeat=order):
                carrier = sum(letter[1] for letter in combo)
                amplitude, operator, signature = self._letter_product(combo)
                key = (tuple(int(v) for v in carrier), signature)
                contribution = (g * amplitude) * operator
                if key in groups:
                    groups[key][0] = groups[key][0] + contribution
                else:
                    groups[key] = [contribution, carrier]

        n_terms = len(groups)
        M = np.zeros((n_terms, self.n_modes + n_tones), dtype=np.int64)
        n_pos = np.zeros((n_terms, max(n_tones, 1)), dtype=np.int64)
        n_neg = np.zeros((n_terms, max(n_tones, 1)), dtype=np.int64)
        signatures: List[Tuple[Tuple[int, bool], ...]] = []
        term_idx, rows, cols, vals = [], [], [], []
        for j, (key, (operator_sum, carrier)) in enumerate(groups.items()):
            M[j] = carrier
            signatures.append(key[1])
            for tone_index, is_conjugate in key[1]:
                if is_conjugate:
                    n_neg[j, tone_index] += 1
                else:
                    n_pos[j, tone_index] += 1
            coo = operator_sum.tocoo()
            # drop structural zeros left behind by cancelling contributions
            keep = coo.data != 0
            r, c, v = coo.row[keep], coo.col[keep], coo.data[keep]
            term_idx.append(np.full(r.size, j, dtype=np.int64))
            rows.append(r.astype(np.int64))
            cols.append(c.astype(np.int64))
            vals.append(v.astype(complex))

        empty_i, empty_c = np.zeros(0, np.int64), np.zeros(0, complex)
        return {
            "M": M,
            "n_pos": n_pos[:, :n_tones] if n_tones else n_pos[:, :0],
            "n_neg": n_neg[:, :n_tones] if n_tones else n_neg[:, :0],
            "term": np.concatenate(term_idx) if term_idx else empty_i,
            "row": np.concatenate(rows) if rows else empty_i,
            "col": np.concatenate(cols) if cols else empty_i,
            "val": np.concatenate(vals) if vals else empty_c,
            "dim": self.dim,
            "n_terms": n_terms,
            "n_tones": n_tones,
            "signatures": signatures,
        }

    # -- analytic effective-rate estimator (Eq. 62) -------------------------
    def effective_rate(self, modes: Sequence[int], n: int,
                       C: Optional[int] = None, eta: Optional[float] = None) -> float:
        r"""RWA process strength g_eff = C g_n eta^k prod_i lambda_is (Eq. 62).

        Parameters
        ----------
        modes : sequence of int
            Participating non-coupler modes, with multiplicity. The pump-quanta
            count is k = n - len(modes).
        n : int
            Non-linear order of the process (must have a defined g_n).
        C : int, optional
            Multinomial coefficient of the process. Defaults to n! (n distinct
            factors); pass it explicitly for degenerate factors.
        eta : float, optional
            Pump amplitude |eta|; defaults to the first tone's envelope peak.

        Returns g_eff in rad/ns.
        """
        g = self.g_n.get(n)
        if g is None:
            raise ValueError(f"no g_{n} defined.")
        n_pump_quanta = n - len(modes)
        if n_pump_quanta < 0:
            raise ValueError("len(modes) cannot exceed n.")
        if eta is None:
            if not self._pump_tones:
                raise ValueError("Set a pump or pass eta explicitly.")
            tone = self._pump_tones[0]
            eta = abs(self._eta(tone, tone.envelope.t_g / 2))
        participation_product = float(np.prod([self.participation[m] for m in modes])) if modes else 1.0
        if C is None:
            C = factorial(n)
        return float(C * g * (eta ** n_pump_quanta) * participation_product)

    def iswap_rate(self, a: int, b: int, eta: Optional[float] = None) -> float:
        """iSWAP coupling g_eff = 6 g3 lambda_as lambda_bs |eta| (rad/ns; Eqs. 55/73);
        `eta` defaults to the first tone's envelope peak."""
        return self.effective_rate([a, b], n=3, C=6, eta=eta)

    def peak_eta(self, tone_index: int = 0) -> float:
        """|eta| at t_g/2 -- the perturbative-pump diagnostic (the dressed-mode
        expansion needs |eta| << 1)."""
        tone = self._pump_tones[tone_index]
        return abs(self._eta(tone, tone.envelope.t_g / 2.0))

    # -- gate-metric helper -------------------------------------------------
    @staticmethod
    def _iswap_fidelity_from_U(U: np.ndarray, fit_virtual_z: bool) -> Tuple[float, float]:
        """Leakage-aware average gate fidelity and leakage from a 4x4 projected
        propagator (Pedersen PLA 367, 47 (2007); Wood & Gambetta PRA 97, 032306
        (2018)); virtual-Z phases optionally fitted out (free in software)."""
        d = 4
        U_ideal = _ideal_iswap()
        U_fit = _fit_virtual_z(U, U_ideal) if fit_virtual_z else U
        overlap = np.abs(np.trace(U_ideal.conj().T @ U_fit)) ** 2
        trace_UU = np.real(np.trace(U_fit.conj().T @ U_fit))
        fidelity = (overlap + trace_UU) / (d * (d + 1))
        leakage = 1.0 - trace_UU / d
        return float(fidelity), float(leakage)

    def _subspace_indices(self, a: int, b: int) -> List[int]:
        """Flat indices of |00>, |01>, |10>, |11> on the (a, b) pair, all other
        modes in |0>."""
        indices = []
        for occ_a, occ_b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            occupations = [0] * self.n_modes
            occupations[a], occupations[b] = occ_a, occ_b
            indices.append(self.fock_index(occupations))
        return indices

    # -- QuTiP Hamiltonian export (exact, sparse, fast) ---------------------
    def to_qutip_hamiltonian(self, cutoff_GHz: float = np.inf,
                             sparse: bool = True) -> Any:
        r"""Build the QuTiP list-format QobjEvo, [[O_j, c_j(t)], ...], with CONSTANT
        operators O_j (sparse CSR by default) and scalar coefficients
        c_j(t) = exp(-i Omega_j t) prod_p eta_p(t)^(...). g_n and the participation
        factors are folded into O_j.

        With cutoff_GHz = inf (default) it prunes NOTHING: sum_j c_j(t) O_j
        reproduces hamiltonian_matrix(t) exactly.

        Parameters
        ----------
        cutoff_GHz : float, default inf
            Passed to `expand_terms`; inf keeps the full, exact Hamiltonian.
        sparse : bool, default True
            Store each operator as SciPy CSR (the fast path) rather than dense.

        Returns
        -------
        qutip.QobjEvo or list
            A QobjEvo on QuTiP 5; the raw list on QuTiP 4 (both accepted by the
            solvers, including qt.mesolve / qt.propagator with collapse operators).
        """
        import qutip as qt
        import scipy.sparse as sp

        dims = [self.dims, self.dims]
        pump_tones = list(self._pump_tones)

        # One eta evaluation per (tone, t), shared by every term's coefficient (the
        # solver calls them all at the same t; recursive DRAG makes eta expensive).
        # One slot per tone: interleaved times just miss. The cache lives only as
        # long as this QobjEvo, so a tone mutated between solves (grape) is never stale.
        eta_cache: Dict[int, Tuple[float, complex]] = {}

        def eta_of(tone_index: int, t: float) -> complex:
            hit = eta_cache.get(tone_index)
            if hit is not None and hit[0] == t:
                return hit[1]
            value = self._eta(pump_tones[tone_index], t)
            eta_cache[tone_index] = (t, value)
            return value

        def make_coeff(omega: float, pump_signature: Tuple[Tuple[int, bool], ...]
                       ) -> Callable[[float], complex]:
            if _SOLVER_BACKEND["gpu"]:
                # must be jax.jit'd so qutip_jax builds a hashable JaxJitCoeff (use_gpu)
                import jax
                import jax.numpy as jnp

                def coeff_jax(t: float, **kwargs: Any) -> complex:
                    value = jnp.exp(-1j * omega * t)
                    for tone_index, is_conjugate in pump_signature:
                        eta = self._eta_at(pump_tones[tone_index], t, jnp)
                        value = value * (jnp.conj(eta) if is_conjugate else eta)
                    return value
                return jax.jit(coeff_jax)

            def coeff(t: float, **kwargs: Any) -> complex:
                value = cmath.exp(-1j * omega * t)
                for tone_index, is_conjugate in pump_signature:
                    eta = eta_of(tone_index, t)
                    value *= eta.conjugate() if is_conjugate else eta
                return value
            return coeff

        H: List[Any] = []
        for omega, pump_signature, operator in self.expand_terms(cutoff_GHz):
            matrix = sp.csr_matrix(operator) if sparse else np.asarray(operator, dtype=complex)
            qobj = qt.Qobj(matrix, dims=dims)
            if _SOLVER_BACKEND["gpu"]:
                qobj = qobj.to(_SOLVER_BACKEND["op_dtype"])      # jaxdia for GPU
            if abs(omega) < 1e-9 and not pump_signature:
                H.append(qobj)                                  # genuinely static term
            else:
                H.append([qobj, make_coeff(float(omega), pump_signature)])
        if np.any(self.anharm):                                 # static transmon anharmonicity
            matrix = sp.csr_matrix(self._anharm_op) if sparse else self._anharm_op
            a_qobj = qt.Qobj(matrix, dims=dims)
            if _SOLVER_BACKEND["gpu"]:
                a_qobj = a_qobj.to(_SOLVER_BACKEND["op_dtype"])
            H.append(a_qobj)
        try:
            return qt.QobjEvo(H)
        except Exception:
            return H                                            # QuTiP 4: pass the list

    # -- time evolution & gate metrics (QuTiP compiled solver) --------------
    def _qutip_options(self, atol: float, rtol: float, nsteps: int) -> Any:
        """Return a solver-options object compatible with QuTiP 4 or 5 (and the
        diffrax integrator when the GPU backend is active)."""
        import qutip as qt
        if _SOLVER_BACKEND["gpu"]:
            # DiffraxIntegrator rejects flat atol/rtol/nsteps (KeyError)
            import diffrax
            return {"method": _SOLVER_BACKEND["method"],
                    "stepsize_controller": diffrax.PIDController(rtol=rtol, atol=atol),
                    "max_steps": int(nsteps)}
        try:
            return qt.Options(atol=atol, rtol=rtol, nsteps=nsteps)     # QuTiP 4
        except (AttributeError, TypeError):
            return {"atol": atol, "rtol": rtol, "nsteps": nsteps}       # QuTiP 5 (CPU)

    def _ket(self, start_index: int) -> Any:
        """Computational-basis ket |start_index> as a QuTiP Qobj, converted to the
        dense `jax` data layer when the GPU backend is active."""
        import qutip as qt
        vector = np.zeros(self.dim, dtype=complex)
        vector[start_index] = 1.0
        psi0 = qt.Qobj(vector.reshape(-1, 1), dims=[self.dims, [1] * self.n_modes])
        return psi0.to(_SOLVER_BACKEND["state_dtype"]) if _SOLVER_BACKEND["gpu"] else psi0

    def _sesolve_final(self, H: Any, start_index: int, t_g: float,
                       options: Any) -> np.ndarray:
        """Closed-system evolve a computational-basis ket to t_g; final state (numpy).

        Output points every ~5 ns keep the per-interval step cap (nsteps) from being
        exhausted on long gates ("Excess work done"); they do not change the result.
        """
        import qutip as qt
        psi0 = self._ket(start_index)
        n_out = int(np.clip(np.ceil(t_g / 5.0), 1, 4000)) + 1   # ~one output / 5 ns
        tlist = np.linspace(0.0, t_g, n_out)
        result = qt.sesolve(H, psi0, tlist, options=options)
        return result.states[-1].full().ravel()

    def evolve_state(self, init_occupations: Sequence[int], t_g: float,
                     atol: float = 1e-10, rtol: float = 1e-8,
                     nsteps: int = 500000) -> np.ndarray:
        """Final state |psi(t_g)>, shape (dim,), of a Fock initial state evolved on the
        EXACT Hamiltonian (QuTiP sesolve; `atol`/`rtol`/`nsteps` are ODE options)."""
        H = self.to_qutip_hamiltonian()
        options = self._qutip_options(atol, rtol, nsteps)
        return self._sesolve_final(H, self.fock_index(init_occupations), t_g, options)

    def evolve_trajectory(self, init_occupations: Sequence[int], times: Sequence[float],
                          atol: float = 1e-10, rtol: float = 1e-8,
                          nsteps: int = 500000) -> np.ndarray:
        """States, shape (len(times), dim), of a Fock initial state at EVERY time in
        `times` (sorted, starting at 0) on the EXACT Hamiltonian -- e.g. to map P(t)
        at fixed pump strength (Fig. 8a style)."""
        import qutip as qt
        H = self.to_qutip_hamiltonian()
        options = self._qutip_options(atol, rtol, nsteps)
        psi0 = self._ket(self.fock_index(init_occupations))
        result = qt.sesolve(H, psi0, np.asarray(times, dtype=float), options=options)
        return np.array([state.full().ravel() for state in result.states])

    def propagator_columns(self, a: int, b: int, t_g: float,
                           atol: float = 1e-10, rtol: float = 1e-8,
                           nsteps: int = 500000) -> np.ndarray:
        """4x4 projection of the realised propagator onto the (a, b) computational
        subspace (Pedersen/Wood convention; spectators start in |0>), on the EXACT
        Hamiltonian. Columns evolve |00>, |01>, |10>, |11>."""
        H = self.to_qutip_hamiltonian()
        options = self._qutip_options(atol, rtol, nsteps)
        indices = self._subspace_indices(a, b)
        U = np.zeros((4, 4), dtype=complex)
        for col, start in enumerate(indices):
            psi_final = self._sesolve_final(H, start, t_g, options)
            for row, end in enumerate(indices):
                U[row, col] = psi_final[end]
        return U

    def iswap_fidelity(self, a: int, b: int, t_g: float, fit_virtual_z: bool = True,
                       atol: float = 1e-10, rtol: float = 1e-8,
                       nsteps: int = 500000) -> Tuple[float, float, np.ndarray]:
        """Leakage-aware average iSWAP fidelity on (a, b) on the EXACT Hamiltonian.

        Returns ``(fidelity, leakage, U)``: leakage is 1 - Tr(U^dag U)/4 and U the
        projected propagator. `fit_virtual_z` divides out the optimal (free)
        single-qubit virtual-Z phases first. For the open system see ``open_system``.
        """
        U = self.propagator_columns(a, b, t_g, atol=atol, rtol=rtol, nsteps=nsteps)
        fidelity, leakage = self._iswap_fidelity_from_U(U, fit_virtual_z)
        return fidelity, leakage, U


# ===========================================================================
# Demo / self-test: reproduce Zhou's iSWAP coupling g_eff = 6 g3 la lb |eta|
# (uses the dense reference builder; runs without QuTiP)
# ===========================================================================
if __name__ == "__main__":
    # two qubits (a, b) + a SNAIL coupler (s); qubits are 2-level (Eq. 72 form)
    coupler = ZhouCoupler(
        mode_freqs_GHz=[4.0, 5.0, 7.0],
        coupler_index=2,
        participations={0: 0.15, 1: 0.15},      # lambda_as, lambda_bs = g_is/Delta_is
        nonlinearities={3: 0.10},               # g3 = 100 MHz
        levels=[2, 2, 6],
    )
    # one flat pump at w_b - w_a activates the iSWAP family (Eq. 73)
    pump_freq_GHz = coupler.delta(1, 0) / TWO_PI
    coupler.set_pump(PumpTone(w_p_GHz=pump_freq_GHz,
                              envelope=ConstantPulse(0.06, 50.0), is_eta=True))

    # the DC (time-averaged) matrix element <ge|H|eg> should equal g_eff
    eg = coupler.fock_index([1, 0, 0])
    ge = coupler.fock_index([0, 1, 0])
    pump_period = TWO_PI / coupler.delta(1, 0)
    times = np.linspace(0.0, 40 * pump_period, 8000, endpoint=False)
    dc_matrix_element = np.mean([coupler.hamiltonian_matrix(t)[ge, eg] for t in times])

    print(f"DC <ge|H|eg|/2pi          = {abs(dc_matrix_element) / TWO_PI * 1e3:.4f} MHz")
    print(f"Zhou 6 g3 la lb |eta| /2pi = {coupler.iswap_rate(0, 1) / TWO_PI * 1e3:.4f} MHz")
    print(f"ratio                      = {abs(dc_matrix_element) / coupler.iswap_rate(0, 1):.5f}  (expect 1.0)")

    # transmon anharmonicity: |2>_a should be shifted by alpha_a (diagonal, static)
    alpha_a_GHz = -0.20
    anh = ZhouCoupler(mode_freqs_GHz=[4.0, 5.0, 7.0], coupler_index=2,
                      participations={0: 0.15, 1: 0.15}, nonlinearities={3: 0.10},
                      levels=[3, 3, 6], anharmonicities_GHz={0: alpha_a_GHz})
    H0 = anh.hamiltonian_matrix(0.0)
    e0 = H0[anh.fock_index([0, 0, 0])].real[anh.fock_index([0, 0, 0])]
    e1 = H0[anh.fock_index([1, 0, 0])].real[anh.fock_index([1, 0, 0])]
    e2 = H0[anh.fock_index([2, 0, 0])].real[anh.fock_index([2, 0, 0])]
    shift = ((e2 - e1) - (e1 - e0)) / TWO_PI       # (E2-E1)-(E1-E0) = alpha_a
    print(f"\nanharmonicity |2>_a shift  = {shift:.4f} GHz  (expect {alpha_a_GHz})")