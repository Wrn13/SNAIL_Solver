"""
envelope.py
===========

Pump envelopes, frequency chirps, and the pump-tone container.

This module is the SINGLE SOURCE OF TRUTH for these classes; `zhou_coupler`
re-exports every public name here, so ``from zhou_coupler import RaisedCosine``
keeps working unchanged.

Split of responsibilities
-------------------------
* :class:`Envelope` carries the pulse SHAPE and its peak scale ``amp``. Its
  ``value`` may be complex (the DRAG quadrature and the CRAB I/Q ansatz both
  need that), but ``|value|`` is what the iSWAP normalization integrates.
* :class:`Chirp` carries a time-dependent FREQUENCY offset delta(t) about the
  tone's fixed carrier ``w_p_GHz``, as an accumulated phase Phi(t) = int_0^t delta.
* :class:`PumpTone` binds the two to a carrier.

Why a chirp needs no solver changes
-----------------------------------
A pump letter enters X(t) as ``eta_p(t) e^{-i w_p t}``. Chirping the carrier,
w_p -> w_p + delta(t), is therefore ALGEBRAICALLY IDENTICAL to multiplying the
complex envelope by a phase::

    eta_p(t) e^{-i(w_p t + Phi(t))} = [eta_p(t) e^{-i Phi(t)}] e^{-i w_p t}

so the fixed carrier w_p stays the expansion reference and the chirp rides in the
envelope. A Hamiltonian term carrying k net pump quanta picks up ``e^{-i k Phi(t)}``
automatically, which is exactly the k-quanta net-carrier shift. Nothing in
``_flux_letters``, ``expand_terms`` or ``to_qutip_hamiltonian`` has to know.
This is the time-dependent generalization of the constant ``offset_rad`` trick in
``grape._H``.

Array API and tracing
---------------------
Every shape function is available in an ``xp``-generic form -- ``value_at(t, xp)``,
``deriv_at(t, xp)``, ``Chirp.phase(t, xp)`` -- accepting scalar OR array ``t`` and
built only from ``xp`` primitives, following the convention already used by
``grape._iq_ansatz``. With ``xp=jax.numpy`` these stay trace-clean: no ``float()``
and no ``asarray(..., dtype=)`` is applied to any value that can carry a tracer,
either of which raises (TracerArrayConversionError / ConcretizationTypeError).
Domain guards are ``xp.where`` masks rather than Python ``if``, since a branch on a
traced value cannot compile.

The scalar ``value(t)`` / ``deriv(t)`` methods remain as thin wrappers so that the
existing per-time solver callbacks are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb as _comb
from typing import Any, Callable, Optional, Sequence

import numpy as np

# numpy>=2 renamed trapz -> trapezoid; fall back only if needed (trapz is gone in 2.x).
_trapezoid: Callable[..., float] = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
TWO_PI = 2.0 * np.pi


# ===========================================================================
# Pump envelopes  (shape eps(t) on [0, t_g]; peak scale carried in `amp`)
# ===========================================================================
class Envelope:
    """Base class for a pump-amplitude envelope eps(t) on [0, t_g].

    Subclasses implement `value_at` and `deriv_at` (the ``xp``-generic forms);
    `value` / `deriv` are scalar wrappers around them and need no overriding.
    `area` integrates |eps| over the gate (subclasses with closed forms override
    it).

    Parameters
    ----------
    amp : float
        Peak amplitude. Carries |eps| in rad/ns, or directly |eta| when the
        owning PumpTone has ``is_eta=True``.
    t_g : float
        Gate duration in ns; the envelope is supported on [0, t_g].

    Attributes
    ----------
    is_complex : bool
        Class attribute; True if `value` can return a complex amplitude. Only
        controls the scalar wrapper's cast.
    """

    is_complex: bool = False

    def __init__(self, amp: float, t_g: float) -> None:
        self.amp: float = float(amp)
        self.t_g: float = float(t_g)

    # -- xp-generic core (subclasses implement these) ----------------------
    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Envelope amplitude at time(s) `t` (ns), built from `xp` primitives.

        Accepts a scalar or an array and returns the same shape. Must not branch
        on `t` (use ``xp.where``) so the body stays vectorizable and traceable.
        """
        raise NotImplementedError

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Time derivative d eps/dt at time(s) `t` (ns), built from `xp`."""
        raise NotImplementedError

    # -- scalar wrappers (the per-time solver callback path) ---------------
    def value(self, t: float) -> Any:
        """Envelope amplitude at a single time `t` (ns)."""
        v = self.value_at(float(t), np)
        return complex(v) if self.is_complex else float(v)

    def deriv(self, t: float) -> Any:
        """Time derivative d eps/dt at a single time `t` (ns)."""
        v = self.deriv_at(float(t), np)
        return complex(v) if self.is_complex else float(v)

    # -- higher derivatives (the recursive-DRAG requirement) ---------------
    def jet_at(self, t: Any, order: int, xp: Any = np) -> tuple:
        """Derivatives ``(eps, eps', ..., eps^(order))`` at time(s) `t` (ns).

        Recursive DRAG (:mod:`snail_solver.drag`) differentiates the amplitude
        produced by the previous correction, so a K-fold composition needs the base
        envelope to order K. Every envelope in this module is a finite trigonometric
        polynomial, so the overrides below are all closed forms -- no quadrature, no
        finite differences, and trace-clean like `value_at`/`deriv_at`.

        `order` must be a static Python int (it controls an unrolled loop). The base
        implementation covers orders 0 and 1 by delegating to the existing methods;
        anything higher is a per-subclass responsibility.
        """
        order = int(order)
        if order <= 0:
            return (self.value_at(t, xp),)
        if order == 1:
            return (self.value_at(t, xp), self.deriv_at(t, xp))
        raise NotImplementedError(
            f"{type(self).__name__} does not implement jet_at beyond order 1; "
            f"recursive DRAG with K channels needs order K")

    # -- integrated quantities ---------------------------------------------
    def area(self) -> float:
        """Return integral_0^{t_g} value(t) dt by quadrature (override if a closed
        form exists).

        This is the quantity `set_pump(..., normalize_iswap=...)` divides by to
        calibrate the pi/2 rotation. It is unaffected by a chirp: a chirp lives on
        the PumpTone, not the envelope, and is a pure phase that leaves |eta|
        alone. Do not add a chirp correction here.
        """  # noqa: D205
        ts = np.linspace(0.0, self.t_g, 4001)
        return float(_trapezoid(self.value_at(ts, np), ts))

    def samples(self, n: int) -> np.ndarray:
        """`n` midpoint samples of eps(t) across the gate (for plotting/storage)."""
        ts = (np.arange(int(n)) + 0.5) * (self.t_g / int(n))
        return np.asarray(self.value_at(ts, np), dtype=complex)

    # -- optimizer parameter interface (no free parameters by default) -----
    @property
    def n_params(self) -> int:
        """Number of free real shape parameters."""
        return 0

    def get_params(self) -> np.ndarray:
        """Free shape parameters as a flat real vector."""
        return np.zeros(0)

    def set_params(self, p: Any) -> None:
        """Load a flat real vector as produced by :meth:`get_params`."""
        if np.asarray(p).size != 0:
            raise ValueError(f"{type(self).__name__} takes no shape parameters")

    # -- shared helper ------------------------------------------------------
    def _support(self, t: Any, xp: Any) -> Any:
        """1.0 inside [0, t_g], 0.0 outside -- as a mask, never a branch."""
        return xp.where((t >= 0.0) & (t <= self.t_g), 1.0, 0.0)


class ConstantPulse(Envelope):
    """Flat pump with instantaneous on/off; handy for steady-state rate checks."""

    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Amplitude `amp` for t in [0, t_g], else 0."""
        return self.amp * self._support(t, xp)

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Zero everywhere (flat pulse)."""
        return 0.0 * xp.asarray(t)

    def jet_at(self, t: Any, order: int, xp: Any = np) -> tuple:
        """Value, then zeros: a flat pulse has no interior derivatives."""
        v = self.value_at(t, xp)
        return (v,) + tuple(0.0 * v for _ in range(int(order)))

    def area(self) -> float:
        """Closed form: amp * t_g."""
        return self.amp * self.t_g


class RaisedCosine(Envelope):
    """Hann (raised-cosine) envelope: smooth turn-on/off, vanishing endpoints,.

    value(t) = amp/2 [1 - cos(2 pi t / t_g)] ,  t in [0, t_g] .
    """

    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Hann amplitude at `t` (ns); 0 outside [0, t_g]."""
        return (self.amp * 0.5 * (1.0 - xp.cos(TWO_PI * t / self.t_g))
                * self._support(t, xp))

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Analytic derivative of the Hann window at `t` (ns); 0 outside [0, t_g]."""
        return (self.amp * 0.5 * (TWO_PI / self.t_g) * xp.sin(TWO_PI * t / self.t_g)
                * self._support(t, xp))

    def jet_at(self, t: Any, order: int, xp: Any = np) -> tuple:
        """Closed-form derivatives to any order.

        Only the cosine carries the time dependence, so for n >= 1
        ``d^n/dt^n[-amp/2 cos(w t)] = -amp/2 w^n cos(w t + n pi/2)`` with
        ``w = 2 pi / t_g``. At n = 1 this reproduces :meth:`deriv_at` exactly.

        Note what this shows about the boundaries: ``eps(0) = eps'(0) = 0`` but
        ``eps''(0) = amp/2 w^2 != 0``. A Hann window therefore supports exactly ONE
        clean derivative correction -- see :mod:`snail_solver.drag`.
        """
        w = TWO_PI / self.t_g
        support = self._support(t, xp)
        out = [self.value_at(t, xp)]
        for n in range(1, int(order) + 1):
            out.append(-self.amp * 0.5 * w ** n
                       * xp.cos(w * t + n * (np.pi / 2.0)) * support)
        return tuple(out)

    def area(self) -> float:
        """Closed form: amp * t_g / 2 (exact integral of the Hann window)."""
        return self.amp * self.t_g / 2.0


class SinePowerRamp(Envelope):
    r"""The Li/Calarco/Motzoi Eq. (13) shape: a ramp with m vanishing end derivatives.

    .. math::
        \Omega^{(m)}(t) = \mathrm{amp}\;\mathcal{I}_0
            \int_0^{t} \sin^m\!\big(\pi t' / t_r\big)\,dt' ,\qquad 0 \le t \le t_r

    normalized by :math:`\mathcal{I}_0` so that :math:`\Omega(t_r) = \mathrm{amp}`,
    then held flat and mirrored down over ``[t_g - t_r, t_g]``.

    Why this shape exists
    ---------------------
    Recursive DRAG differentiates the pulse once per suppressed channel, and the
    paper requires the base shape to be m-times differentiable with **all m
    derivatives vanishing at both ends**, "which guarantees the validity of the frame
    transformation". :class:`RaisedCosine` only manages two
    (``eps(0) = eps'(0) = 0`` but ``eps''(0) != 0``), and that is not a small error:
    with ``F^(2)`` innermost the imaginary term dominates as ``t -> 0``, so a shape
    vanishing as ``t^p`` comes out as ``t^(p - 1/2)``. Hann has ``p = 2``, so two
    further derivatives give a ``t^(-1/2)`` DIVERGENCE at both gate edges -- measured,
    and pinned by ``test_hann_diverges_under_the_full_recursion``. Here
    ``eps ~ t^(m+1)``, so ``m = 3`` carries a 3-channel recursion comfortably.

    Keep m as small as the channel count allows: the paper warns that larger m packs
    more high-frequency content into the pulse, which is itself a source of
    non-adiabatic error.

    Reduces exactly to a Hann window
    --------------------------------
    ``SinePowerRamp(amp, t_g, m=1)`` (i.e. ``t_rise = t_g/2``, no plateau) IS
    :class:`RaisedCosine`, to floating-point equality -- the paper says as much
    ("for m = 1 and with zero holding time, the pulse is the same as the Hann
    window"), and a test asserts it. That makes every Hann-specific constant elsewhere
    in the codebase a special case of this family rather than a separate derivation.

    .. note::
       The paper PRINTS the integrand as ``sin^m(pi t'/2 t_r)``, which would make it
       equal 1 (its maximum) at ``t' = t_r`` and so leave ``Omega'(t_r) != 0`` --
       contradicting the very property the equation is introduced to provide, and
       failing its own m=1 claim by 0.25 in amplitude. The ``pi t'/t_r`` reading used
       here vanishes at both ends and reproduces the Hann window to 6e-16.

    Parameters
    ----------
    amp : float
        Peak amplitude (reached on the plateau).
    t_g : float
        Gate duration (ns).
    m : int, default 3
        Number of vanishing derivatives at each edge. Must be >= 1.
    t_rise : float, optional
        Ramp duration. Defaults to ``t_g/2`` -- no plateau, the direct Hann analogue.
        Must satisfy ``2 t_rise <= t_g``.
    """

    def __init__(self, amp: float, t_g: float, m: int = 3,
                 t_rise: Optional[float] = None) -> None:
        super().__init__(amp, t_g)
        self.m = int(m)
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        self.t_rise = float(self.t_g / 2.0 if t_rise is None else t_rise)
        if not (0.0 < self.t_rise <= self.t_g / 2.0 + 1e-12):
            raise ValueError(f"t_rise must be in (0, t_g/2]; got {self.t_rise} "
                             f"with t_g = {self.t_g}")
        # sin^m(x) = sum_j W_j e^{i k_j x},  k_j = m - 2j, from the binomial expansion
        # of ((e^{ix} - e^{-ix}) / 2i)^m. Complex exponentials rather than a
        # parity-split sin/cos series: one code path covers odd and even m, and both
        # the antiderivative and every derivative are then one-liners.
        j = np.arange(self.m + 1)
        self._k = (self.m - 2 * j).astype(float)
        self._W = (np.array([_comb(self.m, int(x)) for x in j], dtype=complex)
                   * (-1.0) ** j / (2.0j) ** self.m)
        self._w = np.pi / self.t_rise                      # angular rate of the ramp
        # I_0, fixed by Omega(t_rise) = amp
        self._I0 = 1.0 / float(self._integral(self.t_rise))

    # -- the ramp, its integral and its derivatives ------------------------
    def _integral(self, t: Any, xp: Any = np) -> Any:
        """``int_0^t sin^m(w t') dt'`` (un-normalized), termwise in the exponentials.

        The basis lives on a trailing axis, so a scalar and an array of times take
        the same code path (the same trick as ``IQFourierEnvelope._basis``).
        """
        k, W, w = self._k, self._W, self._w
        zero = np.abs(k) < 1e-12
        k_safe = np.where(zero, 1.0, k)        # avoid 0/0 in the discarded branch
        tt = xp.asarray(t)[..., None]
        # the k = 0 term integrates to t; every other to (e^{i k w t} - 1)/(i k w)
        terms = xp.where(zero, tt + 0.0j,
                         (xp.exp(1j * k_safe * w * tt) - 1.0) / (1j * k_safe * w))
        return xp.real(xp.sum(terms * W, axis=-1))

    def _ramp_derivs(self, t: Any, order: int, xp: Any) -> list:
        """``R, R', ..., R^(order)`` of the normalized rising ramp at `t`."""
        k, W, w, I0 = self._k, self._W, self._w, self._I0
        out = [I0 * self._integral(t, xp)]
        arg = xp.exp(1j * (k * w) * xp.asarray(t)[..., None])
        for n in range(1, int(order) + 1):
            # R^(n) = I0 * d^(n-1)/dt^(n-1) sin^m(w t)
            out.append(I0 * xp.real(xp.sum(arg * (W * (1j * k * w) ** (n - 1)),
                                           axis=-1)))
        return out

    def jet_at(self, t: Any, order: int, xp: Any = np) -> tuple:
        """Closed-form derivatives to any order, across rise / plateau / fall.

        The three regions are combined with ``xp.where`` masks, never a Python
        branch, so a scalar, an array and a tracer all take the same path. The fall
        is the rise reflected, so its n-th derivative carries ``(-1)^n``.
        """
        order = int(order)
        t = xp.asarray(t)
        t_r, t_g = self.t_rise, self.t_g
        up = self._ramp_derivs(t, order, xp)
        down = self._ramp_derivs(t_g - t, order, xp)
        # Closed on the left at t_r, so the ramp -- not the plateau -- owns the
        # junction. With no plateau (t_rise = t_g/2) that point is the bell's peak,
        # where derivatives above order m are genuinely NON-zero; handing it to the
        # plateau branch would zero them and silently break the m=1/Hann identity.
        rising = t <= t_r
        falling = t >= (t_g - t_r)
        support = self._support(t, xp)
        out = []
        for n in range(order + 1):
            flat = (xp.ones_like(t) if n == 0 else 0.0 * t)    # plateau value
            v = xp.where(rising, up[n],
                         xp.where(falling, (-1.0) ** n * down[n], flat))
            out.append(self.amp * v * support)
        return tuple(out)

    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Envelope amplitude at `t` (ns); 0 outside [0, t_g]."""
        return self.jet_at(t, 0, xp)[0]

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Analytic d eps/dt at `t` (ns); 0 outside [0, t_g]."""
        return self.jet_at(t, 1, xp)[1]

    def area(self) -> float:
        """Closed form: two ramps plus the plateau.

        Exact rather than quadrature because ``set_pump(normalize_iswap=...)``
        divides by this to calibrate the pi/2 rotation.
        """
        k, W, w, t_r = self._k, self._W, self._w, self.t_rise
        zero = np.abs(k) < 1e-12
        k_safe = np.where(zero, 1.0, k)
        # int_0^{t_r} of each antiderivative term
        inner = np.where(zero, t_r ** 2 / 2.0,
                         ((np.exp(1j * k_safe * w * t_r) - 1.0) / (1j * k_safe * w)
                          - t_r) / (1j * k_safe * w))
        ramp_area = self._I0 * float(np.real(np.sum(inner * W)))
        return self.amp * (2.0 * ramp_area + (self.t_g - 2.0 * t_r))


class IQFourierEnvelope(Envelope):
    r"""Complex I/Q envelope: a fixed shape times a truncated Fourier modulation.

    .. math::
        \eta(t) = \mathrm{amp}\;S(t)\;\Big[1
            + \sum_k \big(a^I_k + i\,a^Q_k\big)\sin(\omega_k t)
            + \sum_k \big(b^I_k + i\,b^Q_k\big)\cos(\omega_k t)\Big]

    with :math:`S(t) = \tfrac12[1 - \cos(2\pi t/t_g)]` the Hann shape function.

    This is the ansatz container used by the CRAB / JOPT optimizers in
    ``grape.py``. Two properties make it safe to drop into the existing pump
    machinery:

    * ``value`` returns a COMPLEX amplitude. ``ZhouCoupler._eta`` already treats
      the envelope value as complex (it has to, for the DRAG quadrature), so the
      full QuTiP path -- ``to_qutip_hamiltonian`` -> ``propagator_columns`` /
      ``iswap_fidelity`` -- evaluates an arbitrary shaped pulse with no changes.
      That means an optimizer can use the compiled QuTiP solver as its black box
      instead of a reduced model.
    * With all coefficients zero it reduces EXACTLY to
      :class:`RaisedCosine`, so ``set_pump(..., normalize_iswap=...)`` performed
      with a raised cosine stays valid, ``peak_eta`` is unchanged, and the
      optimizer starts from the sweep's baseline gate.

    The Hann prefactor enforces :math:`\eta(0) = \eta(t_g) = 0` for ANY
    coefficients, so the optimizer cannot produce a pulse with a discontinuous
    turn-on -- the usual CRAB "shape function" role.

    Parameters
    ----------
    amp : float
        Peak amplitude of the unmodulated shape (see :class:`Envelope`).
    t_g : float
        Gate duration (ns).
    freqs : array_like, optional
        Angular frequencies :math:`\omega_k` (rad/ns) of the basis. For CRAB these
        are randomized about the harmonics; ``None`` (default) gives no
        modulation, i.e. a plain raised cosine.
    sin_I, sin_Q, cos_I, cos_Q : array_like, optional
        Coefficients of the sin/cos basis in the in-phase (I) and quadrature (Q)
        components. Each must match ``freqs`` in length; omitted arrays are zero.

    Notes
    -----
    ``area`` integrates the REAL part, matching the leading-order iSWAP
    normalization convention (the in-phase component is what drives the swap);
    with zero coefficients it is exactly ``amp * t_g / 2``.
    """

    is_complex = True

    def __init__(self, amp: float, t_g: float, freqs=None,
                 sin_I=None, sin_Q=None, cos_I=None, cos_Q=None) -> None:
        super().__init__(amp, t_g)
        self.freqs: np.ndarray = (np.zeros(0) if freqs is None
                                  else np.asarray(freqs, dtype=float))
        n = self.freqs.size

        def _co(v):
            if v is None:
                return np.zeros(n)
            arr = np.asarray(v, dtype=float)
            if arr.size != n:
                raise ValueError(f"coefficient array has {arr.size} entries, "
                                 f"expected {n} to match freqs")
            return arr

        self.sin_I, self.sin_Q = _co(sin_I), _co(sin_Q)
        self.cos_I, self.cos_Q = _co(cos_I), _co(cos_Q)

    # -- the modulation factor and its derivative --------------------------
    def _basis(self, t: Any, xp: Any):
        """sin/cos of every basis frequency, broadcast over `t`.

        ``t[..., None] * freqs`` puts the basis on a trailing axis so a scalar and
        an array of times take the same code path; the contractions below then sum
        over that axis.
        """
        arg = xp.asarray(t)[..., None] * self.freqs
        return xp.sin(arg), xp.cos(arg)

    def _mod(self, t: Any, xp: Any = np) -> Any:
        """Modulation M(t); 1 when there is no basis (plain raised cosine)."""
        if self.freqs.size == 0:
            return 1.0 + 0.0j
        s, c = self._basis(t, xp)
        # sum over the trailing basis axis -- NOT float(), which would concretize
        # a tracer and break the JAX gradient path.
        return ((1.0 + xp.sum(s * self.sin_I, axis=-1) + xp.sum(c * self.cos_I, axis=-1))
                + 1j * (xp.sum(s * self.sin_Q, axis=-1) + xp.sum(c * self.cos_Q, axis=-1)))

    def _dmod(self, t: Any, xp: Any = np) -> Any:
        """dM/dt."""
        if self.freqs.size == 0:
            return 0.0 + 0.0j
        w = self.freqs
        s, c = self._basis(t, xp)
        return ((xp.sum(c * (self.sin_I * w), axis=-1)
                 - xp.sum(s * (self.cos_I * w), axis=-1))
                + 1j * (xp.sum(c * (self.sin_Q * w), axis=-1)
                        - xp.sum(s * (self.cos_Q * w), axis=-1)))

    def _shape(self, t: Any, xp: Any = np) -> Any:
        """Hann shape function S(t)."""
        return 0.5 * (1.0 - xp.cos(TWO_PI * t / self.t_g))

    def _dshape(self, t: Any, xp: Any = np) -> Any:
        """dS/dt."""
        return 0.5 * (TWO_PI / self.t_g) * xp.sin(TWO_PI * t / self.t_g)

    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Complex envelope amplitude at `t` (ns); 0 outside [0, t_g]."""
        return self.amp * self._shape(t, xp) * self._mod(t, xp) * self._support(t, xp)

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Analytic d(eta)/dt at `t` (ns); 0 outside [0, t_g]."""
        return (self.amp * (self._dshape(t, xp) * self._mod(t, xp)
                            + self._shape(t, xp) * self._dmod(t, xp))
                * self._support(t, xp))

    def _shape_derivs(self, t: Any, order: int, xp: Any) -> list:
        """``S, S', ..., S^(order)`` for the Hann shape function."""
        W = TWO_PI / self.t_g
        return [self._shape(t, xp)] + [
            -0.5 * W ** n * xp.cos(W * t + n * (np.pi / 2.0))
            for n in range(1, int(order) + 1)]

    def _mod_derivs(self, t: Any, order: int, xp: Any) -> list:
        """``M, M', ..., M^(order)`` for the Fourier modulation."""
        out = [self._mod(t, xp)]
        if self.freqs.size == 0:            # M == 1: every derivative vanishes
            zero = 0.0 * xp.asarray(t)
            return out + [zero + 0.0j for _ in range(int(order))]
        w = self.freqs
        arg = xp.asarray(t)[..., None] * w
        for n in range(1, int(order) + 1):
            s = xp.sin(arg + n * (np.pi / 2.0))
            c = xp.cos(arg + n * (np.pi / 2.0))
            wn = w ** n
            out.append((xp.sum(s * (self.sin_I * wn), axis=-1)
                        + xp.sum(c * (self.cos_I * wn), axis=-1))
                       + 1j * (xp.sum(s * (self.sin_Q * wn), axis=-1)
                               + xp.sum(c * (self.cos_Q * wn), axis=-1)))
        return out

    def jet_at(self, t: Any, order: int, xp: Any = np) -> tuple:
        """Closed-form derivatives to any order, by Leibniz over ``S(t) M(t)``.

        Both factors are finite trigonometric polynomials, so each is differentiated
        by phase-shifting its arguments; the product rule is then the plain binomial
        sum. At order 1 this reproduces :meth:`deriv_at` exactly.
        """
        order = int(order)
        S = self._shape_derivs(t, order, xp)
        M = self._mod_derivs(t, order, xp)
        support = self._support(t, xp)
        return tuple(self.amp * support * sum(
            _comb(n, j) * S[j] * M[n - j] for j in range(n + 1))
            for n in range(order + 1))

    def area(self) -> float:
        """Integral of Re[eta] over the gate (see Notes); closed form when the
        basis is empty."""
        if self.freqs.size == 0:
            return self.amp * self.t_g / 2.0
        ts = np.linspace(0.0, self.t_g, 4001)
        return float(_trapezoid(np.real(self.value_at(ts, np)), ts))

    # -- flat parameter-vector interface for optimizers --------------------
    @property
    def n_params(self) -> int:
        """Number of free real coefficients (4 per basis frequency)."""
        return 4 * self.freqs.size

    def get_params(self) -> np.ndarray:
        """Coefficients as one flat real vector [sin_I, sin_Q, cos_I, cos_Q]."""
        return np.concatenate([self.sin_I, self.sin_Q, self.cos_I, self.cos_Q])

    def set_params(self, p) -> None:
        """Load a flat real vector as produced by :meth:`get_params`."""
        p = np.asarray(p, dtype=float)
        n = self.freqs.size
        if p.size != 4 * n:
            raise ValueError(f"expected {4 * n} parameters, got {p.size}")
        self.sin_I, self.sin_Q, self.cos_I, self.cos_Q = (
            p[:n].copy(), p[n:2 * n].copy(), p[2 * n:3 * n].copy(), p[3 * n:].copy())


# ===========================================================================
# Frequency chirp
# ===========================================================================
def _legendre_stack(u: Any, degree: int, xp: Any):
    """Shifted-Legendre values P_0(u) .. P_degree(u), by the standard recurrence.

    Built with the recurrence rather than ``numpy.polynomial`` so the same code
    runs under ``jax.numpy`` (and so it vectorizes over an array of `u`).
    """
    ones = xp.ones_like(u)
    out = [ones]
    if degree >= 1:
        out.append(u)
    for k in range(1, degree):
        # (k+1) P_{k+1} = (2k+1) u P_k - k P_{k-1}
        out.append(((2 * k + 1) * u * out[k] - k * out[k - 1]) / (k + 1))
    return out


def _legendre_deriv_stack(u: Any, degree: int, order: int, xp: Any):
    """``P[m][k] = d^m P_k / du^m`` for m <= `order`, k <= `degree`.

    The same recurrence as :func:`_legendre_stack`, differentiated m times in place.
    Since ``u P_k`` is a product with a LINEAR factor, Leibniz truncates after two
    terms::

        (k+1) P_{k+1}^(m) = (2k+1) [ u P_k^(m) + m P_k^(m-1) ] - k P_{k-1}^(m)

    Used by :meth:`Chirp.detuning_jet`; kept here beside the value recurrence so the
    two cannot drift.
    """
    zero = 0.0 * u
    ones = xp.ones_like(u)
    stack = [_legendre_stack(u, degree, xp)]
    for m in range(1, int(order) + 1):
        prev = stack[m - 1]
        row = [zero]
        if degree >= 1:
            row.append(ones if m == 1 else zero)          # P_1 = u
        for k in range(1, degree):
            row.append(((2 * k + 1) * (u * row[k] + m * prev[k])
                        - k * row[k - 1]) / (k + 1))
        stack.append(row)
    return stack


class Chirp:
    r"""A time-dependent pump-frequency offset delta(t) about a fixed carrier.

    The offset is a Legendre polynomial in the normalized gate time
    :math:`u = 2t/t_g - 1 \in [-1, 1]`:

    .. math::
        \delta(t) = 2\pi \sum_k c_k P_k(u) ,\qquad
        \Phi(t) = \int_0^t \delta(t')\,dt'

    The Legendre basis is used (rather than raw powers of `t`) because its terms
    are orthogonal over the gate, which keeps an optimizer's parameters from
    fighting each other: `c_0` is the mean detuning, `c_1` the linear chirp rate,
    and higher terms add structure without shifting the mean.

    :math:`\Phi` is obtained in CLOSED FORM, not by quadrature, from
    :math:`\int_{-1}^{u} P_k = (P_{k+1} - P_{k-1})/(2k+1)` for k >= 1 and
    :math:`\int_{-1}^{u} P_0 = u + 1`:

    .. math::
        \Phi(t) = \pi t_g \Big[ c_0 (u+1)
                  + \sum_{k\ge 1} c_k \frac{P_{k+1}(u) - P_{k-1}(u)}{2k+1} \Big]

    Special cases worth knowing:

    * ``coeffs_GHz = [c0]`` is a CONSTANT detuning, and is exactly equivalent to
      building the tone at ``w_p_GHz + c0`` (see the module docstring). This is a
      free regression test and is asserted in ``test_physics``.
    * ``coeffs_GHz = [c0, c1]`` is a linear chirp sweeping from ``c0 - c1`` to
      ``c0 + c1`` GHz across the gate.

    Parameters
    ----------
    coeffs_GHz : sequence of float
        Legendre coefficients of delta(t)/2pi, in GHz. Empty (or all-zero) means
        no chirp.
    t_g : float
        Gate duration (ns); sets the normalization of `u`.

    Notes
    -----
    A chirp is a pure PHASE: it leaves ``|eta(t)|`` untouched. Therefore it does
    not change ``Envelope.area()``, the ``normalize_iswap`` amplitude calibration,
    or ``peak_eta``. Do not "fix" those to account for it.

    `t` is clipped to [0, t_g] before evaluation. The envelope is zero outside the
    gate so this changes no observable, but it stops a high-degree polynomial from
    reporting a meaningless multi-thousand-radian phase just outside the support.
    """

    def __init__(self, coeffs_GHz: Sequence[float], t_g: float) -> None:
        self.coeffs_GHz: np.ndarray = np.asarray(coeffs_GHz, dtype=float).ravel()
        self.t_g: float = float(t_g)

    def __repr__(self) -> str:  # noqa: D105
        return f"Chirp(coeffs_GHz={list(self.coeffs_GHz)}, t_g={self.t_g})"

    @property
    def is_trivial(self) -> bool:
        """True if this chirp is identically zero (so callers can skip the phase)."""
        return self.coeffs_GHz.size == 0 or not np.any(self.coeffs_GHz)

    def _u(self, t: Any, xp: Any):
        """Normalized gate time u = 2t/t_g - 1, clipped to [-1, 1]."""
        return xp.clip(2.0 * xp.asarray(t) / self.t_g - 1.0, -1.0, 1.0)

    def detuning(self, t: Any, xp: Any = np) -> Any:
        """Instantaneous frequency offset delta(t) in rad/ns."""
        n = self.coeffs_GHz.size
        if n == 0:
            return 0.0 * xp.asarray(t)
        u = self._u(t, xp)
        P = _legendre_stack(u, n - 1, xp)
        total = sum(c * P[k] for k, c in enumerate(self.coeffs_GHz))
        return TWO_PI * total

    def detuning_jet(self, t: Any, order: int, xp: Any = np) -> tuple:
        """``delta, delta', ..., delta^(order)`` at time(s) `t`, in rad/ns^(1+m).

        Recursive DRAG applied verbatim (Eq. 4 of Li/Calarco/Motzoi) differentiates
        ``Omega^n / Delta(t)`` as a whole, so on a CHIRPED tone it needs derivatives
        of the beat, not just its value. Those did not exist before this method:
        ``Delta(t) = Delta_0 - k delta(t)``, so ``Delta^(m) = -k delta^(m)``.

        With ``u = 2t/t_g - 1`` the chain rule gives
        ``d^m delta/dt^m = 2 pi (2/t_g)^m sum_k c_k P_k^(m)(u)``.

        Outside the gate :meth:`_u` CLIPS, so ``delta`` is constant there and every
        derivative is genuinely zero. The support mask below states that explicitly
        rather than leaning on ``xp.clip``'s subgradient, which is backend-dependent
        at the boundary. The 0th entry is left unmasked so it is exactly
        :meth:`detuning` -- changing that would move the DRAG beat outside the gate.
        """
        order = int(order)
        n = self.coeffs_GHz.size
        if n == 0:
            zero = 0.0 * xp.asarray(t)
            return tuple(zero for _ in range(order + 1))
        u = self._u(t, xp)
        P = _legendre_deriv_stack(u, n - 1, order, xp)
        support = xp.where((t >= 0.0) & (t <= self.t_g), 1.0, 0.0)
        du_dt = 2.0 / self.t_g
        # order 0 from the stack we already have -- calling self.detuning() here would
        # rebuild the whole Legendre recurrence a second time, which showed up as ~10x
        # redundant work in the scalar solver callback.
        out = [TWO_PI * sum(c * P[0][k] for k, c in enumerate(self.coeffs_GHz))]
        for m in range(1, order + 1):
            total = sum(c * P[m][k] for k, c in enumerate(self.coeffs_GHz))
            out.append(TWO_PI * du_dt ** m * total * support)
        return tuple(out)

    def phase(self, t: Any, xp: Any = np) -> Any:
        """Accumulated chirp phase Phi(t) = int_0^t delta(t') dt', in radians."""
        n = self.coeffs_GHz.size
        if n == 0:
            return 0.0 * xp.asarray(t)
        u = self._u(t, xp)
        # need P up to degree n (the k = n-1 term uses P_{k+1} = P_n)
        P = _legendre_stack(u, n, xp)
        total = self.coeffs_GHz[0] * (u + 1.0)
        for k in range(1, n):
            total = total + self.coeffs_GHz[k] * (P[k + 1] - P[k - 1]) / (2 * k + 1)
        return np.pi * self.t_g * total

    # -- flat parameter-vector interface for optimizers --------------------
    @property
    def n_params(self) -> int:
        """Number of free chirp coefficients."""
        return int(self.coeffs_GHz.size)

    def get_params(self) -> np.ndarray:
        """Chirp coefficients (GHz) as a flat real vector."""
        return self.coeffs_GHz.copy()

    def set_params(self, p) -> None:
        """Load a flat real vector as produced by :meth:`get_params`."""
        p = np.asarray(p, dtype=float).ravel()
        if p.size != self.coeffs_GHz.size:
            raise ValueError(f"expected {self.coeffs_GHz.size} chirp coefficients, "
                             f"got {p.size}")
        self.coeffs_GHz = p.copy()


def make_chirp(coeffs_GHz: Optional[Sequence[float]], t_g: float) -> Optional[Chirp]:
    """Build a :class:`Chirp`, or None when there is nothing to apply.

    Returning None for an absent/zero chirp keeps the un-chirped solver path
    byte-identical to before this feature existed (``_eta`` skips the phase
    entirely), which is what makes the "chirp is inert when unset" regression
    test meaningful.
    """
    if coeffs_GHz is None:
        return None
    chirp = Chirp(coeffs_GHz, t_g)
    return None if chirp.is_trivial else chirp


#: Envelope kinds addressable by name, for ``config["envelope"]`` and
#: ``find_stark_resonance.scan(shape=...)``. The Hann default is deliberately
#: unchanged: ``sine_power`` is opt-in, because switching the base shape moves the
#: amplitude/area algebra the tune-up is built on (see ``tune_up.area_factor``).
ENVELOPE_KINDS = {
    "raised_cosine": RaisedCosine,
    "constant": ConstantPulse,
    "sine_power": SinePowerRamp,
}


def envelope_from_config(config: Dict[str, Any], t_g: float, amp: float = 1.0):
    """The envelope THIS device is configured to play, at a real gate length.

    Every probe that claims to be "the actual gate pulse" has to be built from the
    configured shape, not from a hardcoded one. ``tune_up.chirp_from_measured_shift``
    already reads the envelope off the config for exactly this reason -- selecting a
    different base shape (which recursive DRAG requires past one channel, see
    :class:`SinePowerRamp`) must not leave a calibration tracking a pulse the solver
    is not playing. This is the same rule for the chevron probes.

    Parameters
    ----------
    config : dict
        Merged device configuration. Reads ``envelope`` (default
        ``"raised_cosine"``) and, for ``sine_power``, ``envelope_m`` and
        ``envelope_rise_frac``.
    t_g : float
        Gate duration (ns). ``envelope_rise_frac`` is a FRACTION of it, which is what
        keeps the shape -- and hence the chirp built from it -- independent of `t_g`.
    amp : float, default 1.0
        Peak amplitude before any iSWAP normalization.

    Returns
    -------
    Envelope
    """
    kind = str(config.get("envelope", "raised_cosine"))
    cls = ENVELOPE_KINDS.get(kind)
    if cls is None:
        raise ValueError(f"unknown envelope {kind!r}; known: {sorted(ENVELOPE_KINDS)}")
    kw: Dict[str, Any] = {}
    if cls is SinePowerRamp:
        kw = {"m": int(config.get("envelope_m", 3)),
              "t_rise": float(t_g) * float(config.get("envelope_rise_frac", 0.5))}
    return cls(amp=float(amp), t_g=float(t_g), **kw)


# ===========================================================================
# DRAG channels
# ===========================================================================
@dataclass(frozen=True)
class DragChannel:
    """One off-resonant process for recursive DRAG to suppress.

    See :mod:`snail_solver.drag`. A tone carrying several of these applies one
    substitution ``F^(n_photon)_{Delta(t)}`` per channel, composed innermost-first.

    Frozen so it can be carried as STATIC data in a ``jax_engine`` pulse spec (it
    controls unrolled Python loops and must never be traced) and hashed as a jit
    static argument.

    Parameters
    ----------
    beat_GHz : float
        ``Delta_0``, the static beat of the suppressed process.
    n_pump : int, default 1
        Pump quanta the process carries, i.e. how its beat MOVES under a chirp:
        ``Delta(t) = 2 pi beat_GHz - n_pump delta(t)``. This is the codebase's
        long-standing ``drag_n_pump``; see ``sweep_common._PUMP_QUANTA``.
    n_photon : int, default 1
        The paper's ``n`` in ``F^(n)`` -- the exponent the drive is raised to.

        Deliberately SEPARATE from `n_pump` even though the two are the same
        physical integer for every channel this device produces. They enter in
        different places (``n_pump`` in the denominator, ``n_photon`` as a numerator
        exponent), and fusing them would silently promote every existing
        ``drag_n_pump=2`` call site -- the subharmonic sweeps, ``--drag-n-pump 2`` --
        from first-order to second-order DRAG. :meth:`from_collision` sets both when
        that IS what you want.
    mode : str, default "perturbative"
        ``"perturbative"`` is Eq. (4). ``"givens"`` (Eq. 7) is not implemented.
    quotient_rule : bool, default False
        Whether ``d/dt`` acts on ``Omega^n / Delta`` as a whole (Eq. 4 verbatim) or
        only on ``Omega^n``. False reproduces this codebase's historical first-order
        arithmetic bit-for-bit; on an UNCHIRPED tone the two are identical anyway
        (``Delta' = 0``).
    kappa : float, optional
        Coupling-per-unit-drive ``g = kappa Omega`` of this process. Required only by
        ``mode="givens"``.
    """

    beat_GHz: float
    n_pump: int = 1
    n_photon: int = 1
    mode: str = "perturbative"
    quotient_rule: bool = False
    kappa: Optional[float] = None

    @classmethod
    def from_collision(cls, beat_GHz: float, kind: str, **kw: Any) -> "DragChannel":
        """Channel for a collision labelled by ``sweep_common._nearest_collision``.

        Sets BOTH `n_pump` and `n_photon` from the collision's pump-quanta count, and
        turns the quotient rule on -- i.e. the paper's scheme applied verbatim. This
        is the auto-fill path; construct :class:`DragChannel` directly for full control.
        """
        from snail_solver.sweep_common import _PUMP_QUANTA
        k = int(_PUMP_QUANTA.get(str(kind), 1))
        kw.setdefault("quotient_rule", True)
        return cls(float(beat_GHz), n_pump=k, n_photon=max(k, 1), **kw)


# ===========================================================================
# Pump tone
# ===========================================================================
@dataclass
class PumpTone:
    """One pump tone applied to the coupler.

    Parameters
    ----------
    w_p_GHz : float
        Pump frequency f_p (GHz); the dressed amplitude oscillates at this rate
        inside X(t). This is the FIXED reference carrier -- any time dependence
        of the frequency belongs in `chirp`, not here.
    envelope : Envelope
        Shape eps_p(t). Its `amp` carries |eps_p| (rad/ns), or directly the
        dimensionless displaced amplitude |eta_p| when `is_eta` is True.
    phi_p : float, default 0.0
        Pump phase (rad); enters as eta_p -> |eta_p| e^{i phi_p}.
    is_eta : bool, default True
        If True the envelope amplitude is already eta_p; if False it is the bare
        pump eps_p and eta_p is computed from eta = 2 w_p/(w_p^2 - w_s^2) eps
        (Eq. 50).
    drag : bool, default False
        If True, add the first-order DRAG quadrature (Motzoi et al., PRL 103,
        110501 (2009)): eta(t) -> eta(t) - i d/dt[eta(t)] / delta. The derivative
        quadrature cancels, to leading order, the adiabatic excitation of a
        process detuned by `delta_drag_GHz`. It suppresses an OFF-resonant
        spectator and is singular as the detuning -> 0 (an on-resonant collision
        needs frequency allocation, not DRAG).
    delta_drag_GHz : float, optional
        STATIC beat Delta_0 (GHz) of the targeted off-resonant process; the
        quadrature is -d eta/dt / (2 pi Delta(t)). Required if `drag`. On a chirped
        tone this is only the t=0 value of the beat -- see `drag_n_pump`.
    drag_n_pump : int, default 1
        Number of PUMP QUANTA the suppressed process carries. On a CHIRPED tone the
        pump sits at w_p + delta(t), so the beat is time-dependent::

            Delta(t) = Delta_0 - drag_n_pump * delta(t)

        matching this codebase's beat convention ``beat = separation - k w_p``
        (``sweep_common._nearest_collision``): k = 1 for a one-pump process, 2 for a
        subharmonic (two-pump) one, and **0 for a static, pump-independent beat**,
        which a chirp must not move. Irrelevant without a chirp (delta = 0), so the
        default leaves every un-chirped tone byte-identical.
    chirp : Chirp, optional
        Time-dependent offset delta(t) of the carrier. Applied as the phase
        e^{-i Phi(t)} on the pump amplitude AFTER the DRAG quadrature -- see
        `ZhouCoupler._eta`. None (default) means an un-chirped tone.

    Notes
    -----
    Sign conventions differ between the two phases here, for historical reasons:
    `phi_p` enters as ``e^{+i phi_p}`` while the chirp enters as ``e^{-i Phi(t)}``,
    which is the sign that matches the ``e^{-i w_p t}`` carrier in X(t). A constant
    chirp `c0` therefore shifts the carrier to ``w_p + c0``, not ``w_p - c0``.

    A chirp interacts with DRAG in exactly ONE place: the denominator. The quadrature
    still differentiates the BASE envelope (the chirp phase is absorbed into the frame
    rotating at the instantaneous pump frequency, so only the amplitude derivative
    enters), but the beat it divides by moves with the pump. Getting this wrong is
    silent: it simply mis-weights the quadrature, most severely near a collision where
    Delta_0 is small and DRAG matters most.
    """

    w_p_GHz: float
    envelope: Envelope
    phi_p: float = 0.0
    is_eta: bool = True
    drag: bool = False
    delta_drag_GHz: Optional[float] = None
    chirp: Optional[Chirp] = None
    drag_n_pump: int = 1
    drag_channels: Optional[Sequence[DragChannel]] = None

    # -- which processes DRAG is suppressing --------------------------------
    def drag_channels_resolved(self) -> tuple:
        """The active :class:`DragChannel` list, innermost-first; ``()`` when DRAG is off.

        Resolution order: an explicit `drag_channels` list wins; otherwise the legacy
        scalar ``drag``/``delta_drag_GHz``/``drag_n_pump`` fields are read as the
        one-channel shorthand; otherwise DRAG is off.

        Returning an EMPTY tuple for an inert tone (rather than a one-element list
        with a zero beat) is deliberate, and mirrors :func:`make_chirp` returning None
        for an absent chirp: it lets the solver skip the correction entirely, which is
        what keeps the DRAG-off path byte-identical to before this feature existed.
        """
        from snail_solver.drag import order_channels
        if self.drag_channels:
            chs = tuple(self.drag_channels)
        elif self.drag and self.delta_drag_GHz not in (None, 0.0):
            chs = (DragChannel(float(self.delta_drag_GHz), n_pump=int(self.drag_n_pump)),)
        else:
            return ()
        return order_channels(chs)

    @property
    def is_legacy_drag(self) -> bool:
        """True when the active DRAG is exactly the historical first-order form.

        One perturbative channel, single-photon, no quotient rule -- the case the
        solver evaluates through its closed-form fast path rather than through jet
        arithmetic, so that every pre-existing result stays bit-identical.
        """
        chs = self.drag_channels_resolved()
        return (len(chs) == 1 and chs[0].n_photon == 1
                and chs[0].mode == "perturbative" and not chs[0].quotient_rule)

    def channel_detuning(self, ch: DragChannel, t: Any, xp: Any = np) -> Any:
        """Instantaneous beat ``Delta_j(t)`` of one channel, in rad/ns."""
        detuning = float(ch.beat_GHz) * TWO_PI
        if self.chirp is None or not ch.n_pump:
            return detuning + 0.0 * xp.asarray(t)
        return detuning - int(ch.n_pump) * self.chirp.detuning(t, xp)

    def channel_detuning_jet(self, ch: DragChannel, t: Any, order: int,
                             xp: Any = np) -> tuple:
        """``Delta_j`` and its derivatives to `order`, in rad/ns.

        ``Delta_j(t) = 2 pi beat - n_pump delta(t)``, so every derivative comes from
        the chirp alone: ``Delta_j^(m) = -n_pump delta^(m)`` for m >= 1.
        """
        value = self.channel_detuning(ch, t, xp)
        order = int(order)
        if self.chirp is None or not ch.n_pump:
            return (value,) + tuple(0.0 * xp.asarray(t) for _ in range(order))
        dj = self.chirp.detuning_jet(t, order, xp)
        return (value,) + tuple(-int(ch.n_pump) * dj[m] for m in range(1, order + 1))

    def channel_detuning_jets(self, channels: Sequence[DragChannel], t: Any,
                              order: int, xp: Any = np) -> list:
        """``Delta_j`` jets for every channel, sharing ONE chirp evaluation.

        Every channel divides the same ``delta(t)`` by a different ``n_pump``, so
        calling :meth:`channel_detuning_jet` per channel rebuilds the whole Legendre
        recurrence once per channel. That is the dominant cost of the scalar solver
        callback (the recurrence ran 10x more often than necessary in a 3-channel
        profile), and it is pure duplication.
        """
        order = int(order)
        chs = tuple(channels)
        zero = 0.0 * xp.asarray(t)
        if self.chirp is None:
            return [(float(c.beat_GHz) * TWO_PI + zero,)
                    + tuple(zero for _ in range(order)) for c in chs]
        dj = self.chirp.detuning_jet(t, order, xp)          # once, not once per channel
        out = []
        for c in chs:
            k = int(c.n_pump)
            if not k:                                        # static, pump-independent
                out.append((float(c.beat_GHz) * TWO_PI + zero,)
                           + tuple(zero for _ in range(order)))
            else:
                out.append((float(c.beat_GHz) * TWO_PI - k * dj[0],)
                           + tuple(-k * dj[m] for m in range(1, order + 1)))
        return out

    # -- the time-dependent DRAG beat ---------------------------------------
    def drag_detuning(self, t: Any, xp: Any = np) -> Any:
        """Instantaneous DRAG beat Delta(t) in rad/ns; xp-generic and trace-clean.

        ``Delta_0 - drag_n_pump * delta(t)``. Returns the constant Delta_0 when the
        tone is un-chirped or ``drag_n_pump == 0``.
        """
        detuning = float(self.delta_drag_GHz or 0.0) * TWO_PI
        if self.chirp is None or not self.drag_n_pump:
            return detuning + 0.0 * xp.asarray(t)
        return detuning - self.drag_n_pump * self.chirp.detuning(t, xp)

    def drag_detuning_floor(self, n: int = 257) -> float:
        """min_t |Delta(t)| over the gate -- the singularity guard.

        A chirp can drive Delta(t) through zero DURING the pulse even when Delta_0 is
        comfortably large, which makes the quadrature diverge mid-gate. That failure
        is invisible in Delta_0 alone, so callers that build a tone should check this
        rather than ``abs(delta_drag_GHz)``.

        Returns ``inf`` when DRAG is off (nothing to guard).

        With several channels this is the min over ALL of them -- one collapsing beat
        is enough to break the pulse. :meth:`drag_detuning_floors` gives the
        per-channel breakdown, which is what an error message needs in order to name
        the offender.
        """
        floors = self.drag_detuning_floors(n)
        return min(floors) if floors else float("inf")

    def drag_detuning_floors(self, n: int = 257) -> list:
        """``min_t |Delta_j(t)|`` per resolved channel, in rad/ns; ``[]`` when off."""
        chs = self.drag_channels_resolved()
        if not chs:
            return []
        t_g = float(getattr(self.envelope, "t_g", 0.0)) or 1.0
        ts = np.linspace(0.0, t_g, int(n))
        return [float(np.min(np.abs(np.asarray(self.channel_detuning(ch, ts, np)))))
                for ch in chs]
