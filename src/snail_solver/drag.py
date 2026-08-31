"""
drag.py
=======

Recursive multi-derivative DRAG -- the single source of truth for the math.

After Li, Calarco & Motzoi, "Experimental error suppression in Cross-Resonance gates
via multi-derivative pulse shaping", npj Quantum Information 10, 66 (2024)
(arXiv:2303.01427).

The substitution operator
-------------------------
For a transition driven by ``n`` drive photons and detuned by ``Delta``, their Eq. (4)
replaces the drive by

.. math::
    \\Omega = \\mathcal{F}^{(n)}_\\Delta(\\tilde\\Omega) :=
        \\Big( \\tilde\\Omega^n - i \\frac{d}{dt}
               \\big[ \\tilde\\Omega^n / \\Delta \\big] \\Big)^{1/n}

At ``n = 1`` this is the familiar single-derivative DRAG of Motzoi et al.,
``eta - i (d eta/dt)/Delta``, which is what this codebase applied before.

Why recursion, and why it matters here
---------------------------------------
The paper's central experimental result is that a SINGLE derivative correction is
insufficient whenever more than one off-resonant transition matters: tuning its one
coefficient can only trade one transition's error against another's. Composing one
substitution per transition suppresses all of them at once -- analytically, with no
free parameters and no calibration (their Eq. 8)::

    Omega_CR = F^(1)_{D21} o F^(1)_{D10} o F^(2)_{D20} (Omega)

That is exactly this device's situation: the pump sits near several collisions at
once, each already located and labelled by ``sweep_common._nearest_collision``.

Two things the paper is emphatic about, both enforced below
------------------------------------------------------------
**The multi-photon substitution goes innermost.** ``F^(n>=2)`` forms ``Omega^n`` and
takes an n-th root, whose branch is unambiguous only while ``Omega`` is still real --
as the raw base shape is. Applying it to an amplitude that an ``F^(1)`` has already
made complex both flips branches and invalidates the perturbative bookkeeping.
:func:`apply_drag` therefore SORTS the channels rather than trusting the caller.
The ``F^(1)`` factors commute among themselves only to the order kept; the residual
is ``O(1/Delta^3)``, which is precisely what the truncation drops anyway.

**The base shape must vanish to high enough order at both ends.** Their Eq. (13)
shape is m-times differentiable with m vanishing derivatives at each edge, "which
guarantees the validity of the frame transformation". A Hann window has
``eps(0) = eps'(0) = 0`` but ``eps''(0) != 0``, so it supports exactly ONE clean
correction. Worse, with ``F^(2)`` innermost the imaginary term dominates as
``t -> 0`` (their ratio ``2p/(t Delta)`` diverges for any ``eps ~ t^p``), giving
``Omega ~ t^(p - 1/2)``; for Hann ``p = 2``, so two further derivatives produce a
``t^(-1/2)`` DIVERGENCE at both gate edges. See :class:`envelope.SinePowerRamp`.

The chirp
---------
The paper's "chirped" pulse is the time-dependent detuning that cancels the residual
IZ error; they approximate it by a constant phase ramp because their IZ is small.
This codebase already carries the general case as :class:`envelope.Chirp`. What the
chirp adds HERE is that a chirped tone makes ``Delta`` itself time-dependent, so
Eq. (4) applied verbatim needs ``Delta'``, ``Delta''``, ... -- supplied by
:meth:`envelope.Chirp.detuning_jet` and threaded in through `delta_derivs`.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from snail_solver.jet import Jet


def order_channels(channels: Sequence[Any]) -> tuple:
    """Channels sorted innermost-first: descending ``n_photon``, ties in caller order.

    Separated out so callers can build their ``Delta_j`` jets in the SAME order the
    recursion will consume them (:func:`apply_drag` sorts again, idempotently, so a
    caller that forgets is corrected rather than silently mis-paired).
    """
    return tuple(sorted(channels, key=lambda c: -int(c.n_photon)))


def required_order(channels: Sequence[Any]) -> int:
    """Derivative order the base envelope and every ``Delta_j`` must be built to.

    Each substitution differentiates its argument exactly once, so a chain of K
    channels consumes K orders and the composed value is what is left at order 0.
    """
    return len(tuple(channels))


def apply_drag(shape_derivs: Sequence[Any], delta_derivs: Sequence[Sequence[Any]],
               channels: Sequence[Any], xp: Any = np) -> Any:
    """Compose ``F^(n_j)_{Delta_j}`` over `channels` and return the corrected amplitude.

    Parameters
    ----------
    shape_derivs : sequence
        ``(eps, eps', ..., eps^(K))`` of the BASE envelope, from
        :meth:`envelope.Envelope.jet_at`. The chirp phase must NOT be folded in here
        -- it is a rotation of the pump frame, not a feature of the pulse shape, and
        differentiating it would inject a spurious ``-i delta(t) eta / Delta`` term.
    delta_derivs : sequence of sequence
        One ``(Delta, Delta', ..., Delta^(K))`` per channel, in rad/ns, in the SAME
        order as `channels` (both are re-sorted together here).
    channels : sequence of DragChannel
        The suppressed processes. Re-sorted innermost-first; see the module docstring.
    xp : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    The corrected complex amplitude at the evaluation point(s) -- the 0th jet
    coefficient, i.e. the value, with all K derivative corrections applied.
    """
    channels = tuple(channels)
    delta_derivs = tuple(delta_derivs)
    if len(channels) != len(delta_derivs):
        raise ValueError(f"got {len(channels)} channels but {len(delta_derivs)} "
                         f"detuning jets; they must correspond one-to-one")
    pairs = sorted(zip(channels, delta_derivs), key=lambda p: -int(p[0].n_photon))

    g = Jet.from_derivs(shape_derivs)
    for ch, dd in pairs:
        if ch.mode != "perturbative":
            raise NotImplementedError(
                f"DRAG mode {ch.mode!r} is not implemented; only 'perturbative' "
                f"(Eq. 4) is available. The Givens variant (Eq. 7) additionally "
                f"needs kappa, the coupling-per-unit-drive of this specific process.")
        n = int(ch.n_photon)
        d = Jet.from_derivs(dd)
        p = g.powi(n)                                  # Omega~^n
        # Eq. (4) differentiates Omega~^n / Delta AS A WHOLE. With a chirped tone
        # Delta is time-dependent, so the quotient rule contributes an extra
        # +i Omega~ Delta'/Delta^2 that the historical first-order code omits.
        # `quotient_rule=False` reproduces that historical arithmetic exactly.
        q = p.div(d, xp).deriv() if ch.quotient_rule else p.deriv().div(d, xp)
        g = p.sub(q.scale(1j)).powf(1.0 / n, xp)
    return g.value
