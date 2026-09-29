"""
drag.py
=======

Recursive multi-derivative DRAG -- the single source of truth for the math.

After Li, Calarco & Motzoi, npj Quantum Information 10, 66 (2024) (arXiv:2303.01427).
For a transition driven by ``n`` photons and detuned by ``Delta``, their Eq. (4)
substitutes

.. math::
    \\Omega = \\mathcal{F}^{(n)}_\\Delta(\\tilde\\Omega) :=
        \\Big( \\tilde\\Omega^n - i \\frac{d}{dt}
               \\big[ \\tilde\\Omega^n / \\Delta \\big] \\Big)^{1/n}

At ``n = 1`` this is Motzoi's single-derivative DRAG, ``eta - i (d eta/dt)/Delta``.
A single coefficient can only trade one off-resonant transition's error against
another's; composing one substitution per transition suppresses all of them with no
free parameters (their Eq. 8), e.g. ``F^(1)_{D21} o F^(1)_{D10} o F^(2)_{D20} (Omega)``.
Here the channels are the collisions labelled by ``sweep_common._nearest_collision``.

Two rules, both enforced below:

* **Multi-photon substitutions go innermost.** ``F^(n>=2)`` takes an n-th root whose
  branch is unambiguous only while ``Omega`` is still real, so :func:`apply_drag`
  SORTS the channels. The ``F^(1)`` factors commute up to ``O(1/Delta^3)``, which the
  truncation drops anyway.
* **The base shape must vanish to high order at both ends** (their Eq. 13). Hann has
  ``eps''(0) != 0``, so it supports ONE clean correction; with ``F^(2)`` innermost a
  shape ``eps ~ t^p`` comes out as ``Omega ~ t^(p - 1/2)`` -- a ``t^(-1/2)``
  DIVERGENCE at both edges for Hann. See :class:`envelope.SinePowerRamp`.

A chirped tone (:class:`envelope.Chirp`) makes ``Delta`` time-dependent, so Eq. (4)
applied verbatim needs ``Delta'``, ``Delta''``, ... -- supplied through `delta_derivs`
from :meth:`envelope.Chirp.detuning_jet`.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from snail_solver.jet import Jet


def order_channels(channels: Sequence[Any]) -> tuple:
    """Channels sorted innermost-first: descending ``n_photon``, ties in caller order.

    Lets callers build their ``Delta_j`` jets in the order the recursion consumes them
    (:func:`apply_drag` re-sorts idempotently, so a caller that forgets is corrected).
    """
    return tuple(sorted(channels, key=lambda c: -int(c.n_photon)))


def required_order(channels: Sequence[Any]) -> int:
    """Derivative order the base envelope and every ``Delta_j`` must be built to:
    each substitution differentiates once, so K channels consume K orders."""
    return len(tuple(channels))


def apply_drag(shape_derivs: Sequence[Any], delta_derivs: Sequence[Sequence[Any]],
               channels: Sequence[Any], xp: Any = np) -> Any:
    """Compose ``F^(n_j)_{Delta_j}`` over `channels` and return the corrected amplitude.

    Parameters
    ----------
    shape_derivs : sequence
        ``(eps, eps', ..., eps^(K))`` of the BASE envelope (:meth:`Envelope.jet_at`).
        The chirp phase must NOT be folded in: it rotates the pump frame, and
        differentiating it would inject a spurious ``-i delta(t) eta / Delta`` term.
    delta_derivs : sequence of sequence
        One ``(Delta, Delta', ..., Delta^(K))`` per channel, rad/ns, in the SAME order
        as `channels` (both are re-sorted together here).
    channels : sequence of DragChannel
        The suppressed processes; re-sorted innermost-first.
    xp : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    The corrected complex amplitude (the jet's value) with all K corrections applied.
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
        # Eq. (4) differentiates Omega~^n / Delta as a whole; on a chirped tone the
        # quotient rule adds +i Omega~ Delta'/Delta^2. `quotient_rule=False`
        # reproduces the historical first-order arithmetic exactly.
        q = p.div(d, xp).deriv() if ch.quotient_rule else p.deriv().div(d, xp)
        g = p.sub(q.scale(1j)).powf(1.0 / n, xp)
    return g.value
