"""
jet.py
======

Truncated Taylor-coefficient ("jet") arithmetic, for exact nested derivatives.

Why this exists
---------------
Recursive DRAG (:mod:`snail_solver.drag`, after Li, Calarco & Motzoi, npj QI 10, 66
(2024)) composes derivative corrections::

    Omega -> ( Omega^n - i d/dt[ Omega^n / Delta(t) ] )^(1/n)

several times over. Each pass differentiates the amplitude produced by the previous
one, so a K-fold composition needs the base envelope -- and the DRAG beat Delta(t),
which a chirp makes time-dependent -- up to the K-th derivative, threaded through
products, quotients and a fractional power. Hand-expanding that for three different
Delta_j(t) under a square root produces an unauditable expression that would then have
to be mirrored by hand into ``jax_engine``.

A jet does it mechanically instead: carry ``(f, f', f'', ...)`` as one object, give it
arithmetic, and the composition becomes the twelve-line loop in ``drag.drag_apply``,
written once and shared by every solve path.

Taylor coefficients, not derivatives
------------------------------------
The stored coefficients are ``a_k = f^(k)(t) / k!``, not ``f^(k)(t)``. That choice is
what makes the arithmetic cheap: multiplication becomes a plain convolution instead of
a binomially-weighted Leibniz sum, division and the fractional power get textbook
three-line recurrences, and differentiation is ``b_k = (k+1) a_{k+1}``. Conversion to
and from ordinary derivatives happens once, at the module boundary
(:meth:`Jet.from_derivs` / :meth:`Jet.derivs`).

Array API and tracing
---------------------
Every operation is built only from ``xp`` primitives and elementwise arithmetic, so a
coefficient may be a scalar, a numpy array of times, or a JAX tracer, following the
same convention as :mod:`snail_solver.envelope`. Two rules keep it trace-clean:

* **The order is a static Python int.** Every loop here runs over ``range(order)``, so
  it unrolls at trace time. The order must never come from a traced value.
* **Divisions use a double ``where``.** See :func:`_safe_div`.
"""
from __future__ import annotations

from math import factorial
from typing import Any, Sequence

import numpy as np


def _safe_div(num: Any, den: Any, xp: Any) -> Any:
    """``num / den``, yielding 0 where ``den`` vanishes -- NaN-free under autodiff.

    The naive ``xp.where(ok, num / den, 0.0)`` is NOT enough. It evaluates ``num/den``
    on the full array first, so the discarded branch still produces ``inf``/``nan``,
    and while ``where`` hides that in the FORWARD pass, reverse-mode differentiation
    multiplies the cotangent by the bad branch's derivative and propagates ``nan``
    back out. Substituting a harmless denominator BEFORE dividing is what makes the
    dead branch numerically inert in both passes.

    This matters here because the jets being divided are pump envelopes, whose leading
    coefficient vanishes identically at the gate endpoints ``t = 0`` and ``t = t_g``.
    """
    ok = den != 0
    den_safe = xp.where(ok, den, 1.0)
    return xp.where(ok, num / den_safe, 0.0 * num)


class Jet:
    """Truncated Taylor coefficients ``a_k = f^(k)(t)/k!`` for ``k = 0 .. order``.

    Each coefficient carries the same shape as the evaluation point(s) ``t``, so one
    ``Jet`` describes a whole time grid at once. Binary operations truncate to the
    lower of the two orders, and :meth:`deriv` lowers the order by one -- which is how
    a chain of K DRAG operators consumes exactly K orders of the base jet.

    Parameters
    ----------
    coeffs : sequence
        The Taylor coefficients, lowest order first. ``coeffs[0]`` is the value.
    """

    __slots__ = ("coeffs",)

    def __init__(self, coeffs: Sequence[Any]) -> None:
        self.coeffs = tuple(coeffs)
        if not self.coeffs:
            raise ValueError("a Jet needs at least the value coefficient")

    # -- construction / extraction -----------------------------------------
    @classmethod
    def from_derivs(cls, derivs: Sequence[Any]) -> "Jet":
        """Build from ordinary derivatives ``(f, f', f'', ...)``, dividing by ``k!``."""
        return cls([d / factorial(k) for k, d in enumerate(derivs)])

    def derivs(self) -> tuple:
        """Ordinary derivatives ``(f, f', f'', ...)``, multiplying by ``k!``."""
        return tuple(a * factorial(k) for k, a in enumerate(self.coeffs))

    @classmethod
    def constant(cls, value: Any, order: int) -> "Jet":
        """A jet whose value is `value` and whose every derivative is zero."""
        return cls([value] + [0.0 * value] * int(order))

    @property
    def order(self) -> int:
        """Highest derivative order carried (``len(coeffs) - 1``)."""
        return len(self.coeffs) - 1

    @property
    def value(self) -> Any:
        """The 0th coefficient, i.e. ``f(t)`` itself."""
        return self.coeffs[0]

    def truncate(self, order: int) -> "Jet":
        """Drop coefficients above `order` (no-op if already at or below it)."""
        return Jet(self.coeffs[:int(order) + 1])

    def __repr__(self) -> str:  # noqa: D105
        return f"Jet(order={self.order})"

    # -- linear structure ---------------------------------------------------
    def _pair(self, other: "Jet") -> int:
        return min(self.order, other.order)

    def add(self, other: "Jet") -> "Jet":
        """Sum, truncated to the lower of the two orders."""
        n = self._pair(other)
        return Jet([self.coeffs[k] + other.coeffs[k] for k in range(n + 1)])

    def sub(self, other: "Jet") -> "Jet":
        """Difference, truncated to the lower of the two orders."""
        n = self._pair(other)
        return Jet([self.coeffs[k] - other.coeffs[k] for k in range(n + 1)])

    def neg(self) -> "Jet":
        """Additive inverse."""
        return Jet([-a for a in self.coeffs])

    def scale(self, c: Any) -> "Jet":
        """Multiply by a constant (may be complex; ``scale(1j)`` is the DRAG factor)."""
        return Jet([c * a for a in self.coeffs])

    # -- ring structure -----------------------------------------------------
    def mul(self, other: "Jet") -> "Jet":
        """Product. In Taylor coefficients this is a plain convolution."""
        n = self._pair(other)
        a, b = self.coeffs, other.coeffs
        return Jet([sum(a[j] * b[k - j] for j in range(k + 1)) for k in range(n + 1)])

    def div(self, other: "Jet", xp: Any = np) -> "Jet":
        """Quotient ``self / other``, by forward substitution.

        From ``self = other * out``, the k-th convolution coefficient gives
        ``out_k = (a_k - sum_{j=1}^{k} b_j out_{k-j}) / b_0``. Every division is by
        ``b_0`` alone, so a single :func:`_safe_div` guard covers all of them.
        """
        n = self._pair(other)
        a, b = self.coeffs, other.coeffs
        out = [_safe_div(a[0], b[0], xp)]
        for k in range(1, n + 1):
            acc = a[k]
            for j in range(1, k + 1):
                acc = acc - b[j] * out[k - j]
            out.append(_safe_div(acc, b[0], xp))
        return Jet(out)

    def powi(self, n: int) -> "Jet":
        """Non-negative integer power, by repeated multiplication.

        Exact and branch-free, unlike :meth:`powf` -- which is why ``F^(n)`` raises to
        the integer power on the way in and only needs the fractional root on the way
        out.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"powi needs a non-negative exponent, got {n}")
        if n == 1:
            return self
        out = Jet.constant(1.0 + 0.0 * self.coeffs[0], self.order)
        for _ in range(n):
            out = out.mul(self)
        return out

    def powf(self, alpha: float, xp: Any = np) -> "Jet":
        """Fractional power ``self ** alpha``, by the standard recurrence.

        Differentiating ``f = g^alpha`` gives ``f' g = alpha f g'``; matching Taylor
        coefficients of ``t^(k-1)`` and isolating ``f_k`` yields

            f_0 = g_0^alpha
            f_k = (1 / (k g_0)) * sum_{j=0}^{k-1} (alpha (k - j) - j) f_j g_{k-j}

        ``alpha == 1`` short-circuits to the identity. That is not just an
        optimization: it is what keeps the ``n_photon = 1`` path -- every existing
        caller -- free of the ``g_0``-vanishing guard below, and hence bit-identical
        to the un-jetted expression.

        ``g_0`` is the pump amplitude, which vanishes at both gate endpoints, so the
        recurrence is guarded by :func:`_safe_div` throughout.
        """
        if float(alpha) == 1.0:
            return self
        g = self.coeffs
        n = self.order
        # complex principal branch: g_0 may be complex once an F^(1) has been applied
        f0 = xp.where(g[0] != 0, xp.asarray(g[0], dtype=complex), 1.0) ** float(alpha)
        f0 = xp.where(g[0] != 0, f0, 0.0 * f0)
        out = [f0]
        for k in range(1, n + 1):
            acc = 0.0
            for j in range(k):
                acc = acc + (float(alpha) * (k - j) - j) * out[j] * g[k - j]
            out.append(_safe_div(acc, k * g[0], xp))
        return Jet(out)

    # -- calculus -----------------------------------------------------------
    def deriv(self) -> "Jet":
        """d/dt, as a jet of one lower order (``b_k = (k+1) a_{k+1}``)."""
        if self.order < 1:
            raise ValueError("cannot differentiate an order-0 jet: build the source "
                             "jet one order higher (a chain of K DRAG operators needs "
                             "the base envelope to order K)")
        return Jet([(k + 1) * self.coeffs[k + 1] for k in range(self.order)])
