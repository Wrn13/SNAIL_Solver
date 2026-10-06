"""Second-order effective Hamiltonian of a g_n X(t)^n coupler in the linear frame.

Model
-----
In the frame U = exp(i sum_k (omega_k t + c_k phi(t)) n_k) the Hamiltonian is

    H_I(t) = H_anh - phidot sum_k c_k n_k + sum_n g_n X_I(t)^n,
    X_I(t) = sum_k lam_k (a_k e^{-i(omega_k t + c_k phi)} + h.c.)
             + eta e^{-i(omega_p t + phi)} + eta* e^{+i(omega_p t + phi)},
    H_anh  = sum_k (alpha_k / 2) a_k^dag a_k^dag a_k a_k.

The pump convention is eta(t) = |eta| e^{-i phi(t)}, so the instantaneous pump
frequency is omega_p + phidot. Frame "A" is c_k = 0 (chirp lives only in the
pump); any other c is a number-diagonal frame and leaves H_anh invariant.

Every normal-ordered monomial of sum_n g_n X^n has a signature
sigma = (s, Delta) with s the net pump number and Delta_k = q_k - p_k for
a_k^{dag p_k} a_k^{q_k}. Its instantaneous carrier (convention e^{-i Omega t}) is

    Omega_sigma(t) = sum_k Delta_k (omega_k + c_k phidot) + s (omega_p + phidot).

Signatures with |Omega| < slow_cutoff form H_slow; the rest form F(t).

Results implemented (derivation in the accompanying notes / chat)
-----------------------------------------------------------------
Time-dependent Schrieffer-Wolff with generator e^{K}, solving
i dK/dt + [K, H_s] = -F to first order, gives the exact identity
H'^{(2)} = (1/2)[K, F] and

    k_sigma = -h/Omega + [h, H_s]/Omega^2 - i s phiddot h/Omega^3 + i hdot/Omega^2,

    H^(2)_slow = sum_{i<j, |Omega_i+Omega_j|<cut} (1/2)(1/Omega_j - 1/Omega_i)[h_i, h_j]
                 e^{-i(Omega_i + Omega_j)t},
    H^(2,alpha) = (1/2) sum_sigma [[h_sigma, H_s], h_{-sigma}] / Omega_sigma^2.

For sigma_j = -sigma_i the first line is James & Jerke's sum [h^dag, h]/Omega
(Can. J. Phys. 85, 625 (2007)); for sigma_i + sigma_j != 0 it is the
Gamel & James generalization (PRA 82, 052106 (2010)). The second line is the
leading alpha-dependence of the denominators, 1/(Omega + alpha m) expanded to
O(alpha/Omega^2).

Chirp: the denominators use the physical (frame-A) carrier Omega + s phidot.
The phiddot term of k cancels identically in the static part of (1/2)[K, F]
(see ``static_from_kick`` and the test suite), so to O(phiddot) the chirp
enters the effective Hamiltonian only through instantaneous denominators;
phiddot appears in the kick (micromotion/leakage) operator. Frame choice c
only re-labels residual carriers of slow terms and adds -phidot sum c_k n_k.

Assumptions that are not derived here
-------------------------------------
* The Hamiltonian form above is a reconstruction of a Zhou-style dressed-mode
  model. The normalization of eta and the coupler participation lam_c must be
  checked against the user's ``expand_terms``.
* Only number-diagonal H_s (H_anh plus static slow bins) enters the
  O(1/Omega^2) correction; [h, h_slow] cross terms with time-dependent slow
  bins are not included.
* Third and higher orders in g_n are not included.

Usage
-----
    from magnus2 import Device, LinearFrameMagnus, example_device
    L = LinearFrameMagnus(example_device(), slow_cutoff=0.1, phidot=0.004)
    H = L.effective_static()                       # NOPoly, number-diagonal
    L.level_energy({"a": 1}, H=H) - L.level_energy({}, H=H)
    L.rung_resolved_shifts([{}, {"a": 1}], {"a": 7, "b": 7, "c": 7})
    static, pairs = L.second_order()               # pairs: slow second-order processes
    K = L.kick_operator()                          # micromotion incl. phiddot term
    E = HighOrderExpansion(L, max_order=5, q_final=2)   # orders 1..5; resonant terms kept slow
    E.effective(3)                                 # slow H through order 3
    E.R[4], E.resonant                             # all order-4 terms; signatures kept as resonant
    Ls = LinearFrameMagnus(example_device(), symbolic=True)   # closed forms in SymPy
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import comb, factorial
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from openfermion import BosonOperator

Coeff = Union[complex, float, int, sp.Expr]
ModeOps = Tuple[Tuple[int, int], ...]
Key = Tuple[int, ModeOps]
Signature = Tuple[int, Tuple[int, ...]]


@lru_cache(maxsize=None)
def _mul_ops(o1: ModeOps, o2: ModeOps) -> Tuple[Tuple[int, ModeOps], ...]:
    """Normal-order the product of two normal-ordered multimode monomials.

    Uses the single-mode Wick identity
    a^{q1} a^{dag p2} = sum_k C(q1,k) C(p2,k) k! a^{dag (p2-k)} a^{(q1-k)},
    applied independently per mode (distinct modes commute).

    Parameters
    ----------
    o1, o2 : ModeOps
        Per-mode exponents ((p, q), ...) of a^{dag p} a^{q}.

    Returns
    -------
    tuple of (int, ModeOps)
        Integer weights and resulting normal-ordered monomials.
    """
    per_mode = []
    for (p1, q1), (p2, q2) in zip(o1, o2):
        per_mode.append([(comb(q1, k) * comb(p2, k) * factorial(k), (p1 + p2 - k, q1 + q2 - k))
                         for k in range(min(q1, p2) + 1)])
    out = []
    for combo in product(*per_mode):
        w = 1
        ops = []
        for wk, pq in combo:
            w *= wk
            ops.append(pq)
        out.append((w, tuple(ops)))
    return tuple(out)


@lru_cache(maxsize=None)
def _qp(o: ModeOps) -> Tuple[int, int]:
    """(total q, total p) of a monomial."""
    return sum(q for _, q in o), sum(p for p, _ in o)


@lru_cache(maxsize=None)
def _mul_ops_q(o1: ModeOps, o2: ModeOps) -> Tuple[Tuple[int, ModeOps, int], ...]:
    """_mul_ops with each result's total q attached."""
    return tuple((w, o, _qp(o)[0]) for w, o in _mul_ops(o1, o2))


@dataclass
class Arith:
    """Coefficient arithmetic policy (numeric complex or SymPy).

    Parameters
    ----------
    symbolic : bool
        If True, coefficients are SymPy expressions.
    tol : float
        Absolute tolerance for dropping numeric coefficients.
    conj_map : dict, optional
        Symbol swaps applied under complex conjugation (eta <-> eta_c).
    """

    symbolic: bool = False
    tol: float = 1e-14
    conj_map: Dict[sp.Symbol, sp.Symbol] = field(default_factory=dict)

    def conj(self, c: Coeff) -> Coeff:
        if self.symbolic:
            c = sp.sympify(c)
            return sp.conjugate(c).subs(self.conj_map, simultaneous=True)
        return complex(c).conjugate()

    def is_zero(self, c: Coeff) -> bool:
        if self.symbolic:
            return sp.expand(c) == 0
        return abs(c) <= self.tol

    def finalize(self, c: Coeff) -> Coeff:
        return sp.expand(c) if self.symbolic else complex(c)


class NOPoly:
    """Normal-ordered polynomial in bosonic modes with net-pump bookkeeping.

    Terms are stored as {(s, ((p_1, q_1), ..., (p_N, q_N))): coeff}, meaning
    coeff * prod_k a_k^{dag p_k} a_k^{q_k}, tagged by net pump number s.

    Parameters
    ----------
    n_modes : int
        Number of bosonic modes.
    terms : dict, optional
        Initial terms.
    """

    __slots__ = ("n_modes", "terms")

    def __init__(self, n_modes: int, terms: Optional[Dict[Key, Coeff]] = None) -> None:
        self.n_modes = n_modes
        self.terms: Dict[Key, Coeff] = {} if terms is None else dict(terms)

    @classmethod
    def scalar(cls, n_modes: int, coeff: Coeff, s: int = 0) -> "NOPoly":
        return cls(n_modes, {(s, ((0, 0),) * n_modes): coeff})

    @classmethod
    def mode_op(cls, n_modes: int, k: int, p: int, q: int, coeff: Coeff = 1) -> "NOPoly":
        ops = [(0, 0)] * n_modes
        ops[k] = (p, q)
        return cls(n_modes, {(0, tuple(ops)): coeff})

    def copy(self) -> "NOPoly":
        return NOPoly(self.n_modes, self.terms)

    def __add__(self, other: "NOPoly") -> "NOPoly":
        out = dict(self.terms)
        for k, c in other.terms.items():
            out[k] = out.get(k, 0) + c
        return NOPoly(self.n_modes, out)

    def __neg__(self) -> "NOPoly":
        return NOPoly(self.n_modes, {k: -c for k, c in self.terms.items()})

    def __sub__(self, other: "NOPoly") -> "NOPoly":
        return self + (-other)

    def scale(self, c: Coeff) -> "NOPoly":
        return NOPoly(self.n_modes, {k: c * v for k, v in self.terms.items()})

    def __mul__(self, other: Union["NOPoly", Coeff]) -> "NOPoly":
        if not isinstance(other, NOPoly):
            return self.scale(other)
        return self.product(other)

    def product(self, other: "NOPoly", qmax: Optional[int] = None) -> "NOPoly":
        """Normal-ordered product, optionally keeping only terms with sum_k q_k <= qmax.

        A product monomial has total q >= max(q_B, q_A - p_B), so pairs that cannot
        reach qmax are skipped before the Wick expansion.
        """
        out: Dict[Key, Coeff] = {}
        get = out.get
        if qmax is None:
            for (s1, o1), c1 in self.terms.items():
                for (s2, o2), c2 in other.terms.items():
                    c12 = c1 * c2
                    for w, o in _mul_ops(o1, o2):
                        key = (s1 + s2, o)
                        out[key] = get(key, 0) + w * c12
            return NOPoly(self.n_modes, out)
        right = [(s2, o2, c2, _qp(o2)) for (s2, o2), c2 in other.terms.items() if _qp(o2)[0] <= qmax]
        for (s1, o1), c1 in self.terms.items():
            q1 = _qp(o1)[0]
            for s2, o2, c2, (_, p2) in right:
                if q1 - p2 > qmax:
                    continue
                c12 = c1 * c2
                for w, o, qo in _mul_ops_q(o1, o2):
                    if qo <= qmax:
                        key = (s1 + s2, o)
                        out[key] = get(key, 0) + w * c12
        return NOPoly(self.n_modes, out)

    def commutator(self, other: "NOPoly", qmax: Optional[int] = None) -> "NOPoly":
        if qmax is None:
            return self * other - other * self
        return self.product(other, qmax) - other.product(self, qmax)

    def truncated(self, qmax: Optional[int]) -> "NOPoly":
        """Drop terms with more than qmax annihilation operators in total."""
        if qmax is None:
            return self
        return NOPoly(self.n_modes, {k: c for k, c in self.terms.items() if sum(q for _, q in k[1]) <= qmax})

    def dagger(self, arith: Arith) -> "NOPoly":
        return NOPoly(self.n_modes, {(-s, tuple((q, p) for p, q in o)): arith.conj(c)
                                     for (s, o), c in self.terms.items()})

    def pruned(self, arith: Arith) -> "NOPoly":
        out = {}
        for k, c in self.terms.items():
            c = arith.finalize(c)
            if not arith.is_zero(c):
                out[k] = c
        return NOPoly(self.n_modes, out)

    def without_scalars(self) -> "NOPoly":
        zero = ((0, 0),) * self.n_modes
        return NOPoly(self.n_modes, {k: c for k, c in self.terms.items() if k[1] != zero})

    def by_signature(self) -> Dict[Signature, "NOPoly"]:
        bins: Dict[Signature, NOPoly] = {}
        for (s, o), c in self.terms.items():
            sig = (s, tuple(q - p for p, q in o))
            bins.setdefault(sig, NOPoly(self.n_modes)).terms[(s, o)] = c
        return bins

    def is_empty(self) -> bool:
        return not self.terms

    def diagonal_element(self, n: Sequence[int]) -> Coeff:
        """<n| H |n> from the s = 0, p_k = q_k terms (falling factorials)."""
        total: Coeff = 0
        for (s, o), c in self.terms.items():
            if s != 0 or any(p != q for p, q in o):
                continue
            w = 1
            for (p, _), nk in zip(o, n):
                if p > nk:
                    w = 0
                    break
                w *= factorial(nk) // factorial(nk - p)
            if w:
                total = total + w * c
        return total

    def to_matrix(self, cutoffs: Sequence[int], subs: Optional[Mapping] = None) -> np.ndarray:
        """Dense matrix on a truncated Fock space (mode order = kron order).

        Pump c-number factors are already in the coefficients; s is ignored, so
        the matrix is the operator multiplying e^{-i Omega t} for one signature.
        """
        lows = []
        for N in cutoffs:
            a = np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1).astype(complex)
            lows.append(a)
        dim = int(np.prod(cutoffs))
        M = np.zeros((dim, dim), dtype=complex)
        for (_, o), c in self.terms.items():
            val = complex(sp.N(sp.sympify(c).subs(subs))) if subs is not None else complex(c)
            op = np.array([[1.0 + 0j]])
            for (p, q), a in zip(o, lows):
                m = np.linalg.matrix_power(a.conj().T, p) @ np.linalg.matrix_power(a, q)
                op = np.kron(op, m)
            M += val * op
        return M

    def apply_to_fock(self, n: Sequence[int]) -> Dict[Tuple[int, ...], Coeff]:
        """H|n> as {m: <m|H|n>}, exact (no Fock cutoff); s is ignored as in to_matrix."""
        out: Dict[Tuple[int, ...], Coeff] = {}
        for (_, o), c in self.terms.items():
            amp, m = 1.0, []
            for (p, q), nk in zip(o, n):
                if q > nk:
                    amp = 0.0
                    break
                low = nk - q
                amp *= np.sqrt(factorial(nk) / factorial(low) * factorial(low + p) / factorial(low))
                m.append(low + p)
            if amp:
                key = tuple(m)
                out[key] = out.get(key, 0) + amp * c
        return out

    def to_boson_operator(self) -> Dict[int, "BosonOperator"]:
        """OpenFermion form {s: BosonOperator}; mode k is OpenFermion index k.

        Interop only: OpenFermion's own +/normal_ordered drop |coeff| < 1e-8 and call
        sympy.simplify on symbolic coefficients, so do the algebra here.
        """
        from openfermion import BosonOperator
        out: Dict[int, BosonOperator] = {}
        for (s, o), c in self.terms.items():
            term = (tuple((k, 1) for k, (p, _) in enumerate(o) for _ in range(p))
                    + tuple((k, 0) for k, (_, q) in enumerate(o) for _ in range(q)))
            out.setdefault(s, BosonOperator()).terms[term] = c
        return out

    @classmethod
    def from_boson_operator(cls, op: "BosonOperator", n_modes: int, s: int = 0) -> "NOPoly":
        """Import an OpenFermion BosonOperator (any ordering) as pump number s."""
        out = NOPoly(n_modes)
        for term, c in op.terms.items():
            P = NOPoly.scalar(n_modes, c, s)
            for k, act in term:
                P = P * NOPoly.mode_op(n_modes, k, act, 1 - act)
            out = out + P
        return out

    def to_string(self, names: Sequence[str], fmt: Callable[[Coeff], str] = str) -> str:
        parts = []
        for (s, o), c in sorted(self.terms.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            ops = []
            for name, (p, q) in zip(names, o):
                if p:
                    ops.append(f"{name}^dag" + (f"^{p}" if p > 1 else ""))
                if q:
                    ops.append(name + (f"^{q}" if q > 1 else ""))
            parts.append(f"({fmt(c)}) " + (" ".join(ops) if ops else "1") + (f"  [s={s}]" if s else ""))
        return "\n".join(parts) if parts else "0"


def opposite(a: Signature, b: Signature) -> bool:
    return a[0] + b[0] == 0 and all(x + y == 0 for x, y in zip(a[1], b[1]))


def format_signature(sig: Signature, names: Sequence[str]) -> str:
    s, d = sig
    body = ", ".join(f"{n}:{x:+d}" for n, x in zip(names, d) if x)
    return f"(s={s:+d}; {body or 'diag'})"


@dataclass
class Device:
    """Device parameters (frequencies in GHz, cycles; any consistent unit works).

    Parameters
    ----------
    modes : list of str
        Mode labels, kron order.
    omega, lam, alpha : dict
        Dressed frequency, participation in X, Kerr (alpha/2 a^dag2 a^2) per mode.
    g : dict
        Nonlinear coefficients {n: g_n} of g_n X^n.
    omega_p : float
        Pump frequency.
    eta : complex
        Pump amplitude in the units of X (normalization must match the user's code).
    """

    modes: List[str]
    omega: Dict[str, float]
    lam: Dict[str, float]
    alpha: Dict[str, float]
    g: Dict[int, float]
    omega_p: float
    eta: complex

    @classmethod
    def from_dict(cls, d: Mapping) -> "Device":
        """Build from {"modes": [{"name","omega","lam","alpha"}], "g": {"3": ...}, "omega_p", "eta"}.

        This schema is this module's own, not the user's device JSON; map keys before calling.
        """
        names = [m["name"] for m in d["modes"]]
        eta = d["eta"]
        if isinstance(eta, (list, tuple)):
            eta = complex(eta[0], eta[1])
        return cls(modes=names,
                   omega={m["name"]: float(m["omega"]) for m in d["modes"]},
                   lam={m["name"]: float(m["lam"]) for m in d["modes"]},
                   alpha={m["name"]: float(m.get("alpha", 0.0)) for m in d["modes"]},
                   g={int(k): float(v) for k, v in d["g"].items()},
                   omega_p=float(d["omega_p"]), eta=complex(eta))

    @classmethod
    def from_json(cls, path: str) -> "Device":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def with_eta(self, eta: complex) -> "Device":
        return Device(list(self.modes), dict(self.omega), dict(self.lam), dict(self.alpha),
                      dict(self.g), self.omega_p, complex(eta))


@dataclass(frozen=True)
class Symbols:
    """SymPy symbols for symbolic mode. eta_c is a formal stand-in for conj(eta)."""

    omega: Dict[str, sp.Symbol]
    lam: Dict[str, sp.Symbol]
    alpha: Dict[str, sp.Symbol]
    g: Dict[int, sp.Symbol]
    omega_p: sp.Symbol
    eta: sp.Symbol
    eta_c: sp.Symbol
    eta_dot: sp.Symbol
    eta_c_dot: sp.Symbol
    phidot: sp.Symbol
    phiddot: sp.Symbol

    @classmethod
    def for_device(cls, dev: Device) -> "Symbols":
        R = dict(real=True)
        return cls(omega={m: sp.Symbol(f"omega_{m}", **R) for m in dev.modes},
                   lam={m: sp.Symbol(f"lambda_{m}", **R) for m in dev.modes},
                   alpha={m: sp.Symbol(f"alpha_{m}", **R) for m in dev.modes},
                   g={n: sp.Symbol(f"g_{n}", **R) for n in dev.g},
                   omega_p=sp.Symbol("omega_p", **R),
                   eta=sp.Symbol("eta", **R), eta_c=sp.Symbol("eta_c", **R),
                   eta_dot=sp.Symbol("etadot", **R), eta_c_dot=sp.Symbol("etadot_c", **R),
                   phidot=sp.Symbol("phidot", **R), phiddot=sp.Symbol("phiddot", **R))

    def subs_map(self, dev: Device, phidot: float = 0.0, phiddot: float = 0.0,
                 eta_dot: complex = 0.0) -> Dict[sp.Symbol, complex]:
        m: Dict[sp.Symbol, complex] = {}
        for k in dev.modes:
            m[self.omega[k]] = dev.omega[k]
            m[self.lam[k]] = dev.lam[k]
            m[self.alpha[k]] = dev.alpha[k]
        for n, v in dev.g.items():
            m[self.g[n]] = v
        m[self.omega_p] = dev.omega_p
        m[self.eta] = dev.eta
        m[self.eta_c] = np.conj(dev.eta)
        m[self.eta_dot] = eta_dot
        m[self.eta_c_dot] = np.conj(eta_dot)
        m[self.phidot] = phidot
        m[self.phiddot] = phiddot
        return m


@dataclass
class FastTerm:
    signature: Signature
    op: NOPoly
    carrier: float
    carrier_sym: Optional[sp.Expr]


@dataclass
class SlowPairTerm:
    """Second-order slow term from a fast pair (i, j); residual carrier in frame A."""

    sig_i: Signature
    sig_j: Signature
    total: Signature
    residual: float
    residual_sym: Optional[sp.Expr]
    op: NOPoly


class LinearFrameMagnus:
    """Second-order effective Hamiltonian of sum_n g_n X^n in the linear frame.

    Parameters
    ----------
    device : Device
        Device parameters.
    symbolic : bool
        Keep coefficients and carriers as SymPy expressions (classification still
        uses the numeric device values).
    slow_cutoff : float
        |Omega| below which a first-order signature is treated as slow.
    near_cutoff : float, optional
        |Omega_i + Omega_j| below which a fast pair contributes a slow
        second-order term. Defaults to slow_cutoff.
    phidot, phiddot : float
        Instantaneous chirp rate and its derivative (frame A, eta = |eta| e^{-i phi}).
    eta_dot : complex
        Envelope derivative used for the hdot term of the kick (numeric mode:
        finite difference; symbolic mode: symbol etadot is used instead).
    tol : float
        Numeric coefficient tolerance.
    """

    def __init__(self, device: Device, *, symbolic: bool = False, slow_cutoff: float = 0.1,
                 near_cutoff: Optional[float] = None, phidot: float = 0.0, phiddot: float = 0.0,
                 eta_dot: complex = 0.0, tol: float = 1e-14) -> None:
        self.dev = device
        self.names = list(device.modes)
        self.N = len(self.names)
        self.symbolic = symbolic
        self.slow_cutoff = slow_cutoff
        self.near_cutoff = slow_cutoff if near_cutoff is None else near_cutoff
        self.phidot = phidot
        self.phiddot = phiddot
        self.eta_dot = eta_dot
        self.syms = Symbols.for_device(device) if symbolic else None
        
        conj_map = {self.syms.eta: self.syms.eta_c, self.syms.eta_c: self.syms.eta,
                    self.syms.eta_dot: self.syms.eta_c_dot,
                    self.syms.eta_c_dot: self.syms.eta_dot} if symbolic else {}
        self.arith = Arith(symbolic=symbolic, tol=tol, conj_map=conj_map)
        self.H_nl = self._build_nonlinear(device)
        self.H_anh = self._build_anharmonic()
        self._classify()

    def _p(self, kind: str, key) -> Coeff:
        if self.symbolic:
            return getattr(self.syms, kind)[key]
        return getattr(self.dev, kind)[key]

    def _build_nonlinear(self, dev: Device) -> NOPoly:
        N = self.N
        if self.symbolic:
            eta, etac = self.syms.eta, self.syms.eta_c
        else:
            eta, etac = dev.eta, np.conj(dev.eta)
        X = NOPoly.scalar(N, eta, s=1) + NOPoly.scalar(N, etac, s=-1)
        for k, m in enumerate(self.names):
            lam = self.syms.lam[m] if self.symbolic else dev.lam[m]
            X = X + NOPoly.mode_op(N, k, 0, 1, lam) + NOPoly.mode_op(N, k, 1, 0, lam)
        H = NOPoly(N)
        power = NOPoly.scalar(N, 1)
        for n in range(1, max(dev.g) + 1):
            power = (power * X).pruned(self.arith)
            if n in dev.g:
                gn = self.syms.g[n] if self.symbolic else dev.g[n]
                H = H + power.scale(gn)
        return H.pruned(self.arith).without_scalars()

    def _build_anharmonic(self) -> NOPoly:
        H = NOPoly(self.N)
        for k, m in enumerate(self.names):
            a = self._p("alpha", m)
            if self.symbolic or a != 0:
                H = H + NOPoly.mode_op(self.N, k, 2, 2, a / 2 if not self.symbolic else a / sp.Integer(2))
        return H.pruned(self.arith)

    def bare_carrier(self, sig: Signature) -> float:
        s, d = sig
        return sum(dk * self.dev.omega[m] for dk, m in zip(d, self.names)) + s * self.dev.omega_p

    def carrier(self, sig: Signature) -> float:
        """Physical instantaneous carrier (frame A): bare + s phidot."""
        return self.bare_carrier(sig) + sig[0] * self.phidot

    def carrier_sym(self, sig: Signature) -> Optional[sp.Expr]:
        if not self.symbolic:
            return None
        s, d = sig
        return (sum(dk * self.syms.omega[m] for dk, m in zip(d, self.names))
                + s * (self.syms.omega_p + self.syms.phidot))

    def _den(self, t: FastTerm) -> Coeff:
        return t.carrier_sym if self.symbolic else t.carrier

    def _classify(self) -> None:
        self.slow: Dict[Signature, NOPoly] = {}
        self.fast: Dict[Signature, FastTerm] = {}
        for sig, op in self.H_nl.by_signature().items():
            Om = self.carrier(sig)
            if abs(Om) < self.slow_cutoff:
                self.slow[sig] = op
            else:
                self.fast[sig] = FastTerm(sig, op, Om, self.carrier_sym(sig))

    def static_slow_part(self) -> NOPoly:
        """Number-diagonal H_s used in the O(1/Omega^2) terms: H_anh + carrier-free slow bins."""
        H = self.H_anh.copy()
        for sig, op in self.slow.items():
            if sig[0] == 0 and not any(sig[1]):
                H = H + op
        return H

    def second_order(self) -> Tuple[NOPoly, List[SlowPairTerm]]:
        """Static diagonal second order and other slow second-order pair terms.

        Returns
        -------
        static : NOPoly
            sum_{Omega>0} [h^dag, h]/Omega (sigma_j = -sigma_i pairs). Number-diagonal.
        slow_pairs : list of SlowPairTerm
            Pairs with sigma_i + sigma_j != 0 and |Omega_i + Omega_j| < near_cutoff.
            Coefficient (1/2)(1/Omega_j - 1/Omega_i)[h_i, h_j]; residual carriers
            include (s_i + s_j) phidot (frame A).
        """
        sigs = list(self.fast)
        static = NOPoly(self.N)
        pairs: List[SlowPairTerm] = []
        for i in range(len(sigs)):
            Ti = self.fast[sigs[i]]
            for j in range(i + 1, len(sigs)):
                Tj = self.fast[sigs[j]]
                delta = Ti.carrier + Tj.carrier
                if abs(delta) > self.near_cutoff:
                    continue
                if self.symbolic:
                    coef = sp.Rational(1, 2) * (1 / self._den(Tj) - 1 / self._den(Ti))
                else:
                    coef = 0.5 * (1.0 / Tj.carrier - 1.0 / Ti.carrier)
                term = Ti.op.commutator(Tj.op).scale(coef).pruned(self.arith)
                if term.is_empty():
                    continue
                if opposite(Ti.signature, Tj.signature):
                    static = static + term
                else:
                    tot = (Ti.signature[0] + Tj.signature[0],
                           tuple(x + y for x, y in zip(Ti.signature[1], Tj.signature[1])))
                    rs = None
                    if self.symbolic:
                        rs = sp.simplify(Ti.carrier_sym + Tj.carrier_sym)
                    pairs.append(SlowPairTerm(Ti.signature, Tj.signature, tot, delta, rs, term))
        return static.pruned(self.arith), pairs

    def anharmonic_correction(self) -> NOPoly:
        """(1/2) sum_sigma [[h_sigma, H_s], h_{-sigma}] / Omega_sigma^2 (leading alpha/Omega)."""
        Hs = self.static_slow_part()
        out = NOPoly(self.N)
        for sig, T in self.fast.items():
            msig = (-sig[0], tuple(-x for x in sig[1]))
            if msig not in self.fast:
                continue
            inner = T.op.commutator(Hs)
            dd = self._den(T)
            out = out + inner.commutator(self.fast[msig].op).scale(
                (sp.Rational(1, 2) if self.symbolic else 0.5) / dd ** 2)
        return out.pruned(self.arith)

    def _hdot(self, sig: Signature) -> Optional[NOPoly]:
        T = self.fast[sig]
        if self.symbolic:
            S = self.syms
            terms = {k: sp.diff(c, S.eta) * S.eta_dot + sp.diff(c, S.eta_c) * S.eta_c_dot
                     for k, c in T.op.terms.items()}
            return NOPoly(self.N, terms).pruned(self.arith)
        if self.eta_dot == 0:
            return None
        eps = 1e-6
        hp = self._build_nonlinear(self.dev.with_eta(self.dev.eta + eps * self.eta_dot)).by_signature()
        hm = self._build_nonlinear(self.dev.with_eta(self.dev.eta - eps * self.eta_dot)).by_signature()
        z = NOPoly(self.N)
        return (hp.get(sig, z) - hm.get(sig, z)).scale(1.0 / (2 * eps)).pruned(self.arith)

    def kick_operator(self, include_anharmonic: bool = True, include_chirp: bool = True,
                      include_envelope: bool = True) -> Dict[Signature, NOPoly]:
        """First-order generator components k_sigma, with K(t) = sum k_sigma e^{-i Theta_sigma(t)}.

        k = -h/Omega + [h, H_s]/Omega^2 - i s phiddot h/Omega^3 + i hdot/Omega^2.
        The state in the effective frame is e^{K}|psi>; |k| sets micromotion/leakage.
        """
        Hs = self.static_slow_part()
        I = sp.I if self.symbolic else 1j
        phidd = self.syms.phiddot if self.symbolic else self.phiddot
        out: Dict[Signature, NOPoly] = {}
        for sig, T in self.fast.items():
            Om = self._den(T)
            k = T.op.scale(-1 / Om)
            if include_anharmonic:
                k = k + T.op.commutator(Hs).scale(1 / Om ** 2)
            if include_chirp and sig[0] != 0:
                k = k + T.op.scale(-I * sig[0] * phidd / Om ** 3)
            if include_envelope:
                hd = self._hdot(sig)
                if hd is not None:
                    k = k + hd.scale(I / Om ** 2)
            out[sig] = k.pruned(self.arith)
        return out

    def static_from_kick(self, kicks: Mapping[Signature, NOPoly]) -> NOPoly:
        """Static part of (1/2)[K, F]: (1/2) sum_sigma [k_sigma, h_{-sigma}].

        Used to verify that the phiddot term of k drops out of H^(2).
        """
        out = NOPoly(self.N)
        half = sp.Rational(1, 2) if self.symbolic else 0.5
        for sig, k in kicks.items():
            msig = (-sig[0], tuple(-x for x in sig[1]))
            if msig in self.fast:
                out = out + k.commutator(self.fast[msig].op).scale(half)
        return out.pruned(self.arith)

    def effective_static(self, include_anharmonic_correction: bool = True) -> NOPoly:
        """H_anh + static slow bins + diagonal second order (+ alpha correction)."""
        static, _ = self.second_order()
        H = self.static_slow_part() + static
        if include_anharmonic_correction:
            H = H + self.anharmonic_correction()
        return H.pruned(self.arith)

    def level_energy(self, state: Mapping[str, int], include_anharmonic_correction: bool = True,
                     H: Optional[NOPoly] = None) -> Coeff:
        """<n| H_eff,static |n> for a Fock state given as {mode: n}."""
        if H is None:
            H = self.effective_static(include_anharmonic_correction)
        n = [state.get(m, 0) for m in self.names]
        val = H.diagonal_element(n)
        return sp.expand(val) if self.symbolic else complex(val).real

    def rung_resolved_shifts(self, states: Sequence[Mapping[str, int]], cutoffs: Mapping[str, int],
                             warn_below: Optional[float] = None) -> List[float]:
        """Second-order diagonal energies with exact rung-resolved denominators (numeric).

        E2(n) = sum_sigma sum_m |<m|h_sigma|n>|^2 / (E_n - E_m + Omega_sigma),
        with E the diagonal of H_s (H_anh plus static slow bins). This resums the
        alpha m / Omega series that ``anharmonic_correction`` truncates at first
        order, and is what the operator result reduces to when alpha = 0.

        Parameters
        ----------
        states : sequence of dict
            Fock states {mode: n}.
        cutoffs : dict
            Per-mode truncation; must exceed max(n) + max monomial degree.
        warn_below : float, optional
            Report |denominator| below this (near-resonant channel) via RuntimeWarning.

        Returns
        -------
        list of float
            E_s(n) + E2(n) for each state (linear frame, absolute; subtract vacuum).
        """
        import warnings
        cuts = [cutoffs[m] for m in self.names]
        Hs = self.numeric(self.static_slow_part())
        dims = np.array(cuts)
        idx = list(np.ndindex(*cuts))
        Ediag = np.array([complex(Hs.diagonal_element(n)).real for n in idx])
        mats = {sig: self.numeric(T.op).to_matrix(cuts) for sig, T in self.fast.items()}
        out = []
        for st in states:
            n = tuple(st.get(m, 0) for m in self.names)
            if any(nk + 3 >= ck for nk, ck in zip(n, cuts)):
                warnings.warn(f"cutoff {cuts} may truncate couplings out of {n}", RuntimeWarning)
            col = int(np.ravel_multi_index(n, dims))
            E2 = 0.0
            for sig, M in mats.items():
                amp = M[:, col]
                nz = np.nonzero(np.abs(amp) > 1e-15)[0]
                if nz.size == 0:
                    continue
                den = Ediag[col] - Ediag[nz] + self.fast[sig].carrier
                if warn_below is not None and np.any(np.abs(den) < warn_below):
                    warnings.warn(f"near-resonant channel {format_signature(sig, self.names)} "
                                  f"from {n}: min |den| = {np.min(np.abs(den)):.4g}", RuntimeWarning)
                E2 += float(np.sum(np.abs(amp[nz]) ** 2 / den))
            out.append(float(Ediag[col]) + E2)
        return out

    def chirp_frame(self, resonant: Signature, mode: str) -> Dict[str, float]:
        """Frame vector c that makes `resonant` carrier chirp-free by rotating only `mode`.

        Solves s + c_mode Delta_mode = 0. Example: exchange (s=+1; a:+1, b:-1) on b gives
        c_b = +1 (frame term -phidot n_b); subharmonic (s=+2; a:-1) on a gives c_a = +2
        (frame term -2 phidot n_a).
        """
        k = self.names.index(mode)
        d = resonant[1][k]
        if d == 0:
            raise ValueError(f"mode {mode} does not appear in signature {resonant}")
        return {mode: -resonant[0] / d}

    def slow_hamiltonian(self, frame: Optional[Mapping[str, float]] = None
                         ) -> Tuple[NOPoly, Dict[Signature, float]]:
        """First-order slow Hamiltonian in frame c with residual carriers.

        Returns
        -------
        H : NOPoly
            H_anh + slow bins - phidot sum_k c_k n_k.
        residual : dict
            Residual carrier of each slow signature in frame c:
            bare + phidot (s + sum_k c_k Delta_k). For non-constant phidot the phase
            is Omega_bare t + (s + c.Delta) phi(t).
        """
        c = {m: 0.0 for m in self.names}
        if frame:
            c.update(frame)
        H = self.H_anh.copy()
        for op in self.slow.values():
            H = H + op
        for k, m in enumerate(self.names):
            if c[m]:
                pd = self.syms.phidot if self.symbolic else self.phidot
                H = H + NOPoly.mode_op(self.N, k, 1, 1, -c[m] * pd)
        res = {sig: self.bare_carrier(sig) + self.phidot * (sig[0] + sum(c[m] * d for m, d in zip(self.names, sig[1])))
               for sig in self.slow}
        return H.pruned(self.arith), res

    def numeric(self, poly: NOPoly) -> NOPoly:
        """Substitute device numbers into a symbolic polynomial."""
        if not self.symbolic:
            return poly
        sm = self.syms.subs_map(self.dev, self.phidot, self.phiddot, self.eta_dot)
        return NOPoly(self.N, {k: complex(sp.N(sp.sympify(c).subs(sm))) for k, c in poly.terms.items()})

    def convergence_report(self, cutoffs: Mapping[str, int], top: int = 10
                           ) -> List[Tuple[float, Signature, float, float]]:
        """Largest ||h_sigma|| / |Omega_sigma| on a truncated space.

        Returns
        -------
        list of (ratio, signature, carrier, norm), sorted descending.
        """
        cuts = [cutoffs[m] for m in self.names]
        rows = []
        for sig, T in self.fast.items():
            op = self.numeric(T.op)
            nrm = float(np.linalg.norm(op.to_matrix(cuts), 2))
            rows.append((nrm / abs(T.carrier), sig, T.carrier, nrm))
        rows.sort(key=lambda r: -r[0])
        return rows[:top]


class HighOrderExpansion:
    """Effective Hamiltonian of the linear-frame H_I(t) to arbitrary order.

    Time-dependent Schrieffer-Wolff (van Vleck) expansion with generator S(t) =
    sum_sigma s_sigma e^{-i delta_sigma t}, where delta_sigma is the detuning (carrier) of
    signature sigma in the linear frame. With psi' = e^{S} psi,

        H' = sum_k ad_S^k(V)/k! + sum_k ad_S^k(W)/(k+1)!,   W = i dS/dt,
        V  = H_anh + sum_n g_n X_I^n.

    Order counting (lambda): every term of V (Kerr and each g_n X^n monomial) is order 1.
    At order n, S_n removes the fast part of the order-n remainder R_n: delta s_n = -fast(R_n),
    W_n = -fast(R_n). The slow part is H_eff^(n), kept with its residual detuning.

    A signature is slow if |delta| < slow_cutoff, or if it is resonant: its strength (largest
    matrix element out of states with at most n_exc quanta) exceeds resonance_ratio |delta| at
    some order. Removing such a term would give s ~ strength/delta > 1 and blow up every higher
    order. Once slow, a signature stays slow at all orders; if it was already removed at a
    lower order the expansion restarts with it slow throughout.

    Consistency with LinearFrameMagnus: order 2 static = second_order() static part, and
    order 3 static = anharmonic_correction() for number-diagonal H_s (tested).

    Parameters
    ----------
    L : LinearFrameMagnus
        Device, slow_cutoff and phidot (numeric mode only; near_cutoff is not used).
    max_order : int
        Highest order computed.
    q_final : int, optional
        Keep only what is needed for matrix elements out of states with at most q_final
        quanta in total: order-n objects are truncated to sum_k q_k <= q_final +
        d (max_order - n), d = largest monomial degree. Exact for those matrix elements.
    resonance_ratio : float or None
        Strength/|delta| above which a term is kept slow; None uses slow_cutoff only.
    n_exc : int
        Quanta of the source states used for the resonance test.
    """

    def __init__(self, L: LinearFrameMagnus, max_order: int = 5, q_final: Optional[int] = None,
                 resonance_ratio: Optional[float] = 1.0, n_exc: int = 2) -> None:
        if L.symbolic:
            raise ValueError("HighOrderExpansion needs a numeric LinearFrameMagnus")
        self.L, self.max_order, self.q_final = L, max_order, q_final
        self.resonance_ratio, self.n_exc = resonance_ratio, n_exc
        self.d = max(max(L.dev.g), 2)
        self._sources = [n for n in product(*[range(n_exc + 1)] * L.N) if sum(n) <= n_exc]
        self.resonant: Dict[Signature, int] = {}  # signature -> order at which it turned resonant
        for _ in range(max_order + 1):
            if not self._run():
                break

    def strength(self, op: NOPoly) -> float:
        """Largest |<m|op|n>| over source states n with at most n_exc quanta (exact)."""
        return max((abs(v) for n in self._sources for v in op.apply_to_fock(n).values()), default=0.0)

    def is_slow(self, sig: Signature) -> bool:
        return abs(self.L.carrier(sig)) < self.L.slow_cutoff or sig in self.resonant

    def _qmax(self, n: int) -> Optional[int]:
        return None if self.q_final is None else self.q_final + self.d * (self.max_order - n)

    def _run(self) -> bool:
        """One pass; returns True if it must restart because a removed signature turned resonant."""
        L, N, top = self.L, self.L.N, self.max_order
        self.R: Dict[int, NOPoly] = {}
        self.H: Dict[int, NOPoly] = {}
        self.S: Dict[int, NOPoly] = {}
        removed = set()
        V = (L.H_anh + L.H_nl).pruned(L.arith)
        T: Dict[int, Dict[int, NOPoly]] = {0: {1: V}}
        U: Dict[int, Dict[int, NOPoly]] = {0: {}}
        for n in range(1, top + 1):
            qn = self._qmax(n)
            R = V if n == 1 else NOPoly(N)
            for k in range(1, n):
                Tk, Uk = NOPoly(N), NOPoly(N)
                for m in range(1, n):
                    if n - m in T[k - 1]:
                        Tk = Tk + self.S[m].commutator(T[k - 1][n - m], qn)
                    if n - m in U[k - 1]:
                        Uk = Uk + self.S[m].commutator(U[k - 1][n - m], qn)
                T.setdefault(k, {})[n] = Tk.truncated(qn).pruned(L.arith)
                U.setdefault(k, {})[n] = Uk.truncated(qn).pruned(L.arith)
                R = R + T[k][n].scale(1 / factorial(k)) + U[k][n].scale(1 / factorial(k + 1))
            R = R.truncated(qn).pruned(L.arith)
            bins = R.by_signature()
            if self.resonance_ratio is not None:
                restart = False
                for sig, op in bins.items():
                    if not self.is_slow(sig) and self.strength(op) > self.resonance_ratio * abs(L.carrier(sig)):
                        self.resonant[sig] = n
                        restart |= sig in removed
                if restart:
                    return True
            self.R[n] = R
            slow, Sn, Wn = NOPoly(N), NOPoly(N), NOPoly(N)
            for sig, op in bins.items():
                if self.is_slow(sig):
                    slow.terms.update(op.terms)
                else:
                    Sn = Sn + op.scale(-1 / L.carrier(sig))
                    Wn = Wn - op
                    removed.add(sig)
            self.H[n], self.S[n] = slow, Sn
            U[0][n] = Wn
        return False

    def effective(self, order: Optional[int] = None) -> NOPoly:
        """Slow Hamiltonian through `order` (residual detunings are kept per signature)."""
        H = NOPoly(self.L.N)
        for n in range(1, (order or self.max_order) + 1):
            H = H + self.H[n]
        return H.pruned(self.L.arith)

    def generator(self, order: Optional[int] = None) -> NOPoly:
        """S = sum_n S_n (all signatures; multiply by e^{-i delta_sigma t} per bin)."""
        S = NOPoly(self.L.N)
        for n in range(1, (order or self.max_order) + 1):
            S = S + self.S[n]
        return S.pruned(self.L.arith)


def example_device() -> Device:
    """Device from the user's allocation. lam_c = 1, alpha_c = 0, eta = 0.9 are ASSUMPTIONS."""
    return Device(modes=["a", "b", "c"],
                  omega={"a": 3.5, "b": 5.7, "c": 4.7},
                  lam={"a": 0.1, "b": 0.1, "c": 1.0},
                  alpha={"a": -0.12, "b": -0.12, "c": 0.0},
                  g={3: 0.06}, omega_p=5.7 - 3.5, eta=0.9)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None, help="JSON in this module's schema (see Device.from_dict)")
    ap.add_argument("--slow-cutoff", type=float, default=0.1)
    ap.add_argument("--near-cutoff", type=float, default=None)
    ap.add_argument("--phidot", type=float, nargs="*", default=[0.0], help="chirp rates (same units as omega)")
    ap.add_argument("--phiddot", type=float, default=0.0)
    ap.add_argument("--max-order", type=int, default=2,
                    help="also print level energies from HighOrderExpansion through this order (3-5)")
    args = ap.parse_args(argv)
    dev = Device.from_json(args.device) if args.device else example_device()
    states = [{}, {"a": 1}, {"b": 1}, {"a": 1, "b": 1}, {"a": 2}, {"b": 2}, {"c": 1}]
    for pd in args.phidot:
        L = LinearFrameMagnus(dev, slow_cutoff=args.slow_cutoff, near_cutoff=args.near_cutoff,
                              phidot=pd, phiddot=args.phiddot)
        print(f"\n=== phidot = {pd:+.4f} ===")
        print("slow first-order signatures:")
        for sig in L.slow:
            print("  ", format_signature(sig, L.names), f"bare carrier {L.bare_carrier(sig):+.4f}")
        H0 = L.effective_static(include_anharmonic_correction=False)
        H1 = L.effective_static(include_anharmonic_correction=True)
        e0 = {tuple(sorted(s.items())): L.level_energy(s, H=H0) for s in states}
        e1 = {tuple(sorted(s.items())): L.level_energy(s, H=H1) for s in states}
        vac0, vac1 = e0[()], e1[()]
        rr = L.rung_resolved_shifts(states, {m: 7 for m in L.names}, warn_below=args.slow_cutoff)
        print("level energies rel. vacuum, linear frame [2nd order | + alpha corr | rung-resolved] MHz:")
        for i, s in enumerate(states):
            k = tuple(sorted(s.items()))
            lab = "|" + ",".join(f"{m}{s.get(m, 0)}" for m in L.names) + ">"
            print(f"   {lab:14s} {1e3 * (e0[k] - vac0):+10.4f} | {1e3 * (e1[k] - vac1):+10.4f}"
                  f" | {1e3 * (rr[i] - rr[0]):+10.4f}")
        zz = (e1[(("a", 1), ("b", 1))] - e1[(("a", 1),)] - e1[(("b", 1),)] + vac1)
        print(f"   ZZ (static, 2nd order + alpha corr): {1e3 * zz:+.5f} MHz")
        _, pairs = L.second_order()
        print(f"slow second-order pair terms: {len(pairs)}")
        for p in pairs[:8]:
            print("  ", format_signature(p.total, L.names), f"residual {p.residual:+.4f}",
                  f"from {format_signature(p.sig_i, L.names)} x {format_signature(p.sig_j, L.names)}")
        print("convergence (||h||/|Omega|, cutoffs a,b:3 c:5):")
        for r, sig, Om, nrm in L.convergence_report({"a": 3, "b": 3, "c": 5}, top=6):
            print(f"   {r:7.4f}  {format_signature(sig, L.names):34s} Omega {Om:+.4f}  ||h|| {nrm:.4f}")
        if args.max_order > 2:
            E = HighOrderExpansion(L, max_order=args.max_order, q_final=2)
            if E.resonant:
                print("kept slow as resonant (strength > |delta|):",
                      ", ".join(f"{format_signature(s, L.names)} (order {n})" for s, n in E.resonant.items()))
            print("static level energies by order (MHz, rel. vacuum):")
            print("   " + " " * 14 + "".join(f"{f'<= {n}':>12s}" for n in range(1, args.max_order + 1)))
            Hs = [E.effective(n) for n in range(1, args.max_order + 1)]
            for s in states:
                lab = "|" + ",".join(f"{m}{s.get(m, 0)}" for m in L.names) + ">"
                n = [s.get(m, 0) for m in L.names]
                z = [0] * L.N
                print(f"   {lab:14s}" + "".join(f"{1e3 * complex(H.diagonal_element(n) - H.diagonal_element(z)).real:+12.4f}"
                                            for H in Hs))


if __name__ == "__main__":
    main()
