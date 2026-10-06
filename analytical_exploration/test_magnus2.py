"""Falsification tests for magnus2.

Run with ``pytest test_magnus2.py`` or ``python test_magnus2.py`` (no pytest needed).

The Floquet benchmark uses commensurate toy frequencies (omega0 = 1, period 2 pi) so
that the exact one-period propagator of the linear-frame Hamiltonian exists; its
quasienergies are gauge invariant and must agree with the effective Hamiltonian up to
third-order terms in g_3.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from magnus2 import Device, HighOrderExpansion, LinearFrameMagnus, NOPoly, Arith, opposite

TOY = dict(modes=["a", "c"], omega={"a": 7.0, "c": 11.0}, lam={"a": 0.3, "c": 1.0},
           alpha={"a": -0.25, "c": 0.0}, g={3: 0.01}, omega_p=2.0, eta=0.5)


def toy_device(**kw) -> Device:
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TOY.items()}
    d.update(kw)
    return Device(**d)


def _random_poly(rng: np.random.Generator, n_modes: int, n_terms: int, max_deg: int) -> NOPoly:
    P = NOPoly(n_modes)
    for _ in range(n_terms):
        ops = tuple((int(rng.integers(0, max_deg + 1)), int(rng.integers(0, max_deg + 1))) for _ in range(n_modes))
        P.terms[(0, ops)] = P.terms.get((0, ops), 0) + complex(rng.normal(), rng.normal())
    return P


def test_product_and_dagger_match_dense_matrices() -> None:
    rng = np.random.default_rng(1)
    ar = Arith()
    cut, low = [12, 12], 3
    keep = [i for i, n in enumerate(np.ndindex(*cut)) if max(n) <= low]
    for _ in range(5):
        A, B = _random_poly(rng, 2, 4, 3), _random_poly(rng, 2, 4, 3)
        MA, MB = A.to_matrix(cut), B.to_matrix(cut)
        AB = (A * B).pruned(ar).to_matrix(cut)
        C = A.commutator(B).pruned(ar).to_matrix(cut)
        D = A.dagger(ar).to_matrix(cut)
        ix = np.ix_(keep, keep)
        assert np.allclose(AB[ix], (MA @ MB)[ix], atol=1e-9)
        assert np.allclose(C[ix], (MA @ MB - MB @ MA)[ix], atol=1e-9)
        assert np.allclose(D[ix], MA.conj().T[ix], atol=1e-9)


def test_boson_operator_round_trip() -> None:
    """to/from OpenFermion preserve the operator, including tiny coefficients."""
    try:
        from openfermion import BosonOperator
    except ImportError:
        try:
            import pytest
        except ImportError:
            print("skipped: openfermion not installed")
            return
        pytest.skip("openfermion not installed")
    rng = np.random.default_rng(2)
    cut = [6, 6]
    A = _random_poly(rng, 2, 6, 2).scale(1e-10)
    (s, op), = A.to_boson_operator().items()
    assert np.allclose(NOPoly.from_boson_operator(op, 2, s).to_matrix(cut), A.to_matrix(cut), atol=1e-22)
    B = NOPoly.from_boson_operator(BosonOperator("0 1 0^", 2.0), 2)  # a b a^dag = b (a^dag a + 1)
    ref = NOPoly.mode_op(2, 0, 1, 1) * NOPoly.mode_op(2, 1, 0, 1, 2.0) + NOPoly.mode_op(2, 1, 0, 1, 2.0)
    assert np.allclose(B.to_matrix(cut), ref.to_matrix(cut))


def test_two_level_sign_convention() -> None:
    """Linear drive eps (q + q^dag) on a Kerr mode, carrier Delta in the linear frame.

    The 1/Omega term gives only a c-number for a linear drive, so the Stark shift of the
    0-1 transition comes entirely from the anharmonicity: exactly
    2 alpha eps^2 / (Delta (Delta + alpha)) (three-level perturbation theory) in the
    rung-resolved form, and 2 alpha eps^2 / Delta^2 at leading order in the operator form.
    """
    eps, Delta, alpha = 0.01, 1.3, -0.2
    dev = Device(modes=["q"], omega={"q": Delta}, lam={"q": 1.0}, alpha={"q": alpha},
                 g={1: eps}, omega_p=100.0, eta=0.0)
    L = LinearFrameMagnus(dev, slow_cutoff=0.05)
    e = L.rung_resolved_shifts([{}, {"q": 1}], {"q": 8})
    exact = 2 * alpha * eps ** 2 / (Delta * (Delta + alpha))
    lead = 2 * alpha * eps ** 2 / Delta ** 2
    H = L.effective_static(include_anharmonic_correction=True)
    op = L.level_energy({"q": 1}, H=H) - L.level_energy({}, H=H)
    assert abs((e[1] - e[0]) - exact) < 1e-12
    assert abs(op - lead) < 1e-12


def _floquet(dev: Device, cutoffs: Sequence[int], omega0: float, n_steps: int
             ) -> Tuple[np.ndarray, np.ndarray]:
    L = LinearFrameMagnus(dev, slow_cutoff=1e-9)
    assert not L.slow, "toy must have no carrier-free first-order terms"
    sigs = list(L.fast)
    Ms = np.array([L.fast[s].op.to_matrix(cutoffs) for s in sigs])
    Om = np.array([L.fast[s].carrier for s in sigs])
    H0 = L.H_anh.to_matrix(cutoffs)
    T = 2 * np.pi / omega0
    dt = T / n_steps
    c1, c2 = 0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6
    U = np.eye(H0.shape[0], dtype=complex)

    def H(t: float) -> np.ndarray:
        return H0 + np.tensordot(np.exp(-1j * Om * t), Ms, axes=1)

    for i in range(n_steps):
        t = i * dt
        B1, B2 = -1j * H(t + c1 * dt), -1j * H(t + c2 * dt)
        U = expm(0.5 * dt * (B1 + B2) - (np.sqrt(3) / 12) * dt ** 2 * (B1 @ B2 - B2 @ B1)) @ U
    lam, V = np.linalg.eig(U)
    return -np.angle(lam) / T, V


def _match(vecs: np.ndarray, col: int) -> int:
    return int(np.argmax(np.abs(vecs[col, :]) ** 2))


def _wrap(x: float, w: float) -> float:
    return (x + w / 2) % w - w / 2


def _effective_matrix(L: LinearFrameMagnus, cutoffs: Sequence[int], anh: bool) -> np.ndarray:
    H = L.effective_static(include_anharmonic_correction=anh).to_matrix(cutoffs)
    _, pairs = L.second_order()
    for p in pairs:
        if abs(p.residual) < 1e-12:
            H = H + p.op.to_matrix(cutoffs)
    return H


def test_floquet_quasienergies() -> None:
    """Exact Floquet vs second order (operator, +alpha, rung-resolved) on the toy device."""
    dev = toy_device()
    cut = [8, 8]
    states = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    eps_F, VF = _floquet(dev, cut, 1.0, 6000)
    L = LinearFrameMagnus(dev, slow_cutoff=0.5, near_cutoff=0.5)
    errs: Dict[str, List[float]] = {"op": [], "op+alpha": [], "rung": []}
    vac = np.ravel_multi_index((0, 0), cut)
    fv = eps_F[_match(VF, vac)]
    for key, anh in (("op", False), ("op+alpha", True)):
        He = _effective_matrix(L, cut, anh)
        w, V = np.linalg.eigh(He)
        ev = w[_match(V, vac)]
        for n in states:
            col = np.ravel_multi_index(n, cut)
            pred = w[_match(V, col)] - ev
            exact = _wrap(eps_F[_match(VF, col)] - fv, 1.0)
            errs[key].append(abs(_wrap(pred - exact, 1.0)))
    # Rung-resolved diagonal plus the second-order slow pair terms (needed: omega_c - omega_a
    # = 2 omega_p makes eta^2 a c^dag a resonant second-order exchange between |10> and |01>).
    sub = [n for n in np.ndindex(4, 4)]
    diag = L.rung_resolved_shifts([{"a": n[0], "c": n[1]} for n in sub], {"a": 9, "c": 9})
    P = sum((p.op.to_matrix(cut) for p in L.second_order()[1] if abs(p.residual) < 1e-12),
            np.zeros((64, 64), complex))
    ix = [np.ravel_multi_index(n, cut) for n in sub]
    Hr = np.diag(diag).astype(complex) + P[np.ix_(ix, ix)]
    w, V = np.linalg.eigh(Hr)
    ev = w[_match(V, sub.index((0, 0)))]
    for n in states:
        col = np.ravel_multi_index(n, cut)
        exact = _wrap(eps_F[_match(VF, col)] - fv, 1.0)
        errs["rung"].append(abs(_wrap(w[_match(V, sub.index(n))] - ev - exact, 1.0)))
    print("Floquet benchmark |error| per state", states)
    for k, v in errs.items():
        print(f"  {k:9s}", " ".join(f"{x:.2e}" for x in v))
    assert max(errs["op+alpha"]) < max(errs["op"])
    assert max(errs["rung"]) < 5e-6


def test_constant_chirp_is_detuned_pump() -> None:
    """phidot enters only as omega_p -> omega_p + phidot (s-weighted carriers).

    Checks the bookkeeping against an exact Floquet run at omega_p + phidot.
    """
    phidot = 0.5
    cut = [8, 8]
    states = [(0, 0), (1, 0), (2, 0), (0, 1)]
    dev = toy_device()
    L = LinearFrameMagnus(dev, slow_cutoff=0.2, near_cutoff=0.2, phidot=phidot)
    Lref = LinearFrameMagnus(toy_device(omega_p=2.0 + phidot), slow_cutoff=0.2, near_cutoff=0.2)
    r = L.rung_resolved_shifts([{"a": n[0], "c": n[1]} for n in states], {"a": 8, "c": 8})
    rref = Lref.rung_resolved_shifts([{"a": n[0], "c": n[1]} for n in states], {"a": 8, "c": 8})
    assert np.allclose(r, rref, atol=1e-14)
    eps_F, VF = _floquet(toy_device(omega_p=2.0 + phidot), cut, 0.5, 12000)
    vac = np.ravel_multi_index((0, 0), cut)
    for i, n in enumerate(states):
        col = np.ravel_multi_index(n, cut)
        exact = _wrap(eps_F[_match(VF, col)] - eps_F[_match(VF, vac)], 0.5)
        assert abs(_wrap((r[i] - r[0]) - exact, 0.5)) < 2e-5


def test_phiddot_drops_out_of_static_second_order() -> None:
    """Static part of (1/2)[K, F] is independent of phiddot (derivation check)."""
    dev = toy_device()
    L0 = LinearFrameMagnus(dev, slow_cutoff=0.5, phiddot=0.0)
    L1 = LinearFrameMagnus(dev, slow_cutoff=0.5, phiddot=0.3)
    S0 = L0.static_from_kick(L0.kick_operator(include_envelope=False))
    S1 = L1.static_from_kick(L1.kick_operator(include_envelope=False))
    ref = L0.effective_static(True) - L0.static_slow_part()
    cut = [6, 6]
    assert np.allclose(S0.to_matrix(cut), S1.to_matrix(cut), atol=1e-13)
    assert np.allclose(S0.to_matrix(cut), ref.to_matrix(cut), atol=1e-13)
    k1 = L1.kick_operator(include_anharmonic=False, include_envelope=False)
    k0 = L0.kick_operator(include_anharmonic=False, include_envelope=False)
    assert any(not np.allclose(k1[s].to_matrix(cut), k0[s].to_matrix(cut)) for s in k0 if s[0] != 0)


def test_symbolic_matches_numeric() -> None:
    dev = toy_device()
    Ls = LinearFrameMagnus(dev, symbolic=True, slow_cutoff=0.5, phidot=0.1)
    Ln = LinearFrameMagnus(dev, symbolic=False, slow_cutoff=0.5, phidot=0.1)
    Hs = Ls.numeric(Ls.effective_static(True))
    Hn = Ln.effective_static(True)
    cut = [6, 6]
    assert np.allclose(Hs.to_matrix(cut), Hn.to_matrix(cut), atol=1e-12)


def test_operator_equals_rung_resolved_when_alpha_zero() -> None:
    dev = toy_device(alpha={"a": 0.0, "c": 0.0})
    L = LinearFrameMagnus(dev, slow_cutoff=0.5)
    states = [{}, {"a": 1}, {"a": 2}, {"c": 1}, {"a": 1, "c": 1}]
    r = L.rung_resolved_shifts(states, {"a": 9, "c": 9})
    H = L.effective_static(False)
    o = [L.level_energy(s, H=H) for s in states]
    assert np.allclose(np.array(r) - r[0], np.array(o) - o[0], atol=1e-12)


def _diag(P: NOPoly) -> NOPoly:
    return NOPoly(P.n_modes, {k: c for k, c in P.terms.items() if k[0] == 0 and all(p == q for p, q in k[1])})


def test_high_order_reproduces_second_order_and_kerr() -> None:
    """Orders 2 and 3 of HighOrderExpansion against the independent LinearFrameMagnus formulas."""
    dev = toy_device()
    L = LinearFrameMagnus(dev, slow_cutoff=0.5, phidot=0.1)
    E = HighOrderExpansion(L, max_order=3)
    cut = [6, 6]
    assert np.allclose(_diag(E.H[2]).to_matrix(cut), L.second_order()[0].to_matrix(cut), atol=1e-15)
    assert np.allclose(_diag(E.H[3]).to_matrix(cut), L.anharmonic_correction().to_matrix(cut), atol=1e-15)


def test_high_order_floquet_converges() -> None:
    """Exact Floquet quasienergies: the error falls with every order through 5."""
    dev = toy_device()
    cut = [8, 8]
    states = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    eps_F, VF = _floquet(dev, cut, 1.0, 6000)
    vac = np.ravel_multi_index((0, 0), cut)
    fv = eps_F[_match(VF, vac)]
    E = HighOrderExpansion(LinearFrameMagnus(dev, slow_cutoff=1e-9), max_order=5)
    assert not E.resonant
    errs = []
    for order in range(1, 6):
        w, V = np.linalg.eigh(E.effective(order).to_matrix(cut))
        ev = w[_match(V, vac)]
        errs.append(max(abs(_wrap((w[_match(V, c)] - ev) - _wrap(eps_F[_match(VF, c)] - fv, 1.0), 1.0))
                        for c in (np.ravel_multi_index(n, cut) for n in states)))
    print("Floquet error by order 1..5:", " ".join(f"{e:.2e}" for e in errs))
    assert all(b < a for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-7


def test_high_order_truncation_is_exact_on_low_states() -> None:
    dev = toy_device()
    L = LinearFrameMagnus(dev, slow_cutoff=0.5, phidot=0.1)
    E, Ef = HighOrderExpansion(L, max_order=4), HighOrderExpansion(L, max_order=4, q_final=2)
    for n in (3, 4):
        for s in [n for n in np.ndindex(3, 3) if sum(n) <= 2]:
            a, b = E.R[n].apply_to_fock(s), Ef.R[n].apply_to_fock(s)
            assert all(abs(a.get(m, 0) - b.get(m, 0)) < 1e-15 for m in set(a) | set(b))


def test_resonant_term_is_kept_slow() -> None:
    """3 omega_p = omega_s + 40 MHz on the resonance_scan device: the order-2 term eta^3 s^dag
    (about 305 MHz at delta = +40 MHz) must stay in H_eff. Removing it gives a generator of
    size ~7.6 and an order-4 term stronger than every first-order term."""
    wa = 3.5
    wb = 1.5 * wa - 0.170
    dev = Device(["a", "b", "s"], {"a": wa, "b": wb, "s": 4.7}, {"a": .1, "b": .1, "s": 1.0},
                 {"a": -.15, "b": -.15, "s": 0.0}, {3: .06}, wb - wa, 1.5)
    L = LinearFrameMagnus(dev, slow_cutoff=0.02)
    E = HighOrderExpansion(L, max_order=4, q_final=2)
    sig = (3, (0, 0, -1))
    assert E.resonant.get(sig) == 2 and sig in E.H[2].by_signature()
    top = {n: max(E.strength(op) for op in E.R[n].by_signature().values()) for n in E.R}
    print("strongest term by order (GHz):", {n: round(g, 3) for n, g in top.items()})
    assert max(top[3], top[4]) < top[1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
