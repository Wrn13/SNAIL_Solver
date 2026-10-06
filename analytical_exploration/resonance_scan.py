"""Rung-resolved scan for resonant first- and second-order processes (linear frame).

Resonance condition for a term with carrier Omega connecting |n> -> |m>:
E_n - E_m + Omega ~ 0, with E the diagonal energies of H_s (H_anh), optionally
Stark-corrected by the rung-resolved second-order shifts at the given eta.
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
from magnus2 import Device, LinearFrameMagnus, format_signature


def scan(dev: Device, sources: Sequence[Tuple[int, ...]], cut: Sequence[int], window: float,
         stark: bool, near_cutoff: float = 0.5) -> List[Tuple]:
    warnings.simplefilter("ignore")
    L = LinearFrameMagnus(dev, slow_cutoff=0.02, near_cutoff=near_cutoff)
    names = L.names
    idx = list(np.ndindex(*cut))
    Hs = L.static_slow_part()
    E = np.array([complex(Hs.diagonal_element(n)).real for n in idx])
    if stark:
        inner = [n for n in idx if all(nk + 3 < ck + 2 for nk, ck in zip(n, cut))]
        sh = L.rung_resolved_shifts([dict(zip(names, n)) for n in idx], {m: c + 3 for m, c in zip(names, cut)})
        E = np.array(sh)
    E = E - E[0]
    rows = []
    terms = [(1, sig, T.op, T.carrier) for sig, T in L.fast.items()]
    terms += [(1, sig, op, L.carrier(sig)) for sig, op in L.slow.items()]
    _, pairs = L.second_order()
    terms += [(2, p.total, p.op, p.residual) for p in pairs]
    mats = {}
    for order, sig, op, Om in terms:
        M = op.to_matrix(cut)
        for n in sources:
            c = int(np.ravel_multi_index(n, cut))
            for r in np.nonzero(np.abs(M[:, c]) > 1e-12)[0]:
                det = E[c] - E[r] + Om
                if abs(det) < window:
                    rows.append((order, format_signature(sig, names), n, idx[r], det, abs(M[r, c])))
    agg: Dict[Tuple, List] = {}
    for o, s, n, m, d, g in rows:
        k = (o, s, n, m)
        if k in agg:
            agg[k][1] += g
        else:
            agg[k] = [d, g]
    return sorted([(k[0], k[1], k[2], k[3], v[0], v[1]) for k, v in agg.items()], key=lambda r: abs(r[4]))


if __name__ == "__main__":
    wa = 3.5
    wb = 1.5 * wa - 0.170
    dev = Device(["a", "b", "s"], {"a": wa, "b": wb, "s": 4.7}, {"a": .1, "b": .1, "s": 1.0},
                 {"a": -.15, "b": -.15, "s": 0.0}, {3: .06}, wb - wa, 1.5)
    src = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 0), (0, 2, 0)]
    for stark in (False, True):
        print(f"\n=== omega_b = {wb:.3f}, omega_p = {wb - wa:.3f}, eta = 1.5, Stark-corrected E: {stark} ===")
        print("ord  signature                      from -> to          detuning(MHz)  |coupling|(MHz)  ratio")
        for o, s, n, m, d, g in scan(dev, src, [5, 5, 5], 0.12, stark):
            print(f" {o}   {s:30s} {str(n):10s}->{str(m):10s} {1e3*d:+9.1f}    {1e3*g:9.3f}   {g/max(abs(d),1e-9):7.3f}")
