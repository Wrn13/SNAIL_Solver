"""Frequency spectrum of the linear-frame Hamiltonian, order by order, vs omega_p and omega_b.

Every term of the time-dependent Schrieffer-Wolff expansion (HighOrderExpansion) is placed at
its detuning delta in the linear frame (e^{-i delta t}) with strength max |<m|h|n>| over the
states n with at most MAX_EXC quanta (exact, no Fock cutoff). Order n is the remainder R_n:
everything generated at lambda^n, where each Kerr or g_n X^n term counts once. Terms with
|delta| < slow_cutoff, or resonant ones (strength > RESONANCE_RATIO |delta|), stay in H_eff;
the rest are removed by the order-n generator.

What the page computes itself, at any (omega_p, omega_b):
  Kerr and order 1  operators are frequency independent; only carriers move.
  order 2           R_2 = [S_1, H_s] + (1/2)[S_1, F_1] is a sum of commutators [h_i, h_j] with
                    coefficients from the detunings, so every commutator is stored once
                    (``pair_tables``).
Precomputed (``line_point``): orders 3-5 along omega_p = |omega_2 - omega_1| (the exchange
lock), at every step of the omega_b sweep, keeping the TOP_K strongest terms per order.

``--check N`` compares the page's order-2 evaluation (``browser_second_order``) with
HighOrderExpansion at N random points. Writes term_spectrum.html from term_spectrum_template.html.
"""
from __future__ import annotations

import base64
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from magnus2 import Device, HighOrderExpansion, LinearFrameMagnus, NOPoly

HERE = Path(__file__).resolve().parent
TIE = 1e-9  # GHz; same tolerance as the page
MAX_EXC = 2
TOP_K = 250
RESONANCE_RATIO = 1.0  # strength / |delta| above which a term is kept in H_eff (HighOrderExpansion default)
# 16-bit log quantization of stored strengths (MHz)
QA = (-3.0, 11.0)

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def op_symbols(names: Sequence[str], snail: str) -> Dict[str, str]:
    """â₁, â₂, ... for the modes in order, ŝ for the SNAIL."""
    out, i = {}, 0
    for m in names:
        if m == snail:
            out[m] = "ŝ"
        else:
            i += 1
            out[m] = "â" + str(i).translate(SUB)
    return out


def monomial_label(o: Sequence[tuple], sym: Sequence[str]) -> str:
    parts = []
    for x, (p, q) in zip(sym, o):
        if p:
            parts.append(f"{x}†" + (str(p).translate(SUP) if p > 1 else ""))
        if q:
            parts.append(x + (str(q).translate(SUP) if q > 1 else ""))
    return " ".join(parts) or "1"


def pump_label(s: int) -> str:
    if s == 0:
        return ""
    base = "η" if s > 0 else "η*"
    return base + (str(abs(s)).translate(SUP) if abs(s) > 1 else "")


def signature_label(sig, sym: Sequence[str]) -> str:
    s, d = sig
    body = ", ".join(f"{x}:{'+' if v > 0 else '−'}{abs(v)}" for x, v in zip(sym, d) if v)
    return f"(s={'+' if s >= 0 else '−'}{abs(s)}; {body or 'diag'})"


def _ordered(op: NOPoly, sym: Sequence[str]) -> list:
    return sorted(op.terms, key=lambda k: (sum(p + q for p, q in k[1]), monomial_label(k[1], sym)))


def meta(order: int, sig, op: NOPoly, sym: Sequence[str], extra: str, n_ops: int = 8) -> dict:
    """Label (pump factor x lowest-degree operator, '+ …' if more), signature, first operators."""
    keys = _ordered(op, sym)
    ops = [" ".join(x for x in (pump_label(s), monomial_label(o, sym)) if x) for s, o in keys]
    return {"o": order, "l": ops[0] + (" + …" if len(ops) > 1 else ""), "sig": signature_label(sig, sym),
            "ops": ops[:n_ops], "nops": len(ops), "s": sig[0], "d": list(sig[1]), "x": extra}


class Strength:
    """max |<m|h|n>| over source states n with at most max_exc quanta in total (exact)."""

    def __init__(self, n_modes: int, max_exc: int) -> None:
        self.max_exc = max_exc
        self.sources = [n for n in np.ndindex(*([max_exc + 1] * n_modes)) if sum(n) <= max_exc]
        self.rows: Dict[tuple, int] = {}  # target-state registry for stored matrix elements

    def elements(self, op: NOPoly) -> Dict[Tuple[int, int], complex]:
        out = {}
        for c, n in enumerate(self.sources):
            for m, v in op.apply_to_fock(n).items():
                out[(self.rows.setdefault(m, len(self.rows)), c)] = complex(v)
        return out

    def __call__(self, op: NOPoly) -> float:
        g = 0.0
        for n in self.sources:
            for v in op.apply_to_fock(n).values():
                g = max(g, abs(v))
        return g


def fixed_terms(L: LinearFrameMagnus, strength: Strength, sym: Sequence[str]) -> Dict[str, dict]:
    """Kerr and order 1: operators do not depend on omega_k or omega_p, only their carriers do."""
    diag = (0, (0,) * L.N)
    out = {}
    for (_, o), c in L.H_anh.terms.items():
        op = NOPoly(L.N, {(0, o): c})
        out[f"K|{o}"] = {**meta(0, diag, op, sym, "Kerr"), "g": strength(op)}
    for sig, op in L.H_nl.by_signature().items():
        g = strength(op)
        if g > 1e-12:
            out[f"1|{sig}"] = {**meta(1, sig, op, sym, ""), "g": g}
    return out


def pair_tables(L: LinearFrameMagnus, strength: Strength, sym: Sequence[str],
                keys: Dict[str, dict]) -> Tuple[list, list]:
    """Frequency-independent ingredients of order 2, R_2 = [S_1, H_s] + (1/2)[S_1, F_1].

    With S_1 = -sum_fast h_i / delta_i, R_2 = sum_{i<j} c_ij [h_i, h_j] over the first-order
    bins plus H_anh (a bin with delta = 0), where c_ij = (1/2)(1/delta_j - 1/delta_i) if both
    are fast, -1/delta_i if only i is fast, +1/delta_j if only j is fast, 0 if both are slow.
    A bin is slow if |delta| < slow_cutoff or its strength exceeds RESONANCE_RATIO |delta|.
    Opposite pairs give static terms, stored per number-diagonal monomial (each its own term);
    the rest are stored as matrix elements on the source states, summed per total signature.

    Returns
    -------
    bins : list of [s, d, strength, t2]
        First-order signatures, then H_anh. strength (GHz) is for the resonance test; t2 is
        the order-2 term with the same signature (-1 if none), for the order-2 restart rule.
    pairs : list of [i, j, kind, term, flat]
        kind 0 (static): flat = [term, re, im, ...]; kind 1: flat = [row, col, re, im, ...].
    """
    diag = (0, (0,) * L.N)
    zero = ((0, 0),) * L.N
    index = {k: i for i, k in enumerate(keys)}
    bins = list(L.H_nl.by_signature().items()) + [(diag, L.H_anh)]
    union: Dict[str, NOPoly] = {}
    pairs = []

    def term(key: str, sig, op: NOPoly, extra: str) -> int:
        union[key] = union.get(key, NOPoly(L.N)) + op
        if key not in index:
            index[key] = len(keys)
            keys[key] = {"o": 2, "s": sig[0], "d": list(sig[1]), "x": extra}
        return index[key]

    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            (si, hi), (sj, hj) = bins[i], bins[j]
            C = hi.commutator(hj).pruned(L.arith)
            if C.is_empty():
                continue
            tot = (si[0] + sj[0], tuple(x + y for x, y in zip(si[1], sj[1])))
            if tot[0] == 0 and not any(tot[1]):
                flat = []
                for (s, o), c in C.terms.items():
                    if o == zero:
                        continue
                    t = term(f"2s|{o}", diag, NOPoly(L.N, {(s, o): c}), "static")
                    keys[list(keys)[t]]["wt"] = strength(NOPoly(L.N, {(s, o): 1.0}))
                    flat += [t, c.real, c.imag]
                if flat:
                    pairs.append([i, j, 0, -1, flat])
            else:
                el = strength.elements(C)
                flat = [x for (r, c), v in el.items() if abs(v) > 1e-15 for x in (r, c, v.real, v.imag)]
                if flat:
                    pairs.append([i, j, 1, term(f"2|{tot}", tot, C, "pair terms"), flat])
    for key, op in union.items():
        m = meta(2, (keys[key]["s"], tuple(keys[key]["d"])), op, sym, keys[key]["x"])
        keys[key].update({k: m[k] for k in ("l", "sig", "ops", "nops")})
    return [[s, list(d), strength(op), index.get(f"2|{(s, d)}", -1)] for (s, d), op in bins], pairs


def _fast(delta: float, g: float, sc: float) -> bool:
    return abs(delta) >= sc - TIE and g <= RESONANCE_RATIO * abs(delta)


def _coef(a: float, b: float, fa: bool, fb: bool) -> float:
    if fa and fb:
        return 0.5 * (1 / b - 1 / a)
    if fa:
        return -1 / a
    if fb:
        return 1 / b
    return 0.0


def browser_second_order(data: dict, dp: float, db: float) -> Dict[int, float]:
    """Python port of the page's order-2 evaluation: {term index: strength in MHz}.

    Mirrors HighOrderExpansion through order 2: a first-order bin is slow if |delta| <
    slow_cutoff or strength > resonance_ratio |delta|; if the order-2 term at a fast bin's
    signature is itself resonant, that bin becomes slow and R_2 is recomputed.
    """
    si, sc, rr = data["sweep_index"], data["slow_cutoff"], data["resonance_ratio"]
    dl = [w + s * dp + d[si] * db for w, (s, d, _, _) in zip(data["bin_w0"], data["bins"])]
    fast = [_fast(x, g, sc) for x, (_, _, g, _) in zip(dl, data["bins"])]
    terms = data["terms"]
    while True:
        acc: Dict[int, dict] = {}
        for i, j, kind, t, flat in data["pairs"]:
            c = _coef(dl[i], dl[j], fast[i], fast[j])
            if c == 0.0:
                continue
            if kind == 0:
                for k in range(0, len(flat), 3):
                    a = acc.setdefault(int(flat[k]), {})
                    a[0] = a.get(0, 0) + c * complex(flat[k + 1], flat[k + 2])
            else:
                a = acc.setdefault(t, {})
                for k in range(0, len(flat), 4):
                    key = (int(flat[k]), int(flat[k + 1]))
                    a[key] = a.get(key, 0) + c * complex(flat[k + 2], flat[k + 3])
        g2 = {t: 1e3 * max(abs(v) for v in a.values()) * terms[t].get("wt", 1.0) for t, a in acc.items()}
        restart = False
        for b, (_, _, _, t2) in enumerate(data["bins"]):
            if fast[b] and t2 >= 0 and g2.get(t2, 0.0) / 1e3 > rr * abs(dl[b]):
                fast[b], restart = False, True
        if not restart:
            return g2


def device_at(base: Device, sweep: str, db: float, dp: float) -> Device:
    omega = dict(base.omega)
    omega[sweep] += db
    return Device(base.modes, omega, base.lam, base.alpha, base.g, base.omega_p + dp, base.eta)


def exchange_offset(base: Device, sweep: str, db: float, qubits: Sequence[str]) -> float:
    """Pump offset that puts omega_p on |omega_2 - omega_1| at mode offset db."""
    om = lambda m: base.omega[m] + (db if m == sweep else 0.0)
    return abs(om(qubits[1]) - om(qubits[0])) - base.omega_p


def _quant(g: float, lo_hi: Tuple[float, float]) -> int:
    lo, hi = lo_hi
    if g <= 10 ** lo:
        return 0
    return 1 + int(round(min(1.0, (np.log10(g) - lo) / (hi - lo)) * 65534))


def line_point(base: Device, sweep: str, db: float, dp: float, slow_cutoff: float, snail: str) -> Dict[int, list]:
    """Orders 3-5 at one point: {order: [(key, meta, strength_MHz, kept_in_H_eff), ...]} (top-K)."""
    warnings.simplefilter("ignore")
    L = LinearFrameMagnus(device_at(base, sweep, db, dp), slow_cutoff=slow_cutoff)
    sym = [op_symbols(L.names, snail)[m] for m in L.names]
    strength = Strength(L.N, MAX_EXC)
    E = HighOrderExpansion(L, max_order=5, q_final=MAX_EXC, resonance_ratio=RESONANCE_RATIO, n_exc=MAX_EXC)
    diag = (0, (0,) * L.N)
    out = {}
    for n in (3, 4, 5):
        items: Dict[str, list] = {}
        for sig, op in E.R[n].by_signature().items():
            if sig == diag:  # static: one term per number-diagonal monomial
                for k, c in op.terms.items():
                    items[f"{n}s|{k[1]}"] = [diag, NOPoly(L.N, {k: c}), 1e3 * abs(c) * strength(NOPoly(L.N, {k: 1.0})), True]
            else:
                items[f"{n}|{sig}"] = [sig, op, 1e3 * strength(op), E.is_slow(sig)]
        keep = [k for k in sorted(items, key=lambda k: -items[k][2])[:TOP_K] if items[k][2] > 1e-3]
        out[n] = [(k, meta(n, items[k][0], items[k][1], sym, "static" if items[k][0] == diag else ""),
                   items[k][2], items[k][3]) for k in keep]
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="compare the page's order-2 evaluation with HighOrderExpansion at N random points")
    ap.add_argument("--line-step", type=float, default=None,
                    help="omega_b spacing (GHz) of the precomputed orders 3-5 (default: the slider step)")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    wa = 3.5
    wb = 1.5 * wa - 0.170
    base = Device(["a", "b", "s"], {"a": wa, "b": wb, "s": 4.7}, {"a": .1, "b": .1, "s": 1.0},
                  {"a": -.15, "b": -.15, "s": 0.0}, {3: .06}, wb - wa, 1.5)
    snail, sweep = "s", "b"
    qubits = [m for m in base.modes if m != snail]
    slow_cutoff = 0.02
    step, span = 0.005, 2.0
    # +-span around nominal, in whole steps; the pump stops at omega_p = 0
    p_lo = -min(span, np.floor(base.omega_p / step) * step)
    p_range, b_range = [round(p_lo, 6), span], [-span, span]

    L = LinearFrameMagnus(base, slow_cutoff=slow_cutoff)
    sym = [op_symbols(L.names, snail)[m] for m in L.names]
    strength = Strength(L.N, MAX_EXC)
    keys: Dict[str, dict] = fixed_terms(L, strength, sym)
    bins, pairs = pair_tables(L, strength, sym, keys)
    for t in keys.values():
        if "g" in t:
            t["g"] = float(f"{t['g'] * 1e3:.6g}")
    carrier0 = lambda s, d: round(sum(dk * base.omega[m] for dk, m in zip(d, base.modes)) + s * base.omega_p, 9)
    bin_w0 = [carrier0(b[0], b[1]) for b in bins]
    for p in pairs:
        p[4] = [float(f"{x:.7g}") for x in p[4]]
    data = {"device": {"modes": base.modes, "omega": base.omega, "lam": base.lam, "alpha": base.alpha,
                       "g": base.g, "eta": base.eta, "wp0": base.omega_p, "sym": op_symbols(base.modes, snail)},
            "sweep": sweep, "sweep_index": base.modes.index(sweep), "step": step,
            "p_range": p_range, "b_range": b_range, "slow_cutoff": slow_cutoff, "max_exc": MAX_EXC,
            "resonance_ratio": RESONANCE_RATIO,
            "bins": bins, "bin_w0": bin_w0, "pairs": pairs}

    if args.check:
        data["terms"] = list(keys.values())
        index = {k: i for i, k in enumerate(keys)}
        zero = ((0, 0),) * L.N
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(args.check):
            dp, db = float(rng.uniform(*p_range)), float(rng.uniform(*b_range))
            E = HighOrderExpansion(LinearFrameMagnus(device_at(base, sweep, db, dp), slow_cutoff=slow_cutoff),
                                   max_order=2, resonance_ratio=RESONANCE_RATIO, n_exc=MAX_EXC)
            ref = {}
            for sig, op in E.R[2].by_signature().items():
                if sig[0] == 0 and not any(sig[1]):
                    for (s, o), c in op.terms.items():
                        if o != zero:
                            ref[index[f"2s|{o}"]] = 1e3 * abs(c) * strength(NOPoly(L.N, {(s, o): 1.0}))
                elif f"2|{sig}" in index:
                    ref[index[f"2|{sig}"]] = 1e3 * strength(op)
                else:
                    assert strength(op) < 1e-12, sig
            got = {t: g for t, g in browser_second_order(data, dp, db).items() if g > 1e-6}
            for t in set(ref) | set(got):
                a, b = ref.get(t, 0.0), got.get(t, 0.0)
                if max(a, b) > 1e-6:
                    worst = max(worst, abs(a - b) / max(abs(a), 1e-3))
        print(f"order-2 check at {args.check} points: worst relative error {worst:.2e}")
        return

    # orders 3-5 along the exchange-locked line
    from joblib import Parallel, delayed
    lstep = args.line_step or step
    dbs = np.round(np.arange(b_range[0], b_range[1] + 1e-9, lstep), 6)
    jobs = [(float(db), round(exchange_offset(base, sweep, float(db), qubits), 9)) for db in dbs]
    # cached, since the line takes ~30 min; the stamp covers device, grid and settings
    # (delete term_spectrum_cache.pkl after changing magnus2 itself)
    import pickle
    cache = HERE / "term_spectrum_cache.pkl"
    stamp = repr((base, slow_cutoff, jobs, MAX_EXC, TOP_K, RESONANCE_RATIO, snail))
    results = None
    if cache.exists():
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved["stamp"] == stamp:
            results = saved["results"]
            print(f"using cached orders 3-5 from {cache.name}")
    if results is None:
        results = Parallel(n_jobs=-1, verbose=5)(delayed(line_point)(base, sweep, db, dp, slow_cutoff, snail)
                                                 for db, dp in jobs)
        with open(cache, "wb") as f:
            pickle.dump({"stamp": stamp, "results": results}, f)
    index = {k: i for i, k in enumerate(keys)}
    words: List[int] = []
    offsets = []
    for res in results:
        offsets.append(len(words))
        for n in (3, 4, 5):
            rows = res[n]
            words.append(len(rows))
            for key, m, g, kept in rows:
                if key not in index:
                    index[key] = len(keys)
                    keys[key] = m
                words += [index[key] | (kept << 15), _quant(g, QA)]  # bit 15: kept in H_eff
    assert len(keys) < 32768
    for t in keys.values():
        t["w0"] = carrier0(t["s"], t["d"])
    data.update({"terms": list(keys.values()),
                 "line": {"dbs": dbs.tolist(), "offsets": offsets, "qa": QA, "top_k": TOP_K,
                          "blob": base64.b64encode(np.asarray(words, dtype="<u2").tobytes()).decode()}})
    tpl = (HERE / "term_spectrum_template.html").read_text()
    out = HERE / "term_spectrum.html"
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False)))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(keys)} terms, {len(pairs)} commutator pairs, "
          f"{len(dbs)} line points)")


if __name__ == "__main__":
    main()
