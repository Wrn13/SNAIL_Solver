"""Frequency spectrum of the linear-frame Hamiltonian, order by order, vs omega_p and omega_b.

Every term of the time-dependent Schrieffer-Wolff expansion (HighOrderExpansion) is placed at
its detuning delta in the linear frame (e^{-i delta t}). A term is one normal-ordered monomial
c eta^s prod_k a_k^{dag p_k} a_k^{q_k} with its own coefficient c (pump amplitudes included);
its strength is |c| max |<m|O|n>| over the unit monomial O and its adjoint, from states n with
at most MAX_EXC quanta (exact, no Fock cutoff). Order n is the remainder R_n: everything
generated at lambda^n, where each Kerr or g_n X^n term counts once.

Which terms stay in H_eff is decided per signature (s, net quanta per mode), as in
HighOrderExpansion: every monomial of a signature shares its detuning and is removed by the
same generator term. A signature stays if |delta| < slow_cutoff or it is resonant (the
largest matrix element of the whole signature > RESONANCE_RATIO |delta|).

What the page computes itself, at any (omega_p, omega_b):
  Kerr and order 1  coefficients are frequency independent; only detunings move.
  order 2           R_2 = [S_1, H_s] + (1/2)[S_1, F_1] is a sum of commutators [h_i, h_j] with
                    coefficients from the detunings, so every commutator is stored once
                    (``pair_tables``), as monomial coefficients and as signature matrix elements.
Precomputed (``line_point``): orders 3-5 along omega_p = |omega_2 - omega_1| (the exchange
lock), at every step of the omega_b sweep, keeping the TOP_K strongest monomials per order.

``--check N`` compares the page's order-2 evaluation (``browser_second_order``), monomial by
monomial, with HighOrderExpansion at N random points. Writes term_spectrum.html from term_spectrum_template.html.
"""
from __future__ import annotations

import base64
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tdsw import Device, HighOrderExpansion, LinearFrameSW, NOPoly

HERE = Path(__file__).resolve().parent
TIE = 1e-9  # GHz; same tolerance as the page
MAX_EXC = 2
TOP_K = 250
LINE_FORMAT = 4  # bump when line_point's output changes (part of the cache stamp)
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


def signature_of(key) -> tuple:
    s, o = key
    return (s, tuple(q - p for p, q in o))


def meta(order: int, key, sym: Sequence[str], strength: "Strength") -> dict:
    """One monomial: label (pump factor, operator), signature, unit-monomial weight."""
    s, o = key
    sig = signature_of(key)
    diag = s == 0 and not any(sig[1])
    return {"o": order, "l": " ".join(x for x in (pump_label(s), monomial_label(o, sym)) if x),
            "sig": signature_label(sig, sym), "s": s, "d": list(sig[1]), "wt": strength.weight(o),
            "x": "static" if diag else ""}


class Strength:
    """max |<m|h|n>| over source states n with at most max_exc quanta in total (exact)."""

    def __init__(self, n_modes: int, max_exc: int) -> None:
        self.max_exc = max_exc
        self.sources = [n for n in np.ndindex(*([max_exc + 1] * n_modes)) if sum(n) <= max_exc]
        self.rows: Dict[tuple, int] = {}  # target-state registry for stored matrix elements
        self._weights: Dict[tuple, float] = {}

    def weight(self, o: tuple) -> float:
        """Strength of the unit monomial o or its adjoint, so h and h^dag (at -delta) agree."""
        if o not in self._weights:
            n = len(o)
            self._weights[o] = max(self(NOPoly(n, {(0, o): 1.0})),
                                   self(NOPoly(n, {(0, tuple((q, p) for p, q in o)): 1.0})))
        return self._weights[o]

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


def fixed_terms(L: LinearFrameSW, strength: Strength, sym: Sequence[str]) -> Dict[str, dict]:
    """Kerr and order 1, one entry per monomial: coefficients do not depend on omega_k or omega_p."""
    out = {}
    for order, H in ((0, L.H_anh), (1, L.H_nl)):
        for key, c in H.terms.items():
            m = meta(order, key, sym, strength)
            if abs(c) * m["wt"] > 1e-12:
                c = complex(c)
                out[f"{order}|{key}"] = {**m, "c": [c.real, c.imag], "g": abs(c) * m["wt"]}
    return out


def pair_tables(L: LinearFrameSW, strength: Strength, sym: Sequence[str],
                keys: Dict[str, dict]) -> Tuple[list, list, list]:
    """Frequency-independent ingredients of order 2, R_2 = [S_1, H_s] + (1/2)[S_1, F_1].

    With S_1 = -sum_fast h_i / delta_i, R_2 = sum_{i<j} c_ij [h_i, h_j] over the first-order
    bins plus H_anh (a bin with delta = 0), where c_ij = (1/2)(1/delta_j - 1/delta_i) if both
    are fast, -1/delta_i if only i is fast, +1/delta_j if only j is fast, 0 if both are slow.
    A bin is slow if |delta| < slow_cutoff or its strength exceeds RESONANCE_RATIO |delta|.
    Each commutator is stored twice: as monomial coefficients (the terms shown) and, for a
    non-static total signature, as matrix elements on the source states (the signature's
    strength, which decides whether it stays in H_eff).

    Returns
    -------
    bins : list of [s, d, strength, k2]
        First-order signatures, then H_anh. strength (GHz) is for the resonance test; k2 indexes
        sig2 for the order-2 signature equal to it (-1 if none), for the order-2 restart rule.
    sig2 : list of [s, d]
        Non-static order-2 signatures.
    pairs : list of [i, j, k2, elements, monomials]
        elements = [row, col, re, im, ...] of the signature sig2[k2] (empty if static);
        monomials = [term, re, im, ...].
    """
    diag = (0, (0,) * L.N)
    zero = ((0, 0),) * L.N
    index = {k: i for i, k in enumerate(keys)}
    bins = list(L.H_nl.by_signature().items()) + [(diag, L.H_anh)]
    sig2: Dict[tuple, int] = {}
    pairs = []
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            (si, hi), (sj, hj) = bins[i], bins[j]
            C = hi.commutator(hj).pruned(L.arith)
            mono = []
            for key, c in C.terms.items():
                if key[1] == zero:  # c-number: a global energy shift
                    continue
                name = f"2|{key}"
                if name not in index:
                    index[name] = len(keys)
                    keys[name] = meta(2, key, sym, strength)
                c = complex(c)
                mono += [index[name], c.real, c.imag]
            if not mono:
                continue
            tot = (si[0] + sj[0], tuple(x + y for x, y in zip(si[1], sj[1])))
            k2, el = -1, []
            if tot != diag:
                k2 = sig2.setdefault(tot, len(sig2))
                el = [x for (r, c), v in strength.elements(C).items() if abs(v) > 1e-15 for x in (r, c, v.real, v.imag)]
            pairs.append([i, j, k2, el, mono])
    return ([[s, list(d), strength(op), sig2.get((s, d), -1)] for (s, d), op in bins],
            [[s, list(d)] for s, d in sig2], pairs)


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


def browser_second_order(data: dict, dp: float, db: float, forced: frozenset = frozenset()
                         ) -> Tuple[Dict[int, complex], List[bool], set]:
    """Python port of the page's order-2 evaluation.

    Returns ({term index: coefficient in GHz}, fast flag per first-order bin, slow order-2
    signatures as (s, d)). Mirrors HighOrderExpansion through order 2: a first-order bin is
    slow if |delta| < slow_cutoff or strength > resonance_ratio |delta|; an order-2 signature
    is slow likewise (strength of the whole signature); if it is a fast first-order bin, that
    bin becomes slow and R_2 is recomputed; a signature once slow stays slow after a restart.
    forced: signatures known to be resonant at higher
    order (on the precomputed line), slow from the start.
    """
    si, sc, rr = data["sweep_index"], data["slow_cutoff"], data["resonance_ratio"]
    det = lambda w0, s, d: w0 + s * dp + d[si] * db
    dl = [det(w, s, d) for w, (s, d, _, _) in zip(data["bin_w0"], data["bins"])]
    d2 = [det(w, s, d) for w, (s, d) in zip(data["sig2_w0"], data["sig2"])]
    fast = [_fast(x, g, sc) and (s, tuple(d)) not in forced for x, (s, d, g, _) in zip(dl, data["bins"])]
    slow2: set = set()  # like HighOrderExpansion.resonant, kept across restarts
    while True:
        mono: Dict[int, complex] = {}
        el: Dict[int, dict] = {}
        for i, j, k2, elements, monomials in data["pairs"]:
            c = _coef(dl[i], dl[j], fast[i], fast[j])
            if c == 0.0:
                continue
            for k in range(0, len(monomials), 3):
                t = int(monomials[k])
                mono[t] = mono.get(t, 0) + c * complex(monomials[k + 1], monomials[k + 2])
            if k2 >= 0:
                a = el.setdefault(k2, {})
                for k in range(0, len(elements), 4):
                    key = (int(elements[k]), int(elements[k + 1]))
                    a[key] = a.get(key, 0) + c * complex(elements[k + 2], elements[k + 3])
        g2 = {k2: max((abs(v) for v in a.values()), default=0.0) for k2, a in el.items()}
        slow2 |= {k2 for k2, g in g2.items() if not _fast(d2[k2], g, sc)}
        restart = False
        for b, (_, _, _, k2) in enumerate(data["bins"]):
            if fast[b] and k2 in slow2:
                fast[b], restart = False, True
        if not restart:
            slow = {(s, tuple(d)) for b, (s, d, _, _) in enumerate(data["bins"]) if not fast[b]}
            slow |= {(data["sig2"][k][0], tuple(data["sig2"][k][1])) for k in slow2} | set(forced)
            return mono, fast, slow


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
    """Orders 3-5 at one point: {order: [(key, meta, strength_MHz, phase, kept_in_H_eff), ...]}.

    One entry per monomial, the TOP_K strongest per order; kept is its signature's verdict.
    out["resonant"] lists every resonant signature: one that turns resonant at order 3-5 after
    being removed at order 1 restarts the expansion, so it also changes orders 1-2 here.
    """
    warnings.simplefilter("ignore")
    L = LinearFrameSW(device_at(base, sweep, db, dp), slow_cutoff=slow_cutoff)
    sym = [op_symbols(L.names, snail)[m] for m in L.names]
    strength = Strength(L.N, MAX_EXC)
    E = HighOrderExpansion(L, max_order=5, q_final=MAX_EXC, resonance_ratio=RESONANCE_RATIO, n_exc=MAX_EXC)
    zero = ((0, 0),) * L.N
    out = {}
    for n in (3, 4, 5):
        items = []
        for key, c in E.R[n].terms.items():
            if key[1] == zero:
                continue
            c = complex(c)
            g = 1e3 * abs(c) * strength.weight(key[1])
            if g > 1e-3:
                items.append((key, c, g))
        items.sort(key=lambda x: -x[2])
        out[n] = [(f"{n}|{key}", meta(n, key, sym, strength), g, float(np.angle(c)), E.is_slow(signature_of(key)))
                  for key, c, g in items[:TOP_K]]
        out[f"total{n}"] = len(items)  # monomials above 1 kHz before the top-K cut
    out["resonant"] = sorted(E.resonant)
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

    L = LinearFrameSW(base, slow_cutoff=slow_cutoff)
    sym = [op_symbols(L.names, snail)[m] for m in L.names]
    strength = Strength(L.N, MAX_EXC)
    keys: Dict[str, dict] = fixed_terms(L, strength, sym)
    bins, sig2, pairs = pair_tables(L, strength, sym, keys)
    for t in keys.values():  # Kerr and order 1 in MHz
        if "g" in t:
            t["g"] = float(f"{t['g'] * 1e3:.6g}")
            t["c"] = [float(f"{x * 1e3:.7g}") for x in t["c"]]
    carrier0 = lambda s, d: round(sum(dk * base.omega[m] for dk, m in zip(d, base.modes)) + s * base.omega_p, 9)
    bin_w0 = [carrier0(b[0], b[1]) for b in bins]
    sig2_w0 = [carrier0(s, d) for s, d in sig2]
    for p in pairs:
        p[3] = [float(f"{x:.7g}") for x in p[3]]
        p[4] = [float(f"{x:.7g}") for x in p[4]]
    data = {"device": {"modes": base.modes, "omega": base.omega, "lam": base.lam, "alpha": base.alpha,
                       "g": base.g, "eta": base.eta, "wp0": base.omega_p, "sym": op_symbols(base.modes, snail)},
            "sweep": sweep, "sweep_index": base.modes.index(sweep), "step": step,
            "p_range": p_range, "b_range": b_range, "slow_cutoff": slow_cutoff, "max_exc": MAX_EXC,
            "resonance_ratio": RESONANCE_RATIO,
            "bins": bins, "bin_w0": bin_w0, "sig2": sig2, "sig2_w0": sig2_w0, "pairs": pairs}

    if args.check:
        data["terms"] = list(keys.values())
        index = {k: i for i, k in enumerate(keys)}
        zero = ((0, 0),) * L.N
        rng = np.random.default_rng(0)
        worst, worst_g, bad_keep, n_terms = 0.0, 0.0, 0, 0
        for _ in range(args.check):
            dp, db = float(rng.uniform(*p_range)), float(rng.uniform(*b_range))
            E = HighOrderExpansion(LinearFrameSW(device_at(base, sweep, db, dp), slow_cutoff=slow_cutoff),
                                   max_order=2, resonance_ratio=RESONANCE_RATIO, n_exc=MAX_EXC)
            got, fast, slow = browser_second_order(data, dp, db)
            ref = {index[f"2|{k}"]: complex(c) for k, c in E.R[2].terms.items() if k[1] != zero and abs(c) > 1e-15}
            for t in set(ref) | set(got):
                a, b = ref.get(t, 0), got.get(t, 0)
                if max(abs(a), abs(b)) > 1e-9:  # GHz
                    worst = max(worst, abs(a - b) / max(abs(a), 1e-6))
                    worst_g = max(worst_g, 1e3 * abs(a - b) * data["terms"][t]["wt"])
            sigs = {signature_of(k) for k in E.R[1].terms} | {signature_of(k) for k in E.R[2].terms}
            for sig in sigs:
                if sig[0] == 0 and not any(sig[1]):
                    continue
                n_terms += 1
                bad_keep += E.is_slow(sig) != (sig in slow or abs(E.L.carrier(sig)) < slow_cutoff - TIE)
        print(f"order-2 check at {args.check} points: worst relative coefficient error {worst:.2e}, "
              f"worst absolute strength error {worst_g:.2e} MHz, keep-rule mismatches {bad_keep}/{n_terms} signatures")
        return

    # orders 3-5 along the exchange-locked line
    from joblib import Parallel, delayed
    lstep = args.line_step or step
    dbs = np.round(np.arange(b_range[0], b_range[1] + 1e-9, lstep), 6)
    jobs = [(float(db), round(exchange_offset(base, sweep, float(db), qubits), 9)) for db in dbs]
    # cached, since the line takes ~30 min; the stamp covers device, grid and settings
    # (delete term_spectrum_cache.pkl after changing tdsw itself)
    import pickle
    cache = HERE / "term_spectrum_cache.pkl"
    stamp = repr((base, slow_cutoff, jobs, MAX_EXC, TOP_K, RESONANCE_RATIO, snail, LINE_FORMAT))
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
    offsets, res_sigs, res_at = [], {}, []
    for res in results:
        offsets.append(len(words))
        res_at.append([res_sigs.setdefault(sig, len(res_sigs)) for sig in res["resonant"]])
        for n in (3, 4, 5):
            rows = res[n]
            words += [len(rows), min(res[f"total{n}"], 65535)]
            for key, m, g, phase, kept in rows:
                if key not in index:
                    index[key] = len(keys)
                    keys[key] = m
                # bit 15: kept in H_eff; phase of the coefficient in units of 2 pi / 65536
                words += [index[key] | (kept << 15), _quant(g, QA), int(round(phase / (2 * np.pi) * 65536)) % 65536]
    assert len(keys) < 32768
    for t in keys.values():
        t["w0"] = carrier0(t["s"], t["d"])
    data.update({"terms": list(keys.values()),
                 "line": {"dbs": dbs.tolist(), "offsets": offsets, "qa": QA, "top_k": TOP_K,
                          "resonant_sigs": [[s, list(d)] for s, d in res_sigs], "resonant": res_at,
                          "blob": base64.b64encode(np.asarray(words, dtype="<u2").tobytes()).decode()}})
    tpl = (HERE / "term_spectrum_template.html").read_text()
    out = HERE / "term_spectrum.html"
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False)))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(keys)} terms, {len(pairs)} commutator pairs, "
          f"{len(dbs)} line points)")


if __name__ == "__main__":
    main()
