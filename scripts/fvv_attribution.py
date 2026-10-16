"""Closed-form FVV analysis at a constant pump amplitude.

    fvv_attribution.py --delta -0.14 --eta 1.3 [--window 0.2] [--order 4]

Per order: the shift of the |10>-|01> resonance relative to eta = 0 (the Stark shift the
chirp follows), and the exchange matrix element <01|F|10>. The second-order shift is
also split by intermediate fast term, the eq. (29b) sum. With --window 0.2 the
a<->s conversion at ~410 MHz counts as fast, so its dispersive shift shows up here
instead of only in the exact evolution.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytical_exploration"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delta", type=float, default=-0.14)
    ap.add_argument("--eta", type=float, default=1.3)
    ap.add_argument("--window", type=float, default=0.2)
    ap.add_argument("--order", type=int, default=4)
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    from floquet import fourier as Fo, model as M, target
    from floquet.expansion import BERNOULLI, floquet_magnus
    cpl, _ = target.pulse(target.column(a.delta))
    f, lam, al, g, dims = M.device_of(cpl)
    car = Fo.carrier_of(f)
    idx = lambda na, nb, ns: (na * dims[1] + nb) * dims[2] + ns
    k10, k01 = idx(1, 0, 0), idx(0, 1, 0)
    z, ex_sig = Fo.zero(), (1, (1, -1, 0))
    det = lambda Mx: float((Mx[k10, k10] - Mx[k01, k01]).real) if Mx is not None else 0.0
    H, H0 = (Fo.coefficients(f, lam, al, g, dims, x) for x in (a.eta, 0.0))
    E, E0 = (floquet_magnus(h, car, a.order, window_GHz=a.window) for h in (H, H0))
    print(f"delta {a.delta * 1e3:+.0f} MHz, |eta| = {a.eta}, secular window {a.window} GHz")
    total = 0.0
    for n in range(1, a.order + 1):
        d = (det(E.F[n].get(z)) - det(E0.F[n].get(z))) * 1e3
        total += d
        e = abs(complex(E.F[n][ex_sig][k01, k10])) * 1e3 if ex_sig in E.F[n] else 0.0
        print(f"  order {n}: resonance shift {d:+8.3f} MHz (through order {n}: {total:+8.3f})"
              f"   exchange {e:7.3f} MHz")
    # eq. (29b): F^(2)_0 = c_1 sum_sigma [Lambda^(1)_sigma, H_-sigma], c_1 = B_1 (-i)
    c1 = BERNOULLI[1] * (-1j)
    split = {}
    for E_, H_, sgn in ((E, H, 1.0), (E0, H0, -1.0)):
        for s1, L in E_.Lambda[1].items():
            if Fo.conj(s1) not in H_:
                continue
            Cm = Fo.comm({s1: L}, {Fo.conj(s1): H_[Fo.conj(s1)]}).get(z)
            key = max(s1, Fo.conj(s1), key=str)
            split[key] = split.get(key, 0.0) + sgn * det(c1 * Cm if Cm is not None else None)
    print("  order-2 shift by intermediate fast term:")
    for key, v in sorted(split.items(), key=lambda kv: -abs(kv[1]))[:10]:
        s = key if key[0] >= 0 else Fo.conj(key)
        print(f"     {v * 1e3:+8.3f} MHz   {Fo.describe(s):16s} (carrier {abs(car(s)) * 1e3:.0f} MHz)")


if __name__ == "__main__":
    main()
