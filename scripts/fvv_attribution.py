"""Closed-form FVV analysis at a constant pump amplitude.

    fvv_attribution.py --delta -0.14 --eta 1.3 [--order 4]

In the charge frame F is static. Per order this prints:

  ridge shift     the change, from eta = 0, of E(|10>) - E(|01>) for the dressed states
                  of F^[n] with the exchange coupling removed. This is the Stark shift the
                  chirp follows; H_0's a<->s and b<->s conversions are included exactly.
  bare shift      the same change for the diagonal elements alone (no dressing)
  exchange        the element <01|F^(n)|10> The second-order shift is split by process with eq. (29b):
F^(2) = sum_{n != 0} [H_n, H_-n] / (2 n w), and H_n = sum of the processes with
Fourier index n. Only a process paired with its own conjugate shifts the diagonal.
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
    ap.add_argument("--order", type=int, default=4)
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from floquet import fourier as Fo, frame as Fr, model as M, target
    cpl, _ = target.pulse(target.column(a.delta))
    f, lam, al, g, dims = M.device_of(cpl)
    idx = lambda na, nb, ns: (na * dims[1] + nb) * dims[2] + ns
    k10, k01 = idx(1, 0, 0), idx(0, 1, 0)
    det = lambda X: float((X[k10, k10] - X[k01, k01]).real) / (2 * np.pi) * 1e3     # MHz
    ex, _ = M.expansion_at(f, lam, al, g, dims, a.eta, a.order)
    ex0, _ = M.expansion_at(f, lam, al, g, dims, 0.0, a.order)
    print(f"delta {a.delta * 1e3:+.0f} MHz, |eta| = {a.eta}, w = 2pi x {ex.omega / (2 * np.pi):.3f} GHz")
    total = 0.0
    for n in range(1, a.order + 1):
        d = det(ex.F[n]) - det(ex0.F[n])
        total += d
        e = abs(ex.F[n][k01, k10]) / (2 * np.pi) * 1e3
        print(f"  order {n}: resonance shift {d:+8.3f} MHz (through order {n}: {total:+8.3f})"
              f"   exchange {e:7.3f} MHz")
    occ = M.frame.number_diagonal(dims) if hasattr(M, "frame") else Fr.number_diagonal(dims)

    def ridge(Fm):
        F = Fm.copy()
        dn = occ[:, None, :] - occ[None, :, :]
        ex_mask = np.all(np.abs(dn) == np.abs(np.array([1, -1, 0])), axis=2) & (dn[..., 0] == -dn[..., 1])
        F[ex_mask] = 0.0                                  # drop the exchange (and its copies)
        ev, V = np.linalg.eigh(F)
        e10 = ev[np.argmax(np.abs(V[k10]) ** 2)]
        e01 = ev[np.argmax(np.abs(V[k01]) ** 2)]
        return float(e10 - e01) / (2 * np.pi) * 1e3
    print("  dressed ridge shift (MHz) through order n:",
          "  ".join(f"{n}: {ridge(ex.F_upto(n)) - ridge(ex0.F_upto(n)):+.3f}" for n in range(1, a.order + 1)))
    w = ex.omega
    split = {}
    for x, sgn in ((a.eta, 1.0), (0.0, -1.0)):
        bins = Fo.coefficients(f, lam, al, g, dims, x)
        for sig, Msig in bins.items():
            m = -Fr.charge(sig)
            c = Fo.conj(sig)
            if m == 0 or c not in bins:
                continue
            A, B = 2 * np.pi * Msig.toarray(), 2 * np.pi * bins[c].toarray()
            key = max(sig, c, key=str)
            split[key] = split.get(key, 0.0) + sgn * det((A @ B - B @ A) / (2 * m * w))
    print("  order-2 shift by process, eq. (29b):")
    for key, v in sorted(split.items(), key=lambda kv: -abs(kv[1]))[:10]:
        s = key if key[0] >= 0 else Fo.conj(key)
        print(f"     {v:+8.3f} MHz   {Fo.describe(s):16s} (carrier {abs(Fo.carrier(s, f)) * 1e3:.0f} MHz,"
              f" Fourier index {-Fr.charge(s):+d})")


if __name__ == "__main__":
    main()
