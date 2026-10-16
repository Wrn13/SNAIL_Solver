"""Floquet-Van Vleck (FVV) effective models of the subharmonic iSWAP gate.

The expansion is the one of Bock, Ernst & Uhrig, "Floquet Theory and Average
Hamiltonian Theory Revisited" (arXiv:2609.17121). Equation numbers below are theirs.

    fourier      H(t) = sum_sigma H_sigma e^{-i Omega_sigma t}           (2), (6a)
                 [n, m] := [H_n, H_m] and nested commutators              (7), (26)
    expansion    dLambda/dt = sum_k B_k/k! ([-i Lambda, H]_k - [i Lambda, F]_k)   (8b)
                 solved order by order in Fourier space                   (27)
                 gauges: FVV, Lambda_0 = 0                                (24), (29), (30)
                         AHT, Lambda(t0) = 0                              (31)
                 closed forms F(1..3), Lambda(1..2)                       (28a-f), (29), (30)
    propagator   U(t, t0) = e^{-i Lambda(t)} e^{-i F (t - t0)} e^{+i Lambda(t0)}   (3)

Two generalisations make the paper's single-frequency theory apply to the gate:

* The Fourier index m becomes a signature sigma = (s, Delta): s pump quanta and a change
  Delta of each mode's occupation, with carrier Omega_sigma = sum_k Delta_k w_k + s w_p.
  The paper's m w is -Omega_sigma, because our coefficients carry e^{-i Omega t}.
* Signatures with |Omega_sigma| < `window_GHz` count as secular and stay in F, with their
  residual carrier. 3 w_p = w_s sits 130 MHz from resonance at delta = -140 MHz with a
  ~110 MHz matrix element, so dividing by its carrier would diverge at first order.
  Eq. (8b) holds pointwise in t, so a slowly time-dependent F is allowed.

Our additions, not in the paper:

* `classical`  the coupler's classical response to the pump (harmonic balance). With it,
  `fourier` gives the Fourier coefficients in the frame displaced by that response.
* `model`      the adiabatic step. The expansion runs at a constant pump amplitude |eta| on
  a grid of |eta|. Coefficients are fitted in |eta|, then the pulse supplies eta(t).
* `target`     the stored figs_ridge_tracked fidelities the models are compared against.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if os.path.join(_ROOT, "src") not in sys.path and os.path.isdir(os.path.join(_ROOT, "src")):
    sys.path.insert(0, os.path.join(_ROOT, "src"))
