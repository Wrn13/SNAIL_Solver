"""Hamiltonian ladder: reduced (Schrieffer-Wolff) models of the ZhouCoupler gate.

`levels` builds the effective Hamiltonian of one device and sorts its terms into the
ladder classes; `engine` swaps it in for ``ZhouCoupler.to_qutip_hamiltonian`` so the
unchanged tune-up and scoring code run on a reduced model; `stored` reads the
figs_ridge_tracked target the reduced fidelities are compared against.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AE = os.path.dirname(_HERE)
if _AE not in sys.path:            # tdsw lives beside this package, not in src/
    sys.path.insert(0, _AE)
