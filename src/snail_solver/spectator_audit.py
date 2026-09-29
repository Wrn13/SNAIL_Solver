#!/usr/bin/env python3
r"""Spectator channel audit: which parasitic processes exist, and how STRONG they are.

``plot_allocation.py`` shows WHERE the curated collision centres are; this tool reads
HOW BAD each one is out of the model. For a spectator at w_spec it builds the 4-mode
system [a, b, coupler, spec] and walks every term of ``ZhouCoupler.expand_terms``:
each carries a carrier detuning Omega_j (the channel's beat) and an operator O_j,
whose matrix elements exciting the spectator from the low-lying computational
states are the parasitic couplings. The enumeration is complete by construction,
including eta^2 / eta^3 (two-pump subharmonic), |2>-involving, anharmonicity-shifted
and coupler-mediated channels.

Per channel it reports

    g            coupling matrix element (MHz), weighted by |eta|^n_pump
    detuning     signed Omega_j / 2 pi (MHz) -- the DRAG beat for that channel
    g/|detuning| first-order excitation amplitude. >= 1: NOT perturbative, no
                 DRAG/virtual-Z fixes it, the placement has to move.
    P_exc        off-resonant excitation estimate min(1, 2 (g/detuning)^2)
    transition    which modes change occupation, and the pump order

and a band scan sweeps the spectator across [w_a, w_b] to find the quiet windows.

CLI
---
    # audit one placement, ranked table + chart
    python -m snail_solver.spectator_audit --device evan_device.json --spec 4.0

    # scan the whole band and mark the quiet windows
    python -m snail_solver.spectator_audit --device evan_device.json --scan-points 241 \
        --out figs/spectator_audit.png --save-npz figs/spectator_audit.npz

    # table only, no figure (quick terminal audit)
    python -m snail_solver.spectator_audit --device evan_device.json --spec 4.0 --no-plot
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi
A, B, COUPLER, SPEC = 0, 1, 2, 3        # mode order, matching sweep_target.py


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
SPEC_MARKERS = ["o", "v", "P", "X", "h", "<"]      # one per spectator, cycled


def _build(config: Dict[str, Any], specs: List[float], t_g_ns: float,
           coupler_levels: Optional[int], s_lv: int):
    """[a, b, coupler, spec_1, ...] with the gate pump set and iSWAP-normalized.

    Each spectator takes participation ``lam_b`` and ``anharm_spec_GHz`` (the
    ``sweep_target.py`` convention). Returns ``(cpl, peak_eta, spec_indices)``.
    """
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, make_chirp
    wa, wb = (float(f) for f in config["qubit_freqs_GHz"])
    ws = float(config["coupler_freq_GHz"])
    q_lv = int(config.get("qubit_levels", 3))
    c_lv = int(coupler_levels or config.get("coupler_levels", 5))
    aq = float(config.get("anharm_qubit_GHz", 0.0))
    a_sp = float(config.get("anharm_spec_GHz", 0.0))
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])
    idxs = [SPEC + k for k in range(len(specs))]
    part = {A: float(config["lam_a"]), B: float(config["lam_b"])}
    part.update({i: float(config["lam_b"]) for i in idxs})
    anh = {A: aq, B: aq}
    anh.update({i: a_sp for i in idxs})
    cpl = ZhouCoupler(mode_freqs_GHz=[wa, wb, ws] + specs, coupler_index=COUPLER,
                      participations=part, nonlinearities=nonlin,
                      levels=[q_lv, q_lv, c_lv] + [s_lv] * len(specs),
                      anharmonicities_GHz=anh)
    # the GATE pump, so it carries the device's chirp (make_chirp -> None when unset)
    cpl.set_pump(PumpTone(w_p_GHz=abs(wb - wa),
                          envelope=RaisedCosine(amp=1.0, t_g=float(t_g_ns)),
                          is_eta=True,
                          chirp=make_chirp(config.get("chirp_coeffs_GHz") or None,
                                           float(t_g_ns))),
                 normalize_iswap=(A, B))
    return cpl, float(cpl.peak_eta()), idxs


def build_with_spectators(config: Dict[str, Any], w_specs_GHz: Sequence[float],
                          t_g_ns: float, *, coupler_levels: Optional[int] = None,
                          spec_levels: Optional[int] = None):
    """:func:`build_with_spectator` for any number of spectators (modes 3, 4, ...).

    The Hilbert dimension multiplies by ``spec_levels`` per spectator, so with two or
    more the default is 2 (enough for |0> -> |1>). Returns
    ``(ZhouCoupler, peak |eta|, spectator mode indices)``.
    """
    specs = [float(w) for w in w_specs_GHz]
    s_lv = int(spec_levels or (3 if len(specs) <= 1 else 2))
    return _build(config, specs, t_g_ns, coupler_levels, s_lv)


def mode_tags(cpl, spec_indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Per-mode identity for labelling: short tag, frequency (GHz) and role."""
    info: Dict[int, Dict[str, Any]] = {}
    for m in range(cpl.n_modes):
        f = float(cpl.omega[m]) / TWO_PI
        if m == A:
            info[m] = dict(tag="a", freq=f, role="qubit a (target)")
        elif m == B:
            info[m] = dict(tag="b", freq=f, role="qubit b (target)")
        elif m == COUPLER:
            info[m] = dict(tag="s", freq=f, role="SNAIL coupler")
        else:
            k = list(spec_indices).index(m) + 1 if m in spec_indices else 0
            info[m] = dict(tag=f"sp{k}", freq=f, role=f"spectator {k}")
    return info


def build_with_spectator(config: Dict[str, Any], w_spec_GHz: float, t_g_ns: float,
                         *, coupler_levels: Optional[int] = None,
                         spec_levels: Optional[int] = None):
    """4-mode [a, b, coupler, spec] system (spectator at ABSOLUTE `w_spec_GHz`) with
    the iSWAP pump set and normalized; t_g fixes |eta|. Returns ``(cpl, peak |eta|)``.
    """
    s_lv = int(spec_levels or config.get("qubit_levels", 3))
    cpl, eta, _idxs = _build(config, [float(w_spec_GHz)], t_g_ns, coupler_levels, s_lv)
    return cpl, eta


def t_g_for_eta(config: Dict[str, Any], eta: float) -> float:
    """Raised-cosine peak-|eta| gate time: t_g = 1 / (12 g3 la lb eta)."""
    return 1.0 / (12.0 * config["g3_GHz"] * config["lam_a"] * config["lam_b"]
                  * float(eta))


# --------------------------------------------------------------------------- #
# Channel enumeration                                                         #
# --------------------------------------------------------------------------- #
def _computational_starts(cpl) -> List[int]:
    """Fock indices of the low-lying states: qubits in {0, 1}, everything else |0>."""
    starts = []
    for na in range(min(2, cpl.dims[A])):
        for nb in range(min(2, cpl.dims[B])):
            occ = [0] * cpl.n_modes
            occ[A], occ[B] = na, nb
            starts.append(cpl.fock_index(occ))
    return starts


def _label(cpl, i_from: int, i_to: int, n_pump: int,
           info: Optional[Dict[int, Dict[str, Any]]] = None) -> str:
    """Compact transition label, e.g. 'a1->0 sp1 0->1 (2 pump)'."""
    if info is None:
        info = {A: dict(tag="a"), B: dict(tag="b"), COUPLER: dict(tag="s")}
        for m in range(cpl.n_modes):
            info.setdefault(m, dict(tag=f"sp{m - SPEC + 1}"))
    o_from, o_to = cpl.decode_index(i_from), cpl.decode_index(i_to)
    parts = [f"{info[m]['tag']}{o_from[m]}->{o_to[m]}"
             for m in range(cpl.n_modes) if o_from[m] != o_to[m]]
    return " ".join(parts) + (f" ({n_pump} pump)" if n_pump else " (static)")


def _process_name(cpl, i_from: int, i_to: int, n_pump: int,
                  info: Dict[int, Dict[str, Any]], is_target: bool = False) -> str:
    """Physics name for a process, e.g. 'qubit A subharmonic', 'A-SNAIL spectator'.

    One mode gains alone -> SUBHARMONIC (``n w_p = w_i``; n=1 is a direct drive);
    one up / one down -> SPECTATOR exchange (pump-assisted for n >= 1); a mode into
    |2> -> LEAKAGE via its partner; two up -> PAIR CREATION.
    """
    o_f, o_t = cpl.decode_index(i_from), cpl.decode_index(i_to)
    delta = {m: o_t[m] - o_f[m] for m in range(cpl.n_modes) if o_t[m] != o_f[m]}
    ups = [m for m, d in delta.items() if d > 0]
    downs = [m for m, d in delta.items() if d < 0]

    def short(m: int) -> str:
        return info[m]["tag"].upper() if info[m]["tag"] != "s" else "SNAIL"

    def long(m: int) -> str:
        t = info[m]["tag"]
        if t == "s":
            return "SNAIL"
        if t in ("a", "b"):
            return f"qubit {t.upper()}"
        return f"spectator {t[2:]}" if t.startswith("sp") else t.upper()

    ORD = {0: "static", 1: "direct drive", 2: "subharmonic",
           3: "3rd subharmonic", 4: "4th subharmonic"}

    if is_target:
        return "target iSWAP (A-B exchange)"

    # one mode changes alone: a subharmonic of that mode (n w_p = w_i)
    if len(ups) == 1 and not downs:
        m = ups[0]
        lvl = "" if o_t[m] <= 1 else f"|{o_f[m]}>->|{o_t[m]}> "
        return f"{long(m)} {lvl}{ORD.get(n_pump, f'{n_pump}-pump')}"
    if len(downs) == 1 and not ups:
        m = downs[0]                          # conjugate branch of the same resonance
        return f"{long(m)} {ORD.get(n_pump, f'{n_pump}-pump')} (emission)"

    # one up, one down: an exchange between the two
    if len(ups) == 1 and len(downs) == 1:
        u, d = ups[0], downs[0]
        if o_t[u] >= 2:                       # driven into |2>: leakage
            return f"{short(u)} |2> leakage (via {short(d)})"
        if n_pump == 0:
            return f"{short(d)}-{short(u)} static exchange"
        if n_pump == 1:
            return f"{short(d)}-{short(u)} spectator"
        return f"{short(d)}-{short(u)} {n_pump}-pump spectator"

    if len(ups) == 2 and not downs:
        return f"{short(ups[0])}+{short(ups[1])} pair creation"
    if len(downs) == 2 and not ups:
        return f"{short(downs[0])}+{short(downs[1])} pair annihilation"
    return _label(cpl, i_from, i_to, n_pump, info)


def _describe(cpl, i_from: int, i_to: int, n_pump: int,
              info: Dict[int, Dict[str, Any]]) -> str:
    """Physical sentence for a process, naming each mode AND its frequency, e.g.
    '2w_p -> s@4.700 |0>->|1>'  or  'w_p: b@5.700 1->0, sp1@4.600 0->1'."""
    o_from, o_to = cpl.decode_index(i_from), cpl.decode_index(i_to)
    moved = [m for m in range(cpl.n_modes) if o_from[m] != o_to[m]]
    bits = [f"{info[m]['tag']}@{info[m]['freq']:.3f} "
            f"|{o_from[m]}>->|{o_to[m]}>" for m in moved]
    drive = ("static" if n_pump == 0 else
             ("w_p" if n_pump == 1 else f"{n_pump}w_p"))
    return f"{drive}: " + ", ".join(bits)


def spectator_channels(cpl, *, window_GHz: float = 1.0, min_g_MHz: float = 1e-3,
                       dedupe_MHz: float = 0.5) -> List[Dict[str, Any]]:
    """Every near-resonant process that EXCITES the spectator, with its strength.

    Matrix elements of each ``expand_terms`` term from a low-lying computational state
    to one with the spectator excited, weighted by ``|eta|^n_pump``. Channels detuned
    beyond `window_GHz` or weaker than `min_g_MHz` are dropped; detunings within
    `dedupe_MHz` merge, keeping the strongest. Sorted by descending ``ratio``; keys
    g_MHz, detuning_MHz, ratio, P_exc, n_pump, transition, perturbative, resonant.
    """
    eta = float(cpl.peak_eta())
    starts = _computational_starts(cpl)

    best: Dict[int, Dict[str, Any]] = {}
    for Omega, pump_sig, O in cpl.expand_terms(cutoff_GHz=window_GHz):
        det = Omega / TWO_PI                                   # signed, GHz
        n_pump = len(pump_sig)
        scale = eta ** n_pump
        Oa = np.asarray(O)
        for i in starts:
            col = Oa[:, i]
            for f in np.nonzero(np.abs(col) > 1e-12)[0]:
                if cpl.decode_index(f)[SPEC] <= cpl.decode_index(i)[SPEC]:
                    continue                                   # spectator not excited
                g = abs(col[f]) * scale / TWO_PI                # GHz
                if g * 1e3 < min_g_MHz:
                    continue
                # on resonance the ratio diverges: flag it, saturate P_exc
                resonant = abs(det) * 1e3 < 1e-3           # within 1 kHz
                ratio = np.inf if resonant else g / abs(det)
                key = int(round(det * 1e3 / max(dedupe_MHz, 1e-6)))
                rec = dict(g_MHz=g * 1e3, detuning_MHz=det * 1e3, ratio=ratio,
                           P_exc=1.0 if resonant else min(1.0, 2.0 * ratio ** 2),
                           n_pump=n_pump, transition=_label(cpl, i, f, n_pump),
                           perturbative=bool(not resonant and ratio < 1.0),
                           resonant=bool(resonant))
                prev = best.get(key)
                if prev is None or rec["ratio"] > prev["ratio"]:
                    best[key] = rec
    return sorted(best.values(), key=lambda r: -r["ratio"])


def drag_verdict(g_MHz: float, det_MHz: float, t_g_ns: float) -> Dict[str, Any]:
    r"""How well first-order DRAG can suppress a channel at (g, detuning).

    First-order DRAG (Motzoi et al. 2009) cancels the adiabatic transient of
    amplitude :math:`\simeq g/|\delta|`, leaving :math:`\sim (g/|\delta|)^2`, so the
    suppression factor is itself :math:`\simeq g/|\delta|` -- DRAG works best where
    it is needed least. It fails outright when :math:`g/|\delta| \geq 1` (not
    perturbative: reallocate) or :math:`|\delta| \lesssim 1/t_g` (inside the pulse
    bandwidth: not adiabatic).

    Returns ratio (g/|det|), bandwidth_MHz, adiabaticity (|det| * t_g), suppression
    (post/pre amplitude, <1 is good), suppression_dB, verdict, colour.
    """
    bw_MHz = 1e3 / float(t_g_ns)                    # pulse spectral width (MHz)
    ad = abs(float(det_MHz)) / bw_MHz               # |det| * t_g, adiabaticity
    ratio = (np.inf if abs(det_MHz) < 1e-9
             else abs(float(g_MHz)) / abs(float(det_MHz)))
    if not np.isfinite(ratio) or ratio >= 1.0:
        verdict, supp, colour = "fails: not perturbative", 1.0, "#b22222"
    elif ad <= 1.0:
        verdict, supp, colour = "fails: inside pulse bandwidth", 1.0, "#d95f02"
    elif ad <= 3.0:
        verdict, supp, colour = "marginal: barely adiabatic", min(1.0, ratio), "#e6ab02"
    elif ratio >= 0.3:
        verdict, supp, colour = "marginal: weak suppression", ratio, "#7570b3"
    else:
        verdict, supp, colour = "DRAG effective", ratio, "#1b7837"
    supp = float(max(supp, 1e-6))
    return dict(ratio=float(ratio), bandwidth_MHz=float(bw_MHz),
                adiabaticity=float(ad), suppression=supp,
                suppression_dB=float(-20.0 * np.log10(supp)),
                verdict=verdict, colour=colour)


def interaction_channels(cpl, *, window_GHz: float = 1.0, t_g_ns: float = 100.0,
                         min_g_MHz: float = 1e-3, dedupe_MHz: float = 0.5,
                         spec_indices: Optional[Sequence[int]] = None,
                         has_spectator: bool = True) -> List[Dict[str, Any]]:
    """EVERY near-resonant process, classified, with strength, detuning and DRAG verdict.

    Unlike :func:`spectator_channels`, the gate and all parasites appear together:

    ``target``     the wanted a<->b exchange, |01> <-> |10> (detuning ~ 0)
    ``spectator``  excites the spectator mode
    ``coupler``    excites the SNAIL coupler (heating)
    ``leakage``    drives a or b into |2>
    ``other``      remaining off-diagonal processes in the low-lying manifold

    ``f_p_res_GHz = f_p + det/n`` is the pump frequency at which an ``n``-pump
    channel becomes resonant (``nan`` for static channels).
    """
    eta = float(cpl.peak_eta())
    f_p = float(cpl._pump_tones[0].w_p_GHz)
    n_modes = cpl.n_modes
    if spec_indices is None:
        spec_indices = ([SPEC] if (has_spectator and n_modes > SPEC) else [])
    spec_indices = list(spec_indices)
    info = mode_tags(cpl, spec_indices)
    # expand_terms carries the HARMONIC carrier only; the anharmonicity is a separate
    # diagonal operator, folded in here. SIGN: Omega is (pump) - (harmonic transition)
    # (a two-pump drive on qubit a gives Omega = 2 w_p - w_a), so the detuning from the
    # FULL transition SUBTRACTS the anharmonic difference:
    #
    #     det = Omega - (E_anh[f] - E_anh[i])
    #
    # Check: E(n) = n w_a + alpha n(n-1)/2, so two pumps hit a |1>->|2> at
    # delta = +alpha/2 (adding would mirror it to -alpha/2, a 3x error in g/|det|).
    E_anh = np.real(np.diag(np.asarray(cpl._anharm_op)))
    starts = _computational_starts(cpl)

    best: Dict[Tuple, Dict[str, Any]] = {}
    for Omega, pump_sig, O in cpl.expand_terms(cutoff_GHz=window_GHz + 0.5):
        n_pump = len(pump_sig)
        scale = eta ** n_pump
        Oa = np.asarray(O)
        for i in starts:
            oi = cpl.decode_index(i)
            col = Oa[:, i]
            for f in np.nonzero(np.abs(col) > 1e-12)[0]:
                if f == i:
                    continue                                    # diagonal -> Stark shift
                of = cpl.decode_index(f)
                det = (Omega - (E_anh[f] - E_anh[i])) / TWO_PI   # signed GHz, anharm-shifted
                if abs(det) > window_GHz:
                    continue
                g = abs(col[f]) * scale / TWO_PI                 # GHz
                if g * 1e3 < min_g_MHz:
                    continue
                # --- classify, and record WHICH mode is the victim
                excited_spec = [m for m in spec_indices if of[m] > oi[m]]
                victim = None
                if excited_spec:
                    cat, victim = "spectator", excited_spec[0]
                elif of[COUPLER] > oi[COUPLER]:
                    cat, victim = "coupler", COUPLER
                elif max(of[A], of[B]) >= 2 and max(of[A], of[B]) > max(oi[A], oi[B]):
                    cat = "leakage"
                    victim = A if of[A] >= 2 else B
                elif ((oi[A], oi[B]) in ((0, 1), (1, 0))
                      and (of[A], of[B]) == (oi[B], oi[A])
                      and of[COUPLER] == 0
                      and all(of[m] == 0 for m in spec_indices)):
                    cat = "target"          # the wanted |01> <-> |10> exchange
                else:
                    cat = "other"
                spec_id = (spec_indices.index(victim) + 1
                           if victim is not None and victim in spec_indices else 0)
                rec = dict(g_MHz=g * 1e3, detuning_MHz=det * 1e3, n_pump=n_pump,
                           # Fock indices, for level-resolved use (see `stark_scales`)
                           i_index=int(i), f_index=int(f),
                           category=cat, transition=_label(cpl, i, f, n_pump, info),
                           name=_process_name(cpl, i, f, n_pump, info,
                                              is_target=(cat == "target")),
                           process=_describe(cpl, i, f, n_pump, info),
                           victim=(-1 if victim is None else int(victim)),
                           victim_tag=("--" if victim is None else info[victim]["tag"]),
                           victim_freq_GHz=(np.nan if victim is None
                                            else info[victim]["freq"]),
                           spectator_id=int(spec_id),
                           f_p_res_GHz=(f_p + det / n_pump if n_pump else np.nan))
                rec.update(drag_verdict(rec["g_MHz"], rec["detuning_MHz"], t_g_ns))
                if cat == "target":
                    rec.update(verdict="resonant by design (the gate)",
                               colour="#111111", suppression=1.0, suppression_dB=0.0)
                key = (cat, rec["victim"],
                       int(round(det * 1e3 / max(dedupe_MHz, 1e-6))))
                prev = best.get(key)
                if prev is None or rec["g_MHz"] > prev["g_MHz"]:
                    best[key] = rec
    out = list(best.values())
    # target first, then strongest parasites by pre-DRAG excitation amplitude
    out.sort(key=lambda r: (r["category"] != "target",
                            -(r["g_MHz"] / max(abs(r["detuning_MHz"]), 1e-6))))
    return out


def level_stark_shifts(cpl, eta: float, window_GHz: float = 2.0) -> np.ndarray:
    r"""Second-order AC-Stark shift of EVERY level, in GHz, at peak drive `eta`.

    The diagonal piece :func:`interaction_channels` skips: every level is pushed by
    the couplings that leave it, summed over pump sidebands with ``eta**n_pump``::

        dE_i = - sum_{k != i} |g_ik|^2 / Delta_ik ,   Delta_ik = Omega + E_k - E_i

    Same ``eta^2`` leading order as the target's Stark law, which is why
    :func:`stark_scales` can express it as a multiple of the chirp. Returns the shift
    per Fock index (GHz).
    """
    E_anh = np.real(np.diag(np.asarray(cpl._anharm_op)))
    n = len(E_anh)
    dE = np.zeros(n, dtype=float)
    for Omega, pump_sig, O in cpl.expand_terms(cutoff_GHz=float(window_GHz) + 0.5):
        Oa = np.asarray(O) * (float(eta) ** len(pump_sig))
        for i in range(n):
            col = Oa[:, i]
            for k in np.nonzero(np.abs(col) > 1e-12)[0]:
                if k == i:
                    continue
                det = Omega + (E_anh[k] - E_anh[i])          # rad/ns
                if abs(det) < 1e-9:
                    continue                                  # resonant: not a shift
                dE[i] -= (abs(col[k]) ** 2) / det
    return dE / TWO_PI                                        # GHz


def stark_scales(config: Dict[str, Any], t_g_ns: float, rows: Sequence[Dict[str, Any]],
                 *, k2_MHz: float, k4_MHz: float = 0.0, eta: float = 1.0,
                 amp_scale: float = 1.0, wp_offset_GHz: float = 0.0,
                 window_GHz: float = 2.0) -> Dict[int, float]:
    r"""``kappa`` per channel: its Stark shift as a MULTIPLE of the chirp.

    The chirp is the target's measured Stark curve ``k2 |eta|^2 + k4 |eta|^4``
    (``tune_up.chirp_from_measured_shift``). Other transitions shift at the same
    leading order, so::

        dDelta_j(t) ~ kappa_j * delta_chirp(t),
        kappa_j = [dE(f) - dE(i)] / [k2 eta^2 + k4 eta^4]

    letting a channel track its detuning by reusing the chirp
    (:attr:`envelope.DragChannel.stark_scale`). Leading-order only: `kappa` drifts
    with drive (~30% over ``eta = 0.6..1.0`` against a chevron survey). `rows` are
    :func:`interaction_channels` rows; `k2_MHz`/`k4_MHz` the MEASURED law the chirp
    uses. Returns ``{_audit_beat_key(beat): kappa}``.
    """
    from snail_solver.device_utils import build_coupler
    cpl, _w_p, _eta_pk = build_coupler(config, float(t_g_ns), float(amp_scale),
                                       float(wp_offset_GHz), None, None,
                                       chirp_coeffs_GHz=[])
    dE = level_stark_shifts(cpl, float(eta), window_GHz=window_GHz)
    ref_MHz = float(k2_MHz) * eta ** 2 + float(k4_MHz) * eta ** 4
    if not np.isfinite(ref_MHz) or abs(ref_MHz) < 1e-12:
        raise ValueError("stark_scales needs a non-zero measured target law")
    out: Dict[int, float] = {}
    for r in rows:
        i, f = r.get("i_index"), r.get("f_index")
        if i is None or f is None:
            continue
        d_MHz = (dE[int(f)] - dE[int(i)]) * 1e3
        out[_audit_beat_key(float(r["detuning_MHz"]) / 1e3)] = float(d_MHz / ref_MHz)
    return out


def _audit_beat_key(beat_GHz: float, dedupe_MHz: float = 0.5) -> int:
    """Bucket a beat so the same process arriving from two enumerations coincides."""
    return int(round(float(beat_GHz) * 1e3 / max(float(dedupe_MHz), 1e-6)))


def select_drag_channels(config: Dict[str, Any], t_g_ns: float, *,
                         max_channels: int = 4,
                         require: Sequence[str] = ("leakage", "coupler"),
                         window_GHz: float = 1.0, amp_scale: float = 1.0,
                         wp_offset_GHz: float = 0.0,
                         spec_abs_GHz: Optional[float] = None,
                         min_g_MHz: float = 1e-3, min_ratio: float = 0.02,
                         max_ratio: float = 0.3,
                         chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                         quotient_rule: bool = True) -> Tuple[tuple, Dict[str, Any]]:
    """Audit every near-resonant channel; pick the ones recursive DRAG should correct.

    Unions two enumerations: :func:`interaction_channels` (every Hamiltonian process;
    `require` categories -- the ``|2>`` ladder and SNAIL heating -- are mandatory)
    and ``sweep_common.collision_drag_channels``'s MODE SUBHARMONICS (``w_i = 2 w_p``),
    also mandatory: a subharmonic exciting qubit A |0>->|1> classifies as ``other``
    and would otherwise be missed.

    Mandatory channels come first, then the strongest correctable parasites by
    ``g/|det|``, up to `max_channels` (each channel costs a unit of ``envelope_m``,
    hence pulse area and ``t_g``). Channels at ``g/|det| >= max_ratio`` are left
    uncorrected (the chirp<->DRAG fixed point diverges); below `min_ratio` a channel
    is negligible and not mandatory. Channels a chirp would sweep through their
    collision are dropped by ``sweep_common._drag_channels_filtered``; every drop is
    reported with a reason.

    Returns ``(channels, audit)``: the ``envelope.DragChannel`` tuple for
    ``run_tune_up(drag_channels=...)``, and a dict with ``rows`` (each with
    ``selected``/``reason``), ``blocking``, ``n_selected``, ``envelope_m_min``,
    ``w_p_GHz``, ``eta_peak``, ``t_g_ns``, ``total_error`` and the selection settings.
    """
    from snail_solver import sweep_common as SC
    from snail_solver.device_utils import build_coupler
    from snail_solver.envelope import DragChannel

    t_g = float(t_g_ns)
    # audit the UNCHIRPED tone: the chirp is what the tune-up is about to calibrate
    cpl, w_p_GHz, eta_peak = build_coupler(config, t_g, float(amp_scale),
                                           float(wp_offset_GHz),
                                           spec_abs_GHz=spec_abs_GHz,
                                           chirp_coeffs_GHz=[])
    rows = [dict(r) for r in
            interaction_channels(cpl, window_GHz=float(window_GHz), t_g_ns=t_g,
                                 min_g_MHz=float(min_g_MHz),
                                 has_spectator=(spec_abs_GHz is not None))]

    # --- the mode subharmonics (no_spectator: no invented beats against wspec = 0)
    sub_cfg = dict(config)
    if spec_abs_GHz is None:
        sub_cfg["no_spectator"] = True
    wa, wb = (float(x) for x in config["qubit_freqs_GHz"])
    ws = float(config["coupler_freq_GHz"])
    wspec = float(spec_abs_GHz) if spec_abs_GHz is not None else 0.0
    sub_chans = SC.collision_drag_channels(
        sub_cfg, wa, wb, ws, wspec, w_p_GHz, n=max(int(max_channels), 8),
        chirp_coeffs_GHz=chirp_coeffs_GHz, t_g=t_g, quotient_rule=quotient_rule)
    sub_by_beat = {(_audit_beat_key(c.beat_GHz), int(c.n_pump)): c
                   for c in sub_chans}

    def _ident(r: Dict[str, Any]) -> Tuple[int, int]:
        return (_audit_beat_key(r["beat_GHz"]), int(r["n_pump"] or 0))

    # --- mark up the audit -------------------------------------------------------
    req = tuple(str(c) for c in require)
    by_beat: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        r["beat_GHz"] = float(r["detuning_MHz"]) / 1e3
        r["n_photon"] = max(int(r.get("n_pump", 1) or 1), 1)
        # same P_exc as spectator_channels, for total_error_estimate / print_table
        _ratio = float(r.get("ratio", 0.0) or 0.0)
        r["P_exc"] = 1.0 if not np.isfinite(_ratio) else min(1.0, 2.0 * _ratio ** 2)
        key = _ident(r)
        r["is_subharmonic"] = key in sub_by_beat
        r["negligible"] = bool(_ratio < float(min_ratio))
        r["too_strong"] = bool(_ratio >= float(max_ratio))
        r["mandatory"] = bool(r["category"] != "target" and not r["negligible"]
                              and (r["category"] in req or r["is_subharmonic"]))
        r["selected"], r["reason"] = False, ""
        by_beat.setdefault(key, r)

    # a subharmonic below the g floor still gets a row, so it is visibly considered
    for key, ch in sub_by_beat.items():
        if key in by_beat:
            continue
        rows.append(dict(category="subharm", name="mode subharmonic (2 w_p)",
                         transition="--", process="--", g_MHz=0.0,
                         detuning_MHz=float(ch.beat_GHz) * 1e3,
                         beat_GHz=float(ch.beat_GHz), n_pump=int(ch.n_pump),
                         n_photon=int(ch.n_photon), ratio=0.0, suppression=1.0,
                         P_exc=0.0,
                         verdict="below the audit's g floor", victim_tag="--",
                         is_subharmonic=True, mandatory=False, negligible=True,
                         too_strong=False, selected=False, reason=""))
        by_beat[key] = rows[-1]

    def _fails(r: Dict[str, Any]) -> bool:
        """Not selectable: DRAG either cannot help, or would not converge."""
        return (str(r.get("verdict", "")).startswith("fails")
                or bool(r.get("too_strong")))

    blocking = [r for r in rows
                if r["category"] != "target"
                and "not perturbative" in str(r.get("verdict", ""))]

    # --- select ------------------------------------------------------------------
    def _rank(r: Dict[str, Any]) -> float:
        return -float(r.get("ratio", 0.0) or 0.0)      # strongest parasite first

    cand = [r for r in rows if r["category"] != "target"]
    mand = sorted((r for r in cand if r["mandatory"] and not _fails(r)), key=_rank)
    rest = sorted((r for r in cand if not r["mandatory"] and not _fails(r)), key=_rank)
    cap = max(int(max_channels), 0)

    chosen: List[Dict[str, Any]] = []
    taken: set = set()
    for r in mand + rest:
        ident = _ident(r)
        if ident in taken:
            # the identical substitution: composing it twice doubles the correction
            r["reason"] = "duplicate substitution (same beat and pump count)"
            continue
        if len(chosen) >= cap:
            r["reason"] = ("capped: mandatory but over --max-drag-channels"
                           if r["mandatory"] else "capped")
            continue
        r["selected"], r["reason"] = True, ("mandatory" if r["mandatory"]
                                            else "strongest remaining parasite")
        taken.add(ident)
        chosen.append(r)
    for r in cand:
        if not r["selected"] and not r["reason"]:
            _hard = str(r.get("verdict", "")).startswith("fails")
            r["reason"] = (
                f"left uncorrected: g/|det| >= {float(max_ratio):g}, the recursion "
                f"would not converge" if r.get("too_strong") and not _hard
                else f"not DRAG-correctable ({r['verdict']})" if _fails(r)
                else f"negligible (g/|det| < {float(min_ratio):g})"
                if r.get("negligible") else "not selected")

    built = tuple(sub_by_beat.get(_ident(r))
                  or DragChannel(float(r["beat_GHz"]), n_pump=int(r["n_pump"]),
                                 n_photon=int(r["n_photon"]),
                                 quotient_rule=bool(quotient_rule))
                  for r in chosen)
    # the shared safety filter: drop channels a chirp would sweep through zero
    channels = SC._drag_channels_filtered(config, built, chirp_coeffs_GHz, t_g)
    kept = {(_audit_beat_key(c.beat_GHz), int(c.n_pump)) for c in channels}
    for r in chosen:
        if _ident(r) not in kept:
            r["selected"], r["reason"] = False, "dropped: inside the DRAG skip window"

    rows.sort(key=lambda r: (r["category"] != "target", not r["selected"], _rank(r)))
    audit = {"rows": rows, "blocking": blocking, "n_selected": len(channels),
             "envelope_m_min": len(channels), "w_p_GHz": float(w_p_GHz),
             "eta_peak": float(eta_peak), "t_g_ns": t_g,
             # The target is resonant by design; including it would peg the budget at 1.
             "total_error": float(total_error_estimate(
                 [r for r in rows if r["category"] != "target"])),
             "max_channels": cap, "require": list(req),
             "min_ratio": float(min_ratio), "max_ratio": float(max_ratio),
             "uncorrected": [r for r in rows
                             if r.get("too_strong") and r["category"] != "target"],
             "n_mandatory": sum(1 for r in rows if r.get("mandatory")),
             "n_capped": sum(1 for r in rows
                             if str(r.get("reason", "")).startswith("capped"))}
    return channels, audit


CATEGORY_STYLE = {
    "target":    dict(colour="#111111", marker="*", label="target iSWAP  $a\\!\\leftrightarrow\\!b$"),
    "spectator": dict(colour="#d62728", marker="o", label="spectator excitation"),
    "coupler":   dict(colour="#2ca02c", marker="s", label="coupler heating"),
    "leakage":   dict(colour="#9467bd", marker="^", label=r"leakage to $|2\rangle$"),
    "other":     dict(colour="#7f7f7f", marker="d", label="other"),
}


def print_interaction_table(channels: Sequence[Dict[str, Any]], f_p_GHz: float,
                            t_g_ns: float, eta: float, top: int = 20) -> None:
    """Ranked interaction table with the DRAG verdict for each channel."""
    bw = 1e3 / t_g_ns
    print(f"\npump f_p = {f_p_GHz:.4f} GHz,  t_g = {t_g_ns:.1f} ns  ->  pulse bandwidth "
          f"1/t_g = {bw:.1f} MHz,  peak |eta| = {eta:.3f}")
    print(f"{len(channels)} near-resonant process(es); DRAG suppression factor ~ g/|det| "
          f"(smaller is better)\n")
    print(f"  {'interaction':<34} {'g(MHz)':>8} {'det(MHz)':>10} "
          f"{'f_p^res':>9} {'g/|det|':>8} {'DRAG':>8}  verdict")
    for c in channels[:int(top)]:
        rs = "  inf" if not np.isfinite(c["ratio"]) else f"{c['ratio']:8.3f}"
        fp = "   --   " if not np.isfinite(c["f_p_res_GHz"]) else f"{c['f_p_res_GHz']:9.4f}"
        db = ("  0.0" if c["suppression"] >= 1.0 else f"{c['suppression_dB']:5.1f}")
        print(f"  {c['name']:<34} {c['g_MHz']:8.3f} "
              f"{c['detuning_MHz']:+10.1f} {fp} {rs} {db:>6}dB  {c['verdict']}")
        print(f"  {'':<34} -> {c['process']}")
    if len(channels) > top:
        print(f"  ... {len(channels) - top} weaker process(es) not shown")


def _style_for(c: Dict[str, Any]) -> Dict[str, Any]:
    """Colour+marker for a channel: category sets the colour, and each SPECIFIC
    spectator gets its own marker so they are distinguishable on the chart."""
    st = dict(CATEGORY_STYLE[c["category"]])
    if c["category"] == "spectator" and c.get("spectator_id", 0) > 0:
        st["marker"] = SPEC_MARKERS[(c["spectator_id"] - 1) % len(SPEC_MARKERS)]
    return st


def plot_interaction_chart(config: Dict[str, Any], channels: Sequence[Dict[str, Any]],
                           out: str, *, f_p_GHz: float, t_g_ns: float, eta: float,
                           w_spec_GHz: Optional[float] = None,
                           mode_info: Optional[Dict[int, Dict[str, Any]]] = None,
                           max_key: int = 14, device_name: str = "") -> None:
    r"""Two views of the interaction landscape, with a numbered key.

    Channels are NUMBERED on both panels (target ``T``); names, strengths and DRAG
    verdicts live in the key panel on the right with the mode inventory.

    LEFT TOP -- each pump-tunable process as a stem at its resonant pump frequency
    :math:`f_p + \delta/n`, height :math:`g`; the shaded strip is :math:`1/t_g`.
    LEFT BOTTOM -- DRAG phase diagram, :math:`g` vs :math:`|\delta|`, with the
    perturbative boundary :math:`g=|\delta|`, the bandwidth line and 10/20/30 dB
    iso-suppression diagonals.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bw = 1e3 / float(t_g_ns)                                  # MHz
    # ---- assign numbers: T for the gate, 1..N for parasites by excitation amplitude
    para = [c for c in channels if c["category"] != "target"]
    para = sorted(para, key=lambda r: -(r["g_MHz"] / max(abs(r["detuning_MHz"]), 1e-6)))
    tgt = [c for c in channels if c["category"] == "target"]
    numbers: Dict[int, str] = {id(c): "T" for c in tgt}
    for k, c in enumerate(para, start=1):
        numbers[id(c)] = str(k)
    tunable = [c for c in channels if np.isfinite(c["f_p_res_GHz"])]

    fig = plt.figure(figsize=(14.6, 9.2), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.52], height_ratios=[1.0, 1.05],
                          wspace=0.04, hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    axk = fig.add_subplot(gs[:, 1]); axk.axis("off")

    def place_numbers(axis, items, min_sep_px: float = 15.0):
        """Draw number badges, each at the first candidate offset (display pixels) that
        clears every badge already placed -- channels are often near-degenerate."""
        fig_ = axis.figure
        fig_.canvas.draw()                     # transforms must be current
        cands = [(0, 9), (0, -17), (14, 5), (-14, 5), (14, -13), (-14, -13),
                 (0, 22), (0, -30), (24, 0), (-24, 0), (24, 14), (-24, 14)]
        placed: List[Tuple[float, float]] = []
        for c, x, y in items:
            px, py = axis.transData.transform((x, y))
            best = cands[0]
            for dx, dy in cands:
                q = (px + dx, py + dy)
                if all((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 > min_sep_px ** 2
                       for p in placed):
                    best = (dx, dy)
                    break
            placed.append((px + best[0], py + best[1]))
            axis.annotate(numbers[id(c)], xy=(x, y), xytext=best,
                          textcoords="offset points", ha="center", va="center",
                          fontsize=7.4, fontweight="bold", color="#222222", zorder=8,
                          bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                                    ec=_style_for(c)["colour"], lw=0.9, alpha=0.95))

    # ---------------- TOP: pump-frequency axis --------------------------------
    gmax = max([c["g_MHz"] for c in channels] + [1.0])
    gmin = min([c["g_MHz"] for c in channels] + [1.0])
    ax.axvspan(f_p_GHz - bw * 1e-3, f_p_GHz + bw * 1e-3, color="#f0c9c9", alpha=0.85,
               zorder=1, label=f"pulse bandwidth $1/t_g$ = {bw:.0f} MHz")
    ax.axvline(f_p_GHz, color="#111111", lw=2.0, zorder=4)
    for c in tunable:
        st = _style_for(c)
        ax.plot([c["f_p_res_GHz"]] * 2, [1e-6, c["g_MHz"]], color=st["colour"],
                lw=1.6, alpha=0.85, zorder=3)
        ax.plot([c["f_p_res_GHz"]], [c["g_MHz"]], marker=st["marker"],
                ms=12 if c["category"] == "target" else 7.5,
                color=st["colour"], mec="white", mew=0.8, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(max(1e-4, 0.3 * gmin), gmax * 12)
    ax.set_xlabel(r"pump frequency at which the process is resonant,"
                  r"  $f_p + \delta/n_{\rm pump}$  (GHz)")
    ax.set_ylabel("coupling $g$ (MHz)")
    ax.text(f_p_GHz, gmax * 6.5, f"  operating pump $f_p$ = {f_p_GHz:.4f} GHz",
            fontsize=8.5, ha="left", va="center")
    hs = []
    for k in ("target", "coupler", "leakage", "other"):
        if any(c["category"] == k for c in channels):
            hs.append(Line2D([], [], color=CATEGORY_STYLE[k]["colour"],
                             marker=CATEGORY_STYLE[k]["marker"], ls="none",
                             label=CATEGORY_STYLE[k]["label"]))
    seen_sp = {}
    for c in channels:
        if c["category"] == "spectator" and c.get("spectator_id", 0) > 0:
            seen_sp.setdefault(c["spectator_id"], (c["victim_tag"], c["victim_freq_GHz"]))
    for sid in sorted(seen_sp):
        tag, fq = seen_sp[sid]
        hs.append(Line2D([], [], color=CATEGORY_STYLE["spectator"]["colour"],
                         marker=SPEC_MARKERS[(sid - 1) % len(SPEC_MARKERS)], ls="none",
                         label=f"{tag} @ {fq:.3f} GHz"))
    ax.legend(handles=hs, fontsize=7.4, loc="lower left", ncol=3, framealpha=0.93)
    place_numbers(ax, sorted([(c, c["f_p_res_GHz"], c["g_MHz"]) for c in tunable],
                             key=lambda t: -t[2]))
    ttl = f"Interaction landscape — {device_name}" if device_name else "Interaction landscape"
    ax.set_title(f"{ttl}\n$t_g$={t_g_ns:.1f} ns,  peak $|\\eta|$={eta:.2f}",
                 fontsize=10.5, pad=8)
    ax.grid(alpha=0.22, which="both")

    # ---------------- BOTTOM: DRAG phase diagram ------------------------------
    if para:
        dets = np.array([max(abs(c["detuning_MHz"]), 1e-3) for c in para])
        gs_ = np.array([c["g_MHz"] for c in para])
        lo_d = max(1e-2, min(dets.min() / 4, bw * 0.5))
        hi_d = dets.max() * 4
        lo_g, hi_g = max(1e-4, gs_.min() / 4), gs_.max() * 4
        dd = np.logspace(np.log10(lo_d), np.log10(hi_d), 200)
        ax2.fill_between(dd, dd, hi_g, color="#b22222", alpha=0.13, zorder=0)
        ax2.axvspan(lo_d, min(bw, hi_d), color="#d95f02", alpha=0.13, zorder=0)
        ax2.plot(dd, dd, color="#b22222", lw=1.4,
                 label=r"$g=|\delta|$  (perturbation theory fails)")
        for r, lab in ((0.3, "10 dB"), (0.1, "20 dB"), (0.03, "30 dB")):
            ax2.plot(dd, r * dd, color="#1b7837", lw=0.9, ls="--", alpha=0.8)
            x_top = hi_g / r
            if x_top <= hi_d:
                xa, ya, va, ha = x_top, hi_g, "top", "right"
            else:
                xa, ya, va, ha = hi_d, r * hi_d, "bottom", "right"
            ax2.annotate(f"{lab}", xy=(xa, ya),
                         xytext=(-3, -3 if va == "top" else 3),
                         textcoords="offset points", ha=ha, va=va,
                         fontsize=7.0, color="#1b7837", clip_on=False)
        ax2.axvline(bw, color="#d95f02", lw=1.4,
                    label=f"pulse bandwidth $1/t_g$ = {bw:.0f} MHz")
        for c in para:
            st = _style_for(c)
            xv = max(abs(c["detuning_MHz"]), 1e-3)
            ax2.plot([xv], [c["g_MHz"]], marker=st["marker"], ms=9, color=st["colour"],
                     mec="white", mew=0.8, zorder=5)
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlim(lo_d, hi_d); ax2.set_ylim(lo_g, hi_g)
        ax2.text(0.018, 0.965, "DRAG cannot help\n($g\\geq|\\delta|$: reallocate)",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=8,
                 color="#8b1a1a",
                 bbox=dict(boxstyle="round", fc="white", ec="#b22222", alpha=0.9))
        ax2.text(0.982, 0.055, "DRAG effective\n(suppression $\\sim g/|\\delta|$)",
                 transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
                 color="#1b7837",
                 bbox=dict(boxstyle="round", fc="white", ec="#1b7837", alpha=0.9))
        ax2.set_xlabel(r"Channel Detuning $|\delta|$ (MHz)")
        ax2.set_ylabel("Interaction Coefficient (MHz)")
        ax2.set_title("Which parasites DRAG can suppress", fontsize=10)
        ax2.legend(fontsize=7.4, loc="lower left", framealpha=0.93)
        ax2.grid(alpha=0.22, which="both")
        place_numbers(ax2, sorted([(c, max(abs(c["detuning_MHz"]), 1e-3), c["g_MHz"])
                                   for c in para], key=lambda t: -t[2]))
    else:
        ax2.text(0.5, 0.5, "no parasitic channel inside the window",
                 transform=ax2.transAxes, ha="center", va="center")
        ax2.set_xticks([]); ax2.set_yticks([])

    # ---------------- RIGHT: mode inventory + numbered interaction key ---------
    y = 0.995
    axk.text(0.0, y, "MODE INVENTORY", fontsize=8.6, fontweight="bold",
             family="monospace", va="top")
    y -= 0.030
    if mode_info:
        for m in sorted(mode_info):
            mi = mode_info[m]
            axk.text(0.0, y, f"  {mi['tag']:<5}{mi['freq']:8.3f} GHz   {mi['role']}",
                     fontsize=7.6, family="monospace", va="top")
            y -= 0.024
    axk.text(0.0, y, f"  {'pump':<5}{f_p_GHz:8.3f} GHz   "
                     f"2w_p = {2 * f_p_GHz:.3f} GHz", fontsize=7.6,
             family="monospace", va="top")
    y -= 0.024
    axk.text(0.0, y, f"  {'1/t_g':<5}{bw:8.1f} MHz   pulse bandwidth",
             fontsize=7.6, family="monospace", va="top")

    y -= 0.045
    axk.text(0.0, y, "INTERACTIONS", fontsize=8.6, fontweight="bold",
             family="monospace", va="top")
    y -= 0.028
    axk.text(0.0, y, f"  {'#':<3}{'process':<30}{'g(MHz)':>8}{'det(MHz)':>10}{'DRAG':>8}",
             fontsize=7.3, family="monospace", va="top", color="#444444")
    y -= 0.023
    axk.plot([0.0, 1.0], [y + 0.010, y + 0.010], color="#bbbbbb", lw=0.8,
             transform=axk.transAxes, clip_on=False)
    for c in (tgt + para)[:int(max_key)]:
        nm = c["name"] if len(c["name"]) <= 30 else c["name"][:29] + "…"
        db = ("  --  " if c["suppression"] >= 1.0 else f"{c['suppression_dB']:5.1f}dB")
        axk.text(0.0, y,
                 f"  {numbers[id(c)]:<3}{nm:<30}{c['g_MHz']:8.2f}"
                 f"{c['detuning_MHz']:+10.1f}{db:>8}",
                 fontsize=7.3, family="monospace", va="top",
                 color=_style_for(c)["colour"])
        y -= 0.0235
    if len(channels) > max_key:
        axk.text(0.0, y, f"  ... {len(channels) - max_key} weaker process(es)",
                 fontsize=7.0, family="monospace", va="top", color="#777777")
        y -= 0.0235
    y -= 0.020
    axk.text(0.0, y, "DRAG column = suppression of that channel,\n"
                     "  ~ g/|det| (blank = cannot help).",
             fontsize=7.0, family="monospace", va="top", color="#555555")
    axk.set_xlim(0, 1); axk.set_ylim(0, 1)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def total_error_estimate(channels: Sequence[Dict[str, Any]]) -> float:
    """Summed excitation estimate over channels, capped at 1 (a rough error budget)."""
    return float(min(1.0, sum(c["P_exc"] for c in channels)))


def print_channel_audit(audit: Dict[str, Any], top: int = 14) -> None:
    """Print a :func:`select_drag_channels` audit before any solve: every channel the
    recursion will NOT touch appears with its reason, not as a silent absence."""
    rows = audit.get("rows", [])
    print(f"  DRAG channel audit: w_p={audit['w_p_GHz']:.4f} GHz  "
          f"t_g={audit['t_g_ns']:.1f} ns  peak|eta|={audit['eta_peak']:.3f}  "
          f"bandwidth 1/t_g={1e3 / max(audit['t_g_ns'], 1e-9):.1f} MHz")
    print(f"  selected {audit['n_selected']}/{audit['max_channels']} channels; "
          f"needs sine_power m >= {audit['envelope_m_min']}; "
          f"parasitic budget ~{audit['total_error']:.2e}")
    print(f"    {'':1s} {'category':9s} {'g(MHz)':>8s} {'det(MHz)':>10s} {'g/|det|':>8s} "
          f"{'k':>2s} {'sub':>3s}  {'verdict':30s}  reason")
    for r in rows[:int(top)]:
        ratio = float(r.get("ratio", 0.0) or 0.0)
        print(f"    {'*' if r.get('selected') else ' '} {r.get('category', '?'):9s} "
              f"{r.get('g_MHz', 0.0):8.3f} {r.get('detuning_MHz', 0.0):10.2f} "
              f"{ratio:8.3f} {int(r.get('n_pump', 0) or 0):2d} "
              f"{'yes' if r.get('is_subharmonic') else '  -':>3s}  "
              f"{str(r.get('verdict', ''))[:30]:30s}  {str(r.get('reason', ''))[:38]}")
    if len(rows) > int(top):
        print(f"    ... {len(rows) - int(top)} weaker channel(s) not shown")
    for u in audit.get("uncorrected", []):
        print(f"    ~~ LEFT UNCORRECTED: {u['name']} g={u['g_MHz']:.2f} MHz "
              f"det={u['detuning_MHz']:.2f} MHz  g/|det|={u.get('ratio', 0):.3f} "
              f">= {audit.get('max_ratio', 0.3):g} -- recursion would not converge")
    for b in audit.get("blocking", []):
        print(f"    !! NOT PERTURBATIVE: {b['name']} g={b['g_MHz']:.2f} MHz "
              f"det={b['detuning_MHz']:.2f} MHz -- frequency allocation, not pulse shaping")


def scan_band(config: Dict[str, Any], t_g_ns: float, *, n_points: int = 161,
              window_GHz: float = 1.0, pad_GHz: float = 0.0,
              coupler_levels: Optional[int] = None,
              verbose: bool = True) -> Dict[str, Any]:
    """Sweep the spectator across [w_a, w_b] (+/- pad) and record its parasitic load.

    Returns w_spec_GHz, worst_ratio, total_P_exc, worst_g_MHz, worst_det_MHz,
    n_nonperturbative, plus t_g_ns / eta / band_GHz.
    """
    wa, wb = sorted(float(f) for f in config["qubit_freqs_GHz"])
    lo, hi = wa - float(pad_GHz), wb + float(pad_GHz)
    grid = np.linspace(lo, hi, int(n_points))
    worst_ratio = np.zeros(grid.size)
    total = np.zeros(grid.size)
    worst_g = np.zeros(grid.size)
    worst_det = np.zeros(grid.size)
    n_bad = np.zeros(grid.size, dtype=int)
    eta_nom = np.nan
    for k, w in enumerate(grid):
        cpl, eta_nom = build_with_spectator(config, float(w), t_g_ns,
                                            coupler_levels=coupler_levels)
        ch = spectator_channels(cpl, window_GHz=window_GHz)
        if ch:
            worst_ratio[k] = ch[0]["ratio"]
            worst_g[k] = ch[0]["g_MHz"]
            worst_det[k] = ch[0]["detuning_MHz"]
            n_bad[k] = sum(1 for c in ch if not c["perturbative"])
        total[k] = total_error_estimate(ch)
        if verbose and (k + 1) % max(1, grid.size // 8) == 0:
            print(f"  scanned {k + 1}/{grid.size}  w_spec={w:.3f} GHz  "
                  f"worst ratio={worst_ratio[k]:.2f}")
    return dict(w_spec_GHz=grid, worst_ratio=worst_ratio, total_P_exc=total,
                worst_g_MHz=worst_g, worst_det_MHz=worst_det,
                n_nonperturbative=n_bad, t_g_ns=float(t_g_ns), eta=float(eta_nom),
                band_GHz=(wa, wb))


def quiet_windows(scan: Dict[str, Any], threshold: float = 1e-3
                  ) -> List[Tuple[float, float]]:
    """Contiguous spectator placements whose total excitation estimate is below
    ``threshold`` -- the intervals where a spectator can actually live."""
    w, tot = scan["w_spec_GHz"], scan["total_P_exc"]
    ok = tot < float(threshold)
    out, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = w[i]
        elif not good and start is not None:
            out.append((float(start), float(w[i - 1])))
            start = None
    if start is not None:
        out.append((float(start), float(w[-1])))
    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def print_table(channels: Sequence[Dict[str, Any]], w_spec_GHz: float,
                eta: float, top: int = 12) -> None:
    """Ranked channel table for one spectator placement."""
    print(f"\nspectator at {w_spec_GHz:.4f} GHz, peak |eta| = {eta:.3f}: "
          f"{len(channels)} channel(s) found")
    if not channels:
        print("  (no spectator-exciting channel inside the window)")
        return
    print(f"  {'g (MHz)':>9} {'detuning':>10} {'g/|det|':>8} {'P_exc':>9}  transition")
    for c in channels[:int(top)]:
        if c.get("resonant"):
            flag, ratio_s = "   <-- ON RESONANCE", "     inf"
        elif not c["perturbative"]:
            flag, ratio_s = "   <-- NOT perturbative", f"{c['ratio']:8.3f}"
        else:
            flag, ratio_s = "", f"{c['ratio']:8.3f}"
        print(f"  {c['g_MHz']:9.3f} {c['detuning_MHz']:+10.2f} {ratio_s} "
              f"{c['P_exc']:9.2e}  {c['transition']}{flag}")
    if len(channels) > top:
        print(f"  ... {len(channels) - top} weaker channel(s) not shown")
    print(f"  total excitation estimate: {total_error_estimate(channels):.3e}")


def plot_chart(config: Dict[str, Any], scan: Optional[Dict[str, Any]],
               channels: Optional[Sequence[Dict[str, Any]]], out: str,
               *, w_spec_GHz: Optional[float] = None,
               threshold: float = 1e-3, device_name: str = "") -> None:
    """Frequency chart: mode layout + strength-weighted channels, over a band scan.

    Top: spectral layout as in ``plot_allocation.py`` plus the audited placement's
    measured channels. Bottom (if `scan`): total excitation vs spectator frequency,
    shaded where non-perturbative, with the quiet windows marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wa, wb = sorted(float(f) for f in config["qubit_freqs_GHz"])
    ws = float(config["coupler_freq_GHz"])
    w_p = abs(wb - wa)

    # curated centres, reused from the existing allocation tool when importable
    try:
        from snail_solver.plot_allocation import allocation_frequencies
        alloc = allocation_frequencies(config)
        families = alloc["families"]
    except Exception:                      # keep the tool standalone if it moves
        families = [
            {"key": "direct", "color": "#555555", "label": "direct",
             "centers": [wa, ws, wb]},
            {"key": "subharm", "color": "#e9a000", "label": "subharmonic",
             "centers": [2.0 * w_p]}]

    n_rows = 2 if scan is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.0, 4.2 * n_rows), dpi=200,
                             sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 1.15][:n_rows]))
    ax = axes[0] if n_rows > 1 else axes

    # ---- top: spectral layout -------------------------------------------
    ax.set_ylim(0.0, 1.0)
    ax.axhspan(0.0, 1.0, xmin=0, xmax=0, color="none")          # keep limits
    ax.axvspan(wa, wb, color="#dfe9f5", alpha=0.7, zorder=0)
    ax.text(0.5 * (wa + wb), 0.055,
            r"physical spectator band  $[\omega_a,\omega_b]$",
            ha="center", va="bottom", fontsize=8.5, color="#2c4a72")
    # mode lines, labelled INSIDE the axes so nothing collides with the title
    for w, lab, col in ((wa, r"$\omega_a$", "#d62728"),
                        (wb, r"$\omega_b$", "#1f77b4"),
                        (ws, r"$\omega_s$ (SNAIL)", "#2ca02c")):
        ax.axvline(w, color=col, lw=2.2, zorder=3)
        ax.text(w, 0.985, f" {lab} {w:.3f} ", ha="center", va="top", fontsize=8.5,
                color=col, rotation=90, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # the pump's 2nd harmonic is the channel that bites hardest here -- label it
    if wa - 0.05 <= 2.0 * w_p <= wb + 0.05:
        ax.axvline(2.0 * w_p, color="#e9a000", lw=1.8, ls="--", zorder=3)
        ax.text(2.0 * w_p, 0.985, f" $2\\omega_p$ {2*w_p:.3f} ", ha="center", va="top",
                fontsize=8.5, color="#b37700", rotation=90, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for fam in families:                   # curated collision centres, in band
        for c in fam["centers"]:
            if wa - 0.05 <= c <= wb + 0.05 and abs(c - 2.0 * w_p) > 1e-9:
                ax.axvline(c, color=fam["color"], lw=1.0, ls=":", alpha=0.75, zorder=2)

    # measured channel strengths at the audited placement, as a readable table
    if channels:
        base = float(w_spec_GHz if w_spec_GHz is not None else 0.5 * (wa + wb))
        ax.axvline(base, color="#000000", lw=1.6, ls="--", zorder=4)
        r0 = float(channels[0]["ratio"])
        head = (r"audited $\omega_{\rm spec}$ = " + f"{base:.3f} GHz    "
                + ("worst channel: ON RESONANCE" if not np.isfinite(r0)
                   else f"worst $g/|\\delta|$ = {r0:.3f}"))
        lines = [head, ""]
        lines.append(f"{'g (MHz)':>8}  {'det (MHz)':>10}  {'g/|det|':>8}   transition")
        for c in channels[:5]:
            rs = "inf" if not np.isfinite(c["ratio"]) else f"{c['ratio']:.3f}"
            lines.append(f"{c['g_MHz']:8.3f}  {c['detuning_MHz']:+10.1f}  {rs:>8}   "
                         f"{c['transition']}")
        tot = total_error_estimate(channels)
        lines.append("")
        lines.append(f"total excitation estimate: {tot:.2e}")
        side = "left" if base > 0.5 * (wa + wb) else "right"
        xa = 0.015 if side == "left" else 0.985
        ax.text(xa, 0.62, "\n".join(lines), transform=ax.transAxes,
                ha=side, va="top", fontsize=7.6, family="monospace", zorder=7,
                bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.94))
    ax.set_yticks([])
    ax.set_ylabel("spectral layout")
    ttl = f"Spectator audit — {device_name}" if device_name else "Spectator audit"
    sub = (f"$\\omega_p$={w_p:.3f} GHz,  $2\\omega_p$={2*w_p:.3f} GHz"
           + (f",  $t_g$={scan['t_g_ns']:.1f} ns,  peak $|\\eta|$={scan['eta']:.2f}"
              if scan else ""))
    ax.set_title(f"{ttl}\n{sub}", fontsize=10.5, pad=10)
    handles = [Line2D([], [], color="#e9a000", lw=1.8, ls="--",
                      label=r"$2\omega_p$ (pump 2nd harmonic)")]
    handles += [Line2D([], [], color=f["color"], lw=1.0, ls=":", label=f["label"])
                for f in families]
    handles += [Line2D([], [], color="k", lw=1.6, ls="--", label="audited placement")]
    ax.legend(handles=handles, fontsize=7.2, loc="lower left", ncol=2, framealpha=0.92)

    # ---- bottom: band scan ----------------------------------------------
    if scan is not None:
        ax2 = axes[1]
        w, tot, bad = scan["w_spec_GHz"], scan["total_P_exc"], scan["n_nonperturbative"]
        ax2.semilogy(w, np.maximum(tot, 1e-12), color="#1f3b73", lw=1.8,
                     label="total excitation estimate")
        ax2.axhline(threshold, color="#888888", ls="--", lw=1.0,
                    label=f"threshold {threshold:g}")
        # shade non-perturbative placements
        inbad = bad > 0
        if inbad.any():
            ax2.fill_between(w, 1e-12, 1.0, where=inbad, color="#b22222", alpha=0.16,
                             step="mid", label="a channel is non-perturbative")
        qw = quiet_windows(scan, threshold)
        for lo_w, hi_w in qw:
            if hi_w > lo_w:
                ax2.axvspan(lo_w, hi_w, color="#2ca02c", alpha=0.18, zorder=0)
        if qw:
            ax2.text(0.005, 0.04,
                     "quiet windows (GHz): "
                     + ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in qw[:4])
                     + (" ..." if len(qw) > 4 else ""),
                     transform=ax2.transAxes, fontsize=8, color="#1a6b1a",
                     bbox=dict(boxstyle="round", fc="white", ec="#2ca02c", alpha=0.85))
        for wv, col in ((wa, "#d62728"), (wb, "#1f77b4"), (ws, "#2ca02c")):
            ax2.axvline(wv, color=col, lw=1.6, alpha=0.8)
        if w_spec_GHz is not None:
            ax2.axvline(float(w_spec_GHz), color="k", lw=1.4, ls="--")
        finite = tot[np.isfinite(tot) & (tot > 0)]
        floor = max(1e-12, (finite.min() / 5.0) if finite.size else 1e-9)
        ax2.set_ylim(min(floor, 0.5 * float(threshold)), 1.6)
        ax2.set_ylabel("estimated spectator excitation")
        ax2.set_xlabel(r"spectator frequency $\omega_{\rm spec}$ (GHz)")
        ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax2.grid(alpha=0.25, which="both")
    else:
        ax.set_xlabel(r"frequency (GHz)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def save_npz(path: str, scan: Optional[Dict[str, Any]],
             channels: Optional[Sequence[Dict[str, Any]]],
             meta: Dict[str, Any],
             interactions: Optional[Sequence[Dict[str, Any]]] = None) -> None:
    """Persist the scan arrays and the audited placement's channel table."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {f"meta_{k}": v for k, v in meta.items()}
    if scan is not None:
        for k in ("w_spec_GHz", "worst_ratio", "total_P_exc", "worst_g_MHz",
                  "worst_det_MHz", "n_nonperturbative"):
            payload[k] = scan[k]
        payload["t_g_ns"] = scan["t_g_ns"]
        payload["eta"] = scan["eta"]
    if channels:
        for key, dtype in (("g_MHz", float), ("detuning_MHz", float),
                           ("ratio", float), ("P_exc", float), ("n_pump", int)):
            payload[f"ch_{key}"] = np.array([c[key] for c in channels], dtype=dtype)
        payload["ch_transition"] = np.array([c["transition"] for c in channels])
    if interactions:
        # dtype None: let numpy infer (the string columns)
        for key, dtype in (("g_MHz", float), ("detuning_MHz", float), ("ratio", float),
                           ("f_p_res_GHz", float), ("adiabaticity", float),
                           ("suppression", float), ("suppression_dB", float),
                           ("n_pump", int), ("category", None), ("victim_tag", None),
                           ("victim_freq_GHz", float), ("spectator_id", int),
                           ("process", None), ("name", None), ("transition", None),
                           ("verdict", None)):
            payload[f"int_{key}"] = np.array([c[key] for c in interactions], dtype=dtype)
    np.savez_compressed(path, **payload)
    print("saved", path)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.spectator_audit", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON")
    ap.add_argument("--spec", default=None,
                    help="ABSOLUTE spectator frequency in GHz, or a comma-separated list "
                         "for several spectators (e.g. 4.6,5.2). Each becomes its own "
                         "mode, labelled sp1, sp2, ... on the chart")
    ap.add_argument("--spec-detuning", default=None,
                    help="spectator(s) by the sweep-convention detuning "
                         "Delta = w_b - w_spec (GHz); comma-separated for several")
    ap.add_argument("--spec-levels", type=int, default=None,
                    help="spectator truncation (default 3 for one spectator, 2 for "
                         "several -- enough to see |0>->|1> excitation)")
    ap.add_argument("--eta", type=float, default=None,
                    help="operating point as peak |eta| (sets t_g); default 1.0")
    ap.add_argument("--t-g-ns", type=float, default=None,
                    help="explicit gate duration (ns); overrides --eta")
    ap.add_argument("--scan-points", type=int, default=161,
                    help="band-scan resolution (0 disables the scan)")
    ap.add_argument("--window-GHz", type=float, default=1.0,
                    help="only report channels detuned by less than this")
    ap.add_argument("--pad-GHz", type=float, default=0.0,
                    help="extend the scan this far beyond [w_a, w_b]")
    ap.add_argument("--threshold", type=float, default=1e-3,
                    help="quiet-window cut on the total excitation estimate")
    ap.add_argument("--coupler-levels", type=int, default=None,
                    help="override coupler truncation (raise at strong drive)")
    ap.add_argument("--top", type=int, default=12, help="table rows to print")
    ap.add_argument("--out", default="figs/spectator_audit.png")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--chart", choices=["interactions", "placement", "both"],
                    default="interactions",
                    help="'interactions' (default): the gate and every parasitic process "
                         "on a pump-frequency axis, weighted by coupling, plus the DRAG "
                         "phase diagram. 'placement': total excitation vs where the "
                         "SPECTATOR sits (the band scan). 'both' writes both figures.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    try:                                   # use the pipeline's resolver when present
        from snail_solver.paths import resolve_device
        device_path = resolve_device(args.device)
    except Exception:
        device_path = args.device
    with open(device_path) as f:
        config = json.load(f)

    t_g = (float(args.t_g_ns) if args.t_g_ns
           else t_g_for_eta(config, args.eta if args.eta else 1.0))
    wa, wb = sorted(float(x) for x in config["qubit_freqs_GHz"])
    print(f"{os.path.basename(device_path)}: w_a={wa} w_b={wb} "
          f"w_s={config['coupler_freq_GHz']} GHz, w_p={abs(wb - wa):.3f} GHz")
    _c, eta_nom = build_with_spectator(config, 0.5 * (wa + wb), t_g,
                                       coupler_levels=args.coupler_levels)
    print(f"t_g = {t_g:.2f} ns  ->  peak |eta| = {eta_nom:.3f}")

    scan = None
    need_scan = (args.chart in ("placement", "both")) and int(args.scan_points) > 0
    if need_scan:
        print(f"scanning the band [{wa - args.pad_GHz:.3f}, {wb + args.pad_GHz:.3f}] "
              f"GHz at {args.scan_points} points ...")
        scan = scan_band(config, t_g, n_points=int(args.scan_points),
                         window_GHz=args.window_GHz, pad_GHz=args.pad_GHz,
                         coupler_levels=args.coupler_levels)
        qw = quiet_windows(scan, args.threshold)
        print(f"\nquiet windows (total excitation < {args.threshold:g}):")
        if qw:
            for lo_w, hi_w in qw:
                print(f"   [{lo_w:.4f}, {hi_w:.4f}] GHz   "
                      f"(width {1e3 * (hi_w - lo_w):.0f} MHz)")
        else:
            print("   NONE -- every placement in the band exceeds the threshold.")
        frac = float(np.mean(scan["n_nonperturbative"] > 0))
        print(f"non-perturbative (g >= detuning) at {100 * frac:.0f}% of placements")

    # which placement(s) to audit in detail
    def _parse_list(txt):
        return [float(x) for x in str(txt).replace(" ", "").split(",") if x]

    w_specs: List[float] = []
    if args.spec is not None:
        w_specs = _parse_list(args.spec)
    elif args.spec_detuning is not None:
        wb_ref = float(config["qubit_freqs_GHz"][1])
        w_specs = [wb_ref - d for d in _parse_list(args.spec_detuning)]
    w_spec = w_specs[0] if w_specs else None
    if w_spec is None and scan is not None:
        w_spec = float(scan["w_spec_GHz"][int(np.argmax(scan["total_P_exc"]))])
        w_specs = [w_spec]
        print(f"\n(no --spec given; auditing the WORST placement found)")
    channels = None
    if w_spec is not None:
        cpl, eta_nom = build_with_spectator(config, float(w_spec), t_g,
                                            coupler_levels=args.coupler_levels)
        channels = spectator_channels(cpl, window_GHz=args.window_GHz)
        print_table(channels, float(w_spec), eta_nom, top=args.top)

    # ---- interaction landscape (the gate + its parasites on a frequency axis) ----
    inter = None
    if args.chart in ("interactions", "both"):
        w_list = w_specs if w_specs else [0.5 * (wa + wb)]
        cpl_i, eta_i, sp_idx = build_with_spectators(
            config, w_list, t_g, coupler_levels=args.coupler_levels,
            spec_levels=args.spec_levels)
        info = mode_tags(cpl_i, sp_idx)
        print("\nmode inventory:")
        for m in sorted(info):
            print(f"   {info[m]['tag']:<5} {info[m]['freq']:8.4f} GHz   {info[m]['role']}")
        inter = interaction_channels(cpl_i, window_GHz=args.window_GHz, t_g_ns=t_g,
                                     spec_indices=sp_idx)
        print_interaction_table(inter, float(cpl_i._pump_tones[0].w_p_GHz), t_g, eta_i,
                                top=args.top)
        if not args.no_plot:
            out_i = (args.out if args.chart == "interactions"
                     else args.out.rsplit(".", 1)[0] + "_interactions.png")
            plot_interaction_chart(config, inter, out_i,
                                   f_p_GHz=float(cpl_i._pump_tones[0].w_p_GHz),
                                   t_g_ns=t_g, eta=eta_i,
                                   w_spec_GHz=(w_list[0] if len(w_list) == 1 else None),
                                   mode_info=info,
                                   device_name=os.path.basename(device_path))

    if args.save_npz:
        save_npz(args.save_npz, scan, channels,
                 dict(device=os.path.basename(device_path), t_g_ns=t_g,
                      eta=eta_nom, w_spec_GHz=(np.nan if w_spec is None else w_spec)),
                 interactions=inter)
    if not args.no_plot and args.chart in ("placement", "both"):
        out_p = (args.out if args.chart == "placement"
                 else args.out.rsplit(".", 1)[0] + "_placement.png")
        plot_chart(config, scan, channels, out_p, w_spec_GHz=w_spec,
                   threshold=args.threshold,
                   device_name=os.path.basename(device_path))


if __name__ == "__main__":
    main()