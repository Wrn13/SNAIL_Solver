"""The swept-channel drop judges the beat the gate plays; replayed tables still plot."""
import os

import numpy as np
import pytest

from snail_solver.envelope import Chirp, DragChannel, PumpTone, SinePowerRamp
from snail_solver.tune_up import TWO_PI, _played_floors_GHz

# delta = +5 MHz, eta* = 1.3, pass C: the ridge chirp (zero-mean, c_0 pinned) that
# swept the A subharmonic's -10 MHz beat through zero while the projection's own
# floors (which carry the pulse-mean shift) reported 9.9 MHz.
CHIRP = [0.0, 0.0, -0.017064439530108045, 0.0, 0.011903997009418141, 0.0,
         -0.0030163923919139003, 0.0, -0.002662]
T_G = 106.838
CHANNELS = [DragChannel(beat_GHz=-0.010, n_pump=2, n_photon=2),
            DragChannel(beat_GHz=0.130, n_pump=2, n_photon=2),
            DragChannel(beat_GHz=0.120, n_pump=1, n_photon=1)]
REPLAYED = os.path.join(os.path.dirname(__file__), os.pardir, "results",
                        "drag_curve_5MHz_2026-09-22", "passA_nodrag", "columns",
                        "col_dm0p005_eta1p3_rabi.npz")


def test_played_floor_catches_the_swept_beat():
    floors = _played_floors_GHz(CHANNELS, CHIRP, T_G)
    assert floors[0] < 0.5e-3                      # inside any skip window
    assert floors[1] > 0.1 and floors[2] > 0.1     # the far beats are clear


def test_played_floor_matches_the_gate_builders_guard():
    tone = PumpTone(w_p_GHz=1.755, envelope=SinePowerRamp(amp=1.0, t_g=T_G, m=3,
                                                          t_rise=0.5 * T_G),
                    is_eta=True, chirp=Chirp(CHIRP, T_G), drag_channels=CHANNELS)
    guard = np.array(tone.drag_detuning_floors(n=4001)) / TWO_PI
    ours = np.array(_played_floors_GHz(CHANNELS, CHIRP, T_G, n=4001))
    assert np.allclose(ours, guard, atol=2e-4)


def test_unchirped_and_static_channels():
    assert _played_floors_GHz(CHANNELS, [], T_G) == pytest.approx([0.010, 0.130, 0.120])
    static = [DragChannel(beat_GHz=-0.002, n_pump=0, n_photon=1)]
    assert _played_floors_GHz(static, CHIRP, T_G) == pytest.approx([0.002])


@pytest.mark.skipif(not os.path.exists(REPLAYED), reason="stored Rabi table absent")
def test_replayed_table_plots(tmp_path):
    from snail_solver.ridge_refit import load_npz_rabi
    from snail_solver.tune_up import plot_rabi_table
    table = load_npz_rabi(REPLAYED)
    assert "delta_MHz" not in table                # stripped on load
    out = plot_rabi_table(table, str(tmp_path / "chev.png"))
    assert os.path.getsize(out) > 0


# -- DRAG beats follow the pump carrier --------------------------------------------
def test_carrier_shift_moves_only_pump_carrying_beats():
    from snail_solver.device_utils import carrier_shifted
    static = DragChannel(beat_GHz=0.050, n_pump=0, n_photon=1)
    beat, chans = carrier_shifted(-0.010, 2, CHANNELS + [static], 0.008)
    assert beat == pytest.approx(-0.026)
    assert [c.beat_GHz for c in chans] == pytest.approx([-0.026, 0.114, 0.112, 0.050])
    assert chans[0].n_pump == 2 and chans[0].n_photon == 2     # only the beat moves
    assert carrier_shifted(None, 1, None, 0.008) == (None, None)


def test_built_gate_beats_include_the_carrier():
    """build_coupler's played beat = what the swept-channel check judges."""
    from snail_solver.device_utils import build_coupler, load_device
    from snail_solver.subharmonic_convergence import config_at_wp
    dev = os.path.join(os.path.dirname(__file__), os.pardir, "devices",
                       "6Gate4.7SNAIL.json")
    if not os.path.exists(dev):
        pytest.skip("device file absent")
    cfg = config_at_wp(load_device(dev), 1.755, branch="above", levels=4)
    off = 0.008285
    cpl, w_p, _ = build_coupler(cfg, T_G, 1.0, off, None, None,
                                chirp_coeffs_GHz=CHIRP, drag_channels=CHANNELS)
    got = np.array(cpl._pump_tones[0].drag_detuning_floors(n=4001)) / TWO_PI
    want = np.array(_played_floors_GHz(CHANNELS, CHIRP, T_G, n=4001,
                                       carrier_offset_GHz=off))
    assert np.allclose(got, want, atol=2e-4)
    # without the carrier the -10 MHz beat would read 0.01 MHz; with it, ~4.9 MHz
    assert want[0] > 4e-3


# -- step 3's chevron sweeps the carrier past a beat ---------------------------------
def test_chevron_probe_switches_off_a_swept_channel_at_that_point_only():
    """delta = -75 MHz: a probe offset of -16.3 MHz puts the A |1>->|2> beat (-30 MHz,
    k=2) at ~0. The probe drops that channel there instead of raising; a probe
    offset that leaves every beat clear keeps all of them."""
    from snail_solver.device_utils import load_device
    from snail_solver.find_stark_resonance import build_chevron_coupler
    from snail_solver.subharmonic_convergence import config_at_wp
    dev = os.path.join(os.path.dirname(__file__), os.pardir, "devices",
                       "6Gate4.7SNAIL.json")
    if not os.path.exists(dev):
        pytest.skip("device file absent")
    cfg = config_at_wp(load_device(dev), 1.675, branch="above", levels=4)
    chans = [DragChannel(beat_GHz=0.150, n_pump=2, n_photon=2),
             DragChannel(beat_GHz=-0.030, n_pump=2, n_photon=2),
             DragChannel(beat_GHz=0.120, n_pump=1, n_photon=1)]
    kw = dict(shape="gate", t_g_ns=T_G, drag_channels=chans, chirp_coeffs_GHz=[])
    swept, _ = build_chevron_coupler(cfg, 1.3, -0.0150, T_G, **kw)
    kept = [c.beat_GHz for c in swept._pump_tones[0].drag_channels_resolved()]
    assert len(kept) == 2 and all(abs(b) > 0.1 for b in kept)
    clear, _ = build_chevron_coupler(cfg, 1.3, 0.0, T_G, **kw)
    assert len(clear._pump_tones[0].drag_channels_resolved()) == 3
