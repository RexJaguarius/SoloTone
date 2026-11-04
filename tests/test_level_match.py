"""
Level matching + master volume: the offline loudness measurement is
deterministic, tracks gain changes, and the app's Match level button and
Master Volume slider move the right (global vs per-profile) values.

Run directly: python tests/test_level_match.py
"""

import sys
import json
import math
import pathlib
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import solotone as st


def _bare_profile(**kw):
    d = {'solotone_profile': 1, 'eq_on': False, 'gate_threshold': 1e-6,
         'in_gain': 1.0, 'out_gain': 1.0, 'pedals': {}}
    d.update(kw)
    return d


def test_measurement_repeats_and_tracks_input_gain():
    a = st.measure_profile_level(_bare_profile())
    b = st.measure_profile_level(_bare_profile())
    assert a == b                                    # fixed-seed test signal
    hot = st.measure_profile_level(_bare_profile(in_gain=2.0))
    assert abs((hot[0] - a[0]) - 6.02) < 0.2         # no NAM: pure linear gain


def test_suggestion_hits_target_and_respects_limits():
    g, note = st.suggest_out_gain_db(-30.0, -15.0)
    assert abs(g - 12.0) < 1e-9 and note == ''
    g, note = st.suggest_out_gain_db(-30.0, -5.0)    # peaks would clip
    assert g < 12.0 and note == 'limited by peaks'
    g, note = st.suggest_out_gain_db(-90.0, -80.0)   # beyond the slider
    assert g == st.OUT_GAIN_RANGE_DB[1] and note


def test_match_button_and_master_volume_in_app():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        st.SESSION_FILE = tmp_path / 'test_session.json'
        st.MIDI_MAP_FILE = tmp_path / 'test_midi.json'
        app = st.App()
        try:
            app.update()
            app._set_master(-12.0)
            assert abs(app.amp.master_gain - 10 ** (-12 / 20)) < 1e-9
            before = app.amp.out_gain
            app._match_level()
            deadline = time.time() + 60
            while str(app.match_btn.cget('state')) == 'disabled' and time.time() < deadline:
                app.update(); time.sleep(0.05)
            assert str(app.match_btn.cget('state')) == 'normal'
            assert app.amp.out_gain != before          # Output Gain moved
            assert abs(app.amp.master_gain - 10 ** (-12 / 20)) < 1e-9   # Master untouched
            assert 'master_gain' not in app._gather_profile_dict('x')   # never in a profile
        finally:
            app.destroy()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('PASS', t.__name__)
    print(f'\n{len(tests)} tests passed.')
