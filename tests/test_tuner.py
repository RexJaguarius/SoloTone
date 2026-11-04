"""
Tuner: the pitch detector finds quiet single-coil signals (with hum) that
the old absolute-level gate threw away, stays silent when nothing is being
played, and the tuner display snaps to the selected tuning's strings
instead of showing every chromatic note it hears.

Run directly: python tests/test_tuner.py
"""

import sys
import math
import pathlib
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import solotone as st

SR = st.SAMPLE_RATE


def _midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)

def _pluck(f0, rms, seed=0, n=12288, fund=1.0, hum_rms=0.0, noise_db=-35):
    """A guitar-ish plucked string: harmonic series with a scalable
    fundamental (a quiet pickup on a wound string is weak at f0), per-
    harmonic decay, white noise and optional 60 Hz hum, at a given RMS."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) + rng.uniform(0.1, 1.5) * SR) / SR
    x = np.zeros(n)
    for h in range(1, 14):
        fh = f0 * h
        if fh > SR / 2.2:
            break
        amp = (1.0 / h) * (fund if h == 1 else 1.0)
        x += amp * np.exp(-2.0 * t * (1 + 0.15 * h)) * np.sin(2 * np.pi * fh * t + rng.uniform(0, 6.28))
    x *= rms / np.sqrt(np.mean(x ** 2))
    x += rng.normal(0, rms * 10 ** (noise_db / 20), n)
    if hum_rms:
        x += hum_rms * math.sqrt(2) * np.sin(2 * np.pi * 60 * np.arange(n) / SR)
    return x.astype(np.float32)

def _detect(sig, tuning_midi=None, hum=True):
    fmin, fmax = st.tuning_search_range(tuning_midi) if tuning_midi else (40.0, 1500.0)
    if hum:
        sig = st.remove_mains_hum(sig, 60.0, SR)
    return st.autocorrelate_pitch(sig, SR, fmin, fmax)


def test_finds_every_open_string_down_to_a_quiet_pickups_level():
    std = [_midi for _midi in (40, 45, 50, 55, 59, 64)]
    for rms in (0.05, 0.005, 0.001):                    # -26 ... -60 dBFS RMS
        for m in std:
            f0 = _midi_hz(m) * 2 ** (13 / 1200)         # 13 cents sharp
            f = _detect(_pluck(f0, rms, seed=m), std)
            assert f is not None, f'no pitch for midi {m} at rms {rms}'
            assert abs(1200 * math.log2(f / f0)) < 6, (m, rms, f, f0)

def test_weak_fundamental_is_not_reported_an_octave_up():
    for m in (40, 45, 50):
        f0 = _midi_hz(m)
        f = _detect(_pluck(f0, 0.002, seed=m, fund=0.15), [40, 45, 50, 55, 59, 64])
        assert f is not None and abs(1200 * math.log2(f / f0)) < 6

def test_hum_louder_than_the_note_is_removed():
    f0 = _midi_hz(45)                                   # A2, near the 120 Hz hum harmonic
    f = _detect(_pluck(f0, 0.002, seed=3, hum_rms=0.004), [40, 45, 50, 55, 59, 64])
    assert f is not None and abs(1200 * math.log2(f / f0)) < 6

def test_nothing_is_reported_when_nothing_is_played():
    n = 12288
    t = np.arange(n) / SR
    rng = np.random.default_rng(5)
    idle = {
        'silence': np.zeros(n),
        'white noise': rng.normal(0, 0.001, n),
        '60 Hz hum': 0.003 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t),
        'hum + harmonics': 0.002 * (np.sin(2 * np.pi * 60 * t) + 0.7 * np.sin(2 * np.pi * 120 * t)
                                    + 0.5 * np.sin(2 * np.pi * 180 * t)),
    }
    for name, sig in idle.items():
        assert _detect(sig.astype(np.float32), [40, 45, 50, 55, 59, 64]) is None, name
        assert _detect(sig.astype(np.float32)) is None, name + ' (chromatic range)'

def test_bass_low_b_is_found():
    f0 = _midi_hz(23)                                    # 5-string low B, ~30.9 Hz
    f = _detect(_pluck(f0, 0.003, seed=9, n=16384), [23, 28, 33, 38, 43])
    assert f is not None and abs(1200 * math.log2(f / f0)) < 8


def test_closest_string_folded_ignores_octave_and_keeps_the_note():
    std = st.GUITAR_TUNING_MIDI['Standard (EADGBe)']
    idx, cents = st.closest_string_folded(82.41, std)
    assert idx == 0 and abs(cents) < 1
    idx, cents = st.closest_string_folded(164.81, std)  # E3: octave error of the low E
    assert idx == 0 and abs(cents) < 1                  # tie with the high E goes to the lower string
    idx, cents = st.closest_string_folded(330.0, std)
    assert idx == 5
    idx, cents = st.closest_string_folded(110.0 * 2 ** (-20 / 1200), std)   # A2, 20 cents flat
    assert idx == 1 and -22 < cents < -18
    # Never a note that isn't a string of this tuning: a G# (between G and A) snaps to one of them.
    idx, _ = st.closest_string_folded(207.65, std)
    assert idx in (2, 3)
    assert st.closest_string_folded(None, std) == (None, 0.0)
    assert st.closest_string_folded(440.0, None) == (None, 0.0)


def test_search_range_follows_the_tuning():
    lo, hi = st.tuning_search_range(st.GUITAR_TUNING_MIDI['Standard (EADGBe)'])
    assert 60 < lo < 70 and 450 < hi < 520
    lo, hi = st.tuning_search_range(st.GUITAR_TUNING_MIDI['Bass Standard (EADG)'])
    assert 30 < lo < 36 and 100 < hi < 150


_APP = None

def test_tuner_end_to_end_with_a_quiet_hummy_signal():
    """The Tuner object itself: fed raw input blocks the way the audio
    callback does, with the amp's hum setting supplying the mains base."""
    tn = st.Tuner()
    tn.amp = st.AmpProcessor()
    tn.running = True
    tn.fmin, tn.fmax = st.tuning_search_range(st.GUITAR_TUNING_MIDI['Standard (EADGBe)'])
    f0 = _midi_hz(43)                                   # G3... a 'G' string, quiet and humming
    sig = _pluck(f0, 0.0015, seed=11, n=24576, hum_rms=0.002)
    for i in range(0, len(sig), 256):
        tn.feed(sig[i:i + 256])
    got = [tn.pitch() for _ in range(4)]
    assert all(g is not None and abs(1200 * math.log2(g / f0)) < 6 for g in got), got


def _app():
    """One App for every display test: the app caches Tk images per root
    window, so building a second App in the same process fails."""
    global _APP
    if _APP is None:
        tmp = tempfile.mkdtemp()
        st.SESSION_FILE = pathlib.Path(tmp) / 'test_session.json'
        st.MIDI_MAP_FILE = pathlib.Path(tmp) / 'test_midi.json'
        _APP = st.App()
        _APP.update()
    _APP.tuner.running = True
    _APP.instrument_var.set('guitar'); _APP._instrument_changed()
    return _APP, None

def _feed(app, freq):
    app.tuner.pitch = lambda f=freq: f
    app._tuner_update()


def test_display_snaps_to_the_strings_of_the_selected_tuning():
    app, tmp = _app()
    try:
        app.tuning_var.set('Standard (EADGBe)'); app._reset_string_highlights()
        assert app.tuner.fmin > 60                      # search range narrowed to this tuning
        _feed(app, 82.6)
        assert app.foot_note.cget('text') == 'E2'
        _feed(app, _midi_hz(40) * 2 ** (-30 / 1200))     # exactly 30 cents flat: still E, not D#
        assert app.foot_note.cget('text') == 'E2'
        assert app.foot_cents.cget('text') == '-30¢'
        _feed(app, 164.8)                               # octave error from the low E: same string
        assert app.foot_note.cget('text') == 'E2'
        for _ in range(3):                              # a new string must win three readings
            _feed(app, 110.4)
        assert app.foot_note.cget('text') == 'A2'
    finally:
        app.tuner.running = False

def test_one_stray_reading_does_not_move_the_display():
    app, tmp = _app()
    try:
        app.tuning_var.set('Standard (EADGBe)'); app._reset_string_highlights()
        _feed(app, 110.0)
        assert app.foot_note.cget('text') == 'A2'
        _feed(app, 147.0)                               # a lone D3 overtone
        assert app.foot_note.cget('text') == 'A2'
        _feed(app, 110.2)
        assert app.foot_note.cget('text') == 'A2'
    finally:
        app.tuner.running = False

def test_chromatic_mode_still_shows_every_note():
    app, tmp = _app()
    try:
        app.tuning_var.set('None (chromatic)'); app._reset_string_highlights()
        assert app.tuner.fmin == 40.0 and app.tuner.fmax == 1500.0
        _feed(app, 103.83)                              # G#2
        assert app.foot_note.cget('text') == 'G#2'
    finally:
        app.tuner.running = False

def test_flat_tunings_and_bass_use_their_own_names_and_ranges():
    app, tmp = _app()
    try:
        app.tuning_var.set('Eb standard'); app._reset_string_highlights()
        _feed(app, 77.78)                               # Eb2
        assert app.foot_note.cget('text') == 'Eb2'
        app.instrument_var.set('bass'); app._instrument_changed()
        assert app.tuning_var.get() == 'Bass Standard (EADG)' and app.tuner.fmax < 200
        _feed(app, 41.2)
        assert app.foot_note.cget('text') == 'E1'
        assert app.str_cents_lbl.cget('text').startswith('str 4')   # 4 strings, lowest is #4
    finally:
        app.tuner.running = False


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('PASS', t.__name__)
    print(f'\n{len(tests)} tests passed.')
