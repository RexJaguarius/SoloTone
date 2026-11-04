"""
Recording and re-amping: the float WAV writer, the Recorder tapping the
audio callback (wet/dry content, Master Volume and metronome kept out of
the file), the offline re-amp render, and the Record tab flow in the real
App.

Run directly: python tests/test_recorder.py
"""

import sys
import pathlib
import tempfile
import time
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import solotone as st


def _sine(n, amp=0.3, f=220.0, sr=st.SAMPLE_RATE):
    return (amp * np.sin(2 * np.pi * f * np.arange(n) / sr)).astype(np.float32)

def _bare_profile(**kw):
    d = {'solotone_profile': 1, 'eq_on': False, 'gate_threshold': 1e-9,
         'in_gain': 1.0, 'out_gain': 1.0, 'pedals': {}}
    d.update(kw)
    return d

def _fake_stream_status():
    return types.SimpleNamespace(input_underflow=False, output_underflow=False)

def _run_callback(amp, signal, block=256):
    """Feed a signal through AmpProcessor._cb the way PortAudio would and
    return the audible output."""
    out = []
    status = _fake_stream_status()
    for i in range(0, len(signal) - block + 1, block):
        indata = signal[i:i + block].reshape(-1, 1)
        outdata = np.zeros((block, 1), dtype=np.float32)
        amp._cb(indata, outdata, block, None, status)
        out.append(outdata[:, 0].copy())
    return np.concatenate(out)


def test_float_wav_is_valid_and_survives_a_crash():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'a.wav'
        w = st.FloatWavWriter(path)
        x = _sine(3 * st.SAMPLE_RATE + 123, amp=1.7)       # over 0 dBFS on purpose
        for i in range(0, len(x), 256):
            w.write(x[i:i + 256])
        # Not closed: the header is re-patched every second, so a crash
        # here still leaves a valid file holding all but the last second.
        try:
            arr, sr = st.read_audio_mono(path)
            assert sr == st.SAMPLE_RATE and len(arr) >= 2 * st.SAMPLE_RATE
            assert np.allclose(arr, x[:len(arr)])
        finally:
            w.close()
        arr, _ = st.read_audio_mono(path)
        assert len(arr) == len(x) and np.allclose(arr, x) and arr.max() > 1.5


def test_recorder_captures_wet_and_dry_without_master_or_metronome():
    with tempfile.TemporaryDirectory() as tmp:
        amp = st.AmpProcessor()
        amp.out_gain = 2.0
        amp.master_gain = 0.1                    # must NOT scale the file
        sig = _sine(st.SAMPLE_RATE)
        wet_p, dry_p = pathlib.Path(tmp) / 'w.wav', pathlib.Path(tmp) / 'd.wav'
        rec = st.Recorder(wet_p, dry_p)
        amp.recorder = rec
        heard = _run_callback(amp, sig)
        amp.recorder = None
        secs = rec.stop()
        assert rec.dropped == 0 and rec.error is None
        assert abs(secs - len(heard) / st.SAMPLE_RATE) < 1e-6
        wet, _ = st.read_audio_mono(wet_p)
        dry, _ = st.read_audio_mono(dry_p)
        assert np.allclose(dry, sig[:len(dry)])                     # raw input, untouched
        assert np.allclose(wet[4096:], sig[4096:len(wet)] * 2.0, atol=1e-5)   # Output Gain yes (past the gate ramp)...
        assert np.allclose(heard, wet * 0.1, atol=1e-5)             # ...Master Volume only in what you hear


def test_dry_track_is_optional_and_looper_is_opt_in():
    with tempfile.TemporaryDirectory() as tmp:
        rec = st.Recorder(pathlib.Path(tmp) / 'w.wav', None)
        assert rec.dry_path is None and rec.include_looper is False
        rec.push(np.zeros(256, dtype=np.float32), np.zeros(256, dtype=np.float32))
        rec.stop()
        assert [p.name for p in pathlib.Path(tmp).iterdir()] == ['w.wav']


def test_reamp_matches_the_chain_and_handles_other_sample_rates():
    sig = _sine(st.SAMPLE_RATE)
    y, sr = st.reamp_render(_bare_profile(out_gain=2.0), sig, st.SAMPLE_RATE)
    assert sr == st.SAMPLE_RATE
    assert len(y) == len(sig) + int(st.REAMP_TAIL_SECONDS * st.SAMPLE_RATE)   # tail added
    assert np.allclose(y[4096:len(sig)], sig[4096:] * 2.0, atol=1e-5)   # past the gate opening
    # Normalize rescales the input peak to -12 dBFS before the chain.
    y, _ = st.reamp_render(_bare_profile(), sig, st.SAMPLE_RATE, normalize=True)
    assert abs(20 * np.log10(np.max(np.abs(y))) - st.REAMP_NORMALIZE_PEAK_DB) < 0.2
    # A 48 kHz source comes back at 48 kHz, same duration.
    sig48 = _sine(48000, sr=48000)
    y, sr = st.reamp_render(_bare_profile(), sig48, 48000)
    if st.HAS_SCIPY:
        assert sr == 48000
        assert abs(len(y) / sr - (1.0 + st.REAMP_TAIL_SECONDS)) < 0.01


def test_reamp_can_be_cancelled():
    import threading
    ev = threading.Event(); ev.set()
    assert st.reamp_render(_bare_profile(), _sine(st.SAMPLE_RATE), st.SAMPLE_RATE, cancel=ev) is None


def test_gate_opens_in_milliseconds_and_closes_in_tens_of_milliseconds():
    att = 0.0
    for _ in range(3):                       # ~17 ms of signal at 256-sample blocks
        att, _r = st.gate_step(att, True, 256)
    assert att > 0.95
    n = 0
    while att > 0.1 and n < 1000:            # then silence: closes with a ~60 ms time constant
        att, _r = st.gate_step(att, False, 256); n += 1
    assert 0.1 < n * 256 / st.SAMPLE_RATE < 0.3
    # Same time constants at any block size.
    a1 = b1 = 0.0
    for _ in range(4): a1, _r = st.gate_step(a1, True, 512)
    for _ in range(8): b1, _r = st.gate_step(b1, True, 256)
    assert abs(a1 - b1) < 1e-9


def test_unique_path_and_safe_filename():
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / 'a.wav'
        assert st.unique_path(p) == p
        p.write_bytes(b'x')
        assert st.unique_path(p).name == 'a (2).wav'
    assert st.safe_filename('AC/DC: Back "in" Black?') == 'ACDC Back in Black'
    assert st.safe_filename('???') == 'SoloTone'


def test_record_tab_flow_in_the_app():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        st.SESSION_FILE = tmp_path / 'test_session.json'
        st.MIDI_MAP_FILE = tmp_path / 'test_midi.json'
        app = st.App()
        try:
            app.update()
            app._rec_dir = str(tmp_path / 'takes')          # skip the first-run folder dialog
            app.amp.running = True                           # stand-in for a live stream
            app._rec_set(True)
            assert app._rec is not None and app.rec_var.get() is True
            sig = _sine(st.SAMPLE_RATE)
            _run_callback(app.amp, sig)
            app.update()
            app._rec_set(False)
            assert app._rec is None and app.rec_var.get() is False
            assert len(app._rec_takes) == 1
            take = app._rec_takes[0]
            wet, _ = st.read_audio_mono(take['wet'])
            dry, _ = st.read_audio_mono(take['dry'])
            assert len(wet) == len(dry) == len(sig) // 256 * 256
            assert 'SoloTone - ' in pathlib.Path(take['wet']).name and take['wet'].endswith('(wet).wav')
            # Re-amp the dry take through the current (bare) sound.
            app.reamp_src_var.set(take['dry'])
            app._reamp_run()
            deadline = time.time() + 60
            while app._reamp_job is not None and time.time() < deadline:
                app.update(); time.sleep(0.05)
            assert app._reamp_result and pathlib.Path(app._reamp_result).exists()
            out, _ = st.read_audio_mono(app._reamp_result)
            assert len(out) > len(dry)                       # tail added
            # Stopping the amp mid-take finishes the file.
            app._rec_set(True)
            _run_callback(app.amp, sig[:2048])
            app._toggle_amp()
            assert app._rec is None and len(app._rec_takes) == 2
        finally:
            app.amp.running = False
            app.destroy()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('PASS', t.__name__)
    print(f'\n{len(tests)} tests passed.')
