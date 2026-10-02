"""
Pedal-chain DSP regression tests — pure numpy in/out, no Tkinter, no
audio device, no display required. Safe to run headless on any CI
runner (Linux/macOS/Windows alike).

Uses a synthetic test signal (a decaying harmonically-rich tone, not a
pure sine) rather than a real guitar capture, so this file is fully
self-contained and needs nothing outside the repo to run. It won't
catch everything a real guitar would (see version.py's build-75
changelog entry for a case where only real playing surfaced the bug),
but it does catch the concrete, structural failure modes that bug fix
was about: NaN/Inf, out-of-range output, state that resets when it
shouldn't, and instability under rapid live parameter changes.

Run directly: python tests/test_pedals_dsp.py
Or with pytest: pytest tests/test_pedals_dsp.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from pedals import PedalChain

SR = 44100
BLOCK = 256


def _test_signal(duration=2.0, sr=SR):
    """A decaying, harmonically-rich tone (fundamental + a few overtones,
    not a pure sine) — closer to a real plucked string than a sine wave,
    while staying fully synthetic/deterministic for CI."""
    t = np.arange(int(duration * sr)) / sr
    f0 = 110.0  # A2, a typical low-string fundamental
    sig = np.zeros_like(t)
    for n, amp in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15), (5, 0.08)):
        sig += amp * np.sin(2 * np.pi * f0 * n * t)
    env = 0.5 * np.exp(-t / 0.9)
    return (sig * env / np.max(np.abs(sig))).astype(np.float32)


_SIGNAL = _test_signal()


def run_chain(sig, fx_name, **kwargs):
    """Feed `sig` through one PedalChain method in real block-size
    chunks (preserving streaming filter/oversampler state across calls,
    same as the real audio callback), every other pedal left bypassed."""
    p = PedalChain(sr=SR)
    for k, v in kwargs.items():
        setattr(p, k, v)
    method = getattr(p, fx_name)
    out = np.zeros(len(sig), dtype=np.float64)
    n = len(sig)
    pos = 0
    while pos < n:
        take = min(BLOCK, n - pos)
        block = sig[pos:pos + take].astype(np.float64)
        if take < BLOCK:
            block = np.pad(block, (0, BLOCK - take))
        y = method(block)
        out[pos:pos + take] = y[:take]
        pos += take
    return out


def assert_sane(sig, label):
    assert np.all(np.isfinite(sig)), f'{label}: NaN/Inf detected'
    peak = float(np.max(np.abs(sig)))
    assert peak <= 1.0001, f'{label}: peak {peak} exceeds clip range'


def test_fuzz_all_modes_and_tones():
    for mode in ('fuzz_face', 'big_muff'):
        for tone in ('full', 'flat', 'scoop'):
            for drv in (1.0, 5.0, 10.0):
                out = run_chain(_SIGNAL, '_fx_fuzz', fuzz_on=True, fuzz_mode=mode,
                                 fuzz_drive=drv, fuzz_vol=0.8, muff_tone=tone)
                assert_sane(out, f'fuzz mode={mode} tone={tone} drive={drv}')


def test_distortion_all_modes_and_tones():
    for mode in ('ds1', 'rat', 'dist_plus', 'metal'):
        for drv in (1.0, 5.0, 10.0):
            for tone in (0.0, 0.5, 1.0):
                out = run_chain(_SIGNAL, '_fx_dist', dist_on=True, dist_mode=mode,
                                 dist_drive=drv, dist_vol=0.8, dist_tone=tone)
                assert_sane(out, f'dist mode={mode} tone={tone} drive={drv}')


def test_asymmetric_clip_dc_bias_stays_bounded():
    """Regression guard for the build-75 fix: MXR Distortion+ and Fuzz
    Face's DC-blocking stage must keep the DC offset small and
    non-growing across the drive range, not climbing with drive the way
    it did before that fix (measured then: +0.045 at drive 5 to +0.062
    at drive 10)."""
    for fx_name, kwargs in (
        ('_fx_dist', dict(dist_on=True, dist_mode='dist_plus', dist_vol=1.0, dist_tone=0.5)),
        ('_fx_fuzz', dict(fuzz_on=True, fuzz_mode='fuzz_face', fuzz_vol=1.0)),
    ):
        drive_attr = 'dist_drive' if fx_name == '_fx_dist' else 'fuzz_drive'
        dcs = []
        for drv in (3.0, 5.0, 7.0, 10.0):
            kw = dict(kwargs)
            kw[drive_attr] = drv
            out = run_chain(_SIGNAL, fx_name, **kw)
            dcs.append(abs(float(np.mean(out))))
        assert max(dcs) < 0.01, f'{fx_name}: DC bias grew too large across drive range: {dcs}'


def test_metal_notch_filter_state_is_continuous():
    """Regression guard for the build-75 fix: Metal's mid-scoop notch
    filter must carry state across blocks instead of resetting to zero
    every call — checked by comparing block-boundary discontinuities
    against the filter's own typical sample-to-sample step size."""
    out = run_chain(_SIGNAL, '_fx_dist', dist_on=True, dist_mode='metal',
                     dist_drive=5.0, dist_vol=1.0, dist_tone=0.5)
    jumps = np.abs(np.diff(out)[BLOCK - 1::BLOCK])
    typical_step = np.mean(np.abs(np.diff(out)))
    assert np.mean(jumps) < typical_step * 5, (
        f'block-boundary jumps ({np.mean(jumps):.4f}) are far bigger than the '
        f'typical sample-to-sample step ({typical_step:.4f}) — filter state is '
        f'likely resetting every block again')


def test_full_chain_all_pedals_engaged():
    p = PedalChain(sr=SR)
    p.comp_on = True
    p.wah_on = True; p.wah_mode = 'auto'
    p.fuzz_on = True; p.fuzz_mode = 'big_muff'; p.fuzz_drive = 8.0
    p.dist_on = True; p.dist_mode = 'metal'; p.dist_drive = 8.0
    p.od_on = True; p.od_mode = 'klon'
    p.chorus_on = True
    p.delay_on = True
    p.reverb_on = True
    out = np.zeros(len(_SIGNAL), dtype=np.float64)
    pos = 0
    while pos < len(_SIGNAL):
        take = min(BLOCK, len(_SIGNAL) - pos)
        block = _SIGNAL[pos:pos + take]
        if take < BLOCK:
            block = np.pad(block, (0, BLOCK - take))
        y = p.process(block, metro_bpm=120)
        out[pos:pos + take] = y[:take]
        pos += take
    assert_sane(out, 'full chain, all 8 pedals engaged')


def test_fuzz_mode_switching_mid_stream_is_stable():
    """Stress test: changing Fuzz mode/tone every single block (as if
    the user were clicking the dropdown live) must not destabilize the
    persistent filter state introduced by the build-75 fix."""
    p = PedalChain(sr=SR)
    p.fuzz_on = True
    p.fuzz_drive = 7.0
    modes = ['fuzz_face', 'big_muff']
    tones = ['full', 'flat', 'scoop']
    out = np.zeros(len(_SIGNAL), dtype=np.float64)
    pos = 0
    i = 0
    while pos < len(_SIGNAL):
        take = min(BLOCK, len(_SIGNAL) - pos)
        block = _SIGNAL[pos:pos + take].astype(np.float64)
        if take < BLOCK:
            block = np.pad(block, (0, BLOCK - take))
        p.fuzz_mode = modes[i % len(modes)]
        p.muff_tone = tones[i % len(tones)]
        y = p._fx_fuzz(block)
        out[pos:pos + take] = y[:take]
        pos += take
        i += 1
    assert_sane(out, 'fuzz mode/tone switched every block')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'PASS {fn.__name__}')
    print(f'\n{len(tests)} tests passed.')
