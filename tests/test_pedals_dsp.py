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
from pedals import PedalChain, DEFAULT_PEDAL_ORDER, AMP_MARKER, normalize_order

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
        y = p.process_post(y, metro_bpm=120)   # the post-amp half of the chain
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


def _run_blocks(fn, signal):
    out = np.zeros(len(signal), dtype=np.float64)
    for pos in range(0, len(signal) - BLOCK + 1, BLOCK):
        out[pos:pos + BLOCK] = fn(signal[pos:pos + BLOCK])
    return out


def test_default_order_puts_time_effects_after_amp():
    assert DEFAULT_PEDAL_ORDER.count(AMP_MARKER) == 1
    i = DEFAULT_PEDAL_ORDER.index(AMP_MARKER)
    assert set(DEFAULT_PEDAL_ORDER[:i]) == {'comp', 'wah', 'pitch', 'fuzz', 'dist', 'od', 'phaser'}
    assert set(DEFAULT_PEDAL_ORDER[i + 1:]) == {'chorus', 'tremolo', 'rotary', 'delay', 'reverb'}


# ── phaser, tremolo, vibrato, doubler, Blues Driver, plate / spring / cloud ──

def test_new_pedals_are_sane_across_their_ranges():
    for rate in (0.05, 0.5, 5.0):
        for depth in (0.0, 0.6, 1.0):
            for fb in (0.0, 0.5, 0.9):
                out = run_chain(_SIGNAL, '_fx_phaser', phaser_on=True, phaser_rate=rate,
                                phaser_depth=depth, phaser_feedback=fb)
                assert_sane(out, f'phaser rate={rate} depth={depth} fb={fb}')
    for rate in (0.5, 5.0, 15.0):
        for depth in (0.0, 0.5, 1.0):
            for shape in (0.0, 0.5, 1.0):
                out = run_chain(_SIGNAL, '_fx_tremolo', tremolo_on=True, tremolo_rate=rate,
                                tremolo_depth=depth, tremolo_shape=shape)
                assert_sane(out, f'tremolo rate={rate} depth={depth} shape={shape}')
    for mode in ('chorus', 'flanger', 'vibrato', 'doubler'):
        for depth in (0.0, 0.5, 1.0):
            out = run_chain(_SIGNAL, '_fx_chorus', chorus_on=True, chorus_mode=mode,
                            chorus_depth=depth, chorus_mix=0.5, chorus_rate=2.0)
            assert_sane(out, f'chorus mode={mode} depth={depth}')
    for drv in (1.0, 5.0, 10.0):
        out = run_chain(_SIGNAL, '_fx_od', od_on=True, od_mode='bd2', od_drive=drv, od_vol=0.8)
        assert_sane(out, f'Blues Driver drive={drv}')
        assert abs(float(np.mean(out))) < 0.01, f'Blues Driver leaves DC at drive {drv}'


def test_tremolo_swings_the_gain_between_one_and_one_minus_depth():
    ones = np.ones(SR, dtype=np.float32) * 0.5
    out = run_chain(ones, '_fx_tremolo', tremolo_on=True, tremolo_rate=4.0,
                    tremolo_depth=0.6, tremolo_shape=0.0)
    assert abs(float(out.max()) - 0.5) < 0.01, out.max()
    assert abs(float(out.min()) - 0.5 * 0.4) < 0.01, out.min()
    flat = run_chain(ones, '_fx_tremolo', tremolo_on=True, tremolo_depth=0.0)
    assert np.allclose(flat, 0.5, atol=1e-6)


def test_phaser_does_not_depend_on_the_block_size():
    def run(block):
        p = PedalChain(sr=SR); p.phaser_on = True; p.phaser_feedback = 0.6
        out = np.zeros(len(_SIGNAL))
        for pos in range(0, len(_SIGNAL) - block + 1, block):
            out[pos:pos + block] = p._fx_phaser(_SIGNAL[pos:pos + block].astype(np.float64))
        return out
    a, b = run(256), run(512)
    n = min(len(a), len(b))
    assert float(np.max(np.abs(a[:n] - b[:n]))) < 1e-9


def test_phaser_notches_the_signal():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.1, SR * 2).astype(np.float32)
    out = run_chain(noise, '_fx_phaser', phaser_on=True, phaser_depth=0.0, phaser_feedback=0.0)
    spec = np.abs(np.fft.rfft(out[SR // 2:])) / np.maximum(np.abs(np.fft.rfft(noise[SR // 2:])), 1e-9)
    assert spec.min() < 0.1, 'a static phaser should carve deep notches in a flat spectrum'


def test_vibrato_has_no_dry_signal_and_doubler_keeps_it():
    imp = np.zeros(SR // 2, dtype=np.float32); imp[0] = 1.0
    vib = run_chain(imp, '_fx_chorus', chorus_on=True, chorus_mode='vibrato', chorus_depth=0.5, chorus_mix=0.3)
    assert abs(float(vib[0])) < 0.05, 'vibrato must be fully wet whatever the Mix says'
    dbl = run_chain(imp, '_fx_chorus', chorus_on=True, chorus_mode='doubler', chorus_depth=0.5, chorus_mix=0.5)
    assert abs(float(dbl[0]) - 0.5) < 0.01, 'doubler keeps the dry signal at the mix amount'
    late = np.argmax(np.abs(dbl[100:]) > 0.3) + 100      # the delayed copy
    assert 0.010 * SR < late < 0.030 * SR, late


def test_new_reverbs_are_stable_and_independent_of_block_size():
    rng = np.random.default_rng(5)
    sig = np.zeros(SR * 3); sig[100] = 0.8; sig[SR:SR + 2000] = rng.normal(0, 0.2, 2000)
    for rtype in ('plate', 'cloud', 'spring'):
        for decay in (0.0, 0.5, 1.0):
            for size in (0.1, 0.6, 1.0):
                p = PedalChain(sr=SR); p.reverb_on = True; p.reverb_type = rtype
                p.reverb_decay, p.reverb_size, p.reverb_mix = decay, size, 1.0
                out = np.zeros(len(sig))
                for pos in range(0, len(sig) - 256 + 1, 256):
                    out[pos:pos + 256] = p._fx_reverb(sig[pos:pos + 256].copy())
                assert np.all(np.isfinite(out)), f'{rtype} decay={decay} size={size}'
                assert np.max(np.abs(out)) < 2.0, f'{rtype} decay={decay} size={size} peak {np.max(np.abs(out))}'

    def run(rtype, block):
        p = PedalChain(sr=SR); p.reverb_on = True; p.reverb_type = rtype; p.reverb_mix = 1.0
        out = np.zeros(len(sig))
        for pos in range(0, len(sig) - block + 1, block):
            out[pos:pos + block] = p._fx_reverb(sig[pos:pos + block].copy())
        return out
    for rtype in ('plate', 'cloud', 'spring'):
        a, b = run(rtype, 256), run(rtype, 768)
        n = min(len(a), len(b))
        assert float(np.max(np.abs(a[:n] - b[:n]))) < 1e-9, f'{rtype} sounds different at different block sizes'


def test_reverb_decay_control_lengthens_the_tail():
    rng = np.random.default_rng(6)
    sig = np.zeros(SR * 4); sig[:2000] = rng.normal(0, 0.3, 2000)
    for rtype in ('plate', 'cloud', 'spring'):
        tails = []
        for decay in (0.1, 0.9):
            p = PedalChain(sr=SR); p.reverb_on = True; p.reverb_type = rtype
            p.reverb_decay, p.reverb_mix = decay, 1.0
            out = np.zeros(len(sig))
            for pos in range(0, len(sig) - 256 + 1, 256):
                out[pos:pos + 256] = p._fx_reverb(sig[pos:pos + 256].copy())
            tails.append(float(np.sqrt(np.mean(out[3 * SR:] ** 2))))
        assert tails[1] > tails[0] * 4, f'{rtype}: decay 0.9 tail {tails[1]} vs decay 0.1 tail {tails[0]}'


def test_room_reverb_is_unchanged_by_the_new_types():
    """The default type is the original Schroeder reverb, bit for bit."""
    a = run_chain(_SIGNAL, '_fx_reverb', reverb_on=True, reverb_mix=0.4)
    b = run_chain(_SIGNAL, '_fx_reverb', reverb_on=True, reverb_mix=0.4, reverb_type='room',
                  reverb_decay=0.9, reverb_damp=0.1)
    assert np.array_equal(a, b)


def test_full_chain_with_every_pedal_engaged():
    p = PedalChain(sr=SR)
    for name in ('comp', 'wah', 'pitch', 'fuzz', 'dist', 'od', 'phaser', 'chorus', 'tremolo', 'rotary', 'delay', 'reverb'):
        setattr(p, name + '_on', True)
    p.od_mode = 'bd2'; p.chorus_mode = 'doubler'; p.reverb_type = 'spring'
    out = np.zeros(len(_SIGNAL), dtype=np.float64)
    for pos in range(0, len(_SIGNAL) - BLOCK + 1, BLOCK):
        y = p.process(_SIGNAL[pos:pos + BLOCK], metro_bpm=120)
        out[pos:pos + BLOCK] = p.process_post(y, metro_bpm=120)
    assert_sane(out, 'full chain, every pedal engaged')


def test_pre_and_post_stages_only_run_their_own_pedals():
    sig = _SIGNAL[:BLOCK * 20]
    # a delay alone, placed after the amp: process() must leave the signal
    # untouched, process_post() must change it.
    p = PedalChain(sr=SR)
    p.delay_on = True; p.delay_time = 0.05; p.delay_mix = 0.5
    pre = _run_blocks(lambda b: p.process(b), sig)
    assert np.allclose(pre, sig, atol=1e-6), 'a post-amp delay altered the pre-amp stage'
    post = _run_blocks(lambda b: p.process_post(b), sig)
    assert not np.allclose(post, sig, atol=1e-3), 'a post-amp delay did nothing in the post stage'
    # same delay moved ahead of the amp marker: now process() applies it
    # and process_post() leaves the signal alone.
    p2 = PedalChain(sr=SR)
    p2.delay_on = True; p2.delay_time = 0.05; p2.delay_mix = 0.5
    p2.order = ['delay', AMP_MARKER] + [c for c in DEFAULT_PEDAL_ORDER if c not in ('delay', AMP_MARKER)]
    pre2 = _run_blocks(lambda b: p2.process(b), sig)
    assert not np.allclose(pre2, sig, atol=1e-3)
    post2 = _run_blocks(lambda b: p2.process_post(b), sig)
    assert np.allclose(post2, sig, atol=1e-6)


def test_mute_silences_both_stages():
    p = PedalChain(sr=SR)
    p.mute = True
    p.reverb_on = True
    blk = _SIGNAL[:BLOCK]
    assert not np.any(p.process(blk))
    assert not np.any(p.process_post(blk))


def test_normalize_order_keeps_legacy_orders_pre_amp():
    legacy_default = ['comp', 'wah', 'fuzz', 'dist', 'od', 'chorus', 'delay', 'reverb']
    # untouched factory order -> upgraded to the new default
    assert normalize_order(legacy_default) == DEFAULT_PEDAL_ORDER
    # a customised old order keeps every pedal it had in the same relative order, all before the
    # amp; the two newer pedals are added (phaser just ahead of the amp, tremolo just after it)
    custom = ['comp', 'od', 'chorus', 'delay', 'reverb', 'wah', 'fuzz', 'dist']
    got = normalize_order(custom)
    assert [c for c in got if c not in ('phaser', 'tremolo', 'rotary', 'pitch')][:len(custom)] == custom
    assert got.index('phaser') == got.index(AMP_MARKER) - 1
    assert got.index('tremolo') == got.index(AMP_MARKER) + 1
    assert sorted(c for c in got if c != AMP_MARKER) == sorted(c for c in DEFAULT_PEDAL_ORDER if c != AMP_MARKER)
    # current-format orders pass through unchanged
    assert normalize_order(DEFAULT_PEDAL_ORDER) == DEFAULT_PEDAL_ORDER
    # junk is rejected
    assert normalize_order(['comp', 'wah']) is None
    assert normalize_order('comp') is None
    assert normalize_order(DEFAULT_PEDAL_ORDER + [AMP_MARKER]) is None
    assert normalize_order(DEFAULT_PEDAL_ORDER[:-1] + ['nonsense']) is None


def test_orders_saved_before_phaser_and_tremolo_get_them_added():
    previous_default = ['comp', 'wah', 'fuzz', 'dist', 'od', AMP_MARKER, 'chorus', 'delay', 'reverb']
    assert normalize_order(previous_default) == DEFAULT_PEDAL_ORDER
    custom = ['delay', 'comp', AMP_MARKER, 'od', 'wah', 'reverb', 'fuzz', 'chorus', 'dist']
    got = normalize_order(custom)
    assert got.index('phaser') == got.index(AMP_MARKER) - 1                  # end of the pre-amp half
    assert got.index('tremolo') == got.index('chorus') + 1                   # right after the chorus
    assert got.index('rotary') == got.index('tremolo') + 1                   # the rotary follows the tremolo
    assert [c for c in got if c not in ('phaser', 'tremolo', 'rotary', 'pitch')] == custom   # nothing else moved
    # an untouched order from the phaser/tremolo era is upgraded to the new factory order; a custom one gains the rotary
    era = ['comp', 'wah', 'fuzz', 'dist', 'od', 'phaser', AMP_MARKER, 'chorus', 'tremolo', 'delay', 'reverb']
    assert normalize_order(era) == DEFAULT_PEDAL_ORDER
    swapped = ['comp', 'wah', 'fuzz', 'dist', 'od', 'phaser', AMP_MARKER, 'delay', 'chorus', 'tremolo', 'reverb']
    got = normalize_order(swapped)
    assert got.index('rotary') == got.index('tremolo') + 1
    assert [c for c in got if c not in ('rotary', 'pitch')] == swapped
    assert normalize_order([c for c in DEFAULT_PEDAL_ORDER if c != 'rotary']) is not None
    assert normalize_order([c for c in DEFAULT_PEDAL_ORDER if c != 'comp']) is None


def test_envelope_filter_follows_the_playing_level():
    """Soft notes keep the filter low, hard notes open it; 'down' does the opposite; every filter type stays sane."""
    t = np.arange(SR // 2) / SR
    tone = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)
    soft, hard = (0.03 * tone).astype(np.float32), (0.35 * tone).astype(np.float32)
    got = {}
    for kind in ('wah', 'bandpass', 'lowpass', 'highpass'):
        for direction in ('up', 'down'):
            for label, sig in (('soft', soft), ('hard', hard)):
                p = PedalChain(sr=SR)
                p.wah_on, p.wah_mode, p.wah_env_type, p.wah_env_dir = True, 'envelope', kind, direction
                out = run_chain(sig, '_fx_wah', wah_on=True, wah_mode='envelope', wah_env_type=kind, wah_env_dir=direction)
                assert_sane(out, f'envelope {kind}/{direction}/{label}')
                got[(kind, direction, label)] = out
    pc = PedalChain(sr=SR)
    pc.wah_on, pc.wah_mode = True, 'envelope'
    pc._fx_wah(soft.astype(np.float64))
    quiet_fc = pc.wah_freq
    pc._fx_wah(hard.astype(np.float64))
    assert pc.wah_freq > quiet_fc * 1.5, 'a hard note should open the envelope filter well above where a soft one left it'
    pc.wah_env_dir = 'down'
    pc._wah_env = 0.0
    pc._fx_wah(hard.astype(np.float64))
    assert pc.wah_freq < 800, 'in Down mode a hard note closes the filter'


def test_envelope_filter_does_not_depend_on_the_block_size():
    sig = _SIGNAL[:SR]
    a = run_chain(sig, '_fx_wah', wah_on=True, wah_mode='envelope')
    p = PedalChain(sr=SR); p.wah_on, p.wah_mode = True, 'envelope'
    b = np.concatenate([p._fx_wah(sig[i:i + 100].astype(np.float64)) for i in range(0, len(sig), 100)])
    assert np.allclose(a[:len(b)], b[:len(a)], atol=1e-6)


def test_tone_bender_is_stable_and_has_a_tighter_low_end_than_the_fuzz_face():
    for drive in (1.0, 5.0, 10.0):
        out = run_chain(_SIGNAL, '_fx_fuzz', fuzz_on=True, fuzz_mode='tone_bender', fuzz_drive=drive, fuzz_vol=0.7)
        assert_sane(out, f'tone bender drive {drive}')
        assert float(np.max(np.abs(out))) > 0.05
    low = (0.3 * np.sin(2 * np.pi * 60 * np.arange(SR) / SR)).astype(np.float32)
    tb = run_chain(low, '_fx_fuzz', fuzz_on=True, fuzz_mode='tone_bender', fuzz_drive=2.0, fuzz_vol=1.0)
    ff = run_chain(low, '_fx_fuzz', fuzz_on=True, fuzz_mode='fuzz_face', fuzz_drive=2.0, fuzz_vol=1.0)
    spec = lambda x: np.abs(np.fft.rfft(x[SR // 2:]))
    assert spec(tb)[30] < spec(ff)[30], 'the Tone Bender input high-pass should leave less 60 Hz fundamental'


def test_rotary_is_stable_and_does_not_depend_on_the_block_size():
    t = np.arange(SR * 2) / SR
    sig = (0.3 * (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t))).astype(np.float32)
    outs = []
    for size in (256, 100):
        p = PedalChain(sr=SR)
        p.rotary_on, p.rotary_speed = True, 'fast'
        outs.append(np.concatenate([p._fx_rotary(sig[i:i + size].astype(np.float64)) for i in range(0, len(sig), size)]))
    assert_sane(outs[0], 'rotary fast')
    m = min(len(outs[0]), len(outs[1]))
    assert np.allclose(outs[0][:m], outs[1][:m], atol=1e-9)
    for speed in ('slow', 'brake'):
        for depth in (0.0, 0.5, 1.0):
            assert_sane(run_chain(sig, '_fx_rotary', rotary_on=True, rotary_speed=speed, rotary_depth=depth), f'rotary {speed} {depth}')


def test_rotary_changes_the_sound_and_depth_zero_leaves_it_alone():
    t = np.arange(SR) / SR
    sig = (0.3 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    swirl = run_chain(sig, '_fx_rotary', rotary_on=True, rotary_speed='fast', rotary_depth=1.0)
    flat = run_chain(sig, '_fx_rotary', rotary_on=True, rotary_speed='fast', rotary_depth=0.0)
    env = lambda x: np.abs(x[SR // 4:]).reshape(-1, 441)[:, :].max(axis=1)
    assert env(swirl).max() - env(swirl).min() > 0.1, 'depth 1 should swell the level as the horn turns'
    assert env(flat).max() - env(flat).min() < 0.02, 'depth 0 should leave a steady tone steady'


def test_rotary_speeds_up_slows_down_and_brakes():
    sig = (0.3 * np.sin(2 * np.pi * 300 * np.arange(441) / SR)).astype(np.float64)
    p = PedalChain(sr=SR)
    p.rotary_on = True
    p._fx_rotary(sig)
    slow_rate = p._rot_rate[0]
    assert abs(slow_rate - p.rotary_slow) < 0.01
    p.rotary_speed = 'fast'
    for _ in range(100):                         # one second
        p._fx_rotary(sig)
    horn, drum = p._rot_rate
    assert slow_rate < horn < p.rotary_fast and drum < horn, 'the rotors coast up to speed, the heavy drum slower'
    for _ in range(600):
        p._fx_rotary(sig)
    assert abs(p._rot_rate[0] - p.rotary_fast) < 0.05
    p.rotary_speed = 'brake'
    for _ in range(600):
        p._fx_rotary(sig)
    assert p._rot_rate[0] < 0.05 and p._rot_rate[1] < p._rot_rate[0] * 0 + 0.5


def _f0(x, sr=SR):
    """Fundamental by autocorrelation (works on a harmonic, guitar-like tone)."""
    x = x - x.mean()
    n = len(x)
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n]
    lo, hi = int(sr / 1000), int(sr / 70)
    k = lo + int(np.argmax(ac[lo:hi]))
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    return sr / (k + 0.5 * (a - c) / (a - 2 * b + c))


def test_pitch_shifter_moves_a_tone_by_the_requested_interval():
    t = np.arange(SR * 2) / SR
    for base in (196.0, 330.0):
        sig = 0.2 * sum(a * np.sin(2 * np.pi * base * n * t) for n, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)))
        for semis in (12, -12, 7, -5, 3):
            p = PedalChain(sr=SR); p.pitch_on, p.pitch_semi, p.pitch_mix = True, semis, 1.0
            out = np.concatenate([p._fx_pitch(sig[i:i + BLOCK]) for i in range(0, len(sig), BLOCK)])[SR // 2:SR]
            want = base * 2 ** (semis / 12)
            cents = 1200 * np.log2(_f0(out) / want)
            assert abs(cents) < 15, f'{base} Hz {semis:+d} semitones: {cents:+.0f} cents off'
            ratio = np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(sig ** 2))
            assert 0.4 < ratio < 1.3, 'the level should stay close to the input'
    # the cents control fine-tunes: +50 cents at 0 semitones
    p = PedalChain(sr=SR); p.pitch_on, p.pitch_semi, p.pitch_cents, p.pitch_mix = True, 0, 50, 1.0
    out = np.concatenate([p._fx_pitch(sig[i:i + BLOCK]) for i in range(0, len(sig), BLOCK)])[SR // 2:SR]
    assert abs(1200 * np.log2(_f0(out) / base) - 50) < 15


def test_pitch_shifter_is_stable_and_does_not_depend_on_the_block_size():
    sig = _SIGNAL[:SR].astype(np.float64)
    def run(block):
        p = PedalChain(sr=SR); p.pitch_on, p.pitch_semi, p.pitch_level2, p.pitch_semi2 = True, 12, 0.5, -5
        return np.concatenate([p._fx_pitch(sig[i:i + block]) for i in range(0, len(sig), block)])
    a, b, c = run(256), run(100), run(1024)
    assert_sane(a, 'pitch shifter')
    n = min(len(a), len(b), len(c))
    assert np.allclose(a[:n], b[:n], atol=1e-6) and np.allclose(a[:n], c[:n], atol=1e-6)


def test_pitch_mix_zero_and_off_leave_the_sound_alone():
    sig = _SIGNAL[:SR // 2].astype(np.float64)
    p = PedalChain(sr=SR)
    assert np.array_equal(p._fx_pitch(sig), sig)                       # off
    p.pitch_on, p.pitch_mix = True, 0.0
    assert np.array_equal(p._fx_pitch(sig), sig)                       # on, but all dry


def test_shimmer_reverb_has_a_tail_with_octave_up_content_and_stays_stable():
    t = np.arange(SR // 4) / SR
    burst = np.concatenate((0.4 * np.sin(2 * np.pi * 330.0 * t), np.zeros(SR * 3)))
    def run(rtype, block):
        p = PedalChain(sr=SR); p.reverb_on, p.reverb_type, p.reverb_mix, p.reverb_decay = True, rtype, 1.0, 0.7
        return np.concatenate([p._fx_reverb(burst[i:i + block].astype(np.float64)) for i in range(0, len(burst), block)])
    sh, cl = run('shimmer', 256), run('cloud', 256)
    assert_sane(sh, 'shimmer reverb')
    tail = slice(SR, SR * 2)
    assert np.sqrt(np.mean(sh[tail] ** 2)) > 1e-3, 'the shimmer tail should ring on'
    assert np.sqrt(np.mean(sh[-SR // 2:] ** 2)) < np.sqrt(np.mean(sh[tail] ** 2)), 'and fade, not build up'
    def band(x, lo, hi):
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return sp[(f > lo) & (f < hi)].sum()
    up_sh = band(sh[tail], 600, 720) / band(sh[tail], 280, 380)          # octave above 330 Hz vs the note itself
    up_cl = band(cl[tail], 600, 720) / band(cl[tail], 280, 380)
    assert up_sh > up_cl * 3, 'the shimmer tail should carry much more octave-up energy than a plain cloud tail'
    other = run('shimmer', 100)
    n = min(len(sh), len(other))
    assert np.allclose(sh[:n], other[:n], atol=1e-6), 'block size must not change the shimmer reverb'


def test_orders_saved_before_the_pitch_shifter_get_it_ahead_of_the_fuzz():
    old_default = ['comp', 'wah', 'fuzz', 'dist', 'od', 'phaser', AMP_MARKER, 'chorus', 'tremolo', 'rotary', 'delay', 'reverb']
    assert normalize_order(old_default) == DEFAULT_PEDAL_ORDER
    custom = ['wah', 'comp', 'od', 'fuzz', 'dist', 'phaser', AMP_MARKER, 'rotary', 'tremolo', 'chorus', 'delay', 'reverb']
    out = normalize_order(custom)
    assert out.index('pitch') == out.index('fuzz') - 1 and [c for c in out if c != 'pitch'] == custom


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'PASS {fn.__name__}')
    print(f'\n{len(tests)} tests passed.')
