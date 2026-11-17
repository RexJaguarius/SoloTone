#!/usr/bin/env python3
"""
Pedal-effects chain around the NAM amp — compressor, wah, fuzz, distortion,
overdrive/boost (Tubescreamer, Klon, Blues Driver, clean boost), phaser,
chorus/flanger/vibrato/doubler, tremolo, delay, reverb (room, plate,
spring, cloud).

Split out of solotone.py into its own module: like nam_engine.py, this
has zero Tkinter/App coupling (plain numpy in, numpy out), so it can be
imported, profiled, or unit-tested in isolation from the rest of the
app — exactly what the build-75 Fuzz/Distortion artifacting fix relied
on (capturing real guitar audio and running it straight through
PedalChain's own methods, block by block, outside the GUI entirely).

Public API: PedalChain, DEFAULT_PEDAL_ORDER, AMP_MARKER, normalize_order. Everything else here
(_SVF, Oversampler, DCBlocker, _hard_clip/_soft_clip/_asymmetric_clip,
_muff_tone) is an internal implementation detail PedalChain itself
uses — nothing outside this module ever referenced them directly, so
they aren't re-exported.
"""

import math

import numpy as np

try:
    import scipy.signal as spsig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Matches solotone.SAMPLE_RATE — duplicated here (not imported) so this
# module has no back-reference to solotone.py at all, the same
# independence nam_engine.py has. Every real call site passes its own
# sr explicitly anyway; this is only ever a fallback default.
SAMPLE_RATE = 44100

class _SVF:
    """State-variable filter — used for wah and tone controls."""
    def __init__(self, sr=44100):
        self.sr = sr
        self.lp = self.bp = self.hp = 0.0

    def process_sample(self, x, fc, q):
        f  = 2.0 * math.sin(math.pi * min(fc, self.sr * 0.49) / self.sr)
        q  = max(0.01, q)
        lp = self.lp + f * self.bp
        hp = x - lp - (1.0 / q) * self.bp
        bp = f * hp + self.bp
        self.lp, self.bp, self.hp = lp, bp, hp
        return lp, bp, hp

    def block_bp(self, x, fc, q):
        out = np.zeros_like(x)
        for i in range(len(x)):
            _, bp, _ = self.process_sample(float(x[i]), fc, q)
            out[i] = bp
        return out

    def block_lp(self, x, fc, q):
        out = np.zeros_like(x)
        for i in range(len(x)):
            lp, _, _ = self.process_sample(float(x[i]), fc, q)
            out[i] = lp
        return out


def _soft_clip(x, drive):
    """Smooth tanh-based soft clip with drive."""
    return np.tanh(x * drive) / math.tanh(drive)


def _asymmetric_clip(x, drive):
    """Asymmetric clip approximating germanium transistor fuzz (Fuzz Face)
    and the MXR Distortion+'s asymmetric diode pair. Positive half: soft
    tanh. Negative half: harder clip at -0.7. The DC bias this
    intentionally introduces (same mechanism a real mismatched-diode
    clipper has) is removed afterward by a DC-blocking filter at the
    call site (PedalChain._fx_fuzz/_fx_dist) rather than here — see the
    comment there for why."""
    pos = np.where(x >= 0, np.tanh(x * drive * 1.4) / math.tanh(drive * 1.4), x)
    neg = np.where(x <  0, np.clip(x * drive * 0.9, -0.75, 0.0), pos)
    # blend
    out = np.where(x >= 0, pos, neg)
    return out


class DCBlocker:
    """Single-pole DC-blocking high-pass (y[n] = x[n] - x[n-1] + R*y[n-1]),
    the same role a coupling capacitor plays after a real asymmetric clip
    stage (Fuzz Face, MXR Distortion+) — removes the DC bias the
    asymmetry deliberately introduces before it eats into the final
    clip's headroom asymmetrically.
    Why this matters: _asymmetric_clip's negative half is a harder,
    lower-ceiling clip than its tanh-shaped positive half (by design —
    that asymmetry IS the fuzz/diode character), which pushes the
    signal's average upward as drive increases. Left uncorrected, that
    growing positive DC bias means the positive side of the waveform
    hits the chain's final symmetric ±1 clip sooner and more often than
    the negative side, which still has slack headroom going unused —
    so harder playing / higher drive clips the attack transients flatter
    and flatter on one side only, read directly as pedals 'petering out'
    or losing punch at higher drive rather than getting more aggressive.
    A real analog circuit solves exactly this with a coupling capacitor
    right after the clipping diodes, for the same reason; this is that
    capacitor's digital equivalent. R very close to 1 keeps the cutoff
    very low (a few Hz) so it only removes slow DC drift, not any real
    low-end guitar content."""
    def __init__(self, r=0.9995):
        self.r = r
        self.x1 = 0.0
        self.y1 = 0.0

    def process(self, x):
        out = np.empty_like(x)
        x1, y1, r = self.x1, self.y1, self.r
        for i in range(len(x)):
            y1 = x[i] - x1 + r * y1
            x1 = x[i]
            out[i] = y1
        self.x1, self.y1 = x1, y1
        return out


def _hard_clip(x, threshold=0.5):
    return np.clip(x, -threshold, threshold) / threshold


class Oversampler:
    """Wraps a nonlinear waveshaping function (clipping) with stateful 4x
    oversampling, to suppress the aliasing naive digital clipping causes.
    A hard (or even a steep soft) clip generates harmonics far above the
    input frequency; any that exceed Nyquist fold back down as
    inharmonic, "digital"-sounding noise that gets worse with both drive
    and input frequency — reported directly against the Fuzz pedal
    ("digital noise... especially past half" drive) and confirmed by
    measurement: spectral energy at non-harmonic bins from a driven Big
    Muff clip dropped 5-20x with this fix in place (bigger drop at
    higher drive, where there's more aliasing to suppress in the first
    place). Running the nonlinearity at 4x the sample rate pushes
    generated harmonics proportionally further from the (now 4x higher)
    Nyquist before they're filtered and decimated back down, so far
    fewer of them fold back into the audible band. Both the anti-imaging
    (upsample) and anti-aliasing (downsample) filters carry their own
    state between calls, so there's no discontinuity at block
    boundaries — the same zi-carrying pattern the rest of this file's
    stateful filters (BiquadState, HumFilter) already use."""
    def __init__(self, factor=4, sr=SAMPLE_RATE):
        self.factor = factor
        if HAS_SCIPY:
            # Cutoff at the original Nyquist, expressed as a fraction of
            # the oversampled rate's (factor times higher) own Nyquist —
            # 0.9x for a safety margin below the true 1/factor point.
            self._sos = spsig.butter(4, 0.9 / factor, output='sos')
            self._zi_up = spsig.sosfilt_zi(self._sos)
            self._zi_dn = spsig.sosfilt_zi(self._sos)

    def process(self, x, fn):
        """Run `fn` (a memoryless nonlinearity) on `x` at `factor`x the
        sample rate. Falls back to calling fn(x) directly if scipy isn't
        installed — same behavior as before this existed."""
        if not HAS_SCIPY:
            return fn(x)
        n = len(x)
        up = np.zeros(n * self.factor, dtype=np.float64)
        up[::self.factor] = np.asarray(x, dtype=np.float64) * self.factor
        up, self._zi_up = spsig.sosfilt(self._sos, up, zi=self._zi_up)
        shaped = fn(up)
        down, self._zi_dn = spsig.sosfilt(self._sos, shaped, zi=self._zi_dn)
        return down[::self.factor]


def _muff_tone(x, mode, sr=44100, zi=None):
    """Big Muff tonestack — three fixed positions on the tone pot.
    Models the high-pass / low-pass blend network.
    full  = tone wide open (bright)
    flat  = tone centred (relatively even)
    scoop = mid scoop (~500 Hz notch, prominent bass+treble)

    Returns (y, new_zi) — the caller must carry new_zi into the next
    call (PedalChain does, via self._muff_zi). Every branch used to call
    lfilter with no zi at all, which defaults to zero initial state on
    every call — fine for a one-shot offline filter, but this runs once
    per audio block (every ~5.8ms at a 256-sample block size), so it was
    resetting the filter's memory at every block boundary instead of
    carrying it forward, the same discontinuity bug fixed in the
    'metal' distortion notch (see PedalChain._fx_dist) — measured there
    at a ~0.1 amplitude jump per boundary, audible as buzzing. 'scoop'
    additionally ran the exact same lfilter call twice (once with
    zi=zeros(2) whose result was thrown away, then again with no zi at
    all) — dead code on top of the same state bug, not a second
    computation of anything real.
    """
    if mode == 'full':
        # Tone pot fully clockwise: emphasise highs via simple HP shelf
        fc = 2000.0
        w0 = 2 * np.pi * fc / sr
        cw = math.cos(w0); sw = math.sin(w0)
        A  = 10 ** (8.0 / 40.0)   # +8 dB shelf
        al = sw / (2 * 0.707)
        b0 =  A * ((A+1) + (A-1)*cw + 2*math.sqrt(A)*al)
        b1 = -2*A*((A-1) + (A+1)*cw)
        b2 =  A * ((A+1) + (A-1)*cw - 2*math.sqrt(A)*al)
        a0 =      (A+1) - (A-1)*cw + 2*math.sqrt(A)*al
        a1 =  2 * ((A-1) - (A+1)*cw)
        a2 =      (A+1) - (A-1)*cw - 2*math.sqrt(A)*al
        b = np.array([b0,b1,b2]) / a0
        a = np.array([1.0, a1/a0, a2/a0])
        if not HAS_SCIPY:
            return x, zi  # fallback: bypass
        if zi is None: zi = np.zeros(2)
        y, zi = spsig.lfilter(b, a, x.astype(np.float64), zi=zi)
        return y.astype(np.float32), zi

    elif mode == 'scoop':
        # Peak notch around 700 Hz  (~-10 dB)
        fc_n, gain_n, Q_n = 700.0, -10.0, 1.4
        w0 = 2*np.pi*fc_n/sr; cw = math.cos(w0); sw = math.sin(w0)
        A  = 10**(gain_n/40.0); al = sw/(2*Q_n)
        b  = np.array([1+al*A, -2*cw, 1-al*A])
        a  = np.array([1+al/A, -2*cw, 1-al/A])
        if not HAS_SCIPY:
            return x, zi
        if zi is None: zi = np.zeros(2)
        y, zi = spsig.lfilter(b/a[0], a/a[0], x.astype(np.float64), zi=zi)
        return y.astype(np.float32), zi

    else:  # flat — gentle mid-presence cut, centre position
        if not HAS_SCIPY:
            return x, zi
        fc, gain, Q = 900.0, -4.0, 0.9
        w0 = 2*np.pi*fc/sr; cw = math.cos(w0); sw = math.sin(w0)
        A  = 10**(gain/40.0); al = sw/(2*Q)
        b  = np.array([1+al*A, -2*cw, 1-al*A])
        a_ = np.array([1+al/A, -2*cw, 1-al/A])
        if zi is None: zi = np.zeros(2)
        y, zi = spsig.lfilter(b/a_[0], a_/a_[0], x.astype(np.float64), zi=zi)
        return y.astype(np.float32), zi


# ── reverb engines (plate, cloud, spring) ───────────────────────────
#
# The original room reverb loops over every sample in Python. These three
# are built so the feedback loops run as numpy vectors instead: every loop
# goes through a delay of at least one "chunk" (_CHUNK samples), and
# reading a delay line D samples back for a chunk of m <= D samples gives
# exactly the same numbers as doing it one sample at a time. So the output
# does not depend on how the audio callback happens to slice the stream.

_CHUNK = 128


def _chunks(n, size):
    pos = 0
    while pos < n:
        take = min(size, n - pos)
        yield pos, pos + take
        pos += take


class _Line:
    """Circular delay line with block read/write. Always read before you
    write within a chunk, and never ask for a delay shorter than the chunk."""
    def __init__(self, size):
        self.n = int(size) + _CHUNK * 2 + 8
        self.buf = np.zeros(self.n)
        self.wp = 0

    def read(self, d, m):
        return np.take(self.buf, np.arange(self.wp - d, self.wp - d + m), mode='wrap')

    def write(self, v):
        m = len(v)
        self.buf[np.arange(self.wp, self.wp + m) % self.n] = v
        self.wp = (self.wp + m) % self.n


def _allpass(line, d, g, x):
    """Schroeder allpass through a delay line (unity gain at every frequency)."""
    vd = line.read(d, len(x))
    v = x - g * vd
    line.write(v)
    return g * v + vd


class _OnePole:
    """y[n] = (1-k) x[n] + k y[n-1], state kept across blocks."""
    def __init__(self):
        self.z = 0.0

    def process(self, x, k):
        if HAS_SCIPY:
            y, zf = spsig.lfilter([1.0 - k], [1.0, -k], x, zi=[self.z * k])
            self.z = float(y[-1]) if len(y) else self.z
            return y
        y = np.empty(len(x)); z = self.z
        for i in range(len(x)):
            z = (1.0 - k) * x[i] + k * z
            y[i] = z
        self.z = z
        return y


class _PlateEngine:
    """Plate reverb after Dattorro's 1997 design: four input diffusers into
    two cross-coupled tank loops, each with a damping filter in the loop.
    Output is a mono sum of taps taken along the tank delays. `big=True` is
    the Cloud voicing: the same network with longer tank delays, a longer
    decay range and extra input diffusion for a washed, ambient tail."""
    _IN = ((142, 0.75), (107, 0.75), (379, 0.625), (277, 0.625))
    _TANK = {'l': (672, 4453, 1800, 3720), 'r': (908, 4217, 2656, 3163)}   # AP1, delay1, AP2, delay2 (at 29761 Hz)
    _MAX_SCALE = 1.5

    def __init__(self, sr, big=False):
        self.big = big
        s = sr / 29761.0
        self.tank_s = s * (1.6 if big else 1.0)
        self.inap = [(_Line(int(L * s) + 4), int(L * s), g) for L, g in self._IN]
        if big:
            self.inap += [(_Line(int(L * s) + 4), int(L * s), 0.5) for L in (211, 163)]
        self.br = {}
        for k, (a1, d1, a2, d2) in self._TANK.items():
            cap = lambda v: _Line(int(v * self.tank_s * self._MAX_SCALE) + 4)
            self.br[k] = dict(ap1=cap(a1), d1=cap(d1), ap2=cap(a2), d2=cap(d2),
                              base=(a1, d1, a2, d2), lp=_OnePole())
        self.lp_in = _OnePole()

    def process(self, x, size, decay, damp):
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        out = np.zeros(n)
        scale = 0.5 + max(0.1, min(1.0, size))
        dec = decay = max(0.0, min(1.0, decay))
        # loop gain per trip round the tank, chosen so the decay control spans about
        # 0.6 s - 6 s (plate) and 2 s - 15 s (cloud)
        g = min(0.94, (0.32 + 0.60 * dec) if self.big else (0.12 + 0.82 * dec ** 1.2))
        k = 0.05 + 0.85 * max(0.0, min(1.0, damp))
        # delays in samples for this size setting
        D = {}
        for key, b in self.br.items():
            a1, d1, a2, d2 = (int(v * self.tank_s * scale) for v in b['base'])
            D[key] = (a1, d1, a2, d2)
        for lo, hi in _chunks(n, _CHUNK):
            xc = self.lp_in.process(x[lo:hi], 0.15) * 0.5
            m = hi - lo
            for line, d, gg in self.inap:
                xc = _allpass(line, d, gg, xc)
            L, R = self.br['l'], self.br['r']
            (la1, ld1, la2, ld2), (ra1, rd1, ra2, rd2) = D['l'], D['r']
            # everything that has to be read from the past this chunk, before any writes
            l_fb = L['d2'].read(ld2, m)
            r_fb = R['d2'].read(rd2, m)
            l_d1 = L['d1'].read(ld1, m)
            r_d1 = R['d1'].read(rd1, m)
            taps = (L['d1'].read(int(ld1 * 0.15), m), L['d1'].read(int(ld1 * 0.70), m),
                    L['d2'].read(int(ld2 * 0.40), m), R['d1'].read(int(rd1 * 0.20), m),
                    R['d1'].read(int(rd1 * 0.80), m), R['d2'].read(int(rd2 * 0.55), m))
            for key, fb, d1out, (a1, d1, a2, d2) in (('l', r_fb, l_d1, D['l']), ('r', l_fb, r_d1, D['r'])):
                b = self.br[key]
                a = _allpass(b['ap1'], a1, 0.7, xc + g * fb)
                b['d1'].write(a)
                v = b['lp'].process(d1out, k) * g
                v = _allpass(b['ap2'], a2, 0.5, v)
                b['d2'].write(np.clip(v, -8.0, 8.0))
            # output level set so the wet signal sits where the room reverb's does
            out[lo:hi] = (1.45 if self.big else 1.6) * (taps[0] + taps[1] - taps[2] + taps[3] - taps[4] + taps[5])
        return out


class _SpringEngine:
    """Spring tank: three springs of different length in parallel. Each is a
    delay loop with a long chain of identical first-order allpass sections
    (the dispersion that makes a spring 'boing' and chirp) and a low-pass
    for the dull top end, fed back on itself."""
    _LENS = (1213, 1511, 1763)          # samples at 44.1 kHz (27.5, 34.3, 40 ms)
    _SECTIONS = 12                      # 12 second-order sections = 24 first-order allpasses

    def __init__(self, sr):
        s = sr / 44100.0
        self.lens = [int(v * s) for v in self._LENS]
        self.lines = [_Line(int(v * 1.5)) for v in self.lens]
        self.lps = [_OnePole() for _ in self.lens]
        self.coef = (0.60, 0.64, 0.68)
        self.sos = [np.tile([a * a, 2 * a, 1.0, 1.0, 2 * a, a * a], (self._SECTIONS, 1)) for a in self.coef]
        self.zi = [np.zeros((self._SECTIONS, 2)) for _ in self.lens]

    def process(self, x, size, decay, damp):
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        out = np.zeros(n)
        scale = 0.6 + 0.8 * max(0.0, min(1.0, size))
        # feedback per trip along the spring: tail of about 0.4 s - 1 s - 4 s at decay 0 / 0.1 / 0.9
        fbg = float(np.interp(max(0.0, min(1.0, decay)), (0.0, 0.1, 0.5, 0.9, 1.0), (0.55, 0.62, 0.855, 0.943, 0.955)))
        k = 0.25 + 0.60 * max(0.0, min(1.0, damp))
        for lo, hi in _chunks(n, 256):
            xc = x[lo:hi] * 0.5
            m = hi - lo
            wet = np.zeros(m)
            for i, line in enumerate(self.lines):
                d = max(m, int(self.lens[i] * scale))
                fb = line.read(d, m)
                u = xc + fbg * fb
                u, self.zi[i] = spsig.sosfilt(self.sos[i], u, zi=self.zi[i])
                u = self.lps[i].process(u, k)
                line.write(np.clip(u, -8.0, 8.0))
                wet += fb
            out[lo:hi] = wet * 2.6          # level set to sit where the room reverb's wet signal does
        return out


class _PitchEngine:
    """Pitch shifter: two read taps sweep along a delay line and the sound is crossfaded between them, so
    each tap is a short grain that is read faster or slower than it was written. Reading faster (ratio > 1)
    raises the pitch, slower lowers it. The two taps sit half a sweep apart and fade with sin^2 / cos^2, which
    sum to one, so the level stays constant. The read position never gets closer than one chunk to the write
    position, which keeps the output the same whatever block size the audio callback uses. Adds about 3 ms of
    delay. Good for octaves, fifths, detune and whammy-style bends; held chords get a slight roughness, as with
    any delay-line shifter."""
    _GRAIN = 2048                       # samples at 44.1 kHz (about 46 ms)

    def __init__(self, sr):
        self.L = max(256, int(self._GRAIN * sr / 44100.0))
        self.base = _CHUNK + 16
        self.line = _Line(self.L + self.base + 8)
        self.phase = 0.0

    def process(self, x, ratio):
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        out = np.zeros(n)
        inc = (1.0 - float(ratio)) / self.L
        for lo, hi in _chunks(n, _CHUNK):
            m = hi - lo
            p1 = (self.phase + inc * np.arange(1, m + 1)) % 1.0
            self.phase = float(p1[-1])
            p2 = (p1 + 0.5) % 1.0
            wp = self.line.wp
            idx = np.arange(m)
            acc = np.zeros(m)
            for p in (p1, p2):
                pos = wp + idx - (self.base + self.L * p)
                i0 = np.floor(pos).astype(np.int64)
                fr = pos - i0
                b = self.line.buf
                v = b.take(i0, mode='wrap') * (1.0 - fr) + b.take(i0 + 1, mode='wrap') * fr
                acc += v * np.sin(np.pi * p) ** 2
            self.line.write(x[lo:hi])
            out[lo:hi] = acc
        return out


class _ShimmerEngine:
    """Shimmer reverb: the Cloud plate with an octave-up pitch shifter in its feedback. What comes out of the
    tail is shifted up an octave and fed back in, so each trip round the loop climbs another octave and
    fades, giving the choir-like rising wash. The feedback is held back by 256 samples so it never depends
    on the block size."""
    _FB_DELAY = _CHUNK * 2

    def __init__(self, sr):
        self.plate = _PlateEngine(sr, big=True)
        self.pitch = _PitchEngine(sr)
        self.fb = _Line(self._FB_DELAY + _CHUNK)
        self.lp = _OnePole()

    def process(self, x, size, decay, damp, shimmer, ratio=2.0):
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        out = np.zeros(n)
        g = 0.55 * max(0.0, min(1.0, shimmer))
        for lo, hi in _chunks(n, _CHUNK):
            m = hi - lo
            xin = x[lo:hi] + g * self.fb.read(self._FB_DELAY, m)
            wet = self.plate.process(xin, size, decay, damp)
            sh = self.pitch.process(wet, ratio)
            self.fb.write(np.tanh(self.lp.process(sh, 0.25)))     # tame the top end and any build-up
            out[lo:hi] = wet
        return out


# Single source of truth for the pedal chain's factory order — used both
# to initialize a new PedalChain and by the Pedals tab's "Reset Order"
# button, so the two can't silently drift apart the way a duplicated
# literal could. Also the order MIDI_MAPPABLE_TARGETS' pedal-related
# entries are listed in, so the Settings tab's MIDI list reads in the
# same sequence as the actual signal chain.
#
# 'amp' is not a pedal: it's a marker for where the NAM amp + cab IR + EQ
# sit in the chain. Everything BEFORE it runs ahead of the amp
# (PedalChain.process — drive/fuzz/wah/compressor, which want to hit the
# amp's input), everything AFTER it runs on the amp's output
# (PedalChain.process_post — time-based effects, which on a real rig sit
# in the effects loop or on the board after the amp; run ahead of a
# distorted amp, delay repeats and reverb tails get crushed and smeared
# by it). Default: chorus, delay and reverb after the amp.
AMP_MARKER = 'amp'
DEFAULT_PEDAL_ORDER = ['comp', 'wah', 'pitch', 'fuzz', 'dist', 'od', 'phaser',
                       AMP_MARKER,
                       'chorus', 'tremolo', 'rotary', 'delay', 'reverb']

# Pedals added after the first 8: a saved order made before they existed
# has no entry for them, so normalize_order() slots them in (phaser just
# ahead of the amp, tremolo after the chorus, rotary after the tremolo). They
# are off by default, so an old profile still sounds exactly as it did.
_NEWER_PEDALS = ('phaser', 'tremolo', 'rotary', 'pitch')

# The previous factory orders (before phaser/tremolo, and before the rotary):
# upgraded to the new default when they were never customised.
_PREVIOUS_DEFAULT_ORDER = ['comp', 'wah', 'fuzz', 'dist', 'od',
                           AMP_MARKER,
                           'chorus', 'delay', 'reverb']
_PREVIOUS_DEFAULT_ORDERS = [
    _PREVIOUS_DEFAULT_ORDER,
    ['comp', 'wah', 'fuzz', 'dist', 'od', 'phaser', AMP_MARKER, 'chorus', 'tremolo', 'delay', 'reverb'],
    ['comp', 'wah', 'fuzz', 'dist', 'od', 'phaser', AMP_MARKER, 'chorus', 'tremolo', 'rotary', 'delay', 'reverb'],
]

# The order before the amp marker existed — every pedal ahead of the amp.
# Kept so normalize_order() can tell "a saved order the user never
# customised" (upgrade it to the new default) from "a custom order"
# (keep it exactly as it was, all pedals pre-amp, so a saved profile
# sounds the same as it always did).
_LEGACY_DEFAULT_ORDER = ['comp', 'wah', 'fuzz', 'dist', 'od', 'chorus', 'delay', 'reverb']


def normalize_order(order):
    """Validate a chain order from a saved profile/session and bring it up
    to the current format. Returns the list to use, or None if it isn't a
    valid permutation of the pedals. An order without the 'amp' marker
    (anything saved before post-amp effects existed) keeps every pedal
    ahead of the amp, exactly as it was — except the untouched factory
    orders, which are upgraded to the new default. An order saved before
    the phaser and tremolo existed gets them added in their default spots."""
    if not isinstance(order, list) or order.count(AMP_MARKER) > 1:
        return None
    pedals = [c for c in order if c != AMP_MARKER]
    known = [c for c in DEFAULT_PEDAL_ORDER if c != AMP_MARKER]
    if len(set(pedals)) != len(pedals) or any(c not in known for c in pedals):
        return None
    required = [c for c in known if c not in _NEWER_PEDALS]
    if any(c not in pedals for c in required):
        return None
    if order == _LEGACY_DEFAULT_ORDER or order in _PREVIOUS_DEFAULT_ORDERS:
        return list(DEFAULT_PEDAL_ORDER)
    out = list(order) if AMP_MARKER in order else list(order) + [AMP_MARKER]
    if 'phaser' not in out:
        out.insert(out.index(AMP_MARKER), 'phaser')
    if 'tremolo' not in out:
        i = out.index('chorus') + 1 if ('chorus' in out and out.index('chorus') > out.index(AMP_MARKER)) \
            else out.index(AMP_MARKER) + 1
        out.insert(i, 'tremolo')
    if 'rotary' not in out:
        i = out.index('tremolo') + 1 if out.index('tremolo') > out.index(AMP_MARKER) else out.index(AMP_MARKER) + 1
        out.insert(i, 'rotary')
    if 'pitch' not in out:                       # ahead of the fuzz, as a whammy sits on a board
        amp = out.index(AMP_MARKER)
        out.insert(out.index('fuzz') if out.index('fuzz') < amp else out.index('phaser'), 'pitch')
    return out


class PedalChain:
    """
    Effects chain around the NAM amp, processed sample-accurately in the amp
    callback: process() runs the pedals ahead of the amp marker in
    self.order, process_post() the ones after it.
    All parameters are plain Python attributes — set from the GUI thread;
    reads in the audio thread are safe because Python attribute access is
    atomic for simple types.

    Default signal order:
        [mute] → compressor → wah → fuzz → distortion → overdrive/boost
               → AMP + CAB + EQ → chorus/flanger → delay → reverb → output
    """
    def __init__(self, sr=44100):
        self.sr = sr

        # ── global ───────────────────────────────────────────
        self.mute         = False

        # ── chain order ──────────────────────────────────────
        # User-reorderable stomp order, including the 'amp' marker that
        # splits pre-amp from post-amp pedals (mute is a global kill
        # switch, not part of the chain, so it isn't in this list).
        # Rearranged from the Pedals tab with the Move up/down buttons.
        self.order        = list(DEFAULT_PEDAL_ORDER)

        # ── compressor (tube-style) ────────────────────────────
        self.comp_on      = False
        self.comp_thresh  = -18.0    # dB
        self.comp_ratio   = 4.0      # N:1
        self.comp_attack  = 5.0      # ms
        self.comp_release = 80.0     # ms
        self.comp_makeup  = 6.0      # dB
        self.comp_warmth  = 0.35     # 0-1 — tube-style saturation blend
        self._comp_env    = 1e-6     # smoothed peak envelope (linear)

        # ── wah ──────────────────────────────────────────────
        self.wah_on        = False
        self.wah_mode      = 'manual'   # 'manual' | 'auto'
        self.wah_range_lo  = 400.0      # Hz — sweep/manual range floor
        self.wah_range_hi  = 2200.0     # Hz — sweep/manual range ceiling
        self.wah_pos       = 0.35       # 0-1 position within range (manual mode)
        self.wah_q         = 4.0
        self.wah_auto_rate  = 2.5       # Hz — auto-sweep speed
        self.wah_auto_depth = 1.0       # 0-1 — auto-sweep intensity (fraction of range)
        # Envelope mode: the filter follows how hard you play (like UltraRig's Envelope Filter)
        self.wah_env_sens   = 0.6       # 0-1 — how far picking dynamics push the filter
        self.wah_env_decay  = 0.4       # 0-1 — how slowly it falls back (40 ms .. 800 ms)
        self.wah_env_type   = 'wah'     # 'wah' | 'bandpass' | 'lowpass' | 'highpass'
        self.wah_env_dir    = 'up'      # 'up' (loud opens the filter) | 'down' (loud closes it)
        self._wah_env       = 0.0       # envelope follower state, carried across blocks
        self.wah_freq       = 800.0     # live Hz value, recomputed every block
        self._wah_svf       = _SVF(sr)
        self._wah_auto_phase = 0.0

        # ── fuzz ─────────────────────────────────────────────
        self.fuzz_on      = False
        self.fuzz_mode    = 'fuzz_face'   # 'fuzz_face' | 'tone_bender' | 'big_muff'
        self.fuzz_drive   = 5.0           # 1–10
        self.fuzz_vol     = 0.7
        self.muff_tone    = 'scoop'       # 'full' | 'flat' | 'scoop'
        self._fuzz_os     = Oversampler(4, sr)   # anti-aliasing for the clip stage
        self._muff_zi     = None   # _muff_tone()'s biquad state, carried across blocks
        self._fuzz_dc     = DCBlocker()   # removes _asymmetric_clip's DC bias (fuzz_face / tone_bender)
        self._tb_hp_zi    = None          # Tone Bender: input high-pass and mid-hump filter state
        self._tb_mid_zi   = None

        # ── overdrive / boost ─────────────────────────────────
        self.od_on        = False
        self.od_mode      = 'tubescreamer'  # 'tubescreamer' | 'boost' | 'klon' | 'bd2'
        self.od_drive     = 4.0
        self.od_vol       = 0.8
        self._od_svf      = _SVF(sr)        # for TS mid-hump
        self._od_zi       = [np.zeros(1), np.zeros(1)]  # HP in, HP out
        self._od_zi_klon  = np.zeros(1)     # Klon's treble-boost filter
        self._od_zi_bd2   = np.zeros(1)     # Blues Driver's presence filter
        self._od_dc       = DCBlocker()     # Blues Driver's asymmetric clip leaves a DC offset

        # ── distortion ─────────────────────────────────────────
        self.dist_on      = False
        self.dist_mode    = 'ds1'   # 'ds1' | 'rat' | 'dist_plus' | 'metal'
        self.dist_drive   = 5.0     # 1-10
        self.dist_tone    = 0.5     # 0 (dark) - 1 (bright)
        self.dist_vol     = 0.7
        self._dist_zi     = np.zeros(1)   # 1st-order tone LP filter state
        self._dist_notch_zi = np.zeros(2)   # 'metal' mid-scoop notch filter state
        self._dist_os     = Oversampler(4, sr)   # anti-aliasing for the clip stage
        self._dist_dc     = DCBlocker()   # removes _asymmetric_clip's DC bias (dist_plus only)

        # ── phaser ───────────────────────────────────────────
        self.phaser_on       = False
        self.phaser_rate     = 0.5       # Hz
        self.phaser_depth    = 0.6       # 0-1, how wide the sweep is
        self.phaser_feedback = 0.3       # 0-0.9, resonance of the notches
        self._ph_phase       = 0.0
        self._ph_z           = [0.0, 0.0, 0.0, 0.0]   # the four allpass stages' states
        self._ph_last        = 0.0       # last output of the stage chain, for the feedback

        # ── tremolo ──────────────────────────────────────────
        self.tremolo_on    = False
        self.tremolo_rate  = 5.0         # Hz
        self.tremolo_depth = 0.5         # 0-1
        self.tremolo_shape = 0.0         # 0 = smooth sine ... 1 = choppy, square-ish
        self._trem_phase   = 0.0

        # ── rotary speaker (Leslie) ──────────────────────────
        self.rotary_on     = False
        self.rotary_speed  = 'slow'      # 'slow' | 'fast' | 'brake'
        self.rotary_slow   = 0.8         # Hz — horn speed on Slow (the drum turns a little slower)
        self.rotary_fast   = 6.7         # Hz — horn speed on Fast
        self.rotary_depth  = 0.6         # 0-1 — how much the swirl moves the sound
        self.rotary_ramp   = 0.5         # 0-1 — how long the rotors take to speed up / slow down
        self.rotary_mix    = 1.0         # 0-1 — dry / wet
        self._rot_ph       = [0.0, 0.0]  # horn, drum rotor angle (cycles)
        self._rot_rate     = [0.0, 0.0]  # horn, drum current speed (Hz): they coast between speeds
        self._rot_hist     = [np.zeros(96), np.zeros(96)]   # delay-line memory per rotor
        self._rot_lp_z     = np.zeros(1)
        self._rot_first    = True

        # ── pitch shifter ─────────────────────────────────────
        self.pitch_on     = False
        self.pitch_semi   = 12           # voice 1 interval in semitones, -12..+12
        self.pitch_cents  = 0            # fine tune for voice 1, -50..+50 cents
        self.pitch_semi2  = 7            # voice 2 interval in semitones
        self.pitch_level2 = 0.0          # 0-1, level of voice 2 (0 = off)
        self.pitch_mix    = 0.5          # 0 = dry ... 1 = shifted voices only
        self._pitch_eng   = [None, None]
        self._pitch_dry   = np.zeros(0)

        # ── chorus / flanger / vibrato / doubler ─────────────
        self.chorus_on    = False
        self.chorus_mode  = 'chorus'     # 'chorus' | 'flanger' | 'vibrato' | 'doubler'
        self.chorus_rate  = 0.5          # Hz
        self.chorus_depth = 0.5          # 0–1
        self.chorus_mix   = 0.5
        self._ch_phase    = 0.0
        _max_del = int(sr * 0.030)       # 30 ms max
        self._ch_buf      = np.zeros(_max_del + 64, dtype=np.float64)
        self._ch_wp       = 0

        # ── delay ─────────────────────────────────────────────
        self.delay_on     = False
        self.delay_mode   = 'tape'       # 'tape' | 'digital'
        self.delay_bpm_link = False
        self.delay_note   = 'dotted8'    # 'dotted8' | 'quarter' | 'half'
        self.delay_time   = 0.375        # seconds (manual, 0.01–2.0)
        self.delay_feedback = 0.4
        self.delay_mix    = 0.35
        self._delay_bpm   = 120          # updated each block from metro
        _max_dly = int(sr * 2.1)
        self._dly_buf     = np.zeros(_max_dly, dtype=np.float64)
        self._dly_wp      = 0
        self._dly_lp_zi   = np.zeros(2)  # tape LP filter state

        # ── reverb (Schroeder) ────────────────────────────────
        self.reverb_on    = False
        self.reverb_type  = 'room'       # 'room' | 'plate' | 'spring' | 'cloud' | 'shimmer'
        self.reverb_mix   = 0.25
        self.reverb_size  = 0.6          # 0–1, scales delay times
        self.reverb_decay = 0.5          # 0-1, length of the tail (plate / spring / cloud; the room reverb has a fixed decay)
        self.reverb_damp  = 0.5          # 0-1, how dark the tail gets (plate / spring / cloud / shimmer)
        self.reverb_shimmer = 0.5        # 0-1, how much of the tail is fed back an octave up (shimmer only)
        self._rev_engines = {}           # built the first time each type is used
        # Comb filter delays (prime-ish lengths in samples at 44100)
        _base = [1557, 1617, 1491, 1422]
        self._comb_bufs   = [np.zeros(int(n * 2), dtype=np.float64) for n in _base]
        self._comb_wps    = [0] * 4
        self._comb_gains  = [0.84, 0.84, 0.84, 0.84]
        # Allpass delays
        _ap = [225, 341]
        self._ap_bufs     = [np.zeros(n * 2, dtype=np.float64) for n in _ap]
        self._ap_wps      = [0] * 2
        self._comb_len    = [int(n * 2) for n in _base]
        self._ap_len      = [n * 2 for n in _ap]

    # ── parameter helpers ─────────────────────────────────────

    def _delay_time_secs(self):
        if self.delay_bpm_link and self._delay_bpm > 0:
            beat = 60.0 / self._delay_bpm
            return {'dotted8': beat * 0.75,
                    'quarter':  beat,
                    'half':     beat * 2.0}[self.delay_note]
        return float(self.delay_time)

    # ── per-block processing ──────────────────────────────────

    def _fx_comp(self, y):
        """Compressor with a soft-knee gain computer (Giannoulis/Massberg/Reiss
        formula) for a musical, non-clamping response, plus an optional
        asymmetric-tanh saturation stage blended in afterward for tube-style
        warmth — the soft knee and the slow one-pole envelope follower already
        give it a rounder, less clampy feel than a hard-knee digital limiter."""
        if not self.comp_on: return y
        n = len(y)
        out = np.empty(n)
        attack_s  = max(0.1, self.comp_attack) / 1000.0
        release_s = max(1.0, self.comp_release) / 1000.0
        a_atk  = math.exp(-1.0 / (self.sr * attack_s))
        a_rel  = math.exp(-1.0 / (self.sr * release_s))
        thresh = self.comp_thresh
        ratio  = max(1.0, self.comp_ratio)
        knee   = 6.0   # dB — fixed soft-knee width
        makeup = 10 ** (self.comp_makeup / 20.0)
        env    = self._comp_env
        for i in range(n):
            x    = float(y[i])
            rect = abs(x)
            if rect > env:
                env = a_atk * env + (1 - a_atk) * rect
            else:
                env = a_rel * env + (1 - a_rel) * rect
            env_db = 20.0 * math.log10(max(env, 1e-9))
            d = env_db - thresh
            if 2 * d < -knee:
                gr_db = 0.0
            elif 2 * abs(d) <= knee:
                gr_db = (1.0/ratio - 1.0) * (d + knee/2.0)**2 / (2.0*knee)
            else:
                gr_db = (1.0/ratio - 1.0) * d
            out[i] = x * (10 ** (gr_db / 20.0)) * makeup
        self._comp_env = env
        warmth = max(0.0, min(1.0, self.comp_warmth))
        if warmth > 0:
            drive = 1.0 + warmth * 2.0
            sat = np.tanh(out * drive) / math.tanh(drive)
            sat = np.where(out >= 0, sat, sat * 0.96)   # slight asymmetry: even-harmonic tube feel
            out = out * (1 - warmth) + sat * warmth
        np.clip(out, -1, 1, out=out)
        return out

    def _fx_wah(self, y):
        if not self.wah_on: return y
        lo, hi = self.wah_range_lo, self.wah_range_hi
        if hi < lo: lo, hi = hi, lo
        if self.wah_mode == 'envelope':
            return self._fx_wah_envelope(y, lo, hi)
        if self.wah_mode == 'auto':
            # Modulated once per block (same coarseness as the chorus LFO
            # above) rather than per-sample — plenty smooth at typical
            # sweep rates given the block sizes this app runs at.
            mid  = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * max(0.0, min(1.0, self.wah_auto_depth))
            fc   = mid + half * math.sin(2 * math.pi * self._wah_auto_phase)
            self._wah_auto_phase = (self._wah_auto_phase +
                                     self.wah_auto_rate * len(y) / self.sr) % 1.0
        else:
            fc = lo + max(0.0, min(1.0, self.wah_pos)) * (hi - lo)
        self.wah_freq = fc
        y = self._wah_svf.block_bp(y, fc, self.wah_q)
        y = y * 3.0
        np.clip(y, -1, 1, out=y)
        return y

    def _fx_wah_envelope(self, y, lo, hi):
        """Envelope filter: a state-variable filter whose centre frequency follows the playing level.
        The envelope follower (fast attack, adjustable release) is a scalar loop; the filter coefficient is
        then worked out for the whole block at once, so only the filter's own recursion runs per sample."""
        n = len(y)
        if n == 0:
            return y
        sr = self.sr
        lo = max(20.0, lo)
        atk = math.exp(-1.0 / (0.004 * sr))
        rel = math.exp(-1.0 / ((0.04 + 0.76 * max(0.0, min(1.0, self.wah_env_decay))) * sr))
        env, mag = self._wah_env, np.abs(y)
        envs = np.empty(n)
        for i in range(n):
            a = mag[i]
            env = a + (env - a) * (atk if a > env else rel)
            envs[i] = env
        self._wah_env = env
        gain_in = 3.0 + 27.0 * max(0.0, min(1.0, self.wah_env_sens))
        pos = 1.0 - np.exp(-envs * gain_in)
        if self.wah_env_dir == 'down':
            pos = 1.0 - pos
        fc = lo * (max(hi, lo * 1.01) / lo) ** pos
        f = 2.0 * np.sin(np.pi * np.minimum(fc, sr * 0.49) / sr)
        kind = self.wah_env_type
        invq = 1.0 / max(0.5, self.wah_q)
        svf = self._wah_svf
        lp, bp = svf.lp, svf.bp
        out = np.empty(n)
        for i in range(n):
            lp += f[i] * bp
            hp = y[i] - lp - invq * bp
            bp += f[i] * hp
            out[i] = lp if kind == 'lowpass' else hp if kind == 'highpass' else bp
        svf.lp, svf.bp = lp, bp
        self.wah_freq = float(fc[-1])
        out *= 3.0 if kind == 'wah' else 1.5 if kind == 'bandpass' else 1.0
        np.clip(out, -1, 1, out=out)
        return out

    def _fx_fuzz(self, y):
        if not self.fuzz_on: return y
        d = max(1.0, self.fuzz_drive)
        if self.fuzz_mode == 'fuzz_face':
            y = self._fuzz_os.process(y, lambda s: _asymmetric_clip(s, d))
            y = self._fuzz_dc.process(y)
        elif self.fuzz_mode == 'tone_bender':
            # British germanium fuzz: more gain than the Fuzz Face, a tighter low end (input high-pass) and a
            # forward mid hump after the clip.
            if HAS_SCIPY:
                if self._tb_hp_zi is None:
                    self._tb_hp_zi = np.zeros(1); self._tb_mid_zi = np.zeros(2)
                a = math.exp(-2 * math.pi * 150.0 / self.sr)
                y, self._tb_hp_zi = spsig.lfilter([(1 + a) / 2, -(1 + a) / 2], [1.0, -a], y, zi=self._tb_hp_zi)
            y = self._fuzz_os.process(y, lambda s: _asymmetric_clip(s, d * 2.2))
            y = self._fuzz_dc.process(y)
            if HAS_SCIPY:
                w0 = 2 * math.pi * 1000.0 / self.sr
                A = 10 ** (6.0 / 40.0); al = math.sin(w0) / (2 * 0.9); cw = math.cos(w0)
                b = np.array([1 + al * A, -2 * cw, 1 - al * A]); aa = np.array([1 + al / A, -2 * cw, 1 - al / A])
                y, self._tb_mid_zi = spsig.lfilter(b / aa[0], aa / aa[0], y, zi=self._tb_mid_zi)
        else:  # big muff — two cascaded hard-clip stages with gain
            def _muff_clip(s):
                s = s * d * 0.3
                s = _hard_clip(s, 0.6)
                s = s * d * 0.3
                s = _hard_clip(s, 0.6)
                return s
            y = self._fuzz_os.process(y, _muff_clip)
            y32, self._muff_zi = _muff_tone(y.astype(np.float32), self.muff_tone, self.sr, self._muff_zi)
            y = y32.astype(np.float64)
        y = y * self.fuzz_vol
        np.clip(y, -1, 1, out=y)
        return y

    def _fx_od(self, y):
        if not self.od_on: return y
        if self.od_mode == 'tubescreamer':
            # HP filter before clip (removes low-end mud)
            if HAS_SCIPY:
                b, a = spsig.butter(1, 720/(self.sr/2), btype='high')
                y, self._od_zi[0] = spsig.lfilter(b, a, y, zi=self._od_zi[0])
            y = _soft_clip(y, max(1.0, self.od_drive) * 0.8)
            # LP after clip (removes harsh highs)
            if HAS_SCIPY:
                b2, a2 = spsig.butter(1, 3200/(self.sr/2), btype='low')
                y, self._od_zi[1] = spsig.lfilter(b2, a2, y, zi=self._od_zi[1])
            # mid-hump (Tubescreamer characteristic)
            y = self._od_svf.block_bp(y, 720, 0.7) * 0.6 + y * 0.4
            y = y * self.od_vol
        elif self.od_mode == 'klon':
            # Klon Centaur: a parallel clean blend plus a gently-clipped
            # path with a treble/presence lift — the "transparent" drive
            # character that thickens without scooping mids the way the
            # Tubescreamer's mid-hump does.
            drive   = max(1.0, self.od_drive)
            clean   = y
            clipped = _soft_clip(y, drive * 0.6)
            if HAS_SCIPY:
                b, a = spsig.butter(1, 2200/(self.sr/2), btype='high')
                hf, self._od_zi_klon = spsig.lfilter(b, a, y, zi=self._od_zi_klon)
                clipped = clipped + hf * 0.25
            y = clean * 0.45 + clipped * 0.55
            np.clip(y, -1, 1, out=y)
            y = y * self.od_vol
        elif self.od_mode == 'bd2':
            # Boss Blues Driver: a wide-range, two-stage gain that stays open
            # and dynamic at low settings and thickens as it goes up. The lows
            # are kept (no pre-clip high-pass) and the upper mids get a gentle
            # lift; the clipping is asymmetric, like the real circuit's.
            drive = max(1.0, self.od_drive)
            k = 0.25 + 0.75 * (drive - 1.0) / 9.0          # share of the clipped path in the mix
            hot = _asymmetric_clip(y, drive * 0.9)
            hot = self._od_dc.process(hot)
            if HAS_SCIPY:
                b, a = spsig.butter(1, 1800/(self.sr/2), btype='high')
                hf, self._od_zi_bd2 = spsig.lfilter(b, a, hot, zi=self._od_zi_bd2)
                hot = hot + hf * 0.3
            y = y * (1.0 - k) * 0.8 + hot * k
            np.clip(y, -1, 1, out=y)
            y = y * self.od_vol
        else:  # clean boost
            y = y * max(1.0, self.od_drive)
            np.clip(y, -1, 1, out=y)
            y = y * self.od_vol
        return y

    def _fx_dist(self, y):
        if not self.dist_on: return y
        d    = max(1.0, self.dist_drive)
        tone = max(0.0, min(1.0, self.dist_tone))
        mode = self.dist_mode

        if mode == 'ds1':
            # Boss DS-1: silicon diodes clipped hard to a low, fixed
            # threshold — the classic boxy, compressed DS-1 crunch.
            clip_fn = lambda s: _hard_clip(s * d * 2.2, 0.3)
        elif mode == 'rat':
            # ProCo RAT: op-amp driven into hard-clipping diodes, a touch
            # rounder-edged than the DS-1's lower threshold.
            clip_fn = lambda s: _hard_clip(s * d * 2.6, 0.45)
        elif mode == 'dist_plus':
            # MXR Distortion+: op-amp into a diode pair to ground — softer
            # and more compressed than either above, asymmetric like real
            # diodes' mismatched forward voltages.
            clip_fn = lambda s: _asymmetric_clip(s, d * 0.9)
        else:  # 'metal' — high-gain, scooped-mid stack
            def clip_fn(s):
                s = s * d * 0.35
                s = _hard_clip(s, 0.55)
                s = s * d * 0.35
                s = _hard_clip(s, 0.55)
                return s
        y = self._dist_os.process(y, clip_fn)
        if mode == 'dist_plus':
            y = self._dist_dc.process(y)

        # Tone: one-pole low-pass sweeping dark->bright; 'metal' adds a mid
        # scoop on top for the classic high-gain scooped-mids voicing.
        if HAS_SCIPY:
            fc = 1200.0 + tone * 4500.0
            b, a = spsig.butter(1, min(fc, self.sr*0.49)/(self.sr/2), btype='low')
            y, self._dist_zi = spsig.lfilter(b, a, y, zi=self._dist_zi)
            if mode == 'metal':
                fc_n, gain_n, Q_n = 500.0, -8.0, 1.2
                w0 = 2*np.pi*fc_n/self.sr; cw = math.cos(w0); sw = math.sin(w0)
                A  = 10**(gain_n/40.0); al = sw/(2*Q_n)
                bn = np.array([1+al*A, -2*cw, 1-al*A])
                an = np.array([1+al/A, -2*cw, 1-al/A])
                # zi carried across blocks (self._dist_notch_zi) — without
                # it, scipy defaults to zero initial state on every call,
                # which reset this filter at every ~5.8ms block boundary
                # and produced a real, measured discontinuity there (~0.1
                # amplitude jump, 23x bigger than the filter's own natural
                # step) — audible as buzzing, same bug class fixed in
                # _muff_tone() below.
                y, self._dist_notch_zi = spsig.lfilter(bn/an[0], an/an[0], y,
                                                        zi=self._dist_notch_zi)

        y = y * self.dist_vol
        np.clip(y, -1, 1, out=y)
        return y

    def _fx_chorus(self, y):
        if not self.chorus_on: return y
        mode = self.chorus_mode
        depth = max(0.0, min(1.0, self.chorus_depth))
        mix = self.chorus_mix
        if mode == 'flanger':
            delay_ms, lfo_frac = 5.0 * depth, 0.5
        elif mode == 'vibrato':
            # the same moving delay as the chorus, but with none of the dry
            # signal: what is left is pure pitch wobble
            delay_ms, lfo_frac, mix = 6.0 * depth, 0.5, 1.0
        elif mode == 'doubler':
            # a fixed short delay with only a slow, shallow drift in it: a
            # second 'take' just behind the first, slightly out of tune
            delay_ms, lfo_frac = 10.0 + 15.0 * depth, 0.1
        else:   # chorus
            delay_ms, lfo_frac = 20.0 * depth, 0.5
        center    = delay_ms / 1000.0 * self.sr
        lfo_range = center * lfo_frac
        mod = math.sin(2 * math.pi * self._ch_phase)
        del_samp  = max(1.0, center + lfo_range * mod)
        self._ch_phase += self.chorus_rate / self.sr
        if self._ch_phase > 1.0: self._ch_phase -= 1.0
        buf = self._ch_buf
        wp  = self._ch_wp
        out = np.zeros(len(y))
        for i in range(len(y)):
            buf[wp] = y[i]
            rp_f = wp - del_samp
            rp   = int(rp_f) % len(buf)
            frac = rp_f - int(rp_f)
            rp2  = (rp + 1) % len(buf)
            delayed = buf[rp] * (1 - frac) + buf[rp2] * frac
            out[i] = y[i] * (1 - mix) + delayed * mix
            wp = (wp + 1) % len(buf)
        self._ch_wp = wp
        self._ch_buf = buf
        return out

    def _fx_phaser(self, y):
        """Four first-order allpass stages swept by a sine LFO, with
        feedback around them, mixed 50/50 with the dry signal: a notch
        pattern that moves up and down the spectrum. The LFO is computed
        per sample, so the sound does not depend on the block size."""
        if not self.phaser_on: return y
        x = np.asarray(y, dtype=np.float64)
        n = len(x)
        rate = max(0.02, float(self.phaser_rate))
        depth = max(0.0, min(1.0, float(self.phaser_depth)))
        fb = max(0.0, min(0.92, float(self.phaser_feedback)))
        ph = self._ph_phase + rate / self.sr * np.arange(n)
        self._ph_phase = float((self._ph_phase + rate * n / self.sr) % 1.0)
        fc = 800.0 * 2.0 ** (depth * 2.2 * np.sin(2.0 * np.pi * ph))   # about 175 Hz - 3.7 kHz at full depth
        t = np.tan(np.pi * np.minimum(fc, 0.45 * self.sr) / self.sr)
        coef = ((t - 1.0) / (t + 1.0)).tolist()
        z0, z1, z2, z3 = self._ph_z
        last = self._ph_last
        xs = x.tolist()
        out = [0.0] * n
        for i in range(n):
            a = coef[i]
            s = xs[i] + fb * last
            v = a * s + z0;  z0 = s - a * v;  s = v
            v = a * s + z1;  z1 = s - a * v;  s = v
            v = a * s + z2;  z2 = s - a * v;  s = v
            v = a * s + z3;  z3 = s - a * v;  s = v
            last = s
            out[i] = 0.5 * (xs[i] + s)
        self._ph_z = [z0, z1, z2, z3]
        self._ph_last = last
        res = np.asarray(out)
        np.clip(res, -1, 1, out=res)
        return res

    def _fx_tremolo(self, y):
        """Amplitude tremolo. Shape blends a smooth sine into a choppy,
        square-ish wave. The gain swings between 1 and 1-depth."""
        if not self.tremolo_on: return y
        x = np.asarray(y, dtype=np.float64)
        n = len(x)
        rate = max(0.1, float(self.tremolo_rate))
        depth = max(0.0, min(1.0, float(self.tremolo_depth)))
        shape = max(0.0, min(1.0, float(self.tremolo_shape)))
        ph = self._trem_phase + rate / self.sr * np.arange(n)
        self._trem_phase = float((self._trem_phase + rate * n / self.sr) % 1.0)
        s = np.sin(2.0 * np.pi * ph)
        sq = np.tanh(6.0 * s) / math.tanh(6.0)
        wave = (1.0 - shape) * s + shape * sq
        return x * (1.0 - depth * (0.5 - 0.5 * wave))

    def _fx_rotary(self, y):
        """Rotary speaker. A crossover splits the sound at 800 Hz: the horn (highs) and the drum (lows) each
        turn at their own speed, which moves the sound closer to and further from the listener (a pitch
        wobble from a modulated delay) and in and out of the cabinet (a level swell). Changing between Slow,
        Fast and Brake makes the rotors speed up and slow down, the horn quicker than the drum, as the
        real thing does. Like the Keeley Dark Side, the main controls are speed, depth and level."""
        if not self.rotary_on: return y
        x = np.asarray(y, dtype=np.float64)
        n = len(x)
        if n == 0:
            return x
        sr = self.sr
        slow = max(0.1, float(self.rotary_slow))
        fast = max(slow, float(self.rotary_fast))
        horn_target = {'slow': slow, 'fast': fast}.get(self.rotary_speed, 0.0)
        targets = (horn_target, horn_target * 0.82)
        ramp = max(0.0, min(1.0, float(self.rotary_ramp)))
        taus = (0.12 + 1.5 * ramp, 0.35 + 3.5 * ramp)
        depth = max(0.0, min(1.0, float(self.rotary_depth)))
        if self._rot_first:                       # start at the speed it was switched on at, not from a standstill
            self._rot_rate = [targets[0], targets[1]]
            self._rot_first = False
        t = (np.arange(n) + 1.0) / sr
        # crossover: a one-pole low-pass and its complement
        a = math.exp(-2.0 * math.pi * 800.0 / sr)
        if HAS_SCIPY:
            lows, self._rot_lp_z = spsig.lfilter([1 - a], [1.0, -a], x, zi=self._rot_lp_z)
        else:
            lows = np.empty(n)
            z = float(self._rot_lp_z[0])
            for i in range(n):
                z = (1 - a) * x[i] + a * z
                lows[i] = z
            self._rot_lp_z = np.array([z])
        parts = (x - lows, lows)                  # horn, drum
        base, swing, am = (24.0, 18.0), (44.0, 30.0), (0.55, 0.35)
        wet = np.zeros(n)
        for k in range(2):
            rates = targets[k] + (self._rot_rate[k] - targets[k]) * np.exp(-t / taus[k])
            phase = self._rot_ph[k] + np.cumsum(rates) / sr
            self._rot_rate[k] = float(rates[-1])
            self._rot_ph[k] = float(phase[-1] % 1.0)
            s = np.sin(2.0 * np.pi * phase)
            buf = np.concatenate((self._rot_hist[k], parts[k]))
            h = len(self._rot_hist[k])
            pos = h + np.arange(n) - (base[k] + swing[k] * depth * 0.5 * (1.0 + s))
            i0 = np.floor(pos).astype(int)
            fr = pos - i0
            sig = buf[i0] * (1.0 - fr) + buf[i0 + 1] * fr
            self._rot_hist[k] = buf[-h:]
            wet += sig * (1.0 - am[k] * depth * (0.5 - 0.5 * np.sin(2.0 * np.pi * phase + 1.0)))
        mix = max(0.0, min(1.0, float(self.rotary_mix)))
        out = x * (1.0 - mix) + wet * mix
        np.clip(out, -1, 1, out=out)
        return out

    def _fx_pitch(self, y):
        """Pitch shifter with one or two voices (octaves, fifths, detune). Mix 0 is the dry sound, 1 the shifted
        voices alone; a small Cents amount at 0 semitones makes a detuner."""
        if not self.pitch_on: return y
        x = np.asarray(y, dtype=np.float64)
        mix = max(0.0, min(1.0, float(self.pitch_mix)))
        if mix <= 0.0 or len(x) == 0:
            return x
        r1 = 2.0 ** ((float(self.pitch_semi) + float(self.pitch_cents) / 100.0) / 12.0)
        if self._pitch_eng[0] is None:
            self._pitch_eng[0] = _PitchEngine(self.sr)
        wet = self._pitch_eng[0].process(x, r1)
        lvl2 = max(0.0, min(1.0, float(self.pitch_level2)))
        if lvl2 > 0.0:
            if self._pitch_eng[1] is None:
                self._pitch_eng[1] = _PitchEngine(self.sr)
            wet = wet + lvl2 * self._pitch_eng[1].process(x, 2.0 ** (float(self.pitch_semi2) / 12.0))
        # the dry path is delayed to match the shifter so the two do not comb against each other
        d = self._pitch_eng[0].base
        hist = self._pitch_dry if len(self._pitch_dry) == d else np.zeros(d)
        buf = np.concatenate((hist, x))
        self._pitch_dry = buf[-d:]
        out = buf[:len(x)] * (1.0 - mix) + wet * mix
        np.clip(out, -1, 1, out=out)
        return out

    def _fx_delay(self, y):
        if not self.delay_on: return y
        dt   = self._delay_time_secs()
        dsmp = min(int(dt * self.sr), len(self._dly_buf) - 1)
        buf  = self._dly_buf
        wp   = self._dly_wp
        out  = np.zeros(len(y))
        for i in range(len(y)):
            rp      = (wp - dsmp) % len(buf)
            delayed = buf[rp]
            # tape mode: one-pole LP on feedback path (warmer repeats)
            if self.delay_mode == 'tape':
                alpha = 1 - math.exp(-2*math.pi * 4500 / self.sr)
                self._dly_lp_zi[0] = self._dly_lp_zi[0] + alpha * (delayed - self._dly_lp_zi[0])
                delayed = self._dly_lp_zi[0]
            buf[wp] = y[i] + delayed * self.delay_feedback
            out[i]  = y[i] * (1 - self.delay_mix) + delayed * self.delay_mix
            wp = (wp + 1) % len(buf)
        self._dly_wp = wp
        self._dly_buf = buf
        return out

    def _fx_reverb(self, y):
        if not self.reverb_on: return y
        rtype = self.reverb_type
        if rtype == 'shimmer':
            eng = self._rev_engines.get('shimmer')
            if eng is None:
                eng = _ShimmerEngine(self.sr)
                self._rev_engines['shimmer'] = eng
            wet = eng.process(y, self.reverb_size, self.reverb_decay, self.reverb_damp, self.reverb_shimmer)
            return np.asarray(y, dtype=np.float64) * (1 - self.reverb_mix) + wet * self.reverb_mix
        if rtype in ('plate', 'cloud', 'spring') and (HAS_SCIPY or rtype != 'spring'):
            eng = self._rev_engines.get(rtype)
            if eng is None:
                eng = (_SpringEngine(self.sr) if rtype == 'spring'
                       else _PlateEngine(self.sr, big=(rtype == 'cloud')))
                self._rev_engines[rtype] = eng
            wet = eng.process(y, self.reverb_size, self.reverb_decay, self.reverb_damp)
            return np.asarray(y, dtype=np.float64) * (1 - self.reverb_mix) + wet * self.reverb_mix
        # Schroeder comb+allpass (the 'room' type)
        sz   = max(0.1, self.reverb_size)
        wet  = np.zeros(len(y))
        # parallel combs
        for ci in range(4):
            clen = max(8, int(self._comb_len[ci] * sz))
            g    = self._comb_gains[ci]
            buf  = self._comb_bufs[ci]
            wp   = self._comb_wps[ci]
            for i in range(len(y)):
                rp      = (wp - clen) % len(buf)
                delayed = buf[rp]
                buf[wp] = y[i] + delayed * g
                wet[i] += delayed
                wp = (wp + 1) % len(buf)
            self._comb_wps[ci] = wp
            self._comb_bufs[ci] = buf
        wet /= 4.0
        # series allpass
        for ai in range(2):
            alen = max(4, int(self._ap_len[ai] * sz))
            buf  = self._ap_bufs[ai]
            wp   = self._ap_wps[ai]
            g    = 0.5
            for i in range(len(wet)):
                rp      = (wp - alen) % len(buf)
                delayed = buf[rp]
                v       = wet[i] + delayed * (-g)
                buf[wp] = wet[i] + v * g
                wet[i]  = delayed + v * g
                wp = (wp + 1) % len(buf)
            self._ap_wps[ai] = wp
            self._ap_bufs[ai] = buf
        return y * (1 - self.reverb_mix) + wet * self.reverb_mix

    _FX_FUNCS = {
        'comp':    '_fx_comp',
        'wah':     '_fx_wah',
        'pitch':   '_fx_pitch',
        'fuzz':    '_fx_fuzz',
        'dist':    '_fx_dist',
        'od':      '_fx_od',
        'phaser':  '_fx_phaser',
        'chorus':  '_fx_chorus',
        'tremolo': '_fx_tremolo',
        'rotary':  '_fx_rotary',
        'delay':   '_fx_delay',
        'reverb':  '_fx_reverb',
    }

    def _split_order(self):
        """(pre, post) pedal lists either side of the amp marker. An order
        with no marker (set directly, bypassing normalize_order) is treated
        as all-pre-amp, the behaviour before post-amp effects existed."""
        order = self.order
        if AMP_MARKER in order:
            i = order.index(AMP_MARKER)
            return order[:i], order[i + 1:]
        return list(order), []

    def _run(self, y, names):
        for name in names:
            method = self._FX_FUNCS.get(name)
            if method is not None:
                y = getattr(self, method)(y)
        np.clip(y, -1, 1, out=y)
        return y.astype(np.float32)

    def process(self, x, metro_bpm=120):
        """Pre-amp stage. x: np.float32[N]. Returns np.float32[N].
        Runs the effects ahead of the 'amp' marker in self.order —
        user-reorderable from the Pedals tab."""
        self._delay_bpm = metro_bpm
        if self.mute:
            return np.zeros(len(x), dtype=np.float32)
        return self._run(x.astype(np.float64), self._split_order()[0])

    def process_post(self, x, metro_bpm=120):
        """Post-amp stage, run on the amp/cab/EQ output. x: np.float32[N].
        Runs the effects after the 'amp' marker. Mute silences this stage
        too, so MUTE still means silence, not a ringing reverb tail."""
        self._delay_bpm = metro_bpm
        post = self._split_order()[1]
        if self.mute:
            return np.zeros(len(x), dtype=np.float32)
        if not post:
            return x
        return self._run(x.astype(np.float64), post)
