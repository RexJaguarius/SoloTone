#!/usr/bin/env python3
"""
Pedal-effects chain (pre-NAM) — compressor, wah, fuzz, distortion,
overdrive/boost, chorus/flanger, delay, reverb.

Split out of solotone.py into its own module: like nam_engine.py, this
has zero Tkinter/App coupling (plain numpy in, numpy out), so it can be
imported, profiled, or unit-tested in isolation from the rest of the
app — exactly what the build-75 Fuzz/Distortion artifacting fix relied
on (capturing real guitar audio and running it straight through
PedalChain's own methods, block by block, outside the GUI entirely).

Public API: PedalChain, DEFAULT_PEDAL_ORDER. Everything else here
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


# Single source of truth for the pedal chain's factory order — used both
# to initialize a new PedalChain and by the Pedals tab's "Reset Order"
# button, so the two can't silently drift apart the way a duplicated
# literal could. Also the order MIDI_MAPPABLE_TARGETS' pedal-related
# entries are listed in, so the Settings tab's MIDI list reads in the
# same sequence as the actual signal chain.
DEFAULT_PEDAL_ORDER = ['comp', 'wah', 'fuzz', 'dist', 'od', 'chorus', 'delay', 'reverb']


class PedalChain:
    """
    Pre-NAM effects chain, processed sample-accurately in the amp callback.
    All parameters are plain Python attributes — set from the GUI thread;
    reads in the audio thread are safe because Python attribute access is
    atomic for simple types.

    Signal order:
        [mute] → compressor → wah → fuzz → distortion → overdrive/boost
               → chorus/flanger → delay → reverb  → output
    """
    def __init__(self, sr=44100):
        self.sr = sr

        # ── global ───────────────────────────────────────────
        self.mute         = False

        # ── chain order ──────────────────────────────────────
        # User-reorderable stomp order (mute is a global kill switch, not
        # part of the chain, so it isn't in this list). Rearranged from the
        # Pedals tab with the Move up/down buttons.
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
        self.wah_freq       = 800.0     # live Hz value, recomputed every block
        self._wah_svf       = _SVF(sr)
        self._wah_auto_phase = 0.0

        # ── fuzz ─────────────────────────────────────────────
        self.fuzz_on      = False
        self.fuzz_mode    = 'fuzz_face'   # 'fuzz_face' | 'big_muff'
        self.fuzz_drive   = 5.0           # 1–10
        self.fuzz_vol     = 0.7
        self.muff_tone    = 'scoop'       # 'full' | 'flat' | 'scoop'
        self._fuzz_os     = Oversampler(4, sr)   # anti-aliasing for the clip stage
        self._muff_zi     = None   # _muff_tone()'s biquad state, carried across blocks
        self._fuzz_dc     = DCBlocker()   # removes _asymmetric_clip's DC bias (fuzz_face only)

        # ── overdrive / boost ─────────────────────────────────
        self.od_on        = False
        self.od_mode      = 'tubescreamer'  # 'tubescreamer' | 'boost' | 'klon'
        self.od_drive     = 4.0
        self.od_vol       = 0.8
        self._od_svf      = _SVF(sr)        # for TS mid-hump
        self._od_zi       = [np.zeros(1), np.zeros(1)]  # HP in, HP out
        self._od_zi_klon  = np.zeros(1)     # Klon's treble-boost filter

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

        # ── chorus / flanger ─────────────────────────────────
        self.chorus_on    = False
        self.chorus_mode  = 'chorus'     # 'chorus' | 'flanger'
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
        self.reverb_mix   = 0.25
        self.reverb_size  = 0.6          # 0–1, scales delay times
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

    def _fx_fuzz(self, y):
        if not self.fuzz_on: return y
        d = max(1.0, self.fuzz_drive)
        if self.fuzz_mode == 'fuzz_face':
            y = self._fuzz_os.process(y, lambda s: _asymmetric_clip(s, d))
            y = self._fuzz_dc.process(y)
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
        delay_ms  = (20.0 if self.chorus_mode == 'chorus' else 5.0) * self.chorus_depth
        center    = delay_ms / 1000.0 * self.sr
        lfo_range = center * 0.5
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
            out[i] = y[i] * (1 - self.chorus_mix) + delayed * self.chorus_mix
            wp = (wp + 1) % len(buf)
        self._ch_wp = wp
        self._ch_buf = buf
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
        # Schroeder comb+allpass
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
        'fuzz':    '_fx_fuzz',
        'dist':    '_fx_dist',
        'od':      '_fx_od',
        'chorus':  '_fx_chorus',
        'delay':   '_fx_delay',
        'reverb':  '_fx_reverb',
    }

    def process(self, x, metro_bpm=120):
        """x: np.float32[N]. Returns np.float32[N].
        Effects run in self.order — user-reorderable from the Pedals tab."""
        self._delay_bpm = metro_bpm
        y = x.astype(np.float64)

        if self.mute:
            return np.zeros(len(x), dtype=np.float32)

        for name in self.order:
            method = self._FX_FUNCS.get(name)
            if method is not None:
                y = getattr(self, method)(y)

        np.clip(y, -1, 1, out=y)
        return y.astype(np.float32)
