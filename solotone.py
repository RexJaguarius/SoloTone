#!/usr/bin/env python3
"""
SoloTone — Guitar Amp Processor · Looper · Tuner · Metronome
Run:
    pip install -r requirements.txt
    python solotone.py

nam_engine.py (same folder) holds the NAM/WaveNet inference engine and
is a required sibling module, not an optional extra — it has no UI
dependency of its own, which is what let it split out cleanly.
"""

import json
import math
import os
import re
import sys

try:
    from version import VERSION_FULL, CHANGELOG
except ImportError:
    VERSION_FULL = 'unknown'
    CHANGELOG = []
from nam_engine import NamLoadError, load_nam
from pedals import PedalChain, DEFAULT_PEDAL_ORDER
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser

import numpy as np

try:
    import sounddevice as sd
except OSError as e:
    sd = None
    _SD_IMPORT_ERROR = e
else:
    _SD_IMPORT_ERROR = None

try:
    import scipy.signal as spsig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import rtmidi
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SAMPLE_RATE = 44100
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
TRANSPOSITIONS = {
    'C  (Concert)': 0,
    'Bb (Trumpet/Clarinet/Tenor Sax)': 2,
    'Eb (Alto Sax/Baritone)': 9,
    'F  (French Horn)': 7,
}

# Keysyms that are pure modifiers — never accepted as a keybind on their
# own (Learn mode ignores a bare tap of these) since they carry no signal
# by themselves and are usually held alongside another key anyway.
KEY_MODIFIER_KEYSYMS = {
    'Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R',
    'Caps_Lock', 'Num_Lock', 'Scroll_Lock', 'Super_L', 'Super_R', 'Menu',
}

# Guitar tuning presets — each entry is a list of 6 (note_name, octave) targets,
# ordered low-to-high (string 6 to string 1).
# Midi note = (octave+1)*12 + note_index
def _midi(note, octave):
    return (octave + 1) * 12 + NOTE_NAMES.index(note)

NOTE_NAMES_FLAT = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']

# Tunings whose own name uses flat nomenclature ("Eb standard") should
# display every string's note as a flat, not the internally-stored sharp
# spelling (D#) — matching how a real hardware tuner would label it.
GUITAR_TUNING_USE_FLATS = {'Eb standard'}

GUITAR_TUNINGS = {
    'Standard (EADGBe)':      [('E',2),('A',2),('D',3),('G',3),('B',3),('E',4)],
    'Eb standard':            [('D#',2),('G#',2),('C#',3),('F#',3),('A#',3),('D#',4)],
    'D standard':             [('D',2),('G',2),('C',3),('F',3),('A',3),('D',4)],
    'Drop D':                 [('D',2),('A',2),('D',3),('G',3),('B',3),('E',4)],
    'Drop C':                 [('C',2),('G',2),('C',3),('F',3),('A',3),('D',4)],
    'Drop B':                 [('B',1),('F#',2),('B',2),('E',3),('G#',3),('C#',4)],
    'Drop A':                 [('A',1),('E',2),('A',2),('D',3),('F#',3),('B',3)],
    'Open G (DGDGBd)':        [('D',2),('G',2),('D',3),('G',3),('B',3),('D',4)],
    'Open D (DADf#Ad)':       [('D',2),('A',2),('D',3),('F#',3),('A',3),('D',4)],
    'Open E (EBEg#Be)':       [('E',2),('B',2),('E',3),('G#',3),('B',3),('E',4)],
    'Open A (EAEAc#e)':       [('E',2),('A',2),('E',3),('A',3),('C#',4),('E',4)],
    'Open C (CGCGCe)':        [('C',2),('G',2),('C',3),('G',3),('C',4),('E',4)],
    'DADGAD':                 [('D',2),('A',2),('D',3),('G',3),('A',3),('D',4)],
    'Double Drop D':          [('D',2),('A',2),('D',3),('G',3),('B',3),('D',4)],
    'None (chromatic)':       None,   # disables string readout, shows raw note
}

# Bass tuning presets — same (note, octave) format, low-to-high. Bass
# fundamentals run roughly an octave below the guitar strings they share a
# name with (e.g. bass E1 vs guitar E2), so these are a separate list
# rather than an extension of GUITAR_TUNINGS, selected via the tuner's
# Guitar/Bass instrument toggle rather than mixed into one dropdown.
BASS_TUNINGS = {
    'Bass Standard (EADG)':   [('E',1),('A',1),('D',2),('G',2)],
    'Bass 5-String (BEADG)':  [('B',0),('E',1),('A',1),('D',2),('G',2)],
    'Bass Drop D (DADG)':     [('D',1),('A',1),('D',2),('G',2)],
    'None (chromatic)':       None,
}

# Pre-compute midi numbers for every tuning, guitar and bass alike, keyed
# by name — the tuner looks numbers up here regardless of which instrument
# mode is currently selected.
ALL_TUNINGS = {**GUITAR_TUNINGS, **BASS_TUNINGS}
GUITAR_TUNING_MIDI = {
    name: [_midi(n, o) for n, o in strings] if strings else None
    for name, strings in ALL_TUNINGS.items()
}

def closest_string(freq_hz, tuning_midi, a4=440.0):
    """Return (string_index 0-5, cents_off) for the closest string target.
    Returns (None, 0) if freq is None or tuning is None."""
    if freq_hz is None or tuning_midi is None:
        return None, 0.0
    midi_detected = 69 + 12 * math.log2(freq_hz / a4)
    best_idx, best_dist = 0, float('inf')
    for i, target in enumerate(tuning_midi):
        dist = abs(midi_detected - target)
        if dist < best_dist:
            best_dist = dist; best_idx = i
    cents = (midi_detected - tuning_midi[best_idx]) * 100.0
    return best_idx, cents
# DEFAULT_ACCENT/DEFAULT_BG are the factory theme, used by the Settings
# tab's Appearance section "Reset to defaults" button. ACCENT/BG (and
# PANEL/ACCENT_DARK/ACCENT_DIM, derived from them — see
# _recompute_derived_theme()) are plain module-level names rather than
# class constants because they're read by dozens of widget-construction
# call sites across every tab; App._apply_theme() mutates them via
# `global` and rebuilds the UI so every widget picks up the new values,
# the same way changing any other Python module global would only take
# effect on the next code that reads it.
DEFAULT_ACCENT = '#1F7E89'   # teal — the app's one highlight color
DEFAULT_BG     = '#121212'   # neutral dark grey

BG          = DEFAULT_BG
PANEL       = '#1c1c1c'      # derived from BG — see _recompute_derived_theme()
BG_WELL     = '#0e0e0e'      # derived from BG — a recessed shade for gauge/progress wells
ACCENT      = DEFAULT_ACCENT
ACCENT_DARK = '#0a2427'      # derived from ACCENT — near-black tint, text/labels drawn ON an ACCENT background
ACCENT_DIM  = '#123a3f'      # derived from ACCENT — a background wash behind ACCENT-colored text/highlights
RED         = '#ff6b6b'
FG          = '#e8eef2'
DIM         = '#7d8891'
FF          = 'DejaVu Sans Mono'

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(c)))) for c in rgb)

def _scale_color(hex_color, factor):
    """Multiply each RGB channel by `factor` (0=black, 1=unchanged),
    clamped to a valid byte — used to derive a darker tint of a color
    (e.g. ACCENT_DARK/ACCENT_DIM from ACCENT) without a separate
    hand-picked literal that could go stale if the base color changes."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((r * factor, g * factor, b * factor))

def _lighten_color(hex_color, amount):
    """Add a flat amount (-255..255) to each RGB channel, clamped —
    used to derive PANEL as a step lighter than BG."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((r + amount, g + amount, b + amount))

def _valid_hex_color(s):
    """True if `s` is a plain '#rrggbb' string — the only shape ever
    written to theme_accent/theme_bg, but session/settings files are
    user-editable text, so anything read back from one is untrusted
    until checked."""
    if not isinstance(s, str) or len(s) != 7 or s[0] != '#':
        return False
    try:
        int(s[1:], 16)
        return True
    except ValueError:
        return False

def _recompute_derived_theme():
    """(Re)derive PANEL/ACCENT_DARK/ACCENT_DIM from the current BG/ACCENT.
    Called once at import with the factory defaults, and again by
    App._apply_theme() whenever the user changes either color from the
    Settings tab's Appearance section."""
    global PANEL, BG_WELL, ACCENT_DARK, ACCENT_DIM
    PANEL       = _lighten_color(BG, 10)
    BG_WELL     = _scale_color(BG, 0.78)
    ACCENT_DARK = _scale_color(ACCENT, 0.18)
    ACCENT_DIM  = _scale_color(ACCENT, 0.35)

_recompute_derived_theme()

# Tk photo images referenced only by ttk styles (never by a widget
# attribute) have nothing else keeping Python from garbage-collecting
# them, which silently blanks the widget — this module-level list is
# that reference, for the app's lifetime. See _rounded_rect_image() /
# App._setup_rounded_buttons().
_UI_IMAGE_REFS = []

def _rounded_rect_image(size, radius, fill, outline=None, outline_width=1, scale=4):
    """A small anti-aliased rounded-rectangle PIL Image, drawn at `scale`x
    and downsampled — Tk/ttk has no native border-radius, so a rounded
    button is one of these handed to ttk as a 9-slice image element
    (see _setup_rounded_buttons) instead of a real vector shape. Used
    directly (transparent background) for the Pedals tab's icons, which
    are drawn onto a Canvas — Tk's Canvas alpha-composites correctly.
    ttk's own image *style element* does not composite this cleanly
    (it showed a stray opaque fringe in testing, likely resize/ringing
    artifacts at the fully-transparent edge), so button assets go through
    _button_asset_image() below instead, which never has a transparent
    pixel to begin with."""
    w, h = size
    pad = outline_width if outline else 0
    big = Image.new('RGBA', (w * scale, h * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.rounded_rectangle(
        [pad * scale // 2, pad * scale // 2,
         w * scale - 1 - pad * scale // 2, h * scale - 1 - pad * scale // 2],
        radius=radius * scale, fill=fill, outline=outline,
        width=(outline_width * scale) if outline else 0)
    return big.resize((w, h), Image.LANCZOS)

def _button_asset_image(size, radius, fill, outline, backdrop, outline_width=1):
    """Like _rounded_rect_image, but composited onto an OPAQUE `backdrop`
    fill rather than a transparent one, so there is no alpha edge for
    ttk's image element to render incorrectly — the corners are simply
    solid backdrop-colored pixels, matching the PANEL-colored frame every
    button in this app is placed on."""
    shape = _rounded_rect_image(size, radius, fill, outline, outline_width)
    base = Image.new('RGBA', size, _hex_to_rgb(backdrop) + (255,))
    base.paste(shape, (0, 0), shape)
    return base

MAX_IR_SLOTS = 5

# ─────────────────────────────────────────────────────────────
# Sound synthesis helpers
# ─────────────────────────────────────────────────────────────

def _hp(x, fc, sr=SAMPLE_RATE):
    dt = 1.0 / sr
    rc = 1.0 / (2 * np.pi * max(1.0, fc))
    a  = rc / (rc + dt)
    y  = np.zeros_like(x)
    py, px = x[0], x[0]
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * (py + x[i] - px)
        py, px = y[i], x[i]
    return y

def _norm(w, g=1.0):
    m = np.max(np.abs(w))
    return (w / m * g).astype(np.float32) if m > 1e-9 else w.astype(np.float32)

def _make_click(f, dur, decay=40.0):
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    return _norm(np.sin(2*np.pi*f*t) * np.exp(-t*decay))

def _make_mech(dur, decay, fc, g=1.0):
    n = int(SAMPLE_RATE * dur); t = np.linspace(0, dur, n, endpoint=False)
    return _norm(_hp(np.random.uniform(-1,1,n), fc) * np.exp(-t*decay), g)

def _make_wood(f, dur, decay, g=1.0):
    n = int(SAMPLE_RATE * dur); t = np.linspace(0, dur, n, endpoint=False)
    env = np.exp(-t*decay)
    return _norm(np.sin(2*np.pi*f*t)*env*0.75 + np.random.uniform(-1,1,n)*np.exp(-t*400)*0.4, g)

def _make_hihat(dur, decay, fc, g=1.0):
    n = int(SAMPLE_RATE * dur); t = np.linspace(0, dur, n, endpoint=False)
    return _norm(_hp(np.random.uniform(-1,1,n), fc) * np.exp(-t*decay), g)

def _make_crash(dur, decay, g=1.0):
    n = int(SAMPLE_RATE * dur); t = np.linspace(0, dur, n, endpoint=False)
    ratios = [1.0, 1.483, 1.932, 2.548, 2.996, 3.386, 4.236, 5.104]
    tone = sum(np.sin(2*np.pi*280*r*t + np.random.uniform(0, 2*np.pi)) for r in ratios) / len(ratios)
    noise = _hp(np.random.uniform(-1,1,n), 3500)
    mix = (tone*0.35 + noise*0.55) * np.exp(-t*decay) * np.minimum(1.0, t/0.004)
    return _norm(mix, g)

def _make_blip(f, dur, decay, g=1.0):
    n = int(SAMPLE_RATE * dur); t = np.linspace(0, dur, n, endpoint=False)
    sq = np.sign(np.sin(2*np.pi*f*t))
    return _norm((0.6*sq + 0.4*np.sin(2*np.pi*f*2*t)) * np.exp(-t*decay), g)

SOUND_KITS = ['Classic Beep', 'Mechanical Click', 'Wood Block', 'Cymbal & Crash', 'Digital Blip']

def build_kit(name):
    if name == 'Classic Beep':
        return dict(accent=_make_click(1750,.05,40), main=_make_click(1100,.045,40), sub=_make_click(750,.03,40))
    if name == 'Mechanical Click':
        return dict(accent=_make_mech(.014,700,3200,1.0), main=_make_mech(.011,900,2500,0.8), sub=_make_mech(.008,1200,2000,0.5))
    if name == 'Wood Block':
        return dict(accent=_make_wood(1500,.06,55,1.0), main=_make_wood(1000,.05,65,0.85), sub=_make_wood(1300,.035,90,0.5))
    if name == 'Cymbal & Crash':
        return dict(accent=_make_crash(1.1,3.2,1.0), main=_make_hihat(.07,170,6500,0.85), sub=_make_hihat(.045,260,7500,0.5))
    if name == 'Digital Blip':
        return dict(accent=_make_blip(1800,.045,55,1.0), main=_make_blip(1200,.04,60,0.8), sub=_make_blip(900,.03,80,0.5))
    raise ValueError(name)

# ─────────────────────────────────────────────────────────────
# Pitch helpers
# ─────────────────────────────────────────────────────────────

def autocorrelate_pitch(signal, sr, fmin=40.0, fmax=1500.0):
    signal = signal.astype(np.float64) - np.mean(signal)
    if np.sqrt(np.mean(signal**2)) < 0.006:
        return None
    windowed = signal * np.hanning(len(signal))
    corr = np.correlate(windowed, windowed, mode='full')
    corr = corr[len(corr)//2:]
    if corr[0] <= 0:
        return None
    lo, hi = max(1, int(sr/fmax)), min(len(corr)-2, int(sr/fmin))
    if hi <= lo:
        return None
    peak = int(np.argmax(corr[lo:hi])) + lo
    if corr[peak] / corr[0] < 0.3:
        return None
    a, b, c = corr[peak-1], corr[peak], corr[peak+1]
    denom = a - 2*b + c
    lag = peak + (0.5*(a-c)/denom if denom != 0 else 0.0)
    if lag <= 0:
        return None
    f = sr / lag
    return f if fmin <= f <= fmax else None

def analyze_pitch(freq, a4=440.0, transpose=0):
    if not freq:
        return None
    n = 12 * math.log2(freq / a4)
    midi = 69 + n
    mr   = round(midi)
    cents = (midi - mr) * 100.0
    dm = mr + transpose
    return dict(name=NOTE_NAMES[dm % 12], octave=dm//12-1, cents=cents, freq=freq)

# ─────────────────────────────────────────────────────────────
# Biquad EQ (3-band: low shelf, mid peak, high shelf)
# ─────────────────────────────────────────────────────────────

def _biquad_coefs(ftype, fc, gain_db, Q, sr):
    A  = 10**(gain_db/40.0)
    w0 = 2*np.pi*fc/sr
    cw, sw = np.cos(w0), np.sin(w0)
    al = sw/(2*Q)
    if ftype == 'lowshelf':
        b0 =  A*((A+1)-(A-1)*cw+2*np.sqrt(A)*al)
        b1 = 2*A*((A-1)-(A+1)*cw)
        b2 =  A*((A+1)-(A-1)*cw-2*np.sqrt(A)*al)
        a0 =    (A+1)+(A-1)*cw+2*np.sqrt(A)*al
        a1 = -2*((A-1)+(A+1)*cw)
        a2 =    (A+1)+(A-1)*cw-2*np.sqrt(A)*al
    elif ftype == 'highshelf':
        b0 =  A*((A+1)+(A-1)*cw+2*np.sqrt(A)*al)
        b1 = -2*A*((A-1)+(A+1)*cw)
        b2 =  A*((A+1)+(A-1)*cw-2*np.sqrt(A)*al)
        a0 =    (A+1)-(A-1)*cw+2*np.sqrt(A)*al
        a1 =  2*((A-1)-(A+1)*cw)
        a2 =    (A+1)-(A-1)*cw-2*np.sqrt(A)*al
    else:  # peak
        b0 = 1+al*A; b1 = -2*cw; b2 = 1-al*A
        a0 = 1+al/A; a1 = -2*cw; a2 = 1-al/A
    return np.array([b0,b1,b2])/a0, np.array([a0,a1,a2])/a0

class BiquadState:
    """5-band EQ (2 shelves + 3 peaking bands, graphic-EQ style spacing)
    plus an overall Level trim applied after the bands — a plain gain
    stage, not a filter, for compensating a heavily cut/boosted EQ
    setting back to a usable output level (like a real EQ pedal's own
    Level knob, centered at 0 dB rather than a boost-only control)."""
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        self.bands = [
            dict(type='lowshelf',  fc=100,  gain=0.0, Q=0.707),
            dict(type='peak',      fc=400,  gain=0.0, Q=1.0),
            dict(type='peak',      fc=1000, gain=0.0, Q=1.0),
            dict(type='peak',      fc=2500, gain=0.0, Q=1.0),
            dict(type='highshelf', fc=6000, gain=0.0, Q=0.707),
        ]
        self.level_db = 0.0
        self._zi = [np.zeros(2) for _ in self.bands]
        self._coefs = [self._calc(i) for i in range(len(self.bands))]

    def _calc(self, i):
        b = self.bands[i]
        b_c, a_c = _biquad_coefs(b['type'], b['fc'], b['gain'], b['Q'], self.sr)
        return b_c, a_c

    def set_gain(self, i, db):
        self.bands[i]['gain'] = db
        self._coefs[i] = self._calc(i)

    def set_level(self, db):
        self.level_db = db

    def process(self, x):
        y = x.astype(np.float64)
        for i, (bc, ac) in enumerate(self._coefs):
            if HAS_SCIPY:
                y, self._zi[i] = spsig.lfilter(bc, ac, y, zi=self._zi[i])
            else:
                # simple direct-form II transposed (good enough for EQ)
                out = np.zeros_like(y)
                z0, z1 = self._zi[i]
                for n in range(len(y)):
                    s = y[n]
                    out[n] = bc[0]*s + z0
                    z0 = bc[1]*s - ac[1]*out[n] + z1
                    z1 = bc[2]*s - ac[2]*out[n]
                self._zi[i] = np.array([z0, z1])
                y = out
        if self.level_db != 0.0:
            y = y * (10.0 ** (self.level_db / 20.0))
        return y.astype(np.float32)

# ─────────────────────────────────────────────────────────────
# Hum filter — mains-frequency notch, for ground-loop/pickup hum
# ─────────────────────────────────────────────────────────────

class HumFilter:
    """Cascaded notch filters at the mains frequency and its next two
    harmonics (60/120/180 Hz, or 50/100/150 Hz), to knock down ground-loop
    or single-coil-pickup hum without touching the rest of the guitar's
    spectrum. Applied to the raw input, ahead of everything else, so the
    tuner and every downstream effect see the cleaned-up signal too.
    Falls back to a silent no-op if scipy isn't installed."""
    def __init__(self, sr=SAMPLE_RATE):
        self.sr   = sr
        self.on   = False
        self.base = 60.0     # 60 Hz (US/Japan) or 50 Hz (most of the rest)
        self._sections = []
        self._build()

    def _build(self):
        self._sections = []
        if not HAS_SCIPY:
            return
        for h in (1, 2, 3):
            f0 = self.base * h
            if f0 >= self.sr / 2 - 50:
                break
            b, a = spsig.iirnotch(f0, 25.0, self.sr)
            self._sections.append([b, a, np.zeros(2)])

    def set_base(self, hz):
        self.base = float(hz)
        self._build()

    def process(self, x):
        if not self.on or not self._sections:
            return x
        y = x.astype(np.float64)
        for sect in self._sections:
            b, a, zi = sect
            y, zi2 = spsig.lfilter(b, a, y, zi=zi)
            sect[2] = zi2
        return y.astype(np.float32)

# ─────────────────────────────────────────────────────────────
# IR convolution (overlap-add)
# ─────────────────────────────────────────────────────────────

def _next_fast_len(n):
    """Smallest 5-smooth integer (only 2/3/5 prime factors) >= n. numpy's
    FFT can be an order of magnitude slower at a size with even one large
    prime factor than at a nearby smooth size — measured directly on a
    real-world IR whose natural convolution length (block + IR - 1)
    factored to include a prime of 1861: 5.8ms/call at that exact size vs
    0.56ms/call at the nearest 5-smooth size, a >10x difference that alone
    was enough to blow the real-time budget and sound like dropped audio."""
    if n <= 1: return 1
    best = None
    p5 = 1
    while p5 < n * 2:
        p35 = p5
        while p35 < n * 2:
            p235 = p35
            while p235 < n:
                p235 *= 2
            if best is None or p235 < best:
                best = p235
            p35 *= 3
        p5 *= 5
    return best


class IRConv:
    def __init__(self):
        self.ir   = None
        self.tail = np.zeros(0, dtype=np.float32)
    def load(self, ir_data):
        ir = np.asarray(ir_data, dtype=np.float32)
        # Normalize by L1 norm (sum of |taps|), not peak. Peak-normalizing an
        # IR that was exported quiet (this app has seen real files with a
        # raw peak of ~0.004) inflates every tap by however much headroom
        # that peak had — for an IR with energy spread across many taps,
        # that can multiply the *convolution* gain far past what the peak
        # number alone suggests. L1-normalizing instead guarantees
        # |output|_inf <= |input|_inf * sum(|h|)/sum(|h|) = |input|_inf, so
        # a bounded guitar signal can't come out the other side amplified.
        s = np.sum(np.abs(ir))
        self.ir   = ir / s if s > 1e-9 else ir
        self.tail = np.zeros(len(self.ir)-1, dtype=np.float32)
        self._H     = None   # cached rfft(ir), keyed by the out_len it was built for
        self._H_len = None
    def process(self, block):
        if self.ir is None:
            return block.astype(np.float32)
        out_len  = len(block) + len(self.ir) - 1
        # Padding the FFT out to the nearest 5-smooth size (rather than the
        # exact out_len) can be an order of magnitude faster — see
        # _next_fast_len — with no change to the result, since the extra
        # zero-padding doesn't affect the first out_len samples of a linear
        # convolution; they're discarded below via the [:out_len] slice.
        fft_len = _next_fast_len(out_len)
        # The IR's own FFT never changes between calls at a fixed block
        # size — only the block does — so it only needs recomputing when
        # fft_len changes (a new IR loaded, or the block size changed),
        # not on every single block.
        if self._H is None or self._H_len != fft_len:
            h_pad = np.zeros(fft_len, dtype=np.float64)
            h_pad[:len(self.ir)] = self.ir
            self._H = np.fft.rfft(h_pad)
            self._H_len = fft_len
        x_pad = np.zeros(fft_len, dtype=np.float64)
        x_pad[:len(block)] = block
        # linear convolution via FFT
        X = np.fft.rfft(x_pad)
        full = np.fft.irfft(X*self._H, n=fft_len).astype(np.float32)[:out_len]
        # overlap-add
        out = full[:len(block)].copy()
        overlap = len(self.tail)
        add_len = min(overlap, len(block))
        out[:add_len] += self.tail[:add_len]
        new_tail_len = len(full) - len(block)
        new_tail = full[len(block):]
        if add_len < overlap:
            combined = np.zeros(max(new_tail_len, overlap-add_len), dtype=np.float32)
            combined[:overlap-add_len] += self.tail[add_len:]
            combined[:new_tail_len]    += new_tail
            self.tail = combined
        else:
            self.tail = new_tail.copy()
        return out

def _read_wav_raw(path):
    """Parse a WAV file's chunks by hand. Returns (fmt_tag, nch, sr, bits, data).

    The stdlib `wave` module refuses to open anything but plain integer PCM
    (format code 1) — it raises 'unknown format' for IEEE-float WAV (format
    code 3) and for WAVE_FORMAT_EXTENSIBLE headers, both of which real IR
    captures and DAW exports use often enough to matter. This walks the
    RIFF chunk list directly (skipping any chunk it doesn't need, e.g. a
    'fact' or 'bext' chunk some exporters add ahead of 'data') so those
    files load instead of erroring out."""
    import struct
    with open(path, 'rb') as f:
        riff = f.read(12)
        if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
            raise ValueError('Not a RIFF/WAVE file')
        fmt_tag = nch = sr = bits = None
        data = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = hdr[:4], struct.unpack('<I', hdr[4:8])[0]
            body = f.read(csize)
            if csize % 2 == 1:
                f.read(1)  # chunks are word-aligned; skip the pad byte
            if cid == b'fmt ':
                fmt_tag, nch, sr = struct.unpack('<HHI', body[:8])
                bits = struct.unpack('<H', body[14:16])[0]
                if fmt_tag == 0xFFFE and len(body) >= 26:
                    # WAVE_FORMAT_EXTENSIBLE: the real format code is the
                    # first two bytes of the sub-format GUID at offset 24
                    fmt_tag = struct.unpack('<H', body[24:26])[0]
            elif cid == b'data':
                data = body
    if fmt_tag is None or data is None:
        raise ValueError('WAV file is missing a fmt or data chunk')
    return fmt_tag, nch, sr, bits, data


def write_wav_mono(path, arr, sr=SAMPLE_RATE):
    """Write a mono float array (range ~[-1, 1]) to a 16-bit PCM WAV file
    using the stdlib wave module — no extra dependency, and every DAW
    reads plain 16-bit PCM without question."""
    import wave as _wave
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype('<i2')
    with _wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def load_ir(path):
    """Load a WAV file as an impulse response. Returns (ir_array, name).
    Supports 8/16/24/32-bit integer PCM and 32/64-bit float WAV, including
    WAVE_FORMAT_EXTENSIBLE headers."""
    fmt_tag, nch, sr, bits, raw = _read_wav_raw(path)
    sw = bits // 8
    total_samples = len(raw) // sw

    if fmt_tag == 3:  # IEEE float
        if sw == 4:
            arr = np.frombuffer(raw, dtype='<f4', count=total_samples).astype(np.float64)
        elif sw == 8:
            arr = np.frombuffer(raw, dtype='<f8', count=total_samples).astype(np.float64)
        else:
            raise ValueError(f"Unsupported float WAV bit depth: {bits}")
    elif fmt_tag == 1:  # integer PCM
        if sw == 1:
            # 8-bit unsigned PCM — centre at 128
            arr = (np.frombuffer(raw, dtype=np.uint8, count=total_samples)
                     .astype(np.float64) - 128.0) / 128.0
        elif sw == 2:
            arr = np.frombuffer(raw, dtype='<i2', count=total_samples).astype(np.float64) / 32768.0
        elif sw == 3:
            # 24-bit PCM — no native numpy dtype; unpack manually
            # Pad each 3-byte sample to 4 bytes (little-endian, sign-extend)
            raw_arr = np.frombuffer(raw, dtype=np.uint8, count=total_samples*3).reshape(-1, 3)
            padded = np.zeros((total_samples, 4), dtype=np.uint8)
            padded[:, :3] = raw_arr
            # Sign-extend: if high bit of byte 2 is set, byte 3 = 0xFF
            padded[:, 3] = np.where(raw_arr[:, 2] & 0x80, 0xFF, 0x00)
            arr = padded.view('<i4').reshape(-1).astype(np.float64) / 2147483648.0
        elif sw == 4:
            arr = np.frombuffer(raw, dtype='<i4', count=total_samples).astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sw} bytes. "
                             f"Expected 8, 16, 24, or 32-bit PCM.")
    else:
        raise ValueError(f"Unsupported WAV format code: {fmt_tag} "
                         f"(only integer PCM and IEEE float are supported)")

    # Take left channel only if stereo/multi-channel
    if nch > 1:
        arr = arr[::nch]

    # Resample if needed (linear interp — fine for IR convolution)
    if sr != SAMPLE_RATE:
        new_len = int(len(arr) * SAMPLE_RATE / sr)
        arr = np.interp(np.linspace(0, len(arr)-1, new_len),
                        np.arange(len(arr)), arr)

    name = os.path.splitext(os.path.basename(path))[0]
    return arr.astype(np.float32), name

# ─────────────────────────────────────────────────────────────
# Metronome engine
# ─────────────────────────────────────────────────────────────

class Metronome:
    """Click-track engine.

    Two operating modes:
      Standalone — owns an OutputStream (used when amp is not running).
      Shared     — pull() is called by AmpProcessor._cb each block; no
                   separate stream is opened, so the output device is not
                   double-claimed.

    pull(frames) is the core: it schedules clicks into the circular mix
    buffer and returns a float32 block ready to add to any output mix.
    """
    def __init__(self, beat_q):
        self.beat_q  = beat_q
        self.sr      = SAMPLE_RATE
        self.bpm     = 120
        self.bpm_m   = 4
        self.subdiv  = 1
        self.volume  = 0.7
        self.mute    = False   # silences click audio only — beat_q keeps firing
        self.lock    = threading.Lock()
        self.stream  = None
        self.running = False
        self.shared  = False   # True when driven by amp callback
        self.amp     = None    # set by App.__init__
        self.sounds  = build_kit('Classic Beep')
        self.mix_len = int(SAMPLE_RATE * 3.0)
        self.mix_buf = np.zeros(self.mix_len, dtype=np.float32)
        self.rpos    = 0
        self._reset()

    def set_kit(self, name):
        k = build_kit(name)
        with self.lock: self.sounds = k

    def _reset(self):
        self.beat     = 0
        self.next_off = 0.0
        self.mix_buf[:] = 0.0
        self.rpos = 0

    def _spe(self):
        return (60.0 / max(1, self.bpm) / self.subdiv) * self.sr

    def _mix_in(self, seg, pos):
        n = min(len(seg), self.mix_len)
        seg = seg[:n]
        end = pos + n
        if end <= self.mix_len:
            self.mix_buf[pos:end] += seg
        else:
            f = self.mix_len - pos
            self.mix_buf[pos:] += seg[:f]
            self.mix_buf[:n-f] += seg[f:]

    def pull(self, frames):
        """Generate and return the next `frames` samples of click audio.
        Called either from _cb (standalone) or from AmpProcessor._cb (shared).
        Must be called with self.lock already held OR from _cb which acquires it.
        This version acquires its own lock — safe from both call sites."""
        with self.lock:
            spe = self._spe()
            o   = self.next_off
            while o < frames:
                oi   = int(round(o))
                si   = self.beat % self.subdiv
                bi   = (self.beat // self.subdiv) % self.bpm_m
                main = si == 0
                kind = ('accent' if bi == 0 else 'main') if main else 'sub'
                if not self.mute:
                    seg = self.sounds[kind] * self.volume
                    self._mix_in(seg, (self.rpos + oi) % self.mix_len)
                if main:
                    try: self.beat_q.put_nowait(bi)
                    except queue.Full: pass
                self.beat += 1; o += spe
            self.next_off = o - frames
            end = self.rpos + frames
            if end <= self.mix_len:
                chunk = self.mix_buf[self.rpos:end].copy()
                self.mix_buf[self.rpos:end] = 0.0
            else:
                f = self.mix_len - self.rpos
                chunk = np.concatenate((self.mix_buf[self.rpos:],
                                        self.mix_buf[:frames - f]))
                self.mix_buf[self.rpos:] = 0.0
                self.mix_buf[:frames - f] = 0.0
            self.rpos = (self.rpos + frames) % self.mix_len
        np.clip(chunk, -1.0, 1.0, out=chunk)
        return chunk

    def _cb(self, out, frames, *_):
        # Standalone mode only — amp not running
        chunk = self.pull(frames)
        out[:, 0] = chunk

    def start(self):
        if sd is None or self.running: return
        self._reset()
        # If the amp is already running, share its output stream
        if self.amp is not None and self.amp.running:
            self.shared  = True
            self.running = True
            return
        self.shared  = False
        self.running = True
        self.stream  = sd.OutputStream(samplerate=self.sr, channels=1,
                                        blocksize=512, callback=self._cb)
        self.stream.start()

    def stop(self):
        self.running = False
        self.shared  = False
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except: pass
            self.stream = None

# ─────────────────────────────────────────────────────────────
# Tuner engine
# ─────────────────────────────────────────────────────────────

class Tuner:
    """Chromatic pitch detector.

    Two operating modes:
      Standalone — opens its own InputStream (used when amp is not running).
      Shared     — fed by AmpProcessor._cb via feed(); no second stream opened.

    The amp callback calls feed() with the raw pre-gain input block every
    time it runs, so switching to the Tuner tab while the amp is running
    works seamlessly with no device-conflict errors.
    """
    def __init__(self):
        # 12288 samples (~278ms at 44.1kHz) instead of the previous 8192
        # (~186ms) — a low string like Drop A1 (55Hz) only gets ~10 full
        # cycles in the old window, which isn't much margin for a stable
        # autocorrelation peak; this gives it ~15.
        self.buf     = np.zeros(12288, dtype=np.float32)
        self.lock    = threading.Lock()
        self.stream  = None
        self.running = False
        self.shared  = False   # True when driven by amp callback
        self.device  = None
        self.fmin    = 40.0    # detection floor — lowered to 24 Hz in bass mode
        self.fmax    = 1500.0  # detection ceiling — lowered to 600 Hz in bass mode
        self._hist   = []      # recent detected freqs, for median smoothing
        # Reference back to the amp, set by App.__init__
        self.amp     = None

    def feed(self, block):
        """Called from AmpProcessor._cb with the raw input block (float32 1-D).
        Safe to call from the audio thread — uses the same lock as pitch()."""
        if not self.running: return
        n = len(block)
        with self.lock:
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = block

    def _cb(self, data, frames, *_):
        # Used only in standalone mode (amp not running)
        with self.lock:
            self.buf = np.roll(self.buf, -frames)
            self.buf[-frames:] = data[:,0]

    def pitch(self):
        with self.lock: d = self.buf.copy()
        f = autocorrelate_pitch(d, SAMPLE_RATE, fmin=self.fmin, fmax=self.fmax)
        if f is None:
            self._hist.clear()
            return None
        # Median of the last few readings rejects a single spurious frame
        # (e.g. an isolated octave error, more common on low/wound strings
        # with lower fundamental-to-harmonic energy ratios) without adding
        # much perceptible lag — a genuine, sustained note still converges
        # in a couple of poll cycles.
        self._hist.append(f)
        if len(self._hist) > 4:
            self._hist.pop(0)
        return float(np.median(self._hist))

    def start(self):
        """Start the tuner. If the amp is already running on the same (or
        default) device, reuse its input stream instead of opening a new one."""
        if sd is None or self.running: return
        # Check whether the amp is running — if so, go shared
        if self.amp is not None and self.amp.running:
            self.shared  = True
            self.running = True
            return
        # Standalone mode
        self.shared  = False
        self.running = True
        try:
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                         blocksize=1024, callback=self._cb,
                                         device=self.device)
            self.stream.start()
        except Exception:
            self.running = False
            raise

    def stop(self):
        self.running = False
        self.shared  = False
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except: pass
            self.stream = None

# Windows exposes the same physical device once per host API it's reachable
# through (MME, DirectSound, WASAPI, WDM-KS) — left unfiltered this turns a
# handful of real devices into 30-50 near-duplicate entries. WASAPI's names
# match Windows' own Sound Settings most closely, so it's the default filter.
PREFERRED_HOST_APIS = ['Windows WASAPI', 'Windows WDM-KS', 'Windows DirectSound', 'MME']

def host_api_names():
    """Host APIs actually present on this system, our preferred ones first."""
    if sd is None: return []
    try:
        names = [h['name'] for h in sd.query_hostapis()]
    except Exception:
        return []
    ordered = [n for n in PREFERRED_HOST_APIS if n in names]
    ordered += [n for n in names if n not in ordered]
    return ordered

def _clean_device_name(name):
    """Sanitize WDM-KS's raw kernel-streaming names, e.g. turning
    'Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0\\n;(soundcore  Q30))'
    into 'Headset (soundcore Q30)'. Names from other host APIs pass through
    with just whitespace normalized — they're already friendly."""
    if not name: return name
    n = re.sub(r'\s+', ' ', name.replace('\r', ' ').replace('\n', ' ')).strip()
    m = re.match(r'^(.*?)\s*\(@', n)
    if not m or not m.group(1).strip():
        return n
    prefix = m.group(1).strip()
    tail = re.search(r';\(([^)]+)\)\)?\s*$', n)
    if tail:
        label = re.sub(r'\s+', ' ', tail.group(1)).strip()
        return f'{prefix} ({label})'
    return prefix

def _is_real_device_name(name):
    """Drop the rare device that reports no name at all (e.g. 'Input ()')
    once the empty parens are stripped — but nothing else. Deciding a
    non-empty name like 'Headphones' isn't "useful enough" would be a
    guess we shouldn't make; the host-API filter above does the real work."""
    return bool(name and re.sub(r'\(\s*\)', '', name).strip())

def _list_devices(kind, host_api=None):
    """kind: 'input' or 'output'. host_api: a name from host_api_names(),
    or None for every host API (the old, noisy behavior)."""
    devs = [(None, 'System default')]
    if sd is None: return devs
    chan_key = 'max_input_channels' if kind == 'input' else 'max_output_channels'
    try:
        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if d.get(chan_key, 0) <= 0:
                continue
            h = hostapis[d['hostapi']]['name']
            if host_api and h != host_api:
                continue
            name = _clean_device_name(d['name'])
            if not _is_real_device_name(name):
                continue
            devs.append((i, name))
    except Exception:
        pass
    return devs

def list_inputs(host_api=None):
    return _list_devices('input', host_api)

def list_outputs(host_api=None):
    return _list_devices('output', host_api)

# ─────────────────────────────────────────────────────────────
# MIDI controller mapping — generic CC/note "Learn" system, not tied to
# any specific hardware. A class-compliant USB-MIDI controller (tested
# against an Akai LPD8) already has a driver on every OS; the only real
# work is reading the standard Note On/Off and Control Change messages
# it sends and mapping them to SoloTone controls. Every target below is
# just a small (app, value) callable, so adding a new mappable control
# later is a one-line registry addition, the same convention
# PEDAL_PROFILE_FIELDS uses for Tone Profiles.
# ─────────────────────────────────────────────────────────────

def _midi_toggle_action(var_name, set_fn):
    """A pedal/global on-off flipped by a pad hit (Note On). Reads and
    flips the current state itself — a MIDI Note On carries no on/off
    information on its own, that's the nature of a momentary pad hit
    being used as a toggle, like a real stompbox footswitch."""
    def action(app, _value):
        var = getattr(app, var_name)
        new_val = not var.get()
        var.set(new_val)
        set_fn(app, new_val)
    # state_getter lets anything that needs to read this target's current
    # on/off state generically (LPD8 pad LED feedback) do so without a
    # target-by-target special case — same convention as nudge_getter.
    action.state_getter = lambda app: bool(getattr(app, var_name).get())
    return action

def _midi_pedal_cc_action(attr, lo, hi):
    """A pedal parameter driven by a knob (CC 0-127, normalized 0-1
    here). Writes straight to the PedalChain attribute — safe from any
    thread per this file's existing convention (plain Python attribute
    assignment is atomic), no amp.lock needed — then syncs the Pedals
    tab's own slider for that attribute, if it's currently built (the
    generic registry populated by _tab_pedals(), so this works for any
    pedal knob without a target-by-target special case)."""
    def action(app, norm):
        v = lo + (hi - lo) * norm
        setattr(app.amp.pedals, attr, v)
        app._pedal_widget_sync(attr, v)
    # nudge_range/nudge_getter let App._nudge_cc_target() step this same
    # target by a fraction of its range without a target-by-target case —
    # used by the keybind system, where a keypress is a discrete nudge
    # rather than a knob's continuous sweep.
    action.nudge_range  = (lo, hi)
    action.nudge_getter = lambda app: getattr(app.amp.pedals, attr)
    return action

def _midi_eq_band_action(band_idx, lo=-15.0, hi=15.0):
    def action(app, norm):
        v = lo + (hi - lo) * norm
        with app.amp.lock: app.amp.eq.set_gain(band_idx, v)
        if hasattr(app, 'eq_sliders'):
            app._set_knob(app.eq_sliders[band_idx], app.eq_val_vars[band_idx],
                           v, f'{v:+.0f} dB')
    action.nudge_range  = (lo, hi)
    action.nudge_getter = lambda app: app.amp.eq.bands[band_idx]['gain']
    return action

def _midi_eq_level_action(lo=-15.0, hi=15.0):
    def action(app, norm):
        v = lo + (hi - lo) * norm
        with app.amp.lock: app.amp.eq.set_level(v)
        if hasattr(app, 'eq_level_sl'):
            app._set_knob(app.eq_level_sl, app.eq_level_val, v, f'{v:+.0f} dB')
    action.nudge_range  = (lo, hi)
    action.nudge_getter = lambda app: app.amp.eq.level_db
    return action

def _midi_toggle_amp_attr_action(var_name, amp_attr, sub_obj=None):
    """Like _midi_toggle_action, but for a flag that lives on `amp` or
    amp.<sub_obj> (eq_on, hum.on) instead of amp.pedals."""
    def action(app, _value):
        var = getattr(app, var_name)
        new_val = not var.get()
        var.set(new_val)
        target = getattr(app.amp, sub_obj) if sub_obj else app.amp
        setattr(target, amp_attr, new_val)
    action.state_getter = lambda app: bool(getattr(app, var_name).get())
    return action

def _midi_gain_action(amp_attr, lo_db, hi_db, slider_attr=None, val_attr=None):
    def action(app, norm):
        db = lo_db + (hi_db - lo_db) * norm
        setattr(app.amp, amp_attr, 10 ** (db / 20.0))
        if slider_attr and hasattr(app, slider_attr):
            app._set_knob(getattr(app, slider_attr), getattr(app, val_attr),
                           db, f'{db:+.1f} dB')
    action.nudge_range  = (lo_db, hi_db)
    action.nudge_getter = lambda app: 20.0 * math.log10(max(1e-9, getattr(app.amp, amp_attr)))
    return action

def _midi_gate_action(lo=0.0, hi=60.0):
    # Gate slider domain is 0-60 representing -dBFS (see App._set_gate).
    def action(app, norm):
        v = lo + (hi - lo) * norm
        app.amp.gate_thr = 10 ** (-v / 20.0)
        if hasattr(app, 'gate_sl'):
            app._set_knob(app.gate_sl, app.gate_val, v, f'{-v:.0f} dBFS')
    action.nudge_range  = (lo, hi)
    action.nudge_getter = lambda app: -20.0 * math.log10(max(1e-9, app.amp.gate_thr))
    return action

def _midi_ir_blend_action(lo=0.0, hi=100.0):
    def action(app, norm):
        v = lo + (hi - lo) * norm
        app.amp.ir_blend = v / 100.0
        if hasattr(app, 'ir_blend_sl'):
            app._set_knob(app.ir_blend_sl, app.ir_blend_lbl, v,
                           f'{100-int(round(v))}% A / {int(round(v))}% B')
    action.nudge_range  = (lo, hi)
    action.nudge_getter = lambda app: app.amp.ir_blend * 100.0
    return action

def _midi_delay_time_action(lo_ms=10.0, hi_ms=2000.0):
    """delay_time is stored in seconds on PedalChain but the Delay tab's
    own slider (and its ms readout) work in milliseconds — a plain
    _midi_pedal_cc_action would either desync the underlying value or
    the on-screen slider, so this converts once, explicitly, instead of
    forcing the generic per-attr sync to assume matching units."""
    def action(app, norm):
        ms = lo_ms + (hi_ms - lo_ms) * norm
        app.amp.pedals.delay_time = ms / 1000.0
        if hasattr(app, 'dly_time_sl'):
            app._updating = True
            app.dly_time_sl.set(ms)
            app._updating = False
            app.dly_time_var.set(f'{int(ms)} ms')
    action.nudge_range  = (lo_ms, hi_ms)
    action.nudge_getter = lambda app: app.amp.pedals.delay_time * 1000.0
    return action

def _midi_wah_pos_action():
    def action(app, norm):
        app.amp.pedals.wah_pos = norm
        app._pedal_widget_sync('wah_pos', norm)
    action.nudge_range  = (0.0, 1.0)
    action.nudge_getter = lambda app: app.amp.pedals.wah_pos
    return action

def _midi_trigger_action(fn):
    def action(app, _value):
        fn(app)
    return action

# Long-press on a pedal's Enable pad (instead of a quick tap) cycles that
# pedal's emulation type, so one pad does double duty: tap = on/off,
# hold = next voicing. Pedals with no "type" of their own (Compressor,
# Reverb, Tuner Mute) simply aren't in here — long-press on those pads
# just does nothing extra.
PEDAL_MODE_CYCLE = {
    'pedal_wah_on':    ('wah_mode',    ['manual', 'auto']),
    'pedal_fuzz_on':   ('fuzz_mode',   ['fuzz_face', 'big_muff']),
    'pedal_dist_on':   ('dist_mode',   ['ds1', 'rat', 'dist_plus', 'metal']),
    'pedal_od_on':     ('od_mode',     ['tubescreamer', 'klon', 'boost']),
    'pedal_chorus_on': ('chorus_mode', ['chorus', 'flanger']),
    'pedal_delay_on':  ('delay_mode',  ['tape', 'digital']),
}

# Each entry: id (stored in the mapping file), label (Settings-tab UI),
# category (grouping), kind ('cc' | 'toggle' | 'trigger'), action.
MIDI_MAPPABLE_TARGETS = [
    # ── Pedals ─────────────────────────────────────────────────
    # Every pedal's targets (on/off pad, then its knobs) are listed
    # together and in DEFAULT_PEDAL_ORDER's sequence — the same order the
    # signal actually runs through — rather than grouping all the on/off
    # toggles first and all the knobs after, so the Settings tab reads as
    # one consistent per-pedal list, top to bottom, matching the chain.
    dict(id='pedal_comp_on',   label='Compressor Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('comp_var', lambda app, v: setattr(app.amp.pedals, 'comp_on', v))),
    dict(id='comp_thresh',   label='Compressor Threshold', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('comp_thresh', -40.0, 0.0)),
    dict(id='comp_makeup',   label='Compressor Makeup Gain', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('comp_makeup', 0.0, 24.0)),

    dict(id='pedal_wah_on',    label='Wah Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('wah_var', lambda app, v: setattr(app.amp.pedals, 'wah_on', v))),
    dict(id='wah_pos',       label='Wah Position', category='Pedals', kind='cc',
         action=_midi_wah_pos_action()),
    # A single CC/knob is a fine way to set Wah Position directly, but a
    # momentary pad can't represent an absolute 0-127 sweep at all — a
    # pad hit is a discrete event, not a continuous value. These two
    # trigger targets are the pad-friendly alternative: press one to
    # step the position up, the other down, reusing the exact same
    # step-size logic (and Settings-tab step-size control) as the
    # existing keyboard Wah nudge.
    dict(id='wah_pos_up',    label='Wah Position Up (pad)', category='Pedals', kind='trigger',
         action=_midi_trigger_action(lambda app: app._nudge_cc_target('wah_pos', 1))),
    dict(id='wah_pos_down',  label='Wah Position Down (pad)', category='Pedals', kind='trigger',
         action=_midi_trigger_action(lambda app: app._nudge_cc_target('wah_pos', -1))),

    dict(id='pedal_fuzz_on',   label='Fuzz Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('fuzz_var', lambda app, v: setattr(app.amp.pedals, 'fuzz_on', v))),
    dict(id='fuzz_drive',    label='Fuzz Drive', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('fuzz_drive', 1.0, 10.0)),

    dict(id='pedal_dist_on',   label='Distortion Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('dist_var', lambda app, v: setattr(app.amp.pedals, 'dist_on', v))),
    dict(id='dist_drive',    label='Distortion Drive', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('dist_drive', 1.0, 10.0)),

    dict(id='pedal_od_on',     label='Overdrive Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('od_var', lambda app, v: setattr(app.amp.pedals, 'od_on', v))),
    dict(id='od_drive',      label='Overdrive Drive', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('od_drive', 1.0, 10.0)),

    dict(id='pedal_chorus_on', label='Chorus/Flanger Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('ch_var', lambda app, v: setattr(app.amp.pedals, 'chorus_on', v))),
    dict(id='chorus_rate',   label='Chorus Rate', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('chorus_rate', 0.1, 5.0)),
    dict(id='chorus_mix',    label='Chorus Mix', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('chorus_mix', 0.0, 1.0)),

    dict(id='pedal_delay_on',  label='Delay Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('dly_var', lambda app, v: setattr(app.amp.pedals, 'delay_on', v))),
    dict(id='delay_time',    label='Delay Time', category='Pedals', kind='cc',
         action=_midi_delay_time_action()),
    dict(id='delay_mix',     label='Delay Mix', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('delay_mix', 0.0, 1.0)),

    dict(id='pedal_reverb_on', label='Reverb Enabled', category='Pedals', kind='toggle',
         action=_midi_toggle_action('rev_var', lambda app, v: setattr(app.amp.pedals, 'reverb_on', v))),
    dict(id='reverb_size',   label='Reverb Size', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('reverb_size', 0.1, 1.0)),
    dict(id='reverb_mix',    label='Reverb Mix', category='Pedals', kind='cc',
         action=_midi_pedal_cc_action('reverb_mix', 0.0, 1.0)),

    # Tuner Mute is a global kill switch ahead of the reorderable chain,
    # not part of it (same distinction PedalChain itself makes) — listed
    # last among the pedal targets rather than implying it has a chain
    # position.
    dict(id='pedal_mute',      label='Tuner Mute', category='Pedals', kind='toggle',
         action=_midi_toggle_action('mute_var', lambda app, v: setattr(app.amp.pedals, 'mute', v))),

    # ── EQ / Amp / Levels ──────────────────────────────────────
    dict(id='eq_on', label='EQ Enabled', category='EQ / Amp', kind='toggle',
         action=_midi_toggle_amp_attr_action('eq_on_var', 'eq_on')),
    dict(id='eq_band_0', label='EQ 100 Hz', category='EQ / Amp', kind='cc', action=_midi_eq_band_action(0)),
    dict(id='eq_band_1', label='EQ 400 Hz', category='EQ / Amp', kind='cc', action=_midi_eq_band_action(1)),
    dict(id='eq_band_2', label='EQ 1 kHz',  category='EQ / Amp', kind='cc', action=_midi_eq_band_action(2)),
    dict(id='eq_band_3', label='EQ 2.5 kHz',category='EQ / Amp', kind='cc', action=_midi_eq_band_action(3)),
    dict(id='eq_band_4', label='EQ 6 kHz',  category='EQ / Amp', kind='cc', action=_midi_eq_band_action(4)),
    dict(id='eq_level',  label='EQ Level',  category='EQ / Amp', kind='cc', action=_midi_eq_level_action()),
    dict(id='in_gain',   label='Input Gain',  category='EQ / Amp', kind='cc',
         action=_midi_gain_action('in_gain', -12.0, 24.0, 'in_gain_sl', 'in_gain_val')),
    dict(id='out_gain',  label='Output Gain', category='EQ / Amp', kind='cc',
         action=_midi_gain_action('out_gain', -24.0, 12.0, 'out_gain_sl', 'out_gain_val')),
    dict(id='gate_thr',  label='Noise Gate Threshold', category='EQ / Amp', kind='cc',
         action=_midi_gate_action()),
    dict(id='hum_on', label='Hum Filter Enabled', category='EQ / Amp', kind='toggle',
         action=_midi_toggle_amp_attr_action('hum_var', 'on', 'hum')),

    # ── Transport / Looper / Tuner (pad triggers) ─────────────
    dict(id='panic_mute_all', label='Panic (Mute All)', category='Transport', kind='toggle',
         action=_midi_toggle_action('panic_var', lambda app, v: app._set_panic(v))),
    dict(id='tap_tempo',   label='Tap Tempo', category='Transport', kind='trigger',
         action=_midi_trigger_action(lambda app: app._tap())),
    dict(id='metro_toggle',label='Metronome Start/Stop', category='Transport', kind='trigger',
         action=_midi_trigger_action(lambda app: app._toggle_metro())),
    dict(id='amp_toggle',  label='Amp Start/Stop', category='Transport', kind='trigger',
         action=_midi_trigger_action(lambda app: app._toggle_amp())),
    dict(id='tuner_toggle',label='Tuner Start/Stop', category='Transport', kind='trigger',
         action=_midi_trigger_action(lambda app: app._toggle_tuner())),
    dict(id='loop_rec',    label='Loop Record / Arm', category='Transport', kind='trigger',
         action=_midi_trigger_action(lambda app: app._loop_rec())),
    dict(id='loop_stop',   label='Loop Stop Recording', category='Transport', kind='trigger',
         action=_midi_trigger_action(lambda app: app.looper.stop_record())),
    dict(id='ir_ab_toggle',label='Cabinet IR A/B Toggle', category='Transport', kind='trigger',
         action=_midi_trigger_action(lambda app: app._toggle_ab())),
    dict(id='ir_blend_toggle', label='Cabinet IR Blend Enabled', category='Transport', kind='toggle',
         action=_midi_toggle_action('ir_blend_on_var', lambda app, v: app._toggle_ir_blend())),
    dict(id='ir_blend_amount', label='Cabinet IR Blend Amount', category='Transport', kind='cc',
         action=_midi_ir_blend_action()),
]
MIDI_TARGETS_BY_ID = {t['id']: t for t in MIDI_MAPPABLE_TARGETS}

def _build_keybind_targets():
    """The Keybinds section (Settings tab) maps the same functions as the
    MIDI Controller section, in the same order — one registry, two UIs —
    so they can't drift apart the way two hand-maintained lists could.
    The one real difference: a keyboard key is a discrete press, not a
    knob's continuous 0-127 sweep, so each 'cc' (continuous) target
    becomes a pad-style Up/Down nudge pair instead, the same idea the
    Wah Position Up/Down MIDI pad targets already use — generalized here
    to every knob via App._nudge_cc_target() rather than one hand-written
    pair per knob. Wah Position itself is skipped in that generalization
    since it already has its own hand-written pair above, reused as-is."""
    out = []
    for t in MIDI_MAPPABLE_TARGETS:
        if t['kind'] != 'cc':
            out.append(t)
            continue
        if f"{t['id']}_up" in MIDI_TARGETS_BY_ID:
            continue   # already has a hand-written pair (e.g. wah_pos) — falls through when reached below
        for suffix, word, direction in ((' Up', 'up', 1), (' Down', 'down', -1)):
            out.append(dict(
                id=f"{t['id']}_kb_{word}", label=t['label'] + suffix, category=t['category'],
                kind='trigger',
                action=_midi_trigger_action(lambda app, tid=t['id'], d=direction: app._nudge_cc_target(tid, d))))
    return out

KEYBIND_MAPPABLE_TARGETS = _build_keybind_targets()
KEYBIND_TARGETS_BY_ID = {t['id']: t for t in KEYBIND_MAPPABLE_TARGETS}

# ─────────────────────────────────────────────────────────────
# LPD8 mk2 pad LED feedback — unofficial. Akai's own tech support says
# there's no documented way to drive this device's pad LEDs from
# incoming MIDI (unlike the original LPD8, which does support it via
# plain Note On messages) — this SysEx sequence is reverse-engineered by
# a third party (github.com/john-kuan/lpd8mk2sysex), not from an Akai
# spec, so treat it as "known to work as of one tester's unit", not a
# guarantee. Everything here is wrapped defensively: worst case if it's
# wrong is the pads just don't light, never a crash or a hang.
# ─────────────────────────────────────────────────────────────

LPD8_MK2_MFR_ID  = 0x47   # Akai Professional
LPD8_MK2_PRODUCT = 0x4C   # LPD8 mk2

# Factory-default Note-mode note numbers for pads 1-8 (channel 1). This
# is the only way to know which physical pad a given incoming note
# belongs to without a pad-index configuration UI of our own — if the
# pads have been remapped to different notes in Akai's Editor, LED
# feedback for that pad simply won't find a match and stays off.
LPD8_MK2_PAD_NOTES = [36, 37, 38, 39, 40, 41, 42, 43]

def _lpd8_pad_index_for_note(note):
    try:
        return LPD8_MK2_PAD_NOTES.index(note)
    except ValueError:
        return None

def _hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _midi_port_base_name(name):
    """Strips a trailing ' <digits>' index rtmidi appends per-direction —
    a single physical device (confirmed against a real LPD8 mk2) can
    enumerate as 'LPD8 mk2 0' on input and 'LPD8 mk2 1' on output, same
    device, different trailing index. Used to match an input port to its
    corresponding output port when their exact names don't line up."""
    return re.sub(r'\s+\d+$', '', name)

def _lpd8_mk2_pad_color_sysex(colors):
    """colors: list of 8 (r,g,b) tuples, each channel 0-255 (the same
    range every other color constant in this file uses) — scaled here to
    the pad protocol's 0-127-per-channel range. Returns the full 8-pad
    'Pad LED Color Update' SysEx message; there's no documented way to
    address a single pad, so every refresh resends all 8."""
    body = bytearray([0xF0, LPD8_MK2_MFR_ID, 0x7F, LPD8_MK2_PRODUCT, 0x06, 0x00, 0x30])
    for r, g, b in colors:
        for c in (r, g, b):
            v = (int(c) * 127) // 255
            body += bytes([0x00, v & 0x7F])
    body.append(0xF7)
    return bytes(body)


class MidiController:
    """Reads MIDI input on its own thread (rtmidi's callback) and queues
    raw (kind, number, value) events for the GUI thread to drain and
    dispatch — Tkinter widgets aren't safe to touch from a non-GUI
    thread, so nothing here calls into the App directly. Nothing in this
    class is Akai-specific; it just decodes standard MIDI Note On/Off
    and Control Change messages, the same for any class-compliant
    controller."""
    def __init__(self, event_q):
        self.event_q = event_q
        self.midi_in = None
        self.midi_out = None   # optional — pad LED feedback (e.g. LPD8 mk2 SysEx) only
        self.port_name = None

    # A single, reused rtmidi.MidiIn() for port enumeration — constructing
    # and immediately discarding a fresh one on every list_ports() call
    # (the Settings tab rebuilds its device dropdown fairly often) is a
    # known trigger for a macOS CoreMIDI crash: thestk/rtmidi#262 and
    # related reports tie it to churning through MidiIn/CoreMIDI client
    # objects, surfaced here as a genuine CI failure — this exact crash
    # (a fatal, uncatchable "PyEval_RestoreThread... GIL" abort, not a
    # normal Python exception) on a GitHub Actions macos-latest runner
    # the first time list_ports() ran inside a real App build. Reusing
    # one instance for the process's lifetime avoids the repeated
    # client-churn entirely.
    _probe_midi_in = None

    @staticmethod
    def list_ports():
        if not HAS_MIDI: return []
        try:
            if MidiController._probe_midi_in is None:
                MidiController._probe_midi_in = rtmidi.MidiIn()
            return list(MidiController._probe_midi_in.get_ports())
        except Exception:
            return []

    def open(self, port_name):
        self.close()
        if not HAS_MIDI: return False
        try:
            midi_in = rtmidi.MidiIn()
            ports = midi_in.get_ports()
            if port_name not in ports:
                return False
            midi_in.open_port(ports.index(port_name))
            midi_in.ignore_types(sysex=True, timing=True, active_sense=True)
            midi_in.set_callback(self._on_message)
            self.midi_in = midi_in
            self.port_name = port_name
        except Exception:
            return False
        # Output port for pad LED feedback — best-effort and entirely
        # optional. Most controllers are input-only as far as this app
        # is concerned, so a missing/failed output port never blocks the
        # input connection above. Matched by base name, not exact name:
        # confirmed directly against a real LPD8 mk2 that rtmidi enumerates
        # its single physical device as "LPD8 mk2 0" (input) and
        # "LPD8 mk2 1" (output) — an exact-string match would silently
        # never find the output port on this hardware at all.
        try:
            midi_out = rtmidi.MidiOut()
            out_ports = midi_out.get_ports()
            out_port_name = port_name if port_name in out_ports else None
            if out_port_name is None:
                base = _midi_port_base_name(port_name)
                out_port_name = next((p for p in out_ports
                                       if _midi_port_base_name(p) == base), None)
            if out_port_name is not None:
                midi_out.open_port(out_ports.index(out_port_name))
                self.midi_out = midi_out
        except Exception:
            self.midi_out = None
        return True

    def close(self):
        if self.midi_in is not None:
            try: self.midi_in.close_port()
            except Exception: pass
            self.midi_in = None
            self.port_name = None
        if self.midi_out is not None:
            try: self.midi_out.close_port()
            except Exception: pass
            self.midi_out = None

    def send_sysex(self, data):
        """Best-effort SysEx send. Returns False (never raises) if there's
        no open output port, or the device rejects/ignores the message —
        the only consequence of either is the pad LEDs staying dark."""
        if self.midi_out is None:
            return False
        try:
            self.midi_out.send_message(list(data))
            return True
        except Exception:
            return False

    def _on_message(self, event, _data=None):
        message, _deltatime = event
        if len(message) < 2:
            return
        status = message[0] & 0xF0
        try:
            if status == 0xB0 and len(message) >= 3:
                self.event_q.put_nowait(('cc', message[1], message[2]))
            elif status == 0x90 and len(message) >= 3:
                kind = 'note_on' if message[2] > 0 else 'note_off'
                self.event_q.put_nowait((kind, message[1], message[2]))
            elif status == 0x80:
                self.event_q.put_nowait(('note_off', message[1], 0))
        except queue.Full:
            pass


# ─────────────────────────────────────────────────────────────
# Pedal chain — pre-NAM effects
# ─────────────────────────────────────────────────────────────

# _SVF, _soft_clip, _asymmetric_clip, DCBlocker, _hard_clip, Oversampler,
# _muff_tone, and DEFAULT_PEDAL_ORDER all moved to pedals.py along with
# PedalChain itself (see the import near the top of this file) — they
# were PedalChain's own internal implementation details and nothing
# outside this block ever referenced them directly.

# Section title shown on the Pedals tab for each pedal — single source of
# truth shared by the _sec() calls that build each section and by
# App.__init__'s accordion-state normalization (see PEDALS_ACCORDION_GROUP
# below), so the two can't drift the way two copies of the same title
# string could.
PEDAL_SECTION_TITLES = {
    'comp':   'Compressor (tube-style)',
    'wah':    'Wah',
    'fuzz':   'Fuzz',
    'dist':   'Distortion',
    'od':     'Overdrive / Boost',
    'chorus': 'Chorus / Flanger',
    'delay':  'Delay',
    'reverb': 'Reverb',
}
# The Pedals tab's 8 effect sections behave as an accordion (see _sec()'s
# `group` param) — opening one collapses whichever of the others was
# open, so at most one is ever expanded at a time. Compressor starts
# open, the rest start closed.
PEDALS_ACCORDION_GROUP = 'pedals'
DEFAULT_OPEN_PEDAL_SECTION = PEDAL_SECTION_TITLES['comp']

# The I/O & Levels tab's sections behave the same way — Audio Devices
# starts open (the thing you most often need right after launch), the
# rest start closed. The Start Amp button lives outside this accordion
# entirely (see App._tab_io) so it's always visible regardless of which
# section is open or how far the tab is scrolled.
IO_ACCORDION_GROUP = 'io'
IO_SECTION_TITLES = ['Audio Devices', 'Levels', 'Noise Gate', 'Hum Filter']
DEFAULT_OPEN_IO_SECTION = 'Audio Devices'

# Pedal Status strip (Pedals tab) — one small stompbox-style icon per
# pedal, each keyed the same way DEFAULT_PEDAL_ORDER/PedalChain.order
# are: (short label baked into the icon, body color). Purely cosmetic —
# see _build_pedal_icon_image()/App._build_pedal_status_strip().
PEDAL_ICON_SPECS = {
    'comp':   ('COMP', '#3d6b84'),
    'wah':    ('WAH',  '#8a3d84'),
    'fuzz':   ('FUZZ', '#c1611f'),
    'dist':   ('DIST', '#a83c3c'),
    'od':     ('OD',   '#c99a2e'),
    'chorus': ('CHOR', '#3d84a8'),
    'delay':  ('DLY',  '#3d8a5c'),
    'reverb': ('REV',  '#6a5a9c'),
}
PEDAL_ICON_SIZE = (56, 72)   # px
_PEDAL_ICON_PHOTO_CACHE = {}   # code -> ImageTk.PhotoImage, built once and reused

def _build_pedal_icon_image(label, fill, size=PEDAL_ICON_SIZE):
    """A small stompbox-style icon: a rounded-rect body (reusing the same
    helper the rounded-button skin uses), a knob, and the pedal's short
    label baked in. Static per pedal type — the live on/off state is a
    separate Canvas oval ("LED") drawn on top of this in the Pedals tab,
    so toggling a pedal only needs to recolor that oval, not regenerate
    the icon bitmap."""
    w, h = size
    img = _rounded_rect_image(size, 10, fill, outline='#000000', outline_width=1)
    d = ImageDraw.Draw(img)
    kr = 9
    kx, ky = w // 2, 26
    d.ellipse([kx - kr, ky - kr, kx + kr, ky + kr],
              fill='#00000040', outline='#ffffff70', width=1)
    d.line([kx, ky - kr + 3, kx, ky], fill='#ffffffb0', width=2)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()   # older Pillow without a size= param
    tb = d.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    d.text(((w - tw) / 2 - tb[0], h - th - 12 - tb[1]), label, fill='#f2f2f2', font=font)
    return img

def _get_pedal_icon_photo(code):
    photo = _PEDAL_ICON_PHOTO_CACHE.get(code)
    if photo is None and HAS_PIL:
        label, fill = PEDAL_ICON_SPECS[code]
        photo = ImageTk.PhotoImage(_build_pedal_icon_image(label, fill))
        _PEDAL_ICON_PHOTO_CACHE[code] = photo
        _UI_IMAGE_REFS.append(photo)
    return photo


# PedalChain itself now lives in pedals.py (imported near the top of
# this file) — split out alongside its helper functions/classes above.

# ─────────────────────────────────────────────────────────────
# Tone player (pitch pipe)
# ─────────────────────────────────────────────────────────────

class TonePlayer:
    def play(self, freq):
        if sd is None: return
        sd.stop()
        t    = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
        wave = 0.25 * np.sin(2*np.pi*freq*t)
        f    = 2000
        env  = np.ones(len(wave)); env[:f] = np.linspace(0,1,f); env[-f:] = np.linspace(1,0,f)
        sd.play((wave*env).astype(np.float32), SAMPLE_RATE, loop=True)
    def stop(self):
        if sd is not None: sd.stop()

# ─────────────────────────────────────────────────────────────
# Process priority — real-time audio needs the OS scheduler on its side
# ─────────────────────────────────────────────────────────────

def _set_process_priority(high):
    """Ask Windows to schedule this process more favorably while the amp
    is running. Even comfortable average headroom doesn't help if the OS
    hands most CPU time to whatever else has focus (a browser, etc.) —
    that's exactly the kind of contention that causes an occasional missed
    callback deadline, heard as a click or a burst of crackle. HIGH_PRIORITY
    (not REALTIME_PRIORITY, which can wedge the whole system if something
    in this process misbehaves) is the standard, safe middle ground real
    audio apps use. Best-effort: silently does nothing on non-Windows
    platforms, or if it fails for any reason."""
    try:
        import ctypes
        HIGH_PRIORITY_CLASS   = 0x00000080
        NORMAL_PRIORITY_CLASS = 0x00000020
        k32 = ctypes.windll.kernel32
        # Default ctypes return-type guessing (32-bit signed int) mangles
        # GetCurrentProcess()'s pseudo-handle on 64-bit Python — it comes
        # back as 0, and every call using it then silently fails. Explicit
        # types are required for this to actually work.
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = k32.GetCurrentProcess()
        k32.SetPriorityClass(
            handle, HIGH_PRIORITY_CLASS if high else NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# Amp processor (live duplex stream)
# ─────────────────────────────────────────────────────────────

class AmpProcessor:
    def __init__(self):
        self.lock       = threading.Lock()
        self.stream     = None
        self.running    = False
        self.in_dev     = None
        self.out_dev    = None
        self.in_gain    = 1.0
        self.out_gain   = 1.0
        self.gate_thr   = 0.01      # noise gate threshold (linear)
        self.gate_open  = False
        self.gate_att   = 0.0
        self.gate_alpha_on  = 1.0 - math.exp(-1.0/(SAMPLE_RATE * 0.003))
        self.gate_alpha_off = 1.0 - math.exp(-1.0/(SAMPLE_RATE * 0.060))
        self.nam        = None      # StreamingNAM | None
        self.nam_on     = False
        self.ir_slots   = [IRConv() for _ in range(MAX_IR_SLOTS)]
        self.ir_loaded  = [False] * MAX_IR_SLOTS
        self.ir_active  = -1        # -1 = bypass
        # IR blending — an extension of the existing A/B slots (ir_ab_a/
        # ir_ab_b on the App side), not a separate pair: when enabled,
        # BOTH of those slots run their own convolution every block
        # (each IRConv keeps its own overlap-add tail, so running two at
        # once is safe) and get mixed at ir_blend (0.0 = all A, 1.0 = all
        # B) instead of only whichever single slot ir_active points at.
        self.ir_blend_on = False
        self.ir_blend    = 0.5
        self.ir_blend_a  = -1
        self.ir_blend_b  = -1
        self.eq         = BiquadState(SAMPLE_RATE)
        self.eq_on      = True
        self.xruns      = 0
        # Global "panic" kill switch — unlike pedals.mute (silences only
        # the live input ahead of the chain), this zeroes the FINAL mixed
        # output in _cb(), after pedals/NAM/IR/EQ/looper/metronome have
        # all already run — so it can't miss a sound source no matter
        # where in the chain it's added later, and everything underneath
        # (looper recording/playback position, metronome scheduling)
        # keeps advancing normally and resumes exactly in sync once
        # un-panicked. Deliberately not saved/restored with the session —
        # reopening the app already-silenced with no visible reason would
        # be its own kind of confusing.
        self.panic      = False
        self.looper     = None   # set by App.__init__
        self.tuner      = None   # set by App.__init__
        self.metro      = None   # set by App.__init__
        self.pedals     = PedalChain(SAMPLE_RATE)
        self.hum        = HumFilter(SAMPLE_RATE)

    def _cb(self, indata, outdata, frames, time_info, status):
        if status.input_underflow or status.output_underflow:
            self.xruns += 1
        raw = indata[:,0].astype(np.float32)
        raw = self.hum.process(raw)   # mains-hum notch, ahead of everything else
        if self.tuner is not None and self.tuner.shared:
            self.tuner.feed(raw)
        # Pedal chain runs pre-NAM, outside amp.lock (has no shared state)
        bpm = self.metro.bpm if self.metro is not None else 120
        x = self.pedals.process(raw, metro_bpm=bpm)
        with self.lock:
            x = x * self.in_gain
            rms = float(np.sqrt(np.mean(x**2)))
            if rms >= self.gate_thr:
                self.gate_att += self.gate_alpha_on  * (1.0 - self.gate_att)
            else:
                self.gate_att += self.gate_alpha_off * (0.0 - self.gate_att)
            x = x * self.gate_att
            if self.nam_on and self.nam is not None:
                try:
                    x = self.nam.process(x)
                    # Safety limiter: some captures have produced runaway
                    # gain (100x+ peak) from this from-scratch inference
                    # engine — left unchecked that reaches the final hard
                    # clip as a shrieking near-square wave, and propagates
                    # into the IR convolution and the looper's recording
                    # too. Only engages when something is clearly wrong
                    # (peak > 1.5); a correctly-calibrated model's normal
                    # output is untouched.
                    peak = float(np.max(np.abs(x))) if x.size else 0.0
                    if peak > 1.5:
                        x = np.tanh(x / peak)
                except: pass
            if (self.ir_blend_on
                    and 0 <= self.ir_blend_a < MAX_IR_SLOTS and self.ir_loaded[self.ir_blend_a]
                    and 0 <= self.ir_blend_b < MAX_IR_SLOTS and self.ir_loaded[self.ir_blend_b]):
                xa = self.ir_slots[self.ir_blend_a].process(x)
                xb = self.ir_slots[self.ir_blend_b].process(x)
                t = self.ir_blend
                x = xa * (1.0 - t) + xb * t
            elif 0 <= self.ir_active < MAX_IR_SLOTS and self.ir_loaded[self.ir_active]:
                x = self.ir_slots[self.ir_active].process(x)
            if self.eq_on:
                x = self.eq.process(x)
            loop_out = np.zeros(frames, dtype=np.float32)
            if self.looper is not None:
                loop_out = self.looper.process(x)
            # Pull metronome clicks if it is sharing our stream.
            # pull() acquires metro.lock internally, so we must NOT hold
            # amp.lock while calling it — but we're already inside amp.lock
            # here. To avoid the nested-lock deadlock risk we instead store
            # the metro reference and pull outside the lock below.
            mix = (x + loop_out) * self.out_gain
        # Metro pull happens outside amp.lock to avoid nested-lock deadlock.
        if hasattr(self, 'metro') and self.metro is not None and                 self.metro.running and self.metro.shared:
            metro_chunk = self.metro.pull(frames)
            mix = mix + metro_chunk
            np.clip(mix, -1.0, 1.0, out=mix)
        else:
            np.clip(mix, -1.0, 1.0, out=mix)
        if self.panic:
            # Everything above (pedals, NAM, looper, metronome) already
            # ran and its state already advanced normally — only the
            # actual audible output gets zeroed, so un-panicking picks
            # back up in sync rather than needing anything restarted.
            mix = np.zeros_like(mix)
        outdata[:,0] = mix
        if outdata.shape[1] > 1:
            outdata[:,1] = mix

    def start(self, in_dev=None, out_dev=None, blocksize=256, extra_settings=None):
        if sd is None or self.running: return
        self.in_dev = in_dev; self.out_dev = out_dev
        # Build the stream and start it BEFORE marking ourselves running —
        # if either raises (e.g. PaErrorCode -9997 "Invalid sample rate"),
        # self.running must stay False or the START AMP button gets stuck
        # thinking the amp is already running and just calls stop() on retry.
        stream = sd.Stream(samplerate=SAMPLE_RATE, blocksize=blocksize,
                            device=(in_dev, out_dev),
                            channels=(1, 2), dtype='float32',
                            callback=self._cb, latency='low',
                            extra_settings=extra_settings)
        stream.start()
        self.stream  = stream
        self.running = True
        _set_process_priority(high=True)

    def stop(self):
        self.running = False
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except: pass
            self.stream = None
        _set_process_priority(high=False)

# ─────────────────────────────────────────────────────────────
# Looper engine
# ─────────────────────────────────────────────────────────────
#
# Architecture
# ────────────
# The Looper lives inside the AmpProcessor audio callback — every block
# the amp processes is also sent to the looper, and the looper's playback
# mix is added back into the output. This keeps everything in a single
# low-latency stream with no extra device opens.
#
# Layers
# ──────
# Up to MAX_LOOP_LAYERS independent layers, each loop_samples long.
# All layers share the same loop length (set before recording starts).
#
# States: IDLE → (arm) → WAITING_FOR_FIRST (countdown beep plays)
#   → RECORDING layer N → PLAYING (layers 0..N-1 loop, N is being recorded)
#   → back to RECORDING next layer OR → DONE (all layers recorded)
#
# Manual mode:  user presses REC to start/stop each layer manually.
# Auto mode:    press START → countdown → records N layers of L seconds
#               automatically, cycling: record → play → record …
#
# The countdown beep and between-layer beep are generated as short sine
# bursts injected directly into the output buffer (no separate stream).

MAX_LOOP_LAYERS = 10

# Looper state constants
_LS_IDLE      = 'idle'
_LS_COUNTDOWN = 'countdown'   # playing a pre-roll beep before first record
_LS_RECORDING = 'recording'
_LS_PLAYING   = 'playing'     # all layers done, looping
_LS_OVERDUB   = 'overdub'     # manual: recording over existing layers


def _beep(freq=880, dur=0.08, sr=SAMPLE_RATE):
    """Short sine burst used for countdown / end-of-loop cue."""
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    env = np.minimum(t/0.005, 1.0) * np.minimum((dur-t)/0.005, 1.0)
    return (0.4 * np.sin(2*np.pi*freq*t) * env).astype(np.float32)


class Looper:
    """
    Thread-safe looper engine designed to be driven from inside an audio
    callback.  Call .process(in_block) every callback; it returns the
    playback block to add to the output.
    """
    def __init__(self):
        self.lock          = threading.Lock()
        self.loop_seconds  = 4.0
        self.loop_samples  = int(SAMPLE_RATE * self.loop_seconds)
        self.max_layers    = 3          # how many layers to record
        self.mode          = 'manual'   # 'manual' | 'auto'
        self.layer_vol     = 0.8        # playback mix volume per layer
        self.loop_vol      = 1.0        # master loop output level

        # layer buffers — allocated when loop length is set
        self.layers        = []         # list of np.float32 arrays, all loop_samples long
        self.layer_mute    = []         # list of bool, parallel to self.layers
        self.n_complete    = 0          # how many layers are fully recorded

        # record head for the currently-recording layer
        self._rec_buf      = None       # np.float32[loop_samples]
        self._rec_pos      = 0

        # playback head (same position for all layers — they're all the same length)
        self._play_pos     = 0

        # countdown beep injection
        self._beep_buf     = np.zeros(0, dtype=np.float32)
        self._beep_pos     = 0

        self.state         = _LS_IDLE

        # status string read by the GUI poll loop (no lock needed — Python GIL)
        self.status_text   = 'Idle'
        self.status_color  = DIM

        # auto-mode bookkeeping
        self._auto_layer   = 0         # which layer we're currently on in auto mode
        self._countdown_samples = int(SAMPLE_RATE * 0.5)   # 0.5s beep before first layer
        self._countdown_pos     = 0

        # event queue: looper → GUI (layer complete, all done, etc.)
        self.event_q       = queue.Queue(maxsize=32)

    # ── public API (called from GUI thread) ──────────────────

    def set_loop_length(self, seconds):
        """Call before starting. Re-allocates buffers and resets."""
        with self.lock:
            if self.state != _LS_IDLE:
                return False
            self.loop_seconds = float(seconds)
            self.loop_samples = int(SAMPLE_RATE * seconds)
            self.layers       = []
            self.layer_mute   = []
            self.n_complete   = 0
            self._rec_buf     = None
            self._play_pos    = 0
        return True

    def set_max_layers(self, n):
        with self.lock: self.max_layers = int(n)

    def set_mode(self, mode):
        """'manual' or 'auto' — only effective while idle."""
        with self.lock:
            if self.state == _LS_IDLE: self.mode = mode

    def arm(self):
        """
        Manual mode: start recording the next layer (or start if idle).
        Auto mode: start the automatic sequence.
        In both cases, a short countdown beep plays first.
        Returns False if already running in an incompatible state.
        """
        with self.lock:
            if self.state in (_LS_COUNTDOWN, _LS_RECORDING, _LS_OVERDUB):
                return False
            if self.state == _LS_IDLE or self.state == _LS_PLAYING:
                self._start_countdown()
            return True

    def stop_record(self):
        """Manual mode: finish recording the current layer and move to playback."""
        with self.lock:
            if self.state in (_LS_RECORDING, _LS_OVERDUB):
                self._finish_layer()

    def clear_all(self):
        """Wipe everything and return to IDLE."""
        with self.lock:
            self.layers      = []
            self.layer_mute  = []
            self.n_complete  = 0
            self._rec_buf    = None
            self._rec_pos    = 0
            self._play_pos   = 0
            self.state       = _LS_IDLE
            self.status_text  = 'Idle'
            self.status_color = DIM

    def clear_last(self):
        """Remove the most recently completed layer."""
        with self.lock:
            if not self.layers:
                return
            self.layers.pop()
            if self.layer_mute:
                self.layer_mute.pop()
            self.n_complete = max(0, self.n_complete - 1)
            if not self.layers:
                self.state = _LS_IDLE
                self.status_text  = 'Idle'
                self.status_color = DIM

    def toggle_mute(self, idx):
        """Mute/unmute one recorded layer's playback (recording is unaffected)."""
        with self.lock:
            if 0 <= idx < len(self.layer_mute):
                self.layer_mute[idx] = not self.layer_mute[idx]

    def get_mix(self):
        """Return the full loop as one mixed np.float32[loop_samples] array,
        honoring per-layer mute and the current layer/master volumes —
        i.e. exactly what you currently hear. Used for WAV export."""
        with self.lock:
            if not self.layers:
                return np.zeros(0, dtype=np.float32)
            mix = np.zeros(self.loop_samples, dtype=np.float32)
            for i, layer in enumerate(self.layers):
                if i < len(self.layer_mute) and self.layer_mute[i]:
                    continue
                mix += layer * self.layer_vol
            mix *= self.loop_vol
            np.clip(mix, -1.0, 1.0, out=mix)
            return mix

    def get_layer(self, idx):
        """Return a copy of one layer's raw (unmixed, full-volume) audio."""
        with self.lock:
            if 0 <= idx < len(self.layers):
                return self.layers[idx].copy()
            return None

    # ── internal helpers (must be called with lock held) ─────

    def _start_countdown(self):
        beep = _beep(freq=880, dur=0.08)
        self._beep_buf  = beep
        self._beep_pos  = 0
        self.state = _LS_COUNTDOWN
        self._countdown_pos = int(SAMPLE_RATE * 0.25)  # 0.25s countdown before rec starts
        self.status_text  = 'Ready…'
        self.status_color = '#ffd166'

    def _start_recording(self):
        self._rec_buf  = np.zeros(self.loop_samples, dtype=np.float32)
        self._rec_pos  = 0
        is_first = (self.n_complete == 0)
        self.state = _LS_RECORDING if is_first else _LS_OVERDUB
        n = self.n_complete + 1
        self.status_text  = f'● REC  layer {n}/{self.max_layers}'
        self.status_color = RED
        try: self.event_q.put_nowait(('rec_start', self.n_complete))
        except queue.Full: pass

    def _finish_layer(self):
        """Commit _rec_buf as a new layer and start playing."""
        if self._rec_buf is None:
            return
        # If the recording was cut short (manual stop before loop end), zero-pad
        if self._rec_pos < self.loop_samples:
            self._rec_buf[self._rec_pos:] = 0.0
        self.layers.append(self._rec_buf.copy())
        self.layer_mute.append(False)
        self.n_complete += 1
        self._rec_buf = None
        self._rec_pos = 0
        # Don't reset play pos — keep layers in sync with ongoing playback
        if self.n_complete >= self.max_layers:
            self.state = _LS_PLAYING
            self.status_text  = f'▶ Playing  {self.n_complete} layer(s)'
            self.status_color = ACCENT
            try: self.event_q.put_nowait(('all_done', self.n_complete))
            except queue.Full: pass
        else:
            self.state = _LS_PLAYING
            self.status_text  = f'▶ Playing  {self.n_complete} layer(s)  — arm for layer {self.n_complete+1}'
            self.status_color = ACCENT
            try: self.event_q.put_nowait(('layer_done', self.n_complete))
            except queue.Full: pass
            # Auto mode: immediately start countdown for next layer
            if self.mode == 'auto':
                self._start_countdown()

    # ── audio callback ───────────────────────────────────────

    def process(self, in_block):
        """
        Called from inside the AmpProcessor callback (with amp.lock held).
        in_block: np.float32[N] — the amp-processed input signal.
        Returns: np.float32[N] — summed playback of all complete layers
                 (to be mixed into the output).
        Also records in_block into the current recording layer if active.
        """
        n = len(in_block)
        out = np.zeros(n, dtype=np.float32)

        # ── countdown state ──────────────────────────────────
        if self.state == _LS_COUNTDOWN:
            # inject beep into output
            rem_beep = len(self._beep_buf) - self._beep_pos
            if rem_beep > 0:
                take = min(n, rem_beep)
                out[:take] += self._beep_buf[self._beep_pos:self._beep_pos+take]
                self._beep_pos += take
            # tick down countdown
            self._countdown_pos -= n
            if self._countdown_pos <= 0:
                self._start_recording()

        # ── recording ────────────────────────────────────────
        if self.state in (_LS_RECORDING, _LS_OVERDUB) and self._rec_buf is not None:
            space = self.loop_samples - self._rec_pos
            take  = min(n, space)
            self._rec_buf[self._rec_pos:self._rec_pos+take] = in_block[:take]
            self._rec_pos += take
            if self._rec_pos >= self.loop_samples:
                # loop length reached — auto-commit
                self._finish_layer()

        # ── playback — all complete layers ───────────────────
        if self.layers and self.state in (_LS_PLAYING, _LS_RECORDING,
                                           _LS_OVERDUB, _LS_COUNTDOWN):
            space = self.loop_samples - self._play_pos
            take  = min(n, space)
            for i, layer in enumerate(self.layers):
                if i < len(self.layer_mute) and self.layer_mute[i]:
                    continue
                out[:take] += layer[self._play_pos:self._play_pos+take] * self.layer_vol
            if take < n:
                # wrap
                wrap = n - take
                for i, layer in enumerate(self.layers):
                    if i < len(self.layer_mute) and self.layer_mute[i]:
                        continue
                    out[take:] += layer[:wrap] * self.layer_vol
                self._play_pos = wrap
            else:
                self._play_pos = (self._play_pos + take) % self.loop_samples

        out *= self.loop_vol
        np.clip(out, -1.0, 1.0, out=out)
        return out

    @property
    def progress(self):
        """0.0–1.0 position within the current loop, for the progress bar."""
        with self.lock:
            if self.loop_samples <= 0: return 0.0
            if self.state in (_LS_RECORDING, _LS_OVERDUB):
                return self._rec_pos / self.loop_samples
            return self._play_pos / self.loop_samples

    @property
    def time_remaining(self):
        """Seconds left in current recording pass (or 0)."""
        with self.lock:
            if self.state not in (_LS_RECORDING, _LS_OVERDUB): return 0.0
            return (self.loop_samples - self._rec_pos) / SAMPLE_RATE


# ─────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────


import json, pathlib

SESSION_FILE = pathlib.Path.home() / '.solotone_session.json'
MIDI_MAP_FILE = pathlib.Path.home() / '.solotone_midi_map.json'

# Default directories — created on first run if absent.
# When frozen (PyInstaller onefile/onedir), __file__ points inside the
# temporary extraction folder, not next to the actual .exe — use
# sys.executable's folder instead so nam/irs/session persist run to run.
if getattr(sys, 'frozen', False):
    _APP_DIR = pathlib.Path(sys.executable).parent
else:
    _APP_DIR = pathlib.Path(__file__).parent
NAM_DIR   = _APP_DIR / 'nam'
IR_DIR    = _APP_DIR / 'irs'
PROFILE_DIR = _APP_DIR / 'profiles'
PEDAL_PRESET_DIR = _APP_DIR / 'pedal_presets'

def _ensure_dirs():
    NAM_DIR.mkdir(exist_ok=True)
    IR_DIR.mkdir(exist_ok=True)
    PROFILE_DIR.mkdir(exist_ok=True)
    PEDAL_PRESET_DIR.mkdir(exist_ok=True)
    for _code in PEDAL_PRESET_FIELDS:
        (PEDAL_PRESET_DIR / _code).mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Tone profiles — a whole recallable sound: NAM/IR paths, pedal chain,
# EQ/gate/gain, hum filter, and guitar tuning. Saved as a small JSON file
# with its own extension so it's clearly distinct from a session
# (session = "how the app was left"; a profile = "a sound worth naming
# and coming back to"). 'solotone_profile' is a format version, not the
# app version, so older profiles keep loading if this ever needs to grow.
# ─────────────────────────────────────────────────────────────

PROFILE_EXT             = '.stprofile'
PROFILE_FORMAT_VERSION  = 1

# Every PedalChain attribute a profile should capture. A plain list rather
# than hand-written save/load code per field, so adding a new pedal
# parameter later (or a whole new pedal) is a one-line addition here
# instead of touching the save/load logic itself.
PEDAL_PROFILE_FIELDS = [
    'mute', 'order',
    'comp_on', 'comp_thresh', 'comp_ratio', 'comp_attack', 'comp_release',
    'comp_makeup', 'comp_warmth',
    'wah_on', 'wah_mode', 'wah_range_lo', 'wah_range_hi', 'wah_pos', 'wah_q',
    'wah_auto_rate', 'wah_auto_depth',
    'fuzz_on', 'fuzz_mode', 'fuzz_drive', 'fuzz_vol', 'muff_tone',
    'dist_on', 'dist_mode', 'dist_drive', 'dist_tone', 'dist_vol',
    'od_on', 'od_mode', 'od_drive', 'od_vol',
    'chorus_on', 'chorus_mode', 'chorus_rate', 'chorus_depth', 'chorus_mix',
    'delay_on', 'delay_mode', 'delay_bpm_link', 'delay_note', 'delay_time',
    'delay_feedback', 'delay_mix',
    'reverb_on', 'reverb_mix', 'reverb_size',
]

# Per-pedal mini-presets — a Tone Profile captures the whole chain at
# once; these capture just ONE pedal's own settings, reusable across
# different profiles (e.g. "my Klon settings" loaded into whatever amp
# you're using today). Derived from PEDAL_PROFILE_FIELDS by prefix rather
# than hand-listed again, so the two can't drift apart the way two
# separately-maintained field lists could — 'mute' and 'order' are
# chain-level, not any one pedal's, so they're deliberately excluded.
PEDAL_PRESET_FIELDS = {
    code: [f for f in PEDAL_PROFILE_FIELDS if f.startswith(prefix)]
    for code, prefix in {
        'comp': 'comp_', 'wah': 'wah_', 'fuzz': 'fuzz_', 'dist': 'dist_',
        'od': 'od_', 'chorus': 'chorus_', 'delay': 'delay_', 'reverb': 'reverb_',
    }.items()
}
PEDAL_PRESET_FIELDS['fuzz'].append('muff_tone')  # Big Muff tone control — doesn't share the fuzz_ prefix

PEDAL_PRESET_EXT            = '.stpedal'
PEDAL_PRESET_FORMAT_VERSION = 1

# ─────────────────────────────────────────────────────────────
# App icon — the "Tally Light" mark, embedded as base64 PNG so the
# window/taskbar icon works from a single file with no extra assets,
# whether run from source or packaged into the Windows .exe.
# ─────────────────────────────────────────────────────────────

_APP_ICON_PNG_64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAATtUlEQVR4nO1beXBUR3r/uvu9eXPp"
    "QAiQsDHIBoERBb4PDhuWGGOHWDII2XF87Cbe1OYPZ6tsJ39kkwIqf2xcZe9WbWXL5fWmyrXGFCVx"
    "mcSxseNIwUDhA2SQYI0MSDE3QhIaSTPzru7Ur+c9eTwWNpedpFZf1dRoZl53f3f/vq9bRKM0SqM0"
    "SqM0SqM0SqM0Sn+QxL6PRZRSbPXq1aympoYdOHCALVy4cMTnWlpaqKamRq1cuVJhGGMM7//vFMBW"
    "rVoFITk+LFy40L9cQaC4lpYWgb9bWlrkmjVrtGKuKrN0FQiMNjU18XHjxrFFixZ5hb83NzeX9vf3"
    "T+SclymlKn3fnyCltPKEYaZp2lLKbiI6GY1Gexljp5YuXdpbONeqVauMwEvk1fAQ40oGNzY2Crg0"
    "YwxC++H3W7dunSGlnEtEtyql5vT19VUTUbkQAoIS59o5vkZSSvI8jzKZDDHGejZv3vw5Y2wfEe1h"
    "jO3au3fv79esWePlr4/3hoaG4bW/Dw9gjY2NPN8CW7dujRPRfCllrVJqvlKqJh6PC8aYFsh1XfJ9"
    "zaPMOYwa0XLQZMATF0IQlGUYORsNDQ1hzO855zuwJBFtr6urG8hXxuV4BbtUN8/X9ptvvjmPiB5X"
    "Si01TXMKmHUcR7+UUn6QyJhSiudkI/yN30ZmhjH9CtbDWBkqizFmRCIRwgvKtG37uBBim5Tyjbq6"
    "uubLVQS7mIcwaSh4Y2NjzLKsOiL6Med8ERiybRtCw7pYGP4dWnLYtUGwKpSE9wJh9QuCwWMCbykM"
    "FZ0AlVKYjJumyS3L0s97nrdLCPEb3/c3hV6Rz/NlK0ApxcPt6JVXXolXVlb+UCn1U8uyqsFkJpMB"
    "U36+6+aPh+AQAoziPZ1OU09PD507d45SqZSOdVA0GqXi4mIaO3YslZeXUyKR0ArJZrPDcxRQfiiJ"
    "WCzGoFTHcTqVUr9ijP3moYceSl+MItiFfsgfuGXLlkcZY/8QjUZnBtaG0Ho8LGKapgEm8Ru+D108"
    "Fovp+P/8889p3759dOTIEert7R0WLHwOY0JFjRkzhq6//nqaM2cOVVdXa+VAceFzGAOvAzmO4wUe"
    "p3VhmqbA89ls9jMiemHv3r2/W7NmjfwmJbCRvly1ahXHwE2bNt1hGMbPI5HIDyCIbdva2lgMb1gw"
    "Ho/DmsellP2GYcxE/FuWpZ3i008/xRZInZ2d2q3DpJbv/oUeEyZNKGTy5MnAEXTrrbfq57PZrIpE"
    "IszzvENEZJWUlEyBMm3blnkKwjMCynQcZ6dt239XX1+/PZdSvp4X2IWEb2pqWh2Lxf6ecy7S6XTo"
    "5iAVjUYFGHRd9wgR/SqbzXbE4/FfO44zBcKfPn2abdy4kdra2nS8g5nQehdKgMMM5SkHHgWFzJgx"
    "g1asWEHXXXedCpRwMp1O/3UkEinnnD9nGMb0IGRgZT0YiojFYgLfZzKZX7S3t/8NvodsF1RAY+Aq"
    "sHwsFvsQMSqlhPCYSEYiESQfMLabiP7ZcZz12Wy2qrS0dBdjbBznXLW2trJ169bR4OAgwTsuRuhv"
    "Uwb4gGs3NDTQXXfdpXxf22NgYGDg3scee6x1y5Yt9UT0bDQavTvw1Fwyxk7EGCsuLuapVGrJihUr"
    "3isMB56/IEAN3n3fvyHnkdILhFeJRALPHrZt+6nW1tZ5dXV1bwghxhcXF+8ionFCCH/79u3s1Vdf"
    "1dsghM+P88shjMUcyCV4f+211+jdd99FwgNvRYlEYntjY+PUurq6Da2trfMzmcyfSSkPJpNJLjGc"
    "cyEZ832lIAhkouv7+r4iMx8pBKCDYO+GQvx4PM4ymcwLe/bsmV5XV/c74PzGxkZkovWmaUJ4b8eO"
    "HQKWh7vD7cOt72pQuBNAqQit999/nxuG4QshkoZhNAKIodB6+OGH19XW1tYM2fbPkpEIs7JZGc9k"
    "WDST4UVDQyYxRrf95CeuyvN8nr/QxIkTGWJEKeViQSwMd3NdV3HOS2tqaq7Hc8D7nPNnksnkPb7v"
    "ex0dHcb69eu1pfJ3gatJ4ZzYIqGE/fv3wzO9RCJxMxJd6Nb/9tvfTq7s65tQ1NurioeGqGhoiCUH"
    "B2nimTPXqMWLrzv17LPjGaYLlMAKk19jY+NMy7LWKaVmB8CFSSmR8Vk2m806jvNTzvlGxlgX5zye"
    "yWTYSy+9xLC3w/pX0/IjUQivk8kkPf/886qkpAQG8z3TvCE6ODiv1HFejZtmUdZxVFYp5kqpDCRi"
    "zzs2qatrzaTOzvb02LEnEr/85QkogYXCr169Wm3atOkazvlH8Xi8MpPJyEOHDvHjx49TUVER3Xjj"
    "jX55eTl2BAz5jDE2w7IstXHjRrZt2zYNZDSCG8ZEKtBzoONL9Yr8cQVzwjADAwO0YMECevzxx5Xj"
    "ODBSR3EqNbXUsnjP+fP+kc5OUW1ZVJVIkMOYGqyoYGnOB6Z3dDxdcfLksX7L+qz05Zf7dKWB+AHu"
    "3rBhwy+Kiooqe3p63HXr1pl79+4NeFBAaeLJJ59UACe+78/A16dOnWK7du3SsZkTnhP5LpHnAffm"
    "PksfQUwUsXKCfJsiwmfsLLAwyiIioF/Mj8JIGHotrPnxxx/Tgvnz2cRJk1TccaqTnFPnsWPqvXfe"
    "EU+UldHc8nIS587pWe0TJ7zO2bOLOqZMeb6iq+uZkqKiyUTUx2F9xM/mzZsnEVGt67py27Ztxu7d"
    "u7WbIebgAefPn6e1a9eyVCoFIIItke3cuVNvd0IYOcbtLKlEMXm3zyN3SR05S+vIXbyM/Btn54Rz"
    "3ZxSLig8zz2DzHvjbD1Wz7GkTs+pksUAB3qtEHl+8MEHZBoGE+m0tG1bvfn22+z+RIIWTJlCzDBI"
    "BS/LdY2q9nYliG7rnD69mvr7zXPPPFOM5kK49d2USCQiqVRKtrW1MQifv40hwfX19dHp06cRDhwK"
    "2b9/v457Bde0bfKrZ5J3+3xS8QQxeAEwAOfkT5lKfnUNmTv+g1hvT84bdE1TILxjkxozltz5i0lO"
    "mEgMVgcPSKzXTiZ/+iwyPtpBouMASSuq124/eJB6urtpqmXxk2fPEqXT9IPqapKA5TmsnvNiKCGb"
    "lZW2Lc6VlU2rams7EBszpkh3cQIW4ogtaBH7eD5Uxd8AGChWKioq9Oeuri5d1AAYKTtL/pzbyLtn"
    "iXZTlh7KDTSEdl18VuXjyVm6nNT4CiIn+2WMh24P4cdXkPPAclLjJhBLDxL5Xm4OPII5haHX8Ofc"
    "rr3NME3tmUcOHybBORWXlNCEsjIS2DYv4GSm76N6Q5ECOb4ss/Lq7gtmXggPz8CjR48eBVgg5jrk"
    "z5hN3h0LiOxMLn6jUeKnjpE4sI/YYD/chyibJYpEyFm8jNSY8lyewFp4eR6p0jL9G57R8R9LEBsc"
    "0HNgLrKiQW7I6LX8mTcRg9KI6MjRo+QpRclYjERxMXUNDuoc5MN7gCPgichDnFMqkSDD9xUZBiCu"
    "fdEtMSSe8ePH62IG3nDy5EmCbaQVI2/2rTmm4XSMkbHzP0kcOkBMSVJWlLw779HhQbBiPEHeTXeQ"
    "2fzvQUeOaUv7s27WvxEsH0uQ6DhIxkfbiWUzpBgnf3oNeXfem2Mmm9FzGMc6yeg/TyfOnKEhKamU"
    "cyotK6O1+/fTz8GraeoQYkjIUtKJsWOpP5mkolQKVnbOJZOpS+oJolQNk09/fz/6VkTSI4bytnQs"
    "Md8js+Ud4ofatTCIXYSAsf3dnBKRDIHvs5lCHyPKpHNZvqiU+KE2MrZvIzIjpKKx3NZ3cJ/2Nnfh"
    "UlKGSbynWytOGIJS/f3UpxSVcE7jS0po58AArWlvp7qJE2lSLEZDnkc7enupZNo0qsRyUpp07Njp"
    "qqam7EUrIISiCAfkCOABrQDPJ/OD97Rw7Owp4kcP5SwJd8ULUWZGyNj9X8QGUqQArg60EhlmsCUq"
    "IpTJbXt00tO/t+/VY/TYEFjFEsSPdpCJ+mDidSSgZNsmLgyyMxkt5FBJCUUjEUqaJh1Ipai9v1//"
    "baPE5pz+HLkQpjfNHrZ7dy8Ko0vygLBBiXDAS2cLJL1Un87wCns2Yjh/r88DQ6J1d5CJIl9PglIW"
    "/M4L5slhCd51mMSRQ9oLFNhXUu9W0vMoG4vRADCJlFRsWfrdk5IMzrUihFLkWBadmDDhNKYcN24c"
    "uyQF5PfqsGPovImXwF4bWPSbgI4V+1KYQoISvvL7CPNgboRFJABLBR0lZPQeKamtv5+uQ92AzjK2"
    "YSkpJSX1xeNkFhWR5/vDyd+4WOGDxsJwSwr1ObagYcYuBuqOJPil/F6wFgSHUYAH8AI5mQylPY+6"
    "cLaA3IFnsC0aBmUMI7c9BoYEfQMs+ypBcAiMdywG7A/XG2nb/D4JCgAvMIjmsb9fC24i2QZbIMjA"
    "5xGMxC92IbjY2bNnNR4A+KmsrBwOif8tCj0A+ARGwd9nzpzR4VQo6oX8k+dNFvbSvvYQvkMCBAwe"
    "GhrSyqiqqrrgEVduZv7VRHe5hDn0OheeC7wgJyFEgU/A64XkyE35JWM873ccTn7lOCp/IOIe0PfU"
    "qVP6MxYFLoBHfDUMkKAkMezruvi5AiVgLCAr5kJVWTAXLI5CberUqZoneCg8ALwWKgAKwvdBfWMP"
    "K6ClpUVnHill+9DQkI8G4rRp07SlMSjM+Nj7EWcQGgciqAtqamqGzwKGhQejQIc33UGy8tocQgwh"
    "76UIHlSXckIleTfdqQusYfgchCTWnj59Ok2YMAG4RJWWlurqFbyGvMOYkOWGG26gsrIyHsiFA1fq"
    "7u5WuguE/l9DQ8NhKWUzytxly5a5mBhJDwPQfMCEjzzyCOAwFMZ831fz5s0r6ALl9nx3wR+Re/dC"
    "cu/7E43ZtRICLP6tFIIfFFgz55C7pJbcu+/VCFAjxWAtWBg8oSkSnFOg+6seeeQRBZ7AM8AaEOuU"
    "KVOotrbWQ1fbcZxP9u/f/2nYBmDBZPp0ZcOGDTMjkciHkUgkMTAw4H7yyScCHSEUQLfccoucNGmS"
    "8DwP/QB0Gcqj0ah6/fXX2Y4dO7Qr+q5HFDHJrnuMyAAgkrqIQflqfPSBxvC6FIaQhTEagCFUhRij"
    "C57ps770ICKKbHmDGBCoadLQ4CB4oqefflq5rouOUI8QYizc/OTJk96ePXvQCtfJ+vbbb4dnm67r"
    "eo7jzF+5cuWHYXuchesHXWC5fv36RfF4fG08Hp8I184/m4PLZbPZf/Q87w3DMNoNw+B9fX3sxRdf"
    "ZNA2ig/loC9QQ968xUSemytpo3FifT0k2veS6Dqcqwx5XliEe3vEIr9qGvmzbtF9AcqmNcgCbDY+"
    "3J6rBxDHvq+t/9xzz2EH8KWU3LbtW0zTXGya5osI1Xze4SHpdLonnU7/6NFHH/3XUFYqTK3hD42N"
    "jRWWZf2V7/tLcKNDKZUWQuz1ff9f6uvr9VH0xo0bX0gmk38Lrba2tho4D0DTRNsVsVtVTe7cRbo0"
    "1pYHvDVMYr3dxI//N/HuMzkBQdEYyXEVJK+ZTGpsUCq7jv4eyjJ3NRPv7NCewRnTXainnnqK5s6d"
    "6wkhjMHBwVfr6+v/ElM1NTXNjUQiT3uedwcRJRljZ4UQ79u2/XJDQ8MXhQcjrDAE87UD6ujosKZN"
    "m+aE52rhrQx4pGmaHxmGMQuHIm+99ZbYsmWLDgVd4qCMHTOWvLmLdPGiBcKugBJVw+agzwcK+4fw"
    "mPAZM0Ls1DEtPOs9p5UBVIeYfuCBB2j58uW+53nC9/0uxtjNrusOFF7RAe/V1dXDGX+kQ1L2TZeT"
    "8i845V9HCZXU1NRUY1nWh77vx03TVFu3buVvvfWWzsSoFCXiGVj8hhk6oamycTnBkSu+1hJjOcEZ"
    "1wKLg5+SOPJZrp5H3pBSJ7b77ruP6uvrJbZfHI5ks9kF+TE9wrUZ1tzcLIJLVl/D2mwkBRQqI8RI"
    "+d/nLfjHsVhsi+u6IhKJqObmZr5582bdgYnieAwlLhJZNEb+pCqSk6p0e0zX+bA8CJWlnSXWfYb4"
    "sU4Sx7p0eDC4vBCUDWqQZcuW0f333y/RkEFGT6fTf9rQ0LD+Asffw6fY3yQfoyug5uZmAy7X1NRU"
    "a1nWBimlYZqmPinasGGD7hsiIZkoVKCIoIWlmyWxBCldwLCc8Jkh3dAEiNUWF4Jcx9HCX3PNNbA6"
    "zZo1y7dtWwCsZTKZJ1auXLk25OFyZWBXooACJSy2LGutEKICh6q2bfOdO3dytK2BzpCNzQjODXmu"
    "KMGWF4YB4h/ZGq0v6ZOLDrPvo14nYI358+fjcBbhaEgpz9u2/cOVK1e+eaXCg65KKYe7e7i+tnbt"
    "2snFxcWvmKZ5P9zUMAyvv7+ft7W1cdwQ+eKLL3QcQzh4Qv4dIRCOsIA5rr32Wpo9ezZuicgxY8Yg"
    "3vUFKdu2P7Bt+8cNDQ2HrobwoKtWy+bH4ebNm/9CCPEz0zSrApSIM30fmOH06dO8u7sb2Rwnzvl3"
    "hBQsXllZKcvKypTARg/fyEHeE4yxf6qtrf015rrYC1AXQ4yuIoVnjMGlqpKKiooniOhHSqlbIGRI"
    "waWJ/GuvSLT5RZoGXUqpdsbYa77vv7Z8+fKecI2RsvnlEqPvgAosxLZu3XqXlPIBKeUCKSWus4zH"
    "/aKckXNVHY7kGGPdnPMOXIYUQrydyWR25V3Pu2pWzydG3+0NcZF/tRXU3NycTKVSFaglPM8rCsw+"
    "YFlWD2Ps9IMPPpgqeN64kgvX/xeIwXoQJLh98q2Kw7MBoPnO+22MvmfK/9+B/O8PHDigwvzxffM0"
    "SqM0SvQHS/8Djpgj1oWwLAsAAAAASUVORK5CYII="
)

_APP_ICON_PNG_32 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAH6ElEQVR4nO1XaWxU1xU+975lZh5j"
    "42Vm7FjGwlsbs9gywg0KtAJRAkUsKtgDSAn50SK1Yf0DbfljW0hVVRVEExmJH5Ea0QbbUxmKBKUo"
    "saEBqoSGrRgMOKaGOMWz2eOZN3jeck913tiNcWhL2/yo1B7pztx331m+e7Z7H8D/6X+d2L8qgIis"
    "paWFzZ079ynZ3t5ebGlpQcYYfukgOzs7pZ6eHrm5uZn/M2biIV6SeZ4Nsn+kaOnSpXzZsmXW1PVT"
    "p05piFiEiIVCCDetIWJGluWo2+0Or1y5Up/KT2DOnz8vWltbxXMBQEQWCoV4MBi0JxWMjY29hIjL"
    "EXEJItYIIQKSJKmSRJsEsG0bLMsyOedhzvkdzvklzvn7T548+TAYDBrEQx5pamoS00PEpj4Q06Th"
    "EydOzOacv4aImzjnc10uF5im6fCRYURE27YdZZxzRoOAECmKAoZhELA7jLGQaZrvNDY2Dky3MR0A"
    "zfHYsWOlubm5exFxq9frzaMX8XgcJUmyNU1jkUiE9fX1scHBQZZIJBzB3NxcKCsrw5qaGgwEAphO"
    "pwmcVFBQ4OjXdT3BGDtmGMbPGhsbB7/ggebmZt7S0gJdXV3fVxSlVdO0QkQkwUgmkzmnadorlmX5"
    "zp49C5cvX2ajo6NZYZbFT7xEM2fOhEWLFuHq1atBVdW4rutnVVX9ptfrLSLetK6PmIZxoODWrbfO"
    "Azh5wRCRU1xOnjxZI8tyr2U5OfdYkqRfplKpN71e7+F0Ov3ttrY27O/v55qmgSzLz8onJxd0XYeK"
    "igqxfft2lpOTczqZTL7h1bSdFsCrAPCCW1UhYVnzN69bd6uzo0PioVDIcX0ymSQFwuPxUHhb1q5d"
    "u9fj8Wx3u90bDh8+LAYePOAz8/NBkmUQQjxzMEkC4hl8+JAfOnRIKIqyxuPx7F27fv0+LZ3en2ea"
    "QhkdFV/55JMyhgi9TU3IyP2zZ89W/X5/KJVKrRkaGhLl5eXjlC+qqm45fvy40nPhAsvTNGZlMgCc"
    "ASgq+Z98PxHIiblpAAgE2eWCRDqNi19+GV/bulXIyeQxJZUKWvG4VuL1sjTn1xf0939XOXjwmhPE"
    "M2fO7NR1/c0DBw7Y8Xhcqqurg927d8P9+/fh4KFDoKAAURgA4SsCpieBf/Yoa5hnyxAEJTWCKCkD"
    "nJEDPPoYpFgUMoiwZ88e+NqsWfBRVxcEFQWKXS57vKxMuu3xvL2go2OfE0xVVRd3d3eL4eFhzM/P"
    "h3v37mEsFhPXrl/nRirFpJeWgNGwGNDldnYo3b8NysX3JwwDgCyDuXg52NVzHA+xzDioVy6B+dFF"
    "uHn1KpS73XaZrvPisjJmGga4h4eFVlLyItTX5zmtlYLOGOOT9VtaWspm5ORI/b29jL84D4xlq4DF"
    "IqCe/jXIf/oY7Dl1YH5jBYAQlHlgLlnurNE74iFeY+kq4HPq4H5fH7g9HimqaUxkMk6PsGfM4ClN"
    "YzA0NP5UOlM5UTKRFyzThMRYAqR8FeDebZA/eA/42CjwwQEAqhTO/9ZFWHIM5D9eAvkPF4AJG3gs"
    "DOaSFc58JJmEcUmCPwsBrbdvw2yPB7TaWqjIzR2HU6eGnwJAtUogqMwEtVfGQfp0EKShh4DU/bQZ"
    "zq7lKx9MuF7J/l25mH1WVUAClsmAeu43YNOGNA10WYYxVYXeaBR+LwSsqqmBIp9vlAHYDgDqA5PG"
    "nYaRTjuucrvdoKfTIClKNssns56qIOuyLz7TIBAE2LSAWrhHVWEwkYAH6bQTsseMgamqjv/4xOa5"
    "ZVkOCFVVIRwOUyiwuLjYCYXDSfG2zM9LbtL4VMP0jnioJwBFyoRAIEDrGA2HQVYUMMkrlkXvPweA"
    "iMnCwkKWTqft8fFxjEQitmEYbN68eWhPKCN3Y4E/W+uOJOUAqWHZOZFhZHkUFRggWJYNjg7bZuFw"
    "2M5kMnRO2AWF1OnRObY5/SQSiV9Q7W/ZssVVVVXFtm3bJlFHrK2tZUXFxWikkiCqayDT9DrYdQ2O"
    "G2H8STYZadDctsGuWwiZxtdBfHUumLoOvoAf6uvrmcvlEqSzurqaBYNBtaGhgTb7thN+6oR0KHR1"
    "dW3Izc3db9t2EQBcj0ajP87Pz3/nxo2b1W/9/LCd80KJZHx9BdizyoE/HADpzg3g8Sj1H8ACH1hz"
    "6kCUVYD06AGoF9+D5Gefiu+9sZ0tXLjwUSwW2xQIBH6AiA2KooRHR0d/unHjxnay7Xg32wayFwW6"
    "8axbty5N83fffXdFUVHRua4TJ+yuzg7mnZnHYcEiMGtqQWheYAZ1bABU3cDTOih9N4Bd/RCSIyNi"
    "7YYNuHnTJikaja4JBoOnp+ueatOhiTucQ4Ts6NGjTo11dHRs6+7uxh/t349VVVVmqd8vKubNx6rV"
    "67Hy1e84g+a0Vhrwi8rKSnPvvn3Y09OD7e3tO0kH6aJT91m2GEyjqciam5vl1tZWq729vcnn8x2J"
    "xWK+3579Hdy8+rE9Eo2gZWavi9Q38nw+Nr++XvrWqlXg9/tHotHozs2bN/9qUsd03X8XwHSavEK1"
    "tbXNqqys/CHnfNN4JlMYj8chkRhzDiG6ERUWFILb7RpBxNDdu3d/smvXrgfTr1//9ndB5xRFR44c"
    "CZSXly9RFKVeCFFCWwKAv5imeW1gYODijh07Hk+X+VIIEZ3vg+cB+zzfD//Rl1EoFOJ+v/8p2Ugk"
    "gs+6dv/X018BfP8mZuyqdf8AAAAASUVORK5CYII="
)


def _set_app_icon(root):
    """Best-effort: set the window/taskbar icon from the embedded PNGs.
    Never fatal — a handful of odd Tk builds lack PNG PhotoImage support."""
    try:
        icons = [tk.PhotoImage(data=_APP_ICON_PNG_64),
                 tk.PhotoImage(data=_APP_ICON_PNG_32)]
        root.iconphoto(True, *icons)
        root._icon_imgs = icons  # keep a reference — Tk drops it otherwise
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# App — redesigned layout
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        global ACCENT, BG   # possibly overwritten below by a saved custom theme
        super().__init__()
        self.title(f'SoloTone  v{VERSION_FULL}')
        _set_app_icon(self)
        self.configure(bg=BG)
        self.geometry('820x720')
        self.minsize(740, 600)

        self.beat_q  = queue.Queue(maxsize=16)
        self.metro   = Metronome(self.beat_q)
        self.tuner   = Tuner()
        self.tone    = TonePlayer()
        self.amp     = AmpProcessor()
        self.looper  = Looper()

        self.amp.looper = self.looper
        self.amp.tuner  = self.tuner
        self.amp.metro  = self.metro
        self.tuner.amp  = self.amp
        self.metro.amp  = self.amp

        self.a4         = tk.DoubleVar(value=440.0)
        self.trans_var    = tk.StringVar(value='C  (Concert)')
        self.tuning_var   = tk.StringVar(value='Standard (EADGBe)')
        self._updating  = False

        # Collapsible Settings/Pedals section state (title -> collapsed bool)
        # and the saved theme colors — both pre-read here, before _style()/
        # _build() below draw anything, so a collapsed section doesn't
        # visibly open then snap shut, and the very first paint already
        # uses any custom accent/background instead of flashing the
        # factory colors first. _load_session() only re-checks the rest
        # of the file later, after the UI already exists.
        self._section_collapsed = {}
        # title -> outer section frame, and group name -> [titles] — built
        # fresh by every _sec() call as tabs are (re)built, never cleared
        # elsewhere, so a single-tab rebuild (_rebuild_pedals_ui) leaves
        # other tabs' entries alone and a full rebuild naturally
        # overwrites every entry as each tab is reconstructed.
        self._section_outer_by_title = {}
        self._section_groups = {}
        try:
            with open(SESSION_FILE) as f:
                _early_session = json.load(f)
            self._section_collapsed = _early_session.get('section_collapsed', {})
            accent = _early_session.get('theme_accent')
            bg     = _early_session.get('theme_bg')
            if _valid_hex_color(accent) or _valid_hex_color(bg):
                if _valid_hex_color(accent): ACCENT = accent
                if _valid_hex_color(bg):     BG = bg
                _recompute_derived_theme()
        except Exception:
            pass

        # Enforce the Pedals accordion's "exactly one open" invariant
        # before first paint — covers a fresh install (no saved state at
        # all), and a session file saved before this feature existed
        # where the ordinary independent-collapse mechanic could have
        # left more than one of these 8 titles open at once.
        pedal_titles = list(PEDAL_SECTION_TITLES.values())
        open_titles = [t for t in pedal_titles
                       if not self._section_collapsed.get(t, t != DEFAULT_OPEN_PEDAL_SECTION)]
        keep_open = open_titles[0] if open_titles else DEFAULT_OPEN_PEDAL_SECTION
        for t in pedal_titles:
            self._section_collapsed[t] = (t != keep_open)

        # Same invariant for the I/O & Levels tab's accordion.
        io_open_titles = [t for t in IO_SECTION_TITLES
                           if not self._section_collapsed.get(t, t != DEFAULT_OPEN_IO_SECTION)]
        io_keep_open = io_open_titles[0] if io_open_titles else DEFAULT_OPEN_IO_SECTION
        for t in IO_SECTION_TITLES:
            self._section_collapsed[t] = (t != io_keep_open)

        _ensure_dirs()

        # Path tracking — updated whenever a file is loaded, saved to session
        self._current_nam_path  = None
        self._current_ir_paths  = [None] * MAX_IR_SLOTS
        # Last folder browsed in each file dialog, so it reopens there next
        # time instead of always resetting to nam/ or irs/
        self._last_nam_dir = None
        self._last_ir_dir  = None
        self._last_profile_dir  = None
        self._current_profile_path = None
        # code -> preset name last saved/loaded for that pedal, or None —
        # survives _rebuild_pedals_ui() (a full tab rebuild, the same
        # mechanism a Tone Profile load already uses) since it lives here
        # on the App instance, not on any one widget that gets destroyed.
        self._pedal_preset_name = {code: None for code in PEDAL_PRESET_FIELDS}

        # MIDI controller mapping (Settings tab) — {('cc'|'note', number): target_id}
        self.midi_q = queue.Queue(maxsize=256)
        self.midi_ctrl = MidiController(self.midi_q)
        self.midi_mappings = {}
        self._midi_learn_target = None
        self._midi_pending_press = {}   # ('cc'|'note', number) -> {'timer', 'fired_long'}, while a pad is held

        # LPD8 mk2 pad LED feedback (experimental, off by default) —
        # lpd8_led_var itself is only created if HAS_MIDI (Settings tab,
        # MIDI Controller section), so every reader of it goes through
        # getattr(self, 'lpd8_led_var', None) rather than assuming it exists.
        self._lpd8_flash_until = {}   # pad index -> time.monotonic() deadline, for trigger-pad flashes
        self._lpd8_last_colors = None   # last 8-tuple sent, so unchanged state isn't resent every poll
        self._load_midi_mappings()

        # Keyboard keybind mapping (Settings tab) — {keysym: target_id},
        # same shape as midi_mappings but keyed by Tk keysym instead of a
        # (kind, number) pair; loaded from the session file in
        # _load_session() since it travels with the player's setup, not
        # with a particular Tone Profile (see KEYBIND_MAPPABLE_TARGETS).
        self.keybind_mappings = {}
        self._keybind_learn_target = None
        self._keybind_pending_press = {}   # keysym -> {'timer', 'fired_long'}, while a key is held
        self._keys_down = set()            # keysyms currently held, to swallow OS key-repeat

        # IR per-slot labels (user-editable)
        self.ir_label_vars = [tk.StringVar(value=f'IR {i+1}') for i in range(MAX_IR_SLOTS)]

        self._style()
        self._build()
        self._load_session()

        if sd is None:
            self.after(100, lambda: messagebox.showwarning(
                'Audio unavailable',
                f'sounddevice/PortAudio not loaded:\n{_SD_IMPORT_ERROR}\n\n'
                'Install PortAudio and restart to use audio features.'))

        self.protocol('WM_DELETE_WINDOW', self._close)
        self._poll_beats()
        self._poll_tuner()
        self._poll_amp()
        self._poll_looper()
        self._poll_midi()
        # Restore file paths after UI is fully ready
        self.after(200, self._restore_file_paths)
        # Reopen the last-used MIDI port, if any and if it's still present.
        # Deferred via after() rather than run synchronously here — see
        # _auto_reconnect_midi()'s docstring for why.
        self.after(250, self._auto_reconnect_midi)

    # ── style ────────────────────────────────────────────────

    def _style(self):
        s = ttk.Style(self)
        try: s.theme_use('clam')
        except: pass
        for name, bg_, fg_, font_ in [
            ('TFrame',     BG,    FG,     (FF,11)),
            ('P.TFrame',   PANEL, FG,     (FF,11)),
            ('TLabel',     BG,    FG,     (FF,11)),
            ('P.TLabel',   PANEL, FG,     (FF,11)),
            ('Dim.TLabel', PANEL, DIM,    (FF,10)),
            ('TNotebook',  BG,    FG,     (FF,11)),
        ]:
            s.configure(name, background=bg_, foreground=fg_, font=font_)
        s.configure('TNotebook.Tab', background=PANEL, foreground=FG,
                    padding=(16,9), font=(FF,11,'bold'))
        s.map('TNotebook.Tab',
              background=[('selected', ACCENT)],
              foreground=[('selected', ACCENT_DARK)])
        s.configure('Go.TButton',  font=(FF,13,'bold'), padding=10)
        s.configure('TButton',     font=(FF,11), padding=6)
        s.configure('TCombobox',   font=(FF,11))
        s.configure('TSpinbox',    font=(FF,11))
        s.configure('TEntry',      font=(FF,10))

        # clam's default Checkbutton/Radiobutton draw their whole label
        # area (not just the indicator box) in the theme's stock light
        # background unless told otherwise — same underlying issue as
        # the unstyled TButton fill, just never noticed until the rest of
        # the theme got dark enough for it to stand out as a visible
        # light-grey box behind every checkbox/radio in the app.
        for name in ('TCheckbutton', 'TRadiobutton'):
            s.configure(name, background=PANEL, foreground=FG, focuscolor=PANEL,
                        indicatorbackground=BG_WELL, indicatorforeground=ACCENT,
                        upperbordercolor=DIM, lowerbordercolor=DIM)
            s.map(name,
                  background=[('active', PANEL)],
                  foreground=[('disabled', DIM)],
                  indicatorbackground=[('selected', ACCENT), ('!selected', BG_WELL)])

        self._setup_rounded_buttons(s)

    # A ttk 9-slice image element's `border` sets a hard MINIMUM button
    # size along each axis (measured: height = 31 + 2*border, regardless
    # of font/padding) — the original border=10 was quietly inflating
    # every button from ~35px to ~51px tall, which broke the footer's
    # tight fixed-height layout (the transpose combobox below About/Stop
    # tone got squeezed and overlapped). border=3 keeps that inflation
    # down to ~2px (visually unnoticeable) while still showing a clearly
    # rounded corner.
    _ROUND_BTN_IMG_SIZE = 14   # source image size (px) before ttk stretches it
    _ROUND_BTN_RADIUS   = 3    # corner radius (px) — "slight," not pill-shaped
    _ROUND_BTN_BORDER   = 3    # 9-slice inset ttk keeps fixed while stretching the middle

    _round_btn_generation = 0   # bumped each call so re-running after a theme change gets fresh element names

    def _setup_rounded_buttons(self, style):
        """Give every ttk.Button a slight border radius. Tk's button
        rendering is fundamentally rectangular — there is no CSS-style
        border-radius property to set — so this works around it the
        standard Tk way: draw the rounded shape once per visual state as
        a small image, then hand it to ttk as a 9-slice ('border=')
        image element, which Tk stretches to fit any button's actual
        width/height while keeping the rounded corners intact. Redefining
        the base 'TButton'/'Go.TButton' layouts here (rather than adding
        a new style name) means every existing ttk.Button call site
        picks this up automatically, with nothing to change per-site.

        Each button asset is composited onto an OPAQUE PANEL-colored
        backdrop (_button_asset_image) rather than left transparent — an
        earlier transparent version showed a stray light/dark fringe at
        the corners and edges in testing, since ttk's image style element
        doesn't composite a transparent PNG's edges the way a Canvas
        does. Baking the real backdrop color in sidesteps that entirely:
        there's no transparent pixel left for anything to render wrong.

        Raw tk.Button widgets (a handful of places that need dynamic
        per-instance colors — REC/ARM, the IR A/B slot buttons, the
        pitch-pipe note buttons) are a different widget class entirely
        and don't go through ttk styles at all, so they're unaffected by
        this and stay square.

        Re-runnable: App._apply_theme() calls this again after the user
        changes the accent/background color, so element names carry a
        generation suffix — ttk errors on redefining an existing element
        name, and there's no public "delete element" call to undo the
        old one first.

        Silently skipped if Pillow isn't installed — buttons stay their
        normal square selves; nothing else about the app depends on it.
        """
        if not HAS_PIL:
            return
        size, radius, border = (self._ROUND_BTN_IMG_SIZE, self._ROUND_BTN_RADIUS,
                                 self._ROUND_BTN_BORDER)
        App._round_btn_generation += 1
        gen = App._round_btn_generation

        def asset(fill, outline):
            img = _button_asset_image((size, size), radius, fill, outline, PANEL, 1)
            photo = ImageTk.PhotoImage(img)
            _UI_IMAGE_REFS.append(photo)
            return photo

        # Plain buttons keep clam's existing neutral fill — only the
        # corners change, so this reads as a refinement, not a recolor.
        normal   = asset('#dcdad5', '#a9a7a1')
        hover    = asset('#e8e6e1', '#b7b5af')
        pressed  = asset('#c7c5c0', '#98968f')
        disabled = asset('#3a3a3a', '#2c2c2c')

        style.element_create(f'Rounded{gen}.button', 'image', normal,
                              ('disabled', disabled),
                              ('pressed', pressed),
                              ('active', hover),
                              border=border, sticky='nsew')
        style.layout('TButton', [
            (f'Rounded{gen}.button', {'sticky': 'nswe', 'children': [
                ('Button.focus', {'sticky': 'nswe', 'children': [
                    ('Button.padding', {'sticky': 'nswe', 'children': [
                        ('Button.label', {'sticky': 'nswe'})]})]})]})])
        style.map('TButton', foreground=[('disabled', DIM)])

        # Go.TButton (Start Amp / Start Tuner) — the one "primary action"
        # style already gets its own name in the code, so it's a natural
        # place to put the accent color front and center instead of
        # leaving it the same neutral grey as every other button. Hover/
        # pressed/outline shades are derived from the current ACCENT
        # rather than hand-picked, so any accent color the user chooses
        # gets sensible-looking button states for free.
        go_outline = _scale_color(ACCENT, 0.55)
        go_normal  = asset(ACCENT, go_outline)
        go_hover   = asset(_lighten_color(ACCENT, 25), go_outline)
        go_pressed = asset(_scale_color(ACCENT, 0.75), _scale_color(ACCENT, 0.4))
        style.element_create(f'RoundedGo{gen}.button', 'image', go_normal,
                              ('disabled', disabled),
                              ('pressed', go_pressed),
                              ('active', go_hover),
                              border=border, sticky='nsew')
        style.layout('Go.TButton', [
            (f'RoundedGo{gen}.button', {'sticky': 'nswe', 'children': [
                ('Button.focus', {'sticky': 'nswe', 'children': [
                    ('Button.padding', {'sticky': 'nswe', 'children': [
                        ('Button.label', {'sticky': 'nswe'})]})]})]})])
        style.map('Go.TButton',
                  foreground=[('disabled', DIM), ('!disabled', FG)])

    # ── top-level layout ─────────────────────────────────────

    def _build(self):
        # ── persistent tuner footer ──────────────────────────
        self._build_footer()

        # ── notebook ─────────────────────────────────────────
        nb = ttk.Notebook(self)
        self._notebook = nb   # kept so _full_rebuild_ui() can restore the selected tab
        nb.pack(fill='both', expand=True, padx=8, pady=(8,0))

        t_io    = ttk.Frame(nb)
        t_chain = ttk.Frame(nb)
        t_loop  = ttk.Frame(nb)
        t_metro = ttk.Frame(nb)

        t_pedals   = ttk.Frame(nb)
        t_settings = ttk.Frame(nb)
        nb.add(t_io,       text='  I/O & Levels  ')
        nb.add(t_chain,    text='  Signal Chain  ')
        nb.add(t_pedals,   text='  Pedals  ')
        nb.add(t_loop,     text='  Looper  ')
        nb.add(t_metro,    text='  Metronome  ')
        nb.add(t_settings, text='  Settings  ')

        self._pedals_tab_root = t_pedals  # kept to allow rebuilding after a profile load
        self._tab_io(t_io)
        self._tab_chain(t_chain)
        self._tab_pedals(t_pedals)
        self._tab_looper(t_loop)
        self._tab_metro(t_metro)
        self._tab_settings(t_settings)

        # Global keybind dispatch — single <KeyPress>/<KeyRelease> binds
        # rather than per-widget or per-key, since mappings are entirely
        # user-configurable on the Settings tab (Learn mode) and looked up
        # by keysym at dispatch time rather than baked into the bind
        # itself. See _on_key_press/_on_key_release.
        self.bind_all('<KeyPress>', self._on_key_press)
        self.bind_all('<KeyRelease>', self._on_key_release)

    def _apply_theme(self, accent_hex, bg_hex):
        """Change the accent/background color and rebuild the whole UI to
        match. Most of this app's widgets are plain tk.Label/tk.Frame
        with a literal bg=PANEL/fg=FG etc. handed to Tk once at creation
        time — unlike a ttk widget, a raw tk widget never re-reads a
        Python variable later, so simply reassigning the ACCENT/BG
        globals wouldn't change anything already on screen. Rebuilding
        every widget is what makes that assignment actually visible:
        each _tab_*()/_build_footer() call reads the current value of
        these globals fresh, the same way _rebuild_pedals_ui() already
        does for a single tab after a profile load, just applied to the
        whole window."""
        global ACCENT, BG
        ACCENT = accent_hex
        BG = bg_hex
        _recompute_derived_theme()
        self._save_session()
        self._full_rebuild_ui()

    def _choose_accent_color(self):
        _, hex_color = colorchooser.askcolor(color=ACCENT, title='Choose accent color', parent=self)
        if hex_color:
            self._apply_theme(hex_color, BG)

    def _choose_bg_color(self):
        _, hex_color = colorchooser.askcolor(color=BG, title='Choose background color', parent=self)
        if hex_color:
            self._apply_theme(ACCENT, hex_color)

    def _reset_theme(self):
        self._apply_theme(DEFAULT_ACCENT, DEFAULT_BG)

    def _full_rebuild_ui(self):
        selected_idx = None
        nb = getattr(self, '_notebook', None)
        if nb is not None:
            try:
                selected_idx = nb.index(nb.select())
            except Exception:
                pass
        for child in self.winfo_children():
            child.destroy()
        self._style()
        self._build()
        if selected_idx is not None:
            try:
                self._notebook.select(selected_idx)
            except Exception:
                pass

    # ── footer (always-visible tuner strip) ──────────────────

    def _build_footer(self):
        # Two-row footer: top row = tuner display + controls + pitch pipe + A4/transpose
        #                 bottom row = tuning preset selector + 6-string readout
        foot = tk.Frame(self, bg=BG_WELL)
        foot.pack(side='bottom', fill='x')

        # ── top row ──────────────────────────────────────────────
        top = tk.Frame(foot, bg=BG_WELL, height=68)
        top.pack(fill='x'); top.pack_propagate(False)

        # gauge canvas
        self.gauge_c = tk.Canvas(top, width=110, height=68,
                                  bg=BG_WELL, highlightthickness=0)
        self.gauge_c.pack(side='left', padx=(8,0))

        # note + freq
        info = tk.Frame(top, bg=BG_WELL)
        info.pack(side='left', padx=8)
        self.foot_note = tk.Label(info, text='—', bg=BG_WELL, fg=ACCENT,
                                   font=(FF, 26, 'bold'))
        self.foot_note.pack(anchor='w')
        self.foot_freq = tk.Label(info, text='', bg=BG_WELL, fg=DIM, font=(FF,9))
        self.foot_freq.pack(anchor='w')

        # cents
        self.foot_cents = tk.Label(top, text='', bg=BG_WELL, fg=DIM, font=(FF,11))
        self.foot_cents.pack(side='left', padx=4)

        # status + start button
        # fill='y' + an inner frame packed with expand=True (and no fill)
        # is the standard Tk way to truly vertically-center a stack of
        # widgets inside a fixed-height row — packing ctrl itself with no
        # fill only centers ctrl's own (possibly lopsided) bounding box,
        # which is what made this look slightly bottom-heavy before.
        ctrl = tk.Frame(top, bg=BG_WELL)
        ctrl.pack(side='left', padx=10, fill='y')
        ctrl_inner = tk.Frame(ctrl, bg=BG_WELL)
        ctrl_inner.pack(expand=True)
        self.foot_status = tk.Label(ctrl_inner, text='Tuner stopped', bg=BG_WELL,
                                     fg=DIM, font=(FF,10))
        self.foot_status.pack(anchor='w')
        btn_row = tk.Frame(ctrl_inner, bg=BG_WELL); btn_row.pack(anchor='w', pady=(3,0))
        self.tuner_btn = ttk.Button(btn_row, text='🎤 START', command=self._toggle_tuner)
        self.tuner_btn.pack(side='left', padx=(0,4))

        # persistent beat light — visible on every tab, so a muted
        # metronome can still be watched silently while recording loops
        beat_frame = tk.Frame(top, bg=BG_WELL); beat_frame.pack(side='left', padx=8)
        tk.Label(beat_frame, text='Beat', bg=BG_WELL, fg=DIM, font=(FF,8)).pack(anchor='w')
        self.beat_light = tk.Canvas(beat_frame, width=20, height=20,
                                     bg=BG_WELL, highlightthickness=0)
        self.beat_light.pack(pady=(2,0))
        self._beat_light_oval = self.beat_light.create_oval(2, 2, 18, 18,
                                                              fill='#2c333b', outline='')

        # pitch pipe
        pp = tk.Frame(top, bg=BG_WELL); pp.pack(side='left', padx=8)
        tk.Label(pp, text='Pitch pipe', bg=BG_WELL, fg=DIM, font=(FF,8)).pack(anchor='w')
        nr = tk.Frame(pp, bg=BG_WELL); nr.pack()
        self.pp_playing_idx = None
        self.pp_buttons = []
        for i, n in enumerate(NOTE_NAMES):
            b = tk.Button(nr, text=n, width=3, font=(FF,7,'bold'),
                          bg='#1e252d', fg=FG, activebackground=ACCENT, relief='flat',
                          command=lambda idx=i: self._toggle_pitch_pipe(idx))
            b.grid(row=0, column=i, padx=1)
            self.pp_buttons.append(b)

        # A4 + transpose + stop — far right
        right = tk.Frame(top, bg=BG_WELL); right.pack(side='right', padx=10)
        a4r = tk.Frame(right, bg=BG_WELL); a4r.pack(anchor='e', pady=(4,1))
        tk.Label(a4r, text='A4:', bg=BG_WELL, fg=DIM, font=(FF,8)).pack(side='left')
        ttk.Button(a4r, text='－', width=2, command=lambda: self._nudge_a4(-1)).pack(side='left', padx=1)
        self.a4_lbl = tk.Label(a4r, text='440', bg=BG_WELL, fg=FG, font=(FF,9), width=4)
        self.a4_lbl.pack(side='left')
        ttk.Button(a4r, text='+', width=2, command=lambda: self._nudge_a4(1)).pack(side='left', padx=1)
        ttk.Button(a4r, text='Stop tone', command=self._stop_pitch_pipe).pack(side='left', padx=4)
        ttk.Button(a4r, text='About', command=self._show_about).pack(side='left', padx=4)
        ttk.Combobox(right, textvariable=self.trans_var, state='readonly', width=22,
                     values=list(TRANSPOSITIONS.keys())).pack(anchor='e')

        # ── bottom row: tuning preset + string readout ───────────
        bot = tk.Frame(foot, bg='#0a0d10')
        bot.pack(fill='x', ipady=4)

        # Packed side='right' and FIRST, before any of this row's
        # left-packed content below — Tk's packer allocates cavity space
        # in pack-call order regardless of side, so claiming this button's
        # space first guarantees it always gets mapped even if the
        # variable-width cents label (str_cents_lbl, packed last, grows
        # once the tuner is actually running) would otherwise have eaten
        # the room. The top row (A4/transpose/Stop tone/About) was tried
        # first and had no slack left for this even in a much wider
        # window — this row had ~80px free instead. Raw tk.Button (not
        # ttk) for the same reason as REC/ARM and the IR A/B buttons: it
        # needs a dynamic per-instance color to read as "engaged".
        self.panic_var = tk.BooleanVar(value=False)
        self.panic_var.trace_add('write', lambda *_a: self._lpd8_refresh_leds())
        self.panic_btn = tk.Button(bot, text='PANIC', font=(FF,8,'bold'), width=7,
                                    bg=BG_WELL, fg=RED, activebackground=RED,
                                    activeforeground=BG_WELL, relief='flat',
                                    command=self._toggle_panic)
        self.panic_btn.pack(side='right', padx=(0,10))

        tk.Label(bot, text='Instrument:', bg='#0a0d10', fg=DIM, font=(FF,9)
                 ).pack(side='left', padx=(10,4))
        self.instrument_var = tk.StringVar(value='guitar')
        for val, txt in [('guitar', 'Guitar'), ('bass', 'Bass')]:
            tk.Radiobutton(bot, text=txt, variable=self.instrument_var, value=val,
                           bg='#0a0d10', fg=DIM, selectcolor='#2a3540',
                           activebackground='#0a0d10', font=(FF,9),
                           command=self._instrument_changed).pack(side='left')

        tk.Label(bot, text='Tuning:', bg='#0a0d10', fg=DIM, font=(FF,9)
                 ).pack(side='left', padx=(14,4))
        self.tuning_cb = ttk.Combobox(bot, textvariable=self.tuning_var,
                                       values=list(GUITAR_TUNINGS.keys()),
                                       state='readonly', width=22)
        self.tuning_cb.pack(side='left', padx=(0,14))
        self.tuning_cb.bind('<<ComboboxSelected>>', lambda e: self._reset_string_highlights())

        # 6 string boxes: low E to high e, left to right
        STRING_LABELS = ['6\nE', '5\nA', '4\nD', '3\nG', '2\nB', '1\ne']
        self.str_boxes   = []
        self.str_labels  = []
        str_frame = tk.Frame(bot, bg='#0a0d10'); str_frame.pack(side='left')
        for i in range(6):
            f = tk.Frame(str_frame, bg='#1a2128', width=48, height=34,
                         highlightthickness=1, highlightbackground='#2c333b')
            f.pack(side='left', padx=2)
            f.pack_propagate(False)
            # string number + note name from preset
            lbl = tk.Label(f, text=STRING_LABELS[i], bg='#1a2128', fg=DIM,
                           font=(FF, 7), justify='center')
            lbl.pack(expand=True)
            self.str_boxes.append(f)
            self.str_labels.append(lbl)

        # cents-to-target label (shown next to string boxes)
        self.str_cents_lbl = tk.Label(bot, text='', bg='#0a0d10', fg=DIM, font=(FF,9))
        self.str_cents_lbl.pack(side='left', padx=8)

        self._draw_gauge(0)
        self._reset_string_highlights()

    # ── helpers ──────────────────────────────────────────────

    def _scale(self, parent, lo, hi, cmd, start, length=280, orient='horizontal', **kw):
        s = ttk.Scale(parent, from_=lo, to=hi, orient=orient,
                      length=length, **kw)
        self._updating = True; s.set(start); self._updating = False
        def _cmd(v):
            if not self._updating: cmd(v)
        s.configure(command=_cmd)
        return s

    def _sec(self, parent, title, color=None, group=None, default_collapsed=False):
        """A titled section. The header is clickable to collapse/expand
        the section's body — state is kept in self._section_collapsed
        (keyed by title) and persisted through the session file, so
        sections stay collapsed the way the user left them next launch.

        color defaults to None (rather than =ACCENT) deliberately: a
        default argument value is evaluated once, when the method is
        defined, so =ACCENT would freeze every section header at
        whatever ACCENT happened to be at import time — invisible to
        App._apply_theme()'s later reassignment of that same global.
        Reading it here instead means every call sees the live color.

        group, if given, makes this an accordion member: expanding this
        section force-collapses every other section registered under the
        same group name (self._section_groups), so at most one member of
        the group is ever open — used for the Pedals tab's 8 effect
        sections (see PEDALS_ACCORDION_GROUP), left None everywhere else
        so every other tab keeps its existing independent collapse
        behavior. default_collapsed sets this section's initial state
        when the session file has no saved entry for `title` yet — the
        Pedals tab uses this to start on Compressor only.
        """
        if color is None:
            color = ACCENT
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill='x', padx=8, pady=6)
        inner = tk.Frame(outer, bg=PANEL)
        collapsed = self._section_collapsed.get(title, default_collapsed)
        hdr = tk.Label(outer, text=self._sec_hdr_text(title, collapsed), bg='#1a2128', fg=color,
                       font=(FF,11,'bold'), anchor='w', cursor='hand2')
        hdr.pack(fill='x')

        # Registered for every section (not just grouped ones) — generic
        # bookkeeping that _jump_to_pedal_section() and the accordion
        # logic below both rely on to reach a DIFFERENT section's own
        # toggle/collapse from outside its closure.
        self._section_outer_by_title[title] = outer
        if group is not None:
            self._section_groups.setdefault(group, []).append(title)

        def _toggle(event=None):
            now = not self._section_collapsed.get(title, default_collapsed)
            self._section_collapsed[title] = now
            if now: inner.pack_forget()
            else:   inner.pack(fill='x', pady=(0,2))
            hdr.configure(text=self._sec_hdr_text(title, now))
            if group is not None and not now:
                # Just expanded, and part of an accordion group — collapse
                # every other member so only this one stays open.
                for other_title in self._section_groups.get(group, []):
                    if other_title == title:
                        continue
                    other_outer = self._section_outer_by_title.get(other_title)
                    fc = getattr(other_outer, '_force_collapse', None) if other_outer else None
                    if fc is not None:
                        fc()

        hdr.bind('<Button-1>', _toggle)
        if not collapsed:
            inner.pack(fill='x', pady=(0,2))

        def _force_expand():
            """Expand this section if it's currently collapsed, otherwise
            do nothing — used by _jump_to_pedal_section() so clicking a
            collapsed pedal's status icon actually reveals its controls
            instead of scrolling to a closed header."""
            if self._section_collapsed.get(title, default_collapsed):
                _toggle()

        def _force_collapse():
            """Collapse this section if it's currently expanded, otherwise
            do nothing — the other half of the accordion group logic
            above (one section's expand forces every sibling's collapse)."""
            if not self._section_collapsed.get(title, default_collapsed):
                _toggle()

        outer._force_expand = _force_expand
        outer._force_collapse = _force_collapse
        return inner

    @staticmethod
    def _sec_hdr_text(title, collapsed):
        return f"  {'▶' if collapsed else '▼'} {title}"

    def _knob_row(self, parent, label, lo, hi, default, cb, fmt=None, length=260):
        row = tk.Frame(parent, bg=PANEL); row.pack(fill='x', padx=10, pady=4)
        tk.Label(row, text=label, bg=PANEL, fg=FG,
                 font=(FF,11), width=22, anchor='w').pack(side='left')
        val_var = tk.StringVar(value=(fmt(default) if fmt else f'{default:+.0f} dB'))
        def _cmd(v):
            fv = float(v)
            val_var.set(fmt(fv) if fmt else f'{fv:+.1f} dB')
            cb(fv)
        sl = self._scale(row, lo, hi, _cmd, default, length=length)
        sl.pack(side='left', padx=8)
        tk.Label(row, textvariable=val_var, bg=PANEL, fg=DIM,
                 font=(FF,10), width=10).pack(side='left')
        return sl, val_var

    def _scrollable(self, parent):
        """Return (canvas, inner_frame) with a vertical scrollbar."""
        c   = tk.Canvas(parent, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient='vertical', command=c.yview)
        c.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        c.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(c, bg=BG)
        wid   = c.create_window((0,0), window=inner, anchor='nw')
        def _resize(e):
            c.configure(scrollregion=c.bbox('all'))
            c.itemconfig(wid, width=e.width)
        inner.bind('<Configure>', _resize)
        c.bind('<Configure>', lambda e: c.itemconfig(wid, width=e.width))
        return inner

    # ── TAB 1: I/O & Levels ──────────────────────────────────

    def _tab_io(self, root):
        # Packed into `root` directly, before the scrollable accordion
        # area below claims the rest of the tab — Start Amp is the one
        # control on this tab you need no matter which section is open or
        # how far you've scrolled, so it lives outside the accordion
        # entirely instead of inside a collapsible "Transport" section.
        tr_row = tk.Frame(root, bg=BG); tr_row.pack(fill='x', padx=8, pady=(8,0))
        self.amp_btn = ttk.Button(tr_row, text='▶  START AMP', style='Go.TButton',
                                   command=self._toggle_amp)
        self.amp_btn.pack(side='left', ipadx=16)
        self.amp_stat = tk.Label(tr_row, text='Stopped', bg=BG, fg=DIM,
                                  font=(FF,11))
        self.amp_stat.pack(side='left', padx=16)

        inner = self._scrollable(root)

        # ── devices ──────────────────────────────────────────
        dev_sec = self._sec(inner, 'Audio Devices',
                             group=IO_ACCORDION_GROUP, default_collapsed=False)

        # Host API filter — Windows reports the same physical device once per
        # API it's reachable through, so leaving this unfiltered turns a
        # handful of real devices into dozens of near-duplicates. Defaults to
        # WASAPI, whose names match Windows' own Sound Settings.
        self._hostapis = host_api_names()
        ha_row = tk.Frame(dev_sec, bg=PANEL); ha_row.pack(fill='x', padx=10, pady=(5,2))
        tk.Label(ha_row, text='Host API:', bg=PANEL, fg=FG,
                 font=(FF,11), width=9).pack(side='left')
        ha_values = self._hostapis + ['All host APIs']
        self.hostapi_var = tk.StringVar(value=(self._hostapis[0] if self._hostapis else 'All host APIs'))
        self.hostapi_box = ttk.Combobox(ha_row, textvariable=self.hostapi_var,
                                         state='readonly', width=24, values=ha_values)
        self.hostapi_box.pack(side='left', padx=4)
        tk.Label(ha_row,
                 text="WASAPI names match Windows' Sound Settings; try WDM-KS for exclusive-mode low latency.",
                 bg=PANEL, fg=DIM, font=(FF,9)).pack(side='left', padx=6)
        self.hostapi_box.bind('<<ComboboxSelected>>', self._hostapi_changed)

        self.amp_inputs  = list_inputs(self._current_host_api())
        self.amp_in_var  = tk.StringVar(value=self.amp_inputs[0][1])

        r1 = tk.Frame(dev_sec, bg=PANEL); r1.pack(fill='x', padx=10, pady=5)
        tk.Label(r1, text='Input:', bg=PANEL, fg=FG,
                 font=(FF,11), width=9).pack(side='left')
        self.amp_in_box = ttk.Combobox(r1, textvariable=self.amp_in_var,
                                        state='readonly', width=48,
                                        values=[l for _,l in self.amp_inputs])
        self.amp_in_box.pack(side='left', padx=4)
        ttk.Button(r1, text='⟳', width=3,
                   command=self._refresh_devices).pack(side='left', padx=4)

        # Tuner device (shared label)
        self.t_inputs  = self.amp_inputs
        self.t_dev_var = tk.StringVar(value=self.amp_inputs[0][1])
        r1b = tk.Frame(dev_sec, bg=PANEL); r1b.pack(fill='x', padx=10, pady=2)
        tk.Label(r1b, text='Tuner input:', bg=PANEL, fg=DIM,
                 font=(FF,10), width=9).pack(side='left')
        tk.Label(r1b,
                 text='Shares amp input when amp is running — or select separately:',
                 bg=PANEL, fg=DIM, font=(FF,9)).pack(side='left', padx=4)
        self.t_dev_box = ttk.Combobox(r1b, textvariable=self.t_dev_var,
                                       state='readonly', width=30,
                                       values=[l for _,l in self.t_inputs])
        self.t_dev_box.pack(side='left', padx=4)
        self.t_dev_box.bind('<<ComboboxSelected>>', self._tuner_dev_changed)
        # Sync self.tuner.device to the displayed selection immediately —
        # otherwise it stays None (system default input, often the wrong
        # device) until the user manually touches this dropdown, which
        # made standalone tuning (amp not running) silently listen to the
        # wrong mic and never detect a plucked string.
        self.tuner.device = next((i for i,l in self.t_inputs
                                   if l == self.t_dev_var.get()), None)

        self.amp_out_devs = list_outputs(self._current_host_api())
        self.amp_out_var = tk.StringVar(value=self.amp_out_devs[0][1])
        r2 = tk.Frame(dev_sec, bg=PANEL); r2.pack(fill='x', padx=10, pady=5)
        tk.Label(r2, text='Output:', bg=PANEL, fg=FG,
                 font=(FF,11), width=9).pack(side='left')
        self.amp_out_box = ttk.Combobox(r2, textvariable=self.amp_out_var, state='readonly',
                                         width=48, values=[l for _,l in self.amp_out_devs])
        self.amp_out_box.pack(side='left', padx=4)

        bs_row = tk.Frame(dev_sec, bg=PANEL); bs_row.pack(fill='x', padx=10, pady=5)
        tk.Label(bs_row, text='Block size:', bg=PANEL, fg=FG,
                 font=(FF,11), width=9).pack(side='left')
        # 64/128 dropped: with a full NAM+IR chain loaded they don't keep up
        # in real-time even after the build-37 speedups (IR convolution cost
        # doesn't shrink with block size the way NAM's now does). 256 is the
        # lowest size confirmed to hold up in real-world testing.
        self.bs_var = tk.StringVar(value='256')
        ttk.Combobox(bs_row, textvariable=self.bs_var, width=7, state='readonly',
                     values=['256','512','1024','2048']).pack(side='left', padx=4)
        tk.Label(bs_row, text='samples  (lower = less latency, more CPU — 256 is the lowest usable with NAM/IR loaded)',
                 bg=PANEL, fg=DIM, font=(FF,10)).pack(side='left', padx=6)

        # ── levels ────────────────────────────────────────────
        lvl_sec = self._sec(inner, 'Levels',
                             group=IO_ACCORDION_GROUP, default_collapsed=True)
        self.in_gain_sl,  self.in_gain_val  = self._knob_row(
            lvl_sec, 'Input Gain (dB)',  -12, 24,  0, self._set_in_gain)
        self.out_gain_sl, self.out_gain_val = self._knob_row(
            lvl_sec, 'Output Gain (dB)', -24, 12, -6, self._set_out_gain)

        # ── noise gate ───────────────────────────────────────
        gate_sec = self._sec(inner, 'Noise Gate',
                              group=IO_ACCORDION_GROUP, default_collapsed=True)
        self.gate_sl, self.gate_val = self._knob_row(
            gate_sec, 'Threshold',  0, 60, 40, self._set_gate,
            fmt=lambda v: f'{-v:.0f} dBFS')

        # ── hum filter ────────────────────────────────────────
        hum_sec = self._sec(inner, 'Hum Filter',
                             group=IO_ACCORDION_GROUP, default_collapsed=True)
        hum_row = tk.Frame(hum_sec, bg=PANEL); hum_row.pack(fill='x', padx=10, pady=6)
        self.hum_var = tk.BooleanVar(value=False)
        self.hum_var.trace_add('write', lambda *_a: self._lpd8_refresh_leds())
        ttk.Checkbutton(hum_row, text='Enabled  (notches mains hum — 60/120/180 Hz or 50/100/150 Hz)',
                        variable=self.hum_var,
                        command=lambda: setattr(self.amp.hum, 'on', self.hum_var.get())
                        ).pack(side='left')
        self.hum_hz_var = tk.StringVar(value='60')
        for val, txt in [('60', '60 Hz'), ('50', '50 Hz')]:
            tk.Radiobutton(hum_row, text=txt, variable=self.hum_hz_var, value=val,
                           bg=PANEL, fg=FG, selectcolor='#2a3540', activebackground=PANEL,
                           command=lambda: self.amp.hum.set_base(float(self.hum_hz_var.get()))
                           ).pack(side='left', padx=(14,0))

    def _current_host_api(self):
        v = getattr(self, 'hostapi_var', None)
        v = v.get() if v else None
        return None if (v is None or v == 'All host APIs') else v

    def _hostapi_changed(self, _=None):
        self._refresh_devices()

    def _refresh_devices(self):
        """Re-scan devices for the currently selected host API (or all of
        them) — called by the ⟳ button and whenever the Host API dropdown
        changes. Resets a selection that no longer exists in the new list
        rather than leaving a stale label showing."""
        ha = self._current_host_api()

        self.amp_inputs = list_inputs(ha)
        self.amp_in_box.configure(values=[l for _,l in self.amp_inputs])
        if self.amp_in_var.get() not in [l for _,l in self.amp_inputs]:
            self.amp_in_var.set(self.amp_inputs[0][1])

        self.t_inputs = self.amp_inputs
        self.t_dev_box.configure(values=[l for _,l in self.t_inputs])
        if self.t_dev_var.get() not in [l for _,l in self.t_inputs]:
            self.t_dev_var.set(self.t_inputs[0][1])
        self._tuner_dev_changed()

        self.amp_out_devs = list_outputs(ha)
        self.amp_out_box.configure(values=[l for _,l in self.amp_out_devs])
        if self.amp_out_var.get() not in [l for _,l in self.amp_out_devs]:
            self.amp_out_var.set(self.amp_out_devs[0][1])

    def _restore_saved_devices(self, s):
        """Audio device selections now persist across runs — but a saved
        device can easily be gone next launch (interface unplugged, USB
        re-enumerated under a new name), so each one is checked against
        what's actually available right now rather than trusted blindly.
        Anything missing falls back to System default (already every
        dropdown's own index-0 entry) with one warning naming exactly
        what wasn't found, instead of silently picking the wrong device
        or leaving a selection that won't resolve when the amp opens its
        stream."""
        saved_hostapi = s.get('hostapi')
        if saved_hostapi and saved_hostapi in self.hostapi_box['values']:
            self.hostapi_var.set(saved_hostapi)
        self._refresh_devices()   # populate the lists for the restored host API

        missing = []
        def _restore_one(var, saved_label, options):
            if not saved_label or saved_label == 'System default':
                return
            if saved_label in [l for _, l in options]:
                var.set(saved_label)
            else:
                missing.append(saved_label)
                var.set('System default')

        _restore_one(self.amp_in_var,  s.get('amp_input'),   self.amp_inputs)
        _restore_one(self.amp_out_var, s.get('amp_output'),  self.amp_out_devs)
        _restore_one(self.t_dev_var,   s.get('tuner_input'), self.t_inputs)
        self._tuner_dev_changed()

        if missing:
            names = '\n'.join(f'  • {m}' for m in missing)
            self.after(300, lambda: messagebox.showwarning(
                'Audio device not found',
                "Couldn't find the previously selected device(s):\n"
                f"{names}\n\n"
                "Falling back to System default. If it's just temporarily "
                "unplugged, reconnect it and pick it again on the I/O & "
                "Levels tab."))

    def _tuner_dev_changed(self, _=None):
        lbl = self.t_dev_var.get()
        self.tuner.device = next((i for i,l in self.t_inputs if l==lbl), None)
        if self.tuner.running and not self.tuner.shared:
            self.tuner.stop()
            try: self.tuner.start()
            except Exception as e: messagebox.showerror('Mic error', str(e))

    def _set_in_gain(self, v):  self.amp.in_gain  = 10**(float(v)/20.0)
    def _set_out_gain(self, v): self.amp.out_gain = 10**(float(v)/20.0)
    def _set_gate(self, v):     self.amp.gate_thr = 10**(-float(v)/20.0)

    def _set_panic(self, v):
        """The one place that actually flips amp.panic — the footer
        button, the MIDI/keybind target, and anything else all funnel
        through here so the button's visual state can never end up out
        of sync with the real flag."""
        self.amp.panic = v
        self.panic_var.set(v)
        if hasattr(self, 'panic_btn'):
            self.panic_btn.configure(bg=(RED if v else BG_WELL),
                                      fg=(BG_WELL if v else RED))

    def _toggle_panic(self):
        self._set_panic(not self.panic_var.get())

    def _toggle_amp(self):
        if self.amp.running:
            self.amp.stop()
            self.amp_btn.configure(text='▶  START AMP')
        else:
            if sd is None:
                messagebox.showerror('No audio', 'PortAudio not available.')
                return
            in_lbl  = self.amp_in_var.get()
            out_lbl = self.amp_out_var.get()
            in_dev  = next((i for i,l in self.amp_inputs      if l==in_lbl),  None)
            out_dev = next((i for i,l in self.amp_out_devs    if l==out_lbl), None)
            # WASAPI shared mode can auto-convert sample rate/channels, so a
            # device whose system-configured rate isn't 44100 (e.g. an audio
            # interface set to 176400 Hz in Windows) still opens — fixes
            # PaErrorCode -9997 "Invalid sample rate" with no change to the
            # DSP pipeline, which stays at SAMPLE_RATE throughout. Resolved
            # per actual device rather than the Host API dropdown, because
            # "System default" (device=None) is PortAudio's own default and
            # may not be a WASAPI device at all regardless of that filter.
            in_extra  = sd.WasapiSettings(auto_convert=True) if self._device_host_api(in_dev,  True)  == 'Windows WASAPI' else None
            out_extra = sd.WasapiSettings(auto_convert=True) if self._device_host_api(out_dev, False) == 'Windows WASAPI' else None
            extra = (in_extra, out_extra) if (in_extra or out_extra) else None
            try:
                self.amp.start(in_dev, out_dev, blocksize=int(self.bs_var.get()),
                                extra_settings=extra)
            except Exception as e:
                messagebox.showerror('Amp error', self._amp_error_text(e)); return
            self.amp_btn.configure(text='■  STOP AMP')

    def _device_host_api(self, dev_idx, is_input):
        """Host API name PortAudio will actually use for this device selector.
        None ("System default") resolves through sd.default.device, since
        that's PortAudio's own default pick and is not guaranteed to match
        whatever this tab's Host API filter currently shows."""
        if sd is None: return None
        try:
            if dev_idx is None:
                pair = sd.default.device
                dev_idx = pair[0] if is_input else pair[1]
            info = sd.query_devices(dev_idx)
            return sd.query_hostapis()[info['hostapi']]['name']
        except Exception:
            return None

    def _amp_error_text(self, e):
        msg = str(e)
        if 'Invalid sample rate' in msg or '-9997' in msg:
            msg += ('\n\nThis device doesn\'t support 44100 Hz directly. Try setting '
                    'Host API to "Windows WASAPI" on this tab (it can convert the '
                    'rate automatically), or pick a different input/output device.')
        return msg

    def _poll_amp(self):
        if self.amp.running:
            lat = ''
            if self.amp.stream:
                try:
                    li, lo = self.amp.stream.latency
                    lat = f'  {li*1000:.0f}ms in / {lo*1000:.0f}ms out'
                except: pass
            xr  = self.amp.xruns
            col = '#ffd166' if xr else ACCENT
            self.amp_stat.configure(
                text=f'Running{lat}' + (f'  ⚠ {xr} xruns' if xr else ''),
                fg=col)
        else:
            self.amp_stat.configure(text='Stopped', fg=DIM)
        self.after(500, self._poll_amp)

    # ── TAB 2: Signal Chain ───────────────────────────────────

    def _tab_chain(self, root):
        inner = self._scrollable(root)

        # ── Profiles ─────────────────────────────────────────
        prof_sec = self._sec(inner, f'Tone Profiles  (.{PROFILE_EXT.lstrip(".")} — NAM/IR + pedals + EQ + tuning)')
        prof_row = tk.Frame(prof_sec, bg=PANEL); prof_row.pack(fill='x', padx=10, pady=6)
        ttk.Button(prof_row, text='Save Profile…', command=self._save_profile).pack(side='left', padx=(0,4))
        ttk.Button(prof_row, text='Load Profile…', command=self._load_profile).pack(side='left', padx=4)
        self.profile_lbl = tk.Label(prof_row, text='No profile loaded', bg=PANEL, fg=DIM,
                                     font=(FF,10))
        self.profile_lbl.pack(side='left', padx=10)

        # ── NAM ──────────────────────────────────────────────
        nam_sec = self._sec(inner, 'NAM Amp Model  (.nam — WaveNet)')
        nam_row = tk.Frame(nam_sec, bg=PANEL); nam_row.pack(fill='x', padx=10, pady=6)
        self.nam_lbl = tk.Label(nam_row, text='No file loaded', bg=PANEL, fg=DIM,
                                 font=(FF,11), width=36, anchor='w')
        self.nam_lbl.pack(side='left')
        ttk.Button(nam_row, text='Load .nam…', command=self._load_nam).pack(side='left', padx=4)
        ttk.Button(nam_row, text='Clear',      command=self._clear_nam).pack(side='left', padx=4)
        self.nam_on_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(nam_row, text='Enabled', variable=self.nam_on_var,
                        command=lambda: setattr(self.amp, 'nam_on', self.nam_on_var.get())
                        ).pack(side='left', padx=10)
        self.nam_info = tk.Label(nam_sec, text='', bg=PANEL, fg=DIM, font=(FF,9))
        self.nam_info.pack(anchor='w', padx=10, pady=(0,4))

        # ── IR slots ─────────────────────────────────────────
        ir_sec = self._sec(inner, 'Cabinet IRs  (up to 5 slots — A/B between any two)')
        self.ir_active_var = tk.IntVar(value=-1)
        self.ir_ab_a = tk.IntVar(value=-1)
        self.ir_ab_b = tk.IntVar(value=-1)

        self.ir_slot_rows = []
        for i in range(MAX_IR_SLOTS):
            row = tk.Frame(ir_sec, bg=PANEL); row.pack(fill='x', padx=8, pady=3)

            # select radio
            rb = tk.Radiobutton(row, variable=self.ir_active_var, value=i,
                                bg=PANEL, fg=FG, selectcolor='#2a3540',
                                activebackground=PANEL,
                                command=lambda idx=i: self._select_ir(idx))
            rb.pack(side='left')

            # user label
            ent = ttk.Entry(row, textvariable=self.ir_label_vars[i], width=14,
                            font=(FF,10))
            ent.pack(side='left', padx=(2,6))

            # file name
            lbl = tk.Label(row, text='(empty)', bg=PANEL, fg=DIM,
                           font=(FF,10), width=28, anchor='w')
            lbl.pack(side='left')

            # A / B buttons
            tk.Label(row, text='A/B:', bg=PANEL, fg=DIM, font=(FF,9)).pack(side='left', padx=(8,2))
            ab_a = tk.Button(row, text='A', width=2, font=(FF,9,'bold'),
                             bg='#1e252d', fg=FG, relief='flat',
                             command=lambda idx=i: self._set_ab('a', idx))
            ab_a.pack(side='left', padx=1)
            ab_b = tk.Button(row, text='B', width=2, font=(FF,9,'bold'),
                             bg='#1e252d', fg=FG, relief='flat',
                             command=lambda idx=i: self._set_ab('b', idx))
            ab_b.pack(side='left', padx=1)

            ttk.Button(row, text='Load…', command=lambda idx=i: self._load_ir(idx)).pack(side='left', padx=6)
            ttk.Button(row, text='✕',     command=lambda idx=i: self._clear_ir(idx)).pack(side='left', padx=2)

            self.ir_slot_rows.append({'lbl': lbl, 'ab_a': ab_a, 'ab_b': ab_b})

        # bypass + A/B toggle row
        bypass_row = tk.Frame(ir_sec, bg=PANEL); bypass_row.pack(fill='x', padx=8, pady=6)
        tk.Radiobutton(bypass_row, text='Bypass IR', variable=self.ir_active_var,
                       value=-1, bg=PANEL, fg=FG, selectcolor='#2a3540',
                       activebackground=PANEL,
                       command=lambda: setattr(self.amp, 'ir_active', -1)
                       ).pack(side='left', padx=(0,16))
        self.ab_btn = ttk.Button(bypass_row, text='⇄  Toggle A/B',
                                  command=self._toggle_ab)
        self.ab_btn.pack(side='left', padx=4)
        self.ab_lbl = tk.Label(bypass_row, text='A: —   B: —', bg=PANEL, fg=DIM,
                                font=(FF,10))
        self.ab_lbl.pack(side='left', padx=10)

        # Blend — runs A's and B's convolutions simultaneously and mixes
        # them at an adjustable ratio, instead of only ever hearing
        # whichever one ir_active/Toggle A/B currently points at. Reuses
        # the same A/B slot assignment above rather than a separate pair
        # of pickers, so setting A/B once covers both toggling and
        # blending. Independent of ir_active: while enabled it overrides
        # whatever radio button is selected, same as flipping a real
        # blend pedal's footswitch.
        blend_row = tk.Frame(ir_sec, bg=PANEL); blend_row.pack(fill='x', padx=8, pady=(0,8))
        self.ir_blend_on_var = tk.BooleanVar(value=False)
        self.ir_blend_on_var.trace_add('write', lambda *_a: self._lpd8_refresh_leds())
        ttk.Checkbutton(blend_row, text='Blend A/B', variable=self.ir_blend_on_var,
                        command=self._toggle_ir_blend).pack(side='left')
        self.ir_blend_sl = self._scale(blend_row, 0, 100, self._set_ir_blend, 50.0, length=200)
        self.ir_blend_sl.pack(side='left', padx=8)
        self.ir_blend_lbl = tk.StringVar(value='50% A / 50% B')
        tk.Label(blend_row, textvariable=self.ir_blend_lbl, bg=PANEL, fg=DIM,
                 font=(FF,10), width=16).pack(side='left')

        # ── 5-band EQ ────────────────────────────────────────
        # Horizontal row of vertical faders, like a hardware graphic EQ
        # pedal, rather than the old stacked horizontal-slider rows —
        # easier to eyeball the overall curve at a glance.
        eq_sec = self._sec(inner, '5-Band EQ')
        eq_hdr = tk.Frame(eq_sec, bg=PANEL); eq_hdr.pack(fill='x', padx=10, pady=(4,0))
        self.eq_on_var = tk.BooleanVar(value=True)
        self.eq_on_var.trace_add('write', lambda *_a: self._lpd8_refresh_leds())
        ttk.Checkbutton(eq_hdr, text='EQ Enabled', variable=self.eq_on_var,
                        command=lambda: setattr(self.amp, 'eq_on', self.eq_on_var.get())
                        ).pack(side='left')
        ttk.Button(eq_hdr, text='Reset all', command=self._eq_reset).pack(side='left', padx=12)

        fader_row = tk.Frame(eq_sec, bg=PANEL); fader_row.pack(padx=10, pady=(10,12))

        def _fader(parent, label, lo, hi, default, cb, fg=FG):
            col = tk.Frame(parent, bg=PANEL); col.pack(side='left', padx=10)
            tk.Label(col, text=label, bg=PANEL, fg=fg, font=(FF,9),
                     width=8, anchor='center').pack()
            val_var = tk.StringVar(value=f'{default:+.0f} dB')
            def _cmd(v):
                fv = float(v)
                val_var.set(f'{fv:+.0f} dB')
                cb(fv)
            # from_=hi, to=lo so the slider reads top=boost/bottom=cut,
            # matching a real fader instead of ttk's default top=from_.
            sl = self._scale(col, hi, lo, _cmd, default, length=140,
                              orient='vertical')
            sl.pack(pady=6)
            tk.Label(col, textvariable=val_var, bg=PANEL, fg=DIM,
                     font=(FF,9), width=8, anchor='center').pack()
            return sl, val_var

        self.eq_sliders = []
        self.eq_val_vars = []
        eq_bands = [('100 Hz', 0), ('400 Hz', 1), ('1 kHz', 2), ('2.5 kHz', 3), ('6 kHz', 4)]
        for freq_str, idx in eq_bands:
            sl, val_var = _fader(
                fader_row, freq_str, -15, 15, 0,
                lambda fv, i=idx: self._eq_band_changed(i, fv))
            self.eq_sliders.append(sl)
            self.eq_val_vars.append(val_var)

        tk.Frame(fader_row, bg='#2c333b', width=1).pack(side='left', fill='y', padx=8)

        self.eq_level_sl, self.eq_level_val = _fader(
            fader_row, 'Level', -15, 15, 0, self._set_eq_level, fg=ACCENT)

    # ── tone profiles ──────────────────────────────────────────

    def _gather_profile_dict(self, name):
        """Collect everything that makes up 'a particular sound' into a
        plain, JSON-able dict: NAM/IR paths (not the audio data itself —
        loaded fresh from disk on the other end, same as session restore),
        the whole pedal chain (via PEDAL_PROFILE_FIELDS, so adding a new
        pedal field later only means adding it to that list), EQ, noise
        gate, input/output gain, hum filter, and the guitar tuning preset."""
        amp = self.amp
        return {
            'solotone_profile': PROFILE_FORMAT_VERSION,
            'name':          name,
            'nam_path':      self._current_nam_path,
            'ir_paths':      list(self._current_ir_paths),
            'ir_active':     amp.ir_active,
            'ir_labels':     [v.get() for v in self.ir_label_vars],
            'eq_on':         amp.eq_on,
            'eq_band_gains': [b['gain'] for b in amp.eq.bands],
            'eq_level':      amp.eq.level_db,
            'gate_threshold': amp.gate_thr,
            'in_gain':       amp.in_gain,
            'out_gain':      amp.out_gain,
            'hum_on':        amp.hum.on,
            'hum_base_hz':   amp.hum.base,
            'pedals':        {f: getattr(amp.pedals, f) for f in PEDAL_PROFILE_FIELDS},
            'tuning':        self.tuning_var.get(),
            # Only the pedal-related slice of the live MIDI map — which
            # knob/pad drives which pedal is part of "this particular
            # sound" the same way the pedal chain itself is, so it
            # travels with the profile. Non-pedal targets (transport,
            # EQ/amp) are physical-controller wiring, not part of the
            # sound, and stay governed by the global MIDI mapping file
            # instead — see _apply_profile.
            'midi_mappings': {f'{k[0]}:{k[1]}': v for k, v in self.midi_mappings.items()
                               if MIDI_TARGETS_BY_ID.get(v, {}).get('category') == 'Pedals'},
        }

    def _set_knob(self, slider, val_var, raw_value, text):
        """Move a _knob_row/_scale-backed slider and its label to match a
        restored value, without re-triggering its own change callback
        (which would just set the exact same value again, but the
        self._updating guard keeps that from ever depending on it)."""
        self._updating = True
        slider.set(raw_value)
        self._updating = False
        val_var.set(text)

    def _apply_profile(self, data, path):
        """Apply a loaded profile dict to the app. A profile is a whole
        sound, not a patch, so NAM/every IR slot is cleared first rather
        than merged with whatever was loaded before it."""
        amp = self.amp
        missing = []

        self._clear_nam()
        for idx in range(MAX_IR_SLOTS):
            self._clear_ir(idx)

        nam_path = data.get('nam_path')
        if nam_path:
            if pathlib.Path(nam_path).exists():
                self._load_nam_from_path(nam_path, silent=True)
            else:
                missing.append(f'NAM: {nam_path}')

        ir_paths = data.get('ir_paths', [None] * MAX_IR_SLOTS)
        for idx, ir_path in enumerate(ir_paths[:MAX_IR_SLOTS]):
            if ir_path:
                if pathlib.Path(ir_path).exists():
                    self._load_ir_from_path(idx, ir_path, silent=True)
                else:
                    missing.append(f'IR slot {idx+1}: {ir_path}')

        if 'ir_labels' in data:
            for i, lbl in enumerate(data['ir_labels'][:MAX_IR_SLOTS]):
                self.ir_label_vars[i].set(lbl)

        ir_active = data.get('ir_active', -1)
        if isinstance(ir_active, int) and -1 <= ir_active < MAX_IR_SLOTS:
            amp.ir_active = ir_active
            self.ir_active_var.set(ir_active)

        if 'eq_on' in data:
            amp.eq_on = bool(data['eq_on'])
            self.eq_on_var.set(amp.eq_on)
        if 'eq_band_gains' in data:
            for i, g in enumerate(data['eq_band_gains'][:len(self.eq_sliders)]):
                with amp.lock: amp.eq.set_gain(i, float(g))
                self._set_knob(self.eq_sliders[i], self.eq_val_vars[i],
                                float(g), f'{float(g):+.0f} dB')
        if 'eq_level' in data:
            g = float(data['eq_level'])
            with amp.lock: amp.eq.set_level(g)
            self._set_knob(self.eq_level_sl, self.eq_level_val, g, f'{g:+.0f} dB')

        if 'gate_threshold' in data:
            thr = max(1e-9, float(data['gate_threshold']))
            amp.gate_thr = thr
            v = -20.0 * math.log10(thr)
            self._set_knob(self.gate_sl, self.gate_val, v, f'{-v:.0f} dBFS')

        if 'in_gain' in data:
            amp.in_gain = float(data['in_gain'])
            v = 20.0 * math.log10(max(1e-9, amp.in_gain))
            self._set_knob(self.in_gain_sl, self.in_gain_val, v, f'{v:+.1f} dB')
        if 'out_gain' in data:
            amp.out_gain = float(data['out_gain'])
            v = 20.0 * math.log10(max(1e-9, amp.out_gain))
            self._set_knob(self.out_gain_sl, self.out_gain_val, v, f'{v:+.1f} dB')

        if 'hum_on' in data:
            amp.hum.on = bool(data['hum_on'])
            self.hum_var.set(amp.hum.on)
        if 'hum_base_hz' in data:
            amp.hum.set_base(float(data['hum_base_hz']))
            self.hum_hz_var.set(str(int(data['hum_base_hz'])))

        pedal_data = data.get('pedals', {})
        if 'order' in pedal_data:
            order = pedal_data['order']
            if isinstance(order, list) and sorted(order) == sorted(amp.pedals.order):
                amp.pedals.order = order
        for field in PEDAL_PROFILE_FIELDS:
            if field == 'order' or field not in pedal_data:
                continue
            setattr(amp.pedals, field, pedal_data[field])
        self._rebuild_pedals_ui()  # also re-applies amp.pedals.order via _relayout_pedals()

        if 'midi_mappings' in data:
            # Replace only the pedal-related slice of the live MIDI map —
            # everything else (transport, EQ/amp) is controller wiring
            # that isn't part of "this particular sound" and is left as
            # the player currently has it. A profile saved before this
            # feature existed has no 'midi_mappings' key at all, so it
            # falls through here and leaves the whole map untouched
            # rather than silently blanking out pedal mappings it never
            # captured.
            self.midi_mappings = {k: v for k, v in self.midi_mappings.items()
                                   if MIDI_TARGETS_BY_ID.get(v, {}).get('category') != 'Pedals'}
            for key_str, tid in data['midi_mappings'].items():
                target = MIDI_TARGETS_BY_ID.get(tid)
                if target is None or target['category'] != 'Pedals':
                    continue
                kind, _, num_str = key_str.partition(':')
                try:
                    self.midi_mappings[(kind, int(num_str))] = tid
                except ValueError:
                    pass
            self._refresh_midi_ui()
            self._save_midi_mappings()

        if 'tuning' in data:
            self._restore_tuning(data['tuning'])

        name = data.get('name') or pathlib.Path(path).stem
        self._current_profile_path = path
        self.profile_lbl.configure(text=f'Profile: {name}', fg=FG)

        if missing:
            missing_str = '\n'.join(f'  • {m}' for m in missing)
            messagebox.showerror(
                'Files missing from profile',
                f'"{name}" refers to files that could not be found:\n\n'
                f'{missing_str}\n\n'
                f'Everything else in the profile was loaded; load the missing '
                f'file(s) manually from the Signal Chain tab if you have them '
                f'elsewhere.'
            )

    def _save_profile(self):
        default_name = pathlib.Path(self._current_profile_path).stem if self._current_profile_path else 'My Tone'
        path = filedialog.asksaveasfilename(
            title='Save Tone Profile',
            initialdir=(self._last_profile_dir or str(PROFILE_DIR)),
            initialfile=default_name + PROFILE_EXT,
            defaultextension=PROFILE_EXT,
            filetypes=[('SoloTone Profile', f'*{PROFILE_EXT}'), ('All files', '*.*')])
        if not path: return
        self._last_profile_dir = str(pathlib.Path(path).parent)
        name = pathlib.Path(path).stem
        data = self._gather_profile_dict(name)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror('Save failed', str(e))
            return
        self._current_profile_path = path
        self.profile_lbl.configure(text=f'Profile: {name}', fg=FG)

    def _load_profile(self):
        path = filedialog.askopenfilename(
            title='Load Tone Profile',
            initialdir=(self._last_profile_dir or str(PROFILE_DIR)),
            filetypes=[('SoloTone Profile', f'*{PROFILE_EXT}'), ('All files', '*.*')])
        if not path: return
        self._last_profile_dir = str(pathlib.Path(path).parent)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror('Load failed', f'Could not read profile:\n{e}')
            return
        if not isinstance(data, dict) or 'solotone_profile' not in data:
            messagebox.showerror('Load failed', 'Not a valid SoloTone profile file.')
            return
        self._apply_profile(data, path)

    # ── per-pedal mini-presets ───────────────────────────────
    # Same JSON-file idea as Tone Profiles, scoped to one pedal's own
    # fields (PEDAL_PRESET_FIELDS) instead of the whole chain, so a
    # favorite pedal setting can be reused across different profiles
    # rather than being locked inside whichever profile it was saved in.

    def _build_pedal_preset_row(self, parent, code):
        row = tk.Frame(parent, bg=PANEL); row.pack(fill='x', padx=10, pady=(6,2))
        ttk.Button(row, text='Save Preset…',
                   command=lambda c=code: self._save_pedal_preset(c)).pack(side='left')
        ttk.Button(row, text='Load Preset…',
                   command=lambda c=code: self._load_pedal_preset(c)).pack(side='left', padx=4)
        name = self._pedal_preset_name.get(code)
        lbl = tk.Label(row, text=(f'Preset: {name}' if name else 'No preset loaded'),
                       bg=PANEL, fg=(FG if name else DIM), font=(FF,9))
        lbl.pack(side='left', padx=10)
        self._pedal_preset_labels = getattr(self, '_pedal_preset_labels', {})
        self._pedal_preset_labels[code] = lbl

    def _save_pedal_preset(self, code):
        default_dir = PEDAL_PRESET_DIR / code
        default_dir.mkdir(exist_ok=True)
        title = PEDAL_SECTION_TITLES[code]
        path = filedialog.asksaveasfilename(
            title=f'Save {title} Preset',
            initialdir=str(default_dir),
            defaultextension=PEDAL_PRESET_EXT,
            filetypes=[('SoloTone Pedal Preset', f'*{PEDAL_PRESET_EXT}'), ('All files', '*.*')])
        if not path: return
        name = pathlib.Path(path).stem
        data = {
            'solotone_pedal_preset': PEDAL_PRESET_FORMAT_VERSION,
            'pedal': code,
            'name': name,
            'fields': {f: getattr(self.amp.pedals, f) for f in PEDAL_PRESET_FIELDS[code]},
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror('Save failed', str(e))
            return
        self._pedal_preset_name[code] = name
        lbl = getattr(self, '_pedal_preset_labels', {}).get(code)
        if lbl is not None:
            lbl.configure(text=f'Preset: {name}', fg=FG)

    def _load_pedal_preset(self, code):
        default_dir = PEDAL_PRESET_DIR / code
        title = PEDAL_SECTION_TITLES[code]
        path = filedialog.askopenfilename(
            title=f'Load {title} Preset',
            initialdir=(str(default_dir) if default_dir.is_dir() else str(PEDAL_PRESET_DIR)),
            filetypes=[('SoloTone Pedal Preset', f'*{PEDAL_PRESET_EXT}'), ('All files', '*.*')])
        if not path: return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror('Load failed', f'Could not read preset:\n{e}')
            return
        if not isinstance(data, dict) or 'solotone_pedal_preset' not in data:
            messagebox.showerror('Load failed', 'Not a valid SoloTone pedal preset file.')
            return
        preset_pedal = data.get('pedal')
        if preset_pedal != code:
            other_title = PEDAL_SECTION_TITLES.get(preset_pedal, preset_pedal)
            messagebox.showerror('Load failed',
                f'This preset is for {other_title}, not {title}.')
            return
        fields = data.get('fields', {})
        for f in PEDAL_PRESET_FIELDS[code]:
            if f in fields:
                setattr(self.amp.pedals, f, fields[f])
        self._pedal_preset_name[code] = data.get('name') or pathlib.Path(path).stem
        self._rebuild_pedals_ui()  # simplest way to reflect every changed slider/radio/checkbox at once

    def _load_nam_from_path(self, path, silent=False):
        """Load a NAM file from an explicit path. Returns True on success.
        If silent=True, skips the error dialog (used during session restore)."""
        try:
            nam, name = load_nam(path)
        except Exception as e:
            if not silent:
                messagebox.showerror('NAM load error', str(e))
            return False
        with self.amp.lock:
            self.amp.nam    = nam
            self.amp.nam_on = True
        self.nam_on_var.set(True)
        self.nam_lbl.configure(text=name, fg=FG)
        self.nam_info.configure(
            text=f'Receptive field: {nam.rf} samples  ({nam.rf/SAMPLE_RATE*1000:.1f} ms)')
        self._current_nam_path = path
        return True

    def _load_nam(self):
        path = filedialog.askopenfilename(title='Load NAM file',
            initialdir=(self._last_nam_dir or str(NAM_DIR)),
            filetypes=[('NAM files','*.nam'),('All files','*.*')])
        if not path: return
        self._last_nam_dir = str(pathlib.Path(path).parent)
        self._load_nam_from_path(path)

    def _clear_nam(self):
        with self.amp.lock: self.amp.nam = None
        self.nam_lbl.configure(text='No file loaded', fg=DIM)
        self.nam_info.configure(text='')
        self._current_nam_path = None

    def _load_ir_from_path(self, idx, path, silent=False):
        """Load an IR WAV from an explicit path into slot idx. Returns True on success.
        If silent=True, skips the error dialog (used during session restore)."""
        try:
            arr, name = load_ir(path)
        except Exception as e:
            if not silent:
                messagebox.showerror('IR load error', str(e))
            return False
        with self.amp.lock:
            self.amp.ir_slots[idx].load(arr)
            self.amp.ir_loaded[idx] = True
        self.ir_slot_rows[idx]['lbl'].configure(text=name, fg=FG)
        self._current_ir_paths[idx] = path
        if self.amp.ir_active == -1:
            self._select_ir(idx)
        return True

    def _load_ir(self, idx):
        path = filedialog.askopenfilename(title=f'Load IR',
            initialdir=(self._last_ir_dir or str(IR_DIR)),
            filetypes=[('WAV files','*.wav'),('All files','*.*')])
        if not path: return
        self._last_ir_dir = str(pathlib.Path(path).parent)
        self._load_ir_from_path(idx, path)

    def _clear_ir(self, idx):
        with self.amp.lock:
            self.amp.ir_slots[idx] = IRConv()
            self.amp.ir_loaded[idx] = False
            if self.amp.ir_active == idx: self.amp.ir_active = -1
        self.ir_slot_rows[idx]['lbl'].configure(text='(empty)', fg=DIM)
        self._current_ir_paths[idx] = None
        self.ir_active_var.set(-1)
        self._refresh_ab_label()
        # The audio thread already no-ops blending against an unloaded
        # slot (falls back to bypass), but leaving the checkbox showing
        # "on" while it's silently doing nothing would be confusing.
        if idx in (self.ir_ab_a.get(), self.ir_ab_b.get()) and self.ir_blend_on_var.get():
            self.ir_blend_on_var.set(False)
            self.amp.ir_blend_on = False

    def _select_ir(self, idx):
        self.amp.ir_active = idx
        self.ir_active_var.set(idx)

    def _set_ab(self, which, idx):
        if which == 'a':
            self.ir_ab_a.set(idx)
            self.amp.ir_blend_a = idx
            self.ir_slot_rows[idx]['ab_a'].configure(bg=ACCENT, fg=ACCENT_DARK)
        else:
            self.ir_ab_b.set(idx)
            self.amp.ir_blend_b = idx
            self.ir_slot_rows[idx]['ab_b'].configure(bg='#2a5c8a', fg=FG)
        self._refresh_ab_label()

    def _toggle_ab(self):
        a, b = self.ir_ab_a.get(), self.ir_ab_b.get()
        if a < 0 and b < 0: return
        cur = self.ir_active_var.get()
        target = b if cur == a else a
        if target >= 0:
            self._select_ir(target)

    def _refresh_ab_label(self):
        a, b = self.ir_ab_a.get(), self.ir_ab_b.get()
        an = self.ir_label_vars[a].get() if a >= 0 else '—'
        bn = self.ir_label_vars[b].get() if b >= 0 else '—'
        self.ab_lbl.configure(text=f'A: {an}   B: {bn}')

    def _toggle_ir_blend(self):
        on = self.ir_blend_on_var.get()
        a, b = self.ir_ab_a.get(), self.ir_ab_b.get()
        if on and (a < 0 or b < 0 or not self.amp.ir_loaded[a] or not self.amp.ir_loaded[b]):
            messagebox.showwarning('Blend needs A and B',
                'Set both an A and a B slot (the A/B buttons on two loaded '
                'IR rows) before enabling Blend.')
            self.ir_blend_on_var.set(False)
            return
        self.amp.ir_blend_a = a
        self.amp.ir_blend_b = b
        self.amp.ir_blend_on = on

    def _set_ir_blend(self, v):
        frac = float(v) / 100.0
        self.amp.ir_blend = frac
        self.ir_blend_lbl.set(f'{100-int(round(v))}% A / {int(round(v))}% B')

    def _eq_band_changed(self, i, db):
        with self.amp.lock: self.amp.eq.set_gain(i, db)

    def _set_eq_level(self, db):
        with self.amp.lock: self.amp.eq.set_level(db)

    def _eq_reset(self):
        self._updating = True
        for sl in self.eq_sliders: sl.set(0)
        self.eq_level_sl.set(0)
        self._updating = False
        for i in range(len(self.eq_sliders)):
            with self.amp.lock: self.amp.eq.set_gain(i, 0.0)
        with self.amp.lock: self.amp.eq.set_level(0.0)
        for vv in self.eq_val_vars: vv.set('+0 dB')
        self.eq_level_val.set('+0 dB')

    # ── TAB 3: Looper ────────────────────────────────────────

    def _tab_looper(self, root):
        # settings bar
        cfg = tk.Frame(root, bg=PANEL); cfg.pack(fill='x', padx=8, pady=8)

        tk.Label(cfg, text='Length:', bg=PANEL, fg=FG, font=(FF,11)).pack(side='left', padx=(10,4))
        self.loop_len_var = tk.StringVar(value='4')
        ttk.Combobox(cfg, textvariable=self.loop_len_var, width=5, state='readonly',
                     values=['1','2','4','5','8','10','16','20','30','60']
                     ).pack(side='left', padx=(0,4))
        tk.Label(cfg, text='s', bg=PANEL, fg=DIM, font=(FF,10)).pack(side='left', padx=(0,12))

        tk.Label(cfg, text='Layers:', bg=PANEL, fg=FG, font=(FF,11)).pack(side='left', padx=(0,4))
        self.loop_layers_var = tk.StringVar(value='3')
        ttk.Combobox(cfg, textvariable=self.loop_layers_var, width=3, state='readonly',
                     values=[str(i) for i in range(1, MAX_LOOP_LAYERS+1)]
                     ).pack(side='left', padx=(0,12))

        tk.Label(cfg, text='Mode:', bg=PANEL, fg=FG, font=(FF,11)).pack(side='left', padx=(0,4))
        self.loop_mode_var = tk.StringVar(value='Manual')
        ttk.Combobox(cfg, textvariable=self.loop_mode_var, width=7, state='readonly',
                     values=['Manual','Auto']).pack(side='left', padx=(0,12))

        # BPM sync button
        ttk.Button(cfg, text='⇄ Sync to BPM',
                   command=self._loop_sync_bpm).pack(side='left', padx=(0,12))

        tk.Label(cfg, text='Layer vol:', bg=PANEL, fg=FG, font=(FF,11)).pack(side='left', padx=(0,4))
        self.loop_vol_var = tk.StringVar(value='80%')
        sl = self._scale(cfg, 0, 100, self._loop_vol_changed, 80, length=90)
        sl.pack(side='left')
        tk.Label(cfg, textvariable=self.loop_vol_var, bg=PANEL, fg=DIM,
                 font=(FF,10), width=5).pack(side='left', padx=4)

        for v in (self.loop_len_var, self.loop_layers_var, self.loop_mode_var):
            v.trace_add('write', lambda *_: self._loop_apply())

        # status
        sf = tk.Frame(root, bg=PANEL); sf.pack(fill='x', padx=8, pady=(0,6))
        self.loop_status_lbl = tk.Label(sf, text='Idle', bg=PANEL, fg=DIM,
                                         font=(FF,15,'bold'), anchor='center')
        self.loop_status_lbl.pack(fill='x', pady=(6,2))
        self.loop_time_lbl = tk.Label(sf, text='', bg=PANEL, fg=DIM,
                                       font=(FF,11), anchor='center')
        self.loop_time_lbl.pack(fill='x')
        self.loop_prog = tk.Canvas(sf, height=22, bg=BG_WELL,
                                    highlightthickness=1, highlightbackground='#2c333b')
        self.loop_prog.pack(fill='x', padx=10, pady=6)
        self.loop_dot_c = tk.Canvas(sf, height=28, bg=PANEL, highlightthickness=0)
        self.loop_dot_c.pack(fill='x', padx=10, pady=(0,8))

        # big rec button
        bf = tk.Frame(root, bg=BG); bf.pack(pady=10)
        self.loop_rec_btn = tk.Button(bf, text='⬤  REC / ARM',
                                       font=(FF,22,'bold'), bg=RED, fg='white',
                                       activebackground='#cc3333',
                                       relief='flat', bd=0, padx=40, pady=22,
                                       command=self._loop_rec)
        self.loop_rec_btn.pack(side='left', padx=12)
        self.loop_stop_btn = tk.Button(bf, text='■  STOP REC',
                                        font=(FF,14,'bold'), bg='#2c333b', fg=FG,
                                        activebackground='#3a4048',
                                        relief='flat', bd=0, padx=20, pady=14,
                                        command=self.looper.stop_record)
        self.loop_stop_btn.pack(side='left', padx=6)

        ctrl2 = tk.Frame(root, bg=BG); ctrl2.pack(pady=4)
        ttk.Button(ctrl2, text='Clear last', command=self._loop_clear_last).pack(side='left', padx=8)
        ttk.Button(ctrl2, text='Clear all',  command=self._loop_clear_all).pack(side='left', padx=8)
        ttk.Button(ctrl2, text='Export Mix…', command=self._loop_export_mix).pack(side='left', padx=8)

        lf = tk.Frame(root, bg=PANEL); lf.pack(fill='x', padx=8, pady=10)
        tk.Label(lf, text='Recorded layers', bg=PANEL, fg=DIM, font=(FF,10)).pack(anchor='w', padx=8, pady=(6,2))
        self.layer_list = tk.Frame(lf, bg=PANEL); self.layer_list.pack(fill='x', padx=8, pady=(0,8))

        self.loop_amp_notice = tk.Label(root,
            text='⚠  Start the Amp (I/O & Levels tab) to enable the looper',
            bg=BG, fg='#ffd166', font=(FF,10))
        self.loop_amp_notice.pack(pady=4)

        self._loop_apply()

    def _loop_sync_bpm(self):
        """Set loop length to a whole number of bars at the current BPM."""
        try:
            bpm  = int(self.bpm_str.get()) if hasattr(self, 'bpm_str') else self.metro.bpm
            bpm  = max(1, bpm)
            bpm_m = self.metro.bpm_m
            try: bars = int(self.loop_layers_var.get())  # reuse layer count as bar count hint
            except: bars = 1
            secs = round((60.0 / bpm) * bpm_m * bars, 2)
            # find nearest option or use free value by picking closest
            options = [1,2,4,5,8,10,16,20,30,60]
            nearest = min(options, key=lambda x: abs(x - secs))
            self.loop_len_var.set(str(nearest))
            # also set a helpful status
        except Exception as e:
            messagebox.showerror('Sync error', str(e))

    def _loop_apply(self):
        try: secs = float(self.loop_len_var.get())
        except: secs = 4.0
        try: layers = int(self.loop_layers_var.get())
        except: layers = 3
        mode = 'auto' if self.loop_mode_var.get() == 'Auto' else 'manual'
        self.looper.set_loop_length(secs)
        self.looper.set_max_layers(layers)
        self.looper.set_mode(mode)
        self._loop_refresh_list()

    def _loop_vol_changed(self, v):
        self.looper.layer_vol = float(v)/100.0
        self.loop_vol_var.set(f'{int(float(v))}%')

    def _loop_rec(self):
        if not self.amp.running:
            messagebox.showinfo('Amp not running',
                'Start the Amp on the I/O & Levels tab first.')
            return
        self.looper.arm()

    def _loop_clear_last(self):
        self.looper.clear_last(); self._loop_refresh_list()

    def _loop_clear_all(self):
        self.looper.clear_all(); self._loop_refresh_list()

    def _loop_toggle_mute(self, idx):
        self.looper.toggle_mute(idx)
        self._loop_refresh_list()

    def _loop_export_mix(self):
        if self.looper.n_complete == 0:
            messagebox.showinfo('Nothing to export', 'Record at least one layer first.')
            return
        path = filedialog.asksaveasfilename(
            title='Export Loop Mix', defaultextension='.wav', initialfile='loop_mix.wav',
            filetypes=[('WAV files', '*.wav'), ('All files', '*.*')])
        if not path: return
        try:
            write_wav_mono(path, self.looper.get_mix(), SAMPLE_RATE)
        except Exception as e:
            messagebox.showerror('Export failed', str(e))
            return
        messagebox.showinfo('Exported', f'Loop mix saved to:\n{path}')

    def _loop_export_layer(self, idx):
        layer = self.looper.get_layer(idx)
        if layer is None:
            return
        path = filedialog.asksaveasfilename(
            title=f'Export Layer {idx+1}', defaultextension='.wav',
            initialfile=f'layer_{idx+1}.wav',
            filetypes=[('WAV files', '*.wav'), ('All files', '*.*')])
        if not path: return
        try:
            write_wav_mono(path, layer, SAMPLE_RATE)
        except Exception as e:
            messagebox.showerror('Export failed', str(e))
            return
        messagebox.showinfo('Exported', f'Layer {idx+1} saved to:\n{path}')

    def _loop_refresh_list(self):
        for w in self.layer_list.winfo_children(): w.destroy()
        n = self.looper.n_complete
        if n == 0:
            tk.Label(self.layer_list, text='No layers recorded yet.',
                     bg=PANEL, fg=DIM, font=(FF,10)).pack(anchor='w')
            return
        for i in range(n):
            muted = i < len(self.looper.layer_mute) and self.looper.layer_mute[i]
            row = tk.Frame(self.layer_list, bg=PANEL); row.pack(fill='x', pady=1)
            tk.Label(row, text=f'  Layer {i+1}', bg=PANEL, fg=(DIM if muted else ACCENT),
                     font=(FF,10,'bold'), width=10).pack(side='left')
            tk.Label(row, text=f'{self.looper.loop_seconds:.1f}s', bg=PANEL, fg=DIM,
                     font=(FF,10), width=8).pack(side='left')
            bar = tk.Canvas(row, width=120, height=14, bg='#1e252d', highlightthickness=0)
            bar.pack(side='left', padx=6)
            bar.create_rectangle(0, 0, 120, 14, fill=(DIM if muted else ACCENT), outline='')
            tk.Button(row, text=('Unmute' if muted else 'Mute'), font=(FF,9),
                      bg=(RED if muted else '#2c333b'), fg='white' if muted else FG,
                      activebackground='#3a4048', relief='flat', bd=0, padx=8, pady=2,
                      command=lambda idx=i: self._loop_toggle_mute(idx)
                      ).pack(side='left', padx=(10,4))
            ttk.Button(row, text='Export…',
                       command=lambda idx=i: self._loop_export_layer(idx)
                       ).pack(side='left', padx=4)

    def _poll_looper(self):
        try:
            while True:
                evt, _ = self.looper.event_q.get_nowait()
                if evt in ('layer_done','all_done','rec_start'):
                    self._loop_refresh_list()
        except queue.Empty: pass

        self.loop_status_lbl.configure(text=self.looper.status_text,
                                        fg=self.looper.status_color)
        tr = self.looper.time_remaining
        if tr > 0:
            self.loop_time_lbl.configure(text=f'{tr:.1f}s remaining')
        elif self.looper.state == _LS_PLAYING:
            pos = self.looper._play_pos / SAMPLE_RATE
            self.loop_time_lbl.configure(
                text=f'{pos:.1f}s / {self.looper.loop_seconds:.1f}s')
        else:
            self.loop_time_lbl.configure(text='')

        # progress bar
        frac = self.looper.progress
        c = self.loop_prog; c.delete('all')
        w = c.winfo_width() or 700; h = c.winfo_height() or 22
        st = self.looper.state
        col = RED if st in (_LS_RECORDING, _LS_OVERDUB) else \
              ACCENT if st == _LS_PLAYING else '#ffd166'
        fw = max(0, int(w * frac))
        if fw: c.create_rectangle(0,0,fw,h, fill=col, outline='')
        c.create_text(w//2, h//2, text=f'{int(frac*100)}%', fill=FG, font=(FF,9))

        # layer dots
        dc = self.loop_dot_c; dc.delete('all')
        dc.update_idletasks()
        dw = dc.winfo_width() or 700
        mx = self.looper.max_layers; done = self.looper.n_complete
        r2, m2 = 10, 16
        gap = (dw-2*m2)/max(1,mx-1) if mx>1 else 0
        for i in range(mx):
            x = m2 + gap*i if mx>1 else dw//2
            if i < done: col2, oc = ACCENT, ACCENT
            elif st in (_LS_RECORDING,_LS_OVERDUB,_LS_COUNTDOWN) and i==done:
                col2, oc = RED, RED
            else: col2, oc = '#1e252d', '#3a4048'
            dc.create_oval(x-r2,4,x+r2,4+2*r2, fill=col2, outline=oc, width=2)
            dc.create_text(x,4+r2, text=str(i+1),
                           fill=ACCENT_DARK if i<done else DIM, font=(FF,8,'bold'))

        # REC button color
        if st in (_LS_RECORDING, _LS_OVERDUB):
            self.loop_rec_btn.configure(bg='#991111', text='⬤  RECORDING…')
        elif st == _LS_COUNTDOWN:
            self.loop_rec_btn.configure(bg='#aa7700', text='⬤  READY…')
        elif st == _LS_PLAYING and done < self.looper.max_layers \
                and self.loop_mode_var.get() == 'Manual':
            self.loop_rec_btn.configure(bg=RED, text=f'⬤  REC layer {done+1}')
        else:
            self.loop_rec_btn.configure(bg=RED, text='⬤  REC / ARM')

        if self.amp.running:
            self.loop_amp_notice.pack_forget()
        else:
            try: self.loop_amp_notice.pack(pady=4)
            except: pass

        self.after(80, self._poll_looper)

    # ── TAB 4: Metronome ─────────────────────────────────────

    def _tab_metro(self, root):
        top = tk.Frame(root, bg=PANEL); top.pack(fill='x', padx=6, pady=(6,10))

        self.bpm_str = tk.StringVar(value='120')
        row = tk.Frame(top, bg=PANEL); row.pack(pady=14)
        ttk.Button(row, text='－', width=3, command=lambda: self._bpm_nudge(-1)).pack(side='left', padx=6)
        tk.Label(row, textvariable=self.bpm_str, bg=PANEL, fg=ACCENT,
                 font=(FF,46,'bold')).pack(side='left', padx=14)
        ttk.Button(row, text='+', width=3, command=lambda: self._bpm_nudge(1)).pack(side='left', padx=6)
        tk.Label(top, text='BPM  (30–250)', bg=PANEL, fg=DIM, font=(FF,10)).pack()

        self.bpm_slider = self._scale(top, 30, 250, self._bpm_from_slider, 120, length=520)
        self.bpm_slider.pack(pady=(10,4), padx=20, fill='x')

        tap_row = tk.Frame(top, bg=PANEL); tap_row.pack(pady=8)
        ttk.Button(tap_row, text='TAP TEMPO', command=self._tap).pack(side='left', padx=8)
        self.tap_lbl = tk.Label(tap_row, text='', bg=PANEL, fg=DIM, font=(FF,10))
        self.tap_lbl.pack(side='left', padx=8)

        self.dot_canvas = tk.Canvas(root, height=90, bg=BG, highlightthickness=0)
        self.dot_canvas.pack(fill='x', padx=20, pady=6)
        self.dots = []

        ctrl = tk.Frame(root, bg=PANEL); ctrl.pack(fill='x', padx=6, pady=10)

        c1 = tk.Frame(ctrl, bg=PANEL); c1.pack(side='left', expand=True, fill='x', padx=10, pady=10)
        tk.Label(c1, text='Beats / measure', bg=PANEL, fg=FG, font=(FF,11)).pack(anchor='w')
        self.beats_var = tk.IntVar(value=4)
        sp = ttk.Spinbox(c1, from_=1, to=12, textvariable=self.beats_var,
                          width=5, command=self._beats_changed)
        sp.pack(anchor='w', pady=4)
        sp.bind('<FocusOut>', lambda e: self._beats_changed())

        c2 = tk.Frame(ctrl, bg=PANEL); c2.pack(side='left', expand=True, fill='x', padx=10, pady=10)
        tk.Label(c2, text='Subdivision', bg=PANEL, fg=FG, font=(FF,11)).pack(anchor='w')
        self.subdiv_var = tk.StringVar(value='Quarter notes')
        cb = ttk.Combobox(c2, textvariable=self.subdiv_var, state='readonly', width=16,
                           values=['Quarter notes','Eighth notes','Triplets','Sixteenth notes'])
        cb.pack(anchor='w', pady=4)
        cb.bind('<<ComboboxSelected>>', lambda e: self._subdiv_changed())

        c3 = tk.Frame(ctrl, bg=PANEL); c3.pack(side='left', expand=True, fill='x', padx=10, pady=10)
        tk.Label(c3, text='Volume', bg=PANEL, fg=FG, font=(FF,11)).pack(anchor='w')
        vol_sl = self._scale(c3, 0, 100, lambda v: setattr(self.metro, 'volume', float(v)/100.0),
                              70, length=140)
        vol_sl.pack(anchor='w', pady=4)
        self.metro_mute_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(c3, text='Mute  (beat light keeps running)',
                        variable=self.metro_mute_var,
                        command=lambda: setattr(self.metro, 'mute', self.metro_mute_var.get())
                        ).pack(anchor='w', pady=(4,0))

        sr = tk.Frame(root, bg=PANEL); sr.pack(fill='x', padx=6, pady=(0,6))
        tk.Label(sr, text='Sound', bg=PANEL, fg=FG, font=(FF,11)).pack(side='left', padx=(10,8))
        self.sound_var = tk.StringVar(value='Classic Beep')
        scb = ttk.Combobox(sr, textvariable=self.sound_var, values=SOUND_KITS,
                            state='readonly', width=20)
        scb.pack(side='left')
        scb.bind('<<ComboboxSelected>>', lambda e: self.metro.set_kit(self.sound_var.get()))
        ttk.Button(sr, text='▶ Preview', command=self._preview).pack(side='left', padx=10)

        self.start_btn = ttk.Button(root, text='▶  START', style='Go.TButton',
                                     command=self._toggle_metro)
        self.start_btn.pack(pady=16, ipadx=20)

        self._rebuild_dots()

    # ── TAB: Settings ────────────────────────────────────────

    def _tab_settings(self, root):
        inner = self._scrollable(root)

        # ── Keybinds ─────────────────────────────────────────
        # Generic keyboard "Learn" mapping — mirrors the MIDI Controller
        # section below field-for-field (KEYBIND_MAPPABLE_TARGETS is
        # MIDI_MAPPABLE_TARGETS with every continuous knob turned into an
        # Up/Down nudge pair, since a keypress is a discrete event, not a
        # knob sweep), so every function mappable to a MIDI controller is
        # also mappable to a keyboard key, in the same chain order.
        kb_sec = self._sec(inner, 'Keybinds')
        tk.Label(kb_sec,
                 text='Click Learn, then press a key. Knobs get an Up/Down '
                      'key pair instead of one key (a keypress nudges, it '
                      "can't sweep like a knob).",
                 bg=PANEL, fg=DIM, font=(FF,9)).pack(anchor='w', padx=10, pady=(6,2))

        step_row = tk.Frame(kb_sec, bg=PANEL); step_row.pack(fill='x', padx=10, pady=(2,6))
        tk.Label(step_row, text='Nudge step size', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.nudge_step_var = tk.DoubleVar(value=5.0)
        self.nudge_step_lbl = tk.StringVar(value='5%')
        def _nudge_step_cb(v):
            fv = float(v)
            self.nudge_step_var.set(fv)
            self.nudge_step_lbl.set(f'{fv:.0f}%')
        self.nudge_step_sl = self._scale(step_row, 1, 25, _nudge_step_cb, 5.0, length=220)
        self.nudge_step_sl.pack(side='left', padx=6)
        tk.Label(step_row, textvariable=self.nudge_step_lbl, bg=PANEL, fg=DIM,
                 font=(FF,10), width=6).pack(side='left')

        self.keybind_learn_status = tk.Label(kb_sec, text='', bg=PANEL, fg=DIM, font=(FF,9))
        self.keybind_learn_status.pack(anchor='w', padx=10, pady=(0,6))

        self.keybind_rows = {}
        cur_cat = None
        for t in KEYBIND_MAPPABLE_TARGETS:
            if t['category'] != cur_cat:
                cur_cat = t['category']
                tk.Label(kb_sec, text=cur_cat, bg=PANEL, fg=ACCENT,
                         font=(FF,10,'bold')).pack(anchor='w', padx=10, pady=(8,2))
            row = tk.Frame(kb_sec, bg=PANEL); row.pack(fill='x', padx=10, pady=1)
            tk.Label(row, text=t['label'], bg=PANEL, fg=FG, font=(FF,10),
                     width=26, anchor='w').pack(side='left')
            map_lbl = tk.Label(row, text=self._keybind_mapping_text(t['id']), bg=PANEL,
                                fg=DIM, font=(FF,9), width=12, anchor='w')
            map_lbl.pack(side='left', padx=6)
            ttk.Button(row, text='Learn', width=7,
                       command=lambda tid=t['id']: self._keybind_start_learn(tid)
                       ).pack(side='left', padx=2)
            ttk.Button(row, text='Clear', width=7,
                       command=lambda tid=t['id']: self._keybind_clear(tid)
                       ).pack(side='left', padx=2)
            self.keybind_rows[t['id']] = map_lbl

        # ── Window ───────────────────────────────────────────
        win_sec = self._sec(inner, 'Window')
        self.always_on_top_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(win_sec, text='Keep window always on top',
                         variable=self.always_on_top_var,
                         command=lambda: self.attributes(
                             '-topmost', self.always_on_top_var.get())
                         ).pack(anchor='w', padx=10, pady=(6,4))

        self.confirm_quit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(win_sec, text='Confirm before quitting while the amp is running',
                         variable=self.confirm_quit_var
                         ).pack(anchor='w', padx=10, pady=(0,6))

        # ── Appearance ───────────────────────────────────────
        appear_sec = self._sec(inner, 'Appearance')
        tk.Label(appear_sec,
                 text='Changing either color rebuilds the whole window '
                      'right away — the tab you’re on stays selected.',
                 bg=PANEL, fg=DIM, font=(FF,9)).pack(anchor='w', padx=10, pady=(6,4))

        accent_row = tk.Frame(appear_sec, bg=PANEL); accent_row.pack(fill='x', padx=10, pady=4)
        tk.Label(accent_row, text='Accent color', bg=PANEL, fg=FG,
                 font=(FF,10), width=14, anchor='w').pack(side='left')
        self.accent_swatch = tk.Label(accent_row, bg=ACCENT, width=4, relief='flat')
        self.accent_swatch.pack(side='left', padx=6)
        ttk.Button(accent_row, text='Choose…', command=self._choose_accent_color).pack(side='left', padx=4)

        bg_row = tk.Frame(appear_sec, bg=PANEL); bg_row.pack(fill='x', padx=10, pady=4)
        tk.Label(bg_row, text='Background color', bg=PANEL, fg=FG,
                 font=(FF,10), width=14, anchor='w').pack(side='left')
        self.bg_swatch = tk.Label(bg_row, bg=BG, width=4, relief='flat')
        self.bg_swatch.pack(side='left', padx=6)
        ttk.Button(bg_row, text='Choose…', command=self._choose_bg_color).pack(side='left', padx=4)

        ttk.Button(appear_sec, text='Reset to Default Theme',
                   command=self._reset_theme).pack(anchor='w', padx=10, pady=(4,8))

        # ── MIDI Controller ────────────────────────────────────
        # Generic CC/note "Learn" mapping — works with any class-compliant
        # USB-MIDI controller (built against an Akai LPD8: 8 pads send
        # Note On/Off, 8 knobs send Control Change), not hardcoded to one
        # device's default CC/note numbers.
        midi_sec = self._sec(inner, 'MIDI Controller')
        if not HAS_MIDI:
            tk.Label(midi_sec,
                     text='python-rtmidi not installed — MIDI control unavailable '
                          '(pip install python-rtmidi, then restart).',
                     bg=PANEL, fg=DIM, font=(FF,10)).pack(anchor='w', padx=10, pady=10)
            return

        dev_row = tk.Frame(midi_sec, bg=PANEL); dev_row.pack(fill='x', padx=10, pady=(8,4))
        tk.Label(dev_row, text='Device:', bg=PANEL, fg=FG, font=(FF,10), width=9).pack(side='left')
        self.midi_port_var = tk.StringVar(value=self.midi_ctrl.port_name or '')
        self.midi_port_box = ttk.Combobox(dev_row, textvariable=self.midi_port_var,
                                           state='readonly', width=32,
                                           values=[])
        # Populated moments later via after(), not synchronously here —
        # see _auto_reconnect_midi()'s docstring for why calling rtmidi
        # during the initial widget build crashes on macOS specifically.
        self.after(0, self._midi_refresh_ports)
        self.midi_port_box.pack(side='left', padx=4)
        self.midi_port_box.bind('<<ComboboxSelected>>', lambda e: self._midi_port_changed())
        ttk.Button(dev_row, text='⟳', width=3, command=self._midi_refresh_ports).pack(side='left', padx=4)
        connected = bool(self.midi_ctrl.port_name)
        self.midi_status_lbl = tk.Label(dev_row,
                                         text=('Connected' if connected else 'Not connected'),
                                         bg=PANEL, fg=(ACCENT if connected else DIM), font=(FF,10))
        self.midi_status_lbl.pack(side='left', padx=10)

        led_row = tk.Frame(midi_sec, bg=PANEL); led_row.pack(fill='x', padx=10, pady=(0,4))
        self.lpd8_led_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(led_row,
                        text='Enable LPD8 mk2 pad LED feedback (experimental — unofficial, '
                             'reverse-engineered SysEx; worst case it just does nothing)',
                        variable=self.lpd8_led_var,
                        command=self._lpd8_led_toggled).pack(anchor='w')

        tk.Label(midi_sec,
                 text='Click Learn, then move a knob or press a pad on your controller.',
                 bg=PANEL, fg=DIM, font=(FF,9)).pack(anchor='w', padx=10, pady=(0,2))
        self.midi_learn_status = tk.Label(midi_sec, text='', bg=PANEL, fg=DIM, font=(FF,9))
        self.midi_learn_status.pack(anchor='w', padx=10, pady=(0,6))

        self.midi_rows = {}
        cur_cat = None
        for t in MIDI_MAPPABLE_TARGETS:
            if t['category'] != cur_cat:
                cur_cat = t['category']
                tk.Label(midi_sec, text=cur_cat, bg=PANEL, fg=ACCENT,
                         font=(FF,10,'bold')).pack(anchor='w', padx=10, pady=(8,2))
            row = tk.Frame(midi_sec, bg=PANEL); row.pack(fill='x', padx=10, pady=1)
            tk.Label(row, text=t['label'], bg=PANEL, fg=FG, font=(FF,10),
                     width=24, anchor='w').pack(side='left')
            map_lbl = tk.Label(row, text=self._midi_mapping_text(t['id']), bg=PANEL,
                                fg=DIM, font=(FF,9), width=12, anchor='w')
            map_lbl.pack(side='left', padx=6)
            ttk.Button(row, text='Learn', width=7,
                       command=lambda tid=t['id']: self._midi_start_learn(tid)
                       ).pack(side='left', padx=2)
            ttk.Button(row, text='Clear', width=7,
                       command=lambda tid=t['id']: self._midi_clear(tid)
                       ).pack(side='left', padx=2)
            self.midi_rows[t['id']] = map_lbl

    # ── MIDI controller ──────────────────────────────────────

    def _midi_mapping_text(self, target_id):
        for key, tid in self.midi_mappings.items():
            if tid == target_id:
                kind, num = key
                return f'CC {num}' if kind == 'cc' else f'Note {num}'
        return '—'

    def _midi_refresh_ports(self):
        ports = MidiController.list_ports()
        self.midi_port_box.configure(values=ports)
        if self.midi_port_var.get() not in ports:
            self.midi_port_var.set('')

    def _auto_reconnect_midi(self):
        """Reopen the last-used MIDI port on launch, if any and if it's
        still present. Deliberately run via self.after() from __init__
        rather than synchronously during Tk widget construction — a
        real, macOS-only crash (fatal GIL error inside
        rtmidi.MidiIn().get_ports(), found by this project's GitHub
        Actions CI matrix) only ever happened when a rtmidi call ran
        during the initial build, before Tk's own event loop had
        started; the exact same call works fine moments later, and an
        isolated test with no Tk involved at all never crashed either
        — pointing at some macOS-specific interaction between
        Tkinter's Cocoa runloop bootstrapping and CoreMIDI's own
        runloop expectations during that narrow pre-mainloop window,
        not anything wrong with the rtmidi call itself."""
        if self._midi_last_port and self._midi_last_port in MidiController.list_ports():
            if self.midi_ctrl.open(self._midi_last_port):
                self.midi_port_var.set(self._midi_last_port)
                self.midi_status_lbl.configure(text='Connected', fg=ACCENT)
                self._lpd8_refresh_leds(force=True)

    def _midi_port_changed(self):
        port = self.midi_port_var.get()
        if not port:
            self.midi_ctrl.close()
            self.midi_status_lbl.configure(text='Not connected', fg=DIM)
            self._save_midi_mappings()
            return
        ok = self.midi_ctrl.open(port)
        self.midi_status_lbl.configure(text=('Connected' if ok else 'Failed to open'),
                                        fg=(ACCENT if ok else RED))
        self._save_midi_mappings()
        self._lpd8_refresh_leds(force=True)

    def _midi_start_learn(self, target_id):
        self._midi_learn_target = target_id
        label = MIDI_TARGETS_BY_ID[target_id]['label']
        self.midi_learn_status.configure(
            text=f'Learning "{label}"… move a knob or hit a pad', fg='#ffd166')

    def _midi_clear(self, target_id):
        self.midi_mappings = {k: v for k, v in self.midi_mappings.items() if v != target_id}
        self._refresh_midi_ui()
        self._lpd8_refresh_leds(force=True)
        self._save_midi_mappings()

    def _refresh_midi_ui(self):
        for tid, lbl in getattr(self, 'midi_rows', {}).items():
            lbl.configure(text=self._midi_mapping_text(tid))

    def _poll_midi(self):
        try:
            while True:
                kind, number, value = self.midi_q.get_nowait()
                self._handle_midi_event(kind, number, value)
        except queue.Empty:
            pass
        self.after(15, self._poll_midi)

    def _handle_midi_event(self, kind, number, value):
        key = ('cc', number) if kind == 'cc' else ('note', number)

        if self._midi_learn_target is not None and kind in ('cc', 'note_on'):
            tid = self._midi_learn_target
            # Drop any existing binding to the same target or the same
            # incoming key, so Learn always leaves a clean 1:1 mapping.
            self.midi_mappings = {k: v for k, v in self.midi_mappings.items()
                                    if v != tid and k != key}
            self.midi_mappings[key] = tid
            self._midi_learn_target = None
            self._refresh_midi_ui()
            self._save_midi_mappings()
            self._lpd8_refresh_leds(force=True)
            kind_word = 'CC' if key[0] == 'cc' else 'Note'
            self.midi_learn_status.configure(
                text=f'Mapped "{MIDI_TARGETS_BY_ID[tid]["label"]}" to {kind_word} {key[1]}',
                fg=ACCENT)
            return

        tid = self.midi_mappings.get(key)
        if tid is None:
            return
        target = MIDI_TARGETS_BY_ID.get(tid)
        if target is None:
            return
        if kind == 'cc' and target['kind'] == 'cc':
            target['action'](self, value / 127.0)
        elif target['kind'] == 'toggle':
            # A quick tap toggles on/off (fires on release, timed against
            # how long the pad was actually held); holding past
            # MIDI_LONG_PRESS_MS instead cycles that pedal's emulation
            # type (if it has one — see PEDAL_MODE_CYCLE) and suppresses
            # the on/off toggle for that press.
            if kind == 'note_on':
                self._midi_arm_toggle(key, tid)
            elif kind == 'note_off':
                self._midi_release_toggle(key, tid)
        elif kind == 'note_on' and target['kind'] == 'trigger':
            target['action'](self, None)
            pad = _lpd8_pad_index_for_note(number)
            if pad is not None:
                self._lpd8_flash_pad(pad)
        # note_off is otherwise ignored — trigger targets act on the press only.

    MIDI_LONG_PRESS_MS = 450

    # ── shared press/hold dispatch ────────────────────────────
    # A "quick tap toggles, holding past MIDI_LONG_PRESS_MS instead cycles
    # the pedal's emulation type" is the same behavior whether the press
    # came from a MIDI pad (note_on/note_off) or a keyboard key
    # (KeyPress/KeyRelease) — these four methods hold that logic once,
    # keyed off whichever `pending` dict the caller passes in
    # (self._midi_pending_press or self._keybind_pending_press), so MIDI
    # and keybinds can never drift into behaving differently.

    def _dispatch_press(self, pending, key, tid, status_label=None):
        self._cancel_pending(pending, key)
        after_id = self.after(self.MIDI_LONG_PRESS_MS,
                               lambda: self._fire_long_press(pending, key, tid, status_label))
        pending[key] = {'timer': after_id, 'fired_long': False}

    def _cancel_pending(self, pending, key):
        pend = pending.pop(key, None)
        if pend and pend['timer'] is not None:
            try: self.after_cancel(pend['timer'])
            except Exception: pass

    def _fire_long_press(self, pending, key, tid, status_label=None):
        pend = pending.get(key)
        if pend is None:
            return
        pend['timer'] = None
        new_val = self._cycle_pedal_mode(tid)
        if new_val is not None:
            # Only suppress the eventual release-toggle when there's
            # actually a mode to cycle to — otherwise a pedal with just
            # one voicing (Compressor, Reverb, Tuner Mute) would go dead
            # on any hold longer than the threshold: the timer fires,
            # cycling does nothing, and the release then does nothing
            # either. With fired_long left False, those pedals just
            # toggle normally on release no matter how long the key/pad
            # was held, same as tapping it.
            pend['fired_long'] = True
            if status_label is not None:
                label = MIDI_TARGETS_BY_ID[tid]['label']
                status_label.configure(text=f'{label}: mode → {new_val}', fg=ACCENT)

    def _dispatch_release(self, pending, key, tid):
        pend = pending.pop(key, None)
        if pend is None:
            return   # e.g. mapping changed mid-press — nothing to release
        if pend['timer'] is not None:
            try: self.after_cancel(pend['timer'])
            except Exception: pass
        if pend['fired_long']:
            return   # already handled as a long press; the release itself does nothing
        MIDI_TARGETS_BY_ID[tid]['action'](self, None)

    def _cycle_pedal_mode(self, tid):
        """Advance the pedal behind `tid` to its next emulation type (see
        PEDAL_MODE_CYCLE) and sync the on-screen control; returns the new
        value, or None if this target has no mode cycle at all."""
        cycle = PEDAL_MODE_CYCLE.get(tid)
        if not cycle:
            return None
        attr, values = cycle
        cur = getattr(self.amp.pedals, attr)
        idx = values.index(cur) if cur in values else -1
        new_val = values[(idx + 1) % len(values)]
        setattr(self.amp.pedals, attr, new_val)
        self._pedal_widget_sync(attr, new_val)
        if attr == 'wah_mode' and hasattr(self, '_wah_refresh_ui'):
            self._wah_refresh_ui()   # show/hide the manual-position vs auto-sweep rows
        return new_val

    def _midi_arm_toggle(self, key, tid):
        self._dispatch_press(self._midi_pending_press, key, tid,
                              getattr(self, 'midi_learn_status', None))

    def _midi_release_toggle(self, key, tid):
        self._dispatch_release(self._midi_pending_press, key, tid)

    def _nudge_cc_target(self, tid, direction):
        """Step any continuous ('cc' kind) mappable target by a fraction
        of its own range, in the given direction — the generic keybind
        (and MIDI pad) alternative to a knob's continuous sweep. Reads
        the target's current value and range via the nudge_getter/
        nudge_range metadata attached to its action (see the _midi_*
        _action factories above) and re-applies through that same
        action(), so a nudge and a knob driving the same target always
        go through identical code."""
        target = MIDI_TARGETS_BY_ID.get(tid)
        if target is None:
            return
        action = target['action']
        getter = getattr(action, 'nudge_getter', None)
        rng = getattr(action, 'nudge_range', None)
        if getter is None or rng is None:
            return
        lo, hi = rng
        if hi == lo:
            return
        step_frac = getattr(self, 'nudge_step_var', None)
        step_frac = (step_frac.get() / 100.0) if step_frac is not None else 0.05
        cur = getter(self)
        norm = (cur - lo) / (hi - lo)
        norm = max(0.0, min(1.0, norm + direction * step_frac))
        action(self, norm)

    def _pedal_widget_sync(self, attr, value):
        """Move the Pedals tab's own control for `attr` to match a value
        that was just set some other way (MIDI, mode-cycling) — a no-op
        if that control isn't currently registered (tab not built yet,
        or this attr has no on-screen widget)."""
        entry = getattr(self, '_pedal_widgets', {}).get(attr)
        if entry is None:
            return
        if entry['kind'] == 'slider':
            self._updating = True
            entry['widget'].set(value)
            self._updating = False
            vv, fmt = entry['val_var'], entry['fmt']
            vv.set(fmt(value) if fmt else f'{value:.2f}')
        elif entry['kind'] == 'radio':
            entry['widget'].set(value)

    def _load_midi_mappings(self):
        self._midi_last_port = None
        try:
            with open(MIDI_MAP_FILE) as f:
                data = json.load(f)
            self._midi_last_port = data.get('port')
            for key_str, tid in data.get('mappings', {}).items():
                if tid not in MIDI_TARGETS_BY_ID:
                    continue
                kind, _, num_str = key_str.partition(':')
                try:
                    self.midi_mappings[(kind, int(num_str))] = tid
                except ValueError:
                    pass
        except Exception:
            pass

    def _save_midi_mappings(self):
        try:
            data = {
                'port': self.midi_ctrl.port_name,
                'mappings': {f'{k[0]}:{k[1]}': v for k, v in self.midi_mappings.items()},
            }
            with open(MIDI_MAP_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ── LPD8 mk2 pad LED feedback (experimental) ──────────────
    # Recomputes all 8 pads' colors from whatever's currently mapped to a
    # note and pushes them in one SysEx message — driven by the *state*
    # (which pads are mapped, and each toggle target's current on/off
    # value), not by hooking every individual place that state can
    # change, so it stays correct regardless of whether a pedal was
    # flipped by mouse, keybind, MIDI, a Tone Profile load, or a preset
    # load. See the trace_add calls on panic_var/hum_var/ir_blend_on_var/
    # eq_on_var and _update_pedal_led() for what actually triggers a
    # recompute; it's cheap to call speculatively since it no-ops
    # instantly whenever the feature's off, unconnected, or nothing
    # actually changed since the last send.

    def _lpd8_led_toggled(self):
        if self.lpd8_led_var.get():
            self._lpd8_last_colors = None   # force a real send even if colors happen to match
            self._lpd8_refresh_leds(force=True)
        else:
            self._lpd8_send_all_off()

    def _lpd8_send_all_off(self):
        """Release every pad back to black — used when the feature is
        turned off, so it doesn't leave the board stuck on whatever
        colors were showing."""
        if self.midi_ctrl.midi_out is None:
            return
        self.midi_ctrl.send_sysex(_lpd8_mk2_pad_color_sysex([(0, 0, 0)] * 8))
        self._lpd8_last_colors = None

    def _lpd8_pad_target_color(self, tid, on):
        """(r,g,b) 0-255 for a toggle-kind target's pad, given its current
        on/off state. Pedal on/off pads reuse that pedal's own icon color
        (PEDAL_ICON_SPECS) rather than one generic 'on' color, so e.g. the
        Fuzz pad and the Chorus pad light differently, matching the
        Pedals tab's own Pedal Status strip."""
        if not on:
            return (0, 0, 0)
        if tid.startswith('pedal_') and tid.endswith('_on'):
            code = tid[len('pedal_'):-len('_on')]
            spec = PEDAL_ICON_SPECS.get(code)
            if spec:
                return _hex_to_rgb(spec[1])
        if tid == 'panic_mute_all':
            return _hex_to_rgb(RED)
        return _hex_to_rgb(ACCENT)

    def _lpd8_flash_pad(self, pad_idx):
        """Brief white flash on a trigger-mapped pad (tap tempo, loop
        record, amp start/stop, ...) — trigger targets have no persistent
        on/off state to show, so a flash is the only feedback that makes
        sense for them, same idea as a real footswitch's momentary LED."""
        self._lpd8_flash_until[pad_idx] = time.monotonic() + 0.12
        self._lpd8_refresh_leds(force=True)
        self.after(130, self._lpd8_refresh_leds)

    def _lpd8_refresh_leds(self, force=False):
        if not getattr(self, 'lpd8_led_var', None) or not self.lpd8_led_var.get():
            return
        if self.midi_ctrl.midi_out is None:
            return
        now = time.monotonic()
        colors = [(0, 0, 0)] * 8
        for (kind, num), tid in self.midi_mappings.items():
            if kind != 'note':
                continue
            pad = _lpd8_pad_index_for_note(num)
            if pad is None:
                continue
            target = MIDI_TARGETS_BY_ID.get(tid)
            if target is None:
                continue
            flash_until = self._lpd8_flash_until.get(pad)
            if flash_until and now < flash_until:
                colors[pad] = (255, 255, 255)
                continue
            elif flash_until:
                del self._lpd8_flash_until[pad]
            if target['kind'] == 'toggle':
                getter = getattr(target['action'], 'state_getter', None)
                on = bool(getter(self)) if getter is not None else False
                colors[pad] = self._lpd8_pad_target_color(tid, on)
            elif target['kind'] == 'trigger':
                colors[pad] = (14, 14, 14)   # dim idle glow: mapped but momentary, nothing to show "on"
        if not force and colors == self._lpd8_last_colors:
            return
        self._lpd8_last_colors = colors
        self.midi_ctrl.send_sysex(_lpd8_mk2_pad_color_sysex(colors))

    # ── Keyboard keybinds ─────────────────────────────────────
    # Mirrors the MIDI Controller section just above: a generic "Learn"
    # mapping over KEYBIND_MAPPABLE_TARGETS instead of a single
    # hardcoded Wah nudge. Saved into the session file (not a Tone
    # Profile) since a keyboard layout is a property of the player's
    # setup, not of a particular sound — see _save_session/_load_session.

    def _keybind_mapping_text(self, target_id):
        for key, tid in self.keybind_mappings.items():
            if tid == target_id:
                return key
        return '—'

    def _keybind_start_learn(self, target_id):
        self._keybind_learn_target = target_id
        label = KEYBIND_TARGETS_BY_ID[target_id]['label']
        self.keybind_learn_status.configure(
            text=f'Learning "{label}"… press a key', fg='#ffd166')

    def _keybind_clear(self, target_id):
        self.keybind_mappings = {k: v for k, v in self.keybind_mappings.items() if v != target_id}
        self._refresh_keybind_ui()
        self._save_session()

    def _refresh_keybind_ui(self):
        for tid, lbl in getattr(self, 'keybind_rows', {}).items():
            lbl.configure(text=self._keybind_mapping_text(tid))

    def _on_key_press(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return   # a text field (IR label, BPM spinbox) is typing, not a keybind
        keysym = event.keysym
        if keysym in KEY_MODIFIER_KEYSYMS:
            return

        if self._keybind_learn_target is not None:
            tid = self._keybind_learn_target
            # Drop any existing binding to the same target or the same
            # incoming key, so Learn always leaves a clean 1:1 mapping.
            self.keybind_mappings = {k: v for k, v in self.keybind_mappings.items()
                                      if v != tid and k != keysym}
            self.keybind_mappings[keysym] = tid
            self._keybind_learn_target = None
            self._refresh_keybind_ui()
            self._save_session()
            label = KEYBIND_TARGETS_BY_ID[tid]['label']
            self.keybind_learn_status.configure(
                text=f'Mapped "{label}" to {keysym}', fg=ACCENT)
            return

        if keysym in self._keys_down:
            return   # OS key-repeat while held — act once per physical press
        tid = self.keybind_mappings.get(keysym)
        if tid is None:
            return
        target = KEYBIND_TARGETS_BY_ID.get(tid)
        if target is None:
            return
        self._keys_down.add(keysym)
        if target['kind'] == 'toggle':
            self._dispatch_press(self._keybind_pending_press, keysym, tid,
                                  getattr(self, 'keybind_learn_status', None))
        elif target['kind'] == 'trigger':
            target['action'](self, None)

    def _on_key_release(self, event):
        keysym = event.keysym
        self._keys_down.discard(keysym)
        tid = self.keybind_mappings.get(keysym)
        if tid is None:
            return
        target = KEYBIND_TARGETS_BY_ID.get(tid)
        if target is not None and target['kind'] == 'toggle':
            self._dispatch_release(self._keybind_pending_press, keysym, tid)

    def _set_bpm(self, val):
        val = int(max(30, min(250, val)))
        self.metro.bpm = val
        self.bpm_str.set(str(val))
        self._updating = True
        self.bpm_slider.set(val)
        self._updating = False

    def _bpm_nudge(self, d):
        cur = int(self.bpm_str.get()) if self.bpm_str.get().isdigit() else 120
        self._set_bpm(cur + d)

    def _bpm_from_slider(self, v):
        val = int(max(30, min(250, round(float(v)))))
        self.metro.bpm = val
        self.bpm_str.set(str(val))

    def _tap(self):
        if not hasattr(self, '_taps'): self._taps = []
        now = time.perf_counter()
        if self._taps and now - self._taps[-1] > 2.5: self._taps.clear()
        self._taps.append(now)
        if len(self._taps) > 5: self._taps.pop(0)
        if len(self._taps) >= 2:
            bpm = round(60.0 / float(np.mean(np.diff(self._taps))))
            self._set_bpm(bpm)
            self.tap_lbl.configure(text=f'≈ {bpm} BPM')
        else:
            self.tap_lbl.configure(text='tap again…')

    def _beats_changed(self):
        try: n = max(1, min(12, int(self.beats_var.get())))
        except: return
        self.metro.bpm_m = n; self._rebuild_dots()

    def _subdiv_changed(self):
        self.metro.subdiv = {'Quarter notes':1,'Eighth notes':2,
                              'Triplets':3,'Sixteenth notes':4}.get(self.subdiv_var.get(), 1)

    def _preview(self):
        if sd is None: return
        k = self.metro.sounds; g = self.metro.volume
        wave = np.concatenate([k['accent']*g,
                               np.zeros(int(SAMPLE_RATE*.15),dtype=np.float32),
                               k['main']*g,
                               np.zeros(int(SAMPLE_RATE*.12),dtype=np.float32),
                               k['sub']*g])
        sd.stop(); sd.play(wave, SAMPLE_RATE)

    def _rebuild_dots(self):
        self.dot_canvas.delete('all'); self.dots = []
        n = self.metro.bpm_m
        self.update_idletasks()
        w = self.dot_canvas.winfo_width() or 640
        r, m = 16, 40
        sp = (w-2*m)/max(1,n-1) if n>1 else 0
        for i in range(n):
            x = m + (sp*i if n>1 else (w-2*m)/2)
            col = RED if i==0 else '#3a4048'
            self.dots.append(self.dot_canvas.create_oval(x-r,29,x+r,61,fill=col,outline=''))

    def _flash(self, bi):
        if bi < len(self.dots):
            for i,d in enumerate(self.dots):
                self.dot_canvas.itemconfig(d, fill=RED if i==0 else '#3a4048')
            self.dot_canvas.itemconfig(self.dots[bi], fill='#ffd166' if bi==0 else ACCENT)
        # Persistent beat light in the footer — updates regardless of
        # which tab is showing, and regardless of the Metronome tab's own
        # dot strip existing, so it works as a silent visual click track.
        self.beat_light.itemconfig(self._beat_light_oval,
                                    fill='#ffd166' if bi==0 else ACCENT)
        self.after(90, self._dim_beat_light)

    def _dim_beat_light(self):
        self.beat_light.itemconfig(self._beat_light_oval, fill='#2c333b')

    def _toggle_metro(self):
        if self.metro.running:
            self.metro.stop(); self.start_btn.configure(text='▶  START')
        else:
            self.metro.start(); self.start_btn.configure(text='■  STOP')

    def _poll_beats(self):
        try:
            while True: self._flash(self.beat_q.get_nowait())
        except queue.Empty: pass
        self.after(20, self._poll_beats)

    # ── footer tuner ─────────────────────────────────────────

    def _nudge_a4(self, d):
        v = max(430, min(450, int(round(self.a4.get())) + d))
        self.a4.set(v); self.a4_lbl.configure(text=str(v))

    def _toggle_pitch_pipe(self, idx):
        """Pitch pipe notes now toggle: clicking the note that's already
        sounding stops it (the reported bug — clicking a note again used to
        just restart the same tone, with no way to tell it was still
        playing or to silence it short of the separate Stop tone button)."""
        if self.pp_playing_idx == idx:
            self.tone.stop()
            self.pp_playing_idx = None
        else:
            freq = self.a4.get() * (2 ** ((idx - 9) / 12.0))
            self.tone.play(freq)
            self.pp_playing_idx = idx
        self._refresh_pitch_pipe_buttons()

    def _stop_pitch_pipe(self):
        self.tone.stop()
        self.pp_playing_idx = None
        self._refresh_pitch_pipe_buttons()

    def _refresh_pitch_pipe_buttons(self):
        for i, b in enumerate(self.pp_buttons):
            b.configure(bg=(ACCENT if i == self.pp_playing_idx else '#1e252d'),
                        fg=(BG if i == self.pp_playing_idx else FG))

    def _show_about(self):
        recent = '\n'.join(
            f'  [{cat:8s}] build {b:>3d}  {date}\n    {desc[:72]}'
            for b, date, cat, desc in CHANGELOG[:8]
        )
        messagebox.showinfo(
            f'SoloTone  v{VERSION_FULL}',
            f'SoloTone — Guitar Amp · Pedals · Looper · Tuner · Metronome\n'
            f'Version {VERSION_FULL}\n\n'
            f'Recent changes:\n{recent}\n\n'
            f'MIT License — Jason S. Gagnon  2026'
        )

    def _toggle_tuner(self):
        if self.tuner.running:
            self.tuner.stop(); self.tuner_btn.configure(text='🎤 START')
            self.foot_status.configure(text='Tuner stopped')
        else:
            if sd is None: messagebox.showerror('No audio','PortAudio not available.'); return
            try: self.tuner.start()
            except Exception as e: messagebox.showerror('Mic error', str(e)); return
            self.tuner_btn.configure(text='■ STOP')
            self.foot_status.configure(text='Listening…')

    def _draw_gauge(self, cents):
        c = self.gauge_c; c.delete('all')
        w = c.winfo_width() or 120; h = c.winfo_height() or 72
        cx, cy = w//2, h-4
        r = min(cx-6, cy+20)
        c.create_arc(cx-r,cy-r,cx+r,cy+r, start=20,extent=140,
                     style='arc', outline='#2c333b', width=8)
        c.create_arc(cx-r,cy-r,cx+r,cy+r, start=85,extent=10,
                     style='arc', outline=ACCENT, width=8)
        cc = max(-50, min(50, cents or 0))
        ang = math.radians(90 - (cc/50)*70)
        col = ACCENT if abs(cc)<5 else ('#ffd166' if abs(cc)<20 else RED)
        c.create_line(cx, cy, cx+(r-6)*math.cos(ang), cy-(r-6)*math.sin(ang),
                      fill=col, width=3, capstyle='round')
        c.create_oval(cx-4,cy-4,cx+4,cy+4, fill=col, outline='')

    def _apply_instrument(self, is_bass):
        """Point the tuning dropdown at the right preset list, and retune
        the autocorrelation pitch detector for bass's much lower range.
        Three changes, not just the obvious one: lowering fmin to 24 Hz
        (a 5-string low B is ~30.9 Hz) turned out not to be enough on its
        own — measured directly, a low B synthesized with realistic
        harmonic content still failed to detect at the default fmax of
        1500 Hz, because autocorrelation stays highly self-correlated at
        very short lags for a signal this much slower than fmax allows,
        and that near-zero-lag plateau outscored the true-period peak.
        Lowering fmax to 600 Hz (still comfortably above every open bass
        string, up to several frets fretted) shrinks the search window
        past that plateau and fixed it in testing. The analysis window
        also needs lengthening to match the lower floor — at the
        guitar-sized buffer, that low B gets under 7 full cycles, thinner
        margin than the ~10-15 cycles the buffer size was originally
        chosen to give lower guitar strings. Returns the tuning dict now
        in effect."""
        tunings = BASS_TUNINGS if is_bass else GUITAR_TUNINGS
        self.tuning_cb.configure(values=list(tunings.keys()))
        self.tuner.fmin = 24.0 if is_bass else 40.0
        self.tuner.fmax = 600.0 if is_bass else 1500.0
        new_buf_len = 16384 if is_bass else 12288
        if len(self.tuner.buf) != new_buf_len:
            with self.tuner.lock:
                self.tuner.buf = np.zeros(new_buf_len, dtype=np.float32)
                self.tuner._hist.clear()
        return tunings

    def _instrument_changed(self):
        is_bass = self.instrument_var.get() == 'bass'
        tunings = self._apply_instrument(is_bass)
        if self.tuning_var.get() not in tunings:
            self.tuning_var.set(next(iter(tunings)))
        self._reset_string_highlights()

    def _restore_tuning(self, name):
        """Set the tuning preset (from a session or profile), switching
        the Guitar/Bass instrument toggle to match first if needed — a key
        present in both dicts (just 'None (chromatic)' today) is treated
        as guitar, since that's the one that existed before bass tunings
        did."""
        if name not in ALL_TUNINGS:
            return
        is_bass = name in BASS_TUNINGS and name not in GUITAR_TUNINGS
        self.instrument_var.set('bass' if is_bass else 'guitar')
        self._apply_instrument(is_bass)
        self.tuning_var.set(name)
        self._reset_string_highlights()

    def _reset_string_highlights(self):
        """Relabel the string boxes from the current tuning preset, showing
        only as many boxes as that tuning has strings — 4 for bass, 5 for
        5-string bass, 6 for guitar — instead of always all six. Every box
        is unpacked and only the needed ones re-packed in order, so the
        left-to-right order stays correct no matter which tunings (and
        string counts) were selected before this one."""
        tuning_name = self.tuning_var.get()
        tuning = ALL_TUNINGS.get(tuning_name)
        use_flats = tuning_name in GUITAR_TUNING_USE_FLATS
        for box in self.str_boxes:
            box.pack_forget()
        if tuning:
            n = len(tuning)
            str_num = [str(n - i) for i in range(n)]
            for i in range(n):
                box, lbl = self.str_boxes[i], self.str_labels[i]
                note, octave = tuning[i]
                if use_flats:
                    note = NOTE_NAMES_FLAT[NOTE_NAMES.index(note)]
                lbl.configure(text=f"{str_num[i]}\n{note}{octave}",
                               fg=DIM, bg='#1a2128')
                box.configure(bg='#1a2128', highlightbackground='#2c333b')
                box.pack(side='left', padx=2)
        self.str_cents_lbl.configure(text='')

    def _update_string_readout(self, freq):
        """Highlight the closest string in the readout and show cents-to-target."""
        tuning_name = self.tuning_var.get()
        tuning_midi = GUITAR_TUNING_MIDI.get(tuning_name)
        # Reset all boxes
        for box, lbl in zip(self.str_boxes, self.str_labels):
            box.configure(bg='#1a2128', highlightbackground='#2c333b')
            lbl.configure(bg='#1a2128', fg=DIM)

        if tuning_midi is None or freq is None:
            self.str_cents_lbl.configure(text='')
            return

        str_idx, cents = closest_string(freq, tuning_midi, a4=self.a4.get())
        if str_idx is None:
            self.str_cents_lbl.configure(text='')
            return

        # Colour: green if within 5c, yellow within 20c, red otherwise
        if abs(cents) < 5:
            bg_col = ACCENT_DIM; hl_col = ACCENT; fg_col = ACCENT
        elif abs(cents) < 20:
            bg_col = '#332800'; hl_col = '#ffd166'; fg_col = '#ffd166'
        else:
            bg_col = '#2d1010'; hl_col = RED; fg_col = RED

        box = self.str_boxes[str_idx]
        lbl = self.str_labels[str_idx]
        box.configure(bg=bg_col, highlightbackground=hl_col)
        lbl.configure(bg=bg_col, fg=fg_col)

        dir_arrow = '▲' if cents > 0 else ('▼' if cents < 0 else '')
        self.str_cents_lbl.configure(
            text=f"str {6-str_idx}  {cents:+.0f}¢ {dir_arrow}",
            fg=fg_col)

    def _poll_tuner(self):
        if self.tuner.running:
            tr   = TRANSPOSITIONS.get(self.trans_var.get(), 0)
            freq = self.tuner.pitch()
            r    = analyze_pitch(freq, a4=self.a4.get(), transpose=tr)
            if r:
                self.foot_note.configure(text=f"{r['name']}{r['octave']}")
                self.foot_freq.configure(text=f"{r['freq']:.1f} Hz")
                self.foot_cents.configure(text=f"{r['cents']:+.0f}¢")
                self._draw_gauge(r['cents'])
                if abs(r['cents']) < 5:
                    self.foot_status.configure(text='In tune ✓', fg=ACCENT)
                else:
                    self.foot_status.configure(
                        text='sharp ▲' if r['cents']>0 else 'flat ▼', fg=DIM)
                self._update_string_readout(freq)
            else:
                self.foot_status.configure(text='Listening… (play a note)', fg=DIM)
                self._update_string_readout(None)
        self.after(90, self._poll_tuner)

    # ── session persistence ───────────────────────────────────

    def _load_session(self):
        try:
            with open(SESSION_FILE) as f:
                s = json.load(f)
            if 'bpm' in s:       self._set_bpm(s['bpm'])
            if 'beats' in s:     self.beats_var.set(s['beats']); self._beats_changed()
            if 'subdiv' in s:    self.subdiv_var.set(s['subdiv']); self._subdiv_changed()
            if 'sound' in s:     self.sound_var.set(s['sound']); self.metro.set_kit(s['sound'])
            if 'metro_mute' in s:
                self.metro_mute_var.set(bool(s['metro_mute']))
                self.metro.mute = self.metro_mute_var.get()
            if 'a4' in s:        self.a4.set(s['a4']); self.a4_lbl.configure(text=str(int(s['a4'])))
            if 'transpose' in s: self.trans_var.set(s['transpose'])
            if 'loop_len' in s:  self.loop_len_var.set(s['loop_len'])
            if 'loop_layers' in s: self.loop_layers_var.set(s['loop_layers'])
            if 'loop_mode' in s: self.loop_mode_var.set(s['loop_mode'])
            if 'tuning' in s: self._restore_tuning(s['tuning'])
            if 'pedal_order' in s:
                order = s['pedal_order']
                if (isinstance(order, list)
                        and sorted(order) == sorted(self.amp.pedals.order)):
                    self.amp.pedals.order = order
                    self._relayout_pedals()
            if 'ir_labels' in s:
                for i, lbl in enumerate(s['ir_labels'][:MAX_IR_SLOTS]):
                    self.ir_label_vars[i].set(lbl)
            if 'blocksize' in s and str(s['blocksize']) in ('256','512','1024','2048'):
                self.bs_var.set(str(s['blocksize']))
            self._restore_saved_devices(s)
            if 'keybind_mappings' in s and isinstance(s['keybind_mappings'], dict):
                for keysym, tid in s['keybind_mappings'].items():
                    if tid in KEYBIND_TARGETS_BY_ID:
                        self.keybind_mappings[keysym] = tid
                self._refresh_keybind_ui()
            if 'nudge_step_pct' in s:
                fv = float(s['nudge_step_pct'])
                self.nudge_step_var.set(fv); self.nudge_step_lbl.set(f'{fv:.0f}%')
                self._updating = True; self.nudge_step_sl.set(fv); self._updating = False
            if 'always_on_top' in s:
                self.always_on_top_var.set(bool(s['always_on_top']))
                self.attributes('-topmost', self.always_on_top_var.get())
            if 'confirm_quit' in s: self.confirm_quit_var.set(bool(s['confirm_quit']))
            if 'lpd8_led_on' in s and hasattr(self, 'lpd8_led_var'):
                self.lpd8_led_var.set(bool(s['lpd8_led_on']))
                self._lpd8_refresh_leds(force=True)
            if 'last_nam_dir' in s and s['last_nam_dir'] and pathlib.Path(s['last_nam_dir']).is_dir():
                self._last_nam_dir = s['last_nam_dir']
            if 'last_ir_dir' in s and s['last_ir_dir'] and pathlib.Path(s['last_ir_dir']).is_dir():
                self._last_ir_dir = s['last_ir_dir']
            # Restore NAM and IR paths — done after UI is ready via after()
            self._pending_session = s
        except Exception:
            self._pending_session = {}
            pass  # no session file yet, or corrupt — just use defaults

    def _restore_file_paths(self):
        """Called once after startup to reload NAM and IR files from the last session.
        Uses silent=True so missing files produce a single summary warning rather
        than one dialog per file."""
        s = getattr(self, '_pending_session', {})
        if not s: return
        missing = []

        # NAM
        nam_path = s.get('nam_path')
        if nam_path:
            if pathlib.Path(nam_path).exists():
                self._load_nam_from_path(nam_path, silent=True)
            else:
                missing.append(f'NAM: {nam_path}')

        # IRs
        ir_paths = s.get('ir_paths', [None]*MAX_IR_SLOTS)
        for idx, path in enumerate(ir_paths[:MAX_IR_SLOTS]):
            if path:
                if pathlib.Path(path).exists():
                    self._load_ir_from_path(idx, path, silent=True)
                else:
                    missing.append(f'IR slot {idx+1}: {path}')

        # Restore active IR selection (after slots are loaded)
        ir_active = s.get('ir_active', -1)
        if isinstance(ir_active, int) and -1 <= ir_active < MAX_IR_SLOTS:
            self.amp.ir_active = ir_active
            self.ir_active_var.set(ir_active)

        # Single warning for all missing files
        if missing:
            missing_str = '\n'.join(f'  • {m}' for m in missing)
            messagebox.showwarning(
                'Files not found',
                f'The following files from your last session could not be found:\n\n'
                f'{missing_str}\n\n'
                f'They may have been moved or deleted. Load them manually from the Signal Chain tab.'
            )

    def _save_session(self):
        try:
            s = {
                'bpm':         self.metro.bpm,
                'beats':       self.metro.bpm_m,
                'subdiv':      self.subdiv_var.get(),
                'sound':       self.sound_var.get(),
                'metro_mute':  self.metro_mute_var.get(),
                'a4':          self.a4.get(),
                'transpose':   self.trans_var.get(),
                'loop_len':    self.loop_len_var.get(),
                'loop_layers': self.loop_layers_var.get(),
                'loop_mode':   self.loop_mode_var.get(),
                'ir_labels':   [v.get() for v in self.ir_label_vars],
                'blocksize':   self.bs_var.get(),
                'tuning':      self.tuning_var.get(),
                'pedal_mute':  self.amp.pedals.mute,
                'pedal_order': self.amp.pedals.order,
                'dly_link':    self.dly_link_var.get(),
                'dly_note':    self.dly_note_var.get(),
                'muff_tone':   self.muff_tone_var.get(),
                'nam_path':    self._current_nam_path,
                'ir_paths':    self._current_ir_paths,
                'ir_active':   self.amp.ir_active,
                'last_nam_dir': self._last_nam_dir,
                'last_ir_dir':  self._last_ir_dir,
                'keybind_mappings': dict(self.keybind_mappings),
                'nudge_step_pct':   self.nudge_step_var.get(),
                'section_collapsed': self._section_collapsed,
                'theme_accent':  ACCENT,
                'theme_bg':      BG,
                'always_on_top': self.always_on_top_var.get(),
                'confirm_quit':  self.confirm_quit_var.get(),
                'lpd8_led_on':   (self.lpd8_led_var.get() if hasattr(self, 'lpd8_led_var') else False),
                'hostapi':       self.hostapi_var.get(),
                'amp_input':     self.amp_in_var.get(),
                'amp_output':    self.amp_out_var.get(),
                'tuner_input':   self.t_dev_var.get(),
            }
            with open(SESSION_FILE, 'w') as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass

    # ── close ────────────────────────────────────────────────


    # ── TAB: Pedals ──────────────────────────────────────────

    def _tab_pedals(self, root):
        inner = self._scrollable(root)
        self._pedals_canvas = inner.master   # the Canvas _scrollable() built inner on top of — see _jump_to_pedal_section()
        self._fx_outer = {}   # name -> outer section frame, for re-packing on reorder
        self._fx_rows  = {}   # name -> {pos_var, up, down}, for the reorder row
        # attr name -> {'kind': 'slider'|'radio', 'widget', 'val_var', 'fmt'}
        # so _pedal_widget_sync() can move the right on-screen control when
        # a pedal parameter changes from something other than this tab's
        # own widgets — MIDI CC, or a long-press mode-cycle.
        self._pedal_widgets = {}
        # code -> on/off BooleanVar, so _update_pedal_led() can read the
        # var directly instead of amp.pedals.<code>_on — a MIDI toggle
        # action calls var.set() *before* it updates the pedal attribute
        # (see _midi_toggle_action), so reading the attribute from inside
        # the var's own write-trace would see the stale, pre-toggle value.
        self._pedal_on_vars = {}
        # Every default below reads the pedal's current live value (rather
        # than a hardcoded literal) so this tab reflects the real state
        # both on first build and after being rebuilt post-profile-load.
        p = self.amp.pedals

        def on_off(sec_frame, attr, label_text):
            """Returns (header_frame, on_var) for a pedal section. Default
            reflects the pedal's current state rather than a hardcoded
            False, so rebuilding this tab after a profile load shows the
            real state instead of resetting the display to off."""
            h = tk.Frame(sec_frame, bg=PANEL); h.pack(fill='x', padx=10, pady=(6,2))
            v = tk.BooleanVar(value=bool(getattr(self.amp.pedals, attr)))
            def _tog():
                setattr(self.amp.pedals, attr, v.get())
            ttk.Checkbutton(h, text=label_text, variable=v, command=_tog).pack(side='left')
            # Tracing the Variable itself (rather than only this checkbox's
            # own command) catches every way it can change — a MIDI pad or
            # a keybind toggling the pedal both call var.set() on this same
            # object — so the status-strip LED above stays in sync no
            # matter which input method flipped the pedal.
            if attr.endswith('_on'):
                code = attr[:-3]
                self._pedal_on_vars[code] = v
                v.trace_add('write', lambda *_a, code=code: self._update_pedal_led(code))
            return h, v

        def _sl(parent, label, lo, hi, default, cb, fmt=None, length=220, attr=None):
            row = tk.Frame(parent, bg=PANEL); row.pack(fill='x', padx=10, pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=FG, font=(FF,10), width=18, anchor='w').pack(side='left')
            vv = tk.StringVar(value=(fmt(default) if fmt else f'{default:.2f}'))
            def _cmd(v):
                fv = float(v)
                vv.set(fmt(fv) if fmt else f'{fv:.2f}')
                cb(fv)
            sl = self._scale(row, lo, hi, _cmd, default, length=length)
            sl.pack(side='left', padx=6)
            tk.Label(row, textvariable=vv, bg=PANEL, fg=DIM, font=(FF,10), width=8).pack(side='left')
            if attr:
                self._pedal_widgets[attr] = dict(kind='slider', widget=sl, val_var=vv, fmt=fmt)
            return sl

        def _radio_row(parent, label, options, attr, default):
            row = tk.Frame(parent, bg=PANEL); row.pack(fill='x', padx=10, pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=FG, font=(FF,10), width=18, anchor='w').pack(side='left')
            v = tk.StringVar(value=default)
            for val, txt in options:
                tk.Radiobutton(row, text=txt, variable=v, value=val,
                               bg=PANEL, fg=FG, selectcolor='#2a3540',
                               activebackground=PANEL,
                               command=lambda a=attr, vv=v: setattr(self.amp.pedals, a, vv.get())
                               ).pack(side='left', padx=6)
            self._pedal_widgets[attr] = dict(kind='radio', widget=v)
            return v

        # ── Mute ─────────────────────────────────────────────
        mute_sec = self._sec(inner, 'Tuner Mute')
        mh = tk.Frame(mute_sec, bg=PANEL); mh.pack(fill='x', padx=10, pady=6)
        self.mute_var = tk.BooleanVar(value=p.mute)
        tk.Checkbutton(mh, text='MUTE  (silence amp input — tune up without the amp hearing you)',
                       variable=self.mute_var, bg=PANEL, fg=FG,
                       selectcolor='#2a3540', activebackground=PANEL,
                       font=(FF,11,'bold'),
                       command=lambda: setattr(self.amp.pedals, 'mute', self.mute_var.get())
                       ).pack(side='left')

        # ── Chain order ──────────────────────────────────────
        order_sec = self._sec(inner, 'Chain Order')
        order_row = tk.Frame(order_sec, bg=PANEL); order_row.pack(fill='x', padx=10, pady=6)
        ttk.Button(order_row, text='Reset to Default Order',
                   command=self._reset_pedal_order).pack(side='left')
        tk.Label(order_row,
                 text=f'Default: {" → ".join(DEFAULT_PEDAL_ORDER)}',
                 bg=PANEL, fg=DIM, font=(FF,9)).pack(side='left', padx=10)

        # ── Pedal status strip ────────────────────────────────
        self._build_pedal_status_strip(inner)

        # ── Compressor (tube-style) ────────────────────────────
        comp_sec = self._sec(inner, PEDAL_SECTION_TITLES['comp'],
                             group=PEDALS_ACCORDION_GROUP, default_collapsed=False)
        self._fx_outer['comp'] = comp_sec.master
        self._fx_reorder_row(comp_sec, 'comp')
        self._build_pedal_preset_row(comp_sec, 'comp')
        _, self.comp_var = on_off(comp_sec, 'comp_on', 'Compressor  Enabled')
        _sl(comp_sec, 'Threshold (dB)', -40.0, 0.0, p.comp_thresh,
            lambda v: setattr(self.amp.pedals, 'comp_thresh', v),
            fmt=lambda v: f'{v:.0f} dB', attr='comp_thresh')
        _sl(comp_sec, 'Ratio', 1.0, 20.0, p.comp_ratio,
            lambda v: setattr(self.amp.pedals, 'comp_ratio', v),
            fmt=lambda v: f'{v:.1f}:1', attr='comp_ratio')
        _sl(comp_sec, 'Attack (ms)', 0.5, 50.0, p.comp_attack,
            lambda v: setattr(self.amp.pedals, 'comp_attack', v),
            fmt=lambda v: f'{v:.1f} ms', attr='comp_attack')
        _sl(comp_sec, 'Release (ms)', 20.0, 500.0, p.comp_release,
            lambda v: setattr(self.amp.pedals, 'comp_release', v),
            fmt=lambda v: f'{v:.0f} ms', attr='comp_release')
        _sl(comp_sec, 'Makeup gain (dB)', 0.0, 24.0, p.comp_makeup,
            lambda v: setattr(self.amp.pedals, 'comp_makeup', v),
            fmt=lambda v: f'{v:+.1f} dB', attr='comp_makeup')
        _sl(comp_sec, 'Warmth', 0.0, 1.0, p.comp_warmth,
            lambda v: setattr(self.amp.pedals, 'comp_warmth', v),
            fmt=lambda v: f'{int(v*100)}%', attr='comp_warmth')

        # ── Wah ──────────────────────────────────────────────
        wah_sec = self._sec(inner, PEDAL_SECTION_TITLES['wah'],
                            group=PEDALS_ACCORDION_GROUP, default_collapsed=True)
        self._fx_outer['wah'] = wah_sec.master
        self._fx_reorder_row(wah_sec, 'wah')
        self._build_pedal_preset_row(wah_sec, 'wah')
        _, self.wah_var = on_off(wah_sec, 'wah_on', 'Wah  Enabled')

        mode_row = tk.Frame(wah_sec, bg=PANEL); mode_row.pack(fill='x', padx=10, pady=3)
        tk.Label(mode_row, text='Mode', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.wah_mode_var = tk.StringVar(value=p.wah_mode)
        self._pedal_widgets['wah_mode'] = dict(kind='radio', widget=self.wah_mode_var)
        def _tog_wah_mode():
            self.amp.pedals.wah_mode = self.wah_mode_var.get()
            self._wah_refresh_ui()
        for val, txt in [('manual','Manual'),('auto','Auto sweep')]:
            tk.Radiobutton(mode_row, text=txt, variable=self.wah_mode_var, value=val,
                           bg=PANEL, fg=FG, selectcolor='#2a3540',
                           activebackground=PANEL, command=_tog_wah_mode
                           ).pack(side='left', padx=6)

        # Range — replaces a single Frequency slider with low/high bounds
        # shared by both the manual position and the auto sweep.
        range_row = tk.Frame(wah_sec, bg=PANEL); range_row.pack(fill='x', padx=10, pady=3)
        tk.Label(range_row, text='Range (Hz)', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.wah_lo_var = tk.StringVar(value=f'{int(p.wah_range_lo)} Hz')
        def _wah_lo_cb(v):
            fv = float(v)
            self.amp.pedals.wah_range_lo = fv
            self.wah_lo_var.set(f'{int(fv)} Hz')
        self._scale(range_row, 100, 2500, _wah_lo_cb, p.wah_range_lo, length=100).pack(side='left', padx=(6,2))
        tk.Label(range_row, textvariable=self.wah_lo_var, bg=PANEL, fg=DIM,
                 font=(FF,9), width=7).pack(side='left')
        tk.Label(range_row, text='to', bg=PANEL, fg=DIM, font=(FF,9)).pack(side='left', padx=4)
        self.wah_hi_var = tk.StringVar(value=f'{int(p.wah_range_hi)} Hz')
        def _wah_hi_cb(v):
            fv = float(v)
            self.amp.pedals.wah_range_hi = fv
            self.wah_hi_var.set(f'{int(fv)} Hz')
        self._scale(range_row, 500, 3500, _wah_hi_cb, p.wah_range_hi, length=100).pack(side='left', padx=(6,2))
        tk.Label(range_row, textvariable=self.wah_hi_var, bg=PANEL, fg=DIM,
                 font=(FF,9), width=7).pack(side='left')
        tk.Label(wah_sec,
                 text='Defaults are guitar-voiced — bass players typically want a lower sweep, e.g. 150–800 Hz',
                 bg=PANEL, fg=DIM, font=(FF,8)).pack(anchor='w', padx=10, pady=(0,2))

        # Manual-mode position — also driven by the Left/Right arrow
        # keybinds configured on the Settings tab.
        self.wah_manual_frame = tk.Frame(wah_sec, bg=PANEL)
        self.wah_manual_frame.pack(fill='x', padx=10, pady=2)
        tk.Label(self.wah_manual_frame, text='Position', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.wah_pos_var = tk.StringVar(value=f'{int(p.wah_pos*100)}%')
        def _wah_pos_cb(v):
            fv = float(v)
            self.amp.pedals.wah_pos = fv
            self.wah_pos_var.set(f'{int(fv*100)}%')
        self.wah_pos_sl = self._scale(self.wah_manual_frame, 0.0, 1.0, _wah_pos_cb, p.wah_pos, length=220)
        self.wah_pos_sl.pack(side='left', padx=6)
        self._pedal_widgets['wah_pos'] = dict(kind='slider', widget=self.wah_pos_sl,
                                               val_var=self.wah_pos_var,
                                               fmt=lambda v: f'{int(v*100)}%')
        tk.Label(self.wah_manual_frame, textvariable=self.wah_pos_var, bg=PANEL, fg=DIM,
                 font=(FF,10), width=8).pack(side='left')
        tk.Label(self.wah_manual_frame, text='◄ ► keys (see Settings tab)',
                 bg=PANEL, fg=DIM, font=(FF,8)).pack(side='left', padx=6)

        # Auto-sweep rate / intensity
        self.wah_auto_frame = tk.Frame(wah_sec, bg=PANEL)
        self.wah_auto_frame.pack(fill='x', padx=0, pady=0)
        _sl(self.wah_auto_frame, 'Sweep rate (Hz)', 0.05, 8.0, p.wah_auto_rate,
            lambda v: setattr(self.amp.pedals, 'wah_auto_rate', v),
            fmt=lambda v: f'{v:.2f} Hz', attr='wah_auto_rate')
        _sl(self.wah_auto_frame, 'Sweep intensity', 0.0, 1.0, p.wah_auto_depth,
            lambda v: setattr(self.amp.pedals, 'wah_auto_depth', v),
            fmt=lambda v: f'{int(v*100)}%', attr='wah_auto_depth')

        _sl(wah_sec, 'Resonance (Q)', 1.0, 8.0, p.wah_q,
            lambda v: setattr(self.amp.pedals, 'wah_q', v), attr='wah_q')

        self._wah_refresh_ui()

        # ── Fuzz ─────────────────────────────────────────────
        fuzz_sec = self._sec(inner, PEDAL_SECTION_TITLES['fuzz'],
                             group=PEDALS_ACCORDION_GROUP, default_collapsed=True)
        self._fx_outer['fuzz'] = fuzz_sec.master
        self._fx_reorder_row(fuzz_sec, 'fuzz')
        self._build_pedal_preset_row(fuzz_sec, 'fuzz')
        _, self.fuzz_var = on_off(fuzz_sec, 'fuzz_on', 'Fuzz  Enabled')
        _radio_row(fuzz_sec, 'Type', [('fuzz_face','Fuzz Face'),('big_muff','Big Muff')],
                   'fuzz_mode', p.fuzz_mode)
        _sl(fuzz_sec, 'Drive', 1.0, 10.0, p.fuzz_drive,
            lambda v: setattr(self.amp.pedals, 'fuzz_drive', v), attr='fuzz_drive')
        _sl(fuzz_sec, 'Volume', 0.1, 1.0, p.fuzz_vol,
            lambda v: setattr(self.amp.pedals, 'fuzz_vol', v),
            fmt=lambda v: f'{int(v*100)}%', attr='fuzz_vol')

        # Big Muff tone toggle
        muff_row = tk.Frame(fuzz_sec, bg=PANEL); muff_row.pack(fill='x', padx=10, pady=3)
        tk.Label(muff_row, text='Big Muff Tone', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.muff_tone_var = tk.StringVar(value=p.muff_tone)
        for val, txt, tip in [
            ('full',  'Full',  'Tone wide open — bright, cutting'),
            ('flat',  'Flat',  'Tone centred — even response'),
            ('scoop', 'Scoop', 'Mid scoop — massive wall-of-sound'),
        ]:
            fr = tk.Frame(muff_row, bg=PANEL); fr.pack(side='left', padx=4)
            tk.Radiobutton(fr, text=txt, variable=self.muff_tone_var, value=val,
                           bg=PANEL, fg=FG, selectcolor='#2a3540', activebackground=PANEL,
                           command=lambda: setattr(self.amp.pedals, 'muff_tone',
                                                   self.muff_tone_var.get())
                           ).pack(anchor='w')
            tk.Label(fr, text=tip, bg=PANEL, fg=DIM, font=(FF,8)).pack(anchor='w')

        # ── Distortion ─────────────────────────────────────────
        dist_sec = self._sec(inner, PEDAL_SECTION_TITLES['dist'],
                             group=PEDALS_ACCORDION_GROUP, default_collapsed=True)
        self._fx_outer['dist'] = dist_sec.master
        self._fx_reorder_row(dist_sec, 'dist')
        self._build_pedal_preset_row(dist_sec, 'dist')
        _, self.dist_var = on_off(dist_sec, 'dist_on', 'Distortion  Enabled')
        _radio_row(dist_sec, 'Type',
                   [('ds1','Boss DS-1'),('rat','ProCo RAT'),
                    ('dist_plus','MXR Dist+'),('metal','Metal')],
                   'dist_mode', p.dist_mode)
        _sl(dist_sec, 'Drive', 1.0, 10.0, p.dist_drive,
            lambda v: setattr(self.amp.pedals, 'dist_drive', v), attr='dist_drive')
        _sl(dist_sec, 'Tone', 0.0, 1.0, p.dist_tone,
            lambda v: setattr(self.amp.pedals, 'dist_tone', v),
            fmt=lambda v: f'{int(v*100)}%', attr='dist_tone')
        _sl(dist_sec, 'Volume', 0.1, 1.0, p.dist_vol,
            lambda v: setattr(self.amp.pedals, 'dist_vol', v),
            fmt=lambda v: f'{int(v*100)}%', attr='dist_vol')

        # ── Overdrive / Boost ─────────────────────────────────
        od_sec = self._sec(inner, PEDAL_SECTION_TITLES['od'],
                           group=PEDALS_ACCORDION_GROUP, default_collapsed=True)
        self._fx_outer['od'] = od_sec.master
        self._fx_reorder_row(od_sec, 'od')
        self._build_pedal_preset_row(od_sec, 'od')
        _, self.od_var = on_off(od_sec, 'od_on', 'OD / Boost  Enabled')
        _radio_row(od_sec, 'Type',
                   [('tubescreamer','Tubescreamer'),('klon','Klon (transparent)'),
                    ('boost','Clean Boost')],
                   'od_mode', p.od_mode)
        _sl(od_sec, 'Drive', 1.0, 10.0, p.od_drive,
            lambda v: setattr(self.amp.pedals, 'od_drive', v), attr='od_drive')
        _sl(od_sec, 'Volume', 0.1, 1.0, p.od_vol,
            lambda v: setattr(self.amp.pedals, 'od_vol', v),
            fmt=lambda v: f'{int(v*100)}%', attr='od_vol')

        # ── Chorus / Flanger ─────────────────────────────────
        ch_sec = self._sec(inner, PEDAL_SECTION_TITLES['chorus'],
                           group=PEDALS_ACCORDION_GROUP, default_collapsed=True)
        self._fx_outer['chorus'] = ch_sec.master
        self._fx_reorder_row(ch_sec, 'chorus')
        self._build_pedal_preset_row(ch_sec, 'chorus')
        _, self.ch_var = on_off(ch_sec, 'chorus_on', 'Chorus / Flanger  Enabled')
        _radio_row(ch_sec, 'Type',
                   [('chorus','Chorus'),('flanger','Flanger')],
                   'chorus_mode', p.chorus_mode)
        _sl(ch_sec, 'Rate (Hz)', 0.1, 5.0, p.chorus_rate,
            lambda v: setattr(self.amp.pedals, 'chorus_rate', v),
            fmt=lambda v: f'{v:.2f} Hz', attr='chorus_rate')
        _sl(ch_sec, 'Depth', 0.0, 1.0, p.chorus_depth,
            lambda v: setattr(self.amp.pedals, 'chorus_depth', v),
            fmt=lambda v: f'{int(v*100)}%', attr='chorus_depth')
        _sl(ch_sec, 'Mix', 0.0, 1.0, p.chorus_mix,
            lambda v: setattr(self.amp.pedals, 'chorus_mix', v),
            fmt=lambda v: f'{int(v*100)}%', attr='chorus_mix')

        # ── Delay ─────────────────────────────────────────────
        dly_sec = self._sec(inner, PEDAL_SECTION_TITLES['delay'],
                            group=PEDALS_ACCORDION_GROUP, default_collapsed=True)
        self._fx_outer['delay'] = dly_sec.master
        self._fx_reorder_row(dly_sec, 'delay')
        self._build_pedal_preset_row(dly_sec, 'delay')
        _, self.dly_var = on_off(dly_sec, 'delay_on', 'Delay  Enabled')
        _radio_row(dly_sec, 'Mode', [('tape','Tape'),('digital','Digital')],
                   'delay_mode', p.delay_mode)

        # BPM link controls
        link_row = tk.Frame(dly_sec, bg=PANEL); link_row.pack(fill='x', padx=10, pady=3)
        tk.Label(link_row, text='BPM Link', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.dly_link_var = tk.BooleanVar(value=p.delay_bpm_link)
        def _tog_link():
            self.amp.pedals.delay_bpm_link = self.dly_link_var.get()
            self._dly_refresh_ui()
        ttk.Checkbutton(link_row, text='Sync to metronome BPM',
                        variable=self.dly_link_var, command=_tog_link
                        ).pack(side='left', padx=6)

        # note value (visible when linked)
        self.dly_note_frame = tk.Frame(dly_sec, bg=PANEL)
        self.dly_note_frame.pack(fill='x', padx=10, pady=2)
        tk.Label(self.dly_note_frame, text='Note value', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.dly_note_var = tk.StringVar(value=p.delay_note)
        for val, txt in [('dotted8','Dotted 8th'),('quarter','Quarter'),('half','Half')]:
            tk.Radiobutton(self.dly_note_frame, text=txt, variable=self.dly_note_var,
                           value=val, bg=PANEL, fg=FG, selectcolor='#2a3540',
                           activebackground=PANEL,
                           command=lambda: setattr(self.amp.pedals, 'delay_note',
                                                   self.dly_note_var.get())
                           ).pack(side='left', padx=6)
        self.dly_note_ms = tk.Label(self.dly_note_frame, text='', bg=PANEL,
                                     fg=DIM, font=(FF,9))
        self.dly_note_ms.pack(side='left', padx=8)

        # manual time slider (visible when not linked)
        self.dly_time_frame = tk.Frame(dly_sec, bg=PANEL)
        self.dly_time_frame.pack(fill='x', padx=10, pady=2)
        tk.Label(self.dly_time_frame, text='Time (ms)', bg=PANEL, fg=FG,
                 font=(FF,10), width=18, anchor='w').pack(side='left')
        self.dly_time_var = tk.StringVar(value=f'{int(p.delay_time*1000)} ms')
        def _dly_time_cb(v):
            self.amp.pedals.delay_time = float(v) / 1000.0
            self.dly_time_var.set(f'{int(float(v))} ms')
        self.dly_time_sl = self._scale(self.dly_time_frame, 10, 2000,
                                        _dly_time_cb, p.delay_time*1000, length=220)
        self.dly_time_sl.pack(side='left', padx=6)
        tk.Label(self.dly_time_frame, textvariable=self.dly_time_var,
                 bg=PANEL, fg=DIM, font=(FF,10), width=8).pack(side='left')

        _sl(dly_sec, 'Feedback', 0.0, 0.95, p.delay_feedback,
            lambda v: setattr(self.amp.pedals, 'delay_feedback', v),
            fmt=lambda v: f'{int(v*100)}%', attr='delay_feedback')
        _sl(dly_sec, 'Mix', 0.0, 1.0, p.delay_mix,
            lambda v: setattr(self.amp.pedals, 'delay_mix', v),
            fmt=lambda v: f'{int(v*100)}%', attr='delay_mix')

        self._dly_refresh_ui()

        # ── Reverb ───────────────────────────────────────────
        rev_sec = self._sec(inner, PEDAL_SECTION_TITLES['reverb'],
                            group=PEDALS_ACCORDION_GROUP, default_collapsed=True)
        self._fx_outer['reverb'] = rev_sec.master
        self._fx_reorder_row(rev_sec, 'reverb')
        self._build_pedal_preset_row(rev_sec, 'reverb')
        _, self.rev_var = on_off(rev_sec, 'reverb_on', 'Reverb  Enabled')
        _sl(rev_sec, 'Size', 0.1, 1.0, p.reverb_size,
            lambda v: setattr(self.amp.pedals, 'reverb_size', v),
            fmt=lambda v: f'{int(v*100)}%', attr='reverb_size')
        _sl(rev_sec, 'Mix', 0.0, 1.0, p.reverb_mix,
            lambda v: setattr(self.amp.pedals, 'reverb_mix', v),
            fmt=lambda v: f'{int(v*100)}%', attr='reverb_mix')

        # Start polling for the BPM-link ms display — guarded so rebuilding
        # this tab (e.g. after a profile load) doesn't stack up a second
        # concurrent polling loop on top of the first.
        if not getattr(self, '_dly_bpm_poll_started', False):
            self._dly_bpm_poll_started = True
            self._poll_dly_bpm()

        # apply the (default, or session-restored-later) chain order to the
        # freshly-built section frames
        self._relayout_pedals()

    def _rebuild_pedals_ui(self):
        """Tear down and rebuild the whole Pedals tab. _tab_pedals() reads
        every control's default from the pedal's current live value, so
        this is the simplest way to bring ~25 pedal controls in sync after
        a profile load sets them all programmatically — cheaper than
        threading a widget reference through every single slider/radio."""
        for child in self._pedals_tab_root.winfo_children():
            child.destroy()
        self._tab_pedals(self._pedals_tab_root)

    def _fx_reorder_row(self, parent, name):
        """Move up/down control at the top of a reorderable pedal section.
        _relayout_pedals() keeps the position label and the disabled ends
        of the chain in sync whenever the order changes."""
        row = tk.Frame(parent, bg=PANEL); row.pack(fill='x', padx=10, pady=(6,0))
        pos_var = tk.StringVar(value='')
        tk.Label(row, textvariable=pos_var, bg=PANEL, fg=DIM, font=(FF,9)).pack(side='left')
        up = ttk.Button(row, text='▲ Move up', width=11,
                         command=lambda: self._move_fx(name, -1))
        up.pack(side='right', padx=(4,0))
        down = ttk.Button(row, text='▼ Move down', width=13,
                           command=lambda: self._move_fx(name, 1))
        down.pack(side='right', padx=(4,0))
        self._fx_rows[name] = dict(pos_var=pos_var, up=up, down=down)

    def _reset_pedal_order(self):
        self.amp.pedals.order = list(DEFAULT_PEDAL_ORDER)
        self._relayout_pedals()

    def _move_fx(self, name, delta):
        order = self.amp.pedals.order
        i = order.index(name)
        j = i + delta
        if not (0 <= j < len(order)):
            return
        order[i], order[j] = order[j], order[i]
        self._relayout_pedals()

    def _relayout_pedals(self):
        """Re-pack the reorderable pedal sections to match amp.pedals.order
        (mute is pinned above all of them, so it's untouched here), and
        refresh each section's position label and Move up/down buttons."""
        order = self.amp.pedals.order
        for name in order:
            self._fx_outer[name].pack_forget()
        for name in order:
            self._fx_outer[name].pack(fill='x', padx=8, pady=6)
        n = len(order)
        for idx, name in enumerate(order):
            row = self._fx_rows[name]
            row['pos_var'].set(f'Chain position {idx+1} of {n}')
            row['up'].configure(state=('disabled' if idx == 0 else 'normal'))
            row['down'].configure(state=('disabled' if idx == n - 1 else 'normal'))
        # The status-strip icons mirror the same order, top of the tab.
        for name in order:
            self._pedal_icon_cells[name].pack_forget()
        for name in order:
            self._pedal_icon_cells[name].pack(side='left', padx=6)

    # Clickable hit-box for the status-strip LED, in the icon canvas's own
    # coordinates — a few px bigger than the drawn oval (w-16,4 to w-4,16)
    # so it's actually easy to hit, not just the exact 12x12 drawn shape.
    _PEDAL_LED_HITBOX = (lambda w: (w - 19, 1, w - 1, 19))(PEDAL_ICON_SIZE[0])

    def _build_pedal_status_strip(self, parent):
        """A row of small pedal icons at the top of the Pedals tab, one
        per pedal in the current chain order, each with a small LED dot
        showing on/off at a glance. Clicking most of an icon jumps the
        tab's scroll position to that pedal's own section below
        (expanding it first if it's currently collapsed) — a quick way
        to get to a pedal's full controls without hunting through the
        scrollable list. Clicking directly on the LED itself instead
        toggles that pedal on/off right from the strip, without needing
        to open its section first — same single click target doing two
        things depending on exactly where on it you click, same idea as
        a MIDI pad's tap-vs-hold already does elsewhere in this app.
        Silently omitted if Pillow isn't installed, same as the
        rounded-button skin."""
        self._pedal_icon_cells = {}
        self._pedal_led_ids = {}
        if not HAS_PIL:
            return
        sec = self._sec(parent, 'Pedal Status')
        strip = tk.Frame(sec, bg=PANEL)
        strip.pack(fill='x', padx=10, pady=8)
        x0, y0, x1, y1 = self._PEDAL_LED_HITBOX
        for code in self.amp.pedals.order:
            photo = _get_pedal_icon_photo(code)
            if photo is None:
                continue
            cell = tk.Frame(strip, bg=PANEL)
            w, h = PEDAL_ICON_SIZE
            c = tk.Canvas(cell, width=w, height=h, bg=PANEL, highlightthickness=0,
                          cursor='hand2')
            c.pack()
            c.create_image(0, 0, anchor='nw', image=photo)
            led = c.create_oval(w - 16, 4, w - 4, 16, outline='#2c2c2c')
            def _on_click(event, code=code):
                if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                    self._toggle_pedal_on_from_strip(code)
                else:
                    self._jump_to_pedal_section(code)
            c.bind('<Button-1>', _on_click)
            cell.pack(side='left', padx=6)
            self._pedal_icon_cells[code] = cell
            self._pedal_led_ids[code] = (c, led)
        self._refresh_all_pedal_leds()

    def _toggle_pedal_on_from_strip(self, code):
        """Flip a pedal's on/off state from its Pedal Status strip LED —
        mirrors exactly what clicking its own Enabled checkbox does
        (same BooleanVar, same amp.pedals attribute write), just from a
        different widget, so MIDI/LED feedback, Tone Profile saving, and
        everything else that watches the pedal's state can't tell the
        difference."""
        v = self._pedal_on_vars.get(code)
        if v is None:
            return
        new_val = not v.get()
        v.set(new_val)
        setattr(self.amp.pedals, code + '_on', new_val)

    def _jump_to_pedal_section(self, code):
        """Scroll the Pedals tab so `code`'s own section is at the top of
        the visible area, expanding it first if it's currently collapsed.
        Position is computed from the section's actual on-screen offset
        within the scrollable inner frame (winfo_y()), not a fixed index
        times a guessed row height, so it stays correct regardless of
        which sections above it are open/collapsed or how tall any of
        them are."""
        outer = self._fx_outer.get(code)
        canvas = getattr(self, '_pedals_canvas', None)
        if outer is None or canvas is None:
            return
        force_expand = getattr(outer, '_force_expand', None)
        if force_expand is not None:
            force_expand()
        outer.update_idletasks()
        inner = outer.master
        total_h = inner.winfo_height()
        if total_h <= 0:
            return
        frac = max(0.0, min(1.0, outer.winfo_y() / total_h))
        canvas.yview_moveto(frac)

    def _update_pedal_led(self, code):
        entry = getattr(self, '_pedal_led_ids', {}).get(code)
        if entry is None:
            return
        canvas, led_id = entry
        var = getattr(self, '_pedal_on_vars', {}).get(code)
        on = bool(var.get()) if var is not None else bool(getattr(self.amp.pedals, code + '_on'))
        # Deliberately RED rather than ACCENT — requested directly: a lit
        # pedal should read as "hot" the same way regardless of whatever
        # accent color is currently chosen in Appearance.
        canvas.itemconfig(led_id, fill=(RED if on else '#3a3a3a'),
                           outline=(_scale_color(RED, 0.4) if on else '#2c2c2c'))
        self._lpd8_refresh_leds()

    def _refresh_all_pedal_leds(self):
        for code in list(getattr(self, '_pedal_led_ids', {}).keys()):
            self._update_pedal_led(code)

    def _wah_refresh_ui(self):
        if self.wah_mode_var.get() == 'auto':
            self.wah_manual_frame.pack_forget()
            self.wah_auto_frame.pack(fill='x', padx=0, pady=0)
        else:
            self.wah_auto_frame.pack_forget()
            self.wah_manual_frame.pack(fill='x', padx=10, pady=2)

    def _dly_refresh_ui(self):
        linked = self.dly_link_var.get()
        if linked:
            self.dly_time_frame.pack_forget()
            self.dly_note_frame.pack(fill='x', padx=10, pady=2)
        else:
            self.dly_note_frame.pack_forget()
            self.dly_time_frame.pack(fill='x', padx=10, pady=2)

    def _poll_dly_bpm(self):
        """Keep the note-value ms display updated as BPM changes."""
        if self.dly_link_var.get():
            bpm = self.metro.bpm
            beat = 60.0 / max(1, bpm)
            ms = {'dotted8': beat*750, 'quarter': beat*1000, 'half': beat*2000
                  }.get(self.dly_note_var.get(), beat*750)
            self.dly_note_ms.configure(text=f'({ms:.0f} ms at {bpm} BPM)')
        self.after(200, self._poll_dly_bpm)

    def _close(self):
        if (getattr(self, 'confirm_quit_var', None) and self.confirm_quit_var.get()
                and self.amp.running):
            if not messagebox.askyesno('Quit SoloTone',
                                        'The amp is still running. Quit anyway?'):
                return
        self._save_session()
        try:
            self.metro.stop(); self.tuner.stop()
            self.tone.stop();  self.amp.stop()
            self.looper.clear_all()
            self.midi_ctrl.close()
        except: pass
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()
