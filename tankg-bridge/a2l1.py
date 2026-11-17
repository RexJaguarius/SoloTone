"""NAM A2-Lite WaveNet (.nam) -> the Tank-G's native 'A2L1' capture blob.

A Python port of `crates/pulsedeck-core/src/a2l.rs` from PulseDeck
(https://github.com/wilianlimasantos/PulseDeck), (c) 2026 Wilian de Lima Santos, MIT License:

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software ... The above
    copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

The formulas are documented in PulseDeck's `protocol.json` (file_transfer.a2l1_quantization). This module replaces
`nam2chip.py`, which was a port of UltraRig's PolyForm-Noncommercial `nam2chip.js`; for every capture tested the two
produce byte-identical blobs (see check_a2l1.py), so this one is a drop-in with a permissive licence.

Pipeline: parse the NAM JSON, take the Lite sub-model of a SlimmableContainer, run the network in float64 over a
calibration signal and record the largest value at every stage, derive fractional bits per stage and per tensor,
quantise (int16 weights, int32 activations, int64 biases), pack little-endian with a CRC32, and re-validate with the
firmware's shift rules so a blob the pedal would reject (LOAD rc=-4) never leaves the host.
"""
import json
import math
import struct
import zlib

import numpy as np

MAGIC = b"A2L1"
VERSION = 1
LSHIFT = 24
FS = 48000
EMBEDDED_GUARD = 1.5          # +3.5 dB margin on the ranges measured with the built-in synthetic signal
MAX_SHIFT = 62
CH = 3
WBITS = 16
ABITS = 32


class ConvertError(Exception):
    pass


NamError = ConvertError          # the name the callers of the old converter used


# ───────────────────────── parsing ─────────────────────────
def _describe_unsupported(d):
    arch = d.get("architecture", "")
    if arch == "WaveNet":
        layers = (d.get("config") or {}).get("layers")
        if isinstance(layers, list) and len(layers) == 2:
            return "This capture is NAM A1: download it again (it now exists as A2)."
    if arch == "LSTM":
        return "This capture is NAM LSTM: download the A2 version."
    if arch in ("ConvNet", "Linear", "Sequential"):
        return 'NAM model "%s": the pedal only runs A2 (WaveNet) captures.' % arch
    return None


def _pick_submodel(d):
    if d.get("architecture") == "SlimmableContainer":
        subs = (d.get("config") or {}).get("submodels")
        if not subs:
            raise ConvertError("This .nam has no sub-models (invalid SlimmableContainer).")
        best = min(subs, key=lambda s: s.get("max_value") if isinstance(s.get("max_value"), (int, float)) else float("inf"))
        return best["model"]
    return d


def _num(v, key, default):
    x = v.get(key) if isinstance(v, dict) else None
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else default


def _activations(l, n):
    a = l.get("activation")
    return list(a) if isinstance(a, list) else [a] * n


def _check_supported(sub):
    errs = []
    if sub.get("architecture") != "WaveNet":
        errs.append("architecture %s (WaveNet A2 only)" % sub.get("architecture", "?"))
    cfg = sub.get("config") or {}
    layers = cfg.get("layers") or []
    if len(layers) != 1:
        errs.append("%d layer arrays (only 1)" % len(layers))
    if cfg.get("head") is not None:
        errs.append("global head (only the array's head_rechannel)")
    cd = cfg.get("condition_dsp")
    if cd is not None and cd is not False:
        errs.append("condition_dsp")
    if errs:
        raise ConvertError("Unsupported model: %s." % "; ".join(errs))
    l = layers[0]
    ks, ds = l.get("kernel_sizes"), l.get("dilations")
    if (not isinstance(ks, list) or not isinstance(ds, list) or len(ks) != len(ds) or not ks
            or l.get("channels") is None or l.get("activation") is None or not isinstance(sub.get("weights"), list)):
        raise ConvertError("Unsupported model: missing fields (kernel_sizes, dilations, channels, activation or weights).")
    n = len(ks)
    if _num(l, "input_size", 1.0) != 1.0 or _num(l, "condition_size", 1.0) != 1.0:
        errs.append("input/condition size != 1")
    if (l.get("head1x1") or {}).get("active") is True:
        errs.append("head1x1 active")
    l11 = l.get("layer1x1") or {}
    if l11.get("active") is not True or _num(l11, "groups", 1.0) != 1.0:
        errs.append("layer1x1 inactive or grouped")
    channels = _num(l, "channels", 0.0)
    if _num(l, "bottleneck", channels) != channels:
        errs.append("bottleneck != channels")
    for k in ("groups_input", "groups_input_mixin"):
        if _num(l, k, 1.0) != 1.0:
            errs.append("%s != 1" % k)
    for k, v in l.items():
        if k.endswith("_film") and isinstance(v, dict) and v.get("active") is True:
            errs.append("FiLM active: %s" % k)
    gating = l.get("gating_mode", "none")
    for g in (gating if isinstance(gating, list) else [gating]):
        if g is not None and g != "none":
            errs.append("gating")
            break
    sec = l.get("secondary_activation")
    if any(x is not None for x in (sec if isinstance(sec, list) else [sec])):
        errs.append("secondary_activation")
    if any((a or {}).get("type") != "LeakyReLU" for a in _activations(l, n)):
        errs.append("activations other than LeakyReLU")
    if _num(l.get("head") or {}, "out_channels", 1.0) != 1.0:
        errs.append("head with more than 1 output")
    if errs:
        raise ConvertError("Unsupported model: %s." % "; ".join(errs))
    if channels != CH:
        raise ConvertError("This network has %g channels; the pedal engine runs the 3-channel A2-Lite network." % channels)
    hk = _num(l.get("head") or {}, "kernel_size", 0.0)
    if not 1.0 <= hk <= 64.0:
        raise ConvertError("Unsupported model: head kernel = %g." % hk)
    for i in range(n):
        if not (1 <= ks[i] <= 15 and 1 <= ds[i] <= 65535):
            raise ConvertError("Unsupported model: kernel/dilation out of range in layer %d." % (i + 1))


# ───────────────────────── the float engine ─────────────────────────
class _Layer:
    pass


class _Engine:
    pass


def _build_engine(sub):
    l = sub["config"]["layers"][0]
    ks, ds = l["kernel_sizes"], l["dilations"]
    n = len(ks)
    hk = int(_num(l["head"], "kernel_size", 0.0))
    head_bias = (l["head"].get("bias") is not False)
    # torch reads the list as float32
    w = np.array([float("nan") if not isinstance(v, (int, float)) else v for v in sub["weights"]], dtype=np.float64)
    w = w.astype(np.float32).astype(np.float64)
    if not np.all(np.isfinite(w)):
        raise ConvertError("The weights contain NaN or infinity.")
    need = CH
    for k in ks:
        need += CH * CH * k + CH + CH + CH * CH + CH
    need += CH * hk + (1 if head_bias else 0) + 1
    if len(w) != need:
        raise ConvertError("Unexpected number of weights: the file has %d, an A2-Lite network of this shape needs %d." % (len(w), need))
    pos = [0]

    def take(k):
        s = w[pos[0]:pos[0] + k].copy()
        pos[0] += k
        return s
    e = _Engine()
    e.rech = take(CH)
    acts = _activations(l, n)
    e.layers = []
    for i in range(n):
        ly = _Layer()
        ly.k, ly.dil = int(ks[i]), int(ds[i])
        s = (acts[i] or {}).get("negative_slope")
        ly.slope = float(s) if isinstance(s, (int, float)) else 0.01
        ly.conv_w = take(CH * CH * ly.k)
        ly.conv_b = take(CH)
        ly.mix_w = take(CH)
        ly.l1_w = take(CH * CH)
        ly.l1_b = take(CH)
        e.layers.append(ly)
    e.head_w = take(CH * hk)
    e.head_b = float(take(1)[0]) if head_bias else 0.0
    hs = (sub["config"]).get("head_scale")
    e.head_scale = float(hs) if isinstance(hs, (int, float)) else 1.0
    e.head_k = hk
    return e


def _receptive_field(e):
    return 1 + sum((l.k - 1) * l.dil for l in e.layers) + e.head_k - 1


def _measure(e, x1):
    """Largest |value| at every stage for input x1 (preceded by receptive_field-1 zeros)."""
    pad = _receptive_field(e) - 1
    n = pad + len(x1)
    xin = np.zeros(n)
    xin[pad:] = x1
    x3 = [e.rech[c] * xin for c in range(CH)]
    r = dict(rech=max(float(np.max(np.abs(a))) for a in x3), z=[], x3=[])
    cur = n
    head = None
    for ly in e.layers:
        k, d = ly.k, ly.dil
        lout = cur - d * (k - 1)
        off = cur - lout
        a = []
        zmax = 0.0
        for co in range(CH):
            z = np.full(lout, ly.conv_b[co])
            for kk in range(k):
                sh = kk * d
                w0 = ly.conv_w[(co * CH) * k + kk]
                w1 = ly.conv_w[(co * CH + 1) * k + kk]
                w2 = ly.conv_w[(co * CH + 2) * k + kk]
                z += w0 * x3[0][sh:sh + lout] + w1 * x3[1][sh:sh + lout] + w2 * x3[2][sh:sh + lout]
            xo = n - lout
            v = z + ly.mix_w[co] * xin[xo:xo + lout]
            zmax = max(zmax, float(np.max(np.abs(v))))
            a.append(np.where(v >= 0.0, v, ly.slope * v))
        r["z"].append(zmax)
        nx, xmax = [], 0.0
        for co in range(CH):
            w0, w1, w2 = ly.l1_w[co * CH], ly.l1_w[co * CH + 1], ly.l1_w[co * CH + 2]
            v = x3[co][off:off + lout] + (w0 * a[0] + w1 * a[1] + w2 * a[2] + ly.l1_b[co])
            xmax = max(xmax, float(np.max(np.abs(v))))
            nx.append(v)
        r["x3"].append(xmax)
        x3 = nx
        if head is not None:
            for c in range(CH):
                a[c] = a[c] + head[c][len(head[c]) - lout:]
        head = a
        cur = lout
    r["head"] = max(float(np.max(np.abs(h))) for h in head)
    hk = e.head_k
    lo = cur - hk + 1
    y = np.full(lo, e.head_b)
    for k in range(hk):
        y += e.head_w[k] * head[0][k:k + lo] + e.head_w[hk + k] * head[1][k:k + lo] + e.head_w[2 * hk + k] * head[2][k:k + lo]
    r["out"] = float(np.max(np.abs(y * e.head_scale)))
    return r


def _merge(rs):
    mx = lambda f: max(f(r) for r in rs)
    return dict(rech=mx(lambda r: r["rech"]), head=mx(lambda r: r["head"]), out=mx(lambda r: r["out"]),
                z=[mx(lambda r, i=i: r["z"][i]) for i in range(len(rs[0]["z"]))],
                x3=[mx(lambda r, i=i: r["x3"][i]) for i in range(len(rs[0]["x3"]))])


# ───────────────────────── calibration signal ─────────────────────────
class _Rng:
    """mulberry32 + Box-Muller: deterministic and portable."""
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (self.s + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.s
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return float(t ^ (t >> 14)) / 4294967296.0

    def g(self):
        a = self.u()
        b = self.u()
        return math.sqrt(-2.0 * math.log(1.0 - a)) * math.cos(2.0 * math.pi * b)


def _sweep(fs, secs):
    n = int(math.floor(secs * fs))
    t_total = (n - 1.0) / fs
    r = fs / 2.0 / 20.0
    lr = math.log(r)
    t = np.arange(n) / fs
    return np.sin(2.0 * math.pi * 20.0 * (np.power(r, t / t_total) - 1.0) * t_total / lr)


def _pluck(fs, f, n, rng, decay):
    length = max(int(round(fs / f)), 2)
    line = [rng.u() * 2.0 - 1.0 for _ in range(length)]
    y = [0.0] * n
    p = 0
    for i in range(n):
        a = line[p]
        b = line[(p + 1) % length]
        y[i] = a
        line[p] = decay * 0.5 * (a + b)
        p = (p + 1) % length
    return y


def _strum(fs, freqs, secs, peak, rng, decay):
    n = int(math.floor(secs * fs))
    y = np.zeros(n)
    for i, f in enumerate(freqs):
        off = int(math.floor(i * 0.012 * fs))
        s = np.array(_pluck(fs, f, n - off, rng, decay))
        y[off:off + len(s)] += s
    m = float(np.max(np.abs(y)))
    if m > 0.0:
        y *= peak / m
    return y


def calibration_signals(fs=float(FS)):
    """Short synthetic calibration set: log sweep, noise, three hot guitar chords."""
    rng = _Rng(12345)
    out = [_sweep(fs, 2.0)]
    n = int(math.floor(2.0 * fs))
    out.append(np.array([min(1.0, max(-1.0, rng.g() * 0.5)) for _ in range(n)]))
    e = [82.41, 123.47, 164.81, 207.65, 246.94, 329.63]
    a = [110.0, 164.81, 220.0, 277.18, 329.63, 440.0]
    d = [73.42, 110.0, 146.83, 220.0, 293.66, 369.99]
    out.append(_strum(fs, e, 2.0, 0.99, rng, 0.9985))
    out.append(_strum(fs, a, 1.5, 0.99, rng, 0.9985))
    out.append(_strum(fs, d, 1.5, 0.99, rng, 0.9985))
    return out


# ───────────────────────── quantisation ─────────────────────────
def _fbits(maxabs, bits):
    if maxabs <= 0.0:
        return bits - 1
    return bits - 1 - (int(math.floor(math.log2(maxabs))) + 1)


def _q_int(v, f, bits):
    lim = 1 << (bits - 1)
    q = np.rint(np.asarray(v, dtype=np.float64) * (2.0 ** f))
    return np.clip(q, -float(lim), float(lim - 1)).astype(np.int64)


def _q_i64(v, f):
    q = np.rint(np.asarray(v, dtype=np.float64) * (2.0 ** f))
    return np.clip(q, -9.223372036854775808e18, 9.223372036854775807e18).astype(np.float64)   # saturating, as Rust's `as`


def _qweight(w, fmax):
    f = min(_fbits(float(np.max(np.abs(w))) if len(w) else 0.0, WBITS), fmax)
    return f, _q_int(w, f, WBITS).astype("<i2")


def _quantize(e, r, name):
    fin = _fbits(1.0, ABITS)
    fx3_0 = _fbits(r["rech"] * 2.0, ABITS)
    fh = _fbits(r["head"] * 2.0, ABITS)
    fout = _fbits(r["out"] * 2.0, ABITS)
    m = dict(name=name, head_k=e.head_k, fin=fin, fx3_0=fx3_0, layers=[])
    m["rech_f"], m["rech_w"] = _qweight(e.rech, fx3_0 - fin + 31)
    fx3_prev = fx3_0
    for i, ly in enumerate(e.layers):
        fz = _fbits(r["z"][i] * 2.0, ABITS)
        fx3 = _fbits(r["x3"][i] * 2.0, ABITS)
        cf, cw = _qweight(ly.conv_w, fz - fx3_prev + 31)
        mf, mw = _qweight(ly.mix_w, fz - fin + 31)
        lf, lw = _qweight(ly.l1_w, fx3 - fz + 31)
        m["layers"].append(dict(
            k=ly.k, dil=ly.dil, fz=fz, fx3=fx3, lmul=int(round(ly.slope * float(1 << LSHIFT))),
            cf=cf, cw=cw, cb=[int(v) for v in _q_i64(ly.conv_b, cf + fx3_prev)],
            mf=mf, mw=mw, lf=lf, lw=lw, lb=[int(v) for v in _q_i64(ly.l1_b, lf + fz)]))
        fx3_prev = fx3
    fh = min(fh, min(l["fz"] for l in m["layers"]) - 1)         # the head-sum shift (Fz - Fh) must stay >= 1
    hw_scaled = e.head_w * e.head_scale
    m["fh"], m["fout"] = fh, fout
    m["hf"], m["hw"] = _qweight(hw_scaled, fout - fh + 31)
    m["hb"] = int(_q_i64(np.array([e.head_b * e.head_scale]), m["hf"] + fh)[0])
    return m


def _i8(v, what):
    if not -128 <= v <= 127:
        raise ConvertError("This model needs %d fractional bits in %s (limit +-127): weights/levels too extreme for the pedal." % (v, what))
    return v & 0xFF


def blob_size(c, head_k, layer_ks):
    size = 36 + 1 + 2 * c + 4 + 1 + 2 * c * head_k + 8
    for k in layer_ks:
        size += 9 + 1 + 2 * c * c * k + 8 * c + 1 + 2 * c + 1 + 2 * c * c + 8 * c
    return size


def _pack(m):
    b = bytearray()
    b += MAGIC
    b += struct.pack("<HBBIHBB", VERSION, WBITS, ABITS, FS, len(m["layers"]), CH, m["head_k"])
    b += bytes([_i8(m["fin"], "Fin"), _i8(m["fx3_0"], "Fx3_0"), _i8(m["fh"], "Fh"), _i8(m["fout"], "Fout")])
    name = m["name"].encode("utf-8")[:16]
    while True:
        try:
            name.decode("utf-8")
            break
        except UnicodeDecodeError:
            name = name[:-1]
    b += name.ljust(16, b"\0")
    b.append(_i8(m["rech_f"], "rechannel"))
    b += m["rech_w"].tobytes()
    for i, l in enumerate(m["layers"]):
        at = lambda s: "%s (layer %d)" % (s, i + 1)
        b.append(l["k"])
        b += struct.pack("<H", l["dil"])
        b.append(_i8(l["fz"], at("Fz")))
        b.append(_i8(l["fx3"], at("Fx3")))
        b += struct.pack("<i", l["lmul"])
        b.append(_i8(l["cf"], at("cF")))
        b += l["cw"].tobytes()
        b += b"".join(struct.pack("<q", v) for v in l["cb"])
        b.append(_i8(l["mf"], at("mF")))
        b += l["mw"].tobytes()
        b.append(_i8(l["lf"], at("lF")))
        b += l["lw"].tobytes()
        b += b"".join(struct.pack("<q", v) for v in l["lb"])
    b.append(_i8(m["hf"], "head"))
    b += m["hw"].tobytes()
    b += struct.pack("<q", m["hb"])
    b += struct.pack("<I", zlib.crc32(bytes(b)) & 0xFFFFFFFF)
    assert len(b) == blob_size(CH, m["head_k"], [l["k"] for l in m["layers"]])
    return bytes(b)


def validate_blob(u):
    """Re-read the blob and apply the firmware's shift rules."""
    n = len(u)
    if n < 44 or u[:4] != MAGIC:
        raise ConvertError("internal: bad blob header")
    if zlib.crc32(u[:n - 4]) & 0xFFFFFFFF != struct.unpack("<I", u[n - 4:])[0]:
        raise ConvertError("internal: CRC mismatch")
    i8 = lambda o: struct.unpack("<b", u[o:o + 1])[0]
    wb, ab = u[6], u[7]
    nl = struct.unpack("<H", u[12:14])[0]
    c, hk = u[14], u[15]
    fin, fx30, fh, fout = i8(16), i8(17), i8(18), i8(19)
    if wb != 16 or ab not in (16, 32) or c == 0 or c > 16 or hk == 0 or hk > 64 or nl == 0:
        raise ConvertError("internal: invalid header")

    def sh(s, what):
        if abs(s) > MAX_SHIFT:
            raise ConvertError("This model needs a shift of %d bits in %s (the engine accepts +-%d): weights/levels too extreme for the pedal." % (s, what, MAX_SHIFT))
    o = 36
    rf = i8(o)
    o += 1 + 2 * c
    sh(rf + fin - fx30, "the input rechannel")
    fprev = fx30
    for i in range(nl):
        k, fz, fx3 = u[o], i8(o + 3), i8(o + 4)
        o += 9
        cf = i8(o)
        o += 1 + 2 * c * c * k + 8 * c
        mf = i8(o)
        o += 1 + 2 * c
        lf = i8(o)
        o += 1 + 2 * c * c + 8 * c
        at = " (layer %d)" % (i + 1)
        sh(cf + fprev - fz, "conv" + at)
        sh(mf + fin - fz, "mixer" + at)
        sh(lf + fz - fx3, "1x1" + at)
        sh(fprev - fx3, "residual" + at)
        sh(fz - fh, "head sum" + at)
        cs = cf + fprev - fz
        if not 0 <= cs <= 31:
            raise ConvertError("This model needs a conv shift of %d bits%s (the engine accepts 0..31)." % (cs, at))
        fprev = fx3
    hf = i8(o)
    o += 1 + 2 * c * hk + 8
    so = hf + fh - fout
    if not 0 <= so <= 31:
        raise ConvertError("This model needs an output shift of %d bits (the engine accepts 0..31)." % so)
    if o + 4 != n:
        raise ConvertError("internal: blob layout mismatch")


# ───────────────────────── public API ─────────────────────────
def convert(path, name=None, signals=None, guard=None):
    """Convert a .nam file. Returns {'blob': bytes, 'name': str, 'layers': n, 'macs': n, 'bytes': n, 'receptive_field': n}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (ValueError, UnicodeDecodeError):
        raise ConvertError("This file is not valid JSON, so it is not a .nam capture.")
    if not isinstance(d, dict) or d.get("architecture") is None or d.get("config") is None:
        raise ConvertError('This does not look like a NAM capture (no "architecture"/"config").')
    msg = _describe_unsupported(d)
    if msg:
        raise ConvertError(msg)
    sub = _pick_submodel(d)
    _check_supported(sub)
    sr = sub.get("sample_rate", d.get("sample_rate", FS))
    if sr != FS:
        raise ConvertError("This capture is %s Hz; the pedal runs at 48 kHz. Download the 48 kHz version." % sr)
    e = _build_engine(sub)
    sigs = signals if signals is not None else calibration_signals()
    ranges = _merge([_measure(e, s) for s in sigs])
    g = guard if guard is not None else (1.0 if signals is not None else EMBEDDED_GUARD)
    if g != 1.0:
        ranges["rech"] *= g
        ranges["head"] *= g
        ranges["out"] *= g
        ranges["z"] = [v * g for v in ranges["z"]]
        ranges["x3"] = [v * g for v in ranges["x3"]]
    nm = name or ((d.get("metadata") or {}).get("name") or "NAM")
    blob = _pack(_quantize(e, ranges, nm))
    validate_blob(blob)
    macs = sum(CH * CH * l.k + CH * CH + CH + CH for l in e.layers) + CH * e.head_k + CH
    # the input level the capture was made at (dBu), when its metadata says: the caller sets the pedal's capture trim from it
    lvl = (d.get("metadata") or {}).get("input_level_dbu")
    in_dbu = float(lvl) if isinstance(lvl, (int, float)) and not isinstance(lvl, bool) and math.isfinite(lvl) else None
    return dict(blob=blob, name=nm, layers=len(e.layers), macs=macs, bytes=len(blob), in_dbu=in_dbu,
                receptive_field=_receptive_field(e))
