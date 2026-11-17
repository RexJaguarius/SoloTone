#!/usr/bin/env python3
"""SoloTone .stprofile -> UltraRig (Tank-G) command plan.  DRY RUN ONLY.

This script never opens a serial port and never imports pyserial. It reads a
profile, maps it onto UltraRig block types using the pedal's own block
catalogue (read from your pedal with --port, or from the file ultrarig_probe.py writes), and prints the
commands a live run would send, plus a CPU / memory estimate and every place
where the mapping is a guess or has no equivalent.

Usage:
    python bridge.py "profiles/My Tone.stprofile" --port COM7
    python bridge.py --all profiles --port COM7          (one summary line per profile)
    python bridge.py PROFILE --catalogue other_fxt.txt

Mapping knowledge comes from SoloTone's own signal flow:
    hum -> pre-amp pedals -> in_gain -> gate -> NAM -> cab IR -> EQ -> post-amp pedals
UltraRig banks: P = before the capture, Q = after it.
"""
import argparse, copy, json, math, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
CPU_BUDGET = 30500          # ticks; the pedal reports 30.5k-31.9k depending on the moment
CPU_WARN = 0.85             # warn above this share of the budget
MEM_POOL = 106480           # bytes of effects pool seen in the MEM reply
IR_MS_DEFAULT = 21          # the pedal's own default Cab IR length
IR_MS_MIN = 5
IR_MEM_BASE, IR_MEM_PER_MS = 7600, 770        # measured: Cab IR block memory grows with its length
IR_TICKS_BASE, IR_TICKS_PER_MS = 4670, 135    # measured: and so does its CPU (the catalogue lists 3727)
CPU_BUDGET_IR = 27000        # measured: the pedal's CPU budget shrinks to ~28000 once an IR is in the chain
DELAY_BYTES_PER_MS = 99.6    # delay memory grows with its time (34864 B at the 350 ms default)
MEM_MARGIN = 4096            # keep this much of the pool free
DEFAULT_OS = 1              # oversampling tier for drive/amp blocks (0 off, 1 = 2x?, 3 = max)


# ───────────────────────── catalogue ─────────────────────────
class Param:
    def __init__(self, idx, name, lo, hi, default):
        self.idx, self.name, self.lo, self.hi, self.default = idx, name, lo, hi, default


class BlockType:
    def __init__(self, tid, name, cost, cat, mem):
        self.id, self.name, self.cost, self.cat, self.mem = tid, name, cost, cat, mem
        self.params = []
        self.os_tiers = None        # (tiers, costs)

    def param(self, name):
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{self.name} has no parameter {name!r}")

    def cost_at(self, os_n):
        if self.os_tiers and os_n in self.os_tiers[0]:
            return self.os_tiers[1][self.os_tiers[0].index(os_n)]
        return self.cost


def load_catalogue(path):
    """The block catalogue from a text file holding the pedal's `FXT` reply (see ultrarig_probe.py)."""
    return parse_catalogue(pathlib.Path(path).read_text(encoding="utf-8").splitlines())


def parse_catalogue(lines):
    """The block catalogue from the lines of the pedal's own `FXT` reply: block types, their parameters and ranges, their CPU and
    memory cost. A send reads it from the connected pedal, so it always matches the firmware that is installed."""
    types, cur = {}, None
    for ln in lines:
        m = re.match(r"^FXT (\d+) (\w+) cost=(\d+) cat=(\w+) mem=(\d+)", ln)
        if m:
            cur = BlockType(int(m[1]), m[2], int(m[3]), m[4], int(m[5]))
            types[cur.name] = cur
            continue
        m = re.match(r"^FXT\s+os tiers=([\d,]+) cost=([\d,]+)", ln)
        if m and cur:
            cur.os_tiers = ([int(x) for x in m[1].split(",")], [int(x) for x in m[2].split(",")])
            continue
        m = re.match(r"^FXT\s+(\d+) (\w+) (-?\d+)\.\.(-?\d+) \((-?\d+)\)", ln)
        if m and cur:
            cur.params.append(Param(int(m[1]), m[2], int(m[3]), int(m[4]), int(m[5])))
    if not types:
        sys.exit("No block types found in the FXT reply")
    return types


# ───────────────────────── plan model ─────────────────────────
class Block:
    def __init__(self, bank, bt, src, note=""):
        self.bank, self.bt, self.src, self.note = bank, bt, src, note
        self.values = {p.name: p.default for p in bt.params}
        self.guesses = []           # parameters that are defaults/guesses rather than derived
        self.code = None            # the SoloTone pedal this block comes from ('od', 'delay'...), None for hum / gate / IR / level blocks
        self.extra_mem = 0          # memory the catalogue does not list (cab IR block, delay time)
        self.extra_ticks = 0
        self.ir_samples = None

    def set(self, name, value, guess=False):
        p = self.bt.param(name)
        v = int(round(value))
        self.values[name] = max(p.lo, min(p.hi, v))
        if v != self.values[name]:
            self.guesses.append(f"{name} clamped {v} -> {self.values[name]}")
        if guess:
            self.guesses.append(f"{name} is a first guess")

    def cost(self):
        return self.bt.cost_at(self.values.get("os_n")) + self.extra_ticks


def set_ir_length(b, ms):
    """Use `ms` of the cab IR: sets the block's Length, the number of samples to upload, and the cost model."""
    b.ir_ms = ms
    b.ir_samples = int(round(ms * 48))
    b.values["largo_ms"] = ms
    b.extra_mem = int(IR_MEM_BASE + IR_MEM_PER_MS * ms) - b.bt.mem
    b.extra_ticks = int(IR_TICKS_BASE + IR_TICKS_PER_MS * ms) - b.bt.cost


class Plan:
    def __init__(self, types):
        self.types, self.blocks, self.warnings, self.uploads = types, [], [], []
        self.fit_notes = []
        self.ampeq = None          # capture Gain/tone/Volume values (dB) when the pedal's AMPEQ is used

    def add(self, bank, tname, src, note=""):
        b = Block(bank, self.types[tname], src, note)
        b.code = getattr(self, "_code", None)
        if "os_n" in b.values:
            b.values["os_n"] = DEFAULT_OS if tname != "amp_hi" else b.values["os_n"]
        self.blocks.append(b)
        return b

    def warn(self, text):
        self.warnings.append(text)


def db(x):
    return 20 * math.log10(max(float(x), 1e-9))


def pct_from_drive(d):          # SoloTone drive knobs run 1-10
    return (float(d) - 1.0) / 9.0 * 100.0


# ───────────────────────── per-pedal mapping ─────────────────────────
# SoloTone's own factory values for the pedals added after the first eight, used when a profile switches one on
# but does not carry every one of its settings (hand-edited, or from a build in between).
_PEDAL_DEFAULTS = {
    "phaser_rate": 0.5, "phaser_depth": 0.6, "phaser_feedback": 0.3,
    "tremolo_rate": 5.0, "tremolo_depth": 0.5, "tremolo_shape": 0.0,
    "chorus_rate": 0.5, "chorus_depth": 0.5, "chorus_mix": 0.5, "chorus_mode": "chorus",
    "reverb_type": "room", "reverb_mix": 0.25, "reverb_size": 0.6, "reverb_decay": 0.5, "reverb_damp": 0.5,
    "reverb_shimmer": 0.5,
    "pitch_semi": 12, "pitch_cents": 0, "pitch_semi2": 7, "pitch_level2": 0.0, "pitch_mix": 0.5,
}


def map_pedal(plan, bank, key, p):
    g = lambda k, d=None: p.get(f"{key}_{k}", d if d is not None else _PEDAL_DEFAULTS.get(f"{key}_{k}"))
    if key == "comp":
        b = plan.add(bank, "comp", "comp")
        b.set("umbral_db", g("thresh")); b.set("ratio_x10", g("ratio") * 10)
        b.set("ataque_ms", g("attack")); b.set("release_ms", g("release"))
        b.set("nivel_db", g("makeup")); b.set("mezcla_pct", 100)
        if g("warmth", 0) > 0:
            plan.warn(f"comp: Warmth {g('warmth')} has no equivalent and is dropped")
    elif key == "wah":
        # UltraRig alpha 11: the Envelope Filter has a Wah voice (filtro_modo 6 = Wah up, 7 = Wah down). It is an
        # envelope wah (opens as you pick harder), not a pedal-position wah or a timed sweep, so it is an approximation.
        if "envfilter" not in plan.types:
            plan.warn("wah is ON in the profile but this firmware has no Envelope Filter (needs UltraRig alpha 10+): skipped")
        else:
            envelope = g("mode") == "envelope"
            b = plan.add(bank, "envfilter", "wah (envelope filter)" if envelope else "wah (envelope wah)")
            if envelope:
                kind = {"lowpass": 0, "bandpass": 1, "highpass": 2, "wah": 6}.get(g("env_type", "wah"), 6)
                down = g("env_dir", "up") == "down"
                b.set("filtro_modo", (kind + 1 if kind == 6 else kind + 3) if down else kind)
                b.set("sens_pct", g("env_sens", 0.6) * 100, guess=True)
                b.set("decay_pct", g("env_decay", 0.4) * 100, guess=True)
            else:
                b.set("filtro_modo", 6)
                b.set("sens_pct", 60, guess=True); b.set("decay_pct", 40, guess=True)
                plan.warn("wah: the pedal's Wah is an envelope wah that opens with your picking, so it can't hold a fixed "
                          "position (manual) or sweep on a timer (auto) like SoloTone's; the sound is close in character only")
            b.set("frec_lo_hz", g("range_lo", 400)); b.set("frec_hi_hz", g("range_hi", 2200))
            b.set("reso_pct", (g("q", 4.0) - 1.0) / 9.0 * 100, guess=True)
            b.set("mezcla_pct", 100)
    elif key == "fuzz":
        if g("mode") == "big_muff":
            b = plan.add(bank, "muff", "fuzz (big_muff)")
            b.set("sustain_pct", pct_from_drive(g("drive")))
            tone = {"scoop": 25, "flat": 50, "full": 80}.get(p.get("muff_tone"), 50)
            b.set("tono_pct", tone, guess=True)
            b.set("nivel_db", db(g("vol")))
        else:
            tb = g("mode") == "tone_bender"
            b = plan.add(bank, "fuzzface", "fuzz (tone_bender)" if tb else "fuzz (fuzz_face)")
            b.set("fuzz_pct", pct_from_drive(g("drive"))); b.set("nivel_db", db(g("vol")))
            if tb:
                if "modelo" in {p_.name for p_ in b.bt.params}:
                    b.set("modelo", 1)
                else:
                    plan.warn("fuzz: Tone Bender needs UltraRig alpha 11 or later; the Fuzz Face model is used instead")
    elif key == "dist":
        mode = g("mode")
        if mode == "rat" or mode == "metal":
            b = plan.add(bank, "rat", f"dist ({mode})")
            b.set("dist_pct", pct_from_drive(g("drive")))
            b.set("filtro_pct", (1 - g("tone")) * 100, guess=True)   # RAT filter: higher = darker
            if mode == "metal":
                plan.warn("dist: 'metal' has no equivalent; using rat (no mid-scoop)")
        else:
            b = plan.add(bank, "ds1", f"dist ({mode})")
            b.set("dist_pct", pct_from_drive(g("drive"))); b.set("tono_pct", g("tone") * 100)
            if mode == "dist_plus":
                plan.warn("dist: 'dist_plus' has no equivalent; using ds1")
        b.set("nivel_db", db(g("vol")))
    elif key == "od":
        mode = g("mode")
        if mode == "klon":
            b = plan.add(bank, "klon", "od (klon)")
            b.set("ganancia_pct", pct_from_drive(g("drive"))); b.set("nivel_db", db(g("vol")))
            b.set("mezcla_pct", 100, guess=True)
        elif mode == "boost":
            b = plan.add(bank, "boost", "od (boost)")
            b.set("ganancia_db", (g("drive") - 1) / 9 * 24, guess=True)
        elif mode == "bd2":
            b = plan.add(bank, "bd2", "od (blues driver)")
            b.set("drive_pct", pct_from_drive(g("drive"))); b.set("nivel_db", db(g("vol")))
            b.set("tono_pct", 50, guess=True)
        else:
            b = plan.add(bank, "ts9", "od (tubescreamer)")
            b.set("drive_pct", pct_from_drive(g("drive"))); b.set("nivel_db", db(g("vol")))
    elif key == "phaser":
        b = plan.add(bank, "phaser", "phaser")
        b.set("vel_chz", g("rate") * 100); b.set("prof_pct", g("depth") * 100)
        b.set("realim_pct", g("feedback") * 100)
    elif key == "tremolo":
        b = plan.add(bank, "tremolo", "tremolo")
        b.set("vel_chz", g("rate") * 100); b.set("prof_pct", g("depth") * 100)
        b.set("forma_pct", g("shape", 0.0) * 100)
    elif key == "pitch":
        # UltraRig pitchpro (id 29): one voice, semitones / cents / mix. Voice 2 has no equivalent.
        b = plan.add(bank, "pitchpro", "pitch shifter")
        b.set("semitonos_st", g("semi")); b.set("cents_ct", g("cents")); b.set("mezcla_pct", g("mix") * 100)
        if g("level2", 0.0) > 0.0:
            plan.warn("pitch: voice 2 has no equivalent on the pedal (one voice sent)")
    elif key == "rotary":
        # UltraRig alpha 11 Rotary (id 39): speed 0 Slow / 1 Fast / 2 Brake, fast rate in centi-Hz.
        if "rotary" not in plan.types:
            plan.warn("rotary is ON in the profile but this firmware has no Rotary block (needs UltraRig alpha 11): skipped")
        else:
            b = plan.add(bank, "rotary", "rotary speaker")
            b.set("velocidad", {"slow": 0, "fast": 1, "brake": 2}.get(g("speed", "slow"), 0))
            b.set("rapido_chz", g("fast", 6.7) * 100)
            b.set("distancia_pct", (1 - g("depth", 0.6)) * 80, guess=True)
            # inercia_pct packs two things (PulseDeck's protocol.json, rotary.inercia_pct): the low 7 bits are the inertia
            # in percent, and value >> 7 is the Slow rate in tenths of a Hz (0 = the default 0.8 Hz). Range 0..2660.
            slow_tenths = max(2, min(20, int(round(g("slow", 0.8) * 10))))
            b.set("inercia_pct", (slow_tenths << 7) | int(round(20 + g("ramp", 0.5) * 60)), guess=True)
            b.set("mezcla_pct", g("mix", 1.0) * 100)
    elif key == "chorus":
        mode = g("mode")
        if mode == "vibrato":
            b = plan.add(bank, "vibrato", "chorus (vibrato)")
            b.set("vel_chz", g("rate") * 100); b.set("prof_pct", g("depth") * 100)
        elif mode == "doubler":
            b = plan.add(bank, "doubler", "chorus (doubler)")
            b.set("retardo_ms", 10 + 15 * g("depth")); b.set("detune_ct", 8, guess=True)
            b.set("mezcla_pct", g("mix") * 100); b.set("nivel_db", 0)
        else:
            flanger = mode == "flanger"
            b = plan.add(bank, "flanger" if flanger else "chorus", f"chorus ({mode})")
            b.set("vel_chz", g("rate") * 100); b.set("prof_pct", g("depth") * 100)
            b.set("mezcla_pct", g("mix") * 100)
            if flanger:
                b.set("realim_pct", 50, guess=True)
    elif key == "delay":
        b = plan.add(bank, "delay", f"delay ({g('mode')})")
        b.set("tiempo_ms", g("time") * 1000); b.set("realim_pct", g("feedback") * 100)
        # measured 2026-11-05: the buffer is sized when the block is added (350 ms = 34848 B) and only ever grows,
        # so a short delay time saves nothing; a longer one costs 99.6 B/ms
        b.extra_mem = int(max(350, b.values["tiempo_ms"]) * DELAY_BYTES_PER_MS) - b.bt.mem
        b.set("mezcla_pct", g("mix") * 100)
        b.set("amort_pct", 45 if g("mode") == "tape" else 15, guess=True)
        if p.get("delay_bpm_link"):
            plan.warn("delay: BPM link is on; fixed time sent (UltraRig 'sync' not mapped)")
    elif key == "reverb":
        rtype = g("type", "room")
        size, mix = g("size") * 100, g("mix") * 100
        decay, damp = g("decay", 0.5) * 100, g("damp", 0.5) * 100
        if rtype == "plate":
            b = plan.add(bank, "plate", "reverb (plate)")
            b.set("tamano_pct", size); b.set("decay_pct", decay); b.set("amort_pct", damp)
            b.set("predelay_ms", 0); b.set("mezcla_pct", mix); b.set("mod_pct", 0)
        elif rtype == "spring":
            b = plan.add(bank, "spring", "reverb (spring)")
            b.set("dwell_pct", size, guess=True); b.set("decay_pct", decay)
            b.set("tono_pct", 100 - damp, guess=True); b.set("resortes_n", 3)
            b.set("mezcla_pct", mix)
        elif rtype == "shimmer":
            b = plan.add(bank, "shimmer", "reverb (shimmer)")
            b.set("tamano_pct", size); b.set("decay_pct", decay); b.set("amort_pct", damp)
            b.set("intervalo", 0, guess=True); b.set("mezcla_pct", mix)
            b.set("shimmer_pct", g("shimmer", 0.5) * 100, guess=True)
        elif rtype == "cloud":
            b = plan.add(bank, "cloud", "reverb (cloud)")
            b.set("tamano_pct", size); b.set("decay_pct", decay); b.set("amort_pct", damp)
            b.set("mod_pct", 30, guess=True); b.set("mezcla_pct", mix)
        else:
            b = plan.add(bank, "reverb2", "reverb")
            b.set("tamano_pct", size); b.set("decay_pct", size, guess=True)
            b.set("mezcla_pct", mix); b.set("predelay_ms", 0); b.set("mod_pct", 0)
            b.set("amort_pct", 40, guess=True)


def nam_verdict(path):
    if not path:
        return "no capture set"
    try:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "file not found on this machine"
    except Exception as e:
        return f"unreadable ({e.__class__.__name__})"
    sr = d.get("sample_rate")
    sub, tag = d, ""
    if d.get("architecture") == "SlimmableContainer":      # A2 file: the pedal runs the smallest submodel
        subs = sorted((d.get("config") or {}).get("submodels") or [], key=lambda s: s.get("max_value", 0))
        if not subs:
            return "broken SlimmableContainer (no submodels)"
        sub, tag = subs[0].get("model") or {}, "A2 container, "
    cfg = sub.get("config") or {}
    layers = cfg.get("layers") or []
    if sub.get("architecture") != "WaveNet":
        return f"{sub.get('architecture')}: not supported (A2 WaveNet only)"
    if len(layers) != 1:
        return f"{len(layers)}-array WaveNet = A1, rejected (needs the A2 version)"
    ch = layers[0].get("channels")
    if ch != 3:
        return f"{tag}{ch}-channel network, pedal runs the 3-channel Lite"
    if sr not in (None, 48000):
        return f"{tag}{sr} Hz (pedal needs 48 kHz)"
    return f"{tag}OK (checked structure only; a2l1 does the full test)"


def build_plan(profile, types, use_ampeq=True, budget=None):
    plan = Plan(types)
    if budget:
        plan.pedal_budget = budget
    P = profile["pedals"]
    order = P.get("order", [])
    split = order.index("amp") if "amp" in order else len(order)
    pre, post = order[:split], order[split + 1:]

    if profile.get("hum_on"):
        b = plan.add("P", "hum", "hum filter")
        b.set("mains_hz", profile.get("hum_base_hz", 50))
    for k in pre:
        if P.get(f"{k}_on"):
            plan._code = k
            map_pedal(plan, "P", k, P)
            plan._code = None
    thr = profile.get("gate_threshold", 0.0)
    if thr and thr > 1e-5:
        if profile.get("gate_mode") == "smart":
            # SoloTone's Smart gate is UltraRig's smart gate (sgate): threshold, depth, release, hush
            b = plan.add("P", "sgate", "smart noise gate")
            b.set("umbral_db", max(-90.0, min(-20.0, db(thr))))
            b.set("rango_att", profile.get("gate_range", 60.0))
            b.set("release_ms", profile.get("gate_release", 120.0))
            b.set("hush_pct", profile.get("gate_hush", 0.5) * 100)
        else:
            b = plan.add("P", "gate", "noise gate")
            b.set("umbral_db", db(thr))
        b.note = "sits just before the capture, as in SoloTone"

    # cab IR
    paths, act = profile.get("ir_paths") or [], profile.get("ir_active", 0)
    ir = paths[act] if act < len(paths) else None
    if ir:
        b = plan.add("Q", "ir", "cab IR")
        b.set("ir_slot", 0, guess=True); b.set("mezcla_pct", 100); b.set("nivel_db", 0)
        set_ir_length(b, IR_MS_DEFAULT)
        plan.uploads.append(f"IR slot {b.values['ir_slot']}: {ir}  (48 kHz mono, <=85 ms; not uploaded in dry run)")
    else:
        plan.warn("no active cab IR in the profile")

    # 5-band EQ -> 6-band graphic EQ (100/400/1k/2.5k/6k  ->  125/315/800/2k/5k)
    gains = profile.get("eq_band_gains") or []
    if use_ampeq:
        plan.ampeq, ampeq_notes = fit_ampeq(profile)
        plan.fit_notes.extend(ampeq_notes)
    elif profile.get("eq_on") and any(abs(x) > 0.05 for x in gains):
        b = plan.add("Q", "geq", "5-band EQ")
        for name, gv in zip(["g125_db", "g315_db", "g800_db", "g2k_db", "g5k_db"], gains):
            b.set(name, gv)
        b.set("nivel_db", profile.get("eq_level", 0.0))
        b.note = "nearest bands, not identical centre frequencies"
    for k in post:
        if P.get(f"{k}_on"):
            plan._code = k
            map_pedal(plan, "Q", k, P)
            plan._code = None

    ig = db(profile.get("in_gain", 1.0)); og = db(profile.get("out_gain", 1.0))
    plan.gain = (ig, og)
    plan.nam = profile.get("nam_path")
    plan.nam_status = nam_verdict(plan.nam)
    if plan.nam:
        plan.uploads.insert(0, f"capture: {plan.nam}  [{plan.nam_status}]  (not uploaded in dry run)")
    if plan.ampeq and abs(plan.ampeq.get("_extra_db", 0.0)) >= 1.0:
        b = plan.add("Q", "boost", "level the Volume range can't carry")       # last block: cuts or lifts the whole sound
        b.set("ganancia_db", max(-12.0, min(24.0, plan.ampeq["_extra_db"])))
    plan.fit_notes = plan.fit_notes + fit_plan(plan)
    if not use_ampeq and (og > 12 or og < -30):
        plan.warn(f"out_gain is {og:+.1f} dB; pedal GAIN is limited to -30..12 dB: level needs matching on the pedal")
    if not use_ampeq and (ig > 12 or ig < -30):
        plan.warn(f"in_gain is {ig:+.1f} dB, outside the pedal's -30..12 dB")
    return plan



# ───────────────────────── scenes ─────────────────────────
# A SoloTone profile can carry up to four scenes (variations: which pedals are on, their settings, EQ and gains). The pedal's
# scenes work the same way, but a scene cannot add or remove blocks, so the preset's chain holds every block any scene
# needs and each scene switches them on or off and sets their knobs (FXE / FXP / AMPEQ, then SNAPSAVE).
SCENE_CODES = ("comp", "wah", "fuzz", "dist", "od", "pitch", "phaser", "chorus", "tremolo", "rotary", "delay", "reverb")
SCENE_EXTRA = ("eq_on", "eq_band_gains", "eq_level", "in_gain", "out_gain")


def scene_slots(profile):
    """[(index, snapshot)] for the scenes a profile holds, or [] if it has fewer than two (then there is nothing to switch)."""
    blk = profile.get("scenes")
    slots = blk.get("slots") if isinstance(blk, dict) else None
    out = [(i, s) for i, s in enumerate((slots or [])[:4]) if isinstance(s, dict) and isinstance(s.get("pedals"), dict)]
    return out if len(out) >= 2 else []


def scene_profile(profile, snap):
    """The profile as it is when `snap` is the active scene."""
    p = copy.deepcopy(profile)
    p["pedals"].update(copy.deepcopy(snap["pedals"]))
    for k in SCENE_EXTRA:
        if k in snap:
            p[k] = copy.deepcopy(snap[k])
    p.pop("scenes", None)
    return p


def union_profile(profile, slots):
    """One profile in which every pedal that is on in ANY scene is on, with the settings of the first scene that uses it."""
    u = scene_profile(profile, slots[0][1])
    for code in SCENE_CODES:
        pref = code + "_"
        src = next((s for _i, s in slots if s["pedals"].get(code + "_on")), None)
        if src is None:
            continue
        for f, v in src["pedals"].items():
            if f.startswith(pref) or (code == "fuzz" and f == "muff_tone"):
                u["pedals"][f] = copy.deepcopy(v)
    return u


def _block_keys(plan):
    """A block is the same block in every scene when it comes from the same SoloTone pedal on the same side of the amp,
    whatever type the fitter made of it (a plate reverb in one scene and a shimmer in another still share one block)."""
    seen, keys = {}, {}
    for b in plan.blocks:
        k = (b.bank, b.src.split(" (")[0])
        n = seen.get(k, 0)
        seen[k] = n + 1
        keys[(k, n)] = b
    return keys


def build_scene_plan(profile, types, use_ampeq=True):
    """build_plan for a profile with scenes. Returns (plan, slots). plan.scene_states lists, per scene, the index (0..3),
    its name, which blocks are on with which values, and its capture EQ; blocks a scene does not use are off in it."""
    slots = scene_slots(profile)
    plan = build_plan(union_profile(profile, slots), types, use_ampeq)
    master = _block_keys(plan)
    names = (profile.get("scenes") or {}).get("names") or []
    plan.scene_states = []
    for idx, snap in slots:
        sp = build_plan(scene_profile(profile, snap), types, use_ampeq)
        blocks = {}
        for key, b in _block_keys(sp).items():
            m = master.get(key)
            if m is None:
                plan.warn(f"scene {'ABCD'[idx]}: {b.src} has no block in the shared chain; left out of that scene")
            else:
                vals = dict(b.values)
                if m.bt.name != b.bt.name:
                    # a block's type cannot change between scenes: the scene gets the shared block with the settings the two
                    # types have in common
                    vals = {k: v for k, v in vals.items() if k in m.values}
                    plan.warn(f"scene {'ABCD'[idx]}: {b.src} is a {m.bt.name} in the shared chain (scenes cannot change a block's type); "
                              f"only the settings both share carry over")
                blocks[id(m)] = (True, vals)
        name = names[idx] if idx < len(names) and isinstance(names[idx], str) else ""
        plan.scene_states.append(dict(index=idx, name=name.strip()[:15] or "Scene " + "ABCD"[idx], blocks=blocks, ampeq=sp.ampeq))
    return plan, slots


# ───────────────────────── capture EQ (AMPEQ) ─────────────────────────
# The pedal has a free "Gain and tone" stage on the capture: Gain (drive before the model, +-12 dB),
# Bass (low shelf 150 Hz), Mid (peak 650 Hz), Treble (high shelf 2 kHz), Presence (high shelf 4.5 kHz),
# each +-12 dB, and Volume (capture output, -24..+12 dB). It costs no CPU or memory and is saved in the
# preset, so SoloTone's input gain, 5-band EQ and output gain go there instead of into blocks.
AMPEQ_BANDS = (("lowshelf", 150.0, 0.707), ("peak", 650.0, 1.0), ("highshelf", 2000.0, 0.707), ("highshelf", 4500.0, 0.707))
SOLOTONE_EQ = (("lowshelf", 100.0, 0.707), ("peak", 400.0, 1.0), ("peak", 1000.0, 1.0), ("peak", 2500.0, 1.0),
               ("highshelf", 6000.0, 0.707))
AMPEQ_RANGE_DB = 12.0
VOLUME_RANGE_DB = (-24.0, 12.0)
VOLUME_SHIFT_DB = 8.0       # leave room above: SoloTone's loudness-matching gains run up to +35 dB, the pedal's Volume only to +12
VOLUME_REF_DB = 1.8         # measured: at equal settings the pedal's capture is ~1.8 dB quieter than SoloTone's
TONE_CORRECTION_GAIN = 0.8  # share of a measured tone error that is corrected (the rest is left as a safety margin)


def _rbj(kind, fc, gain_db, q, f, sr=48000.0):
    """Frequency response (dB) of an RBJ cookbook biquad."""
    import numpy as np
    a_ = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    cw, al = np.cos(w0), np.sin(w0) / (2 * q)
    sq = 2 * np.sqrt(a_) * al
    if kind == "peak":
        b = [1 + al * a_, -2 * cw, 1 - al * a_]; a = [1 + al / a_, -2 * cw, 1 - al / a_]
    elif kind == "lowshelf":
        b = [a_ * ((a_ + 1) - (a_ - 1) * cw + sq), 2 * a_ * ((a_ - 1) - (a_ + 1) * cw), a_ * ((a_ + 1) - (a_ - 1) * cw - sq)]
        a = [(a_ + 1) + (a_ - 1) * cw + sq, -2 * ((a_ - 1) + (a_ + 1) * cw), (a_ + 1) + (a_ - 1) * cw - sq]
    else:
        b = [a_ * ((a_ + 1) + (a_ - 1) * cw + sq), -2 * a_ * ((a_ - 1) + (a_ + 1) * cw), a_ * ((a_ + 1) + (a_ - 1) * cw - sq)]
        a = [(a_ + 1) - (a_ - 1) * cw + sq, 2 * ((a_ - 1) - (a_ + 1) * cw), (a_ + 1) - (a_ - 1) * cw - sq]
    z = np.exp(-1j * 2 * np.pi * f / sr)
    h = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
    return 20 * np.log10(np.abs(h) + 1e-12)


CALIB_PATH = pathlib.Path(__file__).resolve().parent / "calibration.json"
_CALIB = {}


def load_calibration(path=None):
    """Measured pedal-vs-SoloTone corrections (made by make_calibration.py from calibrate_ab.py runs)."""
    global _CALIB
    p = pathlib.Path(path) if path else CALIB_PATH
    try:
        _CALIB = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _CALIB = {}
    return _CALIB


# Profiles that carry no amp or cab of their own (their notes say "it runs through whatever NAM/IR you have loaded")
# are paired with a sibling profile's amp and cab when they go to the pedal, so the preset has a definite sound.
PAIRED_AMP = {"Comfortably Numb (Pink Floyd)": "Comfortably Numb (Pink Floyd) - Telecaster"}
PROFILES_DIR = pathlib.Path(__file__).resolve().parent.parent / "profiles"


def paired_profile(profile, profiles_dir=None):
    """(profile, note). A profile with no capture of its own that has a sibling in PAIRED_AMP gets the sibling's
    NAM capture, cab IRs and active IR; everything else in it (gains, EQ, pedals) is left as it was."""
    name = profile.get("name")
    if profile.get("nam_path") or name not in PAIRED_AMP:
        return profile, None
    root = pathlib.Path(profiles_dir or PROFILES_DIR)
    sib = next(iter(root.rglob(PAIRED_AMP[name] + ".stprofile")), None)
    if sib is None:
        return profile, None
    s = json.loads(sib.read_text(encoding="utf-8"))
    out = dict(profile, nam_path=s.get("nam_path"), ir_paths=s.get("ir_paths"), ir_active=s.get("ir_active", -1))
    return out, "no amp of its own: paired with the capture and cab of '%s'" % PAIRED_AMP[name]


def _nam_stem(profile):
    return pathlib.Path(str(profile.get("nam_path") or "")).stem.lower()


def _calibration_for(profile, drive):
    """(volume correction dB, tone corrections dict or None, how it was found) for this profile."""
    c = _CALIB or load_calibration()
    if not c:
        return 0.0, None, None
    ent = (c.get("profiles") or {}).get(str(profile.get("name") or ""))
    if ent and abs(ent.get("drive", 0.0) - drive) < 0.5:
        return float(ent["volume_offset"]), ent.get("tone"), "measured on this pedal"
    stem = _nam_stem(profile)
    fam = (c.get("captures") or {}).get(stem)
    if fam:
        return float(fam["volume_offset"]), fam.get("tone"), "typical for this capture"
    g = c.get("global") or {}
    return float(g.get("volume_offset", 0.0)), g.get("tone"), "typical for all captures"


def _band_error_curve(tone, f):
    """Piecewise log-frequency interpolation of the measured band errors (dB, pedal minus SoloTone)."""
    import numpy as np
    if not tone:
        return np.zeros_like(f)
    pts = [(80.0, tone["low"]), (320.0, tone["lowmid"]), (1300.0, tone["mid"]), (4500.0, tone["high"])]
    return np.interp(np.log(f), np.log([p[0] for p in pts]), [p[1] for p in pts])


def fit_ampeq(profile, use_calibration=True):
    """SoloTone's input gain / 5-band EQ / output gain -> the pedal's six capture values (dB).
    Returns (values dict, notes list). The EQ is fitted by least squares over 60 Hz - 12 kHz.
    With a calibration file the Volume and tone are corrected by what was measured on the pedal."""
    import numpy as np
    notes = []
    f = np.geomspace(60.0, 12000.0, 120)
    gains = list(profile.get("eq_band_gains") or [0, 0, 0, 0, 0])[:5]
    eq_on = bool(profile.get("eq_on")) and any(abs(x) > 0.05 for x in gains)
    target = np.zeros_like(f)
    if eq_on:
        for (kind, fc, q), g in zip(SOLOTONE_EQ, gains):
            target += _rbj(kind, fc, g, q, f)
    ig = 20 * math.log10(max(float(profile.get("in_gain", 1.0)), 1e-9))
    drive = max(-AMPEQ_RANGE_DB, min(AMPEQ_RANGE_DB, ig))
    vol_corr, tone_corr, how = (0.0, None, None)
    if use_calibration:
        vol_corr, tone_corr, how = _calibration_for(profile, drive)
        if tone_corr:                                   # the pedal measured too bright/dark here: aim the opposite way
            target = target - TONE_CORRECTION_GAIN * _band_error_curve(tone_corr, f)
    basis = np.stack([_rbj(k, fc, 1.0, q, f) for k, fc, q in AMPEQ_BANDS] + [np.ones_like(f)], axis=1)
    sol, *_ = np.linalg.lstsq(basis, target, rcond=None)
    bands = np.clip(sol[:4], -AMPEQ_RANGE_DB, AMPEQ_RANGE_DB)
    resid = float(np.sqrt(np.mean((basis[:, :4] @ bands + sol[4] - target) ** 2)))
    const = float(sol[4])                                   # overall level the fitted curve carries; compensated in Volume
    if eq_on:
        notes.append("EQ: fitted to the pedal's capture tone controls (Bass/Mid/Treble/Presence), average error %.1f dB" % resid)
    og = 20 * math.log10(max(float(profile.get("out_gain", 1.0)), 1e-9))
    if abs(ig - drive) > 0.05:
        notes.append("input gain %+.1f dB limited to the pedal's %+.0f dB Gain range" % (ig, drive))
    level = og + (profile.get("eq_level", 0.0) if eq_on else 0.0) - const + VOLUME_REF_DB - VOLUME_SHIFT_DB
    raw_level = level                                   # what the plain formula says, before the measured correction
    level += vol_corr
    if how:
        notes.append("level matched to SoloTone (%+.1f dB, %s)" % (vol_corr, how))
    volume = max(VOLUME_RANGE_DB[0], min(VOLUME_RANGE_DB[1], level))
    extra = level - volume                              # what the Volume range cannot carry; a Boost block takes it
    if abs(extra) > 0.05:
        notes.append("output level %+.1f dB is outside the pedal's Volume range; %+.1f dB is carried by a Boost block"
                     % (level, max(-12.0, min(24.0, extra))))
    vals = dict(volume=volume, bass=float(bands[0]), mid=float(bands[1]), treble=float(bands[2]),
                presence=float(bands[3]), drive=drive)
    vals["_raw_level"] = raw_level                      # for make_calibration.py; never sent to the pedal
    vals["_extra_db"] = extra
    return vals, notes


def ampeq_command(vals):
    """'AMPEQ volume bass mid treble presence drive' with dB x10 integers, as the control page sends it."""
    order = ("volume", "bass", "mid", "treble", "presence", "drive")
    return "AMPEQ " + " ".join(str(int(round(vals[k] * 10))) for k in order)


# ───────────────────────── automatic fitting ─────────────────────────
FIT_TARGET = 0.84       # share of the CPU budget a sound may use after fitting
HARD_LIMIT = 0.95       # above this even the fitted sound is refused


def _swap(plan, blk, tname):
    nb = Block(blk.bank, plan.types[tname], blk.src, blk.note)
    plan.blocks[plan.blocks.index(blk)] = nb
    return nb


def _eq_from_geq(plan, g):
    e = _swap(plan, g, "eq")
    v = g.values
    e.set("graves_db", v["g125_db"])
    mids = [(400, v["g315_db"]), (1000, v["g800_db"]), (2500, v["g2k_db"])]
    hz, gain = max(mids, key=lambda m: abs(m[1]))
    e.set("medios_hz", hz); e.set("medios_db", gain)
    e.set("agudos_db", v["g5k_db"])
    e.note = "3 bands + one sweepable mid (cheaper than the 6-band graphic EQ)"
    return e


def fit_plan(plan, target=FIT_TARGET, mem_pool=None):
    """Shrink a plan to fit the pedal, least audible change first, and return a
    list of plain-language notes describing what was changed. Nothing here is
    shown as an error: the user just gets the closest sound that runs."""
    notes = []
    pool = (mem_pool or MEM_POOL) - MEM_MARGIN

    def cpu_over():
        return totals(plan)[0] > plan_budget(plan) * target

    def mem_over():
        return totals(plan)[1] > pool

    def over():
        return cpu_over() or mem_over()

    reduced = []
    # 1. (only without AMPEQ) 6-band graphic EQ -> 3 bands + sweepable mid (saves ~2300 ticks)
    if cpu_over():
        for g in [b for b in plan.blocks if b.bt.name == "geq"]:
            _eq_from_geq(plan, g)
            notes.append("EQ: used the pedal's 3-band + sweepable-mid EQ instead of the 6-band one (saves CPU)")
    # 2. lower oversampling on drive/fuzz blocks, biggest saving first (a little more aliasing on hot signals)
    while cpu_over():
        cands = [b for b in plan.blocks if "os_n" in b.values and b.values["os_n"] > 0]
        if not cands:
            break
        b = max(cands, key=lambda x: x.bt.cost_at(x.values["os_n"]) - x.bt.cost_at(0))
        reduced.append((b, b.values["os_n"]))
        b.values["os_n"] = 0
        notes.append("%s: oversampling turned off to save CPU (slightly more fizz when played very hard)" % b.bt.name)
    # 3. use a little less of the cab IR (21 -> 10 ms keeps the cab's character; saves ~8 KB and ~1500 ticks)
    for ms in (10,):
        if over():
            for b in [b for b in plan.blocks if b.bt.name == "ir" and getattr(b, "ir_ms", 0) > ms]:
                set_ir_length(b, ms)
                notes.append("cab IR: using the first %d ms of the IR to save CPU and memory" % ms)
    # 4. cheaper reverb with the same job
    if cpu_over():
        for r in [b for b in plan.blocks if b.bt.name == "reverb2"]:
            n = _swap(plan, r, "plate")
            for k in ("tamano_pct", "decay_pct", "mezcla_pct", "predelay_ms", "amort_pct"):
                n.values[k] = r.values[k]
            notes.append("reverb: plate reverb used instead of the room reverb (cheaper)")
    # 4b. still too much CPU: the spectral reverb is the cheapest big reverb (5400 ticks, 20 KB); alpha 12 removed it
    if cpu_over() and "spectral" in plan.types:
        for r in [x for x in plan.blocks if x.bt.name in ("reverb2", "plate", "hybrid", "cloud", "shimmer")]:
            n = _swap(plan, r, "spectral")
            for k, v in r.values.items():
                if k in {p_.name for p_ in n.bt.params} and k not in ("er_pct", "drip_pct", "resortes_n"):
                    n.values[k] = max(n.bt.param(k).lo, min(n.bt.param(k).hi, v))
            notes.append("reverb: spectral reverb used (much lighter on the pedal's CPU)")
    # 5. memory: step the reverb down through types that need less room (room 54 KB -> hybrid 41 KB
    #    -> spectral 20 KB -> spring 6 KB) until the chain fits the pedal's memory
    for nxt in [t for t in ("hybrid", "spectral", "spring") if t in plan.types]:
        if not mem_over() and not (cpu_over() and nxt != "hybrid"):
            break
        for r in [x for x in plan.blocks if x.bt.name in ("reverb2", "plate", "hybrid", "spectral", "cloud", "shimmer")]:
            if plan.types[nxt].mem >= r.bt.mem:
                continue
            n = _swap(plan, r, nxt)
            for k, v in r.values.items():
                if k in {p.name for p in n.bt.params} and k not in ("er_pct", "drip_pct", "resortes_n"):
                    n.values[k] = max(n.bt.param(k).lo, min(n.bt.param(k).hi, v))
            notes.append("reverb: %s reverb used (the pedal doesn't have memory left for the other types)" % nxt)
    # 5b. only now cut the cab IR hard
    if over():
        for b in [b for b in plan.blocks if b.bt.name == "ir" and getattr(b, "ir_ms", 0) > IR_MS_MIN]:
            set_ir_length(b, IR_MS_MIN)
            notes.append("cab IR: using the first %d ms of the IR to save CPU and memory" % IR_MS_MIN)
    # 6. drop the least important blocks
    for name, why in (("chorus", "chorus/flanger left out"), ("hum", "hum filter left out")):
        if over():
            for b in [b for b in plan.blocks if b.bt.name in (name, "flanger") and name == "chorus" or b.bt.name == name]:
                plan.blocks.remove(b)
                notes.append("%s: %s to fit the pedal" % (b.bt.name, why))
    # later steps may have made room: give oversampling back where it fits (smallest cost first)
    for b, old_os in sorted(reduced, key=lambda r: r[0].bt.cost_at(r[1]) - r[0].bt.cost_at(0)):
        if b not in plan.blocks:
            continue
        b.values["os_n"] = old_os
        if cpu_over():
            b.values["os_n"] = 0
        else:
            notes = [n for n in notes if not n.startswith(b.bt.name + ": oversampling")]
    for prefix in ("reverb:", "cab IR:"):       # several steps in a row: only the final one matters to the user
        rn = [n for n in notes if n.startswith(prefix)]
        if len(rn) > 1:
            notes = [n for n in notes if not n.startswith(prefix)] + [rn[-1]]
    return notes


def plan_budget(plan):
    # alpha 13 (catalogue with the pingpong delay): the engine splits its work over both cores and the pedal reports ~49,500
    # ticks for effects instead of ~30,500; this is only the dry-run default, a send reads the real figure from the pedal
    if "pingpong" in plan.types and not getattr(plan, "pedal_budget", None):
        return 44000 if any(b.bt.name == "ir" for b in plan.blocks) else 49000
    """CPU ticks the pedal has for effects. Read from the pedal when known (plan.pedal_budget): on alpha 11 with the factory
    Stereo setting it was ~20,000 on 2026-11-17, not the ~30,500 measured on alpha 9.1, and it shrinks a little as blocks are added."""
    if getattr(plan, "pedal_budget", None):
        return plan.pedal_budget
    return CPU_BUDGET_IR if any(b.bt.name == "ir" for b in plan.blocks) else CPU_BUDGET


def totals(plan):
    """(CPU ticks, memory bytes). A bypassed block costs no CPU on the pedal (measured 2026-11-17: rotary + delay + Klon, 13207
    ticks on, 0 off) but keeps its memory, so with scenes the CPU is that of the heaviest scene and the memory that of the chain."""
    scenes = getattr(plan, "scene_states", None)
    if scenes:
        cpu = max(sum(b.cost() for b in plan.blocks if id(b) in sc["blocks"]) for sc in scenes)
    else:
        cpu = sum(b.cost() for b in plan.blocks)
    mem = sum(b.bt.mem + b.extra_mem for b in plan.blocks)
    return cpu, mem


# ───────────────────────── output ─────────────────────────
def show(plan, profile, name):
    cpu, mem = totals(plan)
    print(f"=== {profile.get('name', name)}")
    print(f"tuning: {profile.get('tuning')}   capture: {plan.nam_status}")
    bud = plan_budget(plan)
    print(f"estimated pedal load: {cpu}/{bud} ticks ({cpu / bud:.0%}),  effects memory {mem}/{MEM_POOL} B ({mem / MEM_POOL:.0%})\n")
    print("Commands a live run would send (ids assume an empty chain; the pedal assigns the real ones):")
    for i, b in enumerate(plan.blocks):
        print(f"  FXA {b.bank} 99 {b.bt.id:<3d} # id {i}: {b.bt.name:<9s} <- {b.src}  (~{b.cost()} ticks)")
        for p in b.bt.params:
            v = b.values[p.name]
            if v != p.default:
                print(f"  FXP {i} {p.idx} {v:<6d} # {p.name}")
        if b.note:
            print(f"      note: {b.note}")
    if plan.ampeq:
        print("  " + ampeq_command(plan.ampeq) + "   # capture Volume/Bass/Mid/Treble/Presence/Gain (dB x10)")
    print(f"  GAIN {plan.gain[0]:+.1f} {plan.gain[1]:+.1f}  # dB in/out as SoloTone has them (would be clamped to -30..12)")
    for sc in getattr(plan, "scene_states", []):
        print(f"  scene {'ABCD'[sc['index']]} '{sc['name']}': {sum(1 for v in sc['blocks'].values() if v[0])} of {len(plan.blocks)} blocks on"
              f" ({', '.join(b.bt.name for b in plan.blocks if id(b) in sc['blocks'])})")
    if plan.uploads:
        print("\nUploads needed:")
        for u in plan.uploads:
            print("  -", u)
    if plan.fit_notes:
        print("\nFitted to the pedal (automatic):")
        for n in plan.fit_notes:
            print("  ~", n)
    allw = list(plan.warnings)
    if cpu > plan_budget(plan) * CPU_WARN:
        allw.append(f"CPU estimate {cpu / plan_budget(plan):.0%} is over the {CPU_WARN:.0%} comfort line: lower oversampling or drop a block")
    if mem > MEM_POOL:
        allw.append(f"effects memory {mem} B exceeds the {MEM_POOL} B pool: the pedal will refuse a block")
    for b in plan.blocks:
        for gmsg in sorted(set(b.guesses)):
            allw.append(f"{b.bt.name}: {gmsg}")
    if allw:
        print("\nWarnings / guesses:")
        for w in allw:
            print("  !", w)
    print()


def summary_line(plan, profile, name):
    cpu, mem = totals(plan)
    kinds = "+".join(b.bt.name for b in plan.blocks)
    flags = len(plan.warnings) + (cpu > plan_budget(plan) * CPU_WARN) + (mem > MEM_POOL)
    print(f"{name[:52]:<52s} {cpu / plan_budget(plan):5.0%} cpu {mem / 1024:5.1f} kB  {plan.nam_status[:22]:<22s} !{flags}  {kinds}")



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a .stprofile file, or with --all a folder of them")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--catalogue", default=None, help="a file with your pedal's FXT reply (default: fxt_catalogue.txt if it exists)")
    ap.add_argument("--port", metavar="COMPORT", default=None, help="read the block catalogue from the pedal itself (read-only)")
    ap.add_argument("--live", metavar="COMPORT", help="send the plan to the pedal's live (unsaved) chain")
    ap.add_argument("--capture", metavar="COMPORT", help="convert the profile's NAM capture and load it into the pedal (not saved)")
    ap.add_argument("--clear", metavar="COMPORT", help="remove every block from the pedal's live chain")
    a = ap.parse_args()
    if a.clear:
        import live; return live.clear_chain(a.clear)
    if a.port:
        import live
        ped = live.Pedal(a.port, allow={"PING", "FXT"})
        try:
            ped.cmd("PING", total=1.0)
            types = parse_catalogue(ped.cmd("FXT", total=15.0, until=lambda l: l in ("FXT end", "FXT fin")))
        finally:
            ped.s.close()
    else:
        cat = pathlib.Path(a.catalogue) if a.catalogue else HERE / "fxt_catalogue.txt"
        if not cat.exists():
            sys.exit("No block catalogue: pass --port COMx (reads it from your pedal) or --catalogue FILE with its FXT reply "
                     "(python ultrarig_probe.py writes one).")
        types = load_catalogue(cat)
    t = pathlib.Path(a.target)
    files = sorted(t.rglob("*.stprofile")) if a.all else [t]
    for f in files:
        prof = json.loads(f.read_text(encoding="utf-8"))
        plan = build_scene_plan(prof, types)[0] if scene_slots(prof) else build_plan(prof, types)
        if a.capture:
            if not prof.get("nam_path"):
                sys.exit("This profile has no NAM capture set.")
            import live; return live.upload_capture(prof["nam_path"], a.capture)
        if a.live:
            show(plan, prof, f.stem)
            import live; return live.go_live(plan, a.live)
        (summary_line if a.all else show)(plan, prof, f.stem)


if __name__ == "__main__":
    main()
