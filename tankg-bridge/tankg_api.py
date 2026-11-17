"""High-level Tank-G (UltraRig) API used by SoloTone's optional "Tank-G" tab.

SoloTone loads this file only if the tankg-bridge folder exists next to it, so the
UltraRig-derived pieces (the protocol code; the capture converter is now a2l1.py, MIT, ported from PulseDeck) never have to ship with SoloTone.

    ok, why = available()
    ports = find_ports()
    state = read_state(port)                       # presets / captures / IRs already on the pedal
    send_sound(profile_dict, port, preset_slot=31, save=True, overwrite_preset=False,
               audio_loader=fn, log=print, progress=fn)

Safety rules enforced here (not just in the UI):
  * never deletes anything (no PDEL / CDEL / IDEL / PNEW / installer commands exist in the allow-list);
  * never overwrites a capture or IR slot: it takes a free one (or re-uses one with the same name and size);
  * overwrites a preset slot only when overwrite_preset=True;
  * on any failure the load buffers are released and audio is switched back on.
"""
import hashlib
import math
import pathlib
import re
import sys
import time
import zlib

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PEDAL_VID = 0x3654
ALLOW = {"PING", "LVL", "FXT", "FXS", "FXA", "FXP", "FXR", "MODEL", "PLIST", "CLIST", "ILIST",
         "AUDIO", "LDBEGIN", "LDB", "LDEND", "CTRIM", "FXOS", "AMPEQ",
         "IRBEGIN", "IRB", "IREND", "ISAVE", "CSAVE", "PSAVE", "CLOAD", "FXE", "SNAPS", "SNAPSAVE", "SNAPNAME", "SNAP", "TUNER"}
HEADROOM_LIMIT = 0.92                      # measured CPU share above which a lighter version is tried
IR_CAP, IR_CAP_LAST = 4076, 2028           # samples at 48 kHz; slot 7 holds less


class TankGError(Exception):
    pass


# ───────────────────────── discovery ─────────────────────────
def available():
    try:
        import serial  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        return False, "missing Python package: %s (pip install pyserial)" % e.name
    for f in ("bridge.py", "live.py", "a2l1.py"):
        if not (HERE / f).exists():
            return False, "tankg-bridge is missing %s" % f
    return True, ""


def find_ports():
    """[(device, description, looks_like_pedal)] with the pedal first."""
    from serial.tools import list_ports
    out = [(p.device, p.description, p.vid == PEDAL_VID) for p in list_ports.comports()]
    return sorted(out, key=lambda r: (not r[2], r[0]))


def slot_label(n):
    """Preset slot number 0..31 -> '1A'..'8D' (the numbering PSAVE/PLOAD use)."""
    return "%d%s" % (n // 4 + 1, "ABCD"[n % 4])


def _clean(name, limit=23):
    name = re.sub(r"[^\x20-\x7e]", "_", name or "").replace("'", " ").strip()
    return (name or "SoloTone")[:limit]


def _library_name(label, data):
    """A capture/IR library name that identifies the CONTENT: the file's name (trimmed) plus a 4-hex-digit hash of
    the bytes sent, 23 characters at most. Two different files can never share one, and the same file always does."""
    # SHA-1, not CRC-32: a capture blob ends with its own CRC-32, so the CRC of the whole blob is the same
    # constant (0x2144DF1C) for every capture and would tell nothing apart.
    tag = hashlib.sha1(bytes(data)).hexdigest()[:4]
    return _clean(label, 18).rstrip() + " " + tag


# ───────────────────────── pedal session ─────────────────────────
def _open(port, transport=None):
    import live
    ped = live.Pedal(port, allow=ALLOW, transport=transport)
    for _ in range(5):
        if any(l.startswith("PONG") for l in ped.cmd("PING", total=1.0)):
            return ped
    raise TankGError("No answer from the pedal on %s. Is it running UltraRig, and is the UltraRig "
                     "control page / app closed?" % port)


def _tuner_cmd(ped, on, ref=440):
    """The pedal's own tuner: on (output muted, effects off, shown on its display/LEDs) or off (sound comes back)."""
    cmd = "TUNER 1 1 %d" % int(ref) if on else "TUNER 0"
    r = ped.cmd(cmd, total=3.0, until=lambda l: l.startswith("TUNER"))
    if not any(l.startswith("TUNER") for l in r):
        raise TankGError("The pedal did not answer the tuner command: " + " | ".join(r))
    return r


def tuner(port, on, ref=440, transport=None):
    """Turn the pedal's tuner on or off (opens the port for a moment; use LiveSession.tuner while a live session holds it)."""
    ped = _open(port, transport)
    try:
        return _tuner_cmd(ped, on, ref)
    finally:
        if transport is None:
            try:
                ped.s.close()
            except Exception:
                pass


def _lib(ped, cmd):
    """Parse CLIST / ILIST -> ({slot: (name, size)}, slot_count)."""
    rows, slots = {}, 8
    lines = ped.cmd(cmd, total=4.0, until=lambda l: l.startswith(cmd + " end") or l.startswith(cmd + " fin"))
    for l in lines:
        m = re.match(r"^(?:CAP|IRS) (\d+) (\d+) '(.*?)'", l)
        if m:
            rows[int(m[1])] = (m[3], int(m[2]))
        m = re.search(r"\bslots=(\d+)", l)
        if m and l.startswith(cmd + " "):
            slots = int(m[1])
    return rows, slots


def _plist(ped):
    presets = {}
    lines = ped.cmd("PLIST", total=4.0, until=lambda l: l.startswith("PLIST end") or l.startswith("PLIST fin"))
    if not any(l.startswith("PLIST end") or l.startswith("PLIST fin") for l in lines):
        raise TankGError("The pedal's preset list didn't arrive complete; try again.")
    for l in lines:
        m = re.match(r"^PRE (\d+) (\d+) .*?'(.*?)'", l)
        if m:
            presets[int(m[1])] = m[3]
    return presets


def read_state(port, transport=None):
    ped = _open(port, transport)
    pong = next((l for l in ped.cmd("PING", total=1.0) if l.startswith("PONG")), "")
    caps, cap_slots = _lib(ped, "CLIST")
    irs, ir_slots = _lib(ped, "ILIST")
    import live
    chain, cpu = live.read_chain(ped)
    return dict(firmware=pong, presets=_plist(ped), caps=caps, cap_slots=cap_slots, irs=irs,
                ir_slots=ir_slots, chain_blocks=len(chain), cpu=cpu)


# ───────────────────────── IR preparation (port of prepIr in tankg.html) ─────────────────────────
def prep_ir(src48, n48):
    import numpy as np
    x = np.asarray(src48[:n48], dtype=np.float32).copy()
    fade = min(len(x) >> 3, 256)
    if fade:
        i = np.arange(fade, dtype=np.float32)
        x[len(x) - fade:] *= 0.5 * (1 + np.cos(np.pi * i / (fade - 1 or 1)))
    e = float(np.sqrt(np.sum(x.astype(np.float64) ** 2)))
    if not e > 0:
        raise TankGError("the IR is empty")
    x /= e
    pk = float(np.max(np.abs(x)))
    q = np.floor(x / pk * 32767 + 0.5).astype("<i2")
    n = len(q)
    while n > 1 and q[n - 1] == 0:
        n -= 1
    q = q[:n]
    gain = int(math.floor(20000 * math.log10(pk * 32768 / 32767) + 0.5))
    return q, gain


def _apply_state(ped, plan, blocks, ampeq, state, say, label, skip=()):
    """Put the chain in a given state: `blocks` is {id(block): (True, values)}; every other block goes off. `state` is what
    the pedal is in now ({"blocks": {id: (on, values)}, "ampeq": command or None}) and is updated as commands succeed, so only
    the differences are sent. Blocks that go off first, so the ones that come on never push the pedal over its CPU limit for
    a moment; a block the pedal has no CPU for stays off (said, not fatal). Parameters named in `skip` are left alone."""
    import bridge
    for b in sorted(plan.blocks, key=lambda x: id(x) in blocks):
        on, vals = blocks.get(id(b), (False, None))
        cur_on, cur_vals = state["blocks"][id(b)]
        if on != cur_on:
            r = ped.cmd("FXE %d %d" % (b.live_id, 1 if on else 0), total=4.0, until=lambda l: l.startswith("FXE"))
            if not any(l.startswith("FXE ok") for l in r):
                if on and re.search(r"cpu", " ".join(r), re.I):
                    say("note: %s: the %s stays off (the pedal has not enough CPU for it there)" % (label, b.bt.name))
                    continue
                raise TankGError("the pedal refused FXE on the %s block: %s" % (b.bt.name, " | ".join(r)))
            cur_on = on
        for p in b.bt.params:
            new = (vals or {}).get(p.name, cur_vals.get(p.name))
            if p.name in skip or new is None or new == cur_vals.get(p.name):
                continue
            if p.name == "os_n":
                rr = ped.cmd("FXOS %d %d" % (b.live_id, new), total=4.0, until=lambda l: l.startswith("FXOS"))
                ok = any(l.startswith("FXOS ok") for l in rr)
            else:
                rr = ped.cmd("FXP %d %d %d" % (b.live_id, p.idx, new), total=3.0, until=lambda l: l.startswith("FXP"))
                ok = any(l.startswith("FXP ok") for l in rr)
            if not ok:
                raise TankGError("the pedal refused %s.%s=%d: %s" % (b.bt.name, p.name, new, " | ".join(rr)))
            cur_vals[p.name] = new
        state["blocks"][id(b)] = (cur_on, cur_vals)
    if ampeq:
        cmd = bridge.ampeq_command(ampeq)
        if cmd != state.get("ampeq"):
            r = ped.cmd(cmd, total=3.0, until=lambda l: l.startswith("AMPEQ"))
            if not any(re.match(r"^AMPEQ (volume|gain)=", l) for l in r):
                raise TankGError("the pedal refused the capture EQ: " + " | ".join(r))
            state["ampeq"] = cmd


def _save_scenes(ped, plan, say):
    """After the preset is saved: put the chain in each scene's state (blocks on or off, knobs, capture EQ) and SNAPSAVE it.
    A scene cannot add or remove blocks, so the chain holds every block any scene needs. The preset itself is already
    saved; a scene that fails is reported, it never undoes the preset. Returns the scene indexes that were saved."""
    import bridge
    first = plan.scene_states[0]["blocks"]
    state = dict(blocks={id(b): (id(b) in first, dict(b.values)) for b in plan.blocks},   # the chain as it was built: scene 1's blocks on
                 ampeq=bridge.ampeq_command(plan.ampeq) if plan.ampeq else None)
    saved = []
    for sc in plan.scene_states:
        letter = "ABCD"[sc["index"]]
        try:
            _apply_state(ped, plan, sc["blocks"], sc["ampeq"], state, say, "scene " + letter)
            r = ped.cmd("SNAPSAVE %d" % sc["index"], total=8.0, until=lambda l: l.startswith(("SNAPSAVE", "SNAPS", "ERR")))
            if not r or any(l.startswith("ERR") for l in r):
                raise TankGError("SNAPSAVE: " + " | ".join(r or ["no answer"]))
            ped.cmd("SNAPNAME %d %s" % (sc["index"], sc["name"]), total=3.0, until=lambda l: l.startswith(("SNAPNAME", "SNAPS", "ERR")))
            saved.append(sc["index"])
            say("scene %s '%s' saved" % (letter, sc["name"]))
        except TankGError as e:
            say("note: scene %s was not saved: %s" % (letter, e))
    if saved:
        r = ped.cmd("SNAPS", total=3.0, until=lambda l: l.startswith("SNAPS"))
        say("scenes on the pedal: " + " | ".join(r))
    return saved


def _build_chain(ped, plan, added, live):
    """FXA every block, set its parameters (oversampling goes through FXOS), then read the chain
    back and check it. Returns the pedal's CPU line."""
    for b in plan.blocks:
        r = ped.cmd("FXA %s 99 %d" % (b.bank, b.bt.id), total=6.0)
        line = next((l for l in r if l.startswith("FXA id=")), None)
        if not line:
            raise TankGError("The pedal refused the %s block: %s" % (b.bt.name, " | ".join(r)))
        b.live_id = int(re.search(r"id=(\d+)", line)[1])
        added.append(b.live_id)
        scenes = getattr(plan, "scene_states", None)
        if scenes and id(b) not in scenes[0]["blocks"]:
            # used by a later scene only: a bypassed block costs no CPU (measured), so the chain fits as long as the
            # heaviest SCENE does. It is switched on by that scene.
            rr = ped.cmd("FXE %d 0" % b.live_id, total=4.0, until=lambda l: l.startswith("FXE"))
            if not any(l.startswith("FXE ok") for l in rr):
                raise TankGError("The pedal refused to bypass the %s block: %s" % (b.bt.name, " | ".join(rr)))
        for p in b.bt.params:
            v = b.values[p.name]
            if v == p.default:
                continue
            if p.name == "os_n":
                rr = ped.cmd("FXOS %d %d" % (b.live_id, v), total=4.0, until=lambda l: l.startswith("FXOS"))
                if not any(l.startswith("FXOS ok") for l in rr):
                    raise TankGError("The pedal refused oversampling %d on the %s block: %s" % (v, b.bt.name, " | ".join(rr)))
            else:
                rr = ped.cmd("FXP %d %d %d" % (b.live_id, p.idx, v), total=3.0, until=lambda l: l.startswith("FXP"))
                if not any(l.startswith("FXP ok") for l in rr):
                    raise TankGError("The pedal refused %s.%s=%d: %s" % (b.bt.name, p.name, v, " | ".join(rr)))
    chain, cpu_line = live.read_chain(ped)
    bad = 0
    for b in plan.blocks:
        row = next((c for c in chain if c["id"] == b.live_id), None)
        if not row or any(row["p"][p.idx] != b.values[p.name] for p in b.bt.params):
            bad += 1
    if bad:
        raise TankGError("%d block(s) did not read back as sent; nothing was saved." % bad)
    return cpu_line


# ───────────────────────── the main entry point ─────────────────────────
def send_sound(profile, port, preset_slot=None, save=True, overwrite_preset=False, audio_loader=None,
               log=print, progress=None, transport=None, preset_name=None, all_scenes=False, audio_on=True):
    """Put a SoloTone profile dict on the pedal. save=False just plays it (live, nothing written)."""
    import bridge, live, a2l1
    import numpy as np

    say = lambda s: log(s)
    prog = progress or (lambda f, s="": None)
    ped = _open(port, transport)
    # the block catalogue comes from the pedal itself (read-only FXT), so it always matches the installed firmware
    types = bridge.parse_catalogue(ped.cmd("FXT", total=15.0, until=lambda l: l in ("FXT end", "FXT fin")))
    ampeq_ok = any(re.match(r"^AMPEQ (volume|gain)=", l) for l in ped.cmd("AMPEQ", total=2.0, until=lambda l: l.startswith("AMPEQ")))   # bare AMPEQ = read the values
    # A profile with scenes becomes ONE chain holding every block any scene uses, plus the scenes themselves (only when
    # the sound is saved: scenes belong to a stored preset). Playing without saving sends the active scene alone.
    with_scenes = bool((save or all_scenes) and bridge.scene_slots(profile))
    plan = bridge.build_scene_plan(profile, types, ampeq_ok)[0] if with_scenes else bridge.build_plan(profile, types, use_ampeq=ampeq_ok)
    for l in ped.cmd("FXS", total=3.0, until=lambda l: l.startswith(("FXS peak=", "FXS pico="))):
        m = re.match(r"FXS cpu=\d+/(\d+) ticks", l)
        if m and 8000 < int(m[1]) < 40000:
            plan.pedal_budget = int(m[1])
            say("the pedal's CPU budget for effects right now: %d ticks" % plan.pedal_budget)
    if getattr(plan, "pedal_budget", None):
        for n in bridge.fit_plan(plan):
            say("fitted: " + n)
    if with_scenes:
        say("scenes: %s in one shared chain" % ", ".join("%s %s" % ("ABCD"[s["index"]], s["name"]) for s in plan.scene_states))
    if not ampeq_ok:
        say("note: this firmware has no capture EQ, so EQ blocks are used instead")
    cpu, mem = bridge.totals(plan)
    for n in plan.fit_notes:
        say("fitted: " + n)
    if cpu > bridge.plan_budget(plan) * bridge.HARD_LIMIT or mem > bridge.MEM_POOL:
        raise TankGError("This sound is too big for the pedal even after simplifying it (about %d%% of its CPU). "
                         "Nothing was sent." % (cpu * 100 // bridge.plan_budget(plan)))
    for w in plan.warnings:
        say("note: " + w)

    nam_path = profile.get("nam_path")
    blob = None
    if nam_path:
        if not pathlib.Path(nam_path).exists():
            raise TankGError("The NAM capture isn't on this computer: %s" % nam_path)
        prog(0.02, "Converting the capture")
        try:
            blob = a2l1.convert(nam_path)
        except a2l1.NamError as e:
            raise TankGError("This capture can't run on the pedal: %s" % e)
        say("capture: %s, %d layers, %d bytes" % (blob["name"], blob["layers"], len(blob["blob"])))
    else:
        say("note: the profile has no NAM capture; the pedal keeps whatever capture is loaded")

    ir_block = next((b for b in plan.blocks if b.bt.name == "ir"), None)
    ir_src = None
    if ir_block is not None:
        paths = profile.get("ir_paths") or []
        irp = paths[profile.get("ir_active", 0)]
        if audio_loader is None:
            raise TankGError("No audio loader was supplied for the cab IR.")
        if not pathlib.Path(irp).exists():
            raise TankGError("The cab IR isn't on this computer: %s" % irp)
        ir_src = np.asarray(audio_loader(irp), dtype=np.float32)       # mono, already at 48 kHz
        say("cab IR: %s (%d samples at 48 kHz)" % (pathlib.Path(irp).name, len(ir_src)))

    caps, cap_slots = _lib(ped, "CLIST")
    irs, ir_slots = _lib(ped, "ILIST")
    presets = _plist(ped)
    name = _clean(preset_name or profile.get("name") or "SoloTone")
    if save:
        if preset_slot is None or not 0 <= preset_slot <= 31:
            raise TankGError("Pick a preset slot first.")
        if preset_slot in presets and not overwrite_preset:
            raise TankGError("Preset %s ('%s') already exists; not overwritten." %
                             (slot_label(preset_slot), presets[preset_slot]))

    # Which library slots to use (never an occupied one, unless it already holds this very capture / IR).
    cap_slot = ir_slot = None
    cap_reused = False
    if save and blob:
        # Library slots are matched by CONTENT, not by the name stored inside the .nam: several captures are
        # simply called "NAM", and a name+size match would send a preset to someone else's amp.
        cname = _library_name(pathlib.Path(nam_path).stem, blob["blob"])
        same = [s for s, (nm, sz) in caps.items() if nm == cname and sz == len(blob["blob"])]
        free = [s for s in range(cap_slots) if s not in caps]
        if same:
            cap_reused = True
            cap_slot = same[0]
            say("capture library: re-using slot %d (same name and size)" % (cap_slot + 1))
        elif free:
            cap_slot = free[0]
        else:
            raise TankGError("The pedal's capture library is full (%d slots). Free one in the UltraRig page first." % cap_slots)
    ir_reused = False
    iname = None
    if save and ir_block is not None:
        irstem = pathlib.Path(paths[profile.get("ir_active", 0)]).stem
        q0, _g0 = prep_ir(ir_src, min(IR_CAP, ir_block.ir_samples or IR_CAP))
        iname = _library_name(irstem, q0.tobytes())
        same = [s for s, (nm, sz) in irs.items() if nm == iname and sz == len(q0) * 2]
        free = [s for s in range(0, 7) if s not in irs]
        if same:
            ir_reused, ir_slot = True, same[0]
            say("IR library: re-using slot %d (identical IR)" % (ir_slot + 1))
        else:
            ir_slot = free[0] if free else (7 if 7 not in irs else None)
            if ir_slot is None:
                raise TankGError("The pedal's IR library is full. Free one in the UltraRig page first.")

    audio_was_on = None
    scenes_saved = []
    ldopen = irbuf = audio_off = False
    try:
        lvl = " ".join(ped.cmd("LVL"))
        m = re.search(r" m=(\d)", lvl)
        # A sound that has just been sent must be heard: if the pedal was idle (audio off, empty chain, as on a fresh
        # pedal) the send used to leave it silent. Always bring audio back on after the capture load.
        audio_was_on = True
        # Nothing is heard while the sound is being built: blocks start at their default settings (a delay or boost at
        # defaults, a capture with the old gain) and that used to squeal through the speakers. Audio comes back on at
        # the end, once the chain, the capture EQ and the scene are set.
        ped.cmd("AUDIO OFF", total=5.0, until=lambda l: l.startswith("AUDIO"))
        audio_off = True

        # 1. empty the live chain (the sound being sent replaces what is playing)
        prog(0.05, "Clearing the live chain")
        chain, _ = live.read_chain(ped)
        for c in reversed(chain):
            ped.cmd("FXR %d" % c["id"])

        # 2. capture
        if blob:
            data = blob["blob"]
            prog(0.10, "Loading the capture")
            ldopen = True
            r = ped.cmd("LDBEGIN %d" % len(data), total=5.0, until=lambda l: l.startswith("OK LDBEGIN"))
            if not any(l.startswith("OK LDBEGIN") for l in r):
                raise TankGError("The pedal refused the capture: " + " | ".join(r))
            ch = max(1, (190 - len("LDB") - 2 - len(str(len(data)))) // 2)
            for off in range(0, len(data), ch):
                piece = data[off:off + ch]
                okp = "OK LDB %d " % off
                r = ped.cmd("LDB %d %s" % (off, piece.hex()), total=3.0, until=lambda l: l.startswith(okp))
                line = next((l for l in r if l.startswith(okp)), None)
                if not line or int(line.split()[-1]) != len(piece):
                    raise TankGError("Capture transfer failed at byte %d." % off)
                if (off // ch) % 8 == 0:
                    prog(0.10 + 0.25 * off / len(data), "Loading the capture")
            crc = zlib.crc32(data) & 0xFFFFFFFF
            r = ped.cmd("LDEND %08x" % crc, total=70.0, until=lambda l: l.startswith("LOAD"))
            if any(l.startswith("LOAD") for l in r):
                ldopen = False
            if not any(l.startswith("LOAD rc=0") for l in r):
                raise TankGError("The pedal rejected the capture: " + " | ".join(r))
            if blob["in_dbu"] is not None and -20 <= blob["in_dbu"] <= 40:
                ped.cmd("CTRIM live 0 %d" % round(blob["in_dbu"] * 10), total=3.0, until=lambda l: l.startswith("CTRIM"))
            if cap_slot is not None and not cap_reused:
                r = ped.cmd("CSAVE %d %s" % (cap_slot, cname), total=8.0, until=lambda l: l.startswith("CSAVE"))
                if not any(l.startswith("CSAVE ok") for l in r):
                    raise TankGError("The capture loaded but couldn't be saved to a library slot: " + " | ".join(r))
                say("capture saved to library slot %d" % (cap_slot + 1))
            elif cap_slot is not None:
                ped.cmd("CLOAD %d" % cap_slot, total=8.0, until=lambda l: l.startswith("CLOAD"))

        # 3. cab IR
        if ir_block is not None:
            prog(0.40, "Loading the cab IR")
            cap_n = IR_CAP_LAST if ir_slot == 7 else IR_CAP
            n0 = min(cap_n, ir_block.ir_samples or cap_n)
            tries = [n0] + [n for n in (1024, 512, 256) if n < n0]
            done = False
            for k, n48 in enumerate(tries):
                q, gain = prep_ir(ir_src, n48)
                raw = q.tobytes()
                irbuf = True
                r = ped.cmd("IRBEGIN %d %d 48" % (len(q), gain), total=5.0, until=lambda l: l.startswith("OK IRBEGIN"))
                if not any(l.startswith("OK IRBEGIN") for l in r):
                    if any("emory" in l for l in r) and k < len(tries) - 1:
                        continue
                    raise TankGError("The pedal refused the cab IR: " + " | ".join(r))
                ch = max(1, (190 - len("IRB") - 2 - len(str(len(raw)))) // 2)
                for off in range(0, len(raw), ch):
                    piece = raw[off:off + ch]
                    okp = "OK IRB %d " % off
                    r = ped.cmd("IRB %d %s" % (off, piece.hex()), total=3.0, until=lambda l: l.startswith(okp))
                    line = next((l for l in r if l.startswith(okp)), None)
                    if not line or int(line.split()[-1]) != len(piece):
                        raise TankGError("IR transfer failed at byte %d." % off)
                r = ped.cmd("IREND %08x" % (zlib.crc32(raw) & 0xFFFFFFFF), total=6.0, until=lambda l: l.startswith("IR ok"))
                if not any(l.startswith("IR ok") for l in r):
                    raise TankGError("The pedal rejected the cab IR: " + " | ".join(r))
                irbuf = False
                done = True
                say("cab IR loaded (%d samples)" % len(q))
                break
            if not done:
                raise TankGError("The pedal doesn't have memory for the cab IR.")
            if ir_slot is not None:
                final_name = _library_name(irstem, q.tobytes())
                if ir_reused and final_name != iname:
                    # a shorter IR had to be sent than the one planned: it is not what the slot holds
                    free = [s for s in range(0, 7) if s not in irs]
                    ir_slot = free[0] if free else (7 if 7 not in irs else None)
                    ir_reused = False
                    if ir_slot is None:
                        raise TankGError("The pedal's IR library is full. Free one in the UltraRig page first.")
                iname = final_name
                if ir_reused:
                    say("cab IR is already in library slot %d" % (ir_slot + 1))
                else:
                    r = ped.cmd("ISAVE %d %s" % (ir_slot, iname), total=8.0, until=lambda l: l.startswith("ISAVE"))
                    if not any(l.startswith("ISAVE ok") for l in r):
                        raise TankGError("The IR loaded but couldn't be saved to a library slot: " + " | ".join(r))
                    say("cab IR saved to library slot %d" % (ir_slot + 1))

        # 4. the effect chain. If the pedal says it has no room, or the measured CPU leaves too little
        #    headroom, take a lighter version and try again.
        prog(0.55, "Building the chain")
        last_err = None
        steps = ((None, None), (0.80, 0.85), (0.72, 0.70), (0.65, 0.55))     # (CPU target, share of the memory pool)
        pending = None                                  # notes of a lighter plan already worked out (headroom retry)
        for attempt, (tgt, pool_share) in enumerate(steps):
            if tgt is not None:
                extra = pending if pending is not None else bridge.fit_plan(plan, tgt, mem_pool=bridge.MEM_POOL * pool_share)
                pending = None
                if not extra:
                    if last_err is None:
                        break                                   # nothing lighter to offer; keep what was built
                    if attempt == len(steps) - 1:
                        raise last_err
                    continue                                    # the pedal refused but the model sees nothing to trim: tighten more
                for n in extra:
                    say("fitted: " + n)
                say("the pedal needed more headroom; trying a lighter version")
            added_fx = []
            try:
                chain_cpu = _build_chain(ped, plan, added_fx, live)
            except TankGError as e:
                last_err = e
                for i in reversed(added_fx):
                    ped.cmd("FXR %d" % i)
                if attempt < len(steps) - 1 and re.search(r"cpu|memory", str(e), re.I):
                    continue
                raise
            m = re.search(r"cpu=(\d+)/(\d+)", chain_cpu)
            ratio = int(m[1]) / int(m[2]) if m else 0.0
            if ratio > HEADROOM_LIMIT and attempt < len(steps) - 1:
                say("the chain uses %d%% of the pedal's CPU, which leaves little headroom" % round(ratio * 100))
                nxt_tgt, nxt_share = steps[attempt + 1]
                lighter = bridge.fit_plan(plan, nxt_tgt, mem_pool=bridge.MEM_POOL * nxt_share)
                if lighter:                              # only take the chain apart when there is a lighter one to build
                    last_err, pending = None, lighter
                    for i in reversed(added_fx):
                        ped.cmd("FXR %d" % i)
                    continue
            if ratio > 0.88:
                say("note: the chain uses %d%% of the pedal's CPU; it is close to the limit" % round(ratio * 100))
            break
        say("chain verified: %d blocks, %s" % (len(plan.blocks), chain_cpu.replace("FXS ", "")))

        # 4b. capture EQ: SoloTone's input gain, EQ and output gain become the capture's Gain/tone/Volume
        if plan.ampeq:
            cmd = bridge.ampeq_command(plan.ampeq)
            r = ped.cmd(cmd, total=3.0, until=lambda l: l.startswith("AMPEQ"))
            if not any(re.match(r"^AMPEQ (volume|gain)=", l) for l in r):
                raise TankGError("The pedal refused the capture EQ: " + " | ".join(r))
            say("capture EQ set: " + r[0].replace("AMPEQ ", ""))
        if audio_on and audio_off:
            ped.cmd("AUDIO A2L", total=15.0, until=lambda l: l.startswith("AUDIO"))
            audio_off = False

        # 5. save as a preset
        if save:
            on_pedal, _ = live.read_chain(ped)
            if len(on_pedal) != len(plan.blocks):
                raise TankGError("The chain on the pedal (%d blocks) is not the one that was built (%d); nothing was saved."
                                 % (len(on_pedal), len(plan.blocks)))
            prog(0.85, "Saving preset %s" % slot_label(preset_slot))
            if preset_slot in _plist(ped) and not overwrite_preset:
                raise TankGError("Preset %s appeared on the pedal in the meantime; not overwritten." % slot_label(preset_slot))
            r = ped.cmd("PSAVE %d %s" % (preset_slot, name), total=8.0, until=lambda l: l.startswith("PSAVE"))
            if not any(l.startswith("PSAVE ok") for l in r):
                raise TankGError("Couldn't save the preset: " + " | ".join(r))
            say("preset saved: " + " | ".join(r))
            if cap_slot is None and blob:
                say("note: the preset uses whichever capture is loaded at that moment")
            after = _plist(ped)
            if preset_slot not in after:
                raise TankGError("The pedal said OK but preset %s isn't in its list." % slot_label(preset_slot))
            if with_scenes:
                prog(0.92, "Saving the scenes")
                scenes_saved = _save_scenes(ped, plan, say)
        prog(1.0, "Done")
        muted = audio_off
        audio_off = False
        return dict(muted=muted, plan=plan, cpu=cpu, mem=mem, cap_slot=cap_slot, ir_slot=ir_slot, preset_slot=preset_slot if save else None,
                    scenes_saved=scenes_saved)
    finally:
        try:
            if irbuf:
                ped.cmd("IRBEGIN 0 0 48", total=1.5)
            if ldopen:
                ped.cmd("LDBEGIN 0", total=1.5)
            if audio_off:
                ped.cmd("AUDIO A2L", total=15.0, until=lambda l: l.startswith("AUDIO"))
        except Exception:
            pass
        if transport is None:                       # we opened the serial port, so release it (a failure must not leave it locked)
            try:
                ped.s.close()
            except Exception:
                pass


# ───────────────────────── live mode: SoloTone as the remote of the pedal ─────────────────────────
class LiveSession:
    """Plays a profile on the pedal's LIVE chain (nothing is written to its memory: no preset, no library slot) and then keeps
    it in step with SoloTone: sync(profile) sends only what changed since the last call (pedal on/off, knobs, a scene change,
    the EQ and the gains) as FXE / FXP / AMPEQ. A profile with scenes is built as one shared chain, so switching scene is a
    handful of bypass and knob commands. A different capture, IR or chain order needs a new session (start again)."""

    def __init__(self, port, audio_loader=None, log=print, progress=None, transport=None):
        self.port, self.audio_loader, self.log, self.progress, self.transport = port, audio_loader, log, progress, transport
        self.plan = self.types = self.ped = None
        self.state = None
        self.ampeq_ok = True

    def start(self, profile):
        import bridge
        r = send_sound(profile, self.port, save=False, all_scenes=True, audio_loader=self.audio_loader, log=self.log,
                       progress=self.progress, transport=self.transport, audio_on=False)
        self.plan = r["plan"]
        self.types = self.plan.types
        scenes = getattr(self.plan, "scene_states", None)
        first = scenes[0]["blocks"] if scenes else None
        self.state = dict(blocks={id(b): ((id(b) in first) if first is not None else True, dict(b.values)) for b in self.plan.blocks},
                         ampeq=bridge.ampeq_command(self.plan.ampeq) if self.plan.ampeq else None)
        self.ped = _open(self.port, self.transport)
        self.ped.cmd("PING", total=1.5)                              # the first command after opening the port can be lost
        self.sync(profile)                                           # the profile may be on a scene other than the first
        self.ped.cmd("AUDIO A2L", total=15.0, until=lambda l: l.startswith("AUDIO"))   # sound only now: everything is set
        return r

    def sync(self, profile):
        """Send the differences between the pedal's state and `profile` (SoloTone as it is now)."""
        import bridge
        target = bridge.build_plan(profile, self.types, use_ampeq=self.plan.ampeq is not None,
                                   budget=getattr(self.plan, "pedal_budget", None))
        master = bridge._block_keys(self.plan)
        blocks = {}
        for key, tb in bridge._block_keys(target).items():
            mb = master.get(key)
            if mb is None:
                continue                                             # the shared chain has no such block (the fitter dropped it)
            vals = dict(tb.values) if mb.bt.name == tb.bt.name else {k: v for k, v in tb.values.items() if k in mb.values}
            blocks[id(mb)] = (True, vals)
        _apply_state(self.ped, self.plan, blocks, target.ampeq, self.state, self.log, "live", skip=("os_n",))

    def tuner(self, on, ref=440):
        return _tuner_cmd(self.ped, on, ref)

    def close(self):
        if self.ped is not None and self.transport is None:
            try:
                self.ped.s.close()
            except Exception:
                pass
        self.ped = None
