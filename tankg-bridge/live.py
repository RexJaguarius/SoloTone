"""Live mode for bridge.py: sends a plan to the pedal's LIVE chain (never saved).

Allow-list: PING, LVL, FXS (reads) and FXA, FXP, FXR (chain edits). It cannot send
PSAVE / PDEL / PLOAD / CSAVE / ISAVE / uploads / GAIN / AMP. Refuses to run if the
pedal's chain isn't empty; rolls back everything it added if any step is refused.
"""
import re
import sys
import time

LIVE_ALLOWED = {"PING", "LVL", "FXS", "FXA", "FXP", "FXR", "FXOS", "MODEL"}
# Capture upload adds exactly these (and only with allow_upload=True). CSAVE is deliberately absent:
# the capture is loaded into the running engine and nothing is saved to a library slot.
UPLOAD_ALLOWED = {"AUDIO", "LDBEGIN", "LDB", "LDEND", "CTRIM"}
FXS_END = re.compile(r"^FXS (peak|pico)\b")
BLOCK = re.compile(r"^FX ([PQ])(\d+) id=(\d+) (\w+) on=(\d) p=([-\d,]+)")


class Pedal:
    def __init__(self, port, allow_upload=False, allow=None, transport=None):
        if transport is None:
            import serial                  # only needed for live mode
            transport = serial.Serial(port, 115200, timeout=0.05)
        self.s = transport
        self.allow = set(allow) if allow else (LIVE_ALLOWED | (UPLOAD_ALLOWED if allow_upload else set()))

    def cmd(self, c, quiet=0.7, total=5.0, until=None):
        """Send one line and collect the reply. `until(line)` ends the wait as soon as it is true
        (or on an ERR line); without it the reply ends when the pedal goes quiet."""
        word = c.split()[0]
        if word not in self.allow:
            raise RuntimeError("refusing to send " + repr(c))
        if word == "CTRIM" and not c.startswith("CTRIM live "):
            raise RuntimeError("refusing to send " + repr(c))
        self.s.reset_input_buffer()
        self.s.write((c + "\n").encode())
        buf, out, t0, last = b"", [], time.time(), time.time()
        while time.time() - t0 < total:
            ch = self.s.read(4096)
            if ch:
                last = time.time()
                buf += ch
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    ln = raw.decode("utf-8", "replace").rstrip("\r")
                    if ln.strip() and not ln.startswith("@"):
                        out.append(ln)
                if until is not None and any(x.startswith("ERR") or until(x) for x in out):
                    break
                if c.startswith("FXS") and any(FXS_END.match(x) for x in out):
                    break
            elif until is None and out and time.time() - last > quiet:
                break
        return out


def read_chain(ped):
    lines = ped.cmd("FXS")
    chain = []
    for l in lines:
        m = BLOCK.match(l)
        if m:
            chain.append(dict(bank=m[1], pos=int(m[2]), id=int(m[3]), name=m[4],
                              on=int(m[5]), p=[int(x) for x in m[6].split(",")]))
    cpu = next((l for l in lines if l.startswith("FXS cpu=")), "")
    return chain, cpu


def go_live(plan, port, _re=None):
    ped = Pedal(port)
    for _ in range(5):
        if any(l.startswith("PONG") for l in ped.cmd("PING", total=1.0)):
            break
    else:
        sys.exit("No PONG from the pedal: wrong port, or not running UltraRig.")
    chain, cpu = read_chain(ped)
    if chain:
        sys.exit("The pedal's chain is not empty (%d blocks). Not touching it; "
                 "run with --clear first if you want it emptied." % len(chain))
    added = []

    def rollback():
        for i in reversed(added):
            print("   rollback:", " | ".join(ped.cmd("FXR %d" % i)))

    for b in plan.blocks:
        r = ped.cmd("FXA %s 99 %d" % (b.bank, b.bt.id), total=6.0)
        line = next((l for l in r if l.startswith("FXA id=")), None)
        print("FXA %s 99 %d (%s) -> %s" % (b.bank, b.bt.id, b.bt.name, " | ".join(r)))
        if not line:
            print("   !! block refused; rolling back.")
            rollback()
            return
        b.live_id = int(re.search(r"id=(\d+)", line)[1])
        added.append(b.live_id)
        for p in b.bt.params:
            v = b.values[p.name]
            if v != p.default:
                word = "FXOS" if p.name == "os_n" else "FXP"
                cmd = ("FXOS %d %d" % (b.live_id, v)) if word == "FXOS" else ("FXP %d %d %d" % (b.live_id, p.idx, v))
                rr = ped.cmd(cmd)
                print("   %s (%s) -> %s" % (cmd, p.name, " | ".join(rr)))
                if not any(l.startswith(word + " ok") for l in rr):
                    print("   !! parameter refused; rolling back.")
                    rollback()
                    return
    chain, cpu = read_chain(ped)
    print("\nReadback check:")
    bad = 0
    for b in plan.blocks:
        row = next((c for c in chain if c["id"] == b.live_id), None)
        if not row:
            print("  MISSING", b.bt.name)
            bad += 1
            continue
        for p in b.bt.params:
            if row["p"][p.idx] != b.values[p.name]:
                print("  MISMATCH %s.%s: wanted %d, pedal has %d"
                      % (b.bt.name, p.name, b.values[p.name], row["p"][p.idx]))
                bad += 1
    print("  all %d blocks match" % len(plan.blocks) if not bad else "  %d problems" % bad, "|", cpu)
    print("\nThe chain is live and UNSAVED. Cab IR, capture and GAIN were not sent.\n"
          "Run  python bridge.py --clear COMx  to remove it, or power-cycle the pedal.")


def clear_chain(port):
    ped = Pedal(port)
    chain, _ = read_chain(ped)
    for c in reversed(chain):
        print("FXR %d (%s) -> %s" % (c["id"], c["name"], " | ".join(ped.cmd("FXR %d" % c["id"]))))
    chain, cpu = read_chain(ped)
    print("chain now has %d blocks | %s" % (len(chain), cpu))


def upload_capture(path, port):
    """Convert a .nam and load it into the pedal's running engine (not saved to a library slot)."""
    import a2l1
    try:
        r = a2l1.convert(path)
    except a2l1.NamError as e:
        sys.exit("Can't use this capture: %s" % e)
    data, crc = r["blob"], zlib_crc(r["blob"])
    print("Converted %s: %d layers, %d bytes, crc %08x" % (r["name"], r["layers"], len(data), crc))
    ped = Pedal(port, allow_upload=True)
    for _ in range(5):
        if any(l.startswith("PONG") for l in ped.cmd("PING", total=1.0)):
            break
    else:
        sys.exit("No PONG from the pedal.")
    lvl = " ".join(ped.cmd("LVL"))
    m = re.search(r" m=(\d)", lvl)
    was_on = bool(m and m[1] == "2")
    print("audio engine was", "ON" if was_on else "off", "(" + lvl[:60] + "...)")
    ldopen = audio_off = False
    ok = False
    try:
        rr = ped.cmd("AUDIO OFF", total=5.0, until=lambda l: l.startswith("AUDIO"))
        audio_off = was_on
        print("AUDIO OFF ->", " | ".join(rr))
        ldopen = True
        rr = ped.cmd("LDBEGIN %d" % len(data), total=5.0, until=lambda l: l.startswith("OK LDBEGIN"))
        print("LDBEGIN ->", " | ".join(rr))
        if not any(l.startswith("OK LDBEGIN") for l in rr):
            raise RuntimeError("LDBEGIN refused: " + " | ".join(rr))
        ch = max(1, (min(190, 199) - len("LDB") - 2 - len(str(len(data)))) // 2)
        sent = 0
        for off in range(0, len(data), ch):
            piece = data[off:off + ch]
            okp = "OK LDB %d " % off
            rr = ped.cmd("LDB %d %s" % (off, piece.hex()), total=3.0, until=lambda l: l.startswith(okp))
            line = next((l for l in rr if l.startswith(okp)), None)
            if not line or int(line.split()[-1]) != len(piece):
                raise RuntimeError("transfer failed at byte %d: %s" % (off, " | ".join(rr)))
            sent += len(piece)
        print("sent %d bytes in chunks of %d" % (sent, ch))
        rr = ped.cmd("LDEND %08x" % crc, total=70.0, until=lambda l: l.startswith("LOAD"))
        print("LDEND ->", " | ".join(rr))
        if any(l.startswith("LOAD") for l in rr):
            ldopen = False                    # a LOAD reply means the buffer was released
        if not any(l.startswith("LOAD rc=0") for l in rr):
            raise RuntimeError("the pedal rejected the capture: " + " | ".join(rr))
        ok = True
    except Exception as e:
        print("!!", e)
    finally:
        if ldopen:
            print("releasing load buffer ->", " | ".join(ped.cmd("LDBEGIN 0", total=2.0)))
        if audio_off or (ok and was_on):
            print("restoring audio ->", " | ".join(ped.cmd("AUDIO A2L", total=15.0, until=lambda l: l.startswith("AUDIO"))))
    if ok and r["in_dbu"] is not None and -20 <= r["in_dbu"] <= 40:
        rr = ped.cmd("CTRIM live 0 %d" % round(r["in_dbu"] * 10), total=3.0, until=lambda l: l.startswith("CTRIM"))
        print("CTRIM live (input level %.1f dBu) ->" % r["in_dbu"], " | ".join(rr))
    print("\nModel now:", " | ".join(ped.cmd("MODEL", total=2.0, until=lambda l: l.startswith("MODEL"))) if "MODEL" in LIVE_ALLOWED else "")
    print("Capture upload", "OK (loaded, not saved to a library slot)." if ok else "FAILED; the pedal was left as it was.")


def zlib_crc(b):
    import zlib
    return zlib.crc32(b) & 0xFFFFFFFF
