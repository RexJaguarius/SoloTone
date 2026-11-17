"""Back up everything on a Tank-G running UltraRig into a folder: every preset (chain, amp EQ, gain, scenes), every stored
capture and cab IR (as the raw bytes, CRC-checked), and the global settings.

Only documented commands are used; the CREAD / IREAD read-back and the layout of the replies come from PulseDeck's
protocol.json (https://github.com/wilianlimasantos/PulseDeck, MIT) and its crates/pulsedeck-core/src/backup.rs.

READ-ONLY WITH ONE CAVEAT: to read a preset the pedal has to load it (PLOAD, and SNAP for scenes), so the sound that is
playing at that moment is replaced and any unsaved edit is lost. The preset that was loaded is loaded again at the end.
Nothing is written to the pedal's flash. Run it from a terminal (the UltraRig control page must be closed):

    python backup_pedal.py COM7 backups/2026-11-17 --yes

Restoring is not built yet; PulseDeck's rebuild_preset() (backup.rs) shows the sequence (PNEW, FXA, FXP, FXE, PSAVE,
SNAPSAVE for presets; the normal upload sequences for captures and IRs).
"""
import base64
import json
import pathlib
import re
import sys
import time
import zlib

READS = {"PING", "LVL", "PLIST", "PLOAD", "FXS", "AMPEQ", "GAIN", "MODEL", "SNAPS", "SNAPNAME", "SNAP", "PCOLOR",
         "CLIST", "CREAD", "ILIST", "IREAD", "GEQ", "HPF", "LIM", "SPILL", "GLIDE", "SPUR", "FSMODE", "FSA", "FSLED",
         "LEDCFG", "MEM", "FXT"}
GLOBAL_QUERIES = ("GEQ", "HPF", "LIM", "SPILL", "GLIDE", "SPUR", "FSMODE", "FSA", "FSLED", "LEDCFG")
END = {"FXS": lambda l: l.startswith("FXS peak=") or l.startswith("FXS pico="),
       "PLIST": lambda l: l.startswith("PLIST end") or l.startswith("PLIST fin"),
       "CLIST": lambda l: l.startswith("CLIST end") or l.startswith("CLIST fin"),
       "ILIST": lambda l: l.startswith("ILIST end") or l.startswith("ILIST fin"),
       "FSA": lambda l: l.startswith("FSA end") or l.startswith("FSA fin"),
       "FSLED": lambda l: l.startswith("FSLED end") or l.startswith("FSLED fin")}


class BackupError(Exception):
    pass


def kv(line):
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:40] or "item"


def ask(ped, cmd, total=3.0):
    """One command; the reply lines, ending at the command's own terminator (or the first line for single-line replies)."""
    word = cmd.split()[0]
    if word not in READS:
        raise BackupError("refusing to send " + repr(cmd))
    end = END.get(word, lambda l: l.startswith(word))
    return ped.cmd(cmd, total=total, until=end)


def read_blob(ped, cmd, slot, log=print):
    """A stored capture or IR through CREAD / IREAD, every 96-byte piece and the whole checked against its CRC32."""
    info = ask(ped, "%s %d" % (cmd, slot))[0]
    fields = kv(info)
    if "len" not in fields:
        raise BackupError("%s %d: no length in the reply: %s" % (cmd, slot, info))
    n = int(fields["len"])
    data = bytearray()
    while len(data) < n:
        t = ask(ped, "%s %d %d" % (cmd, slot, len(data)))[0].split()
        if len(t) < 6 or int(t[2]) != len(data):
            raise BackupError("%s %d: unexpected piece: %s" % (cmd, slot, " ".join(t)[:80]))
        chunk = base64.b64decode(t[5])
        if not chunk or int(t[3]) != len(chunk):
            raise BackupError("%s %d: piece size mismatch at %d" % (cmd, slot, len(data)))
        if "%08x" % (zlib.crc32(chunk) & 0xFFFFFFFF) != t[4]:
            raise BackupError("%s %d: piece CRC mismatch at %d" % (cmd, slot, len(data)))
        data += chunk
    data = bytes(data[:n])
    if "crc" in fields and "%08x" % (zlib.crc32(data) & 0xFFFFFFFF) != fields["crc"]:
        raise BackupError("%s %d: whole-blob CRC mismatch" % (cmd, slot))
    return info, data


def read_state(ped):
    out = {"fxs": ask(ped, "FXS", total=5.0)}
    for c in ("AMPEQ", "GAIN", "MODEL"):
        try:
            out[c.lower()] = ask(ped, c)
        except Exception as e:            # a firmware without the command: keep going, note it
            out[c.lower()] = ["(unavailable: %s)" % e]
    return out


def backup(ped, out_dir, log=print):
    out = pathlib.Path(out_dir)
    (out / "captures").mkdir(parents=True, exist_ok=True)
    (out / "irs").mkdir(exist_ok=True)
    manifest = dict(tool="backup_pedal.py", created=time.strftime("%Y-%m-%d %H:%M:%S"), warnings=[])
    manifest["ping"] = ask(ped, "PING", total=2.0)
    plist = ask(ped, "PLIST", total=5.0)
    current = next((int(m[1]) for l in plist for m in [re.search(r"current=(-?\d+)", l)] if m and l.startswith("PLIST end")), -1)
    presets = []
    for l in plist:
        m = re.match(r"PRE (\d+) (\d+) (-?\d+) (-?\d+) '(.*)'", l)
        if m:
            presets.append(dict(slot=int(m[1]), blocks=int(m[2]), capture=int(m[3]), ir=int(m[4]), name=m[5], plist=l))
    manifest["plist"] = plist
    log("%d presets, current = %d" % (len(presets), current))
    try:
        for i, p in enumerate(presets):
            r = ask(ped, "PLOAD %d" % p["slot"], total=12.0)
            p["pload"] = r
            p.update(read_state(ped))
            try:
                p["color"] = ask(ped, "PCOLOR")
            except Exception:
                pass
            snaps = ask(ped, "SNAPS")
            p["snaps"] = snaps
            p["scenes"] = []
            bits = snaps[0].split()[1][:4] if snaps and len(snaps[0].split()) > 1 else ""      # 'SNAPS 1010 current=0'
            stored = [int(ch) for ch in bits] if bits and all(ch in "01" for ch in bits) else []
            for k, has in enumerate(stored):
                if has:
                    ask(ped, "SNAP %d" % k, total=12.0)
                    p["scenes"].append(dict(index=k, **read_state(ped)))
            log("  [%d/%d] preset %d '%s': %d blocks, %d scenes" % (i + 1, len(presets), p["slot"], p["name"], p["blocks"], len(p["scenes"])))
    finally:
        if current >= 0:
            try:
                ask(ped, "PLOAD %d" % current, total=12.0)
            except Exception as e:
                manifest["warnings"].append("could not reload preset %d at the end: %s" % (current, e))
    manifest["presets"] = presets
    for kind, lst_cmd, read_cmd, item, folder, ext in (("captures", "CLIST", "CREAD", "CAP", "captures", ".a2l"),
                                                       ("irs", "ILIST", "IREAD", "IRS", "irs", ".ir")):
        entries = []
        for l in ask(ped, lst_cmd, total=5.0):
            m = re.match(r"%s (\d+) (\d+) '(.*)'(.*)" % item, l)
            if m and int(m[2]) > 0:
                entries.append((int(m[1]), m[3], l))
        manifest[kind] = []
        for slot, name, line in entries:
            try:
                info, data = read_blob(ped, read_cmd, slot, log)
            except Exception as e:
                manifest["warnings"].append("%s slot %d (%s): %s" % (kind, slot, name, e))
                log("  %s slot %d FAILED: %s" % (kind, slot, e))
                continue
            f = "%s/%02d_%s%s" % (folder, slot, safe(name), ext)
            (out / f).write_bytes(data)
            manifest[kind].append(dict(slot=slot, name=name, file=f, bytes=len(data), list_line=line, info=info,
                                       crc32="%08x" % (zlib.crc32(data) & 0xFFFFFFFF)))
            log("  %s slot %d '%s': %d bytes, CRC ok" % (kind, slot, name, len(data)))
    manifest["globals"] = {}
    for c in GLOBAL_QUERIES:
        try:
            manifest["globals"][c] = ask(ped, c)
        except Exception as e:
            manifest["globals"][c] = ["(unavailable: %s)" % e]
    (out / "backup.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log("wrote %s (%d presets, %d captures, %d IRs, %d warnings)" % (out / "backup.json", len(presets), len(manifest["captures"]),
                                                                   len(manifest["irs"]), len(manifest["warnings"])))
    return manifest


def main(argv):
    if len(argv) < 3 or "--yes" not in argv:
        sys.exit(__doc__ + "\nGive the port, the folder, and --yes to confirm that the loaded sound may be replaced while it runs.")
    import serial
    from live import Pedal
    s = serial.Serial(argv[1], 115200, timeout=0.05)
    try:
        backup(Pedal(argv[1], allow=READS, transport=s), argv[2])
    finally:
        s.close()


if __name__ == "__main__":
    main(sys.argv)
