"""backup_pedal.py against a fake pedal: no hardware, no serial port."""
import base64, json, pathlib, tempfile, unittest, zlib
import backup_pedal as bk

BLOB = bytes(range(256)) * 3 + b"A2L1-test"


class FakePed:
    def __init__(self):
        self.log = []
        self.current = 1

    def cmd(self, c, quiet=0.7, total=5.0, until=None):
        self.log.append(c)
        w = c.split()
        if w[0] == "PING":
            return ["PONG tankg_fw M5.71 rel ble Nov 12 2026"]
        if w[0] == "PLIST":
            return ["PRE 0 2 0 -1 'Clean one' t=p1,q5 s=0", "PRE 1 1 0 2 'Lead' t=p9 s=0", "PLIST end current=%d" % self.current]
        if w[0] == "PLOAD":
            self.current = int(w[1])
            return ["PLOAD ok %s 'x' missing=0 ms=9" % w[1]]
        if w[0] == "FXS":
            return ["FX P0 id=0 gate on=1 p=-70,120,0,0,0,0,0 ticks=1 l=0 g=-1 fu=0 mem=0", "FXS amp=1 on=1", "FXS peak=1 skipped=0"]
        if w[0] in ("AMPEQ", "GAIN", "MODEL", "PCOLOR"):
            return ["%s ok" % w[0]]
        if w[0] == "SNAPS":
            return ["SNAPS 1010 current=0"]
        if w[0] == "SNAP":
            return ["SNAP ok %s missing=0 ms=3" % w[1]]
        if w[0] == "CLIST":
            return ["CAP 2 %d 'HIWATT Clean'" % len(BLOB), "CLIST end current=2 slots=15"]
        if w[0] == "ILIST":
            return ["IRS 0 %d 'Cab A' fs=48" % len(BLOB), "ILIST end current=0 slots=8"]
        if w[0] in ("CREAD", "IREAD"):
            if len(w) == 2:
                return ["%s %s len=%d crc=%08x id=00000000 aux=0 fs=0 name='x'" % (w[0], w[1], len(BLOB), zlib.crc32(BLOB) & 0xFFFFFFFF)]
            off = int(w[2])
            chunk = BLOB[off:off + 96]
            return ["%s %s %d %d %08x %s" % (w[0], w[1], off, len(chunk), zlib.crc32(chunk) & 0xFFFFFFFF, base64.b64encode(chunk).decode())]
        return ["ERR %s unknown" % w[0]]


class BackupTests(unittest.TestCase):
    def test_backup_reads_everything_and_reloads_the_playing_preset(self):
        ped = FakePed()
        with tempfile.TemporaryDirectory() as d:
            m = bk.backup(ped, d, log=lambda *_: None)
            self.assertEqual([p["slot"] for p in m["presets"]], [0, 1])
            self.assertEqual([s["index"] for s in m["presets"][0]["scenes"]], [0, 2])           # SNAPS 1010
            self.assertEqual(pathlib.Path(d, m["captures"][0]["file"]).read_bytes(), BLOB)
            self.assertEqual(pathlib.Path(d, m["irs"][0]["file"]).read_bytes(), BLOB)
            self.assertEqual(m["warnings"], [])
            self.assertTrue(json.loads(pathlib.Path(d, "backup.json").read_text())["presets"])
        self.assertEqual(ped.current, 1)                                                        # the preset that was loaded is back
        sent = {c.split()[0] for c in ped.log}
        self.assertFalse(sent & {"PSAVE", "CSAVE", "ISAVE", "PDEL", "CDEL", "IDEL", "GSAVE", "LDBEGIN", "FXA", "FXR"})

    def test_a_corrupted_piece_is_reported_not_saved(self):
        ped = FakePed()
        real = ped.cmd

        def bad(c, **kw):
            r = real(c, **kw)
            if c.startswith("CREAD 2 96"):
                r = [r[0].rsplit(" ", 1)[0] + " " + base64.b64encode(b"x" * 96).decode()]
            return r
        ped.cmd = bad
        with tempfile.TemporaryDirectory() as d:
            m = bk.backup(ped, d, log=lambda *_: None)
            self.assertEqual(m["captures"], [])
            self.assertTrue(any("CRC mismatch" in w for w in m["warnings"]))

    def test_refuses_writing_commands(self):
        with self.assertRaises(bk.BackupError):
            bk.ask(FakePed(), "PSAVE 3 x")


if __name__ == "__main__":
    unittest.main()
