"""Offline tests for tankg_api using a fake UltraRig pedal (no hardware, no serial port).

    python -m unittest test_tankg_api -v          (from the tankg-bridge folder)
"""
import hashlib
import json
import pathlib
import re
import sys
import unittest
import zlib

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import bridge          # noqa: E402
import live            # noqa: E402
import tankg_api       # noqa: E402

PROFILE = HERE.parent / "profiles" / "David Bowie" / "Starman (David Bowie) - Telecaster.stprofile"
bridge.CALIB_PATH = HERE / "no_such_calibration.json"      # the tests check the plain formula, not the measured corrections
bridge._CALIB = {}


class FakePedal:
    """Speaks just enough of the UltraRig protocol to exercise tankg_api."""

    def __init__(self):
        self.types = {t.id: t for t in bridge.load_catalogue(HERE / "fxt_catalogue.txt").values()}
        self.chain, self.next_id = [], 0
        self.caps, self.irs, self.presets = {}, {}, {}
        self.audio, self.log, self.out = 2, [], b""
        self.ld = self.ir = None
        self.fail_ldend = False
        self.cpu_limit = None
        self.ampeq = None
        self.snaps = {}                         # scene index -> (chain on/off + params snapshot, ampeq)
        self.fail_snapsave = False
        self.fxe_no_cpu = set()                 # block names the pedal will not switch ON ("not enough CPU")

    def used(self):
        tot = 0
        for b in self.chain:
            t = next(t for t in self.types.values() if t.name == b["name"])
            osi = next((p.idx for p in t.params if p.name == "os_n"), None)
            tot += t.cost_at(b["p"][osi]) if osi is not None else t.cost
        return tot

    # --- transport interface used by live.Pedal
    def reset_input_buffer(self):
        self.out = b""

    def read(self, n):
        d, self.out = self.out[:n], self.out[n:]
        return d

    def write(self, data):
        line = data.decode().strip()
        self.log.append(line)
        for l in self.handle(line):
            self.out += (l + "\r\n").encode()

    # --- protocol
    def handle(self, line):
        w, *a = line.split()
        if w == "PING":
            return ["PONG tankg_fw M5.34 rel amx ble"]
        if w == "FXT":
            return (HERE / "fxt_catalogue.txt").read_text(encoding="utf-8").splitlines() + ["FXT end"]
        if w == "LVL":
            return ["LVL in=-600 out=-600 ci=0 co=0 fx=0/31000 mem=0 sk=0 m=%d g=0/0 b=1200 s=-1 a=1 c=3 p=-1" % self.audio]
        if w == "CLIST":
            return ["CAP %d %d '%s'" % (s, sz, n) for s, (n, sz) in sorted(self.caps.items())] + ["CLIST end current=-1 slots=15"]
        if w == "ILIST":
            return ["IRS %d %d '%s'" % (s, sz, n) for s, (n, sz) in sorted(self.irs.items())] + ["ILIST end current=-1 slots=8"]
        if w == "PLIST":
            return ["PRE %d 3 0 0 '%s' t=p1,q4 s=1" % (n, nm) for n, nm in sorted(self.presets.items())] + ["PLIST end current=-1"]
        if w == "FXS":
            rows = []
            for i, b in enumerate(self.chain):
                rows.append("FX %s%d id=%d %s on=%d p=%s ticks=100 l=0 g=-1 fu=0 mem=0"
                            % (b["bank"], sum(1 for x in self.chain[:i] if x["bank"] == b["bank"]), b["id"], b["name"], b.get("on", 1),
                               ",".join(str(b["p"].get(k, 0)) for k in range(7))))
            return rows + ["FXS amp=1 on=1", "FXS cpu=1000/30000 ticks (3%) mem_fx=0 B", "FXS peak=0 ticks skipped=0"]
        if w == "FXA":
            t = self.types_by_id(int(a[2]))
            if self.cpu_limit is not None and self.used() + t.cost > self.cpu_limit:
                return ["ERR FXA not enough CPU"]
            b = dict(id=self.next_id, bank=a[0], name=t.name, p={p.idx: p.default for p in t.params})
            self.chain.append(b)
            self.next_id += 1
            return ["FXA id=%d %s" % (b["id"], t.name), "FX cpu=1000/30000 ticks (3%) mem_fx=0 B"]
        if w == "FXP":
            b = next(x for x in self.chain if x["id"] == int(a[0]))
            b["p"][int(a[1])] = int(a[2])
            return ["FXP ok %s" % a[2]]
        if w == "AMPEQ":
            if a and a[0] == "info":
                return ["ERR AMPEQ usage"]                          # like M5.34
            if not a:
                v = self.ampeq or [0] * 6
                return ["AMPEQ volume=%d bass=%d mid=%d treble=%d presence=%d drive=%d" % tuple(v)]
            self.ampeq = [int(x) for x in a]
            return ["AMPEQ volume=%d bass=%d mid=%d treble=%d presence=%d drive=%d" % tuple(self.ampeq)]
        if w == "FXE":
            blk = next(x for x in self.chain if x["id"] == int(a[0]))
            if int(a[1]) and blk["name"] in self.fxe_no_cpu:
                return ["ERR FXE not enough CPU"]
            blk["on"] = int(a[1])
            return ["FXE ok cpu=1000/30000 ticks (3%)"]
        if w == "SNAPSAVE":
            if self.fail_snapsave:
                return ["ERR SNAPSAVE no preset"]
            self.snaps[int(a[0])] = dict(on={x["id"]: x.get("on", 1) for x in self.chain}, p={x["id"]: dict(x["p"]) for x in self.chain},
                                         ampeq=list(self.ampeq) if self.ampeq else None, name="")
            return ["SNAPSAVE ok %s" % a[0]]
        if w == "SNAPNAME":
            self.snaps[int(a[0])]["name"] = " ".join(a[1:])
            return ["SNAPNAME ok"]
        if w == "SNAPS":
            return ["SNAPS %s current=0" % "".join("1" if i in self.snaps else "0" for i in range(4))]
        if w == "FXOS":
            b = next(x for x in self.chain if x["id"] == int(a[0]))
            t = next(t for t in self.types.values() if t.name == b["name"])
            b["p"][next(p.idx for p in t.params if p.name == "os_n")] = int(a[1])
            return ["FXOS ok cpu=1000/30000 ticks (3%%) os=%s" % a[1]]
        if w == "FXR":
            self.chain = [x for x in self.chain if x["id"] != int(a[0])]
            return ["FXR ok cpu=0/30000 ticks (0%) mem_fx=0 B"]
        if w == "AUDIO":
            self.audio = 0 if a[0] == "OFF" else 2
            return ["AUDIO %d rc=0 ms=20" % self.audio]
        if w == "LDBEGIN":
            self.ld = [int(a[0]), b""] if int(a[0]) else None
            return ["OK LDBEGIN %s" % a[0]] if self.ld else ["ERR LDBEGIN 0"]
        if w == "LDB":
            chunk = bytes.fromhex(a[1])
            assert int(a[0]) == len(self.ld[1])
            self.ld[1] += chunk
            return ["OK LDB %s %d" % (a[0], len(chunk))]
        if w == "LDEND":
            ok = len(self.ld[1]) == self.ld[0] and "%08x" % (zlib.crc32(self.ld[1]) & 0xFFFFFFFF) == a[0] and not self.fail_ldend
            self.last_blob, self.ld = self.ld[1], None
            return ["LOAD rc=0 name='x' layers=23" if ok else "LOAD rc=-4 bad"]
        if w == "CTRIM":
            return ["CTRIM ok"]
        if w == "CSAVE":
            self.caps[int(a[0])] = (" ".join(a[1:]), len(self.last_blob))
            return ["CSAVE ok"]
        if w == "CLOAD":
            return ["CLOAD ok"]
        if w == "IRBEGIN":
            if int(a[0]) == 0:
                self.ir = None
                return ["ERR IRBEGIN"]
            self.ir = [int(a[0]) * 2, b""]
            return ["OK IRBEGIN"]
        if w == "IRB":
            chunk = bytes.fromhex(a[1])
            self.ir[1] += chunk
            return ["OK IRB %s %d" % (a[0], len(chunk))]
        if w == "IREND":
            ok = len(self.ir[1]) == self.ir[0] and "%08x" % (zlib.crc32(self.ir[1]) & 0xFFFFFFFF) == a[0]
            self.last_ir = self.ir[1]
            return ["IR ok" if ok else "ERR IREND crc"]
        if w == "ISAVE":
            self.irs[int(a[0])] = (" ".join(a[1:]), len(self.last_ir))      # the real pedal lists the size in bytes
            return ["ISAVE ok"]
        if w == "PSAVE":
            self.presets[int(a[0])] = " ".join(a[1:])
            return ["PSAVE ok preset=%s capture=0" % a[0]]
        return ["ERR unknown"]

    def types_by_id(self, tid):
        return self.types[tid]


def ir_loader(_path):
    rng = np.random.default_rng(1)
    return (rng.standard_normal(6000) * np.exp(-np.arange(6000) / 800)).astype(np.float32)


@unittest.skipUnless((HERE / "fxt_catalogue.txt").exists() and PROFILE.exists(),
                     "needs fxt_catalogue.txt (python ultrarig_probe.py writes it from your pedal) and a local profile")
class TankGTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        self.fake = FakePedal()
        self.msgs = []

    def send(self, **kw):
        kw.setdefault("audio_loader", ir_loader)
        return tankg_api.send_sound(self.profile, "FAKE", log=self.msgs.append, transport=self.fake, **kw)

    def test_full_send_saves_everything(self):
        r = self.send(preset_slot=31, save=True)
        self.assertIn(31, self.fake.presets)
        self.assertEqual(r["cap_slot"], 0)
        self.assertEqual(r["ir_slot"], 0)
        self.assertEqual(len(self.fake.caps), 1)
        self.assertEqual(len(self.fake.irs), 1)
        self.assertEqual(self.fake.audio, 2)                      # audio is back on
        self.assertEqual(len(self.fake.chain), len(r["plan"].blocks))
        # the blob the pedal received is the same one a2l1 produces
        import a2l1
        self.assertEqual(self.fake.last_blob, a2l1.convert(self.profile["nam_path"])["blob"])

    def _with_scenes(self):
        import copy
        a = copy.deepcopy(self.profile["pedals"])
        b = copy.deepcopy(self.profile["pedals"])
        b["od_on"], b["od_mode"], b["od_drive"], b["od_vol"] = True, "klon", 4.0, 0.8
        b["delay_on"], b["delay_mix"] = True, 0.5
        a["od_on"] = False
        snap = lambda p, ig: {"pedals": p, "eq_on": False, "eq_band_gains": [0] * 5, "eq_level": 0.0, "in_gain": ig, "out_gain": 1.0}
        self.profile["pedals"] = a
        self.profile["scenes"] = {"active": 0, "names": ["Verse", "Solo", "", ""], "slots": [snap(a, 1.0), snap(b, 1.0), None, None]}

    def test_scenes_become_one_chain_and_saved_scenes(self):
        self._with_scenes()
        r = self.send(preset_slot=31, save=True)
        self.assertEqual(r["scenes_saved"], [0, 1])
        self.assertEqual(sorted(self.fake.snaps), [0, 1])
        self.assertEqual([self.fake.snaps[i]["name"] for i in (0, 1)], ["Verse", "Solo"])
        od = next(b for b in r["plan"].blocks if b.src.startswith(("overdrive", "TS", "ts", "klon", "Klon")) or b.bt.name in ("ts9", "klon"))
        # the overdrive is in the chain, off in scene A and on in scene B
        self.assertEqual(self.fake.snaps[0]["on"][od.live_id], 0)
        self.assertEqual(self.fake.snaps[1]["on"][od.live_id], 1)
        # the delay mix differs between the scenes
        dl = next(b for b in r["plan"].blocks if b.bt.name == "delay")
        mix = next(p.idx for p in dl.bt.params if p.name == "mezcla_pct")
        self.assertNotEqual(self.fake.snaps[0]["p"][dl.live_id][mix], self.fake.snaps[1]["p"][dl.live_id][mix])
        self.assertEqual(self.msgs.count("scenes on the pedal: SNAPS 1100 current=0"), 1)

    def test_a_scene_keeps_going_without_the_block_the_pedal_has_no_cpu_for(self):
        self._with_scenes()
        self.fake.fxe_no_cpu = {"klon", "ts9", "boost"}
        r = self.send(preset_slot=31, save=True)
        self.assertEqual(r["scenes_saved"], [0, 1])
        self.assertTrue(any("stays off" in m for m in self.msgs))
        self.assertIn(1, self.fake.snaps)

    def test_live_session_follows_solotone_without_writing_anything(self):
        self._with_scenes()
        sl = self.profile["scenes"]["slots"]
        ses = tankg_api.LiveSession("FAKE", audio_loader=ir_loader, log=self.msgs.append, transport=self.fake)
        ses.start(self.profile)
        sent = lambda: [l.split()[0] for l in self.fake.log]
        self.assertFalse(set(sent()) & {"PSAVE", "CSAVE", "ISAVE", "SNAPSAVE", "SNAPNAME", "PDEL"})     # live: nothing is stored
        od = next(b for b in ses.plan.blocks if b.code == "od")
        dl = next(b for b in ses.plan.blocks if b.code == "delay")
        fx = lambda b: next(x for x in self.fake.chain if x["id"] == b.live_id)
        self.assertEqual(fx(od).get("on", 1), 0)                  # scene A: the overdrive is bypassed
        n = len(self.fake.log)
        ses.sync(self.profile)                                    # nothing changed: nothing sent
        self.assertEqual(len(self.fake.log), n)
        # scene B
        import bridge
        ses.sync(bridge.scene_profile(self.profile, sl[1]))
        self.assertEqual(fx(od).get("on", 1), 1)
        mix = next(p.idx for p in dl.bt.params if p.name == "mezcla_pct")
        self.assertEqual(fx(dl)["p"][mix], 50)
        cmds = {l.split()[0] for l in self.fake.log[n:]}
        self.assertTrue(cmds <= {"FXE", "FXP", "AMPEQ"}, cmds)    # only the differences, only live commands
        # back to scene A
        ses.sync(bridge.scene_profile(self.profile, sl[0]))
        self.assertEqual(fx(od).get("on", 1), 0)
        ses.close()

    def test_a_profile_without_scenes_also_goes_live(self):
        ses = tankg_api.LiveSession("FAKE", audio_loader=ir_loader, log=self.msgs.append, transport=self.fake)
        ses.start(self.profile)
        n = len(self.fake.log)
        p2 = dict(self.profile); p2["pedals"] = dict(self.profile["pedals"], delay_on=not self.profile["pedals"].get("delay_on", False))
        ses.sync(p2)
        self.assertGreater(len(self.fake.log), n)
        self.assertFalse({"PSAVE", "CSAVE", "SNAPSAVE"} & {l.split()[0] for l in self.fake.log})

    def test_playing_sends_only_the_active_scene(self):
        self._with_scenes()
        self.send(save=False)
        self.assertFalse({l.split()[0] for l in self.fake.log} & {"SNAPSAVE", "SNAPNAME", "PSAVE"})

    def test_a_profile_without_scenes_saves_no_scenes(self):
        r = self.send(preset_slot=31, save=True)
        self.assertEqual(r["scenes_saved"], [])
        self.assertFalse({l.split()[0] for l in self.fake.log} & {"SNAPSAVE", "FXE"})

    def test_a_scene_the_pedal_refuses_does_not_undo_the_preset(self):
        self._with_scenes()
        self.fake.fail_snapsave = True
        r = self.send(preset_slot=31, save=True)
        self.assertIn(31, self.fake.presets)
        self.assertEqual(r["scenes_saved"], [])
        self.assertTrue(any("scene A was not saved" in m for m in self.msgs))

    def test_preview_writes_nothing(self):
        self.send(save=False)
        sent = {l.split()[0] for l in self.fake.log}
        self.assertFalse(sent & {"CSAVE", "ISAVE", "PSAVE"})
        self.assertFalse(self.fake.presets or self.fake.caps or self.fake.irs)
        self.assertTrue(self.fake.chain)

    def test_occupied_preset_is_not_overwritten(self):
        self.fake.presets[31] = "Mine"
        with self.assertRaises(tankg_api.TankGError):
            self.send(preset_slot=31, save=True)
        self.assertEqual(self.fake.presets[31], "Mine")
        self.assertFalse({l.split()[0] for l in self.fake.log} & {"PSAVE", "CSAVE", "ISAVE", "LDBEGIN"})
        self.send(preset_slot=31, save=True, overwrite_preset=True)
        self.assertNotEqual(self.fake.presets[31], "Mine")

    def test_second_send_reuses_the_capture_slot(self):
        self.send(preset_slot=30, save=True)
        self.send(preset_slot=31, save=True)
        self.assertEqual(len(self.fake.caps), 1)
        self.assertEqual(sum(1 for l in self.fake.log if l.startswith("CSAVE")), 1)

    def test_a_library_slot_called_NAM_is_not_taken_for_this_capture(self):
        """Some .nam files are simply named "NAM" inside, and the pedal's own slot 0 is one: a match on name and
        size would send the preset to someone else's amp."""
        self.fake.caps[0] = ("NAM", 4892)
        r = self.send(preset_slot=31, save=True)
        self.assertNotEqual(r["cap_slot"], 0)
        self.assertEqual(self.fake.caps[0], ("NAM", 4892))                 # left exactly as it was
        self.assertEqual(len(self.fake.caps), 2)
        # and sending the same capture again does reuse the slot it made
        r2 = self.send(preset_slot=30, save=True)
        self.assertEqual(r2["cap_slot"], r["cap_slot"])
        self.assertEqual(len(self.fake.caps), 2)

    def test_different_captures_with_the_same_internal_name_get_separate_slots(self):
        import a2l1
        found, seen = {}, set()
        for f in sorted((HERE.parent / "profiles").rglob("*.stprofile")):
            p = json.loads(f.read_text(encoding="utf-8"))
            n = p.get("nam_path")
            if n and n not in seen and pathlib.Path(n).exists():
                seen.add(n)
                try:
                    b = a2l1.convert(n)
                except Exception:
                    continue
                # key: the name stored inside the file; value: {hash of the converted content: path} (SHA-1: every blob ends with its
                # own CRC-32, which makes the CRC of the whole blob the same constant for every capture)
                found.setdefault((b["name"], len(b["blob"])), {})[hashlib.sha1(b["blob"]).hexdigest()] = n
        pair = next((list(v.values()) for v in found.values() if len(v) > 1), None)   # same name, DIFFERENT content
        if pair is None:
            self.skipTest("no two different captures with the same internal name on this machine")
        slots = []
        for i, n in enumerate(pair[:2]):
            self.profile = dict(self.profile, nam_path=n)
            slots.append(self.send(preset_slot=20 + i, save=True)["cap_slot"])
        self.assertNotEqual(slots[0], slots[1])
        self.assertEqual(len(self.fake.caps), 2)

    def test_the_same_cab_ir_uses_one_library_slot(self):
        self.send(preset_slot=30, save=True)
        r = self.send(preset_slot=31, save=True)
        self.assertEqual(len(self.fake.irs), 1)
        self.assertEqual(r["ir_slot"], 0)
        self.assertEqual(sum(1 for l in self.fake.log if l.startswith("ISAVE")), 1)
        self.assertTrue(any("already in library slot" in m or "re-using slot" in m for m in self.msgs))

    def test_full_capture_library_is_refused_without_touching_it(self):
        for i in range(15):
            self.fake.caps[i] = ("other%d" % i, 1)
        with self.assertRaises(tankg_api.TankGError):
            self.send(preset_slot=31, save=True)
        self.assertEqual(len(self.fake.caps), 15)

    def test_rejected_capture_restores_audio_and_buffer(self):
        self.fake.fail_ldend = True
        with self.assertRaises(tankg_api.TankGError):
            self.send(preset_slot=31, save=True)
        self.assertEqual(self.fake.audio, 2)
        self.assertFalse(self.fake.presets)

    def test_dangerous_commands_are_refused(self):
        ped = live.Pedal("FAKE", allow=tankg_api.ALLOW, transport=self.fake)
        for c in ("PDEL 1", "PNEW", "PLOAD 1", "AMP 0", "GAIN 0 0", "CDEL 1"):
            with self.assertRaises(RuntimeError):
                ped.cmd(c)
        self.assertEqual(self.fake.log, [])

    def test_ir_prep_matches_the_javascript_rules(self):
        q, gain = tankg_api.prep_ir(ir_loader(None), 4076)
        self.assertLessEqual(len(q), 4076)
        self.assertEqual(int(np.max(np.abs(q))), 32767)
        self.assertIsInstance(gain, int)

    def test_heavy_sound_is_fitted_not_refused(self):
        heavy = HERE.parent / "profiles" / "Pink Floyd" / "Comfortably Numb (Pink Floyd) - Telecaster.stprofile"
        self.profile = json.loads(heavy.read_text(encoding="utf-8"))
        before = bridge.totals(bridge.build_plan(self.profile, bridge.load_catalogue(HERE / "fxt_catalogue.txt")))[0]
        self.send(preset_slot=31, save=False)                      # must not raise
        names = [b["name"] for b in self.fake.chain]
        self.assertNotIn("geq", names)                              # the EQ lives in the capture's own tone controls
        self.assertNotIn("eq", names)
        self.assertEqual(len(self.fake.ampeq), 6)                   # volume bass mid treble presence drive (dB x10)
        self.assertTrue(all(-240 <= v <= 120 for v in self.fake.ampeq))
        muff = next(b for b in self.fake.chain if b["name"] == "muff")
        self.assertEqual(muff["p"][3], 0)                           # oversampling off
        self.assertTrue(any(m.startswith("fitted:") for m in self.msgs))
        self.assertLess(before, 40000)

    def test_lighter_version_is_tried_when_the_pedal_says_no(self):
        from unittest import mock
        heavy = HERE.parent / "profiles" / "Pink Floyd" / "Comfortably Numb (Pink Floyd) - Telecaster.stprofile"
        self.profile = json.loads(heavy.read_text(encoding="utf-8"))
        plan = bridge.build_plan(self.profile, bridge.load_catalogue(HERE / "fxt_catalogue.txt"))
        first = sum(b.bt.cost_at(b.values.get("os_n")) for b in plan.blocks)
        self.fake.cpu_limit = first - 1000                          # the first attempt is refused by the pedal
        real, calls = bridge.fit_plan, [0]

        def fit(plan, *a, **k):
            calls[0] += 1
            if calls[0] == 1:                                       # the initial fitting inside build_plan
                return real(plan, *a, **k)
            gone = [b for b in plan.blocks if b.bt.name == "chorus"]   # the lighter version: no chorus
            for b in gone:
                plan.blocks.remove(b)
            return ["chorus: left out"] if gone else []
        with mock.patch.object(bridge, "fit_plan", fit):
            self.send(preset_slot=31, save=False)
        self.assertTrue(any("lighter version" in m for m in self.msgs))
        self.assertNotIn("chorus", [b["name"] for b in self.fake.chain])
        self.assertLessEqual(self.fake.used(), first - 1000)

    def test_capture_eq_numbers(self):
        flat = dict(self.profile, eq_on=False, in_gain=1.0, out_gain=10 ** (8 / 20))
        v, notes = bridge.fit_ampeq(flat)
        self.assertEqual((v["bass"], v["mid"], v["treble"], v["presence"], v["drive"]), (0, 0, 0, 0, 0))
        self.assertAlmostEqual(v["volume"], 8 + bridge.VOLUME_REF_DB - bridge.VOLUME_SHIFT_DB, places=3)
        boosted = dict(self.profile, eq_on=True, eq_band_gains=[6, 0, 0, 0, 0], in_gain=10 ** (20 / 20), out_gain=1.0)
        v, notes = bridge.fit_ampeq(boosted)
        self.assertGreater(v["bass"], 3)                            # a low-shelf boost lands on the pedal's Bass
        self.assertEqual(v["drive"], 12.0)                          # +20 dB input gain is limited to the pedal's +12
        self.assertTrue(any("limited" in n for n in notes))

    def _profile_with(self, **pedals):
        prof = json.loads(json.dumps(self.profile))
        for k in [k for k in prof["pedals"] if k.endswith("_on")]:
            prof["pedals"][k] = False
        prof["pedals"].update(pedals)
        prof["pedals"]["order"] = ["comp", "wah", "pitch", "fuzz", "dist", "od", "phaser", "amp", "chorus", "tremolo", "rotary", "delay", "reverb"]
        prof["hum_on"] = False
        return prof

    def test_new_solotone_pedals_map_to_native_ultrarig_blocks(self):
        types = bridge.load_catalogue(HERE / "fxt_catalogue.txt")

        def names(prof):
            return [b.bt.name for b in bridge.build_plan(prof, types).blocks]
        cases = [
            (dict(phaser_on=True, phaser_rate=1.5, phaser_depth=0.8, phaser_feedback=0.4), "phaser"),
            (dict(tremolo_on=True, tremolo_rate=6.0, tremolo_depth=0.7, tremolo_shape=0.5), "tremolo"),
            (dict(chorus_on=True, chorus_mode="vibrato"), "vibrato"),
            (dict(chorus_on=True, chorus_mode="doubler"), "doubler"),
            (dict(od_on=True, od_mode="bd2"), "bd2"),
            (dict(reverb_on=True, reverb_type="plate"), "plate"),
            (dict(reverb_on=True, reverb_type="spring"), "spring"),
            (dict(reverb_on=True, reverb_type="cloud"), "cloud"),
            (dict(reverb_on=True, reverb_type="room"), "reverb2"),
        ]
        for pedals, want in cases:
            prof = self._profile_with(**pedals)
            self.assertIn(want, names(prof), pedals)
        # parameters land in the block's own units and ranges
        plan = bridge.build_plan(self._profile_with(phaser_on=True, phaser_rate=1.5, phaser_depth=0.8,
                                                    phaser_feedback=0.4, tremolo_on=True, tremolo_rate=6.0,
                                                    tremolo_depth=0.7, tremolo_shape=0.5), types)
        ph = next(b for b in plan.blocks if b.bt.name == "phaser")
        self.assertEqual((ph.values["vel_chz"], ph.values["prof_pct"], ph.values["realim_pct"]), (150, 80, 40))
        tr = next(b for b in plan.blocks if b.bt.name == "tremolo")
        self.assertEqual((tr.values["vel_chz"], tr.values["prof_pct"], tr.values["forma_pct"]), (600, 70, 50))
        sp = bridge.build_plan(self._profile_with(reverb_on=True, reverb_type="spring", reverb_decay=0.8,
                                                  reverb_damp=0.3, reverb_mix=0.4), types)
        sb = next(b for b in sp.blocks if b.bt.name == "spring")
        self.assertEqual((sb.values["decay_pct"], sb.values["mezcla_pct"]), (80, 40))

    def test_envelope_filter_and_tone_bender_map_to_the_alpha_11_blocks(self):
        types = bridge.load_catalogue(HERE / "fxt_catalogue.txt")

        def block(name, **pedals):
            return next(b for b in bridge.build_plan(self._profile_with(**pedals), types).blocks if b.bt.name == name)
        env = block("envfilter", wah_on=True, wah_mode="envelope", wah_env_type="lowpass", wah_env_dir="down",
                    wah_env_sens=0.8, wah_env_decay=0.25, wah_range_lo=300, wah_range_hi=2400, wah_q=4.0)
        self.assertEqual((env.values["filtro_modo"], env.values["sens_pct"], env.values["decay_pct"]), (3, 80, 25))
        wah = block("envfilter", wah_on=True, wah_mode="envelope", wah_env_type="wah", wah_env_dir="down")
        self.assertEqual(wah.values["filtro_modo"], 7)
        for kind, up in (("lowpass", 0), ("bandpass", 1), ("highpass", 2), ("wah", 6)):
            self.assertEqual(block("envfilter", wah_on=True, wah_mode="envelope", wah_env_type=kind,
                                   wah_env_dir="up").values["filtro_modo"], up)
        tb = block("fuzzface", fuzz_on=True, fuzz_mode="tone_bender")
        self.assertEqual(tb.values["modelo"], 1)
        self.assertEqual(block("fuzzface", fuzz_on=True, fuzz_mode="fuzz_face").values["modelo"], 0)

    def test_rotary_maps_to_the_alpha_11_rotary_block(self):
        types = bridge.load_catalogue(HERE / "fxt_catalogue.txt")
        plan = bridge.build_plan(self._profile_with(rotary_on=True, rotary_speed="fast", rotary_fast=8.0,
                                                    rotary_mix=0.7), types)
        rot = next(b for b in plan.blocks if b.bt.name == "rotary")
        self.assertEqual((rot.values["velocidad"], rot.values["rapido_chz"], rot.values["mezcla_pct"]), (1, 800, 70))
        slow = bridge.build_plan(self._profile_with(rotary_on=True, rotary_slow=1.5, rotary_ramp=0.5), types)
        inertia = next(b for b in slow.blocks if b.bt.name == "rotary").values["inercia_pct"]
        self.assertEqual((int(inertia) >> 7, int(inertia) & 127), (15, 50))          # Slow rate 1.5 Hz in tenths, inertia 50 %
        self.assertFalse(any("Slow speed is fixed" in w for w in slow.warnings))
        brake = bridge.build_plan(self._profile_with(rotary_on=True, rotary_speed="brake"), types)
        self.assertEqual(next(b for b in brake.blocks if b.bt.name == "rotary").values["velocidad"], 2)

    def test_pitch_shifter_and_shimmer_reverb_map_to_pitchpro_and_shimmer(self):
        types = bridge.load_catalogue(HERE / "fxt_catalogue.txt")
        plan = bridge.build_plan(self._profile_with(pitch_on=True, pitch_semi=-7, pitch_cents=12, pitch_mix=0.65), types)
        pp = next(b for b in plan.blocks if b.bt.name == "pitchpro")
        self.assertEqual((pp.values["semitonos_st"], pp.values["cents_ct"], pp.values["mezcla_pct"]), (-7, 12, 65))
        two = bridge.build_plan(self._profile_with(pitch_on=True, pitch_level2=0.5), types)
        self.assertTrue(any("voice 2" in w for w in two.warnings))
        sh = bridge.build_plan(self._profile_with(reverb_on=True, reverb_type="shimmer", reverb_size=0.4,
                                                  reverb_decay=0.8, reverb_mix=0.3, reverb_shimmer=0.7), types)
        sb = next(b for b in sh.blocks if b.bt.name == "shimmer")
        self.assertEqual((sb.values["tamano_pct"], sb.values["decay_pct"], sb.values["mezcla_pct"], sb.values["shimmer_pct"]),
                         (40, 80, 30, 70))

    def test_smart_gate_maps_to_the_pedals_smart_gate(self):
        types = bridge.load_catalogue(HERE / "fxt_catalogue.txt")
        smart = dict(self._profile_with(), gate_threshold=0.01, gate_mode="smart", gate_range=45.0,
                     gate_release=200.0, gate_hush=0.7)
        plan = bridge.build_plan(smart, types)
        g = next(b for b in plan.blocks if b.bt.name == "sgate")
        self.assertEqual((g.values["umbral_db"], g.values["rango_att"], g.values["release_ms"], g.values["hush_pct"]),
                         (-40, 45, 200, 70))
        standard = bridge.build_plan(dict(self._profile_with(), gate_threshold=0.01), types)
        self.assertIn("gate", [b.bt.name for b in standard.blocks])
        self.assertNotIn("sgate", [b.bt.name for b in standard.blocks])

    def test_a_profile_using_the_new_pedals_sends_and_verifies(self):
        self.profile = dict(self._profile_with(phaser_on=True, tremolo_on=True, chorus_on=True, chorus_mode="doubler",
                                               reverb_on=True, reverb_type="spring"),
                            gate_threshold=0.01, gate_mode="smart", nam_path=self.profile["nam_path"],
                            ir_paths=self.profile.get("ir_paths"), ir_active=self.profile.get("ir_active", 0))
        r = self.send(preset_slot=31, save=False)
        sent = [b["name"] for b in self.fake.chain]
        for want in ("sgate", "phaser", "doubler", "tremolo", "spring"):
            self.assertIn(want, sent)
        self.assertEqual(len(self.fake.chain), len(r["plan"].blocks))

    def test_measured_calibration_is_applied(self):
        flat = dict(self.profile, name="Test profile", nam_path="x/amp one.nam", eq_on=False, in_gain=1.0, out_gain=10 ** (8 / 20))
        plain, _ = bridge.fit_ampeq(flat)
        calib = {"profiles": {"Test profile": {"volume_offset": -10.0, "drive": 0.0,
                                               "tone": {"low": 4.0, "lowmid": 0.0, "mid": -4.0, "high": 0.0}}},
                 "captures": {"amp one": {"volume_offset": -6.0}, "other": {"volume_offset": -1.0}},
                 "global": {"volume_offset": -3.0}}
        try:
            bridge._CALIB = calib
            v, notes = bridge.fit_ampeq(flat)
            self.assertAlmostEqual(v["volume"], plain["volume"] - 10.0, delta=1.0)        # measured (the tone fit shifts it a little)
            self.assertLess(v["bass"], plain["bass"])                                      # pedal was heavy in the lows: cut
            self.assertGreater(v["mid"], plain["mid"])                                     # and dull in the mids: lift
            self.assertTrue(any("measured on this pedal" in n for n in notes))
            v, _ = bridge.fit_ampeq(dict(flat, name="Something else"))
            self.assertAlmostEqual(v["volume"], plain["volume"] - 6.0, places=3)           # same capture
            v, _ = bridge.fit_ampeq(dict(flat, name="Something else", nam_path="y/unknown.nam"))
            self.assertAlmostEqual(v["volume"], plain["volume"] - 3.0, places=3)           # typical for all
            v, _ = bridge.fit_ampeq(dict(flat, in_gain=10 ** (6 / 20)))                    # drive changed: the measurement no longer applies
            self.assertAlmostEqual(v["volume"], plain["volume"] - 6.0, places=3)
        finally:
            bridge._CALIB = {}

    def test_ir_length_changes_cost_not_tone_first(self):
        plan = bridge.build_plan(self.profile, bridge.load_catalogue(HERE / "fxt_catalogue.txt"))
        ir = next(b for b in plan.blocks if b.bt.name == "ir")
        bridge.set_ir_length(ir, 21)
        mem21, ticks21 = ir.bt.mem + ir.extra_mem, ir.cost()
        bridge.set_ir_length(ir, 10)
        self.assertLess(ir.bt.mem + ir.extra_mem, mem21)
        self.assertLess(ir.cost(), ticks21)
        self.assertEqual(ir.values["largo_ms"], 10)
        self.assertEqual(ir.ir_samples, 480)

    def test_slot_labels(self):
        self.assertEqual([tankg_api.slot_label(n) for n in (0, 3, 4, 31)], ["1A", "1D", "2A", "8D"])


if __name__ == "__main__":
    unittest.main()
