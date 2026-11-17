"""Scenes: four variations of a profile, switched live, saved with the profile."""
import copy
import json
import pathlib
import sys
import tempfile
import time
import tkinter as tk
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import solotone as st


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = pathlib.Path(tempfile.mkdtemp())
        st.SESSION_FILE = tmp / 's.json'
        st.MIDI_MAP_FILE = tmp / 'm.json'
        st.ERROR_LOG = tmp / 'e.log'
        st.CRASH_LOG = tmp / 'c.log'
        cls.tmp = tmp
        cls.app = st.App()
        cls.app.geometry('1180x860+20+20')
        cls.app.update()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app.destroy()
        except Exception:
            pass

    def setUp(self):
        self.app._scene_reset()
        self.app._scene_refresh_ui()

    def test_starts_on_scene_a_with_three_empty(self):
        a = self.app
        self.assertEqual(a._scene_active, 0)
        self.assertEqual([x is None for x in a._scenes], [True, True, True, True])
        self.assertEqual(len(a._scene_btns), 4)

    def test_empty_scene_starts_as_a_copy_then_keeps_its_own_edits(self):
        a = self.app
        p = a.amp.pedals
        a._set_pedal_on('od', False)
        a._scene_select(1)
        self.assertEqual(a._scene_active, 1)
        self.assertFalse(p.od_on)                       # a copy of A
        a._set_pedal_on('od', True)                     # edit scene B
        a._pedal_widget_sync('od_drive', 0.9); p.od_drive = 0.9
        a._scene_select(0)
        self.assertFalse(p.od_on)                       # back to A, unchanged
        self.assertNotEqual(p.od_drive, 0.9)
        a._scene_select(1)
        self.assertTrue(p.od_on)
        self.assertAlmostEqual(p.od_drive, 0.9)
        # the open control moved with it
        self.assertTrue(a._pedal_on_vars['od'].get())

    def test_scene_leaves_chain_order_and_capture_alone(self):
        a = self.app
        order = list(a.amp.pedals.order)
        a._scene_select(2)
        a._move_fx('od', -1)
        moved = list(a.amp.pedals.order)
        a._scene_select(0)
        self.assertEqual(a.amp.pedals.order, moved)     # order is the rig, not the scene
        self.assertNotIn('order', a._scenes[0]['pedals'])
        a.amp.pedals.order = order
        a._relayout_pedals()

    def test_a_mode_change_rebuilds_and_applies(self):
        a = self.app
        p = a.amp.pedals
        a._scene_select(1)
        p.delay_mode = 'digital' if p.delay_mode != 'digital' else 'tape'
        want = p.delay_mode
        a._scene_select(0)
        self.assertNotEqual(p.delay_mode, want)
        a._scene_select(1)
        self.assertEqual(p.delay_mode, want)

    def test_scenes_round_trip_through_a_profile_file(self):
        a = self.app
        p = a.amp.pedals
        a._set_pedal_on('reverb', False)
        a._scene_select(1)
        a._set_pedal_on('reverb', True)
        a._scene_names[1] = 'Solo 2'
        d = a._gather_profile_dict('t')
        self.assertEqual(d['scenes']['active'], 1)
        self.assertEqual(d['scenes']['names'][1], 'Solo 2')
        self.assertIsNotNone(d['scenes']['slots'][0])
        path = self.tmp / 't.stprofile'
        path.write_text(json.dumps(d))
        a._scene_reset(); a._set_pedal_on('reverb', False)
        a._load_profile_path(str(path))
        self.assertEqual(a._scene_active, 1)
        self.assertEqual(a._scene_names[1], 'Solo 2')
        self.assertTrue(p.reverb_on)                    # the profile's own values are the active scene
        a._scene_select(0)
        self.assertFalse(p.reverb_on)

    def test_a_profile_without_scenes_is_just_scene_a(self):
        a = self.app
        d = a._gather_profile_dict('old')
        d.pop('scenes')
        path = self.tmp / 'old.stprofile'
        path.write_text(json.dumps(d))
        a._load_profile_path(str(path))
        self.assertEqual(a._scene_active, 0)
        self.assertIsNotNone(a._scenes[0])
        self.assertEqual([x is None for x in a._scenes[1:]], [True, True, True])

    def test_damaged_scene_block_is_ignored(self):
        a = self.app
        d = a._gather_profile_dict('bad')
        d['scenes'] = {'active': 99, 'names': [1, None, 'ok'], 'slots': ['x', 7, {'pedals': 3}, None]}
        path = self.tmp / 'bad.stprofile'
        path.write_text(json.dumps(d))
        a._load_profile_path(str(path))
        self.assertEqual(a._scene_active, 0)
        self.assertEqual(a._scene_names, ['', '', 'ok', ''])

    def test_midi_and_key_targets_exist_and_switch(self):
        a = self.app
        for k in 'abcd':
            self.assertIn(f'scene_{k}', st.MIDI_TARGETS_BY_ID)
            self.assertIn(f'scene_{k}', st.KEYBIND_TARGETS_BY_ID)
        st.MIDI_TARGETS_BY_ID['scene_c']['action'](a, None)
        self.assertEqual(a._scene_active, 2)
        st.MIDI_TARGETS_BY_ID['scene_next']['action'](a, None)
        self.assertEqual(a._scene_active, 3)
        st.MIDI_TARGETS_BY_ID['scene_next']['action'](a, None)
        self.assertEqual(a._scene_active, 0)

    def test_clear_and_rename(self):
        a = self.app
        a._scene_select(1); a._scene_select(0)
        a._scene_names[1] = 'x'
        a._scene_clear(0)                               # the active one can't be cleared
        self.assertIsNotNone(a._scenes[0])
        a._scene_clear(1)
        self.assertIsNone(a._scenes[1])
        self.assertEqual(a._scene_names[1], '')

    def test_library_draws_the_scenes_of_the_selected_profile(self):
        a = self.app
        a._scene_select(1)
        a._set_pedal_on('od', True)
        a._scene_names[1] = 'Lead'
        d = a._gather_profile_dict('lib')
        a._lib_scenes_render(d)
        box = a._lib_scene_box
        grid = box.winfo_children()[1]
        cards = grid.winfo_children()
        self.assertEqual(len(cards), 4)
        def texts(w):
            out = []
            for c in w.winfo_children():
                if isinstance(c, tk.Label):
                    out.append(str(c.cget('text')))
                out += texts(c)
            return out
        self.assertIn('Lead', texts(cards[1]))
        self.assertIn('empty', texts(cards[2]))
        # a profile from before scenes shows its own settings as scene A
        d.pop('scenes')
        a._lib_scenes_render(d)
        cards = a._lib_scene_box.winfo_children()[1].winfo_children()
        self.assertIn('empty', texts(cards[1]))
        self.assertNotIn('empty', texts(cards[0]))
        a._lib_scenes_render(None)
        self.assertEqual(a._lib_scene_box.winfo_children(), [])

    def test_light_switch_is_fast(self):
        a = self.app
        a._scene_select(1)
        a._set_pedal_on('od', True); a._set_pedal_on('delay', False)
        a._scene_select(0)
        a._set_pedal_on('od', False); a._set_pedal_on('delay', True)
        a.update()
        t = time.time()
        for _ in range(4):
            a._scene_select(1); a.update(); a._scene_select(0); a.update()
        self.assertLess((time.time() - t) / 8, 0.4)


if __name__ == '__main__':
    unittest.main()
