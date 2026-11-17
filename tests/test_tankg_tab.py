"""The optional Tank-G tab, tested against a fake add-on (no pedal, no serial port).

Run:  python tests/test_tankg_tab.py
"""
import pathlib
import sys
import tempfile
import time
import types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import solotone as st  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import messagebox  # noqa: E402


class FakeAPI(types.SimpleNamespace):
    pass


class FakeLive:
    """Stands in for tankg_api.LiveSession: records what the app would send to the pedal."""
    instances = []

    def __init__(self, port, audio_loader=None, log=print, progress=None, transport=None):
        self.port, self.started, self.syncs, self.closed = port, None, [], False
        FakeLive.instances.append(self)

    def start(self, profile):
        self.started = profile
        return {}

    def sync(self, profile):
        self.syncs.append(profile)

    def close(self):
        self.closed = True


def make_fake(presets=None):
    calls = []

    def read_state(port):
        return dict(firmware='PONG fake', presets=dict(presets or {}), caps={}, cap_slots=15, irs={},
                    ir_slots=8, chain_blocks=0, cpu='')

    def send_sound(profile, port, preset_slot=None, save=True, overwrite_preset=False, audio_loader=None,
                   log=print, progress=None, **kw):
        calls.append(dict(profile=profile, port=port, slot=preset_slot, save=save, overwrite=overwrite_preset,
                          loader=audio_loader))
        log('fake: sending')
        if progress:
            progress(0.5, 'Halfway')
        return {}

    api = FakeAPI(available=lambda: (True, ''),
                  find_ports=lambda: [('COM3', 'Some Bluetooth port', False), ('COM9', 'USB Serial Device', True)],
                  read_state=read_state, send_sound=send_sound,
                  slot_label=lambda n: '%d%s' % (n // 4 + 1, 'ABCD'[n % 4]), LiveSession=FakeLive)
    return api, calls


def pump(app, cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_loader_returns_none_without_the_addon():
    with tempfile.TemporaryDirectory() as d:
        assert st._load_tankg_api([d]) is None


def test_loader_finds_an_addon_folder():
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / 'tankg_api.py').write_text('MARK = 42\n')
        m = st._load_tankg_api([d])
        assert m is not None and m.MARK == 42


def test_tab_flow_with_a_fake_addon():
    api, calls = make_fake(presets={0: 'Old sound', 1: 'Another'})
    st._load_tankg_api_real = st._load_tankg_api
    st._load_tankg_api = lambda *a, **k: api
    asked = []
    answers = []
    real_ask, real_err = messagebox.askyesno, messagebox.showerror
    messagebox.askyesno = lambda title, msg, **kw: (asked.append((title, msg)), answers.pop(0))[1]
    messagebox.showerror = lambda *a, **k: asked.append(('error', a))
    with tempfile.TemporaryDirectory() as tmp:
        st.SESSION_FILE = pathlib.Path(tmp) / 's.json'
        st.MIDI_MAP_FILE = pathlib.Path(tmp) / 'm.json'
        app = st.App()
        try:
            app.update()
            tabs = [app._notebook.tab(i, 'text').strip() for i in range(app._notebook.index('end'))]
            assert tabs == ['Rig', 'Play', 'Library', 'Settings']           # the Tank-G section lives in Settings, as a pill
            assert 'tankg' in app._settings_pages

            # the panel says up front that the pedal needs the UltraRig firmware, with a working link to it
            def labels(w):
                out = []
                for c in w.winfo_children():
                    if isinstance(c, tk.Label):
                        out.append(c.cget('text'))
                    out += labels(c)
                return out
            texts = ' '.join(labels(app._settings_pages['tankg']))
            assert 'Needs the UltraRig firmware' in texts and 'stock M-VAVE firmware' in texts
            opened = []
            real_open = st.App._open_url
            st.App._open_url = staticmethod(lambda url: opened.append(url))
            try:
                app._notebook.select(st.App.TAB_SETTINGS)
                app._pill_selectors['_settings_sel']('tankg')
                app.update()
                app.tg_ultrarig_link.event_generate('<Button-1>', x=2, y=2)
                app.update()
            finally:
                st.App._open_url = real_open
            assert opened == [st.ULTRARIG_URL] and st.ULTRARIG_URL.startswith('https://github.com/')

            # the pedal's port is picked automatically
            assert app.tg_port_var.get().startswith('COM9') and '(Tank-G)' in app.tg_port_var.get()

            # Read pedal fills in the slots
            app._tg_read()
            assert pump(app, lambda: app._tg_state is not None and not app._tg_busy)
            vals = list(app.tg_slot_box.cget('values'))
            assert len(vals) == 32 and 'in use' in vals[0] and 'empty' in vals[2]
            assert app.tg_slot_box.current() == 2             # first empty slot is pre-selected

            # Play on pedal: no dialog, nothing saved
            app._tg_play()
            assert pump(app, lambda: calls and not app._tg_busy)
            assert calls[-1]['save'] is False and calls[-1]['slot'] is None and not asked

            # The Library page has its own "Load to Tank-G": it loads the selected profile and plays it, saving nothing
            assert app.lib_tg_btn is not None and str(app.lib_tg_btn.cget('state')) != 'disabled'
            prof = pathlib.Path(tmp) / 'lib.stprofile'
            import json
            prof.write_text(json.dumps(app._gather_profile_dict('lib')))
            app._lib_sel = None
            app._library_to_tankg()
            assert 'Pick a profile' in app.lib_tg_lbl.cget('text') and len(calls) == 1
            app._lib_sel = str(prof)
            app._library_to_tankg()
            assert pump(app, lambda: len(calls) == 2 and not app._tg_busy)
            assert calls[-1]['save'] is False and calls[-1]['port'] == 'COM9' and not asked
            assert app._current_profile_path == str(prof)
            assert 'scenes' in calls[-1]['profile']
            assert 'nothing was saved' in app.lib_tg_lbl.cget('text')
            calls.pop()

            # Live mode: SoloTone as the remote of the pedal, nothing stored
            assert app.live_btn is not None
            FakeLive.instances.clear()
            app._live_toggle()                                    # no amp capture loaded: nothing to play, refused
            assert app._live_state == 'off' and not FakeLive.instances and 'capture' in app.live_lbl.cget('text')
            app._current_nam_path = 'amp.nam'
            app._live_toggle()
            assert pump(app, lambda: app._live_state == 'live')
            ses = FakeLive.instances[-1]
            assert ses.port == 'COM9' and ses.started is not None and 'scenes' in ses.started
            assert app._tg_busy                                   # the Tank-G tab's own buttons are locked meanwhile
            app._set_pedal_on('od', not app.amp.pedals.od_on)     # a pedal switched on or off is sent
            assert pump(app, lambda: len(ses.syncs) == 1)
            assert ses.syncs[-1]['pedals']['od_on'] == app.amp.pedals.od_on
            app._scene_select(1)                                  # scene B starts as a copy of A: nothing new to send
            app._set_pedal_on('delay', not app.amp.pedals.delay_on)
            assert pump(app, lambda: len(ses.syncs) == 2)
            app._scene_select(0)                                  # back to A: its state is sent
            assert pump(app, lambda: len(ses.syncs) == 3)
            assert ses.syncs[-1]['scenes']['active'] == 0
            n = len(ses.syncs)
            for _ in range(10):
                app.update(); time.sleep(0.05)
            assert len(ses.syncs) == n                            # idle: nothing is sent
            app._current_nam_path = 'other.nam'                   # a different capture sends the sound again
            assert pump(app, lambda: len(FakeLive.instances) == 2 and app._live_state == 'live')
            assert FakeLive.instances[0].closed and FakeLive.instances[1].started is not None
            app._live_toggle()                                    # stop
            assert pump(app, lambda: app._live_state == 'off' and not app._tg_busy)
            assert FakeLive.instances[-1].closed

            # Loading another profile while live stops live mode (and does not restart it)
            app._current_nam_path = 'amp.nam'
            app._live_toggle()
            assert pump(app, lambda: app._live_state == 'live')
            n_inst = len(FakeLive.instances)
            app._load_profile_path(str(prof))
            assert pump(app, lambda: app._live_state == 'off' and not app._tg_busy)
            assert FakeLive.instances[-1].closed and len(FakeLive.instances) == n_inst
            app._current_nam_path = None

            # Save to a free slot: one confirmation
            answers[:] = [True]
            app._tg_save()
            assert pump(app, lambda: len(calls) == 2 and not app._tg_busy)
            assert calls[-1]['save'] is True and calls[-1]['slot'] == 2 and calls[-1]['overwrite'] is False
            assert len(asked) == 1 and 'preset 1C' in asked[0][1]
            assert calls[-1]['loader'] is st.ir_to_48k_mono
            assert 'nam_path' in calls[-1]['profile'] and 'pedals' in calls[-1]['profile']

            # Cancelling the first confirmation sends nothing
            answers[:] = [False]
            app._tg_save()
            assert pump(app, lambda: len(asked) == 2 and not app._tg_busy)
            assert len(calls) == 2

            # An occupied slot needs a second, explicit confirmation
            app.tg_slot_box.current(0)
            app.tg_slot_box.event_generate('<<ComboboxSelected>>')
            app.update()
            answers[:] = [True, False]                         # agree to send, refuse to replace
            app._tg_save()
            assert pump(app, lambda: len(asked) == 4 and not app._tg_busy)
            assert len(calls) == 2 and 'already holds' in asked[3][1]
            answers[:] = [True, True]
            app._tg_save()
            assert pump(app, lambda: len(calls) == 3 and not app._tg_busy)
            assert calls[-1]['slot'] == 0 and calls[-1]['overwrite'] is True

            # A failure on the pedal is shown, not swallowed
            def boom(*a, **k):
                raise RuntimeError('The pedal said no.')
            api.send_sound = boom
            before = len(asked)
            app._tg_play()
            assert pump(app, lambda: len(asked) > before and not app._tg_busy)
            assert asked[-1][0] == 'error' and 'The pedal said no.' in asked[-1][1][1]
            assert 'The pedal said no.' in app.tg_status_lbl.cget('text')

            # The chosen port survives into the saved session
            app._save_session()
            assert 'COM9' in st.SESSION_FILE.read_text()
        finally:
            messagebox.askyesno, messagebox.showerror = real_ask, real_err
            st._load_tankg_api = st._load_tankg_api_real
            app.destroy()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('PASS', t.__name__)
    print(f'\n{len(tests)} tests passed.')
