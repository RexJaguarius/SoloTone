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
                  slot_label=lambda n: '%d%s' % (n // 4 + 1, 'ABCD'[n % 4]))
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
            assert 'Tank-G' in tabs and tabs.index('Tank-G') < tabs.index('Settings')

            # the panel says up front that the pedal needs the UltraRig firmware, with a working link to it
            def labels(w):
                out = []
                for c in w.winfo_children():
                    if isinstance(c, tk.Label):
                        out.append(c.cget('text'))
                    out += labels(c)
                return out
            texts = ' '.join(labels(app._notebook.nametowidget(app._notebook.tabs()[tabs.index('Tank-G')])))
            assert 'Needs the UltraRig firmware' in texts and 'stock M-VAVE firmware' in texts
            opened = []
            real_open = st.App._open_url
            st.App._open_url = staticmethod(lambda url: opened.append(url))
            try:
                app._notebook.select(tabs.index('Tank-G'))
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
