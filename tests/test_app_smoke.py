"""
Full-app GUI smoke test — builds a real App() (all six tabs, the footer
tuner strip, every widget) and tears it down. Needs a display: on
Windows/macOS that's just whatever's already there; on Linux CI this
needs a virtual display (Xvfb) since there's no real screen.

This exists to catch cross-platform Tkinter/dependency surprises before
they reach a user — a missing system Tk package, a font/theme quirk, an
import that only breaks on one OS — not to verify tone or feel, which
still needs a human on real hardware (see version.py's changelog for
plenty of bugs this kind of test would never have caught).

Run directly: python tests/test_app_smoke.py
Or with pytest: pytest tests/test_app_smoke.py
"""

import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import solotone as st


def test_app_builds_and_closes_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        # Never touch a real user's session/MIDI-map files from a test run.
        st.SESSION_FILE = tmp_path / 'test_session.json'
        st.MIDI_MAP_FILE = tmp_path / 'test_midi.json'

        app = st.App()
        try:
            app.update()
            assert app.winfo_exists()
            # Touch each tab at least once — building a tab's widgets is
            # where a cross-platform surprise (a missing font, an
            # unavailable Tk feature) is most likely to actually surface.
            from tkinter import ttk
            nb = next(w for w in app.winfo_children() if isinstance(w, ttk.Notebook))
            for i in range(len(nb.tabs())):
                nb.select(i)
                app.update()
        finally:
            app.destroy()


if __name__ == '__main__':
    test_app_builds_and_closes_cleanly()
    print('PASS test_app_builds_and_closes_cleanly')
