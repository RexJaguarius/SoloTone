"""
Isolated python-rtmidi probe — no Tkinter involved at all.

Exists specifically to pin down a macOS-only CI crash first seen on the
build-78 GitHub Actions run: a *fatal* (uncatchable) Python error,
"PyEval_RestoreThread: the function must be called with the GIL held,
but the GIL is released", inside rtmidi.MidiIn().get_ports(), during
the very first Settings-tab build of a fresh App(). Never seen on the
Linux or Windows runners in the same matrix run.

Matches a known class of python-rtmidi/CoreMIDI issue (see
thestk/rtmidi#262 and related reports) tied to repeatedly constructing
and discarding MidiIn/CoreMIDI client objects. The fix applied in
MidiController.list_ports() (solotone.py) reuses a single persistent
MidiIn instance instead of a fresh one per call — this file calls
list_ports() directly, several times in a row, with no Tk app around
it at all, specifically to catch a regression back to the old
construct-and-discard pattern before it ever reaches a real app build.

A genuinely fatal error here will abort the whole test process (that's
the nature of the bug being guarded against) rather than raise a normal
Python exception — if this file's process exits non-zero or doesn't
print PROBE_OK, treat that as a real regression, not a flaky test.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import solotone as st


def test_repeated_list_ports_calls_do_not_crash():
    print('HAS_MIDI:', st.HAS_MIDI)
    if not st.HAS_MIDI:
        print('python-rtmidi not installed — nothing to probe, skipping.')
        return
    # The original bug reproduced on the very first call inside a fresh
    # App() build; calling it several times in a row here is a stricter
    # check than that (and would also catch a regression to a fresh
    # MidiIn() per call, which is what actually caused it).
    for i in range(5):
        ports = st.MidiController.list_ports()
        print(f'  call {i}: {ports}')
    assert isinstance(ports, list)


if __name__ == '__main__':
    test_repeated_list_ports_calls_do_not_crash()
    print('PROBE_OK')
