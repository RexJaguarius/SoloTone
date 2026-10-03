# SoloTone — Guitar & Bass Amp Processor · Pedals · Looper · Tuner · Metronome

A fully graphical desktop app (Tkinter) for guitarists and
bassists: a real-time amp/FX processor, a pedal-effects chain, a
multi-layer looper, a chromatic tuner (with a dedicated Bass mode), and a
sample-accurate metronome — all running locally, with no telemetry and no
network access.

## Features

**Amp / FX Processor** (I/O & Levels, Signal Chain tabs)
- Low-latency duplex audio stream: reads your instrument input, writes
  processed audio to your output device in real time
- **NAM (Neural Amp Modeler) amp modeling** via a from-scratch numpy
  WaveNet inference engine — no PyTorch required. Loads `.nam` capture
  files (WaveNet architecture, including SlimmableContainer-wrapped
  files); LSTM-architecture captures are not supported
- Up to **5 cabinet IR slots** (WAV impulse responses — 8/16/24/32-bit
  integer PCM, 32/64-bit float, and `WAVE_FORMAT_EXTENSIBLE` headers, all
  parsed directly rather than through Python's stdlib `wave` module,
  which rejects float and EXTENSIBLE WAVs outright — auto-resampled to
  44.1 kHz), each with an editable label, A/B tagging, and instant A/B
  toggle; overlap-add FFT convolution, sized to the nearest fast FFT
  length so an unlucky IR length can't quietly cost 10x the CPU it should
- **Blend A/B** — mixes the A and B cabinet IRs at an adjustable ratio
  instead of only ever hearing one or the other; both the on/off toggle
  and the blend ratio are mappable to a keybind or MIDI knob like
  anything else in the app
- **5-band EQ** (low shelf 100 Hz, peaking 400 Hz/1 kHz/2.5 kHz, high shelf
  6 kHz, ±15 dB each) laid out as vertical faders in a row, like a
  hardware graphic EQ pedal, plus a separate Level fader (±15 dB) for
  trimming overall output back to a usable level after a heavy cut/boost
- Noise gate with fixed 3 ms attack / 60 ms release
- **Hum Filter** — cascaded notch filters at the mains frequency and its
  next two harmonics (50/60 Hz), applied ahead of everything else
  including the tuner
- Input gain (−12 to +24 dB) and output gain (−24 to +12 dB)
- **Host API selector** (WASAPI/WDM-KS/DirectSound/MME) so the device
  lists match Windows' own Sound Settings instead of showing every
  physical device duplicated once per API
- **Audio device selections persist across sessions** (Host API, Input,
  Output, Tuner input) — if a saved device is gone next launch (unplugged,
  renamed), it falls back to System default with a warning naming exactly
  which one couldn't be found, instead of silently picking the wrong
  device
- The I/O & Levels tab's sections (Audio Devices, Levels, Noise Gate, Hum
  Filter) behave as an accordion, same as Pedals — Audio Devices starts
  open, the rest start closed. **Start Amp** sits above the accordion in
  a fixed row, always visible regardless of which section is open or how
  far the tab is scrolled
- Adjustable audio block size (256/512/1024/2048 samples — smaller sizes
  were dropped after real-world testing showed they can't keep up with a
  full NAM+IR chain), live xrun counter and latency readout
- NAM and IR file paths persist across sessions and reload automatically
  on launch, and the Load NAM/Load IR dialogs remember the last folder
  browsed instead of always resetting to the bundled `nam/`/`irs/` folders

**Pedals** (pre-amp effects chain)
- Tuner Mute
- **Compressor** (tube-style) — Threshold/Ratio/Attack/Release/Makeup gain
  with a soft-knee gain curve, plus a Warmth control that blends in
  asymmetric tube-style saturation
- Wah — Manual mode (position slider, or a keybind/MIDI Up-Down pair)
  or Auto sweep mode (Rate + Intensity), both working within an adjustable
  Range (Hz) instead of a fixed sweep band; the default range is
  guitar-voiced, with an in-app hint that bass typically wants it lower
  (e.g. 150–800 Hz)
- Fuzz — Fuzz Face (asymmetric germanium-style clip) or Big Muff
  (cascaded hard clip with Full/Flat/Scoop tone voicings)
- **Distortion** — 4 selectable voicings: Boss DS-1 (boxy, compressed hard
  clip), ProCo RAT, MXR Distortion+ (asymmetric diode clip), and Metal
  (cascaded clipping with a mid-scoop EQ)
- Overdrive / Boost — Tubescreamer-style mid-hump drive, **Klon**
  (transparent parallel clean + treble-lifted drive), or clean boost
- Chorus / Flanger (rate, depth, mix)
- Delay — Tape or Digital voicing, manual time or BPM-linked to the
  metronome (dotted-8th / quarter / half note)
- Schroeder Reverb (size, mix)
- Every effect has its own bypass, so pedals can be clicked in and out
  independently while playing
- **Per-pedal mini-presets** — each pedal's own section has Save
  Preset…/Load Preset… buttons, capturing just that pedal's settings
  (`.stpedal` files) separately from Tone Profiles, so a favorite pedal
  setting can be reused across different amp profiles instead of being
  locked inside whichever one it was saved in. Loading refuses a preset
  saved for a different pedal rather than silently applying the wrong
  fields
- **Reorderable chain** — Move up/down controls on each pedal let you
  restack the signal order (e.g. fuzz before wah vs. after) instead of a
  fixed sequence; the order persists across sessions. A **Reset to
  Default Order** button restores the factory sequence (Compressor →
  Wah → Fuzz → Distortion → Overdrive → Chorus/Flanger → Delay → Reverb)
  in one click
- **One pedal's controls open at a time** — each pedal's section
  expands/collapses like an accordion: opening one closes whichever
  other was open, so the tab doesn't turn into a long scroll of every
  pedal's controls at once. Compressor is open by default; which pedal
  is open persists across sessions like everything else in Settings
- **Pedal Status strip** — a row of small icons at the top of the tab,
  one per pedal in the current chain order, each with an LED dot that
  lights up **red** when that pedal is on (fixed, regardless of the
  app's accent color — see Appearance). Click most of an icon to jump
  straight to that pedal's own section below, expanding it if it was
  closed; click directly on the LED itself instead to toggle that pedal
  on/off right from the strip, without opening its section at all.
  Requires the optional `Pillow` package (see Setup); the rest of
  the app works the same without it, just without this strip

**Tone Profiles** (Signal Chain tab)
- **Save Profile… / Load Profile…** capture a whole recallable sound in
  one file: NAM path, all 5 IR paths + active slot + labels, the entire
  pedal chain (every effect's parameters plus chain order), any MIDI
  controller mappings for the pedals, 5-band EQ + Level, noise gate
  threshold, input/output gain, hum filter, and the guitar/bass tuning
  preset
- Stored as JSON with its own `.stprofile` extension, in their own
  `profiles/` folder (alongside `nam/` and `irs/`) — built to extend, with
  a format-version number separate from the app version
- Loading a profile whose NAM or IR file can't be found shows an error
  naming exactly which file(s), while still applying everything else in
  the profile
- A profile is a whole sound, not a patch — loading one clears the
  current NAM and all 5 IR slots first, rather than merging with whatever
  was loaded before it

**Looper**
- Up to **10 layers**, loop length 1–60 s (10 preset lengths)
- Manual mode (arm/record/stop per layer) and Auto mode (hands-free
  sequential recording)
- Records the amp's fully processed output (post-NAM, post-IR, post-EQ) —
  the metronome click is mixed in only at the very final output stage,
  after the looper's recording tap, so muted or not, it's never
  captured into a loop layer
- Progress bar, per-layer status dots, and a layer list with durations
- **Per-layer mute** — a Mute/Unmute button on each layer row silences just
  that layer's playback (recording is unaffected); toggle any combination
  on the fly while the loop keeps playing
- **Export to WAV** — each layer has its own Export… button for its raw,
  unmixed audio, and a top-level Export Mix… button saves the full loop
  exactly as currently heard (muted layers excluded, at the current
  volume) — both write standard 16-bit PCM WAV files any DAW can open
- Integrated directly into the amp's audio callback — no separate stream

**Metronome**
- Tempo 30–250 BPM, via +/- buttons, a slider, or **tap tempo**
- Selectable beats-per-measure (1–12) with the downbeat visually accented
- Subdivisions: quarter notes, eighth notes, triplets, sixteenth notes
- Live flashing beat-light display on the Metronome tab, plus a small
  **persistent beat light** in the footer strip (visible on every tab) so
  you can watch the beat while looking at any other part of the app
- Volume control, and a **Mute** toggle that silences the click audio
  while leaving the beat scheduling (and both beat-light indicators)
  running — useful for a silent visual metronome while recording loops
- **5 synthesized sound kits** (no audio samples/downloads — everything is
  generated on the fly with numpy): Classic Beep, Mechanical Click, Wood
  Block, Cymbal & Crash (additive mix buffer so the crash rings out
  naturally under following beats), Digital Blip
- A **Preview** button plays the accent/main/sub sounds of the selected
  kit so you can audition before starting
- Sample-accurate click scheduling (runs in the audio callback itself, so
  timing doesn't drift the way a `sleep()`-based loop would)

**Tuner** (persistent footer strip, visible on every tab)
- Real-time chromatic pitch detection from your microphone or the amp's
  raw input (autocorrelation with parabolic interpolation for sub-cent
  precision), tracking as low as ~40 Hz in Guitar mode
- Large note + octave display, frequency readout, and a sweeping cents
  needle gauge (±50 cents)
- Adjustable A4 calibration, 430–450 Hz
- Transposing-instrument display modes: Concert (C), Bb, Eb, F
- Built-in reference-tone generator (pitch pipe) for all 12 notes
- **Guitar tuning presets** — 14 built-in tunings (Standard, Eb/D
  standard, Drop D/C/B/A, Open G/D/E/A/C, DADGAD, Double Drop D) with a
  string readout that auto-highlights the closest string by cents offset
  (accent color in tune, amber close, red off)
- **Panic** — a button in the footer's bottom row (and a keybind/MIDI
  target, mappable like anything else) that immediately silences all
  output — the live signal, the looper's own already-playing-back loop
  layers, and the metronome click — without stopping or resetting
  anything underneath, so un-panicking picks back up exactly in sync
- **Guitar / Bass instrument toggle** — switches the tuning dropdown to 3
  bass presets (Standard EADG, 5-String BEADG, Drop D DADG), shows only as
  many string boxes as that tuning has strings (4 or 5, not always 6),
  and retunes the detector for bass's lower range (floor down to ~24 Hz
  for a 5-string low B, ceiling lowered too — a naive autocorrelation
  stays falsely self-correlated at very short lags for a fundamental far
  enough below the ceiling, which silently defeats detection otherwise —
  and a longer analysis window). A saved session or profile with a bass
  tuning switches the toggle back to bass automatically on load

**Settings**
- **Sections are collapsible** — click any section's title bar (▼/▶) to
  collapse or expand it; each section remembers its state across
  restarts. The Pedals tab's 8 effect sections are the one exception —
  they behave as an accordion (see Pedals, above) instead of collapsing
  independently
- **Keybinds** — maps any keyboard key to any of the same functions the
  MIDI Controller section below can map to a knob or pad, via the same
  generic "Learn" system (click Learn, then press a key). A knob-style
  control (pedal drive/mix/rate, EQ bands + Level, input/output gain,
  noise gate threshold, wah position) becomes an **Up/Down key pair**
  instead of one key — a keypress is a discrete nudge, not a knob's
  continuous sweep — stepping by a shared, configurable percentage of
  that control's range. Toggle/trigger functions (pedal on/off, tap
  tempo, transport, etc.) map to a single key, with the same tap-toggles/
  hold-cycles-mode behavior as a MIDI pad (see below). Global (skips
  automatically while a text field like an IR label or the BPM spinbox
  has focus). Mappings persist in the app's saved settings, independent
  of whichever Tone Profile is loaded — a keyboard layout is part of your
  setup, not part of a particular sound
- **Window** — keep the app window always on top, and optionally confirm
  before quitting while the amp is still running
- **Appearance** — pick a custom accent color and background color with
  a real color picker (default accent is teal, `#1F7E89`).
  Choosing either rebuilds the whole window immediately — whichever tab
  you're on stays selected — and both colors are saved with the rest of
  your settings. A **Reset to Default Theme** button restores the
  factory colors in one click. The Pedal Status strip's on-LEDs are the
  one exception — they're always red, not the chosen accent (see Pedals,
  above)
- **MIDI Controller** — map any class-compliant USB-MIDI controller's
  knobs (Control Change) and pads/buttons (Note On/Off) to SoloTone
  controls via a generic "Learn" system (click Learn, then move the
  knob or hit the pad — nothing hardcoded to one device). Knobs drive
  ~20 continuous controls (pedal drive/mix/rate, all 5 EQ bands + Level,
  input/output gain, noise gate threshold, wah position); pads toggle
  every pedal on/off plus the tuner mute, or fire one-shot actions (tap
  tempo, amp/metronome/tuner start-stop, loop record/stop, cabinet IR
  A/B). A quick tap on a pedal's on/off pad toggles it; **holding the
  pad past ~450ms instead cycles that pedal's emulation type** (e.g.
  Distortion: DS-1 → RAT → Distortion+ → Metal → …) — one pad, two
  functions, like a footswitch with a hidden second click. Wah Position
  can be mapped either way: as a single knob (absolute sweep) or as a
  **pad pair** (Wah Position Up / Down, each a step-nudge) — a momentary
  pad press can't represent a continuous 0-127 value on its own, so the
  pair is the pad-only alternative. Every pedal slider/mode selector
  updates live on screen when driven by MIDI, not just when you touch it
  with the mouse. **Pedal mappings travel with the loaded Tone Profile**
  (which knob/pad drives which pedal is part of "the sound," the same
  way the pedal chain itself is); mappings for everything else (EQ/amp,
  transport, looper, tuner) stay global across every profile, since
  that's controller wiring, not part of the sound. Requires the optional
  `python-rtmidi` package — the app runs fine without it, just without
  MIDI control
- **LPD8 mk2 pad LED feedback** (experimental, off by default) — an
  "Enable LPD8 mk2 pad LED feedback" checkbox lights each pad to match
  what it's mapped to: a pedal's own color when on and dark when off
  (Panic shows red), and a brief white flash on press for one-shot pads
  like Tap Tempo. This isn't official — Akai doesn't document a way to
  drive this specific model's pad LEDs from incoming MIDI — so it's
  built on a third-party-reverse-engineered SysEx sequence instead;
  verified directly against real hardware (not just that the bytes
  match the spec) before shipping. Every other controller's pads just
  don't light, the same as if the checkbox were off

**Shared audio streams**
When the amp is running, the Metronome and Tuner automatically share its
duplex stream instead of opening their own — this avoids device-conflict
errors that would occur if each tool tried to claim the same hardware
independently.

**Session persistence**
BPM, beats/subdivision, sound kit, A4 calibration, transposition, loop
length/layers/mode, IR slot labels and active slot, block size, guitar
tuning preset, selected pedal states, pedal chain order, keybind mappings,
collapsed-section state, theme colors, and window settings, and the loaded NAM/IR file
paths are all saved on close and restored on next launch.

## Setup

### Windows — packaged .exe

A standalone Windows build is available under `build/dist/` (or from
wherever it was distributed to you) — no Python install required. Just
run `SoloTone-v<version>.exe`. `SoloTone_User_Guide.docx` ships as a
plain sibling file next to it.

### From source (Windows / macOS / Linux)

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python solotone.py
```

### Linux note
`sounddevice` needs PortAudio, and Tk ships separately from Python on most
distributions. If you get an audio or `tkinter` import error, install them
first:

```bash
sudo apt install libportaudio2 portaudio19-dev python3-tk
```

### macOS / Windows
The `sounddevice` wheel bundles PortAudio, so `pip install -r
requirements.txt` is usually all you need. macOS will prompt for
microphone permission the first time you start the tuner — allow it in
System Settings → Privacy & Security → Microphone.

### Audio drivers: no ASIO
SoloTone does not support ASIO. The PortAudio build bundled with
`sounddevice` has no ASIO host API, so ASIO drivers (including ASIO4ALL or an
interface's own ASIO driver) won't appear and aren't needed. On Windows pick
your interface under **WASAPI** (the default) or **WDM-KS** in the Host API
dropdown on the I/O & Levels tab.

## Recording into a DAW

SoloTone doesn't have a plugin (VST/AU) version — it's a standalone app
with its own audio stream — but its output can still be routed into a
DAW for recording:

1. Install a virtual audio mixer, e.g. [VoiceMeeter](https://vb-audio.com/Voicemeeter/)
   (free) or a simpler [VB-CABLE](https://vb-audio.com/Cable/) loopback.
2. Set SoloTone's Output device (I/O & Levels tab) to the virtual
   input the mixer/cable exposes.
3. In your DAW, record from the matching virtual output device.

Plain VB-CABLE alone removes your live monitoring, since its virtual
output makes no sound — VoiceMeeter avoids that by fanning one output to
both your real speakers/headphones (for low-latency monitoring straight
from SoloTone) and the virtual cable feeding the DAW at the same time.

## Notes on accuracy

- The tuner uses time-domain autocorrelation, which works well for
  monophonic input (a single guitar/bass string, a sung or played note).
  Chords or heavy background noise will confuse it, same as with hardware
  clip-on tuners.
- The transposition offsets (Bb=+2, Eb=+9, F=+7 semitones) follow the
  convention used by most commercial tuners for transposing instruments;
  double-check against your instrument's part if it matters for ensemble
  tuning.
- Everything runs locally — no network access, no telemetry.

## Platform support & testing

Windows, macOS, and Linux are all supported. Every push runs an automated
smoke-test matrix on GitHub Actions (`.github/workflows/smoke-test.yml`)
across `ubuntu-latest`, `macos-latest`, and `windows-latest`: dependency
install, the pedal-chain DSP regression suite, an isolated MIDI port-listing
probe, and a full GUI build-and-close of the real app. Run the same checks
locally:

```bash
python tests/test_pedals_dsp.py      # DSP regression tests, no display needed
python tests/test_midi_ports_macos.py
python tests/test_app_smoke.py       # builds the real window; needs a display
```

That proves it launches and processes audio sanely on each OS; it can't judge
tone or feel, which still needs a person with a guitar. Current build:
**1.0.1-rc1** (release candidate 1).

Layout: `solotone.py` (app/GUI), `pedals.py` (pedal-effects DSP),
`nam_engine.py` (Neural Amp Modeler inference), `version.py` (version +
changelog), `tests/`, `website/`, `build/` (Windows .exe packaging).

## Support

SoloTone is free to use. If it's useful to you, you can tip the developer on [Ko-fi](https://ko-fi.com/rexjaguarius).

## License

SoloTone is source-available under the [PolyForm Shield License
1.0.0](LICENSE): you can use it, including commercially (recording, gigging),
and modify it, but you can't use it to build a product that competes with
SoloTone. It is not an OSI-approved open-source license. Third-party
attributions and license texts are in [NOTICES](NOTICES). Outside code
contributions aren't being accepted at this time; bug reports and ideas are
welcome via the contact page or GitHub issues.

## Full documentation

See [SoloTone_User_Guide.docx](SoloTone_User_Guide.docx) for the complete
tab-by-tab user guide, and [version.py](version.py) for the build changelog.
