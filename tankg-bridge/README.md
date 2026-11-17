# SoloTone Tank-G add-on (optional)

Sends a SoloTone profile to an **M-VAVE Tank-G running the third-party UltraRig firmware**, and can drive the pedal live from SoloTone.
It lives in this folder; SoloTone finds it next to `solotone.py` and adds a Tank-G section to Settings, a *Load to Tank-G* button to the
Library and a *Go live* switch to the top bar. Without the folder none of that exists.

> Not made, endorsed or supported by UltraRig or M-VAVE. Installing UltraRig replaces the pedal's stock firmware: read its instructions
> and keep the backup its installer makes. You use this at your own risk.

## What it does
- **Send to a preset slot**: the NAM capture, the cab IR and the whole pedal chain become a preset on the pedal (fitted to its CPU and memory).
  A profile with scenes A to D becomes one shared chain plus the pedal's own scenes.
- **Play on pedal** (nothing saved): loads the sound into the running pedal.
- **Go live**: SoloTone is the pedal's remote. Every pedal you switch, knob you turn, EQ or gain change and scene you pick is sent as you do it.
  Nothing is stored on the pedal. Serial port only, no USB audio.
- **Back up the pedal** (`backup_pedal.py`): reads every preset, capture and IR into a folder. **That folder holds data unique to your pedal: don't share it.**

## Needs
Python packages `pyserial` and `numpy`; a Tank-G with UltraRig (tested on alpha 11 and alpha 13); the UltraRig control page or app **closed**
(only one program can hold the pedal's USB serial port).

## Safety
It only sends commands from a short allow-list, never deletes a preset, capture or IR, never overwrites a capture or IR slot (it uses a free
one, or re-uses one holding the same data), and asks before replacing a preset. Writing to a preset slot is always an explicit action; the
live and play modes write nothing to the pedal's memory.

## Limits worth knowing
- The mappings are approximations: the same pedal name is not the same DSP. The dry-run output flags every parameter that is a guess.
- The pedal's CPU budget varies (and USB audio reserves some): heavy chains are simplified automatically and anything dropped is said.
- A scene cannot change a block's type, so scenes share one reverb type.
- The pedal's own footswitches keep their layout: on the factory layout they load presets, which replaces the live sound.

## Files
| file | purpose |
|---|---|
| `tankg_api.py` | what SoloTone calls: find the pedal, read it, send a sound, `LiveSession` |
| `bridge.py` | SoloTone profile to UltraRig blocks; also a dry-run tool: `python bridge.py PROFILE --port COM7` |
| `live.py` | the serial session and chain commands |
| `a2l1.py` | NAM to the pedal's capture format (port of PulseDeck's converter, MIT) |
| `backup_pedal.py` | read everything off the pedal into a folder |
| `ultrarig_probe.py` | read-only probe; also writes `fxt_catalogue.txt` for offline use |
| `test_*.py` | tests against a fake pedal (they need `fxt_catalogue.txt` from the probe and a local profile) |

The block catalogue (which blocks exist and their ranges) is read from your own pedal when you connect, so it always matches the installed firmware.

## Licenses
SoloTone and this add-on: PolyForm Shield 1.0.0 (see the root `LICENSE`). `a2l1.py` is derived from PulseDeck (c) 2026 Wilian de Lima Santos,
MIT: see `LICENSE-PulseDeck-MIT.txt`. UltraRig is a separate project under its own license and is not included.
