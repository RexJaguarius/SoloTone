"""AI tone building for SoloTone: describe a sound in words, get pedal settings and up to four scenes back.

Nothing in here touches the GUI or the audio engine. It does four things:

  * spec_from_widgets()  describes every pedal setting the app has (type, range, choices), read from the live UI so it can
                         never drift from what the sliders really accept
  * build_prompt()       the text a person can paste into ANY chat AI (the "no key" route)
  * ask_claude()         the same request sent to the Anthropic API with the user's own key (tool use, so the answer is
                         structured); only the request text and the current pedal settings are sent, never audio
  * parse_reply()/clean()  turn the answer (from either route) into safe values: every number is clamped to its range,
                         every choice must be a known one, unknown fields are dropped and reported. Raw model text never
                         reaches the rig.
"""
import json
import re
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-5-5"   # the smallest/cheapest; the player can type a larger one
MAX_SCENES = 4
SCENE_NAME_LEN = 15

# The pedals, as a player would name them, with what each choice does. Keys are the SoloTone pedal codes.
PEDALS = {
    "comp":    "Compressor: evens out dynamics, adds sustain (threshold dB, ratio, attack/release ms, makeup dB, warmth)",
    "wah":     "Wah: manual (position), auto sweep or envelope follower",
    "pitch":   "Pitch shifter: two voices in semitones (+-12) and cents, mix",
    "fuzz":    "Fuzz: Fuzz Face (woolly, guitar-volume sensitive), Tone Bender (gated, spitty), Big Muff (sustaining wall)",
    "dist":    "Distortion: DS-1 (bright), RAT (aggressive, versatile), Dist+ (vintage), Metal (tight, high gain)",
    "od":      "Overdrive/boost: Tubescreamer (mid hump, tightens), Klon (transparent, clean-ish push), Blues Driver, Boost",
    "phaser":  "Phaser: rate Hz, depth, feedback",
    "chorus":  "Chorus / Flanger / Vibrato / Doubler (mode), rate Hz, depth, mix",
    "tremolo": "Tremolo: rate Hz, depth, shape",
    "rotary":  "Rotary speaker (Leslie): slow/fast/brake, speeds in Hz, depth, ramp, mix",
    "delay":   "Delay: tape or digital, time (seconds) or note value synced to the metronome, feedback, mix",
    "reverb":  "Reverb: room, plate, spring, cloud or shimmer; mix, size, decay, damping (shimmer amount for shimmer)",
}
PEDAL_PREFIX = {"comp": "comp_", "wah": "wah_", "pitch": "pitch_", "fuzz": "fuzz_", "dist": "dist_", "od": "od_",
                "phaser": "phaser_", "chorus": "chorus_", "tremolo": "tremolo_", "rotary": "rotary_", "delay": "delay_",
                "reverb": "reverb_"}

# Settings that have no slider in the UI, or are not a slider, so the widgets cannot describe them.
EXTRA_SPECS = {
    "wah_mode":      dict(type="enum", options=["manual", "auto", "envelope"]),
    "wah_range_lo":  dict(type="float", lo=100.0, hi=2500.0),
    "wah_range_hi":  dict(type="float", lo=500.0, hi=3500.0),
    "muff_tone":     dict(type="enum", options=["full", "flat", "scoop"]),
    "delay_bpm_link": dict(type="bool"),
    "delay_note":    dict(type="enum", options=["dotted8", "quarter", "half"]),
    "delay_time":    dict(type="float", lo=0.01, hi=2.0, note="seconds"),
}
# Never set by the AI: chain-level, not part of a sound.
EXCLUDED = {"mute", "order"}

GAIN_SPECS = {
    "input_gain_db":  dict(lo=-12.0, hi=24.0, note="guitar level into the amp capture; more = more amp drive"),
    "output_gain_db": dict(lo=-24.0, hi=42.0, note="final volume"),
}
EQ_BAND_HZ = [100, 400, 1000, 2500, 6000]
EQ_RANGE_DB = (-15.0, 15.0)


def spec_from_widgets(widgets, factory, fields):
    """{field: spec} for every field in `fields` the AI may set. `widgets` is the app's _pedal_widgets (sliders carry lo/hi,
    radio rows carry their options), `factory` the factory values, `fields` PEDAL_PROFILE_FIELDS."""
    spec = {}
    for f in fields:
        if f in EXCLUDED:
            continue
        fv = factory.get(f)
        w = widgets.get(f) or {}
        if f.endswith("_on"):
            s = dict(type="bool")
        elif f in EXTRA_SPECS:
            s = dict(EXTRA_SPECS[f])
        elif w.get("kind") == "slider" and w.get("lo") is not None:
            s = dict(type="int" if isinstance(fv, int) and not isinstance(fv, bool) else "float",
                     lo=w["lo"], hi=w["hi"])
        elif w.get("kind") == "radio" and w.get("options"):
            s = dict(type="enum", options=[o for o, _t in w["options"]])
        else:
            continue                                           # nothing known about its range: leave it to the player
        s["default"] = fv
        if w.get("label"):
            s["label"] = w["label"]
        spec[f] = s
    return spec


def pedal_of(field):
    if field == "muff_tone":
        return "fuzz"
    for code, pre in PEDAL_PREFIX.items():
        if field.startswith(pre):
            return code
    return None


def _num(v):
    return ("%g" % v) if isinstance(v, float) else str(v)


def spec_text(spec):
    """The settings, grouped by pedal, one line each: name, type, range or choices, factory value."""
    out = []
    for code, blurb in PEDALS.items():
        rows = []
        for f, s in spec.items():
            if pedal_of(f) != code:
                continue
            if s["type"] == "bool":
                d = "on/off" if f.endswith("_on") else "true/false"
            elif s["type"] == "enum":
                d = "one of " + ", ".join(s["options"])
            else:
                d = "%s..%s%s" % (_num(s["lo"]), _num(s["hi"]), (" " + s["note"]) if s.get("note") else "")
            rows.append("    %s  [%s, factory %s]" % (f, d, s.get("default")))
        if rows:
            out.append("  " + blurb + "\n" + "\n".join(rows))
    return "\n".join(out)


def tool_schema(spec):
    """JSON schema of the answer (also shown in the copy-paste prompt)."""
    props = {}
    for f, s in spec.items():
        if s["type"] == "bool":
            props[f] = {"type": "boolean"}
        elif s["type"] == "enum":
            props[f] = {"type": "string", "enum": list(s["options"])}
        else:
            props[f] = {"type": "integer" if s["type"] == "int" else "number", "minimum": s["lo"], "maximum": s["hi"]}
    scene = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Scene name, at most %d characters (Verse, Solo, ...)" % SCENE_NAME_LEN},
            "pedals": {"type": "object", "properties": props, "additionalProperties": False,
                       "description": "Only the settings you want; every pedal you do not turn on stays off."},
            "input_gain_db": {"type": "number", "minimum": GAIN_SPECS["input_gain_db"]["lo"],
                              "maximum": GAIN_SPECS["input_gain_db"]["hi"]},
            "output_gain_db": {"type": "number", "minimum": GAIN_SPECS["output_gain_db"]["lo"],
                               "maximum": GAIN_SPECS["output_gain_db"]["hi"]},
            "eq_bands_db": {"type": "array", "items": {"type": "number", "minimum": EQ_RANGE_DB[0], "maximum": EQ_RANGE_DB[1]},
                            "minItems": 5, "maxItems": 5,
                            "description": "Gain in dB at %s Hz" % ", ".join(map(str, EQ_BAND_HZ))},
        },
        "required": ["name", "pedals"],
    }
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "One or two sentences: what you built and why."},
            "scenes": {"type": "array", "items": scene, "minItems": 1, "maxItems": MAX_SCENES},
        },
        "required": ["summary", "scenes"],
    }


SYSTEM = (
    "You are a guitar tone designer inside SoloTone, a software amp and pedalboard. The player has a fixed amp capture "
    "(a Neural Amp Modeler model) and cabinet already loaded; you build the PEDAL settings, the EQ and the gain staging "
    "around it. Be musical and conservative: a usable tone beats an extreme one. Turn on only the pedals the sound needs "
    "(set <pedal>_on true); every pedal you do not mention stays off. Use the pedal chain's fixed order "
    "(comp, wah, pitch, fuzz, dist, od, phaser, chorus, tremolo, rotary, delay, reverb); you cannot change it. "
    "A drive pedal pushes the amp capture: raise input_gain_db rather than stacking drives when you want more saturation. "
    "When asked for several scenes, make them differ by what a foot switch would change in performance (a boost for the solo, "
    "a different delay, a wash of reverb for the outro). Keep the same chain and the same character across scenes. "
    "Return your answer only by calling the tool, or, if you cannot call it, as one JSON object and nothing else."
)


def build_prompt(request, spec, context="", scenes=1):
    """The full text for a chat AI: the task, the settings and their ranges, the current rig, the answer format."""
    n = "one scene" if scenes <= 1 else "%d scenes (A to %s)" % (scenes, "ABCD"[scenes - 1])
    example = {
        "summary": "A warm, compressed lead with a touch of tape delay.",
        "scenes": [{"name": "Lead", "pedals": {"od_on": True, "od_mode": "klon", "od_drive": 3.0, "delay_on": True,
                                                 "delay_mode": "tape", "delay_time": 0.38, "delay_feedback": 0.35,
                                                 "delay_mix": 0.25}, "input_gain_db": 3.0, "eq_bands_db": [0, 0, 1, 0, -1]}],
    }
    return "\n".join([
        SYSTEM, "",
        "TASK: build %s for this sound: %s" % (n, request.strip()), "",
        "THE RIG: " + (context.strip() or "an amp capture and cab that are already loaded"), "",
        "SETTINGS YOU MAY USE (name [range or choices, factory value]); anything else is ignored:", spec_text(spec), "",
        "Also: input_gain_db %g..%g, output_gain_db %g..%g, eq_bands_db = 5 numbers in dB (%s Hz), each %g..%g. "
        "Scene names are at most %d characters." % (
            GAIN_SPECS["input_gain_db"]["lo"], GAIN_SPECS["input_gain_db"]["hi"], GAIN_SPECS["output_gain_db"]["lo"],
            GAIN_SPECS["output_gain_db"]["hi"], ", ".join(map(str, EQ_BAND_HZ)), EQ_RANGE_DB[0], EQ_RANGE_DB[1], SCENE_NAME_LEN),
        "", "ANSWER with exactly one JSON object and no other text, shaped like this example:",
        json.dumps(example, indent=1),
    ])


def user_message(request, context="", scenes=1):
    n = "one scene" if scenes <= 1 else "%d scenes (A to %s)" % (scenes, "ABCD"[scenes - 1])
    return "Build %s for this sound: %s\n\nThe rig: %s" % (
        n, request.strip(), context.strip() or "an amp capture and cab that are already loaded")


# ───────────────────────── answers → safe values ─────────────────────────
def parse_reply(text):
    """Find the JSON object in a chat answer (it may be wrapped in a code fence or in sentences)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("There is no answer to read. Paste the AI's whole reply.")
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError("The reply has no JSON object in it.")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except ValueError as e:
                    raise ValueError("The reply's JSON is damaged (%s). Ask the AI to send it again." % e)
    raise ValueError("The reply's JSON is cut off. Ask the AI to send it again, complete.")


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "on", "yes", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "off", "no", "0"):
        return False
    raise ValueError


def clean(result, spec):
    """Validate an answer. Returns (answer, notes): answer = {'summary', 'scenes': [{'name', 'pedals', 'in_gain_db',
    'out_gain_db', 'eq_bands_db'}]} with every value inside its range; notes = what was dropped or clamped, for the player."""
    notes = []
    if not isinstance(result, dict) or not isinstance(result.get("scenes"), list) or not result["scenes"]:
        raise ValueError("The answer has no scenes in it.")
    scenes = []
    for i, sc in enumerate(result["scenes"][:MAX_SCENES]):
        if not isinstance(sc, dict):
            continue
        letter = "ABCD"[i]
        pedals = {}
        raw = sc.get("pedals") if isinstance(sc.get("pedals"), dict) else {}
        for f, v in raw.items():
            s = spec.get(f)
            if s is None:
                notes.append("scene %s: ignored unknown setting %s" % (letter, f))
                continue
            try:
                if s["type"] == "bool":
                    pedals[f] = _to_bool(v)
                elif s["type"] == "enum":
                    val = next(o for o in s["options"] if str(v).strip().lower() == o.lower())
                    pedals[f] = val
                else:
                    x = float(v)
                    if x != x or x in (float("inf"), float("-inf")):
                        raise ValueError
                    c = min(max(x, s["lo"]), s["hi"])
                    if c != x:
                        notes.append("scene %s: %s %s is outside %s..%s, set to %s" % (letter, f, _num(x), _num(s["lo"]),
                                                                                    _num(s["hi"]), _num(c)))
                    pedals[f] = int(round(c)) if s["type"] == "int" else float(c)
            except (ValueError, TypeError, StopIteration):
                notes.append("scene %s: ignored %s=%r (not a valid value)" % (letter, f, v))
        # a pedal whose settings were given but never switched on: it is meant to be on
        for code in PEDALS:
            on_key = code + "_on"
            if on_key in spec and on_key not in pedals and any(pedal_of(f) == code for f in pedals):
                pedals[on_key] = True
        one = dict(name=str(sc.get("name") or "")[:SCENE_NAME_LEN].strip(), pedals=pedals,
                   in_gain_db=None, out_gain_db=None, eq_bands_db=None)
        for key, gk in (("in_gain_db", "input_gain_db"), ("out_gain_db", "output_gain_db")):
            if sc.get(gk) is not None:
                try:
                    x = float(sc[gk])
                    lo, hi = GAIN_SPECS[gk]["lo"], GAIN_SPECS[gk]["hi"]
                    c = min(max(x, lo), hi)
                    if c != x:
                        notes.append("scene %s: %s %s is outside %g..%g, set to %g" % (letter, gk, _num(x), lo, hi, c))
                    one[key] = c
                except (ValueError, TypeError):
                    notes.append("scene %s: ignored %s" % (letter, gk))
        bands = sc.get("eq_bands_db")
        if isinstance(bands, list) and len(bands) == len(EQ_BAND_HZ):
            try:
                one["eq_bands_db"] = [min(max(float(b), EQ_RANGE_DB[0]), EQ_RANGE_DB[1]) for b in bands]
            except (ValueError, TypeError):
                notes.append("scene %s: ignored the EQ" % letter)
        elif bands is not None:
            notes.append("scene %s: ignored the EQ (it needs %d numbers)" % (letter, len(EQ_BAND_HZ)))
        scenes.append(one)
    if not scenes:
        raise ValueError("The answer has no usable scenes in it.")
    summary = result.get("summary") if isinstance(result.get("summary"), str) else ""
    return dict(summary=summary.strip()[:600], scenes=scenes), notes


# ───────────────────────── the Anthropic API (the player's own key) ─────────────────────────
class AIError(Exception):
    pass


def _http_post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ask_claude(api_key, request, spec, context="", scenes=1, model=DEFAULT_MODEL, timeout=120, post=None):
    """Ask Claude for a tone. Returns the answer dict (not yet cleaned). `post(url, headers, body, timeout)` can be replaced
    in tests. Only the request text, the rig description and the setting names/ranges are sent."""
    if not (api_key or "").strip():
        raise AIError("Enter your Anthropic API key first.")
    body = json.dumps({
        "model": model or DEFAULT_MODEL,
        "max_tokens": 4000,
        "system": SYSTEM,
        "tools": [{"name": "set_tone",
                   "description": "Set the pedal settings (and scenes) for the requested sound.",
                   "input_schema": tool_schema(spec)}],
        "tool_choice": {"type": "tool", "name": "set_tone"},
        "messages": [{"role": "user", "content": user_message(request, context, scenes) + "\n\nSettings and ranges:\n"
                      + spec_text(spec)}],
    }).encode("utf-8")
    headers = {"x-api-key": api_key.strip(), "anthropic-version": API_VERSION, "content-type": "application/json"}
    try:
        raw = (post or _http_post)(API_URL, headers, body, timeout)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8", "replace")).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        why = {401: "The API key was not accepted.", 403: "This key is not allowed to use that model.",
               404: "The model %r was not found." % model, 429: "Too many requests or no credit left on the key.",
               529: "Claude is overloaded; try again in a minute."}.get(e.code, "The API answered %d." % e.code)
        raise AIError(why + (" (" + detail + ")" if detail else ""))
    except (urllib.error.URLError, OSError) as e:
        raise AIError("Could not reach the Anthropic API: %s" % getattr(e, "reason", e))
    try:
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except ValueError:
        raise AIError("The API's answer could not be read.")
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
            return block["input"]
    for block in data.get("content", []):
        if block.get("type") == "text":
            try:
                return parse_reply(block.get("text", ""))
            except ValueError:
                pass
    raise AIError("Claude did not return any settings. Try again, or describe the sound a little differently.")
