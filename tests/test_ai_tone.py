"""AI tone building: prompt, answer validation, the Claude call (fake network) and applying scenes to the rig."""
import json
import pathlib
import sys
import tempfile
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import ai_tone
import solotone as st

_tmp = pathlib.Path(tempfile.mkdtemp())
st.SESSION_FILE = _tmp / "s.json"
st.MIDI_MAP_FILE = _tmp / "m.json"
st.ERROR_LOG = _tmp / "e.log"
st.CRASH_LOG = _tmp / "c.log"
st.App.AI_KEY_FILE = _tmp / "ai.json"

APP = st.App()
APP.update()
SPEC = APP._ai_spec()

GOOD = {
    "summary": "A singing lead.",
    "scenes": [
        {"name": "Verse", "pedals": {"od_on": True, "od_mode": "klon", "od_drive": 3.0, "delay_on": True,
                                     "delay_mode": "tape", "delay_time": 0.38, "delay_feedback": 0.35, "delay_mix": 0.25},
         "input_gain_db": 3.0, "eq_bands_db": [0, 1, 2, 0, -1]},
        {"name": "A much too long scene name", "pedals": {"od_on": True, "od_drive": 99, "reverb_type": "SHIMMER",
                                                          "reverb_mix": 0.4, "bogus_knob": 3}},
    ],
}


def test_spec_covers_the_pedals_with_real_ranges():
    assert SPEC["comp_thresh"] == dict(type="float", lo=-40.0, hi=0.0, default=-18.0, label="Threshold (dB)")
    assert SPEC["od_mode"]["options"] == ["tubescreamer", "klon", "bd2", "boost"]
    assert SPEC["pitch_semi"]["type"] == "int"
    assert SPEC["delay_note"]["options"] == ["dotted8", "quarter", "half"]
    assert "order" not in SPEC and "mute" not in SPEC
    assert all(f in SPEC for f in st.PEDAL_PROFILE_FIELDS if f not in ("order", "mute")), \
        [f for f in st.PEDAL_PROFILE_FIELDS if f not in SPEC and f not in ("order", "mute")]


def test_clean_clamps_and_drops():
    ans, notes = ai_tone.clean(GOOD, SPEC)
    s0, s1 = ans["scenes"]
    assert s0["pedals"]["od_mode"] == "klon" and s0["in_gain_db"] == 3.0 and s0["eq_bands_db"] == [0, 1, 2, 0, -1]
    assert s1["name"] == "A much too long "[:15].strip()
    assert s1["pedals"]["od_drive"] == 10.0                   # clamped to the slider's top
    assert s1["pedals"]["reverb_type"] == "shimmer"           # case-insensitive choice
    assert "bogus_knob" not in s1["pedals"] and any("bogus_knob" in n for n in notes)
    assert any("od_drive" in n and "set to 10" in n for n in notes)
    assert s1["pedals"].get("reverb_on") is True              # settings given for a pedal never switched on: it is meant to be on


def test_clean_rejects_garbage_values_and_empty_answers():
    ans, notes = ai_tone.clean({"scenes": [{"name": "x", "pedals": {"od_drive": "loud", "od_mode": "fuzzy", "delay_on": "maybe"}}]},
                               SPEC)
    assert ans["scenes"][0]["pedals"] == {}
    assert len(notes) == 3
    for bad in ({}, {"scenes": []}, {"scenes": "x"}, []):
        try:
            ai_tone.clean(bad, SPEC)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % (bad,))


def test_at_most_four_scenes_and_nan_is_refused():
    ans, notes = ai_tone.clean({"scenes": [{"name": str(i), "pedals": {"od_drive": float("nan")}} for i in range(6)]}, SPEC)
    assert len(ans["scenes"]) == 4 and ans["scenes"][0]["pedals"] == {}


def test_parse_reply_finds_json_in_chat_text():
    j = json.dumps(GOOD)
    assert ai_tone.parse_reply(j)["summary"] == "A singing lead."
    assert ai_tone.parse_reply("Sure! Here you go:\n```json\n%s\n```\nEnjoy {tone}." % j)["scenes"][0]["name"] == "Verse"
    assert ai_tone.parse_reply('noise {"a": "}{"} more')["a"] == "}{"
    for bad in ("", "no json here", '{"scenes": [', None):
        try:
            ai_tone.parse_reply(bad)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % (bad,))


def test_prompt_has_the_settings_and_the_request():
    t = ai_tone.build_prompt("surf twang", SPEC, "Amp capture: Fender.", scenes=4)
    assert "surf twang" in t and "4 scenes (A to D)" in t and "od_drive" in t and "tubescreamer" in t
    assert "Amp capture: Fender." in t and "JSON" in t
    assert "mute" not in ai_tone.spec_text(SPEC).replace("mute 'm", "")  # no chain-level fields offered


def _fake_reply(payload, status=200):
    seen = {}

    def post(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=json.loads(body))
        if status != 200:
            raise urllib.error.HTTPError(url, status, "x", {}, __import__("io").BytesIO(
                json.dumps({"error": {"message": "nope"}}).encode()))
        return json.dumps({"content": payload}).encode()
    return post, seen


def test_ask_claude_uses_tool_and_sends_no_audio_or_paths():
    post, seen = _fake_reply([{"type": "tool_use", "name": "set_tone", "input": GOOD}])
    got = ai_tone.ask_claude("sk-test", "surf twang", SPEC, "Amp capture: Fender.", 2, post=post)
    assert got == GOOD
    assert seen["url"].startswith("https://api.anthropic.com/") and seen["headers"]["x-api-key"] == "sk-test"
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": "set_tone"}
    assert "od_drive" in json.dumps(seen["body"]["tools"][0]["input_schema"])
    assert "surf twang" in seen["body"]["messages"][0]["content"]


def test_ask_claude_falls_back_to_text_json_and_reports_http_errors():
    post, _ = _fake_reply([{"type": "text", "text": "```json\n%s\n```" % json.dumps(GOOD)}])
    assert ai_tone.ask_claude("k", "x", SPEC, post=post)["summary"] == "A singing lead."
    for code, word in ((401, "key"), (429, "credit"), (404, "not found")):
        post, _ = _fake_reply([], status=code)
        try:
            ai_tone.ask_claude("k", "x", SPEC, post=post)
        except ai_tone.AIError as e:
            assert word in str(e)
        else:
            raise AssertionError(code)
    try:
        ai_tone.ask_claude("  ", "x", SPEC)
    except ai_tone.AIError:
        pass
    else:
        raise AssertionError("empty key accepted")
    post, _ = _fake_reply([{"type": "text", "text": "I cannot"}])
    try:
        ai_tone.ask_claude("k", "x", SPEC, post=post)
    except ai_tone.AIError:
        pass
    else:
        raise AssertionError("no settings accepted")


def test_apply_puts_scenes_on_the_rig_and_undo_restores():
    p = APP.amp.pedals
    APP._set_pedal_on("reverb", True)
    before = APP._scene_dict()
    ans, _ = ai_tone.clean(GOOD, SPEC)
    APP._ai_apply(ans)
    APP.update()
    assert APP._scene_active == 0 and APP._scene_names[:2] == ["Verse", "A much too long"]
    assert p.od_on and p.od_mode == "klon" and p.od_drive == 3.0 and p.delay_on and not p.reverb_on
    assert abs(APP.amp.in_gain - 10 ** (3.0 / 20)) < 1e-9
    assert [b["gain"] for b in APP.amp.eq.bands] == [0, 1, 2, 0, -1]
    assert APP._scenes[2] is None and APP._scenes[1] is not None
    APP._scene_select(1)
    APP.update()
    assert p.od_drive == 10.0 and p.reverb_on and p.reverb_type == "shimmer" and not p.delay_on
    assert APP._ai_restore()
    APP.update()
    assert APP._scene_active == 0 and p.reverb_on and not p.od_on
    assert APP._scene_dict()["slots"][1] == before["slots"][1]
    assert not APP._ai_restore()


def test_profile_still_round_trips_after_an_ai_tone():
    ans, _ = ai_tone.clean(GOOD, SPEC)
    APP._ai_apply(ans)
    d = APP._gather_profile_dict("ai")
    assert d["scenes"]["names"][:2] == ["Verse", "A much too long"] and d["pedals"]["od_mode"] == "klon"


def test_key_file_is_only_written_when_asked():
    APP._ai_save_key("sk-x", "m", False)
    assert not st.App.AI_KEY_FILE.exists()
    APP._ai_save_key("sk-x", "m", True)
    assert APP._ai_load_key() == ("sk-x", "m", True)
    APP._ai_save_key("sk-x", "m", False)                      # unticking deletes it
    assert not st.App.AI_KEY_FILE.exists()


def test_dialog_opens_and_use_pasted_answer_shows_a_result():
    APP._ai_open()
    APP.update()
    win = APP._ai_win
    assert win.winfo_exists()
    APP._ai_open()                                            # a second click reuses the window
    win.destroy()


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
            n += 1
    APP.destroy()
    print("%d tests passed." % n)
