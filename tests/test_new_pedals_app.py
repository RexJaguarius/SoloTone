"""
Tests for the Smart noise gate and the pedals added after the first eight
(phaser, tremolo, vibrato/doubler, Blues Driver, plate/spring/cloud reverb):
the gate's behaviour, how they travel through Tone Profiles and pedal presets,
and how the Pedals tab and the I/O tab present them.

Run directly: python tests/test_new_pedals_app.py
"""

import json
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import solotone as st
from pedals import PedalChain

SR = st.SAMPLE_RATE


def _db(x):
    return 20 * np.log10(max(float(x), 1e-12))


# ── the Smart gate ────────────────────────────────────────────────

def _blocks(gate, x, block=256, **kw):
    out = np.zeros(len(x), dtype=np.float32)
    for pos in range(0, len(x) - block + 1, block):
        out[pos:pos + block] = gate.process(x[pos:pos + block], **kw)
    return out


def test_smart_gate_passes_signal_and_closes_on_noise():
    thr = 0.01
    t = np.arange(SR) / SR
    loud = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    quiet = (0.002 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    g = st.SmartGate()
    out = _blocks(g, loud, thr=thr)
    assert abs(_db(np.sqrt(np.mean(out[SR // 2:] ** 2))) - _db(np.sqrt(np.mean(loud[SR // 2:] ** 2)))) < 0.5, \
        'a signal well over the threshold must pass at full level'
    g = st.SmartGate()
    out = _blocks(g, np.concatenate([loud, quiet, quiet]), thr=thr, range_db=40.0, release_ms=50.0, hush=0.0)
    tail = out[2 * SR:]
    drop = _db(np.sqrt(np.mean(quiet ** 2))) - _db(np.sqrt(np.mean(tail ** 2)))
    assert 33 < drop < 47, f'closed gate should pull noise down by about its depth (40 dB), got {drop:.1f}'


def test_smart_gate_depth_sets_how_far_it_closes():
    thr = 0.01
    quiet = (0.002 * np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR)).astype(np.float32)
    drops = []
    for depth in (12.0, 30.0, 60.0):
        out = _blocks(st.SmartGate(), quiet, thr=thr, range_db=depth, release_ms=30.0, hush=0.0)
        drops.append(_db(np.sqrt(np.mean(quiet[SR:] ** 2))) - _db(np.sqrt(np.mean(out[SR:] ** 2))))
    assert drops[0] < drops[1] < drops[2]
    assert abs(drops[0] - 12) < 3 and abs(drops[1] - 30) < 3


def test_smart_gate_hysteresis_and_hold_keep_a_decaying_note_open():
    thr = 0.01
    sig_hi = 0.1 * np.ones(SR // 4, dtype=np.float32)          # opens it
    sig_mid = 0.007 * np.ones(SR // 2, dtype=np.float32)       # between threshold/2 and threshold
    # a standard-style gate would shut on the middle part; this one stays open until the level is 6 dB under thr
    out = _blocks(st.SmartGate(), np.concatenate([sig_hi, sig_mid]), thr=thr, range_db=60.0, release_ms=30.0, hush=0.0)
    mid = out[len(sig_hi) + SR // 8:]
    assert np.mean(mid) > 0.0065, 'level between the close and open points should stay open'
    # hold: after the signal stops the gate stays open for roughly hold_ms before it starts to fall
    sig = np.concatenate([sig_hi, np.zeros(SR // 2, dtype=np.float32)])
    held = _blocks(st.SmartGate(), sig, thr=thr, hold_ms=200.0, release_ms=20.0, hush=0.0)
    quick = _blocks(st.SmartGate(), sig, thr=thr, hold_ms=0.0, release_ms=20.0, hush=0.0)
    # the input is zero after sample len(sig_hi): compare how much of a small tail is let through
    tail_in = np.concatenate([sig_hi, 0.004 * np.ones(SR // 2, dtype=np.float32)])
    held = _blocks(st.SmartGate(), tail_in, thr=thr, hold_ms=250.0, release_ms=20.0, hush=0.0)
    quick = _blocks(st.SmartGate(), tail_in, thr=thr, hold_ms=0.0, release_ms=20.0, hush=0.0)
    w = slice(len(sig_hi) + 2000, len(sig_hi) + 6000)           # ~50-150 ms after the note
    assert np.mean(np.abs(held[w])) > np.mean(np.abs(quick[w])) * 3


def test_smart_gate_hush_fades_instead_of_chopping():
    thr = 0.01
    n = SR * 2
    ramp_down = np.linspace(0.05, 0.0004, n, dtype=np.float32)     # a note dying away through the threshold
    sig = np.sin(2 * np.pi * 200 * np.arange(n) / SR).astype(np.float32) * ramp_down
    kw = dict(thr=thr, range_db=60.0, release_ms=20.0, hold_ms=0.0)
    soft = _blocks(st.SmartGate(), sig, hush=1.0, **kw)
    hard = _blocks(st.SmartGate(), sig, hush=0.0, **kw)
    # where the level has just fallen under the close point, a plain gate has already dropped to its floor, but
    # with hush the sound is still following the level down: more of it is let through...
    down = np.abs(sig) > 0
    env = np.convolve(np.abs(sig), np.ones(441) / 441, 'same')
    just_under = (env < thr * 0.5) & (env > thr * 0.25)
    assert np.sum(np.abs(soft[just_under])) > 1.5 * np.sum(np.abs(hard[just_under]))
    # ...and well below it both are shut
    deep = env < thr * 0.05
    assert np.sum(np.abs(soft[deep])) < 0.05 * np.sum(np.abs(sig[deep]))


def test_smart_gate_does_not_depend_on_the_block_size():
    rng = np.random.default_rng(2)
    x = (rng.normal(0, 0.01, SR * 2) * (1 + np.sin(np.arange(SR * 2) / 4000.0))).astype(np.float32)
    a = _blocks(st.SmartGate(), x, block=256, thr=0.012)
    b = _blocks(st.SmartGate(), x, block=512, thr=0.012)
    n = min(len(a), len(b))
    # per-64-sample level steps: identical whatever the callback's block size
    assert float(np.max(np.abs(a[:n] - b[:n]))) < 1e-6


def test_offline_chain_uses_the_profile_gate_mode_and_old_profiles_get_the_standard_gate():
    sig = np.zeros(SR, dtype=np.float32)
    t = np.arange(SR // 2) / SR
    sig[:SR // 2] = 0.2 * np.sin(2 * np.pi * 220 * t)
    sig[SR // 2:] = 0.004 * np.sin(2 * np.pi * 220 * np.arange(SR // 2) / SR)

    def run(profile):
        ch = st.OfflineChain(profile)
        out = np.zeros(len(sig), dtype=np.float32)
        for pos in range(0, len(sig) - 256 + 1, 256):
            out[pos:pos + 256] = ch.process(sig[pos:pos + 256])
        return ch, out
    base = {'pedals': {}, 'eq_on': False, 'gate_threshold': 0.01}
    ch, std = run(base)
    assert ch.gate_mode == 'standard'
    ch2, smart = run(dict(base, gate_mode='smart', gate_range=30.0, gate_release=40.0, gate_hold=0.0, gate_hush=0.0))
    assert ch2.gate_mode == 'smart'
    tail = slice(int(SR * 0.8), SR)
    for o in (std, smart):
        assert np.sqrt(np.mean(o[tail] ** 2)) < 0.004 * 0.5      # both gates close on the noise
    assert not np.allclose(std, smart), 'the two modes should not be identical'


# ── fields, profiles and presets ──────────────────────────────────

def test_every_pedal_setting_is_in_the_profile_and_preset_fields():
    """The standing rule: a new pedal parameter that is not in PEDAL_PROFILE_FIELDS silently drops out of saved profiles."""
    p = PedalChain(SR)
    live_only = {'wah_freq'}                                   # recomputed every block, not a setting
    for name in vars(p):
        if name.startswith('_') or name in ('sr', 'mute', 'order') or name in live_only:
            continue
        assert name in st.PEDAL_PROFILE_FIELDS, f'{name} is a pedal setting missing from PEDAL_PROFILE_FIELDS'
    for code in ('phaser', 'tremolo', 'reverb', 'chorus', 'od'):
        assert code in st.PEDAL_PRESET_FIELDS
    assert st.PEDAL_PRESET_FIELDS['phaser'] == ['phaser_on', 'phaser_rate', 'phaser_depth', 'phaser_feedback']
    assert {'reverb_type', 'reverb_decay', 'reverb_damp'} <= set(st.PEDAL_PRESET_FIELDS['reverb'])
    # one section title / board name / icon per pedal in the default order
    for code in (c for c in st.DEFAULT_PEDAL_ORDER if c != st.AMP_MARKER):
        assert code in st.PEDAL_SECTION_TITLES and code in st.PEDAL_ICON_SPECS and code in st.PEDAL_BOARD_NAMES
        assert st.PEDAL_BOARD_SUMMARY(code, p) != ''
        assert f'pedal_{code}_on' in st.MIDI_TARGETS_BY_ID, f'no MIDI on/off target for {code}'
    assert {'phaser_rate', 'phaser_depth', 'tremolo_rate', 'tremolo_depth', 'reverb_decay'} <= set(st.MIDI_TARGETS_BY_ID)
    cycle = st.PEDAL_MODE_CYCLE
    assert 'bd2' in cycle['pedal_od_on'][1] and {'vibrato', 'doubler'} <= set(cycle['pedal_chorus_on'][1])
    assert cycle['pedal_reverb_on'] == ('reverb_type', ['room', 'plate', 'spring', 'cloud', 'shimmer'])


def test_every_shipped_profile_still_loads_into_the_offline_chain():
    root = pathlib.Path(__file__).resolve().parent.parent / 'profiles'
    files = sorted(root.rglob('*.stprofile'))
    assert files, 'no profiles found'
    for f in files:
        data = json.loads(f.read_text(encoding='utf-8'))
        ch = st.OfflineChain(dict(data, nam_path=None, ir_paths=[]))
        assert ch.gate_mode == ('smart' if data.get('gate_mode') == 'smart' else 'standard'), f.name   # no key = the standard gate
        order = ch.pedals.order
        assert 'phaser' in order and 'tremolo' in order and order.count(st.AMP_MARKER) == 1, f.name
        assert ch.pedals.phaser_on is False and ch.pedals.tremolo_on is False
        # a profile made before reverb types existed has none stored and gets the original Room reverb; a newer
        # one (e.g. The Wall set) gets whatever it saved
        assert ch.pedals.reverb_type == data.get('pedals', {}).get('reverb_type', 'room'), f.name


def test_the_pedals_tab_and_profiles_in_the_app():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        st.SESSION_FILE = tmp_path / 'test_session.json'
        st.MIDI_MAP_FILE = tmp_path / 'test_midi.json'
        app = st.App()
        try:
            app.update()
            from tkinter import ttk
            nb = next(w for w in app.winfo_children() if isinstance(w, ttk.Notebook))
            nb.select(0)
            app.update()
            p = app.amp.pedals

            # the board has a tile for every pedal plus the amp; switched-on pedals (and the amp) sit on the chain in
            # signal order, row by row, and switched-off ones wait in the tray underneath
            order = p.order
            cells = app._pedal_icon_cells
            assert set(cells) == set(order) | {'input', 'output'} and len(order) == 13, sorted(cells)
            assert set(app._pedal_sum_lbls) == {c for c in order if c != st.AMP_MARKER}

            def grid_pos(c):
                return (int(cells[c].grid_info()['row']), int(cells[c].grid_info()['column']))

            def chain_now():
                return [c for c in p.order if c == st.AMP_MARKER or app._pedal_is_on(c)]
            for c in ('comp', 'wah', 'fuzz', 'od', 'chorus', 'delay'):
                app._set_pedal_on(c, True)
            app.update(); app.update()
            chain = chain_now()
            pos = [grid_pos(c) for c in chain]
            assert pos == sorted(pos), 'chain tiles must be placed in chain order, row by row'
            tray = [c for c in p.order if c not in chain]
            assert tray and all(grid_pos(c)[0] > max(r for r, _ in pos) for c in tray), 'the tray sits below the chain'
            cols_now = max(col for _, col in pos) + 1
            app.geometry('1700x900'); app.update(); app.update()
            wide = [grid_pos(c) for c in chain]
            assert wide == sorted(wide)
            rows = lambda ps: max(r for r, _ in ps) - min(r for r, _ in ps)
            assert rows(wide) == 0 or max(col for _, col in wide) + 1 > cols_now, \
                'a wider window must let more pedals fit on a row'
            app.geometry('900x900'); app.update(); app.update()

            # the LED sits in its own strip above the icon, so it never covers the artwork
            for c, (led_c, _tag) in app._pedal_side_lbls.items():
                icon_c = app._pedal_tile_parts[c][0]
                assert led_c.winfo_y() + led_c.winfo_height() <= icon_c.winfo_y(), f'{c}: the LED strip overlaps the icon'

            # no pedal is open until one is clicked; then the inspector docks at the bottom with the pedal's own tab
            # the Amp page (NAM, cabinet IRs, EQ) is what the inspector shows at first
            assert app._insp_open == ['amp'] and app._static_pages['amp'].winfo_manager() == 'pack'
            for code in ('input', 'output', 'amp'):                                 # the three fixed tiles open their pages
                app._fixed_tile_click(code) if app._insp_sel != code else None
                app.update()
                assert app._insp_sel == code and app._static_pages[code].winfo_manager() == 'pack', code
                assert app._pedal_icon_cells[code].cget('highlightbackground') == st.ACCENT
            for attr in ('in_gain_sl', 'out_gain_sl', 'master_sl', 'gate_sl', 'hum_var', 'nam_lbl', 'ir_slot_rows',
                         'eq_on_var', 'amp_in_box', 'amp_out_box', 'bs_var', 'hostapi_box'):
                assert hasattr(app, attr), attr                                     # every audio setting is still there
            app._insp_close_all(); app.update()
            assert not app._insp_open and str(app._insp) not in [str(x) for x in app._pedals_paned.panes()]
            app._tile_press('wah', type('E', (), dict(x_root=10, y_root=10))())
            app._tile_release('wah', type('E', (), dict(x_root=10, y_root=10))())          # a click, not a drag
            app.update(); app.update()
            assert app._insp_open == ['wah'] and app._insp_sel == 'wah'
            assert str(app._insp) in [str(x) for x in app._pedals_paned.panes()]
            assert app._pedal_pages['wah'].winfo_manager() == 'pack' and not app._pedal_pages['comp'].winfo_manager()
            assert app._pedal_icon_cells['wah'].cget('highlightbackground') == st.ACCENT      # the open pedal's tile is outlined
            assert app._insp.winfo_y() > app._board_host.winfo_y() and app._insp.winfo_height() >= 150
            # the split sits right below the pedals: the board is only as tall as its tiles, the inspector gets the rest
            app.update(); app.update()
            need = app._board_inner.winfo_reqheight()
            assert abs(app._board_host.winfo_height() - need) <= 12, (app._board_host.winfo_height(), need)
            app._open_pedal('reverb'); app.update()       # the board's tiles are the tabs: one pedal shows at a time
            assert app._insp_open == ['reverb'] and app._pedal_pages['reverb'].winfo_manager() == 'pack'
            assert not app._pedal_pages['wah'].winfo_manager()
            assert app._pedal_icon_cells['reverb'].cget('highlightbackground') == st.ACCENT
            assert app._pedal_icon_cells['wah'].cget('highlightbackground') != st.ACCENT
            assert app._insp_title.cget('text') == st.PEDAL_SECTION_TITLES['reverb']
            app._insp_toggle_max(); app.update()
            assert str(app._board_host) not in [str(x) for x in app._pedals_paned.panes()]
            app._insp_toggle_max(); app.update()
            assert str(app._board_host) in [str(x) for x in app._pedals_paned.panes()]
            # clicking the pedal that is showing closes the panel
            app._tile_press('reverb', type('E', (), dict(x_root=10, y_root=10))())
            app._tile_release('reverb', type('E', (), dict(x_root=10, y_root=10))()); app.update()
            assert not app._insp_open and str(app._insp) not in [str(x) for x in app._pedals_paned.panes()]
            app._open_pedal('wah'); app.update()

            # dragging: dropping on the chain switches a pedal on at that spot, dropping on the tray switches it off
            assert not app._pedal_is_on('dist')
            app._drop_pedal('dist', 'chain', 0); app.update(); app.update()
            assert app._pedal_is_on('dist') and p.dist_on and chain_now()[0] == 'dist'
            assert grid_pos('dist') == min(grid_pos(c) for c in chain_now())
            app._drop_pedal('dist', 'tray', None); app.update(); app.update()
            assert not app._pedal_is_on('dist') and not p.dist_on and 'dist' not in chain_now()
            app._drop_pedal('phaser', 'chain', len(chain_now()) - 1); app.update(); app.update()
            assert app._pedal_is_on('phaser') and chain_now().index('phaser') == len(chain_now()) - 2
            app._set_pedal_on('phaser', False)
            # the earlier/later buttons hop over switched-off pedals (dist is off, between fuzz and od)
            ch0 = chain_now()
            app._move_fx('wah', 1); app.update()
            assert chain_now().index('wah') == ch0.index('wah') + 1
            app._move_fx('wah', 1); app.update()
            assert chain_now().index('wah') == ch0.index('wah') + 2 and p.order.index('wah') > p.order.index('od')
            app._reset_pedal_order(); app.update()
            app._set_pedal_on('comp', False); app._set_pedal_on('wah', False)

            # the notebook tabs: every state has the same padding, so the selected tab is never smaller
            style = ttk.Style(app)
            looks = {tuple(map(str, style.lookup('TNotebook.Tab', 'padding', state)))
                     for state in ([], ['selected'], ['active'], ['!selected'])}
            assert len(looks) == 1, f'tab padding differs between states: {looks}'
            assert style.lookup('TNotebook.Tab', 'foreground', ['selected']) == st.ACCENT_DARK
            assert style.lookup('TNotebook.Tab', 'background', ['selected']) == st.ACCENT

            # the tile text follows the pedal's settings (polled, so call the refresh directly)
            p.reverb_type = 'spring'; p.chorus_mode = 'vibrato'; p.od_mode = 'bd2'
            app._refresh_board_summaries()
            assert app._pedal_sum_lbls['reverb'].cget('text') == 'Spring'
            assert app._pedal_sum_lbls['chorus'].cget('text') == 'Vibrato'
            assert app._pedal_sum_lbls['od'].cget('text') == 'Blues Driver'

            # Decay / Damping are shown for every reverb type except Room
            app.rev_type_var.set('room'); app.update()
            assert not app.rev_extra_frame.outer.winfo_manager()
            app.rev_type_var.set('plate'); app.update()
            assert app.rev_extra_frame.outer.winfo_manager() == 'grid'
            # ...and the MIDI/pad mode-cycle moves the on-screen type radio with it
            p.reverb_type = 'cloud'
            assert app._cycle_pedal_mode('pedal_reverb_on') == 'shimmer'
            assert app.rev_type_var.get() == 'shimmer'
            app.update()
            assert app._rev_page.flow.winfo_children()      # the page is built
            assert app._cycle_pedal_mode('pedal_reverb_on') == 'room'
            assert app.rev_type_var.get() == 'room'

            # round trip through a Tone Profile: every new setting survives, and the gate mode too
            p.phaser_on = True; p.phaser_rate = 1.7; p.phaser_depth = 0.9; p.phaser_feedback = 0.55
            p.tremolo_on = True; p.tremolo_rate = 7.5; p.tremolo_depth = 0.8; p.tremolo_shape = 0.4
            p.reverb_on = True; p.reverb_type = 'spring'; p.reverb_decay = 0.8; p.reverb_damp = 0.2
            p.chorus_mode = 'doubler'; p.od_mode = 'bd2'
            p.rotary_on = True; p.rotary_speed = 'fast'; p.rotary_slow = 1.1; p.rotary_fast = 8.0
            p.rotary_depth = 0.9; p.rotary_ramp = 0.2; p.rotary_mix = 0.7
            p.pitch_on = True; p.pitch_semi = -7; p.pitch_cents = 12; p.pitch_semi2 = 5; p.pitch_level2 = 0.4; p.pitch_mix = 0.65
            p.reverb_shimmer = 0.8
            p.wah_on = True; p.wah_mode = 'envelope'; p.wah_env_sens = 0.8; p.wah_env_decay = 0.2
            p.wah_env_type = 'lowpass'; p.wah_env_dir = 'down'; p.fuzz_mode = 'tone_bender'
            app.amp.gate_mode = 'smart'; app.amp.gate_range = 33.0; app.amp.gate_hold = 80.0
            app.amp.gate_release = 250.0; app.amp.gate_hush = 0.25
            data = app._gather_profile_dict('round trip')
            json.dumps(data)                                           # must be plain JSON
            assert data['pedals']['phaser_rate'] == 1.7 and data['pedals']['reverb_type'] == 'spring'
            assert data['gate_mode'] == 'smart' and data['gate_range'] == 33.0
            # scramble everything, then load
            p.phaser_on = False; p.phaser_rate = 0.1; p.tremolo_on = False; p.reverb_type = 'plate'
            p.reverb_decay = 0.0; p.chorus_mode = 'chorus'; p.od_mode = 'klon'
            p.rotary_on = False; p.rotary_speed = 'slow'; p.rotary_slow = 0.5; p.rotary_fast = 5.0
            p.rotary_depth = 0.1; p.rotary_ramp = 0.9; p.rotary_mix = 0.2
            p.pitch_on = False; p.pitch_semi = 12; p.pitch_cents = 0; p.pitch_semi2 = 0; p.pitch_level2 = 0.0; p.pitch_mix = 0.1
            p.reverb_shimmer = 0.1
            p.wah_mode = 'manual'; p.wah_env_sens = 0.1; p.wah_env_type = 'wah'; p.wah_env_dir = 'up'; p.fuzz_mode = 'fuzz_face'
            app.amp.gate_mode = 'standard'; app.amp.gate_range = 10.0
            app._apply_profile(json.loads(json.dumps(data)), str(tmp_path / 'round trip.stprofile'))
            assert (p.phaser_on, p.phaser_rate, p.phaser_depth, p.phaser_feedback) == (True, 1.7, 0.9, 0.55)
            assert (p.tremolo_on, p.tremolo_rate, p.tremolo_depth, p.tremolo_shape) == (True, 7.5, 0.8, 0.4)
            assert (p.reverb_type, p.reverb_decay, p.reverb_damp) == ('spring', 0.8, 0.2)
            assert (p.chorus_mode, p.od_mode) == ('doubler', 'bd2')
            assert (p.wah_mode, p.wah_env_sens, p.wah_env_decay, p.wah_env_type, p.wah_env_dir) == ('envelope', 0.8, 0.2, 'lowpass', 'down')
            assert p.fuzz_mode == 'tone_bender'
            assert (p.pitch_on, p.pitch_semi, p.pitch_cents, p.pitch_semi2, p.pitch_level2, p.pitch_mix) == (True, -7, 12, 5, 0.4, 0.65)
            assert p.reverb_shimmer == 0.8
            assert app._pedal_sum_lbls['pitch'].cget('text').startswith('-7 st')
            assert (p.rotary_on, p.rotary_speed, p.rotary_slow, p.rotary_fast, p.rotary_depth, p.rotary_ramp,
                    p.rotary_mix) == (True, 'fast', 1.1, 8.0, 0.9, 0.2, 0.7)
            assert app._pedal_sum_lbls['rotary'].cget('text') in ('Slow', 'Fast', 'Brake')
            # the Slow/Fast footswitch trigger flips the setting and the on-screen radio
            assert app._rotary_toggle_speed() == 'slow' and app.amp.pedals.rotary_speed == 'slow'
            assert app._cycle_pedal_mode('pedal_rotary_on') == 'fast'
            app._open_pedal('wah'); app.update()
            assert app.wah_mode_var.get() == 'envelope' and app.wah_env_frame.winfo_manager() == 'pack'
            assert not app.wah_manual_frame.winfo_manager()
            assert (app.amp.gate_mode, app.amp.gate_range, app.amp.gate_hold, app.amp.gate_release,
                    app.amp.gate_hush) == ('smart', 33.0, 80.0, 250.0, 0.25)
            assert app.gate_mode_var.get() == 'smart' and app.gate_smart_frame.winfo_manager() == 'pack'
            assert app.rev_type_var.get() == 'spring'                    # the rebuilt tab shows the loaded type

            # a profile saved before any of this existed: the new pedals come back OFF and at factory
            # values (not left as the previous profile had them), the gate is the standard one, and the
            # old 8-pedal order gets the new pedals added
            old = {k: v for k, v in data.items() if k not in ('gate_mode', 'gate_range', 'gate_release',
                                                               'gate_hold', 'gate_hush')}
            old['pedals'] = {k: v for k, v in data['pedals'].items()
                             if not k.startswith(('phaser_', 'tremolo_', 'reverb_type', 'reverb_decay', 'reverb_damp'))}
            old['pedals']['order'] = ['comp', 'wah', 'fuzz', 'dist', 'od', 'amp', 'chorus', 'delay', 'reverb']
            app._apply_profile(old, str(tmp_path / 'old.stprofile'))
            assert p.phaser_on is False and p.tremolo_on is False
            assert (p.reverb_type, p.reverb_decay) == ('room', 0.5)
            assert app.amp.gate_mode == 'standard' and app.gate_mode_var.get() == 'standard'
            assert 'phaser' in p.order and 'tremolo' in p.order
            assert not app.gate_smart_frame.winfo_manager()
        finally:
            app.destroy()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'PASS {fn.__name__}')
    print(f'\n{len(tests)} tests passed.')
