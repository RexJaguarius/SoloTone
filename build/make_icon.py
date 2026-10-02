#!/usr/bin/env python3
"""
Generate build/solotone.ico from the Tally Light mark (the same icon used
in favicon.svg on the website), for use as the packaged .exe's icon, and
render the embedded taskbar/window PNGs solotone.py uses at runtime.

The mark has no background plate — it's drawn straight onto a transparent
canvas, scaled up to the largest size that keeps every stroke (including
outline width) inside the canvas — so the face reads bigger at the small
sizes a Windows taskbar actually renders an icon at than the original
rounded-square-plate version did.

Run once (or whenever the mark changes):
    pip install pillow
    python build/make_icon.py
"""
import base64
import io
import pathlib
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).parent
OUT  = HERE / 'solotone.ico'

SCALE = 8                 # working resolution multiplier (source viewBox is 0-120)
SIZE  = 120 * SCALE

# Matches solotone.py's current default theme (DEFAULT_BG/DEFAULT_ACCENT) —
# update these by hand if that default ever changes; this script doesn't
# import solotone.py itself (it has its own heavier runtime deps, and this
# is a design asset, not something that should track a user's own
# in-app Appearance color choice).
PLATE   = (38, 38, 38, 255)     # #262626 — neutral dark grey (was a blue-tinted #20262d)
INK     = (173, 173, 173, 255)  # #adadad — neutral light grey (was blue-tinted #aab4bd)
ACCENT  = (31, 126, 137, 255)   # #1f7e89 — teal, matches DEFAULT_ACCENT (was green #3ddc97)
CORAL   = (255, 107, 107, 255)  # #ff6b6b — unchanged; matches solotone.py's RED constant

# Original geometry (below) was designed at a 62x73px scale within a
# 0-120 viewBox, leaving a lot of dead transparent margin once the
# rounded-square background plate is removed. FACE_SCALE enlarges every
# coordinate around (CX, CY) — the original composition's own center —
# by the largest factor that still keeps the ear cups' fill *and* their
# outline stroke inside the 0-120 canvas (they're the widest element):
# solving 60 - 44*S - (3.5*S)/2 >= ~2px margin gives S <= ~1.27.
FACE_SCALE = 1.25
CX, CY = 60.0, 61.0


def t(x, y):
    """Map an original-design coordinate to its enlarged position."""
    return (CX + (x - CX) * FACE_SCALE, CY + (y - CY) * FACE_SCALE)


def tw(w):
    """Scale a stroke width / radius the same way as coordinates."""
    return w * FACE_SCALE


def s(v):
    return v * SCALE


def thick_path(draw, points, width, fill):
    w = s(width)
    for a, b in zip(points, points[1:]):
        draw.line([a, b], fill=fill, width=int(w))
    r = w / 2
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def qb(p0, p1, p2, n=48):
    """Quadratic bezier through already-enlarged (t()-mapped) points,
    returned at working (s()) resolution."""
    pts = []
    for i in range(n + 1):
        frac = i / n
        x = (1-frac)**2*p0[0] + 2*(1-frac)*frac*p1[0] + frac**2*p2[0]
        y = (1-frac)**2*p0[1] + 2*(1-frac)*frac*p1[1] + frac**2*p2[1]
        pts.append((s(x), s(y)))
    return pts


def sp(pt):
    """Scale an already-enlarged (x, y) point to working resolution."""
    return (s(pt[0]), s(pt[1]))


def draw_mark():
    """Draw the Tally Light mark, enlarged and with no background plate,
    onto a transparent SIZE x SIZE canvas."""
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # headphone band: two verticals + a quadratic arc over the top
    band_pts = (
        [sp(t(24, 60)), sp(t(24, 47))]
        + qb(t(24, 47), t(60, 25), t(96, 47))
        + [sp(t(96, 47)), sp(t(96, 60))]
    )
    thick_path(d, band_pts, tw(7), INK)

    # ear cups
    ex0, ey0 = t(16, 50); ex1, ey1 = t(34, 84)
    d.rounded_rectangle([s(ex0), s(ey0), s(ex1), s(ey1)], radius=s(tw(9)),
                         fill=PLATE, outline=INK, width=int(s(tw(3.5))))
    ex0, ey0 = t(86, 50); ex1, ey1 = t(104, 84)
    d.rounded_rectangle([s(ex0), s(ey0), s(ex1), s(ey1)], radius=s(tw(9)),
                         fill=PLATE, outline=INK, width=int(s(tw(3.5))))

    # head
    hx0, hy0 = t(29, 36); hx1, hy1 = t(91, 98)
    d.ellipse([s(hx0), s(hy0), s(hx1), s(hy1)], fill=PLATE, outline=INK,
              width=int(s(tw(4))))

    # eyes (happy arcs, bulging upward)
    thick_path(d, qb(t(40, 63), t(47, 54), t(54, 63)), tw(5), ACCENT)
    thick_path(d, qb(t(66, 63), t(73, 54), t(80, 63)), tw(5), ACCENT)

    # mouth (smile)
    thick_path(d, qb(t(46, 80), t(60, 93), t(74, 80)), tw(5), ACCENT)

    # tally light on the ear cup, with a soft glow
    gx, gy = t(97, 59)
    glow = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gr = tw(10)
    gd.ellipse([s(gx - gr), s(gy - gr), s(gx + gr), s(gy + gr)], fill=(255, 107, 107, 60))
    img = Image.alpha_composite(img, glow)
    d = ImageDraw.Draw(img)
    lr = tw(5.5)
    d.ellipse([s(gx - lr), s(gy - lr), s(gx + lr), s(gy + lr)], fill=CORAL)

    return img


def main():
    img = draw_mark()

    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(OUT, format='ICO', sizes=sizes)
    print(f'Wrote {OUT} ({OUT.stat().st_size} bytes)')

    # Also render the two PNG sizes solotone.py embeds (as base64) for the
    # window/taskbar icon at runtime, and print ready-to-paste constants.
    for label, px in [('_APP_ICON_PNG_64', 64), ('_APP_ICON_PNG_32', 32)]:
        thumb = img.resize((px, px), Image.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        chunks = [b64[i:i+76] for i in range(0, len(b64), 76)]
        print(f'\n{label} = (')
        for c in chunks:
            print(f'    "{c}"')
        print(')')


if __name__ == '__main__':
    main()
