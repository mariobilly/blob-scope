"""Blob Scope engine: the dot-matrix dither + blob-tracker HUD look of Ishan Upadhyay's reel.

render(img, p, t, loop, mask) -> RGB uint8 frame.

A frame depends only on (picture, params, time). Previews match the export exactly, and frames can render in
parallel. All sizes scale with the picture width, so a 640 px preview looks the same as a 1080 px export.

Pipeline: crop to the frame shape -> optional paper cut-out -> camera motion -> tone -> blob analysis -> dots
-> corner halftone -> glitch slices -> HUD (tags, links, main tag, brackets) -> letterbox.
"""
import math
import os
import threading
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- parameters

DEFAULTS = {
    # frame
    "frame": "source",      # source | reel (3:4 art on a black 9:16 canvas) | 9:16 | 4:5 | 1:1 | 16:9
    "paper": False,         # cut the subject out onto paper (needs a mask from the app)
    "paper_color": "#ffffff",
    # dots
    "dots": True,
    "cols": 160,            # dot columns across the picture (the reel: 160 across 720 px)
    "dot": 0.5,             # dot size, as a fraction of its cell
    "shape": "square",      # square | round | plus
    "palette": "source",    # source (per-channel levels) or a named ramp
    "levels": 2,            # luminance levels per dot for the source palette (2 = dark/light)
    "warm": False,          # ramps keep fire-like accents (flames, glowing eyes) warm
    "pattern": "bayer4",    # bayer2 | bayer4 | bayer8
    "spread": 1.0,          # dither strength
    "grid": 0.0,            # darkens the base between the dots (LED-panel look)
    "contrast": 1.15,
    "brightness": 0.0,
    "saturation": 1.35,
    "hue": 0,               # hue rotation in degrees
    "ornament": "top-right",  # corner halftone: off | top-right | corners
    # tracker HUD
    "hud": True,
    "tags": 7,              # red tags on screen at once (on average)
    "life": 1.1,            # seconds a tag lives (on average)
    "links": True,
    "pair_labels": True,
    "main_tag": True,
    "brackets": True,
    "tag_scale": 1.0,
    "tag_color": "#e8222b",
    "link_color": "#ff8c96",
    "id_base": 3500,
    # motion and glitch
    "motion": "drift",      # off | drift | sway | shake | pulse
    "motion_amt": 1.0,
    "cycle": 6.0,           # seconds: the loop length of a photo, the motion cycle of a video
    "glitch": 0.35,
    "seed": 7,
}

FRAMES = ("source", "reel", "9:16", "4:5", "1:1", "16:9")
SHAPES = ("square", "round", "plus")
PATTERNS = {"bayer2": 2, "bayer4": 4, "bayer8": 8}
MOTIONS = ("off", "drift", "sway", "shake", "pulse")
ORNAMENTS = ("off", "top-right", "corners")


def _hex(c, default=(255, 255, 255)):
    try:
        c = str(c).lstrip("#")
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        return default


PALETTES = {  # dark -> light ramps; "source" keeps the picture's own colours
    "cobalt": ["#000000", "#0a0f73", "#1a2fe0", "#4d7dff", "#a9d3ff", "#ffffff"],
    "ink": ["#000000", "#ffffff"],
    "ember": ["#140303", "#7a0d0b", "#e8311a", "#ff9a2e", "#ffe9a8", "#ffffff"],
    "acid": ["#020a03", "#0c4f1b", "#22c94a", "#b8ff6a", "#f5ffe9"],
    "riso": ["#1b1140", "#ff3d8b", "#ffd23f", "#fff7e8"],
    "uv": ["#05010f", "#3a0ca3", "#7a12c2", "#f72585", "#ffd6f5", "#ffffff"],
    "night": ["#000000", "#03122e", "#0d47a1", "#00b4ff", "#c8f4ff"],
}
WARM_RAMP = ["#4a0800", "#e8231a", "#ff8c1a", "#ffe14a"]
_RAMPS = {k: np.array([_hex(c) for c in v], np.float32) / 255 for k, v in PALETTES.items()}
_WARM = np.array([_hex(c) for c in WARM_RAMP], np.float32) / 255

PRESETS = [
    {"id": "tracker", "name": "Blob Tracker", "note": "Own colours in dots, red tracker tags",
     "p": {}},
    {"id": "cobalt", "name": "Cobalt Scan", "note": "Any picture in cobalt blues, fire stays warm",
     "p": {"palette": "cobalt", "contrast": 1.25, "warm": True}},
    {"id": "paper", "name": "Paper Specimen", "note": "Subject cut out onto white paper",
     "p": {"paper": True, "palette": "cobalt", "contrast": 1.25, "warm": True}},
    {"id": "fine", "name": "Fine Grain", "note": "Small dense dots, calm tags",
     "p": {"cols": 240, "dot": 0.55, "tags": 5, "life": 1.6, "glitch": 0.15}},
    {"id": "led", "name": "Chunky LED", "note": "Big round dots, busy tracker",
     "p": {"cols": 72, "dot": 0.8, "shape": "round", "grid": 0.55, "tags": 9, "tag_scale": 1.25}},
    {"id": "ember", "name": "Ember Signal", "note": "Fire ramp with blue-free warm tracking",
     "p": {"palette": "ember", "tag_color": "#111111", "link_color": "#ffd0a0"}},
    {"id": "night", "name": "Night Scope", "note": "Dark cyan scope, black paper",
     "p": {"palette": "night", "grid": 0.6, "tag_color": "#00b4ff", "link_color": "#7fe3ff",
           "ornament": "corners"}},
    {"id": "riso", "name": "Riso Pop", "note": "Pink and yellow print dots",
     "p": {"palette": "riso", "pattern": "bayer8", "tag_color": "#1b1140", "link_color": "#ff3d8b"}},
    {"id": "ink", "name": "Ink Tracker", "note": "Black and white dots, red tags",
     "p": {"palette": "ink", "cols": 180}},
    {"id": "clean", "name": "Tracker Only", "note": "Your picture untouched, just the HUD",
     "p": {"dots": False, "ornament": "off", "glitch": 0.0}},
]


def params(d=None):
    """Defaults + overrides, clamped to safe ranges."""
    p = dict(DEFAULTS)
    for k, v in (d or {}).items():
        if k in DEFAULTS and v is not None:
            p[k] = v
    f = lambda k, lo, hi: min(hi, max(lo, float(p[k])))  # noqa: E731
    p["cols"] = int(round(f("cols", 16, 400)))
    p["dot"] = f("dot", 0.15, 1.0)
    p["levels"] = int(round(f("levels", 2, 8)))
    p["spread"] = f("spread", 0.0, 1.5)
    p["grid"] = f("grid", 0.0, 1.0)
    p["contrast"] = f("contrast", 0.2, 3.0)
    p["brightness"] = f("brightness", -0.6, 0.6)
    p["saturation"] = f("saturation", 0.0, 3.0)
    p["hue"] = f("hue", -180, 180)
    p["tags"] = int(round(f("tags", 0, 30)))
    p["life"] = f("life", 0.2, 6.0)
    p["tag_scale"] = f("tag_scale", 0.4, 3.0)
    p["id_base"] = int(round(f("id_base", 0, 99000)))
    p["motion_amt"] = f("motion_amt", 0.0, 3.0)
    p["cycle"] = f("cycle", 1.0, 60.0)
    p["glitch"] = f("glitch", 0.0, 1.0)
    p["seed"] = int(round(f("seed", 0, 1_000_000)))
    for k in ("dots", "paper", "warm", "hud", "links", "pair_labels", "main_tag", "brackets"):
        p[k] = bool(p[k]) and str(p[k]).lower() not in ("false", "0", "off")
    if p["frame"] not in FRAMES:
        p["frame"] = "source"
    if p["shape"] not in SHAPES:
        p["shape"] = "square"
    if p["palette"] != "source" and p["palette"] not in PALETTES:
        p["palette"] = "source"
    if p["pattern"] not in PATTERNS:
        p["pattern"] = "bayer4"
    if p["motion"] not in MOTIONS:
        p["motion"] = "off"
    if p["ornament"] not in ORNAMENTS:
        p["ornament"] = "off"
    return p


# --------------------------------------------------------------------------- small helpers

def _h(*k):
    """Deterministic hash of integers -> float in [0, 1)."""
    x = 0x9E3779B97F4A7C15
    for v in k:
        x ^= (int(v) & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15 + ((x << 6) & 0xFFFFFFFFFFFFFFFF) + (x >> 2)
        x &= 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
    return (x >> 11) / float(1 << 53)


def _bayer(n):
    m = np.array([[0]], np.float32)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)


BAYER = {n: _bayer(n) for n in (2, 4, 8)}


def _even(v):
    v = int(v)
    return max(2, v - v % 2)


# --------------------------------------------------------------------------- frame shape

def crop_box(w, h, frame):
    """Centered crop (x, y, cw, ch) with even sides for the art area of `frame`."""
    asp = {"reel": 3 / 4, "9:16": 9 / 16, "4:5": 4 / 5, "1:1": 1.0, "16:9": 16 / 9}.get(frame)
    if asp is None:
        cw, ch = w, h
    elif w / h > asp:
        cw, ch = h * asp, h
    else:
        cw, ch = w, w / asp
    cw, ch = _even(min(w, round(cw))), _even(min(h, round(ch)))
    return (w - cw) // 2, (h - ch) // 2, cw, ch


def out_size(w, h, p):
    """Final frame size for a source of w x h: the art crop, or the 9:16 canvas around it for 'reel'."""
    _, _, cw, ch = crop_box(w, h, p["frame"])
    if p["frame"] == "reel":
        return cw, _even(round(cw * 16 / 9))
    return cw, ch


# --------------------------------------------------------------------------- motion

def motion_matrix(w, h, p, t):
    """2x3 affine for the camera move at time t. Every term is periodic in the cycle, so photo loops are
    seamless."""
    m, a = p["motion"], p["motion_amt"]
    if m == "off" or a <= 0:
        return None
    L = p["cycle"]
    ph = 2 * math.pi * t / L
    s, rot, dx, dy = 1.0, 0.0, 0.0, 0.0
    cx, cy = w / 2, h / 2
    if m == "drift":
        s = 1.0 + a * (0.055 + 0.02 * math.sin(ph))
        rot = a * 0.7 * math.sin(ph + 0.5)
        dx = a * 0.014 * w * math.sin(ph + 1.3)
        dy = a * 0.010 * h * math.sin(2 * ph)
    elif m == "sway":  # the head-sway of the reel: rotate around the bottom centre
        cy = h * 0.95
        rot = a * 2.4 * math.sin(ph) + a * 0.6 * math.sin(3 * ph + 1)
        s = 1.0 + a * 0.06
        dx = a * 0.01 * w * math.sin(2 * ph + 0.7)
    elif m == "shake":
        s = 1.0 + a * 0.05
        k = int(math.floor(t * 12)) % max(1, int(round(L * 12)))
        dx = a * 0.008 * w * (_h(p["seed"], k, 1) - 0.5) * 2 + a * 0.008 * w * math.sin(ph)
        dy = a * 0.008 * h * (_h(p["seed"], k, 2) - 0.5) * 2
        rot = a * 0.4 * (_h(p["seed"], k, 3) - 0.5)
    elif m == "pulse":  # punch-in twice per loop, then ease back
        beat = (2 * t / L) % 1.0
        s = 1.0 + a * (0.04 + 0.06 * math.exp(-beat * 6))
        rot = a * 0.3 * math.sin(ph)
    M = cv2.getRotationMatrix2D((cx, cy), rot, s)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


# --------------------------------------------------------------------------- tone + dots

def tone(x, p):
    """x: float32 RGB 0..1 (small grid). Contrast, brightness, saturation, hue."""
    if p["hue"]:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)  # float: H in degrees
        hsv[..., 0] = (hsv[..., 0] + p["hue"]) % 360
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    x = (x - 0.5) * p["contrast"] + 0.5 + p["brightness"]
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    x = lum[..., None] + (x - lum[..., None]) * p["saturation"]
    return np.clip(x, 0.0, 1.0)


LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _cmax(c):
    return np.maximum(np.maximum(c[..., 0], c[..., 1]), c[..., 2])  # much faster than c.max(-1) on 3 channels


def _cmin(c):
    return np.minimum(np.minimum(c[..., 0], c[..., 1]), c[..., 2])


def _offsets(rows, cols, p):
    n = PATTERNS[p["pattern"]]
    thr = np.tile(BAYER[n], (rows // n + 1, cols // n + 1))[:rows, :cols]
    return (thr - 0.5) * p["spread"] + 0.5  # spread 0 -> plain rounding


def _warm_mask(c):
    """c: the picture's own colours (before tone), so a contrast boost cannot turn skin into 'fire'."""
    mx, mn = _cmax(c), _cmin(c)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    # fire, not skin or neon pink: red-orange-yellow (green above blue), strongly saturated and bright
    return (r >= g) & (g > b * 1.15) & (r > b * 1.8) & (sat > 0.72) & (mx > 0.55)


def _shade(c, v):
    """The cell's hue at lightness v: 0 = near-black tint, 0.5 = the pure hue, 1 = near-white tint.

    Two linear ramps (dark tint -> hue -> light tint) folded into one multiply-add per cell.
    """
    hue = c / np.maximum(_cmax(c), 1e-3)[..., None]
    a = np.minimum(v * 2, 1.0)[..., None]
    b = np.maximum(v * 2 - 1, 0.0)[..., None]
    return hue * (0.14 + 0.86 * a - 0.8 * b) + 0.8 * b


def dot_colours(c, p, raw=None):
    """c: tone-mapped float RGB grid, one value per dot. Returns each dot's colour.

    source: each dot is a luminance dither between a dark and a light tint of its own colour (more levels add
    the pure hue in between). Blue areas sparkle with pale-blue dots, light areas get navy specks, fire stays
    warm: the reel. A named palette dithers between the two nearest colours of its ramp.
    """
    rows, cols = c.shape[:2]
    off = _offsets(rows, cols, p)
    lum = c @ LUMA
    if p["palette"] == "source":
        L = p["levels"] - 1
        light = (_cmax(c) + _cmin(c)) * 0.5  # HSL lightness: a pure hue sits at 0.5
        return _shade(c, np.clip(np.floor(light * L + off) / L, 0, 1))
    ramp = _RAMPS[p["palette"]]
    k = len(ramp) - 1
    q = ramp[np.clip(np.floor(lum * k + off), 0, k).astype(np.int32)]
    if p["warm"]:
        warm = _warm_mask(c if raw is None else raw)
        if warm.any():
            kw = len(_WARM) - 1
            wv = np.clip(_cmax(c) * 0.55 + c[..., 1] / np.maximum(c[..., 0], 1e-4) * 0.45, 0, 1)
            q = np.where(warm[..., None], _WARM[np.clip(np.floor(wv * kw + off), 0, kw).astype(np.int32)], q)
    return q


def base_colours(c, p, raw=None):
    """The layer between the dots: the picture's own (tone-mapped) colour, or its place on the ramp."""
    if p["palette"] == "source":  # pushed apart: darks sink to navy so the light dots carry the texture
        light = (_cmax(c) + _cmin(c)) * 0.5
        out = _shade(c, np.clip((light - 0.5) * 1.5 + 0.5, 0, 1))
    else:
        ramp = _RAMPS[p["palette"]]
        k = len(ramp) - 1
        pos = (c @ LUMA) * k
        lo = np.clip(np.floor(pos), 0, k - 1).astype(np.int32)
        fr = (pos - lo)[..., None]
        out = ramp[lo] * (1 - fr) + ramp[lo + 1] * fr
        if p["warm"]:
            warm = _warm_mask(c if raw is None else raw)
            if warm.any():
                kw = len(_WARM) - 1
                wv = np.clip(_cmax(c) * 0.55 + c[..., 1] / np.maximum(c[..., 0], 1e-4) * 0.45, 0, 1) * kw
                wl = np.clip(np.floor(wv), 0, kw - 1).astype(np.int32)
                wf = (wv - wl)[..., None]
                out = np.where(warm[..., None], (_WARM[wl] * (1 - wf) + _WARM[wl + 1] * wf) * 0.62, out)
    return out * (1.0 - 0.75 * p["grid"])


@lru_cache(maxsize=24)
def _dot_mask(W, H, cols, rows, dot, shape):
    fx = (np.arange(W, dtype=np.float32) + 0.5) * cols / W % 1.0 - 0.5
    fy = (np.arange(H, dtype=np.float32) + 0.5) * rows / H % 1.0 - 0.5
    r = dot / 2
    if shape == "round":
        m = fy[:, None] ** 2 + fx[None, :] ** 2 < r * r
    elif shape == "plus":
        ax, ay = np.abs(fx) < r, np.abs(fy) < r
        tx, ty = np.abs(fx) < r * 0.36, np.abs(fy) < r * 0.36
        m = (ay[:, None] & tx[None, :]) | (ty[:, None] & ax[None, :])
    else:
        m = (np.abs(fy) < r)[:, None] & (np.abs(fx) < r)[None, :]
    return m.astype(np.uint8)


def grid_size(W, H, p):
    cols = int(min(p["cols"], W))
    rows = max(1, min(H, int(round(H * cols / W))))
    return cols, rows


def dots(art, p):
    """art: uint8 RGB. Returns the dot-matrix picture at the same size.

    Base layer: the picture's colour on a grid twice as fine as the dots, darkened by `grid`.
    Dot layer: the ordered-dither colour of each cell, drawn as a small square / circle / plus.
    """
    H, W = art.shape[:2]
    cols, rows = grid_size(W, H, p)
    small = cv2.resize(art, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    q8 = (np.clip(dot_colours(tone(small, p), p, small), 0, 1) * 255 + 0.5).astype(np.uint8)
    bc, br = min(W, cols * 2), min(H, rows * 2)
    fine = cv2.resize(art, (bc, br), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    g8 = (np.clip(base_colours(tone(fine, p), p, fine), 0, 1) * 255 + 0.5).astype(np.uint8)
    big_g = cv2.resize(g8, (W, H), interpolation=cv2.INTER_NEAREST)
    big_q = cv2.resize(q8, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = _dot_mask(W, H, cols, rows, round(p["dot"], 3), p["shape"])
    cv2.copyTo(big_q, mask, big_g)
    return big_g


def ornament(img, p, light):
    """Corner halftone: a fading field of small dots in the light (paper) corners."""
    if p["ornament"] == "off":
        return img
    H, W = img.shape[:2]
    step = max(4.0, W / p["cols"] * 2.3)
    R = 0.42 * min(W, H)
    corners = [(W, 0)] if p["ornament"] == "top-right" else [(0, 0), (W, 0), (0, H), (W, H)]
    ink = (12, 22, 120) if p["palette"] in ("source", "cobalt") else tuple(int(c * 255) for c in _RAMPS.get(
        p["palette"], _RAMPS["cobalt"])[1])
    if p["palette"] == "night":
        ink = (0, 180, 255)
    ys = np.arange(step / 2, H, step)
    xs = np.arange(step / 2, W, step)
    gx, gy = np.meshgrid(xs, ys)
    rad = np.zeros_like(gx)
    for cx, cy in corners:
        d = np.hypot(gx - cx, gy - cy)
        rad = np.maximum(rad, np.clip(1 - d / R, 0, 1) ** 1.4)
    rad *= step * 0.22
    # only where the picture is paper-like (very light, or very dark for dark palettes)
    lh, lw = light.shape
    li = light[np.clip((gy / H * lh).astype(int), 0, lh - 1), np.clip((gx / W * lw).astype(int), 0, lw - 1)]
    want = li < 0.12 if p["palette"] == "night" else li > 0.86
    sel = (rad >= 0.6) & want
    out = img.copy()
    for x, y, r in zip(gx[sel], gy[sel], rad[sel]):
        r = max(1, int(round(r)))
        x0, y0 = int(x - r / 2), int(y - r / 2)
        out[max(0, y0):y0 + r, max(0, x0):x0 + r] = ink
    return out


def glitch(img, p, t, loop):
    """Horizontal slice displacement with a red/blue split, in short random bursts (12 Hz clock)."""
    g = p["glitch"]
    if g <= 0:
        return img
    k = int(math.floor(t * 12))
    if loop:
        k %= max(1, int(round(loop * 12)))
    if _h(p["seed"], k, 11) > g * 0.28:
        return img
    H, W = img.shape[:2]
    out = img.copy()
    for j in range(1 + int(_h(p["seed"], k, 12) * 3)):
        y0 = int(_h(p["seed"], k, 20 + j) * H * 0.95)
        hh = max(2, int((0.015 + 0.07 * _h(p["seed"], k, 30 + j)) * H))
        sh = int((_h(p["seed"], k, 40 + j) - 0.5) * 0.16 * W * (0.4 + g))
        band = np.roll(img[y0:y0 + hh], sh, axis=1)
        split = max(1, W // 240)
        band[..., 0] = np.roll(band[..., 0], split, axis=1)
        band[..., 2] = np.roll(band[..., 2], -split, axis=1)
        out[y0:y0 + hh] = band
    return out


# --------------------------------------------------------------------------- blob analysis

AW = 320  # analysis width


def analyse(art, p, mask=None):
    """Find the main blob (the subject) and small feature blobs to tag. Coordinates are normalised 0..1."""
    H, W = art.shape[:2]
    aw = min(AW, W)
    ah = max(8, int(round(H * aw / W)))
    a = cv2.resize(art, (aw, ah), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    light = gray.astype(np.float32) / 255
    if mask is not None:
        subj = (cv2.resize(mask, (aw, ah), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
    else:
        b = max(2, aw // 25)
        border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3),
                                 a[:, -b:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        diff = np.abs(a.astype(np.int16) - bg.astype(np.int16)).max(-1).astype(np.uint8)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        subj = (diff > max(28, otsu)).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        subj = cv2.morphologyEx(cv2.morphologyEx(subj, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    main = None
    n, lab, stats, cent = cv2.connectedComponentsWithStats(subj, connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 0.01 * aw * ah:
            x, y, w, h = stats[i, :4]
            ys, xs = np.nonzero(lab == i)
            # label anchor: the blob pixel nearest its centroid (the centroid can fall outside a C-shape)
            cxp, cyp = cent[i]
            j = int(np.argmin((xs - cxp) ** 2 + (ys - cyp) ** 2))
            main = {"pct": 100.0 * area / (aw * ah), "box": (x / aw, y / ah, (x + w) / aw, (y + h) / ah),
                    "at": ((xs[j] + 0.5) / aw, (ys[j] + 0.5) / ah)}
    cover = subj.mean()
    gap = max(4, aw / 22)
    if 0.02 < cover < 0.92:
        # like the reel, most tags ride the subject's outline (feather tips, beak, neck), a few sit inside it
        r = max(3, aw // 40) | 1
        k = np.ones((r, r), np.uint8)
        band = cv2.dilate(subj, k) - cv2.erode(subj, k)
        inner = cv2.erode(subj, k)
        groups = [(band, 70), (inner, 25)]
    else:
        groups = [(None, 90)]
    feats = []
    for m, n_max in groups:
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=n_max, qualityLevel=0.02, minDistance=gap, mask=m,
                                      blockSize=5)
        if pts is not None:
            feats += [((x + 0.5) / aw, (y + 0.5) / ah) for x, y in pts.reshape(-1, 2)]
    return {"main": main, "feats": feats, "light": light, "gray": gray}


# --------------------------------------------------------------------------- tag schedule

def visible_tags(p, t, loop):
    """Births alive at time t: (birth index, age in seconds). Stateless and periodic when `loop` is set."""
    if p["tags"] <= 0:
        return []
    rate = p["tags"] / p["life"]
    dmax = p["life"] * 1.6
    out = []
    if loop:
        B = max(1, int(round(rate * loop)))
        for i in range(B):
            ti = (i + _h(p["seed"], i, 1)) * loop / B
            age = (t - ti) % loop
            if age < p["life"] * (0.4 + 1.2 * _h(p["seed"], i, 2)):
                out.append((i, age))
    else:
        for i in range(max(0, int((t - dmax) * rate) - 2), int(t * rate) + 2):
            ti = (i + _h(p["seed"], i, 1)) / rate
            age = t - ti
            if 0 <= age < p["life"] * (0.4 + 1.2 * _h(p["seed"], i, 2)):
                out.append((i, age))
    return out


def blob_pct(gray, x, y):
    """Area (% of the frame) of the little blob under a feature: the patch pixels on the feature's side of the
    patch mean, connected to it."""
    ah, aw = gray.shape
    xi, yi = min(aw - 1, int(x * aw)), min(ah - 1, int(y * ah))
    r = max(3, aw // 16)
    x0, y0 = max(0, xi - r), max(0, yi - r)
    patch = gray[y0:yi + r + 1, x0:xi + r + 1]
    thr = patch.mean()
    binm = ((patch > thr) == (gray[yi, xi] > thr)).astype(np.uint8)
    _, lab = cv2.connectedComponents(binm, connectivity=4)
    return 100.0 * float((lab == lab[yi - y0, xi - x0]).sum()) / (aw * ah)


def place_tags(p, t, loop, feats, aspect, gray=None):
    """Give each living birth the free feature nearest its anchor. Returns [(id, x, y, pct, age)]."""
    if not feats:
        return []
    F = np.array(feats, np.float32)
    used = np.zeros(len(F), bool)
    placed = []
    min_gap = 0.075
    for i, age in sorted(visible_tags(p, t, loop)):
        ax, ay = 0.06 + 0.88 * _h(p["seed"], i, 3), 0.06 + 0.88 * _h(p["seed"], i, 4)
        d = (F[:, 0] - ax) ** 2 + ((F[:, 1] - ay) / aspect) ** 2
        d[used] = np.inf
        for j in np.argsort(d)[:12]:
            if not np.isfinite(d[j]) or d[j] > 0.2:
                break
            x, y = F[j]
            if all((x - q[1]) ** 2 + ((y - q[2]) / aspect) ** 2 > min_gap ** 2 for q in placed):
                used[j] = True
                pct = blob_pct(gray, x, y) if gray is not None else 0.0
                placed.append((p["id_base"] + 150 + i, float(x), float(y), pct, age))
                break
    return placed


# --------------------------------------------------------------------------- HUD drawing

FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")


_TL = threading.local()  # FreeType faces are not thread-safe: every render thread keeps its own fonts


def font(size, bold=True):
    cache = getattr(_TL, "fonts", None)
    if cache is None:
        cache = _TL.fonts = {}
    f = cache.get((size, bold))
    if f is None:
        f = ImageFont.load_default()
        for name in (("consolab.ttf", "courbd.ttf") if bold else ("consola.ttf", "cour.ttf")):
            try:
                f = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
                break
            except OSError:
                continue
        cache[(size, bold)] = f
    return f


def _label(d, x, y, text, fnt, fill, ink, W, H, pad):
    """Box with text centred on (x, y), kept inside the frame. Returns the box."""
    l, t_, r, b = fnt.getbbox(text)
    tw, th = r - l, b - t_
    bw, bh = tw + 2 * pad[0], th + 2 * pad[1]
    x0 = int(min(max(0, x - bw / 2), W - bw))
    y0 = int(min(max(0, y - bh / 2), H - bh))
    if fill is not None:
        d.rectangle((x0, y0, x0 + bw - 1, y0 + bh - 1), fill=fill)
    d.text((x0 + pad[0] - l, y0 + pad[1] - t_), text, font=fnt, fill=ink)
    return (x0, y0, x0 + bw, y0 + bh)


def _bracket_ink(light, x, y, W, H):
    lh, lw = light.shape
    v = light[min(lh - 1, max(0, int(y / H * lh))), min(lw - 1, max(0, int(x / W * lw)))]
    return (28, 28, 36, 255) if v > 0.5 else (236, 236, 240, 255)


def draw_hud(img, p, t, loop, info):
    H, W = img.shape[:2]
    feats, main, light = info["feats"], info["main"], info["light"]
    tags = place_tags(p, t, loop, feats, H / W, info.get("gray"))
    if not tags and not main:
        return img
    base = Image.fromarray(img)
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))  # everything is drawn here, then pasted once
    d = ImageDraw.Draw(over)
    fs = max(8, int(round(W * 0.0142 * p["tag_scale"])))
    f_tag, f_pair = font(fs, True), font(max(7, int(fs * 0.86)), False)
    lw = max(1, int(round(W / 900)))
    red = _hex(p["tag_color"], (232, 34, 43))
    pink = _hex(p["link_color"], (255, 140, 150))
    pts = [(tid, x * W, y * H, pct, age) for tid, x, y, pct, age in tags]
    main_id = p["id_base"] + 11
    main_pt = (main["at"][0] * W, main["at"][1] * H) if (main and p["main_tag"]) else None

    # corner brackets around the main blob
    if main and p["brackets"]:
        x0, y0, x1, y1 = main["box"]
        pad = 0.02
        inset = max(2, int(W * 0.004))
        X0, Y0 = max(inset, (x0 - pad) * W), max(inset, (y0 - pad) * H)
        X1, Y1 = min(W - 1 - inset, (x1 + pad) * W), min(H - 1 - inset, (y1 + pad) * H)
        L = 0.075 * min(W, H)
        for cx, cy, sx, sy in ((X0, Y0, 1, 1), (X1, Y0, -1, 1), (X0, Y1, 1, -1), (X1, Y1, -1, -1)):
            ink = _bracket_ink(light, cx + sx * L / 2, cy + sy * L / 2, W, H)
            d.line([(cx, cy + sy * L), (cx, cy), (cx + sx * L, cy)], fill=ink, width=lw)

    # links: each tag to its two nearest, the main tag to its nearest one or two
    if p["links"] and pts:
        pairs = set()
        for a, (ia, xa, ya, _, _) in enumerate(pts):
            near = sorted(((xa - xb) ** 2 + (ya - yb) ** 2, b) for b, (_, xb, yb, _, _) in enumerate(pts) if b != a)
            for dist2, b in near[:2]:
                if dist2 < (0.5 * W) ** 2:
                    pairs.add((min(a, b), max(a, b)))
        alpha = 150
        for a, b in sorted(pairs):
            (ia, xa, ya, _, _), (ib, xb, yb, _, _) = pts[a], pts[b]
            d.line([(xa, ya), (xb, yb)], fill=pink + (alpha,), width=lw)
            if p["pair_labels"] and _h(p["seed"], ia, ib, 5) < 0.7:
                lo, hi = sorted((ia, ib))
                _label(d, (xa + xb) / 2, (ya + yb) / 2 - fs * 0.6, f"{lo}-{hi}", f_pair, None, pink + (220,), W, H,
                       (0, 0))
        if main_pt:
            near = sorted(((main_pt[0] - x) ** 2 + (main_pt[1] - y) ** 2, k) for k, (_, x, y, _, _) in
                          enumerate(pts))
            for n_, (_, k) in enumerate(near[:1 + int(_h(p["seed"], int(t * 2), 6) * 2)]):
                tid, x, y, _, _ = pts[k]
                d.line([main_pt, (x, y)], fill=(170, 170, 178, 150), width=lw)
                if p["pair_labels"] and n_ == 0:
                    _label(d, (main_pt[0] + x) / 2 + fs * 2, (main_pt[1] + y) / 2, f"{main_id}-{tid}", f_pair,
                           None, (150, 150, 160, 230), W, H, (0, 0))

    d2 = d
    pad = (max(2, int(fs * 0.32)), max(1, int(fs * 0.2)))
    for tid, x, y, pct, age in pts:
        flash = age < 0.09  # a tag blinks in inverted for its first couple of frames
        fill, ink = ((255, 255, 255, 255), red + (255,)) if flash else (red + (255,), (255, 255, 255, 255))
        _label(d2, x, y, f"#{tid} {pct:.1f}%", f_tag, fill, ink, W, H, pad)
    if main_pt:
        _label(d2, main_pt[0], main_pt[1], f"#{main_id} {main['pct']:.1f}%", f_tag, (8, 8, 12, 190),
               (255, 255, 255, 255), W, H, pad)
    base.paste(over, (0, 0), over)
    return np.asarray(base)


# --------------------------------------------------------------------------- render

def prepare(img, p, mask=None):
    """Crop to the frame shape and (optionally) put the subject on paper. Returns (art, art_mask)."""
    h, w = img.shape[:2]
    x, y, cw, ch = crop_box(w, h, p["frame"])
    art = img[y:y + ch, x:x + cw]
    m = None
    if mask is not None:
        m = mask[y:y + ch, x:x + cw] if mask.shape[:2] == (h, w) else cv2.resize(
            mask, (w, h), interpolation=cv2.INTER_LINEAR)[y:y + ch, x:x + cw]
    if p["paper"] and m is not None:
        a = (m.astype(np.float32) / 255)[..., None]
        paper = np.array(_hex(p["paper_color"]), np.float32)
        art = (art.astype(np.float32) * a + paper * (1 - a) + 0.5).astype(np.uint8)
    return np.ascontiguousarray(art), m


def render(img, p, t=0.0, video=False, mask=None):
    """img: HxWx3 uint8 RGB source frame. t: seconds.
    video=False: a photo loop, everything repeats every `cycle` seconds. video=True: tag IDs keep counting up.
    mask: optional HxW uint8 subject matte (255 = subject) for the paper cut-out and a cleaner main blob."""
    p = params(p)
    loop = None if video else p["cycle"]
    art, m = prepare(img, p, mask)
    H, W = art.shape[:2]
    M = motion_matrix(W, H, p, t)
    if M is not None:
        border = _hex(p["paper_color"]) if p["paper"] else None
        art = cv2.warpAffine(art, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT if border else cv2.BORDER_REFLECT101,
                             borderValue=border or 0)
        if m is not None:
            m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    info = analyse(art, p, m if p["paper"] else None) if p["hud"] or p["ornament"] != "off" else None
    out = dots(art, p) if p["dots"] else art.copy()
    if p["dots"] and info is not None:
        out = ornament(out, p, info["light"])
    out = glitch(out, p, t, loop)
    if p["hud"] and info is not None:
        out = draw_hud(out, p, t, loop, info)
    if p["frame"] == "reel":
        oh = out_size(W, H, p)[1]
        canvas = np.zeros((oh, W, 3), np.uint8)
        y0 = (oh - H) // 2
        canvas[y0:y0 + H] = out
        out = canvas
    return out


def warm_up():
    """Load the fonts and touch the kernels once so the first preview is quick."""
    try:
        img = np.full((180, 120, 3), 255, np.uint8)
        cv2.circle(img, (60, 90), 40, (20, 40, 220), -1)
        render(img, {}, 0.5)
    except Exception:
        pass
