"""Draws the Blob Scope icon (icon.png + multi-size icon.ico): a dot-matrix blue blob on paper, with a glowing
eye, corner brackets and a red tracker tag, like a frame of the reel."""
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024  # drawn at 2x, then downsampled for clean edges


def draw():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle((24, 24, S - 24, S - 24), radius=200, fill=(244, 246, 255, 255))

    # the blob: a grid of square dots, navy core to pale rim, with pale "sparkle" dots inside
    cx, cy, R = S * 0.5, S * 0.55, S * 0.33
    step = S / 26
    for j in range(27):
        for i in range(27):
            x, y = (i + 0.5) * step, (j + 0.5) * step
            dx, dy = (x - cx) / R, (y - cy) / (R * 0.92)
            a = math.atan2(dy, dx)
            r = math.hypot(dx, dy) + 0.07 * math.sin(2 * a + 2.2) + 0.05 * math.sin(5 * a + 0.4)
            if r > 1.0:
                # sparse halftone in the top-right corner, like the reel
                c = math.hypot(x - S * 0.92, y - S * 0.08) / (S * 0.42)
                if c < 1 and (i + j) % 2 == 0:
                    s = step * 0.32 * (1 - c)
                    if s > 3:
                        d.rectangle((x - s / 2, y - s / 2, x + s / 2, y + s / 2), fill=(12, 22, 120, 255))
                continue
            light = 1 - r
            sparkle = ((i * 7 + j * 13) % 5 == 0 and r < 0.8) or ((i + j * 3) % 4 == 0 and r > 0.78)
            if sparkle:
                col = (200, 222, 255, 255)
            elif r > 0.82:
                col = (77, 125, 255, 255)
            elif r > 0.55:
                col = (26, 47, 224, 255)
            else:
                col = (10, 15, 115, 255)
            s = step * (0.62 if not sparkle else 0.5)
            d.rectangle((x - step / 2, y - step / 2, x + step / 2, y + step / 2), fill=(10, 15, 115, 255)
                        if r < 0.9 else (26, 47, 224, 255))
            d.rectangle((x - s / 2, y - s / 2, x + s / 2, y + s / 2), fill=col)
            _ = light

    # the eye: a little orange-yellow dot cluster
    ex, ey = cx + R * 0.18, cy - R * 0.2
    for j in range(-3, 4):
        for i in range(-3, 4):
            rr = math.hypot(i, j)
            if rr > 3.2:
                continue
            col = (255, 230, 80, 255) if rr < 1.2 else (255, 140, 30, 255) if rr < 2.2 else (232, 35, 26, 255)
            x, y = ex + i * step * 0.62, ey + j * step * 0.62
            s = step * 0.5
            d.rectangle((x - s / 2, y - s / 2, x + s / 2, y + s / 2), fill=col)

    # corner brackets around the blob
    ink, w, L = (24, 26, 36, 255), 16, S * 0.13
    x0, y0, x1, y1 = S * 0.14, S * 0.2, S * 0.86, S * 0.9
    for px, py, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(px, py + sy * L), (px, py), (px + sx * L, py)], fill=ink, width=w, joint="curve")

    # thin link line and the red tracker tag
    d.line([(S * 0.36, S * 0.2), (ex, ey)], fill=(255, 140, 150, 255), width=8)
    tx0, ty0, tx1, ty1 = S * 0.17, S * 0.12, S * 0.6, S * 0.27
    d.rectangle((tx0, ty0, tx1, ty1), fill=(232, 34, 43, 255))
    try:
        f = ImageFont.truetype(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "consolab.ttf"), 118)
        d.text(((tx0 + tx1) / 2, (ty0 + ty1) / 2 + 4), "#3511", font=f, fill=(255, 255, 255, 255), anchor="mm")
    except OSError:
        pass

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((24, 24, S - 24, S - 24), radius=200, fill=255)
    img.paste(tile, (0, 0), mask)
    return img.resize((512, 512), Image.LANCZOS)


if __name__ == "__main__":
    im = draw()
    im.save(os.path.join(HERE, "icon.png"))
    im.save(os.path.join(HERE, "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128),
                                                     (256, 256)])
    print("icon.png and icon.ico written")
