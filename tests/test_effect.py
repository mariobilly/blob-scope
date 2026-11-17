"""Engine tests: dots, palettes, tracker HUD, motion, glitch, frame shapes, determinism.

Run from the app folder: python -m unittest discover -s tests
"""
import os
import sys
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import effect as E  # noqa: E402


def subject(w=360, h=480, colour=(30, 50, 220)):
    """A blue blob with an orange 'eye' and some texture on white paper."""
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 20, 0, 360, colour, -1)
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 7)), (180, 200, 255), -1)
    cv2.circle(img, (w // 2 + 20, h // 2 - 30), 18, (255, 120, 20), -1)
    cv2.circle(img, (w // 2 + 20, h // 2 - 30), 7, (255, 240, 90), -1)
    return img


def no_hud(**k):
    return {"hud": False, "ornament": "off", "glitch": 0, "motion": "off", **k}


class Params(unittest.TestCase):
    def test_clamps_and_defaults(self):
        p = E.params({"cols": 9999, "dot": -1, "palette": "nope", "frame": "weird", "tags": 99, "motion": "x",
                      "hud": "false", "unknown": 1})
        self.assertEqual(p["cols"], 400)
        self.assertEqual(p["dot"], 0.15)
        self.assertEqual(p["palette"], "source")
        self.assertEqual(p["frame"], "source")
        self.assertEqual(p["tags"], 30)
        self.assertEqual(p["motion"], "off")
        self.assertFalse(p["hud"])
        self.assertNotIn("unknown", p)

    def test_presets_are_valid(self):
        ids = [x["id"] for x in E.PRESETS]
        self.assertEqual(len(ids), len(set(ids)))
        for pr in E.PRESETS:
            for k in pr["p"]:
                self.assertIn(k, E.DEFAULTS, f"{pr['id']}: unknown key {k}")


class Dots(unittest.TestCase):
    def test_size_and_grid(self):
        img = subject()
        out = E.render(img, no_hud(), 0)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        # paper stays paper
        self.assertTrue((out[:10, :10] > 245).all())

    def test_dot_pitch_matches_columns(self):
        img = np.zeros((480, 640, 3), np.uint8)
        img[:] = (40, 70, 230)  # flat colour + no dither spread: every dot alike, so the grid shows cleanly
        out = E.render(img, no_hud(cols=80, spread=0), 0).astype(np.float32)
        g = out[160:320, 200:440].mean(-1)
        spec = np.abs(np.fft.rfft(g - g.mean(), axis=1)).mean(0)
        k = np.argmax(spec[3:]) + 3
        period = g.shape[1] / k
        self.assertAlmostEqual(period, 640 / 80, delta=0.6)  # 8 px cells

    def test_blue_stays_blue_and_fire_stays_warm(self):
        img = subject()
        out = E.render(img, no_hud(), 0).astype(int)
        body = out[300:340, 120:160].reshape(-1, 3).mean(0)
        self.assertGreater(body[2], body[0] + 40)  # blue dominates the body
        eye = out[200:220, 190:210].reshape(-1, 3).mean(0)
        self.assertGreater(eye[0], eye[2] + 40)  # the eye is still warm

    def test_light_dots_on_dark_and_dark_dots_on_light(self):
        """The signature look: a mid blue gets pale dots over a navy base."""
        img = np.zeros((200, 200, 3), np.uint8)
        img[:] = (40, 70, 230)
        out = E.render(img, no_hud(cols=40, dot=0.5), 0).astype(int)
        lum = out.mean(-1)
        self.assertGreater(lum.max() - lum.min(), 120)  # both light and dark dots
        self.assertGreater((lum > 180).mean(), 0.02)

    def test_every_palette_and_shape(self):
        img = subject(240, 320)
        for pal in ["source"] + list(E.PALETTES):
            for shape in E.SHAPES:
                out = E.render(img, no_hud(palette=pal, shape=shape), 0)
                self.assertEqual(out.shape, img.shape)
        ramp = (E._RAMPS["ink"] * 255).astype(int)
        out = E.render(img, no_hud(palette="ink", warm=False, grid=0), 0)
        # ink: only black and white dots, plus grey base in between
        self.assertLessEqual(len(np.unique(out.reshape(-1, 3), axis=0)), 300)
        self.assertEqual(ramp.shape, (2, 3))

    def test_scales_with_size(self):
        """A preview and the full-size export look alike: same grid, same mean colour."""
        img = subject(720, 960)
        small = cv2.resize(img, (360, 480), interpolation=cv2.INTER_AREA)
        a = E.render(img, no_hud(), 0)
        b = E.render(small, no_hud(), 0)
        a2 = cv2.resize(a, (36, 48), interpolation=cv2.INTER_AREA).astype(int)
        b2 = cv2.resize(b, (36, 48), interpolation=cv2.INTER_AREA).astype(int)
        self.assertLess(np.abs(a2 - b2).mean(), 14)

    def test_tracker_only_keeps_picture(self):
        img = subject()
        out = E.render(img, {"dots": False, "hud": False, "ornament": "off", "glitch": 0, "motion": "off"}, 0)
        self.assertTrue(np.array_equal(out, img))


class Tracker(unittest.TestCase):
    def test_analysis_finds_subject(self):
        img = subject()
        info = E.analyse(img, E.params())
        self.assertIsNotNone(info["main"])
        self.assertGreater(info["main"]["pct"], 10)
        self.assertLess(info["main"]["pct"], 60)
        x0, y0, x1, y1 = info["main"]["box"]
        self.assertTrue(0.05 < x0 < 0.35 and 0.65 < x1 < 0.95)
        self.assertGreater(len(info["feats"]), 5)

    def test_tag_schedule_count_and_loop(self):
        p = E.params({"tags": 7, "life": 1.1})
        counts = [len(E.visible_tags(p, t / 10, 6.0)) for t in range(60)]
        self.assertGreater(np.mean(counts), 4)
        self.assertLess(np.mean(counts), 11)
        for t in (0.0, 1.3, 4.9):  # periodic
            a, b = E.visible_tags(p, t, 6.0), E.visible_tags(p, t + 6.0, 6.0)
            self.assertEqual([i for i, _ in a], [i for i, _ in b])
            for (_, x), (_, y) in zip(a, b):
                self.assertAlmostEqual(x, y, places=6)

    def test_ids_climb_in_video(self):
        p = E.params({"tags": 7, "life": 1.1})
        early = max(i for i, _ in E.visible_tags(p, 1.0, None))
        late = max(i for i, _ in E.visible_tags(p, 20.0, None))
        self.assertGreater(late, early + 80)  # about 6 new tags a second

    def test_hud_draws_red_tags(self):
        img = subject()
        base = E.render(img, {"hud": False, "glitch": 0, "motion": "off"}, 1.0)
        out = E.render(img, {"glitch": 0, "motion": "off"}, 1.0)
        diff = np.abs(out.astype(int) - base.astype(int)).sum(-1) > 30
        self.assertGreater(diff.mean(), 0.002)
        o = out.astype(int)
        red = (o[..., 0] > 200) & (o[..., 1] < 70) & (o[..., 2] < 80)
        self.assertGreater(red.sum(), 50)

    def test_tags_land_on_the_subject(self):
        img = subject()
        p = E.params({"tags": 10})
        info = E.analyse(img, p)
        tags = E.place_tags(p, 2.0, 6.0, info["feats"], img.shape[0] / img.shape[1], info["gray"])
        self.assertGreater(len(tags), 3)
        for tid, x, y, pct, age in tags:
            px = img[int(y * img.shape[0]), int(x * img.shape[1])]
            # on or right next to the blob, not out on the paper
            near = img[max(0, int(y * img.shape[0]) - 8):int(y * img.shape[0]) + 8,
                       max(0, int(x * img.shape[1]) - 8):int(x * img.shape[1]) + 8]
            self.assertTrue((near.min(axis=(0, 1)) < 240).any(), (tid, x, y, px))
            self.assertGreaterEqual(tid, p["id_base"] + 150)
            self.assertLess(pct, 5)

    def test_no_subject_no_crash(self):
        flat = np.full((240, 320, 3), 128, np.uint8)
        out = E.render(flat, {}, 0.7)
        self.assertEqual(out.shape, flat.shape)
        noise = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        self.assertEqual(E.render(noise, {}, 0.7).shape, noise.shape)


class MotionAndTime(unittest.TestCase):
    def test_deterministic(self):
        img = subject()
        a = E.render(img, {}, 2.37)
        b = E.render(img, {}, 2.37)
        self.assertTrue(np.array_equal(a, b))

    def test_seed_changes_tags(self):
        img = subject()
        a = E.render(img, {"motion": "off", "glitch": 0, "seed": 1}, 2.0)
        b = E.render(img, {"motion": "off", "glitch": 0, "seed": 2}, 2.0)
        self.assertFalse(np.array_equal(a, b))

    def test_every_motion_moves_and_loops(self):
        img = subject()
        for m in E.MOTIONS[1:]:
            p = {"motion": m, "hud": False, "glitch": 0, "ornament": "off", "cycle": 4.0}
            a, b = E.render(img, p, 0.0), E.render(img, p, 1.0)
            self.assertGreater(np.abs(a.astype(int) - b.astype(int)).mean(), 1.0, m)
            end = E.render(img, p, 4.0)  # one cycle later: the same frame (seamless loop)
            self.assertLess(np.abs(a.astype(int) - end.astype(int)).mean(), 0.5, m)

    def test_strength_zero_is_still(self):
        img = subject()
        p = {"motion": "drift", "motion_amt": 0, "hud": False, "glitch": 0}
        self.assertTrue(np.array_equal(E.render(img, p, 0.0), E.render(img, p, 1.7)))

    def test_glitch_bursts(self):
        img = subject()
        p = E.params({"glitch": 1.0, "hud": False, "motion": "off"})
        clean = E.render(img, {**p, "glitch": 0}, 0)
        hits = sum(not np.array_equal(E.render(img, p, k / 12 + 0.01), clean) for k in range(72))
        self.assertGreater(hits, 5)
        self.assertLess(hits, 50)


class Frames(unittest.TestCase):
    def test_shapes(self):
        img = subject(1000, 700)
        for fr, asp in (("1:1", 1.0), ("4:5", 0.8), ("9:16", 9 / 16), ("16:9", 16 / 9)):
            out = E.render(img, {"frame": fr}, 0)
            self.assertAlmostEqual(out.shape[1] / out.shape[0], asp, delta=0.02, msg=fr)
            self.assertEqual(out.shape[0] % 2 + out.shape[1] % 2, 0)

    def test_letterbox_frame(self):
        img = subject(720, 1280)
        out = E.render(img, {"frame": "reel"}, 0)
        self.assertEqual(out.shape[:2], (1280, 720))
        self.assertEqual(E.out_size(720, 1280, E.params({"frame": "reel"})), (720, 1280))
        self.assertTrue((out[:100] == 0).all() and (out[-100:] == 0).all())  # black bars above and below
        self.assertGreater(out[640].mean(), 20)

    def test_paper_cutout(self):
        img = subject()
        img[:] = (img * 0.3).astype(np.uint8)  # dark background
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(mask, (180, 240), (120, 160), 20, 0, 360, 255, -1)
        out = E.render(img, no_hud(paper=True), 0, mask=mask)
        self.assertTrue((out[:15, :15] > 240).all())  # the background became paper


if __name__ == "__main__":
    unittest.main()
