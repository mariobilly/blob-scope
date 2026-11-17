"""Server + ffmpeg pipeline tests on a synthetic clip with audio and accented file names.

Run from the app folder: python -m unittest discover -s tests
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="blobscope-test-")
os.environ["BLOBSCOPE_DATA"] = os.path.join(TMP, "data")
os.environ["BLOBSCOPE_OUT"] = os.path.join(TMP, "out")
os.environ["BLOBSCOPE_STAY"] = "1"
sys.path.insert(0, HERE)

from fastapi.testclient import TestClient  # noqa: E402

import app as A  # noqa: E402
import effect as E  # noqa: E402
import media as M  # noqa: E402

CLIP = os.path.join(TMP, "Clipé ü test.mp4")
PHOTO = os.path.join(TMP, "Fotó 写真.png")


def setUpModule():
    subprocess.run([M.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=240x320:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "1.6", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", CLIP], check=True)
    img = np.full((360, 270, 3), 255, np.uint8)
    cv2.ellipse(img, (135, 190), (90, 120), 15, 0, 360, (30, 50, 220), -1)
    cv2.circle(img, (150, 150), 14, (255, 130, 20), -1)
    M.write_image(PHOTO, img)


def tearDownModule():
    A.close_stream()  # the preview's ffmpeg pipe and the matte worker hold files in TMP open
    A.MATTE.close()
    shutil.rmtree(TMP, ignore_errors=True)


def decode_jpeg(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def wait_job(c, timeout=180):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get("/api/job").json()
        if j["state"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("export did not finish")


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(A.app)

    def test_ping_and_state(self):
        self.assertEqual(self.c.get("/api/ping").json()["app"], "blob-scope")
        st = self.c.get("/api/state").json()
        self.assertIn("cobalt", st["palettes"])
        self.assertEqual(len(st["presets"]), len(E.PRESETS))
        self.assertEqual(st["out_dir"], os.environ["BLOBSCOPE_OUT"])
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)
        self.assertEqual(self.c.get("/static/../app.py").status_code, 404)
        self.assertEqual(self.c.get("/favicon.ico").status_code, 200)

    def test_unsupported_file(self):
        bad = os.path.join(TMP, "notes.txt")
        with open(bad, "w") as f:
            f.write("hi")
        r = self.c.post("/api/load", json={"path": bad})
        self.assertEqual(r.status_code, 400)
        self.assertIn("opens photos", r.json()["detail"])

    def test_photo_preview_thumbs_and_still(self):
        m = self.c.post("/api/load", json={"path": PHOTO}).json()["media"]
        self.assertEqual((m["kind"], m["width"], m["height"]), ("image", 270, 360))
        r = self.c.post("/api/preview", json={"side": 360, "t": 1.0, "params": {}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(decode_jpeg(r.content).shape, (360, 270, 3))
        reel = decode_jpeg(self.c.post("/api/preview", json={"side": 360, "params": {"frame": "reel"}}).content)
        self.assertAlmostEqual(reel.shape[0] / reel.shape[1], 16 / 9, delta=0.02)
        orig = decode_jpeg(self.c.post("/api/preview", json={"side": 360, "original": True}).content)
        self.assertEqual(orig.shape, (360, 270, 3))
        for pr in E.PRESETS:
            if pr["p"].get("paper"):
                continue  # the cut-out needs the matte models: covered by test_paper_preview
            r = self.c.post("/api/thumb", json={"preset": pr["id"], "side": 200, "keep": {"frame": "1:1"}})
            self.assertEqual(r.status_code, 200, pr["id"])
            self.assertEqual(decode_jpeg(r.content).shape[:2], (150, 150))
        self.assertEqual(self.c.post("/api/thumb", json={"preset": "nope"}).status_code, 404)
        j = self.c.post("/api/export", json={"params": {}, "photo_out": "still"}).json()
        self.assertEqual(j["kind"], "still")
        j = wait_job(self.c)
        self.assertEqual(j["state"], "done", j)
        self.assertTrue(j["out"].endswith(".png") and "Blob Scope" in j["out"])
        self.assertEqual(M.read_image(j["out"]).shape, (360, 270, 3))

    def test_photo_loop_export(self):
        self.c.post("/api/load", json={"path": PHOTO})
        j = self.c.post("/api/export", json={"params": {"cycle": 2.0, "frame": "1:1"}, "fps": 25}).json()
        self.assertEqual((j["kind"], j["total"]), ("loop", 50))
        j = wait_job(self.c)
        self.assertEqual(j["state"], "done", j)
        rep = j["report"]
        self.assertEqual(rep["problems"], [])
        self.assertEqual((rep["width"], rep["height"], rep["frames"], rep["audio"]), (270, 270, 50, False))

    def test_video_preview_stream_and_export(self):
        m = self.c.post("/api/load", json={"path": CLIP}).json()["media"]
        self.assertEqual((m["kind"], m["frames"], m["has_audio"]), ("video", 40, True))
        shots = []
        for f in (0, 1, 2, 3, 30, 4):  # forward reads from the open pipe, the jump restarts it
            r = self.c.post("/api/preview", json={"frame": f, "t": f / 25, "side": 320, "params": {}})
            self.assertEqual(r.status_code, 200)
            shots.append(decode_jpeg(r.content))
        self.assertEqual(shots[0].shape, (320, 240, 3))
        self.assertFalse(np.array_equal(shots[0], shots[4]))
        # the streamed frame equals a direct seek of the same frame
        a = A.source(4, 320)
        b = M.read_frame(CLIP, A.S["media"]["info"], 4, 240, 320)
        self.assertLess(np.abs(a.astype(int) - b.astype(int)).mean(), 2)
        j = self.c.post("/api/export", json={"params": {"frame": "reel", "glitch": 0.5}, "max_side": 0}).json()
        self.assertEqual((j["kind"], j["total"]), ("video", 40))
        j = wait_job(self.c)
        self.assertEqual(j["state"], "done", j)
        rep = j["report"]
        self.assertEqual(rep["problems"], [])
        self.assertTrue(rep["audio"])
        self.assertEqual((rep["width"], rep["height"], rep["frames"]), (240, 426, 40))
        self.assertAlmostEqual(rep["duration"], 1.6, delta=0.1)

    def test_power_throttling_opt_out(self):
        self.assertTrue(M.no_throttle())
        p = subprocess.Popen([M.ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "nullsrc", "-t", "3", "-f", "null",
                              "-"], creationflags=M.NO_WINDOW)
        try:
            self.assertTrue(M.no_throttle(p.pid))
        finally:
            p.kill()
            p.wait()

    def test_export_sizes(self):
        p = E.params({"frame": "reel"})
        self.assertEqual(A.export_dims(2160, 3240, p, 1920), (1080, 1620, 1080, 1920))
        p = E.params({})
        self.assertEqual(A.export_dims(3840, 2160, p, 1280), (1280, 720, 1280, 720))
        self.assertEqual(A.export_dims(640, 360, p, 1920), (640, 360, 640, 360))  # never upscales

    def test_cancel(self):
        self.c.post("/api/load", json={"path": PHOTO})
        self.c.post("/api/export", json={"params": {"cycle": 20.0}, "fps": 60})
        time.sleep(0.3)
        self.c.post("/api/cancel", json={})
        j = wait_job(self.c)
        self.assertIn(j["state"], ("cancelled", "done"))
        if j["state"] == "cancelled":
            left = [f for f in os.listdir(os.environ["BLOBSCOPE_OUT"]) if ".part" in f]
            self.assertEqual(left, [])

    def test_paper_preview(self):
        """The real cut-out models (YOLO for people, ISNet otherwise) on the photo."""
        self.c.post("/api/load", json={"path": PHOTO})
        r = self.c.post("/api/preview", json={"side": 360, "params": {"paper": True, "hud": False,
                                                                       "ornament": "off", "motion": "off"}})
        self.assertEqual(r.status_code, 200, r.text[:300])
        out = decode_jpeg(r.content)
        self.assertEqual(out.shape, (360, 270, 3))
        self.assertGreater(out[:20, :20].mean(), 235)  # the white background stays paper
        self.assertGreater(cv2.getNumThreads(), 1)  # the matte models must not leave OpenCV single-threaded
        self.assertNotIn("torch", sys.modules)  # torch + CUDA stay in the worker process (they slow renders 5x)


if __name__ == "__main__":
    unittest.main()
