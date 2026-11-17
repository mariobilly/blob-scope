"""Subject matte for the paper cut-out: YOLO11-seg when there are people, ISNet (rembg) for anything else.

Both run on the GPU when one is available. Importing torch first loads the CUDA / cuDNN libraries that
onnxruntime's CUDA provider needs; without it rembg falls back to the CPU (about 8 s a frame instead of 0.15 s).
The YOLO weights live in this app's models folder. The first cut-out downloads them once from the Ultralytics
releases on GitHub; if that fails, every picture goes through ISNet. rembg downloads its ISNet model once by itself.

The app runs all of this in a separate worker process (Worker below, `python matte.py --serve`): once torch and
CUDA have run inside a process, every CPU render in it gets about 5x slower, so they must not share the server.
"""
import atexit
import json
import os
import struct
import subprocess
import sys
import threading
import urllib.request

import cv2
import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")  # never let Ultralytics fetch anything by itself

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
YOLO_NAMES = ("yolo11m-seg.pt", "yolo11s-seg.pt")
YOLO_RELEASES = ("v8.3.0", "v8.4.0")  # Ultralytics release tags that hold the YOLO11 weights
MAX_SIDE = 1024
_LOCK = threading.Lock()
_S = {}


def _cuda():
    if "cuda" not in _S:
        try:
            import torch
            _S["cuda"] = bool(torch.cuda.is_available())
        except Exception:
            _S["cuda"] = False
    return _S["cuda"]


def yolo_path():
    """The YOLO11-seg weights in the models folder, downloaded once when they are not there yet (None if that fails)."""
    for n in YOLO_NAMES:
        p = os.path.join(MODEL_DIR, n)
        if os.path.isfile(p) and os.path.getsize(p) > 1_000_000:
            return p
    try:
        return _download(YOLO_NAMES[0])
    except Exception as e:
        print("matte: could not download the YOLO weights, using ISNet only:", e, flush=True)
        return None


def _download(name):
    """Fetch an Ultralytics model file from its GitHub releases into the models folder."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, name)
    part = dest + ".part"
    last = None
    for tag in YOLO_RELEASES:
        url = f"https://github.com/ultralytics/assets/releases/download/{tag}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(part) <= 1_000_000:
                raise ValueError("the download is too small")
            os.replace(part, dest)
            print("matte: downloaded", name, "from", url, flush=True)
            return dest
        except Exception as e:  # try the next release tag
            last = e
            if os.path.exists(part):
                os.remove(part)
    raise RuntimeError(f"{name}: {last}")


def _yolo():
    if "yolo" not in _S:
        model, path = None, yolo_path()
        if path:
            try:
                from ultralytics import YOLO
                model = YOLO(path)
            except Exception as e:
                print("matte: YOLO unavailable:", e, flush=True)
        _S["yolo"] = model
    return _S["yolo"]


def _isnet():
    if "isnet" not in _S:
        _cuda()
        from rembg import new_session
        _S["isnet"] = new_session("isnet-general-use", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _S["isnet"]


def _people(img):
    """Union of the person masks, or None when there is no clear person."""
    model = _yolo()
    if model is None:
        return None
    r = model.predict(img, classes=[0], conf=0.35, retina_masks=True, verbose=False,
                      device=0 if _cuda() else "cpu")[0]
    if r.masks is None or len(r.masks) == 0:
        return None
    m = r.masks.data.max(0).values.float().cpu().numpy()
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if (m > 0.5).mean() < 0.02:
        return None
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, img.shape[1] / 600))
    return (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)


def _general(img):
    from PIL import Image
    from rembg import remove
    a = np.asarray(remove(Image.fromarray(img), session=_isnet(), only_mask=True)).astype(np.float32) / 255
    a = np.clip((a - 0.12) / 0.55, 0, 1)  # firm up the half-transparent body that ISNet leaves on busy photos
    return (a * 255 + 0.5).astype(np.uint8)


def subject(img):
    """HxW uint8 matte for an RGB uint8 picture: 255 = subject, 0 = background (in this process)."""
    threads = cv2.getNumThreads()
    try:
        return _subject(img)
    finally:
        # Ultralytics switches OpenCV to a single thread on import; the dot renderer would run ~5x slower
        cv2.setNumThreads(threads)


def _subject(img):
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) \
        if s < 1 else img
    with _LOCK:
        try:
            m = _people(small)
        except Exception as e:
            print("matte: person pass failed:", e, flush=True)
            m = None
        if m is None:
            m = _general(small)
    return m if m.shape == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- worker process

def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        b = f.read(n - len(buf))
        if not b:
            raise EOFError("matte worker pipe closed")
        buf += b
    return bytes(buf)


def _send(f, head, payload=b""):
    h = json.dumps(head).encode()
    f.write(struct.pack("<II", len(h), len(payload)) + h + payload)
    f.flush()


def _recv(f):
    hl, pl = struct.unpack("<II", _read_exact(f, 8))
    return json.loads(_read_exact(f, hl)), _read_exact(f, pl) if pl else b""


def serve():
    """Worker loop: RGB frames in on stdin, mattes out on stdout. Library chatter goes to stderr."""
    try:
        import media
        media.no_throttle()
    except Exception:
        pass
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything printed to stdout by torch / Ultralytics lands in the log, not in the protocol
    sys.stdout = sys.stderr
    inp = sys.stdin.buffer
    while True:
        try:
            head, data = _recv(inp)
        except EOFError:
            return
        try:
            img = np.frombuffer(data, np.uint8).reshape(head["h"], head["w"], 3)
            m = subject(img)
            _send(out, {"ok": True, "h": m.shape[0], "w": m.shape[1]}, m.tobytes())
        except Exception as e:
            _send(out, {"ok": False, "error": str(e)})


class Worker:
    """The matte models in their own process, started on first use, restarted if it dies."""

    def __init__(self, log_path=None):
        self.p, self.log_path, self.lock = None, log_path, threading.Lock()
        atexit.register(self.close)

    def _start(self):
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        try:
            self.p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=log, cwd=APP_DIR,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        finally:
            if log is not subprocess.DEVNULL:
                log.close()  # the worker has its own handle; keeping ours open leaked one per (re)start

    def subject(self, img):
        img = np.ascontiguousarray(img, np.uint8)
        with self.lock:
            for attempt in (0, 1):
                if self.p is None or self.p.poll() is not None:
                    self._start()
                try:
                    _send(self.p.stdin, {"h": img.shape[0], "w": img.shape[1]}, img.tobytes())
                    head, data = _recv(self.p.stdout)
                    break
                except (OSError, EOFError):
                    self.close_locked()
                    if attempt:
                        raise RuntimeError("The cut-out worker stopped (see matte.log)")
        if not head.get("ok"):
            raise RuntimeError("Cut-out failed: " + head.get("error", "unknown error"))
        return np.frombuffer(data, np.uint8).reshape(head["h"], head["w"]).copy()

    def close_locked(self):
        if self.p is not None:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=3)
            except Exception:
                self.p.kill()
            try:
                self.p.stdout.close()  # the reply pipe was left open before
            except Exception:
                pass
            self.p = None

    def close(self):
        with self.lock:
            self.close_locked()


if __name__ == "__main__" and "--serve" in sys.argv:
    serve()
