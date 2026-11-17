# Blob Scope

A blob-tracking dither look for your photos and videos. The picture becomes a grid of small dots in its own colours:
pale dots sparkle on navy, navy specks sit on white paper, and fire stays warm. A tracker HUD runs on top:
- red ID tags (`#3693 0.8%`) on small blobs, with IDs that keep counting up;
- pink links between the tags, with pair labels (`3687-3693`);
- the main-blob tag (`#3511 28.7%`) and corner brackets around the subject;
- a halftone dot corner and glitch slices.

**Everything runs on your computer. No account, no API key, nothing is uploaded.** The two cut-out models used by
**Cut out to paper** download once, for free, the first time you turn it on.

Inspired by the blob-tracking dither look in a reel by [Ishan Upadhyay](https://www.instagram.com/reel/DeFDxdVTXCz/).
Blob Scope is an independent project and shares no code with it.

## Install

You need:
- Windows 10 or 11 (it is built and tested on Windows 11)
- [Python](https://www.python.org/downloads/) 3.11 or newer
- [ffmpeg](https://ffmpeg.org/download.html) on your PATH (ffmpeg and ffprobe), for videos and MP4 export
- Google Chrome or Microsoft Edge: the app opens in its own window
- Optional: an NVIDIA GPU, for a fast paper cut-out. Everything else runs on the CPU.

Install [PyTorch](https://pytorch.org/get-started/locally/) first (with CUDA if you have an NVIDIA GPU). Then:

```
python -m pip install -r requirements.txt
python launch.pyw
```

To run ISNet on the GPU too, install `onnxruntime-gpu` instead of `onnxruntime`.

The first **Cut out to paper** downloads the YOLO11 segmentation model (`yolo11m-seg.pt`, 45 MB) from the
[Ultralytics releases](https://github.com/ultralytics/assets/releases) into `models\`, and rembg downloads its ISNet
model (180 MB) into your user folder.

Optional: `powershell -ExecutionPolicy Bypass -File make_shortcut.ps1` puts a **Blob Scope** shortcut on your Desktop.
You can drop a photo or video onto it, or into the window.

## How it works
1. **Open** a photo (PNG, JPG, WebP, TIFF, BMP) or a video (MP4, MOV, WebM, MKV, AVI, GIF).
2. Pick a **preset**. The cards show each one on your own picture.

   | Preset | Look |
   |---|---|
   | Blob Tracker | Own colours in dots, red tracker tags. |
   | Cobalt Scan | Any picture in cobalt blues; fire stays warm. |
   | Paper Specimen | The subject cut out onto white paper. |
   | Fine Grain | Small, dense dots and calm tags. |
   | Chunky LED | Big round dots and a busy tracker. |
   | Ember Signal | A fire ramp with dark tags. |
   | Night Scope | A dark cyan scope. |
   | Riso Pop | Pink and yellow print dots. |
   | Ink Tracker | Black and white dots, red tags. |
   | Tracker Only | Your picture untouched, with just the HUD. |
3. Fine-tune in the right panel. Double-click a slider to reset it.
   - **Dots**:
     - colours: own colours, or a palette (Cobalt, Ink, Ember, Acid, Riso, Ultraviolet, Night);
     - columns (160 by default);
     - dot size;
     - shape: square, round or plus;
     - Bayer pattern and dither strength;
     - levels (own colours only);
     - "Keep fire warm" (palettes only);
     - panel gaps (an LED-panel look).
   - **Tone**: contrast, brightness, saturation, hue shift.
   - **Tracker**:
     - on/off;
     - how many tags are on screen, and how long each one lives;
     - tag size;
     - links, pair labels, the main tag and brackets;
     - tag and link colours;
     - the first ID number.
   - **Frame**:
     - shape: original, **Letterbox** (3:4 art on a black 9:16 canvas), 9:16, 4:5, 1:1 or 16:9;
     - **Cut out to paper**, which puts the subject on a paper colour (YOLO11-seg for people, ISNet for anything
       else, on the GPU when there is one);
     - corner dots.
   - **Motion**:
     - camera: drift, sway, shake or pulse, with a strength setting;
     - loop / cycle: the loop length of a photo, or the camera cycle of a video;
     - glitch;
     - **seed**: different tag positions, IDs and glitch timing. The same seed always gives the same result.
4. **▶** (or **P**) plays it in the preview, and the bar scrubs through it. Hold **Space** to see the original.
   **R** randomizes, **Ctrl+O** opens a file, **Ctrl+E** exports.
5. **Export**:
   - **Photos** are saved as a seamless MP4 loop (one cycle long, 24/25/30/60 fps) or as a PNG still.
   - **Videos** are saved as H.264 MP4 with the original audio. Every frame is tracked on its own, and tag IDs keep
     climbing through the clip.
   - Size: Max 1920 px (default), Max 1280 px, or full size.
   - Files go to `Videos\Blob Scope` in your user folder. Set the `BLOBSCOPE_OUT` environment variable to use another
     folder. Existing files are never overwritten.
   - Speed: on a fast PC with an NVIDIA GPU, a 6 s 1080×1920 loop takes about 6 s. With the paper cut-out it takes
     about 15 s.

## What's where
- `effect.py`: the engine (numpy + OpenCV + Pillow). It handles the dots, the tracker analysis, the tag schedule,
  the HUD, motion, glitch and frame shapes.
  - A frame depends only on (picture, settings, time), so the preview is exactly the export.
  - Sizes scale with the picture, so a 640 px preview and a 1920 px export match.
- `matte.py`: the paper cut-out. It runs in its own worker process (`python matte.py --serve`).
  - Once torch has run inside a process, every CPU render in that process slows down, so the cut-out models stay
    out of the server.
  - The YOLO weights live in `models\`. The first cut-out downloads them once; if that fails, every picture goes
    through ISNet.
- `media.py`: ffmpeg and ffprobe plumbing:
  - probe;
  - image read and write that works with accented file names;
  - a seekable frame pipe, which makes video previews smooth;
  - the MP4 writer, which keeps the source audio and checks the result with ffprobe.

  It also has `no_throttle()`. Windows slows busy processes that have no window of their own 8–10× after a few
  seconds, so the server, the matte worker and every ffmpeg pipe opt out of that.
- `app.py`: a FastAPI server on `127.0.0.1:47871`. It quits about 20 s after the window closes, unless an export
  is still running.
- `launch.pyw`: the launcher. It opens a Chrome or Edge app window.
- `static/`: the window (HTML, CSS, JS).
- `make_icon.py`, `make_shortcut.ps1`: these rebuild the icon and the Desktop shortcut.
- Your data: `%LOCALAPPDATA%\BlobScope\`: dropped files (`uploads\`), `app.log` and `matte.log`.

## Tests
```
python -m unittest discover -s tests
```
The server tests use FastAPI's TestClient, which needs httpx (`python -m pip install httpx`). There are 32 tests:
- **Engine:**
  - parameter clamping and the presets;
  - dot pitch, and that blue stays blue while fire stays warm;
  - light dots on dark areas and dark dots on light ones;
  - every palette and shape;
  - preview and export scaling;
  - finding the subject, the tag schedule (count, loop, IDs that climb), and that tags land on the subject;
  - determinism and seeds;
  - that every motion moves and loops seamlessly;
  - glitch bursts, frame shapes, the letterbox and the paper cut-out.
- **Server:**
  - photo previews, thumbnails, PNG stills and MP4 loops;
  - video preview streaming, checked against a direct seek;
  - video export on a synthetic clip with audio and an accented file name, checked for frames, size, audio and
    duration;
  - export sizes and cancel;
  - the real cut-out models (downloaded the first time), checking that the server keeps OpenCV multi-threaded and
    never loads torch;
  - the power-throttling opt-out.

## Credits
- Inspiration: the blob-tracking dither look in a reel by [Ishan Upadhyay](https://www.instagram.com/reel/DeFDxdVTXCz/).
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for the people cut-out. Ultralytics and its models
  are licensed under **AGPL-3.0**: check their terms if you plan to use Blob Scope commercially.
- [rembg](https://github.com/danielgatis/rembg) (MIT) with the ISNet model, for everything else.

## Licence

Blob Scope's own code is MIT, see [LICENSE](LICENSE). The libraries and models above keep their own licences.
