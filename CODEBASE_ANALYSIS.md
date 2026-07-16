# Deep-Live-Cam — Full Codebase Analysis

Version in repo: **2.1.6** (`modules/metadata.py` reports internal version `2.1.5`, "GitHub Edition"). Forked/derived from **s0md3v/roop**.

This document explains **what the project is, how it is built, which techniques/models it uses, how data flows through it end-to-end, and how to run it** — all in one place.

---

## 1. What this project is

Deep-Live-Cam is a **real-time face-swap (deepfake) application**. Given:
- one **source** image (a face you want to project), and
- a **target** — an image, a video file, or a **live webcam feed**,

it detects the face(s) in the target, replaces them with the source face, optionally restores/upscales the swapped face, and either saves the result (image/video) or streams it live (webcam mode, for use with OBS/Zoom/etc.).

It ships as a **desktop app** (PySide6/Qt GUI) and also runs **headless from the CLI**.

⚠️ The repo README contains an explicit ethical-use disclaimer (no non-consensual deepfakes, a built-in NSFW filter, legal compliance). That governs any real usage of this tool.

---

## 2. High-level architecture

```
                     ┌────────────────────┐
                     │   run.py (entry)   │
                     └─────────┬──────────┘
                                │
                     ┌──────────▼───────────┐
                     │  modules/core.py     │  argument parsing, execution-provider
                     │  (run/start/destroy) │  selection, orchestration of the whole
                     └───┬───────────────┬──┘  pipeline
                         │               │
             headless?   │               │  no → GUI
                    yes   │               │
                         ▼               ▼
                 start()            modules/ui.py (PySide6/Qt)
                (direct pipeline)    "Start" button → same start()
                                     "Live" button  → webcam preview window

        start() dispatches on target type:
        ┌───────────────────────┬───────────────────────┬────────────────────────┐
        │   image → image       │  video → video         │  webcam (live) mode    │
        └───────────────────────┴───────────────────────┴────────────────────────┘
                     │                       │                        │
                     ▼                       ▼                        ▼
        face_analyser (InsightFace)  in-memory FFmpeg pipe      VideoCapturer thread
        detects/recognises faces    OR disk-based frame         → capture queue
                     │              extraction fallback         → processing worker
                     ▼                       │                    (detect + swap + enhance)
        frame processors pipeline:           ▼                        │
        face_swapper → face_enhancer   frame processors pipeline      ▼
                     │              (same as image path, per frame) processed queue
                     ▼                       │                        │
              output image             FFmpeg re-encode           Qt QTimer polls
                                      + audio restore              → live preview
```

Everything funnels through the same **frame-processor interface** (`process_frame`), so image mode, video mode, and webcam mode all reuse identical face-detection and face-swap code — only the *frame source/sink* differs.

---

## 3. Core techniques & ML models used

| Purpose | Technique / Model | Where |
|---|---|---|
| Face **detection** + **recognition** (embeddings) + **landmarks** | **InsightFace** `buffalo_l` model pack (RetinaFace detector + ArcFace-style recognition + `landmark_2d_106`) | `modules/face_analyser.py` |
| Face **swapping** | **inswapper_128** (ONNX, from `insightface.model_zoo`) — a GAN-based identity-swap model that takes a target face + a source identity embedding and generates a new face | `modules/processors/frame/face_swapper.py` |
| Face **restoration / upscaling** | **GFPGAN** (ONNX export, `gfpgan-1024.onnx`), a StyleGAN2-based blind face-restoration network, run via raw ONNX Runtime (no PyTorch/gfpgan package needed) | `modules/processors/frame/face_enhancer.py`, `_onnx_enhancer.py` |
| Alternative lightweight enhancers | **GPEN** 256/512 variants | `face_enhancer_gpen256.py`, `face_enhancer_gpen512.py` |
| **NSFW / content-safety filter** | `opennsfw2` (Yahoo's open_nsfw model) — probability threshold `0.85` | `modules/predicter.py` |
| **Multi-face clustering** (auto-grouping "who's who" in a video for face-mapping) | **K-Means** clustering over normalized face embeddings, with elbow-method `k` selection (max inertia-drop heuristic) | `modules/cluster_analysis.py` |
| Model runtime | **ONNX Runtime** with pluggable **execution providers**: `CPU`, `CUDA` (NVIDIA), `CoreML` (Apple Silicon), `DirectML` (Windows/AMD/Intel), `ROCm`, `OpenVINO` | `modules/core.py`, model loaders |
| Video I/O | **FFmpeg** subprocess (both classic "extract-frames-then-encode" and a newer **zero-disk-I/O pipe pipeline**) | `modules/utilities.py`, `modules/processors/frame/core.py` |
| GUI | **PySide6 (Qt for Python)** — *not* Tkinter, despite legacy `tkinter_fix.py` left in the repo | `modules/ui.py` |
| Webcam capture | OpenCV `cv2.VideoCapture`, with Windows-specific `pygrabber`/DirectShow handling | `modules/video_capture.py` |

### 3.1 Face swap mechanics (the actual "how")
1. **Detect** faces in the frame → bounding boxes + 5-point keypoints (`fa.det_model.detect`).
2. **Recognize** → 512-d normalized embedding per face (ArcFace-style, `normed_embedding`) — this is the face's "identity vector".
3. **Landmarks** (106 points) are computed **only when needed** (mouth-masking or an enhancer is active) — an optimization to skip a whole ONNX model call when only swapping.
4. **Swap**: `insightface.model_zoo` `inswapper_128` model takes the target face's aligned crop + the source identity embedding and outputs a new face (`bgr_fake`) plus the affine transform `M` used to align it.
5. **Paste-back**: instead of insightface's built-in (slower) paste-back, the code has a custom `_fast_paste_back` + a cached **elliptical, Gaussian-feathered alpha mask** computed once in "aligned-face space" and warped per frame — turning an O(face-size²) blur into an O(crop-area) operation.
6. **Optional post-processing** on top of the raw swap:
   - **Mouth masking** (`create_lower_mouth_mask`/`apply_mouth_area`): cuts the original mouth back in from the pre-swap frame so lip movement/sync isn't destroyed by the swap.
   - **Poisson blending** (OpenCV `seamlessClone`) for smoother edge blending.
   - **Color transfer** to match skin tone.
   - **Opacity blending** (`gpu_add_weighted`) to fade the swap in/out.
   - **Sharpening** and **temporal interpolation** between consecutive frames (smooths flicker in live mode).
7. **Face enhancement** (separate pass, optional): re-aligns the swapped face to a 512×512 FFHQ template, runs it through GFPGAN for detail restoration, and pastes it back.

---

## 4. Module map (what each file does)

```
run.py                          → entry point. Beyond calling `modules.core.run()`, it does
                                   real setup work first: prepends the project root to PATH
                                   so bundled ffmpeg/ffprobe are found, and — because
                                   LD_LIBRARY_PATH/native DLL search paths can't be changed
                                   after the interpreter starts — explicitly registers CUDA/
                                   cuDNN DLL directories on Windows (`os.add_dll_directory`)
                                   and preloads NVIDIA shared libs on Linux (`ctypes.CDLL`,
                                   `RTLD_GLOBAL`) from pip-installed `nvidia-*`/`torch` wheels,
                                   so onnxruntime-gpu can find them when it dlopens its CUDA
                                   execution provider.

modules/
  core.py                       → arg parsing, execution-provider auto-detection,
                                   resource limiting, the `start()` orchestrator that
                                   drives image/video/webcam processing, SIGINT-safe cleanup
  globals.py                    → single shared "settings/state" module (paths, toggles,
                                   face maps, thread locks) — the de facto global config object
  face_analyser.py              → wraps InsightFace FaceAnalysis: detection, recognition,
                                   landmarks; face-mapping helpers (source↔target map,
                                   per-video clustering of unique faces)
  cluster_analysis.py           → K-Means clustering of face embeddings (for "map faces"
                                   auto-grouping across a video)
  predicter.py                  → NSFW probability check (opennsfw2) for image/video/frame
  video_capture.py              → cross-platform webcam capture wrapper (OpenCV + DirectShow
                                   quirks on Windows, empirical FPS measurement)
  capturer.py                   → grabs a single representative frame from a video (for
                                   thumbnails/previews)
  utilities.py                  → ffmpeg wrappers (extract frames, encode video, restore
                                   audio, detect fps/dimensions), temp-file management,
                                   model downloading (`conditional_download`)
  gpu_processing.py              → drop-in GPU-aware replacements for cv2 ops (blur, resize,
                                   color convert, weighted-add) via cv2.cuda GpuMat, with
                                   CPU fallback (disabled by default — see §6)
  onnx_optimize.py               → Apple-Silicon-specific ONNX graph rewrites so CoreML can
                                   run models end-to-end on the Neural Engine instead of
                                   bouncing between CPU/ANE (see §6.3)
  platform_info.py               → OS/arch detection helpers
  ui.py                          → PySide6 GUI: main window, source/target pickers, sliders
                                   (opacity, sharpness, mouth mask size…), face mapper
                                   dialogs, live webcam preview window
  ui_tooltip.py / tkinter_fix.py / gettext.py → UI helpers, legacy shims, i18n
  metadata.py                    → name/version string
  processors/frame/
    core.py                     → frame-processor plugin loader + the two video pipelines:
                                   disk-based (`process_video`) and the newer in-memory
                                   FFmpeg-pipe pipeline (`process_video_in_memory`)
    face_swapper.py (1571 lines) → the swap engine described in §3.1; also owns mouth-mask
                                   creation, face-mask creation, color transfer, CUDA-graph
                                   optimized inference
    face_enhancer.py             → GFPGAN-based restoration (ONNX Runtime session,
                                   FFHQ alignment template, pre/post-processing)
    face_enhancer_gpen256/512.py → lighter-weight alternative enhancers, fetched on demand
                                   from a GitHub release (harisreedhar/Face-Upscalers-ONNX)
    face_masking.py               → mouth/eyes/eyebrow mask geometry + visualization helpers —
                                   **dead code**: nothing in the codebase imports this module
                                   (verified via grep); the masking logic actually used at
                                   runtime is duplicated directly inside face_swapper.py
    _onnx_enhancer.py             → shared ONNX Runtime session/provider construction,
                                   IO-binding (keeps tensors on GPU, no host↔device copy)

tests/                           → unit tests (currently: face_analyser.get_one_face logic,
                                   using stubbed insightface/cv2/numpy modules)
models/                          → GFPGANv1.4.onnx, inswapper_128.onnx, inswapper_128_fp16.onnx
                                   (place these here manually; not committed except a couple)
requirements.txt / pyproject.toml→ dependencies; `pyproject.toml` only configures `ruff` lint
.github/workflows/ruff.yml       → CI: lints the code with ruff on push/PR
```

---

## 5. End-to-end data flow

### 5.1 Image → Image
1. `core.start()` sees `target_path` has an image extension.
2. Optional NSFW check on the target image (`predicter.predict_image`).
3. Copies target → output path, then runs each configured frame processor's `process_image(source_path, output_path, output_path)` **in sequence** (e.g. `face_swapper` then `face_enhancer`), each overwriting the same file.
4. Done — prints elapsed time.

### 5.2 Video → Video
`core.start()` first detects/assumes FPS, then tries **two pipelines**, in order:

**A. In-memory pipe pipeline (default, unless `--map-faces` is set)** — `process_video_in_memory()`:
- Pre-loads the source face embedding once.
- Spawns an **FFmpeg reader** subprocess that decodes the target video to raw `bgr24` frames on stdout, and an **FFmpeg writer** subprocess that encodes raw frames from stdin straight to the output container (`-movflags +faststart`).
- For each raw frame: detect the target face (with a **pipelined detector** — while frame *N* is being swapped, a background thread is already detecting the face in frame *N+1*, overlapping the two GPU-bound steps), run every frame processor's `process_frame`, then write bytes directly to the encoder's stdin.
- Automatically picks a **hardware encoder** when possible (`h264_nvenc`/`hevc_nvenc` for CUDA, `h264_amf`/`hevc_amf` for DirectML) and **falls back to software `libx264`** if the hardware encoder fails.
- Zero PNG files ever touch disk — this is the "major speed-up" path.

**B. Disk-based fallback** (used when `--map-faces` is on, since that needs the classic per-frame face map built from extracted frames, or if the pipe pipeline fails):
- `create_temp` + `extract_frames` (ffmpeg dumps PNGs to a temp dir).
- Each frame processor's `process_video(source_path, temp_frame_paths)` runs, internally using a `ThreadPoolExecutor` (`multi_process_frame`) to batch-parallelize frames across `execution_threads`.
- `create_video()` re-encodes the processed PNGs to a temp video.
- Audio is restored from the original (`restore_audio`) unless `--keep-audio` is off.
- Temp files are cleaned up (`clean_temp`), unless `--keep-frames`.

### 5.3 Live Webcam mode (GUI only)
1. User clicks **Live** in the UI → `_open_webcam_preview(camera_index)` creates a `WebcamPreviewWindow`.
2. That window starts a `VideoCapturer` (OpenCV, negotiates MJPG @ up to 960×540@60fps, measures *actual* FPS empirically since driver-reported FPS is unreliable) and **two background threads**:
   - `_CaptureWorker` — pulls raw frames from the camera into a small `capture_queue` (bounded, to avoid unbounded memory growth / latency buildup).
   - `_ProcessingWorker` — pulls raw frames, runs face detection + the frame-processor pipeline (swap/enhance), pushes results into a `processed_queue`.
3. A Qt `QTimer` in the main/UI thread polls the `processed_queue` at ~2× camera FPS and paints the latest processed frame into the preview `QLabel` — decoupling **capture rate**, **processing rate** (which can lag on slow hardware), and **display rate**.
4. The user points OBS/Zoom/etc. at this window (or a virtual camera) to stream the swapped face live.
5. Changing the source face, opacity, mouth-mask, etc. sliders updates `modules.globals` live and takes effect on the next processed frame — no restart needed.

### 5.4 Face-mapping mode (`--map-faces`, multi-face)
- For **images/videos**: `get_unique_faces_from_target_image/video()` extracts every face, and for video, **clusters embeddings with K-Means** to find the *unique* people appearing, picking the best (highest detection-confidence) representative crop per cluster. The user then manually assigns a **source face to each detected target identity** in a `MapperDialog`/`LiveMapperDialog` (Qt), enabling **"swap different faces onto different people"** in one run.
- For **live mode**, a simplified `simple_map` (list of source faces + target embeddings) is used instead of the full per-frame cluster map, matched by nearest-embedding at runtime.

---

## 6. Performance engineering (why it can run "live")

This fork is unusually aggressive about performance — worth calling out explicitly since it's most of what makes "real-time" plausible on consumer hardware:

1. **Execution-provider auto-detection** (`core.suggest_default_execution_provider`): tries CUDA → ROCm → CoreML → DirectML → CPU, in that preference order, based on what ONNX Runtime reports as available.
2. **FP16 model variant** (`inswapper_128_fp16.onnx`) used automatically when CUDA + PyTorch-CUDA are available (half the memory bandwidth), with FP32 fallback for older GPUs where FP16 can NaN.
3. **CUDA Graphs** (`_init_cuda_graph_session`): captures the swap model's inference graph once and replays it, cutting per-frame kernel-launch overhead on NVIDIA GPUs.
4. **Apple Silicon / CoreML graph rewrites** (`onnx_optimize.py`): four distinct ONNX graph-surgery passes (constant-folding dynamic Shape/Gather chains, decomposing `Pad(reflect)`, decomposing `Split`, widening scalar `Gather` indices) purely to eliminate CPU↔Neural-Engine partition boundaries that ONNX Runtime's CoreML EP would otherwise introduce — with measured wins like "21ms → 4ms" for the detector. Rewritten models are cached to disk with a `_coreml` suffix so the cost is paid once.
5. **Detection/inference overlap**: face detection for the *next* frame runs on a background thread while the *current* frame is being swapped — different hardware units (e.g. GPU vs Apple Neural Engine) work concurrently.
6. **Skip unnecessary model calls**: landmark model (`landmark_2d_106`) is only invoked when mouth-masking or an enhancer needs it; a `detect_one_face_fast`/`detect_many_faces_fast` path skips recognition+landmarks entirely when only bounding boxes are momentarily needed.
7. **Cheap paste-back**: precomputed/cached elliptical feathered alpha mask (§3.1) instead of a per-frame blur that scales with face size.
8. **Adaptive detection caching on Apple Silicon** (`get_faces_optimized`): skips re-detecting faces more often than a tuned interval when running live, using the last known detection.
9. **In-memory FFmpeg pipe pipeline** (§5.2) removes disk I/O from the video path entirely.
10. **Thread-pooled batch processing** for the disk-based fallback (`multi_process_frame`), tuned batch size based on thread count.
11. **`gpu_processing.py` note**: per-operation OpenCV CUDA acceleration (blur/resize/color-convert) is **deliberately disabled by default** — the code's own comment explains that upload/download overhead at webcam resolution outweighs the savings; the real GPU win comes from ONNX Runtime's CUDA/CoreML execution providers doing the heavy detection/swap/enhance inference. It can be forced on via `OPENCV_CUDA_PROCESSING=1`.
12. **Resource limiting** (`limit_resources`): TensorFlow memory-growth mode + OS-level max-memory clamp (`--max-memory`), single-thread `OMP_NUM_THREADS` tuning for CUDA.

---

## 7. Safety / content controls

- `--nsfw-filter` runs `opennsfw2` over the *target* image/video before processing; if the probability of NSFW content exceeds `0.85`, processing is aborted (`ui.check_and_ignore_nsfw`).
- This is opt-in (off by default via CLI), and is one of the README's cited "built-in checks."

---

## 8. Running it

### 8.1 Requirements
- Python 3.11 recommended (3.9+ is the hard floor checked at startup)
- `ffmpeg` on PATH (checked at startup — the app refuses to run without it)
- Models in `models/`: `inswapper_128.onnx` (or the fp16 variant) and a GFPGAN ONNX if you want the enhancer.

> **⚠️ Verified gotcha in this checkout**: `models/` currently contains `GFPGANv1.4.onnx`, but `modules/processors/frame/face_enhancer.py` hard-codes the filename **`gfpgan-1024.onnx`** and will report the model missing (`pre_check()` fails) until a file with that exact name exists in `models/`. Either rename/symlink `GFPGANv1.4.onnx` → `gfpgan-1024.onnx` (only safe if it's actually the same ONNX export the code expects — check the model's expected input size, 1024, against your file first) or source the correct `gfpgan-1024.onnx` export. `inswapper_128.onnx` (554MB) and `inswapper_128_fp16.onnx` (278MB) are both already present and correctly named, so plain `face_swapper` works out of the box; only the `face_enhancer` (GFPGAN) processor is affected. The GPEN-256/512 alternatives are unaffected — they download their own correctly-named models automatically on first use.

### 8.2 Setup
```bash
git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
python3.11 -m venv venv
source venv/bin/activate         # venv\Scripts\activate on Windows
pip install -r requirements.txt
```
Place the downloaded `.onnx` model files into `models/`.

### 8.3 GPU acceleration (optional but recommended for "live")
Pick the path matching your hardware, then pass `--execution-provider <name>`:
- **NVIDIA / CUDA**: install CUDA Toolkit 12.8 + cuDNN 8.9.7, `pip install onnxruntime-gpu==1.21.0`, run with `--execution-provider cuda`.
- **Apple Silicon / CoreML**: `pip install onnxruntime-silicon==1.13.1`, run with `python3.11 run.py --execution-provider coreml`.
- **Windows / DirectML (AMD/Intel/NVIDIA)**: `pip install onnxruntime-directml==1.21.0`, run with `--execution-provider directml`.
- **Intel / OpenVINO**: `pip install onnxruntime-openvino==1.21.0`, run with `--execution-provider openvino`.
- If none configured, it auto-falls-back to CPU.

### 8.4 Running the app

**GUI mode** (no source/target/output args):
```bash
python run.py
```
- Pick a **source face** image.
- Either pick a **target image/video** and click **Start**, or click **Live** to open the webcam preview.
- For multi-face swaps, enable the face-mapper toggle to assign source→target pairs before starting.

**Headless / CLI mode** (passing `-s`/`-t` triggers headless automatically):
```bash
python run.py -s source.jpg -t target.mp4 -o output.mp4 \
  --frame-processor face_swapper face_enhancer \
  --keep-fps --keep-audio --execution-provider cuda --execution-threads 8
```

Key flags (see `modules/core.py:parse_args` for the authoritative list):
```
-s / --source            source face image
-t / --target            target image or video
-o / --output             output path
--frame-processor         face_swapper | face_enhancer | face_enhancer_gpen256 | face_enhancer_gpen512 (repeatable)
--many-faces               swap every detected face with the same source
--map-faces                 use the source↔target identity map (multi-face, manual assignment)
--mouth-mask                 keep original mouth region for natural lip movement
--nsfw-filter                abort if content-safety check trips
--keep-fps / --keep-audio / --keep-frames
--video-encoder {libx264,libx265,libvpx-vp9}   --video-quality [0-51]
--execution-provider {cpu,cuda,coreml,directml,openvino,...}
--execution-threads N
--live-mirror / --live-resizable
-l / --lang                 UI language (see locales/*.json)
```

### 8.5 Tests & lint
```bash
python -m unittest discover tests        # unit tests (face_analyser logic, mocked deps)
ruff check .                             # matches CI (.github/workflows/ruff.yml)
```

---

## 9. Summary of "approach" in one paragraph

Deep-Live-Cam is a **plugin-style frame-processing pipeline**: any target (image, video, or live webcam) is reduced to a stream of frames; each frame is run through **InsightFace** for detection/embedding, an **inswapper_128 ONNX GAN** for identity swapping, and optionally **GFPGAN** for restoration, with a stack of hand-tuned post-processing (mouth masking, Poisson/opacity blending, sharpening, temporal smoothing) to make the result look natural frame-to-frame. The same `process_frame` contract is reused across image/video/webcam so the core AI logic never changes — only how frames are sourced (file read vs. FFmpeg pipe vs. webcam thread) and sunk (file write vs. FFmpeg encode vs. Qt preview) changes. A large fraction of the code is not the AI itself but **hardware-specific performance engineering** — ONNX Runtime execution-provider selection, CUDA Graphs, Apple Neural Engine graph rewrites, in-memory FFmpeg pipes, and detection/inference pipelining — needed to get face-swap inference down to real-time frame budgets.
