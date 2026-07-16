# Deep-Live-Cam — Setup Log & Performance Notes

This documents what was done to get this repo running locally on CPU, and the
levers available to make it smoother — on this machine's CPU, and separately
if a GPU is ever available.

Hardware this was set up on: **Intel Core Ultra 5 225** (10 cores, no NVIDIA GPU).

---

## 1. What we did (setup log)

1. **Cloned the repo** into this folder from `https://github.com/hacksider/Deep-Live-Cam`.
2. **Created a Python 3.11 venv** (`venv/`) — the project needs 3.10/3.11; this
   machine's default `python3` is 3.12, so `python3.11` (already installed) was
   used explicitly.
3. **`pip install -r requirements.txt`** — first attempt failed building
   `insightface`'s C++ extension with `fatal error: Python.h: No such file or
   directory`. Root cause: `python3.11-dev` (the header package matching the
   venv's Python version) wasn't installed — only `python3-dev` for the
   system's default Python 3.12 was present. Fixed with:
   ```bash
   sudo apt-get install -y python3.11-dev
   ```
   Re-ran `pip install -r requirements.txt` — completed clean.
4. **Downloaded the two required ONNX models** into `models/`:
   - `inswapper_128_fp16.onnx` (~265 MB) — the face-swap model
   - `GFPGANv1.4.onnx` (~325 MB) — face enhancer (**not actually used** — see
     step 6; this repo version expects a different filename)
5. **First GUI launch crashed**: `qt.qpa.plugin: Could not load the Qt platform
   plugin "xcb"` — missing system library `libxcb-cursor0`. Fixed with:
   ```bash
   sudo apt-get install -y libxcb-cursor0
   ```
6. **Live preview crashed again**, this time inside the processing thread:
   `FileNotFoundError` from `get_face_enhancer()`. Root cause: the GUI's saved
   `switch_states.json` defaults to `"face_enhancer": true`, but this repo
   version's enhancer code (`modules/processors/frame/face_enhancer.py`)
   expects a file named **`gfpgan-1024.onnx`** — not the `GFPGANv1.4.onnx` the
   README tells you to download (the README is out of sync with this code
   revision). Fixed by disabling the enhancer default:
   ```json
   "fp_ui": {"face_enhancer": false, ...}
   ```
7. **"Nothing happens" on Live**: `switch_states.json` also defaulted to
   `"map_faces": true`. With Map Faces on, the **Live** button opens a
   completely different multi-face mapping dialog instead of doing a simple
   swap. Fixed by setting `"map_faces": false` in the same file.
8. **Lag/smoothness pass** (once it was actually working):
   - Swapped `onnxruntime-gpu` → **`onnxruntime-openvino`** (Intel's own
     inference runtime — see §2 below).
   - Reduced the face detector's fixed analysis resolution from `(640, 640)`
     to `(320, 320)` in `modules/face_analyser.py`'s `DET_SIZE` constant.

**Two important corrections we made to the app's own defaults**, worth
remembering if `switch_states.json` ever gets reset/deleted: **Face Enhancer
should stay "None"** (the bundled model file doesn't match what this code
expects) and **Map Faces should stay off** unless you specifically want the
multi-face mapping workflow.

---

## 2. Running it (current best command, CPU/OpenVINO)

```bash
cd /home/lavesh/Desktop/deep-live-cam
source venv/bin/activate
python run.py --execution-provider openvino --execution-threads 10 --frame-processor face_swapper
```

Then in the GUI: confirm **Map Faces** and **Face Enhancer** are off, **Select
a face** has your source photo, and click **Live** (not Start — Start is only
for processing a saved image/video file, not the webcam).

---

## 3. Enhancing further on CPU

Ranked by expected impact, biggest first:

1. **OpenVINO device target** — `onnxruntime-openvino` doesn't just accelerate
   the CPU path; on an Intel Core Ultra chip like this one, it can also target
   the integrated GPU or NPU via the `OV_DEVICE` environment variable (e.g.
   `OV_DEVICE=GPU` or `OV_DEVICE=NPU` before launching `run.py`, if the
   provider's options expose device selection — check
   `onnxruntime.capi._pybind_state.get_available_openvino_device_ids()` or the
   `provider_options` dict passed when creating the session). Worth
   experimenting with — the NPU/iGPU can be meaningfully faster than the CPU
   cores for this kind of small-model inference, and it's still "no NVIDIA GPU
   needed."
2. **Lower the live preview/capture resolution** further — currently
   640×360 (`PREVIEW_DEFAULT_WIDTH`/`HEIGHT` in `modules/ui.py`). Dropping to
   480×270 cuts the swap step's per-pixel cost proportionally. Diminishing
   returns past a point (too low and the swapped face looks blocky).
3. **`DET_SIZE`** — already dropped to `(320, 320)` (§1.8). Could go to
   `(256, 256)` for more speed if your face stays centered/close to the
   camera; below that, detection starts missing faces that are small, angled,
   or partially out of frame.
4. **Keep everything else off** — Face Enhancer, Poisson Blend, Many Faces,
   Mouth Mask, and Sharpness all cost extra CPU per frame. Only turn one on
   temporarily to judge its effect, then decide if the quality gain is worth
   the FPS cost.
5. **`--execution-threads`** — currently `10`, matching this CPU's core count
   exactly (`nproc` = 10, no hyperthreading). This is already close to
   optimal; going higher just adds thread-scheduling overhead for no benefit.
6. **Quantized/int8 models** — if lag is still an issue after the above, a
   lower-precision (int8) version of `inswapper_128` would run faster on CPU
   at some quality cost. Not bundled with this repo by default — would need
   sourcing or quantizing the fp16 model yourself (e.g. via
   `onnxruntime.quantization`), an extra step beyond what's here today.

---

## 4. Enhancing on GPU (if an NVIDIA GPU becomes available)

This machine has no NVIDIA GPU today, but if you move this to (or add) one:

1. **Install matching CUDA + cuDNN** — per the main README: CUDA Toolkit
   12.8.0 with cuDNN v8.9.7 for the pinned `onnxruntime-gpu==1.23.2`. Version
   matching matters a lot here — a mismatched CUDA/cuDNN pair is the single
   most common cause of "onnxruntime-gpu installed but still silently falls
   back to CPU."
2. **Reinstall the GPU-matched runtime**:
   ```bash
   pip uninstall onnxruntime onnxruntime-openvino
   pip install onnxruntime-gpu==1.23.2
   ```
3. **Run with `--execution-provider cuda`** instead of `openvino`/`cpu`.
4. **Revert `DET_SIZE` back to `(640, 640)`** (or even leave it — GPU inference
   absorbs the larger detector resolution far more easily than CPU does),
   for better face-detection robustness at distance/angle.
5. **Turn Face Enhancer back on** (once pointed at a model file this code
   actually expects — see the `gfpgan-1024.onnx` note in §1.6) — GPU can
   afford GFPGAN's cost far better than CPU can; this was the main quality
   feature disabled to keep CPU playback smooth.
6. **Consider `--execution-provider tensorrt`** instead of plain `cuda` for a
   further speedup — TensorRT compiles the ONNX graph into a hardware-specific
   optimized engine, at the cost of a slower first-run "warm-up" compile step
   per model.
7. **Raise preview/capture resolution** — 1080p becomes realistic on a modern
   GPU (per the tool's own benchmarks, ~60fps @ 1080p on an RTX 4090-class
   card); on CPU that resolution isn't viable at interactive framerates.

---

## 5. Responsible use

This is a real face-swap tool, not this project's stylized VRM avatar system —
worth remembering if a real person's likeness is used: get their consent, and
label shared output as a deepfake, per the upstream project's own guidance.
