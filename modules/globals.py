# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False
live_capture_width: int = 640   # Lower = faster detect/swap on CPU, at the cost of preview sharpness
live_capture_height: int = 360

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {
    "face_enhancer": False,
    "face_enhancer_gpen256": False,
    "face_enhancer_gpen512": False,
    "face_enhancer_codeformer": False,
}

# CodeFormer fidelity weight (0.0-1.0): lower = higher quality/more generated,
# higher = closer to the input face. 0.9 matches CodeFormer's own default.
codeformer_fidelity: float = 0.9

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# Eyes Mask Options
eyes_mask: bool = False            # Keep the target's original eyes instead of the swapped ones
eyes_mask_size: float = 1.0        # Expansion factor for the eye cutout region (relative)

# Beard / Jaw Mask Options
beard_mask: bool = False           # Keep the target's original beard/jawline instead of the swapped one
beard_mask_size: float = 1.0       # Expansion factor for the beard/jaw cutout region (relative)

# Eyebrows Mask Options
eyebrows_mask: bool = False        # Keep the target's original eyebrows instead of the swapped ones
eyebrows_mask_size: float = 1.0    # Expansion factor for the eyebrows cutout region (relative)

# Forehead / Hairline Mask Options
forehead_mask: bool = False        # Keep the target's original forehead/hairline instead of the swapped one
forehead_mask_size: float = 1.0    # Expansion factor for the forehead cutout region (relative)

# Glasses Mask Options — preserves the target's eyeglasses (wider than the eyes mask
# so it also covers the frame/temple area) instead of letting the swap erase them
glasses_mask: bool = False
glasses_mask_size: float = 1.0

# Per-region mask opacity (0.0-1.0): how strongly the target's original pixels are
# restored for each region above. 1.0 = fully restore original (same as before these
# existed), lower values partially blend the swap back in.
mouth_mask_opacity: float = 1.0
eyes_mask_opacity: float = 1.0
beard_mask_opacity: float = 1.0
eyebrows_mask_opacity: float = 1.0
forehead_mask_opacity: float = 1.0
glasses_mask_opacity: float = 1.0

# Max Coverage Mode — when enabled, forces all preserve-masks above off for the
# frame regardless of their individual toggles, maximizing how much of the source
# face is applied (closest achievable approximation of an "exact head swap").
max_coverage_mode: bool = False

# Hairline Feather — extra softening (Gaussian sigma) applied near the top edge of
# the whole-face mask, easing the seam where the swapped face meets the hairline.
hairline_feather: float = 0.0

# Webcam Denoise — light bilateral-filter pre-pass on captured webcam frames before
# detection/swap, to reduce sensor noise (mainly helps low-light webcams).
denoise_webcam: bool = False

# Temporal Mask Stabilization — smooths preserve-mask polygon jitter across
# frames (video/live) to reduce boundary flicker. Snaps to the raw polygon
# instead of blending when the face moves more than a small threshold, so
# real head movement doesn't drag a stale mask behind it.
mask_stabilization: bool = False
mask_stabilization_weight: float = 0.5  # current-frame weight; lower = smoother/laggier

# Pose-Adaptive Mask Sizing — shrinks preserve-mask expansion as the face
# turns away from frontal (estimated from landmarks only, no pose model),
# instead of using the same fixed expansion at every angle.
pose_adaptive_masks: bool = False

# Frequency-Separation Skin Detail Transfer — keeps the target's real skin
# texture (pores/wrinkles) while taking color/tone from the swapped face,
# instead of the swap's own (often smoother/"plastic") skin detail.
# 0.0 = off (default, no change to existing behavior), 1.0 = fully use the
# target's original detail layer.
skin_detail_strength: float = 0.0

# Post-Swap Shape Correction — anisotropically stretches the generated face
# toward the target's actual jaw width/height proportions (estimated from
# landmarks only), so the target's real head shape shows through more than
# the source identity's shape. 0.0 = off (default), 1.0 = full correction.
shape_correction_strength: float = 0.0

# Face-Parsing Precision Mode — uses BiSeNet semantic segmentation (pixel-
# accurate mouth/eyes/eyebrows/glasses regions, plus automatic hair
# occlusion) instead of landmark-geometry approximations for the masks that
# have a matching class. Also makes glasses restoration auto-detect (no
# glasses found -> no-op) instead of always covering a guessed eye band.
face_parsing_masks: bool = False

# Skin Tone Match — color-matches the swapped face crop to the target's skin tone/lighting
skin_tone_match: bool = False

# Local contrast/lighting match (CLAHE) on the swapped face crop
clahe_match: bool = False

# Edge Feather — sigma of the Gaussian blur used to soften the swap boundary
edge_feather: float = 12.0

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
