"""BiSeNet-based face parsing (semantic segmentation) — pixel-accurate
region masks (skin/hair/eyebrows/eyes/glasses/etc.) as a precision upgrade
over the landmark-geometry masks in face_swapper.py / face_masking.py.

This is a utility module consumed by face_swapper.py, not a selectable
frame processor — there is no pre_check/process_image/etc. interface here.

Model: yakhyo/face-parsing resnet18.onnx (MIT license), trained on
CelebAMask-HQ. Label order confirmed directly against the model author's own
reference implementation (utils/common.py ATTRIBUTES list), NOT assumed from
the generic CelebAMask-HQ raw-label ordering (which differs):
    1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g, 7 l_ear,
    8 r_ear, 9 ear_r, 10 nose, 11 mouth, 12 u_lip, 13 l_lip, 14 neck,
    15 neck_l, 16 cloth, 17 hair, 18 hat  (0 = background)

Note: this 19-class scheme has no distinct "forehead" or "beard/facial-hair"
label (forehead is indistinguishable from cheek "skin"), so those two masks
in face_swapper.py stay on landmark geometry regardless of this module.
"""

import os
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import modules.globals
from modules.typing import Face, Frame

NAME = "DLC.FACE-PARSING"
INPUT_SIZE = 512

LABELS = {
    "skin": 1, "l_brow": 2, "r_brow": 3, "l_eye": 4, "r_eye": 5,
    "eye_g": 6, "l_ear": 7, "r_ear": 8, "ear_r": 9, "nose": 10,
    "mouth": 11, "u_lip": 12, "l_lip": 13, "neck": 14, "neck_l": 15,
    "cloth": 16, "hair": 17, "hat": 18,
}

MODEL_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
MODEL_FILE = "resnet18.onnx"

_PARSER = None
_THREAD_LOCK = threading.Lock()

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pre_check() -> bool:
    model_path = os.path.join(models_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        from modules.utilities import conditional_download
        conditional_download(models_dir, [MODEL_URL])
    return os.path.exists(model_path)


def get_face_parser() -> Any:
    global _PARSER
    with _THREAD_LOCK:
        if _PARSER is None:
            import onnxruntime
            model_path = os.path.join(models_dir, MODEL_FILE)
            if not os.path.exists(model_path):
                from modules.utilities import conditional_download
                conditional_download(models_dir, [MODEL_URL])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            providers = modules.globals.execution_providers or ["CPUExecutionProvider"]
            _PARSER = onnxruntime.InferenceSession(model_path, providers=providers)
            print(f"{NAME}: Model loaded successfully.")
    return _PARSER


def _bbox_crop_box(
    face: Face, frame: Frame, pad_ratio: float = 0.35
) -> Optional[Tuple[int, int, int, int]]:
    """Padded bbox around the face, biased upward to include forehead/hair."""
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox.astype(float)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    pad_x = w * pad_ratio
    fh, fw = frame.shape[:2]
    min_x = max(0, int(x1 - pad_x))
    min_y = max(0, int(y1 - h * pad_ratio * 1.5))  # extra headroom for hair/forehead
    max_x = min(fw, int(x2 + pad_x))
    max_y = min(fh, int(y2 + h * pad_ratio))
    if max_x <= min_x or max_y <= min_y:
        return None
    return min_x, min_y, max_x, max_y


def parse_face(face: Face, frame: Frame) -> Optional[Dict[str, Any]]:
    """Runs BiSeNet on a padded bbox crop around `face`.

    Returns {"box": (x1,y1,x2,y2), "class_map": np.ndarray[h,w] uint8} where
    class_map is in the crop's own coordinate space (h,w match the box) and
    values are LABELS indices, or None if parsing wasn't possible.
    """
    box = _bbox_crop_box(face, frame)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        session = get_face_parser()
    except Exception as e:
        print(f"{NAME}: {e}")
        return None

    crop_h, crop_w = crop.shape[:2]
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _MEAN) / _STD
    blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    try:
        input_name = session.get_inputs()[0].name
        outputs = session.run(["output"], {input_name: blob})
        class_map_512 = outputs[0].squeeze(0).argmax(0).astype(np.uint8)
    except Exception as e:
        print(f"{NAME}: Error during parsing inference: {e}")
        return None

    class_map = cv2.resize(class_map_512, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    return {"box": box, "class_map": class_map}


def region_mask(
    parsed: Dict[str, Any], frame_shape: Tuple[int, int], *class_names: str
) -> np.ndarray:
    """Full-frame-sized uint8 binary mask (0/255) for the union of the given
    BiSeNet class names, placed at parsed["box"]."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    class_map = parsed["class_map"]
    x1, y1, x2, y2 = parsed["box"]
    region = np.zeros(class_map.shape, dtype=bool)
    for name in class_names:
        idx = LABELS.get(name)
        if idx is not None:
            region |= (class_map == idx)
    mask[y1:y2, x1:x2] = region.astype(np.uint8) * 255
    return mask
