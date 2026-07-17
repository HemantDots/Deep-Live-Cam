"""CodeFormer face enhancer — ONNX-based face restoration with a fidelity
weight control (0 = higher quality/more generated, 1 = closer to the input).

Unlike GFPGAN/GPEN, CodeFormer's ONNX graph takes the fidelity weight as a
second runtime input rather than a fixed constant, so the alignment/paste-back
code lives here instead of reusing modules.processors.frame._onnx_enhancer's
single-input enhance_face_onnx helper.
"""

from typing import Any, List
import os
import threading

import cv2
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import is_image, is_video
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    THREAD_SEMAPHORE,
)

NAME = "DLC.FACE-ENHANCER-CODEFORMER"
INPUT_SIZE = 512
MODEL_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx"
MODEL_FILE = "codeformer.onnx"

ENHANCER = None
THREAD_LOCK = threading.Lock()

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

# Same 5-point template convention used by _onnx_enhancer.py's GPEN alignment.
_TEMPLATE = np.array([
    [0.31556875, 0.4615741],
    [0.68262291, 0.4615741],
    [0.50009375, 0.6405054],
    [0.34947187, 0.8246919],
    [0.65343645, 0.8246919],
], dtype=np.float32) * INPUT_SIZE


def pre_check() -> bool:
    model_path = os.path.join(models_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        update_status(f"Downloading {MODEL_FILE}...", NAME)
        from modules.utilities import conditional_download
        conditional_download(models_dir, [MODEL_URL])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_enhancer() -> Any:
    global ENHANCER
    with THREAD_LOCK:
        if ENHANCER is None:
            model_path = os.path.join(models_dir, MODEL_FILE)
            if not os.path.exists(model_path):
                from modules.utilities import conditional_download
                conditional_download(models_dir, [MODEL_URL])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"{NAME}: Loading ONNX model from {model_path}")
            ENHANCER = create_onnx_session(model_path)
            warmup_session(ENHANCER)
            print(f"{NAME}: Model loaded successfully.")
    return ENHANCER


def _get_face_affine(face: Any):
    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38], lm106[88], lm106[86], lm106[52], lm106[61],
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, _TEMPLATE, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    return M, cv2.invertAffineTransform(M)


def _preprocess(face_crop: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = (blob - 0.5) / 0.5
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def _postprocess(output: np.ndarray) -> np.ndarray:
    img = output[0].transpose(1, 2, 0)
    img = np.clip(img, -1.0, 1.0)
    img = (img + 1.0) * 0.5 * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame

    M, inv_M = _get_face_affine(face)
    if M is None:
        return temp_frame

    try:
        face_crop = cv2.warpAffine(
            temp_frame, M, (INPUT_SIZE, INPUT_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        blob = _preprocess(face_crop)

        fidelity = getattr(modules.globals, "codeformer_fidelity", 0.9)
        fidelity = max(0.0, min(1.0, fidelity))
        weight = np.array(fidelity, dtype=np.float64)

        with THREAD_SEMAPHORE:
            input_names = [i.name for i in session.get_inputs()]
            image_input_name = next(n for n in input_names if n != "weight")
            outputs = session.run(
                ["output"],
                {image_input_name: blob, "weight": weight},
            )
        enhanced = _postprocess(outputs[0])
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame

    # Feathered-edge paste-back, same scheme as _onnx_enhancer.enhance_face_onnx.
    mask = np.ones((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    border = max(1, INPUT_SIZE // 16)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    h, w = temp_frame.shape[:2]
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0,
    )

    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_enhanced.astype(np.float32) * mask_3ch +
              temp_frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    if detected_faces:
        target_face = detected_faces[0]
    else:
        target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = imread_unicode(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame)
        imwrite_unicode(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = imread_unicode(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    imwrite_unicode(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
