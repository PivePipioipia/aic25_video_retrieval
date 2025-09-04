from __future__ import annotations
import os
from typing import List, Optional
import numpy as np

try:
    import config  # optional
except Exception:
    config = None


def _smooth_sequence(x: np.ndarray, window: int = 5) -> np.ndarray:
    if not isinstance(x, np.ndarray) or x.ndim != 2 or window <= 1:
        return x
    w = min(window, x.shape[0] if x.shape[0] > 0 else 1)
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones((w,), dtype=np.float32) / float(w)
    out = np.vstack([np.convolve(xp[:, j], ker, mode="valid") for j in range(x.shape[1])]).T
    return out.astype(np.float32)


def _normalize_joints(joints: np.ndarray) -> np.ndarray:
    # joints shape: (T, 2*K) or (T, 3*K). Normalize per-frame by torso scale if available
    if not isinstance(joints, np.ndarray) or joints.ndim != 2:
        return joints
    x = joints.astype(np.float32)
    # zero-mean per frame
    x = x - np.mean(x, axis=1, keepdims=True)
    # scale per frame by L2 norm to be roughly scale-invariant
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    return x


def extract_pose_sequence_from_images(image_paths: List[str], smooth_window: int = 5) -> Optional[np.ndarray]:
    """
    Extract a simple pose feature sequence from a list of image paths.
    Returns array of shape (T, D) or None if backend unavailable.
    """
    backend = "mediapipe"
    if config is not None:
        backend = getattr(config, "POSE_BACKEND", backend)

    if backend == "mediapipe":
        try:
            import mediapipe as mp  # type: ignore
            import cv2  # type: ignore
            mp_pose = mp.solutions.pose
            points: List[List[float]] = []
            with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
                for p in image_paths:
                    if not os.path.exists(p):
                        continue
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = pose.process(img_rgb)
                    if not res.pose_landmarks:
                        continue
                    lm = res.pose_landmarks.landmark
                    vec = []
                    for l in lm:
                        vec.extend([l.x, l.y])  # use 2D
                    points.append(vec)
            if not points:
                return None
            arr = np.asarray(points, dtype=np.float32)
            arr = _normalize_joints(arr)
            arr = _smooth_sequence(arr, window=smooth_window)
            return arr
        except Exception:
            return None

    # Unknown backend: return None
    return None


