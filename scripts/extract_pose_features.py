import os
import json
import argparse
from pathlib import Path
from typing import List
import numpy as np

import config
from utils.pose import extract_pose_sequence_from_images


def _load_image_map() -> List[str]:
    with open(config.IMAGE_MAP_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
        rel_paths = [v for _, v in items]
    else:
        rel_paths = list(raw)
    return rel_paths


def _load_scenes(folder: str) -> List[dict]:
    scenes_path = os.path.join(getattr(config, "TRAKE_DATA_DIR", os.path.join(config.DATASET_ROOT, "TRAKE_DATA")), "scenes", f"{folder}.json")
    if not os.path.exists(scenes_path):
        return []
    with open(scenes_path, "r", encoding="utf-8") as f:
        data = json.load(f) or []
    return data


def _collect_image_paths_for_scene(folder: str, scene: dict, rel_all: List[str]) -> List[str]:
    start = int(scene.get("start", 0))
    end = int(scene.get("end", 0))
    abs_list: List[str] = []
    key_root = Path(config.KEYFRAMES_DIR)
    for rp in rel_all:
        rp_str = str(rp).replace("\\", "/")
        if not rp_str.startswith(folder + "/"):
            continue
        name = Path(rp_str).name
        num = "".join([c for c in Path(name).stem if c.isdigit()])
        try:
            idx = int(num)
        except Exception:
            continue
        if idx < start or idx > end:
            continue
        abs_list.append(str(key_root / rp_str))
    abs_list.sort(key=lambda p: int("".join([c for c in Path(p).stem if c.isdigit()]) or 0))
    return abs_list


def main():
    parser = argparse.ArgumentParser(description="Extract pose feature sequences per scene")
    parser.add_argument("video", help="Video id/folder, e.g., L21_V001")
    parser.add_argument("--out_dir", dest="out_dir", default=None, help="Output directory for scene npy files")
    parser.add_argument("--smooth", dest="smooth", type=int, default=5, help="Smoothing window")
    args = parser.parse_args()

    folder = args.video
    scenes = _load_scenes(folder)
    if not scenes:
        raise SystemExit(f"No scenes found for {folder}. Expected JSON under TRAKE_DATA/scenes/{folder}.json")

    rel_all = _load_image_map()

    out_dir = args.out_dir or os.path.join(getattr(config, "TRAKE_DATA_DIR", os.path.join(config.DATASET_ROOT, "TRAKE_DATA")), "features", folder)
    os.makedirs(out_dir, exist_ok=True)

    for i, sc in enumerate(scenes, start=1):
        img_paths = _collect_image_paths_for_scene(folder, sc, rel_all)
        if not img_paths:
            print(f"[WARN] Scene {i:03d} has no keyframes; skipping")
            continue
        seq = extract_pose_sequence_from_images(img_paths, smooth_window=args.smooth)
        if seq is None or not isinstance(seq, np.ndarray) or seq.ndim != 2:
            print(f"[WARN] Failed to extract pose for scene {i:03d}")
            continue
        out_path = os.path.join(out_dir, f"scene_{i:03d}.npy")
        np.save(out_path, seq.astype(np.float32))
        print(f"Saved {out_path} with shape {seq.shape}")


if __name__ == "__main__":
    main()


