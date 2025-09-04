# utils/mapping.py
from __future__ import annotations
import os, csv, glob, json
from typing import Dict, Tuple, Optional
import config

# Trả về: dict[str_rel_path] -> (video_name.mp4, frame_idx:int)
_mapping: Dict[str, Tuple[str, int]] = {}
_fname2trueframe: Dict[str, int] = {}
_loaded = False

# Các tên cột có thể gặp trong CSV map
CAND_COLS_KEYFRAME = ["keyframe", "keyframe_path", "image_path", "rel_path"]
CAND_COLS_VIDEO    = ["video", "video_id", "video_name", "video_file"]
CAND_COLS_FRAME    = ["frame", "frame_id", "frame_idx", "frame_index"]

def _first_hit(cols, header):
    for c in cols:
        if c in header:
            return c
    return None





def _load_fname_trueframe_map():
    path = getattr(config, 'KEYFRAME_FRAME_MAP_JSON', '')
    if not path:
        return
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f) or {}
                if isinstance(data, dict):
                    # chuẩn hoá key: chỉ lấy basename và đảm bảo dạng '0000.jpg'
                    for k, v in data.items():
                        fname = os.path.basename(str(k)).strip()
                        try:
                            _fname2trueframe[fname] = int(v)
                        except Exception:
                            continue
    except Exception:
        pass

def _load_from_json_dir(dir_path: str):
    """
    Đọc toàn bộ map JSON custom trong map_idx/
    Ví dụ: {"001.jpg": 0, "002.jpg": 4, ...}
    -> _mapping["L21_V001/001.jpg"] = ("L21_V001.mp4", 0)
    """
    global _mapping
    import json
    from pathlib import Path

    json_files = sorted(Path(dir_path).glob("map_*.json"))
    for jf in json_files:
        video_id = jf.stem.replace("map_", "")   # map_L21_V001.json -> L21_V001
        with open(jf, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for fname, fidx in raw.items():
            rel = f"{video_id}/{fname}"
            _mapping[rel] = (f"{video_id}.mp4", int(fidx))


def ensure_loaded():
    global _loaded
    if _loaded:
        return
    _load_fname_trueframe_map()

    if os.path.isdir(config.MAP_JSON_DIR):
        _load_from_json_dir(config.MAP_JSON_DIR)

    _loaded = True


def keyframe_to_video_frame(rel_path: str) -> Optional[Tuple[str, int]]:
    """
    rel_path: 'L21_V001/0000.jpg'
    -> ('L21_V001.mp4', 0) nếu có map; None nếu không tìm thấy.
    """
    ensure_loaded()
    rel = rel_path.replace("\\", "/").strip()
    return _mapping.get(rel)
