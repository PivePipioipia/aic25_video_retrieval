import os
import json
from pathlib import Path
import config

def clean_maps():
    maps_dir = Path(config.MAP_JSON_DIR)
    frames_dir = Path(config.KEYFRAMES_DIR)
    out_dir = maps_dir.parent / "map_idx_clean"
    out_dir.mkdir(exist_ok=True)

    total_removed = 0
    for map_file in sorted(maps_dir.glob("map_*.json")):
        video_id = map_file.stem.replace("map_", "")
        with open(map_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cleaned = {}
        removed = []
        for fname, fidx in raw.items():
            img_path = frames_dir / video_id / fname
            if img_path.exists():
                cleaned[fname] = fidx
            else:
                removed.append(fname)

        # lưu file mới
        out_file = out_dir / map_file.name
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

        if removed:
            print(f" {video_id}: removed {len(removed)} missing frames (vd: {removed[:10]})")
            total_removed += len(removed)
        else:
            print(f" {video_id}: no missing frames")

    print(f"\nDone. Tổng cộng đã loại bỏ {total_removed} frames thiếu.")
    print(f"Map sạch lưu ở: {out_dir}")

if __name__ == "__main__":
    clean_maps()
