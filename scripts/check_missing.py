import os
import json
from pathlib import Path
import config

def check_missing():
    maps_dir = Path(config.MAP_JSON_DIR)
    frames_dir = Path(config.KEYFRAMES_DIR)

    total_missing = {}
    for map_file in sorted(maps_dir.glob("map_*.json")):
        video_id = map_file.stem.replace("map_", "")  # map_L21_V001.json -> L21_V001
        with open(map_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        missing = []
        for fname in raw.keys():  # "001.jpg", "002.jpg", ...
            img_path = frames_dir / video_id / fname
            if not img_path.exists():
                missing.append(fname)

        if missing:
            total_missing[video_id] = missing

    if not total_missing:
        print(" Tất cả JSON maps khớp với frames (không thiếu ảnh nào).")
    else:
        print(" Có ảnh bị thiếu:")
        for vid, miss in total_missing.items():
            print(f"- {vid}: {len(miss)} missing frames (vd: {miss[:10]})")

if __name__ == "__main__":
    check_missing()
