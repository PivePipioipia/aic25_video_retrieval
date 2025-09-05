import os
import json
from pathlib import Path

def load_places_vocabulary(categories_file: str):
    """Load danh sách categories từ categories_places365.txt"""
    vocab = []
    with open(categories_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split(" ")[0]  # VD: 'airport/indoor'
            place = token.split("/")[-1]  # lấy 'indoor'
            vocab.append(place)
    return vocab


def load_places_json(root_dir: str):
    """
    Duyệt qua tất cả các subfolder trong root_dir,
    đọc JSON kết quả place (từng video).
    Trả về dict: { "L30_V001": [...], "L30_V002": [...], ... }
    """
    places_data = {}
    root_dir = Path(root_dir)

    for subdir in sorted(root_dir.glob("place_*")):  # VD: place_L30a
        if not subdir.is_dir():
            continue
        for json_file in sorted(subdir.glob("*.json")):
            video_id = json_file.stem.replace("place_", "")  # L30_V001
            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"[WARN] lỗi đọc {json_file}: {e}")
                    data = []
            places_data[video_id] = data

    print(f" Loaded {len(places_data)} videos with place annotations.")
    return places_data


def build_place_vocab_from_json(places_json: dict):
    """
    Xây vocab vi→en dựa trên dữ liệu trong places_json.
    Trả về dict: {"bảo tàng lịch sử tự nhiên": "natural_history_museum",
                  "cửa hàng thú cưng": "pet_shop", ...}
    """
    vi2en = {}
    for video_id, frames in places_json.items():
        for f in frames:
            if "place_vi" in f and "place_en" in f:
                vi = f["place_vi"].strip().lower()
                en = f["place_en"].strip().lower()
                if vi and en:
                    vi2en[vi] = en
    print(f" Built vi→en vocab: {len(vi2en)} entries.")
    return vi2en
