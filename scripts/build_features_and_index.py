# scripts/build_features_and_index.py
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss
import torch
import clip
from pathlib import Path
import config

def build_clip_features_for_video(video_id, rel_abs_pairs, device, out_dir):
    """
    Encode toàn bộ ảnh của 1 video -> lưu thành .npy riêng
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    feats = []
    for rel_path, abs_path in tqdm(rel_abs_pairs, desc=f"Encoding {video_id}"):
        if not os.path.exists(abs_path):
            continue
        img = Image.open(abs_path).convert("RGB")
        with torch.no_grad():
            x = preprocess(img).unsqueeze(0).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))

    if feats:
        feats = np.concatenate(feats, axis=0)
        np.save(out_dir / f"{video_id}.npy", feats)
        print(f" Saved {video_id}.npy – shape {feats.shape}")
    else:
        print(f" Không có ảnh cho {video_id}")
        feats = None
    return feats

def merge_and_build_index(npy_dir, out_feats, out_index):
    """
    Ghép toàn bộ .npy -> features_all.npy và build FAISS index
    """
    all_feats = []
    for f in sorted(npy_dir.glob("*.npy")):
        feats = np.load(f)
        all_feats.append(feats)

    all_feats = np.concatenate(all_feats, axis=0)

    #  Normalize toàn bộ features trước khi build index
    faiss.normalize_L2(all_feats)

    # Lưu ra file npy (đã normalize)
    np.save(out_feats, all_feats)
    print(f" Lưu merged features_all.npy – shape {all_feats.shape}")

    d = all_feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_feats)

    faiss.write_index(index, str(out_index))
    print(f" Lưu FAISS index vào {out_index}")


def main():
    device = "cuda" if config.USE_GPU and torch.cuda.is_available() else "cpu"
    maps_dir = Path(config.MAP_JSON_DIR)
    out_dir = Path(config.CLIP_DIR)   # lưu .npy theo video
    out_dir.mkdir(parents=True, exist_ok=True)

    for map_file in sorted(maps_dir.glob("map_*.json")):
        video_id = map_file.stem.replace("map_", "")
        with open(map_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = sorted(raw.items(), key=lambda kv: int(kv[1]))

        rel_abs_pairs = [
            (f"{video_id}/{fname}", str(Path(config.KEYFRAMES_DIR) / video_id / fname))
            for fname, _ in items
        ]

        if (out_dir / f"{video_id}.npy").exists():
            print(f" Skip {video_id}, đã có file .npy")
            continue
        build_clip_features_for_video(video_id, rel_abs_pairs, device, out_dir)


    merge_and_build_index(out_dir, Path(config.FEATURES_ALL_NPY), Path(config.FAISS_INDEX_BIN))

if __name__ == "__main__":
    main()
