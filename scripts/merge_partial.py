# scripts/merge_partial.py
import numpy as np
from pathlib import Path
import faiss
import config

def merge_and_build_index():
    npy_dir = Path(config.CLIP_DIR)
    out_feats = Path(config.FEATURES_ALL_NPY)
    out_index = Path(config.FAISS_INDEX_BIN)

    all_feats = []
    for f in sorted(npy_dir.glob("*.npy")):
        feats = np.load(f)
        all_feats.append(feats)
        print(f"Loaded {f.name} – shape {feats.shape}")

    if not all_feats:
        print(" Không tìm thấy file .npy nào để merge.")
        return

    all_feats = np.concatenate(all_feats, axis=0)
    np.save(out_feats, all_feats)
    print(f" Lưu merged features_all.npy – shape {all_feats.shape}")

    d = all_feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_feats)
    faiss.write_index(index, str(out_index))
    print(f" Lưu FAISS index vào {out_index}")

if __name__ == "__main__":
    merge_and_build_index()
