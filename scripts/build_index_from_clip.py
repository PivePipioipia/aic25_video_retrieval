
import os, json, re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import faiss
import config

def load_image_map():
    p = Path(config.IMAGE_MAP_JSON)
    raw = json.loads(p.read_text(encoding="utf-8"))
    # chấp nhận cả dict {"0":"L21_V001/001.jpg",...} lẫn list
    if isinstance(raw, dict):
        rel = [v for _, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
    else:
        rel = list(raw)
    return rel

def parse_relpath(relpath:str):
    # "L21_V001/001.jpg" -> ("L21_V001", 0-based index)
    relpath = relpath.replace("\\","/")
    m = re.match(r"^(L\d{2}_V\d{3})/(\d+)\.jpg$", relpath)
    if not m:
        raise ValueError(f"Định dạng keyframe không hợp lệ: {relpath}")
    vid = m.group(1)
    idx = int(m.group(2)) - 1  # 0-based
    return vid, idx

def main():
    rel_paths = load_image_map()
    print(f"Found {len(rel_paths)} keyframes from {config.IMAGE_MAP_JSON}")

    clip_dir = Path(config.CLIP_DIR)
    if not clip_dir.exists():
        raise SystemExit(f"CLIP_DIR không tồn tại: {clip_dir}")

    # nhóm theo video để load .npy theo từng video
    groups = {}
    for gid, rp in enumerate(rel_paths):
        vid, idx = parse_relpath(rp)
        groups.setdefault(vid, []).append((gid, idx))

    # lấy dimension từ 1 file npy đầu tiên
    any_vid = next(iter(groups.keys()))
    sample = np.load(clip_dir / f"{any_vid}.npy", mmap_mode="r")
    d = sample.shape[1]
    out = np.empty((len(rel_paths), d), dtype=np.float32)

    for vid, items in tqdm(groups.items(), desc="Merging provided CLIP features"):
        npy_path = clip_dir / f"{vid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Thiếu file: {npy_path}")
        feats = np.load(npy_path, mmap_mode="r")
        for gid, idx in items:
            if idx < 0 or idx >= feats.shape[0]:
                raise IndexError(f"Frame {idx+1} vượt quá số vector trong {npy_path} (shape {feats.shape})")
            out[gid] = feats[idx]

    # Chuẩn hoá L2 để dùng Inner-Product ~ cosine
    faiss.normalize_L2(out)
    np.save(config.FEATURES_ALL_NPY, out)
    print(f"✅ Saved features to {config.FEATURES_ALL_NPY} with shape {out.shape}")

    # Build FAISS
    d = out.shape[1]
    if config.FAISS_INDEX_TYPE.lower() == "ivf":
        nlist = config.FAISS_NLIST
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(out)
        index.add(out)
        index.nprobe = config.FAISS_NPROBE
    else:
        index = faiss.IndexFlatIP(d)
        index.add(out)

    faiss.write_index(index, config.FAISS_INDEX_BIN)
    print(f"✅ Saved FAISS index to {config.FAISS_INDEX_BIN}")

if __name__ == "__main__":
    main()
