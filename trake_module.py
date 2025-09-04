# trake_module.py
from __future__ import annotations
import os, re, json
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import config  # optional: DTW flags/paths
except Exception:
    config = None

# NOTE: Sẽ dùng MyFaiss và DictImagePath từ app (truyền vào hàm)
# Quy ước: video_name = 'L21_V001.mp4' => folder 'L21_V001'

# ---------- Parse events ----------
def parse_events_from_text(text: str, max_events: int = 4) -> List[str]:
    """
    Tách các event từ mô tả. Hỗ trợ nhiều pattern thường gặp:
    - 'Event1: ... Event2: ...'
    - 'E1: ...; E2: ...'
    - '... | ... | ...'
    - mỗi dòng một event
    """
    if not text:
        return []

    # 1) Tách theo "Event\d+:" hay "E\d+:"
    parts = re.split(r'(?:^|\s)(?:Event|E)\s*\d+\s*:\s*', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts[:max_events]

    # 2) Tách theo ký tự phân cách thường dùng
    for sep in ['|', ';', '->', '→']:
        if sep in text:
            items = [s.strip() for s in text.split(sep)]
            items = [s for s in items if s]
            if len(items) >= 2:
                return items[:max_events]

    # 3) Mỗi dòng là một event
    lines = [l.strip("- \t") for l in text.splitlines() if l.strip()]
    if len(lines) >= 2:
        return lines[:max_events]

    # 4) Nếu không tách được, coi toàn bộ là 1 event
    return [text.strip()][:max_events]

# ---------- Helper ----------
def _folder_from_video(video_name: str) -> str:
    # 'L21_V001.mp4' -> 'L21_V001'
    return os.path.splitext(os.path.basename(video_name))[0]

def _frame_idx_from_relpath(rel_path: str) -> int:
    # 'L21_V001/0123.jpg' -> 123
    base = os.path.splitext(os.path.basename(rel_path))[0]
    digits = ''.join(ch for ch in base if ch.isdigit())
    try:
        return int(digits)
    except:
        return 0

def _try_load_json(path: str) -> Optional[dict]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None

# ---------- Template & scene utilities (for DTW) ----------
def _load_template_sequence() -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    Load template pose features (T0 x D) and event indices [t1..t4] if provided.
    Return (template_seq, template_events) or (None, None) when unavailable.
    """
    try:
        if not config:
            return None, None
        pose_path = getattr(config, "TEMPLATE_POSE_NPY", "")
        events_path = getattr(config, "TEMPLATE_EVENTS_JSON", "")
        tpl = np.load(pose_path) if pose_path and os.path.exists(pose_path) else None
        evs = None
        meta = _try_load_json(events_path)
        if meta:
            # accept list or dict with keys t1..t4
            if isinstance(meta, list):
                evs = [int(x) for x in meta]
            elif isinstance(meta, dict):
                keys = ["t1", "t2", "t3", "t4"]
                evs = [int(meta[k]) for k in keys if k in meta]
        return tpl, evs
    except Exception:
        return None, None

def _list_scenes_for_video(folder: str) -> List[Tuple[int, int]]:
    """Load scene list [(start,end), ...] from TRAKE_DATA/scenes/<folder>.json"""
    if not config:
        return []
    scenes_path = os.path.join(getattr(config, "TRAKE_DATA_DIR", ""), "scenes", f"{folder}.json")
    scenes = _try_load_json(scenes_path) or []
    out: List[Tuple[int, int]] = []
    for item in scenes:
        try:
            out.append((int(item.get("start", 0)), int(item.get("end", 0))))
        except Exception:
            pass
    return out

def _load_scene_data(folder: str, scene_idx: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]]:
    """
    Load scene data:
      - features: (T, D) float16/float32
      - frame_ids: (T,) ints, optional
      - stride: int, optional

    Supports:
      - features/<folder>/scene_XXX.npy  (features only)
      - features/<folder>/scene_XXX.npz  (expects keys: 'features', 'frame_ids', 'stride')
    """
    if not config:
        return None
    feat_dir = os.path.join(getattr(config, "TRAKE_DATA_DIR", ""), "features", folder)
    base = os.path.join(feat_dir, f"scene_{scene_idx:03d}")
    npy_path = base + ".npy"
    npz_path = base + ".npz"
    try:
        if os.path.exists(npy_path):
            feats = np.load(npy_path)
            return feats, None, None
        if os.path.exists(npz_path):
            with np.load(npz_path) as z:
                if "features" in z:
                    feats = z["features"]
                elif "arr" in z:
                    feats = z["arr"]
                else:
                    # fallback: first array in archive
                    feats = z[z.files[0]] if len(z.files) > 0 else None
                if feats is None:
                    return None
                frame_ids = z["frame_ids"] if "frame_ids" in z else None
                stride = int(z["stride"]) if "stride" in z else None
                return feats, frame_ids, stride
    except Exception:
        return None
    return None

def _dtw_align(template: np.ndarray, target: np.ndarray, metric: str = "cosine") -> Tuple[float, List[Tuple[int, int]]]:
    """Minimal DTW alignment. Uses fastdtw if available; otherwise falls back to simple linear mapping.
    Returns (cost, path) where path is list of (i_tpl, j_tgt).
    """
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import cosine

        def dist(a, b):
            if metric == "l2":
                return float(np.linalg.norm(a - b))
            return float(cosine(a, b))

        cost, path = fastdtw(template, target, dist=dist)
        return float(cost), path
    except Exception:
        # linear fallback
        T0, T1 = template.shape[0], target.shape[0]
        path = [(i, min(int(round(i * (T1 - 1) / max(1, (T0 - 1)))), T1 - 1)) for i in range(T0)]
        return 0.0, path

def _map_events_via_path(template_events: List[int], path: List[Tuple[int, int]], scene_start: int, frame_ids: Optional[np.ndarray] = None) -> List[int]:
    tpl2tgt: Dict[int, int] = {}
    for i_tpl, j_tgt in path:
        # keep first occurrence to preserve monotonicity
        if i_tpl not in tpl2tgt:
            tpl2tgt[i_tpl] = j_tgt
    out: List[int] = []
    for e in template_events:
        jt = tpl2tgt.get(int(e), 0)
        if frame_ids is not None and 0 <= int(jt) < len(frame_ids):
            out.append(int(frame_ids[int(jt)]))
        else:
            out.append(scene_start + int(jt))
    return out

def _maybe_align_with_dtw(text_query: str, video_name: str, force_dtw: bool = False) -> Optional[List[int]]:
    if not config:
        return None
    if not (force_dtw or getattr(config, "USE_TRAKE_DTW", False)):
        return None
    folder = _folder_from_video(video_name)
    template_seq, template_events = _load_template_sequence()
    if template_seq is None or not isinstance(template_events, list) or len(template_events) == 0:
        return None
    scenes = _list_scenes_for_video(folder) or []
    if not scenes:
        return None
    best = (None, float("inf"), None)  # (frames, cost, scene_idx)
    metric = getattr(config, "DTW_METRIC", "cosine")
    max_cost = getattr(config, "DTW_MAX_COST", 1e6)
    for si, (s, e) in enumerate(scenes, start=1):
        scene = _load_scene_data(folder, si)
        if scene is None:
            continue
        feat, frame_ids, _ = scene
        if feat is None or not isinstance(feat, np.ndarray) or feat.ndim != 2:
            continue
        # align template to this scene's feature sequence
        cost, path = _dtw_align(template_seq, feat, metric=metric)
        if cost < best[1]:
            mapped = _map_events_via_path(template_events, path, s, frame_ids)
            best = (mapped, cost, si)
    frames, cost, _ = best
    if frames is None or cost >= max_cost:
        return None
    return frames

def align_events(
    text_query: str,
    video_name: str,
    myfaiss,                      # instance Myfaiss
    dict_image_path: Dict[int,str],
    k_per_event: int = 200,
    force_dtw: bool = False
) -> List[int]:
    """
    Trả về list frame_idx cho các event theo thứ tự thời gian trong video đã chọn.
    Chiến lược:
      - Tách event descriptions từ text_query.
      - Với mỗi event: FAISS text_search(event, k) -> lọc các keyframes thuộc folder video.
      - Chọn frame có điểm cao & đảm bảo frame tăng dần (monotonic).
    """
    # Try DTW-based alignment if configured and data available
    frames = _maybe_align_with_dtw(text_query, video_name, force_dtw=force_dtw)
    if isinstance(frames, list) and frames:
        return frames[:4]

    folder = _folder_from_video(video_name)
    events = parse_events_from_text(text_query, max_events=4)
    if not events:
        return []

    chosen_frames: List[int] = []
    last_frame = -1

    for ev in events:
        # search top-K toàn corpus
        _, ids, _, paths = myfaiss.text_search(ev, k=k_per_event)
        # filter về đúng video folder
        cand: List[Tuple[int,int,str]] = []  # (id, frame_idx, rel_path)
        for i, p in zip(ids, paths):
            p = str(p).replace("\\","/")
            # p có dạng 'L21_V001/0000.jpg'
            if p.startswith(folder + "/"):
                cand.append((int(i), _frame_idx_from_relpath(p), p))
        # sort theo similarity đã có (giữ thứ tự hiện tại), rồi enforce thứ tự thời gian
        selected = None
        for cid, fidx, rp in cand:
            if fidx > last_frame:
                selected = fidx
                break
        # nếu không có frame lớn hơn last_frame (hiếm), lấy frame đầu tiên
        if selected is None and cand:
            selected = cand[0][1]

        if selected is not None:
            chosen_frames.append(selected)
            last_frame = selected
        else:
            # không có ứng viên trong video (event này bỏ trống)
            chosen_frames.append(last_frame if last_frame >= 0 else 0)

    return chosen_frames

# ---------- (Optional) Shot detection via TransNetV2 ----------
# Bạn có thể tích hợp thật sự TransNetV2 tại đây, ví dụ:
# def detect_shots(video_path: str) -> List[Tuple[int,int]]:
#     try:
#         from transnetv2 import TransNetV2
#         # TODO: load model + run inference -> trả về list (start_frame, end_frame)
#     except Exception:
#         return []
# Sau đó, trong align_events() bạn có thể giới hạn candidate vào các shot phù hợp để nhanh & chính xác hơn.
