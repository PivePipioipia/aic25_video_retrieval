import clip
import torch
import numpy as np

# keyword extraction (optional)
try:
    from underthesea import pos_tag
except ImportError:
    pos_tag = None

# sentence-BERT hybrid (optional)
try:
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
except ImportError:
    sbert_model = None

# translation
from utils.query_processing import Translation
translator = Translation()


def encode_text(query: str, model, device="cpu"):
    """Encode 1 câu ngắn bằng CLIP"""
    # 🔑 Dịch sang tiếng Anh trước
    query = translator(query)

    tokens = clip.tokenize(query, truncate=True).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


def encode_long_query(query: str, model, device="cpu"):
    """Chia query dài thành nhiều câu → lấy trung bình embedding"""
    # 🔑 Dịch sang tiếng Anh trước
    query = translator(query)

    sentences = [s.strip() for s in query.replace(";", ".").split(",") if s.strip()]
    vecs = []
    for s in sentences:
        tokens = clip.tokenize(s, truncate=True).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            vecs.append(emb.cpu().numpy())
    if not vecs:
        return None
    mean_vec = np.mean(vecs, axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec, keepdims=True)
    return mean_vec.reshape(1, -1).astype(np.float32)


def extract_keywords(query: str):
    """Rút keywords bằng underthesea (nếu có)"""
    if pos_tag is None:
        return query
    tagged = pos_tag(query)
    keywords = [w for w, t in tagged if t in ["N", "Np", "A"]]
    return " ".join(keywords)


def encode_hybrid(query: str, clip_model, device="cpu", alpha=0.7):
    """Hybrid CLIP + SBERT (nếu có)"""
    # 🔑 CLIP: dịch sang tiếng Anh trước
    clip_vec = encode_long_query(query, clip_model, device)

    if sbert_model is None:
        return clip_vec

    # SBERT: giữ nguyên tiếng Việt gốc để tận dụng semantic
    sbert_vec = sbert_model.encode([query], normalize_embeddings=True)

    # Normalize lại tổng hợp
    hybrid_vec = alpha * clip_vec + (1 - alpha) * sbert_vec
    hybrid_vec = hybrid_vec / np.linalg.norm(hybrid_vec, keepdims=True)
    return hybrid_vec.astype(np.float32)
