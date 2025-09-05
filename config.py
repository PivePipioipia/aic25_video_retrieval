import os

# ========== ROOTS ==========
DATASET_ROOT = os.getenv("DATASET_ROOT", r"D:\AIC_DATA1_fromKG")

# ========== DATA PATHS ==========
# Dùng frames và map bạn tự tạo
KEYFRAMES_DIR = os.path.join(DATASET_ROOT, "frames")
CLIP_DIR      = os.path.join(DATASET_ROOT, "clip-features")
MAP_JSON_DIR  = os.path.join(DATASET_ROOT, "map_idx_clean")

# ========== APP INPUT/OUTPUT FILES ==========
IMAGE_MAP_JSON   = os.path.join(os.getcwd(), "image_path.json")
FAISS_INDEX_BIN  = os.path.join(os.getcwd(), "faiss_normal_ViT.bin")
FEATURES_ALL_NPY = os.path.join(os.getcwd(), "features_all.npy")
RESULTS_XLSX     = os.path.join(os.getcwd(), "results.xlsx")

# Vocab cho Places365
PLACES_VOCAB_FILE = os.path.join(DATASET_ROOT, "vocab", "categories_places365.txt")

# Thư mục chứa các JSON kết quả Places
PLACES_JSON_DIR = r"D:\AIC_DATA1_fromKG\places"

# ========== FAISS SETTINGS ==========
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX", "flat")
FAISS_NLIST      = int(os.getenv("FAISS_NLIST", "4096"))
FAISS_NPROBE     = int(os.getenv("FAISS_NPROBE", "16"))

# ========== RUNTIME ==========
USE_GPU = os.getenv("USE_GPU", "0") == "1"  # không có GPU thì mặc định là False

# ========== MAPPINGS ==========
KEYFRAME_FRAME_MAP_JSON = os.getenv("KEYFRAME_FRAME_MAP_JSON", "")

# ========== TRAKE ==========
USE_TRAKE_DTW = os.getenv("USE_TRAKE_DTW", "0") == "1"
TRAKE_DATA_DIR = os.getenv("TRAKE_DATA_DIR", os.path.join(DATASET_ROOT, "TRAKE_DATA"))
TEMPLATE_POSE_NPY    = os.getenv("TEMPLATE_POSE_NPY", os.path.join(TRAKE_DATA_DIR, "templates", "high_jump_pose.npy"))
TEMPLATE_EVENTS_JSON = os.getenv("TEMPLATE_EVENTS_JSON", os.path.join(TRAKE_DATA_DIR, "templates", "high_jump_events.json"))
POSE_BACKEND = os.getenv("POSE_BACKEND", "mediapipe")
DTW_METRIC    = os.getenv("DTW_METRIC", "cosine")  # cosine | l2
DTW_MAX_COST  = float(os.getenv("DTW_MAX_COST", "1e6"))
REFINE_WINDOW = int(os.getenv("REFINE_WINDOW", "10"))
FPS           = float(os.getenv("FPS", "30"))
