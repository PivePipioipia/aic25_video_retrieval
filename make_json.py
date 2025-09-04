# make_json.py
from pathlib import Path
import json
import config

ROOT = Path(config.KEYFRAMES_DIR)  # ví dụ: F:\AIC2025_DATASET\keyframes

# lấy mọi ảnh trong toàn bộ cây thư mục (jpg/jpeg/png)
EXTS = {".jpg", ".jpeg", ".png"}
imgs = [p for p in ROOT.rglob("*") if p.suffix.lower() in EXTS]

# sắp xếp ổn định theo thư mục và tên file
imgs = sorted(imgs, key=lambda p: (p.parent.as_posix(), p.name))

# ánh xạ id -> đường dẫn tương đối (vd "L21_V001/0000.jpg")
mapping = {i: str(p.relative_to(ROOT).as_posix()) for i, p in enumerate(imgs)}

with open(config.IMAGE_MAP_JSON, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"✅ Đã tạo {config.IMAGE_MAP_JSON} với {len(mapping)} ảnh")
if imgs:
    print("Ví dụ:", 0, "->", str(imgs[0].relative_to(ROOT).as_posix()))
