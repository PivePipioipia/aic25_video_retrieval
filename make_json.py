from pathlib import Path
import json
import config

ROOT = Path(config.KEYFRAMES_DIR)

EXTS = {".jpg", ".jpeg", ".png"}
imgs = [p for p in ROOT.rglob("*") if p.suffix.lower() in EXTS]

imgs = sorted(imgs, key=lambda p: (p.parent.as_posix(), p.name))

mapping = {i: str(p.relative_to(ROOT).as_posix()) for i, p in enumerate(imgs)}

with open(config.IMAGE_MAP_JSON, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f" Đã tạo {config.IMAGE_MAP_JSON} với {len(mapping)} ảnh")
if imgs:
    print("Ví dụ:", 0, "->", str(imgs[0].relative_to(ROOT).as_posix()))
