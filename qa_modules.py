# qa_modules.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional
from PIL import Image

import config

# ====== Lazy singletons ======
_OCR = None
_YOLO = None
_BLIP_PROCESSOR = None
_BLIP_MODEL = None

def _on_device():
    return "cuda" if config.USE_GPU else "cpu"

# ---------- OCR (PaddleOCR) ----------
def _load_ocr():
    global _OCR
    if _OCR is not None:
        return _OCR
    try:
        from paddleocr import PaddleOCR
        # NOTE: lang='en' hoạt động ổn; nếu bạn có checkpoint 'vi' hãy đổi lang='vi'
        _OCR = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=config.USE_GPU)
    except Exception as e:
        _OCR = e  # giữ lỗi để báo lại
    return _OCR

def _run_ocr(abs_image_path: str) -> Optional[str]:
    ocr = _load_ocr()
    if isinstance(ocr, Exception):
        return None
    try:
        res = ocr.ocr(abs_image_path, cls=True)
        # Gộp text thành 1 dòng ngắn gọn
        texts = []
        for page in res:
            for line in page:
                txt = line[1][0]
                if txt: texts.append(str(txt))
        return " ".join(texts)[:512] if texts else None
    except Exception:
        return None

# ---------- YOLOv8 (ultralytics) ----------
def _load_yolo():
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    try:
        from ultralytics import YOLO
        # model nhẹ, nhanh; đổi 'yolov8m.pt' nếu cần chính xác hơn
        _YOLO = YOLO("yolov8n.pt")
        _YOLO.to(_on_device())
    except Exception as e:
        _YOLO = e
    return _YOLO

def _run_detect_count(abs_image_path: str) -> Optional[Dict[str, int]]:
    yolo = _load_yolo()
    if isinstance(yolo, Exception):
        return None
    try:
        res = yolo.predict(abs_image_path, conf=0.35, verbose=False, device=_on_device())
        if not res:
            return None
        cls_names = res[0].names
        counts: Dict[str,int] = {}
        for c in res[0].boxes.cls.tolist():
            label = cls_names.get(int(c), str(int(c)))
            counts[label] = counts.get(label, 0) + 1
        return counts if counts else None
    except Exception:
        return None

# ---------- BLIP Caption (transformers) ----------
def _load_blip():
    global _BLIP_PROCESSOR, _BLIP_MODEL
    if _BLIP_MODEL is not None:
        return _BLIP_PROCESSOR, _BLIP_MODEL
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _BLIP_MODEL.to(_on_device())
    except Exception as e:
        _BLIP_PROCESSOR, _BLIP_MODEL = e, e
    return _BLIP_PROCESSOR, _BLIP_MODEL

def _run_caption(abs_image_path: str) -> Optional[str]:
    proc, mdl = _load_blip()
    if isinstance(proc, Exception) or isinstance(mdl, Exception):
        return None
    try:
        image = Image.open(abs_image_path).convert("RGB")
        inputs = proc(image, return_tensors="pt").to(_on_device())
        out = mdl.generate(**inputs, max_length=30)
        text = proc.decode(out[0], skip_special_tokens=True)
        return text
    except Exception:
        return None

# ---------- Public API ----------
def suggest_answer_bundle(abs_image_path: str, question_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Trả về gợi ý Answer cho Q&A dựa trên ảnh:
      - ocr: text OCR (nếu có)
      - count: dict đếm object (person, car, ...)
      - caption: mô tả ảnh
      - hints: list gợi ý trích từ count/caption
    """
    ocr = _run_ocr(abs_image_path)
    counts = _run_detect_count(abs_image_path)
    caption = _run_caption(abs_image_path)

    hints = []
    if counts:
        if 'person' in counts:
            hints.append(f"số người: {counts['person']}")
        # thêm vài lớp hay gặp
        for key in ['car','bus','bicycle','motorcycle']:
            if key in counts:
                hints.append(f"{key}: {counts[key]}")
    if caption:
        hints.append(f"mô tả: {caption}")

    return {
        "ocr": ocr,
        "count": counts,
        "caption": caption,
        "hints": hints
    }
