"""Отдельная проверка YOLO и OCR на настоящем изображении"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / ".cache/ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
os.environ.setdefault("EASYOCR_MODULE_PATH", str(ROOT / ".cache/easyocr"))

from PIL import Image, ImageOps
from src.backends import EasyOCRBackend, YoloDetector
from src.cli import save_json
from src.verification import cv_stats


def main():
    parser = argparse.ArgumentParser(description="Проверить детектор и OCR без Qwen")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/verifiers.json"))
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    Path(".cache").mkdir(exist_ok=True)
    detector = YoloDetector(".cache/yolo11n.pt")
    detections = detector.detect(image)
    print(f"YOLO: {len(detections)} объектов", flush=True)
    texts = EasyOCRBackend().detect(image)
    print(f"OCR: {len(texts)} текстовых областей", flush=True)
    save_json(args.out, {
        "image": str(args.image), "note": "Результаты проверяющих модулей, не ответ Qwen",
        "detections": [{"region": item.region.model_dump(mode="json"), "score": item.score,
                        "cv": cv_stats(image, item.region.bbox)} for item in detections],
        "ocr": [{"region": item.region.model_dump(mode="json"), "score": item.score} for item in texts],
    })


if __name__ == "__main__":
    main()
