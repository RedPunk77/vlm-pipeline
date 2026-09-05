"""Синтетическое демо без Qwen, OCR и скачивания весов"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw
from src.backends import Evidence
from src.cli import save_json
from src.evaluation import evaluate
from src.schema import Region
from src.system import VisualGroundingSystem


class DemoVLM:
    model_id = "synthetic-demo-not-qwen"

    def __init__(self, answer):
        self.answer = answer

    def generate(self, image):
        return json.dumps(self.answer), image.size


class ColorDetector:
    def detect(self, image):
        import cv2
        import numpy as np

        rgb = np.asarray(image)
        mask = ((rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)).astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            results.append(Evidence(Region(label="rectangle", kind="object", bbox=(x, y, x + w, y + h)), 1.0))
        return results


class EmptyOCR:
    def detect(self, image):
        return []


def main(count, out_dir, seed=42):
    if count < 1:
        raise ValueError("count должен быть положительным")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records, manifest = [], []
    for index in range(count):
        x, y = rng.randint(30, 100), rng.randint(30, 90)
        box = [x, y, x + rng.randint(55, 100), y + rng.randint(50, 85)]
        truth = {"objects": [{"label": "rectangle", "kind": "object", "bbox": box, "text": None}]}
        image = Image.new("RGB", (256, 224), "white")
        ImageDraw.Draw(image).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=(240, 30, 30))
        path = out / f"image_{index:04d}.png"
        image.save(path)
        noisy_box = [max(0, box[0] - rng.randint(0, 15)), max(0, box[1] - rng.randint(0, 15)), box[2] + rng.randint(0, 15), box[3] + rng.randint(0, 15)]
        answer = {"objects": [{"label": "rectangle", "kind": "object", "bbox": noisy_box, "text": None}]}
        if rng.random() < 0.35:
            answer["objects"].append({"label": "cat", "kind": "object", "bbox": [5, 5, 25, 25], "text": None})
        system = VisualGroundingSystem(DemoVLM(answer), ColorDetector(), EmptyOCR(), box_policy="detector")
        result = system.run_pipeline(path)
        result["ground_truth"] = truth
        records.append(result)
        manifest.append({"image": path.name, **truth})
    save_json(out / "outputs.json", records)
    report = evaluate(records)
    report.update(mode="synthetic_demo", seed=seed, note="Проверка кода на прямоугольниках, не метрики Qwen")
    save_json(out / "metrics_report.json", report)
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in manifest) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Демо сохранено в {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Синтетическое демо visual grounding")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", default="examples/demo_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.count, args.out, args.seed)
