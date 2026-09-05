"""Пайплайн с сохранением исходного ответа и причин фильтрации"""
import time
from pathlib import Path

from PIL import Image, ImageOps

from src.schema import Scene, parse_scene
from src.verification import verify


class VisualGroundingSystem:
    def __init__(self, vlm, detector, ocr, match_iou=0.3, min_score=0.4, box_policy="preserve"):
        self.vlm = vlm
        self.detector = detector
        self.ocr = ocr
        self.match_iou = match_iou
        self.min_score = min_score
        if box_policy not in ("preserve", "detector"):
            raise ValueError("box_policy должен быть preserve или detector")
        self.box_policy = box_policy

    def run_pipeline(self, image_path):
        started = time.perf_counter()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        raw, model_size = self.vlm.generate(image)
        error = None
        try:
            baseline = parse_scene(raw, model_size)
            sx, sy = image.width / model_size[0], image.height / model_size[1]
            for region in baseline.objects:
                x1, y1, x2, y2 = region.bbox
                region.bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            baseline.check_size(image.size)
        except ValueError as exc:
            baseline = None
            error = str(exc)
        detections, texts = [], []
        if baseline is None:
            verified, audit = None, []
        else:
            detections = self.detector.detect(image)
            texts = self.ocr.detect(image) if any(r.kind == "text" for r in baseline.objects) else []
            for item in detections + texts:
                Scene(objects=[item.region]).check_size(image.size)
            verified, audit = verify(baseline, image, detections, texts, self.match_iou, self.min_score, self.box_policy)
        return {
            "image": str(Path(image_path)), "image_size": list(image.size),
            "model": getattr(self.vlm, "model_id", type(self.vlm).__name__),
            "prompt_version": getattr(self.vlm, "prompt_version", None),
            "model_revision": getattr(self.vlm, "revision", None),
            "config": {"match_iou": self.match_iou, "min_score": self.min_score, "box_policy": self.box_policy},
            "raw": raw, "parse_error": error,
            "baseline": baseline.model_dump(mode="json") if baseline is not None else None,
            "verified": verified.model_dump(mode="json") if verified is not None else None,
            "evidence": {"detector": [{"region": item.region.model_dump(mode="json"), "score": item.score} for item in detections],
                         "ocr": [{"region": item.region.model_dump(mode="json"), "score": item.score} for item in texts]},
            "audit": audit, "seconds": round(time.perf_counter() - started, 3),
        }
