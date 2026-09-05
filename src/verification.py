"""Проверка областей по независимым источникам"""
import math

from src.schema import Scene, iou, normalize


def cv_stats(image, bbox):
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox
    crop = np.asarray(image.convert("L"))[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
    return {"contrast_std": float(crop.std()), "edge_fraction": float((cv2.Canny(crop, 60, 160) > 0).mean())}


def verify(scene, image, detections, ocr, match_iou=0.3, min_score=0.4, box_policy="preserve", recover_score=None, object_labels=None):
    if not 0 < match_iou <= 1 or not 0 <= min_score <= 1:
        raise ValueError("Пороги должны лежать в диапазоне от 0 до 1, IoU строго больше 0")
    if box_policy not in ("preserve", "detector"):
        raise ValueError("box_policy должен быть preserve или detector")
    if recover_score is not None and not 0 <= recover_score <= 1:
        raise ValueError("Порог восстановления должен лежать от 0 до 1")
    evidence = detections + ocr
    candidates = []
    for pi, proposal in enumerate(scene.objects):
        for ei, item in enumerate(evidence):
            region = item.region
            same = proposal.kind == region.kind and normalize(proposal.label) == normalize(region.label)
            if proposal.kind == "text":
                same = same and normalize(proposal.text) == normalize(region.text or "")
            overlap = iou(proposal.bbox, region.bbox)
            if same and item.score >= min_score and overlap >= match_iou:
                candidates.append((overlap, pi, ei))
    # Один результат детектора не может подтвердить несколько объектов VLM
    matches, used = {}, set()
    for overlap, pi, ei in sorted(candidates, reverse=True):
        if pi not in matches and ei not in used:
            matches[pi] = (ei, overlap)
            used.add(ei)
    kept, audit = [], []
    for pi, proposal in enumerate(scene.objects):
        stats = cv_stats(image, proposal.bbox)
        entry = {"proposal": proposal.model_dump(), "cv": stats}
        if pi in matches:
            ei, overlap = matches[pi]
            item = evidence[ei]
            box = proposal.bbox if box_policy == "preserve" else item.region.bbox
            corrected = proposal.model_copy(update={"bbox": box})
            kept.append(corrected)
            entry.update(status="supported", source="ocr" if proposal.kind == "text" else "detector", match_iou=overlap, evidence_score=item.score, evidence_bbox=item.region.bbox, box_policy=box_policy, bbox_changed=box != proposal.bbox)
        else:
            entry.update(status="unconfirmed", source=None)
        # Границы и контраст сами по себе не доказывают, что перед нами кот или машина
        audit.append(entry)
    if recover_score is not None:
        allowed = None if object_labels is None else {normalize(label) for label in object_labels}
        for ei, item in sorted(enumerate(detections), key=lambda pair: pair[1].score, reverse=True):
            region = item.region
            if ei in used or item.score < recover_score or region.kind != "object":
                continue
            if allowed is not None and normalize(region.label) not in allowed:
                continue
            if any(normalize(region.label) == normalize(existing.label) and iou(region.bbox, existing.bbox) >= match_iou for existing in kept):
                continue
            kept.append(region.model_copy())
            audit.append({"proposal": None, "status": "added_by_detector", "source": "detector",
                          "region": region.model_dump(mode="json"), "evidence_score": item.score,
                          "cv": cv_stats(image, region.bbox)})
    return Scene(objects=kept).check_size(image.size), audit
