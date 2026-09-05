"""Метрики по разметке, а не по мнению проверяющей модели"""
from src.schema import Scene, iou, normalize


def evaluate(records, threshold=0.5):
    if not records:
        raise ValueError("Нельзя оценить пустой набор")
    if not 0 < threshold <= 1:
        raise ValueError("Порог IoU должен быть больше 0 и не больше 1")
    report = {"dataset_size": len(records), "iou_threshold": threshold}
    for stage in ("baseline", "verified"):
        predicted = ground_truth = matched = valid = exact = 0
        overlap_sum = 0.0
        for record in records:
            truth = Scene.model_validate(record["ground_truth"]).objects
            ground_truth += len(truth)
            value = record[stage]
            if value is None:
                continue
            proposals = Scene.model_validate(value).objects
            valid += 1
            predicted += len(proposals)
            candidates = []
            for pi, proposal in enumerate(proposals):
                for ti, target in enumerate(truth):
                    same = proposal.kind == target.kind and normalize(proposal.label) == normalize(target.label)
                    if proposal.kind == "text":
                        same = same and normalize(proposal.text) == normalize(target.text)
                    overlap = iou(proposal.bbox, target.bbox)
                    if same and overlap > 0:
                        candidates.append((overlap, pi, ti))
            seen_p, seen_t = set(), set()
            image_matches = 0
            # Жадное сопоставление по IoU, каждая рамка используется один раз
            for overlap, pi, ti in sorted(candidates, reverse=True):
                if pi in seen_p or ti in seen_t:
                    continue
                seen_p.add(pi)
                seen_t.add(ti)
                overlap_sum += overlap
                image_matches += overlap >= threshold
            matched += image_matches
            exact += image_matches == len(proposals) == len(truth)
        report[stage] = {
            "hallucination_rate": (predicted - matched) / predicted if predicted else None,
            "grounding_iou": overlap_sum / ground_truth if ground_truth else None,
            "structured_accuracy": exact / len(records),
            "schema_valid_rate": valid / len(records),
            "recall": matched / ground_truth if ground_truth else None,
            "predicted_objects": predicted, "ground_truth_objects": ground_truth,
            "matched_objects": matched,
        }
    return report
