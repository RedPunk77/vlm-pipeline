"""Проверки ошибок, сопоставления и честного подсчёта метрик"""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.backends import Evidence
from src.cli import load_manifest
from src.evaluation import evaluate
from src.schema import Region, Scene, iou, parse_scene
from src.system import VisualGroundingSystem
from src.verification import verify


def region(label="cat", bbox=(10, 10, 40, 40), kind="object", text=None):
    return Region(label=label, bbox=bbox, kind=kind, text=text)


class SchemaTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 10, 10], [10, 10, 20, 20]), 0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_schema_rejects_bad_regions(self):
        for box in ([1, 1, 0, 2], [-1, 0, 4, 4], [0, 0, float("inf"), 4]):
            with self.assertRaises(ValueError):
                region(bbox=box)
        with self.assertRaises(ValueError):
            region(kind="text")
        with self.assertRaises(ValueError):
            parse_scene('{"objects": [], "verified": true}', (100, 100))
        with self.assertRaises(ValueError):
            parse_scene(Scene(objects=[region()]).model_dump_json(), (20, 20))

    def test_markdown_and_invalid_json(self):
        self.assertEqual(parse_scene('```json\n{"objects": []}\n```', (100, 100)).objects, [])
        with self.assertRaises(ValueError):
            parse_scene('Ответ: {"objects": []}', (100, 100))


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), "white")

    def test_match_corrects_box_and_deduplicates(self):
        proposal = region(bbox=(9, 9, 41, 41))
        result, audit = verify(Scene(objects=[proposal, proposal]), self.image, [Evidence(region(), 0.9)], [], box_policy="detector")
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.objects[0].bbox, region().bbox)
        self.assertEqual(sorted(x["status"] for x in audit), ["supported", "unconfirmed"])

    def test_default_preserves_box_but_still_filters(self):
        proposal = region(bbox=(9, 9, 41, 41))
        scene = Scene(objects=[proposal, region("dog")])
        result, audit = verify(scene, self.image, [Evidence(region(), 0.9)], [])
        self.assertEqual(result.objects, [proposal])
        self.assertEqual(audit[0]["evidence_bbox"], region().bbox)
        self.assertFalse(audit[0]["bbox_changed"])
        self.assertEqual(audit[1]["status"], "unconfirmed")
        self.assertEqual(scene.objects[0].bbox, proposal.bbox)

    def test_unknown_box_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            verify(Scene(objects=[]), self.image, [], [], box_policy="unknown")

    def test_wrong_class_low_confidence_and_wrong_location(self):
        for evidence in (Evidence(region("dog"), 0.9), Evidence(region(), 0.1), Evidence(region(bbox=(60, 60, 80, 80)), 0.9)):
            result, _ = verify(Scene(objects=[region()]), self.image, [evidence], [])
            self.assertEqual(result.objects, [])

    def test_text_requires_matching_text_and_location(self):
        text = region(label="text", kind="text", text="HELLO")
        for value, count in ((" hello ", 1), ("WORLD", 0)):
            result, _ = verify(Scene(objects=[text]), self.image, [], [Evidence(region(label="text", kind="text", text=value), 0.9)])
            self.assertEqual(len(result.objects), count)


class MetricsTests(unittest.TestCase):
    def test_empty_predictions_do_not_look_perfect(self):
        scene = Scene(objects=[region()]).model_dump(mode="json")
        result = evaluate([{"ground_truth": scene, "baseline": scene, "verified": {"objects": []}}])
        self.assertEqual(result["baseline"]["structured_accuracy"], 1)
        self.assertIsNone(result["verified"]["hallucination_rate"])
        self.assertEqual(result["verified"]["recall"], 0)
        self.assertEqual(result["verified"]["grounding_iou"], 0)
        self.assertEqual(result["verified"]["structured_accuracy"], 0)

    def test_duplicates_are_false_positives(self):
        truth = Scene(objects=[region()]).model_dump(mode="json")
        duplicate = Scene(objects=[region(), region()]).model_dump(mode="json")
        result = evaluate([{"ground_truth": truth, "baseline": duplicate, "verified": truth}])
        self.assertEqual(result["baseline"]["hallucination_rate"], 0.5)
        self.assertEqual(result["baseline"]["recall"], 1)

    def test_parse_failure_counts_against_accuracy(self):
        result = evaluate([{"ground_truth": {"objects": []}, "baseline": None, "verified": None}])
        self.assertEqual(result["baseline"]["schema_valid_rate"], 0)
        self.assertEqual(result["baseline"]["structured_accuracy"], 0)
        with self.assertRaises(ValueError):
            evaluate([])


class PipelineTests(unittest.TestCase):
    def test_resize_and_invalid_response(self):
        class VLM:
            def generate(self, image):
                return Scene(objects=[region(bbox=(5, 5, 20, 20))]).model_dump_json(), (50, 50)

        class Detector:
            def detect(self, image):
                return [Evidence(region(), 0.9)]

        class OCR:
            def detect(self, image):
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (100, 100)).save(path)
            system = VisualGroundingSystem(VLM(), Detector(), OCR())
            result = system.run_pipeline(path)
            self.assertEqual(result["baseline"]["objects"][0]["bbox"], [10, 10, 40, 40])
            system.vlm.generate = lambda image: ("broken", (100, 100))
            result = system.run_pipeline(path)
            self.assertIsNone(result["verified"])
            self.assertIsNotNone(result["parse_error"])
            manifest = Path(directory) / "manifest.jsonl"
            row = json.dumps({"image": "image.png", "objects": []})
            manifest.write_text(row + "\n" + row)
            with self.assertRaises(ValueError):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
