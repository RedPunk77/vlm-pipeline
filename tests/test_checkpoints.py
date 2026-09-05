"""Проверка защиты от смешивания результатов разных экспериментов"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from src.checkpoints import file_hash, read_checkpoint, validate_config
from src.cli import save_json
from src.schema import Scene


class CheckpointTests(unittest.TestCase):
    def test_limit_can_change_but_model_cannot(self):
        validate_config({"model": "qwen", "limit": 1}, {"model": "qwen", "limit": 800, "resume": True})
        with self.assertRaises(ValueError):
            validate_config({"model": "qwen"}, {"model": "different"})

    def test_changed_image_and_incomplete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            output = Path(directory) / "result.json"
            Image.new("RGB", (20, 20)).save(image)
            scene = Scene(objects=[])
            record = {"image_sha256": file_hash(image), "image_size": [20, 20],
                      "baseline": {"objects": []}, "verified": {"objects": []},
                      "ground_truth": scene.model_dump(mode="json")}
            save_json(output, record)
            self.assertEqual(read_checkpoint(output, image, scene), record)
            Image.new("RGB", (20, 20), "white").save(image)
            with self.assertRaises(ValueError):
                read_checkpoint(output, image, scene)
            record["image_sha256"] = file_hash(image)
            record["verified"] = None
            save_json(output, record)
            with self.assertRaises(ValueError):
                read_checkpoint(output, image, scene)


class ResumeIntegrationTests(unittest.TestCase):
    def test_completed_run_resumes_without_loading_models(self):
        import io
        from unittest.mock import patch
        from src.cli import main

        class VLM:
            def generate(self, image):
                return '{"objects": []}', image.size

        class Detector:
            def detect(self, image):
                return []

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            output = Path(directory) / 'run'
            Image.new('RGB', (20, 20)).save(image)
            args = ['grounded-vlm', 'infer', str(image), '--out', str(output)]
            with patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO), \
                 patch('src.backends.QwenBackend', return_value=VLM()), \
                 patch('src.backends.YoloDetector', return_value=Detector()), \
                 patch('src.backends.EasyOCRBackend', return_value=Detector()):
                main()
            before = (output / 'outputs.json').read_bytes()
            with patch('sys.argv', args + ['--resume']), patch('sys.stdout', new_callable=io.StringIO), \
                 patch('src.backends.QwenBackend', side_effect=AssertionError('Повторная загрузка модели')), \
                 patch('src.backends.YoloDetector', side_effect=AssertionError('Повторная загрузка детектора')):
                main()
            self.assertEqual(before, (output / 'outputs.json').read_bytes())
