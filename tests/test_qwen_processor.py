"""Регрессия повторного ресайза, выполняется при наличии локального процессора"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from src.backends import QwenBackend


class QwenProcessorTests(unittest.TestCase):
    def test_actual_processor_keeps_prompt_coordinate_system(self):
        folders = list(Path('.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots').glob('*'))
        if not folders or not all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'qwen_vl_utils')):
            self.skipTest('Нет локального процессора или зависимостей моделей')
        from transformers import AutoProcessor
        import torch

        try:
            processor = AutoProcessor.from_pretrained(folders[0], local_files_only=True, use_fast=False)
        except OSError:
            self.skipTest('Процессор ещё не скачан полностью')
        backend = QwenBackend.__new__(QwenBackend)
        backend.processor = processor
        backend.max_pixels = 200704
        backend.object_labels = ['cat']
        backend.max_new_tokens = 1
        captured = {}

        def generate(**kwargs):
            captured['grid'] = kwargs['image_grid_thw'][0].tolist()
            eos = torch.tensor([[processor.tokenizer.eos_token_id]])
            return torch.cat([kwargs['input_ids'], eos], dim=1)

        backend.model = SimpleNamespace(device=torch.device('cpu'), generate=generate)
        _, size = backend.generate(Image.new('RGB', (640, 299)))
        _, grid_h, grid_w = captured['grid']
        patch = processor.image_processor.patch_size
        self.assertEqual((grid_w * patch, grid_h * patch), size)
