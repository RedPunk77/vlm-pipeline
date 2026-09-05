"""Реальные вызовы Qwen, детектора и OCR, веса загружаются только при запуске"""
from dataclasses import dataclass
from pathlib import Path

from src.schema import Region
from src.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt


@dataclass
class Evidence:
    region: Region
    score: float


class QwenBackend:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="auto", max_new_tokens=2048, max_pixels=1280 * 28 * 28, object_labels=None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_id = model_id
        self.max_pixels = max_pixels
        self.object_labels = object_labels
        self.max_new_tokens = max_new_tokens
        if Path(model_id).is_dir():
            model_path = model_id
        else:
            from huggingface_hub import snapshot_download

            # Сначала дожидаемся всего снимка, затем работаем без сетевых запросов процессора
            model_path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"], max_workers=2)
        self.revision = Path(model_path).name
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False, min_pixels=256 * 28 * 28, max_pixels=self.max_pixels)
        kwargs = {"torch_dtype": "auto"}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True, **kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    prompt_version = PROMPT_VERSION

    def generate(self, image):
        import torch
        from qwen_vl_utils import process_vision_info

        # Координаты запрашиваем в размере, который увидит vision encoder
        messages = [{"role": "user", "content": [{"type": "image", "image": image, "min_pixels": 256 * 28 * 28, "max_pixels": self.max_pixels}]}]
        images, videos = process_vision_info(messages)
        width, height = images[0].size
        prompt = build_prompt(width, height, self.object_labels)
        messages[0]["content"].append({"type": "text", "text": prompt})
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt", do_resize=False).to(self.model.device)
        # Изображение уже подготовлено qwen-vl-utils, второй ресайз сдвинет систему координат
        _, grid_h, grid_w = inputs.image_grid_thw[0].tolist()
        patch = self.processor.image_processor.patch_size
        if (grid_w * patch, grid_h * patch) != (width, height):
            raise ValueError("Размер vision grid не совпал с координатами запроса")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return raw, (width, height)


class YoloDetector:
    def __init__(self, weights="yolo11n.pt", confidence=0.25):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = confidence

    def detect(self, image):
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        return [Evidence(Region(label=result.names[int(cls)], kind="object", bbox=box), float(score))
                for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]


class EasyOCRBackend:
    def __init__(self, languages=("en", "ru"), gpu=False):
        self.languages = list(languages)
        self.gpu = gpu
        self.reader = None

    def detect(self, image):
        import numpy as np

        if self.reader is None:
            import easyocr

            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, model_storage_directory=".cache/easyocr", verbose=False)
        evidence = []
        for polygon, text, confidence in self.reader.readtext(np.asarray(image)):
            xs, ys = zip(*polygon)
            width, height = image.size
            box = (max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys)))
            if text.strip() and box[2] > box[0] and box[3] > box[1]:
                evidence.append(Evidence(Region(label="text", kind="text", bbox=box, text=text), float(confidence)))
        return evidence
