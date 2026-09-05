"""Загрузка весов Qwen обычным HTTP, без Xet"""
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    path = snapshot_download(
        "Qwen/Qwen2.5-VL-3B-Instruct", cache_dir=Path(".cache/huggingface/hub"),
        allow_patterns=["*.json", "*.safetensors", "*.txt"], max_workers=2,
    )
    print(f"Веса готовы: {path}")
