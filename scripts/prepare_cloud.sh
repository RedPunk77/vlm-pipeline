#!/usr/bin/env bash
# Подготовка Linux-сервера с NVIDIA, полный эксперимент здесь не запускается
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python - <<'PY'
from pathlib import Path
import torch
from huggingface_hub import snapshot_download

if not torch.cuda.is_available():
    raise SystemExit('CUDA недоступна, проверь образ PyTorch и подключённую GPU')
print('GPU:', torch.cuda.get_device_name(0))
revision='66285546d2b821cf421d4f5eb2576359d3770cd3'
cache=Path('.cache/huggingface/hub')
snapshot=snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', revision=revision, cache_dir=cache,
                           allow_patterns=['*.json','*.safetensors','*.txt'], max_workers=2)
if Path(snapshot).name != revision:
    raise SystemExit('Ревизия модели не совпала с зафиксированной')
# Локальный offline-запуск по имени модели использует именно этот снимок
refs=cache/'models--Qwen--Qwen2.5-VL-3B-Instruct/refs'
refs.mkdir(parents=True,exist_ok=True)
(refs/'main').write_text(revision)
print('Окружение и веса готовы, теперь можно измерить скорость на validation')
PY
