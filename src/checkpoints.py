"""Продолжение прогона без смешивания разных настроек и входных данных"""
import hashlib
import json
from pathlib import Path

from src.schema import Scene


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(previous, current):
    # Можно расширить limit, но менять модель, пороги или сам набор нельзя
    ignored = {"resume", "out", "limit"}
    left = {key: value for key, value in previous.items() if key not in ignored}
    right = {key: value for key, value in current.items() if key not in ignored}
    if left != right:
        changed = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
        raise ValueError("Настройки отличаются от прошлого прогона: " + ", ".join(changed))


def read_checkpoint(path, image_path, truth):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("image_sha256") != file_hash(image_path):
        raise ValueError(f"Изображение изменилось: {image_path}")
    expected = truth.model_dump(mode="json") if truth is not None else None
    if record.get("ground_truth") != expected:
        raise ValueError(f"Разметка изменилась: {image_path}")
    for stage in ("baseline", "verified"):
        if stage not in record:
            raise ValueError(f"Неполный результат: отсутствует {stage}")
        if record[stage] is not None:
            Scene.model_validate(record[stage]).check_size(record["image_size"])
    if (record["baseline"] is None) != (record["verified"] is None):
        raise ValueError("Неполный результат: выполнены не все этапы")
    if record["baseline"] is None and not record.get("parse_error"):
        raise ValueError("Пустой результат без причины ошибки")
    return record
