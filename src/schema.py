"""Общий формат ответов модели и разметки"""
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    label: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]
    kind: Literal["object", "text"]
    text: str | None = None

    @model_validator(mode="after")
    def check_region(self):
        x1, y1, x2, y2 = self.bbox
        if not self.label.strip() or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("Пустая метка или неверные координаты bbox")
        if self.kind == "text" and not (self.text and self.text.strip()):
            raise ValueError("Для текстовой области нужен непустой text")
        if self.kind == "object" and self.text is not None:
            raise ValueError("У обычного объекта text должен быть null")
        return self


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: list[Region]

    def check_size(self, size):
        width, height = size
        if any(r.bbox[2] > width or r.bbox[3] > height for r in self.objects):
            raise ValueError("Координаты выходят за границы изображения")
        return self


def parse_scene(raw: str, size) -> Scene:
    # Убираем только внешнюю Markdown обёртку, сломанный JSON не чиним молча
    value = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", value)
    if fenced:
        value = fenced.group(1)
    return Scene.model_validate(json.loads(value)).check_size(size)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def normalize(value):
    return " ".join(value.casefold().split())
