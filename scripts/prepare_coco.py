"""Подготовка воспроизводимой выборки COCO val2017 без скачивания всех изображений"""
import argparse
import hashlib
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from src.cli import save_json
from src.schema import Region

ANNOTATIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(destination)


def select_rows(annotation, count, seed):
    categories = {item["id"]: item["name"] for item in annotation["categories"]}
    by_image = defaultdict(list)
    crowd = set()
    for item in annotation["annotations"]:
        by_image[item["image_id"]].append(item)
        if item.get("iscrowd", 0):
            crowd.add(item["image_id"])
    # Crowd-области требуют отдельного протокола оценки, такие изображения исключаем целиком
    eligible = sorted((row for row in annotation["images"] if row["id"] not in crowd), key=lambda row: row["id"])
    if count < 1 or count > len(eligible):
        raise ValueError(f"Нужно от 1 до {len(eligible)} изображений")
    selected = random.Random(seed).sample(eligible, count)
    rows = []
    for image in selected:
        objects = []
        for item in by_image[image["id"]]:
            x, y, width, height = item["bbox"]
            bbox = [max(0, x), max(0, y), min(image["width"], x + width), min(image["height"], y + height)]
            region = Region(label=categories[item["category_id"]], kind="object", bbox=bbox)
            objects.append(region.model_dump(mode="json"))
        rows.append({"image": f"images/{image['id']:012d}.jpg", "coco_image_id": image["id"], "objects": objects})
    return rows, sorted(categories.values())


def main():
    parser = argparse.ArgumentParser(description="Скачать выборку COCO val2017 с рамками")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/coco-smoke"))
    parser.add_argument("--annotations", type=Path, help="Готовый instances_val2017.json, если уже скачан")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count должен быть положительным")
    annotation_path = args.annotations or Path(".cache/coco/instances_val2017.json")
    if not args.annotations and not annotation_path.exists():
        archive = Path(".cache/coco/annotations_trainval2017.zip")
        print("Скачиваю архив аннотаций COCO, около 250 МБ", flush=True)
        download(ANNOTATIONS_URL, archive)
        with zipfile.ZipFile(archive) as source:
            payload = source.read("annotations/instances_val2017.json")
        annotation_path.write_bytes(payload)
    payload = annotation_path.read_bytes()
    rows, labels = select_rows(json.loads(payload), args.count, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        path = args.out / row["image"]
        download(f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{row['coco_image_id']:012d}.jpg", path)
        with Image.open(path) as image:
            image.verify()
        print(f"Изображения: {index + 1}/{len(rows)}", flush=True)
    manifest = args.out / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    save_json(args.out / "labels.json", labels)
    save_json(args.out / "dataset.json", {
        "source": "COCO val2017", "annotations_url": ANNOTATIONS_URL,
        "annotations_sha256": hashlib.sha256(payload).hexdigest(),
        "count": len(rows), "seed": args.seed, "exclude_images_with_crowd": True,
        "scope": "Только 80 классов объектов COCO, OCR здесь не оценивается",
        "image_ids": [row["coco_image_id"] for row in rows],
    })
    print(f"Набор готов: {manifest}")


if __name__ == "__main__":
    main()
