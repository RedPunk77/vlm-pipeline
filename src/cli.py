"""Запуск модели на одном изображении или на размеченном наборе"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageOps

from src.checkpoints import file_hash, read_checkpoint, validate_config
from src.evaluation import evaluate
from src.schema import Scene
from src.prompts import PROMPT_VERSION
from src.system import VisualGroundingSystem
from src.visualization import save_comparison


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path, limit=None):
    path = Path(path)
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = (path.parent / row["image"]).resolve()
        if image_path in seen:
            raise ValueError(f"Повтор изображения в наборе: {image_path}")
        seen.add(image_path)
        with Image.open(image_path) as image:
            size = ImageOps.exif_transpose(image).size
        truth = Scene.model_validate({"objects": row["objects"]}).check_size(size)
        rows.append((image_path, truth))
    if not rows:
        raise ValueError("В manifest нет изображений")
    return rows[:limit] if limit else rows


def main():
    parser = argparse.ArgumentParser(description="Visual grounding и проверка ответов Qwen")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("infer", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("input", type=Path, help="Изображение или manifest JSONL")
        child.add_argument("--resume", action="store_true", help="Продолжить прогон с теми же настройками")
        child.add_argument("--out", type=Path, default=Path("outputs") / command)
        child.add_argument("--object-labels", type=Path, help="JSON список классов для оценки только объектов")
        child.add_argument("--max-new-tokens", type=int, default=2048)
        child.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
        child.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
        child.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        child.add_argument("--detector", default="yolo11n.pt")
        child.add_argument("--languages", nargs="+", default=["en", "ru"])
        child.add_argument("--match-iou", type=float, default=0.3)
        child.add_argument("--box-policy", choices=["preserve", "detector"], default="preserve", help="Оставить рамки Qwen или заменить рамками проверяющего модуля")
        child.add_argument("--min-score", type=float, default=0.4)
        if command == "evaluate":
            child.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 < args.match_iou <= 1 or not 0 <= args.min_score <= 1:
        parser.error("Проверь пороги IoU и confidence")
    if args.command == "evaluate" and args.limit is not None and args.limit < 1:
        parser.error("limit должен быть положительным")
    if args.command == "evaluate":
        rows = load_manifest(args.input, args.limit)
    else:
        with Image.open(args.input) as image:
            image.verify()
        rows = [(args.input, None)]
    if args.max_new_tokens < 1 or args.max_pixels < 256 * 28 * 28:
        parser.error("max-new-tokens должен быть положительным, max-pixels не меньше 200704")
    labels = None
    if args.object_labels:
        labels = json.loads(args.object_labels.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label.strip() for label in labels):
            parser.error("object-labels должен содержать непустой JSON список названий классов")
        if any(r.kind != "object" or r.label not in labels for _, truth in rows if truth is not None for r in truth.objects):
            parser.error("В разметке есть области вне выбранных классов объектов")
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    Path(".cache/ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(Path(".cache/easyocr").resolve()))
    from src.backends import EasyOCRBackend, QwenBackend, YoloDetector

    args.out.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["prompt_version"] = PROMPT_VERSION
    config["object_labels"] = labels
    config["input"] = str(args.input.resolve())
    config["input_sha256"] = file_hash(args.input)
    config_path = args.out / "config.json"
    if args.resume:
        if not config_path.exists():
            parser.error("Для resume нужен config.json предыдущего прогона")
        validate_config(json.loads(config_path.read_text(encoding="utf-8")), config)
    elif config_path.exists() or any(args.out.glob("[0-9][0-9][0-9][0-9][0-9].json")):
        parser.error("В папке уже есть прогон, выбери новую --out или добавь --resume")
    cached = {}
    if args.resume:
        # Проверяем весь доступный кеш до загрузки тяжёлых моделей
        for index, (image_path, truth) in enumerate(rows):
            path = args.out / f"{index:05d}.json"
            if path.exists():
                cached[index] = read_checkpoint(path, image_path, truth)
    save_json(config_path, config)
    system = None
    records = []
    for index, (image_path, truth) in enumerate(rows):
        if index in cached:
            result = cached[index]
            print(f"Из кеша {index + 1}/{len(rows)}: {image_path.name}", flush=True)
        else:
            if system is None:
                print("Загружаю Qwen и детектор", flush=True)
                system = VisualGroundingSystem(QwenBackend(args.model, args.device, args.max_new_tokens, args.max_pixels, labels), YoloDetector(args.detector), EasyOCRBackend(args.languages), args.match_iou, args.min_score, args.box_policy)
            result = system.run_pipeline(image_path)
            result["image_sha256"] = file_hash(image_path)
            if truth is not None:
                result["ground_truth"] = truth.model_dump(mode="json")
            save_json(args.out / f"{index:05d}.json", result)
            print(f"Обработано {index + 1}/{len(rows)}: {image_path.name}", flush=True)
        records.append(result)
        save_comparison(image_path, result, args.out / f"{index:05d}.jpg")
    save_json(args.out / "outputs.json", records)
    if args.command == "evaluate":
        report = evaluate(records)
        report["manifest_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        save_json(args.out / "metrics_report.json", report)
    print(f"Результаты сохранены в {args.out}")


if __name__ == "__main__":
    main()
