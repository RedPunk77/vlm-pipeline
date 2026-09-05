"""Фиксированные непересекающиеся validation и test для эксперимента COCO800"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from scripts.prepare_coco import download, select_rows
from src.checkpoints import file_hash
from src.cli import save_json

EXCLUDED_IDS = [118515, 22935, 286507]


def make_splits(annotation, validation_count=20, test_count=800, seed=42):
    if validation_count < 1 or test_count < 1:
        raise ValueError('Обе выборки должны быть непустыми')
    filtered = dict(annotation)
    filtered['images'] = [item for item in annotation['images'] if item['id'] not in EXCLUDED_IDS]
    rows, labels = select_rows(filtered, validation_count + test_count, seed)
    return rows[:validation_count], rows[validation_count:], labels


def main():
    parser = argparse.ArgumentParser(description='Зафиксировать 20 validation и 800 test изображений COCO')
    parser.add_argument('--annotations', type=Path, default=Path('.cache/coco/instances_val2017.json'))
    parser.add_argument('--out', type=Path, default=Path('data/coco800'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-count', type=int, default=20)
    parser.add_argument('--test-count', type=int, default=800)
    parser.add_argument('--download', choices=['none', 'validation', 'all'], default='all')
    args = parser.parse_args()
    validation, test, labels = make_splits(json.loads(args.annotations.read_text()), args.validation_count, args.test_count, args.seed)
    metadata = {'dataset': 'COCO val2017', 'seed': args.seed, 'excluded_development_ids': EXCLUDED_IDS,
                'exclude_images_with_crowd': True, 'annotations_sha256': file_hash(args.annotations),
                'validation_ids': [row['coco_image_id'] for row in validation],
                'test_ids': [row['coco_image_id'] for row in test],
                'scope': 'Только объекты COCO, качество OCR в этом эксперименте не оценивается'}
    metadata_path = args.out / 'splits.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('В папке уже зафиксирована другая выборка')
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(metadata_path, metadata)
    save_json(args.out / 'labels.json', labels)
    for name, rows in [('validation', validation), ('test', test)]:
        value = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n'
        (args.out / f'{name}.jsonl').write_text(value, encoding='utf-8')
    rows = [] if args.download == 'none' else validation if args.download == 'validation' else validation + test

    def fetch(row):
        target = args.out / row['image']
        for attempt in range(3):
            try:
                download(f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{row['coco_image_id']:012d}.jpg", target)
                with Image.open(target) as image:
                    image.verify()
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(fetch, rows), 1):
            if index % 20 == 0 or index == len(rows):
                print(f'Скачано и проверено {index}/{len(rows)}', flush=True)
    print(f'Выборки зафиксированы: validation={len(validation)}, test={len(test)}', flush=True)


if __name__ == '__main__':
    main()
