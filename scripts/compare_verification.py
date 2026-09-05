"""Сравнение политик рамок на одних ответах Qwen и одних подтверждениях"""
import argparse
from copy import deepcopy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageOps
from src.backends import Evidence, EasyOCRBackend, YoloDetector
from src.checkpoints import file_hash
from src.cli import save_json
from src.evaluation import evaluate
from src.schema import Region, Scene
from src.verification import verify
from src.visualization import save_comparison
from scripts.report_results import build_report


def unpack_evidence(value, size):
    evidence = [Evidence(Region.model_validate(item['region']), item['score']) for item in value]
    for item in evidence:
        Scene(objects=[item.region]).check_size(size)
        if not 0 <= item.score <= 1:
            raise ValueError('Уверенность подтверждения вне диапазона от 0 до 1')
    return evidence


def replay(record, image, detections, texts, policy):
    result = deepcopy(record)
    result['config']['box_policy'] = policy
    if 'seconds' in result:
        result['original_pipeline_seconds'] = result.pop('seconds')
    started = time.perf_counter()
    if record['baseline'] is not None:
        scene = Scene.model_validate(record['baseline']).check_size(image.size)
        verified, audit = verify(scene, image, detections, texts,
                                 record['config']['match_iou'], record['config']['min_score'], policy)
        result['verified'] = verified.model_dump(mode='json')
        result['audit'] = audit
    result['reverification_seconds'] = round(time.perf_counter() - started, 4)
    result['evidence'] = {
        'detector': [{'region': item.region.model_dump(mode='json'), 'score': item.score} for item in detections],
        'ocr': [{'region': item.region.model_dump(mode='json'), 'score': item.score} for item in texts],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Сравнить preserve и detector без повторного вызова Qwen')
    parser.add_argument('input', type=Path, help='Сохранённый outputs.json с разметкой')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--detector', default='.cache/yolo11n.pt')
    parser.add_argument('--languages', nargs='+', default=['en', 'ru'])
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding='utf-8'))
    evaluate(records)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Папка результата не пустая, выбери новую --out')
    # Сначала проверяем входные файлы, потом запускаем проверяющие модули
    for record in records:
        if file_hash(record['image']) != record['image_sha256']:
            raise ValueError('Изображение изменилось: ' + record['image'])
    Path('.cache/ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('.cache/ultralytics').resolve()))
    os.environ.setdefault('EASYOCR_MODULE_PATH', str(Path('.cache/easyocr').resolve()))
    detector, ocr = None, None
    groups = {'preserve': [], 'detector': []}
    evidence_sources = []
    for index, record in enumerate(records):
        with Image.open(record['image']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        if list(image.size) != record['image_size']:
            raise ValueError('Размер изображения не совпал с сохранённым')
        detections, texts = [], []
        if record['baseline'] is None:
            origin = 'not_needed_parse_failure'
        elif 'evidence' in record:
            detections = unpack_evidence(record['evidence']['detector'], image.size)
            texts = unpack_evidence(record['evidence']['ocr'], image.size)
            origin = 'saved'
        else:
            # В старых отчётах подтверждения не сохранялись, получаем их один раз для обоих режимов
            if detector is None:
                detector = YoloDetector(args.detector)
            detections = detector.detect(image)
            if any(item['kind'] == 'text' for item in record['baseline']['objects']):
                if ocr is None:
                    ocr = EasyOCRBackend(args.languages)
                texts = ocr.detect(image)
            for item in detections + texts:
                Scene(objects=[item.region]).check_size(image.size)
            origin = 'recomputed'
        evidence_sources.append(origin)
        for policy, group in groups.items():
            result = replay(record, image, detections, texts, policy)
            result['evidence_origin'] = origin
            group.append(result)
            save_json(args.out / policy / f'{index:05d}.json', result)
            save_comparison(record['image'], result, args.out / policy / f'{index:05d}.jpg')
        print(f'Сравнено {index + 1}/{len(records)}', flush=True)
    historical = evaluate(records)
    comparison = {'dataset_size': len(records), 'baseline': historical['baseline'],
                  'historical_verified': historical['verified']}
    for policy, group in groups.items():
        report = evaluate(group)
        comparison[policy] = report['verified']
        save_json(args.out / policy / 'outputs.json', group)
        save_json(args.out / policy / 'metrics_report.json', report)
        (args.out / policy / 'report.md').write_text(build_report(group), encoding='utf-8')
    save_json(args.out / 'comparison.json', comparison)
    save_json(args.out / 'config.json', {
        'input': str(args.input.resolve()), 'input_sha256': file_hash(args.input),
        'detector': args.detector, 'detector_sha256': file_hash(args.detector) if Path(args.detector).is_file() else None,
        'languages': args.languages, 'evidence_sources': evidence_sources,
        'note': 'Ответы Qwen неизменны, оба режима получают одинаковые подтверждения, разметка используется только для метрик',
    })
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
