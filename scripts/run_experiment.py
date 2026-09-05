"""Полный эксперимент: validation, выбор политики, затем фиксированный test"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageOps
from src.checkpoints import file_hash
from src.cli import save_json
from src.evaluation import evaluate
from src.schema import Scene
from src.verification import verify
from scripts.compare_verification import unpack_evidence
from scripts.report_results import build_report

VARIANTS = [
    {'name': 'filter_preserve', 'box_policy': 'preserve', 'recover_score': None},
    {'name': 'filter_detector', 'box_policy': 'detector', 'recover_score': None},
    {'name': 'recover_060', 'box_policy': 'preserve', 'recover_score': 0.6},
    {'name': 'recover_080', 'box_policy': 'preserve', 'recover_score': 0.8},
]


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(list((ROOT / 'src').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def selection_score(metrics):
    # Критерий фиксируется до первого просмотра validation-результатов
    return (0.4 * metrics['structured_accuracy'] + 0.4 * (metrics['grounding_iou'] or 0)
            + 0.2 * (metrics['recall'] or 0) - 0.2 * (metrics['hallucination_rate'] or 0))


def main():
    parser = argparse.ArgumentParser(description='Запустить validation и тест на 800 изображениях')
    parser.add_argument('--data', type=Path, default=Path('data/coco800'))
    parser.add_argument('--out', type=Path, default=Path('outputs/experiment800'))
    parser.add_argument('--device', choices=['mps', 'cuda', 'cpu', 'auto'], default='mps')
    parser.add_argument('--max-new-tokens', type=int, default=1536)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    split = json.loads((args.data / 'splits.json').read_text())
    if len(split['test_ids']) != 800 or len(split['validation_ids']) != 20:
        raise ValueError('Этот протокол требует ровно 20 validation и 800 test изображений')
    if set(split['test_ids']) & set(split['validation_ids']):
        raise ValueError('Выборки пересекаются')
    qwen_cache = ROOT / '.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct'
    qwen_revision = (qwen_cache / 'refs/main').read_text().strip()
    protocol = {
        'qwen_model': 'Qwen/Qwen2.5-VL-3B-Instruct', 'qwen_revision': qwen_revision,
        'detector_sha256': file_hash(ROOT / '.cache/yolo11n.pt'),
        'split_sha256': file_hash(args.data / 'splits.json'),
        'validation_manifest_sha256': file_hash(args.data / 'validation.jsonl'),
        'test_manifest_sha256': file_hash(args.data / 'test.jsonl'),
        'source_sha256': source_hash(), 'device': args.device, 'max_pixels': 200704,
        'max_new_tokens': args.max_new_tokens, 'match_iou': 0.3, 'min_score': 0.4,
        'variants': VARIANTS,
        'selection': '0.4 * structured_accuracy + 0.4 * grounding_iou + 0.2 * recall - 0.2 * hallucination_rate',
        'tie_break': 'Порядок variants, при равенстве выбирается более ранний',
        'versions': {name: version(name) for name in ['torch', 'transformers', 'ultralytics', 'pydantic', 'numpy']},
        'scope': 'Объекты COCO, OCR не оценивается, уже разобранные изображения исключены',
    }
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists():
        if not args.resume:
            parser.error('Эксперимент уже есть, добавь --resume')
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Протокол или код изменился, нужна новая папка эксперимента')
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error('Папка не пустая, выбери другую --out')
    save_json(protocol_path, protocol)

    def status(phase, **extra):
        save_json(args.out / 'status.json', {'phase': phase, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra})
        print(phase, flush=True)

    def run_split(name, variant):
        folder = args.out / name
        argv = [sys.executable, '-m', 'src.cli', 'evaluate', str(args.data / f'{name}.jsonl'),
                '--object-labels', str(args.data / 'labels.json'), '--device', args.device,
                '--detector', '.cache/yolo11n.pt', '--max-pixels', '200704',
                '--max-new-tokens', str(args.max_new_tokens), '--box-policy', variant['box_policy'], '--out', str(folder)]
        if variant['recover_score'] is not None:
            argv += ['--recover-score', str(variant['recover_score'])]
        if (folder / 'config.json').exists():
            argv += ['--resume']
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE='1')
        with (folder / 'run.log').open('a', encoding='utf-8') as log:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((folder / 'outputs.json').read_text())

    try:
        status('validation')
        validation = run_split('validation', VARIANTS[0])
        if {record.get('model_revision') for record in validation} != {qwen_revision}:
            raise ValueError('Ревизия Qwen в validation не совпала с протоколом')
        if len(validation) != 20:
            raise ValueError('Validation завершён не полностью')
        labels = json.loads((args.data / 'labels.json').read_text())
        candidates = []
        for variant in VARIANTS:
            records = []
            for record in validation:
                result = dict(record)
                if record['baseline'] is not None:
                    with Image.open(record['image']) as source:
                        image = ImageOps.exif_transpose(source).convert('RGB')
                    detections = unpack_evidence(record['evidence']['detector'], image.size)
                    texts = unpack_evidence(record['evidence']['ocr'], image.size)
                    scene, _ = verify(Scene.model_validate(record['baseline']), image, detections, texts,
                                      box_policy=variant['box_policy'], recover_score=variant['recover_score'], object_labels=labels)
                    result['verified'] = scene.model_dump(mode='json')
                records.append(result)
            metrics = evaluate(records)
            candidates.append({'variant': variant, 'metrics': metrics, 'selection_score': selection_score(metrics['verified'])})
        selected = max(candidates, key=lambda value: value['selection_score'])
        frozen = {'selected': selected['variant'], 'candidates': candidates, 'source_sha256': protocol['source_sha256']}
        chosen_path = args.out / 'selection.json'
        if chosen_path.exists() and json.loads(chosen_path.read_text()) != frozen:
            raise ValueError('Зафиксированный выбор политики изменился')
        save_json(chosen_path, frozen)
        if source_hash() != protocol['source_sha256']:
            raise ValueError('Код изменён во время validation, тест не запускается')
        status('test', selected=selected['variant'])
        records = run_split('test', selected['variant'])
        if {record.get('model_revision') for record in records} != {qwen_revision}:
            raise ValueError('Ревизия Qwen в тесте не совпала с протоколом')
        if len(records) != 800:
            raise ValueError('Тест завершён не полностью')
        if source_hash() != protocol['source_sha256']:
            raise ValueError('Код изменён во время теста, итог нельзя считать зафиксированным экспериментом')
        report = evaluate(records)
        report['selected_policy'] = selected['variant']
        # Отдельный baseline показывает вклад одного YOLO, независимо от валидности ответа Qwen
        detector_only = []
        for record in records:
            scene = {'objects': [item['region'] for item in record['evidence']['detector'] if item['score'] >= 0.6]}
            detector_only.append({'ground_truth': record['ground_truth'], 'baseline': scene, 'verified': scene})
        report['yolo_only_confidence_060'] = evaluate(detector_only)['baseline']
        save_json(args.out / 'final_metrics.json', report)
        (args.out / 'report.md').write_text(build_report(records), encoding='utf-8')
        status('completed', dataset_size=800, selected=selected['variant'])
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
