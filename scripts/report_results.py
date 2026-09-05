"""Короткий отчёт по завершённому прогону без повторного запуска моделей"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate


def fmt(value):
    return '—' if value is None else f'{value:.3f}'.replace('.', ',')


def build_report(records):
    report = evaluate(records)
    names = {
        'hallucination_rate': 'Доля предсказаний без пары',
        'grounding_iou': 'Grounding IoU',
        'structured_accuracy': 'Точность полного ответа',
        'schema_valid_rate': 'Валидный JSON',
        'recall': 'Recall',
    }
    lines = ['# Результаты прогона', '', f'Изображений: {len(records)}', '',
             'Порог IoU для успешной пары: 0,5, доли показаны от 0 до 1', '',
             '| Метрика | До проверки | После проверки |', '| --- | --- | --- |']
    for key, title in names.items():
        lines.append(f'| {title} | {fmt(report["baseline"][key])} | {fmt(report["verified"][key])} |')
    lines += ['', '## По изображениям', '',
              '| Изображение | Областей в разметке | Предсказаний до → после | Recall до → после | IoU до → после | JSON |',
              '| --- | --- | --- | --- | --- | --- |']
    failures = []
    for index, record in enumerate(records):
        metrics = evaluate([record])
        before, after = metrics['baseline'], metrics['verified']
        name = Path(record['image']).name.replace('|', '\\|')
        status = 'Ошибка' if record['baseline'] is None else 'Валиден'
        lines.append(f'| {name} | {before["ground_truth_objects"]} | '
                     f'{before["predicted_objects"]} → {after["predicted_objects"]} | '
                     f'{fmt(before["recall"])} → {fmt(after["recall"])} | '
                     f'{fmt(before["grounding_iou"])} → {fmt(after["grounding_iou"])} | {status} |')
        notes = []
        if record.get('parse_error'):
            notes.append('ответ не прошёл проверку схемы')
        if before['matched_objects'] < before['ground_truth_objects']:
            notes.append('до проверки есть пропущенные или плохо локализованные области')
        if after['matched_objects'] < before['matched_objects']:
            notes.append('после проверки стало меньше успешных совпадений')
        if after['grounding_iou'] is not None and before['grounding_iou'] is not None and after['grounding_iou'] < before['grounding_iou']:
            notes.append('после проверки IoU снизился')
        if after['predicted_objects'] > after['matched_objects']:
            notes.append('после проверки остались предсказания без пары в разметке')
        if notes:
            failures.append(f'- {name}: ' + '; '.join(notes))
    lines += ['', '## На что посмотреть', '']
    lines += failures or ['На этом наборе перечисленные проверки не выявили ошибок']
    timed = [record['seconds'] for record in records if 'seconds' in record]
    if timed:
        lines += ['', f'Время обработки: {sum(timed):.1f} с суммарно, {sum(timed) / len(timed):.1f} с в среднем, без загрузки модели'.replace('.', ',')]
    lines += ['', 'Ошибка схемы не считается пустым правильным ответом, а отсутствующий объект в разметке даёт 0 в среднем IoU', '',
              'Выводы относятся только к этому набору, маленький пилот не заменяет оценку на отдельной большой выборке', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Собрать Markdown отчёт из outputs.json')
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding='utf-8'))
    result = build_report(records)
    out = args.out or args.input.with_name('report.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result, encoding='utf-8')
    print(f'Отчёт сохранён в {out}')


if __name__ == '__main__':
    main()
