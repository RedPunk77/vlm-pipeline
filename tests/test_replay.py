"""Проверка сравнения политик без изменения исходных ответов"""
from copy import deepcopy
import unittest
from PIL import Image

from scripts.compare_verification import replay, unpack_evidence
from src.backends import Evidence
from src.schema import Region


class ReplayTests(unittest.TestCase):
    def test_same_input_and_evidence_only_change_selected_box(self):
        record = {'raw': 'исходный ответ', 'seconds': 10,
                  'config': {'match_iou': 0.3, 'min_score': 0.4},
                  'baseline': {'objects': [{'label': 'cat', 'kind': 'object', 'text': None, 'bbox': [10, 10, 40, 40]}]},
                  'verified': None, 'audit': []}
        original = deepcopy(record)
        evidence = [Evidence(Region(label='cat', kind='object', bbox=(8, 8, 42, 42)), 0.9)]
        image = Image.new('RGB', (60, 60))
        keep = replay(record, image, evidence, [], 'preserve')
        replace = replay(record, image, evidence, [], 'detector')
        self.assertEqual(record, original)
        self.assertEqual(keep['baseline'], replace['baseline'])
        self.assertEqual(keep['raw'], replace['raw'])
        self.assertEqual(keep['evidence'], replace['evidence'])
        self.assertEqual(keep['verified']['objects'][0]['bbox'], [10, 10, 40, 40])
        self.assertEqual(replace['verified']['objects'][0]['bbox'], [8, 8, 42, 42])
        self.assertNotIn('seconds', keep)
        self.assertEqual(keep['original_pipeline_seconds'], 10)
        self.assertEqual(unpack_evidence(keep['evidence']['detector'], image.size), evidence)

    def test_invalid_json_stays_failed_in_both_policies(self):
        record = {'config': {'match_iou': 0.3, 'min_score': 0.4},
                  'baseline': None, 'verified': None, 'parse_error': 'ошибка', 'audit': []}
        for policy in ('preserve', 'detector'):
            result = replay(record, Image.new('RGB', (20, 20)), [], [], policy)
            self.assertIsNone(result['verified'])
            self.assertEqual(result['parse_error'], 'ошибка')
