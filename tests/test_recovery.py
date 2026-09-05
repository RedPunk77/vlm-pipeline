"""Проверка добавления пропущенных объектов с явным источником"""
import unittest
from PIL import Image
from src.backends import Evidence
from src.schema import Region, Scene
from src.verification import verify
from scripts.prepare_experiment import make_splits, EXCLUDED_IDS


class RecoveryTests(unittest.TestCase):
    def test_recovers_missing_objects_without_duplicates_or_wrong_scope(self):
        image = Image.new('RGB', (100, 100))
        cat = Region(label='cat', kind='object', bbox=(10, 10, 30, 30))
        dog = Region(label='dog', kind='object', bbox=(50, 50, 90, 90))
        evidence = [Evidence(cat, 0.99), Evidence(cat, 0.98), Evidence(dog, 0.85)]
        result, audit = verify(Scene(objects=[cat]), image, evidence, [], recover_score=0.8)
        self.assertEqual(result.objects, [cat, dog])
        self.assertEqual([x['status'] for x in audit], ['supported', 'added_by_detector'])
        result, _ = verify(Scene(objects=[cat]), image, evidence, [], recover_score=0.8, object_labels=['cat'])
        self.assertEqual(result.objects, [cat])
        result, _ = verify(Scene(objects=[cat]), image, evidence, [], recover_score=0.9)
        self.assertEqual(result.objects, [cat])

    def test_split_is_fixed_disjoint_and_excludes_development(self):
        annotation = {'images': [{'id': x, 'width': 100, 'height': 100} for x in list(range(10)) + EXCLUDED_IDS], 'annotations': [], 'categories': []}
        first, second, _ = make_splits(annotation, 3, 7)
        a, b = {r['coco_image_id'] for r in first}, {r['coco_image_id'] for r in second}
        self.assertFalse(a & b)
        self.assertFalse((a | b) & set(EXCLUDED_IDS))
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 7)
        self.assertEqual((first, second), make_splits(annotation, 3, 7)[:2])
