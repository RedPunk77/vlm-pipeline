"""Проверка выборки COCO без скачивания датасета"""
import unittest
from scripts.prepare_coco import select_rows


class CocoTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "categories": [{"id": 1, "name": "person"}],
            "images": [{"id": i, "width": 100, "height": 80} for i in range(5)],
            "annotations": [
                {"image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 30], "iscrowd": 1},
            ],
        }

    def test_crowd_exclusion_and_xywh_conversion(self):
        rows, labels = select_rows(self.data, 4, 42)
        self.assertNotIn(1, [r["coco_image_id"] for r in rows])
        person = next(r for r in rows if r["coco_image_id"] == 0)
        self.assertEqual(person["objects"][0]["bbox"], [10, 20, 40, 60])
        self.assertEqual(labels, ["person"])

    def test_reproducibility_and_count(self):
        self.assertEqual(select_rows(self.data, 2, 42), select_rows(self.data, 2, 42))
        for count in (0, 5):
            with self.assertRaises(ValueError):
                select_rows(self.data, count, 42)
