import base64
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import web_app


class WebAppApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_data_dir = web_app.DATA_DIR
        web_app.DATA_DIR = Path(self.temp_dir.name) / "data" / "gestures"
        self.app = web_app.create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        web_app.DATA_DIR = self.original_data_dir
        self.temp_dir.cleanup()

    @staticmethod
    def image_data_url():
        image = np.full((64, 96, 3), 180, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")

    def test_label_and_sample_lifecycle(self):
        response = self.client.post("/api/labels", json={"label": "Victory_Custom"})
        self.assertEqual(201, response.status_code)
        self.assertEqual("Victory_Custom", response.json["label"])

        response = self.client.post(
            "/api/samples/victory_custom",
            json={"image": self.image_data_url()},
        )
        self.assertEqual(201, response.status_code)
        filename = response.json["name"]

        response = self.client.get("/api/dataset")
        counts = {item["name"]: item["count"] for item in response.json["labels"]}
        self.assertEqual(1, counts["Victory_Custom"])
        self.assertIn("none", counts)

        response = self.client.delete(f"/api/samples/victory_custom/{filename}")
        self.assertEqual(200, response.status_code)
        response = self.client.delete("/api/labels/Victory_Custom")
        self.assertEqual(200, response.status_code)

    def test_only_none_is_protected(self):
        response = self.client.delete("/api/labels/none")
        self.assertEqual(400, response.status_code)

    def test_canonical_label_normalization(self):
        self.assertEqual("none", web_app.normalize_label("None"))
        self.assertEqual("Thumb_Up", web_app.normalize_label("Thumb_Up"))
        self.assertEqual("My_Custom", web_app.normalize_label("My_Custom"))

    def test_training_requires_none_and_one_target_class(self):
        labels = [{"name": "none", "count": 10}]
        self.assertIn("目标手势", web_app.dataset_training_issue(labels))
        labels.append({"name": "ok", "count": 10})
        self.assertIsNone(web_app.dataset_training_issue(labels))
        labels[1]["count"] = 0
        self.assertIn("ok", web_app.dataset_training_issue(labels))

    def test_invalid_label_is_rejected(self):
        response = self.client.post("/api/labels", json={"label": "../escape"})
        self.assertEqual(400, response.status_code)


class TrainingParamsTest(unittest.TestCase):
    def test_accepts_valid_parameters(self):
        params = web_app.validate_training_params({
            "epochs": "20",
            "batch_size": "8",
            "learning_rate": "0.001",
            "dropout_rate": "0.05",
            "model_name": "my-gestures",
        })
        self.assertEqual(20, params["epochs"])
        self.assertEqual("my-gestures", params["model_name"])

    def test_rejects_unsafe_model_name(self):
        with self.assertRaises(ValueError):
            web_app.validate_training_params({"model_name": "../model"})


class GestureRankingTest(unittest.TestCase):
    def test_extracts_custom_labels_from_exported_task(self):
        models = sorted(web_app.EXPORT_DIR.glob("*/gesture_recognizer.task"))
        if not models:
            self.skipTest("No exported gesture task available")
        labels = web_app.custom_labels_for_model(models[-1])
        self.assertIn("none", labels)
        self.assertIn("ok", labels)
        self.assertTrue(web_app.is_independent_label_set(labels))

    def test_independent_label_set_requires_none_and_target(self):
        self.assertTrue(web_app.is_independent_label_set(["none", "ok"]))
        self.assertFalse(web_app.is_independent_label_set(["ok"]))
        self.assertFalse(web_app.is_independent_label_set(["none"]))

    def test_winner_is_real_classifier_argmax(self):
        ranking = [
            {"label": "ok", "score": 0.41, "source": "model"},
            {"label": "Open_Palm", "score": 0.36, "source": "model"},
            {"label": "none", "score": 0.23, "source": "model"},
        ]
        winner = web_app.choose_classifier_winner(ranking)
        self.assertEqual("ok", winner["label"])
        self.assertEqual(0.41, winner["score"])


if __name__ == "__main__":
    unittest.main()
