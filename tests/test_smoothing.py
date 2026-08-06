import unittest

from gesture_demo.smoothing import GestureSmoother


class GestureSmootherTest(unittest.TestCase):
    def test_returns_prediction_after_enough_votes(self):
        smoother = GestureSmoother(window_size=5, min_votes=3)
        self.assertIsNone(smoother.update("Left", "Victory", 0.8))
        self.assertIsNone(smoother.update("Left", "None", 0.0))
        self.assertIsNone(smoother.update("Left", "Victory", 0.9))
        prediction = smoother.update("Left", "Victory", 1.0)
        self.assertEqual("Victory", prediction.label)
        self.assertAlmostEqual(0.9, prediction.score)

    def test_none_never_becomes_a_stable_event(self):
        smoother = GestureSmoother(window_size=3, min_votes=2)
        self.assertIsNone(smoother.update("Right", "None", 0.0))
        self.assertIsNone(smoother.update("Right", "None", 0.0))

    def test_hands_have_independent_histories(self):
        smoother = GestureSmoother(window_size=2, min_votes=2)
        smoother.update("Left", "Victory", 0.8)
        smoother.update("Right", "Thumb_Up", 0.7)
        self.assertEqual("Victory", smoother.update("Left", "Victory", 0.9).label)
        self.assertEqual("Thumb_Up", smoother.update("Right", "Thumb_Up", 0.8).label)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            GestureSmoother(window_size=0, min_votes=1)
        with self.assertRaises(ValueError):
            GestureSmoother(window_size=3, min_votes=4)


if __name__ == "__main__":
    unittest.main()
