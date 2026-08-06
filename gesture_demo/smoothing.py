"""Small, dependency-free temporal smoothing helpers."""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class StablePrediction:
    """A gesture that is stable enough to expose to the application."""

    label: str
    score: float


class GestureSmoother:
    """Turn noisy per-frame classifications into stable gesture events.

    Histories are keyed by handedness (``Left``/``Right``). This is sufficient
    for the common one-left-hand + one-right-hand webcam use case and keeps the
    demo independent from an object tracker.
    """

    def __init__(self, window_size: int = 7, min_votes: int = 4) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if not 1 <= min_votes <= window_size:
            raise ValueError("min_votes must be between 1 and window_size")
        self._window_size = window_size
        self._min_votes = min_votes
        self._history: Dict[str, Deque[Tuple[str, float]]] = {}

    def update(self, hand_key: str, label: str, score: float) -> Optional[StablePrediction]:
        history = self._history.setdefault(hand_key, deque(maxlen=self._window_size))
        history.append((label, score))

        counts = Counter(item[0] for item in history)
        winner, votes = counts.most_common(1)[0]
        if winner == "None" or votes < self._min_votes:
            return None

        winner_scores = [item_score for item_label, item_score in history if item_label == winner]
        return StablePrediction(winner, sum(winner_scores) / len(winner_scores))

    def reset(self) -> None:
        self._history.clear()
