#!/usr/bin/env python3
"""Run real-time, fully local hand gesture recognition from a webcam."""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import mediapipe as mp

from gesture_demo.smoothing import GestureSmoother


HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

LABELS_ZH = {
    "None": "None",
    "Closed_Fist": "Closed Fist",
    "Open_Palm": "Open Palm",
    "Pointing_Up": "Pointing Up",
    "Thumb_Down": "Thumb Down",
    "Thumb_Up": "Thumb Up",
    "Victory": "Victory",
    "ILoveYou": "I Love You",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/gesture_recognizer.task"),
        help="Path to a built-in or Model Maker exported .task file.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-gesture-confidence", type=float, default=0.55)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--min-votes", type=int, default=4)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror frames before recognition (mirroring is friendlier for webcams).",
    )
    return parser.parse_args()


def validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def draw_hand(frame, landmarks: Sequence, color: Tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    points = [
        (max(0, min(width - 1, int(item.x * width))),
         max(0, min(height - 1, int(item.y * height))))
        for item in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 5, color, 1, cv2.LINE_AA)


def draw_text_lines(frame, lines: Iterable[str], origin=(18, 32)) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (15, 15, 15), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (245, 245, 245), 2, cv2.LINE_AA)
        y += 30


def main() -> int:
    args = parse_args()
    try:
        for value, name in (
            (args.min_detection_confidence, "--min-detection-confidence"),
            (args.min_presence_confidence, "--min-presence-confidence"),
            (args.min_tracking_confidence, "--min-tracking-confidence"),
            (args.min_gesture_confidence, "--min-gesture-confidence"),
        ):
            validate_probability(value, name)
        smoother = GestureSmoother(args.smoothing_window, args.min_votes)
    except ValueError as exc:
        print(f"Invalid argument: {exc}", file=sys.stderr)
        return 2

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(
            f"Model not found: {model_path}\n"
            "Run `python scripts/download_default_model.py`, or pass a trained model "
            "with `--model /path/to/gesture_recognizer.task`.",
            file=sys.stderr,
        )
        return 2

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(model_path),
        delegate=mp.tasks.BaseOptions.Delegate.CPU,
    )
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        print(f"Could not open camera index {args.camera}.", file=sys.stderr)
        return 1

    last_timestamp_ms = -1
    fps = 0.0
    previous_time = time.monotonic()

    try:
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera returned an empty frame.", file=sys.stderr)
                    return 1
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                overlay = []
                for index, landmarks in enumerate(result.hand_landmarks):
                    handedness = "Hand"
                    if index < len(result.handedness) and result.handedness[index]:
                        handedness = result.handedness[index][0].category_name or handedness

                    raw_label, raw_score = "None", 0.0
                    if index < len(result.gestures) and result.gestures[index]:
                        category = result.gestures[index][0]
                        if category.score >= args.min_gesture_confidence:
                            raw_label = category.category_name or "None"
                            raw_score = float(category.score)

                    stable = smoother.update(handedness, raw_label, raw_score)
                    color = (80, 220, 120) if handedness == "Left" else (80, 170, 255)
                    draw_hand(frame, landmarks, color)
                    display_label = LABELS_ZH.get(raw_label, raw_label)
                    if stable is None:
                        overlay.append(f"{handedness}: {display_label} {raw_score:.2f} (stabilizing)")
                    else:
                        stable_label = LABELS_ZH.get(stable.label, stable.label)
                        overlay.append(f"{handedness}: {stable_label} {stable.score:.2f}")

                now = time.monotonic()
                frame_seconds = max(now - previous_time, 1e-6)
                current_fps = 1.0 / frame_seconds
                fps = current_fps if fps == 0.0 else 0.9 * fps + 0.1 * current_fps
                previous_time = now
                draw_text_lines(frame, [f"FPS: {fps:.1f}", *overlay, "Q/ESC: quit | R: reset"])
                cv2.imshow("Local MediaPipe Gesture Demo", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    smoother.reset()
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
