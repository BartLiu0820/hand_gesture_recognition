#!/usr/bin/env python3
"""Capture labelled webcam images in MediaPipe Model Maker folder format."""

import argparse
import re
import time
from pathlib import Path

import cv2


SAFE_LABEL = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Gesture label; use 'none' for background samples.")
    parser.add_argument("--output", type=Path, default=Path("data/gestures"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Auto-capture interval in seconds; 0 means Space key only.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N images; 0 means unlimited.")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SAFE_LABEL.fullmatch(args.label):
        raise SystemExit("--label may only contain letters, numbers, '_' and '-'.")
    if args.interval < 0 or args.limit < 0:
        raise SystemExit("--interval and --limit cannot be negative.")

    label = "none" if args.label.lower() == "none" else args.label
    label_dir = (args.output / label).resolve()
    label_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}.")

    captured = 0
    last_capture = time.monotonic()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Camera returned an empty frame.")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            preview = frame.copy()
            status = f"Label: {label} | Captured: {captured} | Space: save | Q: quit"
            cv2.putText(preview, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gesture Dataset Collector", preview)

            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            should_capture = key == ord(" ") or (args.interval > 0 and now - last_capture >= args.interval)
            if should_capture:
                filename = label_dir / f"{time.time_ns()}.jpg"
                if not cv2.imwrite(str(filename), frame):
                    raise SystemExit(f"Could not write {filename}")
                captured += 1
                last_capture = now
                print(filename)
            if key in (ord("q"), 27) or (args.limit and captured >= args.limit):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
