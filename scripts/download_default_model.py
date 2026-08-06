#!/usr/bin/env python3
"""Download Google's built-in Gesture Recognizer model for the first demo run."""

import argparse
import hashlib
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("models/gesture_recognizer.task"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        print(f"Already exists: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    try:
        print(f"Downloading {args.url}")
        urllib.request.urlretrieve(args.url, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Saved: {output}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
