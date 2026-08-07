#!/usr/bin/env python3
"""Import the Xiaobu gesture URL lists into the local training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np


LABEL_MAPPING = {
    "OK": "ok",
    "张开手掌": "open_palm",
    "握拳": "closed_fist",
    "比耶": "victory",
    "点赞": "thumb_up",
    "比心": "finger_heart",
    "666": "gesture_666",
}
EXCLUDED_LABELS = {
    "未识别": "Needs manual cleaning before it can safely become the none class.",
}
USABLE_STATUSES = {"imported", "existing", "manually_restored"}
SOURCE_NAME = re.compile(r"^(?P<collector>[0-9]+)-(?P<timestamp>[0-9]+)\.[A-Za-z0-9]+$")


@dataclass(frozen=True)
class SourceSample:
    source_label: str
    target_label: str
    collector_id: str
    captured_at_ms: int
    url: str
    output_name: str


@dataclass
class ImportRecord:
    source_label: str
    target_label: str
    collector_id: str
    captured_at_ms: int
    url: str
    output_file: str
    status: str
    width: int | None = None
    height: int | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("小布gesture"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/gestures"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/xiaobu_import_manifest.json"),
    )
    parser.add_argument(
        "--manual-approvals",
        type=Path,
        default=Path("data/xiaobu_manual_approvals.json"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-image-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manual_approvals(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise SystemExit(f"Unsupported manual approvals schema: {path}")
    approvals: dict[str, dict] = {}
    for item in payload.get("approved_samples", []):
        url = item.get("url")
        target_label = item.get("target_label")
        if not isinstance(url, str) or target_label not in LABEL_MAPPING.values():
            raise SystemExit(f"Invalid manual approval in {path}: {item}")
        if url in approvals:
            raise SystemExit(f"Duplicate manual approval URL in {path}: {url}")
        approvals[url] = item
    return approvals


def load_sources(source_dir: Path) -> list[SourceSample]:
    samples: list[SourceSample] = []
    seen_urls: set[str] = set()
    for source_label, target_label in LABEL_MAPPING.items():
        source_file = source_dir / source_label
        if not source_file.is_file():
            raise SystemExit(f"Missing Xiaobu source list: {source_file}")
        for line_number, raw_line in enumerate(
            source_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            url = raw_line.strip()
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise SystemExit(f"Invalid URL at {source_file}:{line_number}: {url}")
            match = SOURCE_NAME.fullmatch(Path(parsed.path).name)
            if not match:
                raise SystemExit(
                    f"Unexpected source filename at {source_file}:{line_number}: {url}"
                )
            if url in seen_urls:
                raise SystemExit(f"Duplicate URL in Xiaobu source lists: {url}")
            seen_urls.add(url)
            collector_id = match.group("collector")
            captured_at_ms = int(match.group("timestamp"))
            short_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
            output_name = (
                f"xiaobu_{collector_id}_{captured_at_ms}_{short_hash}.jpg"
            )
            samples.append(
                SourceSample(
                    source_label=source_label,
                    target_label=target_label,
                    collector_id=collector_id,
                    captured_at_ms=captured_at_ms,
                    url=url,
                    output_name=output_name,
                )
            )
    return samples


def download_image(
    sample: SourceSample,
    *,
    timeout: float,
    max_image_bytes: int,
) -> tuple[SourceSample, np.ndarray | None, str | None]:
    request = urllib.request.Request(
        sample.url,
        headers={"User-Agent": "gesture-dataset-importer/1.0"},
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=ssl.create_default_context(),
        ) as response:
            payload = response.read(max_image_bytes + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return sample, None, f"download_error: {exc}"
    if not payload:
        return sample, None, "empty_response"
    if len(payload) > max_image_bytes:
        return sample, None, f"image_exceeds_{max_image_bytes}_bytes"
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return sample, None, "image_decode_failed"
    return sample, image, None


def write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("OpenCV could not encode JPEG")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.jpg", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded.tobytes())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def make_record(
    sample: SourceSample,
    data_dir: Path,
    status: str,
    *,
    image: np.ndarray | None = None,
    error: str | None = None,
) -> ImportRecord:
    height, width = image.shape[:2] if image is not None else (None, None)
    output_path = data_dir / sample.target_label / sample.output_name
    return ImportRecord(
        source_label=sample.source_label,
        target_label=sample.target_label,
        collector_id=sample.collector_id,
        captured_at_ms=sample.captured_at_ms,
        url=sample.url,
        output_file=str(output_path),
        status=status,
        width=width,
        height=height,
        error=error,
    )


def write_manifest(
    path: Path,
    *,
    source_dir: Path,
    data_dir: Path,
    manual_approvals_path: Path,
    approved_urls: set[str],
    min_detection_confidence: float,
    records: list[ImportRecord],
) -> None:
    summary = Counter(record.status for record in records)
    per_label = Counter(
        record.target_label
        for record in records
        if record.status in USABLE_STATUSES
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "data_dir": str(data_dir),
        "label_mapping": LABEL_MAPPING,
        "excluded_labels": EXCLUDED_LABELS,
        "manual_review": {
            "approvals_file": str(manual_approvals_path),
            "approved_url_count": len(approved_urls),
        },
        "hand_filter": {
            "engine": "mediapipe.solutions.hands",
            "mediapipe_version": mp.__version__,
            "min_detection_confidence": min_detection_confidence,
        },
        "summary": dict(sorted(summary.items())),
        "usable_images_per_label": dict(sorted(per_label.items())),
        "records": [asdict(record) for record in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.timeout <= 0 or args.max_image_bytes < 1:
        raise SystemExit("--timeout and --max-image-bytes must be positive")
    if not 0.0 <= args.min_detection_confidence <= 1.0:
        raise SystemExit("--min-detection-confidence must be between 0 and 1")

    source_dir = args.source_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    manual_approvals_path = args.manual_approvals.expanduser().resolve()
    manual_approvals = load_manual_approvals(manual_approvals_path)
    samples = load_sources(source_dir)
    print(f"Found {len(samples)} target-gesture URLs across {len(LABEL_MAPPING)} labels.")
    print("Excluded 未识别 from none pending manual cleaning.")
    if args.dry_run:
        for label, count in sorted(Counter(s.target_label for s in samples).items()):
            print(f"  {label}: {count}")
        return 0

    records: list[ImportRecord] = []
    pending: list[SourceSample] = []
    for sample in samples:
        output_path = data_dir / sample.target_label / sample.output_name
        if output_path.is_file():
            records.append(make_record(sample, data_dir, "existing"))
        else:
            pending.append(sample)

    processed = len(records)
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
    ) as hands, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_image,
                sample,
                timeout=args.timeout,
                max_image_bytes=args.max_image_bytes,
            ): sample
            for sample in pending
        }
        for future in as_completed(futures):
            sample, image, error = future.result()
            processed += 1
            if error:
                records.append(make_record(sample, data_dir, "rejected", error=error))
            else:
                assert image is not None
                manually_approved = sample.url in manual_approvals
                if manually_approved:
                    has_hand = True
                else:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    detection = hands.process(rgb)
                    has_hand = bool(detection.multi_hand_landmarks)
                if not has_hand:
                    records.append(
                        make_record(
                            sample,
                            data_dir,
                            "rejected",
                            image=image,
                            error="no_hand_detected",
                        )
                    )
                else:
                    output_path = data_dir / sample.target_label / sample.output_name
                    try:
                        write_jpeg(output_path, image)
                    except OSError as exc:
                        records.append(
                            make_record(
                                sample,
                                data_dir,
                                "rejected",
                                image=image,
                                error=f"write_error: {exc}",
                            )
                        )
                    else:
                        status = "manually_restored" if manually_approved else "imported"
                        records.append(make_record(sample, data_dir, status, image=image))
            if processed % 100 == 0 or processed == len(samples):
                print(f"Processed {processed}/{len(samples)}")

    records.sort(key=lambda item: (item.target_label, item.captured_at_ms, item.url))
    write_manifest(
        manifest_path,
        source_dir=source_dir,
        data_dir=data_dir,
        manual_approvals_path=manual_approvals_path,
        approved_urls=set(manual_approvals),
        min_detection_confidence=args.min_detection_confidence,
        records=records,
    )
    summary = Counter(record.status for record in records)
    print("Import summary:", dict(sorted(summary.items())))
    print(f"Manifest: {manifest_path}")
    return 0 if not summary.get("rejected") else 2


if __name__ == "__main__":
    raise SystemExit(main())
