#!/usr/bin/env python3
"""Train and export a custom MediaPipe Gesture Recognizer .task model."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_MODEL_MAKER_VERSION = "0.2.1.4"
REQUIRED_LABELS = ("none",)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/gestures"))
    parser.add_argument("--export-dir", type=Path, default=Path("exported_model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--layer-widths", type=int, nargs="*", default=[])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--install-model",
        type=Path,
        default=None,
        help="Optionally copy the exported .task to this path after training.",
    )
    return parser.parse_args()


def require_exact_model_maker_version() -> None:
    try:
        installed = importlib.metadata.version("mediapipe-model-maker")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "mediapipe-model-maker is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    if installed != REQUIRED_MODEL_MAKER_VERSION:
        raise SystemExit(
            f"Refusing to train with mediapipe-model-maker {installed}; "
            f"this project requires exactly {REQUIRED_MODEL_MAKER_VERSION}."
        )


def validate_dataset(dataset_path: Path) -> list:
    if not dataset_path.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_path}")
    labels = sorted(item.name for item in dataset_path.iterdir() if item.is_dir())
    if "none" not in labels:
        raise SystemExit("Dataset must contain the Model Maker background label 'none'.")
    target_labels = [label for label in labels if label != "none"]
    if not target_labels:
        raise SystemExit("Dataset needs 'none' plus at least one target gesture label.")
    empty = [
        label for label in labels
        if not any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in (dataset_path / label).iterdir()
        )
    ]
    if empty:
        raise SystemExit(f"These label directories are empty: {', '.join(empty)}")
    return ["none", *[label for label in labels if label != "none"]]


def main() -> int:
    args = parse_args()
    require_exact_model_maker_version()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")
    if not 0 < args.train_fraction < 1:
        raise SystemExit("--train-fraction must be between 0 and 1.")
    if not 0 < args.validation_fraction < 1 - args.train_fraction:
        raise SystemExit("--validation-fraction must leave a non-empty test fraction.")

    dataset_path = args.data.expanduser().resolve()
    export_dir = args.export_dir.expanduser().resolve()
    labels = validate_dataset(dataset_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    import tensorflow as tf
    from mediapipe_model_maker import gesture_recognizer

    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print(f"Labels: {labels}")
    print("Extracting hand landmarks from dataset images...")
    data = gesture_recognizer.Dataset.from_folder(
        dirname=str(dataset_path),
        hparams=gesture_recognizer.HandDataPreprocessingParams(
            shuffle=True,
            min_detection_confidence=args.min_detection_confidence,
        ),
    )
    train_data, rest_data = data.split(args.train_fraction)
    validation_share_of_rest = args.validation_fraction / (1.0 - args.train_fraction)
    validation_data, test_data = rest_data.split(validation_share_of_rest)

    hparams = gesture_recognizer.HParams(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        export_dir=str(export_dir),
    )
    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=args.dropout_rate,
        layer_widths=args.layer_widths,
    )
    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options,
        hparams=hparams,
    )
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )
    loss, accuracy = model.evaluate(test_data, batch_size=1)
    print(f"Test loss: {loss:.6f}")
    print(f"Test accuracy: {accuracy:.4%}")
    model.export_model()

    exported_task = export_dir / "gesture_recognizer.task"
    if not exported_task.is_file():
        raise SystemExit(f"Training finished, but expected model was not found: {exported_task}")
    print(f"Exported model: {exported_task}")

    manifest = {
        "schema_version": 1,
        "architecture": "independent_gesture_classifier",
        "model_file": exported_task.name,
        "model_sha256": hashlib.sha256(exported_task.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mediapipe_model_maker_version": REQUIRED_MODEL_MAKER_VERSION,
        "labels": labels,
        "required_labels": list(REQUIRED_LABELS),
        "target_labels": [label for label in labels if label not in REQUIRED_LABELS],
        "background_label": "none",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout_rate": args.dropout_rate,
            "layer_widths": args.layer_widths,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
            "min_detection_confidence": args.min_detection_confidence,
            "seed": args.seed,
        },
        "evaluation": {"loss": float(loss), "accuracy": float(accuracy)},
        "reused_components": ["palm_detector", "hand_landmarker", "gesture_embedder"],
        "unity_top5_required": False,
    }
    manifest_path = export_dir / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported manifest: {manifest_path}")

    if args.install_model:
        install_path = args.install_model.expanduser().resolve()
        install_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported_task, install_path)
        print(f"Installed model: {install_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
