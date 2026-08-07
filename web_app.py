#!/usr/bin/env python3
"""Local browser UI for collecting, training, and testing gesture models."""

import base64
import binascii
import importlib.metadata
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from mediapipe.tasks.python.metadata import metadata as task_metadata


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "gestures"
MODELS_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "exported_model"
TRAIN_IMAGE = "gesture-trainer:0.2.1.4-independent"
MODEL_MAKER_VERSION = "0.2.1.4"
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
TOP_GESTURE_RESULTS = 5
REQUIRED_LABELS = ("none",)


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 7 * 1024 * 1024
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dataset_layout()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def api_status():
        return jsonify(build_status())

    @app.get("/api/dataset")
    def api_dataset():
        return jsonify(dataset_summary())

    @app.post("/api/labels")
    def api_create_label():
        payload = request.get_json(silent=True) or {}
        label = normalize_label(payload.get("label", ""))
        label_dir = DATA_DIR / label
        existed = label_dir.exists()
        label_dir.mkdir(parents=True, exist_ok=True)
        return jsonify({"ok": True, "label": label, "created": not existed}), 201

    @app.delete("/api/labels/<label>")
    def api_delete_label(label: str):
        label = normalize_label(label)
        if label in REQUIRED_LABELS:
            return api_error("none 是 Model Maker 必需的背景标签，不能删除。", 400)
        label_dir = DATA_DIR / label
        if not label_dir.is_dir():
            return api_error("标签不存在。", 404)
        shutil.rmtree(label_dir)
        return jsonify({"ok": True})

    @app.get("/api/samples/<label>")
    def api_samples(label: str):
        label = normalize_label(label)
        label_dir = DATA_DIR / label
        if not label_dir.is_dir():
            return api_error("标签不存在。", 404)
        files = image_files(label_dir)
        files.sort(key=lambda item: item.stat().st_mtime_ns, reverse=True)
        return jsonify({
            "label": label,
            "count": len(files),
            "items": [
                {
                    "name": item.name,
                    "url": f"/api/samples/{label}/{item.name}",
                    "source": "upload" if item.name.startswith("upload-") else "camera",
                }
                for item in files[:40]
            ],
        })

    @app.get("/api/samples/<label>/<filename>")
    def api_sample_file(label: str, filename: str):
        label = normalize_label(label)
        if Path(filename).name != filename or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            return api_error("文件名无效。", 400)
        return send_from_directory(DATA_DIR / label, filename)

    @app.post("/api/samples/<label>")
    def api_save_sample(label: str):
        label = normalize_label(label)
        payload = request.get_json(silent=True) or {}
        frame = decode_data_image(payload.get("image", ""))
        label_dir = DATA_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)
        prefix = "upload-" if payload.get("source") == "upload" else ""
        filename = f"{prefix}{time.time_ns()}-{uuid.uuid4().hex[:6]}.jpg"
        target = label_dir / filename
        if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            return api_error("图片保存失败。", 500)
        return jsonify({
            "ok": True,
            "name": filename,
            "url": f"/api/samples/{label}/{filename}",
            "count": len(image_files(label_dir)),
        }), 201

    @app.delete("/api/samples/<label>/<filename>")
    def api_delete_sample(label: str, filename: str):
        label = normalize_label(label)
        if Path(filename).name != filename:
            return api_error("文件名无效。", 400)
        target = DATA_DIR / label / filename
        if not target.is_file() or target.suffix.lower() not in IMAGE_EXTENSIONS:
            return api_error("样本不存在。", 404)
        target.unlink()
        return jsonify({"ok": True})

    @app.get("/api/models")
    def api_models():
        return jsonify({"models": list_models()})

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True) or {}
        model_path = resolve_model(payload.get("model", ""))
        frame = decode_data_image(payload.get("image", ""))
        try:
            prediction = recognizer_manager.predict(frame, model_path)
        except ValueError as exc:
            return api_error(str(exc), 400)
        except RuntimeError as exc:
            return api_error(f"模型推理失败：{exc}", 500)
        return jsonify(prediction)

    @app.post("/api/environment/build")
    def api_build_environment():
        if not shutil.which("docker"):
            return api_error("未检测到 Docker Desktop，请安装并启动后重试。", 409)
        try:
            job = job_manager.start(
                kind="environment",
                command=[
                    "docker", "build", "--platform", "linux/amd64",
                    "-f", "Dockerfile.train", "-t", TRAIN_IMAGE, ".",
                ],
                cwd=ROOT,
                total_epochs=0,
            )
        except JobConflict as exc:
            return api_error(str(exc), 409)
        return jsonify(job), 202

    @app.post("/api/train")
    def api_train():
        payload = request.get_json(silent=True) or {}
        issue = dataset_training_issue()
        if issue:
            return api_error(issue, 400)
        try:
            params = validate_training_params(payload)
        except ValueError as exc:
            return api_error(str(exc), 400)

        mode = training_mode()
        if mode == "unavailable":
            return api_error(
                "训练环境尚未就绪。Apple Silicon 请先安装 Docker Desktop，"
                "然后点击“准备训练环境”。",
                409,
            )

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = params["model_name"] or f"gesture-{stamp}"
        run_dir = f"{run_name}-{stamp}"
        train_args = [
            "--data", "data/gestures",
            "--export-dir", f"exported_model/{run_dir}",
            "--epochs", str(params["epochs"]),
            "--batch-size", str(params["batch_size"]),
            "--learning-rate", str(params["learning_rate"]),
            "--dropout-rate", str(params["dropout_rate"]),
        ]
        container_name = None
        if mode == "docker":
            container_name = f"gesture-training-{uuid.uuid4().hex[:8]}"
            command = [
                "docker", "run", "--rm", "--platform", "linux/amd64",
                "--name", container_name,
                "-v", f"{DATA_DIR.parent.resolve()}:/workspace/data",
                "-v", f"{EXPORT_DIR.resolve()}:/workspace/exported_model",
                TRAIN_IMAGE,
                *train_args,
            ]
        else:
            command = [sys.executable, "train.py", *train_args]

        try:
            job = job_manager.start(
                kind="training",
                command=command,
                cwd=ROOT,
                total_epochs=params["epochs"],
                container_name=container_name,
            )
        except JobConflict as exc:
            return api_error(str(exc), 409)
        return jsonify({**job, "output_model": f"exported_model/{run_dir}/gesture_recognizer.task"}), 202

    @app.get("/api/job")
    def api_job():
        return jsonify(job_manager.snapshot())

    @app.post("/api/job/cancel")
    def api_cancel_job():
        if not job_manager.cancel():
            return api_error("当前没有正在运行的任务。", 409)
        return jsonify(job_manager.snapshot())

    @app.errorhandler(413)
    def request_too_large(_error):
        return api_error("上传图片过大。", 413)

    @app.errorhandler(ValueError)
    def invalid_value(error):
        return api_error(str(error), 400)

    return app


def api_error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def normalize_label(value: str) -> str:
    raw_label = str(value).strip()
    if not SAFE_LABEL.fullmatch(raw_label):
        raise ValueError("标签只能包含字母、数字、下划线和短横线，长度 1～40。")
    if raw_label.lower() == "none":
        return "none"
    if DATA_DIR.is_dir():
        existing = next(
            (
                item.name for item in DATA_DIR.iterdir()
                if item.is_dir() and item.name.lower() == raw_label.lower()
            ),
            None,
        )
        if existing:
            return existing
    return raw_label


def ensure_dataset_layout() -> None:
    """Create the required background label and normalize legacy casing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    legacy_none = DATA_DIR / "None"
    canonical_none = DATA_DIR / "none"
    if legacy_none.is_dir() and not canonical_none.exists():
        legacy_none.rename(canonical_none)
    (DATA_DIR / "none").mkdir(parents=True, exist_ok=True)


def image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return [
        item for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]


def decode_data_image(data_url: str):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("图片数据无效。")
    header, encoded = data_url.split(",", 1)
    if header not in {"data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"}:
        raise ValueError("仅支持 JPEG、PNG 或 WebP 图片。")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片 Base64 数据无效。") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("图片为空或超过 5 MB。")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("图片无法解码。")
    if frame.shape[0] < 32 or frame.shape[1] < 32:
        raise ValueError("图片尺寸过小。")
    return frame


def dataset_summary() -> Dict:
    ensure_dataset_layout()
    labels = []
    total = 0
    for directory in sorted((item for item in DATA_DIR.iterdir() if item.is_dir()), key=lambda p: p.name):
        count = len(image_files(directory))
        total += count
        labels.append({
            "name": directory.name,
            "count": count,
            "recommended": 300 if directory.name == "none" else 200,
            "required": directory.name in REQUIRED_LABELS,
            "background": directory.name == "none",
        })
    return {"labels": labels, "total": total, "training_issue": dataset_training_issue(labels)}


def dataset_training_issue(labels: Optional[List[Dict]] = None) -> Optional[str]:
    labels = labels if labels is not None else dataset_summary_without_issue()
    by_name = {item["name"]: item["count"] for item in labels}
    if "none" not in by_name:
        return "缺少 Model Maker 必需的 none 背景标签。"
    gesture_labels = [name for name in by_name if name != "none"]
    if not gesture_labels:
        return "请至少添加一个目标手势标签。"
    empty = [name for name, count in by_name.items() if count == 0]
    if empty:
        return f"这些标签还没有样本：{', '.join(empty)}。"
    too_small = [name for name, count in by_name.items() if count < 10]
    if too_small:
        return f"这些标签至少需要 10 张样本：{', '.join(too_small)}。"
    return None


def dataset_summary_without_issue() -> List[Dict]:
    if not DATA_DIR.is_dir():
        return []
    return [
        {"name": item.name, "count": len(image_files(item))}
        for item in DATA_DIR.iterdir() if item.is_dir()
    ]


def installed_model_maker_version() -> Optional[str]:
    try:
        return importlib.metadata.version("mediapipe-model-maker")
    except importlib.metadata.PackageNotFoundError:
        return None


def docker_image_ready() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", TRAIN_IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def training_mode() -> str:
    version = installed_model_maker_version()
    is_arm_mac = sys.platform == "darwin" and platform.machine() == "arm64"
    if version == MODEL_MAKER_VERSION and not is_arm_mac:
        return "native"
    if docker_image_ready():
        return "docker"
    return "unavailable"


def build_status() -> Dict:
    docker_installed = bool(shutil.which("docker"))
    version = installed_model_maker_version()
    image_ready = docker_image_ready() if docker_installed else False
    is_arm_mac = sys.platform == "darwin" and platform.machine() == "arm64"
    if version == MODEL_MAKER_VERSION and not is_arm_mac:
        mode = "native"
    elif image_ready:
        mode = "docker"
    else:
        mode = "unavailable"
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "mediapipe": mp.__version__,
        "model_maker_required": MODEL_MAKER_VERSION,
        "model_maker_installed": version,
        "docker_installed": docker_installed,
        "docker_image_ready": image_ready,
        "training_mode": mode,
        "job": job_manager.snapshot(),
    }


def list_models() -> List[Dict]:
    models = []
    seen = set()
    for base, kind in ((EXPORT_DIR, "训练"), (MODELS_DIR, "内置")):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.task"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            resolved = path.resolve()
            if resolved in seen or not resolved.is_relative_to(ROOT):
                continue
            seen.add(resolved)
            labels = custom_labels_for_model(path)
            models.append({
                "path": str(resolved.relative_to(ROOT)),
                "name": path.parent.name if base == EXPORT_DIR else path.stem,
                "kind": kind,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "labels": labels,
                "independent": is_independent_label_set(labels),
            })
    return models


def is_independent_label_set(labels: List[str]) -> bool:
    label_set = set(labels)
    return "none" in label_set and bool(label_set.difference(REQUIRED_LABELS))


def resolve_model(value: str) -> Path:
    candidate = (ROOT / str(value)).resolve()
    if not candidate.is_relative_to(ROOT) or candidate.suffix != ".task" or not candidate.is_file():
        raise ValueError("请选择项目内存在的 .task 模型。")
    allowed_roots = (MODELS_DIR.resolve(), EXPORT_DIR.resolve())
    if not any(candidate.is_relative_to(base) for base in allowed_roots):
        raise ValueError("模型必须位于 models 或 exported_model 目录。")
    return candidate


def validate_training_params(payload: Dict) -> Dict:
    try:
        epochs = int(payload.get("epochs", 20))
        batch_size = int(payload.get("batch_size", 8))
        learning_rate = float(payload.get("learning_rate", 0.001))
        dropout_rate = float(payload.get("dropout_rate", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError("训练参数格式不正确。") from exc
    if not 1 <= epochs <= 500:
        raise ValueError("训练轮数必须在 1～500 之间。")
    if not 1 <= batch_size <= 256:
        raise ValueError("批大小必须在 1～256 之间。")
    if not 0.000001 <= learning_rate <= 1:
        raise ValueError("学习率必须在 0.000001～1 之间。")
    if not 0 <= dropout_rate < 1:
        raise ValueError("Dropout 必须在 0～1 之间。")
    model_name = str(payload.get("model_name", "")).strip().lower()
    if model_name and not SAFE_MODEL_NAME.fullmatch(model_name):
        raise ValueError("模型名只能包含字母、数字、下划线和短横线。")
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "model_name": model_name,
    }


class RecognizerManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._baseline_recognizer = None
        self._model_recognizers = {}
        self._model_path: Optional[Path] = None
        self._timestamp_ms = 0

    def _load(self, model_path: Path) -> None:
        self._close_recognizers()
        labels = custom_labels_for_model(model_path)
        if not is_independent_label_set(labels):
            raise ValueError(
                "所选模型不可用于独立分类测试；需要包含 none 和至少 1 个目标手势。"
            )
        official_model_path = MODELS_DIR / "gesture_recognizer.task"
        if not official_model_path.is_file():
            raise ValueError("缺少官方 gesture_recognizer.task 模型。")
        self._baseline_recognizer = self._create_recognizer(official_model_path)
        for label in labels:
            self._model_recognizers[label] = self._create_recognizer(
                model_path, model_label=label
            )
        self._model_path = model_path
        self._timestamp_ms = 0

    @staticmethod
    def _create_recognizer(
        model_path: Path,
        model_label: Optional[str] = None,
    ):
        canned_options = mp.tasks.components.processors.ClassifierOptions(
            max_results=1, score_threshold=1.0
        ) if model_label else mp.tasks.components.processors.ClassifierOptions(
            max_results=1, score_threshold=0.0
        )
        custom_options = mp.tasks.components.processors.ClassifierOptions(
            category_allowlist=[model_label], score_threshold=0.0
        ) if model_label else mp.tasks.components.processors.ClassifierOptions(
            max_results=1, score_threshold=1.0
        )
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=canned_options,
            custom_gesture_classifier_options=custom_options,
        )
        return mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def _close_recognizers(self) -> None:
        recognizers = list(self._model_recognizers.values())
        if self._baseline_recognizer is not None:
            recognizers.append(self._baseline_recognizer)
        for recognizer in recognizers:
            recognizer.close()
        self._baseline_recognizer = None
        self._model_recognizers = {}

    def predict(self, frame, model_path: Path) -> Dict:
        with self._lock:
            if not self._model_recognizers or self._model_path != model_path:
                self._load(model_path)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now_ms = time.monotonic_ns() // 1_000_000
            self._timestamp_ms = max(now_ms, self._timestamp_ms + 1)
            # A MediaPipe task turns the input into a Packet whose holder may be
            # consumed by its graph. Reusing one mp.Image across independent task
            # runners can leave the downstream tensor packet with multiple owners
            # and fail in ConcatenateTensorVectorCalculator. Give every runner its
            # own image storage instead.
            baseline_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb.copy()
            )
            baseline_result = self._baseline_recognizer.recognize_for_video(
                baseline_image, self._timestamp_ms
            )
            model_results = {}
            for label, recognizer in self._model_recognizers.items():
                model_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb.copy()
                )
                model_results[label] = recognizer.recognize_for_video(
                    model_image, self._timestamp_ms
                )

        hands = []
        for index, landmarks in enumerate(baseline_result.hand_landmarks):
            handedness = "Hand"
            if index < len(baseline_result.handedness) and baseline_result.handedness[index]:
                handedness = baseline_result.handedness[index][0].category_name or handedness
            model_gestures = rank_label_results(model_results, index, "model")
            hands.append({
                "handedness": handedness,
                "baseline_gesture": category_for_hand(baseline_result, index, "official"),
                "model_gestures": model_gestures,
                "winner": choose_classifier_winner(model_gestures),
                "top_gestures": model_gestures[:TOP_GESTURE_RESULTS],
                "landmarks": [{"x": item.x, "y": item.y, "z": item.z} for item in landmarks],
            })
        return {
            "ok": True,
            "model": str(model_path.relative_to(ROOT)),
            "independent_model": True,
            "hands": hands,
        }


def custom_labels_for_model(model_path: Path) -> List[str]:
    try:
        with zipfile.ZipFile(model_path) as outer:
            hand_gesture_task = outer.read("hand_gesture_recognizer.task")
        with zipfile.ZipFile(io.BytesIO(hand_gesture_task)) as inner:
            custom_classifier = inner.read("custom_gesture_classifier.tflite")
        displayer = task_metadata.MetadataDisplayer.with_model_buffer(custom_classifier)
        label_files = [
            name for name in displayer.get_packed_associated_file_list()
            if name.lower().endswith("labels.txt")
        ]
        if not label_files:
            return []
        content = displayer.get_associated_file_buffer(label_files[0]).decode("utf-8")
        return [label.strip() for label in content.splitlines() if label.strip()]
    except (KeyError, UnicodeDecodeError, ValueError, zipfile.BadZipFile):
        return []


def rank_label_results(results: Dict, index: int, source: str) -> List[Dict]:
    rankings = []
    for expected_label, result in results.items():
        if index >= len(result.gestures) or not result.gestures[index]:
            continue
        category = result.gestures[index][0]
        rankings.append({
            "label": category.category_name or expected_label,
            "score": float(category.score),
            "source": source,
        })
    return sorted(
        rankings, key=lambda item: item["score"], reverse=True
    )[:TOP_GESTURE_RESULTS]


def category_for_hand(result, index: int, source: str) -> Optional[Dict]:
    if index >= len(result.gestures) or not result.gestures[index]:
        return None
    category = result.gestures[index][0]
    return {
        "label": category.category_name or "None",
        "score": float(category.score),
        "source": source,
    }


def choose_classifier_winner(rankings: List[Dict]) -> Dict:
    if not rankings:
        return {"label": "none", "score": 0.0, "source": "model"}
    return rankings[0]


class JobConflict(RuntimeError):
    pass


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._container_name: Optional[str] = None
        self._cancel_requested = False
        self._state = self._empty_state()

    @staticmethod
    def _empty_state() -> Dict:
        return {
            "id": None,
            "kind": None,
            "status": "idle",
            "progress": 0,
            "epoch": 0,
            "total_epochs": 0,
            "logs": [],
            "started_at": None,
            "finished_at": None,
            "message": "暂无任务",
        }

    def start(
        self,
        kind: str,
        command: List[str],
        cwd: Path,
        total_epochs: int,
        container_name: Optional[str] = None,
    ) -> Dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise JobConflict("已有任务正在运行，请等待完成或先取消。")
            job_id = uuid.uuid4().hex
            self._cancel_requested = False
            self._container_name = container_name
            self._state = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "progress": 1,
                "epoch": 0,
                "total_epochs": total_epochs,
                "logs": [],
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "message": "正在准备训练环境" if kind == "environment" else "正在提取手部关键点",
            }
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            thread = threading.Thread(target=self._watch, args=(self._process, job_id), daemon=True)
            thread.start()
            return self.snapshot()

    def _watch(self, process: subprocess.Popen, job_id: str) -> None:
        epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")
        build_pattern = re.compile(r"\[(?:[^]]*\s)?(\d+)/(\d+)\]")
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            with self._lock:
                if self._state["id"] != job_id:
                    continue
                self._state["logs"].append(line)
                self._state["logs"] = self._state["logs"][-500:]
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    epoch, total = map(int, epoch_match.groups())
                    self._state["epoch"] = epoch
                    self._state["total_epochs"] = total
                    self._state["progress"] = min(95, 20 + int(epoch / total * 70))
                    self._state["message"] = f"正在训练：第 {epoch}/{total} 轮"
                elif self._state["kind"] == "environment":
                    build_match = build_pattern.search(line)
                    if build_match:
                        current, total = map(int, build_match.groups())
                        self._state["progress"] = min(95, int(current / total * 90))
                    if "Downloading " in line:
                        self._state["progress"] = max(self._state["progress"], 75)
                        self._state["message"] = "正在下载训练依赖，首次构建需要几分钟"
                    elif "Installing collected packages:" in line:
                        self._state["progress"] = max(self._state["progress"], 82)
                        self._state["message"] = "正在安装 TensorFlow 与 Model Maker"
                    elif "exporting layers" in line:
                        self._state["progress"] = max(self._state["progress"], 90)
                        self._state["message"] = "正在保存训练镜像"
                elif "Extracting hand landmarks" in line:
                    self._state["progress"] = 8
                    self._state["message"] = "正在提取手部关键点"
                elif "Test accuracy:" in line:
                    self._state["progress"] = 97
                    self._state["message"] = line
                elif "Exported model:" in line:
                    self._state["message"] = "模型已导出"

        return_code = process.wait()
        with self._lock:
            if self._state["id"] != job_id:
                return
            self._state["finished_at"] = datetime.now().isoformat(timespec="seconds")
            if self._cancel_requested:
                self._state["status"] = "cancelled"
                self._state["message"] = "任务已取消"
            elif return_code == 0:
                self._state["status"] = "success"
                self._state["progress"] = 100
                self._state["message"] = "训练环境已就绪" if self._state["kind"] == "environment" else "训练完成，模型已导出"
            else:
                self._state["status"] = "error"
                self._state["message"] = f"任务失败（退出码 {return_code}）"
            self._process = None
            self._container_name = None

    def snapshot(self) -> Dict:
        with self._lock:
            state = dict(self._state)
            state["logs"] = list(self._state["logs"])
            return state

    def cancel(self) -> bool:
        with self._lock:
            process = self._process
            container_name = self._container_name
            if process is None or process.poll() is not None:
                return False
            self._cancel_requested = True
        if container_name and shutil.which("docker"):
            subprocess.run(
                ["docker", "stop", "--time", "5", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            process.terminate()
        return True


recognizer_manager = RecognizerManager()
job_manager = JobManager()
app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
