#!/usr/bin/env python3
"""
YOLO11 .pt + MiVOLO TorchScript .pt GPU inference.

Supported sources:
- Webcam: --source 0
- Video : --source ./test_video.mp4
- Image : --source ./test_video.jpg
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11 PT + MiVOLO TorchScript PT video/webcam/image inference."
    )
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--mivolo", required=True)
    parser.add_argument("--source", default="0")
    parser.add_argument("--device", default="0")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mivolo-imgsz", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--max-persons", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=("all", "round_robin"),
        default="all",
    )

    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)

    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Use FP32 instead of FP16.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
    )
    parser.add_argument("--save-video", default=None)

    # Optional conversion from normalized age output to years.
    parser.add_argument("--min-age", type=float, default=None)
    parser.add_argument("--max-age", type=float, default=None)
    parser.add_argument("--avg-age", type=float, default=None)

    parser.add_argument(
        "--output",
        default="yolo_mivolo_pt_video_latency.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="yolo_mivolo_pt_video_summary.csv",
    )
    return parser.parse_args()


def resolve_cuda_device(device_text: str) -> Tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch CUDA를 사용할 수 없습니다.\n"
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
        )

    if device_text.isdigit():
        index = int(device_text)
    elif device_text.startswith("cuda:"):
        index = int(device_text.split(":", 1)[1])
    else:
        raise ValueError("--device에는 0 또는 cuda:0을 지정하세요.")

    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA 장치 {index}가 없습니다. 장치 수={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}"), str(index)


class SourceReader:
    def __init__(
        self,
        source: str,
        camera_width: int,
        camera_height: int,
    ) -> None:
        self.source_type: str
        self.capture: Optional[cv2.VideoCapture] = None
        self.image: Optional[np.ndarray] = None
        self.fps = 30.0

        suffix = Path(source).suffix.lower()

        if source.isdigit():
            self.source_type = "webcam"
            self.capture = cv2.VideoCapture(int(source))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        elif suffix in IMAGE_EXTENSIONS:
            self.source_type = "image"
            self.image = cv2.imread(source)
            if self.image is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {source}")

        else:
            self.source_type = "video"
            self.capture = cv2.VideoCapture(source)
            if self.capture.isOpened():
                value = self.capture.get(cv2.CAP_PROP_FPS)
                if value and value > 0:
                    self.fps = float(value)

        if self.capture is not None and not self.capture.isOpened():
            raise RuntimeError(f"입력을 열 수 없습니다: {source}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.source_type == "image":
            assert self.image is not None
            return True, self.image.copy()

        assert self.capture is not None
        return self.capture.read()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


class MiVOLOTorchScript:
    """
    Supports common TorchScript forward forms:
    1) model(concat_face_body)        -> [N, 6, H, W]
    2) model(face, body)              -> two [N, 3, H, W] tensors
    3) model(body, face)
    4) model(face)                    -> face-only fallback
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        input_size: int,
        half: bool,
        min_age: Optional[float],
        max_age: Optional[float],
        avg_age: Optional[float],
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"MiVOLO TorchScript 모델을 찾을 수 없습니다: {path}"
            )

        try:
            self.model = torch.jit.load(str(path), map_location=device)
        except Exception as exc:
            raise RuntimeError(
                "mivolo_v2.pt를 torch.jit.load()로 읽지 못했습니다.\n"
                "현재 파일이 실제 TorchScript 모델인지 확인하세요."
            ) from exc

        self.device = device
        self.input_size = int(input_size)
        self.half = bool(half)
        self.dtype = torch.float16 if self.half else torch.float32

        self.min_age = min_age
        self.max_age = max_age
        self.avg_age = avg_age

        self.model.eval()
        self.model.to(device)

        if self.half:
            self.model.half()
        else:
            self.model.float()

        self.forward_mode = ""
        self._print_schema()
        self._detect_forward_mode()

    @property
    def has_age_metadata(self) -> bool:
        return all(
            value is not None
            for value in (self.min_age, self.max_age, self.avg_age)
        )

    def _print_schema(self) -> None:
        try:
            print("MiVOLO forward schema:", self.model.forward.schema, flush=True)
        except Exception:
            print("MiVOLO forward schema: 확인 불가", flush=True)

    def _dummy_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        face = torch.randn(
            1,
            3,
            self.input_size,
            self.input_size,
            device=self.device,
            dtype=self.dtype,
        )
        body = torch.randn_like(face)
        return face, body

    def _call_mode(
        self,
        mode: str,
        face: torch.Tensor,
        body: torch.Tensor,
    ) -> Any:
        if mode == "concat":
            return self.model(torch.cat((face, body), dim=1))
        if mode == "face_body":
            return self.model(face, body)
        if mode == "body_face":
            return self.model(body, face)
        if mode == "face_only":
            return self.model(face)
        raise ValueError(f"지원하지 않는 mode: {mode}")

    def _detect_forward_mode(self) -> None:
        face, body = self._dummy_pair()
        errors = []

        with torch.inference_mode():
            for mode in ("concat", "face_body", "body_face", "face_only"):
                try:
                    output = self._call_mode(mode, face, body)
                    torch.cuda.synchronize(self.device)
                    self.forward_mode = mode
                    shape = self._output_description(output)
                    print(
                        f"MiVOLO TorchScript 입력 방식: {mode}",
                        flush=True,
                    )
                    print(
                        f"MiVOLO 시험 출력: {shape}",
                        flush=True,
                    )
                    return
                except Exception as exc:
                    errors.append(
                        f"{mode}: {type(exc).__name__}: {str(exc)[:300]}"
                    )

        details = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(
            "TorchScript 모델의 입력 형식을 자동 판별하지 못했습니다.\n"
            f"시도 결과:\n{details}"
        )

    @staticmethod
    def _output_description(output: Any) -> str:
        if torch.is_tensor(output):
            return f"Tensor shape={tuple(output.shape)}"
        if isinstance(output, (tuple, list)):
            return (
                f"{type(output).__name__}("
                + ", ".join(
                    f"Tensor{tuple(item.shape)}"
                    if torch.is_tensor(item)
                    else type(item).__name__
                    for item in output
                )
                + ")"
            )
        if isinstance(output, dict):
            return f"dict keys={list(output.keys())}"
        return type(output).__name__

    def preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            image_bgr,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        array = rgb.astype(np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        array = np.transpose(array, (2, 0, 1))
        array = np.ascontiguousarray(array)

        return (
            torch.from_numpy(array)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
        )

    def make_inputs(
        self,
        face_crop: np.ndarray,
        body_crop: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.preprocess_image(face_crop),
            self.preprocess_image(body_crop),
        )

    def forward(
        self,
        face_tensor: torch.Tensor,
        body_tensor: torch.Tensor,
    ) -> Any:
        with torch.inference_mode():
            return self._call_mode(
                self.forward_mode,
                face_tensor,
                body_tensor,
            )

    def decode(self, output: Any) -> Tuple[float, str, float]:
        """
        Common MiVOLO output: [male_logit, female_logit, normalized_age]
        If the exported model uses another output layout, timing remains valid, but displayed age/gender can require model-specific decoding.
        """
        age_raw = float("nan")
        gender = "unknown"
        gender_score = float("nan")

        if isinstance(output, dict):
            if "age" in output:
                age_values = torch.as_tensor(output["age"]).detach().float().cpu().reshape(-1)
                if age_values.numel():
                    age_raw = float(age_values[0])

            gender_value = output.get(
                "gender_probability",
                output.get("gender", None),
            )
            if gender_value is not None:
                values = (
                    torch.as_tensor(gender_value)
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )
                if values.numel():
                    probability = float(values[0])
                    gender = "female" if probability >= 0.5 else "male"
                    gender_score = max(probability, 1.0 - probability)

        else:
            if isinstance(output, (tuple, list)):
                tensor_items = [item for item in output if torch.is_tensor(item)]
                if len(tensor_items) == 1:
                    tensor = tensor_items[0]
                elif len(tensor_items) >= 2:
                    age_values = tensor_items[0].detach().float().cpu().reshape(-1)
                    if age_values.numel():
                        age_raw = float(age_values[0])

                    gender_values = tensor_items[1].detach().float().cpu().reshape(-1)
                    if gender_values.numel():
                        probability = float(gender_values[0])
                        gender = "female" if probability >= 0.5 else "male"
                        gender_score = max(probability, 1.0 - probability)
                    tensor = None
                else:
                    tensor = None
            elif torch.is_tensor(output):
                tensor = output
            else:
                tensor = None

            if tensor is not None:
                tensor = tensor.detach().float().cpu()
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)

                if tensor.ndim >= 2 and tensor.shape[-1] >= 3:
                    logits = tensor[0, :2]
                    probabilities = torch.softmax(logits, dim=-1)
                    index = int(torch.argmax(probabilities))
                    gender = "male" if index == 0 else "female"
                    gender_score = float(probabilities[index])
                    age_raw = float(tensor[0, 2])
                elif tensor.numel():
                    age_raw = float(tensor.reshape(-1)[0])

        if self.has_age_metadata and np.isfinite(age_raw):
            age = (
                age_raw
                * (float(self.max_age) - float(self.min_age))
                + float(self.avg_age)
            )
        else:
            age = age_raw

        return age, gender, gender_score


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(
            f"OpenCV 얼굴 검출기를 읽지 못했습니다: {cascade_path}"
        )
    return detector


def extract_face_crop(
    detector: cv2.CascadeClassifier,
    body_crop: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces):
        x, y, width, height = max(
            faces,
            key=lambda box: int(box[2]) * int(box[3]),
        )

        margin_x = int(width * 0.20)
        margin_y = int(height * 0.20)
        body_height, body_width = body_crop.shape[:2]

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(body_width, x + width + margin_x)
        y2 = min(body_height, y + height + margin_y)

        crop = body_crop[y1:y2, x1:x2]
        if crop.size:
            return crop, False

    # Fallback: upper part of the detected person.
    height, width = body_crop.shape[:2]
    upper_height = max(1, int(height * 0.45))
    horizontal_margin = int(width * 0.15)

    fallback = body_crop[
        0:upper_height,
        horizontal_margin:max(horizontal_margin + 1, width - horizontal_margin),
    ]
    return (fallback if fallback.size else body_crop), True


def clip_box(
    values: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, values)
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    return x1, y1, x2, y2


def get_person_boxes(
    result: Any,
    frame_shape: Sequence[int],
    max_persons: int,
) -> List[Tuple[int, int, int, int]]:
    frame_height, frame_width = frame_shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []

    if result.boxes is None:
        return boxes

    for box in result.boxes:
        if int(box.cls[0].item()) != 0:
            continue

        values = box.xyxy[0].detach().cpu().tolist()
        boxes.append(
            clip_box(values, frame_width, frame_height)
        )

    boxes.sort(
        key=lambda box: (
            (box[2] - box[0]) * (box[3] - box[1])
        ),
        reverse=True,
    )
    return boxes[:max_persons]


def create_video_writer(
    output_path: Optional[str],
    fps: float,
    frame_shape: Sequence[int],
) -> Optional[cv2.VideoWriter]:
    if output_path is None:
        return None

    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"결과 영상을 생성할 수 없습니다: {output_path}"
        )
    return writer


def write_summary(
    output_path: Path,
    records: List[Dict[str, Any]],
) -> None:
    metrics = (
        "source_read_ms",
        "yolo_preprocess_ms",
        "yolo_inference_ms",
        "yolo_postprocess_ms",
        "yolo_total_ms",
        "yolo_wall_ms",
        "face_detection_total_ms",
        "mivolo_preprocess_total_ms",
        "mivolo_cuda_inference_total_ms",
        "mivolo_cuda_inference_per_call_ms",
        "mivolo_wall_inference_total_ms",
        "mivolo_wall_inference_per_call_ms",
        "mivolo_postprocess_total_ms",
        "mivolo_total_ms",
        "mivolo_total_per_call_ms",
        "pipeline_ms",
        "render_ms",
        "display_ms",
        "end_to_end_ms",
        "actual_fps",
    )

    rows = []
    for metric in metrics:
        values = np.asarray(
            [float(record[metric]) for record in records],
            dtype=np.float64,
        )

        # Per-call metrics are NaN on frames where MiVOLO was not called.
        # Exclude those frames instead of treating them as 0 ms.
        values = values[np.isfinite(values)]

        if values.size == 0:
            rows.append(
                {
                    "metric": metric,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "p95": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p95": float(np.percentile(values, 95)),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup은 0 이상이어야 합니다.")
    if args.frames < 1:
        raise ValueError("--frames는 1 이상이어야 합니다.")
    if args.max_persons < 1:
        raise ValueError("--max-persons는 1 이상이어야 합니다.")

    yolo_path = Path(args.yolo)
    mivolo_path = Path(args.mivolo)

    if not yolo_path.exists():
        raise FileNotFoundError(
            f"YOLO 모델을 찾을 수 없습니다: {yolo_path}"
        )
    if not mivolo_path.exists():
        raise FileNotFoundError(
            f"MiVOLO 모델을 찾을 수 없습니다: {mivolo_path}"
        )

    device, yolo_device = resolve_cuda_device(args.device)
    half = not args.no_half

    print("PyTorch version :", torch.__version__, flush=True)
    print("CUDA version    :", torch.version.cuda, flush=True)
    print("CUDA device     :", torch.cuda.get_device_name(device), flush=True)
    print("Precision       :", "FP16" if half else "FP32", flush=True)

    source = SourceReader(
        source=args.source,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )

    print("Source type     :", source.source_type, flush=True)
    print("Source          :", args.source, flush=True)

    yolo_model = YOLO(str(yolo_path), task="detect")

    try:
        mivolo_model = MiVOLOTorchScript(
            model_path=str(mivolo_path),
            device=device,
            input_size=args.mivolo_imgsz,
            half=half,
            min_age=args.min_age,
            max_age=args.max_age,
            avg_age=args.avg_age,
        )
    except RuntimeError:
        if half:
            print(
                "[알림] FP16 로딩/시험 실행에 실패하여 FP32로 다시 시도합니다.",
                flush=True,
            )
            mivolo_model = MiVOLOTorchScript(
                model_path=str(mivolo_path),
                device=device,
                input_size=args.mivolo_imgsz,
                half=False,
                min_age=args.min_age,
                max_age=args.max_age,
                avg_age=args.avg_age,
            )
            half = False
        else:
            raise

    face_detector = create_face_detector()

    # Warm-up
    dummy_frame = np.zeros(
        (args.imgsz, args.imgsz, 3),
        dtype=np.uint8,
    )
    dummy_face = torch.randn(
        1,
        3,
        args.mivolo_imgsz,
        args.mivolo_imgsz,
        device=device,
        dtype=mivolo_model.dtype,
    )
    dummy_body = torch.randn_like(dummy_face)

    for index in range(1, args.warmup + 1):
        yolo_model.predict(
            source=dummy_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=yolo_device,
            quantize=16 if half else 32,
            verbose=False,
        )
        mivolo_model.forward(dummy_face, dummy_body)

        if index % 5 == 0 or index == args.warmup:
            print(f"Warm-up {index}/{args.warmup}", flush=True)

    torch.cuda.synchronize(device)

    records: List[Dict[str, Any]] = []
    video_writer: Optional[cv2.VideoWriter] = None
    round_robin_index = 0

    try:
        for frame_index in range(1, args.frames + 1):
            end_to_end_start = time.perf_counter()

            read_start = time.perf_counter()
            ok, frame = source.read()
            read_end = time.perf_counter()

            if not ok or frame is None:
                print("영상이 끝났거나 프레임을 읽지 못했습니다.", flush=True)
                break

            if video_writer is None and args.save_video:
                video_writer = create_video_writer(
                    output_path=args.save_video,
                    fps=source.fps,
                    frame_shape=frame.shape,
                )

            pipeline_start = time.perf_counter()

            torch.cuda.synchronize(device)
            yolo_wall_start = time.perf_counter()

            result = yolo_model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=yolo_device,
                quantize=16 if half else 32,
                verbose=False,
            )[0]

            torch.cuda.synchronize(device)
            yolo_wall_end = time.perf_counter()

            yolo_preprocess_ms = float(
                result.speed.get("preprocess", 0.0)
            )
            yolo_inference_ms = float(
                result.speed.get("inference", 0.0)
            )
            yolo_postprocess_ms = float(
                result.speed.get("postprocess", 0.0)
            )
            yolo_total_ms = (
                yolo_preprocess_ms
                + yolo_inference_ms
                + yolo_postprocess_ms
            )
            yolo_wall_ms = (
                yolo_wall_end - yolo_wall_start
            ) * 1000.0

            person_boxes = get_person_boxes(
                result=result,
                frame_shape=frame.shape,
                max_persons=args.max_persons,
            )

            if args.strategy == "round_robin" and person_boxes:
                selected_boxes = [
                    person_boxes[
                        round_robin_index % len(person_boxes)
                    ]
                ]
                round_robin_index += 1
            else:
                selected_boxes = person_boxes

            face_detection_total_ms = 0.0
            mivolo_preprocess_total_ms = 0.0
            mivolo_cuda_inference_total_ms = 0.0
            mivolo_wall_inference_total_ms = 0.0
            mivolo_postprocess_total_ms = 0.0
            mivolo_total_ms = 0.0
            fallback_faces = 0
            labels = []

            for box in selected_boxes:
                x1, y1, x2, y2 = box
                body_crop = frame[y1:y2, x1:x2]
                if body_crop.size == 0:
                    continue

                face_detection_start = time.perf_counter()
                face_crop, used_fallback = extract_face_crop(
                    face_detector,
                    body_crop,
                )
                face_detection_end = time.perf_counter()

                mivolo_one_start = time.perf_counter()

                preprocess_start = time.perf_counter()
                face_tensor, body_tensor = mivolo_model.make_inputs(
                    face_crop,
                    body_crop,
                )
                torch.cuda.synchronize(device)
                preprocess_end = time.perf_counter()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                wall_start = time.perf_counter()
                start_event.record()
                output = mivolo_model.forward(
                    face_tensor,
                    body_tensor,
                )
                end_event.record()
                torch.cuda.synchronize(device)
                wall_end = time.perf_counter()

                postprocess_start = time.perf_counter()
                age, gender, gender_score = mivolo_model.decode(output)
                postprocess_end = time.perf_counter()

                mivolo_one_end = time.perf_counter()

                face_detection_total_ms += (
                    face_detection_end - face_detection_start
                ) * 1000.0
                mivolo_preprocess_total_ms += (
                    preprocess_end - preprocess_start
                ) * 1000.0
                mivolo_cuda_inference_total_ms += float(
                    start_event.elapsed_time(end_event)
                )
                mivolo_wall_inference_total_ms += (
                    wall_end - wall_start
                ) * 1000.0
                mivolo_postprocess_total_ms += (
                    postprocess_end - postprocess_start
                ) * 1000.0
                mivolo_total_ms += (
                    mivolo_one_end - mivolo_one_start
                ) * 1000.0

                if used_fallback:
                    fallback_faces += 1

                labels.append(
                    (
                        box,
                        age,
                        gender,
                        gender_score,
                        used_fallback,
                    )
                )

            mivolo_calls = len(labels)

            if mivolo_calls > 0:
                mivolo_cuda_inference_per_call_ms = (
                    mivolo_cuda_inference_total_ms / mivolo_calls
                )
                mivolo_wall_inference_per_call_ms = (
                    mivolo_wall_inference_total_ms / mivolo_calls
                )
                mivolo_total_per_call_ms = (
                    mivolo_total_ms / mivolo_calls
                )
            else:
                mivolo_cuda_inference_per_call_ms = float("nan")
                mivolo_wall_inference_per_call_ms = float("nan")
                mivolo_total_per_call_ms = float("nan")

            torch.cuda.synchronize(device)
            pipeline_end = time.perf_counter()
            pipeline_ms = (
                pipeline_end - pipeline_start
            ) * 1000.0

            render_start = time.perf_counter()
            annotated = frame.copy()

            for x1, y1, x2, y2 in person_boxes:
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            for (
                box,
                age,
                gender,
                gender_score,
                used_fallback,
            ) in labels:
                x1, y1, _, _ = box
                if np.isfinite(age):
                    label = f"age={age:.1f}"

                    if age <= 7.0:
                        label += " | UNDER 7"
                else:
                    label = "age=unknown"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 255),
                    2,
                )

            cv2.putText(
                annotated,
                (
                    f"persons={len(person_boxes)} "
                    f"calls={len(labels)} "
                    f"pipeline={pipeline_ms:.1f}ms"
                ),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            render_end = time.perf_counter()
            render_ms = (
                render_end - render_start
            ) * 1000.0

            if video_writer is not None:
                video_writer.write(annotated)

            display_ms = 0.0
            key = -1

            if not args.no_display:
                display_start = time.perf_counter()
                cv2.imshow(
                    "YOLO11 PT + MiVOLO TorchScript",
                    annotated,
                )
                key = cv2.waitKey(1) & 0xFF
                display_end = time.perf_counter()
                display_ms = (
                    display_end - display_start
                ) * 1000.0

            end_to_end_end = time.perf_counter()
            end_to_end_ms = (
                end_to_end_end - end_to_end_start
            ) * 1000.0

            record: Dict[str, Any] = {
                "frame": frame_index,
                "source_type": source.source_type,
                "detected_persons": len(person_boxes),
                "mivolo_calls": len(labels),
                "fallback_faces": fallback_faces,
                "source_read_ms": (
                    read_end - read_start
                ) * 1000.0,
                "yolo_preprocess_ms": yolo_preprocess_ms,
                "yolo_inference_ms": yolo_inference_ms,
                "yolo_postprocess_ms": yolo_postprocess_ms,
                "yolo_total_ms": yolo_total_ms,
                "yolo_wall_ms": yolo_wall_ms,
                "face_detection_total_ms": face_detection_total_ms,
                "mivolo_preprocess_total_ms": (
                    mivolo_preprocess_total_ms
                ),
                "mivolo_cuda_inference_total_ms": (
                    mivolo_cuda_inference_total_ms
                ),
                "mivolo_cuda_inference_per_call_ms": (
                    mivolo_cuda_inference_per_call_ms
                ),
                "mivolo_wall_inference_total_ms": (
                    mivolo_wall_inference_total_ms
                ),
                "mivolo_wall_inference_per_call_ms": (
                    mivolo_wall_inference_per_call_ms
                ),
                "mivolo_postprocess_total_ms": (
                    mivolo_postprocess_total_ms
                ),
                "mivolo_total_ms": mivolo_total_ms,
                "mivolo_total_per_call_ms": (
                    mivolo_total_per_call_ms
                ),
                "pipeline_ms": pipeline_ms,
                "render_ms": render_ms,
                "display_ms": display_ms,
                "end_to_end_ms": end_to_end_ms,
                "actual_fps": (
                    1000.0 / end_to_end_ms
                    if end_to_end_ms > 0
                    else 0.0
                ),
            }
            records.append(record)

            if frame_index % 10 == 0 or frame_index == args.frames:
                print(
                    f"[{frame_index}/{args.frames}] "
                    f"persons={len(person_boxes)} | "
                    f"YOLO={yolo_inference_ms:.2f} ms | "
                    f"MiVOLO CUDA total={mivolo_cuda_inference_total_ms:.2f} ms | "
                    f"MiVOLO/call={mivolo_cuda_inference_per_call_ms:.2f} ms | "
                    f"Pipeline={pipeline_ms:.2f} ms | "
                    f"FPS={record['actual_fps']:.2f}",
                    flush=True,
                )

            if key == ord("q"):
                break

    finally:
        source.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    if not records:
        raise RuntimeError("측정된 프레임이 없습니다.")

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    summary_path = Path(args.summary_output)
    write_summary(summary_path, records)

    pipeline_values = np.asarray(
        [float(record["pipeline_ms"]) for record in records],
        dtype=np.float64,
    )
    end_to_end_values = np.asarray(
        [float(record["end_to_end_ms"]) for record in records],
        dtype=np.float64,
    )

    mivolo_per_call_values = np.asarray(
        [
            float(record["mivolo_cuda_inference_per_call_ms"])
            for record in records
        ],
        dtype=np.float64,
    )
    mivolo_per_call_values = mivolo_per_call_values[
        np.isfinite(mivolo_per_call_values)
    ]

    print("\n========== 측정 결과 ==========", flush=True)
    print(f"측정 프레임   : {len(records)}", flush=True)
    print(
        f"Pipeline 평균 : {np.mean(pipeline_values):.3f} ms",
        flush=True,
    )
    print(
        f"Pipeline P95  : {np.percentile(pipeline_values, 95):.3f} ms",
        flush=True,
    )
    print(
        f"E2E 평균      : {np.mean(end_to_end_values):.3f} ms",
        flush=True,
    )
    if mivolo_per_call_values.size > 0:
        print(
            f"MiVOLO 1회 CUDA 평균 : "
            f"{np.mean(mivolo_per_call_values):.3f} ms",
            flush=True,
        )
        print(
            f"MiVOLO 1회 CUDA P95  : "
            f"{np.percentile(mivolo_per_call_values, 95):.3f} ms",
            flush=True,
        )
    else:
        print(
            "MiVOLO 1회 CUDA 평균 : 측정값 없음",
            flush=True,
        )
    print(
        f"평균 FPS      : {1000.0 / np.mean(end_to_end_values):.2f}",
        flush=True,
    )
    print(f"CSV           : {output_path.resolve()}", flush=True)
    print(f"Summary CSV   : {summary_path.resolve()}", flush=True)
    if args.save_video:
        print(
            f"결과 영상     : {Path(args.save_video).resolve()}",
            flush=True,
        )
    print("===============================", flush=True)


if __name__ == "__main__":
    main()