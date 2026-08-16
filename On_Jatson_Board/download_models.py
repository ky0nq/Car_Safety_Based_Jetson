from __future__ import annotations

from pathlib import Path
import shutil
import sys
import urllib.request


MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
    "2.7_80x80_MiniFASNetV2.pth": (
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/refs/heads/master/"
        "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    ),
}


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size < 100_000:
        text = temporary.read_bytes()[:200]
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is unexpectedly small: {text!r}")
    temporary.replace(destination)


def main() -> int:
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for filename, url in MODELS.items():
        destination = model_dir / filename
        if destination.exists() and destination.stat().st_size >= 100_000:
            print(f"[SKIP] {filename} already exists")
            continue
        print(f"[DOWNLOAD] {filename}")
        try:
            download(url, destination)
            print(f"[OK] {destination} ({destination.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {filename}: {exc}", file=sys.stderr)
            print(f"        Manual URL: {url}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
