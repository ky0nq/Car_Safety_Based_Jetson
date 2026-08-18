"""
Multitask(Eye+Yawn+Pose) Model Jetson - inference time 
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def benchmark_model(onnx_path: str, img_size: int, warmup: int, runs: int) -> dict:
    available = ort.get_available_providers()
    providers = []
    for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
        if p in available:
            providers.append(p)
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"  입력 이름: {input_names}")

    eye_dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    face_dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    feed = {input_names[0]: eye_dummy, input_names[1]: face_dummy}

    for _ in range(warmup):
        session.run(None, feed)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    return {
        "provider": session.get_providers()[0],
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "p95_ms": float(np.percentile(times, 95)),
        "fps": float(1000.0 / times.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "multitask_resnet18.onnx",
        "multitask_mobilenet_v2.onnx",
        "multitask_mobilevit_xxs.onnx",
    ])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    print(f"onnxruntime available providers: {ort.get_available_providers()}\n")

    results = {}
    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"[skip] {model_path} 없음")
            continue
        print(f"benchmarking {model_path} ...")
        r = benchmark_model(model_path, args.img_size, args.warmup, args.runs)
        results[model_path] = r
        print(
            f"  provider={r['provider']}  "
            f"mean={r['mean_ms']:.2f}ms  std={r['std_ms']:.2f}ms  "
            f"p95={r['p95_ms']:.2f}ms  fps={r['fps']:.1f}\n"
        )

    print("=== 요약 ===")
    print(f"{'model':45s} {'mean(ms)':>10s} {'p95(ms)':>10s} {'fps':>8s}")
    for name, r in results.items():
        print(f"{name:45s} {r['mean_ms']:10.2f} {r['p95_ms']:10.2f} {r['fps']:8.1f}")


if __name__ == "__main__":
    main()