"""R1 GPU MATH-500 benchmark entry point. No GPU execution at v0.1 development time."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsp.evaluate.math500 import run_math500  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="MATH-500 / Qwen3.5 p50c exact decoding protocol")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--confirm-gpu-cost", action="store_true",
                    help="Explicitly acknowledge 500 x 4 x 32768 potential token generation workload")
    a = ap.parse_args()
    if not a.confirm_gpu_cost:
        ap.error("Pass --confirm-gpu-cost to enable potentially expensive benchmark generation")
    print(json.dumps(run_math500(a.model, a.dataset, a.out), indent=2))


if __name__ == "__main__":
    main()
