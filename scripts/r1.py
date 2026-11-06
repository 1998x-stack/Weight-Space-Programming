"""R1 command line; offline patch reconstruction and paired benchmark analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsp.repair.replay import replay, ReplayError  # noqa: E402
from wsp.evaluate.paired_statistics import compare, PairingError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R1: offline faithful BF16 replay and paired statistics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("replay", help="Apply and verify p50c sparse edit exactly")
    rep.add_argument("--base", type=Path, required=True)
    rep.add_argument("--edit", type=Path, required=True)
    rep.add_argument("--out", type=Path, required=True)
    ev = sub.add_parser("compare", help="Compare upstream-compatible per-problem score arrays")
    ev.add_argument("--base", type=Path, required=True)
    ev.add_argument("--edited", type=Path, required=True)
    ev.add_argument("--samples", type=int, default=4)
    ev.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    try:
        output = replay(args.base, args.edit, args.out) if args.cmd == "replay" else compare(
            args.base, args.edited, bootstrap=args.bootstrap, require_samples=args.samples)
    except (ReplayError, PairingError, OSError, KeyError, ValueError) as exc:
        print(f"R1 FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
