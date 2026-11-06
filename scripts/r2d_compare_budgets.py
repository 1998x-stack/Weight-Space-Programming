"""Compare two authentic frozen diagnostic traces; refuses ambiguous runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from wsp.evaluate.budget_sensitivity import BudgetComparisonError, compare_files  # noqa: E402


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description="R2-D1 token budget sensitivity")
    p.add_argument("--short", type=Path, required=True)
    p.add_argument("--long", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():
            raise BudgetComparisonError("Refusing to overwrite comparison")
        result=compare_files(a.short,a.long)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open("x",encoding="utf8") as file:
            json.dump(result,file,ensure_ascii=False,indent=2,allow_nan=False)
            file.write("\n")
        print(json.dumps({k:result[k] for k in (
            "status","short_cap","long_cap","problems",
            "short_cut_count","long_cut_count",
            "short_repetition_candidate_count","long_repetition_candidate_count"
        )}))
        return 0
    except (BudgetComparisonError, ValueError, OSError, KeyError) as exc:
        print(f"R2-D COMPARISON REFUSED: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
