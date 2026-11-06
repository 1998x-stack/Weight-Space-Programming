"""Prepare licensed MATH-500 rows for R1; requires datasets + network.

Dataset revision is mandatory to prevent ambiguous reproducibility.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Pin and prepare HuggingFaceH4/MATH-500 test dataset")
    p.add_argument("--revision", required=True, help="Exact dataset repository commit SHA")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Output already exists; refusing to overwrite")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=a.revision)
    if len(ds) != 500:
        raise ValueError("Unexpected MATH-500 dataset length")
    rows = [{"id":str(r["unique_id"]), "problem":r["problem"], "answer":r["answer"]} for r in ds]
    if len({r["id"] for r in rows}) != 500:
        raise ValueError("Duplicate MATH-500 unique IDs")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows).encode()
    a.out.write_bytes(blob)
    manifest = {"source":"HuggingFaceH4/MATH-500", "revision":a.revision, "split":"test",
                "problems":500, "file_sha256":hashlib.sha256(blob).hexdigest()}
    (a.out.parent / (a.out.name+".manifest.json")).write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
