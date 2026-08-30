#!/usr/bin/env python3
"""Create reproducible train/validation/test JSONL splits, with optional group isolation.

Examples:
  python split_federalist_dataset.py data.jsonl --group-field paper_id
  python split_federalist_dataset.py data.json --train 0.80 --validation 0.10 --test 0.10

If multiple records came from one Federalist paper, use --group-field with the
field that identifies that paper (for example paper_id). This keeps every
example from a paper in exactly one split and avoids source leakage.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def load_records(path: Path):
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("Input file is empty.")
    if path.suffix.lower() == ".json":
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError("A .json input must contain a top-level list of records.")
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every input item must be a JSON object.")
    return records

def fingerprint(record):
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def normalized_text(record):
    return " ".join(json.dumps(record, ensure_ascii=False, sort_keys=True).lower().split())

def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input .jsonl or JSON-array file")
    parser.add_argument("--out-dir", type=Path, default=Path("splits"))
    parser.add_argument("--group-field", help="Field used to keep related examples together, e.g. paper_id")
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--validation", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.train, args.validation, args.test) <= 0 or abs(args.train + args.validation + args.test - 1.0) > 1e-9:
        parser.error("--train, --validation, and --test must be positive and sum to 1.0")

    records = load_records(args.input)
    original_count = len(records)

    unique = []
    seen = set()
    for record in records:
        key = fingerprint(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    records = unique

    groups = {}
    if args.group_field:
        missing = [i for i, record in enumerate(records) if args.group_field not in record]
        if missing:
            raise ValueError(f"{len(missing)} record(s) lack --group-field '{args.group_field}'. Add it or omit --group-field.")
        for record in records:
            value = record[args.group_field]
            group_key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            groups.setdefault(group_key, []).append(record)
    else:
        groups = {str(i): [record] for i, record in enumerate(records)}

    group_items = list(groups.items())
    random.Random(args.seed).shuffle(group_items)
    targets = {
        "train": len(records) * args.train,
        "validation": len(records) * args.validation,
        "test": len(records) * args.test,
    }
    splits = {name: [] for name in targets}

    for _, group_records in group_items:
        deficits = {name: targets[name] - len(splits[name]) for name in targets}
        destination = max(deficits, key=deficits.get)
        splits[destination].extend(group_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_records in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", split_records)

    text_sets = {name: {normalized_text(r) for r in rows} for name, rows in splits.items()}
    overlap = {
        "train_validation": len(text_sets["train"] & text_sets["validation"]),
        "train_test": len(text_sets["train"] & text_sets["test"]),
        "validation_test": len(text_sets["validation"] & text_sets["test"]),
    }
    report = {
        "input_records": original_count,
        "exact_duplicates_removed": original_count - len(records),
        "seed": args.seed,
        "group_field": args.group_field,
        "groups": len(groups),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "normalized_exact_text_overlap": overlap,
    }
    (args.out_dir / "split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
