#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


NAME_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<mode>drop_nodes|drop_relations)-(?P<pct>\d+)pct-seed(?P<seed>\d+)$"
)


def flatten_bucket(prefix: str, payload: Dict[str, Any], row: Dict[str, Any]) -> None:
    for key, value in payload.items():
        if key == "rates" and isinstance(value, dict):
            for rate_key, rate_value in value.items():
                row[f"{prefix}{rate_key}_rate"] = rate_value
        elif isinstance(value, (int, float, str, bool)) or value is None:
            row[f"{prefix}{key}"] = value


def rows_for_file(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    match = NAME_RE.match(path.stem)
    meta = match.groupdict() if match else {}
    rows: List[Dict[str, Any]] = []
    for bucket_name, bucket in sorted((data.get("buckets") or {}).items()):
        row: Dict[str, Any] = {
            "file": str(path),
            "dataset": data.get("dataset", meta.get("dataset", "")),
            "ablation_mode": meta.get("mode", ""),
            "ablation_pct": meta.get("pct", ""),
            "ablation_seed": meta.get("seed", ""),
            "bucket": bucket_name,
            "kb_path": data.get("kb_path", ""),
            "split": data.get("split", ""),
        }
        if isinstance(bucket, dict):
            flatten_bucket("", bucket, row)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten reachability audit JSON files into one table.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        rows.extend(rows_for_file(path))

    write_csv(Path(args.out_csv), rows)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[reachability-summary] rows={len(rows)} csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
