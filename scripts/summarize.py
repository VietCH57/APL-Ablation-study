#!/usr/bin/env python
"""Collect <runs_dir>/<name>/<split>_metrics.json into a Table-2 style table.

    python scripts/summarize.py --runs_dir runs [--split test] [--out results.md]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.metrics import TABLE_KEYS  # noqa: E402

ORDER = ["AL", "P-raw", "PL-raw", "APL-raw", "P-ft", "PL-ft", "APL-ft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", f"{args.split}_metrics.json"))):
        with open(path) as f:
            results[os.path.basename(os.path.dirname(path))] = json.load(f)
    names = [n for n in ORDER if n in results] + sorted(n for n in results if n not in ORDER)
    lines = ["| Model | " + " | ".join(h for _, h in TABLE_KEYS) + " |", "|" + "---|" * (len(TABLE_KEYS) + 1)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{100 * results[n][k]:.2f}" for k, _ in TABLE_KEYS) + " |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
