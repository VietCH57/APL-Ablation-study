#!/usr/bin/env python
"""Score phone predictions made outside this repo (e.g. XLS-R + CTC from
XLSR-300M-Phoneme-Recognition) with exactly the same MD&D metrics as the APL runs.

    python scripts/score_predictions.py \
        --pred /path/to/XLSR-300M-Phoneme-Recognition/runs/apl_ft/best/infer/test/predictions.csv \
        --split test --out_dir runs/XLSR-CTC

--pred is a CSV with columns `row` (line index in the CSV given to infer.py, i.e. in
xlsr_ft_data/<split>.csv, which has the same order as data/<split>.csv) and `hyp`
(space-separated phones). Canonical and human transcript come from data/<split>.csv.
Writes <out_dir>/<split>_metrics.json and <split>_decoded.jsonl, so scripts/summarize.py
picks the run up as one more row of the table.
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.manifest import load_split  # noqa: E402
from apl.metrics import evaluate, format_table_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    utts = load_split(os.path.join(args.data_dir, f"{args.split}.csv"))
    pred = pd.read_csv(args.pred, keep_default_na=False, dtype={"hyp": str})
    if "row" not in pred.columns or "hyp" not in pred.columns:
        sys.exit(f"{args.pred}: needs columns 'row' and 'hyp'")
    hyp = dict(zip(pred["row"].astype(int), pred["hyp"]))
    if sorted(hyp) != list(range(len(utts))):
        sys.exit(f"{args.pred}: rows do not match the {len(utts)} utterances of {args.split}.csv")
    if "ref" in pred.columns:  # sanity check: same reference transcript, same order
        bad = [i for i, r in zip(pred["row"].astype(int), pred["ref"]) if r.split() != utts[i].transcript]
        if bad:
            sys.exit(f"{len(bad)} rows have a different reference than {args.split}.csv, e.g. row {bad[0]}")

    preds = [hyp[i].split() for i in range(len(utts))]
    res = evaluate([u.canonical for u in utts], [u.transcript for u in utts], preds)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.split}_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.out_dir, f"{args.split}_decoded.jsonl"), "w") as f:
        for u, p in zip(utts, preds):
            f.write(json.dumps({"utt_id": u.utt_id, "canonical": " ".join(u.canonical),
                                "human": " ".join(u.transcript), "pred": " ".join(p)}) + "\n")
    print(f"[{args.split}] {format_table_row(res)}")


if __name__ == "__main__":
    main()
