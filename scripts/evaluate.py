#!/usr/bin/env python
"""Decode a split with a trained checkpoint and report Table-2 metrics.

    python scripts/evaluate.py --run_dir runs/APL-ft --split test [--beam 10]

Data paths (data_dir, fbank_dir, phon_dir) are read from <run_dir>/config.yaml.
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.engine import (build_dataset, load_checkpoint, make_loader, run_eval, setup_logging,  # noqa: E402
                        variant_flags)
from apl.gpu import DutyCycle, limit_gpu_memory  # noqa: E402
from apl.metrics import format_table_row  # noqa: E402
from apl.vocab import Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam", type=int, default=10, help="0 = greedy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu_memory_fraction", type=float, default=0.5, help="VRAM cap (0 = none)")
    ap.add_argument("--gpu_duty_pct", type=int, default=50, help="max %% of wall time the GPU is busy")
    args = ap.parse_args()
    setup_logging()
    with open(os.path.join(args.run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    vocab = Vocab.from_file(cfg["vocab"])
    device = torch.device(args.device)
    limit_gpu_memory(args.gpu_memory_fraction, device)
    duty = DutyCycle(args.gpu_duty_pct, device)
    model = load_checkpoint(os.path.join(args.run_dir, args.ckpt)).to(device)
    flags = variant_flags(cfg["variant"])
    ds = build_dataset(args.split, cfg, vocab, flags)
    loader, _ = make_loader(ds, cfg["batch_size"], False, 0, cfg["num_workers"], vocab.pad_id)
    res = run_eval(model, loader, vocab, device, cfg["amp"], beam=args.beam, duty=duty)
    rows = res.pop("_rows")
    tag = f"{args.split}_beam{args.beam}"
    with open(os.path.join(args.run_dir, f"{tag}_decoded.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.run_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(format_table_row(res))


if __name__ == "__main__":
    main()
