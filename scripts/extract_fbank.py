#!/usr/bin/env python
"""Extract 81-dim fbank for every split and global CMVN stats from the training split.

    python scripts/extract_fbank.py --data_dir data --audio_root /path/to/wavs --out_dir feats/fbank

Writes <out_dir>/<utt_id>.npy (raw features) and <out_dir>/cmvn.npz. Re-running skips files
that already exist (CMVN is always recomputed from the train files).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.audio import load_audio, resolve_audio  # noqa: E402
from apl.features import CMVN, compute_fbank  # noqa: E402
from apl.manifest import load_split  # noqa: E402


def _work(job):
    audio_root, rel, out_path = job
    if os.path.isfile(out_path):
        return None
    path = resolve_audio(audio_root, rel)
    if path is None:
        return f"missing audio: {rel}"
    feat = compute_fbank(load_audio(path))
    tmp = out_path + ".tmp.npy"
    np.save(tmp, feat)
    os.replace(tmp, out_path)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out_dir", default="feats/fbank")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    splits = {s: load_split(os.path.join(args.data_dir, f"{s}.csv")) for s in ("train", "dev", "test")}
    jobs = [(args.audio_root, u.path, os.path.join(args.out_dir, u.utt_id + ".npy"))
            for utts in splits.values() for u in utts]
    errors = []
    with ProcessPoolExecutor(args.num_workers) as ex:
        for i, err in enumerate(ex.map(_work, jobs, chunksize=16)):
            if err:
                errors.append(err)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    if errors:
        print("\n".join(errors[:20]))
        sys.exit(f"{len(errors)} files failed")

    total, total_sq, count = np.zeros(81), np.zeros(81), 0
    for u in splits["train"]:
        f = np.load(os.path.join(args.out_dir, u.utt_id + ".npy")).astype(np.float64)
        total += f.sum(0)
        total_sq += (f**2).sum(0)
        count += f.shape[0]
    CMVN.from_stats(total, total_sq, count).save(os.path.join(args.out_dir, "cmvn.npz"))
    print(f"done: {len(jobs)} utterances, CMVN from {count} training frames")


if __name__ == "__main__":
    main()
