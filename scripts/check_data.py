#!/usr/bin/env python
"""Sanity checks on the three split CSVs before anything is extracted.

    python scripts/check_data.py --data_dir data --audio_root /path/to/wavs

Reports: utterances per corpus and split (compare with Table 1 of the paper), speakers per
split and speaker overlap, phones outside the vocab, TIMIT canonical vs transcript, and
audio files that cannot be found under --audio_root.
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.audio import resolve_audio  # noqa: E402
from apl.manifest import load_split  # noqa: E402
from apl.metrics import edit_distance  # noqa: E402
from apl.vocab import Vocab  # noqa: E402

PAPER = {"train": {"timit": 6300, "l2arctic": 1800}, "dev": {"l2arctic": 897}, "test": {"l2arctic": 900}}
PAPER_SPK = {
    "dev": {"EBVS", "THV", "TNI", "BWC", "YDCK", "YBAA"},
    "test": {"NJS", "HQTV", "SVBI", "NCC", "YKWK", "ZHAA"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vocab", default="configs/vocab.json")
    ap.add_argument("--audio_root", default=None)
    args = ap.parse_args()

    vocab = Vocab.from_file(args.vocab)
    print(f"vocab: {vocab.num_classes} classes ({len(vocab.phones)} phones + blank)")
    speakers = {}
    for split in ("train", "dev", "test"):
        utts = load_split(os.path.join(args.data_dir, f"{split}.csv"))
        by_corpus = collections.Counter(u.corpus for u in utts)
        print(f"\n[{split}] {len(utts)} utts {dict(by_corpus)} | paper: {PAPER[split]}")
        l2_spk = sorted({u.speaker for u in utts if u.corpus == "l2arctic"})
        speakers[split] = set(l2_spk)
        print(f"  L2-ARCTIC speakers ({len(l2_spk)}): {' '.join(l2_spk)}")
        if split in PAPER_SPK and set(l2_spk) != PAPER_SPK[split]:
            print(f"  !! differs from the paper's split: {sorted(PAPER_SPK[split])}")
        oov = collections.Counter(p for u in utts for p in u.canonical + u.transcript if p not in vocab.token_to_id)
        print(f"  phones outside vocab: {dict(oov) if oov else 'none'}")
        bad_canon = sum(1 for u in utts for p in u.canonical if p.endswith("*") or p == "err")
        print(f"  accented/err tokens inside canonical: {bad_canon}")
        timit = [u for u in utts if u.corpus == "timit"]
        if timit:
            errs = sum(edit_distance(u.canonical, u.transcript) for u in timit)
            n = sum(len(u.canonical) for u in timit)
            same = sum(u.canonical == u.transcript for u in timit)
            print(f"  TIMIT canonical vs transcript: PER {100 * errs / n:.2f}%, identical {same}/{len(timit)}")
        l2 = [u for u in utts if u.corpus == "l2arctic"]
        if l2:
            errs = sum(edit_distance(u.canonical, u.transcript) for u in l2)
            n = sum(len(u.canonical) for u in l2)
            print(f"  L2-ARCTIC canonical vs transcript: PER {100 * errs / n:.2f}%")
        if args.audio_root:
            missing = [u.path for u in utts if resolve_audio(args.audio_root, u.path) is None]
            print(f"  audio missing: {len(missing)}" + (f" e.g. {missing[:3]}" if missing else ""))
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        ov = speakers[a] & speakers[b]
        print(f"speaker overlap {a}/{b}: {sorted(ov) if ov else 'none'}")


if __name__ == "__main__":
    main()
