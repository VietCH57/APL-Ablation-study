#!/usr/bin/env python
"""Export the APL splits in the format of XLSR-300M-Phoneme-Recognition, so that XLS-R-300M
is fine-tuned for phoneme recognition on exactly the APL training data (TIMIT + 12 L2-ARCTIC
speakers) and selected on the APL dev speakers. Test speakers are never seen.

    python scripts/make_xlsr_finetune_data.py --data_dir data --audio_root /path/to/wavs --out_dir xlsr_ft_data

Writes <out_dir>/{train,dev,test}.csv (columns audio,text,domain) and <out_dir>/vocab.json
(the 68 output phones + [UNK] + [PAD], [PAD] being the CTC blank as that repo expects).
Then, inside XLSR-300M-Phoneme-Recognition:

    python scripts/train.py --config <this repo>/configs/xlsr_finetune.yaml \
        --train_csv <out_dir>/train.csv --dev_csv <out_dir>/dev.csv --vocab_path <out_dir>/vocab.json \
        --output_dir runs/apl_ft
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.audio import resolve_audio  # noqa: E402
from apl.manifest import load_split  # noqa: E402
from apl.vocab import Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--vocab", default="configs/vocab.json")
    ap.add_argument("--out_dir", default="xlsr_ft_data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vocab = Vocab.from_file(args.vocab)
    ft_vocab = {p: i for i, p in enumerate(vocab.phones)}
    ft_vocab["[UNK]"] = len(ft_vocab)
    ft_vocab["[PAD]"] = len(ft_vocab)
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(ft_vocab, f, indent=2)

    for split in ("train", "dev", "test"):
        utts = load_split(os.path.join(args.data_dir, f"{split}.csv"))
        rows = []
        for u in utts:
            path = resolve_audio(args.audio_root, u.path)
            if path is None:
                sys.exit(f"missing audio: {u.path}")
            rows.append({"audio": os.path.abspath(path), "text": " ".join(u.transcript),
                         "domain": "native_en" if u.corpus == "timit" else "l2a"})
        with open(os.path.join(args.out_dir, f"{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["audio", "text", "domain"])
            w.writeheader()
            w.writerows(rows)
        print(f"{split}: {len(rows)} rows")


if __name__ == "__main__":
    main()
