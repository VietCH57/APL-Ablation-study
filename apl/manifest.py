"""Reading the split CSVs (columns: <index>, Path, Canonical, Transcript).

Path examples:
    L2_arctic_WAV/ABA_arctic_a0003      (L2-ARCTIC)
    train/dr6/mabc0/sa1                 (TIMIT train)
    test/dr1/faks0/sa1                  (TIMIT test, used for training as in the paper)

Canonical and Transcript are used exactly as given in the CSV for every corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

L2_PREFIX = "L2_arctic"


@dataclass
class Utterance:
    utt_id: str
    path: str  # relative path from the CSV (no extension)
    corpus: str  # "timit" or "l2arctic"
    speaker: str
    canonical: List[str]
    transcript: List[str]


def utt_id_from_path(path: str) -> str:
    return path.strip().strip("/").replace("/", "__")


def corpus_of(path: str) -> str:
    return "l2arctic" if path.startswith(L2_PREFIX) else "timit"


def speaker_of(path: str) -> str:
    parts = path.strip("/").split("/")
    if corpus_of(path) == "l2arctic":
        return parts[-1].split("_")[0]
    return parts[-2]  # TIMIT: <split>/<dr>/<speaker>/<utt>


def load_split(csv_path: str) -> List[Utterance]:
    df = pd.read_csv(csv_path, index_col=0, keep_default_na=False, dtype=str)
    for col in ("Path", "Canonical", "Transcript"):
        if col not in df.columns:
            raise ValueError(f"{csv_path}: missing column {col}")
    utts: List[Utterance] = []
    for path, canon, trans in zip(df["Path"], df["Canonical"], df["Transcript"]):
        path = path.strip()
        corpus = corpus_of(path)
        utts.append(Utterance(utt_id_from_path(path), path, corpus, speaker_of(path), canon.split(), trans.split()))
    ids = [u.utt_id for u in utts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{csv_path}: duplicate paths")
    return utts
