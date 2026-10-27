"""Dataset over pre-extracted features.

Feature layout on disk (written by scripts/extract_fbank.py and scripts/extract_xlsr.py):
    <fbank_dir>/<utt_id>.npy      (T, 81) float32, raw (CMVN applied here)
    <fbank_dir>/cmvn.npz          global CMVN from the training split
    <phon_dir>/<utt_id>.npy       (L, D) float16 at 20 ms, upsampled here to T frames
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .features import CMVN, align_to_frames
from .manifest import Utterance
from .vocab import Vocab


class APLDataset(Dataset):
    def __init__(
        self,
        utts: Sequence[Utterance],
        vocab: Vocab,
        fbank_dir: str,
        phon_dir: Optional[str] = None,
        use_acoustic: bool = True,
        use_phonetic: bool = True,
        phon_repeat: int = 2,
    ):
        if use_phonetic and not phon_dir:
            raise ValueError("phon_dir is required when the phonetic branch is used")
        self.utts = list(utts)
        self.vocab = vocab
        self.fbank_dir = fbank_dir
        self.phon_dir = phon_dir
        self.use_acoustic = use_acoustic
        self.use_phonetic = use_phonetic
        self.phon_repeat = phon_repeat
        self.cmvn = CMVN.load(os.path.join(fbank_dir, "cmvn.npz"))
        self.canonical = [torch.tensor(vocab.encode(u.canonical), dtype=torch.long) for u in self.utts]
        self.targets = [torch.tensor(vocab.encode(u.transcript), dtype=torch.long) for u in self.utts]
        self._lengths: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.utts)

    def num_frames(self, idx: int) -> int:
        fb = np.load(os.path.join(self.fbank_dir, self.utts[idx].utt_id + ".npy"), mmap_mode="r")
        return int(fb.shape[0])

    @property
    def lengths(self) -> List[int]:
        if self._lengths is None:
            self._lengths = [self.num_frames(i) for i in range(len(self))]
        return self._lengths

    def __getitem__(self, idx: int) -> Dict:
        u = self.utts[idx]
        item = {"index": idx, "utt_id": u.utt_id, "canonical": self.canonical[idx], "target": self.targets[idx]}
        fb_path = os.path.join(self.fbank_dir, u.utt_id + ".npy")
        if self.use_acoustic:
            fb = np.load(fb_path)
            item["fbank"] = torch.from_numpy(self.cmvn(fb))
            num_frames = fb.shape[0]
        else:
            num_frames = int(np.load(fb_path, mmap_mode="r").shape[0])
        if self.use_phonetic:
            emb = np.load(os.path.join(self.phon_dir, u.utt_id + ".npy")).astype(np.float32)
            item["phonetic"] = torch.from_numpy(align_to_frames(emb, num_frames, self.phon_repeat))
        item["num_frames"] = num_frames
        return item


def collate(batch: List[Dict], pad_id: int) -> Dict:
    bsz = len(batch)
    lengths = torch.tensor([b["num_frames"] for b in batch], dtype=torch.long)
    t_max = int(lengths.max())
    out: Dict = {"lengths": lengths, "utt_ids": [b["utt_id"] for b in batch],
                 "indices": torch.tensor([b["index"] for b in batch])}
    for key in ("fbank", "phonetic"):
        if key in batch[0]:
            dim = batch[0][key].shape[1]
            x = torch.zeros(bsz, t_max, dim)
            for i, b in enumerate(batch):
                x[i, : b["num_frames"]] = b[key]
            out[key] = x
    can_len = torch.tensor([len(b["canonical"]) for b in batch], dtype=torch.long)
    can = torch.full((bsz, int(can_len.max())), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        can[i, : can_len[i]] = b["canonical"]
    out["canonical"], out["canonical_lengths"] = can, can_len
    out["targets"] = torch.cat([b["target"] for b in batch])
    out["target_lengths"] = torch.tensor([len(b["target"]) for b in batch], dtype=torch.long)
    return out


class BucketBatchSampler(Sampler[List[int]]):
    """Shuffled fixed-size batches of similar length (less padding, same batch size as the paper).

    Utterances are shuffled, grouped into chunks of `bucket_mult * batch_size`, each chunk is sorted
    by length and cut into batches, and the batches are shuffled.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, bucket_mult: int = 50, seed: int = 0,
                 shuffle: bool = True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket = batch_size * bucket_mult
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.bucket):
            chunk = sorted(idx[s : s + self.bucket], key=lambda i: self.lengths[i])
            batches += [chunk[j : j + self.batch_size] for j in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
