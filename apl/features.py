"""Acoustic features and frame alignment of phonetic embeddings.

Acoustic features (paper 2.1): 81-dim = 80 log Mel filter banks + 1 log energy, 10 ms shift,
16 kHz. Extracted Kaldi-style as in [14] (conf/fbank.conf: --num-mel-bins=80 --use-energy
--window-type=hamming, 25 ms window).

Normalisation: global CMVN as in [14] (steps/make_feat.sh: compute-cmvn-stats on the
training set, apply-cmvn --norm-vars=true on train/dev/test). The APL paper does not mention
any normalisation; this is the baseline's.

Phonetic embeddings (paper 2.2): must have the same number of frames T as X. XLS-R produces
one frame per 20 ms, so each frame is repeated twice (10 ms) and the result is cropped /
edge-padded to T.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

FBANK_DIM = 81


def compute_fbank(wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """(num_samples,) float in [-1, 1] -> (T, 81) float32: [log energy, 80 log mel]."""
    x = torch.from_numpy(wav).float().unsqueeze(0) * 32768.0  # Kaldi works on int16 scale
    feat = kaldi.fbank(
        x,
        sample_frequency=sample_rate,
        num_mel_bins=80,
        use_energy=True,
        window_type="hamming",
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
    )
    return feat.numpy().astype(np.float32)


class CMVN:
    """Global mean / variance normalisation with statistics from the training set."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def from_stats(cls, total: np.ndarray, total_sq: np.ndarray, count: int) -> "CMVN":
        mean = total / count
        var = np.maximum(total_sq / count - mean**2, 1e-10)  # Kaldi floors the variance
        return cls(mean, np.sqrt(var))

    @classmethod
    def load(cls, path: str) -> "CMVN":
        d = np.load(path)
        return cls(d["mean"], d["std"])

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        return (feat - self.mean) / self.std


def align_to_frames(emb: np.ndarray, num_frames: int, repeat: int = 2) -> np.ndarray:
    """Upsample an embedding sequence by frame repetition and crop / edge-pad to `num_frames`."""
    if repeat > 1:
        emb = np.repeat(emb, repeat, axis=0)
    n = emb.shape[0]
    if n >= num_frames:
        return emb[:num_frames]
    pad = np.repeat(emb[-1:], num_frames - n, axis=0)
    return np.concatenate([emb, pad], axis=0)
