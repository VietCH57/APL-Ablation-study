"""Audio loading: wav / flac / NIST SPHERE (TIMIT) -> mono float32 at 16 kHz."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_EXTS = (".wav", ".WAV", ".flac")


def _candidates(rel: str, exts: Sequence[str]):
    rel = rel.strip().strip("/")
    variants = [rel, rel.upper()]
    # TIMIT distributions differ in case: TRAIN/DR1/FCJF0/SA1.WAV vs train/dr1/fcjf0/sa1.wav
    parts = rel.split("/")
    if len(parts) == 4:
        variants.append("/".join(p.upper() for p in parts[:3]) + "/" + parts[3].upper())
        variants.append("/".join(p.lower() for p in parts))
    seen = set()
    for v in variants:
        for e in exts:
            p = v + e
            if p not in seen:
                seen.add(p)
                yield p


@lru_cache(maxsize=None)
def resolve_audio(audio_root: str, rel: str, exts: Sequence[str] = DEFAULT_EXTS) -> Optional[str]:
    """Find the audio file for a CSV `Path` (which has no extension)."""
    if os.path.splitext(rel)[1].lower() in (".wav", ".flac"):
        p = os.path.join(audio_root, rel)
        return p if os.path.isfile(p) else None
    for cand in _candidates(rel, tuple(exts)):
        p = os.path.join(audio_root, cand)
        if os.path.isfile(p):
            return p
    return None


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # libsndfile reads NIST SPHERE too
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        import torch
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(wav), sr, sample_rate).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)
