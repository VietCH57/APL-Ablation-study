"""Shared pieces of training and evaluation."""

from __future__ import annotations

import json
import logging
import os
import random
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import APLDataset, BucketBatchSampler, collate
from .decoding import beam_decode, greedy_decode
from .gpu import DutyCycle, NoDuty
from .manifest import load_split
from .metrics import evaluate
from .model import APLConfig, APLModel, ctc_loss
from .vocab import Vocab

logger = logging.getLogger("apl")


def setup_logging(out_dir: Optional[str] = None, name: str = "train.log") -> None:
    handlers = [logging.StreamHandler()]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(out_dir, name)))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def variant_flags(variant: str) -> Dict[str, bool]:
    v = variant.upper()
    if v not in ("APL", "AL", "PL", "P", "A", "AP"):
        raise ValueError(f"unknown variant {variant}")
    return {"use_acoustic": "A" in v, "use_phonetic": "P" in v, "use_linguistic": "L" in v}


def phonetic_dim(phon_dir: str) -> int:
    meta = os.path.join(phon_dir, "meta.json")
    if os.path.isfile(meta):
        with open(meta) as f:
            d = json.load(f).get("dim")
        if d:
            return int(d)
    first = next(f for f in os.listdir(phon_dir) if f.endswith(".npy"))
    return int(np.load(os.path.join(phon_dir, first), mmap_mode="r").shape[1])


def build_dataset(split: str, cfg: Dict, vocab: Vocab, flags: Dict[str, bool]) -> APLDataset:
    utts = load_split(os.path.join(cfg["data_dir"], f"{split}.csv"))
    return APLDataset(utts, vocab, cfg["fbank_dir"], cfg.get("phon_dir"), flags["use_acoustic"],
                      flags["use_phonetic"])


def make_loader(ds: APLDataset, batch_size: int, train: bool, seed: int, num_workers: int, pad_id: int):
    sampler = BucketBatchSampler(ds.lengths, batch_size, seed=seed, shuffle=train)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True,
                        collate_fn=lambda b: collate(b, pad_id), persistent_workers=num_workers > 0)
    return loader, sampler


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def autocast(device: torch.device, amp: str):
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward(model: APLModel, batch: Dict, return_attention: bool = False) -> Dict[str, torch.Tensor]:
    return model(batch["lengths"], batch.get("fbank"), batch.get("phonetic"), batch.get("canonical"),
                 batch.get("canonical_lengths"), return_attention=return_attention)


@torch.no_grad()
def run_eval(model: APLModel, loader, vocab: Vocab, device, amp: str, beam: int = 0,
             duty: Optional[DutyCycle] = None) -> Dict:
    """Loss + decoding of a whole split. beam=0 -> greedy."""
    duty = duty or NoDuty()
    model.eval()
    ds = loader.dataset
    total_loss, n_batches = 0.0, 0
    preds: Dict[int, List[int]] = {}
    for batch in loader:
        duty.start()
        batch = to_device(batch, device)
        with autocast(device, amp):
            out = forward(model, batch)
        total_loss += ctc_loss(out, batch["targets"], batch["target_lengths"], vocab.blank_id).item()
        n_batches += 1
        duty.stop()  # GPU work of this batch ends here; decoding below runs on the CPU
        if beam > 0:
            hyps = beam_decode(out["log_probs"], out["output_lengths"], vocab.blank_id, beam)
        else:
            hyps = greedy_decode(out["log_probs"], out["output_lengths"], vocab.blank_id)
        duty.throttle()
        for idx, h in zip(batch["indices"].tolist(), hyps):
            preds[idx] = h
    order = sorted(preds)
    canon = [ds.utts[i].canonical for i in order]
    human = [ds.utts[i].transcript for i in order]
    pred = [vocab.decode(preds[i]) for i in order]
    res = evaluate(canon, human, pred)
    res["loss"] = total_loss / max(n_batches, 1)
    res["_rows"] = [{"utt_id": ds.utts[i].utt_id, "canonical": " ".join(c), "human": " ".join(h),
                     "pred": " ".join(p)} for i, c, h, p in zip(order, canon, human, pred)]
    return res


def save_checkpoint(path: str, model: APLModel, extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": model.cfg.__dict__, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: str, map_location="cpu") -> APLModel:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = APLModel(APLConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model
