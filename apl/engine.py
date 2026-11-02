"""Shared pieces of training and evaluation."""

from __future__ import annotations

import json
import logging
import math
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


class FrameBudget:
    """Largest batch_size x max_frames processed in one forward pass.

    A batch of `batch_size` utterances that does not fit in GPU memory is cut into k consecutive
    chunks (the batch is length-sorted, so chunks have similar lengths); the gradients of the
    chunks are accumulated and one optimizer step is taken per batch, so the update is that of the
    full batch except that BatchNorm statistics are computed per chunk. The budget starts at
    `initial` (0 = unlimited) and is halved whenever a chunk runs out of memory.
    """

    def __init__(self, initial: int = 0):
        self.value = initial if initial and initial > 0 else None

    def splits(self, batch: Dict) -> int:
        n = len(batch["lengths"])
        if self.value is None:
            return 1
        frames = n * int(batch["lengths"].max())
        return max(1, min(n, math.ceil(frames / self.value)))

    def shrink(self, batch: Dict, k: int) -> None:
        frames = len(batch["lengths"]) * int(batch["lengths"].max())
        self.value = max(1, frames // (2 * k))


def split_batch(batch: Dict, k: int) -> List[Dict]:
    """Cut a collated batch into k consecutive chunks, trimming padding to each chunk's length."""
    n = len(batch["lengths"])
    if k <= 1:
        return [batch]
    size = math.ceil(n / k)
    tgt_ends = torch.cumsum(batch["target_lengths"], 0)
    chunks = []
    for s in range(0, n, size):
        e = min(n, s + size)
        t_max = int(batch["lengths"][s:e].max())
        c_max = int(batch["canonical_lengths"][s:e].max())
        t0 = 0 if s == 0 else int(tgt_ends[s - 1])
        t1 = int(tgt_ends[e - 1])
        ch = {"lengths": batch["lengths"][s:e], "utt_ids": batch["utt_ids"][s:e],
              "indices": batch["indices"][s:e], "canonical": batch["canonical"][s:e, :c_max],
              "canonical_lengths": batch["canonical_lengths"][s:e], "targets": batch["targets"][t0:t1],
              "target_lengths": batch["target_lengths"][s:e]}
        for key in ("fbank", "phonetic"):
            if key in batch:
                ch[key] = batch[key][s:e, :t_max]
        chunks.append(ch)
    return chunks


def autocast(device: torch.device, amp: str):
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward(model: APLModel, batch: Dict, return_attention: bool = False) -> Dict[str, torch.Tensor]:
    return model(batch["lengths"], batch.get("fbank"), batch.get("phonetic"), batch.get("canonical"),
                 batch.get("canonical_lengths"), return_attention=return_attention)


def train_step(model: APLModel, batch: Dict, optimizer, vocab: Vocab, device, amp: str,
               budget: FrameBudget) -> float:
    """One optimizer step on one batch (split into chunks if it does not fit in memory).
    Returns the batch loss: summed CTC loss / batch size, as in [14]."""
    n = len(batch["lengths"])
    while True:
        k = budget.splits(batch)
        oom = False
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        try:
            for ch in split_batch(batch, k):
                ch = to_device(ch, device)
                with autocast(device, amp):
                    out = forward(model, ch)
                loss = ctc_loss(out, ch["targets"], ch["target_lengths"], vocab.blank_id, batch_size=n)
                loss.backward()
                total += loss.item()
                del out, loss, ch
        except torch.cuda.OutOfMemoryError:
            if k >= n:
                raise
            oom = True
        if not oom:
            optimizer.step()
            return total
        # cleanup outside the except block so the traceback does not keep tensors alive
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        budget.shrink(batch, k)
        logger.warning("CUDA OOM on a batch of %d x %d frames in %d chunk(s); budget -> %d frames, retrying",
                       n, int(batch["lengths"].max()), k, budget.value)


@torch.no_grad()
def run_eval(model: APLModel, loader, vocab: Vocab, device, amp: str, beam: int = 0,
             duty: Optional[DutyCycle] = None, budget: Optional[FrameBudget] = None) -> Dict:
    """Loss + decoding of a whole split. beam=0 -> greedy."""
    duty = duty or NoDuty()
    budget = budget or FrameBudget()
    model.eval()
    ds = loader.dataset
    total_loss, n_batches = 0.0, 0
    preds: Dict[int, List[int]] = {}
    for batch in loader:
        duty.start()
        n = len(batch["lengths"])
        while True:
            k = budget.splits(batch)
            outs, loss_sum, oom = [], 0.0, False
            try:
                for ch in split_batch(batch, k):
                    ch = to_device(ch, device)
                    with autocast(device, amp):
                        out = forward(model, ch)
                    loss_sum += ctc_loss(out, ch["targets"], ch["target_lengths"], vocab.blank_id,
                                         batch_size=n).item()
                    outs.append((ch["indices"], out["log_probs"].float().cpu(), out["output_lengths"].cpu()))
                    del out, ch
            except torch.cuda.OutOfMemoryError:
                if k >= n:
                    raise
                oom = True
            if not oom:
                break
            outs = []
            torch.cuda.empty_cache()
            budget.shrink(batch, k)
            logger.warning("CUDA OOM in evaluation (%d x %d frames); budget -> %d frames, retrying",
                           n, int(batch["lengths"].max()), budget.value)
        total_loss += loss_sum
        n_batches += 1
        duty.stop()  # GPU work of this batch ends here; decoding below runs on the CPU
        for idx, lp, ol in outs:
            if beam > 0:
                hyps = beam_decode(lp, ol, vocab.blank_id, beam)
            else:
                hyps = greedy_decode(lp, ol, vocab.blank_id)
            for i, h in zip(idx.tolist(), hyps):
                preds[i] = h
        duty.throttle()
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
