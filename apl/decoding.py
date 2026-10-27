"""CTC decoding: greedy (dev monitoring) and prefix beam search (paper 2.4: "the recognized
phoneme sequence is obtained by beam-search"; beam width 10 and no language model, as in
[14] conf/ctc_config.yaml: beam_width 10, lm_alpha 0)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List

import numpy as np
import torch


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int) -> List[List[int]]:
    best = log_probs.argmax(-1).cpu().numpy()
    out = []
    for b, n in enumerate(lengths.tolist()):
        seq, prev = [], -1
        for t in best[b, :n]:
            if t != prev and t != blank:
                seq.append(int(t))
            prev = t
        out.append(seq)
    return out


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def prefix_beam_search(log_probs: np.ndarray, blank: int, beam: int = 10, prune: float = -12.0) -> List[int]:
    """Standard CTC prefix beam search over one utterance. log_probs: (T, C)."""
    beams = {(): (0.0, -math.inf)}  # prefix -> (log P ending in blank, log P ending in non-blank)
    for t in range(log_probs.shape[0]):
        lp = log_probs[t]
        cand = np.where(lp > prune)[0]
        if cand.size == 0:
            cand = np.array([int(lp.argmax())])
        nxt = defaultdict(lambda: (-math.inf, -math.inf))
        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            for c in cand:
                c = int(c)
                p = float(lp[c])
                if c == blank:
                    nb, nnb = nxt[prefix]
                    nxt[prefix] = (_logsumexp(nb, total + p), nnb)
                    continue
                new = prefix + (c,)
                nb, nnb = nxt[new]
                if prefix and prefix[-1] == c:
                    # repeated symbol: only extends from a path that ended in blank
                    nxt[new] = (nb, _logsumexp(nnb, pb + p))
                    ob, onb = nxt[prefix]
                    nxt[prefix] = (ob, _logsumexp(onb, pnb + p))
                else:
                    nxt[new] = (nb, _logsumexp(nnb, total + p))
        beams = dict(sorted(nxt.items(), key=lambda kv: _logsumexp(*kv[1]), reverse=True)[:beam])
    best = max(beams.items(), key=lambda kv: _logsumexp(*kv[1]))[0]
    return list(best)


def beam_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int, beam: int = 10) -> List[List[int]]:
    lp = log_probs.float().cpu().numpy()
    return [prefix_beam_search(lp[b, :n], blank, beam) for b, n in enumerate(lengths.tolist())]
