"""APL: acoustic, phonetic and linguistic embeddings for MD&D (Ye et al., ICASSP 2022).

Paper references (section / equation) are given next to each part. Sizes the paper does not
state are those of the baseline it says it reuses ("The parameters of acoustic encoder,
phonetic encoder, linguistic encoder and decoder are the same as baselines", 3.3), taken from
the released code of [14] (github.com/cageyoko/CTC-Attention-Mispronunciation,
conf/ctc_config.yaml and models/model_ctc.py):

    CNN stack   Conv2D(3x3, pad 1) -> BatchNorm -> ReLU -> Dropout, channels 1->32->32,
                stride (time, freq) = (1, 2) then (2, 2)  => T' = ceil(T / 2)
    RNN stack   BiLSTM(384 per direction) -> BatchNorm -> Dropout     (order of Fig. 1b)
    dropout     0.2
    linguistic  Embedding(512) -> BiLSTM(384 per direction) -> Linear (no bias)

Variants (ablation): any non-empty subset of {A, P} for the frame-level query, with or
without L.
    APL  H^Q = [H^a; H^p], attention over the canonical phones (eq. 4-7)
    PL   H^Q = H^p,        attention over the canonical phones
    AL   H^Q = H^a,        attention over the canonical phones (= Baseline-2 / [14])
    P    H^Q = H^p,        no linguistic encoder: y = softmax(W h^Q + b)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class APLConfig:
    use_acoustic: bool = True
    use_phonetic: bool = True
    use_linguistic: bool = True
    num_classes: int = 69  # 68 phones + CTC blank
    num_embeddings: int = 70  # canonical vocabulary incl. blank and pad
    pad_id: int = 69
    acoustic_dim: int = 81
    phonetic_dim: int = 1024
    conv_channels: Tuple[int, ...] = (32, 32)
    conv_kernels: Tuple[Tuple[int, int], ...] = ((3, 3), (3, 3))
    conv_strides: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 2))  # (time, freq)
    conv_paddings: Tuple[Tuple[int, int], ...] = ((1, 1), (1, 1))
    rnn_hidden: int = 384
    acoustic_rnn_layers: int = 4
    phonetic_rnn_layers: int = 1
    embed_dim: int = 512
    dropout: float = 0.2

    @classmethod
    def from_dict(cls, d: Dict) -> "APLConfig":
        d = dict(d)
        for k in ("conv_channels",):
            if k in d:
                d[k] = tuple(d[k])
        for k in ("conv_kernels", "conv_strides", "conv_paddings"):
            if k in d:
                d[k] = tuple(tuple(x) for x in d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def name(self) -> str:
        return ("A" if self.use_acoustic else "") + ("P" if self.use_phonetic else "") + (
            "L" if self.use_linguistic else "")


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class MaskedBatchNorm(nn.Module):
    """BatchNorm whose batch statistics ignore padded frames (otherwise identical to nn.BatchNorm).

    x: (B, C, T, F) with channel_dim=1, or (B, T, C) with channel_dim=2.
    mask: bool, broadcastable to x, True on valid frames.
    """

    def __init__(self, num_features: int, channel_dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.channel_dim = channel_dim
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _shape(self, x: torch.Tensor) -> List[int]:
        s = [1] * x.dim()
        s[self.channel_dim] = -1
        return s

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        m = mask.to(x.dtype).expand_as(x)
        shape = self._shape(x)
        if self.training:
            dims = [d for d in range(x.dim()) if d != self.channel_dim]
            n = m.sum(dims).clamp_min(1.0)
            mean = (x * m).sum(dims) / n
            var = (((x - mean.view(shape)) ** 2) * m).sum(dims) / n
            with torch.no_grad():
                unbiased = var * n / (n - 1).clamp_min(1.0)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased)
        else:
            mean, var = self.running_mean, self.running_var
        y = (x - mean.view(shape)) * torch.rsqrt(var.view(shape) + self.eps)
        y = y * self.weight.view(shape) + self.bias.view(shape)
        return (y * m).to(dtype)


class CNNStack(nn.Module):
    """One CNN stack (paper 2.1): Conv2D -> BatchNorm -> ReLU -> Dropout.

    Padded frames are set to zero after the convolution and after the stack (as with the zero
    padding of [14]); the 2-D BatchNorm is the standard cuDNN one because its activations are
    the largest tensors of the model (batches are length-bucketed, so padding is small).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel, stride, padding, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.kernel, self.stride, self.padding = kernel, stride, padding

    def out_len(self, n: torch.Tensor, axis: int) -> torch.Tensor:
        return (n + 2 * self.padding[axis] - self.kernel[axis]) // self.stride[axis] + 1

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (B, C, T, F)
        x = self.conv(x)
        lengths = self.out_len(lengths, 0)
        mask = lengths_to_mask(lengths, x.size(2))[:, None, :, None].to(x.dtype)
        x = self.drop(F.relu(self.bn(x * mask)))
        return x * mask, lengths


class RNNStack(nn.Module):
    """One RNN stack (paper 2.1): bi-directional LSTM -> BatchNorm -> Dropout."""

    def __init__(self, input_size: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.bn = MaskedBatchNorm(2 * hidden, channel_dim=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = lengths_to_mask(lengths, out.size(1))[:, :, None]
        return self.drop(self.bn(out, mask))


class FrameEncoder(nn.Module):
    """CNN stacks followed by RNN stacks.

    Acoustic encoder E_a (2.1): 2 CNN stacks + 4 RNN stacks on the 81-dim fbank.
    Phonetic encoder E_p (2.2): the same CNN stacks (so that H^p and H^a have the same time
    resolution) + 1 RNN stack on the phonetic embeddings.
    """

    def __init__(self, in_dim: int, n_rnn: int, cfg: APLConfig):
        super().__init__()
        chans = (1,) + tuple(cfg.conv_channels)
        self.cnns = nn.ModuleList(
            CNNStack(chans[i], chans[i + 1], cfg.conv_kernels[i], cfg.conv_strides[i], cfg.conv_paddings[i],
                     cfg.dropout)
            for i in range(len(cfg.conv_channels))
        )
        freq = torch.tensor(in_dim)
        for c in self.cnns:
            freq = c.out_len(freq, 1)
        rnn_in = cfg.conv_channels[-1] * int(freq)
        self.rnns = nn.ModuleList(
            RNNStack(rnn_in if i == 0 else 2 * cfg.rnn_hidden, cfg.rnn_hidden, cfg.dropout) for i in range(n_rnn)
        )
        self.out_dim = 2 * cfg.rnn_hidden

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for c in self.cnns:
            lengths = c.out_len(lengths, 0)
        return lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (B, T, F), zero beyond `lengths`
        x = x.unsqueeze(1)  # (B, 1, T, F)
        for c in self.cnns:
            x, lengths = c(x, lengths)
        b, ch, t, f = x.shape
        x = x.transpose(1, 2).reshape(b, t, ch * f)
        for r in self.rnns:
            x = r(x, lengths)
        return x, lengths


class LinguisticEncoder(nn.Module):
    """Linguistic encoder E_l (2.3, Fig. 1b): Embedding -> BiLSTM -> Linear.

    H^V is the BiLSTM output, H^K = Linear(H^V). The Linear maps to the size of the query so that
    the dot product h^Q_t' (h^K_n)^T of eq. (5) is defined.
    """

    def __init__(self, cfg: APLConfig, key_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(cfg.num_embeddings, cfg.embed_dim, padding_idx=cfg.pad_id)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.rnn_hidden, batch_first=True, bidirectional=True)
        self.key = nn.Linear(2 * cfg.rnn_hidden, key_dim, bias=False)
        self.value_dim = 2 * cfg.rnn_hidden

    def forward(self, s: torch.Tensor, lengths: torch.Tensor):
        e = self.embedding(s)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hv, _ = self.lstm(packed)
        hv, _ = pad_packed_sequence(hv, batch_first=True, total_length=s.size(1))
        hk = self.key(hv)
        return hk, hv


class APLModel(nn.Module):
    def __init__(self, cfg: APLConfig):
        super().__init__()
        if not (cfg.use_acoustic or cfg.use_phonetic):
            raise ValueError("at least one of the acoustic / phonetic branches is required")
        self.cfg = cfg
        self.acoustic = FrameEncoder(cfg.acoustic_dim, cfg.acoustic_rnn_layers, cfg) if cfg.use_acoustic else None
        self.phonetic = FrameEncoder(cfg.phonetic_dim, cfg.phonetic_rnn_layers, cfg) if cfg.use_phonetic else None
        q_dim = sum(e.out_dim for e in (self.acoustic, self.phonetic) if e is not None)
        self.query_dim = q_dim
        if cfg.use_linguistic:
            self.linguistic = LinguisticEncoder(cfg, key_dim=q_dim)
            out_in = self.linguistic.value_dim + q_dim  # [c_t'; h^Q_t']
        else:
            self.linguistic = None
            out_in = q_dim
        self.output = nn.Linear(out_in, cfg.num_classes)  # W, b of eq. (7)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        enc = self.acoustic if self.acoustic is not None else self.phonetic
        return enc.out_lengths(lengths)

    def forward(
        self,
        lengths: torch.Tensor,
        fbank: Optional[torch.Tensor] = None,
        phonetic: Optional[torch.Tensor] = None,
        canonical: Optional[torch.Tensor] = None,
        canonical_lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        hs, out_len = [], None
        if self.acoustic is not None:
            ha, out_len = self.acoustic(fbank, lengths)  # eq. (1)
            hs.append(ha)
        if self.phonetic is not None:
            hp, out_len_p = self.phonetic(phonetic, lengths)  # eq. (2)
            if out_len is not None and not torch.equal(out_len, out_len_p):
                raise RuntimeError("H^a and H^p have different time resolution")
            out_len = out_len_p
            hs.append(hp)
        hq = torch.cat(hs, dim=-1) if len(hs) > 1 else hs[0]  # eq. (4)
        out: Dict[str, torch.Tensor] = {"output_lengths": out_len}

        if self.linguistic is not None:
            hk, hv = self.linguistic(canonical, canonical_lengths)  # eq. (3)
            scores = torch.bmm(hq.float(), hk.float().transpose(1, 2))  # h^Q_t' (h^K_n)^T
            key_mask = lengths_to_mask(canonical_lengths, hk.size(1))[:, None, :]
            scores = scores.masked_fill(~key_mask, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)  # eq. (5)
            c = torch.bmm(alpha, hv.float()).to(hq.dtype)  # eq. (6)
            feats = torch.cat([c, hq], dim=-1)  # [c_t'; h^Q_t']
            if return_attention:
                out["attention"] = alpha
        else:
            feats = hq
        logits = self.output(feats)
        out["log_probs"] = F.log_softmax(logits.float(), dim=-1)  # eq. (7), log domain for CTC
        return out


def ctc_loss(out: Dict[str, torch.Tensor], targets: torch.Tensor, target_lengths: torch.Tensor,
             blank: int) -> torch.Tensor:
    """Summed CTC loss divided by the batch size, as in [14] (CTCLoss(reduction='sum') / batch)."""
    lp = out["log_probs"].transpose(0, 1)  # (T', B, C)
    loss = F.ctc_loss(lp, targets, out["output_lengths"], target_lengths, blank=blank, reduction="sum",
                      zero_infinity=True)
    return loss / lp.size(1)
