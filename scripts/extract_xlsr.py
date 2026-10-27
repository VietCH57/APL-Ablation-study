#!/usr/bin/env python
"""Extract phonetic embeddings from a frozen XLS-R model (paper 2.2: embeddings come from a
pre-trained acoustic model and are fed to the phonetic encoder; the acoustic model is not
updated while training APL).

The embedding is the output of the last Transformer layer (`last_hidden_state`, 1024-dim,
one frame per 20 ms). For a fine-tuned checkpoint this is the input of the CTC head, i.e. the
head itself is dropped.

    # XLS-R-300M as released (no fine-tuning)
    python scripts/extract_xlsr.py --model facebook/wav2vec2-xls-r-300m --out_dir feats/xlsr_raw ...
    # XLS-R fine-tuned for phoneme recognition with XLSR-300M-Phoneme-Recognition
    python scripts/extract_xlsr.py --model /path/to/runs/apl_ft/best --out_dir feats/xlsr_ft ...

--model accepts a Hugging Face id / directory, or a checkpoint directory of
XLSR-300M-Phoneme-Recognition (encoder_config.json + model.pt).
"""

import argparse
import json
import os
import sys
import logging
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.audio import load_audio, resolve_audio  # noqa: E402
from apl.gpu import DutyCycle, limit_gpu_memory  # noqa: E402
from apl.manifest import load_split  # noqa: E402


def load_encoder(model: str):
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    enc_cfg = os.path.join(model, "encoder_config.json")
    weights = os.path.join(model, "model.pt")
    if os.path.isfile(enc_cfg) and os.path.isfile(weights):  # XLSR-300M-Phoneme-Recognition checkpoint
        enc = Wav2Vec2Model(Wav2Vec2Config.from_json_file(enc_cfg))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(enc_state, strict=False)
        # masked_spec_embed is only used for SpecAugment during training
        missing = [k for k in missing if k != "masked_spec_embed"]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        print(f"loaded fine-tuned encoder from {model} (CTC head dropped)")
        return enc
    print(f"loaded {model} from Hugging Face")
    return Wav2Vec2Model.from_pretrained(model)


@torch.inference_mode()
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=-1,
                    help="-1 = last_hidden_state (default); k >= 0 = hidden_states[k] (0 = CNN/projection output)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu_memory_fraction", type=float, default=0.5, help="VRAM cap (0 = none)")
    ap.add_argument("--gpu_duty_pct", type=int, default=50, help="max %% of wall time the GPU is busy")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    limit_gpu_memory(args.gpu_memory_fraction, device)
    duty = DutyCycle(args.gpu_duty_pct, device)
    enc = load_encoder(args.model).to(device).eval()
    utts = [u for s in ("train", "dev", "test") for u in load_split(os.path.join(args.data_dir, f"{s}.csv"))]
    t0, dim = time.time(), None
    for i, u in enumerate(utts):
        out_path = os.path.join(args.out_dir, u.utt_id + ".npy")
        if os.path.isfile(out_path):
            continue
        path = resolve_audio(args.audio_root, u.path)
        if path is None:
            sys.exit(f"missing audio: {u.path}")
        wav = load_audio(path)
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)  # XLS-R expects zero-mean / unit-variance input
        duty.start()
        x = torch.from_numpy(wav).unsqueeze(0).to(device)
        out = enc(x, output_hidden_states=args.layer >= 0)
        h = out.last_hidden_state if args.layer < 0 else out.hidden_states[args.layer]
        h = h[0].float().cpu().numpy().astype(np.float16)
        duty.step()
        dim = h.shape[1]
        tmp = out_path + ".tmp.npy"
        np.save(tmp, h)
        os.replace(tmp, out_path)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(utts)} ({time.time() - t0:.0f}s)", flush=True)
    meta = {"model": args.model, "layer": args.layer, "dim": dim, "frame_shift_ms": 20}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {len(utts)} utterances -> {args.out_dir}")


if __name__ == "__main__":
    main()
