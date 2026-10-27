#!/usr/bin/env python
"""Train one APL variant, then decode dev and test with beam search.

    python scripts/train.py --config configs/apl.yaml --variant APL \
        --phon_dir feats/xlsr_ft --out_dir runs/APL-ft

Learning-rate schedule and model selection follow [14] (steps/train_ctc.py):
  * after every epoch the dev CTC loss is compared with the best so far: an improvement by more
    than `plateau_threshold` resets the counter; a change within +-threshold increments it; a
    worse loss sets it to `plateau_patience` at once;
  * when the counter reaches `plateau_patience`, the best-loss weights and optimizer state are
    restored and the learning rate is multiplied by `lr_decay`;
  * training stops after `max_lr_decays` decays or `max_epochs` epochs;
  * the kept model is the one with the best dev phone accuracy (1 - PER, greedy decoding).
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from apl.engine import (autocast, build_dataset, forward, load_checkpoint, logger, make_loader,  # noqa: E402
                        phonetic_dim, run_eval, save_checkpoint, set_seed, setup_logging, to_device,
                        variant_flags)
from apl.gpu import DutyCycle, limit_gpu_memory  # noqa: E402
from apl.metrics import format_table_row  # noqa: E402
from apl.model import APLConfig, APLModel, ctc_loss  # noqa: E402
from apl.vocab import Vocab  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/apl.yaml")
    known, rest = ap.parse_known_args()
    with open(known.config) as f:
        cfg = yaml.safe_load(f)
    # every top-level scalar key of the YAML can be overridden: --key value
    over = argparse.ArgumentParser()
    for k, v in cfg.items():
        if isinstance(v, dict):
            continue
        typ = type(v) if v is not None else str
        if typ is bool:
            over.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            over.add_argument(f"--{k}", type=typ, default=v)
    over.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    over.add_argument("--max_train_utts", type=int, default=0, help="debug: use only the first N utterances")
    args = over.parse_args(rest)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def train_epoch(model, loader, sampler, optimizer, vocab, device, amp, epoch, duty):
    model.train()
    sampler.set_epoch(epoch)
    total, n = 0.0, 0
    for batch in loader:
        duty.start()
        batch = to_device(batch, device)
        with autocast(device, amp):
            out = forward(model, batch)
        loss = ctc_loss(out, batch["targets"], batch["target_lengths"], vocab.blank_id)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
        duty.step()
    return total / max(n, 1)


def main():
    cfg = parse_args()
    out_dir = cfg["out_dir"]
    setup_logging(out_dir)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    limit_gpu_memory(cfg["gpu_memory_fraction"], device)
    duty = DutyCycle(cfg["gpu_duty_pct"], device)

    flags = variant_flags(cfg["variant"])
    if flags["use_phonetic"] and not cfg.get("phon_dir"):
        sys.exit("--phon_dir is required for variants with the phonetic branch")
    if not flags["use_phonetic"]:
        cfg["phon_dir"] = None
    vocab = Vocab.from_file(cfg["vocab"])
    train_ds = build_dataset("train", cfg, vocab, flags)
    dev_ds = build_dataset("dev", cfg, vocab, flags)
    test_ds = build_dataset("test", cfg, vocab, flags)
    if cfg["max_train_utts"]:
        train_ds.utts = train_ds.utts[: cfg["max_train_utts"]]
        dev_ds.utts = dev_ds.utts[: cfg["max_train_utts"]]
        test_ds.utts = test_ds.utts[: cfg["max_train_utts"]]

    mcfg = APLConfig.from_dict({
        **cfg["model"], **flags,
        "num_classes": vocab.num_classes, "num_embeddings": vocab.num_embeddings, "pad_id": vocab.pad_id,
        "phonetic_dim": phonetic_dim(cfg["phon_dir"]) if flags["use_phonetic"] else 0,
    })
    model = APLModel(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("variant %s | %.1fM parameters | train %d dev %d test %d utts", mcfg.name, n_params / 1e6,
                len(train_ds), len(dev_ds), len(test_ds))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "model_resolved": mcfg.__dict__}, f, sort_keys=False)

    nw = cfg["num_workers"]
    train_loader, train_sampler = make_loader(train_ds, cfg["batch_size"], True, cfg["seed"], nw, vocab.pad_id)
    dev_loader, _ = make_loader(dev_ds, cfg["batch_size"], False, 0, nw, vocab.pad_id)
    test_loader, _ = make_loader(test_ds, cfg["batch_size"], False, 0, nw, vocab.pad_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lr = cfg["lr"]
    loss_best = loss_best_true = float("inf")
    counter, decays, acc_best = 0, 0, float("-inf")
    reload_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
    history = []
    t0 = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        tr_loss = train_epoch(model, train_loader, train_sampler, optimizer, vocab, device, cfg["amp"], epoch, duty)
        dev = run_eval(model, dev_loader, vocab, device, cfg["amp"], beam=0, duty=duty)
        dev_loss, dev_acc = dev["loss"], 1.0 - dev["PER"]

        thr = cfg["plateau_threshold"]
        if dev_loss < loss_best - thr:
            loss_best = loss_best_true = dev_loss
            counter = 0
            reload_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
        elif dev_loss < loss_best + thr:
            counter += 1
            if dev_loss < loss_best and dev_loss < loss_best_true:
                loss_best_true = dev_loss
                reload_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
        else:
            counter = cfg["plateau_patience"]

        improved = dev_acc > acc_best
        if improved:
            acc_best = dev_acc
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, {"epoch": epoch, "dev_PER": dev["PER"]})

        history.append({"epoch": epoch, "lr": lr, "train_loss": tr_loss, "dev_loss": dev_loss,
                        "dev_PER": dev["PER"], "dev_f1": dev["f1"]})
        logger.info("epoch %3d | lr %.2e | train %.3f | dev loss %.3f PER %.2f%% F1 %.2f%% | plateau %d%s | %.1f min",
                    epoch, lr, tr_loss, dev_loss, 100 * dev["PER"], 100 * dev["f1"], counter,
                    " *best*" if improved else "", (time.time() - t0) / 60)

        if counter >= cfg["plateau_patience"]:
            decays += 1
            counter = 0
            loss_best = min(loss_best, loss_best_true)
            model.load_state_dict(reload_state[0])
            optimizer.load_state_dict(reload_state[1])
            lr = cfg["lr"] * cfg["lr_decay"] ** decays
            for g in optimizer.param_groups:
                g["lr"] = lr
            logger.info("restored best-loss weights, lr -> %.2e (decay %d/%d)", lr, decays, cfg["max_lr_decays"])
            if decays >= cfg["max_lr_decays"]:
                break
    save_checkpoint(os.path.join(out_dir, "last.pt"), model, {"epoch": epoch})
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)

    # ---------------------------------------------------------- final decoding with the selected model
    model = load_checkpoint(os.path.join(out_dir, "best.pt")).to(device)
    for split, loader in (("dev", dev_loader), ("test", test_loader)):
        res = run_eval(model, loader, vocab, device, cfg["amp"], beam=cfg["beam"], duty=duty)
        rows = res.pop("_rows")
        with open(os.path.join(out_dir, f"{split}_decoded.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(out_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
        logger.info("[%s, beam %d] %s", split, cfg["beam"], format_table_row(res))
    logger.info("finished in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
