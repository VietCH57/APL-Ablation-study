import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import apl.engine as engine  # noqa: E402
from apl.dataset import collate  # noqa: E402
from apl.engine import FrameBudget, split_batch, train_step  # noqa: E402
from apl.model import APLConfig, APLModel  # noqa: E402
from apl.vocab import Vocab  # noqa: E402

VOCAB = Vocab.from_file(os.path.join(os.path.dirname(__file__), "..", "configs", "vocab.json"))


def _batch(n=6):
    torch.manual_seed(0)
    items = []
    for i in range(n):
        t = 40 + 7 * i
        items.append({"index": i, "utt_id": f"u{i}", "num_frames": t, "fbank": torch.randn(t, 81),
                      "phonetic": torch.randn(t, 16), "canonical": torch.randint(0, 60, (5 + i,)),
                      "target": torch.randint(0, 60, (4 + i,))})
    return collate(items, VOCAB.pad_id)


def _model():
    torch.manual_seed(1)
    cfg = APLConfig(phonetic_dim=16, rnn_hidden=16, embed_dim=8, num_classes=VOCAB.num_classes,
                    num_embeddings=VOCAB.num_embeddings, pad_id=VOCAB.pad_id)
    return APLModel(cfg)


def test_split_batch_partitions_everything():
    b = _batch()
    chunks = split_batch(b, 3)
    assert sum(len(c["lengths"]) for c in chunks) == 6
    assert torch.equal(torch.cat([c["targets"] for c in chunks]), b["targets"])
    for c in chunks:
        assert c["fbank"].shape[1] == int(c["lengths"].max())
        assert c["canonical"].shape[1] == int(c["canonical_lengths"].max())
        assert int(c["target_lengths"].sum()) == len(c["targets"])


def test_chunked_gradient_equals_full_batch():
    b = _batch()
    grads = []
    for k in (1, 3):
        m = _model().eval()  # no dropout, fixed BatchNorm statistics -> exact comparison
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        loss = train_step(m, b, opt, VOCAB, torch.device("cpu"), "none", FrameBudget(0) if k == 1 else
                          FrameBudget(6 * 82 // 3))
        grads.append((loss, [p.grad.clone() for p in m.parameters() if p.grad is not None]))
    assert abs(grads[0][0] - grads[1][0]) < 1e-3
    for g1, g2 in zip(grads[0][1], grads[1][1]):
        assert torch.allclose(g1, g2, atol=1e-4, rtol=1e-3)


def test_oom_shrinks_budget_and_retries(monkeypatch):
    b = _batch()
    real_forward = engine.forward
    calls = []

    def fake_forward(model, batch, **kw):
        calls.append(len(batch["lengths"]))
        if len(batch["lengths"]) > 2:
            raise torch.cuda.OutOfMemoryError("fake")
        return real_forward(model, batch, **kw)

    monkeypatch.setattr(engine, "forward", fake_forward)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    m = _model().train()
    opt = torch.optim.Adam(m.parameters())
    budget = FrameBudget(0)
    loss = train_step(m, b, opt, VOCAB, torch.device("cpu"), "none", budget)
    assert loss > 0 and budget.value is not None
    assert budget.splits(b) >= 3 and calls[0] == 6
