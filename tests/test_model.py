import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apl.decoding import greedy_decode, prefix_beam_search  # noqa: E402
from apl.features import align_to_frames  # noqa: E402
from apl.model import APLConfig, APLModel, ctc_loss  # noqa: E402


def _batch(phon_dim=64):
    torch.manual_seed(0)
    lens = torch.tensor([90, 61, 30])
    fb, ph = torch.randn(3, 90, 81), torch.randn(3, 90, phon_dim)
    for i, n in enumerate(lens):
        fb[i, n:] = 0
        ph[i, n:] = 0
    can = torch.full((3, 12), 69)
    cl = torch.tensor([12, 9, 4])
    for i, n in enumerate(cl):
        can[i, :n] = torch.randint(0, 39, (int(n),))
    return lens, fb, ph, can, cl


@pytest.mark.parametrize("variant", ["APL", "PL", "AL", "P"])
def test_shapes_and_dims(variant):
    cfg = APLConfig(use_acoustic="A" in variant, use_phonetic="P" in variant, use_linguistic="L" in variant,
                    phonetic_dim=64)
    m = APLModel(cfg)
    lens, fb, ph, can, cl = _batch()
    out = m(lens, fb, ph, can, cl)
    assert out["log_probs"].shape == (3, 45, 69)
    assert out["output_lengths"].tolist() == [45, 31, 15]  # T' = ceil(T / 2)
    q = 768 * (("A" in variant) + ("P" in variant))
    assert m.query_dim == q
    if "L" in variant:
        assert m.linguistic.key.weight.shape == (q, 768)  # H^K = Linear(H^V) -> query size
        assert m.output.in_features == 768 + q            # W [c; h^Q]
    else:
        assert m.output.in_features == q
    assert m.output.bias is not None                      # eq. (7) has a bias b
    loss = ctc_loss(out, torch.randint(0, 68, (int(cl.sum()),)), cl, blank=68)
    loss.backward()


def test_paper_layer_counts():
    m = APLModel(APLConfig(phonetic_dim=64))
    assert len(m.acoustic.cnns) == 2 and len(m.acoustic.rnns) == 4
    assert len(m.phonetic.cnns) == 2 and len(m.phonetic.rnns) == 1


def test_padding_does_not_change_outputs():
    m = APLModel(APLConfig(phonetic_dim=64)).eval()
    lens, fb, ph, can, cl = _batch()
    with torch.no_grad():
        full = m(lens, fb, ph, can, cl)["log_probs"]
        alone = m(lens[2:], fb[2:, :30], ph[2:, :30], can[2:, :4], cl[2:])["log_probs"]
    assert torch.allclose(full[2, :15], alone[0, :15], atol=1e-5)


def test_attention_ignores_padded_canonical():
    m = APLModel(APLConfig(phonetic_dim=64)).eval()
    lens, fb, ph, can, cl = _batch()
    with torch.no_grad():
        a = m(lens, fb, ph, can, cl, return_attention=True)["attention"]
    assert torch.all(a[2, :, 4:] == 0)
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))


def test_align_to_frames():
    import numpy as np
    e = np.arange(10, dtype=np.float32).reshape(5, 2)
    out = align_to_frames(e, 11)
    assert out.shape == (11, 2) and (out[0] == out[1]).all() and (out[-1] == e[-1]).all()
    assert align_to_frames(e, 9).shape == (9, 2)


def test_beam_matches_greedy_on_peaky_input():
    torch.manual_seed(0)
    seq = [3, 3, 68, 5, 68, 68, 7, 7, 3]
    lp = torch.full((len(seq), 69), -20.0)
    for t, c in enumerate(seq):
        lp[t, c] = 0.0
    lp = torch.log_softmax(lp, -1)
    g = greedy_decode(lp[None], torch.tensor([len(seq)]), blank=68)[0]
    b = prefix_beam_search(lp.numpy(), blank=68, beam=10)
    assert g == b == [3, 5, 7, 3]
