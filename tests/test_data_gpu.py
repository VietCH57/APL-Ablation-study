import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apl.gpu import DutyCycle  # noqa: E402
from apl.manifest import load_split  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "train.csv")), reason="no data")
def test_canonical_and_transcript_taken_as_is():
    import pandas as pd

    df = pd.read_csv(os.path.join(DATA, "train.csv"), index_col=0, keep_default_na=False)
    utts = load_split(os.path.join(DATA, "train.csv"))
    for u, (_, row) in zip(utts, df.iterrows()):
        assert u.canonical == row["Canonical"].split()
        assert u.transcript == row["Transcript"].split()


def test_duty_cycle_sleeps_as_long_as_busy_at_50pct():
    d = DutyCycle(50, torch.device("cpu"))
    d.enabled, d.ratio = True, 1.0  # force on (it is only enabled on CUDA devices)
    d.start()
    time.sleep(0.05)
    t0 = time.time()
    d.step()
    assert 0.04 <= time.time() - t0 <= 0.2


def test_duty_cycle_validation():
    with pytest.raises(ValueError):
        DutyCycle(0, torch.device("cpu"))
    assert not DutyCycle(100, torch.device("cpu")).enabled
