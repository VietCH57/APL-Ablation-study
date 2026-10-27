"""Sharing the lab GPU: cap this process' VRAM and its share of GPU time.

Same mechanism as XLSR-300M-Phoneme-Recognition (src/phonrec/utils/misc.py):
  * gpu_memory_fraction: torch.cuda.set_per_process_memory_fraction; allocating beyond the cap
    raises an out-of-memory error instead of taking memory from other users (0 = no cap).
  * gpu_duty_pct: after each busy period (one batch) the process sleeps so that the GPU is busy
    at most `pct` % of wall time (100 = no throttling). E.g. 50 -> sleep as long as the batch took.
"""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger("apl")


def limit_gpu_memory(fraction: float, device: torch.device) -> None:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("gpu_memory_fraction must be in [0, 1]")
    if fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        logger.info("GPU memory capped at %.0f%% (%.1f of %.1f GiB)", 100 * fraction, fraction * total, total)


class DutyCycle:
    """Throttle GPU usage to `pct` % of wall time by sleeping after each busy period."""

    def __init__(self, pct: int, device: torch.device):
        if not 1 <= pct <= 100:
            raise ValueError("gpu_duty_pct must be in [1, 100]")
        self.enabled = pct < 100 and device.type == "cuda"
        self.ratio = (100.0 / pct - 1.0) if self.enabled else 0.0
        self.device = device
        self.busy = 0.0
        self._t0 = None
        if self.enabled:
            logger.info("GPU duty cycle limited to %d%%", pct)

    def start(self) -> None:
        if self.enabled:
            self._t0 = time.time()

    def stop(self) -> None:
        if self.enabled and self._t0 is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)  # count the queued GPU work as busy time
            self.busy += time.time() - self._t0
            self._t0 = None

    def throttle(self) -> None:
        if self.enabled and self.busy > 0:
            time.sleep(self.busy * self.ratio)
            self.busy = 0.0

    def step(self) -> None:
        """stop() + throttle(): call at the end of every batch."""
        self.stop()
        self.throttle()


class NoDuty(DutyCycle):
    def __init__(self):
        self.enabled, self.ratio, self.busy, self._t0, self.device = False, 0.0, 0.0, None, None
