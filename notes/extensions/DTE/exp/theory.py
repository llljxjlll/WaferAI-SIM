"""Closed-form theory for the channel-count sweep, independent of the C++ driver.

Derived directly from DTEUnit's scheduler (llm/src/dte/dte_unit.cpp):
  - admitPending(): the pending FIFO fills any free channel slot as soon as
    one exists (unbounded pending queue in legacy/non-fine-grained mode, see
    descriptorCapacity()).
  - Every admitted transfer pays a fixed launch latency L = gamma_cycles +
    tau_launch_cycles before it can compete for the shared LEGACY_BUS.
  - chooseReadySlot() round-robins over ready (BUS_WAIT) channels; when
    several become ready simultaneously it serves them in increasing slot
    index, i.e. original arrival order (verified against the "rr" case in
    llm/src/dte/v0_selftest.cpp).
  - A channel slot is held for the transfer's *entire* lifetime (launch +
    bus_wait + transmit) and only frees when the transfer completes
    (finishPortServices() decrements active_count_ on COMPLETED).

With N transfers issued back-to-back at t=0 against a pool of C channels and
uniform per-transfer launch latency L and bus service time T, this yields a
simple recursion:

  admission[k] = 0                        if k <  C   (first C admitted immediately)
               = completion[k - C]        if k >= C   (admitted when its slot frees)
  ready[k]     = admission[k] + L         (when it starts competing for the bus)
  bus_finish[k]= max(ready[k], bus_finish[k-1]) + T
  completion[k]= bus_finish[k]

completion[k] is also the cycle at which slot (k mod C) frees.
"""
from __future__ import annotations


def completion_cycles(n: int, channels: int, launch: int, transmit: int) -> list[int]:
    completion: list[int] = [0] * n
    bus_finish_prev = 0
    for k in range(n):
        admission = 0 if k < channels else completion[k - channels]
        ready = admission + launch
        bus_finish = max(ready, bus_finish_prev) + transmit
        completion[k] = bus_finish
        bus_finish_prev = bus_finish
    return completion


def saturation_channels(launch: int, transmit: int) -> int:
    """Smallest channel_count beyond which more channels give no further
    speedup for this workload: enough pipeline depth that a freed slot's
    replacement finishes launching before the bus would otherwise idle."""
    import math
    return math.ceil(launch / transmit) + 1
