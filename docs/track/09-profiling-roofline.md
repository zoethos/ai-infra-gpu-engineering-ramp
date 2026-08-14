---
title: "09 · Profiling & roofline"
parent: "C++ ↔ CUDA Track"
nav_order: 9
permalink: /docs/track/09-profiling-roofline/
---

# 09 · Profiling & roofline
{: .no_toc }

**Stage C** · ⏳ ~3h · prerequisites: [module 08](../08-sync-atomics-memory-model/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/09-profiling-roofline)

---

Optimization without profiling is astrology. This module sets up the tools and
one shared mental model — the **roofline** — that works on both chips.

## The roofline in one paragraph

Every kernel has an **arithmetic intensity** I = FLOPs / bytes moved. The chip
has peak compute P (FLOP/s) and peak bandwidth B (B/s). Attainable performance
is `min(P, I × B)`. Below the ridge point (I = P/B) you're **memory-bound** —
more FLOPs are free, more bytes are not; above it you're **compute-bound**.
SAXPY (I ≈ 0.08 FLOP/B) is deep in memory-bound land on both chips; matmul's
intensity grows with tile size, which is *why* tiling works. H100 ridge point
for FP32 ≈ 20 FLOP/B; for BF16 tensor cores ≈ 300 FLOP/B — meaning tensor cores
are only fed by very-high-reuse kernels: matmul and friends.

## The toolboxes

| Need | CPU | GPU |
|---|---|---|
| Where does time go (system) | `perf record` / `perf top`, VTune | **Nsight Systems** (`nsys profile`) |
| Why is this kernel slow | `perf stat` counters, VTune microarch | **Nsight Compute** (`ncu`) |
| Roofline, automated | VTune / Advisor | `ncu --set roofline` |
| Memory errors | ASan, valgrind | `compute-sanitizer --tool memcheck` |
| Races | TSan | `compute-sanitizer --tool racecheck` |
| In-code timing | `steady_clock` (this repo's `bench.hpp`) | CUDA events (`cuda_check.cuh`) |

**The three numbers to read first in `ncu`:** SM throughput %, DRAM throughput
%, and achieved occupancy. High DRAM + low SM = memory-bound (optimize access);
high SM + low DRAM = compute-bound (optimize math/precision); both low = latency
or launch bound (occupancy, sync, small grid).

## Exercises

1. Run `ncu --set roofline` on your module-03 SAXPY and module-06 tiled matmul.
   Place both on the roofline plot. Verify SAXPY sits on the bandwidth slope and
   matmul climbs toward the ridge.
2. Same story on CPU: `perf stat -e cycles,instructions,cache-misses` for both.
   Compute instructions-per-cycle; explain the difference between the two
   programs' IPC.
3. Take the *naive* GPU matmul and, using only `ncu` output (not prior
   knowledge), write down the evidence trail that leads to "uncoalesced B
   accesses / low reuse" as the diagnosis.
4. `nsys profile` a small PyTorch training step (any toy model). Find: kernel
   launch gaps (CPU-bound?), memcpys you didn't expect, and the NCCL kernels if
   you run DDP. This closes the loop to the main ramp.

## Lab

No new code — this module's lab subjects are the binaries you already built:
profile [`03-data-parallel-saxpy`](../03-data-parallel-saxpy/) and
[`06-matmul-tiling`](../06-matmul-tiling/) per exercises 1–3, and a toy PyTorch
step per exercise 4.

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 6 §"Roofline Model Analysis", Ch. 8 (Nsight Systems/Compute workflow, occupancy tuning — this module's toolchain, chapter-length), Ch. 13 (the same workflow applied to PyTorch, with NVTX and HTA).

---

Next: [10 · Advanced GPU](../10-advanced-gpu/)
