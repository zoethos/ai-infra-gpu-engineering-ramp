---
title: "05 · Scan & histogram"
parent: "C++ ↔ CUDA Track"
nav_order: 5
permalink: /docs/track/05-scan-histogram/
---

# 05 · Scan & histogram
{: .no_toc }

**Stage B** · ⏳ ~4h · prerequisites: [module 04](../04-reduction/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/05-scan-histogram)

---

Two ops that stress what reduction only hinted at. **Prefix sum (scan)** looks
inherently serial (`out[i] = out[i-1] + x[i]`) yet parallelizes to O(log n)
depth — it's the workhorse behind stream compaction, sorting, tokenizer offset
computation, and MoE token routing. **Histogram** is the contention benchmark:
many workers incrementing few counters.

## Concept

**Scan, CPU:** three-pass blocked approach — each core scans its chunk, a serial
scan over the per-chunk totals computes offsets, each core adds its offset.
Depth ~2 passes over data; simple and near-bandwidth-bound.

**Scan, GPU:** same three-level idea but inside the chip: warp-level scan with
`__shfl_up_sync`, block-level via shared memory, grid-level via a block-offset
pass. Modern implementations (CUB) use **decoupled look-back** to make it
single-pass — worth reading even if you don't implement it.

**Histogram, both:** naive = atomics on shared counters. CPU fix: per-thread
private histograms, merge at the end (memory permitting). GPU fix: per-*block*
histogram in shared memory (fast atomics), one global merge per block. Same
idea — *privatize, then combine* — different granularity. When the histogram is
too big to privatize, you're in scatter territory and both chips hurt.

## Confrontation

| Question | C++ | CUDA |
|---|---|---|
| Scan strategy | chunk + offsets (3 logical passes) | warp/block/grid hierarchy, decoupled look-back |
| Histogram contention fix | per-thread copies | per-block shared-memory copies |
| Atomic cost when uncontended | ~20 cycles | ~similar, but ×32 lanes can collide at once |
| Worst case | false sharing on counter lines | all 32 lanes hit one bin (full serialization) |
| Library answer | `std::inclusive_scan(par_unseq)` | `cub::DeviceScan`, `cub::DeviceHistogram` |

## Exercises

1. Implement the 3-pass blocked scan with OpenMP; verify against
   `std::inclusive_scan`. Then implement a single-block GPU scan (shared memory,
   any classic scheme); verify against CPU.
2. Histogram 1 B bytes into 256 bins, data (a) uniform random, (b) all the same
   value. Measure both stacks on both distributions. The (b) case is the lesson:
   contention, not bandwidth, is the ceiling.
3. Read CUB's decoupled look-back description and write a half-page summary of
   why it beats the 3-pass scheme (hint: one trip over global memory, not two).
4. Connect to ML: find where scan appears in MoE token dispatch (capacity
   offsets) — e.g. in Megatron or vLLM source — and note the kernel used.

## Lab

- [`cpp/scan_histogram.cpp`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/05-scan-histogram/cpp/scan_histogram.cpp) — exercises 1–2 (3-pass blocked scan; histogram, shared-atomics vs privatized). Build: `g++ -O3 -std=c++20 -fopenmp -march=native cpp/scan_histogram.cpp -o scanhist_cpu`
- [`cuda/scan_histogram.cu`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/05-scan-histogram/cuda/scan_histogram.cu) — exercises 1–2 (single-block shared-memory scan; histogram, global vs block-privatized atomics). Build: `nvcc -O3 -arch=native cuda/scan_histogram.cu -o scanhist_gpu`

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 7 §"Tiling and Data Reuse Using Shared Memory"; Ch. 12 §"Dynamic Scheduling with Atomic Work Queues" (where histogram-style atomics graduate to).

---

Next: [06 · Matmul & tiling](../06-matmul-tiling/)
