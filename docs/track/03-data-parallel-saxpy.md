---
title: "03 · Data parallelism: SAXPY"
parent: "C++ ↔ CUDA Track"
nav_order: 3
permalink: /docs/track/03-data-parallel-saxpy/
---

# 03 · Data parallelism: SAXPY
{: .no_toc }

**Stage A** · ⏳ ~2h · prerequisites: [module 02](../02-memory-hierarchy/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/03-data-parallel-saxpy)

---

`y[i] = a * x[i] + y[i]` — the "hello world" of throughput computing. Trivially
parallel, zero data reuse, so it's a pure **memory bandwidth** benchmark: the
ceiling is `3 * N * 4 bytes / peak_bandwidth`, on both chips. Neither version
will be compute-bound; proving that to yourself is the point.

## Concept

**CPU parallelism is two-level:** across cores (OpenMP `parallel for` splits the
range) and within a core (SIMD — the compiler auto-vectorizes this loop into
AVX2/AVX-512, 8–16 floats per instruction). Max hardware parallelism ≈ cores ×
SIMD width ≈ a few hundred lanes.

**GPU parallelism is SIMT:** you write *scalar* code for one logical thread and
launch a million of them; the warp is the vector unit (32 lanes) but divergence,
masking, and lane management are the hardware's problem, not yours. That
programming-model difference — scalar code, vector execution — is CUDA's core
ergonomic win over intrinsics.

**The grid-stride loop** (in the .cu file) decouples grid size from problem
size, works for any N, and gives perfectly coalesced access. It's the default
idiom in production kernels (PyTorch's elementwise kernels are grid-stride).

## Confrontation

| Question | C++ | CUDA |
|---|---|---|
| Where's the vectorization? | compiler auto-vec (verify with `-fopt-info-vec`) | implicit in the warp |
| Chunking across workers | OpenMP schedules ranges to cores | grid-stride, warp-interleaved |
| Expected bottleneck | DRAM bandwidth | HBM bandwidth |
| Realistic fraction of peak BW | 80–90 % | 85–95 % |
| What would make it 10× slower | stride/gather access | uncoalesced access |

## Build & run

```bash
g++ -O3 -std=c++20 -fopenmp -march=native cpp/saxpy.cpp -o saxpy_cpu && ./saxpy_cpu
nvcc -O3 -arch=native cuda/saxpy.cu -o saxpy_gpu && ./saxpy_gpu
```

## Exercises

1. Compute achieved GB/s for every variant (the harness prints it). Look up your
   chips' peak bandwidth and fill in the % of peak. This number — not the
   GPU-vs-CPU speedup — is the honest metric.
2. CPU: compare `-O3` with and without `-march=native`; check the compiler's
   vectorization report. GPU: change the block size (32 → 1024); explain why it
   barely matters *for this kernel*.
3. Break coalescing on purpose: make thread `i` process elements `[i*chunk,
   (i+1)*chunk)` (each thread walks its own contiguous slice — the natural CPU
   partitioning!). Measure the damage. This is the single most instructive
   wrong-kernel in CUDA.
4. Add the H2D/D2H transfer time to the GPU measurement. At what N does the GPU
   *including transfers* beat the CPU? (This is why "move the whole pipeline to
   the GPU" beats "offload one op".)

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 6 §"CUDA Programming Refresher" and §"Maintaining High Occupancy".

---

Next: [04 · Reduction](../04-reduction/)
