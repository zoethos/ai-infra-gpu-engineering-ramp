---
title: "02 · Memory hierarchy"
parent: "C++ ↔ CUDA Track"
nav_order: 2
permalink: /docs/track/02-memory-hierarchy/
---

# 02 · Memory hierarchy
{: .no_toc }

**Stage A** · ⏳ ~3h · prerequisites: [module 01](../01-execution-model/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/02-memory-hierarchy)

---

Performance on both chips is dominated by memory, not arithmetic. The stacks
solve the same problem — DRAM is slow — with opposite philosophies.

## Concept

**CPU:** a deep *implicit* hierarchy. L1 (~32–48 KB, ~4 cycles) → L2 (~1–2 MB)
→ L3 (~tens of MB, shared) → DRAM (~100 GB/s per socket, ~80 ns). Hardware
decides what to cache, in 64-byte lines, with prefetchers guessing your access
pattern. You optimize by *arranging access patterns the hardware likes*
(sequential, blocked) — you never manage the cache directly.

**GPU:** a shallow, mostly *explicit* hierarchy. Registers (255/thread) →
**shared memory** (up to 228 KB/SM on Hopper — a user-managed scratchpad) → L2
(tens of MB) → **HBM** (3.3 TB/s on H100, 8 TB/s on B200). The cardinal rule is
**coalescing**: the 32 threads of a warp should touch consecutive addresses so
the hardware merges them into few wide transactions. Stride-1 per *warp* is the
GPU's "sequential access".

The inversion to internalize: on CPU, *one thread* should walk memory
sequentially. On GPU, *adjacent threads* should touch adjacent addresses — which
means a single thread walks memory with a stride of `blockDim * gridDim`
(exactly what the grid-stride loop in module 03 does).

## Confrontation

| Question | C++ | CUDA |
|---|---|---|
| Unit of transfer | 64 B cache line | 32 B sectors / 128 B segment per warp |
| Who manages the fast tier? | hardware (LRU-ish) | you (`__shared__`) |
| Good pattern | thread walks stride-1 | warp walks stride-1 (thread stride = warp width) |
| Bad pattern cost | ~2–5× (miss + no prefetch) | ~10–30× (uncoalesced ⇒ 32 transactions) |
| Peak bandwidth | ~0.1–0.5 TB/s (server socket) | 3–8 TB/s (HBM3/3e) |
| What false sharing looks like | line ping-pong between cores | shared-memory bank conflicts |

## Experiments (both stacks)

1. **Stride sweep.** Sum an array with stride 1, 2, 4 … 4096. CPU: bandwidth
   collapses when stride exceeds a cache line and prefetch gives up. GPU: write
   the kernel so *thread i reads `a[i * stride]`* — coalescing dies at stride 2
   already. Plot GB/s vs stride for both. The two curves are this module's exam.
2. **AoS vs SoA.** A struct of 4 floats; sum one field over 100 M elements. On
   GPU the SoA layout is ~4× faster (coalescing); on CPU the gap is smaller but
   real (line utilization). This is why ML frameworks are SoA-shaped.
3. **Pageable vs pinned host memory.** Time `cudaMemcpy` H2D at 1 GB from
   `malloc` vs `cudaMallocHost`. PCIe 5 tops near 60 GB/s only when pinned —
   this single flag is a classic dataloader bottleneck (`pin_memory=True` in
   PyTorch is literally this).

## References

- CUDA C++ Best Practices Guide — "Memory Optimizations".
- Ulrich Drepper, "What Every Programmer Should Know About Memory" (CPU side).

## Lab

- [`cpp/stride_sweep.cpp`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/02-memory-hierarchy/cpp/stride_sweep.cpp) — experiments 1–2 (stride sweep, AoS vs SoA). Build: `g++ -O3 -std=c++20 -march=native cpp/stride_sweep.cpp -o stride_cpu`
- [`cuda/stride_sweep.cu`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/02-memory-hierarchy/cuda/stride_sweep.cu) — experiments 1–3 (coalescing sweep, AoS vs SoA, pinned vs pageable). Build: `nvcc -O3 -arch=native cuda/stride_sweep.cu -o stride_gpu`

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 6 §"GPU Memory Hierarchy", Ch. 7 (coalesced vs uncoalesced access, vectorized access — runs this module's experiment with Nsight evidence).

---

Next: [03 · Data parallelism: SAXPY](../03-data-parallel-saxpy/)
