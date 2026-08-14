---
title: "10 · Advanced GPU"
parent: "C++ ↔ CUDA Track"
nav_order: 10
permalink: /docs/track/10-advanced-gpu/
---

# 10 · Advanced GPU
{: .no_toc }

**Stage D** · ⏳ ~5h · prerequisites: [module 09](../09-profiling-roofline/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/10-advanced-gpu)

---

Where the GPU track pulls away from the CPU track — the hardware here has no
real CPU counterpart (closest: AVX-512/AMX, noted where honest).

## 10a — Warp-level programming

The warp is a 32-lane vector unit you can program *across*: `__shfl_*_sync`
(lane data exchange — module 04 used it), `__ballot_sync` / `__any_sync`
(lane votes), `__match_any_sync` (group lanes by value). Post-Volta rule:
lockstep is not guaranteed; the `_sync` mask parameter is mandatory hygiene.
CPU cousin: AVX-512 shuffles/permutes + mask registers — same idea, but the
compiler rarely finds it for you; on GPU you write it directly.

## 10b — Tensor cores

Matrix-multiply units consuming fragments (m16n8k16-ish) in one instruction,
at 8–16× the FP32 FMA rate, in FP16/BF16/TF32/FP8/**FP4** (Blackwell).
APIs, in ascending order of both power and pain:
1. `wmma::` (portable, fragment-level) — write one 16×16×16 GEMM with it;
2. `mma` PTX / CUTLASS templates — production;
3. **Hopper WGMMA** (warpgroup = 4 warps cooperating, async matmul from shared
   memory) + **TMA** (Tensor Memory Accelerator: hardware bulk async copies
   global→shared, freeing all threads to compute);
4. Blackwell: 5th-gen tensor cores, FP4/FP6, TMEM, and `tcgen05` instructions.
CPU cousin: Intel AMX tiles — same concept (matrix registers + tile ops),
~an order of magnitude behind in throughput.

## 10c — The modern kernel shape (Hopper+)

Production kernels (FlashAttention-3, CUTLASS 3.x) are **asynchronous
pipelines** inside one kernel: TMA prefetches tile k+1 while WGMMA chews tile
k, coordinated by `cuda::pipeline`/mbarriers, often with **thread block
clusters** (blocks sharing "distributed shared memory" across SMs). The mental
model shifts from "threads compute" to "the kernel is a little dataflow
factory" — specialized producer/consumer warps. This is also exactly the
programming model **Triton** generates for you; after this module, read a
Triton matmul and map its pragmas to what you wrote by hand.

## Exercises

1. Rewrite module-04's reduction using only warp intrinsics for the block-level
   combine (`__reduce_add_sync` on CC 8.0+ for ints — compare).
2. WMMA GEMM: 16×16×16 fragments, FP16 in / FP32 accumulate. Verify against
   cuBLAS; measure vs your module-06 tiled FP32 kernel. Expect an uncomfortable
   ratio.
3. On Hopper+ (rent one): take CUTLASS's `hopper_gemm` example, instrument, and
   identify TMA vs WGMMA phases in Nsight Compute. Write half a page mapping
   producer/consumer warps to the module-07 pipeline you built with streams —
   same pattern, one level down.
4. Read the FlashAttention-3 paper's kernel section with all of the above in
   hand. It should now parse as engineering, not magic.

## References

- CUDA C++ Programming Guide: warp functions; WMMA; clusters; TMA.
- CUTLASS docs + `examples/`; the CUTLASS 3.x "CuTe" layout tutorial.
- FlashAttention-3 paper (Shah et al., 2024).

## Lab

- [`cuda/wmma_gemm.cu`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/10-advanced-gpu/cuda/wmma_gemm.cu) — exercise 2 (tensor-core GEMM via `wmma::`, FP16→FP32; one warp per 16×16 tile). Build: `nvcc -O3 -arch=native cuda/wmma_gemm.cu -o wmma_gpu` (CC ≥ 7.0). Deliberately no CPU twin — the honest CPU counterpart is AMX, see §10b.

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 9 (mixed precision, tensor cores, CUTLASS, PTX/SASS), Ch. 10 (intra-kernel pipelining, warp-specialized producer/consumer, thread block clusters — §10c of this lesson, book-length), Ch. 14 (Triton — the compiler that writes this module's kernels for you).
- Ultra-Scale Playbook: fused kernels / Flash Attention sections.

---

Next: [11 · Multi-device](../11-multi-device/)
