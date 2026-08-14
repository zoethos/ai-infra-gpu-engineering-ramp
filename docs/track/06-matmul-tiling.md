---
title: "06 · Matmul & tiling"
parent: "C++ ↔ CUDA Track"
nav_order: 6
permalink: /docs/track/06-matmul-tiling/
---

# 06 · Matmul & tiling
{: .no_toc }

**Stage B** · ⏳ ~4h · prerequisites: [module 05](../05-scan-histogram/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/06-matmul-tiling)

---

C = A·B, the first op in this track with **data reuse** (each element is read N
times) — which means the first op where the memory hierarchy, not bandwidth,
decides everything, and the first where you can approach *compute* peak. It is
also, of course, ~95 % of a transformer's FLOPs.

## Concept

Arithmetic intensity of matmul grows with tile size: a T×T tile does O(T³) math
on O(T²) data. Both stacks exploit this identically in principle — **keep a tile
in the fast tier, do all its math, move on** — with different fast tiers:

**CPU:** cache blocking. Loop order (i,k,j) or (i,j,k)-blocked so that a block
of A stays in L1/L2 and a strip of B streams through. Add OpenMP across the
outer loop, and the compiler vectorizes the inner one. Getting past ~20 % of
peak by hand is real work — which is why BLAS (OpenBLAS/MKL) exists.

**GPU:** shared-memory tiling. Each block cooperatively loads a tile of A and a
tile of B into shared memory (coalesced!), syncs, computes the partial products
from registers, syncs, slides to the next K-tile. This classic kernel reaches
maybe 30–50 % of FP32 peak. The rest of the distance to peak is **tensor
cores** — dedicated matrix units consuming 16×16-ish fragments (WMMA API, or
WGMMA + TMA on Hopper) — and that's cuBLAS/CUTLASS territory, covered in
module 10.

## Confrontation

| Question | C++ | CUDA |
|---|---|---|
| Fast tier for the tile | L1/L2 (implicit, pray) | shared memory (explicit) |
| Tile residency guaranteed? | no — eviction is hardware's call | yes — it's your scratchpad |
| Inner-loop engine | AVX-512 FMA (16 fl/instr) | FFMA, or tensor core fragment op |
| Hand-written vs library gap | ~5–10× vs MKL | ~2–4× vs cuBLAS (FP32), ~10×+ vs TC |
| Library | OpenBLAS / MKL / BLIS | cuBLAS / CUTLASS |

## Build & run

```bash
g++ -O3 -std=c++20 -fopenmp -march=native cpp/matmul.cpp -o matmul_cpu && ./matmul_cpu
nvcc -O3 -arch=native cuda/matmul.cu -o matmul_gpu && ./matmul_gpu
```

## Exercises

1. Both files implement naive and tiled variants. Measure GFLOP/s for all four;
   compute % of your chips' FP32 peak (cores × freq × SIMD × 2, and SMs × freq ×
   cores/SM × 2).
2. Sweep the GPU tile size (16/32) and block shape; find your GPU's sweet spot.
   Then sweep CPU block size against your L1/L2 sizes (`getconf -a | grep CACHE`).
3. Link cuBLAS (`nvcc -lcublas`, call `cublasSgemm`) and MKL/OpenBLAS
   (`-lopenblas`, call `cblas_sgemm`). Fill the final row of your table. Sit
   with the humility.
4. In the GPU kernel, remove one `__syncthreads()` and run under
   `compute-sanitizer --tool racecheck`. This is module 08's cliffhanger.

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 9 (multi-level micro-tiling, kernel fusion, CUTLASS — the chapters between this module's kernel and cuBLAS).
- Ultra-Scale Playbook: the arithmetic-intensity / kernel sections — same tiling story told from the LLM-training side.

---

Next: [07 · Asynchrony & overlap](../07-async-overlap/)
