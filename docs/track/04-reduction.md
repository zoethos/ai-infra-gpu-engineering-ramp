---
title: "04 · Reduction"
parent: "C++ ↔ CUDA Track"
nav_order: 4
permalink: /docs/track/04-reduction/
---

# 04 · Reduction
{: .no_toc }

**Stage B** · ⏳ ~3h · prerequisites: [module 03](../03-data-parallel-saxpy/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/04-reduction)

---

Sum N floats. SAXPY had no interaction between elements; reduction is *all*
interaction — every element meets every other at the root of a tree. How each
stack shapes that tree is the module.

## Concept

**CPU:** the serial loop is already near memory-bandwidth-bound, so parallelism
buys little on one socket — but *the compiler can't even vectorize the naive
loop* without `-ffast-math`, because float addition isn't associative and
reassociation changes the result. OpenMP `reduction(+:sum)` gives each core a
private partial sum and combines at the end: lock-free, one atomic-ish merge per
core.

**GPU:** thousands of threads can't all add into one accumulator (atomics on one
address serialize). The canonical shape is hierarchical:
1. each thread grid-strides, accumulating a private register sum;
2. **warp shuffle** (`__shfl_down_sync`) folds 32 lane sums to 1 without touching
   memory — data moves register-to-register between lanes;
3. one partial per warp lands in shared memory; the first warp folds them;
4. one `atomicAdd` per *block* into the global result.

That 4-level funnel (registers → warp → shared → global atomic) is the template
for softmax denominators, norms, loss values — half of ML's non-matmul kernels.

## Confrontation

| Question | C++ | CUDA |
|---|---|---|
| Why naive parallel version is wrong | data race on `sum` | same, ×10 000 threads |
| Idiomatic fix | `omp reduction`, per-core partials | warp shuffle + block funnel |
| Combine cost | ~cores merges | log₂(32) shuffles + 1 atomic/block |
| Float determinism | order fixed serially; OpenMP order varies by thread count | order varies with grid config |
| Library answer | `std::reduce(std::execution::par_unseq, …)` | `cub::DeviceReduce::Sum` |

**Run the CPU example before reading on** — on a 64M-element sum of small
values, the strict serial float sum *plateaus* (the accumulator grows so large
that new addends fall below its precision and are absorbed entirely — watch it
stick at exactly 2²¹), while the OpenMP and pairwise variants land near the
double reference. Parallel summation isn't just faster here; its tree order is
*numerically better* than the "correct" serial order. The GPU's hierarchical
reduction inherits the same accuracy benefit.

**Determinism note for ML:** both stacks give run-to-run-stable results with a
fixed config, but *different* results across configs — this is one reason
`loss.item()` differs across GPU models and why bitwise-reproducible training is
genuinely hard.

## Build & run

```bash
g++ -O3 -std=c++20 -fopenmp -march=native cpp/reduce.cpp -o reduce_cpu && ./reduce_cpu
nvcc -O3 -arch=native cuda/reduce.cu -o reduce_gpu && ./reduce_gpu
```

## Exercises

1. Write the *wrong* GPU version: every thread `atomicAdd`s into one global
   float. Measure the collapse; explain where the serialization happens.
2. In the shuffle kernel, print how many kernel-wide atomics actually execute
   (grid size / block size). Reduce N to 1024 — is the fancy kernel still worth
   it over a single-block version?
3. CPU: try `std::reduce` with `std::execution::par_unseq` (link TBB on gcc).
   Compare with your OpenMP version — same shape, different runtime.
4. Replace your GPU kernel with `cub::DeviceReduce::Sum` and measure. Lesson:
   in production you call CUB/Thrust; you write the raw version to *understand*
   what you're calling.

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 7 §"Warp Shuffle Intrinsics" — this module's kernel, profiled.

---

Next: [05 · Scan & histogram](../05-scan-histogram/)
