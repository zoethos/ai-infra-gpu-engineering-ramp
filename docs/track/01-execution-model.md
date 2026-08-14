---
title: "01 · Execution model"
parent: "C++ ↔ CUDA Track"
nav_order: 1
permalink: /docs/track/01-execution-model/
---

# 01 · Execution model
{: .no_toc }

**Stage A** · ⏳ ~2h · prerequisites: none — start here · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/01-execution-model)

---

The single most important delta between the stacks. Everything later is a
consequence of this module.

## Concept

**CPU:** a handful of fat cores (8–128), each running one heavyweight thread
(plus an SMT sibling). Threads are expensive (~µs to create, kernel-scheduled),
so you create few and keep them busy. Latency is hidden by *caches and
out-of-order execution* inside each core.

**GPU:** ~100–150 Streaming Multiprocessors (SMs), each juggling thousands of
featherweight threads grouped in **warps of 32** that execute in lockstep
(SIMT). Threads are nearly free; you launch millions. Latency is hidden by
*oversubscription* — when a warp stalls on memory, the SM switches to another
resident warp in one cycle. There is no OS scheduler in the loop.

The launch hierarchy: a **grid** of **blocks**, each block a group of up to 1024
threads that lands on exactly one SM and can cooperate through shared memory and
`__syncthreads()`. Blocks are independent by design — that independence is what
lets the same binary scale from 20 SMs to 150.

## Confrontation

| Question | C++ answer | CUDA answer |
|---|---|---|
| Cost of one thread? | ~µs create, ~8 MB stack, kernel object | ~nothing; register slice on an SM |
| Right number of threads? | ≈ core count | ≥ 10× the number that "fits" (oversubscribe) |
| Who schedules? | OS kernel, preemptive, ~ms slices | SM warp scheduler, per-cycle |
| What hides a cache/memory miss? | OoO window, prefetcher | another resident warp |
| `printf` from 10 000 threads? | interleaved lines, mutex needed for sanity | works (buffered), order undefined |

## Build & run

```bash
g++ -O3 -std=c++20 -pthread cpp/hello_threads.cpp -o hello_cpu && ./hello_cpu
nvcc -O3 -arch=native cuda/hello_kernel.cu -o hello_gpu && ./hello_gpu
```

## Exercises

1. In the CPU version, raise the thread count to 10 000. Watch creation time
   explode. Find the count where wall time is minimized (spoiler: ≈ your core
   count). In the CUDA version, raise the grid to 10 M threads. Nothing bad
   happens. Explain both outcomes in two sentences each.
2. Print `blockIdx`/`threadIdx` for a 4-block launch several times. The *block*
   execution order changes between runs; the order *within a warp* doesn't. Why?
3. Query your GPU with `cudaGetDeviceProperties` (the CUDA example does):
   compute capability, SM count, max resident threads/SM. Compute the maximum
   number of threads in flight and compare with your CPU's.

## References

- CUDA C++ Programming Guide §2 "Programming Model".
- Hennessy & Patterson, "Computer Architecture", GPU chapter — the SIMT section.

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 2 (SMs, threads, warps — the hardware this module runs on), Ch. 6 §"Threads, Warps, Blocks, and Grids".

---

Next: [02 · Memory hierarchy](../02-memory-hierarchy/)
