---
title: "07 · Asynchrony & overlap"
parent: "C++ ↔ CUDA Track"
nav_order: 7
permalink: /docs/track/07-async-overlap/
---

# 07 · Asynchrony & overlap
{: .no_toc }

**Stage C** · ⏳ ~3h · prerequisites: [module 06](../06-matmul-tiling/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/07-async-overlap)

---

Everything so far ran one thing at a time. Real systems overlap: compute with
data movement, compute with compute, CPU work with GPU work. This module is the
direct ancestor of "overlap communication with computation" — the phrase you'll
meet again in DDP/FSDP.

## Concept

**CPU:** `std::async`/futures for one-shot tasks, thread pools + queues for
throughput, C++20 coroutines for structuring it. Overlap is the *default* (the
OS runs anything runnable); your job is dependency management.

**GPU:** the inversion — *everything is already async* with respect to the host.
A kernel launch queues work and returns immediately; `cudaMemcpy` is the odd one
out (it blocks). The ordering tool is the **stream**: a FIFO queue of GPU work.
Same stream ⇒ ordered; different streams ⇒ may overlap (copy engines run
concurrently with SMs). **Events** are the cross-stream sync primitive
(`record` in one stream, `wait` in another). **CUDA Graphs** capture a whole
DAG of launches and replay it with near-zero CPU overhead — the fix for
launch-bound workloads (small kernels, big models — this is why
`torch.cuda.CUDAGraph` and vLLM's graph capture exist).

The canonical exercise — **chunked pipelined transfer**: split a big array into
chunks; stream k does H2D(k), kernel(k), D2H(k). With pinned memory and 3+
streams, transfer hides behind compute almost entirely.

## Confrontation

| Question | C++ | CUDA |
|---|---|---|
| Unit of scheduled work | task/callable | kernel launch / memcpy |
| Ordering primitive | future, latch, queue | stream (FIFO), event |
| What's async by default | nothing (you opt in) | everything device-side (host opts *out* via sync) |
| Overlap copy & compute | DMA is implicit | needs pinned memory + separate streams |
| Amortize dispatch overhead | batch tasks, avoid tiny tasks | CUDA Graphs |
| Deadlock flavor | lock cycles | stream waiting on itself via event misuse |

## Exercises

1. Pipelined transfer: process 1 GB in 1/2/4/8 chunks with that many streams.
   Plot wall time vs chunk count; find where overlap saturates. Repeat with
   pageable (non-pinned) memory and observe overlap silently vanish (the copy
   engine can't DMA pageable memory).
2. Nsight Systems (`nsys profile ./a.out`) on exercise 1 — the timeline showing
   copy/compute overlap is the single most useful picture in GPU engineering.
   Screenshot it into your NOTES.md.
3. CPU mirror: overlap file reads with processing using a 2-thread
   producer/consumer queue; then with `std::async`. Note which was easier to
   get right and why.
4. Launch a 1 µs kernel 10 000 times in a loop; measure. Capture the same loop
   in a CUDA Graph and replay. The ratio you see is why graphs exist.

## Lab

- [`cpp/overlap_pipeline.cpp`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/07-async-overlap/cpp/overlap_pipeline.cpp) — exercise 3 (serial vs double-buffered fetch/compute pipeline). Build: `g++ -O3 -std=c++20 -pthread cpp/overlap_pipeline.cpp -o overlap_cpu`
- [`cuda/streams_pipeline.cu`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/07-async-overlap/cuda/streams_pipeline.cu) — exercises 1 & 4 (1/2/4/8-stream pipelined transfer; 10k-launch loop vs CUDA Graph). Build: `nvcc -O3 -arch=native cuda/streams_pipeline.cu -o streams_gpu`

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 11 (streams, stream-ordered allocator, events, CUDA Graphs — this module, production-grade), Ch. 12 §"Batch Repeated Kernel Launches with CUDA Graphs".

---

Next: [08 · Sync, atomics, memory model](../08-sync-atomics-memory-model/)
