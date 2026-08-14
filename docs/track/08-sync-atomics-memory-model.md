---
title: "08 · Sync, atomics, memory model"
parent: "C++ ↔ CUDA Track"
nav_order: 8
permalink: /docs/track/08-sync-atomics-memory-model/
---

# 08 · Sync, atomics, memory model
{: .no_toc }

**Stage C** · ⏳ ~3h · prerequisites: [module 07](../07-async-overlap/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/08-sync-atomics-memory-model)

---

The module about *correctness under parallelism* — the part you can't benchmark
your way out of.

## Concept

**CPU:** the C++11 memory model. `std::atomic` with orderings from `relaxed` to
`seq_cst`; happens-before edges via acquire/release; mutexes and condition
variables built on top. Races are UB — not "sometimes wrong", *undefined*.

**GPU:** historically ad hoc, now formally aligned: CUDA adopted the C++ memory
model with **scopes** — `cuda::atomic<T, cuda::thread_scope_block>` (visible
within the block) vs `thread_scope_device` vs `thread_scope_system` (CPU+GPU
coherent). Narrower scope = cheaper. Alongside: `__syncthreads()` (block
barrier — *all or none*: calling it inside a divergent branch is deadlock/UB),
warp-level `__syncwarp()`, and **cooperative groups** for tiles and (with
cooperative launch) grid-wide sync.

The deep difference isn't the primitives — it's the *shape of trouble*:
- CPU trouble: lock contention, false sharing, ABA, priority inversion.
- GPU trouble: barrier divergence, forgetting `__syncthreads()` between shared
  memory reuse (module 06 exercise 4!), assuming warp lockstep that Volta+'s
  independent thread scheduling no longer guarantees (hence `_sync` suffixes on
  all warp intrinsics), block-level deadlock from grid-wide spinlocks (blocks
  aren't guaranteed co-resident — never spin on another block without
  cooperative launch).

## Confrontation

| Question | C++ | CUDA |
|---|---|---|
| Formal model | C++11/20 memory model | same model + thread scopes (`cuda::atomic`) |
| Cheapest correct flag | `atomic<bool>` acq/rel | block-scope atomic in shared memory |
| Barrier | `std::barrier` | `__syncthreads()` / `cg::sync()` |
| Race detector | TSan (`-fsanitize=thread`) | `compute-sanitizer --tool racecheck` |
| Classic footgun | data race = UB | `__syncthreads()` in divergent branch |
| Global sync | trivial (join) | needs cooperative launch, or end the kernel |

**Rule that transfers:** ending a kernel and launching the next *is* the GPU's
grid-wide barrier, and it's usually the right one. Kernel boundaries are cheap;
in-kernel global sync is a specialty tool.

## Exercises

1. Write the racy counter on both stacks (plain `int`, many increments). Then
   fix with the *narrowest* correct primitive on each. Run TSan and
   compute-sanitizer on the racy versions; save the reports.
2. Producer/consumer flag: CPU with `relaxed` (broken-ish) vs `acquire/release`
   (correct). GPU: same experiment with `cuda::atomic_ref` scopes — measure
   block-scope vs device-scope atomic throughput.
3. Reproduce the module-06 missing-`__syncthreads()` bug deliberately; explain
   in NOTES.md exactly which threads read stale shared memory and why the
   wrong answers are nondeterministic.
4. Reading: "GPU Concurrency: Weak Behaviours and Programming Assumptions"
   (Alglave et al.) — skim; then the libcu++ `cuda::atomic` docs.

## Lab

- [`cpp/race_and_fix.cpp`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/08-sync-atomics-memory-model/cpp/race_and_fix.cpp) — exercises 1–2 (racy counter → seq_cst → relaxed → sharded+padded). Build: `g++ -O3 -std=c++20 -pthread cpp/race_and_fix.cpp -o race_cpu` (rebuild with `-fsanitize=thread -O1 -g` for the TSan run)
- [`cuda/race_and_fix.cu`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/08-sync-atomics-memory-model/cuda/race_and_fix.cu) — exercises 1–2 (global atomicAdd → warp-aggregated → block-privatized → `cuda::atomic_ref`). Build: `nvcc -O3 -arch=native cuda/race_and_fix.cu -o race_gpu`

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 10 §"Cooperative Groups", Ch. 12 §"Dynamic Scheduling with Atomic Work Queues" (atomics as a scheduling primitive — where this module's counters are headed).

---

Next: [09 · Profiling & roofline](../09-profiling-roofline/)
