---
title: Home
nav_order: 1
permalink: /docs/
---

# AI Infrastructure & GPU Engineering
{: .fs-8 }

A self-paced path from "what is a feature?" to writing CUDA kernels and sharding
training across GPUs — every lesson backed by runnable code you can measure yourself.
{: .fs-5 .fw-300 }

[Start the track](track/){: .btn .btn-primary }
[ML foundations chapter](how-machines-learn/){: .btn }
[Glossary](glossary/){: .btn }

---

## Start here

- **New to machine learning?** Read [How Machines Learn from Data](how-machines-learn/)
  first: features, training vs inference, the learning paradigms, and the modern LLM
  workflow — no code required.
- **Here for the systems side?** Jump into the [C++ ↔ CUDA dual track](track/) and keep
  the [glossary](glossary/) in a tab.

## The path, in four stages

Do the stages in order; inside a stage, do modules in order. Each lesson states its
prerequisite and an honest time estimate — the hours assume you *run the labs*, not just read.

### Stage A — Foundations of parallel execution

*After this stage you can explain why the same loop runs 100× faster on a GPU, and measure it.*

1. [01 · Execution model](track/01-execution-model/) — ~2h
1. [02 · Memory hierarchy](track/02-memory-hierarchy/) — ~3h
1. [03 · Data parallelism: SAXPY](track/03-data-parallel-saxpy/) — ~2h

### Stage B — Parallel algorithms

*After this stage you can build the three algorithmic shapes 90% of kernels reduce to.*

1. [04 · Reduction](track/04-reduction/) — ~3h
1. [05 · Scan & histogram](track/05-scan-histogram/) — ~4h
1. [06 · Matmul & tiling](track/06-matmul-tiling/) — ~4h

### Stage C — Systems craft

*After this stage you can overlap, synchronize and profile like it's a habit.*

1. [07 · Asynchrony & overlap](track/07-async-overlap/) — ~3h
1. [08 · Sync, atomics, memory model](track/08-sync-atomics-memory-model/) — ~3h
1. [09 · Profiling & roofline](track/09-profiling-roofline/) — ~3h

### Stage D — The frontier

*After this stage you can read FlashAttention and CUTLASS as engineering, not magic.*

1. [10 · Advanced GPU](track/10-advanced-gpu/) — ~5h
1. [11 · Multi-device](track/11-multi-device/) — ~4h
1. [12 · Capstone: PyTorch extension](track/12-capstone-pytorch-extension/) — ~6h

## How to study each lesson

1. Read the lesson (concept + the CPU-vs-GPU confrontation table).
2. Build and run the **cpp/** lab, then the **cuda/** lab — source links are on every page.
3. Answer the exercises *in writing* in your own NOTES.md. The questions are chosen so
   the answers differ between the stacks — that delta is the lesson.
4. Only then read the companion book chapters listed at the bottom of the lesson.

## Reference shelf

- [The Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/) — free; the parallelism textbook.
- *AI Systems Performance Engineering* (Fregly, O'Reilly 2025) — spread across the track as per-lesson companion reading.
- *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) — free PDF from the authors; ch. 5 pairs with the ML foundations chapter.
