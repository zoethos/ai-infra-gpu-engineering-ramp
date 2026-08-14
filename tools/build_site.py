#!/usr/bin/env python3
"""Generate the Jekyll (Just the Docs) sources for the GitHub Pages site.

Reads:  cpp-cuda-track/README.md and cpp-cuda-track/*/README.md
Writes: docs/index.md, docs/track/*.md, docs/glossary.md
The site is built by GitHub Pages natively (remote_theme in /_config.yml);
this script only refreshes the markdown sources from the curriculum.
Run:    python3 tools/build_site.py

The ML chapter (docs/how-machines-learn/index.html) is intentionally NOT
generated: it is a hand-designed static page that Jekyll serves verbatim.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TRACK = ROOT / "cpp-cuda-track"
REPO_BLOB = "https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main"

# num, slug, short title, stage, est hours
MODULES = [
    ("01", "01-execution-model",           "Execution model",             "A", 2),
    ("02", "02-memory-hierarchy",          "Memory hierarchy",            "A", 3),
    ("03", "03-data-parallel-saxpy",       "Data parallelism: SAXPY",     "A", 2),
    ("04", "04-reduction",                 "Reduction",                   "B", 3),
    ("05", "05-scan-histogram",            "Scan & histogram",            "B", 4),
    ("06", "06-matmul-tiling",             "Matmul & tiling",             "B", 4),
    ("07", "07-async-overlap",             "Asynchrony & overlap",        "C", 3),
    ("08", "08-sync-atomics-memory-model", "Sync, atomics, memory model", "C", 3),
    ("09", "09-profiling-roofline",        "Profiling & roofline",        "C", 3),
    ("10", "10-advanced-gpu",              "Advanced GPU",                "D", 5),
    ("11", "11-multi-device",              "Multi-device",                "D", 4),
    ("12", "12-capstone-pytorch-extension","Capstone: PyTorch extension", "D", 6),
]
STAGES = {
    "A": ("Stage A — Foundations of parallel execution",
          "explain why the same loop runs 100× faster on a GPU, and measure it"),
    "B": ("Stage B — Parallel algorithms",
          "build the three algorithmic shapes 90% of kernels reduce to"),
    "C": ("Stage C — Systems craft",
          "overlap, synchronize and profile like it's a habit"),
    "D": ("Stage D — The frontier",
          "read FlashAttention and CUTLASS as engineering, not magic"),
}

GLOSSARY = [
    ("tensor", "An N-dimensional array of numbers, the universal data container of deep learning. A PyTorch tensor also remembers device (CPU/GPU), dtype, and how it was computed (for gradients)."),
    ("feature", "One usable input signal for a model — usually one dataset column (numeric, categorical, or engineered from raw facts)."),
    ("target / label", "The answer a supervised model learns to predict; paired with features it forms a training example."),
    ("weights / parameters", "The adjustable numbers inside a model. Training = finding good values for them."),
    ("forward pass", "Running input through the model with current weights to get a prediction. In code: `model(x)`."),
    ("loss", "A single number measuring how wrong the model's predictions are on a batch. The whole network is judged by it."),
    ("gradient", "For every weight, the answer to \"if I nudge this weight up, does the loss go up or down, and how steeply?\" Stored per-parameter in `.grad`, same shape as the weight. Computed by `loss.backward()` via the chain rule."),
    ("optimizer", "The rule that moves weights using gradients. SGD: `W ← W − η·grad`. Adam/AdamW adds momentum (average of recent gradients) and a per-weight adaptive step (divide by √v of squared-gradient average) — at the cost of two extra floats per parameter."),
    ("learning rate (η)", "The step size of each weight update. Too big: training explodes. Too small: it crawls."),
    ("epoch / batch", "A batch is the group of examples processed in one forward+backward step; an epoch is one full pass over the dataset."),
    ("logits", "The raw scores a network outputs before softmax (from \"logistic unit\", 1944 — historically the log of the odds). Free-range numbers on the whole real line, waiting to become probabilities."),
    ("softmax", "Turns a list of scores into positive numbers summing to 1: exponentiate each, divide by the sum. A 'soft' max: the winner gets most of the mass, not all. Used at classifier outputs, LLM next-token distributions, and inside attention."),
    ("ReLU", "Rectified Linear Unit: `max(0, x)` — pass positives, zero out negatives. The 'bend' that makes deep networks non-linear; without it, stacked linear layers collapse into one (matmul of matmul = one matmul)."),
    ("matmul / GEMM", "Matrix multiplication — every output cell is a dot product of a row and a column. ~95% of a transformer's arithmetic. GEMM is its BLAS name (General Matrix Multiply)."),
    ("tiling / blocking", "Splitting matrices into small blocks that fit the fast memory tier (CPU cache / GPU shared memory), doing all math on a block before moving on. Cuts slow-memory traffic by the tile size. The core trick of fast matmul, and of FlashAttention."),
    ("kernel (GPU)", "A function launched on the GPU and executed by thousands of threads in parallel. Not related to OS kernels or ML kernel methods."),
    ("warp", "The GPU's 32-thread execution bundle — threads in a warp execute in lockstep (SIMT). The unit that memory coalescing and shuffle instructions care about."),
    ("coalescing", "When the 32 threads of a warp touch consecutive memory addresses, the hardware merges the accesses into few wide transactions. The single biggest GPU performance factor."),
    ("shared memory (GPU)", "A small, fast, per-block scratchpad you manage explicitly — the GPU's counterpart to the CPU's (implicit) L1 cache. Where tiles live in a tiled kernel."),
    ("cuBLAS / cuDNN / NCCL", "NVIDIA's closed-source workhorses: BLAS (matmul & friends) on GPU; deep-net primitives (convolutions, norms); collective communications (all-reduce & co.) across GPUs. PyTorch calls them under the hood."),
    ("all-reduce", "The collective where every GPU ends up with the sum of everyone's data — how DDP averages gradients. Ring all-reduce moves 2·(R−1)/R of the buffer per rank."),
    ("autograd", "PyTorch's recorder: during forward it builds a graph of operations; `backward()` walks it in reverse applying the chain rule to fill every `.grad`."),
    ("SIMT", "Single Instruction, Multiple Threads — the GPU model: you write scalar code for one thread, the hardware runs it 32-wide per warp. The CPU cousin is SIMD, where the compiler must vectorize explicitly."),
    ("data / tensor / pipeline parallelism", "The three ways to split training across GPUs: copy the model & split the batch (DDP/FSDP); split every matrix across devices (tensor); split the model by layers into an assembly line (pipeline). Large runs compose all three."),
]


def rewrite_lab_links(md: str, slug: str) -> str:
    md = re.sub(r"\]\(((?:cpp|cuda|csrc)/[^)]+|lab\.py)\)",
                rf"]({REPO_BLOB}/cpp-cuda-track/{slug}/\1)", md)
    return md


def strip_first_h1(md: str) -> str:
    lines = md.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).lstrip("\n")


def build_modules():
    for i, (num, slug, short, stage, hours) in enumerate(MODULES):
        src = (TRACK / slug / "README.md").read_text(encoding="utf-8")
        body = rewrite_lab_links(strip_first_h1(src), slug)
        prereq = "none — start here" if i == 0 else \
            f"[module {MODULES[i-1][0]}](../{MODULES[i-1][1]}/)"
        stage_name = STAGES[stage][0].split(" — ")[0]
        meta = (f"**{stage_name}** · ⏳ ~{hours}h · prerequisites: {prereq} · "
                f"[lab source]({REPO_BLOB}/cpp-cuda-track/{slug})\n\n---\n")
        nav = ""
        if i + 1 < len(MODULES):
            nav = (f"\n\n---\n\nNext: [{MODULES[i+1][0]} · {MODULES[i+1][2]}]"
                   f"(../{MODULES[i+1][1]}/)")
        page = f"""---
title: "{num} · {short}"
parent: "C++ ↔ CUDA Track"
nav_order: {i + 1}
permalink: /docs/track/{slug}/
---

# {num} · {short}
{{: .no_toc }}

{meta}
{body}{nav}
"""
        (DOCS / "track" / f"{slug}.md").write_text(page, encoding="utf-8")


def build_track_overview():
    src = (TRACK / "README.md").read_text(encoding="utf-8")
    body = strip_first_h1(src)
    body = re.sub(r"\]\(common/([^)]+)\)", rf"]({REPO_BLOB}/cpp-cuda-track/common/\1)", body)
    page = f"""---
title: "C++ ↔ CUDA Track"
nav_order: 2
has_children: true
permalink: /docs/track/
---

# C++ ↔ CUDA Dual Track

{body}
"""
    (DOCS / "track").mkdir(parents=True, exist_ok=True)
    (DOCS / "track" / "index.md").write_text(page, encoding="utf-8")


def build_glossary():
    items = "\n\n".join(f"{t}\n: {d}" for t, d in GLOSSARY)
    page = f"""---
title: Glossary
nav_order: 3
permalink: /docs/glossary/
---

# Glossary

Every term that made us stop and ask "wait, what does that actually mean?" —
answered in two sentences, with the deeper story linked from the lessons.

{items}
"""
    (DOCS / "glossary.md").write_text(page, encoding="utf-8")


def build_home():
    stage_blocks = []
    for key in "ABCD":
        name, goal = STAGES[key]
        mods = "\n".join(
            f"1. [{num} · {short}](track/{slug}/) — ~{h}h"
            for num, slug, short, st, h in MODULES if st == key)
        stage_blocks.append(f"### {name}\n\n*After this stage you can {goal}.*\n\n{mods}")
    stages = "\n\n".join(stage_blocks)
    page = f"""---
title: Home
nav_order: 1
permalink: /docs/
---

# AI Infrastructure & GPU Engineering
{{: .fs-8 }}

A self-paced path from "what is a feature?" to writing CUDA kernels and sharding
training across GPUs — every lesson backed by runnable code you can measure yourself.
{{: .fs-5 .fw-300 }}

[Start the track](track/){{: .btn .btn-primary }}
[ML foundations chapter](how-machines-learn/){{: .btn }}
[Glossary](glossary/){{: .btn }}

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

{stages}

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
"""
    (DOCS / "index.md").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    build_home()
    build_track_overview()
    build_modules()
    build_glossary()
    print(f"generated {len(MODULES) + 3} markdown pages under docs/")
