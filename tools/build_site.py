#!/usr/bin/env python3
"""Build the GitHub Pages learning site from the repo's markdown sources.

Reads:  cpp-cuda-track/README.md and cpp-cuda-track/*/README.md
Writes: docs/index.html (hub), docs/track/**, docs/glossary/
Run:    python3 tools/build_site.py   (requires: pip install markdown)

Design contract: every page links ../site.css (depth-relative), no external
runtime dependencies, works on any static host.
"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TRACK = ROOT / "cpp-cuda-track"
REPO_BLOB = "https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main"

# num, slug, short title, stage, est hours
MODULES = [
    ("01", "01-execution-model",          "Execution model",            "A", 2),
    ("02", "02-memory-hierarchy",         "Memory hierarchy",           "A", 3),
    ("03", "03-data-parallel-saxpy",      "Data parallelism: SAXPY",    "A", 2),
    ("04", "04-reduction",                "Reduction",                  "B", 3),
    ("05", "05-scan-histogram",           "Scan & histogram",           "B", 4),
    ("06", "06-matmul-tiling",            "Matmul & tiling",            "B", 4),
    ("07", "07-async-overlap",            "Asynchrony & overlap",       "C", 3),
    ("08", "08-sync-atomics-memory-model","Sync, atomics, memory model","C", 3),
    ("09", "09-profiling-roofline",       "Profiling & roofline",       "C", 3),
    ("10", "10-advanced-gpu",             "Advanced GPU",               "D", 5),
    ("11", "11-multi-device",             "Multi-device",               "D", 4),
    ("12", "12-capstone-pytorch-extension","Capstone: PyTorch extension","D", 6),
]
STAGES = {
    "A": ("Stage A — Foundations of parallel execution",
          "after this stage you can explain why the same loop runs 100× faster on a GPU, and measure it"),
    "B": ("Stage B — Parallel algorithms",
          "after this stage you can build the three algorithmic shapes 90% of kernels reduce to"),
    "C": ("Stage C — Systems craft",
          "after this stage you overlap, synchronize and profile like it's a habit"),
    "D": ("Stage D — The frontier",
          "after this stage you read FlashAttention and CUTLASS as engineering, not magic"),
}

GLOSSARY = [
    ("tensor", "An N-dimensional array of numbers, the universal data container of deep learning. A PyTorch tensor also remembers device (CPU/GPU), dtype, and how it was computed (for gradients)."),
    ("feature", "One usable input signal for a model — usually one dataset column (numeric, categorical, or engineered from raw facts)."),
    ("target / label", "The answer a supervised model learns to predict; paired with features it forms a training example."),
    ("weights / parameters", "The adjustable numbers inside a model. Training = finding good values for them."),
    ("forward pass", "Running input through the model with current weights to get a prediction. In code: <code>model(x)</code>."),
    ("loss", "A single number measuring how wrong the model's predictions are on a batch. The whole network is judged by it."),
    ("gradient", "For every weight, the answer to “if I nudge this weight up, does the loss go up or down, and how steeply?” Stored per-parameter in <code>.grad</code>, same shape as the weight. Computed by <code>loss.backward()</code> via the chain rule."),
    ("optimizer", "The rule that moves weights using gradients. SGD: <code>W ← W − η·grad</code>. Adam/AdamW: adds momentum (average of recent gradients) and a per-weight adaptive step (divide by √v of squared-gradient average) — at the cost of two extra floats per parameter."),
    ("learning rate (η)", "The step size of each weight update. Too big: training explodes. Too small: it crawls."),
    ("epoch / batch", "A batch is the group of examples processed in one forward+backward step; an epoch is one full pass over the dataset."),
    ("logits", "The raw scores a network outputs before softmax (from “logistic unit”, 1944 — historically the log of the odds). Free-range numbers on the whole real line, waiting to become probabilities."),
    ("softmax", "Turns a list of scores into positive numbers summing to 1: exponentiate each, divide by the sum. A ‘soft’ max: the winner gets most of the mass, not all. Used at classifier outputs, LLM next-token distributions, and inside attention."),
    ("ReLU", "Rectified Linear Unit: <code>max(0, x)</code> — pass positives, zero out negatives. The ‘bend’ that makes deep networks non-linear; without it, stacked linear layers collapse into one (matmul of matmul = one matmul)."),
    ("matmul / GEMM", "Matrix multiplication — every output cell is a dot product of a row and a column. ~95% of a transformer's arithmetic. GEMM is its BLAS name (General Matrix Multiply)."),
    ("tiling / blocking", "Splitting matrices into small blocks that fit the fast memory tier (CPU cache / GPU shared memory), doing all math on a block before moving on. Cuts slow-memory traffic by the tile size. The core trick of fast matmul, and of FlashAttention."),
    ("kernel (GPU)", "A function launched on the GPU and executed by thousands of threads in parallel. Not related to OS kernels or ML kernel methods."),
    ("warp", "The GPU's 32-thread execution bundle — threads in a warp execute in lockstep (SIMT). The unit that memory coalescing and shuffle instructions care about."),
    ("coalescing", "When the 32 threads of a warp touch consecutive memory addresses, the hardware merges the accesses into few wide transactions. The single biggest GPU performance factor."),
    ("shared memory (GPU)", "A small, fast, per-block scratchpad you manage explicitly — the GPU's counterpart to the CPU's (implicit) L1 cache. Where tiles live in a tiled kernel."),
    ("cuBLAS / cuDNN / NCCL", "NVIDIA's closed-source workhorses: BLAS (matmul & friends) on GPU; deep-net primitives (convolutions, norms); collective communications (all-reduce & co.) across GPUs. PyTorch calls them under the hood."),
    ("all-reduce", "The collective where every GPU ends up with the sum of everyone's data — how DDP averages gradients. Ring all-reduce moves 2·(R−1)/R of the buffer per rank."),
    ("autograd", "PyTorch's recorder: during forward it builds a graph of operations; <code>backward()</code> walks it in reverse applying the chain rule to fill every <code>.grad</code>."),
    ("SIMT", "Single Instruction, Multiple Threads — the GPU model: you write scalar code for one thread, the hardware runs it 32-wide per warp. The CPU cousin is SIMD, where the compiler must vectorize explicitly."),
    ("data / tensor / pipeline parallelism", "The three ways to split training across GPUs: copy the model & split the batch (DDP/FSDP); split every matrix across devices (tensor); split the model by layers into an assembly line (pipeline). Large runs compose all three."),
]

MD = markdown.Markdown(extensions=["tables", "fenced_code"])


def render_md(text: str) -> str:
    MD.reset()
    html = MD.convert(text)
    # wrap tables for mobile horizontal scroll
    html = html.replace("<table>", '<div class="tablewrap"><table>')
    html = html.replace("</table>", "</table></div>")
    return html


def page(title: str, body: str, depth: int, here: str = "") -> str:
    rel = "../" * depth
    def nav(href, label, key):
        cls = ' class="here"' if key == here else ""
        return f'<a href="{rel}{href}"{cls}>{label}</a>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{rel}site.css">
</head>
<body>
<div class="wrap">
<nav class="sitenav">
  <a class="brand" href="{rel}">AI Infra &amp; GPU Ramp</a>
  {nav("", "Home", "home")}
  {nav("how-machines-learn/", "ML Foundations", "ml")}
  {nav("track/", "C++&hairsp;&harr;&hairsp;CUDA Track", "track")}
  {nav("glossary/", "Glossary", "gloss")}
  <a href="https://github.com/trocca/ai-infra-gpu-engineering-ramp">GitHub</a>
</nav>
{body}
<footer class="site">
  Part of <a href="https://github.com/trocca/ai-infra-gpu-engineering-ramp">ai-infra-gpu-engineering-ramp</a>
  — a documented AI infrastructure learning journey. Site rebuilt from the repo's markdown by
  <a href="https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/tools/build_site.py">tools/build_site.py</a>.
</footer>
</div>
</body>
</html>
"""


def rewrite_module_links(html: str, slug: str) -> str:
    # lab files -> GitHub source
    html = re.sub(r'href="((?:cpp|cuda|csrc)/[^"]+|lab\.py)"',
                  rf'href="{REPO_BLOB}/cpp-cuda-track/{slug}/\1"', html)
    # sibling module links ../NN-slug/ already match the site layout — leave them
    return html


def build_modules():
    for i, (num, slug, short, stage, hours) in enumerate(MODULES):
        src = (TRACK / slug / "README.md").read_text(encoding="utf-8")
        content = rewrite_module_links(render_md(src), slug)
        prereq = "none — start here" if i == 0 else \
            f'<a href="../{MODULES[i-1][1]}/">module {MODULES[i-1][0]}</a>'
        stage_name = STAGES[stage][0].split(" — ")[0]
        meta = f"""<div class="metastrip">
  <span class="badge stage{stage}">{stage_name}</span>
  <span>&#8987; ~{hours}h</span>
  <span>&#128280; prerequisites: {prereq}</span>
  <span>&#128193; <a href="{REPO_BLOB}/cpp-cuda-track/{slug}">lab source on GitHub</a></span>
</div>"""
        prev_html = (f'<a href="../{MODULES[i-1][1]}/">&larr; {MODULES[i-1][0]} · {MODULES[i-1][2]}</a>'
                     if i > 0 else '<a href="../">&larr; Track overview</a>')
        next_html = (f'<a href="../{MODULES[i+1][1]}/">{MODULES[i+1][0]} · {MODULES[i+1][2]} &rarr;</a>'
                     if i + 1 < len(MODULES) else '<a href="../../">Finish line — back to Home &rarr;</a>')
        body = f'{meta}\n<div class="content">\n{content}\n</div>\n' \
               f'<div class="pager">{prev_html}<span class="spacer"></span>{next_html}</div>'
        out = DOCS / "track" / slug / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page(f"{num} · {short} — C++ ↔ CUDA Track", body, depth=2, here="track"),
                       encoding="utf-8")


def build_track_overview():
    src = (TRACK / "README.md").read_text(encoding="utf-8")
    html = render_md(src)
    html = re.sub(r'href="common/([^"]+)"', rf'href="{REPO_BLOB}/cpp-cuda-track/common/\1"', html)
    modules_list = "\n".join(
        f'<li><a href="{slug}/">{num} · {short}</a> '
        f'<span class="tiny">(~{hours}h)</span></li>'
        for num, slug, short, stage, hours in MODULES)
    body = (f'<div class="content">\n{html}\n</div>'
            f'<h2>Read the lessons on this site</h2><ol>{modules_list}</ol>'
            f'<div class="pager"><a href="../">&larr; Home</a><span class="spacer"></span>'
            f'<a href="01-execution-model/">Start module 01 &rarr;</a></div>')
    out = DOCS / "track" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page("C++ ↔ CUDA Dual Track — overview", body, depth=1, here="track"),
                   encoding="utf-8")


def build_glossary():
    items = "\n".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY)
    body = f"""<div class="eyebrow">Reference</div>
<h1>Glossary</h1>
<p class="lede">Every term that made us stop and ask &ldquo;wait, what does that actually mean?&rdquo;
— answered in two sentences, with the deeper story linked from the lessons.</p>
<dl class="gloss">{items}</dl>"""
    out = DOCS / "glossary" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page("Glossary — AI Infra & GPU Ramp", body, depth=1, here="gloss"),
                   encoding="utf-8")


def build_hub():
    def stage_card(key, cls):
        name, goal = STAGES[key]
        mods = "\n".join(
            f'<li><a href="track/{slug}/">{num} · {short}</a> <span class="tiny">~{h}h</span></li>'
            for num, slug, short, st, h in MODULES if st == key)
        return f'<div class="card stage {cls}"><h3>{name}</h3>' \
               f'<p class="goal">{goal}</p><ol>{mods}</ol></div>'

    body = f"""<div class="eyebrow">My AI Journey &middot; a documented learning ramp</div>
<h1>AI Infrastructure &amp; GPU Engineering</h1>
<p class="lede">A self-paced path from &ldquo;what is a feature?&rdquo; to writing CUDA kernels and
sharding training across GPUs — with every lesson backed by runnable code you can measure yourself.</p>

<h2>Start here</h2>
<div class="grid2">
  <div class="card"><h3>&#127793; New to machine learning?</h3>
    <p class="muted">Read <a href="how-machines-learn/">How Machines Learn from Data</a> first:
    features, training vs inference, the learning paradigms, and the modern LLM workflow — no code required.</p></div>
  <div class="card"><h3>&#9889; Here for the systems side?</h3>
    <p class="muted">Jump into the <a href="track/">C++&nbsp;&harr;&nbsp;CUDA dual track</a>: the same
    operation implemented on both stacks, confronted module by module. Keep the
    <a href="glossary/">glossary</a> in a tab.</p></div>
</div>

<h2>The path, in four stages</h2>
<p class="muted">Do the stages in order; inside a stage, do modules in order. Each lesson states its
prerequisite and an honest time estimate — the hours assume you run the labs, not just read.</p>
{stage_card("A", "")}
{stage_card("B", "b")}
{stage_card("C", "c")}
{stage_card("D", "d")}

<h2>How to study each lesson</h2>
<div class="card"><ol>
<li>Read the lesson page (concept + the CPU-vs-GPU confrontation table).</li>
<li>Build and run the <b>cpp/</b> lab, then the <b>cuda/</b> lab — links to the exact source files are on every page.</li>
<li>Answer the exercises <i>in writing</i> in your own NOTES.md. The questions are chosen so the answers differ between the stacks — that delta is the lesson.</li>
<li>Only then read the companion book chapters listed at the bottom of the lesson.</li>
</ol></div>

<h2>Reference shelf</h2>
<ul>
<li><a href="https://nanotron-ultrascale-playbook.static.hf.space/">The Ultra-Scale Playbook</a> — free; the parallelism textbook.</li>
<li><i>AI Systems Performance Engineering</i> (Fregly, O&rsquo;Reilly 2025) — spread across the track as per-lesson companion reading.</li>
<li><i>Mathematics for Machine Learning</i> (Deisenroth, Faisal, Ong) — free PDF from the authors; ch. 5 (gradients) pairs with the ML foundations chapter.</li>
</ul>"""
    (DOCS / "index.html").write_text(page("AI Infra & GPU Engineering Ramp", body, depth=0, here="home"),
                                     encoding="utf-8")


if __name__ == "__main__":
    build_modules()
    build_track_overview()
    build_glossary()
    build_hub()
    n = len(list((DOCS / "track").rglob("index.html"))) + 2
    print(f"site built: {n} pages under docs/")
