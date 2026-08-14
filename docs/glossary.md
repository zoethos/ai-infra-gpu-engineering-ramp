---
title: Glossary
nav_order: 4
permalink: /docs/glossary/
---

# Glossary

Every term that made us stop and ask "wait, what does that actually mean?" —
answered in two sentences, with the deeper story linked from the lessons.

tensor
: An N-dimensional array of numbers, the universal data container of deep learning. A PyTorch tensor also remembers device (CPU/GPU), dtype, and how it was computed (for gradients).

feature
: One usable input signal for a model — usually one dataset column (numeric, categorical, or engineered from raw facts).

target / label
: The answer a supervised model learns to predict; paired with features it forms a training example.

weights / parameters
: The adjustable numbers inside a model. Training = finding good values for them.

forward pass
: Running input through the model with current weights to get a prediction. In code: `model(x)`.

loss
: A single number measuring how wrong the model's predictions are on a batch. The whole network is judged by it.

gradient
: For every weight, the answer to "if I nudge this weight up, does the loss go up or down, and how steeply?" Stored per-parameter in `.grad`, same shape as the weight. Computed by `loss.backward()` via the chain rule.

optimizer
: The rule that moves weights using gradients. SGD: `W ← W − η·grad`. Adam/AdamW adds momentum (average of recent gradients) and a per-weight adaptive step (divide by √v of squared-gradient average) — at the cost of two extra floats per parameter.

learning rate (η)
: The step size of each weight update. Too big: training explodes. Too small: it crawls.

epoch / batch
: A batch is the group of examples processed in one forward+backward step; an epoch is one full pass over the dataset.

logits
: The raw scores a network outputs before softmax (from "logistic unit", 1944 — historically the log of the odds). Free-range numbers on the whole real line, waiting to become probabilities.

softmax
: Turns a list of scores into positive numbers summing to 1: exponentiate each, divide by the sum. A 'soft' max: the winner gets most of the mass, not all. Used at classifier outputs, LLM next-token distributions, and inside attention.

ReLU
: Rectified Linear Unit: `max(0, x)` — pass positives, zero out negatives. The 'bend' that makes deep networks non-linear; without it, stacked linear layers collapse into one (matmul of matmul = one matmul).

matmul / GEMM
: Matrix multiplication — every output cell is a dot product of a row and a column. ~95% of a transformer's arithmetic. GEMM is its BLAS name (General Matrix Multiply).

tiling / blocking
: Splitting matrices into small blocks that fit the fast memory tier (CPU cache / GPU shared memory), doing all math on a block before moving on. Cuts slow-memory traffic by the tile size. The core trick of fast matmul, and of FlashAttention.

kernel (GPU)
: A function launched on the GPU and executed by thousands of threads in parallel. Not related to OS kernels or ML kernel methods.

warp
: The GPU's 32-thread execution bundle — threads in a warp execute in lockstep (SIMT). The unit that memory coalescing and shuffle instructions care about.

coalescing
: When the 32 threads of a warp touch consecutive memory addresses, the hardware merges the accesses into few wide transactions. The single biggest GPU performance factor.

shared memory (GPU)
: A small, fast, per-block scratchpad you manage explicitly — the GPU's counterpart to the CPU's (implicit) L1 cache. Where tiles live in a tiled kernel.

cuBLAS / cuDNN / NCCL
: NVIDIA's closed-source workhorses: BLAS (matmul & friends) on GPU; deep-net primitives (convolutions, norms); collective communications (all-reduce & co.) across GPUs. PyTorch calls them under the hood.

all-reduce
: The collective where every GPU ends up with the sum of everyone's data — how DDP averages gradients. Ring all-reduce moves 2·(R−1)/R of the buffer per rank.

autograd
: PyTorch's recorder: during forward it builds a graph of operations; `backward()` walks it in reverse applying the chain rule to fill every `.grad`.

SIMT
: Single Instruction, Multiple Threads — the GPU model: you write scalar code for one thread, the hardware runs it 32-wide per warp. The CPU cousin is SIMD, where the compiler must vectorize explicitly.

data / tensor / pipeline parallelism
: The three ways to split training across GPUs: copy the model & split the batch (DDP/FSDP); split every matrix across devices (tensor); split the model by layers into an assembly line (pipeline). Large runs compose all three.
