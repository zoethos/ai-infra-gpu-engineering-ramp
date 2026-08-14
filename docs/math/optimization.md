---
title: "Optimization: SGD to AdamW"
parent: "Math for AI"
nav_order: 4
permalink: /docs/math/optimization/
---

# Optimization: SGD to AdamW
{: .no_toc }

**Deep dive:** *Mathematics for Machine Learning*, ch. 7 (continuous optimization).

---

## Gradient descent: the base move

```
W ← W − η · ∇L        (η = learning rate)
```

Stand somewhere on the loss landscape, feel the slope under your feet, take a
small step downhill, repeat. **Stochastic** gradient descent (SGD) estimates the
slope from a mini-batch instead of the whole dataset — noisier, vastly cheaper,
and the noise even helps escape bad flat spots.

The learning rate η is the most sensitive dial in deep learning: too large and
the loss explodes, too small and training crawls. Real runs *schedule* it —
warmup, then decay.

## Momentum: stop zigzagging

In narrow valleys the gradient points wall-to-wall instead of along the valley
floor. Keeping a running average of recent gradients — momentum — cancels the
zigzag components and accumulates the consistent direction, like a ball rolling
rather than a hiker teleporting.

```
m ← β·m + (1−β)·grad ;    W ← W − η·m
```

## Adam: momentum + a personal step size per weight

**Adam** (Adaptive Moment Estimation, 2014 — the field's default) keeps *two*
running averages per weight: the gradient (`m`, momentum) and the squared
gradient (`v`, a measure of typical gradient size), then scales each weight's
step by 1/√v:

```
W ← W − η · m / (√v + ε)
```

Weights with chronically large gradients get calmed down; weights with rare,
small gradients get amplified. Every parameter receives its own effective
learning rate, continuously recalibrated — which is why Adam mostly "just works"
at η = 0.001 while plain SGD needs careful tuning.

**The systems bill:** m and v are two extra floats *per parameter*. For a 7B
model in fp32: 28 GB weights + 28 GB gradients + 56 GB Adam state ≈ **112 GB**
— four times the inference footprint. That 4× is why a single 80 GB GPU can't
naively train a 7B model, and why ZeRO/FSDP exist: their first move is
literally *sharding the optimizer states* across GPUs
(see [module 11](../../track/11-multi-device/)).

## AdamW: the variant everyone actually uses

AdamW decouples **weight decay** (the regularization pull toward zero) from the
adaptive machinery, applying it directly to the weights instead of mixing it
into the gradient. Small fix, measurable win — every modern LLM you've heard of
was trained with AdamW.
