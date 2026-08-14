---
title: "Probability, logits & softmax"
parent: "Math for AI"
nav_order: 3
permalink: /docs/math/probability-softmax/
---

# Probability, logits & softmax
{: .no_toc }

**Deep dive:** *Mathematics for Machine Learning*, ch. 6 (probability and distributions).

---

## From scores to probabilities

A classifier's last layer outputs raw scores — **logits** — one per class, any
real number:

```
cat: 3.1    dog: 1.2    fish: −0.4
```

("Logit" is 1944 statistics jargon — *log*istic *unit*, historically the log of
the odds. The useful reading today: scores living on the whole real line,
waiting to be turned into probabilities.)

To say "85% cat" we need numbers that are positive and sum to 1. **Softmax**
does it in three moves:

```
softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ

e^3.1 = 22.2   e^1.2 = 3.3   e^−0.4 = 0.67     (all positive)
sum = 26.2
→ 85%  ·  13%  ·  3%                            (sums to 1)
```

## Why "soft" max

A hard max would output `[1, 0, 0]` — no information about runners-up, and not
differentiable (gradients can't flow through it). Softmax gives the winner most
of the mass but keeps the ranking visible, and it's smooth. Scale all logits up
and softmax sharpens toward hard max; scale them down and it flattens toward
uniform — that scaling knob is exactly the **temperature** parameter of LLM
sampling.

Where you meet it: classifier outputs, the next-token distribution of every LLM
(softmax over ~100k vocabulary logits, once per generated token), and inside
attention — `softmax(Q·Kᵀ/√d)` turns token-affinity scores into listening
weights that sum to 1.

## Cross-entropy: scoring a probability distribution

Training a classifier means comparing the predicted distribution against the
true label. **Cross-entropy loss** is `−log(probability assigned to the correct
class)`: confident-and-right ≈ 0 loss; confident-and-wrong → loss explodes.
Language models train on exactly this, token after token. (PyTorch's
`CrossEntropyLoss` takes *logits*, not probabilities — it applies a numerically
stable log-softmax internally.)

## The numerical trick every implementation needs

`e^100` overflows floating point. Since `softmax(x) = softmax(x − c)` for any
constant, implementations subtract `max(x)` first, making the largest exponent
exactly 0. That "subtract the max" is a reduction — and it's precisely the
max-then-sum funnel implemented in the
[capstone's fused softmax kernel](../../track/12-capstone-pytorch-extension/).
