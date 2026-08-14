---
title: "Linear algebra & matmul"
parent: "Math for AI"
nav_order: 1
permalink: /docs/math/linear-algebra/
---

# Linear algebra & matmul
{: .no_toc }

**Deep dive:** *Mathematics for Machine Learning*, ch. 2 (linear algebra), ch. 3 (analytic geometry).

---

## Vectors and matrices, operationally

A **vector** is a list of numbers — an input example (784 pixel values), a row of
weights, a gradient. A **matrix** is a table of numbers — a whole layer's weights,
a batch of examples stacked row by row. Deep learning never treats these as abstract
objects: they are literally the arrays in memory that kernels crunch.

## The dot product: one neuron's worth of work

The dot product of two vectors multiplies them element by element and sums:

```
a · b = a₁b₁ + a₂b₂ + … + aₙbₙ
```

That is exactly what one artificial neuron does with its inputs and weights
(plus a bias and an activation). Geometrically it measures *alignment* — which is
why attention uses `Q·K` to score how much one token should listen to another.

## Matmul: all the neurons at once

Matrix multiplication is nothing but dot products in bulk: cell `C[i][j]` is the
dot product of row `i` of A with column `j` of B.

```
A = | 1  2 |     B = | 5  6 |      C[0][0] = 1·5 + 2·7 = 19
    | 3  4 |         | 7  8 |      C[0][1] = 1·6 + 2·8 = 22 …
```

A linear layer computing 128 neurons over a batch of 64 examples is a single
(64×784) @ (784×128) matmul. **~95% of a transformer's arithmetic is matmuls** —
Q/K/V projections, attention scores, feed-forward layers. When a GPU's spec sheet
says "1000 TFLOPS", that number is measured on matmul, because matmul is what the
hardware is built around.

## Why matmul is special to hardware: reuse

For n×n matrices, matmul does n³ multiply-adds on only n² data — every element
gets reused n times. That growing compute-per-byte ratio is why
[tiling](../../track/06-matmul-tiling/) works, why tensor cores exist, and why
matmul can approach a chip's compute peak while almost everything else is stuck
at the memory-bandwidth ceiling.

## The one identity that explains ReLU's job

Two linear layers back-to-back collapse into one:

```
W₂ · (W₁ · x) = (W₂ · W₁) · x = W · x
```

Matmul of matmul is a single matmul — depth adds *nothing* without a
non-linearity between layers. That's the algebraic reason `nn.ReLU()` sits
between every pair of `nn.Linear`s: the bend it introduces is what makes depth
expressive. (Full story in the [ML foundations chapter](../../how-machines-learn/).)
