---
title: "Calculus & gradients"
parent: "Math for AI"
nav_order: 2
permalink: /docs/math/calculus-gradients/
---

# Calculus & gradients
{: .no_toc }

**Deep dive:** *Mathematics for Machine Learning*, ch. 5 (vector calculus).

---

## The whole of training in one toy

Predict house prices with `price = w × m²`. Start with a random `w = 7`.
A 50 m² house comes out at 350 € instead of 100,000 € — too low, so raise `w`.
Try again, measure again, adjust again: 7 → 500 → 1,800 → 2,100 (too high) →
2,010 → **2,000**. Found.

That guess-measure-adjust loop *is* gradient descent. Everything below is just
making "which way, and by how much?" precise — and scaling it from one knob to
a trillion.

## Derivative → gradient

The **derivative** of a function at a point is its local slope: nudge the input
a hair, how much does the output move? For a function of *many* inputs — like a
loss that depends on millions of weights — take the derivative with respect to
each input separately (partial derivatives) and collect them in a vector: the
**gradient**.

```
∇L = [ ∂L/∂w₁ , ∂L/∂w₂ , … , ∂L/∂wₙ ]
```

Reading: `∂L/∂wᵢ` answers *"if I nudge weight i up, does the loss rise or fall,
and how steeply?"* The gradient points in the direction of steepest **increase**
of the loss — so training steps go the opposite way.

In PyTorch every parameter gets a twin tensor of the same shape: `weight.grad`.
A 7-billion-parameter model carries 7 billion gradient floats — the second
28 GB in the training memory bill.

## The chain rule: why deep networks are trainable at all

A network is a composition: `loss = f₃(f₂(f₁(x)))`. The chain rule says the
derivative of a composition is the *product of the pieces' derivatives*:

```
∂loss/∂w₁ = ∂loss/∂f₂ · ∂f₂/∂f₁ · ∂f₁/∂w₁
```

**Backpropagation is nothing more than the chain rule applied backwards through
the network, reusing intermediate results.** PyTorch's autograd records every
operation during the forward pass; `loss.backward()` walks that record in
reverse, multiplying local derivatives, filling every `.grad` in one sweep.
Each operation only needs to know its own derivative — the chain rule composes
them. (ReLU's derivative: 1 where it passed, 0 where it blocked. That's why it's
free.)

## Visual recap

The training loop with its algebra, drawn out (labels in Italian — from this
project's study notes): [training-loop diagram](../../how-machines-learn/ciclo-addestramento.svg)
and [why ReLU / non-linearity](../../how-machines-learn/perche-relu.svg).
