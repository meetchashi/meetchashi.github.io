---
layout: blog_post
title: "Why LLMs Are Numerically Fragile: A Researcher's Perspective"
date: 2025-05-17 00:00:00 +0000
tags: [research, LLMs, robustness]
---

One of the more surprising findings from my recent research is how numerically fragile large language models
actually are. We're used to thinking of these systems as remarkably capable — able to reason across domains,
write code, and engage in nuanced conversation. But underneath all that capability lies floating-point arithmetic,
and floating-point arithmetic is not exact.

## The Core Observation

When you perturb a token embedding by a difference smaller than your eye can see on a plot — on the order of
the rounding error in 16-bit floating point — the perturbation doesn't stay small. Through dozens of transformer
layers, each performing a cascade of matrix multiplications, the perturbation amplifies. In our experiments
([arxiv:2604.13206](https://arxiv.org/abs/2604.13206)), we measured amplification factors exceeding **10 million**
by the time the signal reaches the final logits.

That amplification has a direct consequence: the *margin* between the top predicted token and its competitors
collapses. A model that was 99% confident in predicting one token becomes 50/50 on multiple candidates — and
tiny fluctuations in the computation path determine which one wins.

## Why Does This Matter?

This matters for two reasons.

**First, reliability.** We often evaluate LLMs on fixed benchmarks with fixed inputs. But in deployment,
floating-point computation can vary across hardware (different GPU architectures, different CUDA versions,
even thermal throttling). If a model's decision is sitting in this chaotic regime, the same "logical" input
might produce different outputs on different machines. That's a serious concern for any safety-critical
application.

**Second, security.** An adversary who understands this amplification phenomenon can engineer inputs that
sit right at the edge of a token decision boundary. With minimal, often imperceptible perturbations, they
can reliably steer the model toward a different output. This is distinct from the classic "jailbreak" framing —
it's not about manipulating the model's values or safety alignment, it's about exploiting the numerical
geometry of the prediction itself.

## What Does This Mean for Practitioners?

If you're deploying an LLM in a production environment, here are a few things worth thinking about:

- **Don't assume reproducibility across hardware.** If decisions are in a chaotic regime, you may get
  different outputs on different machines for the same input. Consider ensemble-based confidence estimation.
- **Confidence scores aren't calibrated in this regime.** A high softmax probability doesn't mean the
  model is actually certain; it may be one perturbation away from flipping.
- **Adversarial robustness is not just a fine-tuning problem.** The fragility comes from the numerical
  nature of the architecture, not from training misalignment.

## What's Next

We're currently working on methods to detect when a given input sits in this chaotic amplification regime —
essentially a "numerical uncertainty certificate" that flags predictions likely to be unstable. Stay tuned.

If you're interested in the technical details, the preprint is available at
[arXiv:2604.13206](https://arxiv.org/abs/2604.13206) and the codebase is on
[GitHub](https://github.com/meetchashi/llm_rounding_error_instability).
