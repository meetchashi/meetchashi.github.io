#### Worked example: Shaw relative representations

<figure class="pe-scene pe-work-flow pe-b">
<div>Word rows unchanged</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Make q, k, v</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Key + relative-key row</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Match → weights</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Value + relative-value row</div>
</figure>

#### Step 0: Use relationship labels instead of seat labels

Our gap is **query position minus key position**: a positive gap means the key is earlier. This reverses the original paper’s indexing convention, so we relabel the learned table consistently. Clip gaps to −2…2.

| Gap label | Relative-key row | Relative-value row |
|---|---|---|
| -2 | (-0.200, 0.000, 0.000, 0.000) | (0.000, -0.100, 0.000, 0.000) |
| -1 | (-0.100, 0.000, 0.000, 0.000) | (0.000, -0.050, 0.000, 0.000) |
| 0 | (0.000, 0.000, 0.000, 0.000) | (0.000, 0.000, 0.000, 0.000) |
| 1 | (0.100, 0.000, 0.000, 0.000) | (0.000, 0.050, 0.000, 0.000) |
| 2 | (0.200, 0.000, 0.000, 0.000) | (0.000, 0.100, 0.000, 0.000) |

#### Step 1: Follow “the” looking at “cat”

The query is at 4 and the key is at 1. Gap 3 is clipped to 2. For this pair, add **(0.2, 0, 0, 0)** to cat’s key **(0, 1, 0, 1)** → **(0.2, 1, 0, 1)**. The stored cat key has not permanently changed; the extra row belongs to this particular relationship.

#### Step 2: Score the relationship

The query **(1, 0, 1, 0)** dotted with **(0.2, 1, 0, 1)** gives **0.2**. Divide by 2 → **0.100**, rather than the content-only score 0.

| Key | Gap → clipped gap | Content-only score | Score with relative key |
|---|---|---|---|
| 0: the | 4 → 2 | 1.000 | 1.100 |
| 1: cat | 3 → 2 | 0.000 | 0.100 |
| 2: sat | 2 → 2 | 0.500 | 0.600 |
| 3: on | 1 → 1 | 0.500 | 0.550 |
| 4: the | 0 → 0 | 1.000 | 1.000 |

#### Step 3: Change what is gathered as well

Cat’s value is **(1, 0, 1, 0)**. For this gap we add **(0, 0.1, 0, 0)** → **(1, 0.1, 1, 0)**. Thus the weighted mixture carries relationship information as well as content.

#### Step 4: Mix the pair-specific values

Offsets beyond the clipping limit share a label. This head can distinguish “nearby” and “farther away,” but its position table treats gaps 2, 3, and 4 alike.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>28.9%</span><i style="--weight: 28.934%"></i></div> <div><strong>cat <small>position 1</small></strong><span>10.6%</span><i style="--weight: 10.644%"></i></div> <div><strong>sat <small>position 2</small></strong><span>17.5%</span><i style="--weight: 17.549%"></i></div> <div><strong>on <small>position 3</small></strong><span>16.7%</span><i style="--weight: 16.693%"></i></div> <div><strong>the <small>position 4</small></strong><span>26.2%</span><i style="--weight: 26.180%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.273, 0.784, 0.282, 0.727)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.

