#### Worked example: T5 distance buckets

<figure class="pe-scene pe-work-flow pe-b">
<div>Word rows unchanged</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Make q, k, v</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Content scores</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Add bucket biases</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Weights mix original values</div>
</figure>

#### Step 0: Choose a small learned bias table

This is an invented table for **one head**, not T5’s real bucket boundaries or trained values. Direction is “backward” because our comparison uses causal attention.

| Distance bucket | Learned score adjustment |
|---|---|
| self | 0.000 |
| one back | 0.300 |
| two back | 0.100 |
| 3–4 back | -0.200 |
| 5+ back | -0.400 |

#### Step 1: Keep q, k, and v as they were

For the second “the,” q₄ = **(1, 0, 1, 0)**. Its content-only score row is **(1, 0, 0.5, 0.5, 1)**. We have not added a position vector to any word row.

#### Step 2: Look up one bias for each pair

Looking from position 4 to “on” at 3 gives gap 1. Content score **0.5**, bucket bias **+0.3**, new score **0.8**. Looking back to the first “the” gives gap 4: **1.0 − 0.2 = 0.8**.

| Key | Gap | Bucket | Content score | Add bias | New score |
|---|---|---|---|---|---|
| 0: the | 4 | 3–4 back | 1.000 | -0.200 | 0.800 |
| 1: cat | 3 | 3–4 back | 0.000 | -0.200 | -0.200 |
| 2: sat | 2 | two back | 0.500 | 0.100 | 0.600 |
| 3: on | 1 | one back | 0.500 | 0.300 | 0.800 |
| 4: the | 0 | self | 1.000 | 0.000 | 1.000 |

#### Step 3: Gather values with the new weights

The score row changed. The query, key, and value rows did not receive any direct positional transformation in this calculation.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>22.7%</span><i style="--weight: 22.686%"></i></div> <div><strong>cat <small>position 1</small></strong><span>8.3%</span><i style="--weight: 8.346%"></i></div> <div><strong>sat <small>position 2</small></strong><span>18.6%</span><i style="--weight: 18.574%"></i></div> <div><strong>on <small>position 3</small></strong><span>22.7%</span><i style="--weight: 22.686%"></i></div> <div><strong>the <small>position 4</small></strong><span>27.7%</span><i style="--weight: 27.709%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.310, 0.731, 0.269, 0.690)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.

