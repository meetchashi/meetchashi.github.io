### Worked example: learned positions

<figure class="pe-scene pe-work-flow pe-a">
<div>Look up x</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Look up position row p</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Add: h = x + p</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Make q, k, v from h</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Compare and gather</div>
</figure>

#### Step 0: Keep the sentence and choose a position table

Here is one possible toy table. We chose a tidy pattern so you can check the sums; a trained table need not follow any such pattern.

| Position | Word | Position row p | Input after adding: h = x + p |
|---|---|---|---|
| 0 | the | (0.000, 0.000, 0.000, 0.000) | (1.000, 0.000, 1.000, 0.000) |
| 1 | cat | (0.100, 0.000, 0.000, 0.050) | (0.100, 1.000, 0.000, 1.050) |
| 2 | sat | (0.200, 0.000, 0.000, 0.100) | (1.200, 1.000, 0.000, 0.100) |
| 3 | on | (0.300, 0.000, 0.000, 0.150) | (0.300, 0.000, 1.000, 1.150) |
| 4 | the | (0.400, 0.000, 0.000, 0.200) | (1.400, 0.000, 1.000, 0.200) |
| 5 | soft | (0.500, 0.000, 0.000, 0.250) | (1.500, 0.000, 0.000, 1.250) |
| 6 | mat | (0.600, 0.000, 0.000, 0.300) | (0.600, 1.000, 1.000, 0.300) |

#### Step 1: Compare the two copies of “the”

At position 0, **(1.000, 0.000, 1.000, 0.000) + (0.000, 0.000, 0.000, 0.000) = (1.000, 0.000, 1.000, 0.000)**.

At position 4, **(1.000, 0.000, 1.000, 0.000) + (0.400, 0.000, 0.000, 0.200) = (1.400, 0.000, 1.000, 0.200)**.

The lookup embeddings are identical. The rows entering attention are now different.

#### Step 2: Make queries, keys, and values from the combined rows

Our weights give q₄ = k₄ = **(1.400, 0.000, 1.000, 0.200)** and v₄ = **(0.200, 1.000, 0.000, 1.400)**. Notice that **values change too**: they are made from the word-plus-position input, not the original word row.

#### Step 3: Follow one comparison

Against the first “the”: `1.4×1 + 0×0 + 1×1 + 0.2×0 = 2.4`. Divide by 2 → **1.200**. Against itself the dot is `1.4² + 1² + 0.2² = 3.0` → **1.500**.

| Key position | Word | Query dot key | Scaled score |
|---|---|---|---|
| 0 | the | 2.400 | 1.200 |
| 1 | cat | 0.350 | 0.175 |
| 2 | sat | 1.700 | 0.850 |
| 3 | on | 1.650 | 0.825 |
| 4 | the | 3.000 | 1.500 |

#### Step 4: Convert scores to weights, then gather values

The two “the” connections no longer tie. This follows from the added position rows, not from a changed word identity.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>24.4%</span><i style="--weight: 24.386%"></i></div> <div><strong>cat <small>position 1</small></strong><span>8.7%</span><i style="--weight: 8.750%"></i></div> <div><strong>sat <small>position 2</small></strong><span>17.2%</span><i style="--weight: 17.185%"></i></div> <div><strong>on <small>position 3</small></strong><span>16.8%</span><i style="--weight: 16.761%"></i></div> <div><strong>the <small>position 4</small></strong><span>32.9%</span><i style="--weight: 32.918%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.368, 0.741, 0.259, 0.970)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.

