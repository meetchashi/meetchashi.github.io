### Worked example: ALiBi

<figure class="pe-scene pe-work-flow pe-b">
<div>Word rows unchanged</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Make q, k, v</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Content scores</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Subtract distance costs</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Weights mix original values</div>
</figure>

#### Step 0: Choose a rate for this head

Use a fixed **0.25 per token step**. This is a teaching rate; real ALiBi assigns a prescribed set of rates across heads.

#### Step 1: Compute the content matches first

The second “the” has q₄ = **(1, 0, 1, 0)**. Before distance penalties, the score row is **(1, 0, 0.5, 0.5, 1)**. Keys and values remain as listed in the shared setup.

#### Step 2: Subtract the price of distance

For the first “the,” gap 4 costs **0.25×4 = 1**. Its score changes from **1** to **0**. For “on,” gap 1 costs **0.25**; its score changes from **0.5** to **0.25**.

| Key | Gap | Content score | Distance penalty | New score |
|---|---|---|---|---|
| 0: the | 4 | 1.000 | -1.000 | 0.000 |
| 1: cat | 3 | 0.000 | -0.750 | -0.750 |
| 2: sat | 2 | 0.500 | -0.500 | 0.000 |
| 3: on | 1 | 0.500 | -0.250 | 0.250 |
| 4: the | 0 | 1.000 | 0.000 | 1.000 |

#### Step 3: Mix values according to the new weights

The identical “the” keys now receive different weights because one is four steps away and the other is at the query’s own position. A distant key could still win if its content match were strong enough.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>15.4%</span><i style="--weight: 15.445%"></i></div> <div><strong>cat <small>position 1</small></strong><span>7.3%</span><i style="--weight: 7.296%"></i></div> <div><strong>sat <small>position 2</small></strong><span>15.4%</span><i style="--weight: 15.445%"></i></div> <div><strong>on <small>position 3</small></strong><span>19.8%</span><i style="--weight: 19.832%"></i></div> <div><strong>the <small>position 4</small></strong><span>42.0%</span><i style="--weight: 41.983%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.271, 0.773, 0.227, 0.729)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.

