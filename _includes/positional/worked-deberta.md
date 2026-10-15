#### Worked example: DeBERTa’s separate comparisons

<figure class="pe-scene pe-work-flow pe-b">
<div>Content rows unchanged</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Content q / k + position q / k</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Three separate matches</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Add and scale scores</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Weights gather content values</div>
</figure>

#### Step 0: Give content and position their own representations

For the toy, content projections and the position-key projection are identities. The position-query projection halves the first coordinate and copies the other three. For a signed offset g, the relative-position row is **P(g) = (0.1g, 0.2, 0, 0)**. Actual position rows and projections are learned; this formula just specifies our invented table.

#### Step 1: Match the content

“The” at 4 looking at “cat” at 1: content query **(1, 0, 1, 0)** meets content key **(0, 1, 0, 1)** → content–content match **0**.

#### Step 2: Add two directional position matches

Gap = 3. The content query meets position key **P(+3) = (0.3, 0.2, 0, 0)** → **0.3**. Project P(−3) to position query **(−0.15, 0.2, 0, 0)** meets cat’s content key → **0.2**. Both directions participate, using opposite signed offsets.

#### Step 3: Combine the three matches

For cat, raw total **0 + 0.3 + 0.2 = 0.5**. For this three-term, four-coordinate example, divide by **sqrt(3×4) ≈ 3.464** → **0.144**. We use the three-term attention scale rather than quietly treating it as a single dot product.

| Key | Content↔content | Content→position | Position→content | Scaled total |
|---|---|---|---|---|
| 0: the | 2.000 | 0.400 | -0.200 | 0.635 |
| 1: cat | 0.000 | 0.300 | 0.200 | 0.144 |
| 2: sat | 1.000 | 0.200 | 0.100 | 0.375 |
| 3: on | 1.000 | 0.100 | 0.000 | 0.318 |
| 4: the | 2.000 | 0.000 | 0.000 | 0.577 |

#### Step 4: Gather content values

Relative-position representations affected the score through separate matches. They were not blended into each initial token row.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong> <span>24.7%</span><i style="--weight: 24.660%"></i></div> <div><strong>cat <small>position 1</small></strong> <span>15.1%</span><i style="--weight: 15.096%"></i></div> <div><strong>sat <small>position 2</small></strong> <span>19.0%</span><i style="--weight: 19.018%"></i></div> <div><strong>on <small>position 3</small></strong> <span>18.0%</span><i style="--weight: 17.951%"></i></div> <div><strong>the <small>position 4</small></strong> <span>23.3%</span><i style="--weight: 23.276%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.330, 0.659, 0.341, 0.670)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.

