#### Worked example: Transformer-XL across a segment boundary

<figure class="pe-scene pe-work-flow pe-b">
<div>Memory: the · cat</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Current: sat · on · the · soft · mat</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Current queries + all available keys</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Relative-distance matches</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>5 current output rows</div>
</figure>

#### Step 0: Split our sentence into memory and current text

Remember positions **0–1** (“the cat”) from the previous segment. Process positions **2–6** (“sat on the soft mat”) now. For this first-layer toy, pretend the stored rows equal the listed token rows; real memory contains earlier hidden representations.

#### Step 1: Count the rows

The current segment has **5 query rows**. Together, memory and current text supply **7 key and value rows**, making a **5 × 7** score grid before masking. Following global position 4 gives access to keys 0–4, including the remembered “the cat.”

#### Step 2: Describe distance across the boundary

For the current “the” at global 4 and remembered “cat” at global 1, gap = 3. Use relative clock row **(0.141, -0.990, 0.030, 1.000)**. Their segment-local numbers need not be compared as if they were positions in one isolated segment.

#### Step 3: Combine four kinds of match

In this simplified attention calculation, the content projections and relative-position projection are identities. Use learned-bias examples **u = (0.1, 0, 0, 0)** and **b = (0, 0.1, 0, 0)**. The raw score is `q·k + q·R(gap) + u·k + b·R(gap)`. These are a content match, a content–distance match, a content preference, and a distance preference. Divide their sum by 2.

| Key | q·k | q·R | u·k | b·R | Scaled total |
|---|---|---|---|---|---|
| 0: the | 2.000 | -0.717 | 0.100 | -0.065 | 0.659 |
| 1: cat | 0.000 | 0.171 | 0.000 | -0.099 | 0.036 |
| 2: sat | 1.000 | 0.929 | 0.100 | -0.042 | 0.994 |
| 3: on | 1.000 | 0.851 | 0.000 | 0.054 | 0.953 |
| 4: the | 2.000 | 0.000 | 0.100 | 0.100 | 1.100 |

#### Step 4: Gather from memory and current values

Memory extends the set of keys and values; it does not create new output rows for the old segment.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>17.2%</span><i style="--weight: 17.152%"></i></div> <div><strong>cat <small>position 1</small></strong><span>9.2%</span><i style="--weight: 9.201%"></i></div> <div><strong>sat <small>position 2</small></strong><span>24.0%</span><i style="--weight: 23.976%"></i></div> <div><strong>on <small>position 3</small></strong><span>23.0%</span><i style="--weight: 23.011%"></i></div> <div><strong>the <small>position 4</small></strong><span>26.7%</span><i style="--weight: 26.661%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.322, 0.668, 0.332, 0.678)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating this for the five current queries gives five current output rows, each with four numbers. The two memory rows are consulted, not emitted again.

