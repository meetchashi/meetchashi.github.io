### Worked example: sinusoidal positions

<figure class="pe-scene pe-work-flow pe-a">
<div>Look up x</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Generate clock row PE(m)</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Add: h = x + PE(m)</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Make q, k, v from h</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Compare and gather</div>
</figure>

#### Step 0: Make two clock pairs

For four dimensions and base 10,000, the row is **(sin m, cos m, sin(0.01m), cos(0.01m))**. Angles here are in radians. The first pair changes quickly; the second changes slowly.

| Position | Word | Clock row | Input after adding |
|---|---|---|---|
| 0 | the | (0.000, 1.000, 0.000, 1.000) | (1.000, 1.000, 1.000, 1.000) |
| 1 | cat | (0.841, 0.540, 0.010, 1.000) | (0.841, 1.540, 0.010, 2.000) |
| 2 | sat | (0.909, -0.416, 0.020, 1.000) | (1.909, 0.584, 0.020, 1.000) |
| 3 | on | (0.141, -0.990, 0.030, 1.000) | (0.141, -0.990, 1.030, 2.000) |
| 4 | the | (-0.757, -0.654, 0.040, 0.999) | (0.243, -0.654, 1.040, 0.999) |
| 5 | soft | (-0.959, 0.284, 0.050, 0.999) | (0.041, 0.284, 0.050, 1.999) |
| 6 | mat | (-0.279, 0.960, 0.060, 0.998) | (-0.279, 1.960, 1.060, 0.998) |

#### Step 1: Follow the second “the”

At position 4, its original row **(1.000, 0.000, 1.000, 0.000)** receives clock row **(-0.757, -0.654, 0.040, 0.999)**, producing **(0.243, -0.654, 1.040, 0.999)**. The first “the” instead receives **(0.000, 1.000, 0.000, 1.000)**. Same word, different clock reading.

#### Step 2: Make q, k, and v after the addition

q₄ = k₄ = **(0.243, -0.654, 1.040, 0.999)**; v₄ = **(0.999, 1.040, -0.654, 0.243)**. This is the same place in the pipeline as learned absolute positions. Only the source of the added position row changed.

#### Step 3: Multiply corresponding coordinates and add

For the first “the,” k₀ = **(1.000, 1.000, 1.000, 1.000)**. The raw match is `(0.243×1) + (-0.654×1) + (1.040×1) + (0.999×1)` ≈ **1.629**. Divide by 2 → **0.814**.

| Key position | Word | Raw match | Scaled score |
|---|---|---|---|
| 0 | the | 1.629 | 0.814 |
| 1 | cat | 1.207 | 0.603 |
| 2 | sat | 1.103 | 0.551 |
| 3 | on | 3.751 | 1.875 |
| 4 | the | 2.566 | 1.283 |

#### Step 4: Gather the changed value rows

These clock patterns influence both the matches and the values, because both were made after the input addition.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>14.2%</span><i style="--weight: 14.153%"></i></div> <div><strong>cat <small>position 1</small></strong><span>11.5%</span><i style="--weight: 11.460%"></i></div> <div><strong>sat <small>position 2</small></strong><span>10.9%</span><i style="--weight: 10.879%"></i></div> <div><strong>on <small>position 3</small></strong><span>40.9%</span><i style="--weight: 40.889%"></i></div> <div><strong>the <small>position 4</small></strong><span>22.6%</span><i style="--weight: 22.619%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(1.523, 0.801, -0.171, 0.558)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.

