### Worked examples: extending the same four-number head

For this section we switch from the 90°/10° teaching clocks to **standard d = 4 RoPE rates: 1 and 0.01 radians per position**. The sentence, word embeddings, and weight matrices remain the same. Assume a training window of **4,096 tokens** and a desired **4×** extension. The seven words let us inspect the changed arithmetic; they do not demonstrate long-context quality.

| Recipe | Fast pair · radians/step | Slow pair · radians/step | Position 4 · fast / slow angles |
|---|---|---|---|
| Unextended standard RoPE | 1.000 | 0.010000 | 4.000 / 0.040000 rad |
| Position Interpolation | 0.250 | 0.002500 | 1.000 / 0.010000 rad |
| NTK-aware | 1.000 | 0.002500 | 4.000 / 0.010000 rad |
| YaRN | 1.000 | 0.003835 | 4.000 / 0.015341 rad |
| Example LongRoPE candidate | 0.500 | 0.002500 | 2.000 / 0.010000 rad |

#### Position Interpolation

#### Step 0: Compress the position scale

Position m becomes m/4. Equivalently, divide both rotation rates by 4: **1 → 0.25** and **0.01 → 0.0025**. Position 4 takes the old position-1 angles.

#### Step 1: Turn the second “the” with the revised rates

Its pre-rotation query is still **(1, 0, 1, 0)**. The revised rotation gives **(0.540, 0.841, 1.000, 0.010)**. Each key receives its corresponding turn. The value rows stay as listed in the shared setup.

#### Step 2: Compute matches and mix values

Compare that query with the revised keys, divide raw dots by 2, then turn the scores into weights.

| Key | 0: the | 1: cat | 2: sat | 3: on | 4: the |
|---|---|---|---|---|---|
| Scaled score | 0.770 | 0.345 | 0.679 | 0.501 | 1.000 |
| Attention weight | 21.8% | 14.2% | 19.9% | 16.7% | 27.4% |

The second “the” gathers the original value rows into head output **(0.309, 0.659, 0.341, 0.691)**. Embeddings and values were not directly changed by the positional rule. The seven query rows still produce seven output rows.

#### NTK-aware

#### Step 0: Change the base rather than every position

For the usual base-change rule with d = 4 and scale 4, the new base is **10,000 × 4^(4/(4−2)) = 160,000**. Its rates are **1** and **1/sqrt(160,000) = 0.0025**. The fastest rate stays put; the slower one changes.

#### Step 1: Turn the second “the” with the revised rates

Its pre-rotation query is still **(1, 0, 1, 0)**. The revised rotation gives **(-0.654, -0.757, 1.000, 0.010)**. Each key receives its corresponding turn. The value rows stay as listed in the shared setup.

#### Step 2: Compute matches and mix values

Compare that query with the revised keys, divide raw dots by 2, then turn the scores into weights.

| Key | 0: the | 1: cat | 2: sat | 3: on | 4: the |
|---|---|---|---|---|---|
| Scaled score | 0.173 | 0.074 | 0.247 | 0.501 | 1.000 |
| Attention weight | 15.0% | 13.6% | 16.2% | 20.9% | 34.3% |

The second “the” gathers the original value rows into head output **(0.345, 0.702, 0.298, 0.655)**. Embeddings and values were not directly changed by the positional rule. The seven query rows still produce seven output rows.

#### YaRN

#### Step 0: Classify the two clocks by training-window rotations

The fast clock made about **651.9 turns** in 4,096 positions; the slow one made about **6.52**. Using YaRN’s conventional blend thresholds of 1 and 32 turns, the fast pair is preserved. The slow pair falls in the blend region, not the fully compressed region.

#### Step 1: Blend the slow rate and apply the attention-scale correction

The slow pair’s blend weight is **(6.52 − 1) / (32 − 1) ≈ 0.178**. Blend rate 0.01 with rate 0.0025 → **0.003835**. The conventional attention correction multiplies q and k by **1 + 0.1 ln(4) ≈ 1.139**, multiplying dot-product scores by about **1.296**. We include that correction below. A four-coordinate head has no other pairs; a wide head has many intermediate cases.

#### Step 2: Turn the second “the” with the revised rates

Its pre-rotation query is still **(1, 0, 1, 0)**. The revised rotation gives **(-0.654, -0.757, 1.000, 0.015)** before YaRN’s amplitude correction. Each key receives its corresponding turn. The value rows stay as listed in the shared setup.

#### Step 3: Compute matches and mix values

Compare that query with the revised keys, divide raw dots by 2 and apply the attention correction, then turn the scores into weights.

| Key | 0: the | 1: cat | 2: sat | 3: on | 4: the |
|---|---|---|---|---|---|
| Scaled score | 0.224 | 0.099 | 0.320 | 0.651 | 1.296 |
| Attention weight | 13.5% | 11.9% | 14.8% | 20.6% | 39.3% |

The second “the” gathers the original value rows into head output **(0.325, 0.733, 0.267, 0.675)**. Embeddings and values were not directly changed by the positional rule. The seven query rows still produce seven output rows.

#### Example LongRoPE candidate

#### Step 0: Inspect a possible set of searched factors

Imagine the search selected compression factors **2 for the fast pair** and **4 for the slow pair**. Rates become **0.5** and **0.0025**. These factors are invented to illustrate an uneven candidate; this is not a reproduction of LongRoPE’s search, position-dependent choices, or staged training.

#### Step 1: Turn the second “the” with the revised rates

Its pre-rotation query is still **(1, 0, 1, 0)**. The revised rotation gives **(-0.416, 0.909, 1.000, 0.010)**. Each key receives its corresponding turn. The value rows stay as listed in the shared setup.

#### Step 2: Compute matches and mix values

Compare that query with the revised keys, divide raw dots by 2, then turn the scores into weights.

| Key | 0: the | 1: cat | 2: sat | 3: on | 4: the |
|---|---|---|---|---|---|
| Scaled score | 0.292 | 0.502 | 0.691 | 0.501 | 1.000 |
| Attention weight | 14.3% | 17.7% | 21.3% | 17.6% | 29.1% |

The second “the” gathers the original value rows into head output **(0.353, 0.610, 0.390, 0.647)**. Embeddings and values were not directly changed by the positional rule. The seven query rows still produce seven output rows.

A **larger fixed base** uses the same mechanics: for d = 4, base 160,000 gives rates (1, 0.0025) and the NTK-aware rows above. The distinction is how the base is selected—for example as a training configuration versus an extension rule. Long-context adaptation and evaluation still matter.

