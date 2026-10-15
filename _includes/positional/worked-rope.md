### RoPE with d = 4 and seven words, fully worked

<figure class="pe-scene pe-work-flow pe-c">
<div>Look up x</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Make q, k, v</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Rotate q and k only</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Compare rotated rows</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>Weights mix unrotated values</div>
</figure>

#### Step 0: Locate the change in the pipeline

The input embedding row is **not directly modified by RoPE**. Neither is v. The positional turn happens to q and k just before their dot product. Later token representations still change, because the resulting attention weights gather different information.

#### Step 1: Follow the two copies of “the”

At positions **0** and **4**, their lookup rows are both **(1, 0, 1, 0)**. With our chosen first-layer weights, both also have q = k = **(1, 0, 1, 0)** and v = **(0, 1, 0, 1)** before rotation. In later layers, their different contexts can already give them different content rows.

#### Step 2: Split each matching row into two arrows

Cut **(1, 0, 1, 0)** into **pair A = (1, 0)** and **pair B = (1, 0)**. Each pair is an arrow in its own flat plane. Four coordinates make two pairs; 128 coordinates make 64 pairs.

<figure class="pe-scene pe-pair-split">
<strong>One query row: (1, 0, 1, 0)</strong>
<div><span class="pe-pair-fast">Pair A · coordinates 0–1<br>(1, 0) → fast arrow</span> <span class="pe-pair-slow">Pair B · coordinates 2–3<br>(1, 0) → slow arrow</span></div>
<figcaption>The arrows live in separate planes. They are not two arrows competing for one direction.</figcaption>
</figure>

#### Step 3: Give the arrows different speeds

Use **90° per position** for pair A and **10° per position** for pair B. These are deliberately simple teaching speeds. Standard d = 4 RoPE with base 10,000 uses **1 radian** and **0.01 radian** per position instead.

| Position | Word | Pair A angle | Pair B angle |
|---|---|---|---|
| 0 | the | 0° | 0° |
| 1 | cat | 90° | 10° |
| 2 | sat | 180° | 20° |
| 3 | on | 270° | 30° |
| 4 | the | 360° (same direction as 0°) | 40° |
| 5 | soft | 450° (same direction as 90°) | 50° |
| 6 | mat | 540° (same direction as 180°) | 60° |

#### Step 4: Turn an arrow, then rejoin the pairs

An arrow starting at **(1, 0)** becomes **(cos φ, sin φ)** after a turn of φ. First hold the starting content fixed and see what happens if this same “the” row occupies each possible position. This is a position-only demonstration, not a claim that every word has the same vector.

| Position | Fast pair A | Slow pair B | Rejoined “the” row |
|---|---|---|---|
| 0 | (1.000, 0.000) | (1.000, 0.000) | (1.000, 0.000, 1.000, 0.000) |
| 1 | (0.000, 1.000) | (0.985, 0.174) | (0.000, 1.000, 0.985, 0.174) |
| 2 | (-1.000, 0.000) | (0.940, 0.342) | (-1.000, 0.000, 0.940, 0.342) |
| 3 | (0.000, -1.000) | (0.866, 0.500) | (0.000, -1.000, 0.866, 0.500) |
| 4 | (1.000, 0.000) | (0.766, 0.643) | (1.000, 0.000, 0.766, 0.643) |
| 5 | (0.000, 1.000) | (0.643, 0.766) | (0.000, 1.000, 0.643, 0.766) |
| 6 | (-1.000, 0.000) | (0.500, 0.866) | (-1.000, 0.000, 0.500, 0.866) |

<figure class="pe-figure">
<img src="{{ '/assets/images/blog/positional-encodings/rope-seven-positions.svg' | relative_url }}" alt="The same two starting arrows at seven positions: the fast arrow returns to its starting direction at position four, while the slow arrow has moved forty degrees." width="980" height="340" loading="lazy">
<figcaption>At positions 0 and 4 the fast hand repeats, but the slow hand does not. Together the pairs distinguish these two positions.</figcaption>
</figure>

The actual “the” rows are now **(1, 0, 1, 0)** at 0 and **(1, 0, 0.766, 0.643)** at 4. The first pair alone repeats every four steps in this toy. The second pair separates those positions. Multiple speeds reduce this ambiguity over useful ranges; they do not promise an unlimited, unique distance code.

#### Step 5: Check equal gaps while keeping content fixed

Temporarily give the compared query and key the same starting vector **(1, 0, 1, 0)**. We are isolating position, so these are not the actual “on/cat” and “mat/the” content pairs from our sentence.

**Case 1: query at 3, key at 1, gap = 2.**

q = **(0.000, -1.000, 0.866, 0.500)**, k = **(0.000, 1.000, 0.985, 0.174)**.

Fast-pair contribution **-1.000** + slow-pair contribution **0.940** = raw dot product **-0.060**.

**Case 2: query at 6, key at 4, gap = 2.**

q = **(-1.000, 0.000, 0.500, 0.866)**, k = **(1.000, 0.000, 0.766, 0.643)**.

Fast-pair contribution **-1.000** + slow-pair contribution **0.940** = raw dot product **-0.060**.

Both raw dots are **−0.060** to three decimals. Scaled attention divides by 2, giving about **−0.030**, before masking and softmax. The raw dot and the scaled score are different stages of the pipeline.

#### Step 6: See the shortcut for identical starting arrows

Each pair contributes **cos(speed × gap)**, so the raw dot is **cos(90°×gap) + cos(10°×gap)**. This shortcut applies to our chosen identical unit arrows; general queries and keys also bring their starting content into the match.

| Gap | Fast pair | Slow pair | Raw dot |
|---|---|---|---|
| 0 | 1.000 | 1.000 | 2.000 |
| 1 | 0.000 | 0.985 | 0.985 |
| 2 | -1.000 | 0.940 | -0.060 |
| 3 | 0.000 | 0.866 | 0.866 |
| 4 | 1.000 | 0.766 | 1.766 |
| 5 | 0.000 | 0.643 | 0.643 |
| 6 | -1.000 | 0.500 | -0.500 |

#### Step 7: Change the content and observe direction

Keep q = **(1, 0, 1, 0)** but choose k = **(0, 1, 0, 1)**. The raw dot now follows **sin(90°×gap) + sin(10°×gap)**, with gap = query position minus key position.

| Gap | Raw dot for these different starting rows |
|---|---|
| 0 | 0.000 |
| 1 | 1.174 |
| 2 | 0.342 |
| -1 | -1.174 |

Now +1 gives **1.174**, while −1 gives **−1.174**. In this example the match distinguishes which side the key is on. A causal mask will still forbid future keys; these dots alone are not permission to attend to them.

#### Step 8: Return to the actual seven-word sentence

Now use each word’s own starting row from the shared table. The position-only shortcut no longer applies to every pair. Rotate each q and k using its actual position; keep v unchanged.

| Position | Word | Actual rotated q = rotated k | Unrotated v |
|---|---|---|---|
| 0 | the | (1.000, 0.000, 1.000, 0.000) | (0.000, 1.000, 0.000, 1.000) |
| 1 | cat | (-1.000, 0.000, -0.174, 0.985) | (1.000, 0.000, 1.000, 0.000) |
| 2 | sat | (-1.000, -1.000, 0.000, 0.000) | (0.000, 0.000, 1.000, 1.000) |
| 3 | on | (0.000, 0.000, 0.366, 1.366) | (1.000, 1.000, 0.000, 0.000) |
| 4 | the | (1.000, 0.000, 0.766, 0.643) | (0.000, 1.000, 0.000, 1.000) |
| 5 | soft | (0.000, 1.000, -0.766, 0.643) | (1.000, 0.000, 0.000, 1.000) |
| 6 | mat | (0.000, -1.000, 0.500, 0.866) | (0.000, 1.000, 1.000, 0.000) |

For the second “the,” q̂₄ = **(1, 0, 0.766, 0.643)**. Against “cat” at 1, k̂₁ = **(−1, 0, −0.174, 0.985)**. The raw dot is `−1 + (0.766×−0.174) + (0.643×0.985)` ≈ **−0.500**; scaled score **−0.250**. This is content and position working together.

| Key position | Word | Raw dot after rotation | Scaled score |
|---|---|---|---|
| 0 | the | 1.766 | 0.883 |
| 1 | cat | -0.500 | -0.250 |
| 2 | sat | -1.000 | -0.500 |
| 3 | on | 1.158 | 0.579 |
| 4 | the | 2.000 | 1.000 |

#### Step 9: Convert scores to weights and gather the original values

The two “the” embeddings stayed the same at lookup. Their rotated keys changed the comparisons, which changes what the current token gathers.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>29.1%</span><i style="--weight: 29.112%"></i></div> <div><strong>cat <small>position 1</small></strong><span>9.4%</span><i style="--weight: 9.376%"></i></div> <div><strong>sat <small>position 2</small></strong><span>7.3%</span><i style="--weight: 7.302%"></i></div> <div><strong>on <small>position 3</small></strong><span>21.5%</span><i style="--weight: 21.485%"></i></div> <div><strong>the <small>position 4</small></strong><span>32.7%</span><i style="--weight: 32.725%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.309, 0.833, 0.167, 0.691)**.

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.

**The picture to keep:** two stacked turntables, one fast and one slow. Each token places its content arrows
on them. Position turns the tables. Attention compares a query's arrows with a key's arrows; shifting both by
the same amount preserves their relative turns **when their starting content is held fixed**.

<details>
<summary>Optional: why does the absolute position cancel?</summary>
<p>Rotations preserve dot products. Turn q by mθ and k by nθ; undo the same mθ turn on both. You are left comparing q with k turned by (n−m)θ. The starting vectors still determine the content part of the match. In complex-number notation, the same cancellation comes from multiplying one rotated pair by the conjugate of the other.</p>
</details>

