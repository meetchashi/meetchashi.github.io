### Worked examples: changing which information carries position

#### Partial RoPE

#### Step 0: Rotate one pair, leave one pair alone

Use the seven-word setup. Rotate only coordinates 0–1 at **90° per position**. Coordinates 2–3 keep their content values. The second “the” at 4 becomes **(1, 0, 1, 0)** because its rotating pair completes a full turn; the first “the” at 0 is also **(1, 0, 1, 0)**. In this toy they collide. The unrotated content pair does not supply another clock to separate them.

#### Step 1: Compare partial rotations and gather values

Only the first pair’s contribution changes with position. The other pair provides a direct content match.

| Key | 0: the | 1: cat | 2: sat | 3: on | 4: the |
|---|---|---|---|---|---|
| Scaled score | 1.000 | -0.500 | -0.500 | 0.500 | 1.000 |
| Attention weight | 32.8% | 7.3% | 7.3% | 19.9% | 32.8% |

The second “the” gathers the original value rows into head output **(0.272, 0.854, 0.146, 0.728)**. Embeddings and values were not directly changed by the positional rule. The seven query rows still produce seven output rows.

#### Decoupled RoPE

#### Step 0: Use a content path and a separate positional path

Here is a tiny **architecture sketch**, rather than a complete DeepSeek model. Compress each four-number input to a two-number content latent **c = (x₀ + x₂, x₁ + x₃)**. Let content queries/keys copy c; let values copy c too. A separate projected pair for rotation copies **(x₀, x₁)**. Rotate that pair at 90° per position. The combined query/key width is four; value width is two.

#### Step 1: Add the two kinds of match

For the second “the,” content query is **(2, 0)** and its rotated positional query is **(1, 0)**. For cat, content key is **(0, 2)** and its rotated positional key is **(−1, 0)**. Their content dot is **0** and positional dot is **−1**: combined raw dot **−1**, scaled score **−0.5**.

| Key | Content dot | Positional dot | Combined / 2 | Weight |
|---|---|---|---|---|
| 0: the | 4.000 | 1.000 | 2.500 | 41.5% |
| 1: cat | 0.000 | -1.000 | -0.500 | 2.1% |
| 2: sat | 2.000 | -1.000 | 0.500 | 5.6% |
| 3: on | 2.000 | 0.000 | 1.000 | 9.3% |
| 4: the | 4.000 | 1.000 | 2.500 | 41.5% |

#### Step 2: Gather the content values

The output is the two-number content mixture **(1.810, 0.190)**. The memory idea is to cache a compressed content latent plus the positional key component, instead of all full content keys/values for every head. This toy does not implement the real multi-head projections or cache optimizations.

#### NoPE

#### Step 0: Skip explicit position additions and turns

Keep the word rows and q/k/v from the shared setup. Apply the causal mask. The second “the” sees keys 0–4 and has content scores **(1, 0, 0.5, 0.5, 1)**. Its two “the” key matches tie. The first “the,” at 0, can only see itself.

#### Step 1: Compare the two output rows

The first “the” gathers **100%** of its own value **(0, 1, 0, 1)**. The second “the” mixes five available values:

| Key | 0: the | 1: cat | 2: sat | 3: on | 4: the |
|---|---|---|---|---|---|
| Scaled score | 1.000 | 0.000 | 0.500 | 0.500 | 1.000 |
| Attention weight | 27.9% | 10.3% | 16.9% | 16.9% | 27.9% |

The second “the” gathers the original value rows into head output **(0.272, 0.728, 0.272, 0.728)**. Embeddings and values were not directly changed by the positional rule. The seven query rows still produce seven output rows.

Their contexts differ even without a position vector. The causal mask creates structure the model may learn to use; it does not guarantee a reliable numerical position counter.

#### CoPE

#### Step 0: Let the query decide how much each token counts

Use the second “the” and its base scaled scores **(1, 0, 0.5, 0.5, 1)**. Following the paper’s implementation, sigmoid converts these into gates: **(0.731, 0.500, 0.622, 0.622, 0.731)**. Gates are fractional and query-dependent, not hand-assigned word-category labels.

#### Step 1: Count gated contributions back to each key

For each allowed key, add the gates from that key through the query, **including both endpoints**. For “on” at 3, the contextual count is **0.622 + 0.731 = 1.354**, rather than the raw token gap 1. For the first “the” it is about **3.207**, rather than raw gap 4.

#### Step 2: Interpolate a position contribution

Choose a toy learned position-vector table **e[p] = (b[p], 0, 0, 0)**, with **b[0…4] = (0, 0.3, 0.1, −0.2, −0.4)**. Since q = (1, 0, 1, 0), q·e[p] is simply b[p]. At contextual position 1.354, interpolate **64.6% of b[1] + 35.4% of b[2] ≈ 0.229**. Add that contribution to the base content score. This follows the score-level interpolation used in the paper’s example implementation.

| Key | Gate for this token | Inclusive contextual count | Position contribution | New score |
|---|---|---|---|---|
| 0: the | 0.731 | 3.207 | -0.241 | 0.759 |
| 1: cat | 0.500 | 2.476 | -0.043 | -0.043 |
| 2: sat | 0.622 | 1.976 | 0.105 | 0.605 |
| 3: on | 0.622 | 1.354 | 0.229 | 0.729 |
| 4: the | 0.731 | 0.731 | 0.219 | 1.219 |

#### Step 3: Gather unchanged content values

The new count controls a learned relative-position contribution. Initial word embeddings still have four numbers; we have changed the way relationships are measured.

| Key | 0: the | 1: cat | 2: sat | 3: on | 4: the |
|---|---|---|---|---|---|
| Scaled score | 0.759 | -0.043 | 0.605 | 0.729 | 1.219 |
| Attention weight | 20.6% | 9.2% | 17.6% | 20.0% | 32.6% |

The second “the” gathers the original value rows into head output **(0.292, 0.731, 0.269, 0.708)**. Embeddings and values were not directly changed by the positional rule. The seven query rows still produce seven output rows.

