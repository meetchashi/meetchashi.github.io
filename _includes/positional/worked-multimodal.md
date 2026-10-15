### Worked example: a patch in a video

#### Step 0: Give three coordinates three pairs

This time we need **six** query/key coordinates: one pair for time, one for row, one for column. Four numbers can only hold two pairs. Use starting row **(1, 0, 1, 0, 1, 0)** for two patches to isolate geometry. This illustrates axis assignment, not Qwen2-VL’s exact dimensional allocation or frequencies.

#### Step 1: Choose the coordinates

Patch A is at **frame 0, row 1, column 2**. Patch B is at **frame 0, row 0, column 2**. They are in the same frame and column, one row apart. Use rates **10° per frame, 90° per row, 10° per column**.

| Patch | Time pair | Row pair | Column pair | Rejoined row |
|---|---|---|---|---|
| A | (1, 0) | (0, 1) | (0.940, 0.342) | (1, 0, 0, 1, 0.940, 0.342) |
| B | (1, 0) | (1, 0) | (0.940, 0.342) | (1, 0, 1, 0, 0.940, 0.342) |

#### Step 2: Compare pair by pair

The time pair contributes **1**, the row pair contributes **0**, and the column pair contributes **1**. Raw dot = **2**. Divide by sqrt(6) → **0.816**. The query and key comparisons carry three kinds of place at once.

#### Step 3: Gather visual information

With only those two keys for illustration, A matching itself scores **3/sqrt(6) = 1.225**. A matching B scores **0.816**. Softmax gives about **60.1%** to A and **39.9%** to B. If their unrotated value rows are respectively **(1, 0)** and **(0, 1)**, A gathers **(0.601, 0.399)**. Value width can differ from query/key width; rotations do not need to change the values.

