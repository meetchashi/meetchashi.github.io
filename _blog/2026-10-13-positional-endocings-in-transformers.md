---
layout: blog_post
title: "How Transformers Learn Where Words Are: A Visual Guide to Positional Encodings"
date: 2026-10-13 00:00:00 +0000
tags: [transformers, positional-encoding, LLMs, tutorial]
visual_guide: true
---

Imagine handing someone three cards: **dog**, **bites**, and **man**.

You ask what happened. They cannot tell you. Did the dog bite the man, or did the man bite the dog?
The words are all there. What is missing is their order.

<figure class="pe-scene">
<div class="pe-sentence"><span class="pe-token pe-dog">dog</span> <span class="pe-token">bites</span> <span class="pe-token pe-man">man</span> <span class="pe-reading">An unfortunate afternoon.</span></div>
<div class="pe-sentence"><span class="pe-token pe-man">man</span> <span class="pe-token">bites</span> <span class="pe-token pe-dog">dog</span> <span class="pe-reading">A rather different story.</span></div>
<figcaption>Same words. Different order. Different meaning.</figcaption>
</figure>

We see that order immediately because we read from left to right. A Transformer needs a way to use it too.
Its attention mechanism is excellent at comparing words and gathering relevant information. But the basic,
unrestricted attention calculation does not come with a built-in idea of “first,” “before,” or “three places back.”

That is the problem **positional encodings** solve. They give the model information about where things are.

The interesting part is how that idea developed. The first approach sounds obvious: number the words.
Then comes a better question: do we need the number, or do we need the distance between two words?
Then comes a surprisingly elegant answer involving rotating arrows.

Let's build those ideas one at a time. Each method now has a worked walkthrough: the same sentence, the same
small embeddings, and the point where position changes the calculation. RoPE gets a longer two-clock example.
You can follow the numbers without running code. Read the intuition first; the numbered steps make it concrete.

<nav class="pe-route" aria-label="Post reading guide">
<a href="#start-with-attention-what-should-this-word-look-at">Attention</a>
<a href="#our-shared-example-seven-words-four-numbers-per-word">Shared seven-word example</a>
<a href="#1-give-each-word-a-seat-number">Seat numbers</a>
<a href="#2-use-several-clocks-to-describe-a-position">Clocks</a>
<a href="#3-describe-how-far-apart-the-words-are">Distances</a>
<a href="#4-let-distance-nudge-attention-alibi">ALiBi</a>
<a href="#5-turn-position-into-a-rotation-rope">RoPE</a>
<a href="#6-what-happens-when-the-document-gets-longer">Longer documents</a>
</nav>

## Start with attention: what should this word look at?

Consider this sentence:

> The animal didn't cross the street because it was tired.

To make sense of **it**, we look back at **animal**. The tired thing is probably the animal, not the street.
Attention gives the model a way to make connections like this. Each word compares itself with other words,
then gathers more information from the ones that seem useful.

Here is a picture of what we would like it to learn. The connections are illustrative, not measurements from a model.

<figure class="pe-scene">
<div class="pe-connections">
<div class="pe-focus"><span>it</span> <small>Who are we talking about?</small></div>
<div class="pe-candidates"><div class="pe-match pe-strong"><span>animal</span> <small>A promising connection</small></div><div class="pe-match pe-weak"><span>street</span> <small>Less useful here</small></div></div>
</div>
<figcaption>Attention helps a word gather context. Different attention heads can learn different kinds of connections.</figcaption>
</figure>

But relevance is not just about meaning. In **the red ball**, “red” describes the word next to it.
In a long paragraph, a nearby noun might be a better candidate than another noun several sentences away.
Sometimes we need to look far back. Sometimes the word immediately before us matters most.

So the model needs both: **what a word means** and **where it sits relative to other words**.

A small vocabulary note: models actually work with *tokens*, which can be words, pieces of words, or punctuation.
We'll use whole words in the pictures so we can concentrate on the idea.

### Our shared example: seven words, four numbers per word

We will use the same sentence for every walkthrough. The numbers are deliberately small, invented teaching
values—not a trained model's embeddings. Each word is one token in this example, and positions start at **0**.

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Word | the | cat | sat | on | the | soft | mat |


Notice the two copies of **the**. At the initial lookup, the same token gets the same embedding. Let's see exactly
where each design makes their journeys different.

| Position | Word | Embedding x · four numbers |
|---|---|---|
| 0 | the | (1.000, 0.000, 1.000, 0.000) |
| 1 | cat | (0.000, 1.000, 0.000, 1.000) |
| 2 | sat | (1.000, 1.000, 0.000, 0.000) |
| 3 | on | (0.000, 0.000, 1.000, 1.000) |
| 4 | the | (1.000, 0.000, 1.000, 0.000) |
| 5 | soft | (1.000, 0.000, 0.000, 1.000) |
| 6 | mat | (0.000, 1.000, 1.000, 0.000) |


Seven tokens give us a **7 × 4 embedding matrix**. More generally, n tokens with d numbers each give **n × d**.
The columns are learned coordinates, not named properties such as “animal” or “action.”

### A small head we can follow without hiding the numbers

Real models learn weight matrices that make queries, keys, and values from each row. For this toy head we choose
especially simple matrices: **queries and keys copy the input row; values reverse its four coordinates**.
This is a legitimate choice of weights, but a deliberately simple one—not how trained weights usually look.
We write row vectors, so the convention is `q = x W_Q`, `k = x W_K`, `v = x W_V`.

<figure class="pe-scene">
<span class="pe-eyebrow">Our chosen weight transformations · the same in every main walkthrough</span>
<div class="pe-vector-pair">
<div class="pe-vector-panel"><strong>W_Q and W_K · copy the row</strong><div class="pe-number-matrix"><span>1 0 0 0</span> <span>0 1 0 0</span> <span>0 0 1 0</span> <span>0 0 0 1</span></div></div>
<div class="pe-vector-panel"><strong>W_V · reverse the row</strong><div class="pe-number-matrix"><span>0 0 0 1</span> <span>0 0 1 0</span> <span>0 1 0 0</span> <span>1 0 0 0</span></div></div>
</div>
<figcaption>For “the”: x = (1, 0, 1, 0), q = k = (1, 0, 1, 0), and v = (0, 1, 0, 1). Other layers and heads would normally use different learned weights.</figcaption>
</figure>

| Word | q = k before position | v before position |
|---|---|---|
| the | (1.000, 0.000, 1.000, 0.000) | (0.000, 1.000, 0.000, 1.000) |
| cat | (0.000, 1.000, 0.000, 1.000) | (1.000, 0.000, 1.000, 0.000) |
| sat | (1.000, 1.000, 0.000, 0.000) | (0.000, 0.000, 1.000, 1.000) |
| on | (0.000, 0.000, 1.000, 1.000) | (1.000, 1.000, 0.000, 0.000) |
| soft | (1.000, 0.000, 0.000, 1.000) | (1.000, 0.000, 0.000, 1.000) |
| mat | (0.000, 1.000, 1.000, 0.000) | (0.000, 1.000, 1.000, 0.000) |


We will follow the **second “the,” at position 4**, when it looks backward. It may use positions 0 through 4;
**soft** and **mat** are hidden because they are in the future. We apply this same causal mask to compare the designs,
even though some of the original models also use attention that looks both ways.

To compare a query with a key, multiply corresponding coordinates and add. For example, “the” looking at “sat” gives
`1×1 + 0×1 + 1×0 + 0×0 = 1`. With four query/key coordinates, ordinary scaled attention divides this by
`sqrt(4) = 2`, giving a score of **0.5**. DeBERTa's three-term example below uses a different scale.

For n tokens, one head builds **n query rows, n key rows, and n value rows**. Comparing all queries with all keys
makes an **n × n score grid**. We will display one row of that grid so we can see the arithmetic clearly.

The scores become **attention weights** through softmax: exponentiate the allowed scores, then divide each by
their sum. Those weights mix the value rows, giving one updated row for each input token.

<figure class="pe-scene pe-work-flow pe-a">
<div>7 embedding rows</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>7 q / k / v rows</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>7 × 7 scores + mask</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>weights mix values</div> <span class="pe-work-arrow" aria-hidden="true">→</span> <div>7 updated rows</div>
</figure>


For reference, without any explicit position signal, the second “the” has the score row
**(1.000, 0.000, 0.500, 0.500, 1.000)** for keys at positions 0–4. Both “the” keys tie.
The causal mask itself provides some order structure, but there is no explicit distance signal in these matches.

<figure class="pe-scene"><span class="pe-eyebrow">What the second “the” gathers</span><div class="pe-weight-strip"><div><strong>the <small>position 0</small></strong><span>27.9%</span><i style="--weight: 27.926%"></i></div> <div><strong>cat <small>position 1</small></strong><span>10.3%</span><i style="--weight: 10.273%"></i></div> <div><strong>sat <small>position 2</small></strong><span>16.9%</span><i style="--weight: 16.938%"></i></div> <div><strong>on <small>position 3</small></strong><span>16.9%</span><i style="--weight: 16.938%"></i></div> <div><strong>the <small>position 4</small></strong><span>27.9%</span><i style="--weight: 27.926%"></i></div> </div><figcaption>These five weights add to 100% before rounding. “Soft” and “mat” are hidden by the causal mask.</figcaption></figure>

The head output for this token is the weighted mixture of the five value rows: **(0.272, 0.728, 0.272, 0.728)**. 

This is one head’s output **before** an output projection, residual addition, normalization, or feed-forward block. Repeating the calculation for every query gives seven output rows, each with four numbers.


Now we can change one ingredient at a time. In the panels below, **blue changes the input**, **amber changes
relationships or scores**, and **purple changes query/key rotations**. All calculations use full precision and are
displayed rounded to three decimal places.

## 1. Give each word a seat number

Suppose you wanted to restore the order of our three cards. The simplest thing you could do is write a number on each one.

<figure class="pe-scene">
<div class="pe-sentence"><span class="pe-token pe-dog">dog <small>seat 1</small></span> <span class="pe-token">bites <small>seat 2</small></span> <span class="pe-token pe-man">man <small>seat 3</small></span></div>
<div class="pe-sentence"><span class="pe-token pe-man">man <small>seat 1</small></span> <span class="pe-token">bites <small>seat 2</small></span> <span class="pe-token pe-dog">dog <small>seat 3</small></span></div>
<figcaption>“Dog in seat 1” and “dog in seat 3” now carry different information.</figcaption>
</figure>

A Transformer does something similar, with one difference. It represents a word as a list of numbers,
called an **embedding**. Instead of attaching a printed seat number, it adds another list of numbers describing
that position. The model receives a mixture of the word information and the position information.

**Learned absolute positions** let training decide what each position's list should be. Picture a drawer of seat
labels: a label for the first token, another for the second, and so on. Models such as BERT and GPT-2 used this kind
of position table. “Absolute” just means that a label describes a particular place in the sequence.
[The BERT paper](https://arxiv.org/abs/1810.04805) describes its input embeddings, including position embeddings.

{% include positional/worked-learned.md %}

This gives the model a useful starting point. The word has an identity and a place.

But imagine that our drawer contains 512 labels. What happens when a document needs a 513th?
There is no label waiting for it. Extending the table is possible, but the model has not automatically learned how
to use those extra positions.

There is another question hiding here. “Red” can describe “ball” near the start of a sentence or near the end.
Their seat numbers change, but **being next to each other** stays the same. Could our position system make that
relationship easier to recognize?

Before we get there, let's look at a clever alternative to a drawer of learned labels.

## 2. Use several clocks to describe a position

Think about the hands on a clock. The second hand moves quickly. The minute hand moves more slowly.
The hour hand moves more slowly still. Together, they describe the time more fully than any one hand could.

Now imagine a collection of clock hands that advances every time we move one token along a sentence.
One hand takes a large step. Another takes a small step. Another barely moves.
A position is described by **the collection of angles**.

<figure class="pe-figure">
<img src="{{ '/assets/images/blog/positional-encodings/sinusoidal-clocks.svg' | relative_url }}" alt="Moving from the first position to a later position turns a fast clock hand substantially, while a slow clock hand moves only a little." loading="lazy" width="900" height="300">
<figcaption>The fast hand helps distinguish nearby positions. The slow hand changes more gradually over a longer stretch. Neither hand has to do the whole job alone.</figcaption>
</figure>

Why more than one clock? A single hand eventually comes around again. Looking at the minute hand alone,
you cannot tell which hour it is. Having hands that turn at different speeds gives the model a richer pattern
for each position.

The original 2017 Transformer used sine and cosine functions to make these patterns. That is where the name
**sinusoidal positional encoding** comes from. You can think of those functions as a way to write down a clock
hand's position using two numbers. The resulting numbers are added to the word embedding, just like the seat labels.
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduced this design alongside a learned-position alternative.

{% include positional/worked-sinusoidal.md %}

The pleasant thing is that a formula can keep producing patterns beyond the positions in a fixed table.
We do not run out of labels in the same way.

Still, being able to *produce* a label is different from the model knowing what to *do* with it.
A model trained on short passages can struggle with the patterns and relationships in much longer ones.
And we are still handing it a description of each individual position, then asking it to work out the relationship
between two positions.

Let's make that relationship explicit.

## 3. Describe how far apart the words are

Read these two phrases:

- **the red ball**
- **after a surprisingly long afternoon, the red ball**

“Red” moved. “Ball” moved. But “red” is still immediately before “ball.”
That feels like the useful fact, doesn't it?

<figure class="pe-scene">
<div class="pe-distance-example"><span class="pe-prefix">the</span> <span class="pe-token pe-red">red <small>position 2</small></span> <span class="pe-gap">1 step →</span> <span class="pe-token pe-blue-token">ball <small>position 3</small></span></div>
<div class="pe-distance-example"><span class="pe-prefix">… the</span> <span class="pe-token pe-red">red <small>position 9</small></span> <span class="pe-gap">1 step →</span> <span class="pe-token pe-blue-token">ball <small>position 10</small></span></div>
<figcaption>Absolute positions change. The relationship between the words stays the same. Position numbers here are illustrative.</figcaption>
</figure>

This is the idea behind **relative positions**. Instead of telling each word only “you are in seat 9,”
we give its interactions information such as “this word is one step ahead” or “that word is three steps behind.”

Notice where the information is useful. When “red” considers “ball,” attention is already asking whether they belong
together. We can bring their relative position into that very comparison.

### Shaw: learn what “one step away” means

In 2018, Shaw and colleagues introduced learned representations for relative offsets.
Think of little relationship labels: **one before**, **two before**, **one after**.
The same label can be used anywhere in a sentence. The model can learn what to do with it rather than learning
separate versions of the relationship at every pair of seat numbers.

To keep the collection manageable, distant offsets share labels beyond a chosen cutoff. That saves space,
but it also means some far-apart relationships become indistinguishable to the position system.
Their method brings relative information into both the matching and the information gathered.
[Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155).

{% include positional/worked-shaw.md %}

### Transformer-XL: keep the relationship when you turn the page

Now imagine reading a book one page at a time. You want the next page to use what you learned on the previous one.
But if both pages start their numbering at 1, the old seat numbers become confusing.

Transformer-XL reused representations from earlier text segments. Relative positions helped it describe how far
those remembered tokens were from the current tokens, across segment boundaries. The important development was
that position had to work with a continuing memory, not just within one isolated passage.
[Transformer-XL](https://arxiv.org/abs/1901.02860), 2019.

{% include positional/worked-xl.md %}

### T5: we may not need a separate label for every distance

When giving directions, we often say “next door,” “a few houses away,” or “across town.”
We are precise nearby and less precise far away.

T5 takes a similar approach. Nearby distances get fine distinctions. Larger distances are grouped into wider
**buckets**. For each bucket, an attention head learns a number that nudges a connection's score up or down.

<figure class="pe-scene">
<span class="pe-eyebrow">T5 · a rough map of distance buckets</span>
<div class="pe-buckets"><div>1 <small>its own bucket</small></div><div>2 <small>its own bucket</small></div><div>3–4 <small>a shared bucket</small></div><div>5–8 <small>a wider bucket</small></div><div>9–16 <small>wider still</small></div></div>
<figcaption>Illustrative ranges, not T5's exact bucket boundaries. The principle is fine detail nearby, broader groups farther away.</figcaption>
</figure>

Different heads can learn different distance preferences: one might often look nearby, while another pays more
attention farther back. These are preferences, not rules about which words must connect.
[T5](https://arxiv.org/abs/1910.10683), first released in 2019.

{% include positional/worked-t5.md %}

### DeBERTa: keep meaning and position separate while comparing

Earlier, we mixed the seat label into the word embedding. DeBERTa asks whether we can keep the two descriptions
separate for longer.

Imagine two cards for each token: one describes its content; the other helps describe relative position.
Attention can then consider content-to-content matches and how content relates to position, instead of starting
from one blended description. That separation is the intuition behind *disentangled attention*.
[DeBERTa](https://arxiv.org/abs/2006.03654), 2020.

{% include positional/worked-deberta.md %}

These developments are related ideas, not a ladder where every new method makes the previous one obsolete.
Now let's try a distance preference that does not need to be learned at all.

## 4. Let distance nudge attention: ALiBi

Suppose you are reading a paragraph and encounter “it.” Usually, you start looking for its referent nearby.
You can look farther back, but you need a good reason.

**ALiBi** gives attention that sort of preference. A token farther in the past pays a larger distance penalty
when its attention score is calculated. If two candidates have equally good content matches, the closer one wins.

<figure class="pe-scene">
<span class="pe-eyebrow">ALiBi · equal content matches</span>
<div class="pe-costs"><div class="pe-cost"><span>1 step back</span><div class="pe-cost-track"><i style="width: 20%"></i></div> <small>A small distance cost</small></div><div class="pe-cost"><span>3 steps back</span><div class="pe-cost-track"><i style="width: 60%"></i></div> <small>A larger distance cost</small></div><div class="pe-cost"><span>5 steps back</span><div class="pe-cost-track"><i style="width: 100%"></i></div> <small>A larger cost again</small></div></div>
<figcaption>A longer bar means a larger penalty, not a larger attention weight. A strong content match can overcome the penalty.</figcaption>
</figure>

Different heads pay different rates. Some have a strong preference for nearby tokens; others have a gentler
penalty and can look farther back more easily. These rates are fixed rather than learned.

Why is this useful for longer text? “Charge a little more for each extra step” still makes sense at distances
we did not see during training. There is no new seat label to invent. ALiBi demonstrated better length
extrapolation in its experiments. The tradeoff is that a useful detail buried far back also has to overcome
that distance cost. [Train Short, Test Long](https://arxiv.org/abs/2108.12409), 2021.

{% include positional/worked-alibi.md %}

So far we have given tokens labels and given connections distance preferences.
Next, we change the way the comparison is represented.

## 5. Turn position into a rotation: RoPE

Hold out two pencils so they point in the same direction. Now turn one a little.
They are less aligned. Turn it farther, and their alignment changes again.

Now keep the angle between them fixed and turn **both** together. They still have the same alignment.
You have changed the direction each pencil points, but not their relationship.

That simple observation is the heart of **Rotary Position Embedding**, or **RoPE**.

To connect it to attention, we need just two names. A **query** describes what a token is looking for.
A **key** describes what another token offers for matching. Both are lists of numbers learned by the model.
Attention compares them to decide how much information to gather from that token.

Imagine a tiny part of each query and key as an arrow on a flat sheet of paper.
RoPE turns those arrows by an amount determined by their token positions **before comparing them**.
The comparison can then use both the original content and the relative turn.
[RoFormer](https://arxiv.org/abs/2104.09864), 2021.

{% include positional/worked-rope.md %}

### Why many arrows, rather than just two?

Remember the fast and slow clocks? RoPE uses that same idea of several rotation speeds.
Different pairs of numbers in a query or key turn at different rates. Fast turns give detailed changes over nearby
positions. Slower turns give a more gradual signal over longer distances.

There is an important difference from the earlier sinusoidal method. There, clock readings were **added to the
word's representation**. Here, they control **how the query and key are turned before a match**.

<div class="pe-branches pe-two">
<div class="pe-branch pe-a"><span class="pe-eyebrow">Sinusoidal</span><h3>Add a clock reading</h3><p>The word starts its journey carrying meaning mixed with a position pattern.</p></div>
<div class="pe-branch pe-c"><span class="pe-eyebrow">RoPE</span><h3>Turn the matching arrows</h3><p>Position changes the comparison used to decide where attention should look.</p></div>
</div>

The information being gathered, called the **value**, is not rotated by standard RoPE. A useful way to picture it:
we adjust the way we choose a book from the shelf, rather than turning all the writing inside the book.

One thing to be careful about: RoPE is not a rule that says “nearby words always match better.”
Our equal-gap demonstration fixes the starting arrows to make the geometry visible. Real queries and keys differ with content,
and many rotating pairs contribute together. The model learns how to use that positional structure.

## 6. What happens when the document gets longer?

We have a method that describes relative relationships. Surely we can now read a document of any length?

There is a catch. The model learned from documents of certain lengths. A much longer document produces
combinations of clock angles and distances that may be unfamiliar, especially for slowly turning pairs.
Having a position system that accepts a large number does not guarantee that the model will use it well.

Imagine practising with a ruler marked from 0 to 4, then being handed one that runs to 16.
You can read the numbers, but the patterns you learned on the short ruler may not transfer smoothly.

### Position Interpolation: squeeze the longer ruler into the familiar span

One approach is to compress the position scale. To fit four times as many tokens into the old range,
we treat position 12,000 like position 3,000 for the rotations.

<figure class="pe-scene">
<span class="pe-eyebrow">Position Interpolation · four times the text</span>
<div class="pe-ruler-row"><strong>Familiar range</strong><div class="pe-ruler pe-ruler-original"><span>0</span> <span>1</span> <span>2</span> <span>3</span> <span>4</span></div></div>
<div class="pe-ruler-row"><strong>Longer text, compressed</strong><div class="pe-ruler pe-ruler-compressed"><span>0</span> <span>4</span> <span>8</span> <span>12</span> <span>16</span></div></div>
<figcaption>Both rulers occupy the same span. The second fits more positions inside it, so neighbouring positions are packed closer together.</figcaption>
</figure>

This helps keep the rotation range familiar. The cost is that nearby positions also become more similar:
we squeezed the small gaps as well as the large ones. Models can be fine-tuned to adapt to the new scale.
[Position Interpolation](https://arxiv.org/abs/2306.15595), 2023.

### NTK-aware scaling and YaRN: don't slow every clock equally

If the fast clocks help distinguish neighbours, perhaps we should preserve more of their detail while slowing
the clocks that describe longer stretches.

**NTK-aware scaling** changes the spacing of rotation speeds, stretching the slower ones more while leaving the
fastest unchanged. **YaRN** makes a more explicit distinction: keep fast rotations, slow down slow rotations,
and blend the treatment in between. It also adjusts the attention score scale.

<div class="pe-branches">
<div class="pe-branch pe-c"><span class="pe-eyebrow">Position Interpolation</span><h3>Slow every clock</h3><p>Simple compression, including the clocks that distinguish neighbours.</p></div>
<div class="pe-branch pe-c"><span class="pe-eyebrow">NTK-aware</span><h3>Stretch speeds unevenly</h3><p>Preserve the fastest rotation while slowing longer-range ones more.</p></div>
<div class="pe-branch pe-c"><span class="pe-eyebrow">YaRN</span><h3>Treat fast and slow differently</h3><p>Keep fast clocks, slow slow clocks, and blend the middle.</p></div>
</div>

The aim is to gain room for long documents without losing so much detail between neighbouring tokens.
[YaRN](https://arxiv.org/abs/2309.00071), 2023, develops this frequency-selective approach.

Other approaches, such as **LongRoPE**, search for different stretching factors across the rotating pairs.
[LongRoPE](https://arxiv.org/abs/2402.13753) develops that search-based approach.
A larger RoPE base also makes the slower clocks turn more slowly. These are ways of reshaping the position scale,
not promises that every distant detail will be retrieved correctly.

**Where does this change our rows?** All three keep the same token embeddings and the same query/key/value
pipeline as RoPE. They change the **schedule of turns applied to the query and key pairs**. Interpolation compresses
positions uniformly; NTK-aware scaling and YaRN treat rotation speeds differently. The word rows do not get longer,
and no extra word rows are created.

{% include positional/worked-extensions.md %}

Position is one piece of long-context ability. The model also needs training, enough computation and memory,
and attention patterns that can find useful information in a sea of text.

## 7. A picture or a video needs more than “before” and “after”

A sentence is arranged along a line. An image has rows and columns. A video adds time.
Two image patches may be next to each other horizontally, vertically, or neither. A single place in a flattened
list does not describe those relationships very naturally.

**Multimodal RoPE** assigns different groups of rotating pairs to different coordinates.
Some describe horizontal position, some vertical position, and some time.
[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) uses this idea to handle text, images, and video.

<figure class="pe-scene">
<div class="pe-coordinates"><div class="pe-patch-grid" role="img" aria-label="A three by three image grid with the middle-right patch highlighted"><span></span> <span></span> <span></span> <span></span> <span></span> <span class="pe-selected"></span> <span></span> <span></span> <span></span></div><div class="pe-coordinate-labels"><span class="pe-axis-x">Where across? → column</span> <span class="pe-axis-y">Where down? ↓ row</span> <span class="pe-axis-t">When? → video frame</span></div></div>
<figcaption>The highlighted patch has a place across the image and down the image. In a video, it also belongs to a particular moment.</figcaption>
</figure>

**Where does this change our rows?** An image is split into patches that become visual token representations.
Each patch has a row of numbers, just as a text token does. Multimodal RoPE uses different groups of query/key
coordinates for horizontal, vertical, and time rotations. Those groups live **within the rows**; a patch does not
need a separate token row for each coordinate.

{% include positional/worked-multimodal.md %}

The question is still the same one we started with: what information about place will help two pieces of content
relate to each other? The answer now needs more than one direction.

## 8. Other questions researchers are asking

Once the main idea makes sense, a few more developments are easier to place.

**Does every part of the representation need a rotation?** Partial RoPE rotates only some of the query and key
coordinates. The others retain a way to compare content without an explicit positional turn.

**Can we keep a small position component separate?** DeepSeek's multi-head latent attention compresses stored
attention information to save memory. Its decoupled RoPE design uses a separate component for the positional
comparison so that rotation can coexist with that compression.
[DeepSeek-V2](https://arxiv.org/abs/2405.04434).

**Do all layers need explicit position signals?** A model that can look only backward already has an asymmetry:
the first token sees no earlier tokens, while later tokens can see a growing history. This can provide clues about
position, though it does not automatically give every token a reliable counter. Research on **NoPE** explores
what models can learn without explicit positional encodings.
[The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466).

**Are token steps always the right unit?** “Two sentences ago” may be more useful than “37 tokens ago.”
**Contextual Position Encoding**, or CoPE, learns when to advance a position count based on content, rather than
always counting every token equally.
[Contextual Position Encoding](https://arxiv.org/abs/2405.18719).

| Design | What changes in the row journey? |
|---|---|
| Partial RoPE | Rotate only some query/key coordinates; leave the other coordinates unrotated. |
| Decoupled RoPE | Use a separate query/key component for rotation alongside the compressed content path. |
| NoPE | Add no explicit position rows, score bias, or rotation; order clues can come from the causal attention mask. |
| CoPE | Learn content-dependent positional counts and use them in the attention comparison. |

{% include positional/worked-hybrids.md %}

These approaches address different needs. We should not read the history as one method winning every contest.
A system built for short passages, a system built for video, and a system built for long document retrieval may
need different positional choices.

## Put the story together

Go back to our seven-word sentence. We started with content and needed order.
Here are the main answers we explored:

| Idea | Picture to remember | What it gives the model |
|---|---|---|
| Learned absolute positions | A drawer of seat labels | A learned description of each place |
| Sinusoidal positions | Several clock hands | A position pattern generated by a formula |
| Relative representations | “One step before me” | A relationship reused throughout the text |
| T5 distance buckets | “Next door” versus “across town” | Learned preferences for groups of distances |
| ALiBi | A cost for looking farther back | A fixed preference that weakens with distance |
| RoPE | Two arrows turning before a comparison | Content matching that incorporates relative turns |
| RoPE extension methods | Stretching a ruler, or slowing clocks | A position scale adapted to longer text |
| Multimodal RoPE | Across, down, and through time | Relationships in more than one direction |

The move from seat labels to relative relationships is worth remembering. Much of language depends on how words
sit with respect to one another. “The adjective before this noun” is a pattern we can reuse anywhere.
Rotations give attention a compact way to express those relationships, and the later work asks how to preserve
them across longer documents and different kinds of input.

### Compare the same token across the designs

Each row below is the head output for the **second “the”**, using the main walkthrough's four-coordinate setup.
Different outputs make the changed pipeline visible. These invented weights do **not** rank the methods by quality.
Transformer-XL additionally consults the two memory tokens; its row belongs to the current segment.

| Design | Head output for position 4 |
|---|---|
| Learned absolute | (0.368, 0.741, 0.259, 0.970) |
| Sinusoidal | (1.523, 0.801, -0.171, 0.558) |
| Shaw | (0.273, 0.784, 0.282, 0.727) |
| Transformer-XL | (0.322, 0.668, 0.332, 0.678) |
| T5 | (0.310, 0.731, 0.269, 0.690) |
| DeBERTa | (0.330, 0.659, 0.341, 0.670) |
| ALiBi | (0.271, 0.773, 0.227, 0.729) |
| RoPE (90° / 10° toy) | (0.309, 0.833, 0.167, 0.691) |

## Three small thought experiments

No calculations needed. Try answering before opening the explanation.

<details>
<summary>If we put ten words before “the red ball,” what changes?</summary>
<p>The absolute positions change. The gap between “red” and “ball” does not. That is why relative position is a useful description of their relationship. Their full model representations can still change because the added words introduce new context.</p>
</details>

<details>
<summary>With ALiBi, can a faraway word beat a nearby word?</summary>
<p>Yes. The faraway word pays a larger distance penalty, but a sufficiently stronger content match can overcome it. ALiBi gives attention a preference; it does not forbid distant connections.</p>
</details>

<details>
<summary>Why doesn't a clock formula automatically give a model unlimited context?</summary>
<p>We can generate clock readings for a new position without having trained the model to use them well. Longer passages also bring unfamiliar relationships, more competing content, and higher computation and memory costs. A valid position label is only part of the problem.</p>
</details>
