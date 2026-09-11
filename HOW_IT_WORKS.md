# How the J-lens Works — From Zero

*A beginner's guide to every number and every idea in this project.
No prior knowledge assumed beyond "a neural network is a thing that learns".*

---

## Part 1 — Where the numbers come from

Every "magic number" in this project has one home. Here is the complete list,
so nothing feels pulled out of thin air:

| number | what it is | who chose it |
|---|---|---|
| **2560** | how many numbers the model uses to describe one token (`hidden_size`) | the Qwen team, when they designed Qwen3-4B |
| **36** | how many layers (processing steps) the model has | the Qwen team |
| **151,936** | how many different tokens the model knows (its vocabulary) | the Qwen team |
| **4B** | the model has ~4 billion learned numbers ("parameters") in total | the Qwen team |
| **128** | how many text passages we average over (`N_PROMPTS`), and tokens per passage (`SEQ_LEN`) | us, in `.env` (the paper uses 1000 passages) |
| **14–33** | the "workspace" layers — 38% to 92% of the way through 36 layers | the paper's finding, mapped onto our model |
| **25** | how many word-directions we allow when decomposing an activation | the paper (§2.3) |
| **944 MB** | size of our saved lens: 36 matrices × 2560 × 2560 numbers × 4 bytes | arithmetic |

The first four you can see yourself: open
https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json
and look for `hidden_size: 2560`, `num_hidden_layers: 36`,
`vocab_size: 151936`. **2560 is simply a design decision the Qwen engineers
made** — bigger models use wider descriptions (Qwen3-14B uses 5120).

---

## Part 2 — What a language model actually does

A language model does exactly one thing, over and over:

> Given some text so far, guess the next token.

That's it. Everything else — chat, reasoning, translation — is this one trick
repeated.

### Tokens: text becomes numbers

Computers can't read. So text is first chopped into **tokens** — pieces of
words — and each piece is looked up in a big dictionary of 151,936 entries:

```
  "The city where the Eiffel Tower stands"

          becomes the token pieces

  [The] [ city] [ where] [ the] [ E] [iff] [el] [ Tower] [ stands]

          which are just dictionary ID numbers

  [ 785 ] [ 3283 ] [ 1380 ] [ 279 ] [ 468 ] [ ... ]
```

Notice "Eiffel" got chopped into three pieces (`E`, `iff`, `el`) — the
dictionary only has whole entries for common words.

---

## Part 3 — The residual stream: the model's notepad

Here is the single most important picture in this whole project.

Inside the model, **each token gets its own "notepad" with 2560 slots**, each
slot holding one ordinary number (like 0.73 or -1.24):

```
   token " Tower" 's notepad (1 row, 2560 slots):

   ┌──────┬──────┬──────┬──────┬─────────────────┬──────┐
   │ 0.73 │-1.24 │ 0.02 │ 3.10 │   ... 2554 more │-0.88 │
   └──────┴──────┴──────┴──────┴─────────────────┴──────┘
      slot 1  slot 2  slot 3  slot 4      ...      slot 2560
```

This list of 2560 numbers is called the **residual stream** (in the code:
a "hidden state", or `h`). It is the model's entire working memory about that
token. Every thought the model has about " Tower" — *it's a noun, it's tall,
it's probably the Eiffel Tower, so we're talking about Paris* — must be
written somewhere into those 2560 numbers.

**This is where 2560 comes from every time you see it in the code.**

### The assembly line: 36 layers

The notepad doesn't stay fixed. It passes through 36 processing stations
(**layers**), and each one reads it and writes small updates onto it:

```
 token IDs
    │
    ▼
 ┌─────────┐   notepad v0: barely more than "which token am I"
 │ embed   │──────────────┐
 └─────────┘              ▼
                   ┌─────────────┐
                   │  layer 1    │  reads all notepads, writes updates
                   └─────────────┘
                          ▼
                   ┌─────────────┐
                   │  layer 2    │
                   └─────────────┘
                          ▼
                         ...          by layers 14-33 the notepads hold
                          ▼           "ideas": Paris, France, capital...
                   ┌─────────────┐
                   │  layer 36   │
                   └─────────────┘
                          ▼
                 notepad v36: basically "the next token should be ___"
```

In this project we number the notepad *versions*: **layer 0** = the notepad
right after embedding, **layer i** = the notepad after station i has edited
it. So `hs[20]` in the code means "everyone's notepads, version 20".

Two facts about the stations:

1. **They can look sideways.** A station updating " stands"'s notepad can
   read from earlier tokens' notepads (this is "attention"). That's how
   "Eiffel Tower" information flows into later positions.
2. **They only edit, never rewrite.** Each station *adds* its update to the
   existing notepad. That's why version 30 still resembles version 25.

---

## Part 4 — Reading the final notepad: the unembedding

How does a notepad become a word guess? The model has one final tool: a giant
scoring table `W_U` (the **unembedding**) with one row of 2560 numbers per
vocabulary token — 151,936 rows total.

To score token *v*: multiply the notepad's 2560 numbers with row *v*'s
2560 numbers, pairwise, and add them up. One number out — the score.
Do that for all 151,936 rows, and the highest scores are the model's guesses:

```
   final notepad (2560 numbers)
        │
        ▼        score for "Paris"  = 21.7   ← highest, model says "Paris"
   ┌─────────┐   score for "London" = 14.2
   │   W_U   │   score for "banana" = -3.0
   └─────────┘   ... 151,933 more scores
```

This "multiply pairwise and add" operation is called a **dot product**, and
it measures *how much the notepad points in the same direction as that row*.

**The catch:** `W_U` was trained to read the *final* notepad only.
Point it at a middle-layer notepad, and you get mostly garbage — the middle
layers write in their own private shorthand.

You saw this yourself. In experiment 01 the raw notepad at layer 14 under the
token `el`, scored directly with `W_U` (the "logit lens" baseline), gave:

```
  -building, 晚年, 纤, 对企业, sts      ← nonsense
```

---

## Part 5 — The J-lens idea: a translator for the shorthand

Question: could we build a **translator** that converts a middle-layer
notepad into final-notepad language, so `W_U` can read it?

The paper's idea: measure it. Ask, for every slot pair:

> "If I nudge slot *j* of the layer-20 notepad up by a tiny amount,
>  how much does slot *i* of the final notepad change?"

Do you know what a **derivative** is? It's exactly that: a nudge-response
ratio. If nudging j by 0.001 moves i by 0.0007, the derivative is 0.7.
No deep calculus needed — it's "wiggle the input, watch the output".

Now collect *every* answer into a grid:

```
                        which slot you nudged (j = 1 ... 2560)
                        ┌──────┬──────┬──────┬─────
   which output      1  │ 0.70 │ 0.01 │-0.20 │ ...
   slot moved        2  │-0.05 │ 0.33 │ 0.00 │ ...
   (i = 1...2560)    3  │ 0.12 │-0.40 │ 0.91 │ ...
                     ...│  ...
```

That grid — 2560 × 2560 = 6.5 million nudge-response numbers — is the
**Jacobian**, `J`. ("Jacobian" is just the standard name for a grid of
derivatives. The **J** in J-lens stands for it.) One grid per layer, 36 grids,
saved in your `jlens_qwen3-4b.pt` file (hence 944 MB).

Why is this grid a translator? Because for small changes, every complicated
machine behaves like simple multiplication by its Jacobian. `J₂₀ × notepad`
answers: *"if the rest of the network processed this notepad the way it
usually does, what would the final notepad look like?"* Then `W_U` can score
it — and suddenly the shorthand becomes words:

```
   J-lens at layer 14, token 'el':   巴黎(=Paris), 法国(=France), Paris ...
   logit lens at layer 14:           -building, 晚年, 纤 ...      (garbage)
```

Same notepad. The only difference is the translator.

---

## Part 6 — How the computer measures 6.5 million derivatives

Don't let the word "derivative" scare you. In this part you will compute one
yourself, with nothing beyond multiplication and division.

### 6.1 You could measure one by hand

Remember what one cell of the grid means:

> `J[7][12]` = "if I nudge input slot 12 a little, how much does output
> slot 7 move?"

Nothing stops you from literally doing that experiment:

```
step 1: run the model.                     output slot 7 reads   4.200
step 2: add 0.001 to input slot 12.
step 3: run the model again.               output slot 7 reads   4.230
step 4: divide the changes:   0.030 / 0.001  =  30
```

Congratulations: `J[7][12] ≈ 30`. **That is all a derivative is** — two runs
that differ only by your nudge, and the ratio of the changes.

> **Important: PyTorch does NOT do this.** The nudge experiment only tells
> you what the number *means*. Nudging is approximate (how small is "small
> enough"?) and slow. What PyTorch actually does is apply **exact derivative
> formulas, operation by operation** — no nudging anywhere. The next two
> sections explain that real mechanism.

### 6.2 The one idea: sensitivities multiply, like exchange rates

Suppose:

```
1 dollar buys 0.90 euros
1 euro   buys 160  yen
```

How many yen does one extra dollar get you? You don't need to try it —
you multiply the rates: **0.90 × 160 = 144 yen per dollar**.

You have just used the famous "chain rule". That's the entire theorem: when
an effect flows through a chain of steps, the overall sensitivity is the
**product of each step's sensitivity**.

Now a two-step mini-model:

```
   step A:  y = 3·x + 1      (y's sensitivity to x is 3)
   step B:  z = 2·y          (z's sensitivity to y is 2)
```

Nudge x by 0.001. Then y moves by 0.003, so z moves by 0.006. Sensitivity of
z to x: **3 × 2 = 6** — no rerunning, just multiplied rates. Check it against
the hand method: it agrees exactly.

A transformer is nothing more than a very long chain of such simple steps —
multiplies, adds, a few smooth curves — and the sensitivity through the whole
model is a product of the steps' local rates. Bookkeeping, not magic.

### 6.2b What PyTorch *actually* does (the real mechanism)

Three ingredients, no nudging:

**1. A formula table.** For every elementary operation, the derivative is
known *as a formula*, worked out by humans once and shipped inside PyTorch:

```
   operation            its exact derivative rule
   ─────────────────    ────────────────────────────────────────
   y = W · x            sensitivity flows back as  Wᵀ · (incoming)
   y = a + b            incoming sensitivity passes to BOTH a and b
   y = tanh(x)          multiply incoming by (1 − y²)
   softmax, RMSNorm     their own known formulas
```

**2. A recording of the forward pass.** When the model runs, PyTorch builds
a graph: every operation notes what kind it was, which tensors fed it, and
saves any values its derivative formula will need (e.g. `tanh` saves its own
output, because its rule uses `1 − y²`).

**3. The backward walk.** Start at the output with your chosen vector (our
one-hot "needle 7 only"). Visit the recorded operations in reverse order.
At each one, apply its formula from the table to transform the sensitivity —
that's the "multiply the exchange rate" step, done *exactly*, with the saved
values plugged in. Where two paths merge (every layer's "add my update to the
notepad" step!), the sensitivities from both paths are **added**.

When the walk reaches layer 20, what has accumulated is precisely
row 7 of `J₂₀` — exact to floating-point precision, because every step was a
formula, never an approximation.

You can verify this claim: on a small test function, autograd agrees with
the hand-derived calculus formula to `2×10⁻¹⁶` (pure floating-point
round-off), while the nudge method only gets within `7×10⁻⁷`. Autograd IS
the formula method, automated.

One more subtlety: PyTorch never even builds each step's own sensitivity
grid. Each formula directly transforms the incoming *vector* (for `y = W·x`,
the rule is just one matrix multiply by `Wᵀ`). That's why one backward walk
is cheap — about the cost of two forward runs — even though the grid it
contributes a row to has millions of entries.

### 6.3 Two directions to ask — columns vs rows

Picture everything between layer 20 and the output as a box with
**2560 input dials** and **2560 output needles**. There are two ways to fill
in the sensitivity grid:

```
 FORWARD: nudge ONE dial, watch ALL needles.
          → you learn one COLUMN of the grid per experiment.
            "what does dial 12 affect?"

 BACKWARD: pick ONE needle, walk backwards through the chain,
           multiplying exchange rates upstream toward every dial.
          → you learn one ROW of the grid per walk.
            "what is needle 7 sensitive to?"
```

Backward sounds strange, but it's just reading the currency chain
right-to-left: "how many yen per dollar?" can be answered starting from the
yen side — same multiplication, done from the goal end. And the payoff is
big: **one backward walk gives needle 7's sensitivity to every one of the
2560 dials at once** — a complete row, exact, no tiny-nudge error.

Why does PyTorch only ship the backward direction? Because of *training*.
Training asks: "how does the loss — ONE number — depend on all 4 billion
weights?" That question is one row with 4 billion entries: precisely what a
single backward walk delivers. The J-lens simply hijacks this training
machinery and uses it as a measuring instrument.

### 6.4 The bill: one row per pass, 2560 rows

```
 walk backward from needle    1  →  row    1  (2560 numbers)
 walk backward from needle    2  →  row    2
 ...
 walk backward from needle 2560  →  row 2560     grid complete
```

Each walk costs about as much as running the model twice. So one text
passage costs ≈ 5000 model-runs' worth of compute, and we do 128 passages.
**That is your 8.5 hours** — versus milliseconds for an ordinary question,
which needs only a single forward run.

(One optimization in the code: it feeds 16 identical copies of the text at
once and picks a *different* needle in each copy. Copies can't influence
each other, so one backward walk returns 16 exact rows. That's the
`ROWS_PER_BACKWARD=16` setting in `.env` — 160 walks per text instead
of 2560.)

### 6.5 One token's effect on the FUTURE — and a trick to count it

So far we quietly pretended there is one notepad and one output. In reality
**every position has its own notepad and its own output**. Time to face that.

**Why would token 5's notepad affect token 9's output at all?**
Because of attention (Part 3, fact 1): when the model works on token 9, it
*reads the notepads of earlier tokens*. This is not a technicality — it is
the entire point of the workspace. Look at your own spider experiment:

```
  "How many legs does the animal that spins webs have? ..."
                                            ─┬──
                     "spider" gets written   │
                     on THIS token's notepad ┘        answer "8" comes out
                                                      at the LAST position ─┐
                                                                            ▼
```

The inference lives on the notepad at "webs", but the payoff appears many
tokens later. If we measured each notepad only by its effect on *its own*
position's output, we would miss exactly the broadcasts that make the
workspace interesting. So the paper defines the lens as:

> credit token t's notepad with its effect on its own output **plus every
> later position's output**.

**Why never the PAST?** The model reads left to right: when computing token
3's output, tokens 4 and 5 are simply not looked at (the "causal mask").
Like editing a movie: changing minute 30 cannot alter minutes 0–29.

Put both facts in one table. ✔ = can influence, 0 = physically impossible:

```
                        which position's output can it change?
                          z₁     z₂     z₃     z₄     z₅
   nudge token 1's pad:   ✔      ✔      ✔      ✔      ✔     ← affects all
   nudge token 2's pad:   0      ✔      ✔      ✔      ✔
   nudge token 3's pad:   0      0      ✔      ✔      ✔
   nudge token 4's pad:   0      0      0      ✔      ✔
   nudge token 5's pad:   0      0      0      0      ✔     ← only itself
```

What the paper wants for each token is **the sum of the ✔ entries in its
row** ("itself + its future").

**The expensive way:** measure every ✔ cell separately — one backward walk
per (token, output) pair. For 128 tokens that's thousands of extra walks.

**The trick the code uses:** do ONE backward walk, starting not from a
single position's output but from the **sum of ALL positions' outputs**
(the line `s = z.sum(dim=1)` in `jlens/lens.py`). The sensitivity that lands
on token t's notepad is then automatically

```
   (effect on z₁) + (effect on z₂) + ... + (effect on z_T)
```

— its whole row of the table, zeros included. And since the zeros contribute
nothing, that total **equals the sum of just the ✔ entries**: itself + its
future. Exactly the quantity the paper defines.

A tiny numeric check, 3 tokens. Suppose nudging token 2's slot would change
z₂ by 0.4 and z₃ by 0.3 (and z₁ by 0 — impossible, past). The one-walk trick
delivers, at token 2:  0 + 0.4 + 0.3 = **0.7** = self + future. Correct, and
we never had to measure the pieces separately.

Best of all, the single walk drops the right total on **every token's
notepad simultaneously** — token 1 gets its row-sum, token 2 gets its
row-sum, and so on. One walk, all positions, all futures. That is why
computing the lens is merely *expensive* and not *impossible*.

### 6.5b The backward walk frame by frame — where the adding happens

Section 6.5 said the one-walk trick "delivers the row-sum". Here is the
mechanism, slowed down. Three tokens, layer 20, needle slot i. We backprop
the single number sᵢ = z₁ᵢ + z₂ᵢ + z₃ᵢ:

```
start:    a sensitivity of 1 flows into z₁ᵢ, z₂ᵢ and z₃ᵢ (each has
          derivative exactly 1 with respect to their sum)

from z₃ᵢ: the walk descends layers 35→21; through the attention edges it
          reaches notepads 1, 2 AND 3 at layer 20 (forward-time z₃ read all
          three), depositing  ∂z₃ᵢ/∂h₁   ∂z₃ᵢ/∂h₂   ∂z₃ᵢ/∂h₃

from z₂ᵢ: can only reach notepads 1 and 2 — the forward pass never built a
          path from notepad 3 to z₂, so backward has no road there.
          Deposits  ∂z₂ᵢ/∂h₁   ∂z₂ᵢ/∂h₂        (nothing at notepad 3)

from z₁ᵢ: deposits only  ∂z₁ᵢ/∂h₁
```

Now the single mechanical fact that makes the trick work: **when several
backward paths land on the same tensor, autograd ADDS their deposits**
(the accumulation `+=` from 6.2b). So the finished gradient g = [3, 2560]
reads:

```
g[1] = ∂z₁ᵢ/∂h₁ + ∂z₂ᵢ/∂h₁ + ∂z₃ᵢ/∂h₁    = row i of J₍₂₀,₁₎
g[2] =            ∂z₂ᵢ/∂h₂ + ∂z₃ᵢ/∂h₂    = row i of J₍₂₀,₂₎
g[3] =                       ∂z₃ᵢ/∂h₃    = row i of J₍₂₀,₃₎
```

Each position's "self + future" sum was assembled by the walk itself; the
"past" terms were never in the graph, so nothing had to be masked out. And
because the same walk passes through every layer, it deposits such a g on
H₁₉, H₁₈, ... too — row i of ALL 36 layers from one backward.

**Where did the position axis go?** Note that g[1], g[2], g[3] are rows of
*different* matrices — the true sensitivity is per (layer, position). The
code then takes the mean over positions (and later over prompts), so the
file on disk stores ONE matrix per layer: the average of all the J₍ₗ,t₎.
Reading a new prompt multiplies every position's notepad by this same
averaged matrix — nothing position-specific is recomputed. Why is that
allowed? Same argument as Part 7: each individual J₍ₗ,t₎ is full of one-off
detail (this prompt's attention pattern, this position's quirks), but
averaged over thousands of (prompt, position) samples the one-off parts
cancel, and what survives is the model's shared encoding convention —
average thousands of handwriting samples and you are left with the font.
That one averaged linear map still translates well is the paper's central
empirical finding, and it is what experiment 01 checks every time the right
word crystallizes in the grid.

### 6.6 The same story, in exact tensor shapes

Let's drop all analogies and track the real arrays. Example prompt with
**T = 5 tokens** (in your real runs T = 128; the model width is 2560):

```
   position (time snap):    1       2       3       4        5
   token:                 [The] [ animal] [ that] [ spins] [ webs]
```

**The forward pass.** There is not one notepad — there is a SHEET: one row
per position, 2560 slots per row. Every layer takes a sheet in and puts a
sheet out, same shape, because each layer only ADDS updates:

```
   tensor                      shape         meaning
   ────────────────────────    ──────────    ─────────────────────────────
   token ids                   [5]           five dictionary numbers
   H₀   (after embedding)      [5, 2560]     5 notepads, version 0
   H₁   (after layer 1)        [5, 2560]
   ...                           ...
   H₂₀                         [5, 2560]     ← we'll ask about this one
   ...
   Z = H₃₅ (the target)        [5, 2560]     5 final-ish notepads
```

Inside a layer, row t of the OUTPUT sheet is computed from rows 1..t of the
INPUT sheet (attention looks left only). Row t never sees rows t+1..T.

**The full sensitivity object.** "Notepad" = one row, e.g. h₂ = H₂₀[2] with
2560 numbers. "Output" = one row of Z, e.g. z₄ = Z[4], 2560 numbers. Between
ONE input row and ONE output row there is one nudge-response grid of shape
[2560, 2560]. And there are 5×5 = 25 such (input row, output row) pairs:

```
                      ∂z₁      ∂z₂      ∂z₃      ∂z₄      ∂z₅        each cell is
   ∂/∂h₁            [grid]   [grid]   [grid]   [grid]   [grid]       a [2560,2560]
   ∂/∂h₂              0      [grid]   [grid]   [grid]   [grid]       grid; 0 means
   ∂/∂h₃              0        0      [grid]   [grid]   [grid]       the all-zero
   ∂/∂h₄              0        0        0      [grid]   [grid]       grid (past =
   ∂/∂h₅              0        0        0        0      [grid]       unreachable)
```

This is the ✔/0 table from 6.5, now with honest sizes: the complete object
is [5, 5, 2560, 2560]. The paper's definition says: for source row t, ADD UP
row t of this table (its own column plus all future columns), then average
the five results over t. Target: one [2560, 2560] matrix per layer.

**What one backward walk produces.** The code (with 16 copies for 16 needle
choices) runs:

```
   tensor                          shape            what it is
   ─────────────────────────────   ─────────────    ───────────────────────
   batch of ids                    [16, 5]          16 identical copies
   H₂₀                             [16, 5, 2560]
   Z                               [16, 5, 2560]
   s = Z.sum(dim=1)                [16, 2560]       the 5 output rows ADDED
   loss = Σ_b s[b, needle_b]       one number
   g = ∂loss/∂H₂₀                  [16, 5, 2560]    ← the gradient
```

Two things to absorb about `g`:

1. **A gradient always has the same shape as the tensor you differentiate
   with respect to** — one sensitivity number per entry of H₂₀. That's why
   it is [16, 5, 2560], not a matrix.

2. Take one copy b (needle i) and one position, say t = 2. The slice
   `g[b, 2, :]` is 2560 numbers, and by linearity of derivatives it equals

```
   row i of ∂z₁/∂h₂  +  row i of ∂z₂/∂h₂  +  ...  +  row i of ∂z₅/∂h₂
       = 0 (past)    +     [2560 nums]    +  ...  +    [2560 nums]
```

   i.e. **row i of the summed row-2 cells of the big table** — self+future,
   the zeros contributed by causality itself. Every position t got its own
   version of this in the same walk: that is what the axis of length 5 in
   `g` holds.

**The last two shrinking steps** (in `jlens/lens.py`, lines ~106-108):

```
   g.mean over the position axis     [16, 5, 2560] → [16, 2560]
        = 16 finished ROWS of this prompt's J₂₀
   accumulate rows into J            J₂₀ grows toward [2560, 2560]
   ... repeat for 2560/16 = 160 walks → J₂₀ complete for this prompt
   ... average J₂₀ over 128 prompts  → the file on your disk
```

So the entire pipeline, shapes only:

```
 [5]  →  [5,2560]×36 sheets  →  sum rows: [2560]-per-copy  →  backward:
 [5,2560] gradient per copy  →  mean over 5: one row [2560]  →  stack 2560
 rows: [2560,2560]  →  average 128 prompts  →  J₂₀ in jlens_qwen3-4b.pt
```

---

## Part 7 — Why average over 128 texts?

Imagine a friend rolls a die, adds it to a secret number, and tells you only
the total: 9, 5, 8, 6, 10... No single answer reveals the secret. But
average twenty answers and the die's randomness cancels out (its average is
always 3.5), leaving `secret + 3.5` — the constant part survives, the noise
dies.

Measuring the grid on ONE text has the same problem. The result entangles:

- how the model *generally* translates layer-20 shorthand into words ← want
- what it happened to be doing with *this text about Paris*          ← noise

So the grid is measured on 128 different Wikipedia passages — music,
football, wine, military history — and averaged, exactly like the die rolls.
What survives is the model's *stable translation habit*; the per-text noise
cancels. Star photographers do the same thing when they stack many photos of
a faint galaxy.

This is also why the corpus had to be **diverse**: early in this project the
lens was accidentally averaged over just 2 Wikipedia articles, and the
readouts were visibly worse — that "noise" hadn't cancelled, it had voted.

(The paper averages 1000 texts. Its Figure 59 shows even 10 works. Your 128
is a solid middle.)

---

## Part 8 — What you can do once you have J

### Read (experiments 01, 04)

```
 middle notepad ──× J──▶ translated notepad ──× W_U──▶ scored words
```

This is how you saw `spider` at layer 30 for a prompt that never contains
the word — the model *inferred* it, wrote it on the notepad in shorthand,
and the lens translated the shorthand.

### Every word gets an address (the "J-lens vector")

Run the machinery in reverse and you can compute, for any word, the exact
*direction* in notepad-space that makes the lens read that word. Think of it
as each of the 151,936 words having a street address inside layer 20.

### Two ways to score a word: raw vs cosine

Once every word is an arrow (its J-lens vector v̂) and the notepad h is an
arrow too, "which word does the notepad say?" can be scored two ways:

```
 raw score   ⟨v̂, h⟩            = ‖v̂‖ · ‖h‖ · cos(angle)   (dot product)
 cosine      ⟨v̂, h⟩ / ‖v̂‖‖h‖  =              cos(angle)   (direction only)
```

The raw score is what the model's own unembedding computes — but notice it
rewards a word for having a LONG arrow just as much as for pointing the
right way. And a handful of rare tokens (`____`, the non-breaking space...)
happen to have arrows 2–3× longer than the median. In raw rankings they
shout their way to the top of many cells while pointing nowhere in
particular — that is why early readouts can look full of `____` junk.
Cosine divides both lengths out, keeping only the angle: the junk sinks,
and the same cells read `Egypt` or `spider`. The direction was right all
along; it was being outvoted by loud arrows.

In the code (`jlens/lens.py`): `logits` is the raw form, `cosine` the
normalised one, `readout(..., cosine=True)` switches between them. The
workbench defaults to cosine (best for a human scanning tokens); experiment
01 offers `--cosine` as a flag (raw is the more literally-what-the-model-
computes number, so it stays the default there and in `rank`).

### Edit a thought (experiments 02, 05, 06, 07)

Once "France" has an address, you can edit the notepad like a sticky note:
erase exactly the France-amount written at France's address, write the same
amount at Italy's address, touch nothing else, and let the model continue.
From your own run:

```
 Question: "capital of the country famous for the Eiffel Tower?"
 normally:                       Paris  (100%)
 after swapping France → Italy:  Rome   (100%)
```

No word of the prompt changed. The model's *thought* was edited, and the
answer followed the thought. That is the paper's central claim demonstrated:
the notepad content the lens reads is real, and the model genuinely uses it.

---

## Part 9 — Tiny glossary

| term | plain meaning |
|---|---|
| token | a word piece; one of 151,936 dictionary entries |
| residual stream / hidden state / `h` | one token's notepad: 2560 numbers |
| layer | one of 36 stations that edit the notepads |
| `hs[l]` | all notepads, version l |
| unembedding `W_U` | the scoring table: notepad → 151,936 word scores |
| logit lens | scoring a middle notepad with `W_U` directly (mostly fails) |
| derivative | nudge-response ratio: output change ÷ input change |
| Jacobian `J` | the full 2560×2560 grid of nudge-response ratios |
| backward pass | PyTorch tracing derivatives backwards; yields one row of J |
| J-lens | `W_U × (J × notepad)`: translate first, then score |
| J-lens vector | one word's "address" (direction) inside a middle layer |
| workspace (layers 14–33) | the layer range where notepads hold ideas, not just spelling |

## Where each piece lives in the code

```
.env                       every number you might change
jlens/lens.py              measuring J (top half) + reading with it (bottom half)
jlens/interventions.py     the editing tools (steer / remove / swap)
jlens/jspace.py            "which few words is this notepad made of?"
experiments/00...12        the paper's experiments using all of the above
```

Suggested next step: reread Part 5, then open `jlens/lens.py` and find the
line `s = z.sum(dim=1)` — you now know exactly what it does and why.
