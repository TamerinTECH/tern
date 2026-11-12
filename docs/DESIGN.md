# tern - design notes

How the engine gets its speed on Gemma 4, what was measured on the way, and what did not work.
The first version of tern ran Gemma 3 270M on a Raspberry Pi 5 (that history is at the end). This version
runs the Gemma 4 family on x86-64. Numbers: [RESULTS.md](RESULTS.md).

## 1. Two limits decide everything

For every generated token a transformer reads all of its weights once (a "pass"). Two machine limits bound
the speed, measured with `bench/membw_x86.c` and `bench/kernbench.c`:

| | 32 EPYC (Zen 3) cores, AVX2, 8-channel DDR4 | Ryzen 5 8400F, 6 cores, AVX-512, DDR5 at 3600 |
|---|---|---|
| Memory bandwidth, all cores | 85 - 265 GB/s, depending on where the memory lies (section 8) | ~41 GB/s |
| 4-bit matrix kernel, data in cache | ~75-85 billion multiply-adds/s per core, ~2,200 on 32 | ~115-130 per core |

- **One stream is memory-bound.** Gemma 4 31B has 16.5 GB of 4-bit weights: at 190 GB/s a pass takes at least
  87 ms, whatever the arithmetic costs. The arithmetic for one token (31 billion multiply-adds) takes 14 ms on
  the 32 cores: memory is 6x slower than arithmetic.
- **That spare arithmetic is what batching and speculation use.** Evaluating 4 tokens in one pass over the 31B
  costs about 87 + 4 x 11 ms, not 4 x 100 (measured: 154 ms for 4.7 tokens).
- The small models (E2B: 0.7 GB of matrices per pass, 0.5 GB more for the exact output layer) are memory-bound
  too, but a pass is so short (8 ms) that what happens between the matrix products becomes visible (section 7),
  and so does the place of the weights in memory (section 8).

## 2. One pass, many tokens: slots

One pass evaluates up to 32 "slots". A slot is a token with a sequence and an optional parent slot; it
attends to the committed tokens of its sequence plus its ancestors in the same pass. A plain batch, a prompt
chunk and a chain of guessed tokens are all the same thing. Keys and values of the slots are written to
scratch rows of the cache; `model_commit` keeps the chosen path.

**A slot's result does not depend on what else is in the pass.** Each weight row is evaluated in column
chunks of a fixed size (96 blocks of 32 weights), whatever the number of slots, so the float operations for
one slot are the same sequence alone or in company; attention visits the cached rows in position order. This
makes speculative decoding exact: text with and without it is byte-identical (checked for all models).
A first version chose the chunk size from the slot count, and texts diverged at near-ties after ~100 tokens.

## 3. The weight kernel

- **Files as published.** The GGUF is memory-mapped; matrices are repacked once at load, on all threads
  (31B: 2.3 s), into the layout below, in one anonymous arena (on huge or on 4 KB pages: section 8). The file
  pages are released after repacking, so peak memory is about the model size.
- **Activations** are quantized to int8 per block of 32, as llama.cpp does for these matrix types, so the
  numerics are the same class. `tests/ref_gemma4.py` is an independent float implementation used as ground
  truth (RESULTS.md, "Is the output still the model's output?").
- **Layout.** A row is a sequence of groups of 16 blocks (512 weights): 16 scales, then the nibbles arranged
  so that one byte-dot-product instruction adds four products into each 32-bit lane and lane *l* only ever
  sees block *l*. After 8 instructions each lane holds the complete integer sum of one block: one int-to-float
  conversion and one multiply by the 16 block scales serve 512 weights. Rows whose width is not a multiple of
  512 end in groups of 8, 4, 2 and 1 blocks. The same layout serves AVX-512 (one 64-byte vector), AVX2 (two
  halves) and plain C.
- **AVX-512 VNNI** (`vpdpbusd`): 64 multiply-adds per instruction. **AVX2** has no byte dot product:
  `vpmaddubsw` adds two products into 16-bit lanes; with 4-bit weights the eight vectors of a block still fit
  in 16 bits, so a single widening add per 8 blocks finishes the sums.
- Three row kinds: `P4` (GGUF Q4_0: scale, 32 nibbles), `P4M` (Q4_1 and Q4_K: scale and offset), `P8` (Q6_K,
  the output matrix of Google's files: 32 bytes and a scale per 16 weights, exact).
- **Cache blocking** for many slots: the activations of 4 slots for one column chunk and the weights of one
  row block stay in the first-level cache; a row block is reused for every slot group before the next block
  is read from memory.

Measured limits of the kernel: on Zen 3 the integer multiplier is one execution port, and a block of 256
multiply-adds needs 9 multiplies, which caps a core at ~85 billion multiply-adds/s; on Zen 4 a 512-bit load
per cycle is the cap (10 loads per 512 multiply-adds and slot).

## 4. Gemma 4 specifics

Reference for the computation: llama.cpp `src/models/gemma4.cpp` (commit 2cbe0ec).

- **E2B / E4B:** per-layer token inputs (a 256-wide table row per layer and token, gated into each layer), and
  the upper 20 layers share the keys and values of two lower layers. Tokens whose output is not needed (the
  prompt, except its last token) only run the 15 lower layers: prompt processing reads 366 MB per pass
  instead of 1.05 GB.
- **31B:** 60 layers, 32 query heads on 16 or 4 key/value heads. The cache stores each key/value head as its
  own contiguous run of rows, so scoring a head streams memory instead of hopping 8 KB per token.
- **26B-A4B:** a dense feed-forward plus 128 experts per layer, 8 used per token. The slots of a pass choose
  experts independently; each distinct expert is evaluated once for all slots that chose it, and the work
  units of the parallel region are (expert, row range) so that all threads are busy even when one token uses
  only 8 experts.
- **Tokenizer:** byte-pair encoding with ranked merges over text whose spaces became U+2581, same ids as
  llama.cpp.

## 5. Speculative decoding with the model's own drafter

Google publishes a small drafter for every Gemma 4 model ("assistant", the `mtp-*.gguf` files): four layers
that have no keys and values of their own but attend over the main model's cache. A step takes a token and
the hidden state that predicted it and returns the next token and an estimate of the hidden state after it,
which feeds the next step (`assistant_step` in `src/model.c`, after llama.cpp's `gemma4-assistant.cpp`).

tern drafts a few tokens, then evaluates the new token and the guesses as a chain of slots in one pass. Every
guess the model confirms is a token that needed no pass of its own; a wrong guess is discarded. On the same
prompt tern and llama.cpp confirm the same share of guesses (60 of 177 vs 59 of 175): the drafter is the
same, what differs is the cost of the pass that checks them.

## 6. The output layer

The exact output layer multiplies the hidden state h with all 262,144 rows of the output matrix (0.5 GB for
E2B, 0.9 GB for 26B, 1.8 GB for 31B in the engine's 8-bit rows): where a stream is memory-bound that is 10 to
30 % of a token.

The fast output layer (`src/head.c`) reads 15 - 19 % of those bytes and returned the exact layer's token in
every held-out step that was checked (RESULTS.md). It has three parts:

1. **Sketch.** Every row w is kept a second time as `(w's coordinates along u_1..u_32) + a * sign(rest of w)`:
   32 bytes, one bit per weight, one scale. The u are the directions in which final hidden states vary most:
   the leading eigenvectors of the second moment of recorded states, found by orthogonal iteration. They are
   the "plan" (0.2 - 0.7 MB), written by `tern --make-head` from text the model generates. A scan computes for
   every token `sum_k A_k g_k + a * (signs . rest of h)`, the rest of h quantized to int8.
2. **Candidates.** The 1,024 tokens with the best sketch scores are scored again with their full rows, by the
   same kernel as the exact layer.
3. **Core.** Rows that one bit per weight cannot hold are scored with full rows at every step: rows whose
   rest r has `mean(r^2) / mean(|r|)^2 > 1.65`. That ratio is pi/2 = 1.57 for bell-shaped weights (the median
   row has 1.55 - 1.56) and grows when few large weights carry the row. 3.4 - 5.2 % of the rows qualify, the
   most frequent tokens among them: the best token is a core row in 36 - 46 % of steps. The test needs no
   statistics, only the rows.

What each part is for, measured on 6,000 held-out steps per model (hidden states from prompts that were not
used for the plan; "rank" = position of the true best token when all tokens are ordered by sketch score; the
layer is right whenever the rank is at most the number of candidates). Tool: `tools/research/head_candidates.py`.

| Sketch | E2B | 26B-A4B | 31B |
|---|---|---|---|
| signs only: steps where the true best token is not among 4,096 candidates | 0.28 % | 2.35 % | 0.35 % |
| signs only: worst rank | 15,038 | 73,692 | 165,291 |
| 16 directions + signs: worst rank | 275 | 112 | 9,117 |
| 32 directions + signs: worst rank | 172 | 140 | 5,750 |
| 32 directions + signs, core rows exact: worst rank | 172 | 140 | 16 |

- *Why directions.* A 1-bit copy of a row is wrong in proportion to the length of the vector it meets, and a
  hidden state is long in a few directions: one direction (roughly the mean state) holds 7 - 26 % of the
  energy, 32 hold 40 - 54 %. That part is the same at every step and swamps the differences between tokens.
  Eight to sixteen directions remove most of the damage; 32 fit in the same row size on the large models.
- *Why the core.* Token 672 of the 31B is chosen with a logit of 44.3, of which 42.2 come from the rest of the
  row (16 directions). The signs give 1.1 for that part: a few large weights meet a few large coordinates of
  h, and a sign code spreads the row's weight evenly. No number of candidates repairs that, and the tokens it
  hits are frequent ones. All rows of this kind fail the bell-shape test.
- *The scale a.* Three choices were compared: mean|r| (fits the row best), the root mean square, and
  mean(r^2)/mean|r| (keeps the logit of a state that points along the row's signs). The root mean square
  missed least across the three models and is used.
- *The plan is not fragile.* Directions from 500 recorded steps work as well as directions from 100,000.
  Directions taken from the rows of the matrix alone (no recording) do not: worst rank 19,033 on the 26B.
- *It saves memory traffic, not arithmetic.* The sign bits are expanded to bytes and multiplied with the int8
  state, one multiply-add per weight like an exact row. With more than 8 states per call the scan would cost
  as much as the exact sweep, which tern then uses (`FAST_HEAD_SLOTS` in `src/main.c`).

The first version of this layer used 128 directions fitted to the covariance of the logits, two preview
stages, a recorded list of frequent tokens and a per-sequence pool. Its error sits where it hurts: on tokens
outside the fitted space. It agreed with the exact layer in 99.2 % (E2B), 97.2 % (26B-A4B) and 89.3 % (31B) of
steps, and was replaced.

## 7. Between the matrices

A pass is a few hundred short parallel regions (31B: 300 to 400 for one token) with small per-slot steps
between them: norms, quantization, rotations, the routing of the experts. For one stream these steps are
single-threaded while 31 cores wait. Three things were done about that, each checked to leave the tokens
unchanged:

- **The longest steps run on all threads.** `gelu(gate) * up` and the quantization of its result cover 21,504
  values per layer in the 31B (a few milliseconds of a pass when one thread does them): they are cut into pieces of 1,024
  values that the threads take from a counter. A piece is a multiple of the kernel's group of 512 values, so
  every value goes through the same instructions as before. For the 26B-A4B the 128 router logits per layer
  (a float row of 2,816 each) and the middle step of the 8 chosen experts are spread the same way.
- **Short steps do not start a parallel region.** With one or two slots the caller does the step alone.
  A region costs only a microsecond when it is empty (`bench/poolbench.c`), but it ends when its slowest
  thread is done, and there were six of them per layer.
- **Waiting workers read ahead.** While the caller does a short step, the other threads read their share of
  the next matrix into their caches, a kilobyte between two looks for work (`ra_idle` in `src/model.c`).
  Without any read-ahead the body of a 31B pass takes 114 - 118 ms on the VM, with it 104 - 112 ms.

Together: 26B-A4B 44.5 - 47.7 tok/s before, 48.6 - 49.8 after; 31B 8.5 - 9.1 before, 8.9 - 9.2 after; E2B
unchanged (one stream, exact output layer, three alternating runs each on the VM).

## 8. Where the weights lie: a finding on the benchmark VM

One buffer, read by 32 threads in parallel regions of a given size (`bench/burst.c`). First measurement, a
2 GB buffer:

| Region | transparent huge pages (2 MB) | 4 KB pages |
|---|---|---|
| 5 MB | 84 - 90 GB/s | 173 - 194 GB/s |
| 70 MB | 91 - 103 GB/s | 197 - 235 GB/s |
| all 2 GB at once | 91 - 101 GB/s | 215 - 221 GB/s |

An hour later, after other programs had used and freed memory (regions of 70 and 250 MB):

| Buffer | huge pages | 4 KB pages |
|---|---|---|
| 512 MB | 193 - 258 GB/s | 236 - 251 GB/s |
| 2 GB | 147 - 200 GB/s | 248 - 265 GB/s |
| 8 GB | 114 - 142 GB/s | 227 - 255 GB/s |

Memory behind small pages was read at 170 - 265 GB/s in every measurement; memory behind huge pages at
anything between 85 and 260 GB/s, depending on which blocks the allocation happens to receive (three 16 GB
buffers on huge pages, held together: 242, 105 and 104 GB/s). Who writes the buffer first (one
thread or all) makes no difference, and both kinds of buffer are spread over the guest's physical memory in
the same way (read from `/proc/self/pagemap`).

What that means for the engine: the same build with the weights forced onto one page size, one stream,
tok/s with the exact / the fast output layer, measured back to back:

| Model | huge pages | 4 KB pages |
|---|---|---|
| E2B | 91.2 / 104.7 | 96.9 / 118.4 |
| E4B | 50.3 / 59.3 | 53.8 / 62.4 |
| 26B-A4B | 52.3 / 61.7 | 54.6 / 63.4 |
| 31B | 11.1 / 11.8 | 10.2 / 10.3 |

An hour earlier the gap was larger for the small models (E4B 45.4 / 55.1 on huge pages against 53.1 / 62.1)
and the 31B was level (9.3 / 10.5 against 9.9 / 10.4), half an hour later it was level again (10.5 / 11.4
against 10.8 / 10.9). With 32 sequences per pass, where arithmetic and not
memory is the limit, huge pages win: E2B 686 tok/s against 599, E4B 373 against 324.

Small pages cost a little on every page (the processor's read-ahead stops at a page's end, and every page
needs its own address translation); that shows when the matrices are large or when memory is not the limit.
Huge pages can be unlucky as a whole. The rule tern uses, best or equal in every case above: **4 KB pages if
the model is small for the machine (less than 512 MB of weights per thread) and at most 8 sequences are
evaluated per pass, huge pages otherwise**; `TERN_HUGE=1` or `0` overrides it. The rule is fitted to these
measurements, on two machines that are both virtual (on the desktop under WSL 2 small pages were equal or
slightly better for E2B: 17.5 - 19.2 against 15.8 - 19.3 tok/s in four alternating runs).

Our reading of the cause, which could not be verified from inside the VM: the host has more than one memory
node that the guest does not see, and a 2 MB block lies on one of them as a whole, while a matrix on small
pages is spread over both. An engine that measured where its pages are fast and arranged its matrices
accordingly could gain more on such machines.

## 9. What did not work or was not worth it

- Chunk size chosen from the slot count (faster by a few percent): breaks the equality of speculative and
  plain decoding (section 2).
- More than three guessed tokens per pass on the 31B (4 and 5 were tried): more tokens per pass (3.2, 3.5), but
  every guess costs a slot and a drafter step, and the speed stays the same or falls.
- The drafter on top of the fast output layer for the 26B-A4B: every guessed token routes to its own experts,
  so a pass with guesses reads up to four times the expert weights. No gain over the fast layer alone.
- Keys and values stored per token (all heads of a token side by side): with 16 key/value heads the scoring
  loop hops 8 KB per token and misses the cache: 9 ms per 31B pass for 45 tokens of context.
- One shared completion counter in the thread pool: with 32 threads the cache line bounces for every one of
  the ~300-400 parallel regions of a pass; each thread now reports on its own line.
- An output layer that ranks tokens by a low-rank fit of the logits (section 6): fast, and wrong in 1 - 11 %
  of steps.
- From the Raspberry Pi work (Gemma 3 270M, kept as background): clustering the vocabulary and exact cluster
  bounds for the output layer; 1-2 bit screening of the rows without the directions and the core of section 6;
  int8 keys in attention; software prefetch of the next matrix.
