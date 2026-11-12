# tern - measured results (2026-11-12)

**Machine A (all speed numbers unless stated):** OVH cloud VM "R3-256": 32 cores of an AMD EPYC (Milan, Zen 3;
AVX2, no AVX-512), 245 GB RAM, Ubuntu 26.04 (kernel 7.0), gcc 15.2, 32 threads. Nothing else was running.
Its int8 kernel does ~2,200 billion multiply-adds/s on 32 cores; its memory is read at 85 to 265 GB/s
depending on where an allocation lands (see "What the VM does to the numbers").

**Machine B (desktop, section further down):** AMD Ryzen 5 8400F (6 cores, Zen 4, AVX-512), 64 GB DDR5 running
at 3600 MT/s, Windows 11 with WSL 2 (Ubuntu 24.04, gcc 13.3), 6 threads. The machine was in use during the
runs: its numbers are indicative.

**Models:** Google's QAT Q4_0 GGUF files of Gemma 4: E2B (3.35 GB), E4B (5.15 GB), 26B-A4B (14.4 GB, mixture
of experts), 31B (17.65 GB, dense), and the published drafters (`mtp-gemma-4-*-Q4_0.gguf`).
**Reference:** llama.cpp commit 2cbe0ec, built on the same machine with `-DGGML_NATIVE=ON`, same files, 32 and
16 threads (the better one is shown; 32 was better every time).
**Reproduce:** `bench/run_bench.sh` (prompts: `bench/prompts_single.txt`, `bench/prompts_bench.txt`). Every
tern line is one run over 6 prompts of 160 tokens (31B: 4 prompts of 120); in brackets the slowest and the
fastest prompt. Raw logs: [logs/](logs) (`vm-2026-11-12-main.txt` for E2B, E4B and 26B-A4B, `-supplement.txt`
for the 31B, the 32-sequence lines and the time per part of a pass).

## One stream, greedy decoding (tokens per second)

| Model | llama.cpp | tern, exact output layer | tern, fast output layer | tern fast / llama.cpp |
|---|---|---|---|---|
| E2B | 80.7 | 96.9 (95.3 - 97.8) | 118.4 (116.8 - 119.1) | 1.47 |
| E4B | 42.0 | 53.8 (51.0 - 55.9) | 62.4 (58.6 - 64.2) | 1.49 |
| 26B-A4B | 31.8 | 54.6 (52.4 - 56.0) | 63.4 (56.6 - 65.5) | 1.99 |
| 31B | 5.76 | 10.5 (10.3 - 10.7) | 11.4 (11.2 - 11.5) | 1.98 |

The fast output layer returns the same tokens (below). llama.cpp: `llama-bench -p 64 -n 160 -r 2` (31B: 120).

## One stream with the model's drafter (same text, fewer passes)

| Model | llama.cpp, 3 guesses | tern, exact layer: 2 / 3 guesses | tern, fast layer: 2 / 3 guesses | tokens per pass (2 / 3) |
|---|---|---|---|---|
| E2B | 83.7 | 116.8 / 122.4 | 135.1 / 133.3 | 1.85 / 2.01 |
| E4B | 55.0 | 63.6 / 69.0 | 73.8 / 72.3 | 1.94 / 2.14 |
| 26B-A4B | 39.7 | 59.2 / 57.0 | 63.7 / 61.2 | 2.25 / 2.66 |
| 31B | 12.9 | 18.9 / 18.5 | 19.4 / 19.4 | 2.44 / 2.84 |

llama.cpp: `llama-server --spec-type draft-mtp --spec-draft-n-max 3`, same prompts; it accepted 468 of 1,402
(E2B), 505 of 1,310 (E4B), 583 of 1,077 (26B-A4B) and 310 of 492 (31B) drafted tokens. Per prompt the speed
with a drafter varies by a factor of up to 2 (how predictable the text is). For the 26B-A4B the drafter adds
nothing to the fast layer: every guessed token routes to its own experts.

## Many sequences per pass (total tokens per second)

| Model | 8 sequences: llama.cpp | tern, exact | tern, fast layer | 32 sequences: llama.cpp | tern |
|---|---|---|---|---|---|
| E2B | 247 | 421 | 433 | 444 | 674 - 694 |
| E4B | 139 | 223 | 240 | 231 | 368 - 374 |
| 26B-A4B | 98 | 131 | 140 | 189 | 252 - 253 |
| 31B | 26.0 | 38.1 | 39.1 | 41.1 | 53.6 |

The two programs do not do the same work here: `llama-batched-bench` generates 64 tokens (31B: 48) for
identical sequences of 32 prompt tokens; tern runs 64 real prompts (31B: 32), replaces a sequence when it
ends, and the number counts generation time only. With 32 sequences tern uses the exact output layer.

## Where the time of a pass goes (one stream, one prompt, milliseconds)

| Model | attention matrices | feed-forward | experts | per-layer inputs | attention | between the matrices | body | exact output layer | fast output layer |
|---|---|---|---|---|---|---|---|---|---|
| E2B | 0.66 | 5.16 | | 0.42 | 0.42 | 1.74 | 8.40 | 2.04 | 0.65 |
| E4B | 1.50 | 9.90 | | 0.71 | 0.65 | 3.02 | 15.78 | 3.63 | 0.90 |
| 26B-A4B | 2.75 | 1.56 | 5.77 | | 1.14 | 3.22 | 14.45 | 3.47 | 1.13 |
| 31B | 19.75 | 52.13 | | | 6.34 | 8.03 | 86.24 | 6.70 | 1.45 |

(`TERN_PROF=1`.) The fast output layer keeps, next to the model, a sketch of 67 MB (E2B), 101 MB (E4B,
26B-A4B) or 185 MB (31B) and 9,027 to 13,633 exact rows; building it at load takes 0.10 - 0.33 s.

## What the VM does to the numbers

- **Memory is not one speed.** Buffers of the same size are read by the 32 threads at about 105 or about
  240 GB/s, depending on which part of the machine's memory they receive: three 16 GB buffers on huge pages,
  held together: 242, 105 and 104 GB/s (`bench/burst.c`; more in DESIGN.md, section 8). Memory on 4 KB pages
  was read at 170 - 265 GB/s in every measurement, so tern puts the weights of E2B, E4B and the 26B-A4B there;
  the 31B and runs with more than 8 sequences use huge pages. Same build, page size forced, one stream, exact / fast
  layer: E2B 91.2 / 104.7 on huge pages, 96.9 / 118.4 on small ones; E4B 50.3 / 59.3 and 53.8 / 62.4;
  26B-A4B 52.3 / 61.7 and 54.6 / 63.4; 31B 10.5 / 11.4 and 10.8 / 10.9.
- **The 31B right after a 32-sequence run is slow.** `run_bench.sh` repeats its first line at the end: E2B
  96.2 (first: 96.9), E4B 52.1 (53.8), 26B-A4B 55.2 (54.6), but 31B 5.2 (10.5); in the three earlier runs of
  the day 4.9, 5.1 and 8.2. The line follows two runs with 32 sequences; three single runs of the 31B started
  directly after such a run gave 7.2, 7.3 and 7.4 tok/s, runs at other times 10.9 - 12.4. llama.cpp, measured right after, is not
  affected (5.76). We read this as the memory effect above: the new process receives the blocks the large
  one just returned.
- Between runs an hour apart, the same line moved by 5 - 10 %.

## Machine B: desktop, E2B, 6 threads (best of three rounds)

| | llama.cpp | tern, exact output layer | tern, fast output layer |
|---|---|---|---|
| One stream | 16.5 | 18.8 | 25.1 |
| One stream with the drafter: 2 / 3 guesses | - / 29.5 | 34.5 / 35.6 | 41.9 / 38.7 |
| 8 sequences per pass (total) | 63 | 96 | 104 |
| 32 sequences per pass (total) | 100 | 137 | |

Raw log: `logs/desktop-2026-11-12.txt` (three rounds of `bench/run_bench.sh`, 3 prompts of 100 tokens; the
table has the best round of every line; the fast output layer agreed with the exact one in 2,397 of 2,397
steps in each round). The machine was busy: the rounds differ by up to a third for both programs (tern, exact
layer: 18.8, 12.0, 17.0; llama.cpp: 14.4, 14.9, 16.5). Three rounds on a quieter hour, with the build before
the page rule and their log not kept, gave llama.cpp 20.6 and tern 20.2 / 23.7 for one stream, 29.4 against
34.4 / 40.1 with the drafter, 66 against 111 / 122 for 8 sequences and 111 against 144 for 32. Memory
bandwidth of this machine is 41 GB/s: one stream is bound by it (1.6 GB per token with the exact output
layer, 1.2 GB with the fast one), which is why the engines are close until the output layer and the drafter
come in.

## Is the output still the model's output?

1. **Body and output layer vs the exact float model** (`tests/logit_drift.py`: the engine's greedy generation
   is replayed through `tests/ref_gemma4.py`, an independent NumPy implementation with dequantized weights and
   float64 activations):

   | Model | Steps | Same best token | Mean logit error of the chosen token |
   |---|---|---|---|
   | E2B (Google QAT file) | 96 | 95 (99.0 %) | 0.05 |
   | E2B (Unsloth Q4_0 file: Q4_K output matrix, float per-layer inputs) | 64 | 63 (98.4 %) | 0.13 |
   | E4B | 48 | 48 (100 %) | 0.04 |
   | 26B-A4B | 32 | 32 (100 %) | 0.09 |
   | 31B | 24 | 24 (100 %) | 0.03 |

   The two differing steps are near-ties (logits 18.13 vs 18.10; 11.55 vs 11.47). The differences come from
   int8 activations, the same approximation llama.cpp makes.
   A long generation (E2B, Unsloth file, 760 tokens after a 30-token prompt, i.e. well past the 512-token
   sliding window): 394 of the first 400 steps (98.5 %) and 236 of the steps 520-760 (98.3 %) have the float
   model's best token; mean logit error 0.10 and 0.08.
   The engine changed after these runs (steps spread over the threads, read-ahead, page size); every change
   was checked to produce the same token ids as the build before it, for E2B, 26B-A4B and 31B, one stream and
   8 sequences.
2. **Tokenizer and greedy tokens vs llama.cpp** (`tests/compare_ref.py`, E2B Unsloth file): tokenizer identical
   on 17 of 17 test strings (Hebrew, German, Arabic, emoji, CJK, code, whitespace and newline runs, control
   tokens); greedy tokens identical for 4 of 6 prompts (11-40 tokens each). The other two part at steps 36 and
   18; at step 18 the float reference has 6.15 / 5.50 for the two candidates, tern 6.21 / 5.59, llama.cpp a
   near-tie with the order reversed: there tern is the one that follows the model.
3. **Speculative decoding** (drafter): byte-identical text with and without it, for E2B, 26B-A4B and 31B
   (100-120 tokens each), also together with the fast output layer. On the same prompt tern confirms 60 of
   177 guesses, llama.cpp 59 of 175.
4. **Fast output layer vs the exact one** (`--head-check`: both layers get the same hidden states, the
   generation follows the exact one). Plans made from text generated for `tools/make_prompts.py` prompts,
   checked on the prompts of `bench/prompts_bench.txt`, 100 tokens each:

   | Model | Steps | Same token as the exact layer |
   |---|---|---|
   | E2B | 15,395 | 15,395 |
   | E4B | 15,384 | 15,384 |
   | 26B-A4B | 15,480 | 15,480 |
   | 31B | 6,106 | 6,106 |

   On the desktop: 4,584 of 4,584 (AVX-512 build) and 4,587 of 4,587 (AVX2 build) with a plan made there from
   120 other prompts; 3,559 of 3,559 with the plan from the VM; 72 of 72 with the plain C build. Sampling
   with a seed gives the same text with either layer. The layer this one replaced, which ranked tokens by a
   low-rank fit of the logits, agreed in 99.2 % (E2B), 97.2 % (26B-A4B) and 89.3 % (31B) of steps.

## Not measured / limits of these numbers

- Two machines, both virtual, one compiler each. Text quality was not scored with a benchmark; the claim is
  agreement with the reference computations above.
- Contexts beyond ~800 tokens were not benchmarked for speed; attention cost grows with context length.
- The plans of the fast output layer were made and checked on generations from prompts in 12 languages
  (general questions, explanations, code, stories). Its agreement on other material is not measured;
  `--head-check` measures it on any prompt file.
- Speculative decoding is implemented for one stream and greedy decoding.
- Images and audio are not supported. Linux only (WSL 2 works); no ARM kernels for this layout yet.
