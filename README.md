# tern

Developed by TamerinTECH GmbH. MIT licence ([LICENSE](LICENSE)).

A small CPU inference engine for the Gemma 4 family (text), written from scratch in C: about 3,800 lines, no
dependencies beyond libc and pthreads, no GPU. It reads the published GGUF files as they are. It was built to
find out how fast a real LLM runs on CPUs when the engine is designed around the two limits that matter:
memory bandwidth and int8 arithmetic.

This page is the guide for people: what to expect, what you need, how to run it. How it works and what was
tried and failed: [docs/DESIGN.md](docs/DESIGN.md). Every measurement with its method:
[docs/RESULTS.md](docs/RESULTS.md), raw logs in [docs/logs](docs/logs). Agents start at [AGENTS.md](AGENTS.md).

## What to expect

Tokens per second on a 32-core cloud VM (AMD EPYC Milan, AVX2, 245 GB RAM), Google's QAT Q4_0 files, against
llama.cpp (commit 2cbe0ec, same files, same machine, same threads):

| Tokens per second | E2B | E4B | 26B-A4B (experts) | 31B |
|---|---|---|---|---|
| One stream: llama.cpp | 80.7 | 42.0 | 31.8 | 5.8 |
| One stream: tern, exact output layer | 96.9 | 53.8 | 54.6 | 10.5 |
| One stream: tern, fast output layer | 118.4 | 62.4 | 63.4 | 11.4 |
| One stream with the model's drafter: llama.cpp | 83.7 | 55.0 | 39.7 | 12.9 |
| One stream with the drafter: tern, fast output layer | 135.1 | 73.8 | 63.7 | 19.4 |
| 32 sequences per pass, total: llama.cpp | 444 | 231 | 189 | 41 |
| 32 sequences per pass, total: tern | 674 - 694 | 368 - 374 | 252 | 54 |

Neither the fast output layer nor the drafter changes the text: the fast layer chose the exact layer's token
in all 52,365 steps that were checked, and text generated with the drafter is byte-identical. One run per
cell; the VM itself moves by 5 - 10 % between runs, and its memory has a peculiarity that RESULTS.md describes.

On a 6-core desktop (AMD Ryzen 5 8400F, AVX-512, dual-channel DDR5 at 3600 MT/s, WSL 2, the machine in use),
E2B, best of three rounds:

| Tokens per second, E2B | llama.cpp | tern, exact output layer | tern, fast output layer |
|---|---|---|---|
| One stream | 16.5 | 18.8 | 25.1 |
| One stream with the drafter | 29.5 | 35.6 | 41.9 |
| 8 sequences per pass, total | 63 | 96 | 104 |
| 32 sequences per pass, total | 100 | 137 | |

The rounds differ by up to a third there, for both programs, because the machine was busy; on a quieter hour
both were faster (llama.cpp 20.6, tern 20.2 and 23.7 for one stream).

**Your machine.** One stream is bound by memory: every token reads all the weights it uses once. Divide the
rate at which your machine reads memory with all cores (`make benchtools && build/membw`, the `DRAM stream`
line with all threads) by the gigabytes below; tern reached 70 to 100 % of that quotient on the two test
machines.

| GB read per token | E2B | E4B | 26B-A4B | 31B |
|---|---|---|---|---|
| exact output layer | 1.6 | 3.1 | 2.7 | 18.2 |
| fast output layer | 1.2 | 2.4 | 1.9 | 16.7 |

So a desktop that reads 40 GB/s can expect about 20 tok/s for E2B (measured: 19 to 25), and by the same
reckoning about 10 for E4B, 11 to 15 for the 26B-A4B and under 2 for the 31B, about 3 with the drafter. The
last three are estimates, not measurements: the large files were not run on the desktop. With many sequences
per pass arithmetic is the limit instead, and total throughput grows with the cores.

## What you need

- **Linux on x86-64** (WSL 2 works), gcc, make. Tested with gcc 13.3 (Ubuntu 24.04 under WSL 2) and gcc 15.2
  (Ubuntu 26.04).
- **A CPU with AVX2, FMA and F16C** (Intel since Haswell, AMD since Zen). AVX-512 with VNNI is used when the
  CPU has it. For anything else `make PORTABLE=1` builds plain C kernels: correct, and about ten times slower
  (E2B on the desktop: 1.7 tok/s against 18.7). ARM has no fast kernels yet.
- **Memory** for the weights in the engine's layout, measured as peak resident memory:

  | Model | Download | One stream, context 2,048 | With fast layer and drafter | 32 sequences, context 512 | Load time, 32 threads |
  |---|---|---|---|---|---|
  | E2B | 3.35 GB | 1.9 GB | 2.1 GB | 2.2 GB | 0.3 s |
  | E4B | 5.15 GB | 3.7 GB | 3.9 GB | 4.5 GB | 0.5 s |
  | 26B-A4B | 14.4 GB | 15.1 GB | 15.6 GB | 18.4 GB | 1.4 s |
  | 31B | 17.65 GB | 20.3 GB | 21.0 GB | 33.5 GB | 2.5 s |

  A context of 8,192 tokens (`-c 8192`) adds 0.1 GB (E2B), 0.3 GB (E4B), 1.3 GB (26B-A4B) and 5.3 GB (31B).
  E2B and E4B need less memory than their files: the per-layer token tables stay in the mapped file. While
  loading, the file is read once on top of that; the system drops those pages again when memory is short.
- **Cores.** Threads default to one per physical core. With 4 of the VM's cores (`-t 4`) one stream ran at
  40.6 (E2B), 22.6 (E4B), 24.6 (26B-A4B) and 3.8 tok/s (31B).
- Python 3 with NumPy only for the optional checks in `tests/` and the tools in `tools/`.

## Build

    git clone https://github.com/TamerinTECH/tern.git
    cd tern
    make                  # build/tern, for the CPU of this machine (-march=native)
    make test             # kernel self-test, no model needed: "all kernel checks passed"

    make AVX2=1           # build-avx2/tern: the AVX2 code path on a machine that also has AVX-512
    make PORTABLE=1       # build-portable/tern: plain C kernels, any CPU, slow

## Get a model

The files are public on Hugging Face, no account needed. `models/` is ignored by git.

    mkdir -p models && cd models
    curl -LO https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf
    curl -LO https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mtp-gemma-4-E2B-it-Q4_0.gguf   # its drafter

| Model | File (repository on huggingface.co) | Drafter (optional) |
|---|---|---|
| E2B | `gemma-4-E2B_q4_0-it.gguf` (google/gemma-4-E2B-it-qat-q4_0-gguf) | `mtp-gemma-4-E2B-it-Q4_0.gguf` (ggml-org/gemma-4-E2B-it-GGUF), 59 MB |
| E4B | `gemma-4-E4B_q4_0-it.gguf` (google/gemma-4-E4B-it-qat-q4_0-gguf) | `mtp-gemma-4-E4B-it-Q4_0.gguf` (ggml-org/gemma-4-E4B-it-GGUF), 60 MB |
| 26B-A4B | `gemma-4-26B_q4_0-it.gguf` (google/gemma-4-26B-A4B-it-qat-q4_0-gguf) | `mtp-gemma-4-26B-A4B-it-Q4_0.gguf` (ggml-org/gemma-4-26B-A4B-it-GGUF), 252 MB |
| 31B | `gemma-4-31B_q4_0-it.gguf` (google/gemma-4-31B-it-qat-q4_0-gguf) | `mtp-gemma-4-31B-it-Q4_0.gguf` (unsloth/gemma-4-31B-it-qat-GGUF, folder `MTP`), 280 MB |

Other Gemma 4 GGUF files work when their matrices are Q4_0, Q4_1 or Q4_K (the output matrix may also be
Q6_K); Unsloth's E2B Q4_0 file was tested.

## Run

The output below is copied from runs on the VM (long lines of text wrapped here); the lines after the text go
to standard error.

**One prompt.**

    $ build/tern -m models/gemma-4-26B_q4_0-it.gguf -p "Why is the sky blue? Answer in two sentences."
    The sky appears blue because sunlight is scattered in all directions by the gases and particles in Earth's
    atmosphere. This phenomenon, known as Rayleigh scattering, affects shorter wavelengths of light more than
    longer ones, causing blue light to scatter more intensely.

    load 1.41 s | prompt 20 tokens, 228.7 tok/s | generation 53 tokens, 55.75 tok/s (output layer 21% of the time) | weights 14695 MB on 4 KB pages | AVX2, 32 threads
    memory: peak 15083 MB resident (at exit: 14525 MB private + 65 MB mapped files, which the system can reclaim)

**The drafter and the fast output layer**: the same text in fewer passes and with less memory traffic.

    $ build/tern -m models/gemma-4-31B_q4_0-it.gguf --assistant models/mtp-gemma-4-31B-it-Q4_0.gguf \
          --head plans/gemma-4-31B_q4_0-it.head -p "Explain in three sentences why the sea is salty."
    Rainwater becomes slightly acidic as it falls through the air, causing it to erode rocks and release
    minerals like sodium and chloride. These dissolved salts are carried by rivers and streams into the ocean,
    where they accumulate over millions of years. Because water evaporates from the ocean but the salt remains
    behind, the concentration of salinity increases over time.

    load 2.67 s | prompt 19 tokens, 50.0 tok/s | generation 71 tokens, 19.97 tok/s (output layer 2% of the time) | weights 18235 MB on huge pages | AVX2, 32 threads
    speculation: 30 passes for 71 tokens (2.37 tokens per pass); 59 guesses, 40 confirmed (68%)
    drafter: 60 steps, 2.02 ms per step
    fast output layer: 2.547 ms per call, 31 calls, 12438 full rows per token

**Chat** (`-i`; the conversation stays in the cache; an empty line ends it).

    $ build/tern -m models/gemma-4-26B_q4_0-it.gguf --head plans/gemma-4-26B_q4_0-it.head -i
    tern chat - empty line or Ctrl-D to quit

    > What is the capital of France?
    The capital of France is **Paris**.
    [13 tokens, 70.1 tok/s]

    > And what is it famous for? One sentence.
    Paris is world-famous for its iconic landmarks like the Eiffel Tower, its renowned art museums, and its
    reputation as a global center for fashion and gastronomy.
    [36 tokens, 64.8 tok/s]

**Sampling** instead of the most likely token (`--temp`, `--top-k`, `--seed`).

    $ build/tern -m models/gemma-4-26B_q4_0-it.gguf --head plans/gemma-4-26B_q4_0-it.head \
          -p "Write a haiku about a lighthouse." --temp 0.8 --seed 3
    Tall tower of light,
    Guiding ships through stormy seas,
    Safe through the dark night.

**Many prompts**, one per line of a file, up to 32 sequences per pass: the highest total throughput.

    $ build/tern -m models/gemma-4-26B_q4_0-it.gguf -f bench/prompts_bench.txt -np 32 -n 64 --quiet
    160 prompts, 32 at a time | 10087 generated tokens in 53.6 s = 188.2 tok/s total (prompt processing 12.5 s for 3160 tokens included; generation alone 245.3 tok/s) | AVX2, 32 threads

Without `--quiet` the answers are printed as `=== 0`, `=== 1`, ... in the order of the prompts.

**Where the time goes** (`TERN_PROF=1`), here for one token of the 31B:

    body: 41 passes, 1.4 slots per pass | per pass (ms): total 82.620 = attention matrices 18.579 + feed-forward 47.606 + experts 0.000 + per-layer inputs 0.000 + attention 7.461 + other 8.974 | output layer 6.421

`build/tern` without arguments lists all options. The ones that matter most:

| Option | Meaning |
|---|---|
| `-m FILE` | the model (GGUF) |
| `-p TEXT`, `-i`, `-f FILE` | one prompt, a chat, or a file with one prompt per line |
| `-n N` | tokens to generate per prompt (128) |
| `-t N` | threads (one per physical core) |
| `-c N` | context size in tokens (2,048; 512 with `-f`) |
| `-np N` | with `-f`: sequences per pass (4, at most 32) |
| `--assistant FILE` | the model's drafter: guessed tokens are checked in the same pass; `--spec N` guesses per pass (2) |
| `--head PLAN` | the fast output layer; `--head-check` also runs the exact one and counts agreement |
| `--temp T`, `--top-k K`, `--seed N` | sampling |
| `--raw` | do not wrap the prompt in the chat template |

Rules of use: run **one tern process at a time** on a machine (its threads are pinned to the cores and wait by
spinning; two processes on the same cores crawl), and give it an otherwise idle machine when you measure.

## The fast output layer

The last step of every token multiplies the hidden state with all 262,144 rows of the output matrix: 10 to
30 % of the time of one stream. `--head PLAN` replaces that sweep by a scan of a compact copy (32 bytes plus
one bit per weight for each token) and full rows for 10,000 to 15,000 tokens
([docs/DESIGN.md](docs/DESIGN.md), section 6). A plan is 32 directions: those in which the model's final
hidden states vary most. It does not change the weights or the score of any token; it decides which rows are
read. Plans for Google's four QAT files are in `plans/`. For another model file, let the model write some
text and make one:

    $ build/tern -m models/gemma-4-E2B_q4_0-it.gguf -f bench/prompts_bench.txt -np 32 -n 100 --quiet --make-head my.head
    160 prompts, 32 at a time | 15395 generated tokens in 24.8 s = 620.4 tok/s total (...)
    plan: 32 directions from 8192 recorded steps hold 41.6% of the hidden states' energy -> my.head (192 KB)

and check it on other prompts (both layers run, the count says how often they agree):

    $ build/tern -m models/gemma-4-E2B_q4_0-it.gguf --head my.head --head-check -f bench/prompts_test.txt -np 8 -n 100 --quiet
    fast output layer: 2.370 ms per call, 800 calls, 14656 full rows per token | same token as the exact layer: 5919 of 5919 (100.000%)

The exact layer stays the default. With more than 8 sequences per pass the exact layer is used in any case:
the fast one saves memory traffic, not arithmetic.

## Is the output right?

- **Against the exact model.** `tests/ref_gemma4.py` is an independent NumPy implementation with float64
  activations. The engine's best token is the float model's best token in 95 of 96 steps (E2B), 48 of 48
  (E4B), 32 of 32 (26B-A4B) and 24 of 24 (31B); the one difference is a near-tie. The gap comes from int8
  activations, the same approximation llama.cpp makes.
- **Against llama.cpp.** The tokenizer gives the same ids on 17 of 17 test strings (Hebrew, German, Arabic, CJK,
  emoji, code, control tokens).
- **Speed-ups against the plain path.** Text with the drafter is byte-identical to text without it. The fast
  output layer chose the exact layer's token in 52,365 of 52,365 steps on prompts it had not seen.

Details and commands: [docs/RESULTS.md](docs/RESULTS.md), "Is the output still the model's output?".

    make test                                                                # kernels against plain arithmetic
    python3 tests/logit_drift.py build/tern $M /path/to/llama.cpp/gguf-py    # logits against the float reference
    python3 tests/compare_ref.py build/tern $M                               # tokenizer and tokens against a running llama.cpp server

## Measure it yourself

    bench/run_bench.sh models/gemma-4-31B_q4_0-it.gguf models/mtp-gemma-4-31B-it-Q4_0.gguf \
        plans/gemma-4-31B_q4_0-it.head /path/to/llama.cpp/build/bin 32

prints the tables above for one model: tern with the exact and the fast output layer, with the drafter, with 8
and 32 sequences, the agreement of the two output layers, and the same for llama.cpp (pass `-` instead of its
directory to leave it out). llama.cpp was built with
`cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON` at commit 2cbe0ec.

Two switches are worth a try on a new machine: `TERN_HUGE=1` and `TERN_HUGE=0` put the weights on huge pages or
on 4 KB pages. On the benchmark VM that changed the speed by up to 17 %; tern chooses by a rule fitted there
([docs/DESIGN.md](docs/DESIGN.md), section 8).

## How it gets its speed

- Weight rows are repacked once at load into a layout in which one instruction adds 64 products (AVX-512
  VNNI) and one int-to-float conversion serves 512 weights; activations are int8 as in llama.cpp.
- One pass over the weights evaluates up to 32 tokens ("slots"): a batch of sequences, a prompt chunk or a
  chain of guessed tokens. A slot's result does not depend on the others, so speculation is exact.
- Google's drafter models guess the next tokens; the main model checks them in the pass it needs anyway.
- The output layer reads a sketch instead of the whole matrix and still returns the exact token.
- Between the matrix products nothing waits: short steps are spread over the threads, and idle threads read
  the next matrix ahead.

## Limits

- Gemma 4 text models only: no images or audio, no grammar or tool-call parsing, no HTTP server.
- Linux on x86-64 (WSL 2 works). No ARM kernels yet, no Windows-native build, no GPU.
- Speculative decoding is implemented for one stream and greedy decoding.
- Both machines it was measured on are virtual; contexts beyond about 800 tokens were not benchmarked.

## Layout

    src/             the engine: kernels.c, model.c, head.c, main.c, pool.c, tokenizer.c, quant.c, gguf.c, draft.c
    plans/           plans of the fast output layer for Google's QAT files (E2B, E4B, 26B-A4B, 31B)
    tests/           kernel self-test, float reference, comparison with llama.cpp, secret scan
    bench/           run_bench.sh, memory and kernel benchmarks, prompt sets
    tools/           make_prompts.py, build_draft.py; research/ = the studies behind the design
    docs/            DESIGN.md, RESULTS.md, logs/

## Credits

The Gemma 4 models and their drafters are Google's; their own terms apply to the model files, none of which
are in this repository (`plans/` holds 32 directions per model, computed from text the models generated).
The model definition was taken from reading llama.cpp (MIT), which is also the reference for speed and for
the tokenizer; no code was copied from it. Related: [pi-gemma4](https://github.com/TamerinTECH/pi-gemma4),
Gemma 4 on a Raspberry Pi 5.
