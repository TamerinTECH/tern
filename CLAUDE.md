# tern - guide for agents and contributors

`tern` is a CPU inference engine for Gemma 4 text models, written from scratch in C (about 3,800 lines, no
dependencies beyond libc and pthreads). Developed by TamerinTECH GmbH, MIT licence ([LICENSE](LICENSE)).
Humans start with [README.md](README.md). The reasoning behind every design choice is in
[docs/DESIGN.md](docs/DESIGN.md), every number in [docs/RESULTS.md](docs/RESULTS.md) with the raw logs in
`docs/logs/`. Notes about private machines (addresses, paths) live in `CLAUDE.local.md`, which git ignores.

## Build and check

    make                      # build/tern for this CPU (-march=native); needs gcc on Linux or WSL 2
    make AVX2=1               # build-avx2/tern: the AVX2 code path on a machine that also has AVX-512
    make PORTABLE=1           # build-portable/tern: plain C kernels
    make test                 # kernel self-test against double arithmetic, no model needed
    make secrets              # no credentials, addresses or logins in the files of the repository
    make benchtools           # bench/: memory bandwidth, kernel speed, thread-pool cost, reads in short regions

A checkout on a Windows drive is built from WSL (`make BUILD=$HOME/tern/build` keeps the objects off the
Windows file system). Model files are downloaded (README.md, "Get a model"), never committed.

## What must stay true

1. **Gemma 4 only.** The owner's instruction for this project: no Gemma 3 code paths, tests or numbers. The
   one exception is history: `tools/research/` keeps the studies of the first version.
2. **A slot's result does not depend on the other slots of a pass.** Every weight row is evaluated in column
   chunks of a fixed size (`tern_chunk_blocks` in `src/kernels.c`), attention visits cached rows in position
   order, pieces of a split step are multiples of the kernel's group of 512 values. Speculative decoding and
   batches rely on it. Check after any change to kernels or to the pass: same `--ids` output alone and with
   `-f ... -np 8`, and byte-identical text with and without `--assistant`.
3. **The engine computes the model.** `tests/logit_drift.py` replays a generation through an independent
   float implementation (`tests/ref_gemma4.py`); run it when numerics change (it needs gguf-py from a
   llama.cpp checkout and NumPy with a real BLAS). A change that is meant to be exact must give the same
   token ids as the build before it (`--ids`), for E2B, 26B-A4B and 31B.
4. **The fast output layer returns the exact layer's token.** `--head PLAN --head-check -f prompts.txt -np 32
   --quiet` counts agreement on any prompts. Anything below 100 % on a few thousand steps is a finding.
5. **Every number in the docs comes from a run that is described there.** State machine, files, command and
   what was not measured; keep the raw log in `docs/logs/`. Speeds on shared or virtual machines move: print
   the per-prompt range (`bench/run_bench.sh` does) and repeat a line before trusting a difference of 5 %.
6. **One tern process per machine at a time.** Its threads are pinned and wait by spinning; two processes on
   the same cores crawl, and a benchmark taken next to another job is worthless.
7. **Nothing private in git.** No tokens, no model files, no machine addresses or logins: `make secrets`
   before every commit. Commit and push only when the owner asks.

## Where things are

    src/tern.h       the whole internal interface
    src/kernels.c    packed-row dot products (AVX-512 VNNI, AVX2, plain C), activation quantization, GELU,
                     attention helpers. Row layout and its reasons: docs/DESIGN.md section 3
    src/model.c      loading and repacking, the forward pass over "slots", key/value cache, experts and their
                     router, exact output layer, the drafter ("assistant"), read-ahead, page-size rule
    src/head.c       fast output layer: sketch built at load, scan, candidates, core rows; plan making
    src/main.c       command line, chat template, generation loop with speculation, batch mode
    src/pool.c       thread pool for hundreds of short parallel regions per pass (spinning workers, pinning)
    src/tokenizer.c  Gemma 4 tokenizer    src/quant.c  GGUF row formats    src/gguf.c  GGUF reader
    src/draft.c      guesses by copying from the context or from an n-gram table (alternative to the drafter)
    plans/           plans of the fast output layer for Google's four QAT files
    tests/           kern_test.c, ref_gemma4.py (float reference), logit_drift.py, compare_ref.py (vs a
                     running llama.cpp server), check_no_secrets.py
    bench/           run_bench.sh (the speed tables), membw_x86.c, burst.c, kernbench.c, poolbench.c,
                     vnni_probe.c, prompt sets; membw.c is the ARM twin from the first version
    tools/           make_prompts.py (varied prompts in 12 languages), build_draft.py (n-gram table);
                     research/head_candidates.py is the study behind the fast output layer, the other files
                     in research/ are studies from the first version (Gemma 3 270M on a Raspberry Pi 5)

## How a change is checked

1. `make test` in the three builds (`make`, `make AVX2=1`, `make PORTABLE=1`).
2. Token ids against the build before the change: `tern -m M -p "..." -n 120 --ids` for E2B, 26B-A4B, 31B; text
   of `-f bench/prompts_test.txt -np 8 -n 40`.
3. With the drafter and the fast layer: same text as without (`--assistant`, `--head`), `--head-check`.
4. Speed: `bench/run_bench.sh` on an idle machine; `TERN_PROF=1` prints where the time of a pass goes.
5. Update `docs/RESULTS.md` and `docs/DESIGN.md` in the same change, with the measurement that justifies it.
   What was tried and did not pay goes to DESIGN.md section 9.

## Environment switches

`TERN_PROF=1` time per part of a pass (and of the fast output layer); `TERN_RA=0|1|2` read-ahead off, inside
parallel regions, by waiting workers (default 2); `TERN_HUGE=1|0` weights on huge or on 4 KB pages (default: a
rule in `model_load`, docs/DESIGN.md section 8); `TERN_PIN=0` no thread pinning; `TERN_SWEEP=1` time only the
weight matrices; `TERN_HEADBENCH=1` time the scan kernel of the fast output layer; `TERN_VERBOSE=1` with
`--head-check`: print the steps where the layers differ.

## State (2026-11-12)

Runs E2B, E4B, 26B-A4B and 31B from the published GGUF files on x86-64 Linux; validated against the float
reference for all four; faster than llama.cpp commit 2cbe0ec by a factor of 1.2 to 2.0 for one stream on the
benchmark VM (RESULTS.md). Both test machines are virtual; bare metal was never measured.

## Not done

- ARM NEON kernels for the current row layout (the plain C kernels run there, slowly), Windows-native build.
- Placement of the weights by measurement on machines whose memory nodes are hidden (DESIGN.md section 8);
  the page-size rule is fitted to one VM.
- Speculative decoding with sampling (greedy only) and for parallel sequences; choosing the number of guesses
  per pass from measured costs (2 was best on the VM, 3 on a desktop with slow memory).
- A sketch for the drafter's own output layer (151 MB per guess for the large models).
- AVX-512 kernel tuning (row alignment, joint reduction of the accumulators).
- Images and audio; an HTTP server; grammar or tool-call parsing.
