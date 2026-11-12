# Raw logs

The output of the runs that the tables in [../RESULTS.md](../RESULTS.md) and [../DESIGN.md](../DESIGN.md) are
taken from, as the programs printed it. One change was made to every file: the home directory of the
benchmark machine is written `~`.

| File | What it is |
|---|---|
| `vm-2026-11-12-main.txt` | `bench/run_bench.sh` for E2B, E4B, 26B-A4B and 31B on the 32-core VM, tern and llama.cpp. All weights on 4 KB pages (the page rule did not exist yet); the lines `TERN_HUGE=1` are the same build on huge pages. RESULTS.md takes E2B, E4B and 26B-A4B up to 8 sequences from here |
| `vm-2026-11-12-supplement.txt` | the same build with the page rule: 32 sequences for E2B, E4B, 26B-A4B (twice each), the full script for the 31B, then the time per part of a pass (`TERN_PROF=1`) for all four |
| `vm-2026-11-12-memory.txt` | peak memory and load time per model and mode (README.md, "What you need") |
| `vm-2026-11-12-agreement.txt` | plans made from recorded states, then fast output layer against exact on other prompts, 160 prompts per model (64 for the 31B) |
| `vm-2026-11-12-examples.txt` | the commands and output shown in README.md |
| `vm-2026-11-12-memory-blocks.txt` | a 32-sequence run of the 31B, then three 16 GB buffers held together (`bench/burst.c`); again, then three single runs of the 31B |
| `desktop-2026-11-12.txt` | `bench/run_bench.sh` for E2B on the desktop, three rounds |
| `study-<model>-<set>.txt` | `tools/research/head_candidates.py`: the study behind the fast output layer (DESIGN.md, section 6). Set a: kinds of score (26B-A4B only); b: number of directions, scale, precision; c: which rows fail; d: rows that one bit per weight cannot hold |

In the script's lines for llama.cpp, the column before the last is the result: tokens per second for
`llama-bench`, and `S_TG t/s` (generation only) for `llama-batched-bench`.
