#!/usr/bin/env bash
# llama.cpp reference speeds on this machine: one stream and parallel sequences.
# usage: bench/llama_baseline.sh <llama.cpp build/bin dir> <model.gguf> [threads]
set -u
B=$1; M=$2; T=${3:-6}
echo "== machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), $(nproc) logical cpus"
echo "== $(basename "$M"), $T threads"
echo "-- llama-bench (prompt 128 tokens, generate 128 tokens)"
"$B/llama-bench" -m "$M" -t "$T" -p 128 -n 128 -r 2 2>/dev/null | grep -E "^\|" | tail -2
echo "-- llama-batched-bench (32 prompt tokens, 128 generated per sequence)"
"$B/llama-batched-bench" -m "$M" -c 16384 -b 2048 -ub 512 -npp 32 -ntg 128 -npl 1,4,8,16,32 -t "$T" 2>/dev/null | grep -E "^\|"
