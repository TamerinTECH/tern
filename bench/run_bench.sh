#!/usr/bin/env bash
# Speed of tern and (optionally) llama.cpp for one Gemma 4 model on this machine.
# usage: bench/run_bench.sh <model.gguf> [drafter.gguf|-] [head plan|-] [llama.cpp build/bin dir|-] [threads]
# Run on an idle machine. Prompts: bench/prompts_single.txt (one stream) and bench/prompts_bench.txt (parallel).
# A head plan is made with: tern -m model.gguf -f prompts.txt -np 32 -n 100 --quiet --make-head model.head
# Environment: NGEN (tokens per prompt, one stream), NPROMPTS, SPEC (guesses per pass to try), NPAR (parallel
# sequence counts), NPARPROMPTS, NPARGEN, CHECKPROMPTS (prompts for the agreement check of the fast output layer).
set -u
MODEL=$1; ASST=${2:--}; PLAN=${3:--}; LLAMA=${4:--}; T=${5:-$(nproc)}
HERE=$(cd "$(dirname "$0")" && pwd)
TERN=${TERN:-$HERE/../build/tern}
NGEN=${NGEN:-160}; NPROMPTS=${NPROMPTS:-6}; NPAR=${NPAR:-"8 32"}; NPARPROMPTS=${NPARPROMPTS:-64}; NPARGEN=${NPARGEN:-64}; SPEC=${SPEC:-"2 3"}
SINGLE=$HERE/prompts_single.txt; MANY=$HERE/prompts_bench.txt
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
head -n "$NPROMPTS" "$SINGLE" > "$TMP/single.txt"; head -n "$NPARPROMPTS" "$MANY" > "$TMP/many.txt"

echo "== machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) logical cpus, $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo)"
echo "== model: $(basename "$MODEL")  threads: $T"

# one stream: every prompt on its own, tokens and seconds summed (the load time is not counted)
tern_single() {   # extra args
  local tok=0 sec=0 passes=0 line n ps rates="" range
  while IFS= read -r p; do
    line=$("$TERN" -m "$MODEL" -t "$T" -p "$p" -n "$NGEN" "$@" 2>&1 >/dev/null | grep -E "generation|speculation")
    n=$(echo "$line" | sed -n 's/.*generation \([0-9]*\) tokens, \([0-9.]*\) tok\/s.*/\1 \2/p')
    tok=$(echo "$tok $n" | awk '{print $1 + $2}'); sec=$(echo "$sec $n" | awk '{print $1 + $2 / $3}')
    ps=$(echo "$line" | sed -n 's/.*speculation: \([0-9]*\) passes.*/\1/p'); passes=$((passes + ${ps:-0}))
    rates="$rates ${n#* }"
  done < "$TMP/single.txt"
  # slowest and fastest prompt: a wide range without the drafter means that something disturbed the machine
  range=$(echo $rates | tr ' ' '\n' | sort -g | sed -n '1p;$p' | tr '\n' ' ' | awk '{printf "%.1f - %.1f", $1, $2}')
  if [ "$passes" -gt 0 ]; then echo "$tok $sec $passes $range" | awk '{printf "%.1f tok/s  (%d tokens, %.2f tokens per pass; per prompt %s - %s)\n", $1 / $2, $1, $1 / $3, $4, $6}'
  else echo "$tok $sec $range" | awk '{printf "%.1f tok/s  (%d tokens; per prompt %s - %s)\n", $1 / $2, $1, $3, $5}'; fi
}
tern_par() {      # np, extra args
  local np=$1; shift
  "$TERN" -m "$MODEL" -t "$T" -f "$TMP/many.txt" -np "$np" -n "$NPARGEN" --quiet "$@" 2>&1 | sed -n 's/.*generation alone \([0-9.]*\) tok\/s.*/\1 tok\/s total/p'
}
echo "-- tern, one stream, exact output layer:               $(tern_single)"
[ "$PLAN" != "-" ] && echo "-- tern, one stream, fast output layer:                $(tern_single --head "$PLAN")"
if [ -n "${ALT_HUGE:-}" ]; then      # the same with the page size for the weights forced (TERN_HUGE=1: huge pages, 0: 4 KB pages; docs/DESIGN.md section 8)
  echo "-- tern, one stream, exact layer, TERN_HUGE=$ALT_HUGE:         $(TERN_HUGE=$ALT_HUGE tern_single)"
  [ "$PLAN" != "-" ] && echo "-- tern, one stream, fast layer, TERN_HUGE=$ALT_HUGE:          $(TERN_HUGE=$ALT_HUGE tern_single --head "$PLAN")"
fi
if [ "$ASST" != "-" ]; then
  for k in $SPEC; do
    echo "-- tern, one stream, drafter ($k guesses), exact layer:  $(tern_single --assistant "$ASST" --spec "$k")"
    [ "$PLAN" != "-" ] && echo "-- tern, one stream, drafter ($k guesses), fast layer:   $(tern_single --assistant "$ASST" --spec "$k" --head "$PLAN")"
  done
fi
for np in $NPAR; do
  echo "-- tern, $np sequences in parallel, exact output layer: $(tern_par "$np")"
  # with more than 8 sequences per pass tern uses the exact layer anyway (the fast one saves memory traffic, not arithmetic)
  [ "$PLAN" != "-" ] && [ "$np" -le 8 ] && echo "-- tern, $np sequences in parallel, fast output layer:  $(tern_par "$np" --head "$PLAN")"
done
if [ "$PLAN" != "-" ]; then
  head -n "${CHECKPROMPTS:-160}" "$MANY" > "$TMP/check.txt"
  echo "-- fast output layer vs exact, same hidden states:     $("$TERN" -m "$MODEL" -t "$T" -f "$TMP/check.txt" -np 32 -n 100 --quiet --head "$PLAN" --head-check 2>&1 | sed -n 's/.*same token as the exact layer: //p')"
fi

echo "-- tern, one stream, exact output layer, again:        $(tern_single)   (drift of the machine since the first line)"

if [ "$LLAMA" != "-" ]; then
  for t in $T ${LLAMA_ALT_T:-}; do      # llama.cpp is sometimes faster with fewer threads: LLAMA_ALT_T adds a second run
    echo "-- llama.cpp, one stream (llama-bench, $t threads):"
    "$LLAMA/llama-bench" -m "$MODEL" -t "$t" -p 64 -n "$NGEN" -r 2 2>/dev/null | grep -E "tg$NGEN" | sed 's/^/     /'
  done
  if [ "$ASST" != "-" ] && [ -x "$LLAMA/llama-server" ]; then
    for k in ${LLAMA_SPEC:-3}; do
      "$LLAMA/llama-server" -m "$MODEL" -md "$ASST" --spec-type draft-mtp --spec-draft-n-max "$k" -t "$T" -c 4096 --port 8791 --no-webui > "$TMP/srv.log" 2>&1 &
      SRV=$!
      for i in $(seq 1 300); do curl -s http://127.0.0.1:8791/health 2>/dev/null | grep -q ok && break; sleep 1; done
      echo "-- llama.cpp, one stream, drafter ($k guesses):          $(python3 "$HERE/llama_server_speed.py" http://127.0.0.1:8791 "$TMP/single.txt" "$NGEN")"
      kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    done
  fi
  echo "-- llama.cpp, parallel sequences (llama-batched-bench: 32 prompt tokens, $NPARGEN generated per sequence):"
  "$LLAMA/llama-batched-bench" -m "$MODEL" -c 16384 -b 2048 -ub 512 -npp 32 -ntg "$NPARGEN" -npl "$(echo $NPAR | tr ' ' ',')" -t "$T" 2>/dev/null | grep -E "^\|" | sed 's/^/     /'
fi
