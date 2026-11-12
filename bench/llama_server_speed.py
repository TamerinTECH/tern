#!/usr/bin/env python3
"""Generation speed of a running llama.cpp server over a prompt file (one prompt per line, Gemma 4 chat format,
greedy decoding). Prints tokens/s summed over the prompts and, if the server drafts, its acceptance numbers.
usage: llama_server_speed.py <server url> <prompts.txt> <tokens per prompt>"""
import json, sys, urllib.request

url, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = sec = drafted = accepted = 0
for line in open(path, encoding="utf-8"):
    p = line.strip()
    if not p: continue
    body = {"prompt": "<|turn>user\n" + p + "<turn|>\n<|turn>model\n", "n_predict": n, "temperature": 0, "cache_prompt": False, "samplers": ["temperature"]}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=3600))["timings"]
    tok += t["predicted_n"]; sec += t["predicted_ms"] / 1e3
    drafted += t.get("draft_n", 0); accepted += t.get("draft_n_accepted", 0)
extra = f", {accepted} of {drafted} drafted tokens accepted" if drafted else ""
print(f"{tok / sec:.1f} tok/s  ({tok} tokens{extra})")
