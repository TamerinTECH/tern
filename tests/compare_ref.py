#!/usr/bin/env python3
"""Compare tern with a running llama.cpp server (same GGUF): tokenizer ids and greedy tokens + top logprobs.

usage: compare_ref.py <tern binary> <model.gguf> [server url] [tokens per prompt]
       (start the server with: llama-server -m <model.gguf> --port 8791 --no-webui)
"""
import json, subprocess, sys, urllib.request

TERN, MODEL = sys.argv[1], sys.argv[2]
URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8791"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 48
TOK_STRINGS = [
    "Hello world", "  two  spaces and\ttab\n\nnewlines\n\n\n", "שלום עולם, מה שלומך?", "Übergrößenträger straße",
    "def f(x):\n    return x**2  # square", "😀 emoji 🚀 and 日本語のテキスト", "<|turn>user\nHi<turn|>\n<|turn>model\n",
    "price: $1,234.56 (approx.) 100% <b>bold</b>", "مرحبا بالعالم", "naïve café — “quotes” … and a very loooooong wooooord 1234567890",
    "\n", "\n\n\n\n", " leading space", "trailing space ", "tabs\t\tand\r\nwindows newlines", "<|channel>thought\n<channel|>", "x" * 70,
]
PROMPTS = [
    "Why is the sky blue? Answer in three sentences.",
    "Write a Python function that checks whether a number is prime.",
    "מהי בירת צרפת?",
    "Translate to German: The weather is nice today and I would like to go for a walk.",
    "List five fruits.",
    "Explain what a transformer neural network is.",
]

def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def chat(p):
    return f"<|turn>user\n{p}<turn|>\n<|turn>model\n"

bad = 0
for s in TOK_STRINGS:
    ref = post("/tokenize", {"content": s, "add_special": True, "parse_special": True})["tokens"]
    out = subprocess.run([TERN, "-m", MODEL, "-p", s, "--raw", "--tokenize"], capture_output=True, text=True).stdout.split()
    ok = [int(x) for x in out] == ref
    bad += not ok
    if not ok: print("TOKENIZER DIFF", repr(s), "\n  ref ", ref, "\n  tern", out)
print(f"tokenizer: {len(TOK_STRINGS) - bad}/{len(TOK_STRINGS)} identical")

for p in PROMPTS:
    r = post("/completion", {"prompt": chat(p), "n_predict": N, "temperature": 0, "n_probs": 5, "cache_prompt": False,
                             "samplers": ["temperature"], "post_sampling_probs": False})
    ref = r["completion_probabilities"]
    out = subprocess.run([TERN, "-m", MODEL, "-p", p, "-n", str(N), "--ids", "--top", "5"], capture_output=True, text=True).stdout.strip().split("\n")
    same, maxdp, first_diff = 0, 0.0, None
    for i, (line, rp) in enumerate(zip(out, ref)):
        f = line.split()
        tid = int(f[0])
        top = [(int(a.split(":")[0]), float(a.split(":")[1])) for a in f[1:]]
        # tern prints logits; turn the top-5 into differences from the best to compare with logprob differences
        rtop = {t["id"]: t["logprob"] for t in rp["top_logprobs"]}
        rbest = max(rtop.values())
        for t, lg in top:
            if t in rtop:
                maxdp = max(maxdp, abs((lg - top[0][1]) - (rtop[t] - rbest)))
        if tid != rp["id"]:
            first_diff = (i, tid, rp["id"], top[:2], sorted(rtop.items(), key=lambda kv: -kv[1])[:2])
            break
        same += 1
    n = min(len(out), len(ref))
    print(f"{p[:44]!r:48} same greedy tokens: {same}/{n}  max |logit gap diff| over top-5: {maxdp:.3f}" + (f"  first diff: {first_diff}" if first_diff else ""))
