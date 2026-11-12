#!/usr/bin/env python3
"""How far are the engine's logits from the exact float model? Runs tern on a few prompts, then the NumPy
float reference (tests/ref_gemma4.py) on the same token sequence, and compares the best tokens and their logits.
usage: logit_drift.py <tern binary> <model.gguf> <gguf-py dir> [tokens per prompt] [number of prompts]"""
import subprocess, sys, os
TERN, MODEL, GGUFPY = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 48
NP = int(sys.argv[5]) if len(sys.argv) > 5 else 5
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = ["Why is the sky blue? Answer in three sentences.", "Write a Python function that checks whether a number is prime.",
           "Explain what a transformer neural network is.", "Erkläre die Photosynthese in einfachen Worten.", "List five fruits and their colors."][:NP]
tot = same = 0; d1 = []; dg = []
for p in PROMPTS:
    pt = subprocess.run([TERN, "-m", MODEL, "-p", p, "--tokenize"], capture_output=True, text=True).stdout.split()
    out = subprocess.run([TERN, "-m", MODEL, "-p", p, "-n", str(N), "--ids", "--top", "5"], capture_output=True, text=True).stdout.strip().split("\n")
    gen = [l.split()[0] for l in out]
    r = subprocess.run([sys.executable, os.path.join(HERE, "ref_gemma4.py"), MODEL, GGUFPY] + pt + gen[:-1], capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
    ref = r.stdout.strip().split("\n")[len(pt) - 1:]
    ps = pd = 0
    for i, (lt, lr) in enumerate(zip(out, ref)):
        t = [(int(a.split(":")[0]), float(a.split(":")[1])) for a in lt.split()[1:]]
        rr = [(int(a.split(":")[0]), float(a.split(":")[1])) for a in lr.split()]
        tot += 1; ok = t[0][0] == rr[0][0]; same += ok; ps += ok; pd += 1
        d1.append(abs(t[0][1] - dict(rr).get(t[0][0], rr[0][1])))
        rd = dict(rr)
        if t[1][0] in rd and t[0][0] in rd: dg.append(abs((t[0][1] - t[1][1]) - (rd[t[0][0]] - rd[t[1][0]])))
        if not ok and "-v" in sys.argv: print(f"  step {i}: engine {t[:2]} reference {rr[:2]}")
    print(f"{p[:50]!r:54} same best token {ps}/{pd}", flush=True)
print(f"{tot} steps: same best token as the float model {same}/{tot} ({100 * same / tot:.1f}%), "
      f"mean |logit error| of the chosen token {sum(d1) / len(d1):.3f}, mean |error of the top-1/top-2 gap| {sum(dg) / max(1, len(dg)):.3f}")
