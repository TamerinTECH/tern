#!/usr/bin/env python3
"""Draft-predictor study: how well can cheap functions guess the model's token k steps ahead?

Given the final hidden state h_t (which chose x_{t+1}) the model needs a whole pass to get h_{t+1}.
We fit cheap predictors of h_{t+1} (and further) from what is already known, and count how often the output
layer applied to the predicted state gives the token the real model chooses (the acceptance rate).

usage: draft_study.py <model.gguf> <gguf-py dir> <train.bin> <test.bin>
"""
import sys, time
import numpy as np

sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
from gguf.quants import dequantize

EOT, MAXN = 106, 200

def load(path, E=640):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // (4 + 4 * E)
    raw = raw[: n * (4 + 4 * E)].reshape(n, 4 + 4 * E)
    tok = raw[:, :4].copy().view(np.int32)[:, 0]; hid = raw[:, 4:].copy().view(np.float32).reshape(n, E)
    seq = np.zeros(n, dtype=np.int32); s = 0; c = 0    # sequence id per record
    for i in range(n):
        seq[i] = s; c += 1
        if tok[i] == EOT or tok[i] == 1 or c == MAXN: s += 1; c = 0
    return tok, hid, seq

r = GGUFReader(sys.argv[1])
t = {x.name: x for x in r.tensors}["token_embd.weight"]
Emb = dequantize(t.data, t.tensor_type).astype(np.float32)
tok_tr, H_tr, s_tr = load(sys.argv[3]); tok_te, H_te, s_te = load(sys.argv[4])
print(f"train {len(tok_tr)} steps in {s_tr.max() + 1} sequences; test {len(tok_te)} steps in {s_te.max() + 1} sequences")

def pairs(tok, H, seq, depth):
    """rows i where records i..i+depth are in the same sequence: features use h_i and tokens x_{i+1..i+depth}; target h_{i+depth}"""
    n = len(tok) - depth
    ok = seq[:n] == seq[depth:depth + n]
    return np.nonzero(ok)[0]

def feats(tok, H, idx, depth, hist):
    parts = [H[idx - j] if j == 0 else np.where((idx - j >= 0)[:, None], H[np.maximum(idx - j, 0)], 0) for j in range(hist)]
    for d in range(depth): parts.append(Emb[tok[idx + d]] * 25.0)     # tok[i] is the token chosen from h_i, i.e. x_{i+1}
    parts.append(np.ones((len(idx), 1), dtype=np.float32))
    return np.concatenate(parts, axis=1)

def argmax_tokens(Hp):
    out = np.empty(len(Hp), dtype=np.int32)
    for i in range(0, len(Hp), 1024): out[i:i + 1024] = (Hp[i:i + 1024] @ Emb.T).argmax(1)
    return out

for depth in (1, 2, 3):
    for hist in (1, 2):
        itr = pairs(tok_tr, H_tr, s_tr, depth); ite = pairs(tok_te, H_te, s_te, depth)
        ite = ite[np.random.RandomState(0).choice(len(ite), min(3000, len(ite)), replace=False)]
        X = feats(tok_tr, H_tr, itr, depth, hist).astype(np.float64); Y = H_tr[itr + depth].astype(np.float64)
        best = None
        for lam in (1e-2, 1e-1, 1.0):
            A = X.T @ X; A += lam * np.trace(A) / len(A) * np.eye(len(A))
            Wt = np.linalg.solve(A, X.T @ Y)
            Hp = (feats(tok_te, H_te, ite, depth, hist) @ Wt).astype(np.float32)
            acc = np.mean(argmax_tokens(Hp) == tok_te[ite + depth])
            if best is None or acc > best[0]: best = (acc, lam)
        # baseline: most frequent follower of the previous token (bigram table from train)
        print(f"depth {depth} (guess x_t+{depth + 1}), {hist} past state(s): linear predictor acceptance {best[0]:.4f} (ridge {best[1]})", flush=True)
# bigram baseline for depth 1
from collections import Counter, defaultdict
nxt = defaultdict(Counter)
i1 = pairs(tok_tr, H_tr, s_tr, 1)
for a, b in zip(tok_tr[i1], tok_tr[i1 + 1]): nxt[int(a)][int(b)] += 1
ite = pairs(tok_te, H_te, s_te, 1)
hit = sum(1 for a, b in zip(tok_te[ite], tok_te[ite + 1]) if int(a) in nxt and nxt[int(a)].most_common(1)[0][0] == int(b))
print(f"bigram table baseline (depth 1): acceptance {hit / len(ite):.4f}")
