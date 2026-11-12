#!/usr/bin/env python3
"""Simulate the two-tier output layer on recorded data: a frequency-ranked core that is always scanned plus
cluster probing for the tail. usage: head_sim.py <model.gguf> <gguf-py> <train.bin> <test.bin> <cache dir> <clusters.npz>"""
import os, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
from gguf.quants import dequantize
def load(path, E=640):
    raw = np.fromfile(path, dtype=np.uint8); n = len(raw) // (4 + 4 * E)
    raw = raw[: n * (4 + 4 * E)].reshape(n, 4 + 4 * E)
    return raw[:, :4].copy().view(np.int32)[:, 0], raw[:, 4:].copy().view(np.float32).reshape(n, E)
cache = sys.argv[5]
r = GGUFReader(sys.argv[1]); t = {x.name: x for x in r.tensors}["token_embd.weight"]
Emb = dequantize(t.data, t.tensor_type).astype(np.float32); V = Emb.shape[0]
tok_tr, H_tr = load(sys.argv[3]); tok_te, H_te = load(sys.argv[4])
T_tr = np.load(os.path.join(cache, f"top50_train_{len(H_tr)}.npy")); T_te = np.load(os.path.join(cache, f"top50_test_{len(H_te)}.npy"))
cl = np.load(sys.argv[6]); a = cl["assign"]; Cm = cl["centroid"]; K = len(Cm)
truth = T_te[:, 0]
# frequency score: mentions in the top-50, weighted so that winners count most
wt = 1.0 / (1.0 + np.arange(50))
score = np.zeros(V); np.add.at(score, T_tr.ravel(), np.tile(wt, len(T_tr)))
rank = np.empty(V, dtype=np.int64); rank[np.argsort(-score, kind="stable")] = np.arange(V)
S = H_te @ Cm.T
crank = (S > S[np.arange(len(truth)), a[truth]][:, None]).sum(1)     # rank of the winner's cluster
cnt = np.bincount(a, minlength=K)
order = np.argsort(-S, axis=1)
print(f"test steps {len(truth)}; plausible tokens {(score > 0).sum()}")
print("core size | cluster probes P | found | avg rows scanned (core + probed)")
for core in (0, 2000, 4000, 8000, 16000, 32000):
    for P in (0, 32, 64, 128, 256):
        found = (rank[truth] < core) | (crank < P)
        rows = core + (cnt[order[:, :P]].sum(1).mean() if P else 0)
        print(f"  {core:6d} | {P:4d} | {found.mean():.4f} | {rows:7.0f}")

# --- dynamic pool: tokens that were among the top-50 of recent steps of the same sequence
EOT, MAXN = 106, 200
seq = np.zeros(len(tok_te), dtype=np.int32); s = 0; c = 0
for i in range(len(tok_te)):
    seq[i] = s; c += 1
    if tok_te[i] == EOT or tok_te[i] == 1 or c == MAXN: s += 1; c = 0
print("\nrecent-steps pool (union of the top-k of the previous W steps) + core + probes: found | avg pool size")
for k in (10, 50):
    for W in (8, 32, 200):
        for core, P in ((0, 0), (2000, 32), (4000, 64), (8000, 64)):
            hit = 0; n = 0; psz = 0
            for q in range(seq.max() + 1):
                idx = np.nonzero(seq == q)[0]
                for j, i in enumerate(idx):
                    if j == 0: continue          # first step of a sequence: full scan anyway
                    lo = max(0, j - W)
                    pool = np.unique(T_te[idx[lo:j], :k])
                    n += 1; psz += len(pool)
                    hit += (truth[i] in pool) or (rank[truth[i]] < core) or (crank[i] < P)
            print(f"  top-{k:2d} of last {W:3d} steps, core {core:5d}, P {P:3d}: {hit / n:.4f} | {psz / n:6.0f}")
