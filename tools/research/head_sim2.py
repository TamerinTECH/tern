#!/usr/bin/env python3
"""Do the candidate sources of the fast output layer fail on the same steps? Union recall of:
core + cluster probes + recent-steps pool, and the top-N of a low-rank preview over all tokens.
usage: head_sim2.py <model.gguf> <gguf-py> <train.bin> <test.bin> <cache dir> <clusters.npz>"""
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
cl = np.load(sys.argv[6]); a = cl["assign"]; Cm = cl["centroid"]
truth = T_te[:, 0]; n = len(truth)
wt = 1.0 / (1.0 + np.arange(50)); score = np.zeros(V); np.add.at(score, T_tr.ravel(), np.tile(wt, len(T_tr)))
rank = np.empty(V, dtype=np.int64); rank[np.argsort(-score, kind="stable")] = np.arange(V)
S = H_te @ Cm.T; crank = (S > S[np.arange(n), a[truth]][:, None]).sum(1)
EOT, MAXN = 106, 200
seq = np.zeros(n, dtype=np.int32); s = 0; c = 0
for i in range(n):
    seq[i] = s; c += 1
    if tok_te[i] == EOT or tok_te[i] == 1 or c == MAXN: s += 1; c = 0
inpool = np.zeros(n, dtype=bool)
for q in range(seq.max() + 1):
    idx = np.nonzero(seq == q)[0]; seen = set()
    for i in idx:
        inpool[i] = int(truth[i]) in seen
        seen.update(T_te[i].tolist())
C = (H_tr.T.astype(np.float64) @ H_tr) / len(H_tr); C += np.eye(640) * 1e-3 * np.trace(C) / 640
w, U = np.linalg.eigh(C); Ch = (U * np.sqrt(w)) @ U.T; Cih = (U / np.sqrt(w)) @ U.T
w2, Vv = np.linalg.eigh(Ch @ (Emb.T @ Emb).astype(np.float64) @ Ch); Vv = Vv[:, ::-1]
P = Emb @ (Ch @ Vv).astype(np.float32); G = H_te @ (Cih @ Vv).astype(np.float32)
print(f"{n} test steps. candidate set = core + probes + pool; preview = first r hidden-weighted dims, top-N re-scored")
for core, Pp in ((4000, 64), (8000, 128)):
    cand = (rank[truth] < core) | (crank < Pp) | inpool
    print(f"core {core}, {Pp} probes, pool: found {cand.mean():.4f}")
    for rdim in (32, 64, 128):
        prank = np.empty(n, dtype=np.int64)
        for i in range(0, n, 500):
            Z = G[i:i + 500, :rdim] @ P[:, :rdim].T
            prank[i:i + 500] = (Z > Z[np.arange(len(Z)), truth[i:i + 500]][:, None]).sum(1)
        for N in (256, 1024, 4096):
            pv = prank < N
            print(f"   preview r={rdim:3d} N={N:4d}: alone {pv.mean():.4f} | union {np.mean(cand | pv):.4f} | if independent {1 - (1 - cand.mean()) * (1 - pv.mean()):.4f}", flush=True)
