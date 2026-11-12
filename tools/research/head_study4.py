#!/usr/bin/env python3
"""Output-layer study, part 4: screen ALL tokens with very coarse rows (1 or 2 bits per weight), then re-score
the best N exactly. Reports how often the true best token is within the screen's top-N.
usage: head_study4.py <model.gguf> <gguf-py> <train.bin> <test.bin> <cache dir>"""
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
T_te = np.load(os.path.join(cache, f"top50_test_{len(H_te)}.npy"))
sel = np.random.RandomState(3).choice(len(H_te), 3000, replace=False)
Ht = H_te[sel]; truth = T_te[sel, 0]; second = T_te[sel, 1]
NS = (16, 64, 256, 1024)
def report(name, Q, G, bytes_per_row):
    """Q: [V, d] coarse rows, G: [steps, d] transformed hidden states"""
    rank = np.empty(len(G), dtype=np.int64)
    for i in range(0, len(G), 500):
        Z = G[i:i + 500] @ Q.T
        rank[i:i + 500] = (Z > Z[np.arange(len(Z)), truth[i:i + 500]][:, None]).sum(1)
    print(f"{name:44s} {bytes_per_row:4d} B/row = {bytes_per_row * V / 1e6:5.1f} MB | " + " ".join(f"top-{n}: {np.mean(rank < n):.4f}" for n in NS) + f" | worst rank {rank.max()}", flush=True)

def quant(X, bits):
    """per-column scaled uniform quantizer with 2^bits levels (sign for 1 bit)"""
    if bits == 1:
        a = np.abs(X).mean(0, keepdims=True)
        return np.sign(X).astype(np.float32) * a
    s = X.std(0, keepdims=True)
    if bits == 2:   # levels at +-0.45, +-1.51 sigma (Lloyd-Max for a Gaussian)
        return (np.where(np.abs(X) > 0.98 * s, 1.51, 0.45) * np.sign(X) * s).astype(np.float32)
    raise ValueError

report("1 bit, natural basis", quant(Emb, 1), Ht, 80)
report("2 bit, natural basis", quant(Emb, 2), Ht, 160)
# hidden-weighted basis: P = Emb (C^1/2 V), g = (C^-1/2 V)^T h; leading directions carry most of the logit variance
C = (H_tr.T.astype(np.float64) @ H_tr) / len(H_tr); C += np.eye(640) * 1e-3 * np.trace(C) / 640
w, U = np.linalg.eigh(C); Ch = (U * np.sqrt(w)) @ U.T; Cih = (U / np.sqrt(w)) @ U.T
w2, Vv = np.linalg.eigh(Ch @ (Emb.T @ Emb).astype(np.float64) @ Ch); Vv = Vv[:, ::-1]
P = Emb @ (Ch @ Vv).astype(np.float32); G = Ht @ (Cih @ Vv).astype(np.float32)
for rdim in (160, 320, 640):
    report(f"1 bit, weighted basis, first {rdim} dims", quant(P[:, :rdim], 1), G[:, :rdim], rdim // 8)
for rdim in (160, 320):
    report(f"2 bit, weighted basis, first {rdim} dims", quant(P[:, :rdim], 2), G[:, :rdim], rdim // 4)
# random rotation (spreads information evenly) then sign
Rm = np.linalg.qr(np.random.RandomState(0).randn(640, 640))[0].astype(np.float32)
report("1 bit, random rotation", quant(Emb @ Rm, 1), Ht @ Rm, 80)
