#!/usr/bin/env python3
"""Progressive output layer in the hidden-weighted basis, with the real 4-bit quantization:
stage 1: first r1 dims for all tokens -> keep N1; stage 2: first r2 dims for those -> keep N2; then exact.
usage: head_sim3.py <model.gguf> <gguf-py> <train.bin> <test.bin> <cache dir>"""
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
T_te = np.load(os.path.join(cache, f"top50_test_{len(H_te)}.npy")); truth = T_te[:, 0]; n = len(truth)
C = (H_tr.T.astype(np.float64) @ H_tr) / len(H_tr); C += np.eye(640) * 1e-3 * np.trace(C) / 640
w, U = np.linalg.eigh(C); Ch = (U * np.sqrt(w)) @ U.T; Cih = (U / np.sqrt(w)) @ U.T
w2, Vv = np.linalg.eigh(Ch @ (Emb.T @ Emb).astype(np.float64) @ Ch); Vv = Vv[:, ::-1]; w2 = w2[::-1]
P = Emb @ (Ch @ Vv).astype(np.float32); G = H_te @ (Cih @ Vv).astype(np.float32)
cum = np.cumsum(w2) / w2.sum()
print("share of logit variance in the first r dims: " + ", ".join(f"r={k}: {cum[k - 1]:.3f}" for k in (16, 32, 64, 128, 256, 384)))
def blockmax4(X, blk=32):
    nb = X.shape[1] // blk; Xb = X.reshape(len(X), nb, blk)
    mx = np.abs(Xb).max(2, keepdims=True)
    q = np.clip(np.round(Xb / np.maximum(mx, 1e-12) * 8 - 0.5), -8, 7) + 0.5
    return (q / 8 * mx).reshape(X.shape).astype(np.float32)
Pq = blockmax4(P[:, :256])
def ranks(Pm, rdim):
    out = np.empty(n, dtype=np.int64)
    for i in range(0, n, 500):
        Z = G[i:i + 500, :rdim] @ Pm[:, :rdim].T
        out[i:i + 500] = (Z > Z[np.arange(len(Z)), truth[i:i + 500]][:, None]).sum(1)
    return out
for rdim in (32, 64, 128, 256):
    rq = ranks(Pq, rdim); rf = ranks(P, rdim)
    print(f"first {rdim:3d} dims: winner's rank, 4-bit rows: " + " ".join(f"<{N}: {np.mean(rq < N):.4f}" for N in (256, 1024, 4096, 16384, 32768, 65536)) +
          f" | max {rq.max()}   (float rows: <4096: {np.mean(rf < 4096):.4f}, max {rf.max()})", flush=True)
