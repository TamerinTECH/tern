#!/usr/bin/env python3
"""Output-layer study, part 5: how many bits per weight does a full-vocabulary screen need?
usage: head_study5.py <model.gguf> <gguf-py> <train.bin> <test.bin> <cache dir>"""
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
sel = np.random.RandomState(3).choice(len(H_te), 1500, replace=False)
Ht = H_te[sel]; truth = T_te[sel, 0]
print("embedding: column std min/median/max", np.percentile(Emb.std(0), [0, 50, 100]).round(4), " |column mean| max", np.abs(Emb.mean(0)).max().round(4),
      " kurtosis of entries", float(((Emb - Emb.mean(0)) ** 4).mean() / (Emb.var(0).mean() ** 2)))
print("hidden: per-dim rms min/median/max", np.percentile(np.sqrt((H_tr ** 2).mean(0)), [0, 50, 100]).round(2), " |mean| max", np.abs(H_tr.mean(0)).max().round(2))
Zt = Ht @ Emb.T
def report(name, Q, bits):
    Z = Ht @ Q.T
    err = Z - Zt
    rank = (Z > Z[np.arange(len(Z)), truth][:, None]).sum(1)
    werr = err[np.arange(len(Z)), truth]
    print(f"{name:38s} {bits:4.2f} bit/w = {bits * 640 * V / 8e6:5.1f} MB | logit error std {err.std():5.2f}, winner's error mean {werr.mean():+.2f} | " +
          " ".join(f"top-{n}: {np.mean(rank < n):.4f}" for n in (4, 16, 64, 256)), flush=True)
def blockmax(X, bits, blk=32):
    n = X.shape[1] // blk; Xb = X.reshape(len(X), n, blk)
    mx = np.abs(Xb).max(2, keepdims=True); L = 2 ** (bits - 1)
    q = np.clip(np.round(Xb / np.maximum(mx, 1e-12) * L - 0.5), -L, L - 1) + 0.5
    return (q / L * mx).reshape(X.shape).astype(np.float32)
def rowscale(X, bits, clip):
    s = X.std(1, keepdims=True) * clip; L = 2 ** (bits - 1)
    q = np.clip(np.round(X / s * L - 0.5), -L, L - 1) + 0.5
    return (q / L * s).astype(np.float32)
for b in (4, 3, 2): report(f"{b} bit, per block of 32, max-scaled", blockmax(Emb, b), b + 0.5)
for b, c in ((3, 2.5), (2, 2.0), (2, 1.5)): report(f"{b} bit, per row, clip {c} sigma", rowscale(Emb, b, c), b)
# scale every hidden dimension to unit rms first (fold into the embedding): the coarse rows then spend their
# levels where the logit is decided
s = np.sqrt((H_tr ** 2).mean(0)).astype(np.float32)
Es = Emb * s; Hs = Ht / s
def report2(name, Q, bits):
    Z = Hs @ Q.T; err = Z - Zt
    rank = (Z > Z[np.arange(len(Z)), truth][:, None]).sum(1); werr = err[np.arange(len(Z)), truth]
    print(f"{name:38s} {bits:4.2f} bit/w = {bits * 640 * V / 8e6:5.1f} MB | logit error std {err.std():5.2f}, winner's error mean {werr.mean():+.2f} | " +
          " ".join(f"top-{n}: {np.mean(rank < n):.4f}" for n in (4, 16, 64, 256)), flush=True)
for b, c in ((4, 3.0), (3, 2.5), (2, 2.0), (2, 1.5)): report2(f"{b} bit, rms-scaled dims, per row clip {c}", rowscale(Es, b, c), b)
