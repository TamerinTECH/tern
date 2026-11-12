#!/usr/bin/env python3
"""Study of fast output-layer schemes on recorded hidden states.

usage: head_study.py <model.gguf> <gguf-py dir> <train.bin> <test.bin>
Records: int32 argmax token + 640 float32 (final hidden state), written by `tern --dump`.
"""
import sys, time
import numpy as np

sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
from gguf.quants import dequantize

def load(path, E=640):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // (4 + 4 * E)
    raw = raw[: n * (4 + 4 * E)].reshape(n, 4 + 4 * E)
    tok = raw[:, :4].copy().view(np.int32)[:, 0]
    hid = raw[:, 4:].copy().view(np.float32).reshape(n, E)
    return tok, hid

r = GGUFReader(sys.argv[1])
t = {x.name: x for x in r.tensors}["token_embd.weight"]
t0 = time.time()
Emb = dequantize(t.data, t.tensor_type).astype(np.float32)
print(f"embedding {Emb.shape} loaded in {time.time() - t0:.1f}s")
tok_tr, H_tr = load(sys.argv[3]); tok_te, H_te = load(sys.argv[4])
print(f"train steps {len(tok_tr)}, test steps {len(tok_te)}")

hn = np.linalg.norm(H_tr, axis=1); en = np.linalg.norm(Emb, axis=1)
print(f"|h|: mean {hn.mean():.1f} min {hn.min():.1f} max {hn.max():.1f}   |e|: mean {en.mean():.3f} p1 {np.percentile(en, 1):.3f} p99 {np.percentile(en, 99):.3f} max {en.max():.3f}")

# distribution of logits on a few steps
idx = np.random.RandomState(0).choice(len(tok_te), 200, replace=False)
Z = H_te[idx] @ Emb.T
top = np.sort(Z, axis=1)[:, ::-1]
print(f"logits: best mean {top[:, 0].mean():.1f}, 2nd {top[:, 1].mean():.1f}, 10th {top[:, 9].mean():.1f}, 100th {top[:, 99].mean():.1f}, 1000th {top[:, 999].mean():.1f}, median {np.median(Z, axis=1).mean():.1f}, std {Z.std(axis=1).mean():.1f}")
agree = (Z.argmax(1) == tok_te[idx]).mean()
print(f"float argmax == recorded argmax (int8 engine): {agree:.3f}")

# how concentrated are the winners?
uniq, cnt = np.unique(tok_tr, return_counts=True)
order = uniq[np.argsort(-cnt)]
print(f"distinct winning tokens in train: {len(uniq)}")
for k in (1000, 2000, 4000, 8000):
    hot = set(order[:k].tolist())
    print(f"  top-{k} train winners cover {np.mean([x in hot for x in tok_te]):.4f} of test steps")

# low-rank previews: plain PCA of the embedding vs basis weighted by the hidden-state covariance
def basis_plain():
    w, V = np.linalg.eigh((Emb.T @ Emb).astype(np.float64))
    V = V[:, ::-1]
    return V.astype(np.float32), V.astype(np.float32)          # P = Emb @ V, g = V^T h
def basis_weighted():
    C = (H_tr.T.astype(np.float64) @ H_tr) / len(H_tr)
    C += np.eye(C.shape[0]) * 1e-3 * np.trace(C) / C.shape[0]
    w, U = np.linalg.eigh(C)
    Ch = (U * np.sqrt(w)) @ U.T; Cih = (U / np.sqrt(w)) @ U.T
    M = Ch @ (Emb.T @ Emb).astype(np.float64) @ Ch
    w2, V = np.linalg.eigh(M); V = V[:, ::-1]
    return (Ch @ V).astype(np.float32), (Cih @ V).astype(np.float32)   # P = Emb @ (Ch V), g = (Cih V)^T h

nt = min(1500, len(tok_te))
sel = np.random.RandomState(1).choice(len(tok_te), nt, replace=False)
Ht = H_te[sel]
Zt = Ht @ Emb.T
truth = Zt.argmax(1)
for name, fn in (("plain PCA", basis_plain), ("hidden-weighted", basis_weighted)):
    A, B = fn()
    P_all = Emb @ A          # [vocab, 640] rotated embedding
    G_all = Ht @ B           # [steps, 640] rotated hidden
    print(f"--- {name}: fraction of test steps whose true argmax is among the preview's top-N")
    print("   r  " + "".join(f"  N={n:<6}" for n in (16, 64, 256, 1024, 4096)))
    for rdim in (8, 16, 24, 32, 48, 64, 96):
        Zp = G_all[:, :rdim] @ P_all[:, :rdim].T
        tv = Zp[np.arange(nt), truth]
        rank = (Zp > tv[:, None]).sum(1)
        print(f"  {rdim:3d} " + "".join(f"  {np.mean(rank < n):.4f} " for n in (16, 64, 256, 1024, 4096)))
