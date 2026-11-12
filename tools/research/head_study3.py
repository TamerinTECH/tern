#!/usr/bin/env python3
"""Output-layer study, part 3: split the vocabulary into "plausible" tokens (seen among the top-50 logits of
recorded steps) and the rest, cluster the plausible ones finely in the hidden-weighted geometry, the rest
coarsely, and measure recall of the true best token vs rows scanned.

usage: head_study3.py <model.gguf> <gguf-py dir> <train.bin> <test.bin> <cache dir>
"""
import os, sys, time
import numpy as np

sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
from gguf.quants import dequantize

def load(path, E=640):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // (4 + 4 * E)
    raw = raw[: n * (4 + 4 * E)].reshape(n, 4 + 4 * E)
    return raw[:, :4].copy().view(np.int32)[:, 0], raw[:, 4:].copy().view(np.float32).reshape(n, E)

cache = sys.argv[5]
r = GGUFReader(sys.argv[1])
t = {x.name: x for x in r.tensors}["token_embd.weight"]
Emb = dequantize(t.data, t.tensor_type).astype(np.float32)
V = Emb.shape[0]
tok_tr, H_tr = load(sys.argv[3]); tok_te, H_te = load(sys.argv[4])
print(f"train {len(tok_tr)} steps, test {len(tok_te)} steps", flush=True)

def topk(H, k, tag):
    f = os.path.join(cache, f"top{k}_{tag}_{len(H)}.npy")
    if os.path.exists(f): return np.load(f)
    out = np.empty((len(H), k), dtype=np.int32); t0 = time.time()
    for i in range(0, len(H), 1024):
        Z = H[i:i + 1024] @ Emb.T
        idx = np.argpartition(-Z, k, axis=1)[:, :k]
        zz = np.take_along_axis(Z, idx, 1)
        out[i:i + 1024] = np.take_along_axis(idx, np.argsort(-zz, axis=1), 1)
    np.save(f, out); print(f"top-{k} of {tag}: {time.time() - t0:.0f}s", flush=True)
    return out

T_tr = topk(H_tr, 50, "train"); T_te = topk(H_te, 50, "test")
truth = T_te[:, 0]
print(f"engine argmax == float argmax on test: {np.mean(truth == tok_te):.4f}")
for k in (1, 5, 10, 20, 50):
    seen = np.zeros(V, dtype=bool); seen[np.unique(T_tr[:, :k])] = True
    print(f"tokens seen in train top-{k:2d}: {int(seen.sum()):6d}   cover {np.mean(seen[truth]):.4f} of test winners, {np.mean(seen[T_te[:, :5]]):.4f} of test top-5")
half = len(T_tr) // 2
for k in (10, 50):
    seen = np.zeros(V, dtype=bool); seen[np.unique(T_tr[:half, :k])] = True
    print(f"  (half the train data, top-{k}: {int(seen.sum())} tokens, cover {np.mean(seen[truth]):.4f})")

# hidden-weighted geometry
Cov = (H_tr.T.astype(np.float64) @ H_tr) / len(H_tr)
Cov += np.eye(640) * 1e-3 * np.trace(Cov) / 640
w, U = np.linalg.eigh(Cov)
Wh = ((U * np.sqrt(w)) @ U.T).astype(np.float32)

def kmeans(X, K, iters, seed=0):
    rs = np.random.RandomState(seed)
    C = X[rs.choice(len(X), K, replace=False)].copy()
    for it in range(iters):
        Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        a = np.empty(len(X), dtype=np.int32); best = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), 16384):
            s = X[i:i + 16384] @ Cn.T; a[i:i + 16384] = s.argmax(1); best[i:i + 16384] = s.max(1)
        sums = np.zeros((K, X.shape[1]), dtype=np.float64); np.add.at(sums, a, X)
        cnt = np.bincount(a, minlength=K); empty = cnt == 0
        C = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
        if empty.any(): C[empty] = X[np.argsort(best)[: empty.sum()]]
    return a

seen = np.zeros(V, dtype=bool); seen[np.unique(T_tr)] = True
plaus = np.nonzero(seen)[0]; rest = np.nonzero(~seen)[0]
Xw = Emb @ Wh; Xw /= np.maximum(np.linalg.norm(Xw, axis=1, keepdims=True), 1e-9)
nt = min(4000, len(truth)); sel = np.random.RandomState(1).choice(len(truth), nt, replace=False)
Ht = H_te[sel]; tr = truth[sel]
for Kp, Kr in ((2048, 512), (4096, 512), (8192, 512)):
    t0 = time.time()
    a = np.empty(V, dtype=np.int32)
    a[plaus] = kmeans(Xw[plaus], Kp, 6)
    a[rest] = Kp + kmeans(Xw[rest], Kr, 4)
    K = Kp + Kr
    sums = np.zeros((K, 640), dtype=np.float64); np.add.at(sums, a, Emb)
    cnt = np.bincount(a, minlength=K)
    Cm = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    S = Ht @ Cm.T
    order = np.argsort(-S, axis=1)
    rank = (S > S[np.arange(nt), a[tr]][:, None]).sum(1)
    print(f"--- {Kp} clusters over {len(plaus)} plausible tokens (median size {int(np.median(cnt[:Kp]))}) + {Kr} over the other {len(rest)}  [{time.time() - t0:.0f}s]")
    for P in (8, 16, 32, 64, 128, 256):
        rows = cnt[order[:, :P]].sum(1).mean()
        print(f"  P={P:4d}: found {np.mean(rank < P):.4f}   rows scanned {rows:7.0f}   (winner outside plausible set: {np.mean(~seen[tr]):.4f})")
    np.savez(os.path.join(cache, f"clusters_{Kp}_{Kr}.npz"), assign=a, centroid=Cm)
