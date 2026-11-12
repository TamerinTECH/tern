#!/usr/bin/env python3
"""Output-layer study, part 2: cluster the token embeddings (k-means), score cluster centroids, scan only the
rows of the best clusters. Reports how often the true best token is found and how many rows are scanned.

usage: head_study2.py <model.gguf> <gguf-py dir> <train.bin> <test.bin> <K> [iters] [save.npz]
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
    return raw[:, :4].copy().view(np.int32)[:, 0], raw[:, 4:].copy().view(np.float32).reshape(n, E)

K = int(sys.argv[5]); iters = int(sys.argv[6]) if len(sys.argv) > 6 else 8
r = GGUFReader(sys.argv[1])
t = {x.name: x for x in r.tensors}["token_embd.weight"]
Emb = dequantize(t.data, t.tensor_type).astype(np.float32)
V = Emb.shape[0]
tok_tr, H_tr = load(sys.argv[3]); tok_te, H_te = load(sys.argv[4])

# Cluster in the geometry that matters for logits: two tokens are close when their logits move together over
# real hidden states, i.e. cosine similarity after multiplying by the square root of the hidden-state covariance.
WEIGHTED = "--plain" not in sys.argv
Emb_orig = Emb
if WEIGHTED:
    Cov = (H_tr.T.astype(np.float64) @ H_tr) / len(H_tr)
    Cov += np.eye(Cov.shape[0]) * 1e-3 * np.trace(Cov) / Cov.shape[0]
    w, U = np.linalg.eigh(Cov)
    Emb = (Emb @ ((U * np.sqrt(w)) @ U.T).astype(np.float32))
    Emb /= np.maximum(np.linalg.norm(Emb, axis=1, keepdims=True), 1e-9)

def assign(C, chunk=16384):
    out = np.empty(V, dtype=np.int32); best = np.empty(V, dtype=np.float32)
    for i in range(0, V, chunk):
        s = Emb[i:i + chunk] @ C.T
        out[i:i + chunk] = s.argmax(1); best[i:i + chunk] = s.max(1)
    return out, best

rs = np.random.RandomState(0)
C = Emb[rs.choice(V, K, replace=False)].copy()
t0 = time.time()
for it in range(iters):
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    a, best = assign(Cn)
    sums = np.zeros((K, Emb.shape[1]), dtype=np.float64); np.add.at(sums, a, Emb)
    cnt = np.bincount(a, minlength=K)
    empty = cnt == 0
    C = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    if empty.any():   # re-seed empty clusters with the worst-fitting tokens
        worst = np.argsort(best)[: empty.sum()]
        C[empty] = Emb[worst]
    print(f"iter {it}: mean cos to centroid {best.mean():.4f}, empty {int(empty.sum())}, largest {cnt.max()}, {time.time() - t0:.0f}s", flush=True)
Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
a, best = assign(Cn)
cnt = np.bincount(a, minlength=K)
Emb = Emb_orig
sums = np.zeros((K, Emb.shape[1]), dtype=np.float64); np.add.at(sums, a, Emb)
Cmean = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)     # centroid score = mean logit of the members
Cn = Cmean / np.maximum(np.linalg.norm(Cmean, axis=1, keepdims=True), 1e-9)
print(f"cluster sizes: median {int(np.median(cnt))}, p90 {int(np.percentile(cnt, 90))}, max {cnt.max()}")
if len(sys.argv) > 7 and not sys.argv[7].startswith("--"): np.savez(sys.argv[7], assign=a, centroid=Cmean)

nt = min(3000, len(tok_te)); sel = rs.choice(len(tok_te), nt, replace=False)
Ht = H_te[sel]; truth = (Ht @ Emb.T).argmax(1)
tc = a[truth]
for label, Cs in (("mean centroid", Cmean), ("unit centroid", Cn)):
    S = Ht @ Cs.T
    order = np.argsort(-S, axis=1)
    rank = (S > S[np.arange(nt), tc][:, None]).sum(1)
    print(f"--- {label}: true best token's cluster is within the top-P clusters")
    for P in (4, 8, 16, 32, 64, 128, 256):
        rows = cnt[order[:, :P]].sum(1).mean()
        print(f"  P={P:4d}: found {np.mean(rank < P):.4f}   rows scanned {rows:8.0f} ({100 * rows / V:.2f}% of the vocabulary)")
# hot set from train winners + clusters
uniq, c2 = np.unique(tok_tr, return_counts=True)
hot = np.zeros(V, dtype=bool); hot[uniq] = True
S = Ht @ Cmean.T; rank = (S > S[np.arange(nt), tc][:, None]).sum(1)
print(f"--- with all {len(uniq)} train winners always scanned (mean centroid)")
for P in (4, 8, 16, 32, 64):
    print(f"  P={P:4d}: found {np.mean((rank < P) | hot[truth]):.4f}")
miss = sel[(rank >= 64) & ~hot[truth]]
print("misses at P=64 (token ids):", truth[(rank >= 64) & ~hot[truth]][:40].tolist())
