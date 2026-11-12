#!/usr/bin/env python3
"""Study: how to pick the candidates of the fast output layer.

usage: head_candidates.py [--set a|b|c|d] <model.gguf> <gguf-py dir> <train dump> <held-out dump> [steps]

For held-out hidden states (a `tern --dump` file from prompts that were not used for the statistics) the exact
logits are computed in float, and for several cheap approximate scores the rank of the true best token:
if that rank is <= K, a layer that re-scores the K best candidates with exact rows returns the true best token.
The results are in docs/DESIGN.md, section 6. This is the record of the study, not a tool of the engine.

set a (default): which kind of score
  svd R      logits restricted to R directions fitted on the train dump (the layer that was replaced)
  sign       1 bit per weight: sign(w), one scale per row; h as is
  sign/b     the same with one scale per block of 64 weights
  tern       {-1, 0, +1} per weight (1.58 bits), one scale per row
  2bit       4 levels per weight, one step per row
  4bit       16 levels, one step per block of 32 (the Q4_0 layout)
  pca R+sign the R principal directions of h exactly, the rest of each row as sign bits
set b: how many directions, which scale, which precision, directions from few recorded steps or from the rows alone
set c: which rows fail, and what a list of frequent tokens that are always scored exactly would add
set d: the rows that one bit per weight cannot hold (few large weights), and scoring exactly those at every step
"""
import sys, time
import numpy as np

args = sys.argv[1:]
SET = "a"
if "--set" in args:
    i = args.index("--set"); SET = args[i + 1]; del args[i:i + 2]
model, ggufpy, train, held = args[:4]
n_steps = int(args[4]) if len(args) > 4 else 3000
sys.path.insert(0, ggufpy)
from gguf import GGUFReader
from gguf.quants import dequantize

r = GGUFReader(model)
t = {x.name: x for x in r.tensors}["token_embd.weight"]
Emb = dequantize(t.data, t.tensor_type).astype(np.float32)
V, E = Emb.shape
print(f"output matrix {V} x {E}", flush=True)

def load(path):
    raw = np.memmap(path, dtype=np.uint8, mode="r"); n = len(raw) // (4 + 4 * E)
    rec = raw[: n * (4 + 4 * E)].reshape(n, 4 + 4 * E)
    return np.ascontiguousarray(rec[:, 4:]).view(np.float32).reshape(n, E)

H = load(held)
step = max(1, len(H) // n_steps)
H = np.ascontiguousarray(H[::step][:n_steps]); n = len(H)
t0 = time.time()
Z = H @ Emb.T
best = Z.argmax(1); zb = Z[np.arange(n), best]
Zs = np.sort(np.partition(Z, V - 2, axis=1)[:, V - 2:], axis=1)
gap = Zs[:, 1] - Zs[:, 0]
mu, sd = Z.mean(1), Z.std(1)
print(f"{n} held-out steps, exact logits in {time.time() - t0:.0f}s; best logit is {np.median((zb - mu) / sd):.1f} std above the mean (median), "
      f"5% quantile {np.quantile((zb - mu) / sd, 0.05):.1f}; gap to the second: median {np.median(gap):.2f}, <0.1 in {np.mean(gap < 0.1) * 100:.1f}% of steps", flush=True)

KS = [16, 64, 256, 1024, 4096, 16384]
def report(name, Za, bytes_per_token):
    rank = (Za > Za[np.arange(n), best][:, None]).sum(1) + 1
    miss = " ".join(f"{np.mean(rank > k) * 100:7.3f}" for k in KS)
    print(f"{name:<22} {bytes_per_token * V / 1e6:8.0f} MB | miss % at K={KS}: {miss} | worst rank {rank.max()}", flush=True)
    return rank

# ---- statistics of the training dump
Ht = load(train)
C = np.zeros((E, E))
for i in range(0, len(Ht), 32768): C += Ht[i:i + 32768].T.astype(np.float64) @ Ht[i:i + 32768]
C /= len(Ht)
w, U = np.linalg.eigh(C); w = w[::-1]; U = U[:, ::-1]
print(f"train dump: {len(Ht)} steps; energy of h in the first 1/8/64/256 principal directions: "
      + " ".join(f"{w[:k].sum() / w.sum():.3f}" for k in (1, 8, 64, 256)), flush=True)

if SET == "a":
    # ---- svd preview (as in the TERNHD04 plans)
    Creg = C + np.eye(E) * 1e-3 * np.trace(C) / E
    ww, UU = np.linalg.eigh(Creg)
    Ch = (UU * np.sqrt(ww)) @ UU.T; Cih = (UU / np.sqrt(ww)) @ UU.T
    w2, Vv = np.linalg.eigh(Ch @ (Emb.T @ Emb).astype(np.float64) @ Ch)
    Vv = Vv[:, ::-1]
    for R in (64, 128, 256, 512):
        P = Emb @ (Ch @ Vv[:, :R]).astype(np.float32); B = (Cih @ Vv[:, :R]).T.astype(np.float32)
        report(f"svd {R}", (H @ B.T) @ P.T, R)
    del P

    # ---- low-bit sketches of the rows themselves
    def sign_scores(W, Hq, block=0):
        S = np.sign(W).astype(np.float32); S[S == 0] = 1
        if block:
            a = np.abs(W).reshape(V, E // block, block).mean(2)              # LS scale per block
            return Hq @ (S.reshape(V, E // block, block) * a[:, :, None]).reshape(V, E).T
        a = np.abs(W).mean(1)
        return (Hq @ S.T) * a[None, :]

    report("sign", sign_scores(Emb, H), E / 8 + 4)
    report("sign/b64", sign_scores(Emb, H, 64), E / 8 + E / 64 * 2)

    def levels_scores(W, Hq, nlev):                       # uniform mid-rise quantizer, step per row chosen by a small search
        rms = np.sqrt((W * W).mean(1, keepdims=True))
        bestq, beste = None, None
        for f in ((0.6, 0.8, 1.0, 1.2, 1.5) if nlev == 3 else (0.7, 0.85, 1.0, 1.15) if nlev == 4 else (0.25, 0.3, 0.35, 0.4)):
            d = rms * f
            if nlev % 2: q = np.clip(np.rint(W / d), -(nlev // 2), nlev // 2) * d
            else: q = (np.clip(np.floor(W / d), -nlev // 2, nlev // 2 - 1) + 0.5) * d
            e = ((W - q) ** 2).sum(1)
            if beste is None: bestq, beste = q, e
            else:
                m = e < beste; bestq[m] = q[m]; beste[m] = e[m]
        print(f"   {nlev} levels: relative rms error of the rows {np.sqrt(beste.sum() / (W * W).sum()):.3f}", flush=True)
        return Hq @ bestq.T

    report("tern", levels_scores(Emb, H, 3), E * 1.6 / 8 + 4)
    report("2bit", levels_scores(Emb, H, 4), E / 4 + 4)

    def q4(W):                                            # Q4_0-like: blocks of 32, d = max|w| / 8 (signed max), values -8..7
        Wb = W.reshape(V, E // 32, 32)
        am = np.abs(Wb).max(2, keepdims=True); d = am / 7.5
        d[d == 0] = 1
        return (np.clip(np.rint(Wb / d), -8, 7) * d).reshape(V, E)
    report("4bit", H @ q4(Emb).T, E / 32 * 18)

    # ---- principal directions exactly + sign bits of the rest
    for R in (16, 64, 128):
        Ur = U[:, :R].astype(np.float32)
        A = Emb @ Ur                                      # [V, R]
        Res = Emb - A @ Ur.T
        Hp = H @ Ur; Hr = H - Hp @ Ur.T
        report(f"pca {R} + sign", Hp @ A.T + sign_scores(Res, Hr), R + E / 8 + 4)
        if R == 64: report(f"pca {R} + sign/b64", Hp @ A.T + sign_scores(Res, Hr, 64), R + E / 8 + E / 32)

def sign_part(Res, Hr, scale="ls"):
    S = np.sign(Res).astype(np.float32); S[S == 0] = 1
    l1 = np.abs(Res).mean(1)
    a = l1 if scale == "ls" else (Res * Res).mean(1) / l1 if scale == "un" else np.sqrt(l1 * (Res * Res).mean(1) / l1)
    return (Hr @ S.T) * a[None, :]

def split(Ur):
    A = Emb @ Ur; Res = Emb - A @ Ur.T
    Hp = H @ Ur; Hr = H - Hp @ Ur.T
    return A, Res, Hp, Hr

if SET == "b":
    report("sign (no directions)", sign_part(Emb, H), E / 8 + 4)
    for R in (1, 2, 4, 8, 16, 32, 64):
        A, Res, Hp, Hr = split(U[:, :R].astype(np.float32))
        rk = report(f"pca {R} + sign", Hp @ A.T + sign_part(Res, Hr), R + E / 8 + 4)
        if R == 16:
            rk = report(f"pca {R} + sign, scale un", Hp @ A.T + sign_part(Res, Hr, "un"), R + E / 8 + 4)
            worst = np.argsort(-rk)[:8]
            print("   worst steps (rank, token, std above mean, gap to 2nd): " + "  ".join(f"({rk[i]}, {best[i]}, {(zb[i] - mu[i]) / sd[i]:.1f}, {gap[i]:.2f})" for i in worst), flush=True)
            print("   ranks: " + " ".join(f">{k}: {int((rk > k).sum())}" for k in (1, 4, 16, 64, 128, 256, 512, 1024, 2048)), flush=True)
        if R == 32:
            worst = np.argsort(-rk)[:8]
            print("   worst steps (rank, token, std above mean, gap to 2nd): " + "  ".join(f"({rk[i]}, {best[i]}, {(zb[i] - mu[i]) / sd[i]:.1f}, {gap[i]:.2f})" for i in worst), flush=True)
            report(f"pca {R} + sign, scale un", Hp @ A.T + sign_part(Res, Hr, "un"), R + E / 8 + 4)
            report(f"pca {R} + sign, scale mid", Hp @ A.T + sign_part(Res, Hr, "mid"), R + E / 8 + 4)
            # engine-like precision: A as int8 per column, coordinates int8 per block of 16; residual h as int8 / 6 bits, one scale
            cs = np.abs(A).max(0) / 127.0
            A8 = np.clip(np.rint(A / cs), -127, 127).astype(np.float32)
            g = Hp * cs
            gb = g.reshape(n, R // 16, 16); d = np.abs(gb).max(2, keepdims=True) / 127.0; d[d == 0] = 1
            g8 = (np.rint(gb / d) * d).reshape(n, R)
            for bits in (8, 6):
                lim = (1 << (bits - 1)) - 1
                dq = np.abs(Hr).max(1, keepdims=True) / lim
                Hq = np.rint(Hr / dq) * dq
                report(f"pca {R} i8 + sign, h {bits} bit", g8 @ A8.T + sign_part(Res, Hq), R + E / 8 + 4)
            cs = np.abs(A).max(0) / 32767.0
            A16 = np.rint(A / cs).astype(np.float32); g = Hp * cs
            dq = np.abs(g).max(1, keepdims=True) / 1023.0
            report(f"pca {R} i16 + sign", (np.rint(g / dq) * dq) @ A16.T + sign_part(Res, Hr), 2 * R + E / 8 + 4)
    # directions from few recorded steps
    for few in (500, 2000, 10000):
        Hf = Ht[:: max(1, len(Ht) // few)][:few].astype(np.float64)
        wf, Uf = np.linalg.eigh(Hf.T @ Hf / len(Hf))
        A, Res, Hp, Hr = split(Uf[:, ::-1][:, :32].astype(np.float32))
        report(f"pca 32 of {few} steps", Hp @ A.T + sign_part(Res, Hr), 32 + E / 8 + 4)
    # directions without any recording: principal directions of the rows of the output matrix
    wg, Ug = np.linalg.eigh((Emb.T @ Emb).astype(np.float64))
    for R in (16, 64):
        A, Res, Hp, Hr = split(Ug[:, ::-1][:, :R].astype(np.float32))
        report(f"rows' svd {R} + sign", Hp @ A.T + sign_part(Res, Hr), R + E / 8 + 4)
    # only the mean hidden state
    m = Ht.mean(0).astype(np.float64); m /= np.linalg.norm(m)
    A, Res, Hp, Hr = split(m[:, None].astype(np.float32))
    report("mean direction + sign", Hp @ A.T + sign_part(Res, Hr), 1 + E / 8 + 4)
if SET == "c":
    # which rows fail, and what an always-exact list of frequent tokens (the "core") adds
    raw = np.memmap(train, dtype=np.uint8, mode="r"); nrec = len(raw) // (4 + 4 * E)
    ttok = np.ascontiguousarray(raw[: nrec * (4 + 4 * E)].reshape(nrec, 4 + 4 * E)[:, :4]).view(np.int32)[:, 0]
    cnt = np.bincount(ttok, minlength=V)
    order = np.argsort(-cnt, kind="stable")
    print(f"train dump: {int((cnt > 0).sum())} distinct tokens chosen; the 1024 most frequent cover {cnt[order[:1024]].sum() / cnt.sum() * 100:.1f}% of steps, 4096: {cnt[order[:4096]].sum() / cnt.sum() * 100:.1f}%", flush=True)
    cores = {0: np.zeros(V, bool)}
    for c in (1024, 4096):
        m = np.zeros(V, bool); m[order[:c]] = True; cores[c] = m
    rn = np.sqrt((Emb * Emb).sum(1))
    for R in (16, 32, 64):
        A, Res, Hp, Hr = split(U[:, :R].astype(np.float32))
        l1 = np.abs(Res).mean(1); l2 = (Res * Res).mean(1)
        shape = l2 / (l1 * l1)                        # 1.57 for a Gaussian row; larger: a few big weights
        for scale in ("ls", "mid", "un"):
            Za = Hp @ A.T + sign_part(Res, Hr, scale)
            rank = (Za > Za[np.arange(n), best][:, None]).sum(1) + 1
            out = []
            for c in (0, 1024, 4096):
                inc = cores[c][best]
                out.append(f"core {c}: " + " ".join(f"{int(((rank > k) & ~inc).sum())}" for k in (64, 256, 1024, 4096)))
            print(f"pca {R} + sign, scale {scale}: misses of {n} at K=64/256/1024/4096 | " + " | ".join(out) + f" | worst rank {rank.max()}", flush=True)
            if scale == "ls" and R in (16, 64):
                worst = np.argsort(-rank)[:10]
                for i in worst:
                    t = best[i]
                    true_res = float(Res[t] @ Hr[i]); est = float(l1[t] * (np.sign(Res[t]) @ Hr[i]))
                    print(f"     rank {rank[i]:6d} token {t:6d} (chosen {cnt[t]} times in the train dump) row norm {rn[t]:.2f} (median {np.median(rn):.2f}), shape {shape[t]:.2f} (median {np.median(shape):.2f}); "
                          f"logit {zb[i]:.1f} = directions {float(A[t] @ Hp[i]):.1f} + rest {true_res:.1f}, sketch of the rest {est:.1f}", flush=True)
if SET == "d":
    # rows that one bit per weight cannot hold: a few large weights carry the row ("shape" = mean r^2 / mean|r|^2,
    # 1.57 for a Gaussian row). Does scoring those rows exactly at every step remove the misses?
    raw = np.memmap(train, dtype=np.uint8, mode="r"); nrec = len(raw) // (4 + 4 * E)
    ttok = np.ascontiguousarray(raw[: nrec * (4 + 4 * E)].reshape(nrec, 4 + 4 * E)[:, :4]).view(np.int32)[:, 0]
    cnt = np.bincount(ttok, minlength=V)
    for R in (16, 32):
        A, Res, Hp, Hr = split(U[:, :R].astype(np.float32))
        l1 = np.abs(Res).mean(1); l2 = (Res * Res).mean(1)
        shape = l2 / np.maximum(l1 * l1, 1e-30)
        print(f"pca {R}: shape of the rows: median {np.median(shape):.3f}, 90% {np.quantile(shape, 0.9):.3f}, 99% {np.quantile(shape, 0.99):.3f}, 99.9% {np.quantile(shape, 0.999):.3f}; rows above "
              + ", ".join(f"{tau}: {int((shape > tau).sum())}" for tau in (1.65, 1.7, 1.8, 1.9, 2.0, 2.2)), flush=True)
        ch = cnt > 0
        print(f"   rows of tokens chosen in the train dump ({int(ch.sum())}): median shape {np.median(shape[ch]):.3f}, 90% {np.quantile(shape[ch], 0.9):.3f}; share of generated steps whose token has shape above "
              + ", ".join(f"{tau}: {cnt[shape > tau].sum() / cnt.sum() * 100:.1f}%" for tau in (1.65, 1.7, 1.8, 1.9, 2.0, 2.2)), flush=True)
        for scale in ("ls", "mid"):
            Za = Hp @ A.T + sign_part(Res, Hr, scale)
            rank = (Za > Za[np.arange(n), best][:, None]).sum(1) + 1
            for tau in (9.9, 2.0, 1.9, 1.8, 1.7, 1.65):
                inc = (shape > tau)[best]
                # rank among the rows that are not scored exactly anyway
                print(f"   scale {scale}, rows with shape > {tau} exact ({int((shape > tau).sum())} rows, best token among them in {inc.mean() * 100:.1f}% of steps): misses of {n} at K=64/256/1024/4096: "
                      + " ".join(f"{int(((rank > k) & ~inc).sum())}" for k in (64, 256, 1024, 4096)) + f" | worst rank left {int(rank[~inc].max()) if (~inc).any() else 0}", flush=True)
print("done", flush=True)
