#!/usr/bin/env python3
"""Build the n-gram table that tern uses to guess tokens ahead (speculative decoding).

usage: build_draft.py <out.ngram> [--order K] [--min-count N] <dump.bin>...

The dumps come from `tern --dump`: only the token ids are used (text the model itself generated).
For every context of 1..K tokens seen at least N times, the table keeps the most frequent next token and the
share of occurrences it accounts for. A wrong guess costs a little time, never correctness.
"""
import struct, sys
import numpy as np

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = int(args[i + 1]); del args[i:i + 2]; return v
    return default
ORDER, MINC = opt("--order", 4), opt("--min-count", 2)
out, dumps = args[0], args[1:]
E, EOT, MAXN = 640, 106, 200

seqs = []
for d in dumps:
    raw = np.fromfile(d, dtype=np.uint8); n = len(raw) // (4 + 4 * E)
    tok = raw[: n * (4 + 4 * E)].reshape(n, 4 + 4 * E)[:, :4].copy().view(np.int32)[:, 0]
    cur = []
    for t in tok.tolist():
        cur.append(t)
        if t == EOT or t == 1 or len(cur) == MAXN: seqs.append(cur); cur = []
    if cur: seqs.append(cur)
print(f"{sum(len(s) for s in seqs)} tokens in {len(seqs)} sequences")

PRIME, OFFSET, M64 = 1099511628211, 1469598103934665603, (1 << 64) - 1
def hash_tokens(ts):
    h = OFFSET
    for v in ts:
        for b in range(4):
            h ^= (v >> (8 * b)) & 0xff; h = (h * PRIME) & M64
    return h or 1

with open(out, "wb") as f:
    f.write(b"TERNNG01"); f.write(struct.pack("<ii", ORDER, 0))
    for k in range(1, ORDER + 1):
        counts = {}
        for s in seqs:
            for i in range(k, len(s)):
                key = tuple(s[i - k:i])
                c = counts.setdefault(key, {})
                c[s[i]] = c.get(s[i], 0) + 1
        entries = []
        for key, c in counts.items():
            total = sum(c.values())
            if total < MINC: continue
            nxt, cnt = max(c.items(), key=lambda kv: kv[1])
            entries.append((hash_tokens(key), nxt, cnt / total))
        cap = 1
        while cap < 2 * len(entries) + 1: cap *= 2
        keys = np.zeros(cap, dtype=np.uint64); nxts = np.zeros(cap, dtype=np.int32); conf = np.zeros(cap, dtype=np.float32)
        for h, nxt, cf in entries:
            i = h & (cap - 1)
            while keys[i] != 0: i = (i + 1) & (cap - 1)
            keys[i] = h; nxts[i] = nxt; conf[i] = cf
        f.write(struct.pack("<Q", cap)); f.write(keys.tobytes()); f.write(nxts.tobytes()); f.write(conf.tobytes())
        print(f"order {k}: {len(entries)} contexts kept (of {len(counts)}), table {cap} slots")
print("wrote", out)
