#!/usr/bin/env python3
"""Slow, exact float reference of the Gemma 4 forward pass (dequantized weights, float64 activations).
Ground truth for judging how close an engine's logits are. Text-only path of llama.cpp's src/models/gemma4.cpp:
per-layer token inputs and shared key/value layers (E2B, E4B), grouped key/value heads (31B), routed experts (26B-A4B).

usage: ref_gemma4.py <model.gguf> <gguf-py dir> [--dump FILE.npz] <token ids...>
       -> one line per position: top-5 "id:logit"
"""
import sys
import numpy as np

sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
from gguf.quants import dequantize

args = sys.argv[3:]
dump = None
if args and args[0] == "--dump":
    dump = args[1]; args = args[2:]

r = GGUFReader(sys.argv[1])
T = {t.name: t for t in r.tensors}
def W(name):
    t = T[name]
    return dequantize(t.data, t.tensor_type).astype(np.float64)
def rows(name, idx):
    t = T[name]
    return dequantize(t.data[idx], t.tensor_type).astype(np.float64)
def kv(key, default=None):
    f = r.fields.get("gemma4." + key)
    return f.contents() if f is not None else default
def per_layer(key, il, default=None):
    v = kv(key, default)
    return v[il] if isinstance(v, (list, tuple)) else v

E = kv("embedding_length"); NL = kv("block_count"); NH = kv("attention.head_count")
HD_FULL = kv("attention.key_length"); HD_SWA = kv("attention.key_length_swa", HD_FULL)
SWA = kv("attention.sliding_window"); PAT = kv("attention.sliding_window_pattern")
SHARED = kv("attention.shared_kv_layers", 0); PL = kv("embedding_length_per_layer_input", 0)
EPS = kv("attention.layer_norm_rms_epsilon"); BASE = kv("rope.freq_base"); BASE_SWA = kv("rope.freq_base_swa", 10000.0)
CAP = kv("final_logit_softcapping", 0.0)
NX = kv("expert_count", 0); KX = kv("expert_used_count", 0)
KV0 = NL - SHARED                                   # layers below this have their own keys and values
def is_swa(il): return bool(PAT[il])
def kv_src(il): return il if il < KV0 else KV0 - (2 if is_swa(il) else 1)

def rms(x, w=None, eps=EPS):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w
def rope(x, pos, il):            # x: [T, heads, hd]; pairs (i, i + hd/2)
    hd = x.shape[-1]; h = hd // 2
    base = BASE_SWA if is_swa(il) else BASE
    theta = pos[:, None] * (base ** (-2.0 * np.arange(h) / hd))[None, :]
    if not is_swa(il) and "rope_freqs.weight" in T:
        theta = theta / W("rope_freqs.weight")[None, :h]
    c, s = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]
    a, b = x[..., :h], x[..., h:]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))

toks = np.array([int(a) for a in args])
n = len(toks); pos = np.arange(n, dtype=np.float64)
emb_t = T["token_embd.weight"]
x = rows("token_embd.weight", toks) * np.sqrt(E)

pli = None
if PL:
    ple = rows("per_layer_token_embd.weight", toks) * (np.sqrt(PL) / np.sqrt(2.0))      # [n, NL*PL]
    pp = (x @ W("per_layer_model_proj.weight").T).reshape(n, NL, PL)
    pp = rms(pp, W("per_layer_proj_norm.weight") / np.sqrt(2.0), eps=EPS * E)
    pli = pp + ple.reshape(n, NL, PL)

i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
K, V = {}, {}
states = [x.copy()]
for il in range(NL):
    p = f"blk.{il}."
    hd = HD_SWA if is_swa(il) else HD_FULL
    src = kv_src(il)
    nkv = per_layer("attention.head_count_kv", src, 1); rep = NH // nkv
    h = rms(x, W(p + "attn_norm.weight"))
    q = (h @ W(p + "attn_q.weight").T).reshape(n, NH, hd)
    q = rope(rms(q, W(p + "attn_q_norm.weight")), pos, il)
    if il < KV0:
        k = (h @ W(p + "attn_k.weight").T).reshape(n, nkv, hd)
        v = (h @ W(p + "attn_v.weight").T).reshape(n, nkv, hd) if p + "attn_v.weight" in T else k.copy()
        K[il] = rope(rms(k, W(p + "attn_k_norm.weight")), pos, il)
        V[il] = rms(v)
    ks, vs = K[src], V[src]                          # [n, nkv, hd]
    att = np.einsum("tgrd,sgd->grts", q.reshape(n, nkv, rep, hd), ks)      # Gemma 4: no 1/sqrt(hd) scaling
    mask = (j > i) | ((i - j >= SWA) if is_swa(il) else False)
    att = np.where(mask[None, None], -np.inf, att)
    att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
    o = np.einsum("grts,sgd->tgrd", att, vs).reshape(n, NH * hd)
    o = o @ W(p + "attn_output.weight").T
    sa = rms(o, W(p + "post_attention_norm.weight")) + x
    f = rms(sa, W(p + "ffn_norm.weight"))
    f = (gelu(f @ W(p + "ffn_gate.weight").T) * (f @ W(p + "ffn_up.weight").T)) @ W(p + "ffn_down.weight").T
    if p + "ffn_gate_inp.weight" in T:               # routed experts next to the dense feed-forward
        f = rms(f, W(p + "post_ffw_norm_1.weight"))
        x2 = rms(sa, W(p + "pre_ffw_norm_2.weight"))
        lg = (rms(sa) / np.sqrt(E) * W(p + "ffn_gate_inp.scale")) @ W(p + "ffn_gate_inp.weight").T      # [n, NX]
        pr = np.exp(lg - lg.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
        gu_t, dn_t = T[p + "ffn_gate_up_exps.weight"], T[p + "ffn_down_exps.weight"]
        sc = W(p + "ffn_down_exps.scale") if p + "ffn_down_exps.scale" in T else np.ones(NX)
        moe = np.zeros_like(sa)
        cache = {}
        for t in range(n):
            top = np.argsort(-pr[t], kind="stable")[:KX]
            w = pr[t, top]; w = w / max(w.sum(), 6.103515625e-5)
            for e, we in zip(top, w):
                if e not in cache:
                    cache[e] = (dequantize(gu_t.data[e], gu_t.tensor_type).astype(np.float64), dequantize(dn_t.data[e], dn_t.tensor_type).astype(np.float64))
                GU, DN = cache[e]
                gu = GU @ x2[t]; ff = gu.shape[0] // 2
                moe[t] += we * sc[e] * (DN @ (gelu(gu[:ff]) * gu[ff:]))
        f = f + rms(moe, W(p + "post_ffw_norm_2.weight"))
    y = rms(f, W(p + "post_ffw_norm.weight")) + sa
    if pli is not None:
        g = gelu(y @ W(p + "inp_gate.weight").T) * pli[:, il, :]
        y = y + rms(g @ W(p + "proj.weight").T, W(p + "post_norm.weight"))
    if p + "layer_output_scale.weight" in T:
        y = y * W(p + "layer_output_scale.weight")
    x = y
    states.append(x.copy())
hn = rms(x, W("output_norm.weight"))
if dump:
    np.savez(dump, states=np.stack(states).astype(np.float32), hidden=hn.astype(np.float32), pli=(pli if pli is not None else np.zeros(1)).astype(np.float32))
# output layer in slices of the vocabulary (the dequantized matrix of the larger models is several GB)
nv = emb_t.data.shape[0]
lg = np.empty((n, nv))
for a in range(0, nv, 16384):
    lg[:, a:a + 16384] = hn @ dequantize(emb_t.data[a:a + 16384], emb_t.tensor_type).astype(np.float64).T
if CAP: lg = CAP * np.tanh(lg / CAP)
for t in range(n):
    top = np.argsort(-lg[t])[:5]
    print(" ".join(f"{i}:{lg[t, i]:.3f}" for i in top))
