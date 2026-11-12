// tern command line: generate text from a prompt (or many prompts in parallel) and report speed.
#define _GNU_SOURCE
#include "tern.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

// peak resident memory of this process, from /proc (Linux)
static void print_memory(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256]; long hwm = 0, anon = 0, file = 0;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "VmHWM:", 6)) hwm = atol(line + 6);
        else if (!strncmp(line, "RssAnon:", 8)) anon = atol(line + 8);
        else if (!strncmp(line, "RssFile:", 8)) file = atol(line + 8);
    }
    fclose(f);
    fprintf(stderr, "memory: peak %.0f MB resident (at exit: %.0f MB private + %.0f MB mapped files, which the system can reclaim)\n",
            hwm / 1024.0, anon / 1024.0, file / 1024.0);
}

static void usage(void) {
    fprintf(stderr,
        "usage: tern -m model.gguf (-p \"prompt\" | -f prompts.txt | -i) [options]\n"
        "  -n N          tokens to generate per prompt (default 128)\n"
        "  -t N          threads (default: one per physical core)\n"
        "  -c N          context size in tokens (default 2048; 512 with -f)\n"
        "  -f FILE       one prompt per line, generated in parallel (see -np)\n"
        "  -np N         sequences generated together in one pass over the weights (default 4, with -f; at most %d)\n"
        "  -i            interactive chat (reads one line per turn from standard input)\n"
        "  --temp T      sampling temperature (default 0 = always the most likely token)\n"
        "  --top-k K     sample among the K most likely tokens (default 40, at most 64)\n"
        "  --seed N      random seed for sampling\n"
        "  --raw         do not wrap the prompt in the chat template\n"
        "  --ids         print token ids (for comparing with a reference)\n"
        "  --top K       with --ids: print the K best tokens and logits at each step\n"
        "  --tokenize    only print the token ids of the prompt\n"
        "  --quiet       with -f: do not print the generated text\n"
        "  --head PLAN   fast output layer (PLAN: a file made with --make-head); without it the exact layer is used\n"
        "  --keep N      fast output layer: tokens that are scored with their full rows (default 1024)\n"
        "  --head-check  with --head: also run the exact layer, follow it, and count how often the fast one agrees\n"
        "  --make-head FILE  write the plan for the fast output layer from the text generated in this run (use -f\n"
        "                with a few dozen prompts or more), or with --from-dump FILE from states recorded by --dump\n"
        "  --assistant FILE  speculative decoding for one stream with the model's drafter (the \"mtp-...\" GGUF published\n"
        "                with Gemma 4): each pass also checks guessed next tokens. Same text, fewer passes.\n"
        "  --draft FILE  the same with simpler guesses: an n-gram table from tools/build_draft.py, or the word\n"
        "                \"copy\" to guess only by copying from the context\n"
        "  --spec N      guessed tokens per pass (default 2; 3 can be better where memory is slow and cores are few)\n"
        "  --draft-pmin X  stop guessing when the drafter gives its best token less than this probability (default 0)\n"
        "  --draft-conf X  only use n-gram guesses that were right at least this share of the time (default 0.5)\n"
        "  --dump FILE   append (token id, final hidden state) of every generated step to FILE (for research tools)\n", TERN_MAX_SLOTS);
}

typedef struct {
    const char *model_path, *prompt, *file, *dump_path, *draft_path, *asst_path, *head_path, *make_head, *from_dump;
    int keep, check;
    int spec; float draft_conf, draft_pmin;
    int n_gen, n_threads, n_ctx, raw, ids, topk, only_tok, n_par, quiet, interactive, top_k;
    float temp; uint64_t seed;
} opts;

static int encode_prompt(const tokenizer *tk, const char *prompt, int raw, int *toks, int max) {
    if (raw) return tok_encode(tk, prompt, 1, 1, toks, max);
    size_t l = strlen(prompt) + 128; char *text = malloc(l);
    snprintf(text, l, "<|turn>user\n%s<turn|>\n<|turn>model\n", prompt);
    int n = tok_encode(tk, text, 1, 1, toks, max);
    free(text);
    return n;
}

// feed tokens of one sequence in chunks; the last token's hidden state goes to out
static void eval_prompt(model *m, int seq, const int *toks, int n, float *out) {
    slot sl[TERN_MAX_SLOTS]; int path[TERN_MAX_SLOTS]; float *want[TERN_MAX_SLOTS];
    for (int pos = 0; pos < n;) {
        int k = n - pos < TERN_MAX_SLOTS ? n - pos : TERN_MAX_SLOTS;
        for (int i = 0; i < k; i++) { sl[i] = (slot){toks[pos + i], seq, i - 1}; path[i] = i; want[i] = NULL; }
        if (pos + k == n) want[k - 1] = out;
        model_forward(m, sl, k, want);
        model_commit(m, seq, path, k);
        pos += k;
    }
}

static long chk_n, chk_same;   // --head-check counters

// --make-head: final hidden states of the generated steps (a uniform sample of at most REC_MAX of them)
#define REC_MAX 8192
#define PLAN_DIRS 32
static float *rec_h; static long rec_n; static int rec_e; static uint64_t rec_rng = 88172645463325252ULL;
static void record_state(const float *h, int E) {
    if (!rec_h) { rec_h = malloc(sizeof(float) * (size_t)REC_MAX * E); rec_e = E; }
    long i = rec_n++;
    if (i >= REC_MAX) {
        rec_rng ^= rec_rng << 13; rec_rng ^= rec_rng >> 7; rec_rng ^= rec_rng << 17;
        i = (long)(rec_rng % (uint64_t)rec_n);
        if (i >= REC_MAX) return;
    }
    memcpy(rec_h + (size_t)i * E, h, sizeof(float) * (size_t)E);
}
// the same from a file written by --dump (records: token id, E floats)
static int record_from_dump(const char *path, int E) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    float *row = malloc(sizeof(float) * (size_t)E); int tok;
    while (fread(&tok, 4, 1, f) == 1 && fread(row, 4, (size_t)E, f) == (size_t)E) record_state(row, E);
    fclose(f); free(row);
    return 0;
}

typedef struct {
    const opts *o; model *m; tokenizer *tk; fasthead *fh;
    float *hid, *logits; int end_turn, eos; uint64_t rng;
    int ch_open, ch_close, in_channel;      // the model's thought channel is not printed
    FILE *dump;
    drafter *dr; assistant *as; int *hist, n_hist; float *hs[TERN_MAX_SLOTS], *dh[2];      // speculative decoding
    long passes, guessed, confirmed;
} gen_ctx;

static double rnd01(uint64_t *s) {   // xorshift64*
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (double)((*s * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0;
}

// sample from the k best (token, logit) pairs, best first
static int sample_top(const int *tok, const float *logit, int k, float temp, uint64_t *rng) {
    double p[64], sum = 0;
    for (int i = 0; i < k; i++) { p[i] = exp((double)(logit[i] - logit[0]) / temp); sum += p[i]; }
    double r = rnd01(rng) * sum;
    for (int i = 0; i < k; i++) { r -= p[i]; if (r <= 0) return tok[i]; }
    return tok[0];
}

// the next token for the current hidden state (greedy, or sampled when a temperature is set)
static int pick_token(gen_ctx *g) {
    const opts *o = g->o;
    int k = o->top_k < 1 ? 1 : o->top_k > 64 ? 64 : o->top_k;
    if (g->fh) {
        int next, tt[64]; float tl[64], *hp = g->hid;
        head_argmax(g->fh, &hp, 1, &next, tt, tl);
        if (o->check) {
            int ex = model_head_exact(g->m, g->hid, g->logits);
            chk_n++; chk_same += ex == next;
            if (ex != next && getenv("TERN_VERBOSE")) {
                char a[64], b[64]; int la = tok_piece(g->tk, ex, a, 63), lb = tok_piece(g->tk, next, b, 63); a[la] = 0; b[lb] = 0;
                fprintf(stderr, "[fast output layer chose %d '%s' instead of %d '%s']\n", next, b, ex, a);
            }
            return ex;
        }
        if (o->temp > 0) {
            while (k > 1 && tt[k - 1] < 0) k--;
            next = sample_top(tt, tl, k, o->temp, &g->rng);
        }
        return next;
    }
    int next = model_head_exact(g->m, g->hid, g->logits);
    if (o->temp > 0 && !o->ids) {      // k best of the full logits
        int tt[64]; float tl[64]; int V = model_n_vocab(g->m), n = 0;
        for (int v = 0; v < V; v++) {
            float x = g->logits[v];
            if (n == k && x <= tl[k - 1]) continue;
            int i = n < k ? n++ : k - 1;
            while (i > 0 && tl[i - 1] < x) { tl[i] = tl[i - 1]; tt[i] = tt[i - 1]; i--; }
            tl[i] = x; tt[i] = v;
        }
        next = sample_top(tt, tl, n, o->temp, &g->rng);
    }
    return next;
}

// feed tokens to sequence 0 (prompt or the next chat turn)
static void feed(gen_ctx *g, const int *toks, int n) {
    eval_prompt(g->m, 0, toks, n, g->hid);
    memcpy(g->hist + g->n_hist, toks, sizeof(int) * (size_t)n); g->n_hist += n;
}

static void emit(gen_ctx *g, int tok) {
    if (g->o->ids || tok == g->end_turn || tok == g->eos) return;
    if (tok == g->ch_open) { g->in_channel = 1; return; }
    if (tok == g->ch_close) { g->in_channel = 0; return; }
    if (g->in_channel) return;
    char buf[256]; int l = tok_piece(g->tk, tok, buf, sizeof buf);
    fwrite(buf, 1, l, stdout); fflush(stdout);
}

// Generate until the end of the turn or n_gen tokens; returns the number of tokens produced.
// With a drafter (greedy decoding only), each pass also evaluates guessed followers of the new token; the
// guesses the model confirms are accepted without a pass of their own. The text is the same either way.
static int generate(gen_ctx *g, int n_gen, double *t_head) {
    const opts *o = g->o; model *m = g->m; int E = model_n_embd(m), n_out = 0, have_next = 0, next = -1;
    int spec = (g->dr || g->as) && o->temp <= 0 && !o->ids && !g->dump && !o->check && !o->make_head ? o->spec : 0;
    while (n_out < n_gen && model_len(m, 0) < model_n_ctx(m)) {
        if (!have_next) { double th = now_sec(); next = pick_token(g); *t_head += now_sec() - th; }
        have_next = 0;
        if (g->dump) { fwrite(&next, 4, 1, g->dump); fwrite(g->hid, 4, (size_t)E, g->dump); }
        if (o->make_head) record_state(g->hid, E);
        if (o->ids) {
            printf("%d", next);
            for (int k = 0; k < o->topk; k++) {
                int b = 0; for (int v = 1; v < model_n_vocab(m); v++) if (g->logits[v] > g->logits[b]) b = v;
                printf(" %d:%.3f", b, g->logits[b]); g->logits[b] = -INFINITY;
            }
            printf("\n");
        }
        emit(g, next); n_out++;
        if (next == g->end_turn || next == g->eos) break;
        g->hist[g->n_hist++] = next;
        // this token plus guessed followers, as a chain
        slot sl[TERN_MAX_SLOTS]; int ns = 1, path[TERN_MAX_SLOTS]; float *want[TERN_MAX_SLOTS];
        sl[0] = (slot){next, 0, -1};
        const float *dh = g->hid;       // the state that predicted the last token of the chain
        while (ns <= spec && n_out + ns <= n_gen && model_len(m, 0) + ns < model_n_ctx(m)) {
            int d;
            if (g->as) {
                float p = 1.0f, *h2 = g->dh[ns & 1];
                d = assistant_step(g->as, 0, sl[ns - 1].token, dh, h2, o->draft_pmin > 0 ? &p : NULL);
                if (p < o->draft_pmin) break;
                dh = h2;
            } else d = draft_next(g->dr, g->hist, g->n_hist + ns - 1);
            if (d < 0 || d == g->end_turn || d == g->eos) break;
            g->hist[g->n_hist + ns - 1] = d;        // provisional, beyond n_hist
            sl[ns] = (slot){d, 0, ns - 1}; ns++;
        }
        for (int i = 0; i < ns; i++) { want[i] = g->hs[i]; path[i] = i; }
        model_forward(m, sl, ns, want);
        g->passes++;
        if (ns == 1) { model_commit(m, 0, path, 1); memcpy(g->hid, g->hs[0], sizeof(float) * (size_t)E); continue; }
        // what does the model say after each slot?
        double th = now_sec();
        int outs[TERN_MAX_SLOTS];
        if (g->fh) head_argmax(g->fh, g->hs, ns, outs, NULL, NULL);
        else model_head_exact_batch(m, g->hs, ns, outs);
        *t_head += now_sec() - th;
        int acc = 0;
        while (acc < ns - 1 && outs[acc] == sl[acc + 1].token) acc++;
        g->guessed += ns - 1; g->confirmed += acc;
        model_commit(m, 0, path, acc + 1);
        int stop = 0;
        for (int i = 1; i <= acc && !stop; i++) {
            emit(g, sl[i].token); n_out++;
            if (sl[i].token == g->end_turn || sl[i].token == g->eos) stop = 1; else g->n_hist++;
        }
        if (stop) break;
        next = outs[acc]; have_next = 1;
        memcpy(g->hid, g->hs[acc], sizeof(float) * (size_t)E);
    }
    return n_out;
}

static int run_single(const opts *o, model *m, tokenizer *tk, fasthead *fh, double t_load) {
    int E = model_n_embd(m), n_ctx = model_n_ctx(m);
    gen_ctx g = {.o = o, .m = m, .tk = tk, .fh = fh, .rng = o->seed ? o->seed : (uint64_t)(now_sec() * 1e6) | 1};
    g.hid = malloc(sizeof(float) * (size_t)E);
    g.logits = (o->topk || o->temp > 0 || o->check) ? malloc(sizeof(float) * (size_t)model_n_vocab(m)) : NULL;
    g.end_turn = tok_find(tk, "<turn|>"); g.eos = tok_eos(tk);
    g.ch_open = tok_find(tk, "<|channel>"); g.ch_close = tok_find(tk, "<channel|>");
    g.dump = o->dump_path ? fopen(o->dump_path, "ab") : NULL;
    int *toks = malloc(sizeof(int) * (size_t)n_ctx);
    g.hist = malloc(sizeof(int) * (size_t)(n_ctx + TERN_MAX_SLOTS)); g.n_hist = 0;
    for (int i = 0; i < TERN_MAX_SLOTS; i++) g.hs[i] = malloc(sizeof(float) * (size_t)E);
    g.dh[0] = malloc(sizeof(float) * (size_t)E); g.dh[1] = malloc(sizeof(float) * (size_t)E);
    if (o->asst_path) {
        gguf *ga = gguf_open(o->asst_path);
        if (!ga || !(g.as = assistant_load(ga, m))) return 1;
    } else if (o->draft_path) {
        g.dr = draft_load(strcmp(o->draft_path, "copy") ? o->draft_path : NULL, o->draft_conf);
        if (!g.dr) return 1;
    }

    if (o->interactive) {       // chat: one line per turn, the conversation stays in the cache
        char *line = NULL; size_t cap = 0; int turn = 0;
        fprintf(stderr, "tern chat - empty line or Ctrl-D to quit\n");
        for (;;) {
            fprintf(stderr, "\n> "); fflush(stderr);
            ssize_t l = getline(&line, &cap, stdin);
            if (l <= 0) break;
            while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
            if (l == 0) break;
            size_t tl = (size_t)l + 128; char *text = malloc(tl);
            snprintf(text, tl, "%s<|turn>user\n%s<turn|>\n<|turn>model\n", turn ? "<turn|>\n" : "", line);
            int n = tok_encode(tk, text, turn == 0, 1, toks, n_ctx);
            free(text);
            if (model_len(m, 0) + n + 8 >= n_ctx) { fprintf(stderr, "[context is full: start again]\n"); break; }
            feed(&g, toks, n);
            double t0 = now_sec(), th = 0;
            int n_out = generate(&g, o->n_gen, &th);
            double dt = now_sec() - t0;
            fprintf(stderr, "\n[%d tokens, %.1f tok/s]\n", n_out, n_out / dt);
            turn++;
        }
        return 0;
    }
    int n_prompt = encode_prompt(tk, o->prompt, o->raw, toks, n_ctx);
    double tp0 = now_sec();
    feed(&g, toks, n_prompt);
    double t_prompt = now_sec() - tp0, tg0 = now_sec(), t_head = 0;
    int n_out = generate(&g, o->n_gen, &t_head);
    double t_gen = now_sec() - tg0;
    if (g.dump) fclose(g.dump);
    fprintf(stderr, "\n\nload %.2f s | prompt %d tokens, %.1f tok/s | generation %d tokens, %.2f tok/s (output layer %.0f%% of the time) | weights %.0f MB on %s pages | %s, %d threads\n",
            t_load, n_prompt, n_prompt / t_prompt, n_out, n_out / t_gen, 100 * t_head / t_gen, model_weight_bytes(m) / 1e6, tern_huge_pages() ? "huge" : "4 KB", kernels_name(), pool_threads());
    if (g.dr || g.as) fprintf(stderr, "speculation: %ld passes for %d tokens (%.2f tokens per pass); %ld guesses, %ld confirmed (%.0f%%)\n",
                      g.passes, n_out, g.passes ? (double)n_out / g.passes : 0, g.guessed, g.confirmed, g.guessed ? 100.0 * g.confirmed / g.guessed : 0);
    if (g.as) { double ms; long steps; assistant_stats(g.as, &ms, &steps); fprintf(stderr, "drafter: %ld steps, %.2f ms per step\n", steps, ms); }
    return 0;
}

// Many prompts, n_par sequences at a time: every pass evaluates one token of each active sequence.
// The fast output layer saves memory traffic, not arithmetic: with more than FAST_HEAD_SLOTS states per call its
// scan costs as much as the exact sweep, which is then used instead.
#define FAST_HEAD_SLOTS 8
typedef struct { int active, id, n_out, n_prompt; int *out; float *hid, *rec; } seqst;

static int run_batch(const opts *o, model *m, tokenizer *tk, fasthead *fh) {
    int E = model_n_embd(m), n_ctx = model_n_ctx(m), NP = o->n_par;
    FILE *f = fopen(o->file, "r");
    if (!f) { perror(o->file); return 1; }
    FILE *dump = o->dump_path ? fopen(o->dump_path, "ab") : NULL;
    int end_turn = tok_find(tk, "<turn|>"), eos = tok_eos(tk), ch_open = tok_find(tk, "<|channel>"), ch_close = tok_find(tk, "<channel|>");
    seqst *st = calloc((size_t)NP, sizeof *st);
    int *toks = malloc(sizeof(int) * (size_t)n_ctx);
    for (int q = 0; q < NP; q++) {
        st[q].out = malloc(sizeof(int) * (size_t)o->n_gen); st[q].hid = malloc(sizeof(float) * (size_t)E);
        st[q].rec = dump ? malloc(sizeof(float) * (size_t)E * o->n_gen) : NULL;
    }
    char *line = NULL; size_t cap = 0; int next_id = 0, eof = 0, n_active = 0;
    long total_out = 0, total_prompt = 0, n_done = 0; double t_prompt = 0, t0 = now_sec();
    for (;;) {
        for (int q = 0; q < NP && !eof; q++) if (!st[q].active) {     // refill free sequences
            ssize_t l = getline(&line, &cap, f);
            if (l < 0) { eof = 1; break; }
            while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
            if (l == 0) { q--; continue; }
            int n = encode_prompt(tk, line, o->raw, toks, n_ctx - o->n_gen);
            model_truncate(m, q, 0);
            double tp = now_sec();
            eval_prompt(m, q, toks, n, st[q].hid);
            t_prompt += now_sec() - tp; total_prompt += n;
            st[q].active = 1; st[q].id = next_id++; st[q].n_out = 0; st[q].n_prompt = n; n_active++;
        }
        if (n_active == 0) break;
        // output layer for all active sequences in one sweep, then one pass for their next tokens
        float *hp[TERN_MAX_SLOTS]; int who[TERN_MAX_SLOTS], next[TERN_MAX_SLOTS], n = 0;
        for (int q = 0; q < NP; q++) if (st[q].active) { hp[n] = st[q].hid; who[n++] = q; }
        if (fh && (n <= FAST_HEAD_SLOTS || o->check)) {
            head_argmax(fh, hp, n, next, NULL, NULL);
            if (o->check) {
                int ex[TERN_MAX_SLOTS];
                model_head_exact_batch(m, hp, n, ex);
                for (int i = 0; i < n; i++) { chk_n++; chk_same += ex[i] == next[i]; next[i] = ex[i]; }
            }
        } else model_head_exact_batch(m, hp, n, next);
        slot sl[TERN_MAX_SLOTS]; int ns = 0, sq[TERN_MAX_SLOTS];
        for (int i = 0; i < n; i++) {
            seqst *s = &st[who[i]];
            if (s->rec) memcpy(s->rec + (size_t)s->n_out * E, s->hid, sizeof(float) * (size_t)E);
            if (o->make_head) record_state(s->hid, E);
            s->out[s->n_out++] = next[i]; total_out++;
            if (next[i] == end_turn || next[i] == eos || s->n_out >= o->n_gen) {
                if (dump) for (int k = 0; k < s->n_out; k++) { fwrite(&s->out[k], 4, 1, dump); fwrite(s->rec + (size_t)k * E, 4, (size_t)E, dump); }
                if (!o->quiet) {
                    printf("=== %d\n", s->id);
                    for (int k = 0, hide = 0; k < s->n_out; k++) {
                        int t = s->out[k];
                        if (t == ch_open) hide = 1; else if (t == ch_close) hide = 0;
                        else if (!hide && t != end_turn && t != eos) { char buf[256]; int l = tok_piece(tk, t, buf, sizeof buf); fwrite(buf, 1, l, stdout); }
                    }
                    printf("\n");
                }
                s->active = 0; n_active--; n_done++;
            } else { sl[ns] = (slot){next[i], who[i], -1}; sq[ns++] = who[i]; }
        }
        if (ns > 0) {
            float *want[TERN_MAX_SLOTS];
            for (int i = 0; i < ns; i++) want[i] = st[sq[i]].hid;
            model_forward(m, sl, ns, want);
            for (int i = 0; i < ns; i++) model_commit(m, sq[i], &i, 1);
        }
    }
    double dt = now_sec() - t0;
    if (dump) fclose(dump);
    fprintf(stderr, "%ld prompts, %d at a time | %ld generated tokens in %.1f s = %.1f tok/s total (prompt processing %.1f s for %ld tokens included; generation alone %.1f tok/s) | %s, %d threads\n",
            n_done, NP, total_out, dt, total_out / dt, t_prompt, total_prompt, total_out / (dt - t_prompt), kernels_name(), pool_threads());
    return 0;
}

// one thread per physical core that this process may use
static int default_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN); int cores = 0; char seen[4096] = {0};
    for (int c = 0; c < n && c < 1024; c++) {
        char path[96]; int core = c, pkg = 0;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        FILE *f = fopen(path, "r"); if (f) { if (fscanf(f, "%d", &core) != 1) core = c; fclose(f); }
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        f = fopen(path, "r"); if (f) { if (fscanf(f, "%d", &pkg) != 1) pkg = 0; fclose(f); }
        int key = (pkg * 64 + core) & 4095;
        if (!seen[key]) { seen[key] = 1; cores++; }
    }
    return cores < 1 ? 1 : cores > 64 ? 64 : cores;
}

int main(int argc, char **argv) {
    opts o = {.n_gen = 128, .n_threads = 0, .n_ctx = 0, .n_par = 4, .top_k = 40, .spec = 2, .draft_conf = 0.5f, .keep = 1024};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) o.model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) o.prompt = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) o.file = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) o.n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) o.n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) o.n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-np") && i + 1 < argc) o.n_par = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--raw")) o.raw = 1;
        else if (!strcmp(argv[i], "--ids")) o.ids = 1;
        else if (!strcmp(argv[i], "--quiet")) o.quiet = 1;
        else if (!strcmp(argv[i], "--top") && i + 1 < argc) o.topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokenize")) o.only_tok = 1;
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) o.dump_path = argv[++i];
        else if (!strcmp(argv[i], "-i")) o.interactive = 1;
        else if (!strcmp(argv[i], "--head") && i + 1 < argc) o.head_path = argv[++i];
        else if (!strcmp(argv[i], "--keep") && i + 1 < argc) o.keep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--make-head") && i + 1 < argc) o.make_head = argv[++i];
        else if (!strcmp(argv[i], "--from-dump") && i + 1 < argc) o.from_dump = argv[++i];
        else if (!strcmp(argv[i], "--head-check")) o.check = 1;
        else if (!strcmp(argv[i], "--draft") && i + 1 < argc) o.draft_path = argv[++i];
        else if (!strcmp(argv[i], "--assistant") && i + 1 < argc) o.asst_path = argv[++i];
        else if (!strcmp(argv[i], "--draft-pmin") && i + 1 < argc) o.draft_pmin = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) o.spec = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft-conf") && i + 1 < argc) o.draft_conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) o.temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) o.top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) o.seed = strtoull(argv[++i], NULL, 10);
        else { usage(); return 1; }
    }
    if (!o.model_path || (!o.prompt && !o.file && !o.interactive && !(o.make_head && o.from_dump))) { usage(); return 1; }
    if (o.n_par < 1) o.n_par = 1;
    if (o.n_par > TERN_MAX_SLOTS) o.n_par = TERN_MAX_SLOTS;
    if (o.spec > TERN_MAX_SLOTS - 1) o.spec = TERN_MAX_SLOTS - 1;
    if (!o.n_ctx) o.n_ctx = o.file ? 512 : 2048;
    if (o.n_threads <= 0) o.n_threads = default_threads();

    double t0 = now_sec();
    gguf *g = gguf_open(o.model_path);
    if (!g) return 1;
    tokenizer *tk = tok_load(g);
    if (!tk) return 1;
    if (o.only_tok) {
        int *toks = malloc(sizeof(int) * 65536);
        int n = encode_prompt(tk, o.prompt, o.raw, toks, 65536);
        for (int i = 0; i < n; i++) printf("%d%s", toks[i], i + 1 < n ? " " : "\n");
        return 0;
    }
    pool_init(o.n_threads);
    model_params mp = {.n_threads = o.n_threads, .n_ctx = o.n_ctx, .n_seq = o.file ? o.n_par : 1};
    model *m = model_load(g, &mp);
    if (!m) return 1;
    gguf_release_pages(g);
    if (getenv("TERN_SWEEP")) {
        for (int S = 1; S <= TERN_MAX_SLOTS; S *= 2) {
            double t = model_bench_sweep(m, S, S > 4 ? 3 : 10);
            printf("matrices only, %2d slots: %.3f ms/pass  (%.1f passes/s, %.1f slot-tokens/s)\n", S, t * 1e3, 1 / t, S / t);
        }
        return 0;
    }
    if (o.make_head && o.from_dump) {
        if (record_from_dump(o.from_dump, model_n_embd(m))) return 1;
        return head_make_plan(o.make_head, rec_h, rec_n < REC_MAX ? (int)rec_n : REC_MAX, model_n_embd(m), PLAN_DIRS);
    }
    fasthead *fh = NULL;
    if (o.head_path && !o.make_head && !(fh = head_load(o.head_path, m, o.keep))) return 1;
    int rc = o.file ? run_batch(&o, m, tk, fh) : run_single(&o, m, tk, fh, now_sec() - t0);
    if (o.make_head && !rc) rc = head_make_plan(o.make_head, rec_h, rec_n < REC_MAX ? (int)rec_n : REC_MAX, model_n_embd(m), PLAN_DIRS);
    if (getenv("TERN_PROF")) model_print_profile();
    print_memory();
    if (fh) {
        double ms, cands; long calls; head_stats(fh, &ms, &cands, &calls);
        fprintf(stderr, "fast output layer: %.3f ms per call, %ld calls, %.0f full rows per token", ms, calls, cands);
        if (o.check && chk_n) fprintf(stderr, " | same token as the exact layer: %ld of %ld (%.3f%%)", chk_same, chk_n, 100.0 * chk_same / chk_n);
        fprintf(stderr, "\n");
    }
    return rc;
}
