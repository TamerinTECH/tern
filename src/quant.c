// GGUF row formats: sizes and dequantization (used for lookups and for repacking at load; not a hot path).
#include "tern.h"
#include <string.h>

float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) { if (m == 0) u = s << 31; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; u = (s << 31) | ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) u = (s << 31) | 0x7f800000u | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

uint16_t f32_to_f16(float f) {      // round to nearest even
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u, mant = x & 0x7fffffu; int e = (int)((x >> 23) & 0xff) - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);                       // overflow -> inf
    if (e <= 0) {                                                         // subnormal or zero
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u; int shift = 14 - e;
        uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (mant >> 13), rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

const char *quant_type_name(uint32_t type) {
    switch (type) {
        case GT_F32: return "F32"; case GT_F16: return "F16"; case GT_BF16: return "BF16";
        case GT_Q4_0: return "Q4_0"; case GT_Q4_1: return "Q4_1"; case GT_Q8_0: return "Q8_0";
        case GT_Q4_K: return "Q4_K"; case GT_Q5_K: return "Q5_K"; case GT_Q6_K: return "Q6_K";
        default: return "unsupported";
    }
}

size_t quant_row_bytes(uint32_t type, int cols) {
    switch (type) {
        case GT_F32:  return (size_t)cols * 4;
        case GT_F16: case GT_BF16: return (size_t)cols * 2;
        case GT_Q4_0: return cols % 32 ? 0 : (size_t)cols / 32 * 18;
        case GT_Q4_1: return cols % 32 ? 0 : (size_t)cols / 32 * 20;
        case GT_Q8_0: return cols % 32 ? 0 : (size_t)cols / 32 * 34;
        case GT_Q4_K: return cols % 256 ? 0 : (size_t)cols / 256 * 144;
        case GT_Q5_K: return cols % 256 ? 0 : (size_t)cols / 256 * 176;
        case GT_Q6_K: return cols % 256 ? 0 : (size_t)cols / 256 * 210;
        default: return 0;
    }
}

static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

// 6-bit scale and minimum of sub-block j of a Q4_K / Q5_K super-block
static void scale_min_k4(int j, const uint8_t *q, int *d, int *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

int dequant_row(uint32_t type, const uint8_t *src, int cols, float *out) {
    switch (type) {
    case GT_F32: memcpy(out, src, (size_t)cols * 4); return 0;
    case GT_F16: for (int i = 0; i < cols; i++) out[i] = f16_to_f32(rd16(src + 2 * i)); return 0;
    case GT_BF16: for (int i = 0; i < cols; i++) { uint32_t u = (uint32_t)rd16(src + 2 * i) << 16; memcpy(out + i, &u, 4); } return 0;
    case GT_Q4_0:
        for (int b = 0; b < cols / 32; b++, src += 18) {
            float d = f16_to_f32(rd16(src));
            for (int j = 0; j < 16; j++) { out[b * 32 + j] = d * (float)((src[2 + j] & 15) - 8); out[b * 32 + j + 16] = d * (float)((src[2 + j] >> 4) - 8); }
        }
        return 0;
    case GT_Q4_1:
        for (int b = 0; b < cols / 32; b++, src += 20) {
            float d = f16_to_f32(rd16(src)), m = f16_to_f32(rd16(src + 2));
            for (int j = 0; j < 16; j++) { out[b * 32 + j] = d * (float)(src[4 + j] & 15) + m; out[b * 32 + j + 16] = d * (float)(src[4 + j] >> 4) + m; }
        }
        return 0;
    case GT_Q8_0:
        for (int b = 0; b < cols / 32; b++, src += 34) {
            float d = f16_to_f32(rd16(src));
            for (int j = 0; j < 32; j++) out[b * 32 + j] = d * (float)(int8_t)src[2 + j];
        }
        return 0;
    case GT_Q4_K:
        for (int b = 0; b < cols / 256; b++, src += 144) {
            float d = f16_to_f32(rd16(src)), dmin = f16_to_f32(rd16(src + 2));
            const uint8_t *sc = src + 4, *q = src + 16;
            for (int j = 0; j < 4; j++, q += 32) {
                int s1, m1, s2, m2; scale_min_k4(2 * j, sc, &s1, &m1); scale_min_k4(2 * j + 1, sc, &s2, &m2);
                float d1 = d * (float)s1, o1 = dmin * (float)m1, d2 = d * (float)s2, o2 = dmin * (float)m2;
                float *y = out + b * 256 + j * 64;
                for (int l = 0; l < 32; l++) { y[l] = d1 * (float)(q[l] & 15) - o1; y[l + 32] = d2 * (float)(q[l] >> 4) - o2; }
            }
        }
        return 0;
    case GT_Q5_K:
        for (int b = 0; b < cols / 256; b++, src += 176) {
            float d = f16_to_f32(rd16(src)), dmin = f16_to_f32(rd16(src + 2));
            const uint8_t *sc = src + 4, *qh = src + 16, *q = src + 48;
            int u1 = 1, u2 = 2;
            for (int j = 0; j < 4; j++, q += 32, u1 <<= 2, u2 <<= 2) {
                int s1, m1, s2, m2; scale_min_k4(2 * j, sc, &s1, &m1); scale_min_k4(2 * j + 1, sc, &s2, &m2);
                float d1 = d * (float)s1, o1 = dmin * (float)m1, d2 = d * (float)s2, o2 = dmin * (float)m2;
                float *y = out + b * 256 + j * 64;
                for (int l = 0; l < 32; l++) {
                    y[l] = d1 * (float)((q[l] & 15) + (qh[l] & u1 ? 16 : 0)) - o1;
                    y[l + 32] = d2 * (float)((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - o2;
                }
            }
        }
        return 0;
    case GT_Q6_K:
        for (int b = 0; b < cols / 256; b++, src += 210) {
            const uint8_t *ql = src, *qh = src + 128; const int8_t *sc = (const int8_t *)(src + 192);
            float d = f16_to_f32(rd16(src + 208));
            float *y = out + b * 256;
            for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8)
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = ((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32, q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * (float)sc[is] * (float)q1; y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                    y[l + 64] = d * (float)sc[is + 4] * (float)q3; y[l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
        }
        return 0;
    default: return -1;
    }
}
