R"CL(
// finish.cpp and hdr_luma() on the device. RGB images are interleaved float3 rows.

#define NEG_FLOOR 0.0625f
#define CHROMA_FLOOR 4e-3f

inline float lsc_gain(__global const float* L, int cols, int x0, int y0, float fx, float fy, int c) {
    __global const float* a = L + ((size_t)y0 * cols + x0) * 4 + c;
    __global const float* b = a + (size_t)cols * 4;
    return (1 - fy) * ((1 - fx) * a[0] + fx * a[4]) + fy * ((1 - fx) * b[0] + fx * b[4]);
}

__kernel void normalize_mosaic(__global const float* mosaic, __global float* out, int W, int H, int cfa, float4 black4,
                               float white, float4 wb4, __global const float* lsc, int rows, int cols, int cropX,
                               int cropY, float sw, float sh, float lumaStrength) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int ch = cfa_channel(cfa, x, y);
    float black = ch == 0 ? black4.x : ch == 1 ? black4.y : ch == 2 ? black4.z : black4.w;
    float wb = ch == 0 ? wb4.x : ch == 1 ? wb4.y : ch == 2 ? wb4.z : wb4.w;
    float raw = mosaic[(size_t)y * W + x];
    if (x >= 2 && x < W - 2 && y >= 2 && y < H - 2) {
#define M(dx, dy) mosaic[(size_t)(y + (dy)) * W + (x + (dx))]
        float n0 = M(-2, -2), n1 = M(0, -2), n2 = M(2, -2);
        float n3 = M(-2, 0),               n4 = M(2, 0);
        float n5 = M(-2, 2),  n6 = M(0, 2),  n7 = M(2, 2);
#undef M
        float hi = max(max(max(n0, n1), max(n2, n3)), max(max(n4, n5), max(n6, n7)));
        float avg = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7) * 0.125f;
        float thr = hi + max(0.02f * (white - black), 2.f * (hi - avg));
        if (raw > thr) raw = hi;
    }
    float v = (raw - black) / (white - black);
    float gain = 1.f, clipv = 1.f;
    if (rows > 1 && cols > 1) {
        float gy = (y + cropY) / sh * (rows - 1);
        int y0 = min((int)gy, max(0, rows - 2));
        float fy = gy - y0;
        float gx = (x + cropX) / sw * (cols - 1);
        int x0 = min((int)gx, cols - 2);
        float fx = gx - x0;
        float gG = 0.5f * (lsc_gain(lsc, cols, x0, y0, fx, fy, 1) + lsc_gain(lsc, cols, x0, y0, fx, fy, 2));
        float lumaGain = pow(gG, lumaStrength);
        gain = lsc_gain(lsc, cols, x0, y0, fx, fy, ch) / gG * lumaGain;
        clipv = lumaGain;
    }
    out[(size_t)y * W + x] = clamp(v * gain * wb, -NEG_FLOOR, clipv);
}

__kernel void normalize_rgb(__global const float* in, __global float* rgb, int W, int H, float scale, float3 inv, float3 wb,
                            __global const float* lsc, int rows, int cols, int cropX, int cropY, float sw, float sh,
                            float lumaStrength) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float g0 = 1, g1 = 1, g2 = 1, clipv = 1.f;
    if (rows > 1 && cols > 1) {
        float ry = clamp((y + 0.5f) / scale - 0.5f + cropY, 0.f, sh);
        float gy = ry / sh * (rows - 1);
        int y0 = min((int)gy, max(0, rows - 2));
        float fy = gy - y0;
        float rx = clamp((x + 0.5f) / scale - 0.5f + cropX, 0.f, sw);
        float gx = rx / sw * (cols - 1);
        int x0 = min((int)gx, cols - 2);
        float fx = gx - x0;
        float gG = 0.5f * (lsc_gain(lsc, cols, x0, y0, fx, fy, 1) + lsc_gain(lsc, cols, x0, y0, fx, fy, 2));
        float lumaGain = pow(gG, lumaStrength);
        g0 = lsc_gain(lsc, cols, x0, y0, fx, fy, 0) / gG * lumaGain;
        g1 = lumaGain;
        g2 = lsc_gain(lsc, cols, x0, y0, fx, fy, 3) / gG * lumaGain;
        clipv = lumaGain;
    }
    __global const float* s = in + ((size_t)y * W + x) * 3;
    __global float* p = rgb + ((size_t)y * W + x) * 3;
    p[0] = clamp(s[0] * inv.x * g0 * wb.x, -NEG_FLOOR, clipv);
    p[1] = clamp(s[1] * inv.y * g1 * wb.y, -NEG_FLOOR, clipv);
    p[2] = clamp(s[2] * inv.z * g2 * wb.z, -NEG_FLOOR, clipv);
}

// Immerkaer noise estimate on the G_even sub-plane: partial (sum |laplacian|, count) per (row, group)
__kernel void sigma_rows_cfa(__global const float* cfa, int W, int H, int ox, int oy, __global float* out,
                             __local float2* red) {
    int x = get_global_id(0), y = get_global_id(1);
    int w = W / 2, h = H / 2;
    float2 sn = 0;
    if (x >= 1 && x < w - 1 && y >= 1 && y < h - 1) {
#define P(u, v) cfa[(size_t)(2 * (y + (v)) + oy) * W + 2 * (x + (u)) + ox]
        float c = P(0, 0);
        if (c <= 0.9f) {
            float l = 4 * c - 2 * (P(-1, 0) + P(1, 0) + P(0, -1) + P(0, 1)) + (P(-1, -1) + P(1, -1) + P(-1, 1) + P(1, 1));
            sn = (float2)(fabs(l), 1);
        }
#undef P
    }
    sn = group_sum2(red, sn);
    if (get_local_id(0) == 0) vstore2(sn, y * get_num_groups(0) + get_group_id(0), out);
}

__kernel void sigma_rows_rgb(__global const float* rgb, int W, int H, int step, __global float* out, __local float2* red) {
    int x = get_global_id(0), y = get_global_id(1);
    int w = W / step, h = H / step;
    float2 sn = 0;
    if (x >= 1 && x < w - 1 && y >= 1 && y < h - 1) {
#define P(u, v) rgb[((size_t)(step * (y + (v))) * W + step * (x + (u))) * 3 + 1]
        float c = P(0, 0);
        if (c <= 0.9f) {
            float l = 4 * c - 2 * (P(-1, 0) + P(1, 0) + P(0, -1) + P(0, 1)) + (P(-1, -1) + P(1, -1) + P(-1, 1) + P(1, 1));
            sn = (float2)(fabs(l), 1);
        }
#undef P
    }
    sn = group_sum2(red, sn);
    if (get_local_id(0) == 0) vstore2(sn, y * get_num_groups(0) + get_group_id(0), out);
}

__kernel void demosaic_mhc(__global const float* cfa, int W, int H, int cfaType, __global float* rgb) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
#define P(X, Y) at_clamped(cfa, W, H, X, Y)
    int c = cfa_color(cfaType, x, y);
    float v = P(x, y);
    float o0, o1, o2;  // no array indexed by colour: Adreno spills those to memory
    if (c == 1) {
        float rowK = (5 * v + 4 * (P(x - 1, y) + P(x + 1, y)) - (P(x - 2, y) + P(x + 2, y)) -
                      (P(x - 1, y - 1) + P(x + 1, y - 1) + P(x - 1, y + 1) + P(x + 1, y + 1)) + 0.5f * (P(x, y - 2) + P(x, y + 2))) * 0.125f;
        float colK = (5 * v + 4 * (P(x, y - 1) + P(x, y + 1)) - (P(x, y - 2) + P(x, y + 2)) -
                      (P(x - 1, y - 1) + P(x + 1, y - 1) + P(x - 1, y + 1) + P(x + 1, y + 1)) + 0.5f * (P(x - 2, y) + P(x + 2, y))) * 0.125f;
        bool rowRed = cfa_color(cfaType, x + 1, y) == 0;
        o0 = rowRed ? rowK : colK;
        o1 = v;
        o2 = rowRed ? colK : rowK;
    } else {
        float g = (4 * v + 2 * (P(x - 1, y) + P(x + 1, y) + P(x, y - 1) + P(x, y + 1)) - (P(x - 2, y) + P(x + 2, y) + P(x, y - 2) + P(x, y + 2))) * 0.125f;
        float other = (6 * v + 2 * (P(x - 1, y - 1) + P(x + 1, y - 1) + P(x - 1, y + 1) + P(x + 1, y + 1)) -
                       1.5f * (P(x - 2, y) + P(x + 2, y) + P(x, y - 2) + P(x, y + 2))) * 0.125f;
        o0 = c == 0 ? v : other;
        o1 = g;
        o2 = c == 0 ? other : v;
    }
#undef P
    __global float* o = rgb + ((size_t)y * W + x) * 3;
    o[0] = o0; o[1] = o1; o[2] = o2;
}

inline float sample_ch(__global const float* im, int W, int H, float fx, float fy, int c) {
    fx = clamp(fx, 0.f, (float)(W - 1));
    fy = clamp(fy, 0.f, (float)(H - 1));
    int x0 = min((int)fx, max(0, W - 2)), y0 = min((int)fy, max(0, H - 2));
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float ax = fx - x0, ay = fy - y0;
    __global const float* r0 = im + ((size_t)y0 * W + x0) * 3 + c;
    __global const float* r1 = im + ((size_t)y1 * W + x0) * 3 + c;
    int dx = (x1 - x0) * 3;
    float a = r0[0] * (1.f - ax) + r0[dx] * ax;
    float b = r1[0] * (1.f - ax) + r1[dx] * ax;
    return a * (1.f - ay) + b * ay;
}

__kernel void lca_rgb(__global const float* in, int W, int H, float4 lcaK, float lcx, float lcy, float invR2,
                      __global float* out) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float px = (x + 0.5f) - lcx, py = (y + 0.5f) - lcy;
    float rr = (px * px + py * py) * invR2;
    float kr = lcaK.x + lcaK.y * rr, kb = lcaK.z + lcaK.w * rr;
    __global const float* s = in + ((size_t)y * W + x) * 3;
    __global float* d = out + ((size_t)y * W + x) * 3;
    d[0] = sample_ch(in, W, H, x + px * kr, y + py * kr, 0);
    d[1] = s[1];
    d[2] = sample_ch(in, W, H, x + px * kb, y + py * kb, 2);
}

// apply_ccm, then wb_shift's per-channel gains (all 1 when the shift is off)
__kernel void ccm_wb(__global float* rgb, int n, float3 m0, float3 m1, float3 m2, float3 gain) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global float* p = rgb + (size_t)i * 3;
    float r = p[0], g = p[1], b = p[2];
    float o0 = m0.x * r + m0.y * g + m0.z * b;
    float o1 = m1.x * r + m1.y * g + m1.z * b;
    float o2 = m2.x * r + m2.y * g + m2.z * b;
    p[0] = o0 * gain.x; p[1] = o1 * gain.y; p[2] = o2 * gain.z;
}

// display_gain(): sum of log luma over every 8th pixel of every 8th row
// sum of log luma over every 8th pixel of every 8th row: partial sums per (row, group)
__kernel void log_luma_rows(__global const float* rgb, int W, int H, __global float* out, __local float2* red) {
    int c = get_global_id(0), r = get_global_id(1);
    float v = 0;
    if (c * 8 < W) {
        __global const float* p = rgb + ((size_t)(r * 8) * W + c * 8) * 3;
        v = log(1e-4f + max(0.f, 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]));
    }
    float s = group_sum2(red, (float2)(v, 0)).x;
    if (get_local_id(0) == 0) out[r * get_num_groups(0) + get_group_id(0)] = s;
}

__kernel void chroma_down(__global const float* rgb, int W, int H, __global float* cr, __global float* cb, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float sr = 0, sb = 0;
    int n = 0;
    for (int v = 0; v < 4; v++)
        for (int u = 0; u < 4; u++) {
            int X = x * 4 + u, Y = y * 4 + v;
            if (X >= W || Y >= H) continue;
            __global const float* p = rgb + ((size_t)Y * W + X) * 3;
            float L = luma709(p[0], p[1], p[2]);
            float inv = 1.f / (max(L, 0.f) + CHROMA_FLOOR);
            sr += (p[0] - L) * inv; sb += (p[2] - L) * inv; n++;
        }
    cr[y * w + x] = sr / n;
    cb[y * w + x] = sb / n;
}

__kernel void blur_h(__global const float* in, __global float* out, int w, int h, __constant float* k, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    __global const float* s = in + (size_t)y * w;
    float a = 0;
    for (int i = -r; i <= r; i++) a += k[i + r] * s[clamp(x + i, 0, w - 1)];
    out[(size_t)y * w + x] = a;
}
__kernel void blur_v(__global const float* in, __global float* out, int w, int h, __constant float* k, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float a = 0;
    for (int i = -r; i <= r; i++) a += k[i + r] * in[(size_t)clamp(y + i, 0, h - 1) * w + x];
    out[(size_t)y * w + x] = a;
}

__kernel void chroma_apply(__global float* rgb, int W, int H, __global const float* cr, __global const float* cb, int w,
                           int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float gy = (y + 0.5f) / 4 - 0.5f;
    int y0 = clamp((int)floor(gy), 0, h - 1), y1 = min(y0 + 1, h - 1);
    float fy = clamp(gy - y0, 0.f, 1.f);
    float gx = (x + 0.5f) / 4 - 0.5f;
    int x0 = clamp((int)floor(gx), 0, w - 1), x1 = min(x0 + 1, w - 1);
    float fx = clamp(gx - x0, 0.f, 1.f);
#define BIL(im) ((1 - fy) * ((1 - fx) * im[y0 * w + x0] + fx * im[y0 * w + x1]) + fy * ((1 - fx) * im[y1 * w + x0] + fx * im[y1 * w + x1]))
    __global float* p = rgb + ((size_t)y * W + x) * 3;
    float L = luma709(p[0], p[1], p[2]);
    float d = max(L, 0.f) + CHROMA_FLOOR;
    float r = L + d * BIL(cr), b = L + d * BIL(cb);
#undef BIL
    float g = (L - 0.2126f * r - 0.0722f * b) / 0.7152f;
    p[0] = max(0.f, r); p[1] = max(0.f, g); p[2] = max(0.f, b);
}

__kernel void clamp_nonneg(__global float* rgb, int count) {
    int i = get_global_id(0);
    if (i < count) rgb[i] = max(0.f, rgb[i]);
}

// separable Lanczos-3: 6 taps per output column / row, weights precomputed by the host
__kernel void lanczos_h(__global const float* in, int w, int h, __global float* out, int W, __global const int* x0,
                        __global const float* wt) {
    int X = get_global_id(0), y = get_global_id(1);
    if (X >= W || y >= h) return;
    __global const float* s = in + (size_t)y * w * 3;
    float r = 0, g = 0, b = 0;
    for (int i = 0; i < 6; i++) {
        __global const float* p = s + 3 * clamp(x0[X] + i, 0, w - 1);
        float k = wt[X * 6 + i];
        r += k * p[0]; g += k * p[1]; b += k * p[2];
    }
    __global float* o = out + ((size_t)y * W + X) * 3;
    o[0] = r; o[1] = g; o[2] = b;
}
__kernel void lanczos_v(__global const float* mid, int W, int h, __global float* out, int H, __global const int* y0,
                        __global const float* wt) {
    int x = get_global_id(0), Y = get_global_id(1);
    if (x >= 3 * W || Y >= H) return;
    float v = 0;
    for (int i = 0; i < 6; i++) v += wt[Y * 6 + i] * mid[(size_t)clamp(y0[Y] + i, 0, h - 1) * W * 3 + x];
    out[(size_t)Y * W * 3 + x] = max(0.f, v);
}

__kernel void crop_rgb(__global const float* in, int inW, int x0, int y0, __global float* out, int W, int H) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    __global const float* s = in + ((size_t)(y0 + y) * inW + (x0 + x)) * 3;
    __global float* d = out + ((size_t)y * W + x) * 3;
    d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
}

__kernel void hdr_luma_blocks(__global const float* rgb, int W, int H, int B, __global float* out, int bw, int bh) {
    int bx = get_global_id(0), by = get_global_id(1);
    if (bx >= bw || by >= bh) return;
    float s = 0;
    int n = 0;
    for (int y = by * B; y < min(H, (by + 1) * B); y++)
        for (int x = bx * B; x < min(W, (bx + 1) * B); x++, n++) {
            __global const float* q = rgb + ((size_t)y * W + x) * 3;
            s += 0.2126f * q[0] + 0.7152f * q[1] + 0.0722f * q[2];
        }
    out[by * bw + bx] = n ? s / n : 0.f;
}

// ---- local tone mapping (synthetic exposure fusion)

__kernel void luma_of(__global const float* rgb, int n, float floorv, __global float* Y) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global const float* q = rgb + (size_t)i * 3;
    Y[i] = max(floorv, luma709(q[0], q[1], q[2]));
}

__kernel void tm_exposures(__global const float* Y, int n, __global const float* oetf, float gS, float gL,
                           __global float* eS, __global float* eL, __global float* wS) {
    int i = get_global_id(0);
    if (i >= n) return;
    float s = srgb_oetf(oetf, Y[i] * gS), l = srgb_oetf(oetf, Y[i] * gL);
    float clipFade = clamp((1.f - l) / 0.15f, 0.f, 1.f);
    float ws = exp(-(s - 0.5f) * (s - 0.5f) / (2 * 0.2f * 0.2f)) + 1e-4f;
    float wl = exp(-(l - 0.5f) * (l - 0.5f) / (2 * 0.2f * 0.2f)) * clipFade + 1e-4f;
    eS[i] = s; eL[i] = l; wS[i] = ws / (ws + wl);
}

// 5-tap binomial pyramid (Burt & Adelson), clamped borders
__kernel void pyr_down_h(__global const float* in, int iw, int ih, __global float* t, int w) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= ih) return;
    __global const float* r = in + (size_t)y * iw;
    int X = 2 * x;
#define A(i) r[clamp(i, 0, iw - 1)]
    t[(size_t)y * w + x] = (A(X - 2) + 4 * A(X - 1) + 6 * A(X) + 4 * A(X + 1) + A(X + 2)) * (1.f / 16);
#undef A
}
__kernel void pyr_down_v(__global const float* t, int w, int ih, __global float* o, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int Y = 2 * y;
#define A(i) t[(size_t)clamp(i, 0, ih - 1) * w + x]
    o[(size_t)y * w + x] = (A(Y - 2) + 4 * A(Y - 1) + 6 * A(Y) + 4 * A(Y + 1) + A(Y + 2)) * (1.f / 16);
#undef A
}
__kernel void pyr_up_h(__global const float* in, int iw, int ih, __global float* t, int w) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= ih) return;
    __global const float* r = in + (size_t)y * iw;
    int i = x >> 1;
#define A(k) r[clamp(k, 0, iw - 1)]
    t[(size_t)y * w + x] = (x & 1) ? 0.5f * (A(i) + A(i + 1)) : (A(i - 1) + 6 * A(i) + A(i + 1)) * 0.125f;
#undef A
}
// a -= s * pyr_up(...) (vertical half of the upsampling fused with the update)
__kernel void pyr_up_v_sub(__global const float* t, int w, int ih, __global float* a, int h, float s) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int j = y >> 1;
#define A(k) t[(size_t)clamp(k, 0, ih - 1) * w + x]
    float u = (y & 1) ? 0.5f * (A(j) + A(j + 1)) : (A(j - 1) + 6 * A(j) + A(j + 1)) * 0.125f;
#undef A
    a[(size_t)y * w + x] -= s * u;
}

__kernel void tm_blend(__global float* a, __global const float* b, __global const float* w, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    a[i] = w[i] * a[i] + (1 - w[i]) * b[i];
}

// low-resolution tone mapping (finish.cpp local_tonemap): key samples, binned luma, guided-filter gain
__kernel void tm_key_samples(__global const float* rgb, int W, int H, __global float* out, int sw, int sh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= sw || y >= sh) return;
    __global const float* q = rgb + ((size_t)(y * 8) * W + x * 8) * 3;
    out[y * sw + x] = max(1e-6f, luma709(q[0], q[1], q[2]));
}

__kernel void tm_down(__global const float* rgb, int W, int H, __global float* Ys, int sw, int sh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= sw || y >= sh) return;
    float acc = 0;
    int cnt = 0;
    for (int v = 0; v < 4; v++)
        for (int u = 0; u < 4; u++) {
            int X = x * 4 + u, Y = y * 4 + v;
            if (X < W && Y < H) {
                __global const float* q = rgb + ((size_t)Y * W + X) * 3;
                acc += max(1e-6f, luma709(q[0], q[1], q[2]));
                cnt++;
            }
        }
    Ys[y * sw + x] = acc / cnt;
}

__kernel void tm_guide(__global const float* Ys, __global const float* fused, __global const float* eotf, __global float* G,
                       __global float* Lg, __global float* GG, __global float* GL, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float g = log2(Ys[i]), l = log2(max(1e-6f, lut_interp(eotf, fused[i]))) - g;
    G[i] = g; Lg[i] = l; GG[i] = g * g; GL[i] = g * l;
}

__kernel void tm_coef(__global const float* mG, __global const float* mL, __global const float* mGG, __global const float* mGL,
                      __global float* A, __global float* B, int n, float eps) {
    int i = get_global_id(0);
    if (i >= n) return;
    float mg = mG[i], ml = mL[i];
    float a = (mGL[i] - mg * ml) / (max(0.f, mGG[i] - mg * mg) + eps);
    A[i] = a;
    B[i] = ml - a * mg;
}

inline float up4(__global const float* im, int sw, int sh, int x, int y) {
    float fx = clamp((x + 0.5f) / 4 - 0.5f, 0.f, sw - 1.001f), fy = clamp((y + 0.5f) / 4 - 0.5f, 0.f, sh - 1.001f);
    int x0 = (int)fx, y0 = (int)fy;
    float ax = fx - x0, ay = fy - y0;
    int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
    return (1 - ay) * ((1 - ax) * im[y0 * sw + x0] + ax * im[y0 * sw + x1]) + ay * ((1 - ax) * im[y1 * sw + x0] + ax * im[y1 * sw + x1]);
}

__kernel void tm_apply_guided(__global float* rgb, int W, int H, __global const float* mA, __global const float* mB, int sw,
                              int sh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    __global float* q = rgb + ((size_t)y * W + x) * 3;
    float ratio = exp2(up4(mA, sw, sh, x, y) * log2(max(1e-6f, luma709(q[0], q[1], q[2]))) + up4(mB, sw, sh, x, y));
    q[0] *= ratio; q[1] *= ratio; q[2] *= ratio;
}

__kernel void tm_apply(__global float* rgb, int n, __global const float* acc, __global const float* Y,
                       __global const float* eotf) {
    int i = get_global_id(0);
    if (i >= n) return;
    float ratio = lut_interp(eotf, acc[i]) / Y[i];
    __global float* q = rgb + (size_t)i * 3;
    q[0] *= ratio; q[1] *= ratio; q[2] *= ratio;
}

__kernel void tm_whitepoint(__global float* rgb, int n, float lo, float invSpan, float lift1) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global float* q = rgb + (size_t)i * 3;
    float L = luma709(q[0], q[1], q[2]);
    float t = clamp((L - lo) * invSpan, 0.f, 1.f);
    float f = 1.f + lift1 * t * t * (3.f - 2.f * t);
    q[0] *= f; q[1] *= f; q[2] *= f;
}

inline float smooth01_cl(float t) { t = clamp(t, 0.f, 1.f); return t * t * (3.f - 2.f * t); }

inline float srgb_oetf_exact_cl(float x) {
    x = clamp(x, 0.f, 1.f);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

inline float srgb_eotf_unclamped_cl(float y) {
    return y <= 0.04045f ? y / 12.92f : pow((y + 0.055f) / 1.055f, 2.4f);
}

inline float sgn_cbrt_cl(float x) {
    float a = cbrt(fabs(x));
    return x < 0.f ? -a : a;
}

inline float3 linear_srgb_to_oklab_cl(float r, float g, float b) {
    float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;
    float l_ = sgn_cbrt_cl(l), m_ = sgn_cbrt_cl(m), s_ = sgn_cbrt_cl(s);
    return (float3)(
        0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
        1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
        0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_
    );
}

inline float3 oklab_to_linear_srgb_cl(float L, float a, float ob) {
    float l_ = L + 0.3963377774f * a + 0.2158037573f * ob;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * ob;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * ob;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return (float3)(
        +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
        -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
        -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s
    );
}

inline float oklab_max_saturation_cl(float a, float b) {
    float c1 = -1.88170328f * a - 0.80936493f * b;
    float c2 = +1.81444104f * a - 1.19445276f * b;
    float c3 = +0.13110758f * a + 1.81333944f * b;
    float k0, k1, k2, k3, k4, wl, wm, ws;
    if (c1 >= c2 && c1 >= c3) {
        k0 = +1.19086277f; k1 = +1.76576728f; k2 = +0.59662641f; k3 = +0.75515197f; k4 = +0.56771245f;
        wl = +4.0767416621f; wm = -3.3077115913f; ws = +0.2309699292f;
    } else if (c2 >= c3) {
        k0 = +0.73956515f; k1 = -0.45954404f; k2 = +0.08285427f; k3 = +0.12541070f; k4 = +0.14503204f;
        wl = -1.2684380046f; wm = +2.6097574011f; ws = -0.3413193965f;
    } else {
        k0 = +1.35733652f; k1 = -0.00915799f; k2 = -1.15130210f; k3 = -0.50559606f; k4 = +0.00692167f;
        wl = -0.0041960863f; wm = -0.7034186147f; ws = +1.7076147010f;
    }
    float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
    float kl = +0.3963377774f * a + 0.2158037573f * b;
    float km = -0.1055613458f * a - 0.0638541728f * b;
    float ks = -0.0894841775f * a - 1.2914855480f * b;
    for (int it = 0; it < 2; it++) {
        float l_ = 1.f + S * kl, m_ = 1.f + S * km, s_ = 1.f + S * ks;
        float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
        float l1 = 3.f * kl * l_ * l_, m1 = 3.f * km * m_ * m_, s1 = 3.f * ks * s_ * s_;
        float l2 = 6.f * kl * kl * l_, m2 = 6.f * km * km * m_, s2 = 6.f * ks * ks * s_;
        float f = wl * l + wm * m + ws * s;
        float f1 = wl * l1 + wm * m1 + ws * s1;
        float f2 = wl * l2 + wm * m2 + ws * s2;
        float den = f1 * f1 - 0.5f * f * f2;
        if (fabs(den) > 1e-12f) S -= f * f1 / den;
    }
    return S;
}

inline float oklab_max_chroma_cl(float L, float a_, float b_) {
    if (L <= 0.f || L >= 1.f) return 0.f;
    float Smax = oklab_max_saturation_cl(a_, b_);
    float kl = +0.3963377774f * a_ + 0.2158037573f * b_;
    float km = -0.1055613458f * a_ - 0.0638541728f * b_;
    float ks = -0.0894841775f * a_ - 1.2914855480f * b_;
    float l_ = 1.f + Smax * kl, m_ = 1.f + Smax * km, s_ = 1.f + Smax * ks;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    float r0 = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    float g0 = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    float b0 = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
    float Lcusp = cbrt(1.f / max(1e-6f, max(r0, max(g0, b0))));
    float Ccusp = Lcusp * Smax;
    if (L <= Lcusp) return max(0.f, L * Smax);
    float t = Ccusp * (1.f - L) / max(1.f - Lcusp, 1e-6f);
    for (int it = 0; it < 2; it++) {
        l_ = L + t * kl; m_ = L + t * km; s_ = L + t * ks;
        l = l_ * l_ * l_; m = m_ * m_ * m_; s = s_ * s_ * s_;
        float ldt = 3.f * kl * l_ * l_, mdt = 3.f * km * m_ * m_, sdt = 3.f * ks * s_ * s_;
        float ldt2 = 6.f * kl * kl * l_, mdt2 = 6.f * km * km * m_, sdt2 = 6.f * ks * ks * s_;
        float rr = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
        float gg = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
        float bb = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
        float f, f1, f2;
        if (rr >= gg && rr >= bb) {
            f = rr - 1.f;
            f1 = +4.0767416621f * ldt - 3.3077115913f * mdt + 0.2309699292f * sdt;
            f2 = +4.0767416621f * ldt2 - 3.3077115913f * mdt2 + 0.2309699292f * sdt2;
        } else if (gg >= bb) {
            f = gg - 1.f;
            f1 = -1.2684380046f * ldt + 2.6097574011f * mdt - 0.3413193965f * sdt;
            f2 = -1.2684380046f * ldt2 + 2.6097574011f * mdt2 - 0.3413193965f * sdt2;
        } else {
            f = bb - 1.f;
            f1 = -0.0041960863f * ldt - 0.7034186147f * mdt + 1.7076147010f * sdt;
            f2 = -0.0041960863f * ldt2 - 0.7034186147f * mdt2 + 1.7076147010f * sdt2;
        }
        float den = f1 * f1 - 0.5f * f * f2;
        if (fabs(den) > 1e-12f) t -= f * f1 / den;
    }
    return max(0.f, t);
}

inline float oklab_compress_chroma_cl(float C, float Cmax) {
    float knee = 0.85f * Cmax;
    if (C <= knee) return C;
    float cap = 0.996f * Cmax;
    float span = max(cap - knee, 1e-6f);
    float ex = (C - knee) / span;
    return knee + span * (ex / (1.f + ex));
}

inline float3 oklab_gamut_compress_cl(float r, float g, float b) {
    float mn = min(r, min(g, b)), mx = max(r, max(g, b));
    if (mn >= 0.f && mx <= 1.f && mn >= 0.25f * mx && (1.f - mx) >= 0.25f * (1.f - mn)) return (float3)(r, g, b);
    float3 lab = linear_srgb_to_oklab_cl(r, g, b);
    if (lab.x <= 0.f) return (float3)(0.f, 0.f, 0.f);
    if (lab.x >= 1.f) return (float3)(1.f, 1.f, 1.f);
    float C = sqrt(lab.y * lab.y + lab.z * lab.z);
    if (C <= 1e-6f) {
        float v = clamp(lab.x * lab.x * lab.x, 0.f, 1.f);
        return (float3)(v, v, v);
    }
    float a_ = lab.y / C, b_ = lab.z / C;
    float Cmax = oklab_max_chroma_cl(lab.x, a_, b_);
    float Cnew = oklab_compress_chroma_cl(C, Cmax);
    float3 out = (Cnew != C) ? oklab_to_linear_srgb_cl(lab.x, Cnew * a_, Cnew * b_) : (float3)(r, g, b);
    return clamp(out, 0.f, 1.f);
}

__kernel void pixel_look(__global float* rgb, int n, __global const float* oetf, float contrast, float saturation) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global float* q = rgb + (size_t)i * 3;
    float r = q[0], g = q[1], b = q[2];
    float mx0 = max(r, max(g, b));
    const float kKnee = 0.7f;
    if (mx0 > kKnee) {
        float mn = min(r, min(g, b));
        float t = (mx0 - kKnee) / (1.f - kKnee);
        float shoulder = kKnee + (1.f - kKnee) * t / (1.f + t);
        float chroma = clamp(2.f * (mx0 - max(0.f, mn)) / mx0, 0.f, 1.f);
        float sc = (mx0 + (shoulder - mx0) * chroma) / mx0;
        float bleach = smooth01_cl((mx0 - 0.90f) / (3.2f - 0.90f));
        float mx = mx0 * sc;
        r = r * sc + bleach * (mx - r * sc);
        g = g * sc + bleach * (mx - g * sc);
        b = b * sc + bleach * (mx - b * sc);
    }
    float3 comp = oklab_gamut_compress_cl(r, g, b);
    float rd = srgb_oetf(oetf, comp.x);
    float gd = srgb_oetf(oetf, comp.y);
    float bd = srgb_oetf(oetf, comp.z);
    float Ld = clamp(luma709(rd, gd, bd), 0.f, 1.f);
    float s = Ld * Ld * (3.f - 2.f * Ld);
    float Lv = Ld + contrast * (s - Ld);
    float cg = Ld > 1e-6f ? Lv / Ld : 1.f;
    rd *= cg; gd *= cg; bd *= cg;
    float3 lab = linear_srgb_to_oklab_cl(srgb_eotf_unclamped_cl(rd), srgb_eotf_unclamped_cl(gd), srgb_eotf_unclamped_cl(bd));
    if (lab.x <= 0.f) {
        q[0] = 0.f; q[1] = 0.f; q[2] = 0.f;
    } else if (lab.x >= 1.f) {
        q[0] = 1.f; q[1] = 1.f; q[2] = 1.f;
    } else {
        float C = sqrt(lab.y * lab.y + lab.z * lab.z);
        if (C > 1e-6f) {
            float a_ = lab.y / C, b_ = lab.z / C;
            float Cmax = oklab_max_chroma_cl(lab.x, a_, b_);
            float sat = clamp(C / max(Cmax, 1e-5f), 0.f, 1.f);
            float vibW = 1.f - smooth01_cl((Lv - 0.55f) / (0.80f - 0.55f));
            float hiRoll = 1.f - 0.55f * smooth01_cl((Lv - 0.72f) / (1.0f - 0.72f));
            float k = (1.f + (saturation - 1.f) * (1.f - sat) * vibW) * hiRoll;
            float Cnew = oklab_compress_chroma_cl(C * k, Cmax);
            float3 lin = oklab_to_linear_srgb_cl(lab.x, Cnew * a_, Cnew * b_);
            q[0] = srgb_oetf(oetf, clamp(lin.x, 0.f, 1.f));
            q[1] = srgb_oetf(oetf, clamp(lin.y, 0.f, 1.f));
            q[2] = srgb_oetf(oetf, clamp(lin.z, 0.f, 1.f));
        } else {
            q[0] = clamp(rd, 0.f, 1.f);
            q[1] = clamp(gd, 0.f, 1.f);
            q[2] = clamp(bd, 0.f, 1.f);
        }
    }
}

// ---- local contrast (fast guided filter on display luma)

__kernel void lc_down(__global const float* L, int W, int H, __global float* Ls, __global float* LL, int sw, int sh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= sw || y >= sh) return;
    float acc = 0;
    int cnt = 0;
    for (int v = 0; v < 4; v++)
        for (int u = 0; u < 4; u++) {
            int X = x * 4 + u, Y = y * 4 + v;
            if (X < W && Y < H) { acc += L[(size_t)Y * W + X]; cnt++; }
        }
    float m = acc / cnt;
    Ls[y * sw + x] = m;
    LL[y * sw + x] = m * m;
}

__kernel void box_h(__global const float* in, __global float* out, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    __global const float* s = in + (size_t)y * w;
    float a = 0;
    for (int i = -r; i <= r; i++) a += s[clamp(x + i, 0, w - 1)];
    out[(size_t)y * w + x] = a * (1.f / (2 * r + 1));
}
__kernel void box_v(__global const float* in, __global float* out, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float a = 0;
    for (int i = -r; i <= r; i++) a += in[(size_t)clamp(y + i, 0, h - 1) * w + x];
    out[(size_t)y * w + x] = a * (1.f / (2 * r + 1));
}

__kernel void lc_coef(__global const float* mI, __global const float* mII, __global float* A, __global float* B, int n,
                      float eps) {
    int i = get_global_id(0);
    if (i >= n) return;
    float m = mI[i], v = max(0.f, mII[i] - m * m);
    float a = v / (v + eps);
    A[i] = a;
    B[i] = m - a * m;
}


__kernel void lc_apply(__global float* rgb, int W, int H, __global const float* L, __global const float* mA,
                       __global const float* mB, int sw, int sh, float amount) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float l = L[(size_t)y * W + x], base = up4(mA, sw, sh, x, y) * l + up4(mB, sw, sh, x, y);
    float fade = clamp(l / 0.08f, 0.f, 1.f) * clamp((1.f - l) / 0.1f, 0.f, 1.f);
    float delta = amount * fade * (l - base);
    __global float* q = rgb + ((size_t)y * W + x) * 3;
    for (int c = 0; c < 3; c++) q[c] = clamp(q[c] + delta, 0.f, 1.f);
}

inline float hue_weight(float4 win, float ux, float uy) {
    float t = clamp((ux * win.x + uy * win.y - win.z) / (1.f - win.z), 0.f, 1.f);
    return t * t * (3 - 2 * t);
}

// sky, skin = (cos, sin, cos(half_width), 0) in Oklab; sat4 = (skySat, skinSat, skyRot, skinRot)
__kernel void hue_tune(__global float* rgb, int n, float4 sky, float4 skin, float4 sat4) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global float* q = rgb + (size_t)i * 3;
    float rl = srgb_eotf_unclamped_cl(clamp(q[0], 0.f, 1.f));
    float gl = srgb_eotf_unclamped_cl(clamp(q[1], 0.f, 1.f));
    float bl = srgb_eotf_unclamped_cl(clamp(q[2], 0.f, 1.f));
    float3 lab = linear_srgb_to_oklab_cl(rl, gl, bl);
    if (lab.x <= 0.f || lab.x >= 1.f) return;
    float c2 = lab.y * lab.y + lab.z * lab.z;
    if (c2 < 1e-10f) return;
    float C = sqrt(c2), inv = 1.f / C;
    float ux = lab.y * inv, uy = lab.z * inv;
    float cw = clamp(C / 0.025f, 0.f, 1.f);
    float ws = hue_weight(sky, ux, uy) * cw, wk = hue_weight(skin, ux, uy) * cw;
    if (ws + wk <= 0.f) return;
    float sat = 1.f + ws * (sat4.x - 1.f) + wk * (sat4.y - 1.f);
    float th = ws * sat4.z + wk * sat4.w;
    float cs = cos(th), sn = sin(th);
    float nux = ux * cs - uy * sn, nuy = ux * sn + uy * cs;
    float Cmax = oklab_max_chroma_cl(lab.x, nux, nuy);
    float Cnew = oklab_compress_chroma_cl(C * sat, Cmax);
    float3 lin = oklab_to_linear_srgb_cl(lab.x, Cnew * nux, Cnew * nuy);
    q[0] = srgb_oetf_exact_cl(lin.x);
    q[1] = srgb_oetf_exact_cl(lin.y);
    q[2] = srgb_oetf_exact_cl(lin.z);
}

__kernel void sharpen_apply(__global float* rgb, int n, __global const float* L, __global const float* B, float amount,
                            float core) {
    int i = get_global_id(0);
    if (i >= n) return;
    float d = L[i] - B[i];
    d = d > core ? d - core : (d < -core ? d + core : 0.f);
    float delta = amount * d;
    __global float* q = rgb + (size_t)i * 3;
    for (int c = 0; c < 3; c++) q[c] = clamp(q[c] + delta, 0.f, 1.f);
}

// to_rgba8888(): the per-row LCG of the CPU code, jumped ahead to this pixel's first draw
inline uint lcg_jump(uint seed, uint k) {
    uint A = 1, C = 0, a = 1664525u, c = 1013904223u;
    while (k) {
        if (k & 1) { A = A * a; C = C * a + c; }
        c = c * (a + 1);
        a = a * a;
        k >>= 1;
    }
    return A * seed + C;
}

inline void write_dithered(__global const float* p, __global uchar* o, int x, int y) {
    uint seed = lcg_jump(0x9E3779B9u * (uint)(y + 1), (uint)(3 * x));
    for (int c = 0; c < 3; c++) {
        seed = seed * 1664525u + 1013904223u;
        float dither = ((seed >> 8) & 0xFFFF) / 65535.f - 0.5f;
        o[c] = (uchar)clamp((int)round(p[c] * 255.f + dither), 0, 255);
    }
}

__kernel void to_rgba(__global const float* rgb, int W, int H, __global uchar* out) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    __global const float* p = rgb + ((size_t)y * W + x) * 3;
    __global uchar* o = out + ((size_t)y * W + x) * 4;
    write_dithered(p, o, x, y);
    o[3] = 255;
}

__kernel void to_rgb8(__global const float* rgb, int W, int H, __global uchar* out) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    __global const float* p = rgb + ((size_t)y * W + x) * 3;
    __global uchar* o = out + ((size_t)y * W + x) * 3;
    write_dithered(p, o, x, y);
}

__kernel void render_thumb(__global const float* rgb, int inW, int inH, __global uchar* out, int outW, int outH) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= outW || y >= outH) return;
    int x0 = (int)((long)x * inW / outW), x1 = max(x0 + 1, (int)((long)(x + 1) * inW / outW));
    int y0 = (int)((long)y * inH / outH), y1 = max(y0 + 1, (int)((long)(y + 1) * inH / outH));
    float sr = 0, sg = 0, sb = 0;
    for (int sy = y0; sy < y1; sy++)
        for (int sx = x0; sx < x1; sx++) {
            __global const float* p = rgb + ((size_t)sy * inW + sx) * 3;
            sr += p[0]; sg += p[1]; sb += p[2];
        }
    float inv = 1.f / ((y1 - y0) * (x1 - x0));
    __global uchar* o = out + ((size_t)y * outW + x) * 4;
    o[0] = (uchar)clamp((int)round(sr * inv * 255.f), 0, 255);
    o[1] = (uchar)clamp((int)round(sg * inv * 255.f), 0, 255);
    o[2] = (uchar)clamp((int)round(sb * inv * 255.f), 0, 255);
    o[3] = 255;
}
)CL"
