R"CL(
// finish.cpp and hdr_luma() on the device. RGB images are interleaved float3 rows.

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
    float v = (mosaic[(size_t)y * W + x] - black) / (white - black);
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
    out[(size_t)y * W + x] = min(max(0.f, v) * gain * wb, clipv);
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
    p[0] = min(max(0.f, s[0] * inv.x) * g0 * wb.x, clipv);
    p[1] = min(max(0.f, s[1] * inv.y) * g1 * wb.y, clipv);
    p[2] = min(max(0.f, s[2] * inv.z) * g2 * wb.z, clipv);
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
    o[0] = max(0.f, o0); o[1] = max(0.f, o1); o[2] = max(0.f, o2);
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
        v = log(1e-4f + 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]);
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
            sr += p[0] - L; sb += p[2] - L; n++;
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
    float r = L + BIL(cr), b = L + BIL(cb);
#undef BIL
    float g = (L - 0.2126f * r - 0.0722f * b) / 0.7152f;
    p[0] = r; p[1] = g; p[2] = b;
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

__kernel void subsample8(__global const float* Y, int W, int H, __global float* out, int sw, int sh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= sw || y >= sh) return;
    out[y * sw + x] = Y[(size_t)(y * 8) * W + x * 8];
}

__kernel void tm_exposures(__global const float* Y, int n, __global const float* oetf, float gS, float gL,
                           __global float* eS, __global float* eL, __global float* wS) {
    int i = get_global_id(0);
    if (i >= n) return;
    float s = srgb_oetf(oetf, Y[i] * gS), l = srgb_oetf(oetf, Y[i] * gL);
    float ws = exp(-(s - 0.5f) * (s - 0.5f) / (2 * 0.2f * 0.2f)) + 1e-4f;
    float wl = exp(-(l - 0.5f) * (l - 0.5f) / (2 * 0.2f * 0.2f)) + 1e-4f;
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

__kernel void tm_apply(__global float* rgb, int n, __global const float* acc, __global const float* Y,
                       __global const float* eotf) {
    int i = get_global_id(0);
    if (i >= n) return;
    float ratio = lut_interp(eotf, acc[i]) / Y[i];
    __global float* q = rgb + (size_t)i * 3;
    q[0] *= ratio; q[1] *= ratio; q[2] *= ratio;
}

__kernel void pixel_look(__global float* rgb, int n, __global const float* oetf, float contrast, float saturation) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global float* q = rgb + (size_t)i * 3;
    float v0 = q[0], v1 = q[1], v2 = q[2];
    float mx = max(v0, max(v1, v2));
    const float kKnee = 0.7f;
    if (mx > kKnee) {
        float mn = min(v0, min(v1, v2));
        float t = (mx - kKnee) / (1 - kKnee);
        float shoulder = kKnee + (1 - kKnee) * t / (1 + t);
        float chroma = clamp(2 * (mx - max(0.f, mn)) / mx, 0.f, 1.f);
        float g = (mx + (shoulder - mx) * chroma) / mx;
        v0 *= g; v1 *= g; v2 *= g;
        mx *= g;
    }
    float L = min(1.f, max(0.f, luma709(v0, v1, v2)));
    if (mx > 1.f) {
        float k = (1.f - L) / max(1e-6f, mx - L);
        v0 = L + (v0 - L) * k; v1 = L + (v1 - L) * k; v2 = L + (v2 - L) * k;
    }
    float c3[3] = {v0, v1, v2};
    for (int c = 0; c < 3; c++) {
        float v = srgb_oetf(oetf, c3[c]);
        float s = v * v * (3 - 2 * v);
        c3[c] = v + contrast * (s - v);
    }
    float Ld = luma709(c3[0], c3[1], c3[2]);
    float sat = max(c3[0], max(c3[1], c3[2])) - min(c3[0], min(c3[1], c3[2]));
    float k = 1 + (saturation - 1) * (1 - clamp(sat, 0.f, 1.f));
    for (int c = 0; c < 3; c++) q[c] = clamp(Ld + (c3[c] - Ld) * k, 0.f, 1.f);
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

inline float up4(__global const float* im, int sw, int sh, int x, int y) {
    float fx = clamp((x + 0.5f) / 4 - 0.5f, 0.f, sw - 1.001f), fy = clamp((y + 0.5f) / 4 - 0.5f, 0.f, sh - 1.001f);
    int x0 = (int)fx, y0 = (int)fy;
    float ax = fx - x0, ay = fy - y0;
    int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
    return (1 - ay) * ((1 - ax) * im[y0 * sw + x0] + ax * im[y0 * sw + x1]) + ay * ((1 - ax) * im[y1 * sw + x0] + ax * im[y1 * sw + x1]);
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

// sat4 = (skySat, skinSat, skyRot, skinRot); windows (cx, cy, c0)
__kernel void hue_tune(__global float* rgb, int n, float4 sky, float4 skin, float4 sat4) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global float* q = rgb + (size_t)i * 3;
    float Y = luma709(q[0], q[1], q[2]);
    float cb = (q[2] - Y) / 1.8556f, cr = (q[0] - Y) / 1.5748f;
    float c2 = cb * cb + cr * cr;
    if (c2 < 1e-8f) return;
    float inv = 1.f / sqrt(c2), chroma = c2 * inv;
    float cw = clamp(chroma / 0.04f, 0.f, 1.f);
    float ws = hue_weight(sky, cb * inv, cr * inv) * cw, wk = hue_weight(skin, cb * inv, cr * inv) * cw;
    if (ws + wk <= 0.f) return;
    float sat = 1 + ws * (sat4.x - 1) + wk * (sat4.y - 1);
    float th = ws * sat4.z + wk * sat4.w;
    float ncb = sat * (cb - th * cr), ncr = sat * (cr + th * cb);
    float r = Y + 1.5748f * ncr, b = Y + 1.8556f * ncb;
    float g = (Y - 0.2126f * r - 0.0722f * b) / 0.7152f;
    q[0] = clamp(r, 0.f, 1.f); q[1] = clamp(g, 0.f, 1.f); q[2] = clamp(b, 0.f, 1.f);
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

__kernel void to_rgba(__global const float* rgb, int W, int H, __global uchar* out) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    uint seed = lcg_jump(0x9E3779B9u * (uint)(y + 1), (uint)(3 * x));
    __global const float* p = rgb + ((size_t)y * W + x) * 3;
    __global uchar* o = out + ((size_t)y * W + x) * 4;
    for (int c = 0; c < 3; c++) {
        seed = seed * 1664525u + 1013904223u;
        float dither = ((seed >> 8) & 0xFFFF) / 65535.f - 0.5f;
        o[c] = (uchar)clamp((int)round(p[c] * 255.f + dither), 0, 255);
    }
    o[3] = 255;
}
)CL"
