R"CL(
// Indigo + Original Pixel 1 HDR+ finishing on Adreno: coalesced half4 RGB, hardware image2d_t TP bilinear upsampling,
// perceptual Oklab constant-hue gamut compression, and fused tonemap+look + local-tile sharpen+dithered RGB8.

#define NEG_FLOOR 0.0625f
#define CHROMA_FLOOR 4e-3f

// Local-tile hot-pixel despike + constant-memory float4 LSC bilinear + WB normalisation on 16x8 Wave128 workgroup.
__kernel void normalize_mosaic(__global const float* mosaic, __global float* out, int W, int H, int cfa, float4 black4,
                               float white, float4 wb4, __constant float4* lsc4, int rows, int cols, int cropX,
                               int cropY, float sw, float sh, float lumaStrength) {
    __local float mtile[12][20];
    int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    int gx0 = (int)get_group_id(0) * 16 - 2, gy0 = (int)get_group_id(1) * 8 - 2;
    for (int idx = tid; idx < 240; idx += 128) {
        int ty = idx / 20, tx = idx - ty * 20;
        mtile[ty][tx] = mosaic[(size_t)clamp(gy0 + ty, 0, H - 1) * W + clamp(gx0 + tx, 0, W - 1)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = (int)get_group_id(0) * 16 + lx, y = (int)get_group_id(1) * 8 + ly;
    if (x >= W || y >= H) return;
    int ch = cfa_channel(cfa, x, y);
    float black = ch == 0 ? black4.x : ch == 1 ? black4.y : ch == 2 ? black4.z : black4.w;
    float wb = ch == 0 ? wb4.x : ch == 1 ? wb4.y : ch == 2 ? wb4.z : wb4.w;
    float raw = mtile[ly + 2][lx + 2];
    if (x >= 2 && x < W - 2 && y >= 2 && y < H - 2) {
        float n0 = mtile[ly][lx],     n1 = mtile[ly][lx + 2],     n2 = mtile[ly][lx + 4];
        float n3 = mtile[ly + 2][lx],                             n4 = mtile[ly + 2][lx + 4];
        float n5 = mtile[ly + 4][lx], n6 = mtile[ly + 4][lx + 2], n7 = mtile[ly + 4][lx + 4];
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
        int base = y0 * cols + x0;
        float4 g0 = mix(lsc4[base], lsc4[base + 1], fx);
        float4 g1 = mix(lsc4[base + cols], lsc4[base + cols + 1], fx);
        float4 g4 = mix(g0, g1, fy);
        float gG = 0.5f * (g4.y + g4.z);
        float lumaGain = native_exp2(lumaStrength * native_log2(max(gG, 1e-6f)));
        float gCh = ch == 0 ? g4.x : ch == 1 ? g4.y : ch == 2 ? g4.z : g4.w;
        gain = gCh / gG * lumaGain;
        clipv = lumaGain;
    }
    out[(size_t)y * W + x] = clamp(v * gain * wb, -NEG_FLOOR, clipv);
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
    if (get_local_id(0) == 0 && y < h) vstore2(sn, y * get_num_groups(0) + get_group_id(0), out);
}

inline float sample_half4_ch(__global const half* im, int W, int H, float fx, float fy, int c) {
    fx = clamp(fx, 0.f, (float)(W - 1));
    fy = clamp(fy, 0.f, (float)(H - 1));
    int x0 = min((int)fx, max(0, W - 2)), y0 = min((int)fy, max(0, H - 2));
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float ax = fx - x0, ay = fy - y0;
    float v00 = vload_half(((size_t)y0 * W + x0) * 4 + c, im);
    float v10 = vload_half(((size_t)y0 * W + x1) * 4 + c, im);
    float v01 = vload_half(((size_t)y1 * W + x0) * 4 + c, im);
    float v11 = vload_half(((size_t)y1 * W + x1) * 4 + c, im);
    return (1.f - ay) * ((1.f - ax) * v00 + ax * v10) + ay * ((1.f - ax) * v01 + ax * v11);
}

// Lateral CA warp + CCM fused for ultrawide lens
__kernel void lca_ccm_wb(__global const half* in, int W, int H, float4 lcaK, float lcx, float lcy, float invR2,
                         float4 m0, float4 m1, float4 m2, __global half* out) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float px = (x + 0.5f) - lcx, py = (y + 0.5f) - lcy;
    float rr = (px * px + py * py) * invR2;
    float kr = lcaK.x + lcaK.y * rr, kb = lcaK.z + lcaK.w * rr;
    size_t i = (size_t)y * W + x;
    float r = sample_half4_ch(in, W, H, x + px * kr, y + py * kr, 0);
    float g = vload_half(i * 4 + 1, in);
    float b = sample_half4_ch(in, W, H, x + px * kb, y + py * kb, 2);
    float nr = m0.x * r + m0.y * g + m0.z * b;
    float ng = m1.x * r + m1.y * g + m1.z * b;
    float nb = m2.x * r + m2.y * g + m2.z * b;
    vstore_half4((float4)(nr, ng, nb, 0.f), i, out);
}

__kernel void ccm_half4(__global half* rgb, int n, float4 m0, float4 m1, float4 m2) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 p = vload_half4(i, rgb);
    float nr = m0.x * p.x + m0.y * p.y + m0.z * p.z;
    float ng = m1.x * p.x + m1.y * p.y + m1.z * p.z;
    float nb = m2.x * p.x + m2.y * p.y + m2.z * p.z;
    vstore_half4((float4)(nr, ng, nb, 0.f), i, rgb);
}

__kernel void float3_to_half4(__global const float* in, __global half* out, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global const float* p = in + (size_t)i * 3;
    vstore_half4((float4)(p[0], p[1], p[2], 0.f), i, out);
}

// sum of log luma over every 8th pixel of every 8th row
__kernel void log_luma_rows(__global const half* rgb, int W, int H, __global float* out, __local float2* red) {
    int c = get_global_id(0), r = get_global_id(1);
    int gr = (H + 7) / 8;
    float v = 0;
    if (c * 8 < W && r < gr) {
        float4 p = vload_half4((size_t)(r * 8) * W + c * 8, rgb);
        v = log(1e-4f + max(0.f, luma709(p.x, p.y, p.z)));
    }
    float s = group_sum2(red, (float2)(v, 0)).x;
    if (get_local_id(0) == 0 && r < gr) out[r * get_num_groups(0) + get_group_id(0)] = s;
}

__kernel void chroma_down(__global const half* rgb, int W, int H, __global float2* cr_cb, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float2 s = 0.f;
    int n = 0;
    for (int v = 0; v < 4; v++)
        for (int u = 0; u < 4; u++) {
            int X = x * 4 + u, Y = y * 4 + v;
            if (X >= W || Y >= H) continue;
            float4 p = vload_half4((size_t)Y * W + X, rgb);
            float L = luma709(p.x, p.y, p.z);
            float inv = 1.f / (max(L, 0.f) + CHROMA_FLOOR);
            s += (float2)((p.x - L) * inv, (p.z - L) * inv);
            n++;
        }
    cr_cb[y * w + x] = s * (1.f / n);
}

__kernel void blur2_h(__global const float2* in, __global float2* out, int w, int h, __constant float* k, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    __global const float2* s = in + (size_t)y * w;
    float2 a = 0.f;
    for (int i = -r; i <= r; i++) a += k[i + r] * s[clamp(x + i, 0, w - 1)];
    out[(size_t)y * w + x] = a;
}

__kernel void blur2_v_img(__global const float2* in, __write_only image2d_t outImg, int w, int h, __constant float* k, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float2 a = 0.f;
    for (int i = -r; i <= r; i++) a += k[i + r] * in[(size_t)clamp(y + i, 0, h - 1) * w + x];
    write_imagef(outImg, (int2)(x, y), (float4)(a.x, a.y, 0.f, 0.f));
}

// Reconstructs chroma-denoised RGB via hardware TP bilinear sampling, and when writeTone != 0 fuses tm_key_samples + tm_down!
__kernel void chroma_apply_img(__global half* rgb, int W, int H, __read_only image2d_t cr_cb_img, int writeTone,
                               __global float* keySub, int s8w, __global float* Ys, int sw, int sh) {
    __local float ltile[8][16];
    int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    int x = (int)get_group_id(0) * 16 + lx, y = (int)get_group_id(1) * 8 + ly;
    float yval = 0.f;
    if (x < W && y < H) {
        size_t i = (size_t)y * W + x;
        float4 p = vload_half4(i, rgb);
        float L = luma709(p.x, p.y, p.z);
        float d = max(L, 0.f) + CHROMA_FLOOR;
        float2 cbcr = read_imagef(cr_cb_img, kLinearClamp, (float2)((x + 0.5f) * 0.25f, (y + 0.5f) * 0.25f)).xy;
        float r = max(0.f, L + d * cbcr.x);
        float b = max(0.f, L + d * cbcr.y);
        float g = max(0.f, (L - 0.2126f * r - 0.0722f * b) * (1.f / 0.7152f));
        vstore_half4((float4)(r, g, b, 0.f), i, rgb);
        if (writeTone) {
            yval = max(1e-6f, luma709(r, g, b));
            if (((x | y) & 7) == 0) keySub[(y >> 3) * s8w + (x >> 3)] = yval;
        }
    }
    if (!writeTone) return;
    ltile[ly][lx] = yval;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8) {
        int bx = tid & 3, by = tid >> 2;
        int sx = (int)get_group_id(0) * 4 + bx, sy = (int)get_group_id(1) * 2 + by;
        if (sx < sw && sy < sh) {
            float acc = 0.f;
            int cnt = 0;
            for (int v = 0; v < 4; v++)
                for (int u = 0; u < 4; u++) {
                    if (sx * 4 + u < W && sy * 4 + v < H) {
                        acc += ltile[by * 4 + v][bx * 4 + u];
                        cnt++;
                    }
                }
            Ys[sy * sw + sx] = acc / max(cnt, 1);
        }
    }
}

__kernel void clamp_nonneg_half4(__global half* rgb, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 p = vload_half4(i, rgb);
    vstore_half4(max(p, (float4)0.f), i, rgb);
}

// Fused tm_key_samples + tm_down when zoom/aspect ran after chroma_apply
__kernel void tm_prepare(__global const half* rgb, int W, int H, __global float* keySub, int s8w,
                         __global float* Ys, int sw, int sh) {
    __local float ltile[8][16];
    int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    int x = (int)get_group_id(0) * 16 + lx, y = (int)get_group_id(1) * 8 + ly;
    float yval = 0.f;
    if (x < W && y < H) {
        float4 p = vload_half4((size_t)y * W + x, rgb);
        yval = max(1e-6f, luma709(p.x, p.y, p.z));
        if (((x | y) & 7) == 0) keySub[(y >> 3) * s8w + (x >> 3)] = yval;
    }
    ltile[ly][lx] = yval;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8) {
        int bx = tid & 3, by = tid >> 2;
        int sx = (int)get_group_id(0) * 4 + bx, sy = (int)get_group_id(1) * 2 + by;
        if (sx < sw && sy < sh) {
            float acc = 0.f;
            int cnt = 0;
            for (int v = 0; v < 4; v++)
                for (int u = 0; u < 4; u++) {
                    if (sx * 4 + u < W && sy * 4 + v < H) {
                        acc += ltile[by * 4 + v][bx * 4 + u];
                        cnt++;
                    }
                }
            Ys[sy * sw + sx] = acc / max(cnt, 1);
        }
    }
}

// Separable Lanczos-3 on half4 RGB (for 2x crop zoom)
__kernel void lanczos_h_half4(__global const half* in, int w, int h, __global half* out, int W, __global const int* x0,
                              __global const float* wt) {
    int X = get_global_id(0), y = get_global_id(1);
    if (X >= W || y >= h) return;
    size_t rowOff = (size_t)y * w;
    float4 acc = 0.f;
    int b = x0[X];
    for (int i = 0; i < 6; i++) acc += wt[X * 6 + i] * vload_half4(rowOff + (size_t)clamp(b + i, 0, w - 1), in);
    vstore_half4((float4)(acc.x, acc.y, acc.z, 0.f), (size_t)y * W + X, out);
}

__kernel void lanczos_v_half4(__global const half* mid, int W, int h, __global half* out, int H, __global const int* y0,
                              __global const float* wt) {
    int X = get_global_id(0), Y = get_global_id(1);
    if (X >= W || Y >= H) return;
    float4 acc = 0.f;
    int b = y0[Y];
    for (int i = 0; i < 6; i++) acc += wt[Y * 6 + i] * vload_half4((size_t)clamp(b + i, 0, h - 1) * W + X, mid);
    vstore_half4(max(acc, (float4)0.f), (size_t)Y * W + X, out);
}

__kernel void crop_half4(__global const half* in, int inW, int x0, int y0, __global half* out, int W, int H) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float4 p = vload_half4((size_t)(y0 + y) * inW + (x0 + x), in);
    vstore_half4(p, (size_t)y * W + x, out);
}

__kernel void hdr_luma_blocks_half4(__global const half* rgb, int W, int H, int B, __global float* out, int bw, int bh) {
    int bx = get_global_id(0), by = get_global_id(1);
    if (bx >= bw || by >= bh) return;
    float s = 0.f;
    int n = 0;
    for (int y = by * B; y < min(H, (by + 1) * B); y++)
        for (int x = bx * B; x < min(W, (bx + 1) * B); x++, n++) {
            float4 q = vload_half4((size_t)y * W + x, rgb);
            s += luma709(q.x, q.y, q.z);
        }
    out[by * bw + bx] = n ? s / n : 0.f;
}

// ---- local tone mapping (synthetic exposure fusion + vectorised 1/4-res guided filter)

__kernel void tm_exposures(__global const float* Y, int n, float gS, float gL, __global float* eS, __global float* eL,
                           __global float* wS) {
    int i = get_global_id(0);
    if (i >= n) return;
    float s = srgb_oetf_cl(Y[i] * gS), l = srgb_oetf_cl(Y[i] * gL);
    float clipFade = clamp((1.f - l) * (1.f / 0.15f), 0.f, 1.f);
    float ws = native_exp(-(s - 0.5f) * (s - 0.5f) * (1.f / (2.f * 0.2f * 0.2f))) + 1e-4f;
    float wl = native_exp(-(l - 0.5f) * (l - 0.5f) * (1.f / (2.f * 0.2f * 0.2f))) * clipFade + 1e-4f;
    eS[i] = s; eL[i] = l; wS[i] = ws / (ws + wl);
}

__kernel void pyr_down_h(__global const float* in, int iw, int ih, __global float* t, int w) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= ih) return;
    __global const float* r = in + (size_t)y * iw;
    int X = 2 * x;
#define A(i) r[clamp(i, 0, iw - 1)]
    t[(size_t)y * w + x] = (A(X - 2) + 4 * A(X - 1) + 6 * A(X) + 4 * A(X + 1) + A(X + 2)) * (1.f / 16.f);
#undef A
}
__kernel void pyr_down_v(__global const float* t, int w, int ih, __global float* o, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int Y = 2 * y;
#define A(i) t[(size_t)clamp(i, 0, ih - 1) * w + x]
    o[(size_t)y * w + x] = (A(Y - 2) + 4 * A(Y - 1) + 6 * A(Y) + 4 * A(Y + 1) + A(Y + 2)) * (1.f / 16.f);
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
    a[i] = w[i] * a[i] + (1.f - w[i]) * b[i];
}

__kernel void tm_guide4(__global const float* Ys, __global const float* fused, __global float4* out4, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float g = log2(max(1e-6f, Ys[i])), l = log2(max(1e-6f, srgb_eotf_cl(fused[i]))) - g;
    out4[i] = (float4)(g, l, g * g, g * l);
}

__kernel void box4_h(__global const float4* in, __global float4* out, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    __global const float4* s = in + (size_t)y * w;
    float4 a = 0.f;
    for (int i = -r; i <= r; i++) a += s[clamp(x + i, 0, w - 1)];
    out[(size_t)y * w + x] = a * (1.f / (2 * r + 1));
}
__kernel void box4_v(__global const float4* in, __global float4* out, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float4 a = 0.f;
    for (int i = -r; i <= r; i++) a += in[(size_t)clamp(y + i, 0, h - 1) * w + x];
    out[(size_t)y * w + x] = a * (1.f / (2 * r + 1));
}

__kernel void tm_coef2(__global const float4* m4, __global float2* ab, int n, float eps) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 m = m4[i];
    float mg = m.x, ml = m.y;
    float a = (m.w - mg * ml) / (max(0.f, m.z - mg * mg) + eps);
    ab[i] = (float2)(a, ml - a * mg);
}

__kernel void box2_h(__global const float2* in, __global float2* out, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    __global const float2* s = in + (size_t)y * w;
    float2 a = 0.f;
    for (int i = -r; i <= r; i++) a += s[clamp(x + i, 0, w - 1)];
    out[(size_t)y * w + x] = a * (1.f / (2 * r + 1));
}
__kernel void box2_v_img(__global const float2* in, __write_only image2d_t outImg, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float2 a = 0.f;
    for (int i = -r; i <= r; i++) a += in[(size_t)clamp(y + i, 0, h - 1) * w + x];
    a *= (1.f / (2 * r + 1));
    write_imagef(outImg, (int2)(x, y), (float4)(a.x, a.y, 0.f, 0.f));
}

__kernel void tm_wp_samples(__global const float* Ys, int sw, int sh, __read_only image2d_t mAB_img,
                            __global float* wpOut, int wpW, int wpH) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= wpW || y >= wpH) return;
    int sx = 2 * x, sy = 2 * y;
    float y0 = max(1e-6f, Ys[sy * sw + sx]);
    float2 ab = read_imagef(mAB_img, kNearestClamp, (int2)(sx, sy)).xy;
    wpOut[y * wpW + x] = y0 * exp2(ab.x * log2(y0) + ab.y);
}

// ---- Perceptual Oklab constant-hue gamut compression (Bjorn Ottosson) + Indigo/Pixel look

inline float smooth01_cl(float t) { t = clamp(t, 0.f, 1.f); return t * t * (3.f - 2.f * t); }

inline float fast_cbrt_cl(float x) {
    if (x <= 1e-30f) return 0.f;
    uint i = as_uint(x) / 3u + 0x2a517d47u;
    float y = as_float(i);
    y = (2.f * y + x / (y * y)) * (1.f / 3.f);
    y = (2.f * y + x / (y * y)) * (1.f / 3.f);
    y = (2.f * y + x / (y * y)) * (1.f / 3.f);
    return y;
}

inline float sgn_cbrt_cl(float x) {
    float a = fast_cbrt_cl(fabs(x));
    return x < 0.f ? -a : a;
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
    float Lcusp = fast_cbrt_cl(1.f / max(1e-6f, max(r0, max(g0, b0))));
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

inline float3 oklab_gamut_compress_cl(float3 c) {
    float mn = min(c.x, min(c.y, c.z)), mx = max(c.x, max(c.y, c.z));
    if (mn >= 0.f && mx <= 1.f && mn >= 0.25f * mx && (1.f - mx) >= 0.25f * (1.f - mn)) return c;
    float l = 0.4122214708f * c.x + 0.5363325363f * c.y + 0.0514459929f * c.z;
    float m = 0.2119034982f * c.x + 0.6806995451f * c.y + 0.1073969566f * c.z;
    float s = 0.0883024619f * c.x + 0.2817188376f * c.y + 0.6299787005f * c.z;
    float l_ = sgn_cbrt_cl(l), m_ = sgn_cbrt_cl(m), s_ = sgn_cbrt_cl(s);
    float L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
    if (L <= 0.f) return (float3)0.f;
    if (L >= 1.f) return (float3)1.f;
    float a = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
    float ob = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
    float C = sqrt(a * a + ob * ob);
    if (C <= 1e-6f) return (float3)clamp(L * L * L, 0.f, 1.f);
    float a_ = a / C, b_ = ob / C;
    float Cmax = oklab_max_chroma_cl(L, a_, b_);
    float knee = 0.85f * Cmax;
    if (C > knee) {
        float span = max(0.996f * Cmax - knee, 1e-6f);
        float ex = (C - knee) / span;
        float Cnew = knee + span * (ex / (1.f + ex));
        float na = Cnew * a_, nb = Cnew * b_;
        float nl_ = L + 0.3963377774f * na + 0.2158037573f * nb;
        float nm_ = L - 0.1055613458f * na - 0.0638541728f * nb;
        float ns_ = L - 0.0894841775f * na - 1.2914855480f * nb;
        float nl = nl_ * nl_ * nl_, nm = nm_ * nm_ * nm_, ns = ns_ * ns_ * ns_;
        c.x = +4.0767416621f * nl - 3.3077115913f * nm + 0.2309699292f * ns;
        c.y = -1.2684380046f * nl + 2.6097574011f * nm - 0.3413193965f * ns;
        c.z = -0.0041960863f * nl - 0.7034186147f * nm + 1.7076147010f * ns;
    }
    return clamp(c, 0.f, 1.f);
}

// Fuses guided local tonemap upsampling (via hardware TP bilinear image2d_t) + whitepoint lift + Oklab pixel_look!
__kernel void tonemap_and_look(__global half* rgb, int W, int H, __read_only image2d_t mAB_img, float lo, float invSpan,
                               float lift1, float contrast, float saturation) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    float4 p = vload_half4(i, rgb);
    float3 v = p.xyz;

    float2 ab = read_imagef(mAB_img, kLinearClamp, (float2)((x + 0.5f) * 0.25f, (y + 0.5f) * 0.25f)).xy;
    float L0 = luma709(v.x, v.y, v.z);
    float ratio = exp2(ab.x * log2(max(1e-6f, L0)) + ab.y);
    if (lift1 > 0.f) {
        float tw = clamp((L0 * ratio - lo) * invSpan, 0.f, 1.f);
        ratio *= 1.f + lift1 * tw * tw * (3.f - 2.f * tw);
    }
    v *= ratio;

    float mx0 = max(v.x, max(v.y, v.z));
    const float kKnee = 0.7f;
    if (mx0 > kKnee) {
        float mn = min(v.x, min(v.y, v.z));
        float t = (mx0 - kKnee) * (1.f / (1.f - kKnee));
        float shoulder = kKnee + (1.f - kKnee) * t / (1.f + t);
        float chroma = clamp(2.f * (mx0 - max(0.f, mn)) / mx0, 0.f, 1.f);
        float g = (mx0 + (shoulder - mx0) * chroma) / mx0;
        float bleach = smooth01_cl((mx0 - 0.90f) * (1.f / (3.2f - 0.90f)));
        float mx = mx0 * g;
        v = v * g + bleach * (mx - v * g);
    }

    v = oklab_gamut_compress_cl(v);
    v = (float3)(srgb_oetf_cl(v.x), srgb_oetf_cl(v.y), srgb_oetf_cl(v.z));

    float Ld = clamp(luma709(v.x, v.y, v.z), 0.f, 1.f);
    float s = Ld * Ld * (3.f - 2.f * Ld);
    float Lv = Ld + contrast * (s - Ld);
    float cg = Ld > 1e-6f ? Lv / Ld : 1.f;
    v *= cg;
    float sat = max(v.x, max(v.y, v.z)) - min(v.x, min(v.y, v.z));
    float vibW = 1.f - smooth01_cl((Lv - 0.55f) * (1.f / (0.80f - 0.55f)));
    float hiRoll = 1.f - 0.55f * smooth01_cl((Lv - 0.72f) * (1.f / (1.0f - 0.72f)));
    float k = (1.f + (saturation - 1.f) * (1.f - clamp(sat, 0.f, 1.f)) * vibW) * hiRoll;
    float3 vd = Lv + (v - Lv) * k;
    float mxd = max(vd.x, max(vd.y, vd.z));
    if (mxd > 1.f && Lv < 1.f) {
        k *= (1.f - Lv) / max(1e-6f, mxd - Lv);
        vd = Lv + (v - Lv) * k;
    }
    vd = clamp(vd, 0.f, 1.f);
    vstore_half4((float4)(vd.x, vd.y, vd.z, luma709(vd.x, vd.y, vd.z)), i, rgb);
}

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

inline uchar3 dither_rgb8(float3 p, int x, int y) {
    uint seed = lcg_jump(0x9E3779B9u * (uint)(y + 1), (uint)(3 * x));
    seed = seed * 1664525u + 1013904223u;
    float d0 = ((seed >> 8) & 0xFFFF) * (1.f / 65535.f) - 0.5f;
    seed = seed * 1664525u + 1013904223u;
    float d1 = ((seed >> 8) & 0xFFFF) * (1.f / 65535.f) - 0.5f;
    seed = seed * 1664525u + 1013904223u;
    float d2 = ((seed >> 8) & 0xFFFF) * (1.f / 65535.f) - 0.5f;
    return (uchar3)((uchar)clamp((int)round(p.x * 255.f + d0), 0, 255),
                    (uchar)clamp((int)round(p.y * 255.f + d1), 0, 255),
                    (uchar)clamp((int)round(p.z * 255.f + d2), 0, 255));
}

// Fuses display-luma unsharp mask (r <= 4, locked r=3 for radius=0.85) in 16x8 Wave128 __local memory with
// coalesced 32-bit uint stores to the host-mapped unified RGB8 buffer (eliminates unaligned 1-byte WC store stalls!).
__kernel void sharpen_to_rgb8(__global const half* rgb, int W, int H, __constant float* k, int r, float amount,
                              float core, __global uchar* outRgb8) {
    __local float tile[16][24];
    __local float hblur[16][16];
    __local uint rowWords[8][12];
    int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    int gxBase = (int)get_group_id(0) * 16, gyBase = (int)get_group_id(1) * 8;
    int x = gxBase + lx, y = gyBase + ly;

    float4 px = 0.f;
    if (x < W && y < H) {
        px = vload_half4((size_t)y * W + x, rgb);
        tile[ly + r][lx + r] = px.w;
    }

    int gx0 = gxBase - r, gy0 = gyBase - r;
    int tw = 16 + 2 * r, th = 8 + 2 * r, total = tw * th;
    for (int idx = tid; idx < total; idx += 128) {
        int ty = idx / tw, tx = idx - ty * tw;
        bool inCenter = (tx >= r && tx < r + 16 && ty >= r && ty < r + 8 &&
                         (gx0 + tx) < W && (gy0 + ty) < H);
        if (!inCenter) {
            int sx = clamp(gx0 + tx, 0, W - 1), sy = clamp(gy0 + ty, 0, H - 1);
            tile[ty][tx] = vload_half4((size_t)sy * W + sx, rgb).w;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (r == 3) {
        float k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3];
        for (int ry = ly; ry < 14; ry += 8) {
            hblur[ry][lx] = k3 * tile[ry][lx + 3] + k2 * (tile[ry][lx + 2] + tile[ry][lx + 4]) +
                            k1 * (tile[ry][lx + 1] + tile[ry][lx + 5]) + k0 * (tile[ry][lx] + tile[ry][lx + 6]);
        }
    } else {
        for (int ry = ly; ry < th; ry += 8) {
            float acc = 0.f;
            for (int i = 0; i <= 2 * r; i++) acc += k[i] * tile[ry][lx + i];
            hblur[ry][lx] = acc;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < W && y < H) {
        float3 c = px.xyz;
        if (amount > 0.f) {
            float b = 0.f;
            if (r == 3) {
                b = k[3] * hblur[ly + 3][lx] + k[2] * (hblur[ly + 2][lx] + hblur[ly + 4][lx]) +
                    k[1] * (hblur[ly + 1][lx] + hblur[ly + 5][lx]) + k[0] * (hblur[ly][lx] + hblur[ly + 6][lx]);
            } else {
                for (int i = 0; i <= 2 * r; i++) b += k[i] * hblur[ly + i][lx];
            }
            float d = px.w - b;
            d = d > core ? d - core : (d < -core ? d + core : 0.f);
            c = clamp(c + amount * d, 0.f, 1.f);
        }
        uchar3 rgb8 = dither_rgb8(c, x, y);
        __local uchar* dst = ((__local uchar*)rowWords[ly]) + 3 * lx;
        dst[0] = rgb8.x;
        dst[1] = rgb8.y;
        dst[2] = rgb8.z;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (y >= H) return;
    size_t rowByteOff = ((size_t)y * W + (size_t)gxBase) * 3;
    if (gxBase + 16 <= W && (rowByteOff & 3u) == 0u) {
        if (lx < 12) {
            ((__global uint*)(outRgb8 + rowByteOff))[lx] = rowWords[ly][lx];
        }
    } else if (x < W) {
        __local const uchar* src = ((__local const uchar*)rowWords[ly]) + 3 * lx;
        __global uchar* o = outRgb8 + ((size_t)y * W + x) * 3;
        o[0] = src[0];
        o[1] = src[1];
        o[2] = src[2];
    }
}

__kernel void half4_to_rgb8(__global const half* rgb, int W, int H, __global uchar* outRgb8) {
    __local uint rowWords[8][12];
    int lx = get_local_id(0), ly = get_local_id(1);
    int gxBase = (int)get_group_id(0) * 16, gyBase = (int)get_group_id(1) * 8;
    int x = gxBase + lx, y = gyBase + ly;
    if (x < W && y < H) {
        float4 px = vload_half4((size_t)y * W + x, rgb);
        uchar3 rgb8 = dither_rgb8(px.xyz, x, y);
        __local uchar* dst = ((__local uchar*)rowWords[ly]) + 3 * lx;
        dst[0] = rgb8.x; dst[1] = rgb8.y; dst[2] = rgb8.z;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (y >= H) return;
    size_t rowByteOff = ((size_t)y * W + (size_t)gxBase) * 3;
    if (gxBase + 16 <= W && (rowByteOff & 3u) == 0u) {
        if (lx < 12) ((__global uint*)(outRgb8 + rowByteOff))[lx] = rowWords[ly][lx];
    } else if (x < W) {
        __local const uchar* src = ((__local const uchar*)rowWords[ly]) + 3 * lx;
        __global uchar* o = outRgb8 + ((size_t)y * W + x) * 3;
        o[0] = src[0]; o[1] = src[1]; o[2] = src[2];
    }
}
)CL"