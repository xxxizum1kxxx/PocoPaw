R"CL(
// merge.cpp (spatial robust merge) and merge_freq.cpp (frequency-domain Wiener merge) on the device.

inline float frame_weight(float mu0, float mu, float black, float na, float nb, float g, float lo, float hi) {
    float s = max(0.f, mu0 / 4 - black);
    float vRef = na * s + nb, vAlt = g * na * s + g * g * nb;
    float varMu = 4.f * (vRef + vAlt) / 9.f;
    float z = fabs(mu - mu0) / sqrt(max(varMu, 1e-3f));
    return clamp((hi - z) / (hi - lo), 0.f, 1.f) * (vRef / vAlt);
}

inline bool quad_clipped(__global const ushort* F, int W, int sx, int sy, float clip) {
    __global const ushort* r0 = F + (size_t)(2 * sy) * W + 2 * sx;
    __global const ushort* r1 = r0 + W;
    return max(max(r0[0], r0[1]), max(r1[0], r1[1])) >= clip;
}

__kernel void merge_init(__global const ushort* R, int rOff, int W, int qw, int qh, float black, __global float* out,
                         __global float* wsum) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    R += rOff;
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    out[i0] = R[i0] - black; out[i0 + 1] = R[i0 + 1] - black;
    out[i1] = R[i1] - black; out[i1 + 1] = R[i1 + 1] - black;
    wsum[qy * qw + qx] = 1.f;
}

// pass 1 of merge_frames: robustness of one alternate frame per quad
__kernel void merge_rob(__global const float* mean3, int refOff, int kOff, __global const ushort* frames, int fOff, int W,
                        int qw, int qh, __global const short* odx, __global const short* ody, int tOff, int nx, int ny,
                        __constant float* win, float g, float clip, float black, float na, float nb, float lo,
                        float hi, __global float* rob) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    __global const float* mref = mean3 + refOff;
    __global const float* mk = mean3 + kOff;
    __global const ushort* F = frames + fOff;
    odx += tOff;
    ody += tOff;
    int ti[2], tj[2];
    float wx[2], wy[2];
    tile_taps(qx, nx, win, &ti[0], &ti[1], &wx[0], &wx[1]);
    tile_taps(qy, ny, win, &tj[0], &tj[1], &wy[0], &wy[1]);
    float mu = 0;
    bool clipped = false;
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++) {
            int t = tj[b] * nx + ti[c];
            int sx = clamp(qx + odx[t], 0, qw - 1), sy = clamp(qy + ody[t], 0, qh - 1);
            mu += wx[c] * wy[b] * mk[sy * qw + sx];
            clipped |= quad_clipped(F, W, sx, sy, clip);
        }
    rob[qy * qw + qx] = clipped ? 0.f : frame_weight(mref[qy * qw + qx], mu, black, na, nb, g, lo, hi);
}

// pass 3 of merge_frames: accumulate one alternate frame into the reference
__kernel void merge_acc(__global const float* rob, __global const ushort* frames, int fOff, int W, int qw, int qh,
                        __global const short* odx, __global const short* ody, int tOff, int nx, int ny,
                        __constant float* win, float black, __global float* out, __global float* wsum) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    __global const ushort* F = frames + fOff;
    odx += tOff;
    ody += tOff;
    float w = rob[qy * qw + qx];
    if (w <= 0) return;
    int ti[2], tj[2];
    float wx[2], wy[2];
    tile_taps(qx, nx, win, &ti[0], &ti[1], &wx[0], &wx[1]);
    tile_taps(qy, ny, win, &tj[0], &tj[1], &wy[0], &wy[1]);
    float a[4] = {0, 0, 0, 0};
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++) {
            float tw = wx[c] * wy[b];
            int t = tj[b] * nx + ti[c];
            int sx = clamp(qx + odx[t], 0, qw - 1), sy = clamp(qy + ody[t], 0, qh - 1);
            __global const ushort* r0 = F + (size_t)(2 * sy) * W + 2 * sx;
            __global const ushort* r1 = r0 + W;
            a[0] += tw * r0[0]; a[1] += tw * r0[1];
            a[2] += tw * r1[0]; a[3] += tw * r1[1];
        }
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    out[i0] += w * (a[0] - black); out[i0 + 1] += w * (a[1] - black);
    out[i1] += w * (a[2] - black); out[i1 + 1] += w * (a[3] - black);
    wsum[qy * qw + qx] += w;
}

__kernel void merge_norm(__global float* out, __global const float* wsum, int W, int qw, int qh, float black) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    float inv = 1.f / wsum[qy * qw + qx];
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    out[i0] = out[i0] * inv + black; out[i0 + 1] = out[i0 + 1] * inv + black;
    out[i1] = out[i1] * inv + black; out[i1 + 1] = out[i1 + 1] * inv + black;
}

// ---- frequency-domain merge: one 16x16 work group per tile, one work item per (u, v) coefficient,
// the four Bayer planes in the lanes of a float4. The 2D DFT is two passes of 16-point DFTs through
// local memory.

inline float4 load_quad(__global const ushort* F, int W, int qw, int qh, int x, int y) {
    x = clamp(x, 0, qw - 1);
    y = clamp(y, 0, qh - 1);
    __global const ushort* r0 = F + (size_t)(2 * y) * W + 2 * x;
    __global const ushort* r1 = r0 + W;
    return (float4)(r0[0], r0[1], r1[0], r1[1]);
}

// in: lre/lim hold the tile (index v*16+u); out: this item's coefficient. sgn -1 forward, +1 inverse.
inline void dft2d(__local float4* lre, __local float4* lim, __local float4* tre, __local float4* tim,
                  __local const float* tc, __local const float* ts, int u, int v, float sgn, float4* ore, float4* oim) {
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 sr = 0, si = 0;
    for (int x = 0; x < 16; x++) {
        int m = (u * x) & 15;
        float c = tc[m], s = sgn * ts[m];
        float4 a = lre[v * 16 + x], b = lim[v * 16 + x];
        sr += a * c - b * s;
        si += a * s + b * c;
    }
    tre[v * 16 + u] = sr;
    tim[v * 16 + u] = si;
    barrier(CLK_LOCAL_MEM_FENCE);
    sr = 0; si = 0;
    for (int y = 0; y < 16; y++) {
        int m = (v * y) & 15;
        float c = tc[m], s = sgn * ts[m];
        float4 a = tre[y * 16 + u], b = tim[y * 16 + u];
        sr += a * c - b * s;
        si += a * s + b * c;
    }
    *ore = sr;
    *oim = si;
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline float4 local_sum4(__local float4* scratch, float4 v, int i) {
    scratch[i] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (i < s) scratch[i] += scratch[i + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float4 r = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

// tiles (ti, tj) with ti % 2 == px and tj % 2 == py do not overlap, so one launch accumulates without races
__kernel void merge_freq_tiles(__global const ushort* frames, int W, int qw, int qh, int n, int ref, int frameSize,
                               __global const short* odx, __global const short* ody, int nx, int ny, int tileCount,
                               __global const float* sinw, __global const float* tc, __global const float* ts,
                               float4 black, float w2, float c, float spatial, float na, float nb, int px, int py,
                               int ntx, int nty, __global float* out, __global float* tileW, __local float4* lre,
                               __local float4* lim, __local float4* tre, __local float4* tim, __local float* ltw) {
    __local float* ltc = ltw;
    __local float* lts = ltw + 16;
    int u = get_local_id(0), v = get_local_id(1), i = v * 16 + u;
    int ti = 2 * get_group_id(0) + px, tj = 2 * get_group_id(1) + py;
    if (ti >= ntx || tj >= nty) return;
    if (i < 16) { ltc[i] = tc[i]; lts[i] = ts[i]; }
    int x0 = (ti - 1) * 8, y0 = (tj - 1) * 8;
    float wv = sinw[u] * sinw[v];

    float4 p = load_quad(frames + (size_t)ref * frameSize, W, qw, qh, x0 + u, y0 + v);
    float4 mean = local_sum4(tre, p, i) / 256.f;
    float4 sig = na * max(mean - black, (float4)0) + nb;
    float4 noiseD = 2.f * w2 * sig * c;
    lre[i] = p * wv;
    lim[i] = 0;
    float4 r0, i0;
    dft2d(lre, lim, tre, tim, ltc, lts, u, v, -1.f, &r0, &i0);
    float4 ar = r0, ai = i0, wacc = 0;
    int oi = clamp(ti - 1, 0, nx - 1), oj = clamp(tj - 1, 0, ny - 1);
    for (int k = 0; k < n; k++) {
        if (k == ref) continue;
        int t = k * tileCount + oj * nx + oi;
        lre[i] = load_quad(frames + (size_t)k * frameSize, W, qw, qh, x0 + odx[t] + u, y0 + ody[t] + v) * wv;
        lim[i] = 0;
        float4 rk, ik;
        dft2d(lre, lim, tre, tim, ltc, lts, u, v, -1.f, &rk, &ik);
        float4 dr = r0 - rk, di = i0 - ik;
        float4 d2 = dr * dr + di * di;
        float4 a = d2 / (d2 + noiseD);
        ar += rk + a * dr;
        ai += ik + a * di;
        wacc += 1.f - a;
    }
    float4 ws = local_sum4(tre, wacc, i);
    if (i == 0) tileW[tj * ntx + ti] = (ws.x + ws.y + ws.z + ws.w) / (4.f * 256.f);
    float inv = 1.f / n;
    ar *= inv;
    ai *= inv;
    if (spatial > 0 && i) {
        float4 sigM = w2 * sig * spatial / (float)n;
        float sigMs = sigM.x + sigM.y + sigM.z + sigM.w;
        int fu = min(u, 16 - u), fv = min(v, 16 - v);
        float boost = 1.f + 0.25f * sqrt((float)(fu * fu + fv * fv));
        float4 p4 = ar * ar + ai * ai;
        float pw = p4.x + p4.y + p4.z + p4.w;
        float g = pw / (pw + sigMs * boost);
        ar *= g;
        ai *= g;
    }
    lre[i] = ar;
    lim[i] = ai;
    float4 yr, yi;
    dft2d(lre, lim, tre, tim, ltc, lts, u, v, 1.f, &yr, &yi);
    int x = x0 + u, y = y0 + v;
    if (x < 0 || x >= qw || y < 0 || y >= qh) return;
    float4 o = yr * (wv / 256.f);
    size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
    out[r0i] += o.x; out[r0i + 1] += o.y;
    out[r1i] += o.z; out[r1i + 1] += o.w;
}
)CL"
