R"CL(
// Frequency-domain Wiener merge (merge_freq.cpp) via workgroup-cooperative 64-thread Radix-4 16x16 2D FFT.
// 64 work items per tile (4 threads per 16-pt line, zero private arrays, 8.7 KB local memory).

__constant float kSinWin[16] = {
    0.09801714f, 0.29028468f, 0.47139674f, 0.63439328f,
    0.77301045f, 0.88192126f, 0.95694034f, 0.99518473f,
    0.99518473f, 0.95694034f, 0.88192126f, 0.77301045f,
    0.63439328f, 0.47139674f, 0.29028468f, 0.09801714f
};

__constant float kSinDftRe[16] = {
    10.20229724f, -3.31378072f, -0.61139522f, -0.22659465f,
    -0.09993741f, -0.04437770f, -0.01700853f, -0.00391728f,
    0.00000000f,  -0.00391728f, -0.01700853f, -0.04437770f,
    -0.09993741f, -0.22659465f, -0.61139522f, -3.31378072f
};

__constant float kSinDftIm[16] = {
    0.00000000f, -0.65915197f, -0.25324819f, -0.15140570f,
    -0.09993741f, -0.06641592f, -0.04106221f, -0.01969351f,
    0.00000000f,  0.01969351f,  0.04106221f,  0.06641592f,
    0.09993741f,  0.15140570f,  0.25324819f,  0.65915197f
};

inline float4 load_quad(__global const ushort* F, int W, int qw, int qh, int x, int y) {
    x = clamp(x, 0, qw - 1);
    y = clamp(y, 0, qh - 1);
    __global const ushort* r0 = F + (size_t)(2 * y) * W + 2 * x;
    __global const ushort* r1 = r0 + W;
    return (float4)(r0[0], r0[1], r1[0], r1[1]);
}

inline bool shift_inside(int x0, int d, int q) {
    int lo = max(0, -x0), hi = min(16, q - x0);
    return lo >= hi || (x0 + d + lo >= 0 && x0 + d + hi <= q);
}

inline void r4_step(float sgn,
                    float4 xr0, float4 xi0, float4 xr1, float4 xi1,
                    float4 xr2, float4 xi2, float4 xr3, float4 xi3,
                    float4* yr0, float4* yi0, float4* yr1, float4* yi1,
                    float4* yr2, float4* yi2, float4* yr3, float4* yi3) {
    float4 tr0 = xr0 + xr2, ti0 = xi0 + xi2;
    float4 tr1 = xr0 - xr2, ti1 = xi0 - xi2;
    float4 tr2 = xr1 + xr3, ti2 = xi1 + xi3;
    float4 tr3 = xr1 - xr3, ti3 = xi1 - xi3;
    *yr0 = tr0 + tr2;
    *yi0 = ti0 + ti2;
    *yr2 = tr0 - tr2;
    *yi2 = ti0 - ti2;
    *yr1 = tr1 - sgn * ti3;
    *yi1 = ti1 + sgn * tr3;
    *yr3 = tr1 + sgn * ti3;
    *yi3 = ti1 - sgn * tr3;
}

inline float4 local_sum64(__local float4* buf, float4 v, int lid) {
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 32) buf[lid] += buf[lid + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 16) buf[lid] += buf[lid + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 8) buf[lid] += buf[lid + 8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 4) buf[lid] += buf[lid + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 res = buf[0] + buf[1] + buf[2] + buf[3];
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}

// Forward 16x16 2D Radix-4 FFT from windowed real values in lre[v*17 + u] (lim implicitly 0).
// Outputs frequency bins (u = lid>>2, v = (lid&3) + 4*{0,1,2,3}) in registers (or0..3, oi0..3).
inline void fft2d_r4_fwd(__local float4* lre, __local float4* lim, int lid,
                         float w1r, float w1i, float w2r, float w2i, float w3r, float w3i,
                         float4* or0, float4* oi0, float4* or1, float4* oi1,
                         float4* or2, float4* oi2, float4* or3, float4* oi3) {
    int line = lid >> 2, g = lid & 3;
    int base = line * 17;
    float4 yr0, yi0, yr1, yi1, yr2, yi2, yr3, yi3;

    // Row Stage 1 (imaginary input is 0)
    r4_step(-1.f,
            lre[base + g],      (float4)0, lre[base + g + 4],  (float4)0,
            lre[base + g + 8],  (float4)0, lre[base + g + 12], (float4)0,
            &yr0, &yi0, &yr1, &yi1, &yr2, &yi2, &yr3, &yi3);
    lre[base + g]      = yr0;                     lim[base + g]      = yi0;
    lre[base + 4 + g]  = yr1 * w1r - yi1 * w1i;   lim[base + 4 + g]  = yr1 * w1i + yi1 * w1r;
    lre[base + 8 + g]  = yr2 * w2r - yi2 * w2i;   lim[base + 8 + g]  = yr2 * w2i + yi2 * w2r;
    lre[base + 12 + g] = yr3 * w3r - yi3 * w3i;   lim[base + 12 + g] = yr3 * w3i + yi3 * w3r;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Row Stage 2 -> write transposed to (u = g + 4*q, v = line)
    int b4 = base + 4 * g;
    r4_step(-1.f,
            lre[b4],     lim[b4],     lre[b4 + 1], lim[b4 + 1],
            lre[b4 + 2], lim[b4 + 2], lre[b4 + 3], lim[b4 + 3],
            &yr0, &yi0, &yr1, &yi1, &yr2, &yi2, &yr3, &yi3);
    barrier(CLK_LOCAL_MEM_FENCE);
    lre[(g + 0) * 17 + line]  = yr0; lim[(g + 0) * 17 + line]  = yi0;
    lre[(g + 4) * 17 + line]  = yr1; lim[(g + 4) * 17 + line]  = yi1;
    lre[(g + 8) * 17 + line]  = yr2; lim[(g + 8) * 17 + line]  = yi2;
    lre[(g + 12) * 17 + line] = yr3; lim[(g + 12) * 17 + line] = yi3;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Column Stage 1
    r4_step(-1.f,
            lre[base + g],      lim[base + g],      lre[base + g + 4],  lim[base + g + 4],
            lre[base + g + 8],  lim[base + g + 8],  lre[base + g + 12], lim[base + g + 12],
            &yr0, &yi0, &yr1, &yi1, &yr2, &yi2, &yr3, &yi3);
    lre[base + g]      = yr0;                     lim[base + g]      = yi0;
    lre[base + 4 + g]  = yr1 * w1r - yi1 * w1i;   lim[base + 4 + g]  = yr1 * w1i + yi1 * w1r;
    lre[base + 8 + g]  = yr2 * w2r - yi2 * w2i;   lim[base + 8 + g]  = yr2 * w2i + yi2 * w2r;
    lre[base + 12 + g] = yr3 * w3r - yi3 * w3i;   lim[base + 12 + g] = yr3 * w3i + yi3 * w3r;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Column Stage 2 -> registers
    r4_step(-1.f,
            lre[b4],     lim[b4],     lre[b4 + 1], lim[b4 + 1],
            lre[b4 + 2], lim[b4 + 2], lre[b4 + 3], lim[b4 + 3],
            or0, oi0, or1, oi1, or2, oi2, or3, oi3);
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Inverse 16x16 2D Radix-4 FFT from register bins (u = lid>>2, v = (lid&3) + 4*{0,1,2,3}).
// Returns spatial real quads (x = (lid&3) + 4*{0,1,2,3}, y = lid>>2) in (yr0..3).
inline void fft2d_r4_inv(__local float4* lre, __local float4* lim, int lid,
                         float w1r, float w1i, float w2r, float w2i, float w3r, float w3i,
                         float4 xr0, float4 xi0, float4 xr1, float4 xi1,
                         float4 xr2, float4 xi2, float4 xr3, float4 xi3,
                         float4* out0, float4* out1, float4* out2, float4* out3) {
    int line = lid >> 2, g = lid & 3;
    int base = line * 17, b4 = base + 4 * g;
    float4 yr0, yi0, yr1, yi1, yr2, yi2, yr3, yi3;

    // Column Stage 1 (directly from registers; inverse twiddle has negated imag)
    r4_step(1.f, xr0, xi0, xr1, xi1, xr2, xi2, xr3, xi3,
            &yr0, &yi0, &yr1, &yi1, &yr2, &yi2, &yr3, &yi3);
    lre[base + g]      = yr0;                     lim[base + g]      = yi0;
    lre[base + 4 + g]  = yr1 * w1r + yi1 * w1i;   lim[base + 4 + g]  = yi1 * w1r - yr1 * w1i;
    lre[base + 8 + g]  = yr2 * w2r + yi2 * w2i;   lim[base + 8 + g]  = yi2 * w2r - yr2 * w2i;
    lre[base + 12 + g] = yr3 * w3r + yi3 * w3i;   lim[base + 12 + g] = yi3 * w3r - yr3 * w3i;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Column Stage 2 -> transpose back to row-major (v = g + 4*q, u = line)
    r4_step(1.f,
            lre[b4],     lim[b4],     lre[b4 + 1], lim[b4 + 1],
            lre[b4 + 2], lim[b4 + 2], lre[b4 + 3], lim[b4 + 3],
            &yr0, &yi0, &yr1, &yi1, &yr2, &yi2, &yr3, &yi3);
    barrier(CLK_LOCAL_MEM_FENCE);
    lre[(g + 0) * 17 + line]  = yr0; lim[(g + 0) * 17 + line]  = yi0;
    lre[(g + 4) * 17 + line]  = yr1; lim[(g + 4) * 17 + line]  = yi1;
    lre[(g + 8) * 17 + line]  = yr2; lim[(g + 8) * 17 + line]  = yi2;
    lre[(g + 12) * 17 + line] = yr3; lim[(g + 12) * 17 + line] = yi3;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Row Stage 1
    r4_step(1.f,
            lre[base + g],      lim[base + g],      lre[base + g + 4],  lim[base + g + 4],
            lre[base + g + 8],  lim[base + g + 8],  lre[base + g + 12], lim[base + g + 12],
            &yr0, &yi0, &yr1, &yi1, &yr2, &yi2, &yr3, &yi3);
    lre[base + g]      = yr0;                     lim[base + g]      = yi0;
    lre[base + 4 + g]  = yr1 * w1r + yi1 * w1i;   lim[base + 4 + g]  = yi1 * w1r - yr1 * w1i;
    lre[base + 8 + g]  = yr2 * w2r + yi2 * w2i;   lim[base + 8 + g]  = yi2 * w2r - yr2 * w2i;
    lre[base + 12 + g] = yr3 * w3r + yi3 * w3i;   lim[base + 12 + g] = yi3 * w3r - yr3 * w3i;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Row Stage 2 -> spatial outputs in registers (real part only)
    r4_step(1.f,
            lre[b4],     lim[b4],     lre[b4 + 1], lim[b4 + 1],
            lre[b4 + 2], lim[b4 + 2], lre[b4 + 3], lim[b4 + 3],
            out0, &yi0, out1, &yi1, out2, &yi2, out3, &yi3);
}

inline void wiener_bin(float4 r0, float4 i0, float4 rk, float4 ik, float4 mk,
                       int u, int v, int sub, float pxr, float pxi, float fx, float fy,
                       float4 noiseD, float4* ar, float4* ai, float4* wacc) {
    float4 skr, ski;
    if (sub) {
        int sv = v < 8 ? v : (v > 8 ? v - 16 : 0);
        float ay = 0.3926990817f * (float)sv * fy;
        float pyr = cos(ay), pyi = sin(ay);
        float pr = pxr * pyr - pxi * pyi;
        float pi = pxr * pyi + pxi * pyr;
        float wru = kSinDftRe[u], wiu = kSinDftIm[u];
        float wrv = kSinDftRe[v], wiv = kSinDftIm[v];
        float wr = wru * wrv - wiu * wiv;
        float wi = wru * wiv + wiu * wrv;
        skr = rk * pr - ik * pi + mk * wr;
        ski = rk * pi + ik * pr + mk * wi;
    } else {
        skr = rk;
        ski = ik;
    }
    float4 dr = r0 - skr, di = i0 - ski;
    float4 d2 = dr * dr + di * di;
    float4 a = d2 / (d2 + noiseD);
    *ar += skr + a * dr;
    *ai += ski + a * di;
    *wacc += 1.f - a;
}

inline void shrink_bin(float4* ar, float4* ai, int u, int v, float sigMs) {
    if (u == 0 && v == 0) return;
    int fu = min(u, 16 - u), fv = min(v, 16 - v);
    float boost = 1.f + 0.25f * sqrt((float)(fu * fu + fv * fv));
    float4 p4 = (*ar) * (*ar) + (*ai) * (*ai);
    float pw = p4.x + p4.y + p4.z + p4.w;
    float gn = pw / (pw + sigMs * boost);
    *ar *= gn;
    *ai *= gn;
}

__kernel void zero_f(__global float* a, int n) {
    int i = get_global_id(0);
    if (i < n) a[i] = 0.f;
}

__kernel void merge_freq_tiles(__global const ushort* frames, int W, int qw, int qh, int n, int ref, int frameSize,
                               __global const short* odx, __global const short* ody, int nx, int ny, int tileCount,
                               int subOff, float4 black, float w2, float c, float spatial, float na, float nb,
                               int px, int py, int ntx, int nty, __global float* out, __global float* tileW,
                               __local float4* lre, __local float4* lim) {
    int ti = 2 * get_group_id(0) + px, tj = 2 * get_group_id(1) + py;
    if (ti >= ntx || tj >= nty) return;
    int lid = get_local_id(0);
    int u = lid >> 2, g = lid & 3;
    int lu = lid & 15, lv = lid >> 4;
    int x0 = (ti - 1) * 8, y0 = (tj - 1) * 8;

    float ang = -0.3926990817f * (float)g;
    float w1r = cos(ang),       w1i = sin(ang);
    float w2r = cos(2.f * ang), w2i = sin(2.f * ang);
    float w3r = cos(3.f * ang), w3i = sin(3.f * ang);

    // Load reference tile (coalesced 16-wide row reads) & compute tile mean
    __global const ushort* refBase = frames + (size_t)ref * frameSize;
    float4 q0 = load_quad(refBase, W, qw, qh, x0 + lu, y0 + lv);
    float4 q1 = load_quad(refBase, W, qw, qh, x0 + lu, y0 + lv + 4);
    float4 q2 = load_quad(refBase, W, qw, qh, x0 + lu, y0 + lv + 8);
    float4 q3 = load_quad(refBase, W, qw, qh, x0 + lu, y0 + lv + 12);
    float4 mean = local_sum64(lre, q0 + q1 + q2 + q3, lid) * (1.f / 256.f);
    float4 sig = na * max(mean - black, (float4)0) + nb;
    float4 noiseD = 2.f * w2 * sig * c;

    float wlu = kSinWin[lu];
    lre[(lv +  0) * 17 + lu] = q0 * (wlu * kSinWin[lv +  0]);
    lre[(lv +  4) * 17 + lu] = q1 * (wlu * kSinWin[lv +  4]);
    lre[(lv +  8) * 17 + lu] = q2 * (wlu * kSinWin[lv +  8]);
    lre[(lv + 12) * 17 + lu] = q3 * (wlu * kSinWin[lv + 12]);
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 r00, i00, r01, i01, r02, i02, r03, i03;
    fft2d_r4_fwd(lre, lim, lid, w1r, w1i, w2r, w2i, w3r, w3i,
                 &r00, &i00, &r01, &i01, &r02, &i02, &r03, &i03);

    float4 ar0 = r00, ai0 = i00, ar1 = r01, ai1 = i01;
    float4 ar2 = r02, ai2 = i02, ar3 = r03, ai3 = i03;
    float4 wacc = (float4)0;

    int oi = clamp(ti - 1, 0, nx - 1), oj = clamp(tj - 1, 0, ny - 1);
    int su = u < 8 ? u : (u > 8 ? u - 16 : 0);

    for (int k = 0; k < n; k++) {
        if (k == ref) continue;
        int t = k * tileCount + oj * nx + oi;
        int dx = odx[t], dy = ody[t];
        if (!shift_inside(x0, dx, qw) || !shift_inside(y0, dy, qh)) {
            ar0 += r00; ai0 += i00; ar1 += r01; ai1 += i01;
            ar2 += r02; ai2 += i02; ar3 += r03; ai3 += i03;
            continue;
        }
        float fx = subOff > 0 ? odx[subOff + t] * (1.f / 16384.f) : 0.f;
        float fy = subOff > 0 ? ody[subOff + t] * (1.f / 16384.f) : 0.f;
        int sub = (fx != 0.f || fy != 0.f) ? 1 : 0;

        __global const ushort* fBase = frames + (size_t)k * frameSize;
        q0 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv);
        q1 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv + 4);
        q2 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv + 8);
        q3 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv + 12);

        float4 mk = (float4)0;
        if (sub) mk = local_sum64(lre, q0 + q1 + q2 + q3, lid) * (1.f / 256.f);

        lre[(lv +  0) * 17 + lu] = (q0 - mk) * (wlu * kSinWin[lv +  0]);
        lre[(lv +  4) * 17 + lu] = (q1 - mk) * (wlu * kSinWin[lv +  4]);
        lre[(lv +  8) * 17 + lu] = (q2 - mk) * (wlu * kSinWin[lv +  8]);
        lre[(lv + 12) * 17 + lu] = (q3 - mk) * (wlu * kSinWin[lv + 12]);
        barrier(CLK_LOCAL_MEM_FENCE);

        float4 rk0, ik0, rk1, ik1, rk2, ik2, rk3, ik3;
        fft2d_r4_fwd(lre, lim, lid, w1r, w1i, w2r, w2i, w3r, w3i,
                     &rk0, &ik0, &rk1, &ik1, &rk2, &ik2, &rk3, &ik3);

        float ax = sub ? 0.3926990817f * (float)su * fx : 0.f;
        float pxr = sub ? cos(ax) : 1.f;
        float pxi = sub ? sin(ax) : 0.f;
        wiener_bin(r00, i00, rk0, ik0, mk, u, g,      sub, pxr, pxi, fx, fy, noiseD, &ar0, &ai0, &wacc);
        wiener_bin(r01, i01, rk1, ik1, mk, u, g + 4,  sub, pxr, pxi, fx, fy, noiseD, &ar1, &ai1, &wacc);
        wiener_bin(r02, i02, rk2, ik2, mk, u, g + 8,  sub, pxr, pxi, fx, fy, noiseD, &ar2, &ai2, &wacc);
        wiener_bin(r03, i03, rk3, ik3, mk, u, g + 12, sub, pxr, pxi, fx, fy, noiseD, &ar3, &ai3, &wacc);
    }

    float4 ws = local_sum64(lre, wacc, lid);
    if (lid == 0) tileW[tj * ntx + ti] = (ws.x + ws.y + ws.z + ws.w) / (4.f * 256.f);

    float inv = 1.f / (float)n;
    ar0 *= inv; ai0 *= inv; ar1 *= inv; ai1 *= inv;
    ar2 *= inv; ai2 *= inv; ar3 *= inv; ai3 *= inv;
    if (spatial > 0.f) {
        float4 sigM = w2 * sig * spatial * inv;
        float sigMs = sigM.x + sigM.y + sigM.z + sigM.w;
        shrink_bin(&ar0, &ai0, u, g,      sigMs);
        shrink_bin(&ar1, &ai1, u, g + 4,  sigMs);
        shrink_bin(&ar2, &ai2, u, g + 8,  sigMs);
        shrink_bin(&ar3, &ai3, u, g + 12, sigMs);
    }

    float4 yr0, yr1, yr2, yr3;
    fft2d_r4_inv(lre, lim, lid, w1r, w1i, w2r, w2i, w3r, w3i,
                 ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3,
                 &yr0, &yr1, &yr2, &yr3);

    int y = y0 + u;
    if (y < 0 || y >= qh) return;
    float wy = kSinWin[u] * (1.f / 256.f);
    int x = x0 + g;
    if (x >= 0 && x < qw) {
        float4 o = yr0 * (wy * kSinWin[g]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
    x = x0 + g + 4;
    if (x >= 0 && x < qw) {
        float4 o = yr1 * (wy * kSinWin[g + 4]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
    x = x0 + g + 8;
    if (x >= 0 && x < qw) {
        float4 o = yr2 * (wy * kSinWin[g + 8]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
    x = x0 + g + 12;
    if (x >= 0 && x < qw) {
        float4 o = yr3 * (wy * kSinWin[g + 12]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
}

__kernel void raw_to_float(__global const ushort* raw, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = (float)raw[i];
}

inline void pack_hermitian_cl(float4 r00, float4 i00, float4 r01, float4 i01, float4 r02, float4 i02,
                              int u, int g, __local float4* lre, __local float4* lim) {
    if (g > 0) {
        lre[g * 16 + u] = r00;
        lim[g * 16 + u] = i00;
    } else if (u == 0) {
        lre[0] = r00;
        lim[0] = r02;
    } else if (u == 8) {
        lre[8] = r00;
        lim[8] = r02;
    } else if (u < 8) {
        lre[u] = r00;
        lim[u] = i00;
        lre[16 - u] = r02;
        lim[16 - u] = i02;
    }
    lre[(g + 4) * 16 + u] = r01;
    lim[(g + 4) * 16 + u] = i01;
}

inline void wiener_real_bin(float4 r0, float4 rk, float4 ik, float4 mk,
                            int u, int v, int sub, float pxr, float pxi, float fx, float fy,
                            float4 noiseD, float4* dst, float4* wacc) {
    float4 skr;
    if (sub) {
        int sv = v < 8 ? v : (v > 8 ? v - 16 : 0);
        float ay = 0.3926990817f * (float)sv * fy;
        float pyr = cos(ay), pyi = sin(ay);
        float pr = pxr * pyr - pxi * pyi;
        float pi = pxr * pyi + pxi * pyr;
        float wru = kSinDftRe[u], wiu = kSinDftIm[u];
        float wrv = kSinDftRe[v], wiv = kSinDftIm[v];
        float wr = wru * wrv - wiu * wiv;
        skr = rk * pr - ik * pi + mk * wr;
    } else {
        skr = rk;
    }
    float4 dr = r0 - skr;
    float4 d2 = dr * dr;
    float4 a = d2 / (d2 + noiseD);
    *dst += skr + a * dr;
    *wacc += 1.f - a;
}

__kernel void freq_stream_ref(__global const ushort* refFrame, int W, int qw, int qh, int ntx, int nty,
                              __global float4* accRe, __global float4* accIm, __global float4* tileMean,
                              __local float4* lre, __local float4* lim) {
    int ti = get_group_id(0), tj = get_group_id(1);
    if (ti >= ntx || tj >= nty) return;
    int lid = get_local_id(0);
    int u = lid >> 2, g = lid & 3;
    int lu = lid & 15, lv = lid >> 4;
    int x0 = (ti - 1) * 8, y0 = (tj - 1) * 8;

    float ang = -0.3926990817f * (float)g;
    float w1r = cos(ang),       w1i = sin(ang);
    float w2r = cos(2.f * ang), w2i = sin(2.f * ang);
    float w3r = cos(3.f * ang), w3i = sin(3.f * ang);

    float4 q0 = load_quad(refFrame, W, qw, qh, x0 + lu, y0 + lv);
    float4 q1 = load_quad(refFrame, W, qw, qh, x0 + lu, y0 + lv + 4);
    float4 q2 = load_quad(refFrame, W, qw, qh, x0 + lu, y0 + lv + 8);
    float4 q3 = load_quad(refFrame, W, qw, qh, x0 + lu, y0 + lv + 12);
    float4 mean = local_sum64(lre, q0 + q1 + q2 + q3, lid) * (1.f / 256.f);
    int tid = tj * ntx + ti;
    if (lid == 0) tileMean[tid] = mean;

    float wlu = kSinWin[lu];
    lre[(lv +  0) * 17 + lu] = q0 * (wlu * kSinWin[lv +  0]);
    lre[(lv +  4) * 17 + lu] = q1 * (wlu * kSinWin[lv +  4]);
    lre[(lv +  8) * 17 + lu] = q2 * (wlu * kSinWin[lv +  8]);
    lre[(lv + 12) * 17 + lu] = q3 * (wlu * kSinWin[lv + 12]);
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 r00, i00, r01, i01, r02, i02, r03, i03;
    fft2d_r4_fwd(lre, lim, lid, w1r, w1i, w2r, w2i, w3r, w3i,
                 &r00, &i00, &r01, &i01, &r02, &i02, &r03, &i03);

    pack_hermitian_cl(r00, i00, r01, i01, r02, i02, u, g, lre, lim);
    barrier(CLK_LOCAL_MEM_FENCE);
    size_t base = (size_t)tid * 128;
    accRe[base + lid]      = lre[lid];      accIm[base + lid]      = lim[lid];
    accRe[base + lid + 64] = lre[lid + 64]; accIm[base + lid + 64] = lim[lid + 64];
}

__kernel void freq_stream_add(__global const ushort* frames, int W, int qw, int qh, int frameSize,
                              __global const short* odx, __global const short* ody, int nx, int ny, int subOff,
                              float4 black, float w2, float c, float na, float nb, int ntx, int nty,
                              __global float4* accRe, __global float4* accIm, __global const float4* tileMean,
                              __global float* tileW, __local float4* lre, __local float4* lim) {
    int ti = get_group_id(0), tj = get_group_id(1);
    if (ti >= ntx || tj >= nty) return;
    int lid = get_local_id(0);
    int u = lid >> 2, g = lid & 3;
    int lu = lid & 15, lv = lid >> 4;
    int x0 = (ti - 1) * 8, y0 = (tj - 1) * 8;
    int tid = tj * ntx + ti;
    size_t base = (size_t)tid * 128;

    float ang = -0.3926990817f * (float)g;
    float w1r = cos(ang),       w1i = sin(ang);
    float w2r = cos(2.f * ang), w2i = sin(2.f * ang);
    float w3r = cos(3.f * ang), w3i = sin(3.f * ang);

    float4 q0 = load_quad(frames, W, qw, qh, x0 + lu, y0 + lv);
    float4 q1 = load_quad(frames, W, qw, qh, x0 + lu, y0 + lv + 4);
    float4 q2 = load_quad(frames, W, qw, qh, x0 + lu, y0 + lv + 8);
    float4 q3 = load_quad(frames, W, qw, qh, x0 + lu, y0 + lv + 12);
    float wlu = kSinWin[lu];
    lre[(lv +  0) * 17 + lu] = q0 * (wlu * kSinWin[lv +  0]);
    lre[(lv +  4) * 17 + lu] = q1 * (wlu * kSinWin[lv +  4]);
    lre[(lv +  8) * 17 + lu] = q2 * (wlu * kSinWin[lv +  8]);
    lre[(lv + 12) * 17 + lu] = q3 * (wlu * kSinWin[lv + 12]);
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 r00, i00, r01, i01, r02, i02, r03, i03;
    fft2d_r4_fwd(lre, lim, lid, w1r, w1i, w2r, w2i, w3r, w3i,
                 &r00, &i00, &r01, &i01, &r02, &i02, &r03, &i03);

    int oi = clamp(ti - 1, 0, nx - 1), oj = clamp(tj - 1, 0, ny - 1);
    int ot = oj * nx + oi;
    int dx = odx[ot], dy = ody[ot];
    if (!shift_inside(x0, dx, qw) || !shift_inside(y0, dy, qh)) {
        pack_hermitian_cl(r00, i00, r01, i01, r02, i02, u, g, lre, lim);
        barrier(CLK_LOCAL_MEM_FENCE);
        accRe[base + lid]      += lre[lid];      accIm[base + lid]      += lim[lid];
        accRe[base + lid + 64] += lre[lid + 64]; accIm[base + lid + 64] += lim[lid + 64];
        return;
    }

    float fx = subOff > 0 ? odx[subOff + ot] * (1.f / 16384.f) : 0.f;
    float fy = subOff > 0 ? ody[subOff + ot] * (1.f / 16384.f) : 0.f;
    int sub = (fx != 0.f || fy != 0.f) ? 1 : 0;

    __global const ushort* fBase = frames + (size_t)frameSize;
    q0 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv);
    q1 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv + 4);
    q2 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv + 8);
    q3 = load_quad(fBase, W, qw, qh, x0 + dx + lu, y0 + dy + lv + 12);

    float4 mk = (float4)0;
    if (sub) mk = local_sum64(lre, q0 + q1 + q2 + q3, lid) * (1.f / 256.f);

    lre[(lv +  0) * 17 + lu] = (q0 - mk) * (wlu * kSinWin[lv +  0]);
    lre[(lv +  4) * 17 + lu] = (q1 - mk) * (wlu * kSinWin[lv +  4]);
    lre[(lv +  8) * 17 + lu] = (q2 - mk) * (wlu * kSinWin[lv +  8]);
    lre[(lv + 12) * 17 + lu] = (q3 - mk) * (wlu * kSinWin[lv + 12]);
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 rk0, ik0, rk1, ik1, rk2, ik2, rk3, ik3;
    fft2d_r4_fwd(lre, lim, lid, w1r, w1i, w2r, w2i, w3r, w3i,
                 &rk0, &ik0, &rk1, &ik1, &rk2, &ik2, &rk3, &ik3);

    lre[lid]      = accRe[base + lid];      lim[lid]      = accIm[base + lid];
    lre[lid + 64] = accRe[base + lid + 64]; lim[lid + 64] = accIm[base + lid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 mean = tileMean[tid];
    float4 sig = na * max(mean - black, (float4)0) + nb;
    float4 noiseD = 2.f * w2 * sig * c;

    int su = u < 8 ? u : (u > 8 ? u - 16 : 0);
    float ax = sub ? 0.3926990817f * (float)su * fx : 0.f;
    float pxr = sub ? cos(ax) : 1.f;
    float pxi = sub ? sin(ax) : 0.f;
    float4 wacc = (float4)0;

    if (g > 0) {
        int s = g * 16 + u;
        float4 ar = lre[s], ai = lim[s], wt = (float4)0;
        wiener_bin(r00, i00, rk0, ik0, mk, u, g, sub, pxr, pxi, fx, fy, noiseD, &ar, &ai, &wt);
        lre[s] = ar; lim[s] = ai;
        wacc += 2.f * wt;
    } else if (u == 0 || u == 8) {
        float4 ar0 = lre[u], ar2 = lim[u];
        wiener_real_bin(r00, rk0, ik0, mk, u, 0, sub, pxr, pxi, fx, fy, noiseD, &ar0, &wacc);
        wiener_real_bin(r02, rk2, ik2, mk, u, 8, sub, pxr, pxi, fx, fy, noiseD, &ar2, &wacc);
        lre[u] = ar0; lim[u] = ar2;
    } else if (u < 8) {
        float4 ar0 = lre[u], ai0 = lim[u], wt0 = (float4)0;
        wiener_bin(r00, i00, rk0, ik0, mk, u, 0, sub, pxr, pxi, fx, fy, noiseD, &ar0, &ai0, &wt0);
        lre[u] = ar0; lim[u] = ai0;
        wacc += 2.f * wt0;
        int s2 = 16 - u;
        float4 ar2 = lre[s2], ai2 = lim[s2], wt2 = (float4)0;
        wiener_bin(r02, i02, rk2, ik2, mk, u, 8, sub, pxr, pxi, fx, fy, noiseD, &ar2, &ai2, &wt2);
        lre[s2] = ar2; lim[s2] = ai2;
        wacc += 2.f * wt2;
    }
    {
        int s1 = (g + 4) * 16 + u;
        float4 ar1 = lre[s1], ai1 = lim[s1], wt1 = (float4)0;
        wiener_bin(r01, i01, rk1, ik1, mk, u, g + 4, sub, pxr, pxi, fx, fy, noiseD, &ar1, &ai1, &wt1);
        lre[s1] = ar1; lim[s1] = ai1;
        wacc += 2.f * wt1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    accRe[base + lid]      = lre[lid];      accIm[base + lid]      = lim[lid];
    accRe[base + lid + 64] = lre[lid + 64]; accIm[base + lid + 64] = lim[lid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 ws = local_sum64(lre, wacc, lid);
    if (lid == 0) tileW[tid] += (ws.x + ws.y + ws.z + ws.w) / (4.f * 256.f);
}

__kernel void freq_stream_end(__global const float4* accRe, __global const float4* accIm,
                              __global const float4* tileMean, float4 black, float w2, float spatial,
                              float na, float nb, int n, int W, int qw, int qh, int px, int py,
                              int ntx, int nty, __global float* out, __local float4* lre, __local float4* lim) {
    int ti = 2 * get_group_id(0) + px, tj = 2 * get_group_id(1) + py;
    if (ti >= ntx || tj >= nty) return;
    int lid = get_local_id(0);
    int u = lid >> 2, g = lid & 3;
    int x0 = (ti - 1) * 8, y0 = (tj - 1) * 8;
    int tid = tj * ntx + ti;
    size_t base = (size_t)tid * 128;

    lre[lid]      = accRe[base + lid];      lim[lid]      = accIm[base + lid];
    lre[lid + 64] = accRe[base + lid + 64]; lim[lid + 64] = accIm[base + lid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3;
    if (g > 0) {
        int s0 = g * 16 + u;
        ar0 = lre[s0]; ai0 = lim[s0];
    } else if (u == 0) {
        ar0 = lre[0]; ai0 = (float4)0;
    } else if (u == 8) {
        ar0 = lre[8]; ai0 = (float4)0;
    } else if (u < 8) {
        ar0 = lre[u]; ai0 = lim[u];
    } else {
        ar0 = lre[16 - u]; ai0 = -lim[16 - u];
    }
    {
        int s1 = (g + 4) * 16 + u;
        ar1 = lre[s1]; ai1 = lim[s1];
    }
    if (g > 0) {
        int u2 = (16 - u) & 15, v2 = 16 - (g + 8);
        int s2 = v2 * 16 + u2;
        ar2 = lre[s2]; ai2 = -lim[s2];
    } else if (u == 0) {
        ar2 = lim[0]; ai2 = (float4)0;
    } else if (u == 8) {
        ar2 = lim[8]; ai2 = (float4)0;
    } else if (u < 8) {
        ar2 = lre[16 - u]; ai2 = lim[16 - u];
    } else {
        ar2 = lre[u]; ai2 = -lim[u];
    }
    {
        int u3 = (16 - u) & 15, v3 = 16 - (g + 12);
        int s3 = v3 * 16 + u3;
        ar3 = lre[s3]; ai3 = -lim[s3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float inv = 1.f / (float)n;
    ar0 *= inv; ai0 *= inv; ar1 *= inv; ai1 *= inv;
    ar2 *= inv; ai2 *= inv; ar3 *= inv; ai3 *= inv;
    if (spatial > 0.f) {
        float4 mean = tileMean[tid];
        float4 sig = na * max(mean - black, (float4)0) + nb;
        float4 sigM = w2 * sig * spatial * inv;
        float sigMs = sigM.x + sigM.y + sigM.z + sigM.w;
        shrink_bin(&ar0, &ai0, u, g,      sigMs);
        shrink_bin(&ar1, &ai1, u, g + 4,  sigMs);
        shrink_bin(&ar2, &ai2, u, g + 8,  sigMs);
        shrink_bin(&ar3, &ai3, u, g + 12, sigMs);
    }

    float ang = -0.3926990817f * (float)g;
    float w1r = cos(ang),       w1i = sin(ang);
    float w2r = cos(2.f * ang), w2i = sin(2.f * ang);
    float w3r = cos(3.f * ang), w3i = sin(3.f * ang);
    float4 yr0, yr1, yr2, yr3;
    fft2d_r4_inv(lre, lim, lid, w1r, w1i, w2r, w2i, w3r, w3i,
                 ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3,
                 &yr0, &yr1, &yr2, &yr3);

    int y = y0 + u;
    if (y < 0 || y >= qh) return;
    float wy = kSinWin[u] * (1.f / 256.f);
    int x = x0 + g;
    if (x >= 0 && x < qw) {
        float4 o = yr0 * (wy * kSinWin[g]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
    x = x0 + g + 4;
    if (x >= 0 && x < qw) {
        float4 o = yr1 * (wy * kSinWin[g + 4]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
    x = x0 + g + 8;
    if (x >= 0 && x < qw) {
        float4 o = yr2 * (wy * kSinWin[g + 8]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
    x = x0 + g + 12;
    if (x >= 0 && x < qw) {
        float4 o = yr3 * (wy * kSinWin[g + 12]);
        size_t r0i = (size_t)(2 * y) * W + 2 * x, r1i = r0i + W;
        out[r0i] += o.x; out[r0i + 1] += o.y; out[r1i] += o.z; out[r1i + 1] += o.w;
    }
}
)CL"
