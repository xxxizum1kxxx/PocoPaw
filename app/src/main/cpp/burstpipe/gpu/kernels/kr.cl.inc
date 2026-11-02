R"CL(
// kr.cpp on the device: LK refinement of the tile offsets, kernel shape, robustness and the gather.

__kernel void lk_grad(__global const float* m, int off, int w, int h, __global float* gx, __global float* gy) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    m += off;
    gx[y * w + x] = 0.5f * (at_clamped(m, w, h, x + 1, y) - at_clamped(m, w, h, x - 1, y));
    gy[y * w + x] = 0.5f * (at_clamped(m, w, h, x, y + 1) - at_clamped(m, w, h, x, y - 1));
}

inline float bilinear(__global const float* im, int w, int h, float x, float y) {
    x = clamp(x, 0.f, w - 1.001f);
    y = clamp(y, 0.f, h - 1.001f);
    int x0 = (int)x, y0 = (int)y;
    float fx = x - x0, fy = y - y0;
    __global const float* r0 = im + y0 * w + x0;
    __global const float* r1 = r0 + w;
    return (1 - fy) * ((1 - fx) * r0[0] + fx * r0[1]) + fy * ((1 - fx) * r1[0] + fx * r1[1]);
}

// Inverse-compositional Lucas-Kanade from the block-matching offset. 16 work items per tile, one per tile
// column (neighbouring items read neighbouring addresses); column sums go through local memory and every
// item of the tile adds them up in the same order, so all of them step the same way. Work groups of 64 =
// 4 tiles.
__kernel void refine_flow(__global const float* mean3, int refOff, int altOff, __global const float* gxi,
                          __global const float* gyi, int w, int h, __global const short* odx,
                          __global const short* ody, int tOff, int nx, int ny, int iters, __global float* fxo,
                          __global float* fyo, __local float* red) {
    int g0 = get_global_id(0), j = get_global_id(1), lid = get_local_id(0);
    int u = g0 & 15, i = g0 >> 4, base = lid & ~15;
    bool valid = i < nx && j < ny;
    __global const float* mref = mean3 + refOff;
    __global const float* malt = mean3 + altOff;
    odx += tOff; ody += tOff; fxo += tOff; fyo += tOff;
    int t = j * nx + i;
    float dx0 = valid ? odx[t] : 0.f, dy0 = valid ? ody[t] : 0.f;
    int x0 = max(1, i * 8), x1 = min(w - 1, i * 8 + 16);
    int y0 = max(1, j * 8), y1 = min(h - 1, j * 8 + 16);
    int x = i * 8 + u;
    bool col = valid && x >= x0 && x < x1;
    float ra = 0, rb = 0, rc = 0;
    if (col)
        for (int y = y0; y < y1; y++) {
            float gx = gxi[y * w + x], gy = gyi[y * w + x];
            ra += gx * gx; rb += gx * gy; rc += gy * gy;
        }
    red[lid * 3] = ra; red[lid * 3 + 1] = rb; red[lid * 3 + 2] = rc;
    barrier(CLK_LOCAL_MEM_FENCE);
    float a = 0, b = 0, c = 0;
    for (int k = 0; k < 16; k++) { a += red[(base + k) * 3]; b += red[(base + k) * 3 + 1]; c += red[(base + k) * 3 + 2]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    float det = a * c - b * b, tr = a + c;
    bool flat = x1 <= x0 || y1 <= y0 || tr <= 0 || det < 1e-4f * tr * tr;
    bool done = !valid || flat;
    float dx = dx0, dy = dy0;
    for (int it = 0; it < iters; it++) {
        float sx = 0, sy = 0;
        if (!done && col) {
            int ix = (int)floor(dx), iy = (int)floor(dy);
            float fx = dx - ix, fy = dy - iy;
            bool inside = x0 + ix >= 0 && x1 + ix < w - 1 && y0 + iy >= 0 && y1 + iy < h - 1;
            float w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx, w10 = fy * (1 - fx), w11 = fy * fx;
            if (inside) {
                __global const float* A = malt + (y0 + iy) * w + x + ix;
                float p0 = A[0], p1 = A[1];
                for (int y = y0; y < y1; y++) {
                    A += w;
                    float n0 = A[0], n1 = A[1];
                    float r = w00 * p0 + w01 * p1 + w10 * n0 + w11 * n1 - mref[y * w + x];
                    p0 = n0; p1 = n1;
                    sx += gxi[y * w + x] * r;
                    sy += gyi[y * w + x] * r;
                }
            } else {
                for (int y = y0; y < y1; y++) {
                    float r = bilinear(malt, w, h, x + dx, y + dy) - mref[y * w + x];
                    sx += gxi[y * w + x] * r;
                    sy += gyi[y * w + x] * r;
                }
            }
        }
        red[lid * 3] = sx; red[lid * 3 + 1] = sy;
        barrier(CLK_LOCAL_MEM_FENCE);
        float bx = 0, by = 0;
        for (int k = 0; k < 16; k++) { bx += red[(base + k) * 3]; by += red[(base + k) * 3 + 1]; }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!done) {
            float ux = (c * bx - b * by) / det, uy = (a * by - b * bx) / det;
            ux = clamp(ux, -1.f, 1.f); uy = clamp(uy, -1.f, 1.f);
            dx -= ux; dy -= uy;
            if (fabs(ux) + fabs(uy) < 0.01f) done = true;
        }
    }
    if (!valid || u) return;
    if (!flat) {
        dx = clamp(dx, dx0 - 1.f, dx0 + 1.f);
        dy = clamp(dy, dy0 - 1.f, dy0 + 1.f);
    }
    fxo[t] = dx; fyo[t] = dy;
}

// displacement of quad (qx, qy): raised-cosine blend of the 2x2 overlapping tiles
inline float2 flow_at(__global const float* fx, __global const float* fy, int nx, int ny, __constant float* win,
                      int qx, int qy) {
    int i0, i1, j0, j1;
    float wx0, wx1, wy0, wy1;
    tile_taps(qx, nx, win, &i0, &i1, &wx0, &wx1);
    tile_taps(qy, ny, win, &j0, &j1, &wy0, &wy1);
    float dx = wy0 * (wx0 * fx[j0 * nx + i0] + wx1 * fx[j0 * nx + i1]) + wy1 * (wx0 * fx[j1 * nx + i0] + wx1 * fx[j1 * nx + i1]);
    float dy = wy0 * (wx0 * fy[j0 * nx + i0] + wx1 * fy[j0 * nx + i1]) + wy1 * (wx0 * fy[j1 * nx + i0] + wx1 * fy[j1 * nx + i1]);
    return (float2)(dx, dy);
}

// variance-stabilised reference (generalised Anscombe)
__kernel void kr_vst(__global const float* gray, int qw, int qh, float black, float a, float b, __global float* vst) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= qw || y >= qh) return;
    float s = gray[y * qw + x] / 4 - black;
    vst[y * qw + x] = 2.f / a * sqrt(max(0.f, a * s + 0.375f * a * a + b));
}

// inverse kernel covariance per quad from the structure tensor
__kernel void kr_icov(__global const float* vst, int qw, int qh, float kDetail, float kDenoise, float dTh, float dTr,
                      float kStretch, float kShrink, __global float* icov) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= qw || y >= qh) return;
    float txx = 0, txy = 0, tyy = 0;
    for (int v = -1; v <= 0; v++)
        for (int u = -1; u <= 0; u++) {
            int X = x + u, Y = y + v;
            float a00 = at_clamped(vst, qw, qh, X, Y), a10 = at_clamped(vst, qw, qh, X + 1, Y);
            float a01 = at_clamped(vst, qw, qh, X, Y + 1), a11 = at_clamped(vst, qw, qh, X + 1, Y + 1);
            float gx = 0.5f * ((a10 - a00) + (a11 - a01));
            float gy = 0.5f * ((a01 - a00) + (a11 - a10));
            txx += gx * gx; txy += gx * gy; tyy += gy * gy;
        }
    float tr = txx + tyy, det = txx * tyy - txy * txy;
    float disc = sqrt(max(0.f, 0.25f * tr * tr - det));
    float l1 = 0.5f * tr + disc, l2 = max(0.f, 0.5f * tr - disc);
    float e1x = 1, e1y = 0;
    if (fabs(txy) > 1e-12f) { e1x = txy; e1y = l1 - txx; float nrm = hypot(e1x, e1y); e1x /= nrm; e1y /= nrm; }
    else if (tyy > txx) { e1x = 0; e1y = 1; }
    float A = (l1 + l2) > 1e-12f ? 1.f + sqrt((l1 - l2) / (l1 + l2)) : 1.f;
    float D = clamp(1.f - sqrt(l1) / dTr + dTh, 0.f, 1.f);
    float k1 = (2 - A) + (A - 1) / kShrink, k2 = (2 - A) + (A - 1) * kStretch;
    k1 = kDetail * ((1 - D) * k1 + D * kDenoise);
    k2 = kDetail * ((1 - D) * k2 + D * kDenoise);
    float i1 = 1.f / (k1 * k1), i2 = 1.f / (k2 * k2);
    float e2x = -e1y, e2y = e1x;
    __global float* o = icov + (y * qw + x) * 3;
    o[0] = i1 * e1x * e1x + i2 * e2x * e2x;
    o[1] = i1 * e1x * e1y + i2 * e2x * e2y;
    o[2] = i1 * e1y * e1y + i2 * e2y * e2y;
}

__kernel void kr_rob(__global const float* fx, __global const float* fy, int tOff, int nx, int ny,
                     __constant float* win, __global const float* mean3, int refOff, int kOff,
                     __global const ushort* frames, int fOff, int W, int qw, int qh, float g, float clip, float black,
                     float na, float nb, float lo, float hi, __global float* rob, int robOff) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    fx += tOff; fy += tOff; rob += robOff;
    __global const float* mref = mean3 + refOff;
    __global const float* mk = mean3 + kOff;
    __global const ushort* F = frames + fOff;
    float2 d = flow_at(fx, fy, nx, ny, win, qx, qy);
    float mu = bilinear(mk, qw, qh, qx + d.x, qy + d.y);
    int sx = clamp((int)round(qx + d.x), 0, qw - 1), sy = clamp((int)round(qy + d.y), 0, qh - 1);
    rob[qy * qw + qx] = quad_clipped(F, W, sx, sy, clip) ? 0.f : frame_weight(mref[qy * qw + qx], mu, black, na, nb, g, lo, hi);
}

// gathered per output pixel over every frame's 3x3 raw neighbourhood around its aligned position
__kernel void kr_gather(__global const ushort* frames, int frameSize, int n, int ref, int W, int H,
                        __global const float* fx, __global const float* fy, int nx, int ny, int tileCount,
                        __constant float* win, __global const float* rob, __global const float* icov, int qw,
                        int qh, float sc, int OW, int OH, int cfa, float4 blk, int lca, float4 lcaK, float lcx,
                        float lcy, float invR2, __global float* out) {
    int X = get_global_id(0), Y = get_global_id(1);
    if (X >= OW || Y >= OH) return;
    float ly = (Y + 0.5f) / sc, lx = (X + 0.5f) / sc;
    int qy = clamp((int)(ly / 2), 0, qh - 1), ky = clamp((int)round(ly / 2 - 0.5f), 0, qh - 1);
    int qx = clamp((int)(lx / 2), 0, qw - 1), kx = clamp((int)round(lx / 2 - 0.5f), 0, qw - 1);
    __global const float* ic = icov + (ky * qw + kx) * 3;
    float ia = ic[0], ib2 = 2 * ic[1], icc = ic[2];
    // no per-colour arrays: Adreno spills private arrays indexed at run time to memory
    float rdx = 0, rdy = 0, bdx = 0, bdy = 0;
    if (lca) {
        float px = lx - lcx, py = ly - lcy, rr = (px * px + py * py) * invR2;
        float kr = lcaK.x + lcaK.y * rr, kb = lcaK.z + lcaK.w * rr;
        rdx = px * kr; rdy = py * kr; bdx = px * kb; bdy = py * kb;
    }
    float3 num = 0, den = 0;
    for (int k = 0; k < n; k++) {
        float r = 1.f, mx = lx, my = ly;
        if (k != ref) {
            r = rob[(size_t)k * qw * qh + qy * qw + qx];
            if (r <= 0.f) continue;
            float2 d = flow_at(fx + k * tileCount, fy + k * tileCount, nx, ny, win, qx, qy);
            mx += 2 * d.x; my += 2 * d.y;
        }
        if (mx < 0 || my < 0 || mx >= W || my >= H) continue;
        __global const ushort* F = frames + (size_t)k * frameSize;
        int cx = (int)mx, cy = (int)my;
        float mj = mx - 0.5f, mi = my - 0.5f;
        for (int di = -1; di <= 1; di++) {
            int i = cy + di;
            if (i < 0 || i >= H) continue;
            for (int dj = -1; dj <= 1; dj++) {
                int j = cx + dj;
                if (j < 0 || j >= W) continue;
                int ch = cfa_channel(cfa, j, i);
                float ddx = j - mj - (ch == 0 ? rdx : ch == 3 ? bdx : 0.f);
                float ddy = i - mi - (ch == 0 ? rdy : ch == 3 ? bdy : 0.f);
                float bl = ch == 0 ? blk.x : ch == 1 ? blk.y : ch == 2 ? blk.z : blk.w;
                float z = ia * ddx * ddx + ib2 * ddx * ddy + icc * ddy * ddy;
                float w = r * native_exp(-0.5f * max(0.f, z));
                float3 m = (float3)(ch == 0, ch == 1 || ch == 2, ch == 3);
                num += m * (w * (F[(size_t)i * W + j] - bl));
                den += m * w;
            }
        }
    }
    __global float* o = out + ((size_t)Y * OW + X) * 3;
    o[0] = den.x > 1e-30f ? max(0.f, num.x / den.x) : 0.f;
    o[1] = den.y > 1e-30f ? max(0.f, num.y / den.y) : 0.f;
    o[2] = den.z > 1e-30f ? max(0.f, num.z / den.z) : 0.f;
}
)CL"
