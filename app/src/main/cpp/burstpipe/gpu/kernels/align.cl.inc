R"CL(
// align.cpp on the device: gray image, pyramid, 3x3 mean and the coarse-to-fine tile search.

__kernel void gray_from_raw(__global const ushort* raw, int off, int rw, __global float* g, int gw, int gh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= gw || y >= gh) return;
    __global const ushort* a = raw + off + (size_t)(2 * y) * rw + 2 * x;
    __global const ushort* b = a + rw;
    g[y * gw + x] = (float)a[0] + a[1] + b[0] + b[1];
}

__kernel void downsample(__global const float* in, int iw, __global float* out, int ow, int oh, int f) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= ow || y >= oh) return;
    float s = 0;
    for (int v = 0; v < f; v++) {
        __global const float* r = in + (y * f + v) * iw + x * f;
        for (int u = 0; u < f; u++) s += r[u];
    }
    out[y * ow + x] = s * (1.f / (f * f));
}

__kernel void box3_h(__global const float* g, __global float* t, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    __global const float* r = g + y * w;
    t[y * w + x] = r[max(x - 1, 0)] + r[x] + r[min(x + 1, w - 1)];
}
__kernel void box3_v(__global const float* t, __global float* o, int ooff, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    o[ooff + y * w + x] = (t[max(y - 1, 0) * w + x] + t[y * w + x] + t[min(y + 1, h - 1) * w + x]) * (1.f / 9.f);
}

// The tile search of align_frames in three passes per pyramid level, with the same costs (rows summed in
// order) and the same choice (first of equal costs; the CPU's early exit only skips losers):
//   align_cand    cost of the 1 or 3 starting candidates (nearest coarser-level tile + 2 neighbours)
//   align_search  one work item per (tile, vertical offset): row by row, the tile row and the alt row stay in
//                 registers for all 2R+1 horizontal offsets around the best candidate
//   align_pick    the best offset per tile

inline void parent_cands(int x0, int y0, int tile, __global const short* pdx, __global const short* pdy, int pnx, int pny,
                         int ptile, int f, int cand[3][2]) {
    int pstride = ptile / 2;
    float cx = (x0 + tile * 0.5f) / f, cy = (y0 + tile * 0.5f) / f;
    float fx = (cx - ptile * 0.5f) / pstride, fy = (cy - ptile * 0.5f) / pstride;
    int pi = clamp((int)round(fx), 0, pnx - 1), pj = clamp((int)round(fy), 0, pny - 1);
    int ni = clamp(fx > pi ? pi + 1 : pi - 1, 0, pnx - 1), nj = clamp(fy > pj ? pj + 1 : pj - 1, 0, pny - 1);
    int ids[3][2] = {{pi, pj}, {ni, pj}, {pi, nj}};
    for (int c = 0; c < 3; c++) {
        int k = ids[c][1] * pnx + ids[c][0];
        cand[c][0] = pdx[k] * f;
        cand[c][1] = pdy[k] * f;
    }
}

// the starting point of the search: best candidate and its cost
inline int best_cand(__global const float* ccost, int t, int nc, float* best) {
    int b = 0;
    float bs = 3.4e38f;
    for (int c = 0; c < nc; c++) if (ccost[t * 3 + c] < bs) { bs = ccost[t * 3 + c]; b = c; }
    *best = bs;
    return b;
}

// Unrolled by hand with fixed register indices: Adreno's compiler spills private arrays indexed by loop
// variables to memory. Each row loads ref[x0, x0+T) and alt[bx, bx+T+8) once for all horizontal offsets.
__kernel void align_cand16(__global const float* ref, __global const float* alt, int w, int h, int nx, int ny, int l1,
                          __global const short* pdx, __global const short* pdy, int pnx, int pny, int ptile, int f,
                          __global float* ccost) {
    int g = get_global_id(0), j = get_global_id(1), i = g / 3, c = g - i * 3;
    if (i >= nx || j >= ny) return;
    if (c >= (f > 0 ? 3 : 1)) return;
    int x0 = i * 8, y0 = j * 8;
    int cand[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    if (f > 0) parent_cands(x0, y0, 16, pdx, pdy, pnx, pny, ptile, f, cand);
    int bx = x0 + cand[c][0], by = y0 + cand[c][1];
    float4 a0, a1, a2, a3, b0, b1, b2, b3;
    float s = 0, d;
    for (int v = 0; v < 16; v++) {
        if (x0 >= 0 && x0 + 16 <= w && y0 + v >= 0 && y0 + v < h) {
            __global const float* r_ = ref + (y0 + v) * w + x0;
            a0 = vload4(0, r_);
            a1 = vload4(1, r_);
            a2 = vload4(2, r_);
            a3 = vload4(3, r_);
        } else {
            a0 = (float4)(at_clamped(ref, w, h, x0 + 0, y0 + v), at_clamped(ref, w, h, x0 + 1, y0 + v), at_clamped(ref, w, h, x0 + 2, y0 + v), at_clamped(ref, w, h, x0 + 3, y0 + v));
            a1 = (float4)(at_clamped(ref, w, h, x0 + 4, y0 + v), at_clamped(ref, w, h, x0 + 5, y0 + v), at_clamped(ref, w, h, x0 + 6, y0 + v), at_clamped(ref, w, h, x0 + 7, y0 + v));
            a2 = (float4)(at_clamped(ref, w, h, x0 + 8, y0 + v), at_clamped(ref, w, h, x0 + 9, y0 + v), at_clamped(ref, w, h, x0 + 10, y0 + v), at_clamped(ref, w, h, x0 + 11, y0 + v));
            a3 = (float4)(at_clamped(ref, w, h, x0 + 12, y0 + v), at_clamped(ref, w, h, x0 + 13, y0 + v), at_clamped(ref, w, h, x0 + 14, y0 + v), at_clamped(ref, w, h, x0 + 15, y0 + v));
        }
        if (bx >= 0 && bx + 16 <= w && by + v >= 0 && by + v < h) {
            __global const float* r_ = alt + (by + v) * w + bx;
            b0 = vload4(0, r_);
            b1 = vload4(1, r_);
            b2 = vload4(2, r_);
            b3 = vload4(3, r_);
        } else {
            b0 = (float4)(at_clamped(alt, w, h, bx + 0, by + v), at_clamped(alt, w, h, bx + 1, by + v), at_clamped(alt, w, h, bx + 2, by + v), at_clamped(alt, w, h, bx + 3, by + v));
            b1 = (float4)(at_clamped(alt, w, h, bx + 4, by + v), at_clamped(alt, w, h, bx + 5, by + v), at_clamped(alt, w, h, bx + 6, by + v), at_clamped(alt, w, h, bx + 7, by + v));
            b2 = (float4)(at_clamped(alt, w, h, bx + 8, by + v), at_clamped(alt, w, h, bx + 9, by + v), at_clamped(alt, w, h, bx + 10, by + v), at_clamped(alt, w, h, bx + 11, by + v));
            b3 = (float4)(at_clamped(alt, w, h, bx + 12, by + v), at_clamped(alt, w, h, bx + 13, by + v), at_clamped(alt, w, h, bx + 14, by + v), at_clamped(alt, w, h, bx + 15, by + v));
        }
        d = a0.s0 - b0.s0; s += l1 ? fabs(d) : d * d;
        d = a0.s1 - b0.s1; s += l1 ? fabs(d) : d * d;
        d = a0.s2 - b0.s2; s += l1 ? fabs(d) : d * d;
        d = a0.s3 - b0.s3; s += l1 ? fabs(d) : d * d;
        d = a1.s0 - b1.s0; s += l1 ? fabs(d) : d * d;
        d = a1.s1 - b1.s1; s += l1 ? fabs(d) : d * d;
        d = a1.s2 - b1.s2; s += l1 ? fabs(d) : d * d;
        d = a1.s3 - b1.s3; s += l1 ? fabs(d) : d * d;
        d = a2.s0 - b2.s0; s += l1 ? fabs(d) : d * d;
        d = a2.s1 - b2.s1; s += l1 ? fabs(d) : d * d;
        d = a2.s2 - b2.s2; s += l1 ? fabs(d) : d * d;
        d = a2.s3 - b2.s3; s += l1 ? fabs(d) : d * d;
        d = a3.s0 - b3.s0; s += l1 ? fabs(d) : d * d;
        d = a3.s1 - b3.s1; s += l1 ? fabs(d) : d * d;
        d = a3.s2 - b3.s2; s += l1 ? fabs(d) : d * d;
        d = a3.s3 - b3.s3; s += l1 ? fabs(d) : d * d;
    }
    ccost[(j * nx + i) * 3 + c] = s;
}
__kernel void align_search16(__global const float* ref, __global const float* alt, int w, int h, int nx, int ny, int R,
                            int l1, __global const short* pdx, __global const short* pdy, int pnx, int pny, int ptile,
                            int f, __global const float* ccost, __global float* scost) {
    const int side = 2 * R + 1;
    int g = get_global_id(0), j = get_global_id(1), i = g / side, oy = g - i * side;
    if (i >= nx || j >= ny) return;
    int x0 = i * 8, y0 = j * 8, t = j * nx + i;
    int cand[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    if (f > 0) parent_cands(x0, y0, 16, pdx, pdy, pnx, pny, ptile, f, cand);
    float best0;
    int bc = best_cand(ccost, t, f > 0 ? 3 : 1, &best0);
    int bx = x0 + cand[bc][0] - R, by = y0 + cand[bc][1] - R + oy;
    float4 a0, a1, a2, a3, b0, b1, b2, b3, b4, b5;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0, d;
    for (int v = 0; v < 16; v++) {
        if (x0 >= 0 && x0 + 16 <= w && y0 + v >= 0 && y0 + v < h) {
            __global const float* r_ = ref + (y0 + v) * w + x0;
            a0 = vload4(0, r_);
            a1 = vload4(1, r_);
            a2 = vload4(2, r_);
            a3 = vload4(3, r_);
        } else {
            a0 = (float4)(at_clamped(ref, w, h, x0 + 0, y0 + v), at_clamped(ref, w, h, x0 + 1, y0 + v), at_clamped(ref, w, h, x0 + 2, y0 + v), at_clamped(ref, w, h, x0 + 3, y0 + v));
            a1 = (float4)(at_clamped(ref, w, h, x0 + 4, y0 + v), at_clamped(ref, w, h, x0 + 5, y0 + v), at_clamped(ref, w, h, x0 + 6, y0 + v), at_clamped(ref, w, h, x0 + 7, y0 + v));
            a2 = (float4)(at_clamped(ref, w, h, x0 + 8, y0 + v), at_clamped(ref, w, h, x0 + 9, y0 + v), at_clamped(ref, w, h, x0 + 10, y0 + v), at_clamped(ref, w, h, x0 + 11, y0 + v));
            a3 = (float4)(at_clamped(ref, w, h, x0 + 12, y0 + v), at_clamped(ref, w, h, x0 + 13, y0 + v), at_clamped(ref, w, h, x0 + 14, y0 + v), at_clamped(ref, w, h, x0 + 15, y0 + v));
        }
        if (bx >= 0 && bx + 24 <= w && by + v >= 0 && by + v < h) {
            __global const float* r_ = alt + (by + v) * w + bx;
            b0 = vload4(0, r_);
            b1 = vload4(1, r_);
            b2 = vload4(2, r_);
            b3 = vload4(3, r_);
            b4 = vload4(4, r_);
            b5 = vload4(5, r_);
        } else {
            b0 = (float4)(at_clamped(alt, w, h, bx + 0, by + v), at_clamped(alt, w, h, bx + 1, by + v), at_clamped(alt, w, h, bx + 2, by + v), at_clamped(alt, w, h, bx + 3, by + v));
            b1 = (float4)(at_clamped(alt, w, h, bx + 4, by + v), at_clamped(alt, w, h, bx + 5, by + v), at_clamped(alt, w, h, bx + 6, by + v), at_clamped(alt, w, h, bx + 7, by + v));
            b2 = (float4)(at_clamped(alt, w, h, bx + 8, by + v), at_clamped(alt, w, h, bx + 9, by + v), at_clamped(alt, w, h, bx + 10, by + v), at_clamped(alt, w, h, bx + 11, by + v));
            b3 = (float4)(at_clamped(alt, w, h, bx + 12, by + v), at_clamped(alt, w, h, bx + 13, by + v), at_clamped(alt, w, h, bx + 14, by + v), at_clamped(alt, w, h, bx + 15, by + v));
            b4 = (float4)(at_clamped(alt, w, h, bx + 16, by + v), at_clamped(alt, w, h, bx + 17, by + v), at_clamped(alt, w, h, bx + 18, by + v), at_clamped(alt, w, h, bx + 19, by + v));
            b5 = (float4)(at_clamped(alt, w, h, bx + 20, by + v), at_clamped(alt, w, h, bx + 21, by + v), at_clamped(alt, w, h, bx + 22, by + v), at_clamped(alt, w, h, bx + 23, by + v));
        }
        if (side > 0) {
            d = a0.s0 - b0.s0; s0 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b0.s1; s0 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b0.s2; s0 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b0.s3; s0 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s0; s0 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b1.s1; s0 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b1.s2; s0 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b1.s3; s0 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b2.s0; s0 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b2.s1; s0 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b2.s2; s0 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b2.s3; s0 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b3.s0; s0 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b3.s1; s0 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b3.s2; s0 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b3.s3; s0 += l1 ? fabs(d) : d * d;
        }
        if (side > 1) {
            d = a0.s0 - b0.s1; s1 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b0.s2; s1 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b0.s3; s1 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s0; s1 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s1; s1 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b1.s2; s1 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b1.s3; s1 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s0; s1 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b2.s1; s1 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b2.s2; s1 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b2.s3; s1 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b3.s0; s1 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b3.s1; s1 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b3.s2; s1 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b3.s3; s1 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b4.s0; s1 += l1 ? fabs(d) : d * d;
        }
        if (side > 2) {
            d = a0.s0 - b0.s2; s2 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b0.s3; s2 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s0; s2 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s1; s2 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s2; s2 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b1.s3; s2 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s0; s2 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s1; s2 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b2.s2; s2 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b2.s3; s2 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b3.s0; s2 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b3.s1; s2 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b3.s2; s2 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b3.s3; s2 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b4.s0; s2 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b4.s1; s2 += l1 ? fabs(d) : d * d;
        }
        if (side > 3) {
            d = a0.s0 - b0.s3; s3 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s0; s3 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s1; s3 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s2; s3 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s3; s3 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s0; s3 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s1; s3 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s2; s3 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b2.s3; s3 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b3.s0; s3 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b3.s1; s3 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b3.s2; s3 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b3.s3; s3 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b4.s0; s3 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b4.s1; s3 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b4.s2; s3 += l1 ? fabs(d) : d * d;
        }
        if (side > 4) {
            d = a0.s0 - b1.s0; s4 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s1; s4 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s2; s4 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s3; s4 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s0; s4 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s1; s4 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s2; s4 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s3; s4 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b3.s0; s4 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b3.s1; s4 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b3.s2; s4 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b3.s3; s4 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b4.s0; s4 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b4.s1; s4 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b4.s2; s4 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b4.s3; s4 += l1 ? fabs(d) : d * d;
        }
        if (side > 5) {
            d = a0.s0 - b1.s1; s5 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s2; s5 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s3; s5 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s0; s5 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s1; s5 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s2; s5 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s3; s5 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s0; s5 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b3.s1; s5 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b3.s2; s5 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b3.s3; s5 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b4.s0; s5 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b4.s1; s5 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b4.s2; s5 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b4.s3; s5 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b5.s0; s5 += l1 ? fabs(d) : d * d;
        }
        if (side > 6) {
            d = a0.s0 - b1.s2; s6 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s3; s6 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b2.s0; s6 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s1; s6 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s2; s6 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s3; s6 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b3.s0; s6 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s1; s6 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b3.s2; s6 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b3.s3; s6 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b4.s0; s6 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b4.s1; s6 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b4.s2; s6 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b4.s3; s6 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b5.s0; s6 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b5.s1; s6 += l1 ? fabs(d) : d * d;
        }
        if (side > 7) {
            d = a0.s0 - b1.s3; s7 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b2.s0; s7 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b2.s1; s7 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s2; s7 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s3; s7 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b3.s0; s7 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b3.s1; s7 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s2; s7 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b3.s3; s7 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b4.s0; s7 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b4.s1; s7 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b4.s2; s7 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b4.s3; s7 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b5.s0; s7 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b5.s1; s7 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b5.s2; s7 += l1 ? fabs(d) : d * d;
        }
        if (side > 8) {
            d = a0.s0 - b2.s0; s8 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b2.s1; s8 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b2.s2; s8 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s3; s8 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b3.s0; s8 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b3.s1; s8 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b3.s2; s8 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s3; s8 += l1 ? fabs(d) : d * d;
            d = a2.s0 - b4.s0; s8 += l1 ? fabs(d) : d * d;
            d = a2.s1 - b4.s1; s8 += l1 ? fabs(d) : d * d;
            d = a2.s2 - b4.s2; s8 += l1 ? fabs(d) : d * d;
            d = a2.s3 - b4.s3; s8 += l1 ? fabs(d) : d * d;
            d = a3.s0 - b5.s0; s8 += l1 ? fabs(d) : d * d;
            d = a3.s1 - b5.s1; s8 += l1 ? fabs(d) : d * d;
            d = a3.s2 - b5.s2; s8 += l1 ? fabs(d) : d * d;
            d = a3.s3 - b5.s3; s8 += l1 ? fabs(d) : d * d;
        }
    }
    __global float* o = scost + (size_t)t * side * side + oy * side;
    if (side > 0) o[0] = s0;
    if (side > 1) o[1] = s1;
    if (side > 2) o[2] = s2;
    if (side > 3) o[3] = s3;
    if (side > 4) o[4] = s4;
    if (side > 5) o[5] = s5;
    if (side > 6) o[6] = s6;
    if (side > 7) o[7] = s7;
    if (side > 8) o[8] = s8;
}

__kernel void align_cand8(__global const float* ref, __global const float* alt, int w, int h, int nx, int ny, int l1,
                          __global const short* pdx, __global const short* pdy, int pnx, int pny, int ptile, int f,
                          __global float* ccost) {
    int g = get_global_id(0), j = get_global_id(1), i = g / 3, c = g - i * 3;
    if (i >= nx || j >= ny) return;
    if (c >= (f > 0 ? 3 : 1)) return;
    int x0 = i * 4, y0 = j * 4;
    int cand[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    if (f > 0) parent_cands(x0, y0, 8, pdx, pdy, pnx, pny, ptile, f, cand);
    int bx = x0 + cand[c][0], by = y0 + cand[c][1];
    float4 a0, a1, b0, b1;
    float s = 0, d;
    for (int v = 0; v < 8; v++) {
        if (x0 >= 0 && x0 + 8 <= w && y0 + v >= 0 && y0 + v < h) {
            __global const float* r_ = ref + (y0 + v) * w + x0;
            a0 = vload4(0, r_);
            a1 = vload4(1, r_);
        } else {
            a0 = (float4)(at_clamped(ref, w, h, x0 + 0, y0 + v), at_clamped(ref, w, h, x0 + 1, y0 + v), at_clamped(ref, w, h, x0 + 2, y0 + v), at_clamped(ref, w, h, x0 + 3, y0 + v));
            a1 = (float4)(at_clamped(ref, w, h, x0 + 4, y0 + v), at_clamped(ref, w, h, x0 + 5, y0 + v), at_clamped(ref, w, h, x0 + 6, y0 + v), at_clamped(ref, w, h, x0 + 7, y0 + v));
        }
        if (bx >= 0 && bx + 8 <= w && by + v >= 0 && by + v < h) {
            __global const float* r_ = alt + (by + v) * w + bx;
            b0 = vload4(0, r_);
            b1 = vload4(1, r_);
        } else {
            b0 = (float4)(at_clamped(alt, w, h, bx + 0, by + v), at_clamped(alt, w, h, bx + 1, by + v), at_clamped(alt, w, h, bx + 2, by + v), at_clamped(alt, w, h, bx + 3, by + v));
            b1 = (float4)(at_clamped(alt, w, h, bx + 4, by + v), at_clamped(alt, w, h, bx + 5, by + v), at_clamped(alt, w, h, bx + 6, by + v), at_clamped(alt, w, h, bx + 7, by + v));
        }
        d = a0.s0 - b0.s0; s += l1 ? fabs(d) : d * d;
        d = a0.s1 - b0.s1; s += l1 ? fabs(d) : d * d;
        d = a0.s2 - b0.s2; s += l1 ? fabs(d) : d * d;
        d = a0.s3 - b0.s3; s += l1 ? fabs(d) : d * d;
        d = a1.s0 - b1.s0; s += l1 ? fabs(d) : d * d;
        d = a1.s1 - b1.s1; s += l1 ? fabs(d) : d * d;
        d = a1.s2 - b1.s2; s += l1 ? fabs(d) : d * d;
        d = a1.s3 - b1.s3; s += l1 ? fabs(d) : d * d;
    }
    ccost[(j * nx + i) * 3 + c] = s;
}
__kernel void align_search8(__global const float* ref, __global const float* alt, int w, int h, int nx, int ny, int R,
                            int l1, __global const short* pdx, __global const short* pdy, int pnx, int pny, int ptile,
                            int f, __global const float* ccost, __global float* scost) {
    const int side = 2 * R + 1;
    int g = get_global_id(0), j = get_global_id(1), i = g / side, oy = g - i * side;
    if (i >= nx || j >= ny) return;
    int x0 = i * 4, y0 = j * 4, t = j * nx + i;
    int cand[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    if (f > 0) parent_cands(x0, y0, 8, pdx, pdy, pnx, pny, ptile, f, cand);
    float best0;
    int bc = best_cand(ccost, t, f > 0 ? 3 : 1, &best0);
    int bx = x0 + cand[bc][0] - R, by = y0 + cand[bc][1] - R + oy;
    float4 a0, a1, b0, b1, b2, b3;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0, d;
    for (int v = 0; v < 8; v++) {
        if (x0 >= 0 && x0 + 8 <= w && y0 + v >= 0 && y0 + v < h) {
            __global const float* r_ = ref + (y0 + v) * w + x0;
            a0 = vload4(0, r_);
            a1 = vload4(1, r_);
        } else {
            a0 = (float4)(at_clamped(ref, w, h, x0 + 0, y0 + v), at_clamped(ref, w, h, x0 + 1, y0 + v), at_clamped(ref, w, h, x0 + 2, y0 + v), at_clamped(ref, w, h, x0 + 3, y0 + v));
            a1 = (float4)(at_clamped(ref, w, h, x0 + 4, y0 + v), at_clamped(ref, w, h, x0 + 5, y0 + v), at_clamped(ref, w, h, x0 + 6, y0 + v), at_clamped(ref, w, h, x0 + 7, y0 + v));
        }
        if (bx >= 0 && bx + 16 <= w && by + v >= 0 && by + v < h) {
            __global const float* r_ = alt + (by + v) * w + bx;
            b0 = vload4(0, r_);
            b1 = vload4(1, r_);
            b2 = vload4(2, r_);
            b3 = vload4(3, r_);
        } else {
            b0 = (float4)(at_clamped(alt, w, h, bx + 0, by + v), at_clamped(alt, w, h, bx + 1, by + v), at_clamped(alt, w, h, bx + 2, by + v), at_clamped(alt, w, h, bx + 3, by + v));
            b1 = (float4)(at_clamped(alt, w, h, bx + 4, by + v), at_clamped(alt, w, h, bx + 5, by + v), at_clamped(alt, w, h, bx + 6, by + v), at_clamped(alt, w, h, bx + 7, by + v));
            b2 = (float4)(at_clamped(alt, w, h, bx + 8, by + v), at_clamped(alt, w, h, bx + 9, by + v), at_clamped(alt, w, h, bx + 10, by + v), at_clamped(alt, w, h, bx + 11, by + v));
            b3 = (float4)(at_clamped(alt, w, h, bx + 12, by + v), at_clamped(alt, w, h, bx + 13, by + v), at_clamped(alt, w, h, bx + 14, by + v), at_clamped(alt, w, h, bx + 15, by + v));
        }
        if (side > 0) {
            d = a0.s0 - b0.s0; s0 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b0.s1; s0 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b0.s2; s0 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b0.s3; s0 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s0; s0 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b1.s1; s0 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b1.s2; s0 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b1.s3; s0 += l1 ? fabs(d) : d * d;
        }
        if (side > 1) {
            d = a0.s0 - b0.s1; s1 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b0.s2; s1 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b0.s3; s1 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s0; s1 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s1; s1 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b1.s2; s1 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b1.s3; s1 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s0; s1 += l1 ? fabs(d) : d * d;
        }
        if (side > 2) {
            d = a0.s0 - b0.s2; s2 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b0.s3; s2 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s0; s2 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s1; s2 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s2; s2 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b1.s3; s2 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s0; s2 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s1; s2 += l1 ? fabs(d) : d * d;
        }
        if (side > 3) {
            d = a0.s0 - b0.s3; s3 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s0; s3 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s1; s3 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s2; s3 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b1.s3; s3 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s0; s3 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s1; s3 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s2; s3 += l1 ? fabs(d) : d * d;
        }
        if (side > 4) {
            d = a0.s0 - b1.s0; s4 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s1; s4 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s2; s4 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b1.s3; s4 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s0; s4 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s1; s4 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s2; s4 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b2.s3; s4 += l1 ? fabs(d) : d * d;
        }
        if (side > 5) {
            d = a0.s0 - b1.s1; s5 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s2; s5 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b1.s3; s5 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s0; s5 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s1; s5 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s2; s5 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b2.s3; s5 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s0; s5 += l1 ? fabs(d) : d * d;
        }
        if (side > 6) {
            d = a0.s0 - b1.s2; s6 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b1.s3; s6 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b2.s0; s6 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s1; s6 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s2; s6 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b2.s3; s6 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b3.s0; s6 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s1; s6 += l1 ? fabs(d) : d * d;
        }
        if (side > 7) {
            d = a0.s0 - b1.s3; s7 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b2.s0; s7 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b2.s1; s7 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s2; s7 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b2.s3; s7 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b3.s0; s7 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b3.s1; s7 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s2; s7 += l1 ? fabs(d) : d * d;
        }
        if (side > 8) {
            d = a0.s0 - b2.s0; s8 += l1 ? fabs(d) : d * d;
            d = a0.s1 - b2.s1; s8 += l1 ? fabs(d) : d * d;
            d = a0.s2 - b2.s2; s8 += l1 ? fabs(d) : d * d;
            d = a0.s3 - b2.s3; s8 += l1 ? fabs(d) : d * d;
            d = a1.s0 - b3.s0; s8 += l1 ? fabs(d) : d * d;
            d = a1.s1 - b3.s1; s8 += l1 ? fabs(d) : d * d;
            d = a1.s2 - b3.s2; s8 += l1 ? fabs(d) : d * d;
            d = a1.s3 - b3.s3; s8 += l1 ? fabs(d) : d * d;
        }
    }
    __global float* o = scost + (size_t)t * side * side + oy * side;
    if (side > 0) o[0] = s0;
    if (side > 1) o[1] = s1;
    if (side > 2) o[2] = s2;
    if (side > 3) o[3] = s3;
    if (side > 4) o[4] = s4;
    if (side > 5) o[5] = s5;
    if (side > 6) o[6] = s6;
    if (side > 7) o[7] = s7;
    if (side > 8) o[8] = s8;
}

__kernel void align_pick(int tile, int nx, int ny, int R, __global const short* pdx, __global const short* pdy, int pnx,
                         int pny, int ptile, int f, __global const float* ccost, __global const float* scost,
                         __global short* dx, __global short* dy) {
    int i = get_global_id(0), j = get_global_id(1);
    if (i >= nx || j >= ny) return;
    int x0 = i * (tile / 2), y0 = j * (tile / 2), t = j * nx + i, side = 2 * R + 1;
    int cand[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    if (f > 0) parent_cands(x0, y0, tile, pdx, pdy, pnx, pny, ptile, f, cand);
    float best;
    int bc = best_cand(ccost, t, f > 0 ? 3 : 1, &best);
    int cx = cand[bc][0], cy = cand[bc][1], bx = cx, by = cy;
    __global const float* sc = scost + (size_t)t * side * side;
    for (int oy = -R; oy <= R; oy++)
        for (int ox = -R; ox <= R; ox++) {
            if (!ox && !oy) continue;
            float s = sc[(oy + R) * side + ox + R];
            if (s < best) { best = s; bx = cx + ox; by = cy + oy; }
        }
    dx[t] = (short)bx;
    dy[t] = (short)by;
}

__kernel void align_subpixel_cost(__global const float* refM3, int refOff, __global const float* altM3, int altOff,
                                  int w, int h, int nx, int ny, __global const short* dx, __global const short* dy,
                                  __global float* sc3) {
    int g = get_global_id(0), j = get_global_id(1), i = g / 3, oy = (g - i * 3) - 1;
    if (i >= nx || j >= ny) return;
    __global const float* ref = refM3 + refOff;
    __global const float* alt = altM3 + altOff;
    int t = j * nx + i;
    int x0 = i * 8, y0 = j * 8;
    int bx = x0 + (int)dx[t] - 1, by = y0 + (int)dy[t] + oy;
    float s0 = 0, s1 = 0, s2 = 0;
    float4 a0, a1, a2, a3, b0, b1, b2, b3;
    float b16, b17;
    for (int v = 0; v < 16; v++) {
        if (x0 >= 0 && x0 + 16 <= w && y0 + v >= 0 && y0 + v < h) {
            __global const float* rp = ref + (y0 + v) * w + x0;
            a0 = vload4(0, rp); a1 = vload4(1, rp); a2 = vload4(2, rp); a3 = vload4(3, rp);
        } else {
            int yy = clamp(y0 + v, 0, h - 1);
            a0 = (float4)(at_clamped(ref, w, h, x0 + 0, yy), at_clamped(ref, w, h, x0 + 1, yy), at_clamped(ref, w, h, x0 + 2, yy), at_clamped(ref, w, h, x0 + 3, yy));
            a1 = (float4)(at_clamped(ref, w, h, x0 + 4, yy), at_clamped(ref, w, h, x0 + 5, yy), at_clamped(ref, w, h, x0 + 6, yy), at_clamped(ref, w, h, x0 + 7, yy));
            a2 = (float4)(at_clamped(ref, w, h, x0 + 8, yy), at_clamped(ref, w, h, x0 + 9, yy), at_clamped(ref, w, h, x0 + 10, yy), at_clamped(ref, w, h, x0 + 11, yy));
            a3 = (float4)(at_clamped(ref, w, h, x0 + 12, yy), at_clamped(ref, w, h, x0 + 13, yy), at_clamped(ref, w, h, x0 + 14, yy), at_clamped(ref, w, h, x0 + 15, yy));
        }
        if (bx >= 0 && bx + 18 <= w && by + v >= 0 && by + v < h) {
            __global const float* ap = alt + (by + v) * w + bx;
            b0 = vload4(0, ap); b1 = vload4(1, ap); b2 = vload4(2, ap); b3 = vload4(3, ap);
            b16 = ap[16]; b17 = ap[17];
        } else {
            int yy = clamp(by + v, 0, h - 1);
            b0 = (float4)(at_clamped(alt, w, h, bx + 0, yy), at_clamped(alt, w, h, bx + 1, yy), at_clamped(alt, w, h, bx + 2, yy), at_clamped(alt, w, h, bx + 3, yy));
            b1 = (float4)(at_clamped(alt, w, h, bx + 4, yy), at_clamped(alt, w, h, bx + 5, yy), at_clamped(alt, w, h, bx + 6, yy), at_clamped(alt, w, h, bx + 7, yy));
            b2 = (float4)(at_clamped(alt, w, h, bx + 8, yy), at_clamped(alt, w, h, bx + 9, yy), at_clamped(alt, w, h, bx + 10, yy), at_clamped(alt, w, h, bx + 11, yy));
            b3 = (float4)(at_clamped(alt, w, h, bx + 12, yy), at_clamped(alt, w, h, bx + 13, yy), at_clamped(alt, w, h, bx + 14, yy), at_clamped(alt, w, h, bx + 15, yy));
            b16 = at_clamped(alt, w, h, bx + 16, yy);
            b17 = at_clamped(alt, w, h, bx + 17, yy);
        }
        float4 d0 = a0 - b0, d1 = a1 - b1, d2 = a2 - b2, d3 = a3 - b3;
        s0 += dot(d0, d0) + dot(d1, d1) + dot(d2, d2) + dot(d3, d3);
        d0 = a0 - (float4)(b0.s1, b0.s2, b0.s3, b1.s0);
        d1 = a1 - (float4)(b1.s1, b1.s2, b1.s3, b2.s0);
        d2 = a2 - (float4)(b2.s1, b2.s2, b2.s3, b3.s0);
        d3 = a3 - (float4)(b3.s1, b3.s2, b3.s3, b16);
        s1 += dot(d0, d0) + dot(d1, d1) + dot(d2, d2) + dot(d3, d3);
        d0 = a0 - (float4)(b0.s2, b0.s3, b1.s0, b1.s1);
        d1 = a1 - (float4)(b1.s2, b1.s3, b2.s0, b2.s1);
        d2 = a2 - (float4)(b2.s2, b2.s3, b3.s0, b3.s1);
        d3 = a3 - (float4)(b3.s2, b3.s3, b16, b17);
        s2 += dot(d0, d0) + dot(d1, d1) + dot(d2, d2) + dot(d3, d3);
    }
    __global float* o = sc3 + ((size_t)t * 3 + (oy + 1)) * 3;
    o[0] = s0; o[1] = s1; o[2] = s2;
}

__kernel void align_subpixel_fit(int nx, int ny, __global const float* sc3, __global float* fx, __global float* fy) {
    int i = get_global_id(0), j = get_global_id(1);
    if (i >= nx || j >= ny) return;
    int t = j * nx + i;
    __global const float* c = sc3 + (size_t)t * 9;
    float c00 = c[0], c01 = c[1], c02 = c[2];
    float c10 = c[3], c11 = c[4], c12 = c[5];
    float c20 = c[6], c21 = c[7], c22 = c[8];
    float gx = ((c02 - c00) + (c12 - c10) + (c22 - c20)) * (1.f / 6.f);
    float gy = ((c20 - c00) + (c21 - c01) + (c22 - c02)) * (1.f / 6.f);
    float hxx = ((c02 - 2.f * c01 + c00) + (c12 - 2.f * c11 + c10) + (c22 - 2.f * c21 + c20)) * (1.f / 3.f);
    float hyy = ((c20 - 2.f * c10 + c00) + (c21 - 2.f * c11 + c01) + (c22 - 2.f * c12 + c02)) * (1.f / 3.f);
    float hxy = (c22 - c20 - c02 + c00) * 0.25f;
    float tr = hxx + hyy;
    if (tr <= 0.f) { fx[t] = 0.f; fy[t] = 0.f; return; }
    float det = hxx * hyy - hxy * hxy;
    float disc = sqrt(max(0.f, 0.25f * tr * tr - det));
    float l1 = 0.5f * tr + disc, l2 = 0.5f * tr - disc;
    float cmin = max(c11, 1e-6f);
    if (l1 <= 0.8f * cmin) { fx[t] = 0.f; fy[t] = 0.f; return; }
    float e1x = 1.f, e1y = 0.f;
    if (fabs(hxy) > 1e-12f * tr) {
        if (hxx >= hyy) { e1x = l1 - hyy; e1y = hxy; }
        else { e1x = hxy; e1y = l1 - hxx; }
        float nrm = hypot(e1x, e1y);
        e1x /= nrm; e1y /= nrm;
    } else if (hyy > hxx) {
        e1x = 0.f; e1y = 1.f;
    }
    float e2x = -e1y, e2y = e1x;
    float g1 = gx * e1x + gy * e1y, g2 = gx * e2x + gy * e2y;
    float mu = 0.02f * cmin;
    float s1 = -g1 * l1 / (l1 * l1 + mu * mu);
    float s2 = (l2 > max(0.05f * l1, 0.8f * cmin)) ? (-g2 * l2 / (l2 * l2 + mu * mu)) : 0.f;
    float ox = clamp(s1 * e1x + s2 * e2x, -0.5f, 0.5f);
    float oy = clamp(s1 * e1y + s2 * e2y, -0.5f, 0.5f);
    float dlx = c10 - c11, drx = c12 - c11;
    float dly = c01 - c11, dry = c21 - c11;
    if (dlx <= 0.f || drx <= 0.f || max(dlx, drx) > 12.f * min(dlx, drx)) ox = 0.f;
    if (dly <= 0.f || dry <= 0.f || max(dly, dry) > 12.f * min(dly, dry)) oy = 0.f;
    fx[t] = ox; fy[t] = oy;
}
)CL"
