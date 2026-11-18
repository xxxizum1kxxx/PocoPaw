R"CL(
// align.cpp on the device: fused gray + level-1 downsample, local-tile 3x3 mean, and zero-spill tile search.

inline float raw_quad_sum(__global const ushort* raw, int off, int rw, int x, int y) {
    __global const ushort* a = raw + off + (size_t)(2 * y) * rw + 2 * x;
    __global const ushort* b = a + rw;
    ushort2 r0 = vload2(0, a);
    ushort2 r1 = vload2(0, b);
    return (float)r0.x + (float)r0.y + (float)r1.x + (float)r1.y;
}

// Fuses gray_from_raw (level 0: gw x gh) and 2x2 downsample (level 1: pw1 x ph1) in a single RAW read pass.
__kernel void gray_down2(__global const ushort* raw, int off, int rw, __global float* g0, int gw, int gh,
                         __global float* g1, int pw1, int ph1) {
    int x2 = get_global_id(0), y2 = get_global_id(1);
    int x = 2 * x2, y = 2 * y2;
    if (x >= gw || y >= gh) return;
    float v00 = raw_quad_sum(raw, off, rw, x, y);
    g0[y * gw + x] = v00;
    float v10 = 0.f, v01 = 0.f, v11 = 0.f;
    if (x + 1 < gw) {
        v10 = raw_quad_sum(raw, off, rw, x + 1, y);
        g0[y * gw + x + 1] = v10;
    }
    if (y + 1 < gh) {
        v01 = raw_quad_sum(raw, off, rw, x, y + 1);
        g0[(y + 1) * gw + x] = v01;
        if (x + 1 < gw) {
            v11 = raw_quad_sum(raw, off, rw, x + 1, y + 1);
            g0[(y + 1) * gw + x + 1] = v11;
        }
    }
    if (x2 < pw1 && y2 < ph1) {
        g1[y2 * pw1 + x2] = 0.25f * (v00 + v10 + v01 + v11);
    }
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

// Single-pass 3x3 box mean in local memory on a 16x8 Wave128 work-group (eliminates intermediate qw*qh DRAM pass).
__kernel void box3_3x3(__global const float* g, __global float* o, int ooff, int w, int h) {
    __local float tile[10][18];
    int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    int gx0 = (int)get_group_id(0) * 16 - 1, gy0 = (int)get_group_id(1) * 8 - 1;
    for (int idx = tid; idx < 180; idx += 128) {
        int ty = idx / 18, tx = idx - ty * 18;
        tile[ty][tx] = g[clamp(gy0 + ty, 0, h - 1) * w + clamp(gx0 + tx, 0, w - 1)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int x = (int)get_group_id(0) * 16 + lx, y = (int)get_group_id(1) * 8 + ly;
    if (x >= w || y >= h) return;
    float r0 = tile[ly][lx] + tile[ly][lx + 1] + tile[ly][lx + 2];
    float r1 = tile[ly + 1][lx] + tile[ly + 1][lx + 1] + tile[ly + 1][lx + 2];
    float r2 = tile[ly + 2][lx] + tile[ly + 2][lx + 1] + tile[ly + 2][lx + 2];
    o[ooff + y * w + x] = (r0 + r1 + r2) * (1.f / 9.f);
}

// Scalar register candidate selection (zero runtime-indexed private arrays on Adreno).
inline void parent_cands(int x0, int y0, int tile, __global const short* pdx, __global const short* pdy, int pnx, int pny,
                         int ptile, int f, int2* c0, int2* c1, int2* c2) {
    int pstride = ptile / 2;
    float cx = (x0 + tile * 0.5f) / f, cy = (y0 + tile * 0.5f) / f;
    float fx = (cx - ptile * 0.5f) / pstride, fy = (cy - ptile * 0.5f) / pstride;
    int pi = clamp((int)round(fx), 0, pnx - 1), pj = clamp((int)round(fy), 0, pny - 1);
    int ni = clamp(fx > pi ? pi + 1 : pi - 1, 0, pnx - 1), nj = clamp(fy > pj ? pj + 1 : pj - 1, 0, pny - 1);
    int k0 = pj * pnx + pi, k1 = pj * pnx + ni, k2 = nj * pnx + pi;
    *c0 = (int2)((int)pdx[k0] * f, (int)pdy[k0] * f);
    *c1 = (int2)((int)pdx[k1] * f, (int)pdy[k1] * f);
    *c2 = (int2)((int)pdx[k2] * f, (int)pdy[k2] * f);
}

inline int best_cand(__global const float* ccost, int t, int nc, float* best) {
    int b = 0;
    float bs = ccost[t * 3];
    if (nc > 1) {
        float c1 = ccost[t * 3 + 1];
        if (c1 < bs) { bs = c1; b = 1; }
        float c2 = ccost[t * 3 + 2];
        if (c2 < bs) { bs = c2; b = 2; }
    }
    *best = bs;
    return b;
}

inline float sum_abs4(float4 d) {
    float4 ad = fabs(d);
    return (ad.x + ad.y) + (ad.z + ad.w);
}

__kernel void align_cand16(__global const float* ref, __global const float* alt, int w, int h, int nx, int ny, int l1,
                           __global const short* pdx, __global const short* pdy, int pnx, int pny, int ptile, int f,
                           __global float* ccost) {
    int g = get_global_id(0), j = get_global_id(1), i = g / 3, c = g - i * 3;
    if (i >= nx || j >= ny) return;
    if (c >= (f > 0 ? 3 : 1)) return;
    int x0 = i * 8, y0 = j * 8;
    int2 c0 = (int2)(0, 0), c1 = (int2)(0, 0), c2 = (int2)(0, 0);
    if (f > 0) parent_cands(x0, y0, 16, pdx, pdy, pnx, pny, ptile, f, &c0, &c1, &c2);
    int2 chosen = (c == 0) ? c0 : ((c == 1) ? c1 : c2);
    int bx = x0 + chosen.x, by = y0 + chosen.y;
    bool refIn = (x0 >= 0 && x0 + 16 <= w && y0 >= 0 && y0 + 16 <= h);
    bool altIn = (bx >= 0 && bx + 16 <= w && by >= 0 && by + 16 <= h);
    float4 a0, a1, a2, a3, b0, b1, b2, b3;
    float s = 0;
    for (int v = 0; v < 16; v++) {
        if (refIn) {
            __global const float* r_ = ref + (y0 + v) * w + x0;
            a0 = vload4(0, r_); a1 = vload4(1, r_); a2 = vload4(2, r_); a3 = vload4(3, r_);
        } else {
            a0 = (float4)(at_clamped(ref, w, h, x0 + 0, y0 + v), at_clamped(ref, w, h, x0 + 1, y0 + v), at_clamped(ref, w, h, x0 + 2, y0 + v), at_clamped(ref, w, h, x0 + 3, y0 + v));
            a1 = (float4)(at_clamped(ref, w, h, x0 + 4, y0 + v), at_clamped(ref, w, h, x0 + 5, y0 + v), at_clamped(ref, w, h, x0 + 6, y0 + v), at_clamped(ref, w, h, x0 + 7, y0 + v));
            a2 = (float4)(at_clamped(ref, w, h, x0 + 8, y0 + v), at_clamped(ref, w, h, x0 + 9, y0 + v), at_clamped(ref, w, h, x0 + 10, y0 + v), at_clamped(ref, w, h, x0 + 11, y0 + v));
            a3 = (float4)(at_clamped(ref, w, h, x0 + 12, y0 + v), at_clamped(ref, w, h, x0 + 13, y0 + v), at_clamped(ref, w, h, x0 + 14, y0 + v), at_clamped(ref, w, h, x0 + 15, y0 + v));
        }
        if (altIn) {
            __global const float* r_ = alt + (by + v) * w + bx;
            b0 = vload4(0, r_); b1 = vload4(1, r_); b2 = vload4(2, r_); b3 = vload4(3, r_);
        } else {
            b0 = (float4)(at_clamped(alt, w, h, bx + 0, by + v), at_clamped(alt, w, h, bx + 1, by + v), at_clamped(alt, w, h, bx + 2, by + v), at_clamped(alt, w, h, bx + 3, by + v));
            b1 = (float4)(at_clamped(alt, w, h, bx + 4, by + v), at_clamped(alt, w, h, bx + 5, by + v), at_clamped(alt, w, h, bx + 6, by + v), at_clamped(alt, w, h, bx + 7, by + v));
            b2 = (float4)(at_clamped(alt, w, h, bx + 8, by + v), at_clamped(alt, w, h, bx + 9, by + v), at_clamped(alt, w, h, bx + 10, by + v), at_clamped(alt, w, h, bx + 11, by + v));
            b3 = (float4)(at_clamped(alt, w, h, bx + 12, by + v), at_clamped(alt, w, h, bx + 13, by + v), at_clamped(alt, w, h, bx + 14, by + v), at_clamped(alt, w, h, bx + 15, by + v));
        }
        float4 d0 = a0 - b0, d1 = a1 - b1, d2 = a2 - b2, d3 = a3 - b3;
        if (l1) s += sum_abs4(d0) + sum_abs4(d1) + sum_abs4(d2) + sum_abs4(d3);
        else s += dot(d0, d0) + dot(d1, d1) + dot(d2, d2) + dot(d3, d3);
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
    int2 c0 = (int2)(0, 0), c1 = (int2)(0, 0), c2 = (int2)(0, 0);
    if (f > 0) parent_cands(x0, y0, 16, pdx, pdy, pnx, pny, ptile, f, &c0, &c1, &c2);
    float best0;
    int bc = best_cand(ccost, t, f > 0 ? 3 : 1, &best0);
    int2 chosen = (bc == 0) ? c0 : ((bc == 1) ? c1 : c2);
    int bx = x0 + chosen.x - R, by = y0 + chosen.y - R + oy;
    bool refIn = (x0 >= 0 && x0 + 16 <= w && y0 >= 0 && y0 + 16 <= h);
    bool altIn = (bx >= 0 && bx + (15 + side) <= w && by >= 0 && by + 16 <= h);
    float4 a0, a1, a2, a3, b0, b1, b2, b3, b4 = 0, b5 = 0;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0;
    for (int v = 0; v < 16; v++) {
        if (refIn) {
            __global const float* r_ = ref + (y0 + v) * w + x0;
            a0 = vload4(0, r_); a1 = vload4(1, r_); a2 = vload4(2, r_); a3 = vload4(3, r_);
        } else {
            a0 = (float4)(at_clamped(ref, w, h, x0 + 0, y0 + v), at_clamped(ref, w, h, x0 + 1, y0 + v), at_clamped(ref, w, h, x0 + 2, y0 + v), at_clamped(ref, w, h, x0 + 3, y0 + v));
            a1 = (float4)(at_clamped(ref, w, h, x0 + 4, y0 + v), at_clamped(ref, w, h, x0 + 5, y0 + v), at_clamped(ref, w, h, x0 + 6, y0 + v), at_clamped(ref, w, h, x0 + 7, y0 + v));
            a2 = (float4)(at_clamped(ref, w, h, x0 + 8, y0 + v), at_clamped(ref, w, h, x0 + 9, y0 + v), at_clamped(ref, w, h, x0 + 10, y0 + v), at_clamped(ref, w, h, x0 + 11, y0 + v));
            a3 = (float4)(at_clamped(ref, w, h, x0 + 12, y0 + v), at_clamped(ref, w, h, x0 + 13, y0 + v), at_clamped(ref, w, h, x0 + 14, y0 + v), at_clamped(ref, w, h, x0 + 15, y0 + v));
        }
        if (altIn) {
            __global const float* r_ = alt + (by + v) * w + bx;
            b0 = vload4(0, r_); b1 = vload4(1, r_); b2 = vload4(2, r_); b3 = vload4(3, r_);
            if (side > 1) b4 = vload4(4, r_);
            if (side > 5) b5 = vload4(5, r_);
        } else {
            b0 = (float4)(at_clamped(alt, w, h, bx + 0, by + v), at_clamped(alt, w, h, bx + 1, by + v), at_clamped(alt, w, h, bx + 2, by + v), at_clamped(alt, w, h, bx + 3, by + v));
            b1 = (float4)(at_clamped(alt, w, h, bx + 4, by + v), at_clamped(alt, w, h, bx + 5, by + v), at_clamped(alt, w, h, bx + 6, by + v), at_clamped(alt, w, h, bx + 7, by + v));
            b2 = (float4)(at_clamped(alt, w, h, bx + 8, by + v), at_clamped(alt, w, h, bx + 9, by + v), at_clamped(alt, w, h, bx + 10, by + v), at_clamped(alt, w, h, bx + 11, by + v));
            b3 = (float4)(at_clamped(alt, w, h, bx + 12, by + v), at_clamped(alt, w, h, bx + 13, by + v), at_clamped(alt, w, h, bx + 14, by + v), at_clamped(alt, w, h, bx + 15, by + v));
            if (side > 1) b4 = (float4)(at_clamped(alt, w, h, bx + 16, by + v), at_clamped(alt, w, h, bx + 17, by + v), at_clamped(alt, w, h, bx + 18, by + v), at_clamped(alt, w, h, bx + 19, by + v));
            if (side > 5) b5 = (float4)(at_clamped(alt, w, h, bx + 20, by + v), at_clamped(alt, w, h, bx + 21, by + v), at_clamped(alt, w, h, bx + 22, by + v), at_clamped(alt, w, h, bx + 23, by + v));
        }
        #define COST4(u0, u1, u2, u3) (l1 ? (sum_abs4(a0 - (u0)) + sum_abs4(a1 - (u1)) + sum_abs4(a2 - (u2)) + sum_abs4(a3 - (u3))) \
                                          : (dot(a0 - (u0), a0 - (u0)) + dot(a1 - (u1), a1 - (u1)) + dot(a2 - (u2), a2 - (u2)) + dot(a3 - (u3), a3 - (u3))))
        if (side > 0) s0 += COST4(b0, b1, b2, b3);
        if (side > 1) s1 += COST4((float4)(b0.s123, b1.s0), (float4)(b1.s123, b2.s0), (float4)(b2.s123, b3.s0), (float4)(b3.s123, b4.s0));
        if (side > 2) s2 += COST4((float4)(b0.s23, b1.s01), (float4)(b1.s23, b2.s01), (float4)(b2.s23, b3.s01), (float4)(b3.s23, b4.s01));
        if (side > 3) s3 += COST4((float4)(b0.s3, b1.s012), (float4)(b1.s3, b2.s012), (float4)(b2.s3, b3.s012), (float4)(b3.s3, b4.s012));
        if (side > 4) s4 += COST4(b1, b2, b3, b4);
        if (side > 5) s5 += COST4((float4)(b1.s123, b2.s0), (float4)(b2.s123, b3.s0), (float4)(b3.s123, b4.s0), (float4)(b4.s123, b5.s0));
        if (side > 6) s6 += COST4((float4)(b1.s23, b2.s01), (float4)(b2.s23, b3.s01), (float4)(b3.s23, b4.s01), (float4)(b4.s23, b5.s01));
        if (side > 7) s7 += COST4((float4)(b1.s3, b2.s012), (float4)(b2.s3, b3.s012), (float4)(b3.s3, b4.s012), (float4)(b4.s3, b5.s012));
        if (side > 8) s8 += COST4(b2, b3, b4, b5);
        #undef COST4
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
    int2 c0 = (int2)(0, 0), c1 = (int2)(0, 0), c2 = (int2)(0, 0);
    if (f > 0) parent_cands(x0, y0, 8, pdx, pdy, pnx, pny, ptile, f, &c0, &c1, &c2);
    int2 chosen = (c == 0) ? c0 : ((c == 1) ? c1 : c2);
    int bx = x0 + chosen.x, by = y0 + chosen.y;
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
    int2 c0 = (int2)(0, 0), c1 = (int2)(0, 0), c2 = (int2)(0, 0);
    if (f > 0) parent_cands(x0, y0, 8, pdx, pdy, pnx, pny, ptile, f, &c0, &c1, &c2);
    float best0;
    int bc = best_cand(ccost, t, f > 0 ? 3 : 1, &best0);
    int2 chosen = (bc == 0) ? c0 : ((bc == 1) ? c1 : c2);
    int bx = x0 + chosen.x - R, by = y0 + chosen.y - R + oy;
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
                         __global short* dx, __global short* dy, int outOff) {
    int i = get_global_id(0), j = get_global_id(1);
    if (i >= nx || j >= ny) return;
    int x0 = i * (tile / 2), y0 = j * (tile / 2), t = j * nx + i, side = 2 * R + 1;
    int2 c0 = (int2)(0, 0), c1 = (int2)(0, 0), c2 = (int2)(0, 0);
    if (f > 0) parent_cands(x0, y0, tile, pdx, pdy, pnx, pny, ptile, f, &c0, &c1, &c2);
    float best;
    int bc = best_cand(ccost, t, f > 0 ? 3 : 1, &best);
    int2 chosen = (bc == 0) ? c0 : ((bc == 1) ? c1 : c2);
    int cx = chosen.x, cy = chosen.y, bx = cx, by = cy;
    __global const float* sc = scost + (size_t)t * side * side;
    for (int oy = -R; oy <= R; oy++)
        for (int ox = -R; ox <= R; ox++) {
            if (!ox && !oy) continue;
            float s = sc[(oy + R) * side + ox + R];
            if (s < best) { best = s; bx = cx + ox; by = cy + oy; }
        }
    dx[outOff + t] = (short)bx;
    dy[outOff + t] = (short)by;
}
)CL"