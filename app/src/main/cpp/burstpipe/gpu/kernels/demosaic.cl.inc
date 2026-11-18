R"CL(
// RCD demosaicing (Luis Sanz Rodriguez, librtprocess rcd.cc, GPL-3) packed in coalesced FP16 vectors
// with FP32 internal math, and rcd_rb_at_g + rcd_compose + ccm_wb fused into rcd_finish_rgb.

#define RCD_EPS 1e-5f
#define RCD_EPSSQ 1e-10f
#define RCD_PED 0.0625f

inline float rc(__global const float* cfa, int W, int H, int x, int y) {
    return clamp((cfa[(size_t)clamp(y, 0, H - 1) * W + clamp(x, 0, W - 1)] + RCD_PED) * (1.f / (1.f + RCD_PED)), 0.f, 1.f);
}
inline float sq(float x) { return x * x; }
inline float intp(float a, float b, float c) { return a * (b - c) + c; }
inline float disc(float central, float neigh) { return fabs(0.5f - central) < fabs(0.5f - neigh) ? neigh : central; }

// Stores un-squared high-pass differences (dv, dh, dp, dq) in a single coalesced half4 buffer (halves DRAM vs 4 floats,
// and squaring in FP32 inside rcd_dir avoids any FP16 underflow near RCD_EPSSQ = 1e-10f).
__kernel void rcd_hpf(__global const float* cfa, int W, int H, __global half* hpf) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    float c = C(0, 0);
    float dv = (C(0, -3) - C(0, -1) - C(0, 1) + C(0, 3)) - 3.f * (C(0, -2) + C(0, 2)) + 6.f * c;
    float dh = (C(-3, 0) - C(-1, 0) - C(1, 0) + C(3, 0)) - 3.f * (C(-2, 0) + C(2, 0)) + 6.f * c;
    float dp = (C(-3, -3) - C(-1, -1) - C(1, 1) + C(3, 3)) - 3.f * (C(-2, -2) + C(2, 2)) + 6.f * c;
    float dq = (C(3, -3) - C(1, -1) - C(-1, 1) + C(-3, 3)) - 3.f * (C(2, -2) + C(-2, 2)) + 6.f * c;
#undef C
    vstore_half4((float4)(dv, dh, dp, dq), (size_t)y * W + x, hpf);
}

// Packs (VH, LPF) into half2 and PQ into half.
__kernel void rcd_dir(__global const float* cfa, int W, int H, int cfaType, __global const half* hpf,
                      __global half* vh_lpf, __global half* pqBuf) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    int xm = max(x - 1, 0), xp = min(x + 1, W - 1), ym = max(y - 1, 0), yp = min(y + 1, H - 1);
    float4 hc = vload_half4(i, hpf);
    float v_ym = vload_half4((size_t)ym * W + x, hpf).x;
    float v_yp = vload_half4((size_t)yp * W + x, hpf).x;
    float h_xm = vload_half4((size_t)y * W + xm, hpf).y;
    float h_xp = vload_half4((size_t)y * W + xp, hpf).y;
    float vs = max(RCD_EPSSQ, sq(v_ym) + sq(hc.x) + sq(v_yp));
    float hs = max(RCD_EPSSQ, sq(h_xm) + sq(hc.y) + sq(h_xp));
    float vh = vs / (vs + hs);
    if (cfa_color(cfaType, x, y) == 1) {
        vstore_half2((float2)(vh, 0.f), i, vh_lpf);
        return;
    }
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    float lpf = C(0, 0) + 0.5f * (C(0, -1) + C(0, 1) + C(-1, 0) + C(1, 0)) + 0.25f * (C(-1, -1) + C(1, -1) + C(-1, 1) + C(1, 1));
#undef C
    vstore_half2((float2)(vh, lpf), i, vh_lpf);

#define HPF_AT(xx, yy) vload_half4((size_t)clamp(yy, 0, H - 1) * W + clamp(xx, 0, W - 1), hpf)
    int a = (x - 1) | 1;
    float4 p0 = HPF_AT(a, y - 1), p1 = HPF_AT(x | 1, y), p2 = HPF_AT(a + 2, y + 1);
    float4 q0 = HPF_AT(a + 2, y - 1), q2 = HPF_AT(a, y + 1);
#undef HPF_AT
    float ps = max(RCD_EPSSQ, sq(p0.z) + sq(p1.z) + sq(p2.z));
    float qs = max(RCD_EPSSQ, sq(q0.w) + sq(p1.w) + sq(q2.w));
    vstore_half(ps / (ps + qs), i, pqBuf);
}

// step 3: green at R/B
__kernel void rcd_green(__global const float* cfa, int W, int H, int cfaType, __global const half* vh_lpf, __global half* G) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    float ci = C(0, 0);
    if (cfa_color(cfaType, x, y) == 1 || x < 4 || y < 4 || x >= W - 4 || y >= H - 4) {
        vstore_half(ci, i, G);
        return;
    }
    float N_Grad = RCD_EPS + (fabs(C(0, -1) - C(0, 1)) + fabs(ci - C(0, -2))) + (fabs(C(0, -1) - C(0, -3)) + fabs(C(0, -2) - C(0, -4)));
    float S_Grad = RCD_EPS + (fabs(C(0, -1) - C(0, 1)) + fabs(ci - C(0, 2))) + (fabs(C(0, 1) - C(0, 3)) + fabs(C(0, 2) - C(0, 4)));
    float W_Grad = RCD_EPS + (fabs(C(-1, 0) - C(1, 0)) + fabs(ci - C(-2, 0))) + (fabs(C(-1, 0) - C(-3, 0)) + fabs(C(-2, 0) - C(-4, 0)));
    float E_Grad = RCD_EPS + (fabs(C(-1, 0) - C(1, 0)) + fabs(ci - C(2, 0))) + (fabs(C(1, 0) - C(3, 0)) + fabs(C(2, 0) - C(4, 0)));
    float2 vhl = vload_half2(i, vh_lpf);
    float l = vhl.y;
    float N_Est = C(0, -1) * (l + l) / (RCD_EPS + l + vload_half2(i - 2 * W, vh_lpf).y);
    float S_Est = C(0, 1) * (l + l) / (RCD_EPS + l + vload_half2(i + 2 * W, vh_lpf).y);
    float W_Est = C(-1, 0) * (l + l) / (RCD_EPS + l + vload_half2(i - 2, vh_lpf).y);
    float E_Est = C(1, 0) * (l + l) / (RCD_EPS + l + vload_half2(i + 2, vh_lpf).y);
#undef C
    float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
    float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);
    float nb = 0.25f * ((vload_half2(i - W - 1, vh_lpf).x + vload_half2(i - W + 1, vh_lpf).x) +
                        (vload_half2(i + W - 1, vh_lpf).x + vload_half2(i + W + 1, vh_lpf).x));
    vstore_half(intp(disc(vhl.x, nb), H_Est, V_Est), i, G);
}

// step 4.2: R and B at R/B sites stored together as half2(R, B)
__kernel void rcd_rb_at_rb(__global const float* cfa, int W, int H, int cfaType, __global const half* G,
                           __global const half* pqBuf, __global half* rbBuf) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int col = cfa_color(cfaType, x, y);
    if (col == 1) return;
    size_t i = (size_t)y * W + x;
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    float own = C(0, 0), other = own;
    if (x >= 4 && y >= 4 && x < W - 4 && y < H - 4) {
        float g0 = vload_half(i, G);
#define GG(dx, dy) vload_half(i + (dy) * W + (dx), G)
        float nb = 0.25f * (vload_half(i - W - 1, pqBuf) + vload_half(i - W + 1, pqBuf) +
                            vload_half(i + W - 1, pqBuf) + vload_half(i + W + 1, pqBuf));
        float pqDisc = disc(vload_half(i, pqBuf), nb);
        float NW_Grad = RCD_EPS + fabs(C(-1, -1) - C(1, 1)) + fabs(C(-1, -1) - C(-3, -3)) + fabs(g0 - GG(-2, -2));
        float NE_Grad = RCD_EPS + fabs(C(1, -1) - C(-1, 1)) + fabs(C(1, -1) - C(3, -3)) + fabs(g0 - GG(2, -2));
        float SW_Grad = RCD_EPS + fabs(C(1, -1) - C(-1, 1)) + fabs(C(-1, 1) - C(-3, 3)) + fabs(g0 - GG(-2, 2));
        float SE_Grad = RCD_EPS + fabs(C(-1, -1) - C(1, 1)) + fabs(C(1, 1) - C(3, 3)) + fabs(g0 - GG(2, 2));
        float NW_Est = C(-1, -1) - GG(-1, -1);
        float NE_Est = C(1, -1) - GG(1, -1);
        float SW_Est = C(-1, 1) - GG(-1, 1);
        float SE_Est = C(1, 1) - GG(1, 1);
#undef GG
        float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
        float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);
        other = g0 + intp(pqDisc, Q_Est, P_Est);
    }
#undef C
    float2 rb = (col == 0) ? (float2)(own, other) : (float2)(other, own);
    vstore_half2(rb, i, rbBuf);
}

// Fuses step 4.3 (R & B at G in parallel float2) + rcd_compose (scalar border, zero private arrays) + CCM into half4 RGB.
__kernel void rcd_finish_rgb(__global const float* cfa, int W, int H, int cfaType, __global const half* G,
                             __global const half* vh_lpf, __global const half* rbBuf, int applyCcm,
                             float4 m0, float4 m1, float4 m2, __global half* rgbOut) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    const float span = 1.f + RCD_PED;
    float r, g, b;

    if (x >= 9 && y >= 9 && x < W - 9 && y < H - 9) {
        g = vload_half(i, G);
        int col = cfa_color(cfaType, x, y);
        if (col != 1) {
            float2 rb = vload_half2(i, rbBuf);
            r = rb.x;
            b = rb.y;
        } else {
            float vh0 = vload_half2(i, vh_lpf).x;
            float nb = 0.25f * ((vload_half2(i - W - 1, vh_lpf).x + vload_half2(i - W + 1, vh_lpf).x) +
                                (vload_half2(i + W - 1, vh_lpf).x + vload_half2(i + W + 1, vh_lpf).x));
            float vhDisc = disc(vh0, nb);
            float N1 = RCD_EPS + fabs(g - vload_half(i - 2 * W, G));
            float S1 = RCD_EPS + fabs(g - vload_half(i + 2 * W, G));
            float W1 = RCD_EPS + fabs(g - vload_half(i - 2, G));
            float E1 = RCD_EPS + fabs(g - vload_half(i + 2, G));
            float gmw = vload_half(i - W, G), gpw = vload_half(i + W, G);
            float gm1 = vload_half(i - 1, G), gp1 = vload_half(i + 1, G);

            float2 xmw = vload_half2(i - W, rbBuf), xpw = vload_half2(i + W, rbBuf);
            float2 xm3w = vload_half2(i - 3 * W, rbBuf), xp3w = vload_half2(i + 3 * W, rbBuf);
            float2 xm1 = vload_half2(i - 1, rbBuf), xp1 = vload_half2(i + 1, rbBuf);
            float2 xm3 = vload_half2(i - 3, rbBuf), xp3 = vload_half2(i + 3, rbBuf);

            float2 SNabs = fabs(xmw - xpw);
            float2 EWabs = fabs(xm1 - xp1);
            float2 N_Grad = N1 + SNabs + fabs(xmw - xm3w);
            float2 S_Grad = S1 + SNabs + fabs(xpw - xp3w);
            float2 W_Grad = W1 + EWabs + fabs(xm1 - xm3);
            float2 E_Grad = E1 + EWabs + fabs(xp1 - xp3);
            float2 N_Est = xmw - gmw;
            float2 S_Est = xpw - gpw;
            float2 W_Est = xm1 - gm1;
            float2 E_Est = xp1 - gp1;
            float2 V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad);
            float2 H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad);
            float2 rb = g + vhDisc * (H_Est - V_Est) + V_Est;
            r = rb.x;
            b = rb.y;
        }
        r = r * span - RCD_PED;
        g = g * span - RCD_PED;
        b = b * span - RCD_PED;
    } else {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
        float n0 = 0.f, n1 = 0.f, n2 = 0.f;
        for (int i1 = y - 1; i1 < y + 2; i1++)
            for (int j1 = x - 1; j1 < x + 2; j1++) {
                if (i1 < 0 || i1 >= H || j1 < 0 || j1 >= W) continue;
                int c = cfa_color(cfaType, j1, i1);
                float v = rc(cfa, W, H, j1, i1);
                if (c == 0) { s0 += v; n0 += 1.f; }
                else if (c == 1) { s1 += v; n1 += 1.f; }
                else { s2 += v; n2 += 1.f; }
            }
        int c = cfa_color(cfaType, x, y);
        float v = rc(cfa, W, H, x, y);
        r = (c == 0 ? v : s0 / max(n0, 1.f)) * span - RCD_PED;
        g = (c == 1 ? v : s1 / max(n1, 1.f)) * span - RCD_PED;
        b = (c == 2 ? v : s2 / max(n2, 1.f)) * span - RCD_PED;
    }

    if (applyCcm) {
        float nr = m0.x * r + m0.y * g + m0.z * b;
        float ng = m1.x * r + m1.y * g + m1.z * b;
        float nb = m2.x * r + m2.y * g + m2.z * b;
        r = nr; g = ng; b = nb;
    }
    vstore_half4((float4)(r, g, b, 0.f), i, rgbOut);
}
)CL"