R"CL(
// RCD demosaicing (Luis Sanz Rodriguez, librtprocess rcd.cc, GPL-3) as whole-image passes. Inside the
// 9-pixel border every pass reproduces the tiled CPU code; the border uses bayerborder_demosaic's
// 3x3 average. Values are the clamped [0, 1] mosaic, as RCD's LIM01 input.

#define RCD_EPS 1e-5f
#define RCD_EPSSQ 1e-10f

inline float rc(__global const float* cfa, int W, int H, int x, int y) {
    return clamp(cfa[(size_t)clamp(y, 0, H - 1) * W + clamp(x, 0, W - 1)], 0.f, 1.f);
}
inline float sq(float x) { return x * x; }
inline float intp(float a, float b, float c) { return a * (b - c) + c; }

// step 1.1 / 1.2 / 4.0: squared high-pass of the colour difference, vertical, horizontal and diagonal
__kernel void rcd_hpf(__global const float* cfa, int W, int H, __global float* V, __global float* Hh, __global float* P,
                      __global float* Q) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    float c = C(0, 0);
    size_t i = (size_t)y * W + x;
    V[i] = sq((C(0, -3) - C(0, -1) - C(0, 1) + C(0, 3)) - 3.f * (C(0, -2) + C(0, 2)) + 6.f * c);
    Hh[i] = sq((C(-3, 0) - C(-1, 0) - C(1, 0) + C(3, 0)) - 3.f * (C(-2, 0) + C(2, 0)) + 6.f * c);
    P[i] = sq((C(-3, -3) - C(-1, -1) - C(1, 1) + C(3, 3)) - 3.f * (C(-2, -2) + C(2, 2)) + 6.f * c);
    Q[i] = sq((C(3, -3) - C(1, -1) - C(-1, 1) + C(-3, 3)) - 3.f * (C(2, -2) + C(-2, 2)) + 6.f * c);
#undef C
}

// step 1.2 (VH_Dir), step 2 (low-pass at R/B) and step 4.1 (PQ_Dir at R/B). The CPU stores the diagonal
// high-pass only at odd columns and reads it through half-indices, so column c reads column c | 1.
__kernel void rcd_dir(__global const float* cfa, int W, int H, int cfaType, __global const float* V,
                      __global const float* Hh, __global const float* P, __global const float* Q, __global float* VH,
                      __global float* LPF, __global float* PQ) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    int xm = max(x - 1, 0), xp = min(x + 1, W - 1), ym = max(y - 1, 0), yp = min(y + 1, H - 1);
    float vs = max(RCD_EPSSQ, V[(size_t)ym * W + x] + V[i] + V[(size_t)yp * W + x]);
    float hs = max(RCD_EPSSQ, Hh[(size_t)y * W + xm] + Hh[i] + Hh[(size_t)y * W + xp]);
    VH[i] = vs / (vs + hs);
    if (cfa_color(cfaType, x, y) == 1) { LPF[i] = 0; PQ[i] = 0; return; }
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    LPF[i] = C(0, 0) + 0.5f * (C(0, -1) + C(0, 1) + C(-1, 0) + C(1, 0)) + 0.25f * (C(-1, -1) + C(1, -1) + C(-1, 1) + C(1, 1));
#undef C
#define AT(B, xx, yy) B[(size_t)clamp(yy, 0, H - 1) * W + clamp(xx, 0, W - 1)]
    int a = (x - 1) | 1;
    float ps = max(RCD_EPSSQ, AT(P, a, y - 1) + AT(P, x | 1, y) + AT(P, a + 2, y + 1));
    float qs = max(RCD_EPSSQ, AT(Q, a + 2, y - 1) + AT(Q, x | 1, y) + AT(Q, a, y + 1));
#undef AT
    PQ[i] = ps / (ps + qs);
}

inline float disc(float central, float neigh) { return fabs(0.5f - central) < fabs(0.5f - neigh) ? neigh : central; }

// step 3: green at R/B
__kernel void rcd_green(__global const float* cfa, int W, int H, int cfaType, __global const float* VH,
                        __global const float* LPF, __global float* G) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    float ci = C(0, 0);
    if (cfa_color(cfaType, x, y) == 1 || x < 4 || y < 4 || x >= W - 4 || y >= H - 4) { G[i] = ci; return; }
    float N_Grad = RCD_EPS + (fabs(C(0, -1) - C(0, 1)) + fabs(ci - C(0, -2))) + (fabs(C(0, -1) - C(0, -3)) + fabs(C(0, -2) - C(0, -4)));
    float S_Grad = RCD_EPS + (fabs(C(0, -1) - C(0, 1)) + fabs(ci - C(0, 2))) + (fabs(C(0, 1) - C(0, 3)) + fabs(C(0, 2) - C(0, 4)));
    float W_Grad = RCD_EPS + (fabs(C(-1, 0) - C(1, 0)) + fabs(ci - C(-2, 0))) + (fabs(C(-1, 0) - C(-3, 0)) + fabs(C(-2, 0) - C(-4, 0)));
    float E_Grad = RCD_EPS + (fabs(C(-1, 0) - C(1, 0)) + fabs(ci - C(2, 0))) + (fabs(C(1, 0) - C(3, 0)) + fabs(C(2, 0) - C(4, 0)));
    float l = LPF[i];
    float N_Est = C(0, -1) * (l + l) / (RCD_EPS + l + LPF[i - 2 * W]);
    float S_Est = C(0, 1) * (l + l) / (RCD_EPS + l + LPF[i + 2 * W]);
    float W_Est = C(-1, 0) * (l + l) / (RCD_EPS + l + LPF[i - 2]);
    float E_Est = C(1, 0) * (l + l) / (RCD_EPS + l + LPF[i + 2]);
#undef C
    float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
    float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);
    float nb = 0.25f * ((VH[i - W - 1] + VH[i - W + 1]) + (VH[i + W - 1] + VH[i + W + 1]));
    G[i] = intp(disc(VH[i], nb), H_Est, V_Est);
}

// step 4.2: the opposite colour at R/B; R and B planes get the native sample there too
__kernel void rcd_rb_at_rb(__global const float* cfa, int W, int H, int cfaType, __global const float* G,
                           __global const float* PQ, __global float* R, __global float* B) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int col = cfa_color(cfaType, x, y);
    if (col == 1) return;
    size_t i = (size_t)y * W + x;
#define C(dx, dy) rc(cfa, W, H, x + (dx), y + (dy))
    float own = C(0, 0), other = own;
    if (x >= 4 && y >= 4 && x < W - 4 && y < H - 4) {
        float g0 = G[i];
#define GG(dx, dy) G[i + (dy) * W + (dx)]
        float nb = 0.25f * (PQ[i - W - 1] + PQ[i - W + 1] + PQ[i + W - 1] + PQ[i + W + 1]);
        float pqDisc = disc(PQ[i], nb);
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
    if (col == 0) { R[i] = own; B[i] = other; }
    else { B[i] = own; R[i] = other; }
}

// step 4.3: R and B at green
__kernel void rcd_rb_at_g(int W, int H, int cfaType, __global const float* G, __global const float* VH,
                          __global float* R, __global float* B) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    if (cfa_color(cfaType, x, y) != 1) return;
    size_t i = (size_t)y * W + x;
    if (x < 4 || y < 4 || x >= W - 4 || y >= H - 4) { R[i] = G[i]; B[i] = G[i]; return; }
    float nb = 0.25f * ((VH[i - W - 1] + VH[i - W + 1]) + (VH[i + W - 1] + VH[i + W + 1]));
    float vhDisc = disc(VH[i], nb);
    float g1 = G[i];
    float N1 = RCD_EPS + fabs(g1 - G[i - 2 * W]);
    float S1 = RCD_EPS + fabs(g1 - G[i + 2 * W]);
    float W1 = RCD_EPS + fabs(g1 - G[i - 2]);
    float E1 = RCD_EPS + fabs(g1 - G[i + 2]);
    float gmw = G[i - W], gpw = G[i + W], gm1 = G[i - 1], gp1 = G[i + 1];
    for (int c = 0; c < 2; c++) {
        __global float* X = c == 0 ? R : B;
        float SNabs = fabs(X[i - W] - X[i + W]);
        float EWabs = fabs(X[i - 1] - X[i + 1]);
        float N_Grad = N1 + SNabs + fabs(X[i - W] - X[i - 3 * W]);
        float S_Grad = S1 + SNabs + fabs(X[i + W] - X[i + 3 * W]);
        float W_Grad = W1 + EWabs + fabs(X[i - 1] - X[i - 3]);
        float E_Grad = E1 + EWabs + fabs(X[i + 1] - X[i + 3]);
        float N_Est = X[i - W] - gmw;
        float S_Est = X[i + W] - gpw;
        float W_Est = X[i - 1] - gm1;
        float E_Est = X[i + 1] - gp1;
        float V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad);
        float H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad);
        X[i] = g1 + intp(vhDisc, H_Est, V_Est);
    }
}

// interleave; the 9-pixel frame is bayerborder_demosaic on the unclamped mosaic
__kernel void rcd_compose(__global const float* cfa, int W, int H, int cfaType, __global const float* R,
                          __global const float* G, __global const float* B, __global float* rgb) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    __global float* o = rgb + i * 3;
    if (x >= 9 && y >= 9 && x < W - 9 && y < H - 9) {
        o[0] = max(0.f, R[i]); o[1] = max(0.f, G[i]); o[2] = max(0.f, B[i]);
        return;
    }
    float sum[3] = {0, 0, 0}, cnt[3] = {0, 0, 0};
    for (int i1 = y - 1; i1 < y + 2; i1++)
        for (int j1 = x - 1; j1 < x + 2; j1++) {
            if (i1 < 0 || i1 >= H || j1 < 0 || j1 >= W) continue;
            int c = cfa_color(cfaType, j1, i1);
            sum[c] += max(0.f, cfa[(size_t)i1 * W + j1]);
            cnt[c] += 1;
        }
    int c = cfa_color(cfaType, x, y);
    float v = max(0.f, cfa[i]);
    float r = c == 0 ? v : sum[0] / cnt[0];
    float g = c == 1 ? v : sum[1] / cnt[1];
    float b = c == 2 ? v : sum[2] / cnt[2];
    o[0] = max(0.f, r); o[1] = max(0.f, g); o[2] = max(0.f, b);
}
)CL"
