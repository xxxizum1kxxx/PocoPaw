#include "kr.h"

#if defined(__ARM_NEON) && !defined(KR_NO_NEON)
#include <arm_neon.h>
#define KR_NEON 1
#else
#define KR_NEON 0
#endif

namespace {

#if KR_NEON
// exp(x) for x <= 0 (Cephes expf: Cody-Waite range reduction + degree-5 polynomial, ~1e-7 relative)
inline float32x4_t exp_neg(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t f = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    f = vfmsq_f32(f, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, f);
    p = vfmaq_f32(vaddq_f32(f, vdupq_n_f32(1.f)), p, vmulq_f32(f, f));
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
#endif

float g_win[2 * kStride];

struct WinInit {
    WinInit() {
        for (int u = 0; u < 2 * kStride; u++) g_win[u] = 0.5f - 0.5f * std::cos(2.f * (float)M_PI * (u + 0.5f) / (2 * kStride));
    }
} g_winInit;

inline float bilinear(const ImageF& im, float x, float y) {
    x = clampf(x, 0.f, im.w - 1.001f);
    y = clampf(y, 0.f, im.h - 1.001f);
    int x0 = (int)x, y0 = (int)y;
    float fx = x - x0, fy = y - y0;
    const float* r0 = im.row(y0) + x0;
    const float* r1 = im.row(y0 + 1) + x0;
    return (1 - fy) * ((1 - fx) * r0[0] + fx * r0[1]) + fy * ((1 - fx) * r1[0] + fx * r1[1]);
}

// displacement (gray px) at quad (qx,qy): raised-cosine blend of the 2x2 overlapping tiles
inline void flow_at(const Flow& f, int qx, int qy, float& dx, float& dy) {
    int bi = qx / kStride, bj = qy / kStride;
    int ux = qx - bi * kStride, uy = qy - bj * kStride;
    int i0 = clampi(bi - 1, 0, f.nx - 1), i1 = clampi(bi, 0, f.nx - 1);
    int j0 = clampi(bj - 1, 0, f.ny - 1), j1 = clampi(bj, 0, f.ny - 1);
    float wx0 = g_win[ux + kStride], wx1 = g_win[ux], wy0 = g_win[uy + kStride], wy1 = g_win[uy];
    auto F = [&](const std::vector<float>& v, int i, int j) { return v[(size_t)j * f.nx + i]; };
    dx = wy0 * (wx0 * F(f.fx, i0, j0) + wx1 * F(f.fx, i1, j0)) + wy1 * (wx0 * F(f.fx, i0, j1) + wx1 * F(f.fx, i1, j1));
    dy = wy0 * (wx0 * F(f.fy, i0, j0) + wx1 * F(f.fy, i1, j0)) + wy1 * (wx0 * F(f.fy, i0, j1) + wx1 * F(f.fy, i1, j1));
}

float lerp_snr(float snr, float a, float b) { return a + (b - a) * clampf((snr - 6.f) / 24.f, 0.f, 1.f); }

}  // namespace

Flow zero_flow(const Offsets& o) {
    Flow f;
    f.nx = o.nx; f.ny = o.ny;
    f.fx.assign((size_t)o.nx * o.ny, 0.f);
    f.fy = f.fx;
    return f;
}

LkGrad lk_gradients(const ImageF& mref) {
    LkGrad g;
    g.gx = ImageF(mref.w, mref.h);
    g.gy = ImageF(mref.w, mref.h);
    parallel_for(mref.h, [&](int y) {
        for (int x = 0; x < mref.w; x++) {
            g.gx.at(x, y) = 0.5f * (mref.atc(x + 1, y) - mref.atc(x - 1, y));
            g.gy.at(x, y) = 0.5f * (mref.atc(x, y + 1) - mref.atc(x, y - 1));
        }
    });
    return g;
}

Flow refine_flow(const ImageF& mref, const ImageF& malt, const Offsets& o, int iters, const LkGrad& g,
                 const NoiseModel& nm, float black) {
    Flow f = zero_flow(o);
    f.err.assign(f.fx.size(), 1e3f);
    parallel_for(o.ny, [&](int j) {
        for (int i = 0; i < o.nx; i++) {
            float dx = o.x(i, j), dy = o.y(i, j);
            int x0 = std::max(1, i * kStride), x1 = std::min(mref.w - 1, i * kStride + kTile);
            int y0 = std::max(1, j * kStride), y1 = std::min(mref.h - 1, j * kStride + kTile);
            double a = 0, b = 0, c = 0;
            for (int y = y0; y < y1; y++) {
                const float* gx = g.gx.row(y);
                const float* gy = g.gy.row(y);
                float ra = 0, rb = 0, rc = 0;
                for (int x = x0; x < x1; x++) { ra += gx[x] * gx[x]; rb += gx[x] * gy[x]; rc += gy[x] * gy[x]; }
                a += ra; b += rb; c += rc;
            }
            double det = a * c - b * b, tr = a + c;
            if (x1 <= x0 || y1 <= y0 || tr <= 0 || det < 1e-4 * tr * tr) {
                f.fx[(size_t)j * o.nx + i] = dx; f.fy[(size_t)j * o.nx + i] = dy;
                continue;
            }
            {
                // displacement covariance ~ var(residual) * H^-1; the residual is the difference of two
                // 3x3 means of 4-pixel sums. Use the weaker direction (smallest Hessian eigenvalue).
                double s = 0;
                for (int y = y0; y < y1; y += 4) for (int x = x0; x < x1; x += 4) s += mref.at(x, y);
                s /= ((y1 - y0 + 3) / 4) * ((x1 - x0 + 3) / 4);
                double varRes = 2.0 * 4.0 * nm.var(s / 4 - black) / 9.0;
                double lmin = 0.5 * tr - std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
                f.err[(size_t)j * o.nx + i] = (float)std::sqrt(varRes / std::max(lmin, 1e-9));
            }
            for (int it = 0; it < iters; it++) {
                double bx = 0, by = 0;
                int ix = (int)std::floor(dx), iy = (int)std::floor(dy);
                float fx = dx - ix, fy = dy - iy;
                bool inside = x0 + ix >= 0 && x1 + ix < malt.w - 1 && y0 + iy >= 0 && y1 + iy < malt.h - 1;
                for (int y = y0; y < y1; y++) {
                    const float* gx = g.gx.row(y);
                    const float* gy = g.gy.row(y);
                    const float* rr = mref.row(y);
                    float sx = 0, sy = 0;
                    if (inside) {
                        // whole tile inside the image: fixed bilinear weights, no clamping
                        const float* a0 = malt.row(y + iy) + ix;
                        const float* a1 = malt.row(y + iy + 1) + ix;
                        float w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx, w10 = fy * (1 - fx), w11 = fy * fx;
                        for (int x = x0; x < x1; x++) {
                            float r = w00 * a0[x] + w01 * a0[x + 1] + w10 * a1[x] + w11 * a1[x + 1] - rr[x];
                            sx += gx[x] * r; sy += gy[x] * r;
                        }
                    } else {
                        for (int x = x0; x < x1; x++) {
                            float r = bilinear(malt, x + dx, y + dy) - rr[x];
                            sx += gx[x] * r; sy += gy[x] * r;
                        }
                    }
                    bx += sx; by += sy;
                }
                float sx = (float)((c * bx - b * by) / det), sy = (float)((a * by - b * bx) / det);
                sx = clampf(sx, -1.f, 1.f); sy = clampf(sy, -1.f, 1.f);
                dx -= sx; dy -= sy;
                if (std::fabs(sx) + std::fabs(sy) < 0.01f) break;
            }
            // never move more than one gray pixel away from the block-matching optimum
            dx = clampf(dx, o.x(i, j) - 1.f, o.x(i, j) + 1.f);
            dy = clampf(dy, o.y(i, j) - 1.f, o.y(i, j) + 1.f);
            f.fx[(size_t)j * o.nx + i] = dx; f.fy[(size_t)j * o.nx + i] = dy;
        }
    });
    return f;
}

ImageF kr_merge(const std::vector<Raw>& frames, int ref, const std::vector<Flow>& flows,
                const ImageF& grayRef, const std::vector<ImageF>& mean3, const Meta& m,
                const NoiseModel& nm, const MergeOpts& o, KrParams p, KrStats* st) {
    const Raw& R = frames[ref];
    const int W = R.w, H = R.h, qw = W / 2, qh = H / 2, n = (int)frames.size();
    const float black = (m.black[0] + m.black[1] + m.black[2] + m.black[3]) / 4;

    // burst SNR (dB) of the reference frame, as in the IPOL implementation
    double s2 = 0, v2 = 0;
    for (int y = 0; y < H; y += 7)
        for (int x = 0; x < W; x += 7) {
            double s = R.at(x, y) - black;
            if (s <= 0 || R.at(x, y) >= m.white) continue;
            s2 += s * s; v2 += nm.var(s);
        }
    float snr = v2 > 0 ? (float)(10.0 * std::log10(s2 / v2)) : 30.f;
    if (p.kDetail < 0) p.kDetail = lerp_snr(snr, 0.33f, 0.25f);
    if (p.kDenoise < 0) p.kDenoise = lerp_snr(snr, 5.0f, 3.0f);
    if (p.dTh < 0) p.dTh = lerp_snr(snr, 0.81f, 0.71f);
    if (p.dTr < 0) p.dTr = lerp_snr(snr, 1.24f, 1.0f);

    // kernel shape from the variance-stabilised reference (generalised Anscombe transform, so the
    // structure-tensor thresholds are in noise-sigma units at every brightness)
    ImageF vst(qw, qh);
    const double a = nm.a, b = nm.b;
    parallel_for(qh, [&](int y) {
        for (int x = 0; x < qw; x++) {
            double s = grayRef.at(x, y) / 4 - black;
            vst.at(x, y) = (float)(2.0 / a * std::sqrt(std::max(0.0, a * s + 0.375 * a * a + b)));
        }
    });
    ImageF icov(qw, qh, 3);
    parallel_for(qh, [&](int y) {
        for (int x = 0; x < qw; x++) {
            float txx = 0, txy = 0, tyy = 0;
            for (int v = -1; v <= 0; v++)
                for (int u = -1; u <= 0; u++) {
                    int X = x + u, Y = y + v;
                    float gx = 0.5f * ((vst.atc(X + 1, Y) - vst.atc(X, Y)) + (vst.atc(X + 1, Y + 1) - vst.atc(X, Y + 1)));
                    float gy = 0.5f * ((vst.atc(X, Y + 1) - vst.atc(X, Y)) + (vst.atc(X + 1, Y + 1) - vst.atc(X + 1, Y)));
                    txx += gx * gx; txy += gx * gy; tyy += gy * gy;
                }
            float tr = txx + tyy, det = txx * tyy - txy * txy;
            float disc = std::sqrt(std::max(0.f, 0.25f * tr * tr - det));
            float l1 = 0.5f * tr + disc, l2 = std::max(0.f, 0.5f * tr - disc);
            float e1x = 1, e1y = 0;
            if (std::fabs(txy) > 1e-12f) { e1x = txy; e1y = l1 - txx; float nrm = std::hypot(e1x, e1y); e1x /= nrm; e1y /= nrm; }
            else if (tyy > txx) { e1x = 0; e1y = 1; }
            float A = (l1 + l2) > 1e-12f ? 1.f + std::sqrt((l1 - l2) / (l1 + l2)) : 1.f;
            float D = clampf(1.f - std::sqrt(l1) / p.dTr + p.dTh, 0.f, 1.f);
            float k1 = (2 - A) + (A - 1) / p.kShrink, k2 = (2 - A) + (A - 1) * p.kStretch;
            k1 = p.kDetail * ((1 - D) * k1 + D * p.kDenoise);
            k2 = p.kDetail * ((1 - D) * k2 + D * p.kDenoise);
            float i1 = 1.f / (k1 * k1), i2 = 1.f / (k2 * k2);
            float e2x = -e1y, e2y = e1x;
            float* o = &icov.at(x, y);
            o[0] = i1 * e1x * e1x + i2 * e2x * e2x;
            o[1] = i1 * e1x * e1y + i2 * e2x * e2y;
            o[2] = i1 * e1y * e1y + i2 * e2y * e2y;
        }
    });
    vst = ImageF();

    // per-frame robustness at quad resolution: local-mean mismatch against the noise model (with the
    // frame's exposure weight, zero where it clipped), then a min filter so a rejection also covers its
    // neighbourhood
    std::vector<ImageF> rob(n);
    std::vector<double> frameW(n, 1.0);
    for (int k = 0; k < n; k++) {
        if (k == ref) continue;
        rob[k] = ImageF(qw, qh);
        const float g = o.g(k), clip = o.clipAt(k);
        const Raw& F = frames[k];
        parallel_for(qh, [&](int qy) {
            for (int qx = 0; qx < qw; qx++) {
                float dx, dy;
                flow_at(flows[k], qx, qy, dx, dy);
                float mu = bilinear(mean3[k], qx + dx, qy + dy);
                int sx = clampi((int)std::lround(qx + dx), 0, qw - 1), sy = clampi((int)std::lround(qy + dy), 0, qh - 1);
                bool clipped = std::max(std::max(F.at(2 * sx, 2 * sy), F.at(2 * sx + 1, 2 * sy)),
                                        std::max(F.at(2 * sx, 2 * sy + 1), F.at(2 * sx + 1, 2 * sy + 1))) >= clip;
                rob[k].at(qx, qy) = clipped ? 0.f : frame_weight(mean3[ref].at(qx, qy), mu, black, nm, g, o);
            }
        });
        min_filter(rob[k], o.dilate);
        std::vector<double> rowSum(qh, 0.0);
        parallel_for(qh, [&](int qy) {
            double s = 0;
            for (int qx = 0; qx < qw; qx++) s += std::min(rob[k].at(qx, qy), 1.f);
            rowSum[qy] = s;
        });
        double s = 0;
        for (double v : rowSum) s += v;
        frameW[k] = s / ((double)qw * qh);
    }

    // accumulation, gathered per output pixel
    const float sc = p.scale;
    const int OW = (int)std::lround(W * sc), OH = (int)std::lround(H * sc);
    ImageF out(OW, OH, 3);
    float blk[4];
    for (int c = 0; c < 4; c++) blk[c] = m.black[c];
#if KR_NEON
    // a 4-wide window starting at an even/odd column of an even/odd row: per-lane black level and
    // colour masks (lane 3 lies outside the 3x3 support and is masked off)
    float32x4_t blackV[4], maskV[4][3];
    for (int pr = 0; pr < 2; pr++)
        for (int pc = 0; pc < 2; pc++) {
            float bl[4], mk[3][4];
            for (int l = 0; l < 4; l++) {
                int ch = cfa_channel(m.cfa, pc + l, pr);
                bl[l] = blk[ch];
                int col = ch == 0 ? 0 : (ch == 3 ? 2 : 1);
                for (int c = 0; c < 3; c++) mk[c][l] = (l < 3 && c == col) ? 1.f : 0.f;
            }
            blackV[pr * 2 + pc] = vld1q_f32(bl);
            for (int c = 0; c < 3; c++) maskV[pr * 2 + pc][c] = vld1q_f32(mk[c]);
        }
    const float laneInit[4] = {0, 1, 2, 3};
    const float32x4_t lane = vld1q_f32(laneInit);
#endif
    const bool lca = p.lcaR[0] != 0 || p.lcaR[1] != 0 || p.lcaB[0] != 0 || p.lcaB[1] != 0;
    const float fcx = (m.full_w() - 1) * 0.5f, fcy = (m.full_h() - 1) * 0.5f, invR2 = 1.f / (fcx * fcx + fcy * fcy);
    const float lcx = fcx - m.cropX, lcy = fcy - m.cropY;
    parallel_for(OH, [&](int Y) {
        float* o = out.row(Y);
        float ly = (Y + 0.5f) / sc;
        int qy = clampi((int)(ly / 2), 0, qh - 1);
        int ky = clampi((int)std::lround(ly / 2 - 0.5f), 0, qh - 1);
        // displacements along this quad row, once per quad instead of once per output pixel
        thread_local std::vector<float> fl;
        fl.resize((size_t)n * qw * 2);
        for (int k = 0; k < n; k++) {
            if (k == ref) continue;
            float* d = &fl[(size_t)k * qw * 2];
            for (int qx = 0; qx < qw; qx++) flow_at(flows[k], qx, qy, d[2 * qx], d[2 * qx + 1]);
        }
        for (int X = 0; X < OW; X++, o += 3) {
            float lx = (X + 0.5f) / sc;
            int qx = clampi((int)(lx / 2), 0, qw - 1);
            int kx = clampi((int)std::lround(lx / 2 - 0.5f), 0, qw - 1);
            const float* ic = &icov.at(kx, ky);
            float num[3] = {0, 0, 0}, den[3] = {0, 0, 0};
            // per-colour shift of the query point (R, G, B), constant over the 3x3 support
            float qdx[3] = {0, 0, 0}, qdy[3] = {0, 0, 0};
            if (lca) {
                float px = lx - lcx, py = ly - lcy, rr = (px * px + py * py) * invR2;
                float kr = p.lcaR[0] + p.lcaR[1] * rr, kb = p.lcaB[0] + p.lcaB[1] * rr;
                qdx[0] = px * kr; qdy[0] = py * kr; qdx[2] = px * kb; qdy[2] = py * kb;
            }
#if KR_NEON
            float32x4_t accN[4], accD[4];
            for (int i = 0; i < 4; i++) { accN[i] = vdupq_n_f32(0.f); accD[i] = vdupq_n_f32(0.f); }
            const float32x4_t ia = vdupq_n_f32(ic[0]), ib2 = vdupq_n_f32(2 * ic[1]), icc = vdupq_n_f32(ic[2]), zero = vdupq_n_f32(0.f);
            float32x4_t sdx[4], sdy[4];
            if (lca)
                for (int pi = 0; pi < 4; pi++) {
                    sdx[pi] = vaddq_f32(vmulq_n_f32(maskV[pi][0], qdx[0]), vmulq_n_f32(maskV[pi][2], qdx[2]));
                    sdy[pi] = vaddq_f32(vmulq_n_f32(maskV[pi][0], qdy[0]), vmulq_n_f32(maskV[pi][2], qdy[2]));
                }
#endif
            for (int k = 0; k < n; k++) {
                float r = 1.f, mx = lx, my = ly;
                if (k != ref) {
                    r = rob[k].at(qx, qy);
                    if (r <= 0.f) continue;
                    const float* d = &fl[((size_t)k * qw + qx) * 2];
                    mx += 2 * d[0]; my += 2 * d[1];
                }
                if (mx < 0 || my < 0 || mx >= W || my >= H) continue;
                const Raw& F = frames[k];
                int cx = (int)mx, cy = (int)my;
                float mj = mx - 0.5f, mi = my - 0.5f;
#if KR_NEON
                if (lca && cx >= 1 && cx + 2 < W && cy >= 1 && cy + 1 < H) {
                    const float32x4_t ddx0 = vaddq_f32(vdupq_n_f32(cx - 1 - mj), lane);
                    int pc = (cx - 1) & 1;
                    for (int i = cy - 1; i <= cy + 1; i++) {
                        int pi = (i & 1) * 2 + pc;
                        float32x4_t ddx = vsubq_f32(ddx0, sdx[pi]), ddy = vsubq_f32(vdupq_n_f32(i - mi), sdy[pi]);
                        float32x4_t z = vfmaq_f32(vmulq_f32(vmulq_f32(ia, ddx), ddx), vmulq_f32(ib2, ddx), ddy);
                        z = vfmaq_f32(z, vmulq_f32(icc, ddy), ddy);
                        float32x4_t w = vmulq_n_f32(exp_neg(vmulq_n_f32(vmaxq_f32(z, zero), -0.5f)), r);
                        float32x4_t v = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vld1_u16(F.row(i) + cx - 1))), blackV[pi]);
                        accN[pi] = vfmaq_f32(accN[pi], w, v);
                        accD[pi] = vaddq_f32(accD[pi], w);
                    }
                    continue;
                }
                if (cx >= 1 && cx + 2 < W && cy >= 1 && cy + 1 < H) {
                    float32x4_t ddx = vaddq_f32(vdupq_n_f32(cx - 1 - mj), lane);
                    float32x4_t ax2 = vmulq_f32(vmulq_f32(ia, ddx), ddx);
                    float32x4_t bx = vmulq_f32(ib2, ddx);
                    int pc = (cx - 1) & 1;
                    for (int i = cy - 1; i <= cy + 1; i++) {
                        float ddy = i - mi;
                        float32x4_t z = vfmaq_n_f32(vaddq_f32(ax2, vdupq_n_f32(ic[2] * ddy * ddy)), bx, ddy);
                        float32x4_t w = vmulq_n_f32(exp_neg(vmulq_n_f32(vmaxq_f32(z, zero), -0.5f)), r);
                        int pi = (i & 1) * 2 + pc;
                        float32x4_t v = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vld1_u16(F.row(i) + cx - 1))), blackV[pi]);
                        accN[pi] = vfmaq_f32(accN[pi], w, v);
                        accD[pi] = vaddq_f32(accD[pi], w);
                    }
                    continue;
                }
#endif
                for (int di = -1; di <= 1; di++) {
                    int i = cy + di;
                    if (i < 0 || i >= H) continue;
                    for (int dj = -1; dj <= 1; dj++) {
                        int j = cx + dj;
                        if (j < 0 || j >= W) continue;
                        int ch = cfa_channel(m.cfa, j, i);
                        int col = ch == 0 ? 0 : (ch == 3 ? 2 : 1);
                        float ddx = j - mj - qdx[col], ddy = i - mi - qdy[col];
                        float z = ic[0] * ddx * ddx + 2 * ic[1] * ddx * ddy + ic[2] * ddy * ddy;
                        float w = r * std::exp(-0.5f * std::max(0.f, z));
                        num[col] += w * (F.at(j, i) - blk[ch]);
                        den[col] += w;
                    }
                }
            }
#if KR_NEON
            for (int pi = 0; pi < 4; pi++)
                for (int c = 0; c < 3; c++) {
                    num[c] += vaddvq_f32(vmulq_f32(accN[pi], maskV[pi][c]));
                    den[c] += vaddvq_f32(vmulq_f32(accD[pi], maskV[pi][c]));
                }
#endif
            for (int c = 0; c < 3; c++) o[c] = den[c] > 1e-30f ? std::max(0.f, num[c] / den[c]) : 0.f;
        }
    });

    if (st) {
        st->snrDb = snr;
        st->kDetail = p.kDetail; st->kDenoise = p.kDenoise; st->dTh = p.dTh; st->dTr = p.dTr;
        double s = 0;
        for (int k = 0; k < n; k++) if (k != ref) s += frameW[k];
        st->meanWeight = n > 1 ? s / (n - 1) : 0;
        // sub-pixel phase coverage (JSR "tap" idea): per tile, how many of the 4x4 half-pixel phase
        // bins inside one Bayer period (2 raw px) are hit by some frame, relative to what n frames allow
        const Flow& f0 = flows[ref];
        double cov = 0;
        int tiles = 0;
        for (int t = 0; t < f0.nx * f0.ny; t++) {
            uint16_t bins = 0;
            for (int k = 0; k < n; k++) {
                float px = 2 * flows[k].fx[t], py = 2 * flows[k].fy[t];
                px -= 2 * std::floor(px / 2); py -= 2 * std::floor(py / 2);
                int bx = std::min(3, (int)(px * 2)), by = std::min(3, (int)(py * 2));
                bins |= (uint16_t)(1u << (by * 4 + bx));
            }
            int hit = 0;
            for (int bit = 0; bit < 16; bit++) hit += (bins >> bit) & 1;
            cov += (double)hit / std::min(n, 16);
            tiles++;
        }
        st->phaseCoverage = tiles ? cov / tiles : 0;
    }
    return out;
}

SubpixelStats subpixel_diversity(const std::vector<Flow>& flows, int ref) {
    SubpixelStats st;
    const int n = (int)flows.size();
    if (n < 2) return st;
    const Flow& f0 = flows[ref];
    const size_t nt = (size_t)f0.nx * f0.ny;
    // a tile counts when every alternate frame's flow is known to better than 0.1 gray px (0.2 raw px)
    constexpr float kMaxErr = 0.1f;
    double spread = 0, err = 0, cov = 0;
    size_t used = 0;
    std::vector<float> scores;
    for (size_t t = 0; t < nt; t++) {
        float e = 0;
        bool ok = true;
        for (int k = 0; k < n && ok; k++) {
            if (k == ref) continue;
            if (flows[k].err.empty() || flows[k].err[t] > kMaxErr) ok = false;
            else e = std::max(e, flows[k].err[t]);
        }
        if (!ok) continue;
        // displacements in raw px (the reference contributes 0). Robust spread: median distance from the
        // per-axis median, so a single mis-registered frame does not count as diversity.
        std::vector<float> xs(n), ys(n), dev(n);
        uint16_t bins = 0;
        for (int k = 0; k < n; k++) {
            float px = 2 * flows[k].fx[t], py = 2 * flows[k].fy[t];
            xs[k] = px; ys[k] = py;
            px -= 2 * std::floor(px / 2); py -= 2 * std::floor(py / 2);
            int bx = std::min(3, (int)(px * 2)), by = std::min(3, (int)(py * 2));
            bins |= (uint16_t)(1u << (by * 4 + bx));
        }
        auto median = [](std::vector<float> v) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
        float mx = median(xs), my = median(ys);
        for (int k = 0; k < n; k++) dev[k] = std::hypot(xs[k] - mx, ys[k] - my);
        double sp = median(dev), er = 2 * e;  // both in raw px
        int hit = 0;
        for (int bit = 0; bit < 16; bit++) hit += (bins >> bit) & 1;
        double c = (double)hit / std::min(n, 16);
        // spread must clear the estimation noise by 3 sigma, and saturates at half a Bayer period
        double excess = clampf((float)((sp - 3 * er) / 0.5), 0.f, 1.f);
        scores.push_back((float)(excess * c));
        spread += sp; err += er; cov += c;
        used++;
    }
    // too few measurable tiles (flat or very noisy scene): report no diversity, i.e. stay at 1x
    if (used >= std::max<size_t>(50, nt / 200)) {
        // median over tiles: the burst-level decision should follow the majority of the frame
        std::nth_element(scores.begin(), scores.begin() + scores.size() / 2, scores.end());
        st.score = scores[scores.size() / 2];
        st.meanSpread = spread / used;
        st.meanErr = err / used;
        st.coverage = cov / used;
    }
    st.confidentTiles = (double)used / nt;
    return st;
}
