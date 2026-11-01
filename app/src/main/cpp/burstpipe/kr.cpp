#include "kr.h"

namespace {

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

Flow refine_flow(const ImageF& mref, const ImageF& malt, const Offsets& o, int iters) {
    Flow f = zero_flow(o);
    parallel_for(o.ny, [&](int j) {
        for (int i = 0; i < o.nx; i++) {
            float dx = o.x(i, j), dy = o.y(i, j);
            int x0 = std::max(1, i * kStride), x1 = std::min(mref.w - 1, i * kStride + kTile);
            int y0 = std::max(1, j * kStride), y1 = std::min(mref.h - 1, j * kStride + kTile);
            double a = 0, b = 0, c = 0;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++) {
                    float gx = 0.5f * (mref.atc(x + 1, y) - mref.at(x - 1, y));
                    float gy = 0.5f * (mref.atc(x, y + 1) - mref.at(x, y - 1));
                    a += gx * gx; b += gx * gy; c += gy * gy;
                }
            double det = a * c - b * b, tr = a + c;
            if (x1 <= x0 || y1 <= y0 || tr <= 0 || det < 1e-4 * tr * tr) {
                f.fx[(size_t)j * o.nx + i] = dx; f.fy[(size_t)j * o.nx + i] = dy;
                continue;
            }
            for (int it = 0; it < iters; it++) {
                double bx = 0, by = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) {
                        float gx = 0.5f * (mref.atc(x + 1, y) - mref.at(x - 1, y));
                        float gy = 0.5f * (mref.atc(x, y + 1) - mref.at(x, y - 1));
                        float r = bilinear(malt, x + dx, y + dy) - mref.at(x, y);
                        bx += gx * r; by += gy * r;
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
                const NoiseModel& nm, float robustLo, float robustHi, KrParams p, KrStats* st) {
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

    // per-frame robustness at quad resolution: local-mean mismatch against the noise model,
    // then a 3x3 minimum so a rejection also covers its neighbourhood
    std::vector<ImageF> rob(n);
    std::vector<double> frameW(n, 1.0);
    for (int k = 0; k < n; k++) {
        if (k == ref) continue;
        ImageF r0(qw, qh);
        parallel_for(qh, [&](int qy) {
            for (int qx = 0; qx < qw; qx++) {
                float mu0 = mean3[ref].at(qx, qy);
                float varMu = (float)(2.0 * 4.0 * nm.var(mu0 / 4 - black) / 9.0);
                float dx, dy;
                flow_at(flows[k], qx, qy, dx, dy);
                float mu = bilinear(mean3[k], qx + dx, qy + dy);
                float z = std::fabs(mu - mu0) / std::sqrt(std::max(varMu, 1e-3f));
                r0.at(qx, qy) = clampf((robustHi - z) / (robustHi - robustLo), 0.f, 1.f);
            }
        });
        rob[k] = ImageF(qw, qh);
        std::vector<double> rowSum(qh, 0.0);
        parallel_for(qh, [&](int qy) {
            double s = 0;
            for (int qx = 0; qx < qw; qx++) {
                float v = 1.f;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) v = std::min(v, r0.atc(qx + dx, qy + dy));
                rob[k].at(qx, qy) = v;
                s += v;
            }
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
    parallel_for(OH, [&](int Y) {
        float* o = out.row(Y);
        float ly = (Y + 0.5f) / sc;
        int qy = clampi((int)(ly / 2), 0, qh - 1);
        int ky = clampi((int)std::lround(ly / 2 - 0.5f), 0, qh - 1);
        for (int X = 0; X < OW; X++, o += 3) {
            float lx = (X + 0.5f) / sc;
            int qx = clampi((int)(lx / 2), 0, qw - 1);
            int kx = clampi((int)std::lround(lx / 2 - 0.5f), 0, qw - 1);
            const float* ic = &icov.at(kx, ky);
            float num[3] = {0, 0, 0}, den[3] = {0, 0, 0};
            for (int k = 0; k < n; k++) {
                float r = 1.f, mx = lx, my = ly;
                if (k != ref) {
                    r = rob[k].at(qx, qy);
                    if (r <= 0.f) continue;
                    float dx, dy;
                    flow_at(flows[k], qx, qy, dx, dy);
                    mx += 2 * dx; my += 2 * dy;
                }
                if (mx < 0 || my < 0 || mx >= W || my >= H) continue;
                const Raw& F = frames[k];
                int cx = (int)mx, cy = (int)my;
                float mj = mx - 0.5f, mi = my - 0.5f;
                for (int di = -1; di <= 1; di++) {
                    int i = cy + di;
                    if (i < 0 || i >= H) continue;
                    float ddy = i - mi;
                    for (int dj = -1; dj <= 1; dj++) {
                        int j = cx + dj;
                        if (j < 0 || j >= W) continue;
                        float ddx = j - mj;
                        float z = ic[0] * ddx * ddx + 2 * ic[1] * ddx * ddy + ic[2] * ddy * ddy;
                        float w = r * std::exp(-0.5f * std::max(0.f, z));
                        int ch = cfa_channel(m.cfa, j, i);
                        int col = ch == 0 ? 0 : (ch == 3 ? 2 : 1);
                        num[col] += w * (F.at(j, i) - blk[ch]);
                        den[col] += w;
                    }
                }
            }
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
