#include "merge.h"

namespace {

float g_win[2 * kStride];

struct WinInit {
    WinInit() {
        for (int u = 0; u < 2 * kStride; u++) g_win[u] = 0.5f - 0.5f * std::cos(2.f * (float)M_PI * (u + 0.5f) / (2 * kStride));
    }
} g_winInit;

// The 2x2 tiles overlapping quad q, with window weights. Window pairs sum to 1.
struct TileTaps {
    int i[2], j[2];
    float wx[2], wy[2];
};

inline TileTaps taps(int qx, int qy, const Offsets& o) {
    TileTaps t;
    int bi = qx / kStride, bj = qy / kStride;
    int ux = qx - bi * kStride, uy = qy - bj * kStride;
    t.i[0] = clampi(bi - 1, 0, o.nx - 1); t.i[1] = clampi(bi, 0, o.nx - 1);
    t.j[0] = clampi(bj - 1, 0, o.ny - 1); t.j[1] = clampi(bj, 0, o.ny - 1);
    t.wx[0] = g_win[ux + kStride]; t.wx[1] = g_win[ux];
    t.wy[0] = g_win[uy + kStride]; t.wy[1] = g_win[uy];
    return t;
}

}  // namespace

NoiseModel estimate_noise(const ImageF& gref, const ImageF& galt, const Offsets& off, float black) {
    const int kBins = 48;
    float maxS = 0;
    for (float v : gref.d) maxS = std::max(maxS, v);
    float range = std::max(1.f, maxS / 4 - black);
    std::vector<std::vector<float>> bins(kBins);
    for (int qy = 2; qy < gref.h - 2; qy += 3)
        for (int qx = 2; qx < gref.w - 2; qx += 3) {
            TileTaps t = taps(qx, qy, off);
            float a = 0;
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 2; c++) {
                    int ox = off.x(t.i[c], t.j[b]), oy = off.y(t.i[c], t.j[b]);
                    a += t.wx[c] * t.wy[b] * galt.atc(qx + ox, qy + oy);
                }
            float s = gref.at(qx, qy) / 4 - black;
            if (s < 0) continue;
            int k = std::min(kBins - 1, (int)(s / range * kBins));
            float d = gref.at(qx, qy) - a;
            bins[k].push_back(d * d);
        }
    // per-bin robust variance of the difference of two 4-pixel sums: var_d = 2 * 4 * var_raw
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int k = 0; k < kBins; k++) {
        auto& v = bins[k];
        if (v.size() < 400) continue;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        double varRaw = v[v.size() / 2] / 0.4549 / 8.0;  // median(chi2_1) = 0.4549
        double x = (k + 0.5) * range / kBins, w = (double)v.size();
        sw += w; sx += w * x; sy += w * varRaw; sxx += w * x * x; sxy += w * x * varRaw;
    }
    NoiseModel nm;
    double den = sw * sxx - sx * sx;
    if (sw > 0 && den > 0) {
        nm.a = std::max(1e-3, (sw * sxy - sx * sy) / den);
        nm.b = std::max(0.25, (sy - nm.a * sx) / sw);
    }
    return nm;
}

ImageF merge_frames(const std::vector<Raw>& frames, int ref, const std::vector<Offsets>& offs,
                    const std::vector<ImageF>& mean3, const NoiseModel& nm, float black,
                    float robustLo, float robustHi, MergeStats* st) {
    const Raw& R = frames[ref];
    int qw = R.w / 2, qh = R.h / 2, n = (int)frames.size();
    ImageF out(R.w, R.h);
    std::vector<double> rowW(qh, 0.0);
    parallel_for(qh, [&](int qy) {
        double wacc = 0;
        for (int qx = 0; qx < qw; qx++) {
            float acc[4], wsum = 1.f;
            int rx = 2 * qx, ry = 2 * qy;
            acc[0] = R.at(rx, ry); acc[1] = R.at(rx + 1, ry); acc[2] = R.at(rx, ry + 1); acc[3] = R.at(rx + 1, ry + 1);
            float mu0 = mean3[ref].at(qx, qy);
            // variance of a 3x3 mean of 4-pixel sums, for the difference of two frames
            float varMu = (float)(2.0 * 4.0 * nm.var(mu0 / 4 - black) / 9.0);
            float invSigma = 1.f / std::sqrt(std::max(varMu, 1e-3f));
            for (int k = 0; k < n; k++) {
                if (k == ref) continue;
                const Offsets& o = offs[k];
                TileTaps t = taps(qx, qy, o);
                float a[4] = {0, 0, 0, 0}, mu = 0;
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) {
                        float w = t.wx[c] * t.wy[b];
                        int sx = clampi(qx + o.x(t.i[c], t.j[b]), 0, qw - 1);
                        int sy = clampi(qy + o.y(t.i[c], t.j[b]), 0, qh - 1);
                        const Raw& F = frames[k];
                        a[0] += w * F.at(2 * sx, 2 * sy); a[1] += w * F.at(2 * sx + 1, 2 * sy);
                        a[2] += w * F.at(2 * sx, 2 * sy + 1); a[3] += w * F.at(2 * sx + 1, 2 * sy + 1);
                        mu += w * mean3[k].at(sx, sy);
                    }
                // robustness: linear ramp from robustLo to robustHi sigmas of local-mean mismatch
                float z = std::fabs(mu - mu0) * invSigma;
                float w = clampf((robustHi - z) / (robustHi - robustLo), 0.f, 1.f);
                for (int c = 0; c < 4; c++) acc[c] += w * a[c];
                wsum += w;
            }
            float inv = 1.f / wsum;
            out.at(rx, ry) = acc[0] * inv; out.at(rx + 1, ry) = acc[1] * inv;
            out.at(rx, ry + 1) = acc[2] * inv; out.at(rx + 1, ry + 1) = acc[3] * inv;
            wacc += (wsum - 1.f) / std::max(1, n - 1);
        }
        rowW[qy] = wacc / qw;
    });
    if (st) { double s = 0; for (double v : rowW) s += v; st->meanWeight = s / qh; }
    return out;
}
