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

NoiseModel NoiseCurve::at(float frameIso, float scale) const {
    NoiseModel nm;
    const int n = (int)iso.size();
    if (n == 0) return nm;
    double a0, b0;
    if (frameIso <= iso[0]) {
        a0 = a[0] * frameIso / iso[0];
        b0 = b[0];
    } else if (frameIso >= iso[n - 1]) {
        a0 = a[n - 1];
        b0 = b[n - 1];
    } else {
        int i = 0;
        while (frameIso > iso[i + 1]) i++;
        double t = (frameIso - iso[i]) / (iso[i + 1] - iso[i]);
        a0 = a[i] + t * (a[i + 1] - a[i]);
        b0 = b[i] + t * (b[i + 1] - b[i]);
    }
    nm.a = a0 * scale;
    nm.b = b0 * scale * scale;
    return nm;
}

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

void min_filter(ImageF& img, int r) {
    if (r <= 0) return;
    const int w = img.w, h = img.h;
    ImageF tmp(w, h);
    parallel_for(h, [&](int y) {
        const float* s = img.row(y);
        float* d = tmp.row(y);
        for (int x = 0; x < w; x++) {
            float v = s[x];
            for (int u = std::max(0, x - r); u <= std::min(w - 1, x + r); u++) v = std::min(v, s[u]);
            d[x] = v;
        }
    });
    parallel_for(h, [&](int y) {
        float* d = img.row(y);
        for (int x = 0; x < w; x++) {
            float v = tmp.at(x, y);
            for (int t = std::max(0, y - r); t <= std::min(h - 1, y + r); t++) v = std::min(v, tmp.at(x, t));
            d[x] = v;
        }
    });
}

bool scale_to_reference(std::vector<Raw>& frames, int ref, const Meta& m, MergeOpts& o, float& refWhite) {
    const int n = (int)frames.size();
    std::vector<double> e(n);
    for (int k = 0; k < n; k++)
        e[k] = m.exposureNs[std::min(k, (int)m.exposureNs.size() - 1)] * (m.iso.empty() ? 100 : m.iso[std::min(k, (int)m.iso.size() - 1)]);
    double eMax = *std::max_element(e.begin(), e.end()), eMin = *std::min_element(e.begin(), e.end());
    refWhite = (float)m.white;
    if (eMax < eMin * 1.1) return false;
    o.gain.assign(n, 1.f);
    o.clip.assign(n, 1e30f);
    // frame k gets factor eMax / e[k] >= 1 (exact for the longest); the reference ends up at S = eMax / e[ref]
    const float black = (m.black[0] + m.black[1] + m.black[2] + m.black[3]) / 4;
    for (int k = 0; k < n; k++) {
        float f = (float)(eMax / e[k]);
        o.gain[k] = (float)(e[ref] / e[k]);
        o.clip[k] = (float)((m.white * 0.98 - black) * f + black);
        if (f == 1.f) continue;
        Raw& F = frames[k];
        F.make_writable();
        parallel_for(F.h, [&](int y) {
            uint16_t* p = F.row(y);
            for (int x = 0; x < F.w; x++) {
                float v = (p[x] - black) * f + black;
                p[x] = (uint16_t)clampi((int)std::lround(v), 0, 65535);
            }
        });
    }
    refWhite = (float)((m.white - black) * (eMax / e[ref]) + black);
    return true;
}

ImageF merge_frames(const std::vector<Raw>& frames, int ref, const std::vector<Offsets>& offs,
                    const std::vector<ImageF>& mean3, const NoiseModel& nm, float black,
                    const MergeOpts& o, MergeStats* st) {
    const Raw& R = frames[ref];
    const int qw = R.w / 2, qh = R.h / 2, n = (int)frames.size();
    ImageF out(R.w, R.h), wsum(qw, qh);
    parallel_for(qh, [&](int qy) {
        for (int qx = 0; qx < qw; qx++) {
            int rx = 2 * qx, ry = 2 * qy;
            out.at(rx, ry) = R.at(rx, ry) - black; out.at(rx + 1, ry) = R.at(rx + 1, ry) - black;
            out.at(rx, ry + 1) = R.at(rx, ry + 1) - black; out.at(rx + 1, ry + 1) = R.at(rx + 1, ry + 1) - black;
            wsum.at(qx, qy) = 1.f;
        }
    });
    double wTotal = 0;
    ImageF rob(qw, qh);
    for (int k = 0; k < n; k++) {
        if (k == ref) continue;
        const Offsets& of = offs[k];
        const Raw& F = frames[k];
        const float g = o.g(k), clip = o.clipAt(k);
        // pass 1: robustness per quad; pass 2: dilate the rejections; pass 3: accumulate
        parallel_for(qh, [&](int qy) {
            for (int qx = 0; qx < qw; qx++) {
                TileTaps t = taps(qx, qy, of);
                float mu = 0;
                bool clipped = false;
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) {
                        int sx = clampi(qx + of.x(t.i[c], t.j[b]), 0, qw - 1);
                        int sy = clampi(qy + of.y(t.i[c], t.j[b]), 0, qh - 1);
                        mu += t.wx[c] * t.wy[b] * mean3[k].at(sx, sy);
                        clipped |= std::max(std::max(F.at(2 * sx, 2 * sy), F.at(2 * sx + 1, 2 * sy)),
                                            std::max(F.at(2 * sx, 2 * sy + 1), F.at(2 * sx + 1, 2 * sy + 1))) >= clip;
                    }
                rob.at(qx, qy) = clipped ? 0.f : frame_weight(mean3[ref].at(qx, qy), mu, black, nm, g, o);
            }
        });
        min_filter(rob, o.dilate);
        std::vector<double> rowW(qh, 0.0);
        parallel_for(qh, [&](int qy) {
            double acc = 0;
            for (int qx = 0; qx < qw; qx++) {
                float w = rob.at(qx, qy);
                if (w <= 0) continue;
                TileTaps t = taps(qx, qy, of);
                float a[4] = {0, 0, 0, 0};
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) {
                        float tw = t.wx[c] * t.wy[b];
                        int sx = clampi(qx + of.x(t.i[c], t.j[b]), 0, qw - 1);
                        int sy = clampi(qy + of.y(t.i[c], t.j[b]), 0, qh - 1);
                        a[0] += tw * F.at(2 * sx, 2 * sy); a[1] += tw * F.at(2 * sx + 1, 2 * sy);
                        a[2] += tw * F.at(2 * sx, 2 * sy + 1); a[3] += tw * F.at(2 * sx + 1, 2 * sy + 1);
                    }
                int rx = 2 * qx, ry = 2 * qy;
                out.at(rx, ry) += w * (a[0] - black); out.at(rx + 1, ry) += w * (a[1] - black);
                out.at(rx, ry + 1) += w * (a[2] - black); out.at(rx + 1, ry + 1) += w * (a[3] - black);
                wsum.at(qx, qy) += w;
                acc += std::min(w, 1.f);
            }
            rowW[qy] = acc;
        });
        for (double v : rowW) wTotal += v;
    }
    parallel_for(qh, [&](int qy) {
        for (int qx = 0; qx < qw; qx++) {
            float inv = 1.f / wsum.at(qx, qy);
            int rx = 2 * qx, ry = 2 * qy;
            out.at(rx, ry) = out.at(rx, ry) * inv + black; out.at(rx + 1, ry) = out.at(rx + 1, ry) * inv + black;
            out.at(rx, ry + 1) = out.at(rx, ry + 1) * inv + black; out.at(rx + 1, ry + 1) = out.at(rx + 1, ry + 1) * inv + black;
        }
    });
    if (st) st->meanWeight = n > 1 ? wTotal / ((double)qw * qh * (n - 1)) : 0;
    return out;
}
