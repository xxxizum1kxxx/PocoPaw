#include "merge.h"

#include <cstring>
#if defined(__ARM_NEON) && !defined(BURSTPIPE_NO_NEON)
#include <arm_neon.h>
#define MERGE_NEON 1
#endif

namespace {

#ifdef MERGE_NEON
// the 2x2 Bayer quad at (2 sx, 2 sy) as {top-left, top-right, bottom-left, bottom-right}
inline float32x4_t load_quad(const Raw& F, int sx, int sy) {
    const uint16_t* p = F.row(2 * sy) + 2 * sx;
    uint32_t a, b;
    memcpy(&a, p, 4);
    memcpy(&b, p + F.w, 4);
    return vcvtq_f32_u32(vmovl_u16(vreinterpret_u16_u32(vset_lane_u32(b, vdup_n_u32(a), 1))));
}
#endif

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

#ifdef MERGE_NEON
// merge_frames pass 1 for quads qx..qx+3 (qx a multiple of 4, so one tile column); false where a tap
// reaches past the left / right edge and the scalar path has to clamp per quad
bool rob_quads4(ImageF& rob, int qx, int qy, const Offsets& of, const ImageF& m3, const ImageF& m3ref, float black,
                const NoiseModel& nm, float g, const MergeOpts& o) {
    const int qw = rob.w, qh = rob.h;
    TileTaps t = taps(qx, qy, of);
    const int ux = qx % kStride;
    const float32x4_t wx[2] = {vld1q_f32(g_win + ux + kStride), vld1q_f32(g_win + ux)};
    float32x4_t mu = vdupq_n_f32(0);
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++) {
            int sx = qx + of.x(t.i[c], t.j[b]);
            if (sx < 0 || sx + 4 > qw) return false;
            int sy = clampi(qy + of.y(t.i[c], t.j[b]), 0, qh - 1);
            mu = vfmaq_f32(mu, vmulq_n_f32(wx[c], t.wy[b]), vld1q_f32(m3.row(sy) + sx));
        }
    const float a = (float)nm.a, bb = (float)nm.b;
    float32x4_t mu0 = vld1q_f32(m3ref.row(qy) + qx);
    float32x4_t sig = vmaxq_f32(vdupq_n_f32(0.f), vsubq_f32(vmulq_n_f32(mu0, 0.25f), vdupq_n_f32(black)));
    float32x4_t vRef = vfmaq_n_f32(vdupq_n_f32(bb), sig, a), vAlt = vfmaq_n_f32(vdupq_n_f32(g * g * bb), sig, g * a);
    float32x4_t varMu = vmulq_n_f32(vaddq_f32(vRef, vAlt), 4.f / 9.f);
    float32x4_t z = vdivq_f32(vabdq_f32(mu, mu0), vsqrtq_f32(vmaxq_f32(varMu, vdupq_n_f32(1e-3f))));
    float32x4_t ramp = vmulq_n_f32(vsubq_f32(vdupq_n_f32(o.robustHi), z), 1.f / (o.robustHi - o.robustLo));
    ramp = vminq_f32(vmaxq_f32(ramp, vdupq_n_f32(0.f)), vdupq_n_f32(1.f));
    vst1q_f32(rob.row(qy) + qx, vmulq_f32(ramp, vdivq_f32(vRef, vAlt)));
    return true;
}
#endif

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
    // offsets in the outer loop so the inner loops run over contiguous pixels (vectorised)
    parallel_for(h, [&](int y) {
        const float* s = img.row(y);
        float* d = tmp.row(y);
        std::copy(s, s + w, d);
        for (int u = 1; u <= r; u++)
            for (int x = 0; x < w; x++) d[x] = std::min(d[x], std::min(s[std::max(0, x - u)], s[std::min(w - 1, x + u)]));
    });
    parallel_for(h, [&](int y) {
        float* d = img.row(y);
        std::copy(tmp.row(y), tmp.row(y) + w, d);
        for (int u = 1; u <= r; u++) {
            const float* a = tmp.row(std::max(0, y - u));
            const float* b = tmp.row(std::min(h - 1, y + u));
            for (int x = 0; x < w; x++) d[x] = std::min(d[x], std::min(a[x], b[x]));
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
        const bool checkClip = !o.clip.empty();
        // pass 1: robustness per quad; pass 2: dilate the rejections; pass 3: accumulate
        parallel_for(qh, [&](int qy) {
            for (int qx = 0; qx < qw; qx++) {
#ifdef MERGE_NEON
                // 4 quads in one tile column share the offsets, so their 3x3 means are contiguous
                if (!checkClip && (qx & 3) == 0 && qx + 4 <= qw && rob_quads4(rob, qx, qy, of, mean3[k], mean3[ref], black, nm, g, o)) {
                    qx += 3;
                    continue;
                }
#endif
                TileTaps t = taps(qx, qy, of);
                float mu = 0;
                bool clipped = false;
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) {
                        int sx = clampi(qx + of.x(t.i[c], t.j[b]), 0, qw - 1);
                        int sy = clampi(qy + of.y(t.i[c], t.j[b]), 0, qh - 1);
                        mu += t.wx[c] * t.wy[b] * mean3[k].at(sx, sy);
                        if (!checkClip) continue;
#ifdef MERGE_NEON
                        clipped |= vmaxvq_f32(load_quad(F, sx, sy)) >= clip;
#else
                        clipped |= std::max(std::max(F.at(2 * sx, 2 * sy), F.at(2 * sx + 1, 2 * sy)),
                                            std::max(F.at(2 * sx, 2 * sy + 1), F.at(2 * sx + 1, 2 * sy + 1))) >= clip;
#endif
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
                int rx = 2 * qx, ry = 2 * qy;
#ifdef MERGE_NEON
                float32x4_t q = vdupq_n_f32(0);
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) {
                        int sx = clampi(qx + of.x(t.i[c], t.j[b]), 0, qw - 1);
                        int sy = clampi(qy + of.y(t.i[c], t.j[b]), 0, qh - 1);
                        q = vfmaq_n_f32(q, load_quad(F, sx, sy), t.wx[c] * t.wy[b]);
                    }
                float32x4_t d = vmulq_n_f32(vsubq_f32(q, vdupq_n_f32(black)), w);
                float* o0 = out.row(ry) + rx;
                float* o1 = out.row(ry + 1) + rx;
                vst1_f32(o0, vadd_f32(vld1_f32(o0), vget_low_f32(d)));
                vst1_f32(o1, vadd_f32(vld1_f32(o1), vget_high_f32(d)));
#else
                float a[4] = {0, 0, 0, 0};
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) {
                        float tw = t.wx[c] * t.wy[b];
                        int sx = clampi(qx + of.x(t.i[c], t.j[b]), 0, qw - 1);
                        int sy = clampi(qy + of.y(t.i[c], t.j[b]), 0, qh - 1);
                        a[0] += tw * F.at(2 * sx, 2 * sy); a[1] += tw * F.at(2 * sx + 1, 2 * sy);
                        a[2] += tw * F.at(2 * sx, 2 * sy + 1); a[3] += tw * F.at(2 * sx + 1, 2 * sy + 1);
                    }
                out.at(rx, ry) += w * (a[0] - black); out.at(rx + 1, ry) += w * (a[1] - black);
                out.at(rx, ry + 1) += w * (a[2] - black); out.at(rx + 1, ry + 1) += w * (a[3] - black);
#endif
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
