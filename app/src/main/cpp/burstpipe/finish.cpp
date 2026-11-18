#include "finish.h"

#include <csetjmp>
#include <cstdio>
#include <cstring>

#include "third_party/libultrahdr/include/jpeglib.h"

namespace {

float srgb_oetf_exact(float x) { return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f; }
float srgb_eotf_exact(float y) { return y <= 0.04045f ? y / 12.92f : std::pow((y + 0.055f) / 1.055f, 2.4f); }

// 1D LUTs with linear interpolation; the OETF table is indexed in sqrt space for shadow precision
struct Lut {
    static const int N = 4096;
    float oetf[N + 1], eotf[N + 1];
    Lut() {
        for (int i = 0; i <= N; i++) {
            float t = (float)i / N;
            oetf[i] = srgb_oetf_exact(t * t);
            eotf[i] = srgb_eotf_exact(t);
        }
    }
} g_lut;

inline float lut_interp(const float* T, float t) {
    float f = clampf(t, 0.f, 1.f) * Lut::N;
    int i = std::min((int)f, Lut::N - 1);
    return T[i] + (f - i) * (T[i + 1] - T[i]);
}
inline float srgb_oetf(float x) { return lut_interp(g_lut.oetf, std::sqrt(clampf(x, 0.f, 1.f))); }
inline float srgb_eotf(float y) { return lut_interp(g_lut.eotf, y); }
inline float luma709(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

inline float fast_cbrt(float x) {
    if (x <= 1e-30f) return 0.f;
    uint32_t i;
    memcpy(&i, &x, 4);
    i = i / 3u + 0x2a517d47u;
    float y;
    memcpy(&y, &i, 4);
    y = (2.f * y + x / (y * y)) * (1.f / 3.f);
    y = (2.f * y + x / (y * y)) * (1.f / 3.f);
    y = (2.f * y + x / (y * y)) * (1.f / 3.f);
    return y;
}

inline float sgn_cbrt(float x) {
    float a = fast_cbrt(std::fabs(x));
    return x < 0.f ? -a : a;
}

inline void linear_srgb_to_oklab(float r, float g, float b, float& L, float& a, float& ob) {
    float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;
    float l_ = sgn_cbrt(l), m_ = sgn_cbrt(m), s_ = sgn_cbrt(s);
    L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
    a = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
    ob = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
}

inline void oklab_to_linear_srgb(float L, float a, float ob, float& r, float& g, float& b) {
    float l_ = L + 0.3963377774f * a + 0.2158037573f * ob;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * ob;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * ob;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    r = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
}

inline float oklab_max_saturation(float a, float b) {
    float c1 = -1.88170328f * a - 0.80936493f * b;
    float c2 = +1.81444104f * a - 1.19445276f * b;
    float c3 = +0.13110758f * a + 1.81333944f * b;
    float k0, k1, k2, k3, k4, wl, wm, ws;
    if (c1 >= c2 && c1 >= c3) {
        k0 = +1.19086277f; k1 = +1.76576728f; k2 = +0.59662641f; k3 = +0.75515197f; k4 = +0.56771245f;
        wl = +4.0767416621f; wm = -3.3077115913f; ws = +0.2309699292f;
    } else if (c2 >= c3) {
        k0 = +0.73956515f; k1 = -0.45954404f; k2 = +0.08285427f; k3 = +0.12541070f; k4 = +0.14503204f;
        wl = -1.2684380046f; wm = +2.6097574011f; ws = -0.3413193965f;
    } else {
        k0 = +1.35733652f; k1 = -0.00915799f; k2 = -1.15130210f; k3 = -0.50559606f; k4 = +0.00692167f;
        wl = -0.0041960863f; wm = -0.7034186147f; ws = +1.7076147010f;
    }
    float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
    float kl = +0.3963377774f * a + 0.2158037573f * b;
    float km = -0.1055613458f * a - 0.0638541728f * b;
    float ks = -0.0894841775f * a - 1.2914855480f * b;
    for (int it = 0; it < 2; it++) {
        float l_ = 1.f + S * kl, m_ = 1.f + S * km, s_ = 1.f + S * ks;
        float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
        float l1 = 3.f * kl * l_ * l_, m1 = 3.f * km * m_ * m_, s1 = 3.f * ks * s_ * s_;
        float l2 = 6.f * kl * kl * l_, m2 = 6.f * km * km * m_, s2 = 6.f * ks * ks * s_;
        float f = wl * l + wm * m + ws * s;
        float f1 = wl * l1 + wm * m1 + ws * s1;
        float f2 = wl * l2 + wm * m2 + ws * s2;
        float den = f1 * f1 - 0.5f * f * f2;
        if (std::fabs(den) > 1e-12f) S -= f * f1 / den;
    }
    return S;
}

inline float oklab_max_chroma(float L, float a_, float b_) {
    if (L <= 0.f || L >= 1.f) return 0.f;
    float Smax = oklab_max_saturation(a_, b_);
    float kl = +0.3963377774f * a_ + 0.2158037573f * b_;
    float km = -0.1055613458f * a_ - 0.0638541728f * b_;
    float ks = -0.0894841775f * a_ - 1.2914855480f * b_;
    float l_ = 1.f + Smax * kl, m_ = 1.f + Smax * km, s_ = 1.f + Smax * ks;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    float r0 = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    float g0 = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    float b0 = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
    float Lcusp = fast_cbrt(1.f / std::max(1e-6f, std::max(r0, std::max(g0, b0))));
    float Ccusp = Lcusp * Smax;
    if (L <= Lcusp) return std::max(0.f, L * Smax);
    float t = Ccusp * (1.f - L) / std::max(1.f - Lcusp, 1e-6f);
    for (int it = 0; it < 2; it++) {
        l_ = L + t * kl; m_ = L + t * km; s_ = L + t * ks;
        l = l_ * l_ * l_; m = m_ * m_ * m_; s = s_ * s_ * s_;
        float ldt = 3.f * kl * l_ * l_, mdt = 3.f * km * m_ * m_, sdt = 3.f * ks * s_ * s_;
        float ldt2 = 6.f * kl * kl * l_, mdt2 = 6.f * km * km * m_, sdt2 = 6.f * ks * ks * s_;
        float rr = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
        float gg = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
        float bb = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
        float f, f1, f2;
        if (rr >= gg && rr >= bb) {
            f = rr - 1.f;
            f1 = +4.0767416621f * ldt - 3.3077115913f * mdt + 0.2309699292f * sdt;
            f2 = +4.0767416621f * ldt2 - 3.3077115913f * mdt2 + 0.2309699292f * sdt2;
        } else if (gg >= bb) {
            f = gg - 1.f;
            f1 = -1.2684380046f * ldt + 2.6097574011f * mdt - 0.3413193965f * sdt;
            f2 = -1.2684380046f * ldt2 + 2.6097574011f * mdt2 - 0.3413193965f * sdt2;
        } else {
            f = bb - 1.f;
            f1 = -0.0041960863f * ldt - 0.7034186147f * mdt + 1.7076147010f * sdt;
            f2 = -0.0041960863f * ldt2 - 0.7034186147f * mdt2 + 1.7076147010f * sdt2;
        }
        float den = f1 * f1 - 0.5f * f * f2;
        if (std::fabs(den) > 1e-12f) t -= f * f1 / den;
    }
    return std::max(0.f, t);
}

inline float oklab_compress_chroma(float C, float Cmax) {
    constexpr float kKneeFrac = 0.85f;
    constexpr float kCapFrac = 0.996f;
    float knee = kKneeFrac * Cmax;
    if (C <= knee) return C;
    float cap = kCapFrac * Cmax;
    float span = std::max(cap - knee, 1e-6f);
    float ex = (C - knee) / span;
    return knee + span * (ex / (1.f + ex));
}

inline void oklab_gamut_compress(float& r, float& g, float& b) {
    float mn = std::min(r, std::min(g, b)), mx = std::max(r, std::max(g, b));
    // Deeply in-gamut linear RGB satisfies C <= 0.85 * Cmax, where compression is an exact no-op.
    if (mn >= 0.f && mx <= 1.f && mn >= 0.25f * mx && (1.f - mx) >= 0.25f * (1.f - mn)) return;
    float L, a, ob;
    linear_srgb_to_oklab(r, g, b, L, a, ob);
    if (L <= 0.f) { r = g = b = 0.f; return; }
    if (L >= 1.f) { r = g = b = 1.f; return; }
    float C = std::sqrt(a * a + ob * ob);
    if (C <= 1e-6f) {
        float v = clampf(L * L * L, 0.f, 1.f);
        r = g = b = v;
        return;
    }
    float a_ = a / C, b_ = ob / C;
    float Cmax = oklab_max_chroma(L, a_, b_);
    float Cnew = oklab_compress_chroma(C, Cmax);
    if (Cnew != C) oklab_to_linear_srgb(L, Cnew * a_, Cnew * b_, r, g, b);
    r = clampf(r, 0.f, 1.f);
    g = clampf(g, 0.f, 1.f);
    b = clampf(b, 0.f, 1.f);
}

double illuminant_cct(int code) {
    switch (code) {
        case 1: case 4: case 9: return 5500.0;
        case 2: return 4000.0;
        case 3: return 2850.0;
        case 10: return 6500.0;
        case 11: return 7500.0;
        case 12: return 6430.0;
        case 13: return 5000.0;
        case 14: return 4150.0;
        case 15: return 3450.0;
        case 17: return 2856.0;
        case 18: return 4874.0;
        case 19: return 6774.0;
        case 20: return 5503.0;
        case 21: return 6504.0;
        case 22: return 7504.0;
        case 23: return 5003.0;
        case 24: return 3200.0;
        default: return 0.0;
    }
}

double xy_to_cct_robertson(double x, double y) {
    static const double kRt[31][4] = {
        {0, 0.18006, 0.26352, -0.24341},   {10, 0.18066, 0.26589, -0.25479},  {20, 0.18133, 0.26846, -0.26876},
        {30, 0.18208, 0.27119, -0.28539},  {40, 0.18293, 0.27407, -0.30470},  {50, 0.18388, 0.27709, -0.32675},
        {60, 0.18494, 0.28021, -0.35156},  {70, 0.18611, 0.28342, -0.37915},  {80, 0.18740, 0.28668, -0.40955},
        {90, 0.18880, 0.28997, -0.44278},  {100, 0.19032, 0.29326, -0.47888}, {125, 0.19462, 0.30141, -0.58204},
        {150, 0.19962, 0.30921, -0.70471}, {175, 0.20525, 0.31647, -0.84901}, {200, 0.21142, 0.32312, -1.0182},
        {225, 0.21807, 0.32909, -1.2168},  {250, 0.22511, 0.33439, -1.4512},  {275, 0.23247, 0.33904, -1.7298},
        {300, 0.24010, 0.34308, -2.0637},  {325, 0.24702, 0.34655, -2.4681},  {350, 0.25591, 0.34951, -2.9641},
        {375, 0.26400, 0.35200, -3.5814},  {400, 0.27218, 0.35407, -4.3633},  {425, 0.28039, 0.35577, -5.3762},
        {450, 0.28863, 0.35714, -6.7262},  {475, 0.29685, 0.35823, -8.5955},  {500, 0.30505, 0.35907, -11.324},
        {525, 0.31320, 0.35968, -15.628},  {550, 0.32129, 0.36011, -23.325},  {575, 0.32931, 0.36038, -40.770},
        {600, 0.33724, 0.36051, -116.45},
    };
    const double den = -2.0 * x + 12.0 * y + 3.0;
    if (std::fabs(den) < 1e-12) return 5000.0;
    const double u = 4.0 * x / den, v = 6.0 * y / den;
    double prevD = 0.0;
    for (int i = 0; i < 31; i++) {
        const double t = kRt[i][3];
        const double d = ((v - kRt[i][2]) - t * (u - kRt[i][1])) / std::hypot(1.0, t);
        if (i > 0 && ((d >= 0.0 && prevD <= 0.0) || (d <= 0.0 && prevD >= 0.0))) {
            const double f = std::fabs(prevD) / (std::fabs(prevD) + std::fabs(d) + 1e-12);
            const double mired = kRt[i - 1][0] + f * (kRt[i][0] - kRt[i - 1][0]);
            return 1e6 / std::max(mired, 1.0);
        }
        prevD = d;
    }
    return prevD > 0.0 ? 1e6 / 600.0 : 50000.0;
}

double det3(const double m[9]) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

bool invert3(const double m[9], double inv[9]) {
    double d = det3(m);
    if (!std::isfinite(d) || std::fabs(d) < 1e-7) return false;
    double id = 1.0 / d;
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * id;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * id;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * id;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * id;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * id;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * id;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * id;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * id;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * id;
    return true;
}

void mul3(const double a[9], const double b[9], double o[9]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
}

void mul3v(const double m[9], const double v[3], double o[3]) {
    for (int r = 0; r < 3; r++) o[r] = m[r * 3] * v[0] + m[r * 3 + 1] * v[1] + m[r * 3 + 2] * v[2];
}

bool mat_valid3(const float f[9], double out[9]) {
    for (int i = 0; i < 9; i++) {
        if (!std::isfinite(f[i])) return false;
        out[i] = f[i];
    }
    return std::fabs(det3(out)) > 1e-6;
}

// 5-tap binomial pyramid (Burt & Adelson)
ImageF pyr_down(const ImageF& in) {
    int w = (in.w + 1) / 2, h = (in.h + 1) / 2;
    ImageF t(w, in.h), o(w, h);
    parallel_for(in.h, [&](int y) {
        for (int x = 0; x < w; x++) {
            int X = 2 * x;
            t.at(x, y) = (in.atc(X - 2, y) + 4 * in.atc(X - 1, y) + 6 * in.atc(X, y) + 4 * in.atc(X + 1, y) + in.atc(X + 2, y)) * (1.f / 16);
        }
    });
    parallel_for(h, [&](int y) {
        int Y = 2 * y;
        for (int x = 0; x < w; x++)
            o.at(x, y) = (t.atc(x, Y - 2) + 4 * t.atc(x, Y - 1) + 6 * t.atc(x, Y) + 4 * t.atc(x, Y + 1) + t.atc(x, Y + 2)) * (1.f / 16);
    });
    return o;
}

ImageF pyr_up(const ImageF& in, int w, int h) {
    ImageF t(w, in.h), o(w, h);
    parallel_for(in.h, [&](int y) {
        for (int x = 0; x < w; x++) {
            int i = x >> 1;
            t.at(x, y) = (x & 1) ? 0.5f * (in.atc(i, y) + in.atc(i + 1, y))
                                 : (in.atc(i - 1, y) + 6 * in.atc(i, y) + in.atc(i + 1, y)) * 0.125f;
        }
    });
    parallel_for(h, [&](int y) {
        int j = y >> 1;
        for (int x = 0; x < w; x++)
            o.at(x, y) = (y & 1) ? 0.5f * (t.atc(x, j) + t.atc(x, j + 1))
                                 : (t.atc(x, j - 1) + 6 * t.atc(x, j) + t.atc(x, j + 1)) * 0.125f;
    });
    return o;
}

// a -= s * b
void sub_inplace(ImageF& a, const ImageF& b, float s = 1.f) { parallel_for(a.h, [&](int y) { float* p = a.row(y); const float* q = b.row(y); for (int x = 0; x < a.w * a.c; x++) p[x] -= s * q[x]; }); }

ImageF gaussian_blur(const ImageF& in, float sigma) {
    int r = std::max(1, (int)std::ceil(2.5f * sigma));
    std::vector<float> k(2 * r + 1);
    float s = 0;
    for (int i = -r; i <= r; i++) s += k[i + r] = std::exp(-0.5f * i * i / (sigma * sigma));
    for (float& v : k) v /= s;
    ImageF t(in.w, in.h), o(in.w, in.h);
    parallel_for(in.h, [&](int y) {
        for (int x = 0; x < in.w; x++) { float a = 0; for (int i = -r; i <= r; i++) a += k[i + r] * in.atc(x + i, y); t.at(x, y) = a; }
    });
    parallel_for(in.h, [&](int y) {
        for (int x = 0; x < in.w; x++) { float a = 0; for (int i = -r; i <= r; i++) a += k[i + r] * t.atc(x, y + i); o.at(x, y) = a; }
    });
    return o;
}

// box mean of a single-channel image, radius r (clamped borders), separable running sums
ImageF box_mean(const ImageF& in, int r) {
    ImageF tmp(in.w, in.h), out(in.w, in.h);
    const float inv = 1.f / (2 * r + 1);
    parallel_for(in.h, [&](int y) {
        const float* s = in.row(y);
        float* d = tmp.row(y);
        float acc = 0;
        for (int x = -r - 1; x < r; x++) acc += s[clampi(x, 0, in.w - 1)];
        for (int x = 0; x < in.w; x++) {
            acc += s[clampi(x + r, 0, in.w - 1)] - s[clampi(x - r - 1, 0, in.w - 1)];
            d[x] = acc * inv;
        }
    });
    const int bands = 64;
    parallel_for(bands, [&](int b) {
        int x0 = in.w * b / bands, x1 = in.w * (b + 1) / bands;
        std::vector<float> acc(x1 - x0, 0.f);
        for (int y = -r - 1; y < r; y++) { const float* s = tmp.row(clampi(y, 0, in.h - 1)); for (int x = x0; x < x1; x++) acc[x - x0] += s[x]; }
        for (int y = 0; y < in.h; y++) {
            const float* a = tmp.row(clampi(y + r, 0, in.h - 1));
            const float* s = tmp.row(clampi(y - r - 1, 0, in.h - 1));
            float* d = out.row(y);
            for (int x = x0; x < x1; x++) { acc[x - x0] += a[x] - s[x]; d[x] = acc[x - x0] * inv; }
        }
    });
    return out;
}

}  // namespace

// Small negative excursion allowed through WB + demosaic + CCM so zero-mean shadow noise stays
// unbiased before chroma_denoise averages it away, preventing shadow noise rectification into pink.
constexpr float kNegFloor = 1.f / 16.f;

ImageF normalize_mosaic(const ImageF& mosaic, const Meta& m, const LookParams& p) {
    ImageF out(mosaic.w, mosaic.h);
    bool lsc = m.lscRows > 1 && m.lscCols > 1;
    float sw = m.full_w() - 1.f, sh = m.full_h() - 1.f;
    parallel_for(mosaic.h, [&](int y) {
        float gy = lsc ? (y + m.cropY) / sh * (m.lscRows - 1) : 0;
        int y0 = std::min((int)gy, std::max(0, m.lscRows - 2));
        float fy = gy - y0;
        for (int x = 0; x < mosaic.w; x++) {
            int ch = cfa_channel(m.cfa, x, y);
            float raw = mosaic.at(x, y);
            if (x >= 2 && x < mosaic.w - 2 && y >= 2 && y < mosaic.h - 2) {
                float n0 = mosaic.at(x - 2, y - 2), n1 = mosaic.at(x, y - 2), n2 = mosaic.at(x + 2, y - 2);
                float n3 = mosaic.at(x - 2, y),                             n4 = mosaic.at(x + 2, y);
                float n5 = mosaic.at(x - 2, y + 2), n6 = mosaic.at(x, y + 2), n7 = mosaic.at(x + 2, y + 2);
                float hi = std::max(std::max(std::max(n0, n1), std::max(n2, n3)), std::max(std::max(n4, n5), std::max(n6, n7)));
                float avg = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7) * 0.125f;
                float thr = hi + std::max(0.02f * (m.white - m.black[ch]), 2.f * (hi - avg));
                if (raw > thr) raw = hi;
            }
            float v = (raw - m.black[ch]) / (m.white - m.black[ch]);
            float gain = 1.f, clip = 1.f;
            if (lsc) {
                float gx = (x + m.cropX) / sw * (m.lscCols - 1);
                int x0 = std::min((int)gx, m.lscCols - 2);
                float fx = gx - x0;
                auto G = [&](int c) {
                    const float* L = m.lsc.data();
                    auto at = [&](int xx, int yy) { return L[((size_t)yy * m.lscCols + xx) * 4 + c]; };
                    return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
                };
                float gG = 0.5f * (G(1) + G(2));
                float lumaGain = std::pow(gG, p.lscLumaStrength);
                gain = G(ch) / gG * lumaGain;
                clip = lumaGain;
            }
            out.at(x, y) = clampf(v * gain * m.wb[ch], -kNegFloor, clip);
        }
    });
    return out;
}

void correct_lca(ImageF& rgb, const Meta& m, const float lcaR[2], const float lcaB[2]) {
    if (rgb.w < 2 || rgb.h < 2) return;
    if (lcaR[0] == 0 && lcaR[1] == 0 && lcaB[0] == 0 && lcaB[1] == 0) return;
    const int W = rgb.w, H = rgb.h;
    const float fcx = (m.full_w() - 1) * 0.5f, fcy = (m.full_h() - 1) * 0.5f;
    const float invR2 = 1.f / (fcx * fcx + fcy * fcy);
    const float lcx = fcx - m.cropX, lcy = fcy - m.cropY;
    auto sample = [&](const ImageF& im, float fx, float fy, int c) {
        fx = clampf(fx, 0.f, (float)(W - 1));
        fy = clampf(fy, 0.f, (float)(H - 1));
        int x0 = std::min((int)fx, W - 2), y0 = std::min((int)fy, H - 2);
        float ax = fx - x0, ay = fy - y0;
        const float* r0 = im.row(y0) + (size_t)x0 * 3 + c;
        const float* r1 = im.row(y0 + 1) + (size_t)x0 * 3 + c;
        float a = r0[0] * (1.f - ax) + r0[3] * ax;
        float b = r1[0] * (1.f - ax) + r1[3] * ax;
        return a * (1.f - ay) + b * ay;
    };
    ImageF out(W, H, 3);
    parallel_for(H, [&](int y) {
        const float* src = rgb.row(y);
        float* dst = out.row(y);
        float py = (y + 0.5f) - lcy;
        for (int x = 0; x < W; x++, src += 3, dst += 3) {
            float px = (x + 0.5f) - lcx;
            float rr = (px * px + py * py) * invR2;
            float kr = lcaR[0] + lcaR[1] * rr, kb = lcaB[0] + lcaB[1] * rr;
            dst[0] = sample(rgb, x + px * kr, y + py * kr, 0);
            dst[1] = src[1];
            dst[2] = sample(rgb, x + px * kb, y + py * kb, 2);
        }
    });
    rgb = std::move(out);
}

void apply_ccm(ImageF& rgb, const float M[9]) {
    parallel_for(rgb.h, [&](int y) {
        float* p = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, p += 3) {
            float r = p[0], g = p[1], b = p[2];
            p[0] = M[0] * r + M[1] * g + M[2] * b;
            p[1] = M[3] * r + M[4] * g + M[5] * b;
            p[2] = M[6] * r + M[7] * g + M[8] * b;
        }
    });
}

bool compute_dual_illuminant_ccm(const DualColorCal& cal, const float wb[4], float outCcm[9]) {
    if (cal.illuminants <= 0) return false;
    double neu[3] = {cal.neutral[0], cal.neutral[1], cal.neutral[2]};
    bool neuOk = std::isfinite(neu[0]) && std::isfinite(neu[1]) && std::isfinite(neu[2]) &&
                 neu[0] > 1e-5 && neu[1] > 1e-5 && neu[2] > 1e-5;
    if (!neuOk) {
        double g = 0.5 * (wb[1] + wb[2]);
        if (!(wb[0] > 1e-5f && g > 1e-5 && wb[3] > 1e-5f && std::isfinite(wb[0]) && std::isfinite(g) && std::isfinite(wb[3])))
            return false;
        neu[0] = g / wb[0];
        neu[1] = 1.0;
        neu[2] = g / wb[3];
    }

    double cm1[9], cm2[9], fm1[9], fm2[9];
    const bool cm1Ok = mat_valid3(cal.colorMatrix[0], cm1);
    const bool cm2Ok = cal.illuminants >= 2 && mat_valid3(cal.colorMatrix[1], cm2);
    const bool fm1Ok = mat_valid3(cal.forwardMatrix[0], fm1);
    const bool fm2Ok = cal.illuminants >= 2 && mat_valid3(cal.forwardMatrix[1], fm2);
    const double t1 = illuminant_cct(cal.illuminant[0]);
    const double t2 = cal.illuminants >= 2 ? illuminant_cct(cal.illuminant[1]) : 0.0;
    if (!cm1Ok && !fm1Ok) return false;

    double w = 1.0;
    if (cal.illuminants >= 2 && cm1Ok && cm2Ok && t1 > 0.0 && t2 > 0.0) {
        const double m1 = 1e6 / t1, m2 = 1e6 / t2;
        double T = 5000.0;
        for (int it = 0; it < 32; it++) {
            const double mt = 1e6 / clampf((float)T, 1500.f, 50000.f);
            w = std::fabs(m1 - m2) < 1e-6 ? 0.5 : std::min(1.0, std::max(0.0, (mt - m2) / (m1 - m2)));
            double cm[9], invCm[9], xyz[3];
            for (int i = 0; i < 9; i++) cm[i] = w * cm1[i] + (1.0 - w) * cm2[i];
            if (!invert3(cm, invCm)) return false;
            mul3v(invCm, neu, xyz);
            const double sum = xyz[0] + xyz[1] + xyz[2];
            if (!std::isfinite(sum) || sum <= 1e-9) return false;
            const double tn = xy_to_cct_robertson(xyz[0] / sum, xyz[1] / sum);
            if (!std::isfinite(tn)) return false;
            if (std::fabs(tn - T) < 0.01) { T = tn; break; }
            T = it >= 12 ? 0.5 * (T + tn) : tn;
        }
        const double mt = 1e6 / clampf((float)T, 1500.f, 50000.f);
        w = std::fabs(m1 - m2) < 1e-6 ? 0.5 : std::min(1.0, std::max(0.0, (mt - m2) / (m1 - m2)));
    }

    static const double kBrad[9] = {
        0.8951, 0.2664, -0.1614,
        -0.7502, 1.7135, 0.0367,
        0.0389, -0.0685, 1.0296,
    };
    static const double kXyzToSrgb[9] = {
        3.2404542, -1.5371385, -0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
        0.0556434, -0.2040259, 1.0572252,
    };
    static const double kD50[3] = {0.9642, 1.0, 0.8249};
    static const double kD65[3] = {0.95047, 1.0, 1.08883};
    double invBrad[9];
    if (!invert3(kBrad, invBrad)) return false;

    double M[9];
    if (fm1Ok && (w >= 1.0 || fm2Ok || cal.illuminants < 2)) {
        double fm[9];
        for (int i = 0; i < 9; i++) fm[i] = (w >= 1.0 || !fm2Ok) ? fm1[i] : (w * fm1[i] + (1.0 - w) * fm2[i]);
        double cD50[3], cD65[3];
        mul3v(kBrad, kD50, cD50);
        mul3v(kBrad, kD65, cD65);
        double diagBrad[9] = {}, tA[9], adapt[9], d50ToSrgb[9];
        for (int i = 0; i < 3; i++) diagBrad[i * 4] = cD65[i] / cD50[i];
        mul3(diagBrad, kBrad, tA);
        mul3(invBrad, tA, adapt);
        mul3(kXyzToSrgb, adapt, d50ToSrgb);
        mul3(d50ToSrgb, fm, M);
    } else if (cm1Ok && (w >= 1.0 || cm2Ok || cal.illuminants < 2)) {
        double cm[9], camToXyz[9], xyzW[3];
        for (int i = 0; i < 9; i++) cm[i] = (w >= 1.0 || !cm2Ok) ? cm1[i] : (w * cm1[i] + (1.0 - w) * cm2[i]);
        if (!invert3(cm, camToXyz)) return false;
        mul3v(camToXyz, neu, xyzW);
        if (!std::isfinite(xyzW[1]) || xyzW[1] <= 1e-6) return false;
        for (int i = 0; i < 3; i++) xyzW[i] /= xyzW[1];
        double cW[3], cD65[3];
        mul3v(kBrad, xyzW, cW);
        mul3v(kBrad, kD65, cD65);
        if (cW[0] <= 1e-6 || cW[1] <= 1e-6 || cW[2] <= 1e-6) return false;
        double diagBrad[9] = {}, tA[9], adapt[9], wbCamToXyz[9], diagNeu[9] = {}, tB[9];
        for (int i = 0; i < 3; i++) { diagBrad[i * 4] = cD65[i] / cW[i]; diagNeu[i * 4] = neu[i]; }
        mul3(diagBrad, kBrad, tA);
        mul3(invBrad, tA, adapt);
        mul3(camToXyz, diagNeu, wbCamToXyz);
        mul3(adapt, wbCamToXyz, tB);
        mul3(kXyzToSrgb, tB, M);
    } else {
        return false;
    }

    for (int r = 0; r < 3; r++) {
        double sum = M[r * 3] + M[r * 3 + 1] + M[r * 3 + 2];
        if (!std::isfinite(sum) || sum <= 0.1) return false;
        for (int c = 0; c < 3; c++) M[r * 3 + c] /= sum;
    }
    if (!std::isfinite(det3(M)) || std::fabs(det3(M)) <= 1e-6) return false;
    for (int i = 0; i < 9; i++) outCcm[i] = (float)M[i];
    return true;
}

bool ccm_usable(const float ccm[9]) {
    double m[9];
    if (!mat_valid3(ccm, m)) return false;
    double dev = 0.0;
    for (int i = 0; i < 9; i++) dev = std::max(dev, std::fabs(m[i] - (i % 4 == 0 ? 1.0 : 0.0)));
    return dev > 1e-3;
}

void upscale_lanczos(ImageF& rgb, int W, int H) {
    if (W == rgb.w && H == rgb.h) return;
    constexpr int kA = 3;
    constexpr float kPi = 3.14159265f;
    struct Taps { int x0; float w[2 * kA]; };
    auto taps = [&](int n, int N) {
        std::vector<Taps> t(N);
        const float s = (float)n / N;
        for (int X = 0; X < N; X++) {
            float c = (X + 0.5f) * s - 0.5f;
            int x0 = (int)std::floor(c) - kA + 1;
            float sum = 0;
            for (int i = 0; i < 2 * kA; i++) {
                float d = std::fabs(c - (x0 + i));
                float v = d < 1e-6f ? 1.f : (d >= kA ? 0.f : kA * std::sin(kPi * d) * std::sin(kPi * d / kA) / (kPi * kPi * d * d));
                t[X].w[i] = v;
                sum += v;
            }
            for (int i = 0; i < 2 * kA; i++) t[X].w[i] /= sum;
            t[X].x0 = x0;
        }
        return t;
    };
    const int w = rgb.w, h = rgb.h;
    std::vector<Taps> tx = taps(w, W), ty = taps(h, H);
    ImageF mid(W, h, 3);
    parallel_for(h, [&](int y) {
        const float* s = rgb.row(y);
        float* o = mid.row(y);
        for (int X = 0; X < W; X++) {
            float r = 0, g = 0, b = 0;
            for (int i = 0; i < 2 * kA; i++) {
                const float* p = s + 3 * clampi(tx[X].x0 + i, 0, w - 1);
                float k = tx[X].w[i];
                r += k * p[0]; g += k * p[1]; b += k * p[2];
            }
            o[3 * X] = r; o[3 * X + 1] = g; o[3 * X + 2] = b;
        }
    });
    rgb = ImageF();
    ImageF out(W, H, 3);
    parallel_for(H, [&](int Y) {
        float* o = out.row(Y);
        const float* rows[2 * kA];
        for (int i = 0; i < 2 * kA; i++) rows[i] = mid.row(clampi(ty[Y].x0 + i, 0, h - 1));
        for (int x = 0; x < 3 * W; x++) {
            float v = 0;
            for (int i = 0; i < 2 * kA; i++) v += ty[Y].w[i] * rows[i][x];
            // Lanczos rings below zero next to dark edges; the tone curve expects linear values >= 0
            o[x] = std::max(0.f, v);
        }
    });
    rgb = std::move(out);
}

void aspect_crop_box(int inW, int inH, float aspect, int& x0, int& y0, int& cw, int& ch) {
    if (aspect <= 0.f || inW <= 0 || inH <= 0) {
        x0 = 0; y0 = 0; cw = inW; ch = inH;
        return;
    }
    const float r = aspect >= 1.f ? aspect : 1.f / aspect;
    if (inW >= inH) {
        if ((double)inW / inH > r) {
            ch = inH & ~1;
            cw = std::min(inW, (int)std::lround(ch * r)) & ~1;
        } else {
            cw = inW & ~1;
            ch = std::min(inH, (int)std::lround(cw / r)) & ~1;
        }
    } else {
        if ((double)inH / inW > r) {
            cw = inW & ~1;
            ch = std::min(inH, (int)std::lround(cw * r)) & ~1;
        } else {
            ch = inH & ~1;
            cw = std::min(inW, (int)std::lround(ch / r)) & ~1;
        }
    }
    cw = std::max(2, cw);
    ch = std::max(2, ch);
    x0 = ((inW - cw) / 2) & ~1;
    y0 = ((inH - ch) / 2) & ~1;
}

void crop_aspect(ImageF& rgb, float aspect) {
    int x0, y0, cw, ch;
    aspect_crop_box(rgb.w, rgb.h, aspect, x0, y0, cw, ch);
    if (cw == rgb.w && ch == rgb.h) return;
    ImageF out(cw, ch, 3);
    parallel_for(ch, [&](int y) {
        const float* s = rgb.row(y0 + y) + (size_t)x0 * 3;
        std::copy(s, s + (size_t)cw * 3, out.row(y));
    });
    rgb = std::move(out);
}

void chroma_denoise(ImageF& rgb, float strength) {
    if (strength <= 0) {
        parallel_for(rgb.h, [&](int y) {
            float* p = rgb.row(y);
            for (int x = 0; x < rgb.w * 3; x++) p[x] = std::max(0.f, p[x]);
        });
        return;
    }
    constexpr float kChromaFloor = 4e-3f;
    const int f = 4;
    int w = (rgb.w + f - 1) / f, h = (rgb.h + f - 1) / f;
    ImageF cr(w, h), cb(w, h);
    parallel_for(h, [&](int y) {
        for (int x = 0; x < w; x++) {
            float sr = 0, sb = 0; int n = 0;
            for (int v = 0; v < f; v++)
                for (int u = 0; u < f; u++) {
                    int X = x * f + u, Y = y * f + v;
                    if (X >= rgb.w || Y >= rgb.h) continue;
                    const float* p = &rgb.at(X, Y);
                    float L = luma709(p[0], p[1], p[2]);
                    float inv = 1.f / (std::max(L, 0.f) + kChromaFloor);
                    sr += (p[0] - L) * inv; sb += (p[2] - L) * inv; n++;
                }
            cr.at(x, y) = sr / n; cb.at(x, y) = sb / n;
        }
    });
    cr = gaussian_blur(cr, 1.0f * strength);
    cb = gaussian_blur(cb, 1.0f * strength);
    parallel_for(rgb.h, [&](int y) {
        float gy = (y + 0.5f) / f - 0.5f;
        int y0 = clampi((int)std::floor(gy), 0, h - 1), y1 = std::min(y0 + 1, h - 1);
        float fy = clampf(gy - y0, 0.f, 1.f);
        for (int x = 0; x < rgb.w; x++) {
            float gx = (x + 0.5f) / f - 0.5f;
            int x0 = clampi((int)std::floor(gx), 0, w - 1), x1 = std::min(x0 + 1, w - 1);
            float fx = clampf(gx - x0, 0.f, 1.f);
            auto bil = [&](const ImageF& im) {
                return (1 - fy) * ((1 - fx) * im.at(x0, y0) + fx * im.at(x1, y0)) + fy * ((1 - fx) * im.at(x0, y1) + fx * im.at(x1, y1));
            };
            float* p = &rgb.at(x, y);
            float L = luma709(p[0], p[1], p[2]);
            float d = std::max(L, 0.f) + kChromaFloor;
            float r = L + d * bil(cr), b = L + d * bil(cb);
            float g = (L - 0.2126f * r - 0.0722f * b) / 0.7152f;
            p[0] = std::max(0.f, r); p[1] = std::max(0.f, g); p[2] = std::max(0.f, b);
        }
    });
}

// Laplacian pyramid exposure fusion (Mertens et al. 2007) of a bright and a dark synthetic exposure of Y, as in HDR+
// finishing; returns the fused display-referred luma
static ImageF fuse_exposures(const ImageF& Y, float gS, float gL) {
    ImageF eS(Y.w, Y.h), eL(Y.w, Y.h), wS(Y.w, Y.h);
    parallel_for(Y.h, [&](int y) {
        for (int x = 0; x < Y.w; x++) {
            float s = srgb_oetf(Y.at(x, y) * gS), l = srgb_oetf(Y.at(x, y) * gL);
            float clipFade = clampf((1.f - l) / 0.15f, 0.f, 1.f);
            float ws = std::exp(-(s - 0.5f) * (s - 0.5f) / (2 * 0.2f * 0.2f)) + 1e-4f;
            float wl = std::exp(-(l - 0.5f) * (l - 0.5f) / (2 * 0.2f * 0.2f)) * clipFade + 1e-4f;
            eS.at(x, y) = s; eL.at(x, y) = l; wS.at(x, y) = ws / (ws + wl);
        }
    });
    int levels = 1;
    int minTop = Y.w > 2000 ? 24 * kToneSub : 24;
    for (int m = std::min(Y.w, Y.h); m > minTop; m /= 2) levels++;
    std::vector<ImageF> gs, gl, gw;
    gs.push_back(std::move(eS)); gl.push_back(std::move(eL)); gw.push_back(std::move(wS));
    for (int l = 1; l < levels; l++) { gs.push_back(pyr_down(gs.back())); gl.push_back(pyr_down(gl.back())); gw.push_back(pyr_down(gw.back())); }
    // Gaussian -> Laplacian in place, fine to coarse (level l+1 is still Gaussian when level l is processed)
    for (int l = 0; l < levels - 1; l++) {
        sub_inplace(gs[l], pyr_up(gs[l + 1], gs[l].w, gs[l].h));
        sub_inplace(gl[l], pyr_up(gl[l + 1], gl[l].w, gl[l].h));
    }
    // blend into gs, then collapse coarse to fine
    for (int l = 0; l < levels; l++)
        parallel_for(gs[l].h, [&](int y) {
            float* a = gs[l].row(y); const float* b = gl[l].row(y); const float* w = gw[l].row(y);
            for (int x = 0; x < gs[l].w; x++) a[x] = w[x] * a[x] + (1 - w[x]) * b[x];
        });
    gl.clear(); gw.clear();
    for (int l = levels - 2; l >= 0; l--) {
        sub_inplace(gs[l], pyr_up(gs[l + 1], gs[l].w, gs[l].h), -1.f);
        gs[l + 1] = ImageF();
    }
    return std::move(gs[0]);
}

// log2 / exp2 to ~1e-4 for the per-pixel tone gain (polynomial fits on the mantissa / fraction)
static inline float fast_log2(float x) {
    uint32_t i;
    memcpy(&i, &x, 4);
    float e = (float)((int)(i >> 23) - 127);
    i = (i & 0x7FFFFFu) | 0x3F800000u;
    float t;
    memcpy(&t, &i, 4);
    t -= 1.f;
    return e + t * (1.44253315f + t * (-0.716105633f + t * (0.439540816f + t * (-0.225090121f + t * 0.059238473f))));
}

static inline float fast_exp2(float x) {
    x = clampf(x, -126.f, 126.f);
    float n = std::floor(x), f = x - n;
    float p = 1.000007283f + f * (0.692931289f + f * (0.241710262f + f * (0.051666877f + f * 0.013676531f)));
    uint32_t i;
    memcpy(&i, &p, 4);
    i += (uint32_t)((int)n << 23);
    memcpy(&p, &i, 4);
    return p;
}

ToneInfo local_tonemap(ImageF& rgb, const LookParams& p) {
    auto Yat = [&](int x, int y) { const float* q = rgb.row(y) + (size_t)x * 3; return std::max(1e-6f, luma709(q[0], q[1], q[2])); };
    std::vector<float> samp;
    samp.reserve((size_t)((rgb.h + 7) / 8) * ((rgb.w + 7) / 8));
    double lsum = 0;
    for (int y = 0; y < rgb.h; y += 8)
        for (int x = 0; x < rgb.w; x += 8) { float v = Yat(x, y); samp.push_back(v); lsum += std::log(v + 1e-4); }
    float key = (float)std::exp(lsum / samp.size());
    std::nth_element(samp.begin(), samp.begin() + (size_t)(samp.size() * 0.995), samp.end());
    float p995 = samp[(size_t)(samp.size() * 0.995)];
    float gL = clampf(p.targetKey / key, 1.f, 64.f);
    float maxRatio = p.compression > 0 ? p.compression : 8.f;
    float gS = clampf(1.f / p995, gL / maxRatio, gL);

    // fusion on the binned luma; its gain (log2 fused / input) is brought back to full size by a fast guided filter
    // (He & Sun 2015) on log2 luma, so it follows the edges the binning blurred
    constexpr int kSub = kToneSub;
    const int sw = (rgb.w + kSub - 1) / kSub, sh = (rgb.h + kSub - 1) / kSub;
    ImageF Ys(sw, sh);
    parallel_for(sh, [&](int y) {
        for (int x = 0; x < sw; x++) {
            float acc = 0; int cnt = 0;
            for (int v = 0; v < kSub; v++) for (int u = 0; u < kSub; u++) {
                int X = x * kSub + u, Yy = y * kSub + v;
                if (X < rgb.w && Yy < rgb.h) { acc += Yat(X, Yy); cnt++; }
            }
            Ys.at(x, y) = acc / cnt;
        }
    });
    ImageF fused = fuse_exposures(Ys, gS, gL);
    ImageF G(sw, sh), Lg(sw, sh), GG(sw, sh), GL(sw, sh);
    parallel_for(sh, [&](int y) {
        for (int x = 0; x < sw; x++) {
            float g = fast_log2(Ys.at(x, y)), l = fast_log2(std::max(1e-6f, srgb_eotf(fused.at(x, y)))) - g;
            G.at(x, y) = g; Lg.at(x, y) = l; GG.at(x, y) = g * g; GL.at(x, y) = g * l;
        }
    });
    ImageF mG = box_mean(G, kToneGuideRadius), mL = box_mean(Lg, kToneGuideRadius), mGG = box_mean(GG, kToneGuideRadius),
           mGL = box_mean(GL, kToneGuideRadius);
    ImageF A(sw, sh), B(sw, sh);
    parallel_for(sh, [&](int y) {
        for (int x = 0; x < sw; x++) {
            float mg = mG.at(x, y), ml = mL.at(x, y);
            float a = (mGL.at(x, y) - mg * ml) / (std::max(0.f, mGG.at(x, y) - mg * mg) + kToneGuideEps);
            A.at(x, y) = a; B.at(x, y) = ml - a * mg;
        }
    });
    ImageF mA = box_mean(A, kToneGuideRadius), mB = box_mean(B, kToneGuideRadius);

    // Estimate post-tonemap 99.5th percentile whitepoint directly on the 1/4-res guided luma (stride 2 = 8x8 grid),
    // folding whitepoint lift into the single full-res guided-gain pass.
    std::vector<float> s2;
    s2.reserve((size_t)((sh + 1) / 2) * ((sw + 1) / 2));
    for (int y = 0; y < sh; y += 2)
        for (int x = 0; x < sw; x += 2) {
            float y0 = std::max(1e-6f, Ys.at(x, y));
            s2.push_back(y0 * fast_exp2(mA.at(x, y) * fast_log2(y0) + mB.at(x, y)));
        }
    size_t idx = (size_t)(s2.size() * 0.995);
    std::nth_element(s2.begin(), s2.begin() + idx, s2.end());
    const float hi = s2[idx];
    constexpr float kWhite = 0.72f, kMaxLift = 1.35f;
    const float lift = clampf(kWhite / std::max(hi, 1e-6f), 1.f, kMaxLift);
    const float lo = p.targetKey;
    const bool doLift = lift > 1.001f && hi > lo * 1.5f;
    const float invSpan = doLift ? 1.f / (hi - lo) : 0.f;
    const float lift1 = doLift ? (lift - 1.f) : 0.f;

    std::vector<int> x0s(rgb.w);
    std::vector<float> axs(rgb.w);
    for (int x = 0; x < rgb.w; x++) {
        float fx = clampf((x + 0.5f) / kSub - 0.5f, 0.f, sw - 1.001f);
        x0s[x] = (int)fx;
        axs[x] = fx - x0s[x];
    }
    parallel_for(rgb.h, [&](int y) {
        float fy = clampf((y + 0.5f) / kSub - 0.5f, 0.f, sh - 1.001f);
        int y0 = (int)fy, y1 = std::min(y0 + 1, sh - 1);
        float ay = fy - y0;
        std::vector<float> ra(sw + 1), rb(sw + 1);
        for (int x = 0; x < sw; x++) {
            ra[x] = (1 - ay) * mA.at(x, y0) + ay * mA.at(x, y1);
            rb[x] = (1 - ay) * mB.at(x, y0) + ay * mB.at(x, y1);
        }
        ra[sw] = ra[sw - 1];
        rb[sw] = rb[sw - 1];
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) {
            int x0 = x0s[x];
            float ax = axs[x];
            float a = ra[x0] + ax * (ra[x0 + 1] - ra[x0]), b = rb[x0] + ax * (rb[x0 + 1] - rb[x0]);
            float L0 = luma709(q[0], q[1], q[2]);
            float ratio = fast_exp2(a * fast_log2(std::max(1e-6f, L0)) + b);
            if (doLift) {
                float t = clampf((L0 * ratio - lo) * invSpan, 0.f, 1.f);
                ratio *= 1.f + lift1 * t * t * (3.f - 2.f * t);
            }
            q[0] *= ratio; q[1] *= ratio; q[2] *= ratio;
        }
    });
    return {gS, gL, key};
}

void pixel_look(ImageF& rgb, const LookParams& p) {
    auto smooth01 = [](float t) { t = clampf(t, 0.f, 1.f); return t * t * (3.f - 2.f * t); };
    parallel_for(rgb.h, [&](int y) {
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) {
            float r = q[0], g = q[1], b = q[2];
            float mx0 = std::max(r, std::max(g, b));
            // coloured highlights (lamp-lit wood, skin) roll off on the max channel with hue kept, while bright
            // speculars and light sources (monitors, LEDs) bleach smoothly toward neutral white
            constexpr float kKnee = 0.7f;
            if (mx0 > kKnee) {
                float mn = std::min(r, std::min(g, b));
                float t = (mx0 - kKnee) / (1 - kKnee);
                float shoulder = kKnee + (1 - kKnee) * t / (1 + t);
                float chroma = clampf(2 * (mx0 - std::max(0.f, mn)) / mx0, 0.f, 1.f);
                float sc = (mx0 + (shoulder - mx0) * chroma) / mx0;
                float bleach = smooth01((mx0 - 0.90f) / (3.2f - 0.90f));
                float mx = mx0 * sc;
                r = r * sc + bleach * (mx - r * sc);
                g = g * sc + bleach * (mx - g * sc);
                b = b * sc + bleach * (mx - b * sc);
            }
            // constant-hue perceptual Oklab gamut compression (skips deeply in-gamut pixels in 4 comparisons)
            oklab_gamut_compress(r, g, b);
            q[0] = srgb_oetf(r); q[1] = srgb_oetf(g); q[2] = srgb_oetf(b);
            // luminance S-curve avoids warping R/G/B at different slopes near white (which twists cool tints cyan)
            float Ld = clampf(luma709(q[0], q[1], q[2]), 0.f, 1.f);
            float s = Ld * Ld * (3 - 2 * Ld);
            float Lv = Ld + p.contrast * (s - Ld);
            float cg = Ld > 1e-6f ? Lv / Ld : 1.f;
            for (int c = 0; c < 3; c++) q[c] *= cg;
            float sat = std::max(q[0], std::max(q[1], q[2])) - std::min(q[0], std::min(q[1], q[2]));
            float vibW = 1.f - smooth01((Lv - 0.55f) / (0.80f - 0.55f));
            float hiRoll = 1.f - 0.55f * smooth01((Lv - 0.72f) / (1.0f - 0.72f));
            float k = (1.f + (p.saturation - 1.f) * (1.f - clampf(sat, 0.f, 1.f)) * vibW) * hiRoll;
            float mxd = 0.f;
            for (int c = 0; c < 3; c++) mxd = std::max(mxd, Lv + (q[c] - Lv) * k);
            if (mxd > 1.f && Lv < 1.f) k *= (1.f - Lv) / std::max(1e-6f, mxd - Lv);
            for (int c = 0; c < 3; c++) q[c] = clampf(Lv + (q[c] - Lv) * k, 0.f, 1.f);
        }
    });
}

float sharpen_core(float visibleSigma) {
    // clean bursts (visible sigma 0.003-0.007) keep the fixed 0.004
    return std::max(0.004f, 0.6f * visibleSigma);
}

float noise_limited_key(float targetKey, float visibleSigma) {
    // visible sigma goes linearly with the key; above the cap the key follows its square root, at most 1 EV down
    constexpr float kKeyNoiseCap = 0.015f;
    if (visibleSigma <= kKeyNoiseCap) return targetKey;
    return targetKey * std::max(0.5f, std::sqrt(kKeyNoiseCap / visibleSigma));
}

void sharpen_luma(ImageF& rgb, float amount, float radius, float core) {
    if (amount <= 0) return;
    const int W = rgb.w, H = rgb.h;
    ImageF L(W, H);
    parallel_for(H, [&](int y) {
        const float* q = rgb.row(y);
        float* l = L.row(y);
        for (int x = 0; x < W; x++, q += 3) l[x] = luma709(q[0], q[1], q[2]);
    });
    ImageF B = gaussian_blur(L, radius);
    parallel_for(H, [&](int y) {
        const float* lr = L.row(y);
        const float* br = B.row(y);
        float* q = rgb.row(y);
        for (int x = 0; x < W; x++, q += 3) {
            float d = lr[x] - br[x];
            d = d > core ? d - core : (d < -core ? d + core : 0.f);
            float delta = amount * d;
            for (int c = 0; c < 3; c++) q[c] = clampf(q[c] + delta, 0.f, 1.f);
        }
    });
}

std::vector<uint8_t> to_rgb8(const ImageF& rgb, int ds, int& w, int& h) {
    ds = std::max(1, ds);
    w = rgb.w / ds; h = rgb.h / ds;
    std::vector<uint8_t> buf((size_t)w * h * 3);
    parallel_for(h, [&](int y) {
        uint32_t seed = 0x9E3779B9u * (y + 1);
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                float s = 0;
                for (int v = 0; v < ds; v++) for (int u = 0; u < ds; u++) s += rgb.at(x * ds + u, y * ds + v, c);
                s /= ds * ds;
                seed = seed * 1664525u + 1013904223u;
                float dither = ((seed >> 8) & 0xFFFF) / 65535.f - 0.5f;
                buf[((size_t)y * w + x) * 3 + c] = (uint8_t)clampi((int)std::lround(s * 255.f + dither), 0, 255);
            }
    });
    return buf;
}

namespace {

struct JpegError {
    jpeg_error_mgr mgr;
    jmp_buf jump;
};

void jpeg_fail(j_common_ptr c) { longjmp(reinterpret_cast<JpegError*>(c->err)->jump, 1); }

// one complete baseline JPEG of h rows (standard Huffman tables, so every strip's tables match)
std::vector<uint8_t> jpeg_one(const uint8_t* px, int w, int h, int channels, int quality, bool chroma420) {
    jpeg_compress_struct c;
    JpegError err;
    c.err = jpeg_std_error(&err.mgr);
    err.mgr.error_exit = jpeg_fail;
    unsigned char* buf = nullptr;
    unsigned long size = 0;
    if (setjmp(err.jump)) {
        jpeg_destroy_compress(&c);
        free(buf);
        return {};
    }
    jpeg_create_compress(&c);
    jpeg_mem_dest(&c, &buf, &size);
    c.image_width = w;
    c.image_height = h;
    c.input_components = channels;
    c.in_color_space = channels == 1 ? JCS_GRAYSCALE : channels == 3 ? JCS_RGB : JCS_EXT_RGBA;
    jpeg_set_defaults(&c);
    jpeg_set_quality(&c, quality, TRUE);
    if (channels > 1) c.comp_info[0].h_samp_factor = c.comp_info[0].v_samp_factor = chroma420 ? 2 : 1;
    jpeg_start_compress(&c, TRUE);
    const size_t stride = (size_t)w * channels;
    while (c.next_scanline < c.image_height) {
        JSAMPROW row = const_cast<JSAMPROW>(px + c.next_scanline * stride);
        jpeg_write_scanlines(&c, &row, 1);
    }
    jpeg_finish_compress(&c);
    std::vector<uint8_t> out(buf, buf + size);
    jpeg_destroy_compress(&c);
    free(buf);
    return out;
}

}  // namespace

std::vector<uint8_t> encode_jpeg(const uint8_t* px, int w, int h, int channels, int quality, bool chroma420) {
    // horizontal strips encoded in parallel, joined into one scan with a restart marker between them (DRI = one
    // strip's MCUs): the strips' entropy-coded segments are exactly what a single encoder with that restart
    // interval would write
    const int mcu = channels > 1 && chroma420 ? 16 : 8;
    const int mcuRows = (h + mcu - 1) / mcu, mcuCols = (w + mcu - 1) / mcu;
    // finer strips let ThreadPool rebalance away from slower little cores on big.LITTLE SoCs
    int rowsPer = std::max(2, (mcuRows + 4 * g_threads - 1) / (4 * g_threads));
    rowsPer = std::min(rowsPer, 65535 / mcuCols);
    const int strips = rowsPer > 0 ? (mcuRows + rowsPer - 1) / rowsPer : 1;
    if (strips <= 1 || rowsPer <= 0) return jpeg_one(px, w, h, channels, quality, chroma420);
    std::vector<std::vector<uint8_t>> part(strips);
    parallel_for(strips, [&](int s) {
        int y0 = s * rowsPer * mcu, y1 = std::min(h, y0 + rowsPer * mcu);
        part[s] = jpeg_one(px + (size_t)y0 * w * channels, w, y1 - y0, channels, quality, chroma420);
    });
    // the scan data of each strip starts after its SOS segment and ends before EOI
    auto scan_start = [](const std::vector<uint8_t>& j, size_t* sosAt, size_t* sofAt) -> size_t {
        size_t p = 2;
        while (p + 4 <= j.size() && j[p] == 0xFF) {
            int marker = j[p + 1], len = (j[p + 2] << 8) | j[p + 3];
            if (marker >= 0xC0 && marker <= 0xC2 && sofAt) *sofAt = p;
            if (marker == 0xDA) {
                if (sosAt) *sosAt = p;
                return p + 2 + len;
            }
            p += 2 + len;
        }
        return 0;
    };
    for (auto& j : part)
        if (j.size() < 4 || scan_start(j, nullptr, nullptr) == 0) return jpeg_one(px, w, h, channels, quality, chroma420);
    size_t sos = 0, sof = 0;
    scan_start(part[0], &sos, &sof);
    size_t total = part[0].size() + 6;
    for (int s = 1; s < strips; s++) total += part[s].size();
    std::vector<uint8_t> out;
    out.reserve(total);
    out.insert(out.end(), part[0].begin(), part[0].begin() + sos);
    out[sof + 5] = (uint8_t)(h >> 8);
    out[sof + 6] = (uint8_t)(h & 0xFF);
    const int ri = rowsPer * mcuCols;
    const uint8_t dri[6] = {0xFF, 0xDD, 0x00, 0x04, (uint8_t)(ri >> 8), (uint8_t)(ri & 0xFF)};
    out.insert(out.end(), dri, dri + 6);
    out.insert(out.end(), part[0].begin() + sos, part[0].end() - 2);
    for (int s = 1; s < strips; s++) {
        out.push_back(0xFF);
        out.push_back((uint8_t)(0xD0 + ((s - 1) & 7)));
        out.insert(out.end(), part[s].begin() + scan_start(part[s], nullptr, nullptr), part[s].end() - 2);
    }
    out.push_back(0xFF);
    out.push_back(0xD9);
    return out;
}

bool write_jpeg(const std::string& path, const ImageF& rgb, int quality, int ds) {
    int w, h;
    std::vector<uint8_t> buf = to_rgb8(rgb, ds, w, h);
    std::vector<uint8_t> jpg = encode_jpeg(buf.data(), w, h, 3, quality);
    FILE* f = jpg.empty() ? nullptr : fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(jpg.data(), 1, jpg.size(), f) == jpg.size();
    return fclose(f) == 0 && ok;
}

bool apply_look_preset(const std::string& name, LookParams& p) {
    if (name == "indigo" || name == "default" || name == "pixel" || name == "night") {
        // Locked Project Indigo + Original Pixel 1 HDR+ look: SLR-like shadow depth & highlight rolloff,
        // faithful natural colour, organic fine grain (localContrast = 0), and halo-free capture sharpening.
        p = LookParams();
        p.lscLumaStrength = 0.85f;
        p.chromaDenoise = -1.f;
        p.targetKey = 0.20f;
        p.compression = 5.0f;
        p.contrast = 0.20f;
        p.saturation = 1.08f;
        p.sharpen = 0.35f;
        p.sharpenRadius = 0.85f;
        p.localContrast = 0.0f;
        return true;
    }
    return false;
}

void to_rgba8888(const ImageF& rgb, uint8_t* out, size_t stride) {
    parallel_for(rgb.h, [&](int y) {
        uint32_t seed = 0x9E3779B9u * (y + 1);
        const float* p = rgb.row(y);
        uint8_t* o = out + (size_t)y * stride;
        for (int x = 0; x < rgb.w; x++, p += 3, o += 4) {
            for (int c = 0; c < 3; c++) {
                seed = seed * 1664525u + 1013904223u;
                float dither = ((seed >> 8) & 0xFFFF) / 65535.f - 0.5f;
                o[c] = (uint8_t)clampi((int)std::lround(p[c] * 255.f + dither), 0, 255);
            }
            o[3] = 255;
        }
    });
}

void rgb8_to_rgba8888(const uint8_t* rgb8, int w, int h, uint8_t* out, size_t stride) {
    parallel_for(h, [&](int y) {
        const uint8_t* p = rgb8 + (size_t)y * w * 3;
        uint8_t* o = out + (size_t)y * stride;
        for (int x = 0; x < w; x++, p += 3, o += 4) {
            o[0] = p[0];
            o[1] = p[1];
            o[2] = p[2];
            o[3] = 255;
        }
    });
}

void to_rgba8888_thumb(const ImageF& rgb, uint8_t* out, size_t stride, int outW, int outH) {
    if (rgb.w <= 0 || rgb.h <= 0 || outW <= 0 || outH <= 0) return;
    parallel_for(outH, [&](int y) {
        int y0 = (int)((long)y * rgb.h / outH), y1 = std::max(y0 + 1, (int)((long)(y + 1) * rgb.h / outH));
        int syStep = std::max(1, (y1 - y0) / 4);
        uint8_t* o = out + (size_t)y * stride;
        for (int x = 0; x < outW; x++, o += 4) {
            int x0 = (int)((long)x * rgb.w / outW), x1 = std::max(x0 + 1, (int)((long)(x + 1) * rgb.w / outW));
            int sxStep = std::max(1, (x1 - x0) / 4);
            float sr = 0, sg = 0, sb = 0;
            int cnt = 0;
            for (int sy = y0; sy < y1; sy += syStep) {
                const float* row = rgb.row(sy);
                for (int sx = x0; sx < x1; sx += sxStep, cnt++) {
                    const float* p = row + (size_t)sx * 3;
                    sr += p[0]; sg += p[1]; sb += p[2];
                }
            }
            float inv = 255.f / cnt;
            o[0] = (uint8_t)clampi((int)std::lround(sr * inv), 0, 255);
            o[1] = (uint8_t)clampi((int)std::lround(sg * inv), 0, 255);
            o[2] = (uint8_t)clampi((int)std::lround(sb * inv), 0, 255);
            o[3] = 255;
        }
    });
}

void rgb8_to_rgba8888_thumb(const uint8_t* rgb8, int inW, int inH, uint8_t* out, size_t stride, int outW, int outH) {
    if (!rgb8 || inW <= 0 || inH <= 0 || outW <= 0 || outH <= 0) return;
    parallel_for(outH, [&](int y) {
        int y0 = (int)((long)y * inH / outH), y1 = std::max(y0 + 1, (int)((long)(y + 1) * inH / outH));
        int syStep = std::max(1, (y1 - y0) / 4);
        uint8_t* o = out + (size_t)y * stride;
        for (int x = 0; x < outW; x++, o += 4) {
            int x0 = (int)((long)x * inW / outW), x1 = std::max(x0 + 1, (int)((long)(x + 1) * inW / outW));
            int sxStep = std::max(1, (x1 - x0) / 4);
            uint32_t sr = 0, sg = 0, sb = 0, cnt = 0;
            for (int sy = y0; sy < y1; sy += syStep) {
                const uint8_t* row = rgb8 + (size_t)sy * inW * 3;
                for (int sx = x0; sx < x1; sx += sxStep, cnt++) {
                    const uint8_t* p = row + (size_t)sx * 3;
                    sr += p[0]; sg += p[1]; sb += p[2];
                }
            }
            uint32_t half = cnt >> 1;
            o[0] = (uint8_t)((sr + half) / cnt);
            o[1] = (uint8_t)((sg + half) / cnt);
            o[2] = (uint8_t)((sb + half) / cnt);
            o[3] = 255;
        }
    });
}
