#include "finish.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

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

ImageF normalize_mosaic(const ImageF& mosaic, const Meta& m, const LookParams& p) {
    ImageF out(mosaic.w, mosaic.h);
    bool lsc = m.lscRows > 1 && m.lscCols > 1;
    parallel_for(mosaic.h, [&](int y) {
        float gy = lsc ? (float)y / (mosaic.h - 1) * (m.lscRows - 1) : 0;
        int y0 = std::min((int)gy, std::max(0, m.lscRows - 2));
        float fy = gy - y0;
        for (int x = 0; x < mosaic.w; x++) {
            int ch = cfa_channel(m.cfa, x, y);
            float v = (mosaic.at(x, y) - m.black[ch]) / (m.white - m.black[ch]);
            float gain = 1.f, clip = 1.f;
            if (lsc) {
                float gx = (float)x / (mosaic.w - 1) * (m.lscCols - 1);
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
            out.at(x, y) = std::min(std::max(0.f, v) * gain * m.wb[ch], clip);
        }
    });
    return out;
}

void normalize_rgb(ImageF& rgb, const Meta& m, const LookParams& p, float scale) {
    bool lsc = m.lscRows > 1 && m.lscCols > 1;
    float sw = m.width - 1.f, sh = m.height - 1.f;
    float wbG = 0.5f * (m.wb[1] + m.wb[2]);
    float inv[3] = {1.f / (m.white - m.black[0]), 2.f / (2 * m.white - m.black[1] - m.black[2]), 1.f / (m.white - m.black[3])};
    parallel_for(rgb.h, [&](int y) {
        float ry = clampf((y + 0.5f) / scale - 0.5f, 0.f, sh);
        float gy = lsc ? ry / sh * (m.lscRows - 1) : 0;
        int y0 = std::min((int)gy, std::max(0, m.lscRows - 2));
        float fy = gy - y0;
        float* px = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, px += 3) {
            float gain[3] = {1, 1, 1}, clip = 1.f;
            if (lsc) {
                float rx = clampf((x + 0.5f) / scale - 0.5f, 0.f, sw);
                float gx = rx / sw * (m.lscCols - 1);
                int x0 = std::min((int)gx, m.lscCols - 2);
                float fx = gx - x0;
                auto G = [&](int c) {
                    const float* L = m.lsc.data();
                    auto at = [&](int xx, int yy) { return L[((size_t)yy * m.lscCols + xx) * 4 + c]; };
                    return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
                };
                float gG = 0.5f * (G(1) + G(2));
                float lumaGain = std::pow(gG, p.lscLumaStrength);
                gain[0] = G(0) / gG * lumaGain; gain[1] = lumaGain; gain[2] = G(3) / gG * lumaGain;
                clip = lumaGain;
            }
            float wb[3] = {m.wb[0], wbG, m.wb[3]};
            for (int c = 0; c < 3; c++) px[c] = std::min(std::max(0.f, px[c] * inv[c]) * gain[c] * wb[c], clip);
        }
    });
}

ImageF demosaic_mhc(const ImageF& cfa, int cfaType) {
    ImageF rgb(cfa.w, cfa.h, 3);
    auto P = [&](int x, int y) { return cfa.atc(x, y); };
    parallel_for(cfa.h, [&](int y) {
        for (int x = 0; x < cfa.w; x++) {
            int c = cfa_color(cfaType, x, y);
            float v = P(x, y);
            float out[3];
            out[c] = v;
            if (c == 1) {
                // Malvar-He-Cutler: colour present on this row vs. this column
                float rowK = (5 * v + 4 * (P(x - 1, y) + P(x + 1, y)) - (P(x - 2, y) + P(x + 2, y))
                              - (P(x - 1, y - 1) + P(x + 1, y - 1) + P(x - 1, y + 1) + P(x + 1, y + 1)) + 0.5f * (P(x, y - 2) + P(x, y + 2))) * 0.125f;
                float colK = (5 * v + 4 * (P(x, y - 1) + P(x, y + 1)) - (P(x, y - 2) + P(x, y + 2))
                              - (P(x - 1, y - 1) + P(x + 1, y - 1) + P(x - 1, y + 1) + P(x + 1, y + 1)) + 0.5f * (P(x - 2, y) + P(x + 2, y))) * 0.125f;
                int rowColor = cfa_color(cfaType, x + 1, y);
                out[rowColor] = rowK;
                out[2 - rowColor] = colK;
            } else {
                out[1] = (4 * v + 2 * (P(x - 1, y) + P(x + 1, y) + P(x, y - 1) + P(x, y + 1)) - (P(x - 2, y) + P(x + 2, y) + P(x, y - 2) + P(x, y + 2))) * 0.125f;
                out[2 - c] = (6 * v + 2 * (P(x - 1, y - 1) + P(x + 1, y - 1) + P(x - 1, y + 1) + P(x + 1, y + 1))
                              - 1.5f * (P(x - 2, y) + P(x + 2, y) + P(x, y - 2) + P(x, y + 2))) * 0.125f;
            }
            float* o = &rgb.at(x, y);
            for (int k = 0; k < 3; k++) o[k] = std::max(0.f, out[k]);
        }
    });
    return rgb;
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

void chroma_denoise(ImageF& rgb, float strength) {
    if (strength <= 0) return;
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
                    sr += p[0] - L; sb += p[2] - L; n++;
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
            float r = L + bil(cr), b = L + bil(cb);
            float g = (L - 0.2126f * r - 0.0722f * b) / 0.7152f;
            p[0] = r; p[1] = g; p[2] = b;
        }
    });
}

ToneInfo local_tonemap(ImageF& rgb, const LookParams& p) {
    ImageF Y(rgb.w, rgb.h);
    parallel_for(rgb.h, [&](int y) {
        const float* q = rgb.row(y);
        float* o = Y.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) o[x] = std::max(1e-6f, luma709(q[0], q[1], q[2]));
    });
    std::vector<float> samp;
    double lsum = 0;
    for (int y = 0; y < Y.h; y += 8)
        for (int x = 0; x < Y.w; x += 8) { float v = Y.at(x, y); samp.push_back(v); lsum += std::log(v + 1e-4); }
    float key = (float)std::exp(lsum / samp.size());
    std::nth_element(samp.begin(), samp.begin() + (size_t)(samp.size() * 0.995), samp.end());
    float p995 = samp[(size_t)(samp.size() * 0.995)];
    float gL = clampf(p.targetKey / key, 1.f, 64.f);
    float maxRatio = p.compression > 0 ? p.compression : 8.f;
    float gS = clampf(1.f / p995, gL / maxRatio, gL);

    // synthetic short / long exposures, display referred
    ImageF eS(Y.w, Y.h), eL(Y.w, Y.h), wS(Y.w, Y.h);
    parallel_for(Y.h, [&](int y) {
        for (int x = 0; x < Y.w; x++) {
            float s = srgb_oetf(Y.at(x, y) * gS), l = srgb_oetf(Y.at(x, y) * gL);
            float ws = std::exp(-(s - 0.5f) * (s - 0.5f) / (2 * 0.2f * 0.2f)) + 1e-4f;
            float wl = std::exp(-(l - 0.5f) * (l - 0.5f) / (2 * 0.2f * 0.2f)) + 1e-4f;
            eS.at(x, y) = s; eL.at(x, y) = l; wS.at(x, y) = ws / (ws + wl);
        }
    });
    // Laplacian pyramid exposure fusion (Mertens et al. 2007), as in HDR+ finishing
    int levels = 1;
    for (int m = std::min(Y.w, Y.h); m > 24; m /= 2) levels++;
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
    ImageF& acc = gs[0];
    parallel_for(rgb.h, [&](int y) {
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) {
            float ratio = srgb_eotf(acc.at(x, y)) / Y.at(x, y);
            q[0] *= ratio; q[1] *= ratio; q[2] *= ratio;
        }
    });
    return {gS, gL, key};
}

void pixel_look(ImageF& rgb, const LookParams& p) {
    parallel_for(rgb.h, [&](int y) {
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) {
            float mx = std::max(q[0], std::max(q[1], q[2]));
            // coloured highlights (lamp-lit wood, skin) roll off on the max channel with hue and saturation kept,
            // instead of the leading channel clipping first; neutrals are left for the gamut fit below
            constexpr float kKnee = 0.7f;
            if (mx > kKnee) {
                float mn = std::min(q[0], std::min(q[1], q[2]));
                float t = (mx - kKnee) / (1 - kKnee);
                float shoulder = kKnee + (1 - kKnee) * t / (1 + t);
                float chroma = clampf(2 * (mx - std::max(0.f, mn)) / mx, 0.f, 1.f);
                float g = (mx + (shoulder - mx) * chroma) / mx;
                for (int c = 0; c < 3; c++) q[c] *= g;
                mx *= g;
            }
            // fit out-of-gamut highlights by desaturating toward luminance instead of hard per-channel clipping
            float L = std::min(1.f, std::max(0.f, luma709(q[0], q[1], q[2])));
            if (mx > 1.f) { float k = (1.f - L) / std::max(1e-6f, mx - L); for (int c = 0; c < 3; c++) q[c] = L + (q[c] - L) * k; }
            for (int c = 0; c < 3; c++) {
                float v = srgb_oetf(q[c]);
                float s = v * v * (3 - 2 * v);
                q[c] = v + p.contrast * (s - v);
            }
            float Ld = luma709(q[0], q[1], q[2]);
            float sat = std::max(q[0], std::max(q[1], q[2])) - std::min(q[0], std::min(q[1], q[2]));
            float k = 1 + (p.saturation - 1) * (1 - clampf(sat, 0, 1));
            for (int c = 0; c < 3; c++) q[c] = clampf(Ld + (q[c] - Ld) * k, 0.f, 1.f);
        }
    });
}

void sharpen_luma(ImageF& rgb, float amount, float radius) {
    if (amount <= 0) return;
    ImageF L(rgb.w, rgb.h);
    parallel_for(rgb.h, [&](int y) { const float* q = rgb.row(y); for (int x = 0; x < rgb.w; x++, q += 3) L.at(x, y) = luma709(q[0], q[1], q[2]); });
    ImageF B = gaussian_blur(L, radius);
    const float core = 0.004f;
    parallel_for(rgb.h, [&](int y) {
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) {
            float d = L.at(x, y) - B.at(x, y);
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

std::vector<uint8_t> encode_jpeg(const uint8_t* px, int w, int h, int channels, int quality) {
    std::vector<uint8_t> out;
    stbi_write_jpg_to_func([](void* ctx, void* data, int size) {
        auto* o = static_cast<std::vector<uint8_t>*>(ctx);
        o->insert(o->end(), (uint8_t*)data, (uint8_t*)data + size);
    }, &out, w, h, channels, px, quality);
    return out;
}

bool write_jpeg(const std::string& path, const ImageF& rgb, int quality, int ds) {
    int w, h;
    std::vector<uint8_t> buf = to_rgb8(rgb, ds, w, h);
    return stbi_write_jpg(path.c_str(), w, h, 3, buf.data(), quality) != 0;
}

void wb_shift(ImageF& rgb, float cool) {
    if (cool == 0) return;
    // linear sRGB: trade red for blue at constant Rec.709 luminance
    float r = 1 - cool, b = 1 + cool;
    float g = (1 - 0.2126f * r - 0.0722f * b) / 0.7152f;
    parallel_for(rgb.h, [&](int y) {
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) { q[0] *= r; q[1] *= g; q[2] *= b; }
    });
}

void local_contrast(ImageF& rgb, float amount, float radiusFrac) {
    if (amount <= 0) return;
    // self-guided filter (He et al. 2010) on display luma: edge-preserving base, so boosting the
    // detail layer adds clarity without halos around strong edges
    ImageF L(rgb.w, rgb.h);
    parallel_for(rgb.h, [&](int y) { const float* q = rgb.row(y); float* o = L.row(y); for (int x = 0; x < rgb.w; x++, q += 3) o[x] = luma709(q[0], q[1], q[2]); });
    // fast guided filter (He & Sun 2015): coefficients on a 4x subsampled grid, upsampled bilinearly
    constexpr int kSub = 4;
    const int sw = (rgb.w + kSub - 1) / kSub, sh = (rgb.h + kSub - 1) / kSub;
    int r = std::max(1, (int)std::lround(radiusFrac * std::max(rgb.w, rgb.h) / kSub));
    const float eps = 0.01f * 0.01f * 25;
    ImageF Ls(sw, sh), LL(sw, sh);
    parallel_for(sh, [&](int y) {
        for (int x = 0; x < sw; x++) {
            float acc = 0; int cnt = 0;
            for (int v = 0; v < kSub; v++) for (int u = 0; u < kSub; u++) {
                int X = x * kSub + u, Y = y * kSub + v;
                if (X < rgb.w && Y < rgb.h) { acc += L.at(X, Y); cnt++; }
            }
            float m = acc / cnt;
            Ls.at(x, y) = m; LL.at(x, y) = m * m;
        }
    });
    ImageF mI = box_mean(Ls, r), mII = box_mean(LL, r);
    ImageF A(sw, sh), B(sw, sh);
    parallel_for(sh, [&](int y) {
        for (int x = 0; x < sw; x++) {
            float m = mI.at(x, y), v = std::max(0.f, mII.at(x, y) - m * m);
            float a = v / (v + eps);
            A.at(x, y) = a; B.at(x, y) = m - a * m;
        }
    });
    ImageF mA = box_mean(A, r), mB = box_mean(B, r);
    auto up = [&](const ImageF& im, int x, int y) {
        float fx = clampf((x + 0.5f) / kSub - 0.5f, 0.f, sw - 1.001f), fy = clampf((y + 0.5f) / kSub - 0.5f, 0.f, sh - 1.001f);
        int x0 = (int)fx, y0 = (int)fy;
        float ax = fx - x0, ay = fy - y0;
        int x1 = std::min(x0 + 1, sw - 1), y1 = std::min(y0 + 1, sh - 1);
        return (1 - ay) * ((1 - ax) * im.at(x0, y0) + ax * im.at(x1, y0)) + ay * ((1 - ax) * im.at(x0, y1) + ax * im.at(x1, y1));
    };
    parallel_for(rgb.h, [&](int y) {
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) {
            float l = L.at(x, y), base = up(mA, x, y) * l + up(mB, x, y);
            // fade the boost out in deep shadows (noise) and near white (clipping)
            float fade = clampf(l / 0.08f, 0.f, 1.f) * clampf((1.f - l) / 0.1f, 0.f, 1.f);
            float delta = amount * fade * (l - base);
            for (int c = 0; c < 3; c++) q[c] = clampf(q[c] + delta, 0.f, 1.f);
        }
    });
}

void hue_tune(ImageF& rgb, const LookParams& p) {
    if (p.skySat == 1 && p.skinSat == 1 && p.skyHue == 0 && p.skinHue == 0) return;
    // hue windows in the display-referred BT.709 CbCr plane (Cb axis = 0 deg, Cr axis = 90 deg):
    // sky blue around -30 deg (+-40), skin around 122 deg (+-30). Trig-free: the window is a smoothstep
    // of cos(hue - centre) from the chroma direction's dot product with the centre direction.
    struct Win { float cx, cy, c0; };
    auto mk = [](float centre, float half) {
        float r = centre * (float)M_PI / 180.f;
        return Win{std::cos(r), std::sin(r), std::cos(half * (float)M_PI / 180.f)};
    };
    const Win sky = mk(-30.f, 40.f), skin = mk(122.f, 30.f);
    auto weight = [](const Win& w, float ux, float uy) {
        float t = clampf((ux * w.cx + uy * w.cy - w.c0) / (1.f - w.c0), 0.f, 1.f);
        return t * t * (3 - 2 * t);
    };
    const float skyRot = p.skyHue * (float)M_PI / 180.f, skinRot = p.skinHue * (float)M_PI / 180.f;
    parallel_for(rgb.h, [&](int y) {
        float* q = rgb.row(y);
        for (int x = 0; x < rgb.w; x++, q += 3) {
            float Y = luma709(q[0], q[1], q[2]);
            float cb = (q[2] - Y) / 1.8556f, cr = (q[0] - Y) / 1.5748f;
            float c2 = cb * cb + cr * cr;
            if (c2 < 1e-8f) continue;
            float inv = 1.f / std::sqrt(c2), chroma = c2 * inv;
            float cw = clampf(chroma / 0.04f, 0.f, 1.f);
            float ws = weight(sky, cb * inv, cr * inv) * cw, wk = weight(skin, cb * inv, cr * inv) * cw;
            if (ws + wk <= 0.f) continue;
            float sat = 1 + ws * (p.skySat - 1) + wk * (p.skinSat - 1);
            float th = ws * skyRot + wk * skinRot;  // a few degrees: first-order rotation is enough
            float ncb = sat * (cb - th * cr), ncr = sat * (cr + th * cb);
            float r = Y + 1.5748f * ncr, b = Y + 1.8556f * ncb;
            float g = (Y - 0.2126f * r - 0.0722f * b) / 0.7152f;
            q[0] = clampf(r, 0.f, 1.f); q[1] = clampf(g, 0.f, 1.f); q[2] = clampf(b, 0.f, 1.f);
        }
    });
}

bool apply_look_preset(const std::string& name, LookParams& p) {
    if (name == "default") {
        // a little brighter and crisper than the neutral parameters, still well below pixel's noise
        p = LookParams();
        p.targetKey = 0.23f; p.sharpen = 0.8f; p.localContrast = 0.3f;
        return true;
    }
    if (name == "pixel") {
        // brighter mid-tones, stronger compression and local contrast, crisper, slightly cool,
        // deeper sky blue and gentler skin saturation
        p = LookParams();
        p.targetKey = 0.22f; p.compression = 6.f; p.contrast = 0.26f; p.saturation = 1.24f;
        p.sharpen = 0.9f; p.localContrast = 0.6f; p.wbCool = 0.025f;
        p.skySat = 1.18f; p.skyHue = -4.f; p.skinSat = 0.92f; p.skinHue = 3.f;
        return true;
    }
    if (name == "night") {
        // keep night dark: lower key, softer compression, stronger chroma denoise, gentle sharpening
        p = LookParams();
        p.targetKey = 0.11f; p.compression = 4.f; p.contrast = 0.18f; p.saturation = 1.05f;
        p.sharpen = 0.4f; p.localContrast = 0.15f; p.chromaDenoiseScale = 1.6f;
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
