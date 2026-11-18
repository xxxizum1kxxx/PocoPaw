// Frequency-domain robust merge (Hasinoff et al. 2016, Sec. 5; parameters after hdr-plus-swift).
// The four Bayer planes are processed together, one per SIMD lane.
#include "merge.h"

namespace {

typedef float f4 __attribute__((ext_vector_type(4)));

constexpr int T = 16;  // tile size in plane pixels (= quads), same grid as the alignment tiles
constexpr int S = 8;   // stride (half overlap)

float g_sin[T];        // sine window; its square sums to 1 under half overlap (analysis + synthesis)
float g_twr[T / 2], g_twi[T / 2];
float g_wr[T], g_wi[T]; // 1D DFT of g_sin, so mean * W decouples DC window cusp from phase shift
int g_rev[T];

struct Init {
    Init() {
        for (int u = 0; u < T; u++) g_sin[u] = std::sin((float)M_PI * (u + 0.5f) / T);
        for (int k = 0; k < T / 2; k++) { g_twr[k] = std::cos(2 * (float)M_PI * k / T); g_twi[k] = -std::sin(2 * (float)M_PI * k / T); }
        for (int k = 0; k < T; k++) {
            double re = 0, im = 0;
            for (int u = 0; u < T; u++) {
                double a = -2.0 * M_PI * k * u / T;
                re += g_sin[u] * std::cos(a);
                im += g_sin[u] * std::sin(a);
            }
            g_wr[k] = (float)re;
            g_wi[k] = (float)im;
        }
        for (int i = 0; i < T; i++) {
            int r = 0;
            for (int b = 0; b < 4; b++) if (i & (1 << b)) r |= 1 << (3 - b);
            g_rev[i] = r;
        }
    }
} g_init;

// in-place radix-2 FFT of 16 points at re[k*stride], im[k*stride]; inverse is unnormalised
inline void fft16(f4* re, f4* im, int stride, bool inverse) {
    for (int i = 0; i < T; i++) {
        int j = g_rev[i];
        if (j > i) { std::swap(re[i * stride], re[j * stride]); std::swap(im[i * stride], im[j * stride]); }
    }
    for (int len = 2; len <= T; len <<= 1) {
        int half = len / 2, step = T / len;
        for (int s = 0; s < T; s += len)
            for (int k = 0; k < half; k++) {
                float wr = g_twr[k * step], wi = inverse ? -g_twi[k * step] : g_twi[k * step];
                f4& ar = re[(s + k) * stride]; f4& ai = im[(s + k) * stride];
                f4& br = re[(s + k + half) * stride]; f4& bi = im[(s + k + half) * stride];
                f4 tr = br * wr - bi * wi, ti = br * wi + bi * wr;
                br = ar - tr; bi = ai - ti;
                ar += tr; ai += ti;
            }
    }
}

inline void fft2d(f4* re, f4* im, bool inverse) {
    for (int v = 0; v < T; v++) fft16(re + v * T, im + v * T, 1, inverse);
    for (int u = 0; u < T; u++) fft16(re + u, im + u, T, inverse);
}

// windowed tile of the 4 Bayer planes, quad origin (x0,y0), clamped at the image border.
// When `zeroMean` is true, subtracts the tile's per-plane mean before windowing so continuous
// Fourier phase shifts only act on zero-mean texture (preventing window-cusp DC ringing).
inline f4 load_tile(const Raw& F, int x0, int y0, f4* re, f4* im, bool zeroMean = false) {
    const int qw = F.w / 2, qh = F.h / 2;
    f4 s = 0;
    f4 raw[T * T];
    for (int v = 0; v < T; v++) {
        int y = clampi(y0 + v, 0, qh - 1);
        const uint16_t* r0 = F.row(2 * y);
        const uint16_t* r1 = F.row(2 * y + 1);
        for (int u = 0; u < T; u++) {
            int x = clampi(x0 + u, 0, qw - 1);
            f4 p = {(float)r0[2 * x], (float)r0[2 * x + 1], (float)r1[2 * x], (float)r1[2 * x + 1]};
            s += p;
            raw[v * T + u] = p;
        }
    }
    f4 mean = s / (float)(T * T);
    f4 sub = zeroMean ? mean : (f4)0;
    for (int v = 0; v < T; v++)
        for (int u = 0; u < T; u++) {
            int i = v * T + u;
            re[i] = (raw[i] - sub) * (g_sin[u] * g_sin[v]);
            im[i] = 0;
        }
    return mean;
}

// the tile at x0 shifted by d still lands inside [0, q) wherever the unshifted tile does
inline bool shift_inside(int x0, int d, int q) {
    int lo = std::max(0, -x0), hi = std::min(T, q - x0);
    return lo >= hi || (x0 + d + lo >= 0 && x0 + d + hi <= q);
}

inline f4 load_f4(const float* p) { return f4{p[0], p[1], p[2], p[3]}; }
inline void store_f4(float* p, f4 v) { p[0] = v[0]; p[1] = v[1]; p[2] = v[2]; p[3] = v[3]; }

// Hermitian 128-bin packing for a 16x16 real-signal 2D DFT:
// slot 0: re = Re(0,0), im = Re(0,8)
// slot 1..7: v=0, u=1..7
// slot 8: re = Re(8,0), im = Re(8,8)
// slot 9..15: v=8, u=16-slot (i.e. u=1..7 at slot=16-u)
// slot v*16+u (v=1..7, u=0..15): bin (u,v)
inline void pack_hermitian(const f4* r, const f4* i, float* ar, float* ai) {
    store_f4(ar + 0 * 4, r[0 * T + 0]);
    store_f4(ai + 0 * 4, r[8 * T + 0]);
    store_f4(ar + 8 * 4, r[0 * T + 8]);
    store_f4(ai + 8 * 4, r[8 * T + 8]);
    for (int u = 1; u < 8; u++) {
        store_f4(ar + u * 4, r[u]);
        store_f4(ai + u * 4, i[u]);
        store_f4(ar + (16 - u) * 4, r[8 * T + u]);
        store_f4(ai + (16 - u) * 4, i[8 * T + u]);
    }
    for (int v = 1; v < 8; v++)
        for (int u = 0; u < T; u++) {
            int s = v * 16 + u;
            store_f4(ar + s * 4, r[v * T + u]);
            store_f4(ai + s * 4, i[v * T + u]);
        }
}

inline void unpack_hermitian(const float* ar, const float* ai, f4* r, f4* i) {
    r[0 * T + 0] = load_f4(ar + 0 * 4); i[0 * T + 0] = (f4)0;
    r[8 * T + 0] = load_f4(ai + 0 * 4); i[8 * T + 0] = (f4)0;
    r[0 * T + 8] = load_f4(ar + 8 * 4); i[0 * T + 8] = (f4)0;
    r[8 * T + 8] = load_f4(ai + 8 * 4); i[8 * T + 8] = (f4)0;
    for (int u = 1; u < 8; u++) {
        f4 r0 = load_f4(ar + u * 4), i0 = load_f4(ai + u * 4);
        r[u] = r0; i[u] = i0;
        r[16 - u] = r0; i[16 - u] = -i0;
        f4 r8 = load_f4(ar + (16 - u) * 4), i8 = load_f4(ai + (16 - u) * 4);
        r[8 * T + u] = r8; i[8 * T + u] = i8;
        r[8 * T + (16 - u)] = r8; i[8 * T + (16 - u)] = -i8;
    }
    for (int v = 1; v < 8; v++)
        for (int u = 0; u < T; u++) {
            int s = v * 16 + u;
            f4 rv = load_f4(ar + s * 4), iv = load_f4(ai + s * 4);
            r[v * T + u] = rv; i[v * T + u] = iv;
            int u2 = (16 - u) & 15, v2 = 16 - v;
            r[v2 * T + u2] = rv; i[v2 * T + u2] = -iv;
        }
}

}  // namespace

ImageF merge_frames_freq(const std::vector<Raw>& frames, int ref, const std::vector<Offsets>& offs,
                         const Meta& m, const NoiseModel& nm, float c, float spatial, MergeStats* st) {
    const Raw& R = frames[ref];
    const int qw = R.w / 2, qh = R.h / 2, n = (int)frames.size();
    const Offsets& o0 = offs[ref];
    // black level per lane (quad position), and the summed squared window = noise gain of the DFT
    f4 black;
    for (int l = 0; l < 4; l++) black[l] = m.black[cfa_channel(m.cfa, l & 1, l >> 1)];
    float w2 = 0;
    for (int v = 0; v < T; v++) for (int u = 0; u < T; u++) w2 += g_sin[u] * g_sin[u] * g_sin[v] * g_sin[v];

    ImageF out(R.w, R.h);
    std::fill(out.d.begin(), out.d.end(), 0.f);
    // tiles at origins t*S for t = -1 .. ceil(q/S): every quad is covered by four tiles
    const int ntx = (qw + S - 1) / S + 1, nty = (qh + S - 1) / S + 1;
    std::vector<double> rowW(nty, 0.0);
    for (int parity = 0; parity < 2; parity++) {
        // tile rows of equal parity do not overlap, so they can be accumulated concurrently
        parallel_for((nty + 1 - parity) / 2, [&](int jj) {
            int tj = 2 * jj + parity;
            alignas(16) f4 r0[T * T], i0[T * T], rk[T * T], ik[T * T], ar[T * T], ai[T * T];
            double wsum = 0;
            for (int ti = 0; ti < ntx; ti++) {
                int x0 = (ti - 1) * S, y0 = (tj - 1) * S;
                f4 mean = load_tile(R, x0, y0, r0, i0, false);
                fft2d(r0, i0, false);
                // per-coefficient noise variance of a frame difference, from the raw noise model
                f4 sig;
                for (int l = 0; l < 4; l++) sig[l] = (float)nm.var(mean[l] - black[l]);
                f4 noiseD = 2.f * w2 * sig * c;
                for (int i = 0; i < T * T; i++) { ar[i] = r0[i]; ai[i] = i0[i]; }
                int ai_ = clampi(ti - 1, 0, o0.nx - 1), aj_ = clampi(tj - 1, 0, o0.ny - 1);
                float tileW = 0;
                for (int k = 0; k < n; k++) {
                    if (k == ref) continue;
                    int dx = offs[k].x(ai_, aj_), dy = offs[k].y(ai_, aj_);
                    // clamping would repeat the border into streaks: the reference stands in for this frame
                    if (!shift_inside(x0, dx, qw) || !shift_inside(y0, dy, qh)) {
                        for (int i = 0; i < T * T; i++) { ar[i] += r0[i]; ai[i] += i0[i]; }
                        continue;
                    }
                    float fx = offs[k].subx(ai_, aj_), fy = offs[k].suby(ai_, aj_);
                    bool sub = (fx != 0.f || fy != 0.f);
                    f4 mk = load_tile(frames[k], x0 + dx, y0 + dy, rk, ik, sub);
                    fft2d(rk, ik, false);
                    float pxr[T], pxi[T], pyr[T], pyi[T];
                    if (sub) {
                        for (int m = 0; m < T; m++) {
                            int sm = m < 8 ? m : (m > 8 ? m - 16 : 0);
                            float ax = (2.f * (float)M_PI / T) * (sm * fx);
                            float ay = (2.f * (float)M_PI / T) * (sm * fy);
                            pxr[m] = std::cos(ax); pxi[m] = std::sin(ax);
                            pyr[m] = std::cos(ay); pyi[m] = std::sin(ay);
                        }
                    }
                    f4 wacc = 0;
                    for (int v = 0; v < T; v++)
                        for (int u = 0; u < T; u++) {
                            int i = v * T + u;
                            f4 skr, ski;
                            if (sub) {
                                float pr = pxr[u] * pyr[v] - pxi[u] * pyi[v];
                                float pi = pxr[u] * pyi[v] + pxi[u] * pyr[v];
                                float wr = g_wr[u] * g_wr[v] - g_wi[u] * g_wi[v];
                                float wi = g_wr[u] * g_wi[v] + g_wi[u] * g_wr[v];
                                skr = rk[i] * pr - ik[i] * pi + mk * wr;
                                ski = rk[i] * pi + ik[i] * pr + mk * wi;
                            } else {
                                skr = rk[i];
                                ski = ik[i];
                            }
                            f4 dr = r0[i] - skr, di = i0[i] - ski;
                            f4 d2 = dr * dr + di * di;
                            f4 a = d2 / (d2 + noiseD);
                            // pairwise Wiener merge: keep the alternate where it agrees, the reference where not
                            ar[i] += skr + a * dr;
                            ai[i] += ski + a * di;
                            wacc += 1.f - a;
                        }
                    tileW += (wacc[0] + wacc[1] + wacc[2] + wacc[3]) / (4.f * T * T);
                }
                float inv = 1.f / n;
                // optional spatial shrinkage of the merged tile (HDR+ Sec. 5.2), stronger at high frequencies
                f4 sigM = w2 * sig * spatial / (float)n;
                float sigMs = sigM[0] + sigM[1] + sigM[2] + sigM[3];
                for (int v = 0; v < T; v++)
                    for (int u = 0; u < T; u++) {
                        int i = v * T + u;
                        ar[i] *= inv; ai[i] *= inv;
                        if (spatial > 0 && i) {
                            int fu = std::min(u, T - u), fv = std::min(v, T - v);
                            float boost = 1.f + 0.25f * std::sqrt((float)(fu * fu + fv * fv));
                            // one gain for all four planes: independent shrinkage of G_even and G_odd
                            // leaves a Bayer-period mismatch that demosaicing turns into a maze pattern
                            f4 p4 = ar[i] * ar[i] + ai[i] * ai[i];
                            float p = p4[0] + p4[1] + p4[2] + p4[3];
                            float g = p / (p + sigMs * boost);
                            ar[i] *= g; ai[i] *= g;
                        }
                    }
                fft2d(ar, ai, true);
                const float norm = 1.f / (T * T);
                for (int v = 0; v < T; v++) {
                    int y = y0 + v;
                    if (y < 0 || y >= qh) continue;
                    float* o0r = out.row(2 * y);
                    float* o1r = out.row(2 * y + 1);
                    for (int u = 0; u < T; u++) {
                        int x = x0 + u;
                        if (x < 0 || x >= qw) continue;
                        f4 p = ar[v * T + u] * (norm * g_sin[u] * g_sin[v]);
                        o0r[2 * x] += p[0]; o0r[2 * x + 1] += p[1];
                        o1r[2 * x] += p[2]; o1r[2 * x + 1] += p[3];
                    }
                }
                wsum += n > 1 ? tileW / (n - 1) : 0;
            }
            rowW[tj] = wsum / ntx;
        });
    }
    if (st) { double s = 0; for (double v : rowW) s += v; st->meanWeight = s / nty; }
    return out;
}

FreqStreamMerge::FreqStreamMerge(const Raw& ref, const Meta& m, float c, float spatial)
    : ref_(ref), qw_(ref.w / 2), qh_(ref.h / 2), ntx_((qw_ + S - 1) / S + 1), nty_((qh_ + S - 1) / S + 1),
      c_(c), spatial_(spatial) {
    for (int l = 0; l < 4; l++) black_[l] = m.black[cfa_channel(m.cfa, l & 1, l >> 1)];
    w2_ = 0.f;
    for (int v = 0; v < T; v++) for (int u = 0; u < T; u++) w2_ += g_sin[u] * g_sin[u] * g_sin[v] * g_sin[v];
    const size_t nTiles = (size_t)ntx_ * nty_;
    ar_.resize(nTiles * 128 * 4);
    ai_.resize(nTiles * 128 * 4);
    mean_.resize(nTiles * 4);
    tileW_.assign(nTiles, 0.0);
    parallel_for(nty_, [&](int tj) {
        alignas(16) f4 r0[T * T], i0[T * T];
        for (int ti = 0; ti < ntx_; ti++) {
            int x0 = (ti - 1) * S, y0 = (tj - 1) * S;
            f4 mean = load_tile(ref_, x0, y0, r0, i0, false);
            fft2d(r0, i0, false);
            size_t tid = (size_t)tj * ntx_ + ti;
            store_f4(mean_.data() + tid * 4, mean);
            pack_hermitian(r0, i0, ar_.data() + tid * 128 * 4, ai_.data() + tid * 128 * 4);
        }
    });
}

void FreqStreamMerge::add(const Raw& F, const Offsets& of, const NoiseModel& nm) {
    parallel_for(nty_, [&](int tj) {
        alignas(16) f4 r0[T * T], i0[T * T], rk[T * T], ik[T * T];
        for (int ti = 0; ti < ntx_; ti++) {
            int x0 = (ti - 1) * S, y0 = (tj - 1) * S;
            size_t tid = (size_t)tj * ntx_ + ti;
            float* arp = ar_.data() + tid * 128 * 4;
            float* aip = ai_.data() + tid * 128 * 4;
            int ai_ = clampi(ti - 1, 0, of.nx - 1), aj_ = clampi(tj - 1, 0, of.ny - 1);
            int dx = of.x(ai_, aj_), dy = of.y(ai_, aj_);

            load_tile(ref_, x0, y0, r0, i0, false);
            fft2d(r0, i0, false);
            if (!shift_inside(x0, dx, qw_) || !shift_inside(y0, dy, qh_)) {
                auto add_real = [&](int slot, f4 v, float* dst) { store_f4(dst + slot * 4, load_f4(dst + slot * 4) + v); };
                add_real(0, r0[0 * T + 0], arp);
                add_real(0, r0[8 * T + 0], aip);
                add_real(8, r0[0 * T + 8], arp);
                add_real(8, r0[8 * T + 8], aip);
                for (int u = 1; u < 8; u++) {
                    add_real(u, r0[u], arp);
                    add_real(u, i0[u], aip);
                    add_real(16 - u, r0[8 * T + u], arp);
                    add_real(16 - u, i0[8 * T + u], aip);
                }
                for (int v = 1; v < 8; v++)
                    for (int u = 0; u < T; u++) {
                        int s = v * 16 + u;
                        add_real(s, r0[v * T + u], arp);
                        add_real(s, i0[v * T + u], aip);
                    }
                continue;
            }

            f4 mean = load_f4(mean_.data() + tid * 4);
            f4 sig;
            for (int l = 0; l < 4; l++) sig[l] = (float)nm.var(mean[l] - black_[l]);
            f4 noiseD = 2.f * w2_ * sig * c_;

            float fx = of.subx(ai_, aj_), fy = of.suby(ai_, aj_);
            bool sub = (fx != 0.f || fy != 0.f);
            f4 mk = load_tile(F, x0 + dx, y0 + dy, rk, ik, sub);
            fft2d(rk, ik, false);
            float pxr[T], pxi[T], pyr[T], pyi[T];
            if (sub) {
                for (int m = 0; m < T; m++) {
                    int sm = m < 8 ? m : (m > 8 ? m - 16 : 0);
                    float ax = (2.f * (float)M_PI / T) * (sm * fx);
                    float ay = (2.f * (float)M_PI / T) * (sm * fy);
                    pxr[m] = std::cos(ax); pxi[m] = std::sin(ax);
                    pyr[m] = std::cos(ay); pyi[m] = std::sin(ay);
                }
            }

            f4 wacc = 0;
            auto wiener_complex = [&](int slot, int u, int v, float mult) {
                int i = v * T + u;
                f4 skr, ski;
                if (sub) {
                    float pr = pxr[u] * pyr[v] - pxi[u] * pyi[v];
                    float pi = pxr[u] * pyi[v] + pxi[u] * pyr[v];
                    float wr = g_wr[u] * g_wr[v] - g_wi[u] * g_wi[v];
                    float wi = g_wr[u] * g_wi[v] + g_wi[u] * g_wr[v];
                    skr = rk[i] * pr - ik[i] * pi + mk * wr;
                    ski = rk[i] * pi + ik[i] * pr + mk * wi;
                } else {
                    skr = rk[i];
                    ski = ik[i];
                }
                f4 dr = r0[i] - skr, di = i0[i] - ski;
                f4 d2 = dr * dr + di * di;
                f4 a = d2 / (d2 + noiseD);
                store_f4(arp + slot * 4, load_f4(arp + slot * 4) + skr + a * dr);
                store_f4(aip + slot * 4, load_f4(aip + slot * 4) + ski + a * di);
                wacc += mult * (1.f - a);
            };
            auto wiener_real = [&](int slot, int u, int v, float* dst) {
                int i = v * T + u;
                f4 skr = sub ? (rk[i] * (pxr[u] * pyr[v] - pxi[u] * pyi[v]) - ik[i] * (pxr[u] * pyi[v] + pxi[u] * pyr[v]) +
                                mk * (g_wr[u] * g_wr[v] - g_wi[u] * g_wi[v]))
                             : rk[i];
                f4 dr = r0[i] - skr;
                f4 d2 = dr * dr;
                f4 a = d2 / (d2 + noiseD);
                store_f4(dst + slot * 4, load_f4(dst + slot * 4) + skr + a * dr);
                wacc += 1.f - a;
            };

            wiener_real(0, 0, 0, arp);
            wiener_real(0, 0, 8, aip);
            wiener_real(8, 8, 0, arp);
            wiener_real(8, 8, 8, aip);
            for (int u = 1; u < 8; u++) {
                wiener_complex(u, u, 0, 2.f);
                wiener_complex(16 - u, u, 8, 2.f);
            }
            for (int v = 1; v < 8; v++)
                for (int u = 0; u < T; u++)
                    wiener_complex(v * 16 + u, u, v, 2.f);

            tileW_[tid] += (wacc[0] + wacc[1] + wacc[2] + wacc[3]) / (4.f * T * T);
        }
    });
    added_++;
}

double FreqStreamMerge::neff() const {
    if (added_ <= 0 || tileW_.empty() || ntx_ <= 0 || nty_ <= 0) return 1.0;
    double s = 0.0;
    for (double v : tileW_) s += v;
    return 1.0 + s / ((double)ntx_ * nty_);
}

ImageF FreqStreamMerge::result(const NoiseModel& nm, MergeStats* st) {
    if (added_ == 0) {
        ImageF out(ref_.w, ref_.h);
        for (size_t i = 0; i < out.d.size(); i++) out.d[i] = ref_.d[i];
        if (st) st->meanWeight = 0.0;
        ar_.clear(); ai_.clear(); mean_.clear(); tileW_.clear(); ref_ = Raw();
        return out;
    }
    const int n = added_ + 1;
    const float inv = 1.f / n;
    ImageF out(ref_.w, ref_.h);
    std::fill(out.d.begin(), out.d.end(), 0.f);
    for (int parity = 0; parity < 2; parity++) {
        parallel_for((nty_ + 1 - parity) / 2, [&](int jj) {
            int tj = 2 * jj + parity;
            alignas(16) f4 ar[T * T], ai[T * T];
            for (int ti = 0; ti < ntx_; ti++) {
                size_t tid = (size_t)tj * ntx_ + ti;
                unpack_hermitian(ar_.data() + tid * 128 * 4, ai_.data() + tid * 128 * 4, ar, ai);
                f4 mean = load_f4(mean_.data() + tid * 4);
                f4 sig;
                for (int l = 0; l < 4; l++) sig[l] = (float)nm.var(mean[l] - black_[l]);
                f4 sigM = w2_ * sig * spatial_ / (float)n;
                float sigMs = sigM[0] + sigM[1] + sigM[2] + sigM[3];
                for (int v = 0; v < T; v++)
                    for (int u = 0; u < T; u++) {
                        int i = v * T + u;
                        ar[i] *= inv; ai[i] *= inv;
                        if (spatial_ > 0 && i) {
                            int fu = std::min(u, T - u), fv = std::min(v, T - v);
                            float boost = 1.f + 0.25f * std::sqrt((float)(fu * fu + fv * fv));
                            f4 p4 = ar[i] * ar[i] + ai[i] * ai[i];
                            float p = p4[0] + p4[1] + p4[2] + p4[3];
                            float g = p / (p + sigMs * boost);
                            ar[i] *= g; ai[i] *= g;
                        }
                    }
                fft2d(ar, ai, true);
                int x0 = (ti - 1) * S, y0 = (tj - 1) * S;
                const float norm = 1.f / (T * T);
                for (int v = 0; v < T; v++) {
                    int y = y0 + v;
                    if (y < 0 || y >= qh_) continue;
                    float* o0r = out.row(2 * y);
                    float* o1r = out.row(2 * y + 1);
                    for (int u = 0; u < T; u++) {
                        int x = x0 + u;
                        if (x < 0 || x >= qw_) continue;
                        f4 p = ar[v * T + u] * (norm * g_sin[u] * g_sin[v]);
                        o0r[2 * x] += p[0]; o0r[2 * x + 1] += p[1];
                        o1r[2 * x] += p[2]; o1r[2 * x + 1] += p[3];
                    }
                }
            }
        });
    }
    if (st) {
        double s = 0;
        for (double v : tileW_) s += v;
        st->meanWeight = s / ((double)added_ * ntx_ * nty_);
    }
    ar_.clear(); ai_.clear(); mean_.clear(); tileW_.clear(); ref_ = Raw();
    return out;
}
