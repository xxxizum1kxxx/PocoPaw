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
