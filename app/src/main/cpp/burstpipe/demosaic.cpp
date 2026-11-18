#include "demosaic.h"

#include "finish.h"
#include "third_party/rtprocess/librtprocess.h"

bool parse_demosaic(const std::string& s, Demosaic& d) {
    if (s == "rcd" || s == "mhc" || s == "amaze") {
        d = Demosaic::RCD;
        return true;
    }
    return false;
}

const char* demosaic_name(Demosaic) { return "rcd"; }

ImageF demosaic(const ImageF& cfa, int cfaType, Demosaic) {
    const int W = cfa.w, H = cfa.h;
    unsigned pattern[2][2];
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++) pattern[y][x] = (unsigned)cfa_color(cfaType, x, y);

    // RCD clamps its input to [0, 65536]. A small normalized pedestal keeps zero-mean shadow noise
    // positive through demosaicing without rectifying shadow chroma.
    constexpr float kPed = 1.f / 16.f;
    constexpr float kScale = 65536.f;
    ImageF in(W, H);
    parallel_for(H, [&](int y) {
        const float* s = cfa.row(y);
        float* d = in.row(y);
        for (int x = 0; x < W; x++) d[x] = clampf((s[x] + kPed) / (1.f + kPed), 0.f, 1.f) * kScale;
    });
    ImageF planes[3] = {ImageF(W, H), ImageF(W, H), ImageF(W, H)};

    constexpr int kPad = 24;
    const int nStrips = std::max(1, std::min(H / 96, 3 * g_threads));
    auto noCancel = [](double) { return false; };
    parallel_for(nStrips, [&](int s) {
        int y0 = (int)((long)H * s / nStrips) & ~1, y1 = s + 1 == nStrips ? H : (int)((long)H * (s + 1) / nStrips) & ~1;
        int a = std::max(0, y0 - kPad), b = std::min(H, y1 + kPad), h = b - a;
        std::vector<const float*> raw(h);
        std::vector<float*> out[3];
        std::vector<float> scratch((size_t)3 * (y0 - a + b - y1) * W);
        size_t used = 0;
        for (int c = 0; c < 3; c++) out[c].resize(h);
        for (int r = 0; r < h; r++) {
            int y = a + r;
            raw[r] = in.row(y);
            for (int c = 0; c < 3; c++) {
                if (y >= y0 && y < y1) out[c][r] = planes[c].row(y);
                else { out[c][r] = scratch.data() + used; used += W; }
            }
        }
        rcd_demosaic(W, h, raw.data(), out[0].data(), out[1].data(), out[2].data(), pattern, noCancel, 2, false, false);
    });

    ImageF rgb(W, H, 3);
    constexpr float kUnscale = (1.f + kPed) / kScale;
    parallel_for(H, [&](int y) {
        float* o = rgb.row(y);
        const float *r = planes[0].row(y), *g = planes[1].row(y), *bl = planes[2].row(y);
        for (int x = 0; x < W; x++, o += 3) {
            o[0] = r[x] * kUnscale - kPed;
            o[1] = g[x] * kUnscale - kPed;
            o[2] = bl[x] * kUnscale - kPed;
        }
    });
    return rgb;
}
