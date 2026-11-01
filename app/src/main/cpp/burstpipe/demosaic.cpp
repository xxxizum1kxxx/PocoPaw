#include "demosaic.h"

#include "finish.h"
#include "third_party/rtprocess/librtprocess.h"

bool parse_demosaic(const std::string& s, Demosaic& d) {
    if (s == "mhc") d = Demosaic::MHC;
    else if (s == "rcd") d = Demosaic::RCD;
    else if (s == "amaze") d = Demosaic::AMAZE;
    else return false;
    return true;
}

const char* demosaic_name(Demosaic d) {
    switch (d) {
        case Demosaic::RCD: return "rcd";
        case Demosaic::AMAZE: return "amaze";
        default: return "mhc";
    }
}

ImageF demosaic(const ImageF& cfa, int cfaType, Demosaic method) {
    if (method == Demosaic::MHC) return demosaic_mhc(cfa, cfaType);
    const int W = cfa.w, H = cfa.h;
    unsigned pattern[2][2];
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++) pattern[y][x] = (unsigned)cfa_color(cfaType, x, y);

    // RCD clamps its input to [0, 65536]; AMaZE takes explicit input/output scales
    const float scale = method == Demosaic::RCD ? 65536.f : 1.f;
    ImageF in(W, H);
    parallel_for(H, [&](int y) {
        const float* s = cfa.row(y);
        float* d = in.row(y);
        for (int x = 0; x < W; x++) d[x] = std::max(0.f, s[x]) * scale;
    });
    ImageF planes[3] = {ImageF(W, H), ImageF(W, H), ImageF(W, H)};

    // strips start on even rows so every strip sees the same CFA phase; the overlap is larger
    // than both algorithms' borders (RCD 9, AMaZE 16) and is written to scratch rows
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
        if (method == Demosaic::RCD)
            rcd_demosaic(W, h, raw.data(), out[0].data(), out[1].data(), out[2].data(), pattern, noCancel, 2, false, false);
        else
            amaze_demosaic(W, h, 0, 0, W, h, raw.data(), out[0].data(), out[1].data(), out[2].data(), pattern, noCancel,
                           1.0, 0, 1.f, 1.f);
    });

    ImageF rgb(W, H, 3);
    const float inv = 1.f / scale;
    parallel_for(H, [&](int y) {
        float* o = rgb.row(y);
        const float *r = planes[0].row(y), *g = planes[1].row(y), *bl = planes[2].row(y);
        for (int x = 0; x < W; x++, o += 3) {
            o[0] = std::max(0.f, r[x] * inv);
            o[1] = std::max(0.f, g[x] * inv);
            o[2] = std::max(0.f, bl[x] * inv);
        }
    });
    return rgb;
}
