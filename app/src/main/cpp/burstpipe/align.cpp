#include "align.h"

#include <climits>

namespace {

const int kFactor[3] = {2, 4, 4};           // level l -> l+1
const int kTileL[4] = {16, 16, 16, 8};
const int kRadius[4] = {1, 4, 4, 4};

ImageF downsample(const ImageF& in, int f) {
    ImageF out(in.w / f, in.h / f);
    float inv = 1.f / (f * f);
    parallel_for(out.h, [&](int y) {
        for (int x = 0; x < out.w; x++) {
            float s = 0;
            for (int v = 0; v < f; v++) {
                const float* r = in.row(y * f + v) + x * f;
                for (int u = 0; u < f; u++) s += r[u];
            }
            out.at(x, y) = s * inv;
        }
    });
    return out;
}

struct Grid {
    int tile, stride, nx, ny;
    Grid(const ImageF& im, int t) : tile(t), stride(t / 2) {
        nx = (im.w + stride - 1) / stride;
        ny = (im.h + stride - 1) / stride;
    }
};

float tile_cost(const ImageF& ref, const ImageF& alt, int x0, int y0, int t, int ox, int oy, bool l1, float best) {
    float s = 0;
    bool inside = x0 >= 0 && y0 >= 0 && x0 + t <= ref.w && y0 + t <= ref.h &&
                  x0 + ox >= 0 && y0 + oy >= 0 && x0 + ox + t <= alt.w && y0 + oy + t <= alt.h;
    for (int v = 0; v < t; v++) {
        if (inside) {
            const float* a = ref.row(y0 + v) + x0;
            const float* b = alt.row(y0 + v + oy) + x0 + ox;
            if (l1) for (int u = 0; u < t; u++) s += std::fabs(a[u] - b[u]);
            else for (int u = 0; u < t; u++) { float d = a[u] - b[u]; s += d * d; }
        } else {
            for (int u = 0; u < t; u++) {
                float d = ref.atc(x0 + u, y0 + v) - alt.atc(x0 + u + ox, y0 + v + oy);
                s += l1 ? std::fabs(d) : d * d;
            }
        }
        if (s > best) return s;  // early out
    }
    return s;
}

}  // namespace

ImageF gray_from_raw(const Raw& r) {
    ImageF g(r.w / 2, r.h / 2);
    parallel_for(g.h, [&](int y) {
        const uint16_t* a = r.row(2 * y);
        const uint16_t* b = a + r.w;
        float* o = g.row(y);
        for (int x = 0; x < g.w; x++) o[x] = (float)a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1];
    });
    return g;
}

std::vector<ImageF> build_pyramid(const ImageF& g0) {
    std::vector<ImageF> p;
    p.push_back(g0);
    for (int l = 0; l < 3; l++) p.push_back(downsample(p.back(), kFactor[l]));
    return p;
}

ImageF box3(const ImageF& g) {
    ImageF t(g.w, g.h), o(g.w, g.h);
    parallel_for(g.h, [&](int y) {
        for (int x = 0; x < g.w; x++) t.at(x, y) = g.atc(x - 1, y) + g.at(x, y) + g.atc(x + 1, y);
    });
    parallel_for(g.h, [&](int y) {
        for (int x = 0; x < g.w; x++) o.at(x, y) = (t.atc(x, y - 1) + t.at(x, y) + t.atc(x, y + 1)) * (1.f / 9.f);
    });
    return o;
}

Offsets align_frames(const std::vector<ImageF>& ref, const std::vector<ImageF>& alt) {
    std::vector<int16_t> pdx, pdy;  // offsets of the coarser level
    Grid pg(ref[3], kTileL[3]);
    for (int l = 3; l >= 0; l--) {
        Grid g(ref[l], kTileL[l]);
        std::vector<int16_t> dx((size_t)g.nx * g.ny), dy(dx.size());
        bool l1 = (l == 0);
        int R = kRadius[l];
        parallel_for(g.ny, [&](int j) {
            for (int i = 0; i < g.nx; i++) {
                int x0 = i * g.stride, y0 = j * g.stride;
                // candidate initial offsets from the coarser level: nearest parent tile + 2 neighbours
                int cand[3][2] = {{0, 0}, {0, 0}, {0, 0}};
                int nc = 1;
                if (l < 3) {
                    int f = kFactor[l];
                    float cx = (x0 + g.tile * 0.5f) / f, cy = (y0 + g.tile * 0.5f) / f;
                    float fx = (cx - pg.tile * 0.5f) / pg.stride, fy = (cy - pg.tile * 0.5f) / pg.stride;
                    int pi = clampi((int)std::lround(fx), 0, pg.nx - 1), pj = clampi((int)std::lround(fy), 0, pg.ny - 1);
                    int ni = clampi(fx > pi ? pi + 1 : pi - 1, 0, pg.nx - 1), nj = clampi(fy > pj ? pj + 1 : pj - 1, 0, pg.ny - 1);
                    int ids[3][2] = {{pi, pj}, {ni, pj}, {pi, nj}};
                    for (int c = 0; c < 3; c++) {
                        size_t k = (size_t)ids[c][1] * pg.nx + ids[c][0];
                        cand[c][0] = pdx[k] * f;
                        cand[c][1] = pdy[k] * f;
                    }
                    nc = 3;
                }
                int bx = cand[0][0], by = cand[0][1];
                float best = 3.4e38f;
                for (int c = 0; c < nc; c++) {
                    float s = tile_cost(ref[l], alt[l], x0, y0, g.tile, cand[c][0], cand[c][1], l1, best);
                    if (s < best) { best = s; bx = cand[c][0]; by = cand[c][1]; }
                }
                int cx = bx, cy = by;
                for (int oy = -R; oy <= R; oy++)
                    for (int ox = -R; ox <= R; ox++) {
                        if (!ox && !oy) continue;
                        float s = tile_cost(ref[l], alt[l], x0, y0, g.tile, cx + ox, cy + oy, l1, best);
                        if (s < best) { best = s; bx = cx + ox; by = cy + oy; }
                    }
                dx[(size_t)j * g.nx + i] = (int16_t)bx;
                dy[(size_t)j * g.nx + i] = (int16_t)by;
            }
        });
        pdx.swap(dx);
        pdy.swap(dy);
        pg = g;
    }
    Offsets o;
    o.nx = pg.nx;
    o.ny = pg.ny;
    o.dx = std::move(pdx);
    o.dy = std::move(pdy);
    return o;
}
