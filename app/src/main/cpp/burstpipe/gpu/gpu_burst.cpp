#include "gpu_burst.h"

#include <set>

namespace {

// the frequency merge's per-coefficient DFT is ~10x slower than the CPU on Adreno 725
std::set<std::string> g_off = {"freq"};

cl_float4 f4(float a, float b, float c, float d) {
    cl_float4 v;
    v.s[0] = a; v.s[1] = b; v.s[2] = c; v.s[3] = d;
    return v;
}

// same windows and tables as align/merge/kr/merge_freq/finish build on the CPU
struct Tables {
    float win[16], sinw[16], tc[16], ts[16], w2 = 0;
    std::vector<float> oetf, eotf;
    Tables() : oetf(4097), eotf(4097) {
        for (int u = 0; u < 16; u++) win[u] = 0.5f - 0.5f * std::cos(2.f * (float)M_PI * (u + 0.5f) / (2 * kStride));
        for (int u = 0; u < 16; u++) sinw[u] = std::sin((float)M_PI * (u + 0.5f) / 16);
        for (int m = 0; m < 16; m++) { tc[m] = std::cos(2 * (float)M_PI * m / 16); ts[m] = std::sin(2 * (float)M_PI * m / 16); }
        for (int v = 0; v < 16; v++) for (int u = 0; u < 16; u++) w2 += sinw[u] * sinw[u] * sinw[v] * sinw[v];
        for (int i = 0; i <= 4096; i++) {
            float t = (float)i / 4096;
            float x = t * t;
            oetf[i] = x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
            eotf[i] = t <= 0.04045f ? t / 12.92f : std::pow((t + 0.055f) / 1.055f, 2.4f);
        }
    }
};
const Tables& tables() { static Tables t; return t; }

float lerp_snr(float snr, float a, float b) { return a + (b - a) * clampf((snr - 6.f) / 24.f, 0.f, 1.f); }

size_t gs(int v) { return (size_t)std::max(0, v); }
int groups64(int n) { return (n + 63) / 64; }
const LocalMem kRed{64 * 2 * sizeof(float)};

}  // namespace

void GpuBurst::set_disabled(const std::string& off) {
    g_off.clear();
    size_t p = 0;
    while (p <= off.size()) {
        size_t e = off.find(',', p);
        if (e == std::string::npos) e = off.size();
        if (e > p) g_off.insert(off.substr(p, e - p));
        p = e + 1;
    }
}

bool GpuBurst::enabled(const char* stage) { return !g_off.count(stage) && !g_off.count("all"); }

std::unique_ptr<GpuBurst> GpuBurst::open(int width, int height, int slots, std::string& why) {
    Gpu* g = Gpu::get();
    if (!g) { why = Gpu::describe(); return nullptr; }
    auto lock = g->try_acquire();
    if (!lock.owns_lock()) { why = "GPU busy with another burst"; return nullptr; }
    std::unique_ptr<GpuBurst> b(new GpuBurst(*g, std::move(lock)));
    try {
        b->n_ = slots;
        b->W_ = width;
        b->H_ = height;
        b->qw_ = b->W_ / 2;
        b->qh_ = b->H_ / 2;
        b->frames_ = g->alloc((size_t)b->W_ * b->H_ * 2 * slots);
        b->mean3_ = g->alloc((size_t)b->qw_ * b->qh_ * 4 * slots);
        const Tables& t = tables();
        b->win_ = g->alloc(sizeof t.win);
        g->write(b->win_, t.win, sizeof t.win);
        b->oetf_ = g->alloc(t.oetf.size() * 4);
        g->write(b->oetf_, t.oetf.data(), t.oetf.size() * 4);
        b->eotf_ = g->alloc(t.eotf.size() * 4);
        g->write(b->eotf_, t.eotf.data(), t.eotf.size() * 4);
    } catch (const GpuError& e) {
        why = e.what();
        return nullptr;
    }
    return b;
}

std::unique_ptr<GpuBurst> GpuBurst::create(const std::vector<Raw>& frames, std::string& why) {
    if (frames.empty()) { why = "no frames"; return nullptr; }
    std::unique_ptr<GpuBurst> b = open(frames[0].w, frames[0].h, (int)frames.size(), why);
    if (!b) return nullptr;
    try {
        const size_t fs = (size_t)b->W_ * b->H_;
        for (int k = 0; k < b->n_; k++) b->g_.write(b->frames_, frames[k].row(0), fs * 2, fs * 2 * k);
    } catch (const GpuError& e) {
        why = e.what();
        return nullptr;
    }
    return b;
}

std::unique_ptr<GpuBurst> GpuBurst::create_stream(int width, int height, std::string& why) { return open(width, height, 2, why); }

GImg GpuBurst::upload(const ImageF& im) {
    GImg g = g_.image(im.w, im.h, im.c);
    g_.write(g.b, im.d.data(), g.count() * 4);
    return g;
}

ImageF GpuBurst::download(const GImg& g) {
    ImageF im(g.w, g.h, g.c);
    g_.read(g.b, im.d.data(), g.count() * 4);
    return im;
}

std::string GpuBurst::profile() { return g_.take_profile(); }

// ---- align

namespace {
const int kFactor[3] = {2, 4, 4}, kTileL[4] = {16, 16, 16, 8}, kRadius[4] = {1, 4, 4, 4};
}

std::vector<GImg> GpuBurst::pyramid(int k) {
    const int qn = qw_ * qh_;
    std::vector<GImg> p(4);
    p[0] = g_.image(qw_, qh_);
    g_.run("gray_from_raw", gs(qw_), gs(qh_), frames_, k * W_ * H_, W_, p[0], qw_, qh_);
    GImg t = g_.image(qw_, qh_);
    g_.run("box3_h", gs(qw_), gs(qh_), p[0], t, qw_, qh_);
    g_.run("box3_v", gs(qw_), gs(qh_), t, mean3_, k * qn, qw_, qh_);
    for (int l = 0; l < 3; l++) {
        const GImg& in = p[l];
        p[l + 1] = g_.image(in.w / kFactor[l], in.h / kFactor[l]);
        g_.run("downsample", gs(p[l + 1].w), gs(p[l + 1].h), in, in.w, p[l + 1], p[l + 1].w, p[l + 1].h, kFactor[l]);
    }
    return p;
}

Offsets GpuBurst::align_pyramids(const std::vector<GImg>& refPyr, const std::vector<GImg>& altPyr) {
    Buf pdx, pdy;
    int pnx = 0, pny = 0, ptile = 0;
    for (int l = 3; l >= 0; l--) {
        const GImg& r = refPyr[l];
        int tile = kTileL[l], stride = tile / 2;
        int nx = (r.w + stride - 1) / stride, ny = (r.h + stride - 1) / stride;
        const int R = kRadius[l], side = 2 * R + 1, f = l < 3 ? kFactor[l] : 0, l1 = l == 0 ? 1 : 0;
        const size_t nt = (size_t)nx * ny;
        Buf dx = g_.alloc(nt * 2), dy = g_.alloc(nt * 2), cc = g_.alloc(nt * 3 * 4), sc = g_.alloc(nt * side * side * 4);
        const std::string sfx = std::to_string(tile);
        g_.run(("align_cand" + sfx).c_str(), gs(nx * 3), gs(ny), r, altPyr[l], r.w, r.h, nx, ny, l1, pdx, pdy, pnx, pny, ptile, f, cc);
        g_.run(("align_search" + sfx).c_str(), gs(nx * side), gs(ny), r, altPyr[l], r.w, r.h, nx, ny, R, l1, pdx, pdy, pnx, pny,
               ptile, f, cc, sc);
        g_.run("align_pick", gs(nx), gs(ny), tile, nx, ny, R, pdx, pdy, pnx, pny, ptile, f, cc, sc, dx, dy);
        pdx = dx; pdy = dy; pnx = nx; pny = ny; ptile = tile;
    }
    Offsets o;
    o.nx = pnx; o.ny = pny;
    o.dx.resize((size_t)pnx * pny);
    o.dy.resize(o.dx.size());
    g_.read(pdx, o.dx.data(), o.dx.size() * 2);
    g_.read(pdy, o.dy.data(), o.dy.size() * 2);
    return o;
}

void GpuBurst::align(int ref, int alt, std::vector<Offsets>& offs, ImageF& grayRef, ImageF& grayAlt) {
    std::vector<GImg> refPyr = pyramid(ref);
    grayRef_ = refPyr[0];
    offs.assign(n_, Offsets());
    for (int k = 0; k < n_; k++) {
        if (k == ref) continue;
        std::vector<GImg> altPyr = pyramid(k);
        if (k == alt) grayAlt = download(altPyr[0]);
        offs[k] = align_pyramids(refPyr, altPyr);
    }
    Offsets z;
    z.nx = (qw_ + kStride - 1) / kStride;
    z.ny = (qh_ + kStride - 1) / kStride;
    z.dx.assign((size_t)z.nx * z.ny, 0);
    z.dy = z.dx;
    offs[ref] = z;
    grayRef = download(grayRef_);
    aligned_ = true;
}

void GpuBurst::set_aligned(const std::vector<ImageF>& mean3, const ImageF& grayRef) {
    const size_t qn = (size_t)qw_ * qh_;
    for (int k = 0; k < n_; k++) g_.write(mean3_, mean3[k].d.data(), qn * 4, qn * 4 * k);
    grayRef_ = upload(grayRef);
    aligned_ = true;
}

void GpuBurst::upload_offsets(const std::vector<Offsets>& offs) {
    tnx_ = offs[0].nx;
    tny_ = offs[0].ny;
    nt_ = tnx_ * tny_;
    std::vector<int16_t> dx((size_t)nt_ * n_), dy(dx.size());
    for (int k = 0; k < n_; k++) {
        std::copy(offs[k].dx.begin(), offs[k].dx.end(), dx.begin() + (size_t)k * nt_);
        std::copy(offs[k].dy.begin(), offs[k].dy.end(), dy.begin() + (size_t)k * nt_);
    }
    odx_ = g_.alloc(dx.size() * 2);
    ody_ = g_.alloc(dy.size() * 2);
    g_.write(odx_, dx.data(), dx.size() * 2);
    g_.write(ody_, dy.data(), dy.size() * 2);
}

// ---- merges

double GpuBurst::merge_spatial(int ref, const std::vector<Offsets>& offs, const NoiseModel& nm, float black, const MergeOpts& o) {
    upload_offsets(offs);
    const int qn = qw_ * qh_, fs = W_ * H_;
    GImg out = g_.image(W_, H_), wsum = g_.image(qw_, qh_), rob = g_.image(qw_, qh_), tmp = g_.image(qw_, qh_);
    g_.run("merge_init", gs(qw_), gs(qh_), frames_, ref * fs, W_, qw_, qh_, black, out, wsum);
    const int slots = qh_ * groups64(qw_);
    Buf rows = g_.alloc((size_t)slots * n_ * 4);
    for (int k = 0; k < n_; k++) {
        if (k == ref) continue;
        g_.run("merge_rob", gs(qw_), gs(qh_), mean3_, ref * qn, k * qn, frames_, k * fs, W_, qw_, qh_, odx_, ody_, k * nt_,
               tnx_, tny_, win_, o.g(k), o.clipAt(k), black, nm.a, nm.b, o.robustLo, o.robustHi, rob);
        if (o.dilate > 0) {
            g_.run("min_filter_h", gs(qw_), gs(qh_), rob, 0, tmp, qw_, qh_, o.dilate);
            g_.run("min_filter_v", gs(qw_), gs(qh_), tmp, rob, 0, qw_, qh_, o.dilate);
        }
        g_.run_local("row_sums", gs(qw_), gs(qh_), 64, 1, rob, 0, qw_, qh_, 1.f, rows, k * slots, kRed);
        g_.run("merge_acc", gs(qw_), gs(qh_), rob, frames_, k * fs, W_, qw_, qh_, odx_, ody_, k * nt_, tnx_, tny_, win_, black,
               out, wsum);
    }
    g_.run("merge_norm", gs(qw_), gs(qh_), out, wsum, W_, qw_, qh_, black);
    std::vector<float> rs((size_t)slots * n_);
    g_.read(rows, rs.data(), rs.size() * 4);
    double wTotal = 0;
    for (int k = 0; k < n_; k++)
        if (k != ref) for (int y = 0; y < slots; y++) wTotal += rs[(size_t)k * slots + y];
    merged_ = out;
    kr_ = false;
    g_.finish();
    return n_ > 1 ? wTotal / ((double)qw_ * qh_ * (n_ - 1)) : 0;
}

void GpuBurst::stream_ref(const Raw& ref, float black, ImageF* gray) {
    const size_t fs = (size_t)W_ * H_;
    g_.write(frames_, ref.row(0), fs * 2, 0);
    refPyr_ = pyramid(0);
    if (gray) *gray = download(refPyr_[0]);
    sOut_ = g_.image(W_, H_);
    sWsum_ = g_.image(qw_, qh_);
    sRob_ = g_.image(qw_, qh_);
    sTmp_ = g_.image(qw_, qh_);
    sRows_ = g_.alloc((size_t)qh_ * groups64(qw_) * 4);
    g_.run("merge_init", gs(qw_), gs(qh_), frames_, 0, W_, qw_, qh_, black, sOut_, sWsum_);
    sWeight_ = 0;
    sAdded_ = 0;
}

Offsets GpuBurst::stream_align(const Raw& frame, ImageF* gray) {
    const size_t fs = (size_t)W_ * H_;
    g_.write(frames_, frame.row(0), fs * 2, fs * 2);
    std::vector<GImg> altPyr = pyramid(1);
    if (gray) *gray = download(altPyr[0]);
    return align_pyramids(refPyr_, altPyr);
}

void GpuBurst::stream_merge(const Offsets& of, const NoiseModel& nm, float black, const MergeOpts& o) {
    const int qn = qw_ * qh_, fs = W_ * H_;
    tnx_ = of.nx;
    tny_ = of.ny;
    nt_ = tnx_ * tny_;
    if (!odx_) {
        odx_ = g_.alloc((size_t)nt_ * 2);
        ody_ = g_.alloc((size_t)nt_ * 2);
    }
    g_.write(odx_, of.dx.data(), (size_t)nt_ * 2);
    g_.write(ody_, of.dy.data(), (size_t)nt_ * 2);
    g_.run("merge_rob", gs(qw_), gs(qh_), mean3_, 0, qn, frames_, fs, W_, qw_, qh_, odx_, ody_, 0, tnx_, tny_, win_, 1.f, 1e30f,
           black, nm.a, nm.b, o.robustLo, o.robustHi, sRob_);
    if (o.dilate > 0) {
        g_.run("min_filter_h", gs(qw_), gs(qh_), sRob_, 0, sTmp_, qw_, qh_, o.dilate);
        g_.run("min_filter_v", gs(qw_), gs(qh_), sTmp_, sRob_, 0, qw_, qh_, o.dilate);
    }
    const int slots = qh_ * groups64(qw_);
    g_.run_local("row_sums", gs(qw_), gs(qh_), 64, 1, sRob_, 0, qw_, qh_, 1.f, sRows_, 0, kRed);
    g_.run("merge_acc", gs(qw_), gs(qh_), sRob_, frames_, fs, W_, qw_, qh_, odx_, ody_, 0, tnx_, tny_, win_, black, sOut_, sWsum_);
    std::vector<float> rs(slots);
    g_.read(sRows_, rs.data(), rs.size() * 4);
    for (float v : rs) sWeight_ += v;
    sAdded_++;
}

double GpuBurst::stream_end(float black) {
    g_.run("merge_norm", gs(qw_), gs(qh_), sOut_, sWsum_, W_, qw_, qh_, black);
    merged_ = sOut_;
    kr_ = false;
    sOut_ = sWsum_ = sRob_ = sTmp_ = GImg();
    sRows_.reset();
    refPyr_.clear();
    drop_frames();
    g_.finish();
    return sAdded_ > 0 ? sWeight_ / ((double)qw_ * qh_ * sAdded_) : 0;
}

double GpuBurst::merge_freq(int ref, const std::vector<Offsets>& offs, const Meta& m, const NoiseModel& nm, float c, float spatial) {
    upload_offsets(offs);
    const Tables& t = tables();
    Buf sinw = g_.alloc(64), tc = g_.alloc(64), ts = g_.alloc(64);
    g_.write(sinw, t.sinw, 64);
    g_.write(tc, t.tc, 64);
    g_.write(ts, t.ts, 64);
    float bl[4];
    for (int l = 0; l < 4; l++) bl[l] = m.black[cfa_channel(m.cfa, l & 1, l >> 1)];
    GImg out = g_.image(W_, H_);
    g_.zero(out.b);
    const int ntx = (qw_ + 7) / 8 + 1, nty = (qh_ + 7) / 8 + 1;
    Buf tileW = g_.alloc((size_t)ntx * nty * 4);
    const LocalMem lm{256 * sizeof(cl_float4)};
    for (int py = 0; py < 2; py++)
        for (int px = 0; px < 2; px++) {
            size_t gx = (size_t)(ntx - px + 1) / 2 * 16, gy = (size_t)(nty - py + 1) / 2 * 16;
            g_.run_local("merge_freq_tiles", gx, gy, 16, 16, frames_, W_, qw_, qh_, n_, ref, W_ * H_, odx_, ody_, tnx_, tny_, nt_,
                         sinw, tc, ts, f4(bl[0], bl[1], bl[2], bl[3]), t.w2, c, spatial, nm.a, nm.b, px, py, ntx, nty, out, tileW,
                         lm, lm, lm, lm, LocalMem{32 * sizeof(float)});
        }
    std::vector<float> tw((size_t)ntx * nty);
    g_.read(tileW, tw.data(), tw.size() * 4);
    double s = 0;
    if (n_ > 1) for (float v : tw) s += v / (n_ - 1);
    merged_ = out;
    kr_ = false;
    return s / ((double)ntx * nty);
}

void GpuBurst::merge_kr(const Raw& R, int ref, const std::vector<Offsets>& offs, const Meta& m, const NoiseModel& nm,
                        const MergeOpts& o, KrParams p, KrStats* st) {
    upload_offsets(offs);
    const int W = W_, H = H_, qw = qw_, qh = qh_, n = n_, qn = qw * qh, fs = W * H;
    const float black = (m.black[0] + m.black[1] + m.black[2] + m.black[3]) / 4;

    double s2 = 0, v2 = 0;
    for (int y = 0; y < H; y += 7)
        for (int x = 0; x < W; x += 7) {
            double s = R.at(x, y) - black;
            if (s <= 0 || R.at(x, y) >= m.white) continue;
            s2 += s * s; v2 += nm.var(s);
        }
    float snr = v2 > 0 ? (float)(10.0 * std::log10(s2 / v2)) : 30.f;
    if (p.kDetail < 0) p.kDetail = lerp_snr(snr, 0.33f, 0.25f);
    if (p.kDenoise < 0) p.kDenoise = lerp_snr(snr, 5.0f, 3.0f);
    if (p.dTh < 0) p.dTh = lerp_snr(snr, 0.81f, 0.71f);
    if (p.dTr < 0) p.dTr = lerp_snr(snr, 1.24f, 1.0f);

    GImg gx = g_.image(qw, qh), gy = g_.image(qw, qh);
    g_.run("lk_grad", gs(qw), gs(qh), mean3_, ref * qn, qw, qh, gx, gy);
    Buf fx = g_.alloc((size_t)nt_ * n * 4), fy = g_.alloc((size_t)nt_ * n * 4);
    g_.zero(fx);
    g_.zero(fy);
    for (int k = 0; k < n; k++) {
        if (k == ref) continue;
        g_.run_local("refine_flow", gs(16 * tnx_), gs(tny_), 64, 1, mean3_, ref * qn, k * qn, gx, gy, qw, qh, odx_, ody_, k * nt_,
                     tnx_, tny_, 3, fx, fy, LocalMem{64 * 3 * sizeof(float)});
    }
    gx = GImg();
    gy = GImg();

    GImg vst = g_.image(qw, qh), icov = g_.image(qw, qh, 3);
    g_.run("kr_vst", gs(qw), gs(qh), grayRef_, qw, qh, black, nm.a, nm.b, vst);
    g_.run("kr_icov", gs(qw), gs(qh), vst, qw, qh, p.kDetail, p.kDenoise, p.dTh, p.dTr, p.kStretch, p.kShrink, icov);
    vst = GImg();

    Buf rob = g_.alloc((size_t)qn * n * 4), rows = g_.alloc((size_t)qh * groups64(qw) * n * 4);
    GImg tmp = g_.image(qw, qh);
    for (int k = 0; k < n; k++) {
        if (k == ref) continue;
        g_.run("kr_rob", gs(qw), gs(qh), fx, fy, k * nt_, tnx_, tny_, win_, mean3_, ref * qn, k * qn, frames_, k * fs, W, qw, qh,
               o.g(k), o.clipAt(k), black, nm.a, nm.b, o.robustLo, o.robustHi, rob, k * qn);
        if (o.dilate > 0) {
            g_.run("min_filter_h", gs(qw), gs(qh), rob, k * qn, tmp, qw, qh, o.dilate);
            g_.run("min_filter_v", gs(qw), gs(qh), tmp, rob, k * qn, qw, qh, o.dilate);
        }
        g_.run_local("row_sums", gs(qw), gs(qh), 64, 1, rob, k * qn, qw, qh, 1.f, rows, k * qh * groups64(qw), kRed);
    }
    tmp = GImg();

    const float sc = p.scale;
    const int OW = (int)std::lround(W * sc), OH = (int)std::lround(H * sc);
    GImg out = g_.image(OW, OH, 3);
    const bool lca = p.lcaR[0] != 0 || p.lcaR[1] != 0 || p.lcaB[0] != 0 || p.lcaB[1] != 0;
    const float fcx = (m.full_w() - 1) * 0.5f, fcy = (m.full_h() - 1) * 0.5f, invR2 = 1.f / (fcx * fcx + fcy * fcy);
    const float lcx = fcx - m.cropX, lcy = fcy - m.cropY;
    g_.run("kr_gather", gs(OW), gs(OH), frames_, fs, n, ref, W, H, fx, fy, tnx_, tny_, nt_, win_, rob, icov, qw, qh, sc, OW, OH,
           m.cfa, f4(m.black[0], m.black[1], m.black[2], m.black[3]), lca ? 1 : 0, f4(p.lcaR[0], p.lcaR[1], p.lcaB[0], p.lcaB[1]),
           lcx, lcy, invR2, out);

    const int slots = qh * groups64(qw);
    std::vector<float> rs((size_t)slots * n);
    g_.read(rows, rs.data(), rs.size() * 4);
    if (st) {
        st->snrDb = snr;
        st->kDetail = p.kDetail; st->kDenoise = p.kDenoise; st->dTh = p.dTh; st->dTr = p.dTr;
        double s = 0;
        for (int k = 0; k < n; k++) {
            if (k == ref) continue;
            double fw = 0;
            for (int y = 0; y < slots; y++) fw += rs[(size_t)k * slots + y];
            s += fw / ((double)qw * qh);
        }
        st->meanWeight = n > 1 ? s / (n - 1) : 0;
        std::vector<float> hx((size_t)nt_ * n), hy(hx.size());
        g_.read(fx, hx.data(), hx.size() * 4);
        g_.read(fy, hy.data(), hy.size() * 4);
        double cov = 0;
        for (int t = 0; t < nt_; t++) {
            uint16_t bins = 0;
            for (int k = 0; k < n; k++) {
                float px = 2 * hx[(size_t)k * nt_ + t], py = 2 * hy[(size_t)k * nt_ + t];
                px -= 2 * std::floor(px / 2); py -= 2 * std::floor(py / 2);
                int bx = std::min(3, (int)(px * 2)), by = std::min(3, (int)(py * 2));
                bins |= (uint16_t)(1u << (by * 4 + bx));
            }
            int hit = 0;
            for (int bit = 0; bit < 16; bit++) hit += (bins >> bit) & 1;
            cov += (double)hit / std::min(n, 16);
        }
        st->phaseCoverage = nt_ ? cov / nt_ : 0;
    }
    merged_ = out;
    kr_ = true;
    g_.finish();
}

void GpuBurst::drop_frames() {
    frames_.reset();
    mean3_.reset();
    grayRef_ = GImg();
    odx_.reset();
    ody_.reset();
}

ImageF GpuBurst::download_merged() { return download(merged_); }

void GpuBurst::set_merged(const ImageF& im, bool kernelRegression) {
    merged_ = upload(im);
    kr_ = kernelRegression;
}

// ---- finishing

GImg GpuBurst::luma(const GImg& rgb, float floorv) {
    GImg Y = g_.image(rgb.w, rgb.h);
    g_.run("luma_of", gs(rgb.w * rgb.h), 1, rgb, rgb.w * rgb.h, floorv, Y);
    return Y;
}

GImg GpuBurst::gaussian(const GImg& in, float sigma) {
    int r = std::max(1, (int)std::ceil(2.5f * sigma));
    std::vector<float> k(2 * r + 1);
    float s = 0;
    for (int i = -r; i <= r; i++) s += k[i + r] = std::exp(-0.5f * i * i / (sigma * sigma));
    for (float& v : k) v /= s;
    Buf kb = g_.alloc(k.size() * 4);
    g_.write(kb, k.data(), k.size() * 4);
    GImg t = g_.image(in.w, in.h), o = g_.image(in.w, in.h);
    g_.run("blur_h", gs(in.w), gs(in.h), in, t, in.w, in.h, kb, r);
    g_.run("blur_v", gs(in.w), gs(in.h), t, o, in.w, in.h, kb, r);
    return o;
}

GImg GpuBurst::box(const GImg& in, int r) {
    GImg t = g_.image(in.w, in.h), o = g_.image(in.w, in.h);
    g_.run("box_h", gs(in.w), gs(in.h), in, t, in.w, in.h, r);
    g_.run("box_v", gs(in.w), gs(in.h), t, o, in.w, in.h, r);
    return o;
}

GImg GpuBurst::pyr_down(const GImg& in) {
    int w = (in.w + 1) / 2, h = (in.h + 1) / 2;
    GImg t = g_.image(w, in.h), o = g_.image(w, h);
    g_.run("pyr_down_h", gs(w), gs(in.h), in, in.w, in.h, t, w);
    g_.run("pyr_down_v", gs(w), gs(h), t, w, in.h, o, h);
    return o;
}

void GpuBurst::pyr_up_sub(const GImg& coarse, GImg& fine, float s) {
    GImg t = g_.image(fine.w, coarse.h);
    g_.run("pyr_up_h", gs(fine.w), gs(coarse.h), coarse, coarse.w, coarse.h, t, fine.w);
    g_.run("pyr_up_v_sub", gs(fine.w), gs(fine.h), t, fine.w, coarse.h, fine, fine.h, s);
}

GImg GpuBurst::demosaic_rcd(const GImg& cfa, int cfaType) {
    const int W = cfa.w, H = cfa.h;
    GImg V = g_.image(W, H), Hh = g_.image(W, H), P = g_.image(W, H), Q = g_.image(W, H);
    g_.run("rcd_hpf", gs(W), gs(H), cfa, W, H, V, Hh, P, Q);
    GImg VH = g_.image(W, H), LPF = g_.image(W, H), PQ = g_.image(W, H);
    g_.run("rcd_dir", gs(W), gs(H), cfa, W, H, cfaType, V, Hh, P, Q, VH, LPF, PQ);
    V = Hh = P = Q = GImg();
    GImg G = g_.image(W, H), R = g_.image(W, H), B = g_.image(W, H);
    g_.run("rcd_green", gs(W), gs(H), cfa, W, H, cfaType, VH, LPF, G);
    g_.run("rcd_rb_at_rb", gs(W), gs(H), cfa, W, H, cfaType, G, PQ, R, B);
    g_.run("rcd_rb_at_g", gs(W), gs(H), W, H, cfaType, G, VH, R, B);
    GImg rgb = g_.image(W, H, 3);
    g_.run("rcd_compose", gs(W), gs(H), cfa, W, H, cfaType, R, G, B, rgb);
    return rgb;
}

void GpuBurst::chroma_denoise(float strength) {
    if (strength <= 0) return;
    const int W = rgb_.w, H = rgb_.h, w = (W + 3) / 4, h = (H + 3) / 4;
    GImg cr = g_.image(w, h), cb = g_.image(w, h);
    g_.run("chroma_down", gs(w), gs(h), rgb_, W, H, cr, cb, w, h);
    cr = gaussian(cr, 1.0f * strength);
    cb = gaussian(cb, 1.0f * strength);
    g_.run("chroma_apply", gs(W), gs(H), rgb_, W, H, cr, cb, w, h);
}

void GpuBurst::upscale(int W, int H) {
    if (W == rgb_.w && H == rgb_.h) return;
    constexpr int kA = 3;
    constexpr float kPi = 3.14159265f;
    auto taps = [&](int n, int N, std::vector<int>& x0, std::vector<float>& wt) {
        x0.resize(N);
        wt.resize((size_t)N * 2 * kA);
        const float s = (float)n / N;
        for (int X = 0; X < N; X++) {
            float c = (X + 0.5f) * s - 0.5f;
            int b = (int)std::floor(c) - kA + 1;
            float sum = 0;
            for (int i = 0; i < 2 * kA; i++) {
                float d = std::fabs(c - (b + i));
                float v = d < 1e-6f ? 1.f : (d >= kA ? 0.f : kA * std::sin(kPi * d) * std::sin(kPi * d / kA) / (kPi * kPi * d * d));
                wt[(size_t)X * 2 * kA + i] = v;
                sum += v;
            }
            for (int i = 0; i < 2 * kA; i++) wt[(size_t)X * 2 * kA + i] /= sum;
            x0[X] = b;
        }
    };
    const int w = rgb_.w, h = rgb_.h;
    std::vector<int> tx0, ty0;
    std::vector<float> txw, tyw;
    taps(w, W, tx0, txw);
    taps(h, H, ty0, tyw);
    Buf bx0 = g_.alloc(tx0.size() * 4), bxw = g_.alloc(txw.size() * 4), by0 = g_.alloc(ty0.size() * 4), byw = g_.alloc(tyw.size() * 4);
    g_.write(bx0, tx0.data(), tx0.size() * 4);
    g_.write(bxw, txw.data(), txw.size() * 4);
    g_.write(by0, ty0.data(), ty0.size() * 4);
    g_.write(byw, tyw.data(), tyw.size() * 4);
    GImg mid = g_.image(W, h, 3);
    g_.run("lanczos_h", gs(W), gs(h), rgb_, w, h, mid, W, bx0, bxw);
    rgb_ = GImg();
    GImg out = g_.image(W, H, 3);
    g_.run("lanczos_v", gs(3 * W), gs(H), mid, W, h, out, H, by0, byw);
    rgb_ = out;
}

GImg GpuBurst::fuse_exposures(const GImg& Y, float gS, float gL) {
    const int W = Y.w, H = Y.h, n = W * H;
    GImg eS = g_.image(W, H), eL = g_.image(W, H), wS = g_.image(W, H);
    g_.run("tm_exposures", gs(n), 1, Y, n, oetf_, gS, gL, eS, eL, wS);
    int levels = 1;
    for (int m = std::min(W, H); m > 24; m /= 2) levels++;
    std::vector<GImg> gsP, glP, gwP;
    gsP.push_back(eS); glP.push_back(eL); gwP.push_back(wS);
    eS = eL = wS = GImg();
    for (int l = 1; l < levels; l++) { gsP.push_back(pyr_down(gsP.back())); glP.push_back(pyr_down(glP.back())); gwP.push_back(pyr_down(gwP.back())); }
    for (int l = 0; l < levels - 1; l++) {
        pyr_up_sub(gsP[l + 1], gsP[l], 1.f);
        pyr_up_sub(glP[l + 1], glP[l], 1.f);
    }
    for (int l = 0; l < levels; l++) g_.run("tm_blend", gs(gsP[l].w * gsP[l].h), 1, gsP[l], glP[l], gwP[l], gsP[l].w * gsP[l].h);
    glP.clear();
    gwP.clear();
    for (int l = levels - 2; l >= 0; l--) {
        pyr_up_sub(gsP[l + 1], gsP[l], -1.f);
        gsP[l + 1] = GImg();
    }
    return gsP[0];
}

ToneInfo GpuBurst::local_tonemap(const LookParams& p) {
    const int W = rgb_.w, H = rgb_.h, n = W * H;
    const int s8w = (W + 7) / 8, s8h = (H + 7) / 8;
    GImg sub = g_.image(s8w, s8h);
    g_.run("tm_key_samples", gs(s8w), gs(s8h), rgb_, W, H, sub, s8w, s8h);
    std::vector<float> samp((size_t)s8w * s8h);
    g_.read(sub.b, samp.data(), samp.size() * 4);
    double lsum = 0;
    for (float v : samp) lsum += std::log(v + 1e-4);
    float key = (float)std::exp(lsum / samp.size());
    std::nth_element(samp.begin(), samp.begin() + (size_t)(samp.size() * 0.995), samp.end());
    float p995 = samp[(size_t)(samp.size() * 0.995)];
    float gL = clampf(p.targetKey / key, 1.f, 64.f);
    float maxRatio = p.compression > 0 ? p.compression : 8.f;
    float gS = clampf(1.f / p995, gL / maxRatio, gL);

    if (p.toneFull) {
        GImg Y = luma(rgb_, 1e-6f);
        GImg acc = fuse_exposures(Y, gS, gL);
        g_.run("tm_apply", gs(n), 1, rgb_, n, acc, Y, eotf_);
        return {gS, gL, key};
    }
    const int sw = (W + kToneSub - 1) / kToneSub, sh = (H + kToneSub - 1) / kToneSub, sn = sw * sh;
    GImg Ys = g_.image(sw, sh);
    g_.run("tm_down", gs(sw), gs(sh), rgb_, W, H, Ys, sw, sh);
    GImg fused = fuse_exposures(Ys, gS, gL);
    GImg G = g_.image(sw, sh), Lg = g_.image(sw, sh), GG = g_.image(sw, sh), GL = g_.image(sw, sh);
    g_.run("tm_guide", gs(sn), 1, Ys, fused, eotf_, G, Lg, GG, GL, sn);
    const int r = kToneGuideRadius;
    GImg mG = box(G, r), mL = box(Lg, r), mGG = box(GG, r), mGL = box(GL, r);
    GImg A = g_.image(sw, sh), B = g_.image(sw, sh);
    g_.run("tm_coef", gs(sn), 1, mG, mL, mGG, mGL, A, B, sn, kToneGuideEps);
    GImg mA = box(A, r), mB = box(B, r);
    g_.run("tm_apply_guided", gs(W), gs(H), rgb_, W, H, mA, mB, sw, sh);
    return {gS, gL, key};
}

void GpuBurst::local_contrast(float amount, float radiusFrac) {
    if (amount <= 0) return;
    const int W = rgb_.w, H = rgb_.h;
    GImg L = luma(rgb_, -3.4e38f);
    constexpr int kSub = 4;
    const int sw = (W + kSub - 1) / kSub, sh = (H + kSub - 1) / kSub;
    int r = std::max(1, (int)std::lround(radiusFrac * std::max(W, H) / kSub));
    const float eps = 0.01f * 0.01f * 25;
    GImg Ls = g_.image(sw, sh), LL = g_.image(sw, sh);
    g_.run("lc_down", gs(sw), gs(sh), L, W, H, Ls, LL, sw, sh);
    GImg mI = box(Ls, r), mII = box(LL, r);
    GImg A = g_.image(sw, sh), B = g_.image(sw, sh);
    g_.run("lc_coef", gs(sw * sh), 1, mI, mII, A, B, sw * sh, eps);
    GImg mA = box(A, r), mB = box(B, r);
    g_.run("lc_apply", gs(W), gs(H), rgb_, W, H, L, mA, mB, sw, sh, amount);
}

void GpuBurst::sharpen(float amount, float radius, float core) {
    if (amount <= 0) return;
    const int n = rgb_.w * rgb_.h;
    GImg L = luma(rgb_, -3.4e38f);
    GImg B = gaussian(L, radius);
    g_.run("sharpen_apply", gs(n), 1, rgb_, n, L, B, amount, core);
}

void GpuBurst::finish(const Meta& m, const LookParams& look, Demosaic dm, float zoom, bool keepHdr, FinishOut& out) {
    LookParams lp = look;
    Timer t;
    auto lap = [&](double& field) { g_.finish(); field = t.ms(); t = Timer(); };
    const int W = merged_.w, H = merged_.h;
    const bool lsc = m.lscRows > 1 && m.lscCols > 1;
    Buf lscBuf = g_.alloc(lsc ? m.lsc.size() * 4 : 16);
    if (lsc) g_.write(lscBuf, m.lsc.data(), m.lsc.size() * 4);
    const float sw = m.full_w() - 1.f, sh = m.full_h() - 1.f;
    auto sigma_from = [&](const Buf& rows, int count) {
        std::vector<float> v((size_t)count * 2);
        g_.read(rows, v.data(), v.size() * 4);
        double s = 0, n = 0;
        for (int i = 0; i < count; i++) { s += v[2 * i]; n += v[2 * i + 1]; }
        return n ? std::sqrt(M_PI / 2) / (6.0 * n) * s : 0;
    };
    if (kr_) {
        rgb_ = g_.image(W, H, 3);
        float wbG = 0.5f * (m.wb[1] + m.wb[2]);
        cl_float4 inv = f4(1.f / (m.white - m.black[0]), 2.f / (2 * m.white - m.black[1] - m.black[2]), 1.f / (m.white - m.black[3]), 0);
        g_.run("normalize_rgb", gs(W), gs(H), merged_, rgb_, W, H, 1.f, inv, f4(m.wb[0], wbG, m.wb[3], 0), lscBuf, lsc ? m.lscRows : 0,
               lsc ? m.lscCols : 0, m.cropX, m.cropY, sw, sh, lp.lscLumaStrength);
        lap(out.ft.lscwb);
        int h = H / 2, slots = h * groups64(W / 2);
        Buf rows = g_.alloc((size_t)std::max(1, slots) * 8);
        g_.run_local("sigma_rows_rgb", gs(W / 2), gs(h), 64, 1, rgb_, W, H, 2, rows, kRed);
        out.sigma = sigma_from(rows, slots);
    } else {
        GImg cfa = g_.image(W, H);
        g_.run("normalize_mosaic", gs(W), gs(H), merged_, cfa, W, H, m.cfa, f4(m.black[0], m.black[1], m.black[2], m.black[3]), m.white,
               f4(m.wb[0], m.wb[1], m.wb[2], m.wb[3]), lscBuf, lsc ? m.lscRows : 0, lsc ? m.lscCols : 0, m.cropX, m.cropY, sw, sh,
               lp.lscLumaStrength);
        lap(out.ft.lscwb);
        int ox = 0, oy = 0;
        for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) if (cfa_channel(m.cfa, x, y) == 1) { ox = x; oy = y; }
        int h = H / 2, slots = h * groups64(W / 2);
        Buf rows = g_.alloc((size_t)std::max(1, slots) * 8);
        g_.run_local("sigma_rows_cfa", gs(W / 2), gs(h), 64, 1, cfa, W, H, ox, oy, rows, kRed);
        out.sigma = sigma_from(rows, slots);
        if (dm == Demosaic::MHC) {
            rgb_ = g_.image(W, H, 3);
            g_.run("demosaic_mhc", gs(W), gs(H), cfa, W, H, m.cfa, rgb_);
        } else if (dm == Demosaic::RCD) {
            rgb_ = demosaic_rcd(cfa, m.cfa);
        } else {
            rgb_ = upload(demosaic(download(cfa), m.cfa, dm));
        }
        lap(out.ft.demosaic);
    }

    const int n = rgb_.w * rgb_.h;
    float r = 1, g = 1, b = 1;
    if (lp.wbCool != 0) { r = 1 - lp.wbCool; b = 1 + lp.wbCool; g = (1 - 0.2126f * r - 0.0722f * b) / 0.7152f; }
    const float* M = m.ccm;
    g_.run("ccm_wb", gs(n), 1, rgb_, n, f4(M[0], M[1], M[2], 0), f4(M[3], M[4], M[5], 0), f4(M[6], M[7], M[8], 0), f4(r, g, b, 0));
    lap(out.ft.color);

    const int gr = (rgb_.h + 7) / 8, gc = (rgb_.w + 7) / 8;
    Buf lr = g_.alloc((size_t)gr * groups64(gc) * 4);
    g_.run_local("log_luma_rows", gs(gc), gs(gr), 64, 1, rgb_, rgb_.w, rgb_.h, lr, kRed);
    std::vector<float> lv((size_t)gr * groups64(gc));
    g_.read(lr, lv.data(), lv.size() * 4);
    double ls = 0;
    for (float v : lv) ls += v;
    long cnt = (long)gr * ((rgb_.w + 7) / 8);
    const float logKey = (float)std::exp(ls / std::max(1L, cnt));
    lp.targetKey = noise_limited_key(lp.targetKey, (float)out.sigma * clampf(lp.targetKey / logKey, 1.f, 64.f));
    out.targetKey = lp.targetKey;
    float dgain = clampf(lp.targetKey / logKey, 1.f, 64.f);
    out.visibleSigma = out.sigma * dgain;
    float chroma = lp.chromaDenoise;
    if (chroma < 0) chroma = clampf((float)out.visibleSigma / 0.01f, 1.f, 6.f);
    chroma *= lp.chromaDenoiseScale;
    out.chroma = chroma;
    chroma_denoise(chroma);
    lap(out.ft.chroma);
    if (zoom > 1.f) {
        upscale(m.full_w(), m.full_h());
        lap(out.tUpscale);
    }
    if (keepHdr) {
        HdrLuma o;
        const int bw = (rgb_.w + o.block - 1) / o.block, bh = (rgb_.h + o.block - 1) / o.block;
        GImg y = g_.image(bw, bh);
        g_.run("hdr_luma_blocks", gs(bw), gs(bh), rgb_, rgb_.w, rgb_.h, o.block, y, bw, bh);
        o.y = download(y);
        out.hdr = std::move(o);
    }
    t = Timer();
    out.tone = local_tonemap(lp);
    lap(out.ft.tonemap);
    out.localContrast = noise_scaled_local_contrast(lp.localContrast, (float)out.visibleSigma);
    g_.run("pixel_look", gs(rgb_.w * rgb_.h), 1, rgb_, rgb_.w * rgb_.h, oetf_, lp.contrast, lp.saturation);
    local_contrast((float)out.localContrast, lp.localRadius);
    if (!(lp.skySat == 1 && lp.skinSat == 1 && lp.skyHue == 0 && lp.skinHue == 0)) {
        auto win = [](float centre, float half) {
            float rr = centre * (float)M_PI / 180.f;
            return f4(std::cos(rr), std::sin(rr), std::cos(half * (float)M_PI / 180.f), 0);
        };
        g_.run("hue_tune", gs(rgb_.w * rgb_.h), 1, rgb_, rgb_.w * rgb_.h, win(-30.f, 40.f), win(122.f, 30.f),
               f4(lp.skySat, lp.skinSat, lp.skyHue * (float)M_PI / 180.f, lp.skinHue * (float)M_PI / 180.f));
    }
    lap(out.ft.look);
    out.sharpenCore = sharpen_core((float)out.visibleSigma);
    sharpen(lp.sharpen, lp.sharpenRadius, (float)out.sharpenCore);
    lap(out.ft.sharpen);
    merged_ = GImg();
}

void GpuBurst::render(uint8_t* rgba, size_t stride) {
    Buf o = g_.alloc((size_t)rgb_.w * rgb_.h * 4);
    g_.run("to_rgba", gs(rgb_.w), gs(rgb_.h), rgb_, rgb_.w, rgb_.h, o);
    g_.read_rows(o, rgba, (size_t)rgb_.w * 4, rgb_.h, stride);
}

ImageF GpuBurst::download_rgb() { return download(rgb_); }
