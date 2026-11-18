#include "gpu_burst.h"

#include <set>

namespace {

std::set<std::string> g_off;

cl_float4 f4(float a, float b, float c, float d) {
    cl_float4 v;
    v.s[0] = a; v.s[1] = b; v.s[2] = c; v.s[3] = d;
    return v;
}

struct Tables {
    float win[16];
    Tables() {
        for (int u = 0; u < 16; u++) win[u] = 0.5f - 0.5f * std::cos(2.f * (float)M_PI * (u + 0.5f) / (2 * kStride));
    }
};
const Tables& tables() { static Tables t; return t; }

size_t gs(int v) { return (size_t)std::max(0, v); }
int groups64(int n) { return (n + 63) / 64; }
const LocalMem kRed{64 * 2 * sizeof(float)};

// Persistent host RGB8 buffer reused across shots under Gpu's exclusive lock (avoids 48 MB mmap page faults per shot).
std::vector<uint8_t> g_hostRgb8;

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

std::unique_ptr<GpuBurst> GpuBurst::create_stream(int width, int height, std::string& why) { return open(width, height, 3, why); }

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
    p[1] = g_.image(qw_ / kFactor[0], qh_ / kFactor[0]);
    g_.run("gray_down2", gs((qw_ + 1) / 2), gs((qh_ + 1) / 2), frames_, k * W_ * H_, W_, p[0], qw_, qh_, p[1], p[1].w, p[1].h);
    g_.run_local("box3_3x3", gs(qw_), gs(qh_), 16, 8, p[0], mean3_, k * qn, qw_, qh_);
    for (int l = 1; l < 3; l++) {
        const GImg& in = p[l];
        p[l + 1] = g_.image(in.w / kFactor[l], in.h / kFactor[l]);
        g_.run("downsample", gs(p[l + 1].w), gs(p[l + 1].h), in, in.w, p[l + 1], p[l + 1].w, p[l + 1].h, kFactor[l]);
    }
    return p;
}

void GpuBurst::align_pyramids_to(const std::vector<GImg>& refPyr, const std::vector<GImg>& altPyr, Buf& outDx, Buf& outDy, int outOff) {
    Buf pdx, pdy;
    int pnx = 0, pny = 0, ptile = 0;
    for (int l = 3; l >= 0; l--) {
        const GImg& r = refPyr[l];
        int tile = kTileL[l], stride = tile / 2;
        int nx = (r.w + stride - 1) / stride, ny = (r.h + stride - 1) / stride;
        const int R = kRadius[l], side = 2 * R + 1, f = l < 3 ? kFactor[l] : 0, l1 = l == 0 ? 1 : 0;
        const size_t nt = (size_t)nx * ny;
        Buf cc = g_.alloc(nt * 3 * 4), sc = g_.alloc(nt * side * side * 4);
        const char* candK = tile == 16 ? "align_cand16" : "align_cand8";
        const char* searchK = tile == 16 ? "align_search16" : "align_search8";
        g_.run(candK, gs(nx * 3), gs(ny), r, altPyr[l], r.w, r.h, nx, ny, l1, pdx, pdy, pnx, pny, ptile, f, cc);
        g_.run(searchK, gs(nx * side), gs(ny), r, altPyr[l], r.w, r.h, nx, ny, R, l1, pdx, pdy, pnx, pny, ptile, f, cc, sc);
        if (l == 0) {
            g_.run("align_pick", gs(nx), gs(ny), tile, nx, ny, R, pdx, pdy, pnx, pny, ptile, f, cc, sc, outDx, outDy, outOff);
        } else {
            Buf dx = g_.alloc(nt * 2), dy = g_.alloc(nt * 2);
            g_.run("align_pick", gs(nx), gs(ny), tile, nx, ny, R, pdx, pdy, pnx, pny, ptile, f, cc, sc, dx, dy, 0);
            pdx = dx; pdy = dy; pnx = nx; pny = ny; ptile = tile;
        }
    }
}

void GpuBurst::align(int ref, int alt, std::vector<Offsets>& offs, ImageF& grayRef, ImageF& grayAlt) {
    std::vector<GImg> refPyr = pyramid(ref);
    grayRef_ = refPyr[0];
    tnx_ = (qw_ + kStride - 1) / kStride;
    tny_ = (qh_ + kStride - 1) / kStride;
    nt_ = tnx_ * tny_;
    size_t offBytes = (((size_t)nt_ * n_ * 2) + 3) & ~size_t(3);
    odx_ = g_.alloc(offBytes);
    ody_ = g_.alloc(offBytes);
    g_.zero(odx_);
    g_.zero(ody_);

    GImg altGrayGpu;
    for (int k = 0; k < n_; k++) {
        if (k == ref) continue;
        std::vector<GImg> altPyr = pyramid(k);
        if (k == alt) altGrayGpu = altPyr[0];
        align_pyramids_to(refPyr, altPyr, odx_, ody_, k * nt_);
    }

    std::vector<int16_t> allDx((size_t)nt_ * n_), allDy((size_t)nt_ * n_);
    g_.read_async(odx_, allDx.data(), (size_t)nt_ * n_ * 2);
    g_.read_async(ody_, allDy.data(), (size_t)nt_ * n_ * 2);
    if (alt >= 0 && altGrayGpu.w > 0) {
        grayAlt = ImageF(altGrayGpu.w, altGrayGpu.h);
        g_.read_async(altGrayGpu.b, grayAlt.d.data(), altGrayGpu.count() * 4);
    }
    grayRef = ImageF(grayRef_.w, grayRef_.h);
    g_.read(grayRef_.b, grayRef.d.data(), grayRef_.count() * 4);

    offs.assign(n_, Offsets());
    for (int k = 0; k < n_; k++) {
        offs[k].nx = tnx_;
        offs[k].ny = tny_;
        offs[k].dx.assign(allDx.begin() + (size_t)k * nt_, allDx.begin() + (size_t)(k + 1) * nt_);
        offs[k].dy.assign(allDy.begin() + (size_t)k * nt_, allDy.begin() + (size_t)(k + 1) * nt_);
    }
    aligned_ = true;
    offsOnGpu_ = true;
}

void GpuBurst::set_aligned(const std::vector<ImageF>& mean3, const ImageF& grayRef) {
    const size_t qn = (size_t)qw_ * qh_;
    for (int k = 0; k < n_; k++) g_.write(mean3_, mean3[k].d.data(), qn * 4, qn * 4 * k);
    grayRef_ = upload(grayRef);
    aligned_ = true;
    offsOnGpu_ = false;
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
    offsOnGpu_ = true;
}

// ---- spatial merge (batch & streaming)

double GpuBurst::merge_spatial(int ref, const std::vector<Offsets>& offs, const NoiseModel& nm, float black, const MergeOpts& o) {
    if (!offsOnGpu_) upload_offsets(offs);
    const int qn = qw_ * qh_, fs = W_ * H_;
    GImg out = g_.image(W_, H_), wsum = g_.image(qw_, qh_), rob = g_.half_image(qw_, qh_), tmp;
    if (o.dilate > 1) tmp = g_.half_image(qw_, qh_);
    g_.run("merge_init", gs(qw_), gs(qh_), frames_, ref * fs, W_, qw_, qh_, black, out, wsum);

    bool unitGain = true;
    for (int k = 0; k < n_; k++) if (k != ref && std::fabs(o.g(k) - 1.f) > 1e-4f) unitGain = false;

    const int slots = qh_ * groups64(qw_);
    Buf rows = g_.alloc((size_t)slots * (unitGain ? 1 : n_) * 4);
    for (int k = 0; k < n_; k++) {
        if (k == ref) continue;
        g_.run("merge_rob", gs(qw_), gs(qh_), mean3_, ref * qn, k * qn, frames_, k * fs, W_, qw_, qh_, odx_, ody_, k * nt_,
               tnx_, tny_, win_, o.g(k), o.clipAt(k), black, nm.a, nm.b, o.robustLo, o.robustHi, rob);
        if (o.dilate <= 1) {
            g_.run_local("min3_merge_acc", gs(qw_), gs(qh_), 16, 8, rob, o.dilate, frames_, k * fs, W_, qw_, qh_, odx_, ody_,
                         k * nt_, tnx_, tny_, win_, black, out, wsum);
            if (!unitGain) g_.run_local("row_sums_half", gs(qw_), gs(qh_), 64, 1, rob, 0, qw_, qh_, 1.f, rows, k * slots, kRed);
        } else {
            g_.run("min_filter_h_half", gs(qw_), gs(qh_), rob, 0, tmp, qw_, qh_, o.dilate);
            g_.run("min_filter_v_half", gs(qw_), gs(qh_), tmp, rob, 0, qw_, qh_, o.dilate);
            if (!unitGain) g_.run_local("row_sums_half", gs(qw_), gs(qh_), 64, 1, rob, 0, qw_, qh_, 1.f, rows, k * slots, kRed);
            g_.run_local("min3_merge_acc", gs(qw_), gs(qh_), 16, 8, rob, 0, frames_, k * fs, W_, qw_, qh_, odx_, ody_, k * nt_,
                         tnx_, tny_, win_, black, out, wsum);
        }
    }
    g_.run("merge_norm", gs(qw_), gs(qh_), out, wsum, W_, qw_, qh_, black);
    double meanW = 0;
    if (n_ > 1) {
        if (unitGain) {
            g_.run_local("row_sums", gs(qw_), gs(qh_), 64, 1, wsum, 0, qw_, qh_, 0.f, rows, 0, kRed);
            std::vector<float> rs(slots);
            g_.read(rows, rs.data(), rs.size() * 4);
            double wTotal = 0;
            for (float v : rs) wTotal += v;
            meanW = std::max(0.0, (wTotal - (double)qw_ * qh_)) / ((double)qw_ * qh_ * (n_ - 1));
        } else {
            std::vector<float> rs((size_t)slots * n_);
            g_.read(rows, rs.data(), rs.size() * 4);
            double wTotal = 0;
            for (int k = 0; k < n_; k++)
                if (k != ref) for (int y = 0; y < slots; y++) wTotal += rs[(size_t)k * slots + y];
            meanW = wTotal / ((double)qw_ * qh_ * (n_ - 1));
        }
    } else {
        g_.flush();
    }
    merged_ = out;
    if (g_.profiling()) g_.finish();
    return meanW;
}

void GpuBurst::stream_ref(const Raw& ref, float black, ImageF* gray) {
    const size_t fs = (size_t)W_ * H_;
    g_.write(frames_, ref.row(0), fs * 2, 0);
    refPyr_ = pyramid(0);
    if (gray) *gray = download(refPyr_[0]);
    sOut_ = g_.image(W_, H_);
    sWsum_ = g_.image(qw_, qh_);
    sRob_ = g_.half_image(qw_, qh_);
    tnx_ = (qw_ + kStride - 1) / kStride;
    tny_ = (qh_ + kStride - 1) / kStride;
    nt_ = tnx_ * tny_;
    sOdx_ = g_.alloc((size_t)nt_ * kMaxStreamFrames * 2);
    sOdy_ = g_.alloc((size_t)nt_ * kMaxStreamFrames * 2);
    sRows_ = g_.alloc((size_t)qh_ * groups64(qw_) * 4);
    g_.run("merge_init", gs(qw_), gs(qh_), frames_, 0, W_, qw_, qh_, black, sOut_, sWsum_);
    sAdded_ = 0;
    g_.flush();
}

void GpuBurst::stream_align(const Raw& frame, Offsets* hostOffs, ImageF* gray) {
    const size_t fs = (size_t)W_ * H_;
    sCurSlot_ = 1 + (sAdded_ & 1);
    sCurOff_ = std::min(sAdded_, kMaxStreamFrames - 1) * nt_;
    g_.write(frames_, frame.row(0), fs * 2, fs * 2 * sCurSlot_);
    std::vector<GImg> altPyr = pyramid(sCurSlot_);
    align_pyramids_to(refPyr_, altPyr, sOdx_, sOdy_, sCurOff_);
    if (hostOffs) {
        hostOffs->nx = tnx_;
        hostOffs->ny = tny_;
        hostOffs->dx.resize((size_t)nt_);
        hostOffs->dy.resize((size_t)nt_);
        g_.read_async(sOdx_, hostOffs->dx.data(), (size_t)nt_ * 2, (size_t)sCurOff_ * 2);
        if (gray) {
            g_.read_async(sOdy_, hostOffs->dy.data(), (size_t)nt_ * 2, (size_t)sCurOff_ * 2);
            *gray = download(altPyr[0]);
        } else {
            g_.read(sOdy_, hostOffs->dy.data(), (size_t)nt_ * 2, (size_t)sCurOff_ * 2);
        }
    } else if (gray) {
        *gray = download(altPyr[0]);
    }
}

void GpuBurst::stream_merge(const NoiseModel& nm, float black, const MergeOpts& o) {
    const int qn = qw_ * qh_, fs = W_ * H_;
    g_.run("merge_rob", gs(qw_), gs(qh_), mean3_, 0, sCurSlot_ * qn, frames_, sCurSlot_ * fs, W_, qw_, qh_, sOdx_, sOdy_,
           sCurOff_, tnx_, tny_, win_, 1.f, 1e30f, black, nm.a, nm.b, o.robustLo, o.robustHi, sRob_);
    if (o.dilate <= 1) {
        g_.run_local("min3_merge_acc", gs(qw_), gs(qh_), 16, 8, sRob_, o.dilate, frames_, sCurSlot_ * fs, W_, qw_, qh_, sOdx_,
                     sOdy_, sCurOff_, tnx_, tny_, win_, black, sOut_, sWsum_);
    } else {
        if (!sTmp_.b) sTmp_ = g_.half_image(qw_, qh_);
        g_.run("min_filter_h_half", gs(qw_), gs(qh_), sRob_, 0, sTmp_, qw_, qh_, o.dilate);
        g_.run("min_filter_v_half", gs(qw_), gs(qh_), sTmp_, sRob_, 0, qw_, qh_, o.dilate);
        g_.run_local("min3_merge_acc", gs(qw_), gs(qh_), 16, 8, sRob_, 0, frames_, sCurSlot_ * fs, W_, qw_, qh_, sOdx_, sOdy_,
                     sCurOff_, tnx_, tny_, win_, black, sOut_, sWsum_);
    }
    sAdded_++;
    g_.flush();
}

double GpuBurst::stream_end(float black, double* maxShift) {
    g_.run("merge_norm", gs(qw_), gs(qh_), sOut_, sWsum_, W_, qw_, qh_, black);
    double meanW = 0;
    if (sAdded_ > 0) {
        const int slots = qh_ * groups64(qw_);
        g_.run_local("row_sums", gs(qw_), gs(qh_), 64, 1, sWsum_, 0, qw_, qh_, 0.f, sRows_, 0, kRed);
        int tracked = std::min(sAdded_, kMaxStreamFrames);
        std::vector<int16_t> hx, hy;
        if (maxShift && nt_ > 0) {
            hx.resize((size_t)tracked * nt_);
            hy.resize((size_t)tracked * nt_);
            g_.read_async(sOdx_, hx.data(), hx.size() * 2);
            g_.read_async(sOdy_, hy.data(), hy.size() * 2);
        }
        std::vector<float> rs(slots);
        g_.read(sRows_, rs.data(), rs.size() * 4);
        double wTotal = 0;
        for (float v : rs) wTotal += v;
        meanW = std::max(0.0, (wTotal - (double)qw_ * qh_)) / ((double)qw_ * qh_ * sAdded_);
        if (maxShift && nt_ > 0) {
            double ms = *maxShift;
            for (int k = 0; k < tracked; k++) {
                double sx = 0, sy = 0;
                const int16_t* dx = hx.data() + (size_t)k * nt_;
                const int16_t* dy = hy.data() + (size_t)k * nt_;
                for (int i = 0; i < nt_; i++) { sx += dx[i]; sy += dy[i]; }
                ms = std::max(ms, std::hypot(sx / nt_, sy / nt_));
            }
            *maxShift = ms;
        }
    } else {
        g_.flush();
    }
    merged_ = sOut_;
    sOut_ = sWsum_ = sRob_ = sTmp_ = GImg();
    sRows_.reset();
    sOdx_.reset();
    sOdy_.reset();
    refPyr_.clear();
    drop_frames();
    if (g_.profiling()) g_.finish();
    return meanW;
}

void GpuBurst::drop_frames() {
    frames_.reset();
    mean3_.reset();
    grayRef_ = GImg();
    odx_.reset();
    ody_.reset();
    offsOnGpu_ = false;
}

ImageF GpuBurst::download_merged() {
    if (merged_.w <= 0 || !merged_.b) throw GpuError("no merged mosaic on GPU");
    return download(merged_);
}

void GpuBurst::set_merged(const ImageF& im) { merged_ = upload(im); }

// ---- finishing

Buf GpuBurst::gaussian_weights(float sigma, int maxRadius, int& outR) {
    int r = std::max(1, (int)std::ceil(2.5f * sigma));
    if (maxRadius > 0) r = std::min(r, maxRadius);
    outR = r;
    std::vector<float> k(2 * r + 1);
    float s = 0;
    for (int i = -r; i <= r; i++) s += k[i + r] = std::exp(-0.5f * i * i / (sigma * sigma));
    for (float& v : k) v /= s;
    Buf kb = g_.alloc(k.size() * 4);
    g_.write(kb, k.data(), k.size() * 4);
    return kb;
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

void GpuBurst::demosaic_rcd_and_color(const GImg& cfa, const Meta& m, const float lcaR[2], const float lcaB[2]) {
    const int W = cfa.w, H = cfa.h;
    const bool hasLca = lcaR && lcaB && W >= 2 && H >= 2 &&
                        (lcaR[0] != 0 || lcaR[1] != 0 || lcaB[0] != 0 || lcaB[1] != 0);
    const float* M = m.ccm;
    const cl_float4 m0 = f4(M[0], M[1], M[2], 0), m1 = f4(M[3], M[4], M[5], 0), m2 = f4(M[6], M[7], M[8], 0);

    GImg hpf = g_.half_image(W, H, 4);
    g_.run("rcd_hpf", gs(W), gs(H), cfa, W, H, hpf);
    GImg vhLpf = g_.half_image(W, H, 2), pq = g_.half_image(W, H, 1);
    g_.run("rcd_dir", gs(W), gs(H), cfa, W, H, m.cfa, hpf, vhLpf, pq);
    hpf = GImg();
    GImg G = g_.half_image(W, H, 1), rb = g_.half_image(W, H, 2);
    g_.run("rcd_green", gs(W), gs(H), cfa, W, H, m.cfa, vhLpf, G);
    g_.run("rcd_rb_at_rb", gs(W), gs(H), cfa, W, H, m.cfa, G, pq, rb);
    pq = GImg();

    rgb_ = g_.half_image(W, H, 4);
    g_.run("rcd_finish_rgb", gs(W), gs(H), cfa, W, H, m.cfa, G, vhLpf, rb, hasLca ? 0 : 1, m0, m1, m2, rgb_);
    G = vhLpf = rb = GImg();

    if (hasLca) {
        const float fcx = (m.full_w() - 1) * 0.5f, fcy = (m.full_h() - 1) * 0.5f;
        const float invR2 = 1.f / (fcx * fcx + fcy * fcy);
        const float lcx = fcx - m.cropX, lcy = fcy - m.cropY;
        GImg out = g_.half_image(W, H, 4);
        g_.run("lca_ccm_wb", gs(W), gs(H), rgb_, W, H, f4(lcaR[0], lcaR[1], lcaB[0], lcaB[1]), lcx, lcy, invR2, m0, m1, m2, out);
        rgb_ = out;
    }
}

void GpuBurst::chroma_and_prepare(float strength, bool writeTone, Buf& keySub, int s8w, GImg& Ys, int sw, int sh) {
    const int W = rgb_.w, H = rgb_.h;
    if (strength <= 0.f) {
        g_.run("clamp_nonneg_half4", gs(W * H), 1, rgb_, W * H);
        if (writeTone) g_.run_local("tm_prepare", gs(W), gs(H), 16, 8, rgb_, W, H, keySub, s8w, Ys, sw, sh);
        return;
    }
    const int cw = (W + 3) / 4, ch = (H + 3) / 4;
    GImg crCb = g_.image(cw, ch, 2), tmp2 = g_.image(cw, ch, 2);
    Img2D crCbImg = g_.image2d(cw, ch);
    int r = 1;
    Buf kb = gaussian_weights(1.0f * strength, 0, r);
    g_.run("chroma_down", gs(cw), gs(ch), rgb_, W, H, crCb, cw, ch);
    g_.run("blur2_h", gs(cw), gs(ch), crCb, tmp2, cw, ch, kb, r);
    g_.run("blur2_v_img", gs(cw), gs(ch), tmp2, crCbImg, cw, ch, kb, r);
    g_.run_local("chroma_apply_img", gs(W), gs(H), 16, 8, rgb_, W, H, crCbImg, writeTone ? 1 : 0, keySub, s8w, Ys, sw, sh);
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
    GImg mid = g_.half_image(W, h, 4);
    g_.run("lanczos_h_half4", gs(W), gs(h), rgb_, w, h, mid, W, bx0, bxw);
    rgb_ = GImg();
    GImg out = g_.half_image(W, H, 4);
    g_.run("lanczos_v_half4", gs(W), gs(H), mid, W, h, out, H, by0, byw);
    rgb_ = out;
}

void GpuBurst::crop_aspect(float aspect) {
    int x0, y0, cw, ch;
    aspect_crop_box(rgb_.w, rgb_.h, aspect, x0, y0, cw, ch);
    if (cw == rgb_.w && ch == rgb_.h) return;
    GImg out = g_.half_image(cw, ch, 4);
    g_.run("crop_half4", gs(cw), gs(ch), rgb_, rgb_.w, x0, y0, out, cw, ch);
    rgb_ = out;
}

GImg GpuBurst::fuse_exposures(const GImg& Y, float gS, float gL) {
    const int W = Y.w, H = Y.h, n = W * H;
    GImg eS = g_.image(W, H), eL = g_.image(W, H), wS = g_.image(W, H);
    g_.run("tm_exposures", gs(n), 1, Y, n, gS, gL, eS, eL, wS);
    int levels = 1;
    int minTop = W > 2000 ? 24 * kToneSub : 24;
    for (int m = std::min(W, H); m > minTop; m /= 2) levels++;
    std::vector<GImg> gsP, glP, gwP;
    gsP.push_back(eS); glP.push_back(eL); gwP.push_back(wS);
    eS = eL = wS = GImg();
    for (int l = 1; l < levels; l++) {
        gsP.push_back(pyr_down(gsP.back()));
        glP.push_back(pyr_down(glP.back()));
        gwP.push_back(pyr_down(gwP.back()));
    }
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

ToneInfo GpuBurst::local_tonemap_and_look(const LookParams& p, Buf& keySub, int s8w, int s8h, GImg& Ys, int sw, int sh) {
    const size_t nKey = (size_t)s8w * s8h;
    const float* ksPtr = static_cast<const float*>(g_.map_read(keySub));
    std::vector<float> samp(ksPtr, ksPtr + nKey);
    g_.unmap(keySub);
    double lsum = 0;
    for (float v : samp) lsum += std::log(v + 1e-4);
    float key = (float)std::exp(lsum / samp.size());
    std::nth_element(samp.begin(), samp.begin() + (size_t)(samp.size() * 0.995), samp.end());
    float p995 = samp[(size_t)(samp.size() * 0.995)];
    float gL = clampf(p.targetKey / key, 1.f, 64.f);
    float maxRatio = p.compression > 0 ? p.compression : 8.f;
    float gS = clampf(1.f / p995, gL / maxRatio, gL);

    const int sn = sw * sh;
    GImg fused = fuse_exposures(Ys, gS, gL);
    GImg guide4 = g_.image(sw, sh, 4), tmp4 = g_.image(sw, sh, 4), m4 = g_.image(sw, sh, 4);
    g_.run("tm_guide4", gs(sn), 1, Ys, fused, guide4, sn);
    fused = GImg();
    g_.run("box4_h", gs(sw), gs(sh), guide4, tmp4, sw, sh, kToneGuideRadius);
    g_.run("box4_v", gs(sw), gs(sh), tmp4, m4, sw, sh, kToneGuideRadius);
    guide4 = tmp4 = GImg();

    GImg ab = g_.image(sw, sh, 2), tmpAb = g_.image(sw, sh, 2);
    Img2D mABImg = g_.image2d(sw, sh);
    g_.run("tm_coef2", gs(sn), 1, m4, ab, sn, kToneGuideEps);
    m4 = GImg();
    g_.run("box2_h", gs(sw), gs(sh), ab, tmpAb, sw, sh, kToneGuideRadius);
    g_.run("box2_v_img", gs(sw), gs(sh), tmpAb, mABImg, sw, sh, kToneGuideRadius);
    ab = tmpAb = GImg();

    const int wpW = (sw + 1) / 2, wpH = (sh + 1) / 2;
    Buf wpSub = g_.alloc_host((size_t)wpW * wpH * 4);
    g_.run("tm_wp_samples", gs(wpW), gs(wpH), Ys, sw, sh, mABImg, wpSub, wpW, wpH);
    Ys = GImg();

    const size_t nWp = (size_t)wpW * wpH;
    const float* wpPtr = static_cast<const float*>(g_.map_read(wpSub));
    std::vector<float> wpVec(wpPtr, wpPtr + nWp);
    g_.unmap(wpSub);
    size_t idx = (size_t)(wpVec.size() * 0.995);
    std::nth_element(wpVec.begin(), wpVec.begin() + idx, wpVec.end());
    const float hi = wpVec[idx];
    constexpr float kWhite = 0.72f, kMaxLift = 1.35f;
    const float lift = clampf(kWhite / std::max(hi, 1e-6f), 1.f, kMaxLift);
    const float lo = p.targetKey;
    const bool doLift = lift > 1.001f && hi > lo * 1.5f;
    const float invSpan = doLift ? 1.f / (hi - lo) : 0.f;
    const float lift1 = doLift ? (lift - 1.f) : 0.f;

    g_.run("tonemap_and_look", gs(rgb_.w), gs(rgb_.h), rgb_, rgb_.w, rgb_.h, mABImg, lo, invSpan, lift1, p.contrast, p.saturation);
    return {gS, gL, key};
}

void GpuBurst::finish(const Meta& m, const LookParams& look, Demosaic, float zoom, bool keepHdr, FinishOut& out, float aspect,
                      const float lcaR[2], const float lcaB[2]) {
    rgb8Ptr_ = nullptr;
    LookParams lp = look;
    Timer tStage;
    const int W = merged_.w, H = merged_.h;
    const bool lsc = m.lscRows > 1 && m.lscCols > 1;
    Buf lscBuf = g_.alloc(lsc ? m.lsc.size() * 4 : 16);
    if (lsc) g_.write(lscBuf, m.lsc.data(), m.lsc.size() * 4);
    const float swFull = m.full_w() - 1.f, shFull = m.full_h() - 1.f;

    // Enqueue normalize_mosaic + sigma_rows_cfa + RCD + CCM + log_luma_rows as one continuous GPU batch (Sync #1).
    GImg cfa = g_.image(W, H);
    g_.run_local("normalize_mosaic", gs(W), gs(H), 16, 8, merged_, cfa, W, H, m.cfa,
                 f4(m.black[0], m.black[1], m.black[2], m.black[3]), m.white,
                 f4(m.wb[0], m.wb[1], m.wb[2], m.wb[3]), lscBuf, lsc ? m.lscRows : 0, lsc ? m.lscCols : 0, m.cropX,
                 m.cropY, swFull, shFull, lp.lscLumaStrength);

    int ox = 0, oy = 0;
    for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) if (cfa_channel(m.cfa, x, y) == 1) { ox = x; oy = y; }
    const int sigSlots = (H / 2) * groups64(W / 2);
    Buf sigmaRows = g_.alloc((size_t)std::max(1, sigSlots) * 8);
    g_.run_local("sigma_rows_cfa", gs(W / 2), gs(H / 2), 64, 1, cfa, W, H, ox, oy, sigmaRows, kRed);

    demosaic_rcd_and_color(cfa, m, lcaR, lcaB);
    cfa = GImg();

    const int gr = (rgb_.h + 7) / 8, gc = (rgb_.w + 7) / 8;
    const int logSlots = gr * groups64(gc);
    Buf logRows = g_.alloc((size_t)std::max(1, logSlots) * 4);
    g_.run_local("log_luma_rows", gs(gc), gs(gr), 64, 1, rgb_, rgb_.w, rgb_.h, logRows, kRed);

    std::vector<float> sv((size_t)sigSlots * 2);
    std::vector<float> lv((size_t)logSlots);
    g_.read_async(sigmaRows, sv.data(), sv.size() * 4);
    g_.read(logRows, lv.data(), lv.size() * 4);
    out.ft.demosaic = tStage.ms();
    tStage = Timer();

    double sigS = 0, sigN = 0;
    for (int i = 0; i < sigSlots; i++) { sigS += sv[2 * i]; sigN += sv[2 * i + 1]; }
    out.sigma = sigN > 0 ? std::sqrt(M_PI / 2) / (6.0 * sigN) * sigS : 0.0;

    double ls = 0;
    for (float v : lv) ls += v;
    const float logKey = (float)std::exp(ls / std::max(1L, (long)gr * gc));
    lp.targetKey = noise_limited_key(lp.targetKey, (float)out.sigma * clampf(lp.targetKey / logKey, 1.f, 64.f));
    out.targetKey = lp.targetKey;
    const float dgain = clampf(lp.targetKey / logKey, 1.f, 64.f);
    out.visibleSigma = out.sigma * dgain;
    float chroma = lp.chromaDenoise;
    if (chroma < 0) chroma = clampf((float)out.visibleSigma / 0.01f, 1.f, 6.f);
    chroma *= lp.chromaDenoiseScale;
    out.chroma = chroma;

    const bool needGeom = (zoom > 1.f) || (aspect > 0.f);
    int s8w = (rgb_.w + 7) / 8, s8h = (rgb_.h + 7) / 8;
    int sw = (rgb_.w + kToneSub - 1) / kToneSub, sh = (rgb_.h + kToneSub - 1) / kToneSub;
    Buf keySub = needGeom ? g_.alloc(16) : g_.alloc_host((size_t)s8w * s8h * 4);
    GImg Ys;
    if (!needGeom) Ys = g_.image(sw, sh);
    else { Ys.w = 1; Ys.h = 1; Ys.c = 1; Ys.b = g_.alloc(16); }

    chroma_and_prepare(chroma, !needGeom, keySub, s8w, Ys, sw, sh);
    if (needGeom) {
        if (zoom > 1.f) upscale(m.full_w(), m.full_h());
        if (aspect > 0.f) crop_aspect(aspect);
        s8w = (rgb_.w + 7) / 8;
        s8h = (rgb_.h + 7) / 8;
        sw = (rgb_.w + kToneSub - 1) / kToneSub;
        sh = (rgb_.h + kToneSub - 1) / kToneSub;
        keySub = g_.alloc_host((size_t)s8w * s8h * 4);
        Ys = g_.image(sw, sh);
        g_.run_local("tm_prepare", gs(rgb_.w), gs(rgb_.h), 16, 8, rgb_, rgb_.w, rgb_.h, keySub, s8w, Ys, sw, sh);
    }
    if (keepHdr) {
        HdrLuma o;
        const int bw = (rgb_.w + o.block - 1) / o.block, bh = (rgb_.h + o.block - 1) / o.block;
        GImg y = g_.image(bw, bh);
        g_.run("hdr_luma_blocks_half4", gs(bw), gs(bh), rgb_, rgb_.w, rgb_.h, o.block, y, bw, bh);
        o.y = download(y);
        out.hdr = std::move(o);
    }

    // Fused local_tonemap + Oklab pixel_look, then fused local-tile sharpen + dithered zero-copy RGB8!
    out.tone = local_tonemap_and_look(lp, keySub, s8w, s8h, Ys, sw, sh);
    out.ft.tonemap = tStage.ms();
    tStage = Timer();

    outW_ = rgb_.w;
    outH_ = rgb_.h;
    out.sharpenCore = sharpen_core((float)out.visibleSigma);
    const size_t rgb8Bytes = (size_t)outW_ * outH_ * 3;
    rgb8_ = g_.alloc(rgb8Bytes);
    if (lp.sharpen > 0.f) {
        int sr = 3;
        Buf kb = gaussian_weights(lp.sharpenRadius, 4, sr);
        g_.run_local("sharpen_to_rgb8", gs(outW_), gs(outH_), 16, 8, rgb_, outW_, outH_, kb, sr, lp.sharpen,
                     (float)out.sharpenCore, rgb8_);
    } else {
        g_.run_local("half4_to_rgb8", gs(outW_), gs(outH_), 16, 8, rgb_, outW_, outH_, rgb8_);
    }
    rgb_ = GImg();
    if (g_hostRgb8.size() < rgb8Bytes) g_hostRgb8.resize(rgb8Bytes);
    g_.read(rgb8_, g_hostRgb8.data(), rgb8Bytes);
    rgb8_.reset();
    merged_ = GImg();
    rgb8Ptr_ = g_hostRgb8.data();
    out.ft.sharpen = tStage.ms();
}

void GpuBurst::render(uint8_t* rgba, size_t stride) {
    if (!rgb8Ptr_ || outW_ <= 0 || outH_ <= 0) throw GpuError("no finished GPU RGB8 image");
    rgb8_to_rgba8888(rgb8Ptr_, outW_, outH_, rgba, stride);
}

std::vector<uint8_t> GpuBurst::render_rgb8() {
    if (!rgb8Ptr_ || outW_ <= 0 || outH_ <= 0) return {};
    return std::vector<uint8_t>(rgb8Ptr_, rgb8Ptr_ + (size_t)outW_ * outH_ * 3);
}

void GpuBurst::render_thumb(uint8_t* rgba, size_t stride, int outW, int outH) {
    if (!rgb8Ptr_ || outW_ <= 0 || outH_ <= 0) throw GpuError("no finished GPU RGB8 image");
    rgb8_to_rgba8888_thumb(rgb8Ptr_, outW_, outH_, rgba, stride, outW, outH);
}

ImageF GpuBurst::download_rgb() {
    if (!rgb8Ptr_ || outW_ <= 0 || outH_ <= 0) return {};
    ImageF im(outW_, outH_, 3);
    parallel_for(outH_, [&](int y) {
        const uint8_t* s = rgb8Ptr_ + (size_t)y * outW_ * 3;
        float* d = im.row(y);
        for (int x = 0; x < outW_ * 3; x++) d[x] = s[x] * (1.f / 255.f);
    });
    return im;
}
