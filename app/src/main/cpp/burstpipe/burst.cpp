#include "burst.h"

int g_threads = (int)std::max(1u, std::thread::hardware_concurrency());

namespace {

// Immerkaer (1996) fast noise sigma estimate on the G_even sub-plane of a normalised mosaic
double noise_sigma(const ImageF& cfa, int cfaType) {
    int ox = 0, oy = 0;
    for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) if (cfa_channel(cfaType, x, y) == 1) { ox = x; oy = y; }
    int w = cfa.w / 2, h = cfa.h / 2;
    double s = 0;
    long n = 0;
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            auto P = [&](int u, int v) { return cfa.at(2 * (x + u) + ox, 2 * (y + v) + oy); };
            double c = P(0, 0);
            if (c > 0.9) continue;
            double l = 4 * c - 2 * (P(-1, 0) + P(1, 0) + P(0, -1) + P(0, 1)) + (P(-1, -1) + P(1, -1) + P(-1, 1) + P(1, 1));
            s += std::fabs(l);
            n++;
        }
    return n ? std::sqrt(M_PI / 2) / (6.0 * n) * s : 0;
}

// same estimator on the G channel of an RGB image, sampled on a grid matching one Bayer G plane
double noise_sigma_rgb(const ImageF& rgb, int step) {
    int w = rgb.w / step, h = rgb.h / step;
    double s = 0;
    long n = 0;
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            auto P = [&](int u, int v) { return rgb.at(step * (x + u), step * (y + v), 1); };
            double c = P(0, 0);
            if (c > 0.9) continue;
            double l = 4 * c - 2 * (P(-1, 0) + P(1, 0) + P(0, -1) + P(0, 1)) + (P(-1, -1) + P(1, -1) + P(-1, 1) + P(1, 1));
            s += std::fabs(l);
            n++;
        }
    return n ? std::sqrt(M_PI / 2) / (6.0 * n) * s : 0;
}

// exposure gain the tone mapper will apply to reach the target key (log-average luma)
float display_gain(const ImageF& rgb, float targetKey) {
    double s = 0;
    long n = 0;
    for (int y = 0; y < rgb.h; y += 8)
        for (int x = 0; x < rgb.w; x += 8) {
            const float* p = rgb.row(y) + (size_t)x * 3;
            s += std::log(1e-4 + 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]);
            n++;
        }
    return clampf(targetKey / (float)std::exp(s / std::max(1L, n)), 1.f, 64.f);
}

}  // namespace

BurstSession::BurstSession(const Meta& m, std::vector<Raw> frames, int ref)
    : m_(m), frames_(std::move(frames)), ref_(clampi(ref, 0, (int)frames_.size() - 1)) {}

void BurstSession::align() {
    Timer t;
    int n = (int)frames_.size();
    if (n > 1 && (int)m_.exposureNs.size() >= n) {
        auto expo = [&](int k) { return m_.exposureNs[k] * (m_.iso.size() > (size_t)k ? m_.iso[k] : 100); };
        int r = ref_;
        for (int k = 0; k < n; k++) if (expo(k) < expo(r) * 0.9) r = k;
        float refWhite;
        if (scale_to_reference(frames_, r, m_, mo_, refWhite)) {
            ref_ = r;
            m_.white = refWhite;
            bracket_ = true;
        }
    }
    gray_.resize(n);
    mean3_.resize(n);
    for (int k = 0; k < n; k++) { gray_[k] = gray_from_raw(frames_[k]); mean3_[k] = box3(gray_[k]); }
    offs_.resize(n);
    std::vector<ImageF> refPyr = build_pyramid(gray_[ref_]);
    for (int k = 0; k < n; k++) {
        if (k == ref_) {
            Offsets z;
            z.nx = (gray_[ref_].w + kStride - 1) / kStride;
            z.ny = (gray_[ref_].h + kStride - 1) / kStride;
            z.dx.assign((size_t)z.nx * z.ny, 0);
            z.dy = z.dx;
            offs_[k] = z;
            continue;
        }
        offs_[k] = align_frames(refPyr, build_pyramid(gray_[k]));
        double sx = 0, sy = 0;
        for (size_t i = 0; i < offs_[k].dx.size(); i++) { sx += offs_[k].dx[i]; sy += offs_[k].dy[i]; }
        sx /= offs_[k].dx.size();
        sy /= offs_[k].dy.size();
        maxShift_ = std::max(maxShift_, std::hypot(sx, sy));
    }
    tAlign_ = t.ms();

    t = Timer();
    float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    if (n > 1) {
        // a frame with the reference's exposure, so the residual is noise only
        int alt = ref_ == 0 ? 1 : 0;
        for (int k = 0; k < n; k++) if (k != ref_ && std::fabs(mo_.g(k) - 1.f) < 0.05f) { alt = k; break; }
        nm_ = estimate_noise(gray_[ref_], gray_[alt], offs_[alt], black);
    }
    tNoise_ = t.ms();
}

void BurstSession::merge(const BurstOptions& o) {
    Timer t;
    int n = (int)frames_.size();
    float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    opt_ = o;
    // the frequency merge has no exposure weighting
    if (bracket_) opt_.freqMerge = false;
    mo_.robustLo = o.robustLo;
    mo_.robustHi = o.robustHi;
    mo_.dilate = o.dilate;
    kr_ = o.kernelRegression;
    if (kr_) {
        std::vector<Flow> flows(n);
        LkGrad lkg = lk_gradients(mean3_[ref_]);
        for (int k = 0; k < n; k++)
            flows[k] = k == ref_ ? zero_flow(offs_[k]) : refine_flow(mean3_[ref_], mean3_[k], offs_[k], 3, lkg, nm_, black);
        KrParams kp;
        for (int i = 0; i < 2; i++) { kp.lcaR[i] = o.lcaR[i]; kp.lcaB[i] = o.lcaB[i]; }
        rgb_ = kr_merge(frames_, ref_, flows, gray_[ref_], mean3_, m_, nm_, mo_, kp, &ks_);
        meanWeight_ = ks_.meanWeight;
    } else if (n > 1) {
        MergeStats ms;
        mosaic_ = opt_.freqMerge ? merge_frames_freq(frames_, ref_, offs_, m_, nm_, o.freqC, o.freqSpatial, &ms)
                                 : merge_frames(frames_, ref_, offs_, mean3_, nm_, black, mo_, &ms);
        meanWeight_ = ms.meanWeight;
    } else {
        mosaic_ = ImageF(m_.width, m_.height);
        const uint16_t* src = frames_[ref_].d;
        for (size_t i = 0; i < mosaic_.d.size(); i++) mosaic_.d[i] = src[i];
    }
    frames_.clear();
    gray_.clear();
    mean3_.clear();
    offs_.clear();
    tMerge_ = t.ms();
}

void BurstSession::finish(const LookParams& lp, bool keepHdr) {
    Timer t;
    if (kr_) {
        normalize_rgb(rgb_, m_, lp, 1.f);
        ft_.lscwb = t.ms();
        sigma_ = noise_sigma_rgb(rgb_, 2);
    } else {
        ImageF cfa = normalize_mosaic(mosaic_, m_, lp);
        ft_.lscwb = t.ms();
        sigma_ = noise_sigma(cfa, m_.cfa);
        mosaic_ = ImageF();
        t = Timer();
        rgb_ = demosaic(cfa, m_.cfa, opt_.demosaic);
        ft_.demosaic = t.ms();
    }
    t = Timer(); apply_ccm(rgb_, m_.ccm); wb_shift(rgb_, lp.wbCool); ft_.color = t.ms();
    float chroma = lp.chromaDenoise;
    visibleSigma_ = sigma_ * display_gain(rgb_, lp.targetKey);
    if (chroma < 0) chroma = clampf((float)visibleSigma_ / 0.01f, 1.f, 6.f);
    chroma *= lp.chromaDenoiseScale;
    chroma_ = chroma;
    t = Timer(); chroma_denoise(rgb_, chroma); ft_.chroma = t.ms();
    hdr_ = keepHdr ? hdr_luma(rgb_, 1) : HdrLuma();
    t = Timer(); tone_ = local_tonemap(rgb_, lp); ft_.tonemap = t.ms();
    t = Timer(); pixel_look(rgb_, lp); local_contrast(rgb_, lp.localContrast, lp.localRadius); hue_tune(rgb_, lp); ft_.look = t.ms();
    t = Timer(); sharpen_luma(rgb_, lp.sharpen, lp.sharpenRadius); ft_.sharpen = t.ms();
}

void BurstSession::render(uint8_t* rgba, size_t stride) const { to_rgba8888(rgb_, rgba, stride); }

std::vector<uint8_t> BurstSession::encode_ultrahdr(int quality, float headroom) {
    if (hdr_.y.w == 0) return {};
    Timer t;
    std::vector<uint8_t> file = ::encode_ultrahdr(rgb_, 1, hdr_, tone_.gainLong, headroom, quality, &us_);
    tUhdr_ = t.ms();
    return file;
}

std::string BurstSession::summary() const {
    char buf[1024];
    std::string mode = kr_ ? "kr" : std::string("classic/") + (opt_.freqMerge ? "freq" : "spatial") + "/" + demosaic_name(opt_.demosaic);
    if (bracket_) {
        mode += " bracket gains";
        for (float g : mo_.gain) { char b[16]; snprintf(b, sizeof b, " %.2f", g); mode += b; }
    }
    if (kr_ && (opt_.lcaR[0] != 0 || opt_.lcaB[0] != 0)) mode += " lca";
    snprintf(buf, sizeof buf,
             "%s %dx%d ref=%d threads=%d | noise var=%.4f*S+%.3f weight=%.3f shift=%.1f | sigma=%.5f visible=%.4f chroma=%.2f | "
             "tone key=%.4f short=%.2f long=%.2f | ms align %.0f noise %.0f merge %.0f lsc+wb %.0f demosaic %.0f ccm %.0f "
             "chroma %.0f tonemap %.0f look %.0f sharpen %.0f",
             mode.c_str(), m_.width, m_.height, ref_, g_threads, nm_.a, nm_.b, meanWeight_, maxShift_, sigma_,
             visibleSigma_, chroma_, tone_.keyBefore, tone_.gainShort, tone_.gainLong, tAlign_, tNoise_, tMerge_,
             ft_.lscwb, ft_.demosaic, ft_.color, ft_.chroma, ft_.tonemap, ft_.look, ft_.sharpen);
    std::string s = buf;
    if (us_.bytes) {
        snprintf(buf, sizeof buf, " | ultrahdr max boost %.2f gain map %zu B file %zu B %.0f ms", us_.maxBoost, us_.gainMapBytes, us_.bytes, tUhdr_);
        s += buf;
    } else if (tUhdr_ > 0) {
        snprintf(buf, sizeof buf, " | ultrahdr skipped, max boost %.2f", us_.maxBoost);
        s += buf;
    }
    return s;
}
