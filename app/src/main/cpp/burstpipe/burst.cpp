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

void BurstSession::crop(float zoom) {
    if (zoom <= 1.f) return;
    Timer t;
    const int W = m_.width, H = m_.height;
    const int cw = (int)std::lround(W / zoom) & ~1, ch = (int)std::lround(H / zoom) & ~1;
    cropX0_ = ((W - cw) / 2) & ~1;
    cropY0_ = ((H - ch) / 2) & ~1;
    m_.fullW = m_.full_w();
    m_.fullH = m_.full_h();
    m_.cropX += cropX0_;
    m_.cropY += cropY0_;
    m_.width = cw;
    m_.height = ch;
    m_.rowStride = cw * 2;
    zoom_ = zoom;
    for (Raw& r : frames_) r = cropped(r);
    tCrop_ = t.ms();
}

// a private copy of the crop() window of a full-sensor frame
Raw BurstSession::cropped(const Raw& r) const {
    Raw c;
    c.w = m_.width;
    c.h = m_.height;
    c.own = std::make_shared<std::vector<uint16_t>>((size_t)c.w * c.h);
    for (int y = 0; y < c.h; y++) {
        const uint16_t* s = r.d + (size_t)(cropY0_ + y) * r.w + cropX0_;
        std::copy(s, s + c.w, c.own->data() + (size_t)y * c.w);
    }
    c.d = c.own->data();
    return c;
}

void BurstSession::align() {
    Timer t;
    int n = (int)frames_.size();
    refScale_ = 1.f;
    if (n > 1 && (int)m_.exposureNs.size() >= n) {
        auto expo = [&](int k) { return m_.exposureNs[k] * (m_.iso.size() > (size_t)k ? m_.iso[k] : 100); };
        int r = ref_;
        for (int k = 0; k < n; k++) if (expo(k) < expo(r) * 0.9) r = k;
        float refWhite;
        if (scale_to_reference(frames_, r, m_, mo_, refWhite)) {
            float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
            refScale_ = (refWhite - black) / (m_.white - black);
            ref_ = r;
            m_.white = refWhite;
            bracket_ = true;
        }
    }
    // a frame with the reference's exposure, so the noise fit's residual is noise only
    int alt = -1;
    if (n > 1) {
        alt = ref_ == 0 ? 1 : 0;
        for (int k = 0; k < n; k++) if (k != ref_ && std::fabs(mo_.g(k) - 1.f) < 0.05f) { alt = k; break; }
    }
    if (gpuWanted_) {
        std::string why;
        gpu_ = GpuBurst::create(frames_, why);
        if (!gpu_) gpuNote_ = why;
    }
    bool onGpu = false;
    if (gpu_ && GpuBurst::enabled("align")) {
        try {
            ImageF gRef, gAlt;
            gpu_->align(ref_, alt, offs_, gRef, gAlt);
            gray_.assign(n, ImageF());
            gray_[ref_] = std::move(gRef);
            if (alt >= 0) gray_[alt] = std::move(gAlt);
            mean3_.clear();
            onGpu = true;
            gpuSteps_ += " align";
        } catch (const std::exception& e) {
            gpu_failed("align", e);
        }
    }
    if (!onGpu) {
        cpu_align_state();
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
        }
    }
    for (int k = 0; k < n; k++) {
        if (k == ref_) continue;
        double sx = 0, sy = 0;
        for (size_t i = 0; i < offs_[k].dx.size(); i++) { sx += offs_[k].dx[i]; sy += offs_[k].dy[i]; }
        sx /= offs_[k].dx.size();
        sy /= offs_[k].dy.size();
        maxShift_ = std::max(maxShift_, std::hypot(sx, sy));
    }
    tAlign_ = t.ms();

    t = Timer();
    float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    calibrated_ = !curve_.empty() && (int)m_.iso.size() > ref_;
    if (calibrated_) nm_ = curve_.at((float)m_.iso[ref_], refScale_);
    else if (n > 1) nm_ = estimate_noise(gray_[ref_], gray_[alt], offs_[alt], black);
    tNoise_ = t.ms();
}

// gray images and 3x3 means of every frame on the host, for the CPU align and merges (also after a GPU
// align, which keeps them on the device)
void BurstSession::cpu_align_state() {
    int n = (int)frames_.size();
    gray_.resize(n);
    mean3_.resize(n);
    for (int k = 0; k < n; k++) {
        if (gray_[k].w == 0) gray_[k] = gray_from_raw(frames_[k]);
        if (mean3_[k].w == 0) mean3_[k] = box3(gray_[k]);
    }
}

void BurstSession::gpu_failed(const char* step, const std::exception& e) {
    gpuNote_ = std::string(step) + " failed (" + e.what() + "), CPU from there";
    gpu_.reset();
}

void BurstSession::merge(const BurstOptions& o) {
    if (streamed_) {
        opt_.demosaic = o.demosaic;
        return;
    }
    Timer t;
    int n = (int)frames_.size();
    float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    opt_ = o;
    // the frequency merge has no exposure weighting
    if (bracket_) opt_.freqMerge = false;
    mo_.robustLo = o.robustLo > 0 ? o.robustLo : (calibrated_ ? kRobustCalibrated[0] : kRobustEstimated[0]);
    mo_.robustHi = o.robustHi > 0 ? o.robustHi : (calibrated_ ? kRobustCalibrated[1] : kRobustEstimated[1]);
    mo_.dilate = o.dilate;
    kr_ = o.kernelRegression;
    KrParams kp;
    for (int i = 0; i < 2; i++) { kp.lcaR[i] = o.lcaR[i]; kp.lcaB[i] = o.lcaB[i]; }
    const char* step = kr_ ? "kr" : opt_.freqMerge && n > 1 ? "freq" : "merge";
    bool onGpu = false;
    if (gpu_ && GpuBurst::enabled(step)) {
        try {
            if (!gpu_->aligned()) {
                cpu_align_state();
                gpu_->set_aligned(mean3_, gray_[ref_]);
            }
            if (kr_) {
                gpu_->merge_kr(frames_[ref_], ref_, offs_, m_, nm_, mo_, kp, &ks_);
                meanWeight_ = ks_.meanWeight;
            } else if (opt_.freqMerge && n > 1) {
                meanWeight_ = gpu_->merge_freq(ref_, offs_, m_, nm_, o.freqC, o.freqSpatial);
            } else {
                meanWeight_ = gpu_->merge_spatial(ref_, offs_, nm_, black, mo_);
            }
            gpu_->drop_frames();
            onGpu = true;
            gpuSteps_ += std::string(" ") + step;
        } catch (const std::exception& e) {
            gpu_failed(step, e);
        }
    }
    if (onGpu) {
    } else if (kr_) {
        cpu_align_state();
        std::vector<Flow> flows(n);
        LkGrad lkg = lk_gradients(mean3_[ref_]);
        for (int k = 0; k < n; k++)
            flows[k] = k == ref_ ? zero_flow(offs_[k]) : refine_flow(mean3_[ref_], mean3_[k], offs_[k], 3, lkg, nm_, black);
        rgb_ = kr_merge(frames_, ref_, flows, gray_[ref_], mean3_, m_, nm_, mo_, kp, &ks_);
        meanWeight_ = ks_.meanWeight;
    } else if (n > 1) {
        if (!opt_.freqMerge) cpu_align_state();
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

void BurstSession::begin_stream(const BurstOptions& o) {
    streaming_ = true;
    opt_ = o;
    opt_.kernelRegression = false;
    opt_.freqMerge = false;
    kr_ = false;
    ref_ = -1;
    calibrated_ = !curve_.empty() && !m_.iso.empty();
    if (calibrated_) {
        nm_ = curve_.at((float)m_.iso[0]);
        noiseReady_ = true;
    }
    mo_ = MergeOpts();
    mo_.robustLo = o.robustLo > 0 ? o.robustLo : (calibrated_ ? kRobustCalibrated[0] : kRobustEstimated[0]);
    mo_.robustHi = o.robustHi > 0 ? o.robustHi : (calibrated_ ? kRobustCalibrated[1] : kRobustEstimated[1]);
    mo_.dilate = o.dilate;
    if (gpuWanted_) {
        std::string why;
        gpu_ = GpuBurst::create_stream(m_.width, m_.height, why);
        if (!gpu_) gpuNote_ = why;
    }
}

int BurstSession::add_frame(Raw f) {
    if (!streaming_ || streamed_) throw std::runtime_error("add_frame outside begin_stream / end_frames");
    if (zoom_ > 1.f) {
        Timer t;
        f = cropped(f);
        tCrop_ += t.ms();
    }
    streamAdded_++;
    if (ref_ >= 0) {
        stream_frame(f);
        return ref_;
    }
    frames_.push_back(std::move(f));
    if ((int)frames_.size() < kStreamCandidates) return -1;
    stream_start();
    return ref_;
}

void BurstSession::stream_start() {
    Timer t;
    ref_ = sharpest_frame(frames_, (int)frames_.size());
    streamRef_ = frames_[ref_];
    const float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    if (gpu_ && GpuBurst::enabled("align") && GpuBurst::enabled("merge")) {
        try {
            gpu_->stream_ref(streamRef_, black, noiseReady_ ? nullptr : &streamGrayRef_);
            streamGpu_ = true;
            gpuSteps_ += " align merge (streamed)";
        } catch (const std::exception& e) {
            gpu_failed("stream", e);
        }
    } else if (gpu_) {
        gpu_->drop_frames();
    }
    if (!streamGpu_) stream_cpu_start();
    tAlign_ += t.ms();
    for (int k = 0; k < (int)frames_.size(); k++)
        if (k != ref_) stream_frame(frames_[k]);
    frames_.clear();
}

void BurstSession::stream_cpu_start() {
    const float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    if (streamGrayRef_.w == 0) streamGrayRef_ = gray_from_raw(streamRef_);
    streamRefPyr_ = build_pyramid(streamGrayRef_);
    streamMean3Ref_ = box3(streamGrayRef_);
    streamMerge_ = std::make_unique<SpatialMerge>(streamRef_, black);
    streamMerged_ = 0;
}

void BurstSession::stream_frame(const Raw& f) {
    const float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    auto note = [&](const Offsets& of) {
        double sx = 0, sy = 0;
        for (size_t i = 0; i < of.dx.size(); i++) { sx += of.dx[i]; sy += of.dy[i]; }
        maxShift_ = std::max(maxShift_, std::hypot(sx / of.dx.size(), sy / of.dy.size()));
        streamMerged_++;
    };
    auto fitNoise = [&](const ImageF& gray, const Offsets& of) {
        if (noiseReady_) return;
        Timer t;
        nm_ = estimate_noise(streamGrayRef_, gray, of, black);
        noiseReady_ = true;
        tNoise_ = t.ms();
    };
    Timer t;
    ImageF gray;
    if (streamGpu_) {
        try {
            Offsets of = gpu_->stream_align(f, noiseReady_ ? nullptr : &gray);
            tAlign_ += t.ms();
            fitNoise(gray, of);
            t = Timer();
            gpu_->stream_merge(of, nm_, black, mo_);
            tMerge_ += t.ms();
            note(of);
            return;
        } catch (const std::exception& e) {
            // the frames merged on the device are lost; the CPU starts again from the reference
            gpu_failed("stream", e);
            streamGpu_ = false;
            stream_cpu_start();
            t = Timer();
        }
    }
    gray = gray_from_raw(f);
    Offsets of = align_frames(streamRefPyr_, build_pyramid(gray));
    tAlign_ += t.ms();
    fitNoise(gray, of);
    t = Timer();
    streamMerge_->add(f, of, box3(gray), streamMean3Ref_, nm_, 1.f, 1e30f, false, mo_);
    tMerge_ += t.ms();
    note(of);
}

int BurstSession::end_frames() {
    if (!streaming_ || streamed_) throw std::runtime_error("end_frames outside begin_stream");
    if (ref_ < 0) {
        if (frames_.empty()) throw std::runtime_error("no frames added");
        stream_start();
    }
    Timer t;
    const float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    if (streamGpu_) {
        try {
            meanWeight_ = gpu_->stream_end(black);
        } catch (const std::exception& e) {
            gpu_failed("stream", e);
            streamGpu_ = false;
            stream_cpu_start();
        }
    }
    if (!streamGpu_) {
        MergeStats ms;
        mosaic_ = streamMerge_->result(&ms);
        meanWeight_ = ms.meanWeight;
    }
    streamMerge_.reset();
    streamRef_ = Raw();
    streamRefPyr_.clear();
    streamMean3Ref_ = ImageF();
    streamGrayRef_ = ImageF();
    streamed_ = true;
    tMerge_ += t.ms();
    return ref_;
}

std::vector<uint8_t> BurstSession::encode_dng(const DngColor& c, const DngExtras& x) {
    if (kr_) return {};
    Timer t;
    ImageF fromGpu;
    const ImageF* mosaic = &mosaic_;
    if (mosaic_.w == 0) {
        if (!gpu_ || !gpu_->merged()) return {};
        fromGpu = gpu_->download_merged();
        mosaic = &fromGpu;
    }
    DngExtras ex = x;
    ex.baselineExposure = std::log2(refScale_);
    std::vector<uint8_t> file = ::encode_dng(*mosaic, m_, c, ex);
    tDng_ = t.ms();
    return file;
}

void BurstSession::finish(const LookParams& lp, bool keepHdr) {
    if (gpu_ && GpuBurst::enabled("finish")) {
        try {
            if (!gpu_->merged()) gpu_->set_merged(kr_ ? rgb_ : mosaic_, kr_);
            GpuBurst::FinishOut o;
            gpu_->finish(m_, lp, opt_.demosaic, zoom_, keepHdr, o);
            sigma_ = o.sigma;
            visibleSigma_ = o.visibleSigma;
            chroma_ = o.chroma;
            localContrast_ = o.localContrast;
            targetKey_ = o.targetKey;
            sharpenCore_ = o.sharpenCore;
            tUpscale_ = o.tUpscale;
            tone_ = o.tone;
            ft_ = o.ft;
            hdr_ = std::move(o.hdr);
            mosaic_ = ImageF();
            rgb_ = ImageF();
            outW_ = gpu_->width();
            outH_ = gpu_->height();
            gpuFinished_ = true;
            gpuSteps_ += " finish";
            gpuProfile_ = gpu_->profile();
            return;
        } catch (const std::exception& e) {
            if (mosaic_.w == 0 && rgb_.w == 0) {
                try {
                    (kr_ ? rgb_ : mosaic_) = gpu_->download_merged();
                } catch (const std::exception&) {
                    gpu_.reset();
                    throw;
                }
            }
            gpu_failed("finish", e);
        }
    } else if (gpu_ && gpu_->merged()) {
        (kr_ ? rgb_ : mosaic_) = gpu_->download_merged();
    }
    if (gpu_) gpuProfile_ = gpu_->profile();
    gpu_.reset();
    finish_cpu(lp, keepHdr);
    outW_ = rgb_.w;
    outH_ = rgb_.h;
}

void BurstSession::finish_cpu(const LookParams& look, bool keepHdr) {
    Timer t;
    LookParams lp = look;
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
    lp.targetKey = noise_limited_key(lp.targetKey, (float)(sigma_ * display_gain(rgb_, lp.targetKey)));
    targetKey_ = lp.targetKey;
    visibleSigma_ = sigma_ * display_gain(rgb_, lp.targetKey);
    if (chroma < 0) chroma = clampf((float)visibleSigma_ / 0.01f, 1.f, 6.f);
    chroma *= lp.chromaDenoiseScale;
    chroma_ = chroma;
    t = Timer(); chroma_denoise(rgb_, chroma); ft_.chroma = t.ms();
    if (zoom_ > 1.f) { t = Timer(); upscale_lanczos(rgb_, m_.full_w(), m_.full_h()); tUpscale_ = t.ms(); }
    hdr_ = keepHdr ? hdr_luma(rgb_, 1) : HdrLuma();
    t = Timer(); tone_ = local_tonemap(rgb_, lp); ft_.tonemap = t.ms();
    localContrast_ = noise_scaled_local_contrast(lp.localContrast, (float)visibleSigma_);
    t = Timer(); pixel_look(rgb_, lp); local_contrast(rgb_, localContrast_, lp.localRadius); hue_tune(rgb_, lp); ft_.look = t.ms();
    sharpenCore_ = sharpen_core((float)visibleSigma_);
    t = Timer(); sharpen_luma(rgb_, lp.sharpen, lp.sharpenRadius, (float)sharpenCore_); ft_.sharpen = t.ms();
}

void BurstSession::render(uint8_t* rgba, size_t stride) const {
    if (gpuFinished_ && gpu_) gpu_->render(rgba, stride);
    else to_rgba8888(rgb_, rgba, stride);
}

std::vector<uint8_t> BurstSession::encode_ultrahdr(int quality, float headroom) {
    if (hdr_.y.w == 0) return {};
    Timer t;
    if (gpuFinished_ && gpu_ && rgb_.w == 0) rgb_ = gpu_->download_rgb();
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
    if (streamed_) mode += " streamed " + std::to_string(streamMerged_ + 1) + "/" + std::to_string(streamAdded_);
    snprintf(buf, sizeof buf,
             "%s %dx%d ref=%d threads=%d | noise var=%.4f*S+%.3f %s robust %.2f-%.2f weight=%.3f shift=%.1f | sigma=%.5f visible=%.4f chroma=%.2f local=%.2f | "
             "tone key=%.4f target=%.3f short=%.2f long=%.2f core=%.4f | ms align %.0f noise %.0f merge %.0f lsc+wb %.0f demosaic %.0f ccm %.0f "
             "chroma %.0f tonemap %.0f look %.0f sharpen %.0f",
             mode.c_str(), m_.width, m_.height, ref_, g_threads, nm_.a, nm_.b, calibrated_ ? "calibrated" : "estimated", mo_.robustLo,
             mo_.robustHi, meanWeight_, maxShift_, sigma_,
             visibleSigma_, chroma_, localContrast_, tone_.keyBefore, targetKey_, tone_.gainShort, tone_.gainLong, sharpenCore_, tAlign_, tNoise_, tMerge_,
             ft_.lscwb, ft_.demosaic, ft_.color, ft_.chroma, ft_.tonemap, ft_.look, ft_.sharpen);
    std::string s = buf;
    if (zoom_ > 1.f) {
        snprintf(buf, sizeof buf, " | zoom %.2fx crop at (%d, %d) %.0f ms upscale to %dx%d %.0f ms", zoom_, m_.cropX, m_.cropY, tCrop_,
                 m_.full_w(), m_.full_h(), tUpscale_);
        s += buf;
    }
    if (tDng_ > 0) {
        snprintf(buf, sizeof buf, " | dng x%d %.0f ms", dng_scale(m_.white), tDng_);
        s += buf;
    }
    if (us_.bytes) {
        snprintf(buf, sizeof buf, " | ultrahdr max boost %.2f gain map %zu B file %zu B %.0f ms", us_.maxBoost, us_.gainMapBytes, us_.bytes, tUhdr_);
        s += buf;
    } else if (tUhdr_ > 0) {
        snprintf(buf, sizeof buf, " | ultrahdr skipped, max boost %.2f", us_.maxBoost);
        s += buf;
    }
    if (gpuWanted_) {
        s += " | gpu:" + (gpuSteps_.empty() ? std::string(" none") : gpuSteps_);
        if (!gpuNote_.empty()) s += " (" + gpuNote_ + ")";
        if (!gpuProfile_.empty()) s += " | gpu kernels " + gpuProfile_;
    }
    return s;
}
