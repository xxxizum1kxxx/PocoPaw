#include "burst.h"

int g_threads = (int)std::max(1u, std::thread::hardware_concurrency());

namespace {

// Immerkaer (1996) fast noise sigma estimate on the G_even sub-plane of a normalised mosaic
double noise_sigma(const ImageF& cfa, int cfaType) {
    int ox = 0, oy = 0;
    for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) if (cfa_channel(cfaType, x, y) == 1) { ox = x; oy = y; }
    int w = cfa.w / 2, h = cfa.h / 2;
    if (w <= 2 || h <= 2) return 0;
    std::vector<double> rowS(h, 0.0);
    std::vector<long> rowN(h, 0);
    parallel_for(h - 2, [&](int yi) {
        int y = yi + 1;
        double s = 0;
        long n = 0;
        for (int x = 1; x < w - 1; x++) {
            auto P = [&](int u, int v) { return cfa.at(2 * (x + u) + ox, 2 * (y + v) + oy); };
            double c = P(0, 0);
            if (c > 0.9) continue;
            double l = 4 * c - 2 * (P(-1, 0) + P(1, 0) + P(0, -1) + P(0, 1)) + (P(-1, -1) + P(1, -1) + P(-1, 1) + P(1, 1));
            s += std::fabs(l);
            n++;
        }
        rowS[y] = s;
        rowN[y] = n;
    });
    double s = 0;
    long n = 0;
    for (int y = 1; y < h - 1; y++) { s += rowS[y]; n += rowN[y]; }
    return n ? std::sqrt(M_PI / 2) / (6.0 * n) * s : 0;
}

// exposure gain the tone mapper will apply to reach the target key (log-average luma)
float display_gain(const ImageF& rgb, float targetKey) {
    int rows = (rgb.h + 7) / 8;
    if (rows <= 0 || rgb.w <= 0) return 1.f;
    std::vector<double> rowS(rows, 0.0);
    parallel_for(rows, [&](int r) {
        int y = r * 8;
        double ls = 0;
        for (int x = 0; x < rgb.w; x += 8) {
            const float* p = rgb.row(y) + (size_t)x * 3;
            ls += std::log(1e-4 + std::max(0.0, 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]));
        }
        rowS[r] = ls;
    });
    double s = 0;
    for (double v : rowS) s += v;
    long n = (long)rows * ((rgb.w + 7) / 8);
    return clampf(targetKey / (float)std::exp(s / std::max(1L, n)), 1.f, 64.f);
}

}  // namespace

BurstSession::BurstSession(const Meta& m, std::vector<Raw> frames, int ref)
    : m_(m), frames_(std::move(frames)), ref_(clampi(ref, 0, (int)frames_.size() - 1)) {}

void BurstSession::crop(float zoom) {
    if (zoom <= 1.f) return;
    ScopedBigCoreAffinity bigPin;
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

Raw BurstSession::cropped(const Raw& r) const {
    Raw c;
    c.w = m_.width;
    c.h = m_.height;
    c.own = std::make_shared<std::vector<uint16_t>>((size_t)c.w * c.h);
    parallel_for(c.h, [&](int y) {
        const uint16_t* s = r.d + (size_t)(cropY0_ + y) * r.w + cropX0_;
        std::copy(s, s + c.w, c.own->data() + (size_t)y * c.w);
    });
    c.d = c.own->data();
    return c;
}

void BurstSession::align() {
    ScopedBigCoreAffinity bigPin;
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
    calibrated_ = !curve_.empty() && (int)m_.iso.size() > ref_;
    int alt = -1;
    if (n > 1 && !calibrated_) {
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
    if (calibrated_) nm_ = curve_.at((float)m_.iso[ref_], refScale_);
    else if (n > 1) {
        if (alt < 0) alt = ref_ == 0 ? 1 : 0;
        if (gray_[alt].w == 0) gray_[alt] = gray_from_raw(frames_[alt]);
        nm_ = estimate_noise(gray_[ref_], gray_[alt], offs_[alt], black);
    }
    tNoise_ = t.ms();
}

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
    ScopedBigCoreAffinity bigPin;
    if (streamed_) {
        opt_.demosaic = Demosaic::RCD;
        for (int i = 0; i < 2; i++) { opt_.lcaR[i] = o.lcaR[i]; opt_.lcaB[i] = o.lcaB[i]; }
        return;
    }
    Timer t;
    int n = (int)frames_.size();
    float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    opt_ = o;
    opt_.kernelRegression = false;
    opt_.freqMerge = false;
    opt_.demosaic = Demosaic::RCD;
    mo_.robustLo = o.robustLo > 0 ? o.robustLo : (calibrated_ ? kRobustCalibrated[0] : kRobustEstimated[0]);
    mo_.robustHi = o.robustHi > 0 ? o.robustHi : (calibrated_ ? kRobustCalibrated[1] : kRobustEstimated[1]);
    mo_.dilate = o.dilate;

    bool onGpu = false;
    if (gpu_ && GpuBurst::enabled("merge")) {
        try {
            if (!gpu_->aligned()) {
                cpu_align_state();
                gpu_->set_aligned(mean3_, gray_[ref_]);
            }
            meanWeight_ = gpu_->merge_spatial(ref_, offs_, nm_, black, mo_);
            gpu_->drop_frames();
            onGpu = true;
            gpuSteps_ += " merge";
        } catch (const std::exception& e) {
            gpu_failed("merge", e);
        }
    }
    if (!onGpu) {
        if (n > 1) {
            cpu_align_state();
            MergeStats ms;
            mosaic_ = merge_frames(frames_, ref_, offs_, mean3_, nm_, black, mo_, &ms);
            meanWeight_ = ms.meanWeight;
        } else {
            mosaic_ = ImageF(m_.width, m_.height);
            const uint16_t* src = frames_[ref_].d;
            parallel_for(m_.height, [&](int y) {
                const uint16_t* sr = src + (size_t)y * m_.width;
                float* dr = mosaic_.row(y);
                for (int x = 0; x < m_.width; x++) dr[x] = sr[x];
            });
        }
    }
    frames_.clear();
    gray_.clear();
    mean3_.clear();
    offs_.clear();
    tMerge_ = t.ms();
}

void BurstSession::begin_stream(const BurstOptions& o) {
    ScopedBigCoreAffinity bigPin;
    streaming_ = true;
    opt_ = o;
    opt_.kernelRegression = false;
    opt_.freqMerge = false;
    opt_.demosaic = Demosaic::RCD;
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
    ScopedBigCoreAffinity bigPin;
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
            if (noiseReady_) {
                gpu_->stream_align(f, nullptr, nullptr);
                tAlign_ += t.ms();
                t = Timer();
                gpu_->stream_merge(nm_, black, mo_);
                tMerge_ += t.ms();
                streamMerged_++;
            } else {
                Offsets of;
                gpu_->stream_align(f, &of, &gray);
                tAlign_ += t.ms();
                fitNoise(gray, of);
                t = Timer();
                gpu_->stream_merge(nm_, black, mo_);
                tMerge_ += t.ms();
                note(of);
            }
            return;
        } catch (const std::exception& e) {
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
    ScopedBigCoreAffinity bigPin;
    if (!streaming_ || streamed_) throw std::runtime_error("end_frames outside begin_stream");
    if (ref_ < 0) {
        if (frames_.empty()) throw std::runtime_error("no frames added");
        stream_start();
    }
    Timer t;
    const float black = (m_.black[0] + m_.black[1] + m_.black[2] + m_.black[3]) / 4;
    if (streamGpu_) {
        try {
            meanWeight_ = gpu_->stream_end(black, &maxShift_);
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
    ScopedBigCoreAffinity bigPin;
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
    ScopedBigCoreAffinity bigPin;
    if (gpu_ && GpuBurst::enabled("finish")) {
        try {
            if (!gpu_->merged()) gpu_->set_merged(mosaic_);
            GpuBurst::FinishOut o;
            gpu_->finish(m_, lp, Demosaic::RCD, zoom_, keepHdr, o, aspect_, opt_.lcaR, opt_.lcaB);
            sigma_ = o.sigma;
            visibleSigma_ = o.visibleSigma;
            chroma_ = o.chroma;
            localContrast_ = 0.0;
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
            if (Gpu::get() && Gpu::get()->profiling()) gpuProfile_ = gpu_->profile();
            return;
        } catch (const std::exception& e) {
            if (mosaic_.w == 0) {
                try {
                    mosaic_ = gpu_->download_merged();
                } catch (const std::exception&) {
                    gpu_.reset();
                    throw;
                }
            }
            gpu_failed("finish", e);
        }
    } else if (gpu_ && gpu_->merged()) {
        mosaic_ = gpu_->download_merged();
    }
    if (gpu_ && Gpu::get() && Gpu::get()->profiling()) gpuProfile_ = gpu_->profile();
    gpu_.reset();
    finish_cpu(lp, keepHdr);
    outW_ = rgb_.w;
    outH_ = rgb_.h;
}

void BurstSession::finish_cpu(const LookParams& look, bool keepHdr) {
    Timer t;
    LookParams lp = look;
    ImageF cfa = normalize_mosaic(mosaic_, m_, lp);
    ft_.lscwb = t.ms();
    sigma_ = noise_sigma(cfa, m_.cfa);
    mosaic_ = ImageF();
    t = Timer();
    rgb_ = demosaic(cfa, m_.cfa, Demosaic::RCD);
    cfa = ImageF();
    correct_lca(rgb_, m_, opt_.lcaR, opt_.lcaB);
    ft_.demosaic = t.ms();
    t = Timer(); apply_ccm(rgb_, m_.ccm); ft_.color = t.ms();
    float chroma = lp.chromaDenoise;
    float dgain0 = display_gain(rgb_, lp.targetKey);
    lp.targetKey = noise_limited_key(lp.targetKey, (float)(sigma_ * dgain0));
    targetKey_ = lp.targetKey;
    visibleSigma_ = sigma_ * clampf(dgain0 * (lp.targetKey / std::max(1e-6f, look.targetKey)), 1.f, 64.f);
    if (chroma < 0) chroma = clampf((float)visibleSigma_ / 0.01f, 1.f, 6.f);
    chroma *= lp.chromaDenoiseScale;
    chroma_ = chroma;
    t = Timer(); chroma_denoise(rgb_, chroma); ft_.chroma = t.ms();
    if (zoom_ > 1.f) { t = Timer(); upscale_lanczos(rgb_, m_.full_w(), m_.full_h()); tUpscale_ = t.ms(); }
    if (aspect_ > 0.f) crop_aspect(rgb_, aspect_);
    hdr_ = keepHdr ? hdr_luma(rgb_, 1) : HdrLuma();
    t = Timer(); tone_ = local_tonemap(rgb_, lp); ft_.tonemap = t.ms();
    localContrast_ = 0.0;
    t = Timer(); pixel_look(rgb_, lp); ft_.look = t.ms();
    sharpenCore_ = sharpen_core((float)visibleSigma_);
    t = Timer(); sharpen_luma(rgb_, lp.sharpen, lp.sharpenRadius, (float)sharpenCore_); ft_.sharpen = t.ms();
}

void BurstSession::render(uint8_t* rgba, size_t stride) const {
    ScopedBigCoreAffinity bigPin;
    if (gpuFinished_ && gpu_) gpu_->render(rgba, stride);
    else to_rgba8888(rgb_, rgba, stride);
}

void BurstSession::render_thumb(uint8_t* rgba, size_t stride, int outW, int outH) const {
    ScopedBigCoreAffinity bigPin;
    if (gpuFinished_ && gpu_) gpu_->render_thumb(rgba, stride, outW, outH);
    else to_rgba8888_thumb(rgb_, rgba, stride, outW, outH);
}

std::vector<uint8_t> BurstSession::encode_photo(int quality, bool wantUltraHdr, float headroom, bool* isUltraHdr) {
    ScopedBigCoreAffinity bigPin;
    if (isUltraHdr) *isUltraHdr = false;
    int w = outW_, h = outH_;
    if (w <= 0 || h <= 0) return {};
    const uint8_t* px = nullptr;
    std::vector<uint8_t> cpuPx;
    if (gpuFinished_ && gpu_ && gpu_->rgb8_ptr() != nullptr) {
        px = gpu_->rgb8_ptr();
    } else {
        cpuPx = to_rgb8(rgb_, 1, w, h);
        px = cpuPx.empty() ? nullptr : cpuPx.data();
    }
    if (!px) return {};
    std::vector<uint8_t> base = ::encode_jpeg(px, w, h, 3, quality, true);
    if (wantUltraHdr && hdr_.y.w > 0 && !base.empty()) {
        Timer t;
        std::vector<uint8_t> uhdr = encode_ultrahdr_rgb8(px, w, h, base, hdr_, tone_.gainLong, headroom, &us_);
        tUhdr_ = t.ms();
        if (!uhdr.empty()) {
            if (isUltraHdr) *isUltraHdr = true;
            return uhdr;
        }
    }
    return base;
}

std::vector<uint8_t> BurstSession::encode_ultrahdr(int quality, float headroom) {
    ScopedBigCoreAffinity bigPin;
    if (hdr_.y.w == 0) return {};
    Timer t;
    if (gpuFinished_ && gpu_ && rgb_.w == 0) rgb_ = gpu_->download_rgb();
    std::vector<uint8_t> file = ::encode_ultrahdr(rgb_, 1, hdr_, tone_.gainLong, headroom, quality, &us_);
    tUhdr_ = t.ms();
    return file;
}

std::string BurstSession::summary() const {
    char buf[1024];
    std::string mode = "indigo/spatial/rcd";
    if (bracket_) {
        mode += " bracket gains";
        for (float g : mo_.gain) { char b[16]; snprintf(b, sizeof b, " %.2f", g); mode += b; }
    }
    if (opt_.lcaR[0] != 0 || opt_.lcaR[1] != 0 || opt_.lcaB[0] != 0 || opt_.lcaB[1] != 0) mode += " lca";
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
    if (aspect_ > 0.f && outW_ > 0 && outH_ > 0) {
        snprintf(buf, sizeof buf, " | aspect %dx%d", outW_, outH_);
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
