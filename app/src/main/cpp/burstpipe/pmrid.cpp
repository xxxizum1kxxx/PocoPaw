#include "pmrid.h"

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "ort_runtime.h"

namespace {

constexpr float kAnchorK = 0.96793133f;
constexpr float kAnchorS = 2.97945257f;
constexpr float kRangeV = 959.0f;
constexpr float kInpScale = 256.0f;

bool file_readable(const std::string& p) {
    if (p.empty()) return false;
    struct stat st{};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

inline int reflect_idx(int i, int n) {
    if (n <= 1) return 0;
    const int period = 2 * (n - 1);
    const int r = i % period;
    return r < n ? r : period - r;
}

float interp_alpha_knots(float kEff) {
    static constexpr float kLogX[3] = {-1.1f, -0.5f, 0.0f};
    static constexpr float kAlphaY[3] = {0.90f, 0.95f, 0.97f};
    const float lx = std::log10(std::max(1e-6f, kEff));
    if (lx <= kLogX[0]) return kAlphaY[0];
    if (lx >= kLogX[2]) return kAlphaY[2];
    if (lx <= kLogX[1]) {
        const float t = (lx - kLogX[0]) / (kLogX[1] - kLogX[0]);
        return kAlphaY[0] + t * (kAlphaY[1] - kAlphaY[0]);
    }
    const float t = (lx - kLogX[1]) / (kLogX[2] - kLogX[1]);
    return kAlphaY[1] + t * (kAlphaY[2] - kAlphaY[1]);
}

struct CachedSession {
    std::mutex mu;
    std::string key;
    std::unique_ptr<OrtSessionRunner> runner;
};

CachedSession& pmrid_cache() {
    // Process-lifetime heap singleton: avoids atexit ReleaseSession after dlopen'd libonnxruntime teardown.
    static auto* c = new CachedSession();
    return *c;
}

}  // namespace

float pmrid_blend_alpha(float kEff, float lowNoiseCutoff) {
    if (!is_finite_f32_val(kEff) || kEff <= 0.f || !is_finite_f32_val(lowNoiseCutoff)) return 0.f;
    float alpha = interp_alpha_knots(kEff);
    if (lowNoiseCutoff > 0.f) {
        const float lo = 0.5f * lowNoiseCutoff;
        const float hi = lowNoiseCutoff;
        const float t = clampf((kEff - lo) / std::max(1e-6f, hi - lo), 0.f, 1.f);
        const float gate = t * t * (3.f - 2.f * t);
        alpha *= gate;
    }
    return is_finite_f32_val(alpha) ? alpha : 0.f;
}

bool pmrid_should_run(
    const Meta& m,
    const NoiseModel& nm,
    int frameCount,
    double meanWeight,
    float lowNoiseCutoff,
    float* outKEff,
    float* outBEff,
    float* outNeff,
    float* outAlpha) {
    auto fail = [&]() {
        if (outKEff) *outKEff = 0.f;
        if (outBEff) *outBEff = 0.f;
        if (outNeff) *outNeff = 1.f;
        if (outAlpha) *outAlpha = 0.f;
        return false;
    };
    if (!is_finite_f64_val(nm.a) || !is_finite_f64_val(nm.b) || !is_finite_f64_val(meanWeight) ||
        !is_finite_f32_val(lowNoiseCutoff) || !is_finite_f32_val(m.white) ||
        !is_finite_f32_val(m.black[0]) || !is_finite_f32_val(m.black[1]) ||
        !is_finite_f32_val(m.black[2]) || !is_finite_f32_val(m.black[3]) ||
        nm.a < 0.0 || nm.b < 0.0) {
        return fail();
    }
    const double black = (static_cast<double>(m.black[0]) + m.black[1] + m.black[2] + m.black[3]) * 0.25;
    const double whiteD = static_cast<double>(m.white);
    if (!is_finite_f64_val(black) || !is_finite_f64_val(whiteD) || whiteD <= black) {
        return fail();
    }
    const double dynRange = whiteD - black;
    if (!is_finite_f64_val(dynRange) || dynRange <= 0.0) {
        return fail();
    }
    const double dynScaleD = static_cast<double>(kRangeV) / dynRange;
    const float dynScale = static_cast<float>(dynScaleD);
    const double neff = 1.0 + std::max(0, frameCount - 1) * std::max(0.0, meanWeight);
    const float neffF = static_cast<float>(neff);
    if (!is_finite_f64_val(dynScaleD) ||
        !is_finite_f32_val(dynScale) || dynScale <= 0.f ||
        !is_finite_f64_val(neff) || neff <= 0.0 || !is_finite_f32_val(neffF)) {
        return fail();
    }
    const float kEff = static_cast<float>(std::max(1e-6, nm.a * dynScaleD / neff));
    const float bEff = static_cast<float>(std::max(0.0, nm.b * dynScaleD * dynScaleD / neff));
    if (!is_finite_f32_val(kEff) || !is_finite_f32_val(bEff)) {
        return fail();
    }
    const float cvtK = kAnchorK / kEff;
    const float cvtB = (bEff / (kEff * kEff) - kAnchorS / (kAnchorK * kAnchorK)) * kAnchorK;
    const float hi = kRangeV * cvtK + cvtB;
    if (!is_finite_f32_val(cvtK) || cvtK <= 0.f || !is_finite_f32_val(cvtB) || !is_finite_f32_val(hi)) {
        return fail();
    }
    const float alpha = pmrid_blend_alpha(kEff, lowNoiseCutoff);
    if (!is_finite_f32_val(alpha) || alpha <= 0.f) {
        if (outKEff) *outKEff = kEff;
        if (outBEff) *outBEff = bEff;
        if (outNeff) *outNeff = neffF;
        if (outAlpha) *outAlpha = 0.f;
        return false;
    }
    if (outKEff) *outKEff = kEff;
    if (outBEff) *outBEff = bEff;
    if (outNeff) *outNeff = neffF;
    if (outAlpha) *outAlpha = alpha;
    return true;
}

std::string resolve_pmrid_model(const std::string& explicitPath, const char* argv0) {
    if (!explicitPath.empty()) {
        return file_readable(explicitPath) ? explicitPath : std::string();
    }
    std::vector<std::string> candidates = {
        "calib/pmrid.onnx",
        "assets/models/pmrid.onnx",
        "models/pmrid.onnx",
        "../PocoPaw/app/src/main/assets/models/pmrid.onnx",
        "../../PocoPaw/app/src/main/assets/models/pmrid.onnx",
        "app/src/main/assets/models/pmrid.onnx",
    };
    if (argv0 && *argv0) {
        std::string exe(argv0);
        const size_t slash = exe.rfind('/');
        const std::string exeDir = slash == std::string::npos ? "." : exe.substr(0, slash);
        candidates.push_back(exeDir + "/pmrid.onnx");
        candidates.push_back(exeDir + "/../calib/pmrid.onnx");
        candidates.push_back(exeDir + "/../../PocoPaw/app/src/main/assets/models/pmrid.onnx");
    }
    for (const auto& p : candidates) {
        if (file_readable(p)) return p;
    }
    return {};
}

bool pmrid_denoise_mosaic(
    ImageF& mosaic,
    const Meta& m,
    const NoiseModel& nm,
    int frameCount,
    double meanWeight,
    float lowNoiseCutoff,
    const std::string& modelPath,
    AccelBackend backend,
    PmridStats* st,
    const char* argv0) {
    Timer timer;
    PmridStats localStats;

    if (mosaic.w < 2 || mosaic.h < 2 || mosaic.c != 1) {
        localStats.note = "invalid mosaic dimensions";
        if (st) *st = localStats;
        return false;
    }

    float kEff = 0.f, bEff = 0.f, neff = 1.f, alpha = 0.f;
    const bool active = pmrid_should_run(m, nm, frameCount, meanWeight, lowNoiseCutoff, &kEff, &bEff, &neff, &alpha);
    localStats.kEff = kEff;
    localStats.bEff = bEff;
    localStats.neff = neff;
    localStats.alpha = alpha;

    if (!active) {
        localStats.note = "gated (low noise)";
        localStats.ms = timer.ms();
        if (st) *st = localStats;
        return false;
    }

    {
        std::atomic<bool> inOk{true};
        parallel_for(mosaic.h, [&](int y) {
            if (!all_finite_f32(mosaic.row(y), (size_t)mosaic.w)) {
                inOk.store(false, std::memory_order_relaxed);
            }
        });
        if (!inOk.load(std::memory_order_relaxed)) {
            localStats.note = "non-finite input mosaic";
            localStats.ms = timer.ms();
            if (st) *st = localStats;
            return false;
        }
    }

    const std::string resolved = resolve_pmrid_model(modelPath, argv0);
    if (resolved.empty()) {
        localStats.note = modelPath.empty() ? "pmrid.onnx not found" : ("model not found: " + modelPath);
        localStats.ms = timer.ms();
        if (st) *st = localStats;
        return false;
    }

    const double blackD = (static_cast<double>(m.black[0]) + m.black[1] + m.black[2] + m.black[3]) * 0.25;
    const double whiteD = static_cast<double>(m.white);
    if (!is_finite_f64_val(blackD) || !is_finite_f64_val(whiteD) || whiteD <= blackD) {
        localStats.note = "non-finite dynamic range";
        localStats.ms = timer.ms();
        if (st) *st = localStats;
        return false;
    }
    const double dynRangeD = whiteD - blackD;
    const double dynScaleD = static_cast<double>(kRangeV) / dynRangeD;
    const double invDynScaleD = dynRangeD / static_cast<double>(kRangeV);
    const float black = static_cast<float>(blackD);
    const float dynScale = static_cast<float>(dynScaleD);
    if (!is_finite_f64_val(dynRangeD) || dynRangeD <= 0.0 ||
        !is_finite_f64_val(dynScaleD) || !is_finite_f64_val(invDynScaleD) || invDynScaleD <= 0.0 ||
        !is_finite_f32_val(black) || !is_finite_f32_val(dynScale) || dynScale <= 0.f) {
        localStats.note = "non-finite dynamic range";
        localStats.ms = timer.ms();
        if (st) *st = localStats;
        return false;
    }

    const int H = mosaic.h & ~1;
    const int W = mosaic.w & ~1;
    const int hh = H / 2;
    const int hw = W / 2;
    const int Hp = ((hh + 31) / 32) * 32;
    const int Wp = ((hw + 31) / 32) * 32;

    const float cvtK = kAnchorK / kEff;
    const float cvtB = (bEff / (kEff * kEff) - kAnchorS / (kAnchorK * kAnchorK)) * kAnchorK;
    const float lo = cvtB;
    const float hi = kRangeV * cvtK + cvtB;
    if (!is_finite_f32_val(cvtK) || cvtK <= 0.f || !is_finite_f32_val(cvtB) || !is_finite_f32_val(hi)) {
        localStats.note = "non-finite K-Sigma transform";
        localStats.ms = timer.ms();
        if (st) *st = localStats;
        return false;
    }
    const float invVScale = kInpScale / kRangeV;
    const int cfa = m.cfa;

    FloatBuf inp((size_t)4 * Hp * Wp);
    const size_t planeStride = (size_t)Hp * Wp;

    parallel_for(Hp, [&](int py) {
        const int syBase = reflect_idx(py, hh);
        for (int px = 0; px < Wp; px++) {
            const int sxBase = reflect_idx(px, hw);
            for (int sy2 = 0; sy2 < 2; sy2++) {
                const int ry = 2 * syBase + sy2;
                const int my = (cfa == 2 || cfa == 3) ? (H - 1 - ry) : ry;
                for (int sx2 = 0; sx2 < 2; sx2++) {
                    const int rx = 2 * sxBase + sx2;
                    const int mx = (cfa == 1 || cfa == 3) ? (W - 1 - rx) : rx;
                    const int c = sy2 * 2 + sx2;
                    const double xDnD = (static_cast<double>(mosaic.at(mx, my)) - blackD) * dynScaleD;
                    const float xDn = (is_finite_f64_val(xDnD) && std::abs(xDnD) <= 1e30)
                        ? static_cast<float>(xDnD)
                        : (xDnD < 0.0 ? -1e30f : 1e30f);
                    const float ks = clampf(xDn * cvtK + cvtB, lo, hi);
                    inp[(size_t)c * planeStride + (size_t)py * Wp + px] = ks * invVScale;
                }
            }
        }
    });

    std::vector<float> out;
    std::vector<int64_t> outShape;
    std::string ortErr;
    {
        CachedSession& cache = pmrid_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        const std::string cacheKey = resolved + "#" + std::to_string(static_cast<int>(backend));
        if (!cache.runner || cache.key != cacheKey) {
            cache.runner = OrtSessionRunner::create(resolved, backend, 4, &ortErr);
            cache.key = cache.runner ? cacheKey : std::string();
        }
        if (!cache.runner) {
            localStats.note = ortErr.empty() ? "ORT session init failed" : ortErr;
            localStats.ms = timer.ms();
            if (st) *st = localStats;
            return false;
        }
        localStats.ep = cache.runner->active_ep();
        const int64_t inShape[4] = {1, 4, Hp, Wp};
        if (!cache.runner->run(inp.data(), inShape, 4, out, outShape, &ortErr)) {
            localStats.note = ortErr.empty() ? "ORT inference failed" : ortErr;
            localStats.ms = timer.ms();
            if (st) *st = localStats;
            return false;
        }
    }

    if (outShape.size() != 4 || outShape[0] != 1 || outShape[1] != 4 || outShape[2] != Hp || outShape[3] != Wp ||
        out.size() != (size_t)4 * planeStride) {
        localStats.note = "unexpected PMRID output tensor shape";
        localStats.ms = timer.ms();
        if (st) *st = localStats;
        return false;
    }

    {
        std::atomic<bool> outOk{true};
        parallel_for(4 * Hp, [&](int r) {
            if (!all_finite_f32(out.data() + (size_t)r * Wp, (size_t)Wp)) {
                outOk.store(false, std::memory_order_relaxed);
            }
        });
        if (!outOk.load(std::memory_order_relaxed)) {
            localStats.note = "non-finite PMRID output";
            localStats.ms = timer.ms();
            if (st) *st = localStats;
            return false;
        }
    }

    const float outToDnScale = kRangeV / kInpScale;
    constexpr double kMaxF32 = static_cast<double>(std::numeric_limits<float>::max());
    parallel_for(hh, [&](int py) {
        for (int px = 0; px < hw; px++) {
            for (int sy2 = 0; sy2 < 2; sy2++) {
                const int ry = 2 * py + sy2;
                const int my = (cfa == 2 || cfa == 3) ? (H - 1 - ry) : ry;
                for (int sx2 = 0; sx2 < 2; sx2++) {
                    const int rx = 2 * px + sx2;
                    const int mx = (cfa == 1 || cfa == 3) ? (W - 1 - rx) : rx;
                    const int c = sy2 * 2 + sx2;
                    const double xDnD = (static_cast<double>(mosaic.at(mx, my)) - blackD) * dynScaleD;
                    const float xDn = (is_finite_f64_val(xDnD) && std::abs(xDnD) <= 1e30)
                        ? static_cast<float>(xDnD)
                        : (xDnD < 0.0 ? -1e30f : 1e30f);
                    const float rawOut = out[(size_t)c * planeStride + (size_t)py * Wp + px];
                    const float predDn = std::min((rawOut * outToDnScale - cvtB) / cvtK, kRangeV);
                    const float blended = alpha * predDn + (1.f - alpha) * xDn;
                    const double outValD = static_cast<double>(blended) * invDynScaleD + blackD;
                    const float outVal = is_finite_f64_val(outValD)
                        ? static_cast<float>(std::max(-kMaxF32, std::min(kMaxF32, outValD)))
                        : mosaic.at(mx, my);
                    mosaic.at(mx, my) = is_finite_f32_val(outVal) ? outVal : mosaic.at(mx, my);
                }
            }
        }
    });

    localStats.ran = true;
    localStats.ms = timer.ms();
    if (st) *st = localStats;
    return true;
}
