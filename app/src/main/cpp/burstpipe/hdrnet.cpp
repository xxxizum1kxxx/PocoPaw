#include "hdrnet.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>

#include "ort_runtime.h"

namespace {

constexpr float kHdrnetMu = 128.f;
constexpr float kHdrnetMax = 16.f;
constexpr float kHdrnetInvLogMax = 1.f / 7.62510714f;  // 1 / ln(1 + 128 * 16) = 1 / ln(2049)

inline float hdrnet_encode(float c) {
    return std::log1p(kHdrnetMu * clampf(c, 0.f, kHdrnetMax)) * kHdrnetInvLogMax;
}

inline float srgb_eotf_unclamped_cpu(float y) {
    if (y <= 0.04045f) return y / 12.92f;
    return std::pow((y + 0.055f) / 1.055f, 2.4f);
}

bool file_readable(const std::string& p) {
    if (p.empty()) return false;
    struct stat st{};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

struct CachedSession {
    std::mutex mu;
    std::string key;
    std::unique_ptr<OrtSessionRunner> runner;
};

CachedSession& hdrnet_cache() {
    // Process-lifetime heap singleton: avoids atexit teardown ordering issues against dlopen'd ORT.
    static auto* c = new CachedSession();
    return *c;
}

void fill_default_guide(std::vector<float>& guide) {
    guide.assign(67, 0.f);
    guide[0] = 1.f;
    guide[5] = 1.f;
    guide[10] = 1.f;
    for (int m = 0; m < 3; m++) {
        for (int k = 0; k < 17; k++) {
            guide[12 + m * 17 + k] = k / 16.f;
        }
    }
    guide[63] = 0.2126f;
    guide[64] = 0.7152f;
    guide[65] = 0.0722f;
    guide[66] = 0.f;
}

inline float pwl17(const float* t, float v) {
    const float f = clampf(v, 0.f, 1.f) * 16.f;
    const int k = std::min(static_cast<int>(f), 15);
    const float a = f - k;
    return t[k] + a * (t[k + 1] - t[k]);
}

}  // namespace

std::string resolve_hdrnet_model(const std::string& explicitPath, const char* argv0) {
    if (!explicitPath.empty()) {
        return file_readable(explicitPath) ? explicitPath : std::string();
    }
    std::vector<std::string> candidates = {
        "calib/hdrnet.onnx",
        "assets/models/hdrnet.onnx",
        "models/hdrnet.onnx",
        "../PocoPaw/app/src/main/assets/models/hdrnet.onnx",
        "../../PocoPaw/app/src/main/assets/models/hdrnet.onnx",
        "app/src/main/assets/models/hdrnet.onnx",
    };
    if (argv0 && *argv0) {
        std::string exe(argv0);
        const size_t slash = exe.rfind('/');
        const std::string exeDir = slash == std::string::npos ? "." : exe.substr(0, slash);
        candidates.push_back(exeDir + "/hdrnet.onnx");
        candidates.push_back(exeDir + "/../calib/hdrnet.onnx");
        candidates.push_back(exeDir + "/../../PocoPaw/app/src/main/assets/models/hdrnet.onnx");
    }
    for (const auto& p : candidates) {
        if (file_readable(p)) return p;
    }
    return {};
}

void hdrnet_downsample_cpu(const ImageF& rgb, float* nchw256) {
    if (rgb.w <= 0 || rgb.h <= 0 || !nchw256) return;
    constexpr int kS = 256;
    const size_t plane = (size_t)kS * kS;
    const float sxScale = (float)rgb.w / (float)kS;
    const float syScale = (float)rgb.h / (float)kS;
    parallel_for(kS, [&](int y) {
        const float fy = clampf((y + 0.5f) * syScale - 0.5f, 0.f, (float)std::max(0, rgb.h - 1));
        const int y0 = (int)fy;
        const int y1 = std::min(y0 + 1, rgb.h - 1);
        const float ay = fy - y0;
        const float* row0 = rgb.row(y0);
        const float* row1 = rgb.row(y1);
        for (int x = 0; x < kS; x++) {
            const float fx = clampf((x + 0.5f) * sxScale - 0.5f, 0.f, (float)std::max(0, rgb.w - 1));
            const int x0 = (int)fx;
            const int x1 = std::min(x0 + 1, rgb.w - 1);
            const float ax = fx - x0;
            const float* p00 = row0 + (size_t)x0 * 3;
            const float* p10 = row0 + (size_t)x1 * 3;
            const float* p01 = row1 + (size_t)x0 * 3;
            const float* p11 = row1 + (size_t)x1 * 3;
            const size_t idx = (size_t)y * kS + x;
            for (int c = 0; c < 3; c++) {
                const float u00 = hdrnet_encode(p00[c]);
                const float u10 = hdrnet_encode(p10[c]);
                const float u01 = hdrnet_encode(p01[c]);
                const float u11 = hdrnet_encode(p11[c]);
                const float v = (1.f - ay) * ((1.f - ax) * u00 + ax * u10) +
                                ay * ((1.f - ax) * u01 + ax * u11);
                nchw256[(size_t)c * plane + idx] = v;
            }
        }
    });
}

bool hdrnet_predict(
    const float* nchw256,
    const std::string& modelPath,
    AccelBackend backend,
    std::vector<float>& gridOut,
    std::vector<float>& guideOut,
    HdrnetStats* st,
    const char* argv0) {
    Timer timer;
    HdrnetStats local;
    if (!nchw256) {
        local.note = "null HDRnet input";
        if (st) *st = local;
        return false;
    }

    const std::string resolved = resolve_hdrnet_model(modelPath, argv0);
    if (resolved.empty()) {
        local.note = modelPath.empty() ? "hdrnet.onnx not found" : ("model not found: " + modelPath);
        if (st) *st = local;
        return false;
    }

    std::vector<std::vector<float>> outputsData;
    std::vector<std::vector<int64_t>> outputsShape;
    std::string ortErr;
    {
        CachedSession& cache = hdrnet_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        const std::string cacheKey = resolved + "#" + std::to_string(static_cast<int>(backend));
        if (!cache.runner || cache.key != cacheKey) {
            cache.runner = OrtSessionRunner::create(resolved, backend, 4, &ortErr);
            cache.key = cache.runner ? cacheKey : std::string();
        }
        if (!cache.runner) {
            local.note = ortErr.empty() ? "ORT session init failed" : ortErr;
            local.ortMs = timer.ms();
            if (st) *st = local;
            return false;
        }
        local.ep = cache.runner->active_ep();
        const int64_t inShape[4] = {1, 3, 256, 256};
        if (!cache.runner->run_multi(nchw256, inShape, 4, outputsData, outputsShape, &ortErr)) {
            local.note = ortErr.empty() ? "ORT inference failed" : ortErr;
            local.ortMs = timer.ms();
            if (st) *st = local;
            return false;
        }
    }
    local.ortMs = timer.ms();

    if (outputsData.empty() || outputsData[0].size() != 16u * 16u * 8u * 12u) {
        local.note = "unexpected HDRnet grid shape";
        if (st) *st = local;
        return false;
    }
    if (!all_finite_f32(outputsData[0].data(), outputsData[0].size())) {
        local.note = "non-finite HDRnet grid";
        if (st) *st = local;
        return false;
    }
    gridOut = std::move(outputsData[0]);

    bool guideOk = outputsData.size() >= 2 && outputsData[1].size() == 67u &&
                   all_finite_f32(outputsData[1].data(), outputsData[1].size());
    if (guideOk) {
        guideOut = std::move(outputsData[1]);
    } else {
        fill_default_guide(guideOut);
    }

    local.ran = true;
    if (st) *st = local;
    return true;
}

bool hdrnet_tonemap(
    ImageF& rgb,
    const LookParams& lp,
    AccelBackend backend,
    ToneInfo& outTone,
    HdrnetStats* st,
    const char* argv0) {
    if (rgb.w <= 0 || rgb.h <= 0 || rgb.c != 3) return false;
    outTone = compute_tone_info(rgb, lp);

    std::vector<float> nchw((size_t)3 * 256 * 256);
    hdrnet_downsample_cpu(rgb, nchw.data());

    std::vector<float> grid, guide;
    HdrnetStats local;
    if (!hdrnet_predict(nchw.data(), lp.hdrnetModelPath, backend, grid, guide, &local, argv0)) {
        if (st) *st = local;
        return false;
    }

    Timer sliceTimer;
    const int W = rgb.w, H = rgb.h;
    const float keyScale = clampf(lp.targetKey / 0.22f, 0.35f, 1.25f);
    const float* gPtr = grid.data();
    const float* gdPtr = guide.data();

    parallel_for(H, [&](int y) {
        const float fy = clampf((y + 0.5f) * (16.f / H) - 0.5f, 0.f, 15.f);
        const int y0 = std::min(static_cast<int>(fy), 14), y1 = y0 + 1;
        const float ay = fy - y0;
        float* q = rgb.row(y);
        for (int x = 0; x < W; x++, q += 3) {
            const float ur = hdrnet_encode(q[0]);
            const float ug = hdrnet_encode(q[1]);
            const float ub = hdrnet_encode(q[2]);

            const float c0 = gdPtr[0] * ur + gdPtr[1] * ug + gdPtr[2] * ub + gdPtr[3];
            const float c1 = gdPtr[4] * ur + gdPtr[5] * ug + gdPtr[6] * ub + gdPtr[7];
            const float c2 = gdPtr[8] * ur + gdPtr[9] * ug + gdPtr[10] * ub + gdPtr[11];
            const float p0 = pwl17(gdPtr + 12, c0);
            const float p1 = pwl17(gdPtr + 29, c1);
            const float p2 = pwl17(gdPtr + 46, c2);
            const float gd = clampf(gdPtr[63] * p0 + gdPtr[64] * p1 + gdPtr[65] * p2 + gdPtr[66], 0.f, 1.f);

            const float fx = clampf((x + 0.5f) * (16.f / W) - 0.5f, 0.f, 15.f);
            const float fz = clampf(gd * 7.f, 0.f, 7.f);
            const int x0 = std::min(static_cast<int>(fx), 14), x1 = x0 + 1;
            const int z0 = std::min(static_cast<int>(fz), 6), z1 = z0 + 1;
            const float ax = fx - x0, az = fz - z0;

            const float w000 = (1.f - ax) * (1.f - ay) * (1.f - az);
            const float w100 = ax * (1.f - ay) * (1.f - az);
            const float w010 = (1.f - ax) * ay * (1.f - az);
            const float w110 = ax * ay * (1.f - az);
            const float w001 = (1.f - ax) * (1.f - ay) * az;
            const float w101 = ax * (1.f - ay) * az;
            const float w011 = (1.f - ax) * ay * az;
            const float w111 = ax * ay * az;

            const float* g000 = gPtr + (((size_t)y0 * 16 + x0) * 8 + z0) * 12;
            const float* g100 = gPtr + (((size_t)y0 * 16 + x1) * 8 + z0) * 12;
            const float* g010 = gPtr + (((size_t)y1 * 16 + x0) * 8 + z0) * 12;
            const float* g110 = gPtr + (((size_t)y1 * 16 + x1) * 8 + z0) * 12;
            const float* g001 = gPtr + (((size_t)y0 * 16 + x0) * 8 + z1) * 12;
            const float* g101 = gPtr + (((size_t)y0 * 16 + x1) * 8 + z1) * 12;
            const float* g011 = gPtr + (((size_t)y1 * 16 + x0) * 8 + z1) * 12;
            const float* g111 = gPtr + (((size_t)y1 * 16 + x1) * 8 + z1) * 12;

            for (int c = 0; c < 3; c++) {
                const int b = c * 4;
                const float a0 = w000 * g000[b + 0] + w100 * g100[b + 0] + w010 * g010[b + 0] + w110 * g110[b + 0] +
                                 w001 * g001[b + 0] + w101 * g101[b + 0] + w011 * g011[b + 0] + w111 * g111[b + 0];
                const float a1 = w000 * g000[b + 1] + w100 * g100[b + 1] + w010 * g010[b + 1] + w110 * g110[b + 1] +
                                 w001 * g001[b + 1] + w101 * g101[b + 1] + w011 * g011[b + 1] + w111 * g111[b + 1];
                const float a2 = w000 * g000[b + 2] + w100 * g100[b + 2] + w010 * g010[b + 2] + w110 * g110[b + 2] +
                                 w001 * g001[b + 2] + w101 * g101[b + 2] + w011 * g011[b + 2] + w111 * g111[b + 2];
                const float a3 = w000 * g000[b + 3] + w100 * g100[b + 3] + w010 * g010[b + 3] + w110 * g110[b + 3] +
                                 w001 * g001[b + 3] + w101 * g101[b + 3] + w011 * g011[b + 3] + w111 * g111[b + 3];
                const float yd = clampf(a0 * ur + a1 * ug + a2 * ub + a3, 0.f, 1.25f);
                q[c] = srgb_eotf_unclamped_cpu(yd) * keyScale;
            }
        }
    });

    local.sliceMs = sliceTimer.ms();
    if (st) *st = local;
    return true;
}
