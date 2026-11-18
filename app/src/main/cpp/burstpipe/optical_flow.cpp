#include "optical_flow.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

#if (defined(__ARM_NEON) || defined(__aarch64__)) && !defined(BURSTPIPE_NO_NEON)
#include <arm_neon.h>
#define FLOW_HAS_NEON 1
#endif

#include "ort_runtime.h"

namespace {

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

CachedSession& flow_cache() {
    // Process-lifetime heap singleton: avoids atexit teardown ordering issues against dlopen'd ORT.
    static auto* c = new CachedSession();
    return *c;
}

float tile_cost_gray(const ImageF& ref, const ImageF& alt, int x0, int y0, int t, int ox, int oy, bool l1, float best) {
    float s = 0.f;
    const bool inside = x0 >= 0 && y0 >= 0 && x0 + t <= ref.w && y0 + t <= ref.h &&
                        x0 + ox >= 0 && y0 + oy >= 0 && x0 + ox + t <= alt.w && y0 + oy + t <= alt.h;
#ifdef FLOW_HAS_NEON
    if (inside && t == 16) {
        for (int v = 0; v < 16; v++) {
            const float* a = ref.row(y0 + v) + x0;
            const float* b = alt.row(y0 + v + oy) + x0 + ox;
            float32x4_t a0 = vld1q_f32(a),      b0 = vld1q_f32(b);
            float32x4_t a1 = vld1q_f32(a + 4),  b1 = vld1q_f32(b + 4);
            float32x4_t a2 = vld1q_f32(a + 8),  b2 = vld1q_f32(b + 8);
            float32x4_t a3 = vld1q_f32(a + 12), b3 = vld1q_f32(b + 12);
            float32x4_t sum0, sum1, sum2, sum3;
            if (l1) {
                sum0 = vabdq_f32(a0, b0);
                sum1 = vabdq_f32(a1, b1);
                sum2 = vabdq_f32(a2, b2);
                sum3 = vabdq_f32(a3, b3);
            } else {
                float32x4_t d0 = vsubq_f32(a0, b0), d1 = vsubq_f32(a1, b1);
                float32x4_t d2 = vsubq_f32(a2, b2), d3 = vsubq_f32(a3, b3);
                sum0 = vmulq_f32(d0, d0);
                sum1 = vmulq_f32(d1, d1);
                sum2 = vmulq_f32(d2, d2);
                sum3 = vmulq_f32(d3, d3);
            }
            float32x4_t rowSum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
            s += vaddvq_f32(rowSum);
            if (s > best) return s;
        }
        return s;
    }
#endif
    for (int v = 0; v < t; v++) {
        if (inside) {
            const float* a = ref.row(y0 + v) + x0;
            const float* b = alt.row(y0 + v + oy) + x0 + ox;
            if (l1) {
                for (int u = 0; u < t; u++) s += std::fabs(a[u] - b[u]);
            } else {
                for (int u = 0; u < t; u++) {
                    const float d = a[u] - b[u];
                    s += d * d;
                }
            }
        } else {
            for (int u = 0; u < t; u++) {
                const float d = ref.atc(x0 + u, y0 + v) - alt.atc(x0 + u + ox, y0 + v + oy);
                s += l1 ? std::fabs(d) : d * d;
            }
        }
        if (s > best) return s;
    }
    return s;
}

inline void parabolic_fit_2d_flow(const float c[3][3], float& fx, float& fy) {
    const float gx = ((c[0][2] - c[0][0]) + (c[1][2] - c[1][0]) + (c[2][2] - c[2][0])) * (1.f / 6.f);
    const float gy = ((c[2][0] - c[0][0]) + (c[2][1] - c[0][1]) + (c[2][2] - c[0][2])) * (1.f / 6.f);
    const float hxx = ((c[0][2] - 2 * c[0][1] + c[0][0]) + (c[1][2] - 2 * c[1][1] + c[1][0]) +
                       (c[2][2] - 2 * c[2][1] + c[2][0])) * (1.f / 3.f);
    const float hyy = ((c[2][0] - 2 * c[1][0] + c[0][0]) + (c[2][1] - 2 * c[1][1] + c[0][1]) +
                       (c[2][2] - 2 * c[1][2] + c[0][2])) * (1.f / 3.f);
    const float hxy = (c[2][2] - c[2][0] - c[0][2] + c[0][0]) * 0.25f;

    const float tr = hxx + hyy;
    if (tr <= 0.f) { fx = 0.f; fy = 0.f; return; }
    const float det = hxx * hyy - hxy * hxy;
    const float disc = std::sqrt(std::max(0.f, 0.25f * tr * tr - det));
    const float l1 = 0.5f * tr + disc, l2 = 0.5f * tr - disc;
    const float cmin = std::max(c[1][1], 1e-6f);
    if (l1 <= 0.8f * cmin) { fx = 0.f; fy = 0.f; return; }

    float e1x = 1.f, e1y = 0.f;
    if (std::fabs(hxy) > 1e-12f * tr) {
        if (hxx >= hyy) { e1x = l1 - hyy; e1y = hxy; }
        else { e1x = hxy; e1y = l1 - hxx; }
        const float nrm = std::hypot(e1x, e1y);
        e1x /= nrm;
        e1y /= nrm;
    } else if (hyy > hxx) {
        e1x = 0.f;
        e1y = 1.f;
    }
    const float e2x = -e1y, e2y = e1x;
    const float g1 = gx * e1x + gy * e1y;
    const float g2 = gx * e2x + gy * e2y;

    const float mu = 0.02f * cmin;
    const float s1 = -g1 * l1 / (l1 * l1 + mu * mu);
    const float s2 = (l2 > std::max(0.05f * l1, 0.8f * cmin)) ? (-g2 * l2 / (l2 * l2 + mu * mu)) : 0.f;
    fx = clampf(s1 * e1x + s2 * e2x, -0.5f, 0.5f);
    fy = clampf(s1 * e1y + s2 * e2y, -0.5f, 0.5f);

    const float c0 = c[1][1];
    const float dlx = c[1][0] - c0, drx = c[1][2] - c0;
    const float dly = c[0][1] - c0, dry = c[2][1] - c0;
    if (dlx <= 0.f || drx <= 0.f || std::max(dlx, drx) > 12.f * std::min(dlx, drx)) fx = 0.f;
    if (dly <= 0.f || dry <= 0.f || std::max(dly, dry) > 12.f * std::min(dly, dry)) fy = 0.f;
}

inline float sample_bilinear(const float* plane, int w, int h, float x, float y) {
    const float fx = clampf(x, 0.f, (float)std::max(0, w - 1));
    const float fy = clampf(y, 0.f, (float)std::max(0, h - 1));
    const int x0 = (int)fx, y0 = (int)fy;
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    const float ax = fx - x0, ay = fy - y0;
    const float v00 = plane[(size_t)y0 * w + x0];
    const float v10 = plane[(size_t)y0 * w + x1];
    const float v01 = plane[(size_t)y1 * w + x0];
    const float v11 = plane[(size_t)y1 * w + x1];
    return (1.f - ay) * ((1.f - ax) * v00 + ax * v10) + ay * ((1.f - ax) * v01 + ax * v11);
}

inline float sample_gray_bilinear(const ImageF& im, float x, float y) {
    const float fx = clampf(x, 0.f, (float)std::max(0, im.w - 1));
    const float fy = clampf(y, 0.f, (float)std::max(0, im.h - 1));
    const int x0 = (int)fx, y0 = (int)fy;
    const int x1 = std::min(x0 + 1, im.w - 1), y1 = std::min(y0 + 1, im.h - 1);
    const float ax = fx - x0, ay = fy - y0;
    return (1.f - ay) * ((1.f - ax) * im.at(x0, y0) + ax * im.at(x1, y0)) +
           ay * ((1.f - ax) * im.at(x0, y1) + ax * im.at(x1, y1));
}

}  // namespace

std::string resolve_flow_model(const std::string& explicitPath, const char* argv0) {
    if (!explicitPath.empty()) {
        return file_readable(explicitPath) ? explicitPath : std::string();
    }
    std::vector<std::string> candidates = {
        "calib/sea_raft.onnx",
        "assets/models/sea_raft.onnx",
        "models/sea_raft.onnx",
        "../PocoPaw/app/src/main/assets/models/sea_raft.onnx",
        "../../PocoPaw/app/src/main/assets/models/sea_raft.onnx",
        "app/src/main/assets/models/sea_raft.onnx",
    };
    if (argv0 && *argv0) {
        std::string exe(argv0);
        const size_t slash = exe.rfind('/');
        const std::string exeDir = slash == std::string::npos ? "." : exe.substr(0, slash);
        candidates.push_back(exeDir + "/sea_raft.onnx");
        candidates.push_back(exeDir + "/../calib/sea_raft.onnx");
        candidates.push_back(exeDir + "/../../PocoPaw/app/src/main/assets/models/sea_raft.onnx");
    }
    for (const auto& p : candidates) {
        if (file_readable(p)) return p;
    }
    return {};
}

bool assess_alignment_residual(
    const ImageF& grayRef,
    const ImageF& grayAlt,
    const Offsets& offs,
    float black,
    float* outGlobRatio,
    float* outP999Ratio,
    float* outOutlierFrac) {
    if (grayRef.w < 16 || grayRef.h < 16 || grayAlt.w != grayRef.w || grayAlt.h != grayRef.h || offs.nx <= 0 || offs.ny <= 0) {
        if (outGlobRatio) *outGlobRatio = 0.f;
        if (outP999Ratio) *outP999Ratio = 0.f;
        if (outOutlierFrac) *outOutlierFrac = 0.f;
        return false;
    }

    // Unbiased Immerkaer Laplacian noise-floor estimate on grayRef.
    // For white noise N(0, sigma^2), L/6 ~ N(0, sigma^2); the 20th percentile of |L/6|
    // equals Phi^-1(0.60) * sigma = 0.25334710 * sigma. For two aligned frames with independent
    // noise sigma, E[|n_ref - n_alt|] = (2 / sqrt(pi)) * sigma = 1.12837917 * sigma.
    std::vector<float> lapSamples;
    lapSamples.reserve((size_t)((grayRef.h / 2) * (grayRef.w / 2)) / 4 + 16);
    constexpr float kInv6Phi60 = 1.f / (6.f * 0.25334710f);
    for (int y = 2; y < grayRef.h - 2; y += 4) {
        const float* r0 = grayRef.row(y - 1);
        const float* r1 = grayRef.row(y);
        const float* r2 = grayRef.row(y + 1);
        for (int x = 2; x < grayRef.w - 2; x += 4) {
            const float l = 4.f * r1[x] - 2.f * (r1[x - 1] + r1[x + 1] + r0[x] + r2[x]) +
                            (r0[x - 1] + r0[x + 1] + r2[x - 1] + r2[x + 1]);
            lapSamples.push_back(std::fabs(l) * kInv6Phi60);
        }
    }
    constexpr float kTwoFrameDiffMae = 1.12837917f;  // 2 / sqrt(pi)
    float sigmaRef = 4.f;
    if (!lapSamples.empty()) {
        const size_t p20Idx = lapSamples.size() / 5;
        std::nth_element(lapSamples.begin(), lapSamples.begin() + p20Idx, lapSamples.end());
        sigmaRef = std::max(4.f, kTwoFrameDiffMae * lapSamples[p20Idx]);
    }

    const int nTiles = offs.nx * offs.ny;
    std::vector<float> rels(nTiles, 0.f);
    std::vector<float> maes(nTiles, 0.f);
    const float black4 = 4.f * black;
    parallel_for(offs.ny, [&](int j) {
        for (int i = 0; i < offs.nx; i++) {
            const int x0 = i * kStride, y0 = j * kStride;
            const int ox = offs.x(i, j), oy = offs.y(i, j);
            float mae = 0.f, meanRef = 0.f;
            if (x0 >= 0 && y0 >= 0 && x0 + kTile <= grayRef.w && y0 + kTile <= grayRef.h &&
                x0 + ox >= 0 && y0 + oy >= 0 && x0 + ox + kTile <= grayAlt.w && y0 + oy + kTile <= grayAlt.h) {
                for (int v = 0; v < kTile; v++) {
                    const float* a = grayRef.row(y0 + v) + x0;
                    const float* b = grayAlt.row(y0 + v + oy) + x0 + ox;
                    for (int u = 0; u < kTile; u++) {
                        meanRef += a[u];
                        mae += std::fabs(a[u] - b[u]);
                    }
                }
            } else {
                for (int v = 0; v < kTile; v++) {
                    for (int u = 0; u < kTile; u++) {
                        const float rv = grayRef.atc(x0 + u, y0 + v);
                        const float av = grayAlt.atc(x0 + u + ox, y0 + v + oy);
                        meanRef += rv;
                        mae += std::fabs(rv - av);
                    }
                }
            }
            mae *= (1.f / (kTile * kTile));
            meanRef *= (1.f / (kTile * kTile));
            const size_t idx = (size_t)j * offs.nx + i;
            maes[idx] = mae;
            rels[idx] = mae / std::sqrt(std::max(0.f, meanRef - black4) + 16.f);
        }
    });

    double maeSum = 0.0;
    for (float v : maes) maeSum += v;
    const float globRatio = static_cast<float>(maeSum / std::max(1, nTiles)) / sigmaRef;

    std::vector<float> sorted = rels;
    const size_t medIdx = sorted.size() / 2;
    std::nth_element(sorted.begin(), sorted.begin() + medIdx, sorted.end());
    const float med = std::max(0.1f, sorted[medIdx]);

    const size_t p999Idx = std::min(sorted.size() - 1, (size_t)(sorted.size() * 0.999));
    std::nth_element(sorted.begin(), sorted.begin() + p999Idx, sorted.end());
    const float p999Ratio = sorted[p999Idx] / med;

    int outliers = 0;
    const float outThresh = std::max(1.5f, 5.f * med);
    for (float r : rels) {
        if (r > outThresh) outliers++;
    }
    const float outlierFrac = static_cast<float>(outliers) / std::max(1, nTiles);

    if (outGlobRatio) *outGlobRatio = globRatio;
    if (outP999Ratio) *outP999Ratio = p999Ratio;
    if (outOutlierFrac) *outOutlierFrac = outlierFrac;
    return (globRatio > 3.0f) || (p999Ratio > 12.0f && outlierFrac >= 0.004f);
}

bool estimate_sea_raft_offsets(
    const ImageF& grayRef,
    const ImageF& grayAlt,
    const ImageF& mean3Ref,
    const ImageF& mean3Alt,
    float black,
    float white,
    const std::string& modelPath,
    AccelBackend backend,
    Offsets& outOffs,
    std::string* outEp,
    std::string* outNote,
    const char* argv0) {
    if (grayRef.w < 16 || grayRef.h < 16 || grayAlt.w != grayRef.w || grayAlt.h != grayRef.h) {
        if (outNote) *outNote = "invalid gray dimensions";
        return false;
    }

    const std::string resolved = resolve_flow_model(modelPath, argv0);
    if (resolved.empty()) {
        if (outNote) *outNote = modelPath.empty() ? "sea_raft.onnx not found" : ("model not found: " + modelPath);
        return false;
    }

    // Working resolution capped at max long side 384 (multiple of 8) to bound CorrBlock O((HW/64)^2) memory.
    const int maxSide = 384;
    const int longSide = std::max(grayRef.w, grayRef.h);
    const float downScale = longSide > maxSide ? (float)maxSide / (float)longSide : 1.f;
    auto round8 = [](int v) { return std::max(64, ((v + 4) / 8) * 8); };
    const int flowW = std::min(std::max(8, (grayRef.w / 8) * 8), round8((int)std::lround(grayRef.w * downScale)));
    const int flowH = std::min(std::max(8, (grayRef.h / 8) * 8), round8((int)std::lround(grayRef.h * downScale)));

    const float black4 = 4.f * black;
    const float span4 = std::max(1.f, 4.f * white - black4);
    std::vector<float> samp;
    samp.reserve((size_t)(grayRef.w / 4 + 1) * (grayRef.h / 4 + 1));
    for (int y = 0; y < grayRef.h; y += 4) {
        const float* row = grayRef.row(y);
        for (int x = 0; x < grayRef.w; x += 4) samp.push_back(row[x]);
    }
    float p99 = black4 + span4;
    if (!samp.empty()) {
        const size_t p99Idx = std::min(samp.size() - 1, (size_t)(samp.size() * 0.99));
        std::nth_element(samp.begin(), samp.begin() + p99Idx, samp.end());
        p99 = samp[p99Idx];
    }
    const float normScale = clampf(p99 - black4, 0.05f * span4, span4);
    const float invNormScale = 1.f / normScale;
    struct GammaLut {
        float t[1025];
        GammaLut() {
            for (int i = 0; i <= 1024; i++) {
                const float u = i / 1024.f;
                t[i] = std::pow(u * u, 1.f / 2.2f) * 255.f;
            }
        }
    };
    static const GammaLut kGammaLut;
    auto gammaSample = [](float v) {
        const float f = std::sqrt(clampf(v, 0.f, 1.f)) * 1024.f;
        const int i = std::min(static_cast<int>(f), 1023);
        return kGammaLut.t[i] + (f - i) * (kGammaLut.t[i + 1] - kGammaLut.t[i]);
    };

    const size_t planeSz = (size_t)flowW * flowH;
    std::vector<float> inp(6 * planeSz);
    const float sxScale = (float)grayRef.w / (float)flowW;
    const float syScale = (float)grayRef.h / (float)flowH;
    parallel_for(flowH, [&](int y) {
        const float sy = clampf((y + 0.5f) * syScale - 0.5f, 0.f, (float)(grayRef.h - 1));
        for (int x = 0; x < flowW; x++) {
            const float sx = clampf((x + 0.5f) * sxScale - 0.5f, 0.f, (float)(grayRef.w - 1));
            const float rNorm = clampf((sample_gray_bilinear(grayRef, sx, sy) - black4) * invNormScale, 0.f, 1.f);
            const float aNorm = clampf((sample_gray_bilinear(grayAlt, sx, sy) - black4) * invNormScale, 0.f, 1.f);
            const float rVal = gammaSample(rNorm);
            const float aVal = gammaSample(aNorm);
            const size_t idx = (size_t)y * flowW + x;
            inp[0 * planeSz + idx] = rVal;
            inp[1 * planeSz + idx] = rVal;
            inp[2 * planeSz + idx] = rVal;
            inp[3 * planeSz + idx] = aVal;
            inp[4 * planeSz + idx] = aVal;
            inp[5 * planeSz + idx] = aVal;
        }
    });

    std::vector<float> flowOut;
    std::vector<int64_t> outShape;
    std::string ortErr;
    {
        CachedSession& cache = flow_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        const std::string cacheKey = resolved + "#" + std::to_string(static_cast<int>(backend));
        if (!cache.runner || cache.key != cacheKey) {
            cache.runner = OrtSessionRunner::create(resolved, backend, 4, &ortErr);
            cache.key = cache.runner ? cacheKey : std::string();
        }
        if (!cache.runner) {
            if (outNote) *outNote = ortErr.empty() ? "ORT session init failed" : ortErr;
            return false;
        }
        if (outEp) *outEp = cache.runner->active_ep();
        const int64_t inShape[4] = {1, 6, flowH, flowW};
        if (!cache.runner->run(inp.data(), inShape, 4, flowOut, outShape, &ortErr)) {
            if (outNote) *outNote = ortErr.empty() ? "ORT inference failed" : ortErr;
            return false;
        }
    }

    if (outShape.size() != 4 || outShape[0] != 1 || outShape[1] != 2 ||
        outShape[2] != flowH || outShape[3] != flowW || flowOut.size() != 2 * planeSz) {
        if (outNote) *outNote = "unexpected SEA-RAFT output shape";
        return false;
    }
    if (!all_finite_f32(flowOut.data(), flowOut.size())) {
        if (outNote) *outNote = "non-finite SEA-RAFT output";
        return false;
    }

    const float* uPlane = flowOut.data();
    const float* vPlane = flowOut.data() + planeSz;
    const int nx = (grayRef.w + kStride - 1) / kStride;
    const int ny = (grayRef.h + kStride - 1) / kStride;
    Offsets res;
    res.nx = nx;
    res.ny = ny;
    res.dx.resize((size_t)nx * ny);
    res.dy.resize((size_t)nx * ny);
    res.fx.assign((size_t)nx * ny, 0.f);
    res.fy.assign((size_t)nx * ny, 0.f);

    const float toFlowX = (float)flowW / (float)grayRef.w;
    const float toFlowY = (float)flowH / (float)grayRef.h;
    const float toGrayX = (float)grayRef.w / (float)flowW;
    const float toGrayY = (float)grayRef.h / (float)flowH;
    const ImageF& rSub = mean3Ref.w == grayRef.w ? mean3Ref : grayRef;
    const ImageF& aSub = mean3Alt.w == grayAlt.w ? mean3Alt : grayAlt;

    parallel_for(ny, [&](int j) {
        for (int i = 0; i < nx; i++) {
            const int x0 = i * kStride, y0 = j * kStride;
            const float cx = (x0 + kTile * 0.5f) * toFlowX - 0.5f;
            const float cy = (y0 + kTile * 0.5f) * toFlowY - 0.5f;
            const float uGray = sample_bilinear(uPlane, flowW, flowH, cx, cy) * toGrayX;
            const float vGray = sample_bilinear(vPlane, flowW, flowH, cx, cy) * toGrayY;
            int bx = clampi((int)std::lround(uGray), -grayRef.w / 2, grayRef.w / 2);
            int by = clampi((int)std::lround(vGray), -grayRef.h / 2, grayRef.h / 2);

            // Fine-level (l == 0) R = 2 L1 snapping removes downsampling quantization error.
            const int cx0 = bx, cy0 = by;
            float best = tile_cost_gray(grayRef, grayAlt, x0, y0, kTile, cx0, cy0, true, 3.4e38f);
            for (int oy = -2; oy <= 2; oy++) {
                for (int ox = -2; ox <= 2; ox++) {
                    if (!ox && !oy) continue;
                    const float s = tile_cost_gray(grayRef, grayAlt, x0, y0, kTile, cx0 + ox, cy0 + oy, true, best);
                    if (s < best) {
                        best = s;
                        bx = cx0 + ox;
                        by = cy0 + oy;
                    }
                }
            }

            const size_t idx = (size_t)j * nx + i;
            res.dx[idx] = static_cast<int16_t>(bx);
            res.dy[idx] = static_cast<int16_t>(by);

            float c3[3][3];
            for (int oy = -1; oy <= 1; oy++) {
                for (int ox = -1; ox <= 1; ox++) {
                    c3[oy + 1][ox + 1] = tile_cost_gray(rSub, aSub, x0, y0, kTile, bx + ox, by + oy, false, 3.4e38f);
                }
            }
            parabolic_fit_2d_flow(c3, res.fx[idx], res.fy[idx]);
        }
    });

    outOffs = std::move(res);
    return true;
}
