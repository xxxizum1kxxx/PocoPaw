#include "semantic_seg.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>

#include "ort_runtime.h"

namespace {

bool file_readable(const std::string& p) {
    if (p.empty()) return false;
    struct stat st{};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

inline float smooth01(float t) {
    t = clampf(t, 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

struct CachedSession {
    std::mutex mu;
    std::string key;
    std::unique_ptr<OrtSessionRunner> runner;
};

CachedSession& seg_cache() {
    // Process-lifetime heap singleton: avoids atexit teardown ordering issues against dlopen'd ORT.
    static auto* c = new CachedSession();
    return *c;
}

int normalize_rot90(int degrees) {
    int d = ((degrees % 360) + 360) % 360;
    if (d >= 45 && d < 135) return 90;
    if (d >= 135 && d < 225) return 180;
    if (d >= 225 && d < 315) return 270;
    return 0;
}

inline float sample_plane_bilinear(const float* p, int w, int h, float fx, float fy) {
    fx = clampf(fx, 0.f, (float)std::max(0, w - 1));
    fy = clampf(fy, 0.f, (float)std::max(0, h - 1));
    const int x0 = (int)fx, y0 = (int)fy;
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    const float ax = fx - x0, ay = fy - y0;
    return (1.f - ay) * ((1.f - ax) * p[(size_t)y0 * w + x0] + ax * p[(size_t)y0 * w + x1]) +
           ay * ((1.f - ax) * p[(size_t)y1 * w + x0] + ax * p[(size_t)y1 * w + x1]);
}

}  // namespace

std::string resolve_seg_model(const std::string& explicitPath, const char* argv0) {
    if (!explicitPath.empty()) {
        return file_readable(explicitPath) ? explicitPath : std::string();
    }
    std::vector<std::string> candidates = {
        "calib/semantic_seg.onnx",
        "assets/models/semantic_seg.onnx",
        "models/semantic_seg.onnx",
        "../PocoPaw/app/src/main/assets/models/semantic_seg.onnx",
        "../../PocoPaw/app/src/main/assets/models/semantic_seg.onnx",
        "app/src/main/assets/models/semantic_seg.onnx",
    };
    if (argv0 && *argv0) {
        std::string exe(argv0);
        const size_t slash = exe.rfind('/');
        const std::string exeDir = slash == std::string::npos ? "." : exe.substr(0, slash);
        candidates.push_back(exeDir + "/semantic_seg.onnx");
        candidates.push_back(exeDir + "/../calib/semantic_seg.onnx");
        candidates.push_back(exeDir + "/../../PocoPaw/app/src/main/assets/models/semantic_seg.onnx");
    }
    for (const auto& p : candidates) {
        if (file_readable(p)) return p;
    }
    return {};
}

std::vector<float> map_face_boxes_to_roi(
    const std::vector<float>& fullBoxes,
    const Meta& m,
    int aspectX0,
    int aspectY0,
    int aspectW,
    int aspectH,
    int preAspectW,
    int preAspectH) {
    std::vector<float> out;
    const size_t count = fullBoxes.size() / 4;
    if (count == 0) return out;

    const float fullW = (float)std::max(1, m.full_w());
    const float fullH = (float)std::max(1, m.full_h());
    const float preW = (float)std::max(1, preAspectW);
    const float preH = (float)std::max(1, preAspectH);
    const float cropW = (float)std::max(1, m.width);
    const float cropH = (float)std::max(1, m.height);

    const float u0 = (m.cropX + cropW * (aspectX0 / preW)) / fullW;
    const float v0 = (m.cropY + cropH * (aspectY0 / preH)) / fullH;
    const float uw = std::max(1e-5f, (cropW * (aspectW / preW)) / fullW);
    const float vh = std::max(1e-5f, (cropH * (aspectH / preH)) / fullH);

    out.reserve(count * 4);
    for (size_t i = 0; i < count; i++) {
        const float rawX0 = fullBoxes[i * 4 + 0];
        const float rawY0 = fullBoxes[i * 4 + 1];
        const float rawX1 = fullBoxes[i * 4 + 2];
        const float rawY1 = fullBoxes[i * 4 + 3];
        if (!is_finite_f32_val(rawX0) || !is_finite_f32_val(rawY0) ||
            !is_finite_f32_val(rawX1) || !is_finite_f32_val(rawY1)) continue;
        const float bx0 = clampf(std::min(rawX0, rawX1), 0.f, 1.f);
        const float by0 = clampf(std::min(rawY0, rawY1), 0.f, 1.f);
        const float bx1 = clampf(std::max(rawX0, rawX1), 0.f, 1.f);
        const float by1 = clampf(std::max(rawY0, rawY1), 0.f, 1.f);
        const float rx0 = clampf((bx0 - u0) / uw, 0.f, 1.f);
        const float ry0 = clampf((by0 - v0) / vh, 0.f, 1.f);
        const float rx1 = clampf((bx1 - u0) / uw, 0.f, 1.f);
        const float ry1 = clampf((by1 - v0) / vh, 0.f, 1.f);
        if (rx1 - rx0 < 1e-3f || ry1 - ry0 < 1e-3f) continue;
        out.push_back(rx0);
        out.push_back(ry0);
        out.push_back(rx1);
        out.push_back(ry1);
    }
    return out;
}

std::vector<float> downsample_rgb256(const ImageF& rgb) {
    constexpr int kS = 256;
    std::vector<float> out((size_t)kS * kS * 3, 0.f);
    if (rgb.w <= 0 || rgb.h <= 0 || rgb.c != 3) return out;
    parallel_for(kS, [&](int y) {
        const int y0 = (int)((long)y * rgb.h / kS);
        const int y1 = std::max(y0 + 1, (int)((long)(y + 1) * rgb.h / kS));
        const int sy = std::max(1, (y1 - y0) / 4);
        for (int x = 0; x < kS; x++) {
            const int x0 = (int)((long)x * rgb.w / kS);
            const int x1 = std::max(x0 + 1, (int)((long)(x + 1) * rgb.w / kS));
            const int sx = std::max(1, (x1 - x0) / 4);
            float sr = 0.f, sg = 0.f, sb = 0.f;
            int cnt = 0;
            for (int yy = y0; yy < y1; yy += sy) {
                const float* row = rgb.row(yy) + (size_t)x0 * 3;
                const int stride = sx * 3;
                for (int xx = x0; xx < x1; xx += sx, row += stride) {
                    sr += row[0];
                    sg += row[1];
                    sb += row[2];
                    cnt++;
                }
            }
            const float inv = 1.f / std::max(1, cnt);
            float* dst = out.data() + ((size_t)y * kS + x) * 3;
            dst[0] = clampf(sr * inv, 0.f, 1.f);
            dst[1] = clampf(sg * inv, 0.f, 1.f);
            dst[2] = clampf(sb * inv, 0.f, 1.f);
        }
    });
    return out;
}

bool semantic_seg_from_rgb256(
    const std::vector<float>& rgb256,
    const std::string& modelPath,
    const std::vector<float>& mappedFaceBoxes,
    int orientationDegrees,
    AccelBackend backend,
    SemanticMask& outMask,
    SemanticSegStats* st,
    const char* argv0) {
    Timer timer;
    SemanticSegStats local;
    local.faceBoxCount = static_cast<int>(mappedFaceBoxes.size() / 4);
    outMask = SemanticMask();

    constexpr int kS = 256;
    const int rot = normalize_rot90(orientationDegrees);
    std::vector<float> modelFace, modelSky;
    int mwModel = 0, mhModel = 0;

    const std::string resolved = resolve_seg_model(modelPath, argv0);
    if (!resolved.empty() && rgb256.size() == (size_t)kS * kS * 3) {
        constexpr size_t kPlane = (size_t)kS * kS;
        std::vector<float> nchw(3 * kPlane);
        for (int uy = 0; uy < kS; uy++) {
            for (int ux = 0; ux < kS; ux++) {
                int sx = ux, sy = uy;
                if (rot == 90) { sx = uy; sy = kS - 1 - ux; }
                else if (rot == 180) { sx = kS - 1 - ux; sy = kS - 1 - uy; }
                else if (rot == 270) { sx = kS - 1 - uy; sy = ux; }
                const float* p = rgb256.data() + ((size_t)sy * kS + sx) * 3;
                const size_t dstIdx = (size_t)uy * kS + ux;
                nchw[0 * kPlane + dstIdx] = clampf(p[0], 0.f, 1.f);
                nchw[1 * kPlane + dstIdx] = clampf(p[1], 0.f, 1.f);
                nchw[2 * kPlane + dstIdx] = clampf(p[2], 0.f, 1.f);
            }
        }

        std::vector<float> ortOut;
        std::vector<int64_t> outShape;
        std::string ortErr;
        bool ortOk = false;
        {
            CachedSession& cache = seg_cache();
            std::lock_guard<std::mutex> lk(cache.mu);
            const std::string cacheKey = resolved + "#" + std::to_string(static_cast<int>(backend));
            if (!cache.runner || cache.key != cacheKey) {
                cache.runner = OrtSessionRunner::create(resolved, backend, 4, &ortErr);
                cache.key = cache.runner ? cacheKey : std::string();
            }
            if (cache.runner) {
                local.ep = cache.runner->active_ep();
                const int64_t inShape[4] = {1, 3, kS, kS};
                ortOk = cache.runner->run(nchw.data(), inShape, 4, ortOut, outShape, &ortErr);
            }
        }

        if (ortOk && outShape.size() == 4 && outShape[0] == 1 && outShape[1] == 2 &&
            outShape[2] > 0 && outShape[3] > 0 &&
            ortOut.size() == (size_t)2 * outShape[2] * outShape[3]) {
            const bool finite = all_finite_f32(ortOut.data(), ortOut.size());
            if (finite) {
                const int uh = static_cast<int>(outShape[2]);
                const int uw = static_cast<int>(outShape[3]);
                const size_t uPlane = (size_t)uh * uw;
                // Rotate mask back counter-clockwise from upright to sensor coordinates.
                if (rot == 90 || rot == 270) {
                    mwModel = uh;
                    mhModel = uw;
                } else {
                    mwModel = uw;
                    mhModel = uh;
                }
                modelFace.resize((size_t)mwModel * mhModel);
                modelSky.resize((size_t)mwModel * mhModel);
                for (int sy = 0; sy < mhModel; sy++) {
                    for (int sx = 0; sx < mwModel; sx++) {
                        int ux = sx, uy = sy;
                        if (rot == 90) { ux = uw - 1 - sy; uy = sx; }
                        else if (rot == 180) { ux = uw - 1 - sx; uy = uh - 1 - sy; }
                        else if (rot == 270) { ux = sy; uy = uh - 1 - sx; }
                        const size_t sIdx = (size_t)sy * mwModel + sx;
                        const size_t uIdx = (size_t)uy * uw + ux;
                        modelFace[sIdx] = clampf(ortOut[0 * uPlane + uIdx], 0.f, 1.f);
                        modelSky[sIdx] = clampf(ortOut[1 * uPlane + uIdx], 0.f, 1.f);
                    }
                }
                local.usedModel = true;
            } else {
                local.note = "non-finite seg output";
            }
        } else if (!ortErr.empty()) {
            local.note = ortErr;
        }
    } else if (!modelPath.empty() && resolved.empty()) {
        local.note = "model not found: " + modelPath;
    }

    if (!local.usedModel && local.faceBoxCount == 0) {
        if (local.note.empty()) local.note = "seg model missing and no faceBoxes";
        local.ms = timer.ms();
        if (st) *st = local;
        return false;
    }

    constexpr int kMaskW = 64;
    constexpr int kMaskH = 64;
    outMask.w = kMaskW;
    outMask.h = kMaskH;
    outMask.data.assign((size_t)kMaskW * kMaskH * 2, 0.f);

    double sumFace = 0.0, sumSky = 0.0;
    for (int y = 0; y < kMaskH; y++) {
        const float v = (y + 0.5f) / kMaskH;
        const float my = local.usedModel ? (y + 0.5f) * mhModel / (float)kMaskH - 0.5f : 0.f;
        for (int x = 0; x < kMaskW; x++) {
            const float u = (x + 0.5f) / kMaskW;
            float face = 0.f, sky = 0.f;
            if (local.usedModel) {
                const float mx = (x + 0.5f) * mwModel / (float)kMaskW - 0.5f;
                face = sample_plane_bilinear(modelFace.data(), mwModel, mhModel, mx, my);
                sky = sample_plane_bilinear(modelSky.data(), mwModel, mhModel, mx, my);
            }
            for (int b = 0; b < local.faceBoxCount; b++) {
                const float x0 = mappedFaceBoxes[b * 4 + 0];
                const float y0 = mappedFaceBoxes[b * 4 + 1];
                const float x1 = mappedFaceBoxes[b * 4 + 2];
                const float y1 = mappedFaceBoxes[b * 4 + 3];
                const float cx = 0.5f * (x0 + x1);
                const float cy = 0.5f * (y0 + y1);
                const float rx = std::max(0.01f, 0.5f * (x1 - x0));
                const float ry = std::max(0.01f, 0.5f * (y1 - y0));
                const float d = std::hypot((u - cx) / rx, (v - cy) / ry);
                const float prior = 1.f - smooth01((d - 0.75f) / 0.50f);
                face = std::max(face, 0.85f * prior);
                sky *= (1.f - prior);
            }
            face = clampf(face, 0.f, 1.f);
            sky = clampf(sky, 0.f, 1.f);
            const size_t idx = ((size_t)y * kMaskW + x) * 2;
            outMask.data[idx + 0] = face;
            outMask.data[idx + 1] = sky;
            sumFace += face;
            sumSky += sky;
        }
    }

    const double norm = (double)(kMaskW * kMaskH);
    local.ran = true;
    local.meanFace = static_cast<float>(sumFace / norm);
    local.meanSky = static_cast<float>(sumSky / norm);
    local.ms = timer.ms();
    if (st) *st = local;
    return true;
}
