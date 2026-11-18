#pragma once
#include <string>
#include <vector>

#include "accel_backend.h"
#include "common.h"
#include "finish.h"

struct HdrnetStats {
    bool ran = false;
    double ortMs = 0.0;
    double sliceMs = 0.0;
    std::string ep;
    std::string note;
};

// Resolves hdrnet.onnx path. If explicitPath is non-empty, returns it only if readable
// (never substitutes a different file for an explicit missing path).
std::string resolve_hdrnet_model(const std::string& explicitPath, const char* argv0 = nullptr);

// Downsamples scene-linear RGB to log-compressed [1, 3, 256, 256] planar NCHW input for HDRnet.
void hdrnet_downsample_cpu(const ImageF& rgb, float* nchw256);

// Runs HDRnet ONNX predictor on [1, 3, 256, 256] log-compressed input, returning 16x16x8x12 grid
// (24,576 floats) and 67-float guidance parameters.
bool hdrnet_predict(
    const float* nchw256,
    const std::string& modelPath,
    AccelBackend backend,
    std::vector<float>& gridOut,
    std::vector<float>& guideOut,
    HdrnetStats* st = nullptr,
    const char* argv0 = nullptr);

// Applies learned HDRnet bilateral grid tone mapping to scene-linear RGB in-place.
// Populates outTone (same pre-tonemap key/gainShort/gainLong contract as local_tonemap).
// Returns false if model missing/invalid (leaving rgb untouched for clean fallback to MERTENS).
bool hdrnet_tonemap(
    ImageF& rgb,
    const LookParams& lp,
    AccelBackend backend,
    ToneInfo& outTone,
    HdrnetStats* st = nullptr,
    const char* argv0 = nullptr);
