#pragma once
#include <string>

#include "accel_backend.h"
#include "common.h"
#include "merge.h"

struct PmridStats {
    bool ran = false;
    float kEff = 0.f;
    float bEff = 0.f;
    float neff = 1.f;
    float alpha = 0.f;
    double ms = 0.0;
    std::string ep;
    std::string note;
};

// Interpolated ALPHA_KNOTS blend weight modulated by smoothstep low-noise gating.
// When lowNoiseCutoff > 0 and kEff <= 0.5f * lowNoiseCutoff, returns 0.f.
float pmrid_blend_alpha(float kEff, float lowNoiseCutoff);

// Computes effective noise parameters and checks whether PMRID should run (alpha > 0).
// Allows callers to skip GPU mosaic download when low-noise gating disables PMRID.
bool pmrid_should_run(
    const Meta& m,
    const NoiseModel& nm,
    int frameCount,
    double meanWeight,
    float lowNoiseCutoff,
    float* outKEff = nullptr,
    float* outBEff = nullptr,
    float* outNeff = nullptr,
    float* outAlpha = nullptr);

// Resolves pmrid.onnx path. If explicitPath is non-empty, returns it only if readable
// (never substitutes a different file for an explicit missing path).
std::string resolve_pmrid_model(const std::string& explicitPath, const char* argv0 = nullptr);

// Runs PMRID learned raw denoising on a merged Bayer mosaic (raw DN, black level included).
// Returns true iff mosaic was updated; false leaves mosaic untouched (clean fallback).
bool pmrid_denoise_mosaic(
    ImageF& mosaic,
    const Meta& m,
    const NoiseModel& nm,
    int frameCount,
    double meanWeight,
    float lowNoiseCutoff,
    const std::string& modelPath,
    AccelBackend backend,
    PmridStats* st = nullptr,
    const char* argv0 = nullptr);
