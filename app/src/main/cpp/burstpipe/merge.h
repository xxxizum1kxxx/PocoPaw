#pragma once
#include "align.h"

// Raw-domain noise model: variance = a * signal + b (signal in raw DN above black).
// Camera2 exposes this as SENSOR_NOISE_PROFILE, but this HAL does not, so it is
// estimated from the residual between the reference and an aligned frame.
struct NoiseModel {
    double a = 1, b = 4;
    double var(double s) const { return a * std::max(0.0, s) + b; }
};

// sample of frame k's gray/mean image at reference quad (qx,qy), blending the
// 4 overlapping tiles with raised-cosine weights
NoiseModel estimate_noise(const ImageF& gref, const ImageF& galt, const Offsets& off, float black);

struct MergeStats { double meanWeight = 0; };

// Spatial-domain robust merge on the Bayer mosaic (HDR+ bracketing / Super Res Zoom style
// per-pixel deghosting). Returns a float mosaic in raw DN (black not subtracted).
ImageF merge_frames(const std::vector<Raw>& frames, int ref, const std::vector<Offsets>& offs,
                    const std::vector<ImageF>& mean3, const NoiseModel& nm, float black,
                    float robustLo, float robustHi, MergeStats* st);
