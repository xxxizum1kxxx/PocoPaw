#pragma once
#include "align.h"

// Raw-domain noise model: variance = a * signal + b (signal in raw DN above black).
// Camera2 exposes this as SENSOR_NOISE_PROFILE, but this HAL does not, so it comes from an offline
// calibration (NoiseCurve) or is estimated from the residual between the reference and an aligned frame.
struct NoiseModel {
    double a = 1, b = 4;
    double var(double s) const { return a * std::max(0.0, s) + b; }
};

// Per-ISO noise measured offline (tools/noise_calib.py curve), in raw DN of an unscaled frame; linear in ISO
// between the points, a proportional to ISO below the first, held above the last.
struct NoiseCurve {
    std::vector<float> iso, a, b;
    bool empty() const { return iso.empty(); }
    // for a frame whose values above black were multiplied by `scale` (bracket scaling)
    NoiseModel at(float frameIso, float scale = 1.f) const;
};

// Robustness ramps (in noise sigmas) for each noise source: the residual estimate runs 10-25% low, so the
// calibrated model needs a proportionally tighter ramp for the same ghost rejection (RESULTS.md "noise-cal").
constexpr float kRobustEstimated[2] = {2.f, 4.5f};
constexpr float kRobustCalibrated[2] = {1.75f, 3.9f};

// sample of frame k's gray/mean image at reference quad (qx,qy), blending the
// 4 overlapping tiles with raised-cosine weights
NoiseModel estimate_noise(const ImageF& gref, const ImageF& galt, const Offsets& off, float black);

struct MergeStats { double meanWeight = 0; };

// Per-frame robustness options shared by the spatial and kernel-regression merges. Bracketed bursts are
// scaled to the reference exposure before merging (scale_to_reference); `gain` keeps each frame's factor
// relative to the reference so the noise model and the exposure weighting know about it.
struct MergeOpts {
    float robustLo = 1.f, robustHi = 3.f;  // local-mean mismatch ramp, in noise sigmas
    int dilate = 1;                         // motion mask dilation: min filter radius on the robustness map, in quads
    std::vector<float> gain;                // per frame, < 1 for longer exposures than the reference; empty = all 1
    std::vector<float> clip;                // per frame raw level (after scaling) treated as clipped; empty = none
    float g(int k) const { return gain.empty() ? 1.f : gain[k]; }
    float clipAt(int k) const { return clip.empty() ? 1e30f : clip[k]; }
};

// Robustness of one alternate-frame quad: linear ramp of the 3x3 local-mean mismatch (z-score against
// the summed noise of both frames), times the inverse-variance weight of the frame's exposure.
inline float frame_weight(float mu0, float mu, float black, const NoiseModel& nm, float g, const MergeOpts& o) {
    float s = std::max(0.f, mu0 / 4 - black);
    float a = (float)nm.a, b = (float)nm.b;
    float vRef = a * s + b, vAlt = g * a * s + g * g * b;
    float varMu = 4.f * (vRef + vAlt) / 9.f;
    float z = std::fabs(mu - mu0) / std::sqrt(std::max(varMu, 1e-3f));
    return clampf((o.robustHi - z) / (o.robustHi - o.robustLo), 0.f, 1.f) * (vRef / vAlt);
}

// separable running minimum over a (2r+1)^2 window, edges clamped
void min_filter(ImageF& img, int r);

// Bracketed bursts: scales every frame's raw values (above black) to the reference exposure, in place,
// with a common factor so the longest exposure keeps integer DN. Returns the options' gain / clip arrays
// and the new white level of the reference. No-op (returns false) when all exposures match.
bool scale_to_reference(std::vector<Raw>& frames, int ref, const Meta& m, MergeOpts& o, float& refWhite);

// Spatial-domain robust merge on the Bayer mosaic (HDR+ bracketing / Super Res Zoom style
// per-pixel deghosting). Returns a float mosaic in raw DN (black not subtracted).
ImageF merge_frames(const std::vector<Raw>& frames, int ref, const std::vector<Offsets>& offs,
                    const std::vector<ImageF>& mean3, const NoiseModel& nm, float black,
                    const MergeOpts& o, MergeStats* st);

// merge_frames one frame at a time, for merging while the burst is still being captured: start from the
// reference, add() each aligned frame (it may be freed right after), then result(). Same output as
// merge_frames given the frames in the same order.
class SpatialMerge {
public:
    SpatialMerge(const Raw& ref, float black);
    // g / clip: the frame's MergeOpts::g(k) / clipAt(k); checkClip as !o.clip.empty()
    void add(const Raw& F, const Offsets& of, const ImageF& mean3, const ImageF& mean3Ref, const NoiseModel& nm, float g,
             float clip, bool checkClip, const MergeOpts& o);
    ImageF result(MergeStats* st);
    int added() const { return added_; }

private:
    ImageF out_, wsum_, rob_;
    float black_;
    double wTotal_ = 0;
    int added_ = 0;
};

// Frequency-domain alternative (HDR+ 2016 Sec. 5): 16x16 half-overlapping tiles per Bayer plane on the
// alignment grid, pairwise Wiener merge A = |D|^2 / (|D|^2 + c * sigma_D^2) against the reference,
// optional spatial shrinkage of the merged spectrum (`spatial` > 0). Same output as merge_frames.
ImageF merge_frames_freq(const std::vector<Raw>& frames, int ref, const std::vector<Offsets>& offs,
                         const Meta& m, const NoiseModel& nm, float c, float spatial, MergeStats* st);
