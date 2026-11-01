#pragma once
#include "merge.h"

// Per-tile displacement in gray pixels (integer block-matching offset + sub-pixel refinement),
// on the same overlapping tile grid as Offsets.
struct Flow {
    int nx = 0, ny = 0;
    std::vector<float> fx, fy;
    // per-tile standard error of the refined displacement (gray px), from the LK Hessian and the noise
    // model; empty for the reference / unrefined flows
    std::vector<float> err;
};

// Inverse-compositional Lucas-Kanade refinement of the block-matching offsets
// (Lafenetre et al., IPOL 2023, Alg. 6), run on the 3x3-averaged gray images.
// central-difference gradients of the reference, shared by every frame's refinement
struct LkGrad { ImageF gx, gy; };
LkGrad lk_gradients(const ImageF& mref);
Flow refine_flow(const ImageF& mref, const ImageF& malt, const Offsets& o, int iters, const LkGrad& g,
                 const NoiseModel& nm, float black);

// Sub-pixel diversity of the burst, in [0,1]: over tiles whose flow error is small, how far the
// displacement spread across frames exceeds the estimation noise, times the share of half-pixel phase
// bins it reaches. Static bursts score near 0 even though LK noise alone fills the bins.
struct SubpixelStats { double score = 0, confidentTiles = 0, meanSpread = 0, meanErr = 0, coverage = 0; };
SubpixelStats subpixel_diversity(const std::vector<Flow>& flows, int ref);
constexpr double kAutoScaleThreshold = 0.3;
Flow zero_flow(const Offsets& o);

struct KrParams {
    float scale = 1;
    // < 0: interpolated from the burst SNR like the IPOL reference implementation
    float kDetail = -1, kDenoise = -1, dTh = -1, dTr = -1;
    float kStretch = 4, kShrink = 2;
    // lateral chromatic aberration: channel X is sampled around p + (p - c) * (kX1 + kX2 * (r / rmax)^2)
    // (tools/lca_calib.py); all zero = off
    float lcaR[2] = {0, 0}, lcaB[2] = {0, 0};
};

struct KrStats {
    double snrDb = 0, meanWeight = 0, phaseCoverage = 0;
    float kDetail = 0, kDenoise = 0, dTh = 0, dTr = 0;
};

// Handheld multi-frame super-resolution merge (Wronski et al. 2019): every raw sample of every
// aligned frame contributes to the output RGB grid through an anisotropic Gaussian kernel shaped
// by the local structure tensor, times a per-frame robustness weight. Replaces merge + demosaic.
// Returns linear camera RGB in raw DN above black, at `scale` times the sensor resolution.
ImageF kr_merge(const std::vector<Raw>& frames, int ref, const std::vector<Flow>& flows,
                const ImageF& grayRef, const std::vector<ImageF>& mean3, const Meta& m,
                const NoiseModel& nm, const MergeOpts& o, KrParams p, KrStats* st);
