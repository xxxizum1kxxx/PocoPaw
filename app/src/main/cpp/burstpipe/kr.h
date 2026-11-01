#pragma once
#include "merge.h"

// Per-tile displacement in gray pixels (integer block-matching offset + sub-pixel refinement),
// on the same overlapping tile grid as Offsets.
struct Flow {
    int nx = 0, ny = 0;
    std::vector<float> fx, fy;
};

// Inverse-compositional Lucas-Kanade refinement of the block-matching offsets
// (Lafenetre et al., IPOL 2023, Alg. 6), run on the 3x3-averaged gray images.
Flow refine_flow(const ImageF& mref, const ImageF& malt, const Offsets& o, int iters);
Flow zero_flow(const Offsets& o);

struct KrParams {
    float scale = 1;
    // < 0: interpolated from the burst SNR like the IPOL reference implementation
    float kDetail = -1, kDenoise = -1, dTh = -1, dTr = -1;
    float kStretch = 4, kShrink = 2;
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
                const NoiseModel& nm, float robustLo, float robustHi, KrParams p, KrStats* st);
