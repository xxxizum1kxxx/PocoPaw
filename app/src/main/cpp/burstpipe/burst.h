#pragma once
#include "demosaic.h"
#include "finish.h"
#include "kr.h"
#include "merge.h"
#include "ultrahdr.h"

// How the frames are merged; see pipeline/out/p1/RESULTS.md for the measurements behind the defaults.
struct BurstOptions {
    bool kernelRegression = false;
    float robustLo = 2.f, robustHi = 4.5f;
    int dilate = 1;                      // motion mask dilation, in quads
    Demosaic demosaic = Demosaic::RCD;   // classic path only
    bool freqMerge = false;              // classic path: frequency-domain Wiener merge (night)
    float freqC = 64.f, freqSpatial = 2.f;
    float lcaR[2] = {0, 0}, lcaB[2] = {0, 0};  // kernel regression only
};

// One shutter press worth of processing, split into the steps the app runs as pipeline stages:
// align() -> merge() -> finish() -> render() / encode_ultrahdr(). Frames are views; they may be
// released after merge(). A burst whose frames differ in exposure x ISO is a bracket: the shortest
// exposure becomes the reference and the others are scaled to it before alignment.
class BurstSession {
public:
    BurstSession(const Meta& m, std::vector<Raw> frames, int ref);

    void align();
    void merge(const BurstOptions& o);
    // keepHdr: also keeps the scene-linear luminance needed by encode_ultrahdr()
    void finish(const LookParams& look, bool keepHdr);
    int width() const { return rgb_.w; }
    int height() const { return rgb_.h; }
    void render(uint8_t* rgba, size_t stride) const;
    std::vector<uint8_t> encode_ultrahdr(int quality, float headroom);
    // One line of measurements for the log: noise model, merge weight, tone gains, timings.
    std::string summary() const;

private:
    Meta m_;
    std::vector<Raw> frames_;
    int ref_ = 0;
    bool kr_ = false, bracket_ = false;
    BurstOptions opt_;
    MergeOpts mo_;
    std::vector<ImageF> gray_, mean3_;
    std::vector<Offsets> offs_;
    NoiseModel nm_;
    ImageF mosaic_, rgb_;
    HdrLuma hdr_;
    UltraHdrStats us_;
    double meanWeight_ = 1, sigma_ = 0, visibleSigma_ = 0, chroma_ = 0, maxShift_ = 0;
    KrStats ks_;
    ToneInfo tone_{};
    double tAlign_ = 0, tNoise_ = 0, tMerge_ = 0, tUhdr_ = 0;
    FinishTimes ft_;
};
