#pragma once
#include "finish.h"
#include "kr.h"
#include "merge.h"

// One shutter press worth of processing, split into the steps the app runs as pipeline stages:
// align() -> merge() -> finish() -> render(). Frames are views; they may be released after merge().
class BurstSession {
public:
    BurstSession(const Meta& m, std::vector<Raw> frames, int ref);

    void align();
    void merge(bool kernelRegression, float robustLo, float robustHi);
    void finish(const LookParams& look);
    int width() const { return rgb_.w; }
    int height() const { return rgb_.h; }
    void render(uint8_t* rgba, size_t stride) const;
    // One line of measurements for the log: noise model, merge weight, tone gains, timings.
    std::string summary() const;

private:
    Meta m_;
    std::vector<Raw> frames_;
    int ref_ = 0;
    bool kr_ = false;
    std::vector<ImageF> gray_, mean3_;
    std::vector<Offsets> offs_;
    NoiseModel nm_;
    ImageF mosaic_, rgb_;
    double meanWeight_ = 1, sigma_ = 0, visibleSigma_ = 0, chroma_ = 0, maxShift_ = 0;
    KrStats ks_;
    ToneInfo tone_{};
    double tAlign_ = 0, tNoise_ = 0, tMerge_ = 0;
    FinishTimes ft_;
};
