#pragma once
#include "demosaic.h"
#include "finish.h"
#include "gpu/gpu_burst.h"
#include "kr.h"
#include "merge.h"
#include "ultrahdr.h"

// How the frames are merged; see pipeline/out/p1/RESULTS.md for the measurements behind the defaults.
struct BurstOptions {
    bool kernelRegression = false;
    float robustLo = 0, robustHi = 0;    // 0: kRobustCalibrated / kRobustEstimated by the noise source
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
// With use_gpu(), every step runs on the GPU (gpu/gpu_burst.h) where it can and falls back to the CPU
// code, step by step, when there is no GPU or a GPU step fails.
class BurstSession {
public:
    BurstSession(const Meta& m, std::vector<Raw> frames, int ref);

    // before align()
    void use_gpu(bool on) { gpuWanted_ = on; }
    // Before align(): the camera's calibrated noise at each ISO, used instead of estimating it from the
    // burst (works for single frames too). Empty keeps the estimate.
    void set_noise_curve(NoiseCurve c) { curve_ = std::move(c); }
    // Digital zoom: before align(), cuts every frame to its centre 1/zoom (CFA phase kept) into
    // private copies; finish() scales the result back to the sensor size.
    void crop(float zoom);
    void align();
    void merge(const BurstOptions& o);
    // keepHdr: also keeps the scene-linear luminance needed by encode_ultrahdr()
    void finish(const LookParams& look, bool keepHdr);
    int width() const { return outW_; }
    int height() const { return outH_; }
    void render(uint8_t* rgba, size_t stride) const;
    std::vector<uint8_t> encode_ultrahdr(int quality, float headroom);
    // One line of measurements for the log: noise model, merge weight, tone gains, timings.
    std::string summary() const;

private:
    void gpu_failed(const char* step, const std::exception& e);
    void cpu_align_state();
    void finish_cpu(const LookParams& look, bool keepHdr);

    bool gpuWanted_ = false, gpuFinished_ = false;
    std::unique_ptr<GpuBurst> gpu_;
    std::string gpuSteps_, gpuNote_, gpuProfile_;
    int outW_ = 0, outH_ = 0;
    Meta m_;
    std::vector<Raw> frames_;
    int ref_ = 0;
    float zoom_ = 1.f;
    bool kr_ = false, bracket_ = false;
    BurstOptions opt_;
    MergeOpts mo_;
    std::vector<ImageF> gray_, mean3_;
    std::vector<Offsets> offs_;
    NoiseModel nm_;
    NoiseCurve curve_;
    bool calibrated_ = false;
    ImageF mosaic_, rgb_;
    HdrLuma hdr_;
    UltraHdrStats us_;
    double meanWeight_ = 1, sigma_ = 0, visibleSigma_ = 0, chroma_ = 0, localContrast_ = 0, maxShift_ = 0, targetKey_ = 0,
           sharpenCore_ = 0;
    KrStats ks_;
    ToneInfo tone_{};
    double tCrop_ = 0, tAlign_ = 0, tNoise_ = 0, tMerge_ = 0, tUpscale_ = 0, tUhdr_ = 0;
    FinishTimes ft_;
};
