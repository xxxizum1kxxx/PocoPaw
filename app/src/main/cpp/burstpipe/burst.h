#pragma once
#include "demosaic.h"
#include "dng.h"
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

    // Merging while the burst is captured: a session made with no frames, after use_gpu / set_noise_curve /
    // crop, takes begin_stream() and then each frame through add_frame() as it arrives, and end_frames()
    // after the last one; merge() then only records the options' demosaic. Classic spatial merge of one
    // exposure only. The first kStreamCandidates frames are held to pick the reference (sharpest_frame);
    // every later frame is aligned and merged on arrival and not kept. add_frame / end_frames return the
    // reference's index (in arrival order) once it is chosen, else -1; from then on every frame but the
    // reference may be freed after add_frame returns, the reference after end_frames.
    static constexpr int kStreamCandidates = 3;
    void begin_stream(const BurstOptions& o);
    int add_frame(Raw frame);
    int end_frames();
    // After merge(), before finish(): the merged mosaic as a 16-bit DNG (dng.h); empty after kernel regression,
    // which merges to RGB.
    std::vector<uint8_t> encode_dng(const DngColor& c, const DngExtras& x);
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
    Raw cropped(const Raw& r) const;
    void stream_start();
    void stream_frame(const Raw& f);
    void stream_cpu_start();

    bool gpuWanted_ = false, gpuFinished_ = false;
    std::unique_ptr<GpuBurst> gpu_;
    std::string gpuSteps_, gpuNote_, gpuProfile_;
    int outW_ = 0, outH_ = 0;
    Meta m_;
    std::vector<Raw> frames_;
    int ref_ = 0;
    float zoom_ = 1.f;
    int cropX0_ = 0, cropY0_ = 0;
    bool kr_ = false, bracket_ = false;
    float refScale_ = 1.f;  // bracket factor applied to the reference's raw values above black
    bool streaming_ = false, streamed_ = false, streamGpu_ = false, noiseReady_ = false;
    int streamAdded_ = 0, streamMerged_ = 0;
    Raw streamRef_;
    std::unique_ptr<SpatialMerge> streamMerge_;
    std::vector<ImageF> streamRefPyr_;
    ImageF streamMean3Ref_, streamGrayRef_;
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
    double tCrop_ = 0, tAlign_ = 0, tNoise_ = 0, tMerge_ = 0, tUpscale_ = 0, tUhdr_ = 0, tDng_ = 0;
    FinishTimes ft_;
};
