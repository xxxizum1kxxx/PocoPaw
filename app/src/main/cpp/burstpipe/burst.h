#pragma once
#include "demosaic.h"
#include "dng.h"
#include "finish.h"
#include "gpu/gpu_burst.h"
#include "merge.h"
#include "ultrahdr.h"

struct BurstOptions {
    bool kernelRegression = false;
    float robustLo = 0, robustHi = 0;    // 0: kRobustCalibrated / kRobustEstimated by the noise source
    int dilate = 1;                      // motion mask dilation, in quads
    Demosaic demosaic = Demosaic::RCD;
    bool freqMerge = false;
    float freqC = 64.f, freqSpatial = 2.f;
    float lcaR[2] = {0, 0}, lcaB[2] = {0, 0};
};

// One shutter press worth of Indigo HDR+ processing:
// align() -> merge() -> finish() -> render_thumb() / encode_photo().
// With use_gpu(), every stage runs on Adreno OpenCL and falls back cleanly to the CPU on GpuError.
class BurstSession {
public:
    BurstSession(const Meta& m, std::vector<Raw> frames, int ref);

    void use_gpu(bool on) { gpuWanted_ = on; }
    void set_noise_curve(NoiseCurve c) { curve_ = std::move(c); }
    void crop(float zoom);
    void set_aspect(float aspect) { aspect_ = aspect; }
    void align();
    void merge(const BurstOptions& o);

    static constexpr int kStreamCandidates = 3;
    void begin_stream(const BurstOptions& o);
    int add_frame(Raw frame);
    int end_frames();
    std::vector<uint8_t> encode_dng(const DngColor& c, const DngExtras& x);
    void finish(const LookParams& look, bool keepHdr);
    int width() const { return outW_; }
    int height() const { return outH_; }
    void render(uint8_t* rgba, size_t stride) const;
    void render_thumb(uint8_t* rgba, size_t stride, int outW, int outH) const;
    std::vector<uint8_t> encode_photo(int quality, bool wantUltraHdr, float headroom, bool* isUltraHdr);
    std::vector<uint8_t> encode_ultrahdr(int quality, float headroom);
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
    float zoom_ = 1.f, aspect_ = 0.f;
    int cropX0_ = 0, cropY0_ = 0;
    bool bracket_ = false;
    float refScale_ = 1.f;
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
    ToneInfo tone_{};
    double tCrop_ = 0, tAlign_ = 0, tNoise_ = 0, tMerge_ = 0, tUpscale_ = 0, tUhdr_ = 0, tDng_ = 0;
    FinishTimes ft_;
};
