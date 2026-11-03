#pragma once
#include <mutex>

#include "../demosaic.h"
#include "../finish.h"
#include "../kr.h"
#include "../merge.h"
#include "../ultrahdr.h"
#include "cl_runtime.h"

// The burst pipeline on the GPU (OpenCL). Mirrors align / merge / merge_freq / kr / finish stage by stage
// on device-resident buffers: the frames are uploaded once, the merged image stays on the device through
// finishing, and only the RGBA result (plus the small statistics the CPU code computes on the host) comes
// back. Each call throws GpuError on failure; the caller then reruns that stage on the CPU.
class GpuBurst {
public:
    // Null (with the reason) when there is no usable GPU, another burst holds it, or the frames do not fit.
    static std::unique_ptr<GpuBurst> create(const std::vector<Raw>& frames, std::string& why);

    // Stages listed in `off` ("align,merge,freq,kr,finish") run on the CPU, for bisecting a GPU problem;
    // the default is "freq".
    static void set_disabled(const std::string& off);
    static bool enabled(const char* stage);

    // gray images, 3x3 means and tile offsets of every frame; the gray images of `ref` and `alt` are read
    // back for the noise fit
    void align(int ref, int alt, std::vector<Offsets>& offs, ImageF& grayRef, ImageF& grayAlt);
    bool aligned() const { return aligned_; }
    // 3x3 means and reference gray image of a CPU align, for a GPU merge after it
    void set_aligned(const std::vector<ImageF>& mean3, const ImageF& grayRef);

    // merges keep their result on the device; frames come from the device, offsets from the host
    double merge_spatial(int ref, const std::vector<Offsets>& offs, const NoiseModel& nm, float black, const MergeOpts& o);
    double merge_freq(int ref, const std::vector<Offsets>& offs, const Meta& m, const NoiseModel& nm, float c, float spatial);
    // `ref` frame on the host for the SNR estimate, as kr_merge
    void merge_kr(const Raw& refFrame, int ref, const std::vector<Offsets>& offs, const Meta& m, const NoiseModel& nm,
                  const MergeOpts& o, KrParams p, KrStats* st);
    // frees the frames after the merge
    void drop_frames();
    // the merged mosaic (classic) or RGB (kernel regression) for a CPU finish; finish() keeps it intact
    // until it returns
    ImageF download_merged();
    bool merged() const { return merged_.w > 0; }
    // a CPU merge's result, for a GPU finish
    void set_merged(const ImageF& im, bool kernelRegression);

    struct FinishOut {
        double sigma = 0, visibleSigma = 0, chroma = 0, localContrast = 0, tUpscale = 0, targetKey = 0, sharpenCore = 0;
        ToneInfo tone{};
        FinishTimes ft;
        HdrLuma hdr;
    };
    void finish(const Meta& m, const LookParams& lp, Demosaic dm, float zoom, bool keepHdr, FinishOut& out);
    int width() const { return rgb_.w; }
    int height() const { return rgb_.h; }
    void render(uint8_t* rgba, size_t stride);
    ImageF download_rgb();

    // per-kernel device times since the last call, when profiling is on
    std::string profile();

private:
    GpuBurst(Gpu& g, std::unique_lock<std::mutex> lock) : g_(g), lock_(std::move(lock)) {}
    GImg upload(const ImageF& im);
    ImageF download(const GImg& g);
    void upload_offsets(const std::vector<Offsets>& offs);
    void robustness_rows(const GImg& rob, int k, std::vector<double>& rowSums);
    GImg gaussian(const GImg& in, float sigma);
    GImg box(const GImg& in, int r);
    GImg pyr_down(const GImg& in);
    void pyr_up_sub(const GImg& coarse, GImg& fine, float s);
    GImg demosaic_rcd(const GImg& cfa, int cfaType);
    void chroma_denoise(float strength);
    void upscale(int W, int H);
    ToneInfo local_tonemap(const LookParams& p);
    GImg fuse_exposures(const GImg& Y, float gS, float gL);
    void local_contrast(float amount, float radiusFrac);
    void sharpen(float amount, float radius, float core);
    GImg luma(const GImg& rgb, float floorv);

    Gpu& g_;
    std::unique_lock<std::mutex> lock_;
    int n_ = 0, W_ = 0, H_ = 0, qw_ = 0, qh_ = 0;
    Buf frames_, mean3_, win_, odx_, ody_, oetf_, eotf_;
    int nt_ = 0, tnx_ = 0, tny_ = 0;
    GImg grayRef_;
    bool aligned_ = false, kr_ = false;
    GImg merged_, rgb_;
};
