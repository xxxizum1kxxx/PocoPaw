#pragma once
#include <mutex>

#include "../demosaic.h"
#include "../finish.h"
#include "../merge.h"
#include "../ultrahdr.h"
#include "cl_runtime.h"

// The Indigo HDR+ burst pipeline on Adreno OpenCL. Frames stay device-resident through alignment, spatial merge,
// RCD demosaic, chroma denoise, guided exposure fusion, perceptual Oklab look, and local-tile capture sharpening,
// writing dithered RGB8 directly into a zero-copy host-mapped unified memory buffer.
class GpuBurst {
public:
    static std::unique_ptr<GpuBurst> create(const std::vector<Raw>& frames, std::string& why);

    static void set_disabled(const std::string& off);
    static bool enabled(const char* stage);

    void align(int ref, int alt, std::vector<Offsets>& offs, ImageF& grayRef, ImageF& grayAlt);
    bool aligned() const { return aligned_; }
    void set_aligned(const std::vector<ImageF>& mean3, const ImageF& grayRef);

    double merge_spatial(int ref, const std::vector<Offsets>& offs, const NoiseModel& nm, float black, const MergeOpts& o);
    void drop_frames();

    // Streaming align + spatial merge with GPU-resident offsets and ping-pong upload slots (zero per-frame host stalls).
    static std::unique_ptr<GpuBurst> create_stream(int width, int height, std::string& why);
    void stream_ref(const Raw& ref, float black, ImageF* gray);
    void stream_align(const Raw& frame, Offsets* hostOffs, ImageF* gray);
    void stream_merge(const NoiseModel& nm, float black, const MergeOpts& o);
    double stream_end(float black, double* maxShift = nullptr);

    ImageF download_merged();
    bool merged() const { return merged_.w > 0 && static_cast<bool>(merged_.b); }
    void set_merged(const ImageF& im);

    struct FinishOut {
        double sigma = 0, visibleSigma = 0, chroma = 0, tUpscale = 0, targetKey = 0, sharpenCore = 0;
        ToneInfo tone{};
        FinishTimes ft;
        HdrLuma hdr;
    };
    void finish(const Meta& m, const LookParams& lp, Demosaic dm, float zoom, bool keepHdr, FinishOut& out, float aspect = 0.f,
                const float lcaR[2] = nullptr, const float lcaB[2] = nullptr);
    int width() const { return outW_; }
    int height() const { return outH_; }
    const uint8_t* rgb8_ptr() const { return rgb8Ptr_; }
    void render(uint8_t* rgba, size_t stride);
    std::vector<uint8_t> render_rgb8();
    void render_thumb(uint8_t* rgba, size_t stride, int outW, int outH);
    ImageF download_rgb();

    std::string profile();
    ~GpuBurst() = default;

private:
    static constexpr int kMaxStreamFrames = 16;

    GpuBurst(Gpu& g, std::unique_lock<std::mutex> lock) : g_(g), lock_(std::move(lock)) {}
    static std::unique_ptr<GpuBurst> open(int width, int height, int slots, std::string& why);
    std::vector<GImg> pyramid(int k);
    void align_pyramids_to(const std::vector<GImg>& refPyr, const std::vector<GImg>& altPyr, Buf& outDx, Buf& outDy, int outOff);
    GImg upload(const ImageF& im);
    ImageF download(const GImg& g);
    void upload_offsets(const std::vector<Offsets>& offs);
    Buf gaussian_weights(float sigma, int maxRadius, int& outR);
    GImg pyr_down(const GImg& in);
    void pyr_up_sub(const GImg& coarse, GImg& fine, float s);
    void demosaic_rcd_and_color(const GImg& cfa, const Meta& m, const float lcaR[2], const float lcaB[2]);
    void chroma_and_prepare(float strength, bool writeTone, Buf& keySub, int s8w, GImg& Ys, int sw, int sh);
    void upscale(int W, int H);
    void crop_aspect(float aspect);
    ToneInfo local_tonemap_and_look(const LookParams& p, Buf& keySub, int s8w, int s8h, GImg& Ys, int sw, int sh);
    GImg fuse_exposures(const GImg& Y, float gS, float gL);

    Gpu& g_;
    std::unique_lock<std::mutex> lock_;
    int n_ = 0, W_ = 0, H_ = 0, qw_ = 0, qh_ = 0, outW_ = 0, outH_ = 0;
    Buf frames_, mean3_, win_, odx_, ody_;
    int nt_ = 0, tnx_ = 0, tny_ = 0;
    GImg grayRef_;
    bool aligned_ = false, offsOnGpu_ = false;
    GImg merged_, rgb_;
    Buf rgb8_;
    const uint8_t* rgb8Ptr_ = nullptr;
    std::vector<GImg> refPyr_;
    GImg sOut_, sWsum_, sRob_, sTmp_;
    Buf sRows_, sOdx_, sOdy_;
    int sAdded_ = 0, sCurSlot_ = 1, sCurOff_ = 0;
};
