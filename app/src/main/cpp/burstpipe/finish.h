#pragma once
#include "common.h"

struct LookParams {
    float lscLumaStrength = 0.85f;  // 1 = fully remove vignetting, <1 keeps a little (and less corner noise)
    float chromaDenoise = -1.f;     // 0 disables, < 0 scales with the visible noise level
    float compression = 0.f;        // max gain ratio between synthetic exposures, 0 = auto
    float targetKey = 0.18f;        // linear mid-grey target for the bright synthetic exposure
    float contrast = 0.22f;         // S-curve blend
    float saturation = 1.12f;       // vibrance-style boost
    float sharpen = 0.35f;
    float sharpenRadius = 0.85f;
    float localContrast = 0.f;      // retained for JNI struct layout; locked to 0 in Indigo look
    float chromaDenoiseScale = 1.f; // multiplies the automatic chroma denoise strength
};

// local_tonemap: fusion at 1/kToneSub of each side, gain upsampled by a guided filter of this radius (binned px) and
// epsilon (log2 luma squared)
constexpr int kToneSub = 4;
constexpr int kToneGuideRadius = 2;
constexpr float kToneGuideEps = 0.02f;

struct FinishTimes { double lscwb = 0, demosaic = 0, color = 0, chroma = 0, tonemap = 0, look = 0, sharpen = 0; };

// Bayer float mosaic (raw DN) -> linear camera RGB, WB + LSC applied, normalised to [0,1]
ImageF normalize_mosaic(const ImageF& mosaic, const Meta& m, const LookParams& p);
void correct_lca(ImageF& rgb, const Meta& m, const float lcaR[2], const float lcaB[2]);
void apply_ccm(ImageF& rgb, const float ccm[9]);
struct DualColorCal {
    int illuminants = 0;            // 0 = absent, 1 or 2 sets
    int illuminant[2] = {21, 17};   // EXIF LightSource tags
    float colorMatrix[2][9] = {};   // XYZ -> camera (SENSOR_COLOR_TRANSFORMn), row-major
    float forwardMatrix[2][9] = {}; // white-balanced camera -> XYZ D50 (SENSOR_FORWARD_MATRIXn), row-major
    float neutral[3] = {0, 0, 0};   // AsShotNeutral in camera RGB; <= 0 falls back to 1/wb
};
bool compute_dual_illuminant_ccm(const DualColorCal& cal, const float wb[4], float outCcm[9]);
bool ccm_usable(const float ccm[9]);
void chroma_denoise(ImageF& rgb, float strength);
// separable Lanczos-3 resize of a linear RGB image to W x H (for the zoom crop back to sensor size)
void upscale_lanczos(ImageF& rgb, int W, int H);
void aspect_crop_box(int inW, int inH, float aspect, int& x0, int& y0, int& cw, int& ch);
void crop_aspect(ImageF& rgb, float aspect);
struct ToneInfo { float gainShort, gainLong, keyBefore; };
ToneInfo local_tonemap(ImageF& rgb, const LookParams& p);    // HDR+ synthetic exposure fusion, in-place, output linear
void pixel_look(ImageF& rgb, const LookParams& p);          // Oklab gamut + sRGB OETF + S-curve + vibrance
// display-luma unsharp mask; detail below `core` (sharpen_core) is left alone so grain is not boosted
void sharpen_luma(ImageF& rgb, float amount, float radius, float core);
float sharpen_core(float visibleSigma);
// targetKey lowered where reaching it would lift the visible noise above kKeyNoiseCap (dark scenes, high ISO)
float noise_limited_key(float targetKey, float visibleSigma);
bool apply_look_preset(const std::string& name, LookParams& p);   // indigo | default | pixel | night
bool write_jpeg(const std::string& path, const ImageF& rgb, int quality, int downscale);
std::vector<uint8_t> to_rgb8(const ImageF& rgb, int downscale, int& w, int& h);  // dithered, box-downscaled
// Display-referred RGB -> RGBA8888 rows (Android ARGB_8888 memory order), dithered
void to_rgba8888(const ImageF& rgb, uint8_t* out, size_t stride);
void rgb8_to_rgba8888(const uint8_t* rgb8, int w, int h, uint8_t* out, size_t stride);
// Box-subsampled display-referred RGB -> RGBA8888 thumbnail of size outW x outH
void to_rgba8888_thumb(const ImageF& rgb, uint8_t* out, size_t stride, int outW, int outH);
void rgb8_to_rgba8888_thumb(const uint8_t* rgb8, int inW, int inH, uint8_t* out, size_t stride, int outW, int outH);
// chroma420: 4:2:0 at any quality (otherwise 4:4:4 above 90), as Android's Bitmap.compress
// libjpeg-turbo, strip-parallel; channels 1 (gray), 3 (RGB) or 4 (RGBA, alpha ignored)
std::vector<uint8_t> encode_jpeg(const uint8_t* px, int w, int h, int channels, int quality, bool chroma420 = false);
