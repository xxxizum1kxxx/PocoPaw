#pragma once
#include "common.h"

struct LookParams {
    float lscLumaStrength = 0.85f;  // 1 = fully remove vignetting, <1 keeps a little (and less corner noise)
    float chromaDenoise = -1.f;     // 0 disables, < 0 scales with the visible noise level
    float compression = 0.f;        // max gain ratio between synthetic exposures, 0 = auto
    float targetKey = 0.18f;        // linear mid-grey target for the bright synthetic exposure
    float contrast = 0.22f;         // S-curve blend
    float saturation = 1.12f;       // vibrance-style boost
    float sharpen = 0.6f;
    float sharpenRadius = 1.0f;
};

struct FinishTimes { double lscwb = 0, demosaic = 0, color = 0, chroma = 0, tonemap = 0, look = 0, sharpen = 0; };

// Bayer float mosaic (raw DN) -> linear camera RGB, WB + LSC applied, normalised to [0,1]
ImageF normalize_mosaic(const ImageF& mosaic, const Meta& m, const LookParams& p);
// Linear camera RGB in raw DN above black (e.g. from kr_merge, at `scale` x sensor size) -> same normalisation
void normalize_rgb(ImageF& rgb, const Meta& m, const LookParams& p, float scale);
ImageF demosaic_mhc(const ImageF& cfa, int cfaType);
void apply_ccm(ImageF& rgb, const float ccm[9]);
void chroma_denoise(ImageF& rgb, float strength);
struct ToneInfo { float gainShort, gainLong, keyBefore; };
ToneInfo local_tonemap(ImageF& rgb, const LookParams& p);    // HDR+ synthetic exposure fusion, in-place, output linear
void pixel_look(ImageF& rgb, const LookParams& p);          // sRGB OETF + S-curve + vibrance, output display-referred
void sharpen_luma(ImageF& rgb, float amount, float radius);
// Display-referred RGB -> RGBA8888 rows (Android ARGB_8888 memory order), dithered
void to_rgba8888(const ImageF& rgb, uint8_t* out, size_t stride);
