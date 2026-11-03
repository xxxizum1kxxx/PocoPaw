#pragma once
#include "common.h"

// HDR+ style coarse-to-fine tile alignment (Hasinoff et al. 2016, Sec. 4).
// Works on the 2x2-binned grayscale image, so offsets are in gray pixels
// (= 2 raw pixels, which keeps the Bayer phase intact).
constexpr int kTile = 16;            // finest-level tile size (gray px)
constexpr int kStride = kTile / 2;   // tiles overlap by half

struct Offsets {
    int nx = 0, ny = 0;
    std::vector<int16_t> dx, dy;
    int16_t& x(int i, int j) { return dx[(size_t)j * nx + i]; }
    int16_t& y(int i, int j) { return dy[(size_t)j * nx + i]; }
    int16_t x(int i, int j) const { return dx[(size_t)j * nx + i]; }
    int16_t y(int i, int j) const { return dy[(size_t)j * nx + i]; }
};

ImageF gray_from_raw(const Raw& r);                       // sum of each 2x2 quad
std::vector<ImageF> build_pyramid(const ImageF& g0);      // downsample factors 2, 4, 4
Offsets align_frames(const std::vector<ImageF>& ref, const std::vector<ImageF>& alt);
ImageF box3(const ImageF& g);                             // 3x3 mean
// gradient energy of the 4x4-binned centre (3/4 of each side); higher is sharper at equal exposure
double sharpness(const Raw& r);
// the sharpest of the first `count` frames, the reference a handheld burst should align to
int sharpest_frame(const std::vector<Raw>& frames, int count = 3);
