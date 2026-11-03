#pragma once
#include "common.h"

// Colour of the capturing camera as Camera2 reports it (the tags DngCreator writes): row-major 3x3
// matrices per calibration illuminant (EXIF LightSource codes), and the reference frame's neutral.
struct DngColor {
    int illuminants = 0;  // 1 or 2 sets below
    int illuminant[2] = {21, 17};
    float colorMatrix[2][9] = {};    // XYZ -> camera (SENSOR_COLOR_TRANSFORMn)
    float forwardMatrix[2][9] = {};  // white-balanced camera -> XYZ D50 (SENSOR_FORWARD_MATRIXn)
    float neutral[3] = {1, 1, 1};    // AsShotNeutral
};

struct DngExtras {
    int orientation = 1;  // EXIF / TIFF orientation
    // stops; a bracket merge extends the white level past the reference's, so renderers need this to keep its brightness
    float baselineExposure = 0;
    std::string make, model, software;
    // 8-bit RGB preview in the sensor's orientation, in a SubIFD of the raw IFD0 (as DngCreator); empty = none
    std::vector<uint8_t> thumb;
    int thumbW = 0, thumbH = 0;
};

// The merged mosaic in raw DN (black not subtracted) at 16 bits: values times kDngScale (fewer when the
// white level would not fit), keeping 4 of the bits the merge adds below one DN. Black and white levels scale with it.
constexpr int kDngScale = 16;
int dng_scale(float white);

// DNG 1.4 of a Bayer mosaic (Meta gives size, CFA, black / white, and the lens shading written as one OpcodeList2
// GainMap per CFA position, placed on the digital-zoom crop), little-endian, uncompressed.
std::vector<uint8_t> encode_dng(const ImageF& mosaic, const Meta& m, const DngColor& c, const DngExtras& x);
