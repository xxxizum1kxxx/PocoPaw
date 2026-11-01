#pragma once
#include "common.h"

// Ultra HDR (JPEG + gain map, ISO 21496-1 / Adobe XMP) output.
// The HDR rendition is the scene-linear image after colour correction, exposed by the tone mapper's
// long-exposure gain so that 1.0 = SDR white, with a soft roll-off to `headroom`.
constexpr int kGainMapBlock = 4;
constexpr float kMinBoost = 1.05f;  // below this peak gain no gain map is written

struct HdrLuma {
    ImageF y;        // block-averaged scene-linear luminance (Rec.709), kGainMapBlock x kGainMapBlock blocks
    int block = kGainMapBlock;
};
HdrLuma hdr_luma(const ImageF& linearRgb, int downscale);

struct UltraHdrStats { float maxBoost = 1, meanBoost = 1, boostedFrac = 0; size_t bytes = 0, gainMapBytes = 0; };
// the whole Ultra HDR JPEG file; empty on failure or when the peak gain is below kMinBoost
std::vector<uint8_t> encode_ultrahdr(const ImageF& sdr, int downscale, const HdrLuma& hdr, float exposure, float headroom,
                                     int quality, UltraHdrStats* st = nullptr);
bool write_ultrahdr(const std::string& path, const ImageF& sdr, int downscale, const HdrLuma& hdr, float exposure,
                    float headroom, int quality, UltraHdrStats* st = nullptr);
