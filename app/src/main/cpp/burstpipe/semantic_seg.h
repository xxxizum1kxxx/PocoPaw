#pragma once
#include <string>
#include <vector>

#include "accel_backend.h"
#include "common.h"

struct SemanticMask {
    int w = 0, h = 0;
    std::vector<float> data;  // interleaved [face, sky] in [0, 1], size = w * h * 2
    bool valid() const { return w > 0 && h > 0 && data.size() == (size_t)w * h * 2; }
};

struct SemanticSegStats {
    bool ran = false;
    bool usedModel = false;
    int faceBoxCount = 0;
    float meanFace = 0.f, meanSky = 0.f;
    double ms = 0.0;
    std::string ep;
    std::string note;
};

// Resolves semantic_seg.onnx path. If explicitPath is non-empty, returns it only if readable
// (never substitutes a different file for an explicit missing path).
std::string resolve_seg_model(const std::string& explicitPath, const char* argv0 = nullptr);

// Maps normalized sensor-orientation face boxes [x0, y0, x1, y1, ...] into the active
// zoom + aspect crop ROI in [0, 1]^2. Out-of-bounds or inverted coordinates are clamped safely.
std::vector<float> map_face_boxes_to_roi(
    const std::vector<float>& fullBoxes,
    const Meta& m,
    int aspectX0,
    int aspectY0,
    int aspectW,
    int aspectH,
    int preAspectW,
    int preAspectH);

// Box-averages display-referred sRGB [0,1] image to interleaved 256x256x3 floats (matching GPU seg_downsample).
std::vector<float> downsample_rgb256(const ImageF& rgb);

// Runs TopFormer-T ONNX segmentation (rotated clockwise by orientationDegrees and restored CCW)
// and fuses smooth elliptical priors from mappedFaceBoxes into a 64x64 [face, sky] mask.
// Returns true iff outMask is valid (either ONNX model ran or mappedFaceBoxes is non-empty).
bool semantic_seg_from_rgb256(
    const std::vector<float>& rgb256,
    const std::string& modelPath,
    const std::vector<float>& mappedFaceBoxes,
    int orientationDegrees,
    AccelBackend backend,
    SemanticMask& outMask,
    SemanticSegStats* st = nullptr,
    const char* argv0 = nullptr);
