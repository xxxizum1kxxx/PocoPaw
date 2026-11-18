#pragma once
#include <string>

#include "accel_backend.h"
#include "align.h"
#include "common.h"

enum class FlowMethod { BLOCK_MATCH = 0, SEA_RAFT = 1, AUTO = 2 };

inline const char* flow_method_name(FlowMethod m) {
    switch (m) {
        case FlowMethod::SEA_RAFT: return "searaft";
        case FlowMethod::AUTO: return "auto";
        default: return "block_match";
    }
}

inline FlowMethod parse_flow_method(int v) {
    if (v == 1) return FlowMethod::SEA_RAFT;
    if (v == 2) return FlowMethod::AUTO;
    return FlowMethod::BLOCK_MATCH;
}

inline bool parse_flow_method_str(const std::string& s, FlowMethod& out) {
    if (s == "block_match" || s == "bm") { out = FlowMethod::BLOCK_MATCH; return true; }
    if (s == "searaft" || s == "sea_raft") { out = FlowMethod::SEA_RAFT; return true; }
    if (s == "auto") { out = FlowMethod::AUTO; return true; }
    return false;
}

struct FlowStats {
    int raftFrames = 0;
    int totalFrames = 0;
    double raftMs = 0.0;
    float maxGlobRatio = 0.f;
    float maxP999Ratio = 0.f;
    std::string ep;
    std::string note;
};

// Resolves sea_raft.onnx path. If explicitPath is non-empty, returns it only if readable
// (never substitutes a different file for an explicit missing path).
std::string resolve_flow_model(const std::string& explicitPath, const char* argv0 = nullptr);

// Residual detector on tile-aligned gray images; returns true if block matching left large
// global or localized unaligned motion warranting SEA-RAFT refinement.
bool assess_alignment_residual(
    const ImageF& grayRef,
    const ImageF& grayAlt,
    const Offsets& offs,
    float black,
    float* outGlobRatio = nullptr,
    float* outP999Ratio = nullptr,
    float* outOutlierFrac = nullptr);

// Estimates tile Offsets via SEA-RAFT ONNX optical flow followed by l==0 (R=2) L1 + parabolic
// subpixel snapping. Returns true iff outOffs was populated; false leaves outOffs untouched.
bool estimate_sea_raft_offsets(
    const ImageF& grayRef,
    const ImageF& grayAlt,
    const ImageF& mean3Ref,
    const ImageF& mean3Alt,
    float black,
    float white,
    const std::string& modelPath,
    AccelBackend backend,
    Offsets& outOffs,
    std::string* outEp = nullptr,
    std::string* outNote = nullptr,
    const char* argv0 = nullptr);
