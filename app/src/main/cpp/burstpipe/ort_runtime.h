#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "accel_backend.h"

// By-value IEEE-754 finiteness checks using volatile byte loads so -O3 -ffast-math
// (-ffinite-math-only) cannot fold bitmask checks on inlined/SSA float or double values.
inline bool is_finite_f32_val(float x) {
    const volatile float v = x;
    const volatile unsigned char* p = reinterpret_cast<const volatile unsigned char*>(&v);
    unsigned char buf[4];
    for (int i = 0; i < 4; ++i) buf[i] = p[i];
    uint32_t u = 0;
    std::memcpy(&u, buf, sizeof(u));
    return (u & 0x7F800000u) != 0x7F800000u;
}

inline bool is_finite_f64_val(double x) {
    const volatile double v = x;
    const volatile unsigned char* p = reinterpret_cast<const volatile unsigned char*>(&v);
    unsigned char buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = p[i];
    uint64_t u = 0;
    std::memcpy(&u, buf, sizeof(u));
    return (u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

// Fast bitwise IEEE-754 float32 finiteness check for externally-populated heap buffers
// (e.g. ORT output tensors). WARNING: Do NOT pass stack-local scalars or stack-local
// arrays under -ffast-math — LLVM value-tracking may fold the check to true; use
// is_finite_f32_val() / is_finite_f64_val() instead.
inline bool is_finite_f32_at(const float* p) {
    uint32_t u = 0;
    std::memcpy(&u, reinterpret_cast<const unsigned char*>(p), sizeof(u));
    return (u & 0x7F800000u) != 0x7F800000u;
}

inline bool all_finite_f32(const float* p, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!is_finite_f32_at(p + i)) return false;
    }
    return true;
}

// Dynamically loads ONNX Runtime at run time (`libonnxruntime.dylib` / `libonnxruntime.so`)
// via ORT_API_MANUAL_INIT + dlopen/dlsym("OrtGetApiBase") so there is no link-time dependency.
bool ort_runtime_available();
std::string ort_runtime_version();
std::vector<std::string> ort_available_providers();

class OrtSessionRunner {
public:
    ~OrtSessionRunner();
    OrtSessionRunner(const OrtSessionRunner&) = delete;
    OrtSessionRunner& operator=(const OrtSessionRunner&) = delete;

    static std::unique_ptr<OrtSessionRunner> create(
        const std::string& modelPath,
        AccelBackend backend,
        int threads,
        std::string* error = nullptr);

    bool run(
        const float* inputData,
        const int64_t* inputShape,
        int inputRank,
        std::vector<float>& outputData,
        std::vector<int64_t>& outputShape,
        std::string* error = nullptr);

    bool run_multi(
        const float* inputData,
        const int64_t* inputShape,
        int inputRank,
        std::vector<std::vector<float>>& outputsData,
        std::vector<std::vector<int64_t>>& outputsShape,
        std::string* error = nullptr);

    const std::string& active_ep() const { return activeEp_; }
    int intra_op_threads() const { return threads_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string activeEp_ = "CPU";
    int threads_ = 4;

    OrtSessionRunner() = default;
};
