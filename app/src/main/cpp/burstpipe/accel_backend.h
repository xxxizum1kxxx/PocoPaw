#pragma once
#include <cstdint>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

// Accelerator routing policy across burst image-processing and ML inference stages.
enum class AccelBackend {
    AUTO = 0,
    CPU = 1,
    GPU_OPENCL = 2,
    NPU_DSP = 3,
};

const char* accel_backend_name(AccelBackend b);
AccelBackend parse_accel_backend(int v);
bool parse_accel_backend_str(const std::string& s, AccelBackend& out);

// Dynamically detects top `maxCount` highest-frequency CPU cores on Linux/Android
// (on Snapdragon 7+ Gen 2 / marble: X2 prime + 3x A710 big cores = {7, 6, 5, 4}).
// Returns empty on macOS or systems without sysfs cpufreq entries.
std::vector<int> detect_big_prime_cpus(int maxCount = 4);

// Formats `session.intra_op_thread_affinities` with exact `threads - 1` semicolon-separated
// entries (ORT rule: worker threads = intra_op_num_threads - 1; caller thread uses ScopedCpuAffinity).
std::string format_ort_intra_op_affinities(const std::vector<int>& cpus, int threads);

// Pins the calling thread to `cpus` on Linux/Android during ORT Run(), restoring the previous mask
// on destruction so the calling thread also stays off little cores.
class ScopedCpuAffinity {
public:
    explicit ScopedCpuAffinity(const std::vector<int>& cpus);
    ~ScopedCpuAffinity();
    ScopedCpuAffinity(const ScopedCpuAffinity&) = delete;
    ScopedCpuAffinity& operator=(const ScopedCpuAffinity&) = delete;

private:
#if defined(__linux__)
    bool active_ = false;
    cpu_set_t prevMask_{};
#endif
};

struct AccelCapabilities {
    bool openclAvailable = false;
    std::string openclDevice;
    bool ortAvailable = false;
    std::string ortVersion;
    std::vector<std::string> ortProviders;
    bool nnapiAvailable = false;
    bool qnnAvailable = false;
    std::string qnnNote;
    bool cdspFastRpcAvailable = false;
    bool cdspUnsignedPd = false;
    uint32_t cdspArch = 0;
    uint32_t cdspHvx128 = 0;
    std::string cdspNote;
    int mlThreads = 4;
    std::vector<int> bigPrimeCpuIds;

    std::string summary() const;
};

AccelCapabilities probe_accel_capabilities();
