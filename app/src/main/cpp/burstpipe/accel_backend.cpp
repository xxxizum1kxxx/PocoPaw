#include "accel_backend.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "ort_runtime.h"

#if defined(__linux__)
#include <unistd.h>
#endif

const char* accel_backend_name(AccelBackend b) {
    switch (b) {
        case AccelBackend::AUTO: return "auto";
        case AccelBackend::CPU: return "cpu";
        case AccelBackend::GPU_OPENCL: return "gpu";
        case AccelBackend::NPU_DSP: return "npu";
    }
    return "auto";
}

AccelBackend parse_accel_backend(int v) {
    switch (v) {
        case 1: return AccelBackend::CPU;
        case 2: return AccelBackend::GPU_OPENCL;
        case 3: return AccelBackend::NPU_DSP;
        default: return AccelBackend::AUTO;
    }
}

bool parse_accel_backend_str(const std::string& s, AccelBackend& out) {
    std::string lo = s;
    std::transform(lo.begin(), lo.end(), lo.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (lo == "auto") { out = AccelBackend::AUTO; return true; }
    if (lo == "cpu") { out = AccelBackend::CPU; return true; }
    if (lo == "gpu" || lo == "opencl" || lo == "gpu_opencl") { out = AccelBackend::GPU_OPENCL; return true; }
    if (lo == "npu" || lo == "dsp" || lo == "npu_dsp" || lo == "cdsp") { out = AccelBackend::NPU_DSP; return true; }
    return false;
}

std::vector<int> detect_big_prime_cpus(int maxCount) {
    std::vector<int> result;
    if (maxCount <= 0) return result;
#if defined(__linux__)
    struct CoreFreq {
        int cpu;
        long khz;
    };
    std::vector<CoreFreq> cores;
    for (int i = 0; i < 64; i++) {
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        std::ifstream f(path);
        long khz = 0;
        if (f >> khz && khz > 0) cores.push_back({i, khz});
    }
    if (cores.empty()) return result;
    std::sort(cores.begin(), cores.end(), [](const CoreFreq& a, const CoreFreq& b) {
        if (a.khz != b.khz) return a.khz > b.khz;
        return a.cpu > b.cpu;
    });
    const int n = std::min(maxCount, static_cast<int>(cores.size()));
    result.reserve(n);
    for (int i = 0; i < n; i++) result.push_back(cores[i].cpu);
#endif
    return result;
}

std::string format_ort_intra_op_affinities(const std::vector<int>& cpus, int threads) {
    if (threads <= 1 || cpus.empty()) return "";
    std::vector<int> sorted = cpus;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    std::string list;
    for (size_t i = 0; i < sorted.size(); i++) {
        if (i > 0) list += ',';
        list += std::to_string(sorted[i]);
    }
    std::string out;
    const int workerCount = threads - 1;
    for (int i = 0; i < workerCount; i++) {
        if (i > 0) out += ';';
        out += list;
    }
    return out;
}

ScopedCpuAffinity::ScopedCpuAffinity(const std::vector<int>& cpus) {
#if defined(__linux__)
    if (cpus.empty()) return;
    CPU_ZERO(&prevMask_);
    if (sched_getaffinity(0, sizeof(prevMask_), &prevMask_) != 0) return;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &mask);
    }
    if (sched_setaffinity(0, sizeof(mask), &mask) == 0) active_ = true;
#else
    (void)cpus;
#endif
}

ScopedCpuAffinity::~ScopedCpuAffinity() {
#if defined(__linux__)
    if (active_) sched_setaffinity(0, sizeof(prevMask_), &prevMask_);
#endif
}

namespace {

bool probe_opencl_presence(std::string& devOut) {
#if defined(__APPLE__)
    const char* path = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
    const char* path = "libOpenCL.so";
#endif
    void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        devOut = "unavailable";
        return false;
    }
    using ClGetPlatformIds = int (*)(uint32_t, void**, uint32_t*);
    using ClGetDeviceIds = int (*)(void*, uint64_t, uint32_t, void**, uint32_t*);
    using ClGetDeviceInfo = int (*)(void*, uint32_t, size_t, void*, size_t*);
    auto getPlats = reinterpret_cast<ClGetPlatformIds>(dlsym(lib, "clGetPlatformIDs"));
    auto getDevs = reinterpret_cast<ClGetDeviceIds>(dlsym(lib, "clGetDeviceIDs"));
    auto getInfo = reinterpret_cast<ClGetDeviceInfo>(dlsym(lib, "clGetDeviceInfo"));
    bool ok = false;
    if (getPlats && getDevs && getInfo) {
        void* plat = nullptr;
        uint32_t np = 0;
        if (getPlats(1, &plat, &np) == 0 && np > 0) {
            void* dev = nullptr;
            uint32_t nd = 0;
            // CL_DEVICE_TYPE_GPU = 4, CL_DEVICE_NAME = 0x102B
            if (getDevs(plat, 4, 1, &dev, &nd) == 0 && nd > 0) {
                char name[256] = {};
                if (getInfo(dev, 0x102B, sizeof(name) - 1, name, nullptr) == 0) devOut = name;
                else devOut = "OpenCL GPU";
                ok = true;
            }
        }
    }
    if (!ok && devOut.empty()) devOut = "no GPU device";
    dlclose(lib);
    return ok;
}

void probe_cdsp_fastrpc(AccelCapabilities& cap) {
#if defined(__ANDROID__)
    void* lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        cap.cdspFastRpcAvailable = false;
        cap.cdspNote = "libcdsprpc.so unavailable; custom HVX skel requires proprietary Hexagon SDK";
        return;
    }
    cap.cdspFastRpcAvailable = true;
    struct DspCapability { uint32_t domain, attribute, capability; };
    using HandleControl = int (*)(uint32_t, void*, uint32_t);
    auto control = reinterpret_cast<HandleControl>(dlsym(lib, "remote_handle_control"));
    if (control) {
        auto query = [&](uint32_t attr, uint32_t& dst) {
            DspCapability c{3, attr, 0};
            if (control(2, &c, sizeof(c)) == 0) dst = c.capability;
        };
        uint32_t upd = 0;
        query(1, upd);
        query(3, cap.cdspHvx128);
        query(6, cap.cdspArch);
        cap.cdspUnsignedPd = upd != 0;
    }
    cap.cdspNote = "FastRPC probed; custom cDSP skel compilation requires proprietary Hexagon SDK (unavailable)";
    dlclose(lib);
#else
    cap.cdspFastRpcAvailable = false;
    cap.cdspNote = "Hexagon cDSP unavailable on host; proprietary Hexagon SDK required for skel compilation";
#endif
}

}  // namespace

AccelCapabilities probe_accel_capabilities() {
    AccelCapabilities cap;
    cap.openclAvailable = probe_opencl_presence(cap.openclDevice);
    cap.ortAvailable = ort_runtime_available();
    if (cap.ortAvailable) {
        cap.ortVersion = ort_runtime_version();
        cap.ortProviders = ort_available_providers();
        cap.nnapiAvailable = std::find(cap.ortProviders.begin(), cap.ortProviders.end(), "NnapiExecutionProvider") != cap.ortProviders.end();
        cap.qnnAvailable = std::find(cap.ortProviders.begin(), cap.ortProviders.end(), "QNNExecutionProvider") != cap.ortProviders.end();
    }
    cap.qnnNote = cap.qnnAvailable
        ? "QNN execution provider available"
        : "QNN unavailable (proprietary Qualcomm QNN SDK not bundled; NPU_DSP uses NNAPI -> CPU fallback)";
    cap.bigPrimeCpuIds = detect_big_prime_cpus(4);
    cap.mlThreads = 4;
    probe_cdsp_fastrpc(cap);
    return cap;
}

std::string AccelCapabilities::summary() const {
    std::ostringstream os;
    os << "opencl gpu: " << (openclAvailable ? openclDevice : "unavailable (" + openclDevice + ")") << "\n";
    os << "onnx runtime: " << (ortAvailable ? (ortVersion.empty() ? "available" : ortVersion) : "unavailable") << "\n";
    os << "ort providers: ";
    if (ortProviders.empty()) {
        os << "none\n";
    } else {
        for (size_t i = 0; i < ortProviders.size(); i++) {
            if (i > 0) os << ", ";
            os << ortProviders[i];
        }
        os << "\n";
    }
    os << "nnapi ep: " << (nnapiAvailable ? "available" : "unavailable") << "\n";
    os << "qnn ep: " << qnnNote << "\n";
    os << "ml threads: " << mlThreads;
    if (!bigPrimeCpuIds.empty()) {
        os << " (big+prime cpus ";
        for (size_t i = 0; i < bigPrimeCpuIds.size(); i++) {
            if (i > 0) os << ",";
            os << bigPrimeCpuIds[i];
        }
        os << ")";
    }
    os << "\n";
    os << "cdsp fastrpc: " << (cdspFastRpcAvailable ? "available" : "unavailable");
    if (cdspFastRpcAvailable) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), " (arch 0x%x, HVX %u, unsigned PD %u)",
                      cdspArch, cdspHvx128, cdspUnsignedPd ? 1u : 0u);
        os << buf;
    }
    os << "\n";
    os << "hexagon sdk boundary: " << cdspNote;
    return os.str();
}
