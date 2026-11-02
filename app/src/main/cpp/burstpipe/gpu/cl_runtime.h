#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// OpenCL is loaded with dlopen (Adreno's libOpenCL.so is a vendor library, not part of the NDK; on the
// Mac it is OpenCL.framework), so a phone without it falls back to the CPU instead of failing to load.
// Every failure inside a GPU stage throws GpuError; BurstSession catches it and reruns the stage on the CPU.
struct GpuError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Gpu;

// A device buffer from Gpu's pool; the storage goes back to the pool when the last copy is dropped.
class Buf {
public:
    Buf() = default;
    cl_mem mem() const { return p_ ? p_->mem : nullptr; }
    size_t bytes() const { return p_ ? p_->bytes : 0; }
    explicit operator bool() const { return p_ != nullptr; }
    void reset() { p_.reset(); }

private:
    friend class Gpu;
    struct Block {
        Gpu* gpu = nullptr;
        cl_mem mem = nullptr;
        size_t bytes = 0, cap = 0;
        ~Block();
    };
    std::shared_ptr<Block> p_;
};

// A float image on the device: w x h pixels of c interleaved channels.
struct GImg {
    Buf b;
    int w = 0, h = 0, c = 1;
    size_t count() const { return (size_t)w * h * c; }
};

// __local scratch argument of a kernel
struct LocalMem { size_t bytes; };

class Gpu {
public:
    // The process-wide device, created on first use; null when there is no usable OpenCL GPU.
    static Gpu* get();
    // Directory for compiled program binaries (the app's cacheDir); set before the first get().
    static void set_cache_dir(const std::string& dir);
    // Records per-kernel device time (queue profiling); for session_check and the developer log.
    static void set_profiling(bool on);
    // One "property: value" line per device property, or "unavailable: why".
    static std::string describe();
    // Returns pooled device memory to the driver, if the device was ever created (after a capture).
    static void release_memory();

    // Serialises bursts: a second session that finds the GPU busy runs on the CPU.
    std::unique_lock<std::mutex> try_acquire() { return std::unique_lock<std::mutex>(use_, std::try_to_lock); }

    Buf alloc(size_t bytes);
    GImg image(int w, int h, int c = 1) { GImg g; g.w = w; g.h = h; g.c = c; g.b = alloc(g.count() * sizeof(float)); return g; }
    void write(const Buf& b, const void* src, size_t bytes, size_t offset = 0);
    void read(const Buf& b, void* dst, size_t bytes, size_t offset = 0);
    // rows of `rowBytes` from a tightly packed buffer into memory with `dstStride` bytes per row
    void read_rows(const Buf& b, void* dst, size_t rowBytes, size_t rows, size_t dstStride);
    void zero(const Buf& b);
    template <class... A>
    void run(const char* kernel, size_t gx, size_t gy, const A&... args) {
        cl_kernel k = kernel_(kernel);
        cl_uint i = 0;
        (set_(k, i++, args), ...);
        size_t g[2] = {gx, gy};
        enqueue_(k, kernel, gy > 1 ? 2 : 1, g, nullptr);
    }
    // explicit work-group size; the global size is rounded up to a multiple of it
    template <class... A>
    void run_local(const char* kernel, size_t gx, size_t gy, size_t lx, size_t ly, const A&... args) {
        cl_kernel k = kernel_(kernel);
        cl_uint i = 0;
        (set_(k, i++, args), ...);
        size_t g[2] = {(gx + lx - 1) / lx * lx, (gy + ly - 1) / ly * ly}, l[2] = {lx, ly};
        enqueue_(k, kernel, 2, g, l);
    }
    void finish();
    // releases pooled device memory
    void trim();
    // "kernel ms, ..." of the profiled kernels since the last call, slowest first
    std::string take_profile();

    size_t max_alloc() const { return maxAlloc_; }
    const std::string& name() const { return name_; }

    ~Gpu();

private:
    Gpu() = default;
    bool init(std::string& why);
    bool build(std::string& why);
    cl_kernel kernel_(const char* name);
    void enqueue_(cl_kernel k, const char* name, cl_uint dims, const size_t* g, const size_t* l);
    void set_(cl_kernel k, cl_uint i, const Buf& b);
    void set_(cl_kernel k, cl_uint i, const GImg& g) { set_(k, i, g.b); }
    void set_(cl_kernel k, cl_uint i, const LocalMem& l);    void set_(cl_kernel k, cl_uint i, int v) { setv_(k, i, sizeof v, &v); }
    void set_(cl_kernel k, cl_uint i, unsigned v) { setv_(k, i, sizeof v, &v); }
    void set_(cl_kernel k, cl_uint i, float v) { setv_(k, i, sizeof v, &v); }
    void set_(cl_kernel k, cl_uint i, double v) { float f = (float)v; setv_(k, i, sizeof f, &f); }
    void set_(cl_kernel k, cl_uint i, const cl_float4& v) { setv_(k, i, sizeof v, &v); }
    void setv_(cl_kernel k, cl_uint i, size_t n, const void* v);
    void release_(Buf::Block* b);
    void collect_();
    friend struct Buf::Block;

    cl_context ctx_ = nullptr;
    cl_device_id dev_ = nullptr;
    cl_command_queue q_ = nullptr;
    cl_program prog_ = nullptr;
    std::map<std::string, cl_kernel> kernels_;
    std::multimap<size_t, cl_mem> pool_;
    std::mutex poolMu_, use_;
    size_t maxAlloc_ = 0;
    std::string name_, driver_;
    bool profiling_ = false;
    std::vector<std::pair<std::string, cl_event>> events_;
    std::map<std::string, double> profile_;
};

void check_cl(cl_int err, const char* what);
