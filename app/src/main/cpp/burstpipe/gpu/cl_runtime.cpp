#include "cl_runtime.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

#define CL_FUNCS(X)                                                                                                    \
    X(clGetPlatformIDs) X(clGetPlatformInfo) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext)                  \
    X(clReleaseContext) X(clCreateCommandQueue) X(clReleaseCommandQueue) X(clCreateBuffer) X(clCreateImage)           \
    X(clReleaseMemObject) X(clCreateProgramWithSource) X(clCreateProgramWithBinary) X(clBuildProgram)                 \
    X(clGetProgramBuildInfo) X(clGetProgramInfo) X(clReleaseProgram) X(clCreateKernel) X(clCreateKernelsInProgram)    \
    X(clGetKernelInfo) X(clGetKernelWorkGroupInfo) X(clReleaseKernel) X(clSetKernelArg) X(clEnqueueNDRangeKernel)     \
    X(clEnqueueReadBuffer) X(clEnqueueWriteBuffer) X(clEnqueueReadBufferRect) X(clEnqueueMapBuffer)                   \
    X(clEnqueueUnmapMemObject) X(clEnqueueFillBuffer) X(clFlush) X(clFinish) X(clGetEventProfilingInfo)               \
    X(clReleaseEvent)

struct Api {
#define X(f) decltype(&::f) f = nullptr;
    CL_FUNCS(X)
#undef X
    void* lib = nullptr;
    bool load(std::string& why) {
        const char* paths[] = {
#ifdef __APPLE__
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
#else
            "libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so",
#endif
        };
        for (const char* p : paths)
            if ((lib = dlopen(p, RTLD_NOW | RTLD_LOCAL))) break;
        if (!lib) { why = std::string("no OpenCL library: ") + dlerror(); return false; }
#define X(f)                                                                    \
    f = reinterpret_cast<decltype(&::f)>(dlsym(lib, #f));                       \
    if (!f) { why = "OpenCL library lacks " #f; return false; }
        CL_FUNCS(X)
#undef X
        return true;
    }
} cl;

const char* const kSource[] = {
#include "kernels/common.cl.inc"
    ,
#include "kernels/align.cl.inc"
    ,
#include "kernels/merge.cl.inc"
    ,
#include "kernels/finish.cl.inc"
    ,
#include "kernels/demosaic.cl.inc"
};
const char* const kBuildOptions = "-cl-mad-enable";

std::string g_cacheDir;
bool g_profiling = false;
std::mutex g_initMu;
Gpu* g_gpu = nullptr;
bool g_tried = false;
std::string g_why;

std::string dev_str(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (cl.clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "";
    std::string s(n, '\0');
    cl.clGetDeviceInfo(d, what, n, s.data(), nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

template <class T>
T dev_val(cl_device_id d, cl_device_info what) {
    T v{};
    cl.clGetDeviceInfo(d, what, sizeof v, &v, nullptr);
    return v;
}

uint64_t fnv1a(const std::string& s, uint64_t h = 1469598103934665603ull) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

}  // namespace

void check_cl(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        char buf[160];
        snprintf(buf, sizeof buf, "OpenCL error %d in %s", err, what);
        throw GpuError(buf);
    }
}

Buf::Block::~Block() {
    if (gpu && mem) gpu->release_(this);
}

Img2D::Block::~Block() {
    if (gpu && mem) gpu->release_img_(this);
}

void Gpu::set_cache_dir(const std::string& dir) { g_cacheDir = dir; }
void Gpu::set_profiling(bool on) { g_profiling = on; }

void Gpu::release_memory() {
    Gpu* g;
    {
        std::lock_guard<std::mutex> lk(g_initMu);
        g = g_gpu;
    }
    if (g) g->trim();
}

void Gpu::trim_excess(size_t maxBytes) {
    Gpu* g;
    {
        std::lock_guard<std::mutex> lk(g_initMu);
        g = g_gpu;
    }
    if (g) g->trim_to(maxBytes);
}

Gpu* Gpu::get() {
    std::lock_guard<std::mutex> lk(g_initMu);
    if (g_tried) return g_gpu;
    g_tried = true;
    Gpu* g = new Gpu();
    if (!g->init(g_why)) { delete g; return nullptr; }
    g_gpu = g;
    return g_gpu;
}

std::string Gpu::describe() {
    Gpu* g = get();
    if (!g) return "unavailable: " + g_why;
    cl_device_id d = g->dev_;
    std::ostringstream o;
    o << "device: " << g->name_ << "\n";
    o << "version: " << dev_str(d, CL_DEVICE_VERSION) << "\n";
    o << "driver: " << g->driver_ << "\n";
    o << "compute units: " << dev_val<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS) << "\n";
    o << "max clock MHz: " << dev_val<cl_uint>(d, CL_DEVICE_MAX_CLOCK_FREQUENCY) << "\n";
    o << "max work group: " << dev_val<size_t>(d, CL_DEVICE_MAX_WORK_GROUP_SIZE) << "\n";
    o << "local memory KB: " << dev_val<cl_ulong>(d, CL_DEVICE_LOCAL_MEM_SIZE) / 1024 << "\n";
    o << "global memory MB: " << dev_val<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE) / (1 << 20) << "\n";
    o << "max allocation MB: " << g->maxAlloc_ / (1 << 20) << "\n";
    o << "image support: " << (g->imageSupport_ ? "yes" : "no") << "\n";
    std::string ext = dev_str(d, CL_DEVICE_EXTENSIONS), qcom;
    std::istringstream es(ext);
    for (std::string e; es >> e;) if (e.rfind("cl_qcom_", 0) == 0) qcom += (qcom.empty() ? "" : " ") + e;
    o << "fp16: " << (ext.find("cl_khr_fp16") != std::string::npos ? "yes" : "no") << "\n";
    o << "qcom extensions: " << (qcom.empty() ? "none" : qcom);
    return o.str();
}

bool Gpu::init(std::string& why) {
    if (!cl.load(why)) return false;
    cl_uint np = 0;
    if (cl.clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) { why = "no OpenCL platform"; return false; }
    std::vector<cl_platform_id> ps(np);
    cl.clGetPlatformIDs(np, ps.data(), nullptr);
    for (cl_platform_id p : ps) {
        cl_uint nd = 0;
        if (cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev_, &nd) == CL_SUCCESS && nd > 0) break;
        dev_ = nullptr;
    }
    if (!dev_) { why = "no OpenCL GPU device"; return false; }
    name_ = dev_str(dev_, CL_DEVICE_NAME);
    driver_ = dev_str(dev_, CL_DRIVER_VERSION);
    maxAlloc_ = (size_t)dev_val<cl_ulong>(dev_, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    imageSupport_ = dev_val<cl_bool>(dev_, CL_DEVICE_IMAGE_SUPPORT) == CL_TRUE;
    if (dev_val<size_t>(dev_, CL_DEVICE_MAX_WORK_GROUP_SIZE) < 256) { why = "work groups smaller than 256"; return false; }

    cl_int err = 0;
    // cl_qcom_perf_hint + cl_qcom_priority_hint: request highest Adreno clock & priority immediately
    const cl_context_properties kPerfHint = 0x40C2, kPerfHigh = 0x40C3;
    const cl_context_properties kPrioHint = 0x40C9, kPrioHigh = 0x40CA;
    std::string ext = dev_str(dev_, CL_DEVICE_EXTENSIONS);
    bool hasPerf = ext.find("cl_qcom_perf_hint") != std::string::npos;
    bool hasPrio = ext.find("cl_qcom_priority_hint") != std::string::npos;
    std::vector<cl_context_properties> props;
    if (hasPerf) { props.push_back(kPerfHint); props.push_back(kPerfHigh); }
    if (hasPrio) { props.push_back(kPrioHint); props.push_back(kPrioHigh); }
    props.push_back(0);

    ctx_ = cl.clCreateContext(props.size() > 1 ? props.data() : nullptr, 1, &dev_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS && hasPerf) {
        cl_context_properties perfOnly[] = {kPerfHint, kPerfHigh, 0};
        ctx_ = cl.clCreateContext(perfOnly, 1, &dev_, nullptr, nullptr, &err);
    }
    if (err != CL_SUCCESS) ctx_ = cl.clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { why = "clCreateContext failed " + std::to_string(err); return false; }

    profiling_ = g_profiling;
    q_ = cl.clCreateCommandQueue(ctx_, dev_, profiling_ ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    if (err != CL_SUCCESS) { why = "clCreateCommandQueue failed " + std::to_string(err); return false; }
    return build(why);
}

void Gpu::precreate_kernels_() {
    cl_uint nk = 0;
    if (cl.clCreateKernelsInProgram(prog_, 0, nullptr, &nk) != CL_SUCCESS || nk == 0) return;
    std::vector<cl_kernel> ks(nk);
    if (cl.clCreateKernelsInProgram(prog_, nk, ks.data(), nullptr) != CL_SUCCESS) return;
    for (cl_kernel k : ks) {
        size_t n = 0;
        if (cl.clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &n) == CL_SUCCESS && n > 0) {
            std::string name(n, '\0');
            cl.clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, n, name.data(), nullptr);
            while (!name.empty() && name.back() == '\0') name.pop_back();
            size_t maxWg = 0;
            cl.clGetKernelWorkGroupInfo(k, dev_, CL_KERNEL_WORK_GROUP_SIZE, sizeof maxWg, &maxWg, nullptr);
            kernels_[name] = {k, maxWg};
        } else {
            cl.clReleaseKernel(k);
        }
    }
}

bool Gpu::build(std::string& why) {
    std::string src;
    for (const char* s : kSource) src += s;
    std::string cacheFile;
    if (!g_cacheDir.empty()) {
        char hex[17];
        snprintf(hex, sizeof hex, "%016llx", (unsigned long long)fnv1a(name_ + driver_ + kBuildOptions, fnv1a(src)));
        cacheFile = g_cacheDir + "/burstpipe-" + hex + ".clbin";
    }
    cl_int err = 0;
    if (!cacheFile.empty()) {
        std::ifstream f(cacheFile, std::ios::binary);
        std::vector<unsigned char> bin((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (!bin.empty()) {
            const unsigned char* p = bin.data();
            size_t n = bin.size();
            cl_int st = 0;
            prog_ = cl.clCreateProgramWithBinary(ctx_, 1, &dev_, &n, &p, &st, &err);
            if (err == CL_SUCCESS && st == CL_SUCCESS && cl.clBuildProgram(prog_, 1, &dev_, kBuildOptions, nullptr, nullptr) == CL_SUCCESS) {
                precreate_kernels_();
                return true;
            }
            if (prog_) cl.clReleaseProgram(prog_);
            prog_ = nullptr;
        }
    }
    const char* s = src.c_str();
    prog_ = cl.clCreateProgramWithSource(ctx_, 1, &s, nullptr, &err);
    if (err != CL_SUCCESS) { why = "clCreateProgramWithSource failed"; return false; }
    if (cl.clBuildProgram(prog_, 1, &dev_, kBuildOptions, nullptr, nullptr) != CL_SUCCESS) {
        size_t n = 0;
        cl.clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        cl.clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        why = "kernel build failed: " + log.substr(0, 2000);
        fprintf(stderr, "%s\n", why.c_str());
        return false;
    }
    if (!cacheFile.empty()) {
        size_t n = 0;
        if (cl.clGetProgramInfo(prog_, CL_PROGRAM_BINARY_SIZES, sizeof n, &n, nullptr) == CL_SUCCESS && n > 0) {
            std::vector<unsigned char> bin(n);
            unsigned char* p = bin.data();
            if (cl.clGetProgramInfo(prog_, CL_PROGRAM_BINARIES, sizeof p, &p, nullptr) == CL_SUCCESS) {
                std::string tmp = cacheFile + ".tmp";
                std::ofstream f(tmp, std::ios::binary);
                f.write((const char*)bin.data(), (std::streamsize)n);
                f.close();
                if (f) std::rename(tmp.c_str(), cacheFile.c_str());
            }
        }
    }
    precreate_kernels_();
    return true;
}

Gpu::~Gpu() {
    trim();
    for (auto& k : kernels_) cl.clReleaseKernel(k.second.k);
    if (prog_) cl.clReleaseProgram(prog_);
    if (q_) cl.clReleaseCommandQueue(q_);
    if (ctx_) cl.clReleaseContext(ctx_);
}

Gpu::KernelInfo Gpu::kernel_(const char* name) {
    auto it = kernels_.find(name);
    if (it != kernels_.end()) return it->second;
    cl_int err = 0;
    cl_kernel k = cl.clCreateKernel(prog_, name, &err);
    check_cl(err, name);
    size_t maxWg = 0;
    cl.clGetKernelWorkGroupInfo(k, dev_, CL_KERNEL_WORK_GROUP_SIZE, sizeof maxWg, &maxWg, nullptr);
    KernelInfo ki{k, maxWg};
    kernels_[name] = ki;
    return ki;
}

void Gpu::setv_(cl_kernel k, cl_uint i, size_t n, const void* v) { check_cl(cl.clSetKernelArg(k, i, n, v), "clSetKernelArg"); }

void Gpu::set_(cl_kernel k, cl_uint i, const Buf& b) {
    cl_mem m = b.mem();
    check_cl(cl.clSetKernelArg(k, i, sizeof m, &m), "clSetKernelArg(buffer)");
}

void Gpu::set_(cl_kernel k, cl_uint i, const Img2D& im) {
    cl_mem m = im.mem();
    check_cl(cl.clSetKernelArg(k, i, sizeof m, &m), "clSetKernelArg(image2d)");
}

void Gpu::set_(cl_kernel k, cl_uint i, const LocalMem& l) { check_cl(cl.clSetKernelArg(k, i, l.bytes, nullptr), "clSetKernelArg(local)"); }

void Gpu::enqueue_(cl_kernel k, const char* name, cl_uint dims, const size_t* g, const size_t* l) {
    if (g[0] == 0 || (dims > 1 && g[1] == 0)) return;
    cl_event ev = nullptr;
    check_cl(cl.clEnqueueNDRangeKernel(q_, k, dims, nullptr, g, l, 0, nullptr, profiling_ ? &ev : nullptr), name);
    if (ev) events_.emplace_back(name, ev);
}

Buf Gpu::alloc_impl_(size_t bytes, bool hostAlloc) {
    bytes = std::max<size_t>(bytes, 4);
    if (bytes > maxAlloc_) throw GpuError("allocation of " + std::to_string(bytes >> 20) + " MB exceeds the device limit");
    Buf b;
    b.p_ = std::make_shared<Buf::Block>();
    b.p_->gpu = this;
    b.p_->bytes = bytes;
    b.p_->hostAlloc = hostAlloc;
    {
        std::lock_guard<std::mutex> lk(poolMu_);
        auto& pl = hostAlloc ? hostPool_ : pool_;
        auto it = pl.lower_bound(bytes);
        if (it != pl.end() && it->first <= bytes + bytes / 4) {
            b.p_->cap = it->first;
            b.p_->mem = it->second;
            poolBytes_ -= it->first;
            pl.erase(it);
            return b;
        }
    }
    cl_int err = 0;
    cl_mem_flags flags = CL_MEM_READ_WRITE | (hostAlloc ? CL_MEM_ALLOC_HOST_PTR : 0);
    b.p_->mem = cl.clCreateBuffer(ctx_, flags, bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        trim();
        b.p_->mem = cl.clCreateBuffer(ctx_, flags, bytes, nullptr, &err);
    }
    if (err != CL_SUCCESS) b.p_->mem = nullptr;
    check_cl(err, "clCreateBuffer");
    b.p_->cap = bytes;
    return b;
}

Buf Gpu::alloc(size_t bytes) { return alloc_impl_(bytes, false); }
Buf Gpu::alloc_host(size_t bytes) { return alloc_impl_(bytes, true); }

Img2D Gpu::image2d(int w, int h) {
    if (!imageSupport_ || w <= 0 || h <= 0) throw GpuError("OpenCL 2D images unsupported");
    uint64_t key = ((uint64_t)(uint32_t)w << 32) | (uint32_t)h;
    Img2D im;
    im.p_ = std::make_shared<Img2D::Block>();
    im.p_->gpu = this;
    im.p_->w = w;
    im.p_->h = h;
    {
        std::lock_guard<std::mutex> lk(poolMu_);
        auto it = imgPool_.find(key);
        if (it != imgPool_.end()) {
            im.p_->mem = it->second;
            poolBytes_ -= (size_t)w * h * 16;
            imgPool_.erase(it);
            return im;
        }
    }
    cl_image_format fmt{CL_RGBA, CL_FLOAT};
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)w;
    desc.image_height = (size_t)h;
    cl_int err = 0;
    im.p_->mem = cl.clCreateImage(ctx_, CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS) {
        trim();
        im.p_->mem = cl.clCreateImage(ctx_, CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &err);
    }
    if (err != CL_SUCCESS) im.p_->mem = nullptr;
    check_cl(err, "clCreateImage");
    return im;
}

void Gpu::release_(Buf::Block* b) {
    if (b->mapped) {
        cl.clEnqueueUnmapMemObject(q_, b->mem, b->mapped, 0, nullptr, nullptr);
        b->mapped = nullptr;
    }
    std::lock_guard<std::mutex> lk(poolMu_);
    (b->hostAlloc ? hostPool_ : pool_).emplace(b->cap, b->mem);
    poolBytes_ += b->cap;
    b->mem = nullptr;
}

void Gpu::release_img_(Img2D::Block* b) {
    uint64_t key = ((uint64_t)(uint32_t)b->w << 32) | (uint32_t)b->h;
    std::lock_guard<std::mutex> lk(poolMu_);
    imgPool_.emplace(key, b->mem);
    poolBytes_ += (size_t)b->w * b->h * 16;
    b->mem = nullptr;
}

void Gpu::trim() { trim_to(0); }

void Gpu::trim_to(size_t maxBytes) {
    std::lock_guard<std::mutex> lk(poolMu_);
    if (maxBytes == 0) {
        for (auto& e : pool_) cl.clReleaseMemObject(e.second);
        for (auto& e : hostPool_) cl.clReleaseMemObject(e.second);
        for (auto& e : imgPool_) cl.clReleaseMemObject(e.second);
        pool_.clear();
        hostPool_.clear();
        imgPool_.clear();
        poolBytes_ = 0;
        return;
    }
    while (poolBytes_ > maxBytes && (!pool_.empty() || !hostPool_.empty())) {
        bool useHost = pool_.empty() || (!hostPool_.empty() && std::prev(hostPool_.end())->first > std::prev(pool_.end())->first);
        auto& pl = useHost ? hostPool_ : pool_;
        auto it = std::prev(pl.end());
        cl.clReleaseMemObject(it->second);
        poolBytes_ -= it->first;
        pl.erase(it);
    }
    while (poolBytes_ > maxBytes && !imgPool_.empty()) {
        auto it = std::prev(imgPool_.end());
        int w = (int)(it->first >> 32), h = (int)(uint32_t)it->first;
        cl.clReleaseMemObject(it->second);
        size_t sz = (size_t)w * h * 16;
        poolBytes_ = poolBytes_ > sz ? poolBytes_ - sz : 0;
        imgPool_.erase(it);
    }
}

void Gpu::write(const Buf& b, const void* src, size_t bytes, size_t offset) {
    check_cl(cl.clEnqueueWriteBuffer(q_, b.mem(), CL_TRUE, offset, bytes, src, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
}

void Gpu::read(const Buf& b, void* dst, size_t bytes, size_t offset) {
    check_cl(cl.clEnqueueReadBuffer(q_, b.mem(), CL_TRUE, offset, bytes, dst, 0, nullptr, nullptr), "clEnqueueReadBuffer");
}

void Gpu::read_async(const Buf& b, void* dst, size_t bytes, size_t offset) {
    check_cl(cl.clEnqueueReadBuffer(q_, b.mem(), CL_FALSE, offset, bytes, dst, 0, nullptr, nullptr), "clEnqueueReadBuffer(async)");
}

void* Gpu::map_read(const Buf& b) {
    if (!b) return nullptr;
    if (b.p_->mapped) return b.p_->mapped;
    cl_int err = 0;
    void* p = cl.clEnqueueMapBuffer(q_, b.mem(), CL_TRUE, CL_MAP_READ, 0, b.bytes(), 0, nullptr, nullptr, &err);
    check_cl(err, "clEnqueueMapBuffer");
    b.p_->mapped = p;
    if (profiling_) collect_();
    return p;
}

void Gpu::unmap(const Buf& b) {
    if (!b || !b.p_->mapped) return;
    check_cl(cl.clEnqueueUnmapMemObject(q_, b.mem(), b.p_->mapped, 0, nullptr, nullptr), "clEnqueueUnmapMemObject");
    b.p_->mapped = nullptr;
}

void Gpu::read_rows(const Buf& b, void* dst, size_t rowBytes, size_t rows, size_t dstStride) {
    if (dstStride == rowBytes) { read(b, dst, rowBytes * rows); return; }
    size_t org[3] = {0, 0, 0}, region[3] = {rowBytes, rows, 1};
    check_cl(cl.clEnqueueReadBufferRect(q_, b.mem(), CL_TRUE, org, org, region, rowBytes, 0, dstStride, 0, dst, 0, nullptr, nullptr),
             "clEnqueueReadBufferRect");
}

void Gpu::zero(const Buf& b) {
    float z = 0;
    check_cl(cl.clEnqueueFillBuffer(q_, b.mem(), &z, sizeof z, 0, b.bytes() / 4 * 4, 0, nullptr, nullptr), "clEnqueueFillBuffer");
}

void Gpu::flush() {
    check_cl(cl.clFlush(q_), "clFlush");
}

void Gpu::finish() {
    check_cl(cl.clFinish(q_), "clFinish");
    collect_();
}

void Gpu::collect_() {
    for (auto& e : events_) {
        cl_ulong t0 = 0, t1 = 0;
        cl.clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_START, sizeof t0, &t0, nullptr);
        cl.clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_END, sizeof t1, &t1, nullptr);
        profile_[e.first] += (t1 - t0) * 1e-6;
        cl.clReleaseEvent(e.second);
    }
    events_.clear();
}

std::string Gpu::take_profile() {
    finish();
    std::vector<std::pair<double, std::string>> v;
    for (auto& p : profile_) v.emplace_back(p.second, p.first);
    profile_.clear();
    std::sort(v.rbegin(), v.rend());
    std::string s;
    char buf[96];
    for (auto& p : v) {
        snprintf(buf, sizeof buf, "%s%s %.1f", s.empty() ? "" : ", ", p.second.c_str(), p.first);
        s += buf;
    }
    return s;
}
