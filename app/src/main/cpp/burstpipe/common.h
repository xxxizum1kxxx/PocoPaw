#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern int g_threads;

template <class F>
void parallel_for(int n, F&& f) {
    int nt = std::max(1, std::min(g_threads, n));
    if (nt == 1) { for (int i = 0; i < n; i++) f(i); return; }
    std::atomic<int> next(0);
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (int t = 0; t < nt; t++)
        ts.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& t : ts) t.join();
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
};

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Large float buffers are recycled through a process-wide pool. On Android, scudo returns big
// blocks to the kernel on free, so every new image would page-fault again, and on this 5.10
// kernel concurrent faults serialise on mmap_lock. Storage is left uninitialised.
float* pool_alloc(size_t n, size_t& cap);
void pool_free(float* p, size_t cap);
// Returns every pooled buffer to the system; the app calls it when a capture is done.
void pool_trim();

struct FloatBuf {
    float* p = nullptr;
    size_t n = 0, cap = 0;
    FloatBuf() = default;
    explicit FloatBuf(size_t n_) : n(n_) { if (n) p = pool_alloc(n, cap); }
    FloatBuf(const FloatBuf& o) : FloatBuf(o.n) { std::copy(o.p, o.p + n, p); }
    FloatBuf& operator=(const FloatBuf& o) { if (this != &o) { FloatBuf t(o); swap(t); } return *this; }
    FloatBuf(FloatBuf&& o) noexcept { swap(o); }
    FloatBuf& operator=(FloatBuf&& o) noexcept { FloatBuf t(std::move(o)); swap(t); return *this; }
    ~FloatBuf() { if (p) pool_free(p, cap); }
    void swap(FloatBuf& o) noexcept { std::swap(p, o.p); std::swap(n, o.n); std::swap(cap, o.cap); }
    float* data() { return p; }
    const float* data() const { return p; }
    size_t size() const { return n; }
    float& operator[](size_t i) { return p[i]; }
    float operator[](size_t i) const { return p[i]; }
    float* begin() { return p; }
    float* end() { return p + n; }
    const float* begin() const { return p; }
    const float* end() const { return p + n; }
};

struct ImageF {
    int w = 0, h = 0, c = 1;
    FloatBuf d;
    ImageF() = default;
    ImageF(int w_, int h_, int c_ = 1) : w(w_), h(h_), c(c_), d((size_t)w_ * h_ * c_) {}
    float* row(int y) { return d.data() + (size_t)y * w * c; }
    const float* row(int y) const { return d.data() + (size_t)y * w * c; }
    float& at(int x, int y, int k = 0) { return d[((size_t)y * w + x) * c + k]; }
    float at(int x, int y, int k = 0) const { return d[((size_t)y * w + x) * c + k]; }
    float atc(int x, int y, int k = 0) const { return at(clampi(x, 0, w - 1), clampi(y, 0, h - 1), k); }
};

// A view of one tightly packed RAW16 frame; the samples are owned by the caller (a direct
// ByteBuffer on Android) and must outlive every use of the view.
struct Raw {
    int w = 0, h = 0;
    const uint16_t* d = nullptr;
    uint16_t at(int x, int y) const { return d[(size_t)y * w + x]; }
};

struct Meta {
    int width = 0, height = 0, rowStride = 0, cfa = 3;
    float black[4] = {64, 64, 64, 64};
    float white = 1023;
    float wb[4] = {1, 1, 1, 1};  // R, G_even, G_odd, B
    float ccm[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    int lscRows = 0, lscCols = 0;
    std::vector<float> lsc;  // rows*cols*4, order R, G_even, G_odd, B
    float ev = 0;
    std::vector<double> exposureNs;
    std::vector<int> iso;
};

// Bayer channel index (0=R,1=G_even,2=G_odd,3=B) at (x,y) for a Camera2 CFA enum
inline int cfa_channel(int cfa, int x, int y) {
    // layouts row-major for the 2x2 cell: RGGB, GRBG, GBRG, BGGR
    static const int L[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {1, 3, 0, 2}, {3, 1, 2, 0}};
    int ch = L[cfa][(y & 1) * 2 + (x & 1)];
    // green on even rows is G_even, on odd rows G_odd
    if (ch == 1 || ch == 2) ch = (y & 1) ? 2 : 1;
    return ch;
}
// 0=R,1=G,2=B
inline int cfa_color(int cfa, int x, int y) {
    int ch = cfa_channel(cfa, x, y);
    return ch == 0 ? 0 : (ch == 3 ? 2 : 1);
}