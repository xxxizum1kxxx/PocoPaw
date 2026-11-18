#include <map>
#include <mutex>

#include "common.h"

namespace {
std::mutex g_poolMu;
std::multimap<size_t, float*> g_pool;
size_t g_poolBytes = 0;
}  // namespace

float* pool_alloc(size_t n, size_t& cap) {
    {
        std::lock_guard<std::mutex> lk(g_poolMu);
        auto it = g_pool.lower_bound(n);
        if (it != g_pool.end() && it->first <= n + n / 4) {
            cap = it->first;
            float* p = it->second;
            g_poolBytes -= cap * sizeof(float);
            g_pool.erase(it);
            return p;
        }
    }
    cap = n;
    return new float[n];
}

void pool_free(float* p, size_t cap) {
    if (cap < (1u << 16)) { delete[] p; return; }
    std::lock_guard<std::mutex> lk(g_poolMu);
    g_pool.emplace(cap, p);
    g_poolBytes += cap * sizeof(float);
}

void pool_trim_excess(size_t maxBytes) {
    std::lock_guard<std::mutex> lk(g_poolMu);
    while (g_poolBytes > maxBytes && !g_pool.empty()) {
        auto it = std::prev(g_pool.end());
        g_poolBytes -= it->first * sizeof(float);
        delete[] it->second;
        g_pool.erase(it);
    }
}

void pool_trim() {
    pool_trim_excess(0);
}
