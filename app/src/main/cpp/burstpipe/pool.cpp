#include <map>
#include <mutex>

#include "common.h"

namespace {
std::mutex g_poolMu;
std::multimap<size_t, float*> g_pool;
}  // namespace

float* pool_alloc(size_t n, size_t& cap) {
    {
        std::lock_guard<std::mutex> lk(g_poolMu);
        auto it = g_pool.lower_bound(n);
        if (it != g_pool.end() && it->first <= n + n / 4) {
            cap = it->first;
            float* p = it->second;
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
}

void pool_trim() {
    std::lock_guard<std::mutex> lk(g_poolMu);
    for (auto& e : g_pool) delete[] e.second;
    g_pool.clear();
}
