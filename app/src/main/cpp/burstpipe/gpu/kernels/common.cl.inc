R"CL(
// Shared helpers. Every kernel guards its own range so global sizes may be rounded up to Wave128 work-groups.
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t kLinearClamp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t kNearestClamp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant int kCfaLayout[16] = {0, 1, 2, 3, 1, 0, 3, 2, 1, 3, 0, 2, 3, 1, 2, 0};

// Bayer channel index (0 R, 1 G_even, 2 G_odd, 3 B), as cfa_channel() in common.h
inline int cfa_channel(int cfa, int x, int y) {
    int ch = kCfaLayout[cfa * 4 + (y & 1) * 2 + (x & 1)];
    if (ch == 1 || ch == 2) ch = (y & 1) ? 2 : 1;
    return ch;
}
inline int cfa_color(int cfa, int x, int y) {
    int ch = cfa_channel(cfa, x, y);
    return ch == 0 ? 0 : (ch == 3 ? 2 : 1);
}
inline float luma709(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }
inline float at_clamped(__global const float* im, int w, int h, int x, int y) {
    return im[clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)];
}
// raised-cosine window of the overlapping alignment tiles: win[u] for u in [0, 16)
inline void tile_taps(int q, int n, __constant float* win, int* i0, int* i1, float* w0, float* w1) {
    int b = q / 8, u = q - b * 8;
    *i0 = clamp(b - 1, 0, n - 1);
    *i1 = clamp(b, 0, n - 1);
    *w0 = win[u + 8];
    *w1 = win[u];
}

// Hardware SFU sRGB OETF / EOTF (replaces random global LUT gathers while staying within < 1e-6 of exact pow)
inline float srgb_oetf_cl(float x) {
    x = clamp(x, 0.f, 1.f);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * native_powr(x, 1.f / 2.4f) - 0.055f;
}
inline float srgb_eotf_cl(float y) {
    y = clamp(y, 0.f, 1.f);
    return y <= 0.04045f ? y * (1.f / 12.92f) : native_powr((y + 0.055f) * (1.f / 1.055f), 2.4f);
}

// Sum over a 1D work group (every item must call it).
inline float2 group_sum2(__local float2* red, float2 v) {
    int l = get_local_id(0);
    red[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = (int)get_local_size(0) / 2; s > 0; s >>= 1) {
        if (l < s) red[l] += red[l + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return red[0];
}

// partial sums over a w x h float image (when cap > 0, clamps each sample to <= cap)
__kernel void row_sums(__global const float* im, int ioff, int w, int h, float cap, __global float* out, int ooff,
                       __local float2* red) {
    int x = get_global_id(0), y = get_global_id(1);
    float v = 0.f;
    if (x < w && y < h) {
        v = im[ioff + y * w + x];
        if (cap > 0.f) v = min(v, cap);
    }
    float s = group_sum2(red, (float2)(v, 0)).x;
    if (get_local_id(0) == 0 && y < h) out[ooff + y * get_num_groups(0) + get_group_id(0)] = s;
}

__kernel void row_sums_half(__global const half* im, int ioff, int w, int h, float cap, __global float* out, int ooff,
                            __local float2* red) {
    int x = get_global_id(0), y = get_global_id(1);
    float v = 0.f;
    if (x < w && y < h) {
        v = vload_half(ioff + y * w + x, im);
        if (cap > 0.f) v = min(v, cap);
    }
    float s = group_sum2(red, (float2)(v, 0)).x;
    if (get_local_id(0) == 0 && y < h) out[ooff + y * get_num_groups(0) + get_group_id(0)] = s;
}

// separable running minimum over a (2r+1) window on half-float robustness maps (used when dilate > 1)
__kernel void min_filter_h_half(__global const half* in, int ioff, __global half* out, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int base = ioff + y * w;
    float v = vload_half(base + x, in);
    for (int u = max(0, x - r); u <= min(w - 1, x + r); u++) v = min(v, vload_half(base + u, in));
    vstore_half(v, y * w + x, out);
}
__kernel void min_filter_v_half(__global const half* in, __global half* out, int ooff, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = vload_half(y * w + x, in);
    for (int t = max(0, y - r); t <= min(h - 1, y + r); t++) v = min(v, vload_half(t * w + x, in));
    vstore_half(v, ooff + y * w + x, out);
}
)CL"