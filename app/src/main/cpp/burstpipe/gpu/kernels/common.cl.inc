R"CL(
// Shared helpers. Every kernel guards its own range: global sizes are not rounded to a work-group size.

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

// sRGB tables of finish.cpp: OETF indexed in sqrt space, EOTF linear, both 4097 entries
inline float lut_interp(__global const float* T, float t) {
    float f = clamp(t, 0.f, 1.f) * 4096.f;
    int i = min((int)f, 4095);
    return T[i] + (f - i) * (T[i + 1] - T[i]);
}
inline float srgb_oetf(__global const float* T, float x) { return lut_interp(T, sqrt(clamp(x, 0.f, 1.f))); }

__kernel void scale_buffer(__global float* a, int n, float s) {
    int i = get_global_id(0);
    if (i < n) a[i] *= s;
}

// Sum over a 1D work group (every item must call it). The statistics kernels below are launched as
// (columns, rows) with 64-wide groups and write one partial sum per (row, group) for the host to add up:
// one work item per row would leave the GPU nearly idle and read rows far apart.
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

// partial sums of min(v, cap) over a w x h image, for weights read back by the host
__kernel void row_sums(__global const float* im, int ioff, int w, int h, float cap, __global float* out, int ooff,
                       __local float2* red) {
    int x = get_global_id(0), y = get_global_id(1);
    float v = x < w ? min(im[ioff + y * w + x], cap) : 0.f;
    float s = group_sum2(red, (float2)(v, 0)).x;
    if (get_local_id(0) == 0) out[ooff + y * get_num_groups(0) + get_group_id(0)] = s;
}

// separable running minimum over a (2r+1) window, clamped at the edges
__kernel void min_filter_h(__global const float* in, int ioff, __global float* out, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    __global const float* s = in + ioff + y * w;
    float v = s[x];
    for (int u = max(0, x - r); u <= min(w - 1, x + r); u++) v = min(v, s[u]);
    out[y * w + x] = v;
}
__kernel void min_filter_v(__global const float* in, __global float* out, int ooff, int w, int h, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = in[y * w + x];
    for (int t = max(0, y - r); t <= min(h - 1, y + r); t++) v = min(v, in[t * w + x]);
    out[ooff + y * w + x] = v;
}
)CL"
