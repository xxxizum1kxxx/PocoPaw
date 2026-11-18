R"CL(
// Spatial robust merge on the device (symmetric shot-noise variance, FP16 robustness map, fused 3x3 dilation + accumulation).

inline float frame_weight(float mu0, float mu, float black, float na, float nb, float g, float lo, float hi) {
    float s = max(0.f, (mu0 + mu) * 0.125f - black);
    float vRef = na * s + nb, vAlt = g * na * s + g * g * nb;
    float varMu = 4.f * (vRef + vAlt) * (1.f / 9.f);
    float z = fabs(mu - mu0) / sqrt(max(varMu, 1e-3f));
    return clamp((hi - z) / (hi - lo), 0.f, 1.f) * (vRef / vAlt);
}

inline bool quad_clipped(__global const ushort* F, int W, int sx, int sy, float clip) {
    __global const ushort* r0 = F + (size_t)(2 * sy) * W + 2 * sx;
    ushort2 p0 = vload2(0, r0);
    ushort2 p1 = vload2(0, r0 + W);
    return (float)max(max(p0.x, p0.y), max(p1.x, p1.y)) >= clip;
}

__kernel void merge_init(__global const ushort* R, int rOff, int W, int qw, int qh, float black, __global float* out,
                         __global float* wsum) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    R += rOff;
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    ushort2 p0 = vload2(0, R + i0);
    ushort2 p1 = vload2(0, R + i1);
    vstore2((float2)((float)p0.x - black, (float)p0.y - black), 0, out + i0);
    vstore2((float2)((float)p1.x - black, (float)p1.y - black), 0, out + i1);
    wsum[qy * qw + qx] = 1.f;
}

__kernel void merge_rob(__global const float* mean3, int refOff, int kOff, __global const ushort* frames, int fOff, int W,
                        int qw, int qh, __global const short* odx, __global const short* ody, int tOff, int nx, int ny,
                        __constant float* win, float g, float clip, float black, float na, float nb, float lo,
                        float hi, __global half* rob) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    __global const float* mref = mean3 + refOff;
    __global const float* mk = mean3 + kOff;
    __global const ushort* F = frames + fOff;
    odx += tOff;
    ody += tOff;
    int ti0, ti1, tj0, tj1;
    float wx0, wx1, wy0, wy1;
    tile_taps(qx, nx, win, &ti0, &ti1, &wx0, &wx1);
    tile_taps(qy, ny, win, &tj0, &tj1, &wy0, &wy1);

    float mu = 0.f;
    bool clipped = false;
    bool checkClip = clip < 1e20f;

    #define TAP_ROB(tc, tb, wxc, wyb)                                                   \
        do {                                                                            \
            int t_ = (tb) * nx + (tc);                                                  \
            int sx_ = qx + odx[t_], sy_ = qy + ody[t_];                                 \
            if (sx_ < 0 || sx_ >= qw || sy_ < 0 || sy_ >= qh) {                         \
                clipped = true;                                                         \
            } else {                                                                    \
                mu += (wxc) * (wyb) * mk[sy_ * qw + sx_];                               \
                if (checkClip) clipped |= quad_clipped(F, W, sx_, sy_, clip);           \
            }                                                                           \
        } while (0)

    TAP_ROB(ti0, tj0, wx0, wy0);
    TAP_ROB(ti1, tj0, wx1, wy0);
    TAP_ROB(ti0, tj1, wx0, wy1);
    TAP_ROB(ti1, tj1, wx1, wy1);
    #undef TAP_ROB

    float w = clipped ? 0.f : frame_weight(mref[qy * qw + qx], mu, black, na, nb, g, lo, hi);
    vstore_half(w, qy * qw + qx, rob);
}

// Fuses 3x3 min-filter dilation (when dilate == 1, or pass-through when dilate == 0) with merge_acc on a 16x8 Wave128 tile.
__kernel void min3_merge_acc(__global const half* rob, int dilate, __global const ushort* frames, int fOff, int W, int qw,
                             int qh, __global const short* odx, __global const short* ody, int tOff, int nx, int ny,
                             __constant float* win, float black, __global float* out, __global float* wsum) {
    __local float tile[10][18];
    int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    int gx0 = (int)get_group_id(0) * 16 - 1, gy0 = (int)get_group_id(1) * 8 - 1;
    if (dilate > 0) {
        for (int idx = tid; idx < 180; idx += 128) {
            int ty = idx / 18, tx = idx - ty * 18;
            tile[ty][tx] = vload_half(clamp(gy0 + ty, 0, qh - 1) * qw + clamp(gx0 + tx, 0, qw - 1), rob);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int qx = (int)get_group_id(0) * 16 + lx, qy = (int)get_group_id(1) * 8 + ly;
    if (qx >= qw || qy >= qh) return;

    float w;
    if (dilate > 0) {
        float m0 = min(min(tile[ly][lx], tile[ly][lx + 1]), tile[ly][lx + 2]);
        float m1 = min(min(tile[ly + 1][lx], tile[ly + 1][lx + 1]), tile[ly + 1][lx + 2]);
        float m2 = min(min(tile[ly + 2][lx], tile[ly + 2][lx + 1]), tile[ly + 2][lx + 2]);
        w = min(min(m0, m1), m2);
    } else {
        w = vload_half(qy * qw + qx, rob);
    }
    if (w <= 0.f) return;

    __global const ushort* F = frames + fOff;
    odx += tOff;
    ody += tOff;
    int ti0, ti1, tj0, tj1;
    float wx0, wx1, wy0, wy1;
    tile_taps(qx, nx, win, &ti0, &ti1, &wx0, &wx1);
    tile_taps(qy, ny, win, &tj0, &tj1, &wy0, &wy1);

    float2 a0 = 0.f, a1 = 0.f;
    #define TAP_ACC(tc, tb, tw_)                                                        \
        do {                                                                            \
            float tw = (tw_);                                                           \
            int t_ = (tb) * nx + (tc);                                                  \
            int sx_ = clamp(qx + odx[t_], 0, qw - 1);                                   \
            int sy_ = clamp(qy + ody[t_], 0, qh - 1);                                   \
            __global const ushort* r0_ = F + (size_t)(2 * sy_) * W + 2 * sx_;           \
            ushort2 p0_ = vload2(0, r0_);                                               \
            ushort2 p1_ = vload2(0, r0_ + W);                                           \
            a0 += tw * (float2)((float)p0_.x, (float)p0_.y);                            \
            a1 += tw * (float2)((float)p1_.x, (float)p1_.y);                            \
        } while (0)

    TAP_ACC(ti0, tj0, wx0 * wy0);
    TAP_ACC(ti1, tj0, wx1 * wy0);
    TAP_ACC(ti0, tj1, wx0 * wy1);
    TAP_ACC(ti1, tj1, wx1 * wy1);
    #undef TAP_ACC

    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    vstore2(vload2(0, out + i0) + w * (a0 - black), 0, out + i0);
    vstore2(vload2(0, out + i1) + w * (a1 - black), 0, out + i1);
    wsum[qy * qw + qx] += w;
}

__kernel void merge_norm(__global float* out, __global const float* wsum, int W, int qw, int qh, float black) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    float inv = 1.f / wsum[qy * qw + qx];
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    vstore2(vload2(0, out + i0) * inv + black, 0, out + i0);
    vstore2(vload2(0, out + i1) * inv + black, 0, out + i1);
}
)CL"