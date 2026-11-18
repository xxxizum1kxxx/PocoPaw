R"CL(
// merge.cpp (spatial robust merge) and merge_freq.cpp (frequency-domain Wiener merge) on the device.

inline float frame_weight(float mu0, float mu, float black, float na, float nb, float g, float lo, float hi) {
    float s0 = max(0.f, mu0 / 4 - black);
    float s = max(0.f, (mu0 + mu) / 8 - black);
    float vRef = na * s + nb, vAlt = g * na * s + g * g * nb;
    float varMu = 4.f * (vRef + vAlt) / 9.f;
    float z = fabs(mu - mu0) / sqrt(max(varMu, 1e-3f));
    float vRef0 = na * s0 + nb, vAlt0 = g * na * s0 + g * g * nb;
    return clamp((hi - z) / (hi - lo), 0.f, 1.f) * (vRef0 / vAlt0);
}

inline bool quad_clipped(__global const ushort* F, int W, int sx, int sy, float clip) {
    __global const ushort* r0 = F + (size_t)(2 * sy) * W + 2 * sx;
    __global const ushort* r1 = r0 + W;
    return max(max(r0[0], r0[1]), max(r1[0], r1[1])) >= clip;
}

__kernel void merge_init(__global const ushort* R, int rOff, int W, int qw, int qh, float black, __global float* out,
                         __global float* wsum) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    R += rOff;
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    out[i0] = R[i0] - black; out[i0 + 1] = R[i0 + 1] - black;
    out[i1] = R[i1] - black; out[i1 + 1] = R[i1 + 1] - black;
    wsum[qy * qw + qx] = 1.f;
}

// pass 1 of merge_frames: robustness of one alternate frame per quad
__kernel void merge_rob(__global const float* mean3, int refOff, int kOff, __global const ushort* frames, int fOff, int W,
                        int qw, int qh, __global const short* odx, __global const short* ody, int tOff, int nx, int ny,
                        __constant float* win, float g, float clip, float black, float na, float nb, float lo,
                        float hi, __global float* rob) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    __global const float* mref = mean3 + refOff;
    __global const float* mk = mean3 + kOff;
    __global const ushort* F = frames + fOff;
    odx += tOff;
    ody += tOff;
    int ti[2], tj[2];
    float wx[2], wy[2];
    tile_taps(qx, nx, win, &ti[0], &ti[1], &wx[0], &wx[1]);
    tile_taps(qy, ny, win, &tj[0], &tj[1], &wy[0], &wy[1]);
    float mu = 0;
    bool clipped = false;
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++) {
            int t = tj[b] * nx + ti[c];
            int sx = qx + odx[t], sy = qy + ody[t];
            // clamping would repeat the border row / column into streaks
            if (sx < 0 || sx >= qw || sy < 0 || sy >= qh) {
                clipped = true;
                continue;
            }
            mu += wx[c] * wy[b] * mk[sy * qw + sx];
            clipped |= quad_clipped(F, W, sx, sy, clip);
        }
    rob[qy * qw + qx] = clipped ? 0.f : frame_weight(mref[qy * qw + qx], mu, black, na, nb, g, lo, hi);
}

// pass 3 of merge_frames: accumulate one alternate frame into the reference
__kernel void merge_acc(__global const float* rob, __global const ushort* frames, int fOff, int W, int qw, int qh,
                        __global const short* odx, __global const short* ody, int tOff, int nx, int ny,
                        __constant float* win, float black, __global float* out, __global float* wsum) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    __global const ushort* F = frames + fOff;
    odx += tOff;
    ody += tOff;
    float w = rob[qy * qw + qx];
    if (w <= 0) return;
    int ti[2], tj[2];
    float wx[2], wy[2];
    tile_taps(qx, nx, win, &ti[0], &ti[1], &wx[0], &wx[1]);
    tile_taps(qy, ny, win, &tj[0], &tj[1], &wy[0], &wy[1]);
    float a[4] = {0, 0, 0, 0};
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++) {
            float tw = wx[c] * wy[b];
            int t = tj[b] * nx + ti[c];
            int sx = clamp(qx + odx[t], 0, qw - 1), sy = clamp(qy + ody[t], 0, qh - 1);
            __global const ushort* r0 = F + (size_t)(2 * sy) * W + 2 * sx;
            __global const ushort* r1 = r0 + W;
            a[0] += tw * r0[0]; a[1] += tw * r0[1];
            a[2] += tw * r1[0]; a[3] += tw * r1[1];
        }
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    out[i0] += w * (a[0] - black); out[i0 + 1] += w * (a[1] - black);
    out[i1] += w * (a[2] - black); out[i1 + 1] += w * (a[3] - black);
    wsum[qy * qw + qx] += w;
}

__kernel void merge_norm(__global float* out, __global const float* wsum, int W, int qw, int qh, float black) {
    int qx = get_global_id(0), qy = get_global_id(1);
    if (qx >= qw || qy >= qh) return;
    float inv = 1.f / wsum[qy * qw + qx];
    size_t i0 = (size_t)(2 * qy) * W + 2 * qx, i1 = i0 + W;
    out[i0] = out[i0] * inv + black; out[i0 + 1] = out[i0 + 1] * inv + black;
    out[i1] = out[i1] * inv + black; out[i1 + 1] = out[i1 + 1] * inv + black;
}
)CL"
