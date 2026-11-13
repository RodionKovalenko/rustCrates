// CUDA kernels of the engine (compiled at run time with NVRTC). Every kernel computes exactly what the operation
// of the same name in cpu.rs computes; layouts are described in backend.rs. Offsets and sizes are in floats.

typedef long long i64;

// NVRTC compiles without the math headers: -infinity from its bit pattern.
#define NEG_INF __int_as_float(0xff800000)

#define ROWS_PER_BLOCK_THREADS 256

__device__ __forceinline__ float sigmoid_f(float t) {
    if (t >= 0.0f) return 1.0f / (1.0f + expf(-t));
    float e = expf(t);
    return e / (1.0f + e);
}
__device__ __forceinline__ float silu_f(float t) { return t * sigmoid_f(t); }
__device__ __forceinline__ float silu_d(float t) { float s = sigmoid_f(t); return s * (1.0f + t * (1.0f - s)); }

// Sum over the threads of a block (blockDim.x a power of two <= 1024); every thread gets the result.
__device__ float block_sum(float v, float* shared) {
    unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    unsigned warps = (blockDim.x + 31) >> 5;
    v = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();
    v = shared[0];
    __syncthreads();
    return v;
}

__device__ float block_max(float v, float* shared) {
    unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    unsigned warps = (blockDim.x + 31) >> 5;
    v = threadIdx.x < warps ? shared[threadIdx.x] : NEG_INF;
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();
    v = shared[0];
    __syncthreads();
    return v;
}

#define GRID_STRIDE(i, n) for (i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (i64)blockDim.x * gridDim.x)

extern "C" __global__ void fill(float* buf, i64 off, i64 n, float value) {
    GRID_STRIDE(i, n) buf[off + i] = value;
}

extern "C" __global__ void copy(const float* src, i64 src_off, float* dst, i64 dst_off, i64 n) {
    GRID_STRIDE(i, n) dst[dst_off + i] = src[src_off + i];
}

extern "C" __global__ void axpy(float alpha, const float* src, float* dst, i64 n) {
    GRID_STRIDE(i, n) dst[i] += alpha * src[i];
}

extern "C" __global__ void scale(float* buf, i64 n, float alpha) {
    GRID_STRIDE(i, n) buf[i] *= alpha;
}

// Partial sums of squares per block (summed on the host in f64); counts non-finite values.
extern "C" __global__ void sum_squares(const float* buf, i64 n, float* partial, int* non_finite) {
    __shared__ float shared[32];
    float acc = 0.0f;
    int bad = 0;
    GRID_STRIDE(i, n) {
        float v = buf[i];
        if (!isfinite(v)) bad = 1;
        acc += v * v;
    }
    float total = block_sum(acc, shared);
    if (threadIdx.x == 0) partial[blockIdx.x] = total;
    if (bad) atomicOr(non_finite, 1);
}

// f32 -> f16 (round to nearest), stored as 16-bit integers.
extern "C" __global__ void to_half(const float* src, i64 src_off, unsigned short* dst, i64 n) {
    GRID_STRIDE(i, n) {
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(src[src_off + i]));
        dst[i] = h;
    }
}

// out (2k x 2n) = [[Wr, Wi], [-Wi, Wr]]
extern "C" __global__ void build_block(const float* params, i64 re_off, i64 im_off, i64 k, i64 n, float* out) {
    GRID_STRIDE(idx, 4 * k * n) {
        i64 row = idx / (2 * n), col = idx % (2 * n);
        i64 src = (row % k) * n + (col % n);
        bool top = row < k, left = col < n;
        float r = params[re_off + src], im = params[im_off + src];
        out[idx] = top ? (left ? r : im) : (left ? -im : r);
    }
}

// grads[re] += G11 + G22, grads[im] += G12 - G21
extern "C" __global__ void fold_block_grad(const float* g, i64 k, i64 n, float* grads, i64 re_off, i64 im_off) {
    GRID_STRIDE(idx, k * n) {
        i64 row = idx / n, j = idx % n;
        float g11 = g[row * 2 * n + j], g12 = g[row * 2 * n + n + j];
        float g21 = g[(k + row) * 2 * n + j], g22 = g[(k + row) * 2 * n + n + j];
        grads[re_off + idx] += g11 + g22;
        grads[im_off + idx] += g12 - g21;
    }
}

extern "C" __global__ void add_row_vector(float* y, i64 rows, i64 cols, const float* params, i64 off) {
    GRID_STRIDE(idx, rows * cols) y[idx] += params[off + idx % cols];
}

// grads[off + j] += sum_r dy[r, j]: one thread per column (columns are contiguous, so warps read coalesced).
extern "C" __global__ void column_sums(const float* dy, i64 rows, i64 cols, float* grads, i64 off) {
    GRID_STRIDE(j, cols) {
        double acc = 0.0;
        for (i64 r = 0; r < rows; r++) acc += dy[r * cols + j];
        grads[off + j] += (float)acc;
    }
}

// One block per planar row.
extern "C" __global__ void layer_norm(const float* x, i64 w, float eps, const float* params, i64 gamma_off, i64 beta_off, float* y, float* stats) {
    __shared__ float shared[32];
    i64 r = blockIdx.x;
    const float* xr = x + r * 2 * w;
    float sr = 0.0f, si = 0.0f;
    for (i64 j = threadIdx.x; j < w; j += blockDim.x) { sr += xr[j]; si += xr[w + j]; }
    float mre = block_sum(sr, shared) / w, mim = block_sum(si, shared) / w;
    float v = 0.0f;
    for (i64 j = threadIdx.x; j < w; j += blockDim.x) {
        float dr = xr[j] - mre, di = xr[w + j] - mim;
        v += dr * dr + di * di;
    }
    float rstd = rsqrtf(block_sum(v, shared) / w + eps);
    const float* gamma = params + gamma_off;
    const float* beta = params + beta_off;
    for (i64 j = threadIdx.x; j < w; j += blockDim.x) {
        float hr = (xr[j] - mre) * rstd, hi = (xr[w + j] - mim) * rstd;
        float gr = gamma[j], gi = gamma[w + j];
        y[r * 2 * w + j] = hr * gr - hi * gi + beta[j];
        y[r * 2 * w + w + j] = hr * gi + hi * gr + beta[w + j];
    }
    if (threadIdx.x == 0) { stats[3 * r] = mre; stats[3 * r + 1] = mim; stats[3 * r + 2] = rstd; }
}

// One block per row: writes dx and the row's gamma / beta gradient contributions (rows x 4w, summed afterwards).
extern "C" __global__ void layer_norm_backward(const float* x, const float* stats, const float* dy, i64 w, const float* params, i64 gamma_off, float* dx, float* row_grads) {
    __shared__ float shared[32];
    i64 r = blockIdx.x;
    const float* xr = x + r * 2 * w;
    const float* g = dy + r * 2 * w;
    const float* gamma = params + gamma_off;
    float mre = stats[3 * r], mim = stats[3 * r + 1], rstd = stats[3 * r + 2];
    float ur_sum = 0.0f, ui_sum = 0.0f, s = 0.0f;
    float* rg = row_grads + r * 4 * w;
    for (i64 j = threadIdx.x; j < w; j += blockDim.x) {
        float gr = g[j], gi = g[w + j], gmr = gamma[j], gmi = gamma[w + j];
        float ur = gr * gmr + gi * gmi, ui = gi * gmr - gr * gmi;
        float hr = (xr[j] - mre) * rstd, hi = (xr[w + j] - mim) * rstd;
        ur_sum += ur; ui_sum += ui; s += ur * hr + ui * hi;
        rg[j] = gr * hr + gi * hi;
        rg[w + j] = gi * hr - gr * hi;
        rg[2 * w + j] = gr;
        rg[3 * w + j] = gi;
    }
    float ure_mean = block_sum(ur_sum, shared) / w, uim_mean = block_sum(ui_sum, shared) / w;
    float c = block_sum(s, shared) / w;
    for (i64 j = threadIdx.x; j < w; j += blockDim.x) {
        float gr = g[j], gi = g[w + j], gmr = gamma[j], gmi = gamma[w + j];
        float ur = gr * gmr + gi * gmi, ui = gi * gmr - gr * gmi;
        float hr = (xr[j] - mre) * rstd, hi = (xr[w + j] - mim) * rstd;
        dx[r * 2 * w + j] = (ur - ure_mean - hr * c) * rstd;
        dx[r * 2 * w + w + j] = (ui - uim_mean - hi * c) * rstd;
    }
}

__device__ __forceinline__ void rope_cos_sin(i64 position, i64 i, i64 head_dim, float rope_base, float* c, float* s) {
    double theta = (double)position / pow((double)rope_base, 2.0 * (double)i / (double)head_dim);
    theta = fmin(fmax(theta, -1.0), 1.0);
    *c = (float)cos(theta);
    *s = (float)sin(theta);
}

// One thread per (row, head, pair, part).
extern "C" __global__ void rope_to_heads(const float* src, i64 src_ld, i64 re_off, i64 im_off, i64 rows, i64 seq_len, i64 heads, i64 head_dim, float rope_base, float* dst) {
    i64 pairs = head_dim / 2;
    GRID_STRIDE(idx, rows * heads * pairs * 2) {
        i64 part = idx % 2, i = (idx / 2) % pairs, h = (idx / (2 * pairs)) % heads, r = idx / (2 * pairs * heads);
        float c, s;
        rope_cos_sin(r % seq_len, i, head_dim, rope_base, &c, &s);
        i64 off = part == 0 ? re_off : im_off;
        float e = src[r * src_ld + off + h * head_dim + 2 * i], o = src[r * src_ld + off + h * head_dim + 2 * i + 1];
        i64 base = r * 2 * heads * head_dim + h * 2 * head_dim + part * head_dim + 2 * i;
        dst[base] = e * c - o * s;
        dst[base + 1] = e * s + o * c;
    }
}

extern "C" __global__ void rope_from_heads(const float* src, i64 rows, i64 seq_len, i64 heads, i64 head_dim, float rope_base, float* dst, i64 dst_ld, i64 re_off, i64 im_off) {
    i64 pairs = head_dim / 2;
    GRID_STRIDE(idx, rows * heads * pairs * 2) {
        i64 part = idx % 2, i = (idx / 2) % pairs, h = (idx / (2 * pairs)) % heads, r = idx / (2 * pairs * heads);
        float c, s;
        rope_cos_sin(r % seq_len, i, head_dim, rope_base, &c, &s);
        i64 base = r * 2 * heads * head_dim + h * 2 * head_dim + part * head_dim + 2 * i;
        float e = src[base], o = src[base + 1];
        i64 off = part == 0 ? re_off : im_off;
        dst[r * dst_ld + off + h * head_dim + 2 * i] = e * c + o * s;
        dst[r * dst_ld + off + h * head_dim + 2 * i + 1] = -e * s + o * c;
    }
}

// One block per score row.
extern "C" __global__ void softmax_window(float* s, i64 t, i64 window) {
    __shared__ float shared[32];
    i64 row = blockIdx.x, i = row % t;
    i64 first = i > window ? i - window : 0;
    float* v = s + row * t;
    float m = NEG_INF;
    for (i64 j = first + threadIdx.x; j <= i; j += blockDim.x) m = fmaxf(m, v[j]);
    m = block_max(m, shared);
    float sum = 0.0f;
    for (i64 j = threadIdx.x; j < t; j += blockDim.x) {
        float e = (j >= first && j <= i) ? expf(v[j] - m) : 0.0f;
        v[j] = e;
        sum += e;
    }
    sum = block_sum(sum, shared);
    float inv = 1.0f / sum;
    for (i64 j = threadIdx.x; j < t; j += blockDim.x) v[j] *= inv;
}

extern "C" __global__ void softmax_window_backward(const float* p, float* dp, i64 t) {
    __shared__ float shared[32];
    i64 row = blockIdx.x;
    const float* pr = p + row * t;
    float* d = dp + row * t;
    float dot = 0.0f;
    for (i64 j = threadIdx.x; j < t; j += blockDim.x) dot += pr[j] * d[j];
    dot = block_sum(dot, shared);
    for (i64 j = threadIdx.x; j < t; j += blockDim.x) d[j] = pr[j] * (d[j] - dot);
}

extern "C" __global__ void attention_residual(const float* x, const float* att, i64 rows, i64 w, float beta, float* y) {
    GRID_STRIDE(idx, rows * 2 * w) {
        i64 r = idx / (2 * w), j = idx % (2 * w);
        y[idx] = x[idx] + (j < w ? beta * att[r * w + j] : 0.0f);
    }
}

extern "C" __global__ void real_part_scaled(const float* dy, i64 rows, i64 w, float beta, float* out) {
    GRID_STRIDE(idx, rows * w) out[idx] = beta * dy[(idx / w) * 2 * w + idx % w];
}

extern "C" __global__ void gated(const float* h, i64 rows, i64 hidden, float* g) {
    GRID_STRIDE(idx, rows * hidden) {
        i64 r = idx / hidden, j = idx % hidden;
        const float* hr = h + r * 4 * hidden;
        float ar = hr[j], br = hr[hidden + j], ai = hr[2 * hidden + j], bi = hr[3 * hidden + j];
        float sr = silu_f(br), si = silu_f(bi);
        g[r * 2 * hidden + j] = ar * sr - ai * si;
        g[r * 2 * hidden + hidden + j] = ar * si + ai * sr;
    }
}

extern "C" __global__ void gated_backward(const float* h, const float* dg, i64 rows, i64 hidden, float* dh) {
    GRID_STRIDE(idx, rows * hidden) {
        i64 r = idx / hidden, j = idx % hidden;
        const float* hr = h + r * 4 * hidden;
        float ar = hr[j], br = hr[hidden + j], ai = hr[2 * hidden + j], bi = hr[3 * hidden + j];
        float gre = dg[r * 2 * hidden + j], gim = dg[r * 2 * hidden + hidden + j];
        float sr = silu_f(br), si = silu_f(bi);
        float* d = dh + r * 4 * hidden;
        d[j] = gre * sr + gim * si;
        d[2 * hidden + j] = gim * sr - gre * si;
        float ur = gre * ar + gim * ai, ui = gim * ar - gre * ai;
        d[hidden + j] = ur * silu_d(br);
        d[3 * hidden + j] = ui * silu_d(bi);
    }
}

extern "C" __global__ void leaky_residual(const float* o, const float* x, i64 rows, i64 w, float beta, float slope, float* y) {
    GRID_STRIDE(idx, rows * w) {
        i64 r = idx / w, j = idx % w;
        i64 re = r * 2 * w + j, im = re + w;
        float f = o[re] > 0.0f ? beta : beta * slope;
        y[re] = x[re] + f * o[re];
        y[im] = x[im] + f * o[im];
    }
}

extern "C" __global__ void leaky_residual_backward(const float* o, const float* dy, i64 rows, i64 w, float beta, float slope, float* d_o) {
    GRID_STRIDE(idx, rows * w) {
        i64 r = idx / w, j = idx % w;
        i64 re = r * 2 * w + j, im = re + w;
        float f = o[re] > 0.0f ? beta : beta * slope;
        d_o[re] = f * dy[re];
        d_o[im] = f * dy[im];
    }
}

// One block per row: loss of the row to row_loss[r], logits replaced by scale * (softmax - onehot).
extern "C" __global__ void cross_entropy(float* logits, i64 vocab, const unsigned int* targets, i64 t_off, float scale, float* row_loss) {
    __shared__ float shared[32];
    i64 r = blockIdx.x;
    float* l = logits + r * vocab;
    i64 target = targets[t_off + r];
    float m = NEG_INF;
    for (i64 j = threadIdx.x; j < vocab; j += blockDim.x) m = fmaxf(m, l[j]);
    m = block_max(m, shared);
    float sum = 0.0f;
    for (i64 j = threadIdx.x; j < vocab; j += blockDim.x) sum += expf(l[j] - m);
    sum = block_sum(sum, shared);
    float target_logit = l[target];
    __syncthreads();
    float inv = 1.0f / sum;
    for (i64 j = threadIdx.x; j < vocab; j += blockDim.x) l[j] = expf(l[j] - m) * inv * scale;
    __syncthreads();
    if (threadIdx.x == 0) {
        l[target] -= scale;
        row_loss[r] = logf(sum) - (target_logit - m);
    }
}

extern "C" __global__ void embed(const float* params, i64 re_off, i64 im_off, const unsigned int* ids, i64 ids_off, i64 rows, i64 w, float* out) {
    GRID_STRIDE(idx, rows * w) {
        i64 r = idx / w, j = idx % w;
        i64 id = ids[ids_off + r];
        out[r * 2 * w + j] = params[re_off + id * w + j];
        out[r * 2 * w + w + j] = params[im_off + id * w + j];
    }
}

extern "C" __global__ void embed_backward(const float* dx, const unsigned int* ids, i64 ids_off, i64 rows, i64 w, float* grads, i64 re_off, i64 im_off) {
    GRID_STRIDE(idx, rows * w) {
        i64 r = idx / w, j = idx % w;
        i64 id = ids[ids_off + r];
        atomicAdd(grads + re_off + id * w + j, dx[r * 2 * w + j]);
        atomicAdd(grads + im_off + id * w + j, dx[r * 2 * w + w + j]);
    }
}

extern "C" __global__ void adamw(float* params, const float* grads, float* m, float* v, i64 n, float lr, float beta1, float beta2, float eps, float weight_decay, float grad_scale, float c1, float c2) {
    GRID_STRIDE(i, n) {
        float g = grads[i] * grad_scale;
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        float p = params[i] * (1.0f - lr * weight_decay);
        params[i] = p - lr * (mi / c1) / (sqrtf(vi / c2) + eps);
    }
}
