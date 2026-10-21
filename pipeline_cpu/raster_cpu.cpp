// CPU Gaussian rasteriser with any number of colour channels (we render RGB + camera depth in one pass).
// Adapted from OpenSplat's rasterizer/gsplat-cpu/gsplat_cpu.cpp (Piero Toffanin, AGPLv3, itself based on nerfstudio gsplat):
// same per-pixel front-to-back compositing, early termination at T <= 1e-4 and backward pass, generalised from 3 to C channels,
// plus an |dL/dxy| accumulator for absgrad densification. Built at runtime with torch.utils.cpp_extension.load.
#include <torch/extension.h>
#include <vector>
#include <thread>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace {
int nWorkers(int height, size_t floatsPerWorker = 0) {
    int w = (int)std::max(1u, std::thread::hardware_concurrency());
    w = std::min(w, std::max(1, height));
    if (floatsPerWorker > 0) {
        const size_t budget = 256ull << 20;
        w = std::min(w, std::max(1, (int)(budget / (floatsPerWorker * sizeof(float) + 1))));
    }
    return w;
}
template <typename F> void parChunks(int numChunks, int numWorkers, const F &fn) {
    if (numWorkers <= 1) { for (int c = 0; c < numChunks; c++) fn(c, 0); return; }
    std::atomic<int> next(0);
    auto worker = [&](int slot) { int c; while ((c = next.fetch_add(1)) < numChunks) fn(c, slot); };
    std::vector<std::thread> th;
    for (int t = 1; t < numWorkers; t++) th.emplace_back([&worker, t]() { worker(t); });
    worker(0);
    for (auto &t : th) t.join();
}
const int CPW = 4;

struct Px2Gid { std::vector<int32_t> *v; int n; };
}

// returns (img[H,W,C], finalT[H,W], handle)
std::tuple<torch::Tensor, torch::Tensor, int64_t> forward(int width, int height, torch::Tensor xys, torch::Tensor conics,
        torch::Tensor colors, torch::Tensor opac, torch::Tensor bg, torch::Tensor cov2d, torch::Tensor depths) {
    torch::NoGradGuard ng;
    xys = xys.contiguous(); conics = conics.contiguous(); colors = colors.contiguous(); opac = opac.contiguous();
    bg = bg.contiguous(); depths = depths.contiguous();
    const int C = colors.size(1), N = xys.size(0);
    const float *pD = depths.data_ptr<float>();
    auto *px = new std::vector<int32_t>[(size_t)width * height];
    std::vector<int32_t> order(N); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return pD[a] < pD[b]; });
    auto fo = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor out = torch::zeros({height, width, C}, fo), T = torch::ones({height, width}, fo);
    std::vector<uint8_t> doneV((size_t)width * height, 0);
    torch::Tensor sx = (3.0f * torch::sqrt(cov2d.index({"...", 0, 0}))).contiguous(), sy = (3.0f * torch::sqrt(cov2d.index({"...", 1, 1}))).contiguous();
    const float *pCon = conics.data_ptr<float>(), *pXy = xys.data_ptr<float>(), *pSx = sx.data_ptr<float>(), *pSy = sy.data_ptr<float>();
    const float *pOp = opac.data_ptr<float>(), *pCol = colors.data_ptr<float>(), *pBg = bg.data_ptr<float>();
    float *pOut = out.data_ptr<float>(), *pT = T.data_ptr<float>(); uint8_t *pDone = doneV.data();
    const float athr = 1.0f / 255.0f;
    const int nw = nWorkers(height), nb = std::min(height, nw * CPW);
    parChunks(nb, nw, [&](int band, int) {
        for (int k = 0; k < N; k++) {
            int g = order[k];
            float gy = pXy[g * 2 + 1], syy = pSy[g];
            int r0 = std::max(0, (int)std::floor(gy - syy) - 2), r1 = std::min(height, (int)std::ceil(gy + syy) + 2);
            r0 += ((band - r0) % nb + nb) % nb;
            if (r0 >= r1) continue;
            float gx = pXy[g * 2], sxx = pSx[g];
            int c0 = std::max(0, (int)std::floor(gx - sxx) - 2), c1 = std::min(width, (int)std::ceil(gx + sxx) + 2);
            if (c0 >= c1) continue;
            float A = pCon[g * 3], B = pCon[g * 3 + 1], Cc = pCon[g * 3 + 2], op = pOp[g];
            float cut = op > 0 ? std::log(255.0f * op) + 1e-3f : -1.0f;
            const float *col = pCol + (size_t)g * C;
            for (int i = r0; i < r1; i += nb) for (int j = c0; j < c1; j++) {
                size_t p = (size_t)i * width + j;
                if (pDone[p]) continue;
                float dx = gx - j, dy = gy - i;
                float s = 0.5f * (A * dx * dx + Cc * dy * dy) + B * dx * dy;
                if (s < 0.0f || s > cut) continue;
                float al = std::min(0.999f, op * std::exp(-s));
                if (al < athr) continue;
                float t = pT[p], nt = t * (1.0f - al);
                if (nt <= 1e-4f) { pDone[p] = 1; continue; }
                float vis = al * t;
                for (int c = 0; c < C; c++) pOut[p * C + c] += vis * col[c];
                pT[p] = nt; px[p].push_back(g);
            }
        }
        for (int i = band; i < height; i += nb) for (int j = 0; j < width; j++) {
            size_t p = (size_t)i * width + j;
            for (int c = 0; c < C; c++) pOut[p * C + c] += pT[p] * pBg[c];
        }
    });
    return std::make_tuple(out, T, (int64_t) new Px2Gid{px, width * height});
}

void free_handle(int64_t h) { auto *q = (Px2Gid *)h; delete[] q->v; delete q; }

// returns (v_xy[N,2], v_conic[N,3], v_colors[N,C], v_opac[N,1], v_xy_abs[N,2])
std::vector<torch::Tensor> backward(int width, int height, torch::Tensor xys, torch::Tensor conics, torch::Tensor colors,
        torch::Tensor opac, torch::Tensor bg, torch::Tensor finalT, int64_t handle, torch::Tensor v_out, torch::Tensor v_alpha) {
    torch::NoGradGuard ng;
    xys = xys.contiguous(); conics = conics.contiguous(); colors = colors.contiguous(); opac = opac.contiguous();
    v_out = v_out.contiguous(); v_alpha = v_alpha.contiguous(); bg = bg.contiguous(); finalT = finalT.contiguous();
    auto *q = (Px2Gid *)handle; const std::vector<int32_t> *px = q->v;
    const int C = colors.size(1), N = xys.size(0);
    const float *pCon = conics.data_ptr<float>(), *pXy = xys.data_ptr<float>(), *pOp = opac.data_ptr<float>(), *pCol = colors.data_ptr<float>();
    const float *pBg = bg.data_ptr<float>(), *pTf = finalT.data_ptr<float>(), *pvo = v_out.data_ptr<float>(), *pva = v_alpha.data_ptr<float>();
    const float athr = 1.0f / 255.0f;
    const size_t fpw = (size_t)N * (2 + 3 + C + 1 + 2);
    const int nw = nWorkers(height, fpw), nb = std::min(height, nw * CPW);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor bxy = torch::zeros({nw, N, 2}, fo), bcon = torch::zeros({nw, N, 3}, fo), bcol = torch::zeros({nw, N, C}, fo),
                  bop = torch::zeros({nw, N, 1}, fo), babs = torch::zeros({nw, N, 2}, fo);
    float *Pxy = bxy.data_ptr<float>(), *Pcon = bcon.data_ptr<float>(), *Pcol = bcol.data_ptr<float>(), *Pop = bop.data_ptr<float>(), *Pabs = babs.data_ptr<float>();
    parChunks(nb, nw, [&](int band, int slot) {
        float *vxy = Pxy + (size_t)slot * N * 2, *vcon = Pcon + (size_t)slot * N * 3, *vcol = Pcol + (size_t)slot * N * C,
              *vop = Pop + (size_t)slot * N, *vabs = Pabs + (size_t)slot * N * 2;
        std::vector<float> buf(C);
        for (int i = band; i < height; i += nb) for (int j = 0; j < width; j++) {
            size_t p = (size_t)i * width + j;
            float Tf = pTf[p], T = Tf;
            std::fill(buf.begin(), buf.end(), 0.0f);
            const float *go = pvo + p * C;
            const auto &ids = px[p];
            for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
                int g = *it;
                float A = pCon[g * 3], B = pCon[g * 3 + 1], Cc = pCon[g * 3 + 2];
                float dx = pXy[g * 2] - j, dy = pXy[g * 2 + 1] - i;
                float s = 0.5f * (A * dx * dx + Cc * dy * dy) + B * dx * dy;
                if (s < 0.0f) continue;
                float vis = std::exp(-s), al = std::min(0.999f, pOp[g] * vis);
                if (al < athr) continue;
                float ra = 1.0f / (1.0f - al);
                T *= ra;
                float fac = al * T, va = Tf * ra * pva[p];
                const float *col = pCol + (size_t)g * C;
                for (int c = 0; c < C; c++) {
                    vcol[(size_t)g * C + c] += fac * go[c];
                    va += (col[c] * T - buf[c] * ra) * go[c] - Tf * ra * pBg[c] * go[c];
                    buf[c] += col[c] * fac;
                }
                float vs = -pOp[g] * vis * va;
                vcon[g * 3] += 0.5f * vs * dx * dx; vcon[g * 3 + 1] += 0.5f * vs * dx * dy; vcon[g * 3 + 2] += 0.5f * vs * dy * dy;
                float gx = vs * (A * dx + B * dy), gy = vs * (B * dx + Cc * dy);
                vxy[g * 2] += gx; vxy[g * 2 + 1] += gy;
                vabs[g * 2] += std::fabs(gx); vabs[g * 2 + 1] += std::fabs(gy);
                vop[g] += vis * va;
            }
        }
    });
    free_handle(handle);
    return {bxy.sum(0), bcon.sum(0), bcol.sum(0), bop.sum(0), babs.sum(0)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
    m.def("free_handle", &free_handle);
}
