#include <vector>
#include <cmath>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <utility>
#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif
#include "math/array.hpp"
#include "math/blockMatrix.hpp"

using namespace sparse;
using namespace block;

static double drand(uint32_t i)
{
    double s = std::sin((double)i * 0.371 + 1.7) * 43758.5453;
    return s - std::floor(s);
}

// =====================================================================
// 归约小工具：连续内存上的点积 / 2-范数
//   - OpenMP reduction: 每线程私有累加器, 无写冲突、无 false sharing
//   - 内层只触碰两块连续 double, -O3 -march=native 下可自动向量化
// =====================================================================
namespace {

inline double dot_par(const std::vector<double>& a, const std::vector<double>& b)
{
    const std::ptrdiff_t n = (std::ptrdiff_t)a.size();
    double s = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(+:s) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

inline double norm_par(const std::vector<double>& a)
{
    return std::sqrt(dot_par(a, a));
}

// 由逐行 nnz 计数生成 CSR 风格行偏移(前缀和)。
// 串行 O(nrows), 只有整数加法, 成本远低于随后的填充; 换来的是填充阶段
// 行间完全无竞争(无需原子操作), 且每个 vector 只分配一次(零 realloc)。
inline std::vector<uint32_t> row_offsets(const std::vector<uint32_t>& cnt)
{
    std::vector<uint32_t> off(cnt.size() + 1, 0);
    for (size_t i = 0; i < cnt.size(); ++i) off[i + 1] = off[i] + cnt[i];
    return off;
}

} // namespace

// =====================================================================
// 1D Stokes 测试数据:  K = [ A   B^T ]
//                          [ B   0   ]
// =====================================================================
// 【生命周期 + 内存】三块矩阵放进 shared_ptr 持有的结构体:
//   1) 工厂函数返回后局部对象即销毁, lambda 若按引用捕获 -> 悬垂引用(UB)。
//      改为值捕获 shared_ptr: 引用计数 +1, 数据寿命与 lambda 绑定。
//   2) 相比"把 A/B/Bt 各深拷贝一份再按值捕获", 省掉 3 份完整 CSR
//      (大算例下就是 3x 常驻内存)。
struct StokesMatrices
{
    SMatrix K_full;   // 整体鞍点矩阵
    SMatrix A;        // 速度 Laplace (N_vel  x N_vel)
    SMatrix B;        // 散度         (N_pres x N_vel)
    SMatrix Bt;       // 负梯度       (N_vel  x N_pres)
};

struct StokesTestData
{
    std::shared_ptr<StokesMatrices> mats;
    SaddlePointSystem               sys;
    size_t n_vel  = 0;
    size_t n_pres = 0;

    size_t n_total() const { return n_vel + n_pres; }
};

StokesTestData make_1d_stokes_test(uint32_t N_vel)
{
    StokesTestData out;
    out.mats = std::make_shared<StokesMatrices>();

    const uint32_t nv = N_vel;
    const uint32_t np = (N_vel > 0) ? (N_vel - 1u) : 0u;
    out.n_vel  = nv;
    out.n_pres = np;
    const uint32_t N_total = nv + np;

    // ---------------------------------------------------------------
    // 组装策略: 计数 -> 前缀和 -> 并行直填
    //   原先是"先攒一份统一 COO, 再逐元素三分支拆分到 A/B/Bt"。那份统一 COO
    //   是纯中间产物(峰值 2x), 拆分又是串行且 6 个 vector 无 reserve 反复
    //   realloc。现在四份 COO 全部按解析行数一次分配到位, 并行直填。
    // ---------------------------------------------------------------

    // ---- 逐行 nnz 计数 (解析可得, 无需真正遍历) ----
    // A : 行 i -> 对角 + (i>0) + (i<nv-1)
    // B : 行 p -> 恒 2
    // Bt: 行 r -> (r<np) + (r>=1)
    // K : 速度行 -> A 的行数 + 两条压力耦合; 压力行 -> 恒 2
    std::vector<uint32_t> cntA(nv, 0), cntB(np, 0), cntBt(nv, 0), cntK(N_total, 0);
    for (uint32_t i = 0; i < nv; ++i)
        cntA[i] = 1u + (i > 0 ? 1u : 0u) + (i < nv - 1u ? 1u : 0u);
    for (uint32_t p = 0; p < np; ++p)
        cntB[p] = 2u;
    for (uint32_t r = 0; r < nv; ++r)
        cntBt[r] = (r < np ? 1u : 0u) + (r >= 1u ? 1u : 0u);
    for (uint32_t r = 0; r < N_total; ++r)
        cntK[r] = (r < nv) ? (cntA[r] + (r < np ? 1u : 0u) + (r >= 1u ? 1u : 0u)) : 2u;

    std::vector<uint32_t> offA = row_offsets(cntA);
    std::vector<uint32_t> offB = row_offsets(cntB);
    std::vector<uint32_t> offBt = row_offsets(cntBt);
    std::vector<uint32_t> offK = row_offsets(cntK);

    // 计数/行偏移用完即弃: 在分配大块 COO 之前把临时内存还给分配器, 压低峰值
    { std::vector<uint32_t> e;
      cntA.swap(e); cntB.swap(e); cntBt.swap(e); cntK.swap(e); }

    const size_t nnzA = offA[nv], nnzB = offB[np], nnzBt = offBt[nv], nnzK = offK[N_total];
    std::vector<double>   vA(nnzA),  vB(nnzB),  vBt(nnzBt),  vK(nnzK);
    std::vector<uint32_t> rA(nnzA),  rB(nnzB),  rBt(nnzBt),  rK(nnzK);
    std::vector<uint32_t> cA(nnzA),  cB(nnzB),  cBt(nnzBt),  cK(nnzK);

    // ---- 并行填充: 每行只写自己的 [off[r], off[r+1]) 区间 ----
    const std::ptrdiff_t pnv = (std::ptrdiff_t)nv;
    const std::ptrdiff_t pnp = (std::ptrdiff_t)np;
    const std::ptrdiff_t pnt = (std::ptrdiff_t)N_total;

#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel
#endif
    {
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (std::ptrdiff_t ii = 0; ii < pnv; ++ii)
        {
            const uint32_t i = (uint32_t)ii;
            uint32_t w = offA[i];
            rA[w] = i; cA[w] = i; vA[w] = 2.0; ++w;
            if (i > 0)       { rA[w] = i; cA[w] = i - 1; vA[w] = -1.0; ++w; }
            if (i < nv - 1u) { rA[w] = i; cA[w] = i + 1; vA[w] = -1.0; ++w; }
        }

#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (std::ptrdiff_t pp = 0; pp < pnp; ++pp)
        {
            const uint32_t p = (uint32_t)pp;
            uint32_t w = offB[p];
            rB[w] = p; cB[w] = p;     vB[w] = -1.0; ++w;
            rB[w] = p; cB[w] = p + 1; vB[w] =  1.0; ++w;
        }

#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (std::ptrdiff_t rr = 0; rr < pnv; ++rr)
        {
            const uint32_t r = (uint32_t)rr;
            uint32_t w = offBt[r];
            if (r < np)  { rBt[w] = r; cBt[w] = r;     vBt[w] = -1.0; ++w; }
            if (r >= 1u) { rBt[w] = r; cBt[w] = r - 1; vBt[w] =  1.0; ++w; }
        }

#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t rr = 0; rr < pnt; ++rr)
        {
            const uint32_t r = (uint32_t)rr;
            uint32_t w = offK[r];
            if (r < nv)
            {
                // 速度块自身 (A)
                rK[w] = r; cK[w] = r; vK[w] = 2.0; ++w;
                if (r > 0)       { rK[w] = r; cK[w] = r - 1; vK[w] = -1.0; ++w; }
                if (r < nv - 1u) { rK[w] = r; cK[w] = r + 1; vK[w] = -1.0; ++w; }
                // 与压力块的耦合 (B^T)
                if (r < np)  { rK[w] = r; cK[w] = r + nv;       vK[w] = -1.0; ++w; }
                if (r >= 1u) { rK[w] = r; cK[w] = (r - 1) + nv; vK[w] =  1.0; ++w; }
            }
            else
            {
                const uint32_t p = r - nv;                  // 压力行 (B)
                rK[w] = r; cK[w] = p;     vK[w] = -1.0; ++w;
                rK[w] = r; cK[w] = p + 1; vK[w] =  1.0; ++w;
            }
        }
    }

    // 进入 SMatrix 构造(COO -> CSR, 内部还有一次拷贝)之前释放行偏移, 压低峰值
    { std::vector<uint32_t> e;
      offA.swap(e); offB.swap(e); offBt.swap(e); offK.swap(e); }

    out.mats->A      = SMatrix(nv, nv, vA, rA, cA);
    out.mats->B      = SMatrix(np, nv, vB, rB, cB);
    out.mats->Bt     = SMatrix(nv, np, vBt, rBt, cBt);
    out.mats->K_full = SMatrix(N_total, N_total, vK, rK, cK);

    // 【关键修复】值捕获 shared_ptr, 而非引用捕获即将销毁的局部 StokesTestData。
    // 三大块零拷贝地随 lambda 存活。
    auto mats = out.mats;
    out.sys.n_vel  = out.n_vel;
    out.sys.n_pres = out.n_pres;
    out.sys.spmv_A  = [mats](const std::vector<double>& x, std::vector<double>& y)
                      { y = SMatrix::spmv(mats->A,  x); };
    out.sys.spmv_B  = [mats](const std::vector<double>& x, std::vector<double>& y)
                      { y = SMatrix::spmv(mats->B,  x); };
    out.sys.spmv_Bt = [mats](const std::vector<double>& x, std::vector<double>& y)
                      { y = SMatrix::spmv(mats->Bt, x); };

    return out;
}

// =====================================================================
// 测试：GMRES + Schur 补
// =====================================================================
void test_schur_complement(uint32_t N_vel)
{
    auto data = make_1d_stokes_test(N_vel);
    const size_t N_total = data.n_total();
    const std::ptrdiff_t pn = (std::ptrdiff_t)N_total;

    // A 的预处理器 (标量 ILU0)
    auto A_precond = SMatrix::make_ilu0_preconditioner(data.mats->A);

    std::cout << "ilu预处理器加载完成" << std::endl;

    // S 的预处理器 (简化: 单位阵)
    auto S_precond = [](const std::vector<double>& r, std::vector<double>& z) { z = r; };

    // Schur 预处理器
    auto schur_precond = make_schur_preconditioner(data.sys, A_precond, S_precond);

    std::cout << "舒尔补预处理器加载完成" << std::endl;

    // 右端项: b = K x_exact
    std::vector<double> x_exact(N_total), b;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i)
        x_exact[i] = drand((uint32_t)i);
    b = SMatrix::spmv(data.mats->K_full, x_exact);

    // 【关键】调用 SMatrix::gmres 的 std::function 重载
    auto [x, it] = SMatrix::gmres(data.mats->K_full, b, schur_precond, 50);

    std::cout << "gmres求解完成" << std::endl;

    // 相对误差: ||x - x_exact|| / ||x_exact||  (并行归约 + 连续访存)
    double err2 = 0.0, ref2 = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(+:err2) reduction(+:ref2) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i)
    {
        const double d = x[i] - x_exact[i];
        err2 += d * d;
        ref2 += x_exact[i] * x_exact[i];
    }
    const double rel = (ref2 > 0.0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);

    std::printf("[Schur] N_vel=%4u DOF=%6zu GMRES迭代数=%4d 相对误差=%.3e %s\n",
                N_vel, N_total, it + 1, rel, rel < 1e-6 ? "通过" : "失败");
}

int main()
{
    test_schur_complement(100);
    test_schur_complement(500);
    test_schur_complement(1000);
    return 0;
}
