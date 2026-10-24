#include <vector>
#include <cmath>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <chrono>
#include <iostream>
#include <cstdarg>
#include <cstdlib>
#include <cctype>
#include <string>
#include <stdexcept>
#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif

#include "math/array.hpp"
#include "math/blockMatrix.hpp"

using namespace sparse;
using namespace block;

// 确定性伪随机数
static double drand(uint32_t i) {
    double s = std::sin((double)i * 0.371 + 1.7) * 43758.5453;
    return s - std::floor(s);
}

// =====================================================================
// 小工具
// =====================================================================
namespace {

using Clock = std::chrono::high_resolution_clock;

inline double elapsed(Clock::time_point a, Clock::time_point b)
{ return std::chrono::duration<double>(b - a).count(); }

// 平方距离 ||a-b||^2 与参考范数 ||b||^2 一次遍历同时算出 (访存 4n 而不是 6n)
inline void sqdist_par(const std::vector<double>& a, const std::vector<double>& b,
                       double& dist2, double& ref2)
{
    const std::ptrdiff_t n = (std::ptrdiff_t)std::min(a.size(), b.size());
    double d2 = 0.0, r2 = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(+:d2) reduction(+:r2) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i)
    {
        const double d = a[i] - b[i];
        d2 += d * d;
        r2 += b[i] * b[i];
    }
    dist2 = d2; ref2 = r2;
}

// 把 vector 的缓冲区立刻还给分配器 (swap-with-temporary 惯用法):
// 组装完一块矩阵先把它的 COO 归还, 再去组装下一块, 避免所有 COO 同时常驻
template <typename V>
inline void drop(V& v) { V().swap(v); }

} // namespace

// =====================================================================
// 3D MAC 网格 Stokes 算例
//   K = [ A   B^T ]     A : 3 个解耦的 3D 标量 Laplace + alpha*I
//       [ B   0   ]     B : 散度, Lp : 压力 Laplace
//
//   【Schur 补的精确形式】N=4 上逐 Fourier 模数值核对, 误差 1.1e-16:
//     B B^T = Lp                   精确成立 (逐元素差 0)
//     A = alpha*I + nu*L, 三个速度分量的 L 与 Lp 是同一个 7 点算子
//     => S = B A^{-1} B^T 的特征值 = lambda/(alpha + nu*lambda)
//     => S^{-1} = nu*Mp^{-1} + alpha*Lp^{-1}      (本算例 nu=1, Mp=I)
//   这就是 Cahouet-Chabard 形式: 单位阵项是 Laplace 逆, Lp^{-1} 项是质量项。
//   alpha 越大 (非定常 Stokes 的小时间步), Lp^{-1} 项越重要, AMG 越有价值。
// =====================================================================
// 【内存 + 生命周期】五个矩阵放进 shared_ptr 持有的结构体:
//   1) lambda 值捕获 shared_ptr -> 不会像引用捕获那样悬垂
//   2) 相比"把 A/B/Bt 各深拷贝一份再值捕获"省掉 3 份完整 CSR
//      (N=75 时 A 单独一份就是 ~140 MB)
struct Stokes3DMatrices {
    SMatrix K_full;
    SMatrix A;
    SMatrix B;
    SMatrix Bt;
    SMatrix Lp;      // 压力 Laplace
};

struct Stokes3DData {
    std::shared_ptr<Stokes3DMatrices> m;
    uint32_t N_vel  = 0;
    uint32_t N_pres = 0;

    uint32_t N_total() const { return N_vel + N_pres; }
};

Stokes3DData make_3d_stokes_test(uint32_t N, double alpha) {
    Stokes3DData out;
    out.m = std::make_shared<Stokes3DMatrices>();

    const uint32_t N3 = N * N * N;
    const uint32_t nv = 3 * N3;          // 速度自由度
    const uint32_t np = N3;              // 压力自由度
    out.N_vel  = nv;
    out.N_pres = np;
    const uint32_t N_total = nv + np;

    // alpha (Helmholtz 正则化参数) 由调用方传入:
    //   A = alpha*I + nu*L, 扫 alpha 可以观察 Schur 补里"质量项"与
    //   "Laplace 项"的权重变化 —— alpha 大时 S ~ Lp/alpha, AMG 才真正起作用。
    const uint32_t NN  = N * N;

    // idx(i,j,k) = i + j*N + k*N*N   (三方向周期边界)
    auto idx = [N, NN](uint32_t i, uint32_t j, uint32_t k) { return i + j * N + k * NN; };

    // 6 个周期邻居, 顺序与原先一致
    auto neighbors6 = [&](uint32_t p, uint32_t* nb) {
        const uint32_t i = p % N, j = (p / N) % N, k = p / NN;
        nb[0] = idx((i + 1) % N, j, k);     nb[1] = idx((i + N - 1) % N, j, k);
        nb[2] = idx(i, (j + 1) % N, k);     nb[3] = idx(i, (j + N - 1) % N, k);
        nb[4] = idx(i, j, (k + 1) % N);     nb[5] = idx(i, j, (k + N - 1) % N);
    };
    // 沿 comp 方向的"前一格" (MAC 错位网格的 -1/2 偏移)
    auto prev_node = [&](uint32_t comp, uint32_t p) -> uint32_t {
        const uint32_t i = p % N, j = (p / N) % N, k = p / NN;
        if (comp == 0) return idx((i + N - 1) % N, j, k);
        if (comp == 1) return idx(i, (j + N - 1) % N, k);
        return idx(i, j, (k + N - 1) % N);
    };
    // prev_node 的逆: 沿 comp 方向的"后一格"
    auto next_node = [&](uint32_t comp, uint32_t p) -> uint32_t {
        const uint32_t i = p % N, j = (p / N) % N, k = p / NN;
        if (comp == 0) return idx((i + 1) % N, j, k);
        if (comp == 1) return idx(i, (j + 1) % N, k);
        return idx(i, j, (k + 1) % N);
    };

    // ---------------------------------------------------------------
    // 组装策略: 逐行 nnz 是常数 -> 行偏移就是等差数列
    //   A  : 每行 7 (对角 + 6 邻居)
    //   Lp : 每行 7
    //   B  : 每行 6 (3 个分量 x 2)
    //   Bt : 每行 2 (curr 与 prev 各贡献一次)
    //   K  : 速度行 9 (A 的 7 + 两条压力耦合), 压力行 6
    // 于是根本不需要前缀和数组, 填充阶段行间无竞争(无需原子操作),
    // 每块 COO 一次性精确分配(零 realloc), 组装完立刻 drop 掉再装下一块。
    // 原来是无 reserve 的 push_back: 反复 realloc + 峰值 2x, 且五份 COO 同时在场。
    // ---------------------------------------------------------------
    const std::ptrdiff_t pnv = (std::ptrdiff_t)nv;
    const std::ptrdiff_t pnp = (std::ptrdiff_t)np;
    const std::ptrdiff_t pnt = (std::ptrdiff_t)N_total;

    // ---- A: 3 个独立的 3D 标量 Laplace + alpha*I ----
    {
        std::vector<double>   vv((size_t)nv * 7);
        std::vector<uint32_t> rl((size_t)nv * 7), cl((size_t)nv * 7);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t rr = 0; rr < pnv; ++rr)
        {
            const uint32_t r = (uint32_t)rr;
            const uint32_t comp = r / N3, p = r % N3;
            uint32_t w = r * 7;
            rl[w] = r; cl[w] = r; vv[w] = 6.0 + alpha; ++w;
            uint32_t nb[6]; neighbors6(p, nb);
            for (int d = 0; d < 6; ++d)
            { rl[w] = r; cl[w] = comp * N3 + nb[d]; vv[w] = -1.0; ++w; }
        }
        out.m->A = SMatrix(nv, nv, vv, rl, cl);
        drop(vv); drop(rl); drop(cl);
    }

    // ---- Lp: 压力 Laplace ----
    // 【关键改动】周期边界下 Lp 是奇异的 (零空间 = 常数向量), 必须加对角扰动。
    //   原来用 1e-8: 这样 Lp 的条件数 ≈ 1e9, 而 Schur 预处理器要用
    //   Lp^{-1} (S^{-1} 里的 alpha*Lp^{-1} 项) —— 零空间方向上会被放大 1e8 倍以上。
    //   更糟的是 AMG 最粗层是 50x50 稠密 LU, 最后一个主元掉到 1e-8 量级,
    //   一旦跌破代码里的 1e-14 clamp 就是 1e14 倍放大, GMRES 当场发散。
    //   这里取 1e-6: 对算子只有 1.7e-7 的相对扰动 (纯预处理器改动, 精确解不变),
    //   但把条件数压到 ~1e7, 粗层求解不再碰 clamp。
    const double kPresReg = 1e-6;
    {
        std::vector<double>   vv((size_t)np * 7);
        std::vector<uint32_t> rl((size_t)np * 7), cl((size_t)np * 7);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t pp = 0; pp < pnp; ++pp)
        {
            const uint32_t p = (uint32_t)pp;
            uint32_t w = p * 7;
            rl[w] = p; cl[w] = p; vv[w] = 6.0 + kPresReg; ++w;
            uint32_t nb[6]; neighbors6(p, nb);
            for (int d = 0; d < 6; ++d) { rl[w] = p; cl[w] = nb[d]; vv[w] = -1.0; ++w; }
        }
        out.m->Lp = SMatrix(np, np, vv, rl, cl);
        drop(vv); drop(rl); drop(cl);
    }

    // ---- B: 散度 (np x nv) ----
    {
        std::vector<double>   vv((size_t)np * 6);
        std::vector<uint32_t> rl((size_t)np * 6), cl((size_t)np * 6);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t pp = 0; pp < pnp; ++pp)
        {
            const uint32_t p = (uint32_t)pp;
            uint32_t w = p * 6;
            for (uint32_t comp = 0; comp < 3; ++comp)
            {
                const uint32_t u_curr = comp * N3 + p;
                const uint32_t u_prev = comp * N3 + prev_node(comp, p);
                rl[w] = p; cl[w] = u_curr; vv[w] =  1.0; ++w;
                rl[w] = p; cl[w] = u_prev; vv[w] = -1.0; ++w;
            }
        }
        out.m->B = SMatrix(np, nv, vv, rl, cl);
        drop(vv); drop(rl); drop(cl);
    }

    // ---- Bt: 负梯度 (nv x np), 即 B^T ----
    //      速度自由度 u=(comp,p) 恰好出现在两行 B 里:
    //      作为 u_curr 属于压力行 p   -> +1
    //      作为 u_prev 属于压力行 next_node(comp,p) -> -1
    {
        std::vector<double>   vv((size_t)nv * 2);
        std::vector<uint32_t> rl((size_t)nv * 2), cl((size_t)nv * 2);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t uu = 0; uu < pnv; ++uu)
        {
            const uint32_t u = (uint32_t)uu;
            const uint32_t comp = u / N3, p = u % N3;
            uint32_t w = u * 2;
            rl[w] = u; cl[w] = p;                  vv[w] =  1.0; ++w;
            rl[w] = u; cl[w] = next_node(comp, p); vv[w] = -1.0; ++w;
        }
        out.m->Bt = SMatrix(nv, np, vv, rl, cl);
        drop(vv); drop(rl); drop(cl);
    }

    // ---- K_full: 整体鞍点矩阵 ----
    {
        const size_t base = (size_t)nv * 9;          // 压力行区起点
        std::vector<double>   vv(base + (size_t)np * 6);
        std::vector<uint32_t> rl(vv.size()), cl(vv.size());
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t rr = 0; rr < pnt; ++rr)
        {
            const uint32_t r = (uint32_t)rr;
            if (r < nv)
            {
                const uint32_t comp = r / N3, p = r % N3;
                uint32_t w = r * 9;
                // A 部分
                rl[w] = r; cl[w] = r; vv[w] = 6.0 + alpha; ++w;
                uint32_t nb[6]; neighbors6(p, nb);
                for (int d = 0; d < 6; ++d)
                { rl[w] = r; cl[w] = comp * N3 + nb[d]; vv[w] = -1.0; ++w; }
                // B^T 部分: 与压力块 p 及 next_node(comp,p) 的耦合
                rl[w] = r; cl[w] = nv + p;                 vv[w] =  1.0; ++w;
                rl[w] = r; cl[w] = nv + next_node(comp, p); vv[w] = -1.0; ++w;
            }
            else
            {
                const uint32_t p = r - nv;
                uint32_t w = (uint32_t)(base + (size_t)p * 6);
                for (uint32_t comp = 0; comp < 3; ++comp)
                {
                    rl[w] = r; cl[w] = comp * N3 + p;                  vv[w] =  1.0; ++w;
                    rl[w] = r; cl[w] = comp * N3 + prev_node(comp, p); vv[w] = -1.0; ++w;
                }
            }
        }
        out.m->K_full = SMatrix(N_total, N_total, vv, rl, cl);
        drop(vv); drop(rl); drop(cl);
    }

    return out;
}

// =====================================================================
// 速度块预处理器: Block AMG (BS = 3)
// ---------------------------------------------------------------------
//   本组装下速度算子 A = alpha*I + nu*L 是「三个互不耦合的标量 7 点 Laplace」:
//       A[(comp,p), (comp',p')] = delta_{comp,comp'} * M[p,p'],  M = alpha*I + L
//   于是以「节点」为块行、三个速度分量为块内自由度时, 3x3 块正好是 M[p][p']*I_3,
//   块 Frobenius 范数 = |M[p][p']| * sqrt(3) —— 强度连接 / 聚合 / 延拓 / Galerkin
//   全部与标量情形一一对应, 所以 BS=3 的 Block AMG 在数学上等价于对 M 做标量 AMG。
//
//   既然等价, 为什么不直接对 M 做 3 次标量 AMG?
//     (a) 这是本项目 BCSR / Block-AMG 主线的真实用例;
//     (b) 一旦速度算子出现分量耦合 —— grad-div 稳定项 (div u, div v)、变黏度、
//         Coriolis 项 —— 3x3 块就不再是对角的, 标量拆分立刻失效, 而 BCSR 路径
//         无需任何改动。当前算例只是它的退化特例 (代价: 存储 9x, 浮点 3x)。
//
//   与 ILU(0) 的差别: ILU(0) 是串行的前代/回代, 迭代数随 cond(A) 增长
//     (alpha=0.1 时 N=16/24/32 的 cond(A) = 48/70/87) —— 这正是 alpha 较小时
//     S_pre=AMG 的迭代数仍然 45->55->63 上升的原因, 瓶颈在速度块而不在 Schur 补。
//     alpha*I + L 是多重网格的经典最优情形, 换掉 ILU(0) 后速度块不再随网格退化。
//
//   【排列】Schur 预处理器用的是「分量优先」编号 u = comp*N3 + p,
//   而 BCSR 需要「节点优先」编号 u = p*3 + comp, 所以 apply 前后各做一次
//   O(n_vel) 的置换。两趟都是纯 gather/scatter, 可并行。
// =====================================================================
static std::function<void(const std::vector<double>&, std::vector<double>&)>
make_velocity_block_amg(uint32_t N, double alpha, bool verbose,
                        double drop_tol = 0.02, uint32_t row_cap = 40,
                        int n_smooth = 2, bool allow_scalar = true)
{
    using Blk = BlockMatrix<double, 3>;
    const uint32_t NN = N * N;
    const uint32_t N3 = N * N * N;              // 块行数 = 节点数

    auto idx = [N, NN](uint32_t i, uint32_t j, uint32_t k) { return i + j * N + k * NN; };
    auto neighbors6 = [&](uint32_t p, uint32_t* nb) {
        const uint32_t i = p % N, j = (p / N) % N, k = p / NN;
        nb[0] = idx((i + 1) % N, j, k);     nb[1] = idx((i + N - 1) % N, j, k);
        nb[2] = idx(i, (j + 1) % N, k);     nb[3] = idx(i, (j + N - 1) % N, k);
        nb[4] = idx(i, j, (k + 1) % N);     nb[5] = idx(i, j, (k + N - 1) % N);
    };

    // 每行 7 个块; 块值 = M[p][q] * I_3  (对角 6+alpha, 六个邻居 -1)
    // 逐行块数恒定 -> 一次精确分配, 零 realloc; 行间无竞争。
    std::vector<uint32_t> rp((size_t)N3 + 1);
    std::vector<uint32_t> ci((size_t)N3 * 7);
    std::vector<double>   vv((size_t)N3 * 7 * 9, 0.0);

#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t b = 0; b < (std::ptrdiff_t)N3; ++b)
    {
        const uint32_t p = (uint32_t)b;
        uint32_t n6[6]; neighbors6(p, n6);
        uint32_t cols[7];
        cols[0] = p;
        for (int d = 0; d < 6; ++d) cols[d + 1] = n6[d];
        std::sort(cols, cols + 7);              // 块列升序, 符合 CSR 约定
        const uint32_t base = p * 7;
        rp[p] = base;
        for (int d = 0; d < 7; ++d)
        {
            ci[(size_t)base + d] = cols[d];
            const double m = (cols[d] == p) ? (6.0 + alpha) : -1.0;
            double* blk = &vv[(size_t)(base + d) * 9];
            blk[0] = blk[4] = blk[8] = m;       // m * I_3, 其余保持 0
        }
    }
    rp[N3] = N3 * 7;

    Blk Mvel(N3, N3, std::move(rp), std::move(ci), std::move(vv));
    auto amg = make_block_amg_preconditioner(Mvel, 8, verbose, drop_tol, row_cap,
                                             n_smooth, allow_scalar);

    // 工作缓冲随闭包走 —— 不用 static thread_local (理由同 make_schur_preconditioner:
    // 同一 lambda 只有一个闭包类型, static 成员在所有实例间共享, 且 thread_local
    // 的构造/析构与 OpenMP 线程池生命周期交织, 是多线程崩溃源)。
    struct WS { std::vector<double> a, b; };
    auto ws = std::make_shared<WS>();

    return [amg, ws, N3](const std::vector<double>& r, std::vector<double>& z)
    {
        const std::size_t n = r.size();
        if (ws->a.size() != n) { ws->a.resize(n); ws->b.resize(n); }
        std::vector<double>& ra = ws->a;
        std::vector<double>& rb = ws->b;
        const std::ptrdiff_t pn3 = (std::ptrdiff_t)N3;

        // comp-major (comp*N3 + p)  ->  node-major (p*3 + comp)
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static) if(n > 4096)
#endif
        for (std::ptrdiff_t p = 0; p < pn3; ++p)
            for (int c = 0; c < 3; ++c)
                ra[(size_t)p * 3 + c] = r[(size_t)c * N3 + (size_t)p];

        amg(ra, rb);

        // node-major -> comp-major
        z.resize(n);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static) if(n > 4096)
#endif
        for (std::ptrdiff_t p = 0; p < pn3; ++p)
            for (int c = 0; c < 3; ++c)
                z[(size_t)c * N3 + (size_t)p] = rb[(size_t)p * 3 + c];
    };
}

// =====================================================================
// 进度输出
//   【关键】必须显式 flush。stdout 被重定向到文件或管道时是块缓冲 (4 KB),
//   一行几十字节的提示会一直憋在缓冲区里 —— 这就是"跑了一分钟毫无反应"
//   的直接原因之一: 程序其实在算, 只是没吐字。
// =====================================================================
static void logf(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

// =====================================================================
// 结果
// =====================================================================
struct SchurResult {
    uint32_t N        = 0;   // 每边网格数
    uint32_t dof      = 0;   // 总自由度
    double alpha      = 0.0; // Helmholtz 正则化参数
    bool   amg_ok     = false;  // AMG 体检是否通过 (未通过则降级为 S_pre=I)
    int    it_ident = 0;     // 配置1: A^-1=ILU(0),  S^-1=I
    int    it_amg   = 0;     // 配置2: A^-1=ILU(0),  S^-1=AMG(Lp)
    int    it_bamg  = 0;     // 配置3: A^-1=BAMG(3), S^-1=AMG(Lp)
    double rel      = 0.0;   // 相对误差 ||x-x*|| / ||x*|| (取配置3 的解)
    double t_setup  = 0.0;   // 组装 + 预处理
    double t_solve  = 0.0;   // 三次 GMRES 合计
    double t_solve_I    = 0.0;
    double t_solve_amg  = 0.0;
    double t_solve_bamg = 0.0;
    bool   use_bamg   = false;  // 自适应判据: 速度块是否该选 Block AMG
    double cond_A     = 0.0;    // 判据用的 cond(A) 估计
    bool   bamg_built = true;   // 本次是否真的建了 BAMG 层次 (--adapt 下可能跳过)
    int    it_auto    = 0;      // 配置4(=自适应): 由判据在 ILU(0)/BAMG 中选一个
    double t_auto     = 0.0;

    // ---- 度量 (见 gmres_best_of 与"度量口径"处的注释) ----
    int    n_rep    = 1;    // 每个配置重复求解几次, 取最快的一次作代表值
    double t_best   = 0.0;  // 三个固定配置里最快的那一个 —— 事后最优 (oracle)
    int    best_cfg = 0;    // 1/2/3 = ILU0+I / ILU0+AMG / BAMG+AMG
    double spread   = 1.0;  // 计时离散度 max(t_hi/t_lo), 1.00 = 完全稳定
    bool   noisy    = false;// 最快的求解总时长 <30 ms, 比值落在噪声量级
    double ms_i_I    = 0.0; // 单次迭代耗时 (ms/iter) —— 噪声免疫的选型指标
    double ms_i_amg  = 0.0;
    double ms_i_bamg = 0.0;
    double ms_i_auto = 0.0;
};

// =====================================================================
// AMG 体检
//   在用 AMG 当 Schur 预处理器之前, 先单独测一次它: 解 Lp w = rhs,
//   看一次 V-cycle 能把残差降多少, 输出有没有被零空间放大。
//   (a) rhs = Lp * x_true  —— 在值域内, 正常情况残差应有明显下降
//   (b) rhs = 随机         —— 含 O(1) 的常数分量, 专门用来暴露零空间放大:
//       若 |mean(w)| 或 |w|max 远大于 |rhs| 的量级, 说明粗层求解碰到了
//       近奇异主元, 这个 AMG 不能直接拿去当预处理器。
// =====================================================================
struct AmgHealth {
    double res_in_range = 0.0;
    double res_random   = 0.0;
    double wmax         = 0.0;
    double wmean_abs    = 0.0;
    double rhs_scale    = 0.0;
    bool   finite       = true;
};

static AmgHealth check_amg(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& amg,
    const SMatrix& Lp)
{
    AmgHealth h;
    const std::ptrdiff_t n = (std::ptrdiff_t)Lp.rows();
    std::vector<double> x_true(n), rhs(n), w(n), r(n);

#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) x_true[i] = drand((uint32_t)i);

    // (a) rhs 在值域内
    SMatrix::spmv_into(Lp, x_true, rhs);
    amg(rhs, w);
    SMatrix::spmv_into(Lp, w, r);
    {
        double num = 0.0, den = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for reduction(+:num) reduction(+:den) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < n; ++i)
        { const double d = rhs[i] - r[i]; num += d * d; den += rhs[i] * rhs[i]; }
        h.res_in_range = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    }

    // (b) rhs 随机
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) rhs[i] = drand((uint32_t)(i + 7919));
    amg(rhs, w);
    {
        // 【判据必须去均值】Lp 在周期边界下是近奇异的, Lp^{-1} 的范数本来就是
        //   1/lambda_min ≈ 1e6。所以直接看 |w|max 一定会"超标", 但超标的那部分
        //   全部落在常数(零空间)分量上 —— 那是数学上应有的行为, 不是 AMG 坏了。
        //   Schur 预处理器真正要用的也是伪逆 (见 S_pre_amg 的注释), 所以判据
        //   要看的是「去掉均值之后」的量级, 那才是 Lp 在值域上的有效逆范数
        //   (N=16 时 1/lambda_2 ≈ 6.5)。
        double wsum = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for reduction(+:wsum) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < n; ++i) wsum += w[i];
        const double mean = wsum / (double)n;

        double wmax = 0.0, rs = 0.0;
        int bad = 0;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for reduction(max:wmax) reduction(+:rs) \
                                 reduction(+:bad) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < n; ++i)
        {
            const double a = std::fabs(w[i] - mean);
            if (a > wmax) wmax = a;
            rs += std::fabs(rhs[i]);
            if (!std::isfinite(w[i])) ++bad;
        }
        h.wmax      = wmax;                 // 去均值后的量级 (值域上的有效逆范数)
        h.wmean_abs = std::fabs(mean);      // 零空间分量的量级 (仅作参考)
        h.rhs_scale = rs / (double)n;
        h.finite    = (bad == 0);
    }

    SMatrix::spmv_into(Lp, w, r);
    {
        double num = 0.0, den = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for reduction(+:num) reduction(+:den) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < n; ++i)
        { const double d = rhs[i] - r[i]; num += d * d; den += rhs[i] * rhs[i]; }
        h.res_random = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    }
    return h;
}

// =====================================================================
// 速度块预处理器的自适应选择: ILU(0) 还是 Block AMG(3)?
// ---------------------------------------------------------------------
//   【为什么必须有这一步】换 BAMG 之后迭代数确实拉平了 (alpha=0.1 时
//   45/55/63 -> 17/17/22), 但墙钟只在 alpha=0.1 赢, alpha=1/10 反而亏
//   1.9~6.5 倍。原因是 BAMG 单次 apply 的单价是 ILU(0) 的 2.1~2.7 倍
//   (N=32 实测 12.3/14.7/13.6 ms vs 5.4/5.4/6.4 ms), 于是
//       净收益 = 迭代数下降倍数 / 单价倍数
//       alpha=0.1   68->22 步 (3.09x) / 2.28x = 净快 1.37x   -> BAMG
//       alpha=1     24->17 步 (1.41x) / 2.72x = 净慢 1.92x   -> ILU(0)
//       alpha=10    14->14 步 (1.00x) / 2.13x = 净慢 2.11x   -> ILU(0)
//   三组预测与实测墙钟吻合到 5% 以内, 说明决定因素就是「单价 vs 迭代数」。
//
//   【判据】A = alpha*I + nu*L, 3D 7 点 Laplace 的谱 lambda(L) in [0, 12],
//   周期边界下最小非零特征值 lambda1 = 4 sin^2(pi/N), 于是
//       cond(A) = (alpha + 12*nu) / (alpha + nu*lambda1)
//     alpha=0.1 -> 48 / 70 / 87   (随 N 涨, ILU(0) 必然退化)
//     alpha=1   -> 11.3 / 12.5
//     alpha=10  -> 2.17 / 2.19    (几乎对角阵, ILU(0) 一步就够)
//   阈值取 20: 只有 cond(A) > 20 才值得付 BAMG 那 2~3 倍的单价。
//   (阈值是拿本算例的实测墙钟标定的, 换算子要重新标定; 但
//    "cond 越大越该上多重网格" 这个方向是普适的。)
// =====================================================================
static double velocity_cond_est(double alpha, uint32_t N, double nu = 1.0)
{
    const double kPi = 3.14159265358979323846;
    const double s   = std::sin(kPi / (double)N);
    const double l1  = 4.0 * s * s;            // lambda_min(L), 周期边界
    return (alpha + 12.0 * nu) / (alpha + nu * l1);
}
// cond(A) 阈值: 只有 cond(A) > 阈值 才值得付 Block AMG 那笔单价。
//   默认 20 是拿 smooth=2 的实测墙钟标定的: 那时 BAMG 单价是 ILU(0) 的 1.8 倍,
//   盈亏线是"迭代下降 > 1.8x", 而 20 正好落在 alpha=1 (cond 12.5) 和
//   alpha=0.1 (cond 48~87) 之间 —— 实测 9/9 命中事后最优。
//
//   【阈值与单价强耦合, 别只改一边】--amg-smooth 一改, 单价就变, 盈亏线随之
//   移动, 阈值必须重新标定, 否则自适应会"正确地做错误的选择"。
//
//   【已实测: smooth=1 时阈值不应下调, 保持 20】
//     smooth=1 确实把单价压到 1.49 (事前的预测是 1.45~1.55, 单价这半命中),
//     但 BAMG 的迭代数同时涨了 14%~46%, 于是 alpha=1 的迭代下降从 1.41x 掉到
//     1.14x —— 反而更低于新的盈亏线 1.49x。
//     实测: --amg-smooth 1 --adapt-thr 5  -> 命中 6/9, 平均 0.923
//           (alpha=1 三例全错, 各亏约 25%)
//           --amg-smooth 1 --adapt-thr 20 -> 命中 9/9, 平均 1.000
//     教训: 单价降了, 不等于盈亏线降得比迭代收益快。判据要跟着"迭代下降倍数"
//     一起重标, 只看单价会把阈值调反。
//   用 --adapt-thr 扫一遍就能找到新的盈亏点。
static double g_cond_thr = 20.0;
static bool velocity_use_bamg(double alpha, uint32_t N, double nu = 1.0)
{
    return velocity_cond_est(alpha, N, nu) > g_cond_thr;
}

// =====================================================================
// best-of-n 求解计时
// ---------------------------------------------------------------------
//   【为什么必须有】N=16 时一次求解只有 15~25 ms, 而同一份代码、同一个配置,
//   两次运行的墙钟能差 30% 以上 —— 实测同一个 (N=16, alpha=0.1), 上一轮日志
//   报加速 1.34x, 这一轮报 0.77x, 结论直接反号。在这个量级上任何比值都是
//   噪声, 拿它做判断会推出相反的结论。
//
//   【为什么取 min 而不是平均】抖动来自 OS 调度 / 缺页 / turbo 降频, 全是
//   单向把时间拉长的因素; 最小值才是"这段代码在稳定状态下的成本", 取平均
//   等于把这些一次性干扰算进了算法成本里。
//
//   【附带的一致性检查】GMRES 是确定性的, 重复求解的迭代数必须逐次相同。
//   一旦不同, 就说明有未初始化内存或数据竞争 —— 这正是之前那次 8 线程
//   段错误的同类症状, 值得直接报警, 而不是默默取 min 糊过去。
// =====================================================================
struct SolveRun {
    std::vector<double> x;
    int    it = 0;          // gmres 返回值, 调用处 +1 得到迭代数
    double t_lo = 0.0;      // n 次里最快的一次 (代表值)
    double t_hi = 0.0;      // n 次里最慢的一次 (用来看离散度)
    bool   consistent = true;
};

static SolveRun gmres_best_of(
    const SMatrix& K, const std::vector<double>& b,
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& pre,
    int restart, int n_rep)
{
    SolveRun r;
    r.t_lo = 1e300;
    r.it   = -1;
    for (int k = 0; k < n_rep; ++k) {
        const auto t0 = Clock::now();
        auto [x, it] = SMatrix::gmres(K, b, pre, restart);
        const double t = elapsed(t0, Clock::now());
        if (r.it >= 0 && it != r.it) r.consistent = false;
        if (t < r.t_lo) { r.t_lo = t; r.it = it; r.x = std::move(x); }
        if (t > r.t_hi) r.t_hi = t;
    }
    if (r.it < 0) { r.it = 0; r.t_lo = 0.0; }
    return r;
}

// 单次迭代耗时 (ms/iter): 唯一兼具"可比"和"稳定"的选型指标。
//   净墙钟收益 = 迭代数下降倍数 / 单价(ms/iter)倍数 —— 两半都要它。
static double ms_per_iter(double t, int iters)
{
    return (iters > 0) ? (t * 1e3 / (double)iters) : 0.0;
}
static double min3(double a, double b, double c)
{
    double m = a; if (b < m) m = b; if (c < m) m = c; return m;
}

// =====================================================================
// 核心测试：GMRES + Schur 补 + Block AMG
// =====================================================================
SchurResult test_3d_schur_with_amg(uint32_t N, int restart,
                                   double alpha, bool verbose_amg,
                                   double drop_tol = 0.02, uint32_t row_cap = 40,
                                   bool adapt = false, int n_rep = 3,
                                   int n_smooth = 2, bool allow_scalar = true) {
    SchurResult res;

    const uint32_t N3  = N * N * N;
    const uint32_t dof = 4 * N3;
    // 先报规模: GMRES 的 Krylov 基要 (restart+1) * DOF * 8 字节,
    // Arnoldi 正交化的访存量 ~ O(restart^2 * DOF), 这两个数决定了能不能跑。
    logf("  [N=%u] DOF=%u | K 非零元≈%.1fM | GMRES 基 V≈%.0f MB\n",
         N, dof, (33.0 * (double)N3) / 1e6,
         (double)(restart + 1) * (double)dof * 8.0 / 1e6);

    const auto t0 = Clock::now();
    auto data = make_3d_stokes_test(N, alpha);
    const uint32_t N_total = data.N_total();
    const std::ptrdiff_t pn = (std::ptrdiff_t)N_total;
    // alpha 必须与组装 A 时用的完全一致:
    //   S^{-1} = nu*Mp^{-1} + alpha*Lp^{-1}, 系数错了预处理器就不再是 S^{-1}。
    res.N     = N;
    res.alpha = alpha;
    res.dof   = N_total;
    logf("  [N=%u] 五块矩阵组装完成 (%.2f s), 开始建预处理器...\n",
         N, elapsed(t0, Clock::now()));

    // 1. 速度块预处理器, 两个方案对比
    //   (a) ILU(0) —— 基线。串行前代/回代, 迭代数随 cond(A) 增长。
    auto A_precond = SMatrix::make_ilu0_preconditioner(data.m->A);
    logf("  [N=%u] ILU(0) on A 完成 (%.2f s)\n", N, elapsed(t0, Clock::now()));

    //   (b) Block AMG (BS=3) —— 见 make_velocity_block_amg 的详细注释。
    //       alpha*I + L 是多重网格的经典最优情形; M 的对角依赖 alpha,
    //       所以每个 alpha 都要重建一次层次。
    //
    //   【自适应】--adapt 且判据说"这个 alpha 用不上"时, 连层次都不建:
    //       N=32 建一次 BAMG 层次约 0.24 s, 而一次 GMRES 才 0.2~0.3 s ——
    //       单右端项场景下白建一次就足以吃掉全部收益。此时配置 3 直接
    //       退化成 ILU(0) (即与配置 2 相同), 汇总表里会标注 (ILU)。
    const double cond_A     = velocity_cond_est(alpha, N);
    const bool   use_bamg   = velocity_use_bamg(alpha, N);
    const bool   bamg_built = (!adapt) || use_bamg;
    res.cond_A = cond_A; res.use_bamg = use_bamg; res.bamg_built = bamg_built;
    logf("  [N=%u] 自适应判据: cond(A)≈%.1f (阈值 %g) -> 速度块选 %s%s\n",
         N, cond_A, g_cond_thr, use_bamg ? "BAMG(3)" : "ILU(0)",
         bamg_built ? "" : "   [跳过 BAMG 层次构建]");
    const char* cfg3_name = bamg_built ? "BAMG(3)" : "ILU(0)";

    std::function<void(const std::vector<double>&, std::vector<double>&)> A_pre_bamg;
    if (bamg_built) {
        A_pre_bamg = make_velocity_block_amg(N, alpha, verbose_amg, drop_tol, row_cap,
                                             n_smooth, allow_scalar);
        logf("  [N=%u] Block AMG (BS=3) on A 完成 (%.2f s)\n", N, elapsed(t0, Clock::now()));
    } else {
        A_pre_bamg = A_precond;
    }

    // 2. 压力块预处理器 (Block AMG)
    //    将标量 SMatrix 凝聚为 BS=1 的 BlockMatrix, 以适配 Block AMG 接口
    BlockMatrix<double, 1> Lp_blk = BlockMatrix<double, 1>::condense(
        data.m->Lp, data.m->Lp.rows(), data.m->Lp.cols());
    // 三维下用更大的聚合上限: 默认 4 的粗化比太低, 粗算子模板每级膨胀,
    // setup 时间按 n^2 涨。
    auto Lp_amg = make_block_amg_preconditioner(Lp_blk, 8, /*verbose=*/verbose_amg,
                                                drop_tol, row_cap, n_smooth);
    logf("  [N=%u] Block AMG on Lp 完成 (%.2f s)\n", N, elapsed(t0, Clock::now()));

    // 体检: 不合格就降级, 不用一个会输出 1e11 垃圾的预处理器去跑 GMRES
    bool amg_ok = false;
    {
        const AmgHealth hh = check_amg(Lp_amg, data.m->Lp);
        if (verbose_amg) logf("  [N=%u] AMG 体检: 残差(rhs∈值域)=%.3e 残差(rhs 随机)=%.3e | "
             "|w-mean|max=%.3e |mean(w)|=%.3e (|rhs|~%.3e) | %s\n",
             N, hh.res_in_range, hh.res_random, hh.wmax, hh.wmean_abs, hh.rhs_scale,
             hh.finite ? "有限" : "!! 含 NaN/Inf !!");
        amg_ok = hh.finite
              && (hh.wmax <= 1e3 * (hh.rhs_scale + 1e-300))   // 值域上逆范数正常
              && (hh.res_in_range < 0.9);                      // V-cycle 确实在收敛
        if (!amg_ok && verbose_amg)
            logf("  [N=%u]   ^^^ AMG 体检不合格, 本次 S_pre 降级为 I "
                 "(把上面 [BlockAMG] 的分层信息发来即可定位)\n", N);
    }
    res.amg_ok = amg_ok;

    // 3. Schur 预处理器 —— Cahouet-Chabard 形式
    //
    //   【旧公式是错的】原来用 S_pre^{-1} = (6+alpha)*Lp^{-1}, 那是
    //   "B*diag(A)^{-1}*B^T" 型近似, 只在 A 被对角占优时才成立。本算例里
    //   A 的非对角部分就是整个 Laplace, 对角近似完全失效:
    //     N=16 时 S_pre^{-1}S 的谱被拉宽到 [0.55, 10.9] (cond ~20),
    //     而 S_pre = I 反而是 [0.82, 0.99] (cond ~1.2)
    //   —— 这就是日志里"AMG 反而更慢"的真正原因, 不是 AMG 不好。
    //
    //   【正确公式】由 B B^T = Lp 与 A = alpha*I + nu*L 可精确推出
    //     S^{-1} = nu*Mp^{-1} + alpha*Lp^{-1}      (本算例 nu = 1, Mp = I)
    //   换成它之后 S_pre^{-1}S 的谱恒等于 1, 与网格 N、参数 alpha 都无关。
    //
    //   【内存】mutable lambda 自带缓冲, 每次 apply 不再新分配一个 n 长 vector
    const double nu = 1.0;      // A 中 Laplace 的系数 (本组装为 1)
    auto S_pre_amg = [Lp_amg, alpha, nu, w_buf = std::vector<double>{}](
                         const std::vector<double>& r, std::vector<double>& z) mutable
    {
        w_buf.resize(r.size());
        Lp_amg(r, w_buf);                       // AMG 求解 L_p w = r
        z.resize(r.size());
        const std::ptrdiff_t n = (std::ptrdiff_t)r.size();

        // 【关键】Lp^{-1} 用伪逆而不是真逆: 投影掉常数分量。
        //   周期边界下 Lp 的零空间是常数向量, 1/lambda_min 可达 1e6 以上;
        //   rhs 里只要有一点点常数分量, 解就会被放大到 1e6, 直接把 GMRES
        //   的搜索方向撑爆 (这就是原来相对误差 1e+47 的来源)。
        //   去掉均值是安全的: B^T * 1 = 0, 常数分量对速度块没有任何作用,
        //   留在 z_p 里只是给校正向量加了一个 K 的零空间分量。
        double mean = 0.0;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:mean) schedule(static) if(n > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < n; ++i) mean += w_buf[i];
        mean /= (double)n;

        // 两项融合成一次遍历: 访存 3n 而不是 5n (分开写要各读写一遍 z),
        // 且单次线性扫描可被编译器向量化。
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(n > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < n; ++i)
            z[i] = nu * r[i] + alpha * (w_buf[i] - mean);
    };

    // Baseline: S_pre = I
    auto S_pre_I = [](const std::vector<double>& r, std::vector<double>& z) { z = r; };

    // 体检不合格时用它顶上, 保证测试仍能给出有意义的迭代数
    std::function<void(const std::vector<double>&, std::vector<double>&)> S_pre_used;

    // 4. 鞍点系统接口 —— 【关键】值捕获 shared_ptr, 零拷贝且生命周期安全。
    //    spmv_into 复用 y 的缓冲区: 这些 lambda 在每次预处理器 apply 里都要
    //    各调一次, GMRES 跑几百步就是几百次 malloc/free。
    SaddlePointSystem sys;
    sys.n_vel  = data.N_vel;
    sys.n_pres = data.N_pres;
    auto mats = data.m;
    sys.spmv_A  = [mats](const std::vector<double>& x, std::vector<double>& y)
                  { SMatrix::spmv_into(mats->A,  x, y); };
    sys.spmv_B  = [mats](const std::vector<double>& x, std::vector<double>& y)
                  { SMatrix::spmv_into(mats->B,  x, y); };
    sys.spmv_Bt = [mats](const std::vector<double>& x, std::vector<double>& y)
                  { SMatrix::spmv_into(mats->Bt, x, y); };

    if (amg_ok) S_pre_used = S_pre_amg;
    else        S_pre_used = S_pre_I;

    // 5. 右端项 b = K x_exact
    std::vector<double> x_exact(N_total), b(N_total);
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i) x_exact[i] = drand((uint32_t)i);
    logf("  [N=%u] 组装 b = K*x_exact ... ", N);
    SMatrix::spmv_into(data.m->K_full, x_exact, b);
    logf("ok\n");

    const auto t1 = Clock::now();
    res.t_setup = elapsed(t0, t1);

    // 6. 两次求解对比
    //   【计时】两次各记各的。原来两行都用 elapsed(t1, ...) —— 那是累计值,
    //   第二行看到的秒数其实包含了第一次求解, 会把 AMG 读成"很慢"。
    auto schur_I = make_schur_preconditioner(sys, A_precond, S_pre_I);

    // ------------------------------------------------------------------
    // 分段多线程预检
    //   8 线程下曾在"第一次 Schur apply"处段错误。这里把 Schur apply 的每个子
    //   步骤单独跑一遍, 并且 *进入前* 就打印段名 (logf 带 fflush), 于是崩溃时
    //   屏幕上最后那一行段名就是崩溃点 —— 一次运行即可定位, 不需要 gdb。
    //   段 1~3 直接打裸算子 (ILU0 / B / B^T), 段 4~5 走完整 Schur 并重复一次,
    //   重复结果比对能捕获工作缓冲区上的数据竞争。
    // ------------------------------------------------------------------
    bool bamg_finite = true;
    {
        std::vector<double> probe(N_total), z0, z1;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < pn; ++i)
            probe[(size_t)i] = drand((uint32_t)i + 7919u) - 0.5;

        const std::size_t nv = sys.n_vel, np = sys.n_pres;
        std::vector<double> vv(nv), pp(np), oo(nv);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)nv; ++i)
            vv[(size_t)i] = drand((uint32_t)i + 104729u) - 0.5;

        logf("  [N=%u] 预检[1/6] ILU0(A) apply (串行, n=%zu) ... ", N, nv);
        A_precond(vv, oo);
        logf("ok\n");

        // 速度块 Block AMG 是新增的多线程路径, 单独体检一次
        logf("  [N=%u] 预检[2/6] BAMG(BS=3)(A) apply (多线程, n=%zu) ... ", N, nv);
        A_pre_bamg(vv, oo);
        {
            int bad = 0;
#ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:bad) schedule(static)
#endif
            for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)nv; ++i)
                if (!std::isfinite(oo[(size_t)i])) ++bad;
            bamg_finite = (bad == 0);
        }
        logf("%s\n", bamg_finite ? "ok" : "!! 含 NaN/Inf !!");

        logf("  [N=%u] 预检[3/6] spmv  B   (%zu x %zu) ... ", N, np, nv);
        sys.spmv_B(oo, pp);
        logf("ok -> %zu\n", pp.size());

        logf("  [N=%u] 预检[4/6] spmv  B^T (%zu x %zu) ... ", N, nv, np);
        sys.spmv_Bt(pp, oo);
        logf("ok -> %zu\n", oo.size());

        logf("  [N=%u] 预检[5/6] Schur(I) apply #1 ... ", N);
        schur_I(probe, z0);
        logf("ok -> %zu\n", z0.size());

        logf("  [N=%u] 预检[6/6] Schur(I) apply #2 (重复比对) ... ", N);
        schur_I(probe, z1);
        logf("ok -> %zu\n", z1.size());

        if (z0.size() != N_total || z1.size() != N_total)
            throw std::runtime_error("Schur 多线程预检失败：输出长度错误");
        double diff2 = 0.0, ref2 = 0.0;
        sqdist_par(z0, z1, diff2, ref2);
        int bad = 0;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for reduction(+:bad) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < pn; ++i)
            if (!std::isfinite(z0[(size_t)i]) || !std::isfinite(z1[(size_t)i])) ++bad;
        const double repeat_rel = (ref2 > 0.0) ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
        if (bad != 0 || repeat_rel > 1e-13)
            throw std::runtime_error("Schur 多线程预检失败：出现非有限值或重复结果不一致");
        if (verbose_amg)
            logf("  [N=%u] Schur(I) 多线程预检通过 (重复差 %.1e)\n", N, repeat_rel);
        if (!bamg_finite)
            logf("  [N=%u]   ^^^ 速度块 Block AMG 输出含 NaN/Inf, "
                 "配置3 的迭代数不可信\n", N);
    }

    // ---- 配置 1: A^-1 = ILU(0), S^-1 = I   (基线) ----
    logf("  [N=%u] GMRES[1/3] A=ILU(0)  S=I   (best-of-%d) 开始...\n", N, n_rep);
    const SolveRun r1 = gmres_best_of(data.m->K_full, b, schur_I, restart, n_rep);
    res.t_solve_I = r1.t_lo;
    logf("  [N=%u]   -> %d 步 (%.2f s | %.2f ms/iter)\n", N, r1.it + 1, r1.t_lo,
         ms_per_iter(r1.t_lo, r1.it + 1));

    // ---- 配置 2: A^-1 = ILU(0), S^-1 = AMG(Lp) ----
    auto schur_amg = make_schur_preconditioner(sys, A_precond, S_pre_used);
    logf("  [N=%u] GMRES[2/3] A=ILU(0)  S=%s (best-of-%d) 开始...\n", N,
         amg_ok ? "AMG" : "I(降级)", n_rep);
    const SolveRun r2 = gmres_best_of(data.m->K_full, b, schur_amg, restart, n_rep);
    res.t_solve_amg = r2.t_lo;
    logf("  [N=%u]   -> %d 步 (%.2f s | %.2f ms/iter)\n", N, r2.it + 1, r2.t_lo,
         ms_per_iter(r2.t_lo, r2.it + 1));

    // ---- 配置 3: A^-1 = Block AMG(BS=3), S^-1 = AMG(Lp)  (完整配置) ----
    auto schur_full = make_schur_preconditioner(sys, A_pre_bamg, S_pre_used);
    logf("  [N=%u] GMRES[3/3] A=%s S=%s (best-of-%d) 开始...\n", N, cfg3_name,
         amg_ok ? "AMG" : "I(降级)", n_rep);
    const SolveRun r3 = gmres_best_of(data.m->K_full, b, schur_full, restart, n_rep);
    res.t_solve_bamg = r3.t_lo;
    logf("  [N=%u]   -> %d 步 (%.2f s | %.2f ms/iter)\n", N, r3.it + 1, r3.t_lo,
         ms_per_iter(r3.t_lo, r3.it + 1));

    if (!r1.consistent || !r2.consistent || !r3.consistent)
        logf("  [N=%u]   ^^^ 重复求解的迭代数不一致 —— 未初始化内存或数据竞争\n", N);

    const auto t2 = Clock::now();
    res.t_solve = elapsed(t1, t2);
    res.n_rep   = n_rep;
    {
        double sp = 1.0;
        const double s1 = (r1.t_lo > 0.0) ? r1.t_hi / r1.t_lo : 1.0;
        const double s2 = (r2.t_lo > 0.0) ? r2.t_hi / r2.t_lo : 1.0;
        const double s3 = (r3.t_lo > 0.0) ? r3.t_hi / r3.t_lo : 1.0;
        if (s1 > sp) sp = s1;
        if (s2 > sp) sp = s2;
        if (s3 > sp) sp = s3;
        res.spread = sp;
    }

    res.it_ident = r1.it + 1;
    res.it_amg   = r2.it + 1;
    res.it_bamg  = r3.it + 1;
    // 自适应配置不额外跑一遍 GMRES: 它就是从配置 2 / 配置 3 里挑一个
    res.it_auto = res.use_bamg ? res.it_bamg : res.it_amg;
    res.t_auto  = res.use_bamg ? res.t_solve_bamg : res.t_solve_amg;

    // 7. 相对误差
    //   【关键】K 是奇异的: 周期边界下压力只差一个任意常数 (K*[0;1] = 0)。
    //   GMRES 收敛到的是 *某个* 解, 它与 x_exact 可以差一个压力常数,
    //   于是 ||x-x*||/||x*|| 会稳定停在 O(1) (实测 0.43) —— 这不是求解失败,
    //   是度量方式错了。正确做法: 两段都先去掉压力块的均值再比。
    //   (b = K*x_exact 自动满足相容条件: 压力块 sum(B*u) = 0。)
    std::vector<double> xd = r3.x, xe = x_exact;
    {
        const std::ptrdiff_t nv = (std::ptrdiff_t)data.N_vel;
        const std::ptrdiff_t np = (std::ptrdiff_t)data.N_pres;
        double mp = 0.0, me = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for reduction(+:mp) reduction(+:me) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < np; ++i) { mp += xd[nv + i]; me += xe[nv + i]; }
        if (np > 0) { mp /= (double)np; me /= (double)np; }
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < np; ++i) { xd[nv + i] -= mp; xe[nv + i] -= me; }
    }
    double err2 = 0.0, ref2 = 0.0;
    sqdist_par(xd, xe, err2, ref2);
    res.rel = (ref2 > 0.0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);

    // 网格无关性观察: AMG 迭代数不应随 N 增长, 且不应劣于 S_pre=I
    const bool 通过   = (res.rel < 1e-6);
    // 判据留 1 步容差: 迭代数是离散量, 差 1~2 步只是收敛路径的噪声,
    //   不等于"Schur 近似失效"。真正该看的是迭代数随 N 是否增长 (见末尾汇总)。
    // 现在看的是「完整配置 vs 基线」: 速度块和 Schur 补都换成多重网格之后,
    //   迭代数应当不随 N 增长, 且不劣于只有 Schur 补的配置。
    const bool 未劣化 = (res.it_bamg <= res.it_ident + 1);

    // ---- 度量口径 (这一版改掉了两个会骗人的地方) ----
    //   1. 分母换成「事后最优」。原来 加速 / 自适应 两列都以基线 (ILU0 + S=I)
    //      作分母, 而基线是三个配置里最弱的那个 —— 于是「Schur 补从 I 换成
    //      AMG」的收益被算到了「速度块换成 BAMG」头上。N=32/alpha=10 报出的
    //      "自适应 4.66x" 里绝大部分其实是 0.40/0.09, 即 Schur 补的功劳,
    //      跟自适应判据本身一点关系都没有, 而且高得无法解释。
    //   2. 判据的真实损失 = t_best / t_auto:
    //        t_best = 三个固定配置里最快的那一个 (事后才知道, 即 oracle)
    //      1.00 = 判据恰好选中了最优配置; <1 = 选错, 差多少一目了然。
    //      这个比值与基线无关, 只评价判据本身, 跨 alpha / 跨 N 可直接比较。
    //   3. 噪声闸门: 最快的那次求解总时长不足 30 ms 时, OS 抖动与信号同量级,
    //      比值打上 ~ 前缀 —— 可以定性看, 别拿它下结论。想压掉噪声就加大
    //      --repeat (代价是运行时间线性变长)。
    const double t_best = min3(res.t_solve_I, res.t_solve_amg, res.t_solve_bamg);
    res.t_best   = t_best;
    res.best_cfg = (t_best == res.t_solve_I)   ? 1
                 : (t_best == res.t_solve_amg) ? 2 : 3;
    res.ms_i_I    = ms_per_iter(res.t_solve_I,    res.it_ident);
    res.ms_i_amg  = ms_per_iter(res.t_solve_amg,  res.it_amg);
    res.ms_i_bamg = ms_per_iter(res.t_solve_bamg, res.it_bamg);
    res.ms_i_auto = ms_per_iter(res.t_auto,       res.it_auto);
    res.noisy     = (t_best < 0.03);

    const double vs基线 = (res.t_solve_bamg > 1e-9) ? (res.t_solve_I / res.t_solve_bamg) : 0.0;
    const double vs最优 = (res.t_auto > 1e-9) ? (t_best / res.t_auto) : 0.0;

    const char* auto_name = res.use_bamg ? "BAMG3" : "ILU0 ";
    const char* best_name = (res.best_cfg == 1) ? "ILU0+I  "
                          : (res.best_cfg == 2) ? "ILU0+AMG" : "BAMG3   ";
    char sv[32], sa[32];
    std::snprintf(sv, sizeof sv, "%s%.2fx", res.noisy ? "~" : "", vs基线);
    std::snprintf(sa, sizeof sa, "%s%.2fx", res.noisy ? "~" : "", vs最优);

    logf("[3D Stokes] N=%3u a=%-5g DOF=%8u | ILU0+I %4d步 %7.2fms/i | ILU0+%s %4d步 %7.2fms/i | "
         "%s+%s %4d步 %7.2fms/i\n",
         N, alpha, N_total,
         res.it_ident, res.ms_i_I,
         amg_ok ? "AMG" : "I  ", res.it_amg, res.ms_i_amg,
         cfg3_name, amg_ok ? "AMG" : "I", res.it_bamg, res.ms_i_bamg);
    logf("             | vs基线 %6s | 自适应/最优 %6s (选%s 最优%s) | 抖动 %.2f "
         "| 误差=%.2e | %s | %s\n",
         sv, sa, auto_name, best_name, res.spread, res.rel,
         通过 ? "通过" : "失败",
         未劣化 ? "完整配置未劣于基线" : "完整配置迭代更多(需检查)");

    return res;
}

// =====================================================================
// main
//   默认只跑 N = 16 / 24 / 32。
//   N=75 时 DOF 达 1.69e6: Krylov 基 V 就要 (restart+1)*DOF*8 字节
//   (restart=100 时 1.36 GB), Arnoldi 正交化的访存量又按 restart^2 涨 ——
//   在大网格上跑到分钟/十分钟级是正常的, 不是卡死。
//   想复现大网格: ./smatrix_test_app 50 75 --restart 40
//   只跑一个 alpha:  ./smatrix_test_app 16 --alpha 0.1
//   调 Galerkin 稀疏化: ./smatrix_test_app 32 --amg-drop 0.02 --amg-cap 40
//                       (--amg-drop 0 关闭丢弃, 用于对照)
//   自适应选速度块:     ./smatrix_test_app 32 --adapt
//                       (按 cond(A) 判据决定 ILU(0) 还是 BAMG(3);
//                        判据说用不上 BAMG 时连层次都不建, 省掉 setup)
//   改重复次数:         ./smatrix_test_app 16 --repeat 10
//                       (默认 3。小算例一次求解只有十几毫秒, 单次计时被 OS
//                        抖动淹没 —— 实测同一配置先后报 1.34x 和 0.77x。
//                        加大 repeat 能压掉噪声, 代价是运行时间线性变长。)
//   改 AMG 光滑步数:     ./smatrix_test_app 32 --amg-smooth 1
//   关标量快速路径:     ./smatrix_test_app 32 --amg-scalar 0
//                       (A/B 对照: 迭代数应一致, 差的是 ms/iter)
//                       (默认 2。每个 V-cycle 在每层做 2*n+1 次 matvec:
//                        n=2 是 5 次, n=1 是 3 次 —— 这是压 AMG 单价的唯一大杠杆。
//                        实测 smooth=2 时 BAMG 单价是 ILU(0) 的 1.8 倍;
//                        n=1 会把单价压下来, 但 V-cycle 变弱、迭代数会涨,
//                        两者谁赢要看汇总里的 ms/iter 和迭代数两行。
//                        注意: 阈值 cond(A)>20 是按 smooth=2 的单价标定的,
//                        smooth 一变, 自适应阈值必须重新标定。)
//   扫自适应阈值:       ./smatrix_test_app 32 --amg-smooth 1 --adapt-thr 5
//                       (单价降了就要把阈值也降下来, 否则判据会继续选 ILU(0)。
//                        扫几个值看汇总里的"命中 n/m"和 ms/iter 两行即可定位
//                        新的盈亏点。)
//   alpha 扫描的意义: alpha 小 -> S ~ I (单位阵就已经很好, AMG 加成有限);
//                     alpha 大 -> S ~ Lp/alpha, 此时 Lp^{-1} 项主导, AMG 关键。
// =====================================================================
int main(int argc, char** argv) {
    std::vector<uint32_t> Nlist;
    std::vector<double>   alist;      // alpha 扫描列表
    int restart = 40;
    double   g_drop = 0.02;     // Galerkin 相对丢弃阈值
    uint32_t g_cap  = 40;       // 每行最多保留的块数
    bool     g_adapt = false;   // --adapt: 按 cond(A) 自适应选速度块预处理器
    int      g_repeat = 3;      // --repeat: 每个配置重复求解几次, 取最快的一次
    int      g_smooth = 2;      // --amg-smooth: AMG 前/后光滑步数 (1 或 2)
    bool     g_scalar = true;   // --amg-scalar 0: 关掉标量快速路径, 强制走通用 BCSR

    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if (a == "--restart" && i + 1 < argc) { restart = std::atoi(argv[++i]); continue; }
        if (a == "--amg-drop" && i + 1 < argc) { g_drop = std::atof(argv[++i]); continue; }
        if (a == "--amg-cap"  && i + 1 < argc)
        { g_cap = (uint32_t)std::strtoul(argv[++i], nullptr, 10); continue; }
        if (a == "--adapt") { g_adapt = true; continue; }
        if (a == "--repeat" && i + 1 < argc)
        { g_repeat = std::atoi(argv[++i]); continue; }
        if (a == "--amg-smooth" && i + 1 < argc)
        { g_smooth = std::atoi(argv[++i]); continue; }
        if (a == "--amg-scalar" && i + 1 < argc)
        { g_scalar = (std::atoi(argv[++i]) != 0); continue; }
        if (a == "--adapt-thr" && i + 1 < argc)
        { g_cond_thr = std::atof(argv[++i]); continue; }
        if (a == "--alpha" && i + 1 < argc)
        {
            // 逗号分隔:  --alpha 0.1,1,10
            const char* s = argv[++i];
            alist.clear();
            while (*s)
            {
                char* e = nullptr;
                const double v = std::strtod(s, &e);
                if (e == s) break;
                alist.push_back(v);
                s = e;
                if (*s == ',') ++s;
            }
            continue;
        }
        if (!a.empty() && std::isdigit(static_cast<unsigned char>(a[0])))
            Nlist.push_back((uint32_t)std::strtoul(a.c_str(), nullptr, 10));
    }
    if (Nlist.empty()) Nlist = {16, 24, 32};
    if (alist.empty()) alist = {0.1, 1.0, 10.0};
    if (restart < 1) restart = 1;
    if (g_repeat < 1) g_repeat = 1;
    if (g_smooth < 1) g_smooth = 1;

#ifdef CPPNUM_HAVE_OPENMP
    const char* mkl_layer = std::getenv("MKL_THREADING_LAYER");
    const char* mkl_nt    = std::getenv("MKL_NUM_THREADS");
    logf("OpenMP 线程数 = %d (可见核数 %d)%s, restart = %d | "
         "MKL_THREADING_LAYER=%s MKL_NUM_THREADS=%s\n",
         omp_get_max_threads(), omp_get_num_procs(),
         (omp_get_max_threads() <= 1) ? "  <-- 建议 export OMP_NUM_THREADS=<核数>" : "",
         restart, mkl_layer ? mkl_layer : "<未设置>", mkl_nt ? mkl_nt : "<未设置>");
#else
    logf("OpenMP 未启用, restart = %d\n", restart);
#endif
    logf("计时口径: 每个配置 best-of-%d 取最快 | 比值前带 ~ = 该算例最快的求解 "
         "<30 ms, 计时在噪声量级 | AMG 光滑步数 = %d | 自适应阈值 cond(A) > %g\n",
         g_repeat, g_smooth, g_cond_thr);

    std::vector<SchurResult> all;
    for (uint32_t N : Nlist)
    {
        for (size_t ai = 0; ai < alist.size(); ++ai)
        {
            const double alpha = alist[ai];
            logf("=== 3D Stokes + Schur 补 + Block AMG (N = %u, alpha = %g) ===\n",
                 N, alpha);
            const auto t0 = Clock::now();
            // Lp 与 alpha 无关, 层次诊断只在第一个 alpha 打印, 免得刷屏
            const SchurResult r = test_3d_schur_with_amg(N, restart, alpha, ai == 0,
                                                         g_drop, g_cap, g_adapt,
                                                         g_repeat, g_smooth, g_scalar);
            const auto t1 = Clock::now();
            logf("    组装+预处理 %.2f s | 求解墙钟(含 best-of-%d) %.2f s | 合计 %.2f s\n\n",
                 r.t_setup, g_repeat, r.t_solve, elapsed(t0, t1));
            all.push_back(r);
        }
    }

    // ---- 汇总: 迭代数 vs 网格 ----
    //   这才是判断 Schur 预处理器好坏的真正标准 —— 网格无关性。
    //   好的预处理器: AMG 那一行随 N 基本持平; S_pre=I 那一行随 N 增长。
    logf("===== 汇总: 迭代数 vs 网格 (看 AMG 是否网格无关) =====\n");
    for (double a : alist)
    {
        std::string s1, s2, s3, s4;
        for (const SchurResult& r : all)
        {
            if (r.alpha != a) continue;
            char b1[48], b2[48], b3[64], b4[64];
            std::snprintf(b1, sizeof b1, "N=%-3u:%4d步 ", r.N, r.it_ident);
            std::snprintf(b2, sizeof b2, "N=%-3u:%4d步 ", r.N, r.it_amg);
            std::snprintf(b3, sizeof b3, "N=%-3u:%4d步%s ", r.N, r.it_bamg,
                          r.bamg_built ? "" : "(ILU)");
            std::snprintf(b4, sizeof b4, "N=%-3u:%4d步/%s ", r.N, r.it_auto,
                          r.use_bamg ? "BAMG" : "ILU0");
            s1 += b1; s2 += b2; s3 += b3; s4 += b4;
        }
        logf("  alpha=%-5g  A=ILU(0)  S=I    [ %s]\n", a, s1.c_str());
        logf("              A=ILU(0)  S=AMG  [ %s]\n", s2.c_str());
        logf("              A=BAMG(3) S=AMG  [ %s]\n", s3.c_str());
        logf("              A=AUTO    S=AMG  [ %s]\n", s4.c_str());
    }

    // ---- 汇总: 单次迭代耗时 ----
    //   迭代数会被 GMRES 的收敛路径干扰, 总墙钟会被机器负载干扰; 而
    //   ms/iter = 墙钟 / 迭代数 是唯一又可比又稳定的量 —— 它直接等于
    //   「一次预处理器 apply + 一次 spmv」的成本, 也就是算法选型真正在
    //   比较的东西 (净收益 = 迭代数下降倍数 / 单价倍数)。
    //   看 BAMG 值不值, 就看它这一行是不是高得离谱。
    logf("\n===== 汇总: 单次迭代耗时 ms/iter (best-of-%d) =====\n", g_repeat);
    for (double a : alist)
    {
        std::string s1, s2, s3, s4;
        for (const SchurResult& r : all)
        {
            if (r.alpha != a) continue;
            char b1[48], b2[48], b3[48], b4[48];
            std::snprintf(b1, sizeof b1, "N=%-3u:%7.2fms ", r.N, r.ms_i_I);
            std::snprintf(b2, sizeof b2, "N=%-3u:%7.2fms ", r.N, r.ms_i_amg);
            std::snprintf(b3, sizeof b3, "N=%-3u:%7.2fms ", r.N, r.ms_i_bamg);
            std::snprintf(b4, sizeof b4, "N=%-3u:%7.2fms ", r.N, r.ms_i_auto);
            s1 += b1; s2 += b2; s3 += b3; s4 += b4;
        }
        logf("  alpha=%-5g  A=ILU(0)  S=I    [ %s]\n", a, s1.c_str());
        logf("              A=ILU(0)  S=AMG  [ %s]\n", s2.c_str());
        logf("              A=BAMG(3) S=AMG  [ %s]\n", s3.c_str());
        logf("              A=AUTO    S=AMG  [ %s]\n", s4.c_str());
    }

    // ---- 汇总: 自适应判据的命中率 ----
    //   1.00 = 判据恰好选中了事后最优的那个配置, 判据零损失。
    //   这是评价「自适应」本身的唯一正确口径: 拿基线当分母会把 Schur 补的
    //   收益一起算进来, 得到 4.66x 这种虚高且无法解释的数。
    {
        int hits = 0, total = 0, nz = 0;
        double sum = 0.0, worst = 1e300;
        for (const SchurResult& r : all)
        {
            if (r.t_auto <= 1e-9 || r.t_best <= 1e-9) continue;
            const double ratio = r.t_best / r.t_auto;
            ++total;
            sum += ratio;
            if (ratio < worst) worst = ratio;
            if (ratio >= 0.95) ++hits;
            if (r.noisy) ++nz;
        }
        logf("\n===== 汇总: 自适应判据 vs 事后最优 (1.00 = 判据恰好选中最快配置) =====\n");
        for (double a : alist)
        {
            std::string s;
            for (const SchurResult& r : all)
            {
                if (r.alpha != a) continue;
                char b[80];
                std::snprintf(b, sizeof b, "N=%-3u:%.2f(%s/最优%s)%s ",
                              r.N, (r.t_auto > 0.0 ? r.t_best / r.t_auto : 0.0),
                              r.use_bamg ? "BAMG" : "ILU0",
                              (r.best_cfg == 3) ? "BAMG"
                              : (r.best_cfg == 2) ? "ILU0+AMG" : "ILU0+I",
                              r.noisy ? "~" : "");
                s += b;
            }
            logf("  alpha=%-5g [ %s]\n", a, s.c_str());
        }
        logf("  命中 %d/%d | 平均 %.3f | 最差 %.3f%s\n", hits, total,
             (total ? sum / (double)total : 0.0), (total ? worst : 0.0),
             nz ? "  (带 ~ 的算例最快的求解 <30 ms, 计时在噪声量级)" : "");
    }
    return 0;
}
