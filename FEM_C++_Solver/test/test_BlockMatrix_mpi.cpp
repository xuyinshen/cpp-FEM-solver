#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <limits>
#include <memory>
#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef CPPNUM_HAVE_MPI
#include <mpi.h>
#endif
#include "math/array.hpp"
#include "math/blockMatrix.hpp"

using namespace sparse;
using namespace block;

constexpr int BS = 3;                      // 块边长 (模拟 2D NS 同阶元)
using Blk = BlockMatrix<double, BS>;

// ---------- 确定性伪随机 (各进程/各次运行一致) ----------
static double drand(size_t i)
{ double s = std::sin((double)i * 0.371 + 1.7) * 43758.5453; return s - std::floor(s); }

// =====================================================================
// 归约小工具
//   - 全部走 OpenMP reduction: 每线程私有累加器, 无写冲突、无 false sharing
//   - 内层只触碰连续 double 数组, -O3 -march=native 下可自动向量化
// =====================================================================
namespace {

inline double dot_par(const std::vector<double>& a, const std::vector<double>& b)
{
    const std::ptrdiff_t n = (std::ptrdiff_t)a.size();
    double s = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(+:s) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline double max_abs_diff_par(const std::vector<double>& a, const std::vector<double>& b)
{
    const std::ptrdiff_t n = (std::ptrdiff_t)std::min(a.size(), b.size());
    double m = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(max:m) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i)
    {
        const double d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// 由逐行 nnz 计数生成 CSR 风格行偏移(前缀和)。
// 串行 O(nrows) 但只有整数加法, 成本远低于随后的填充; 换来的是填充阶段
// 行间完全无竞争(无需原子操作), 且每个 vector 只分配一次(零 realloc)。
inline std::vector<uint32_t> row_offsets(const std::vector<uint32_t>& cnt)
{
    std::vector<uint32_t> off(cnt.size() + 1, 0);
    for (size_t i = 0; i < cnt.size(); ++i) off[i + 1] = off[i] + cnt[i];
    return off;
}

} // namespace

// =====================================================================
// 构造 node-major 块结构标量测试矩阵
//   2D N x N 节点, 每节点 BS 分量; 对角块 = (4+BS)I + 0.1 耦合, 非对角块 = -I
//   对称 + 块对角占优 => SPD, 保证 Krylov 收敛
// =====================================================================
static SMatrix make_test_scalar(uint32_t N)
{
    const uint32_t nodes = N * N;
    const uint32_t n = nodes * BS;

    // ---- 计数: 每行 nnz 解析可得 (块对角 BS 个 + 最多 4 个邻居) ----
    std::vector<uint32_t> cnt(n, 0);
    for (uint32_t idx = 0; idx < nodes; ++idx)
    {
        const uint32_t i = idx / N, j = idx % N;
        const uint32_t nb = (uint32_t)BS + (j > 0 ? 1u : 0u) + (j < N - 1 ? 1u : 0u)
                                         + (i > 0 ? 1u : 0u) + (i < N - 1 ? 1u : 0u);
        for (int r = 0; r < BS; ++r) cnt[idx * BS + (uint32_t)r] = nb;
    }
    std::vector<uint32_t> off = row_offsets(cnt);
    // 【内存】计数数组用完立即归还, 再去分配三份大 COO, 压低峰值
    { std::vector<uint32_t> e; cnt.swap(e); }

    const size_t nnz = (size_t)off[n];
    std::vector<double>   vals(nnz);
    std::vector<uint32_t> rl(nnz), cl(nnz);

    // ---- 并行填充: 每行只写自己的 [off[row], off[row+1]) 区间 ----
    const std::ptrdiff_t pn = (std::ptrdiff_t)n;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t rr = 0; rr < pn; ++rr)
    {
        const uint32_t row = (uint32_t)rr;
        const uint32_t idx = row / BS, r = row % BS;
        const uint32_t i = idx / N, j = idx % N;
        uint32_t w = off[row];
        for (int c = 0; c < BS; ++c)            // 对角块
        {
            rl[w] = row; cl[w] = idx * BS + (uint32_t)c;
            vals[w] = (r == (uint32_t)c) ? (4.0 + BS) : 0.1;
            ++w;
        }
        if (j > 0)     { rl[w] = row; cl[w] = (idx - 1) * BS + r; vals[w] = -1.0; ++w; }
        if (j < N - 1) { rl[w] = row; cl[w] = (idx + 1) * BS + r; vals[w] = -1.0; ++w; }
        if (i > 0)     { rl[w] = row; cl[w] = (idx - N) * BS + r; vals[w] = -1.0; ++w; }
        if (i < N - 1) { rl[w] = row; cl[w] = (idx + N) * BS + r; vals[w] = -1.0; ++w; }
    }
    { std::vector<uint32_t> e; off.swap(e); }   // 归还后再进 COO->CSR

    return SMatrix(n, n, vals, rl, cl);
}

// ---------- flatten: BlockMatrix -> 标量 CSR (node-major), 用于往返验证 ----------
static SMatrix flatten_blk(const Blk& B)
{
    const uint32_t nbr = B.block_rows(), nbc = B.block_cols();
    const uint32_t n = nbr * BS, m = nbc * BS;

    const auto& rp = B.block_row_ptr();
    const auto& ci = B.block_col_idx();
    const auto& vv = B.values();

    std::vector<uint32_t> srp(n + 1, 0);
    for (uint32_t bi = 0; bi < nbr; ++bi)
    {
        const uint32_t cnt = (rp[bi + 1] - rp[bi]) * BS;
        for (int r = 0; r < BS; ++r) srp[bi * BS + r + 1] = cnt;
    }
    for (uint32_t i = 0; i < n; ++i) srp[i + 1] += srp[i];

    std::vector<uint32_t> sci(srp[n]);
    std::vector<double>   svv(srp[n]);

#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nbr; ++bi)
        for (int r = 0; r < BS; ++r)
        {
            uint32_t w = srp[(uint32_t)bi * BS + r];
            for (uint32_t p = rp[bi]; p < rp[bi + 1]; ++p)
                for (int c = 0; c < BS; ++c)
                {
                    sci[w] = ci[p] * BS + c;
                    // (size_t) 防止 p*BS*BS 在大 NNZ 下 32 位溢出
                    svv[w] = vv[(size_t)p * BS * BS + (size_t)r * BS + c];
                    ++w;
                }
        }
    return SMatrix(n, m, srp, sci, svv);        // CSR 直构
}

// ---------- 小高斯求逆 (bs x bs) ----------
// 【内存】原先是 vector<vector<double>>: bs 次独立堆分配且各行不连续。
//         改为调用方传入的扁平 bs x 2bs 增广矩阵, 调用方可把它做成线程私有
//         复用缓冲, 于是 nb 次求逆总共只分配 T 次而不是 nb 次。
static void test_invert_block(const double* M, double* Minv, double* aug, int bs)
{
    const int ld = 2 * bs;                      // 增广矩阵行 stride
    for (int i = 0; i < bs; ++i)
    {
        double* Ai = &aug[(size_t)i * ld];
        for (int j = 0; j < bs; ++j) Ai[j] = M[i * bs + j];
        for (int j = 0; j < bs; ++j) Ai[bs + j] = 0.0;
        Ai[bs + i] = 1.0;
    }
    for (int k = 0; k < bs; ++k)
    {
        int piv = k;
        double amax = std::fabs(aug[(size_t)k * ld + k]);
        for (int i = k + 1; i < bs; ++i)
        {
            const double v = std::fabs(aug[(size_t)i * ld + k]);
            if (v > amax) { amax = v; piv = i; }
        }
        if (amax == 0.0)                         // 奇异: 退化为零逆, 交由上层判定
        { for (int i = 0; i < bs * bs; ++i) Minv[i] = 0.0; return; }
        if (piv != k)
            for (int j = k; j < ld; ++j) std::swap(aug[(size_t)k * ld + j], aug[(size_t)piv * ld + j]);

        const double d = aug[(size_t)k * ld + k];
        const double* Ak = &aug[(size_t)k * ld];
        for (int i = k + 1; i < bs; ++i)
        {
            double* Ai = &aug[(size_t)i * ld];
            const double f = Ai[k] / d;
            if (f == 0.0) continue;
            for (int j = k; j < ld; ++j) Ai[j] -= f * Ak[j];
        }
    }
    for (int i = bs - 1; i >= 0; --i)
    {
        double* Ai = &aug[(size_t)i * ld];
        for (int j = i + 1; j < bs; ++j)
        {
            const double aij = Ai[j];
            if (aij == 0.0) continue;
            const double* Aj = &aug[(size_t)j * ld];
            for (int c = 0; c < bs; ++c) Ai[bs + c] -= aij * Aj[bs + c];
        }
        const double d = Ai[i];
        for (int c = 0; c < bs; ++c) Ai[bs + c] /= d;
    }
    for (int i = 0; i < bs; ++i)
    {
        const double* Ai = &aug[(size_t)i * ld];
        for (int j = 0; j < bs; ++j) Minv[i * bs + j] = Ai[bs + j];
    }
}

// ---------- 测试用 Block-Jacobi ----------
static std::function<void(const std::vector<double>&, std::vector<double>&)>
make_test_block_jacobi(const Blk& B)
{
    const uint32_t nb = B.block_rows();
    // 【内存】shared_ptr 持有逆块: 闭包按值捕获 shared_ptr, 不复制 nb*BS^2 个 double
    auto inv = std::make_shared<std::vector<double>>((size_t)nb * BS * BS);
    double* I = inv->data();

    const std::ptrdiff_t pnb = (std::ptrdiff_t)nb;
    // 【并行】nb 个对角块求逆彼此独立; 百万级块行时串行求逆是这一段的真实瓶颈
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel
#endif
    {
        // 线程私有增广矩阵: nb 次求逆总共只分配 T 次
        std::vector<double> aug((size_t)BS * (2 * BS));
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t bi = 0; bi < pnb; ++bi)
            test_invert_block(B.diag_block((uint32_t)bi), &I[(size_t)bi * BS * BS], aug.data(), BS);
    }

    return [nb, inv](const std::vector<double>& rhs, std::vector<double>& z)
    {
        z.resize(rhs.size());
        const double* I2 = inv->data();
        const std::ptrdiff_t pnb2 = (std::ptrdiff_t)nb;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t bi = 0; bi < pnb2; ++bi)
        {
            const double* Mi = &I2[(size_t)bi * BS * BS];
            const double* ri = &rhs[(size_t)bi * BS];
            double*       zi = &z[(size_t)bi * BS];
            // 块内 GEMV: BS 是编译期常量 -> 完全展开; simd 显式向量化
            for (int r = 0; r < BS; ++r)
            {
                double a = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
                #pragma omp simd reduction(+:a)
#endif
                for (int c = 0; c < BS; ++c) a += Mi[r * BS + c] * ri[c];
                zi[r] = a;
            }
        }
    };
}

// ---------- 测试用 mini-CG (直接消费 BlockMatrix, 验证求解器衔接) ----------
template <typename Mat, typename Pre>
static std::pair<std::vector<double>, int>
mini_cg(const Mat& M, const std::vector<double>& b, const Pre& solveM, double tol, int maxit)
{
    const size_t n = b.size();
    const std::ptrdiff_t pn = (std::ptrdiff_t)n;

    // 【内存】工作向量一次性分配到 n, 循环内只有赋值没有 realloc
    std::vector<double> x(n, 0.0), r(b), z(n), p, Ap(n);
    solveM(r, z);
    p = z;

    double rz = dot_par(r, z);
    const double nb = std::sqrt(dot_par(b, b));
    int it = 0;
    for (; it < maxit; ++it)
    {
        M.matvec(p, Ap);
        const double pAp = dot_par(p, Ap);
        const double alpha = rz / pAp;

        // axpy 融合: x += alpha*p 与 r -= alpha*Ap 一次遍历完成
        // (两次线性扫描合并成一次, 访存量从 5n 降到 3n)
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for simd
#endif
        for (std::ptrdiff_t i = 0; i < pn; ++i)
        { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }

        if (std::sqrt(dot_par(r, r)) / nb < tol) break;

        solveM(r, z);
        const double rz2 = dot_par(r, z);
        const double beta = rz2 / rz;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for simd
#endif
        for (std::ptrdiff_t i = 0; i < pn; ++i) p[i] = z[i] + beta * p[i];
        rz = rz2;
    }
    return {x, it};
}

// ================= T1: 布局 + condense + matvec =================
static void test_layout_condense(uint32_t N)
{
    SMatrix 标量参照 = make_test_scalar(N);
    Blk 块矩阵 = Blk::condense(标量参照, N * N, N * N);
    SMatrix 展平回标量 = flatten_blk(块矩阵);

    const size_t n = 标量参照.rows();
    std::vector<double> 随机向量(n), 参照结果, 展平结果, 块结果;
    const std::ptrdiff_t pn = (std::ptrdiff_t)n;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i) 随机向量[i] = drand((size_t)i);

    参照结果 = SMatrix::spmv(标量参照, 随机向量);
    展平结果 = SMatrix::spmv(展平回标量, 随机向量);
    块矩阵.matvec(随机向量, 块结果);

    // 并行 max 归约
    double 往返误差 = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(max:往返误差) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i)
    {
        const double d1 = std::fabs(参照结果[i] - 展平结果[i]);
        const double d2 = std::fabs(参照结果[i] - 块结果[i]);
        const double d  = (d1 > d2) ? d1 : d2;
        if (d > 往返误差) 往返误差 = d;
    }

    // 修正: BCSR 块内稠密, 展平 nnz 必然放大, 不再断言 nnz 相等
    const bool 通过 = (往返误差 < 1e-12);
    std::printf("[T1] N=%4u 标量自由度=%8u 块非零数=%9zu 标量nnz=%8zu 展平nnz=%8zu "
                "往返误差=%.3e %s\n",
                N, 标量参照.rows(), 块矩阵.nnz_blocks(),
                (size_t)标量参照.nnz(), (size_t)展平回标量.nnz(),
                往返误差, 通过 ? "通过" : "失败");
}

// ================= T2: transpose 内积恒等式 =================
static void test_transpose(uint32_t N)
{
    SMatrix 标量参照 = make_test_scalar(N);
    Blk 块矩阵 = Blk::condense(标量参照, N * N, N * N);
    Blk 转置矩阵 = 块矩阵.transpose();
    const size_t 自由度 = 块矩阵.rows();
    const std::ptrdiff_t pn = (std::ptrdiff_t)自由度;
    std::vector<double> 左向量(自由度), 右向量(自由度), Av结果, Atu结果;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i)
    { 左向量[i] = drand((size_t)i + 11); 右向量[i] = drand((size_t)i + 29); }

    块矩阵.matvec(右向量, Av结果);
    转置矩阵.matvec(左向量, Atu结果);

    const double 左内积 = dot_par(左向量, Av结果);
    const double 右内积 = dot_par(Atu结果, 右向量);
    const double 内积相对偏差 = std::fabs(左内积 - 右内积) / (std::fabs(左内积) + 1e-30);

    std::printf("[T2] N=%4u <u,Av>=%.10e <A^T u,v>=%.10e 内积相对偏差=%.3e %s\n",
                N, 左内积, 右内积, 内积相对偏差, 内积相对偏差 < 1e-12 ? "通过" : "失败");
}

// ================= T3: diag_block + Block-Jacobi 逐块验证 =================
static void test_diag_jacobi(uint32_t N)
{
    SMatrix 标量参照 = make_test_scalar(N);
    Blk 块矩阵 = Blk::condense(标量参照, N * N, N * N);
    auto 块雅可比求解 = make_test_block_jacobi(块矩阵);
    const uint32_t 块数 = 块矩阵.block_rows();
    std::vector<double> 右端项((size_t)块数 * BS), 解向量;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)右端项.size(); ++i)
        右端项[i] = drand((size_t)i + 41);
    块雅可比求解(右端项, 解向量);

    double 块残差 = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(max:块残差) schedule(static)
#endif
    for (std::ptrdiff_t bb = 0; bb < (std::ptrdiff_t)块数; ++bb)
    {
        const uint32_t bi = (uint32_t)bb;
        const double* 对角块 = 块矩阵.diag_block(bi);
        for (int r = 0; r < BS; ++r)
        {
            double 块乘积累加 = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
            #pragma omp simd reduction(+:块乘积累加)
#endif
            for (int c = 0; c < BS; ++c)
                块乘积累加 += 对角块[r * BS + c] * 解向量[(size_t)bi * BS + c];
            const double d = std::fabs(块乘积累加 - 右端项[(size_t)bi * BS + r]);
            if (d > 块残差) 块残差 = d;
        }
    }
    std::printf("[T3] N=%4u 块数=%7u 块残差=%.3e %s\n",
                N, 块数, 块残差, 块残差 < 1e-10 ? "通过" : "失败");
}

// ================= T4: 端到端 mini-CG + Block-Jacobi =================
static void test_solve(uint32_t N)
{
    SMatrix 标量参照 = make_test_scalar(N);
    Blk 块矩阵 = Blk::condense(标量参照, N * N, N * N);
    auto 块雅可比求解 = make_test_block_jacobi(块矩阵);
    std::vector<double> 右端项(块矩阵.rows());
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)右端项.size(); ++i)
        右端项[i] = 1.0 + 0.5 * drand((size_t)i + 7);

    auto [解向量, 迭代数] = mini_cg(块矩阵, 右端项, 块雅可比求解, 1e-8, 3000);

    std::vector<double> 矩阵乘解; 块矩阵.matvec(解向量, 矩阵乘解);
    const std::ptrdiff_t pn = (std::ptrdiff_t)右端项.size();
    double 残差平方 = 0.0, 右端平方 = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(+:残差平方) reduction(+:右端平方) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i)
    {
        const double d = 右端项[i] - 矩阵乘解[i];
        残差平方 += d * d;
        右端平方 += 右端项[i] * 右端项[i];
    }
    const double 真实相对残差 = std::sqrt(残差平方 / 右端平方);

    std::printf("[T4] N=%4u CG迭代数=%5d 真实相对残差=%.3e %s\n",
                N, 迭代数 + 1, 真实相对残差, 真实相对残差 < 1e-6 ? "通过" : "失败");
}

// ================= T5: MPI distributed_matvec vs 串行 =================
#ifdef CPPNUM_HAVE_MPI
// ---------- 从全局块矩阵切出本进程块行, 并建立 halo 通信表 ----------
static void distribute_and_halo(const Blk& 全局矩阵, int 进程号, int 进程数,
                                uint32_t& 起始块行, Blk& 本地矩阵)
{
    const uint32_t 全局块行数 = 全局矩阵.block_rows();
    起始块行 = (uint32_t)((uint64_t)全局块行数 * 进程号 / 进程数);
    const uint32_t 终止块行 = (uint32_t)((uint64_t)全局块行数 * (进程号 + 1) / 进程数);

    const auto& 全局行指针 = 全局矩阵.block_row_ptr();
    const auto& 全局列索引 = 全局矩阵.block_col_idx();
    const auto& 全局块值   = 全局矩阵.values();

    // 1. 切出本进程块行 (列索引保持全局编号)
    const uint32_t 本地行数 = 终止块行 - 起始块行;
    std::vector<uint32_t> 本地行指针(本地行数 + 1);
    for (uint32_t i = 0; i <= 本地行数; ++i)
        本地行指针[i] = 全局行指针[起始块行 + i] - 全局行指针[起始块行];
    std::vector<uint32_t> 本地列索引(全局列索引.begin() + 全局行指针[起始块行],
                                     全局列索引.begin() + 全局行指针[终止块行]);
    std::vector<double>  本地块值(全局块值.begin() + (size_t)全局行指针[起始块行] * BS * BS,
                                  全局块值.begin() + (size_t)全局行指针[终止块行] * BS * BS);

    auto 归属进程 = [&](uint32_t 全局列) -> int
    { return (int)((uint64_t)全局列 * 进程数 / 全局块行数); };

    // 2. 先收集 ghost 列 (引用了但不属于本进程的块列), 去重排序
    //    —— 必须在下面把 本地列索引 move 进 Blk 之前做完
    std::vector<uint32_t> 幽灵列;
    幽灵列.reserve(本地列索引.size());      // 【内存】上界预留, 避免反复 realloc
    for (size_t k = 0; k < 本地列索引.size(); ++k)
        if (归属进程(本地列索引[k]) != 进程号) 幽灵列.push_back(本地列索引[k]);
    std::sort(幽灵列.begin(), 幽灵列.end());
    幽灵列.erase(std::unique(幽灵列.begin(), 幽灵列.end()), 幽灵列.end());
    // 去重后多余容量归还: 这份数组会常驻到 set_halo
    幽灵列.shrink_to_fit();

    // 3. 组装本地块矩阵 —— 三份数组 move 进去, 不产生 nnz 级别的拷贝
    本地矩阵 = Blk(本地行数, 全局矩阵.block_cols(),
                   std::move(本地行指针), std::move(本地列索引), std::move(本地块值));

    // 3. 按归属进程分组 -> recv 表
    std::vector<int> 接收进程, 接收计数;
    for (size_t k = 0; k < 幽灵列.size(); )
    {
        const int p = 归属进程(幽灵列[k]);
        size_t k2 = k;
        while (k2 < 幽灵列.size() && 归属进程(幽灵列[k2]) == p) ++k2;
        接收进程.push_back(p); 接收计数.push_back((int)(k2 - k)); k = k2;
    }

    // 4. Alltoallv 交换 ghost 请求: 告知 owner "我要你的哪些行", 同时获知 "谁要我的哪些行"
    std::vector<int> 发送计数(进程数, 0), 接收请求计数(进程数, 0);
    for (size_t k = 0; k < 接收进程.size(); ++k) 发送计数[接收进程[k]] = 接收计数[k];
    MPI_Alltoall(发送计数.data(), 1, MPI_INT,
                 接收请求计数.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> 发送位移(进程数, 0), 接收位移(进程数, 0);
    for (int p = 1; p < 进程数; ++p)
    { 发送位移[p] = 发送位移[p - 1] + 发送计数[p - 1];
      接收位移[p] = 接收位移[p - 1] + 接收请求计数[p - 1]; }

    std::vector<uint32_t> 接收请求(接收位移[进程数 - 1] + 接收请求计数[进程数 - 1]);
    MPI_Alltoallv(幽灵列.data(), 发送计数.data(), 发送位移.data(), MPI_UINT32_T,
                  接收请求.data(), 接收请求计数.data(), 接收位移.data(), MPI_UINT32_T,
                  MPI_COMM_WORLD);

    // 5. 组装 send 表: 别人要我的全局行 -> 本地索引 (全局行 - 起始块行)
    std::vector<int> 发送进程, 发送计数表; std::vector<uint32_t> 发送本地索引;
    for (int p = 0; p < 进程数; ++p)
        if (接收请求计数[p] > 0)
        {
            发送进程.push_back(p); 发送计数表.push_back(接收请求计数[p]);
            for (int t = 0; t < 接收请求计数[p]; ++t)
                发送本地索引.push_back(接收请求[接收位移[p] + t] - 起始块行);
        }

    // 6. 注入 halo 表 (最后一个参数 = 本进程 owned 行的全局起点, 关键!)
    本地矩阵.set_halo(接收进程, 接收计数, 幽灵列,
                      发送进程, 发送计数表, 发送本地索引, 起始块行);
}
#endif

#ifdef CPPNUM_HAVE_MPI
static void test_mpi(uint32_t N, int 进程号, int 进程数)
{
    SMatrix 标量参照 = make_test_scalar(N);
    Blk 全局矩阵 = Blk::condense(标量参照, N * N, N * N);
    const size_t 总自由度 = (size_t)全局矩阵.block_rows() * BS;
    std::vector<double> 全局向量(总自由度);
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)总自由度; ++i)
        全局向量[i] = drand((size_t)i + 5);

    std::vector<double> 串行结果; 全局矩阵.matvec(全局向量, 串行结果);   // 黄金标准

    if (进程号 == 0) std::cout << "初始化完毕" << std::endl;

    uint32_t 起始块行; Blk 本地矩阵;
    distribute_and_halo(全局矩阵, 进程号, 进程数, 起始块行, 本地矩阵);
    const uint32_t 终止块行 = (uint32_t)((uint64_t)全局矩阵.block_rows() * (进程号 + 1) / 进程数);
    std::vector<double> 本地向量(全局向量.begin() + (std::ptrdiff_t)起始块行 * BS,
                                 全局向量.begin() + (std::ptrdiff_t)终止块行 * BS), 本地结果;
    本地矩阵.distributed_matvec(本地向量, 本地结果);

    if (进程号 == 0) std::cout << "向量计算完毕" << std::endl;

    // 【内存】只有 rank 0 需要拼接全量结果做比对,
    //         因此用 Gatherv(root=0) 取代 Allgatherv: 省掉 P-1 份 n*8 字节的常驻副本
    int 本地长度 = (int)本地结果.size();
    std::vector<int> 各进程长度(进程数), 位移(进程数, 0);
    MPI_Allgather(&本地长度, 1, MPI_INT, 各进程长度.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int p = 1; p < 进程数; ++p) 位移[p] = 位移[p - 1] + 各进程长度[p - 1];

    std::vector<double> 拼接结果;
    if (进程号 == 0) 拼接结果.resize(位移[进程数 - 1] + 各进程长度[进程数 - 1]);
    MPI_Gatherv(本地结果.data(), 本地长度, MPI_DOUBLE,
                进程号 == 0 ? 拼接结果.data()  : nullptr,
                进程号 == 0 ? 各进程长度.data() : nullptr,
                进程号 == 0 ? 位移.data()       : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (进程号 == 0)
    {
        std::cout << "MPI收集变量完毕" << std::endl;
        const double 全局最大偏差 = max_abs_diff_par(拼接结果, 串行结果);
        std::printf("[T5] N=%4u 进程数=%2d 分布-串行最大偏差=%.3e %s\n",
                    N, 进程数, 全局最大偏差, 全局最大偏差 < 1e-12 ? "通过" : "失败");
    }
}
#endif

// ================= T6: Block-ILU(0) 分解健壮性 =================
// 纯局部计算: 只在 rank 0 上跑, 其余进程直接返回 (见 main 中的 rank 判定)
static void test_block_ilu_pattern(uint32_t N)
{
    SMatrix 标量参照 = make_test_scalar(N);
    Blk 块矩阵 = Blk::condense(标量参照, N * N, N * N);
    auto 求解器 = make_block_ilu0_preconditioner(块矩阵);   // 触发分解

    // 间接验证: 对若干随机右端项施加 M^{-1}, 检查
    //   (a) 无 breakdown (结果全为有限值)   (b) 结果非平凡 (范数 > 0)
    const std::ptrdiff_t pn = (std::ptrdiff_t)块矩阵.rows();
    bool 通过 = true;
    double 最小范数 = std::numeric_limits<double>::max();

    for (uint32_t t = 0; t < 4 && 通过; ++t)
    {
        std::vector<double> r((size_t)pn), z;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < pn; ++i) r[i] = drand((size_t)i + 1000u * t);
        求解器(r, z);

        double 平方和 = 0.0;
        long long 非有限 = 0;
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for reduction(+:平方和) reduction(+:非有限) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < pn; ++i)
        { 平方和 += z[i] * z[i]; if (!std::isfinite(z[i])) ++非有限; }

        if (非有限 != 0 || !(平方和 > 0.0) || !std::isfinite(平方和)) 通过 = false;
        else 最小范数 = std::min(最小范数, std::sqrt(平方和));
    }
    std::printf("[T6] N=%4u 分解完成无breakdown 最小||M^{-1}r||=%.3e %s\n",
                N, 通过 ? 最小范数 : 0.0, 通过 ? "通过" : "失败");
}

// ================= T7: 块ILU vs 块Jacobi 预处理收敛对比 =================
static void test_block_ilu_solve(uint32_t N)
{
    SMatrix 标量参照 = make_test_scalar(N);
    Blk 块矩阵 = Blk::condense(标量参照, N * N, N * N);
    auto 块雅可比 = make_test_block_jacobi(块矩阵);
    auto 块ILU   = make_block_ilu0_preconditioner(块矩阵);

    std::vector<double> 右端项(块矩阵.rows());
    const std::ptrdiff_t pn = (std::ptrdiff_t)右端项.size();
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i) 右端项[i] = 1.0 + 0.5 * drand((size_t)i + 7);

    auto [解J, 步J] = mini_cg(块矩阵, 右端项, 块雅可比, 1e-8, 3000);
    auto [解I, 步I] = mini_cg(块矩阵, 右端项, 块ILU,   1e-8, 3000);
    (void)解J;                                  // 只比较迭代次数, 解本身不参与判定

    std::vector<double> Ax; 块矩阵.matvec(解I, Ax);
    double 残差平方 = 0.0, 右端平方 = 0.0;
#ifdef CPPNUM_HAVE_OPENMP
    #pragma omp parallel for reduction(+:残差平方) reduction(+:右端平方) schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < pn; ++i)
    {
        const double d = 右端项[i] - Ax[i];
        残差平方 += d * d; 右端平方 += 右端项[i] * 右端项[i];
    }
    const double 真实相对残差 = std::sqrt(残差平方 / 右端平方);

    const bool 通过 = (真实相对残差 < 1e-6) && (步I <= 步J);
    std::printf("[T7] N=%4u CG+块Jacobi=%4d步 CG+块ILU=%4d步 真实相对残差=%.3e %s\n",
                N, 步J + 1, 步I + 1, 真实相对残差, 通过 ? "通过" : "失败");
}

// ================= main =================
int main(int argc, char** argv)
{
    int rank = 0, nprocs = 1;
#ifdef CPPNUM_HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#else
    (void)argc; (void)argv; (void)nprocs;
#endif

    for (uint32_t N : {128u, 512u, 1024u})          // 阶数可调: 块数 = N^2, 标量 = 3N^2
    {
        // T1~T4 是纯局部的串行等价性测试: 只在 rank 0 上跑一次。
        // 让 P 个进程各造一遍 3N^2 规模的矩阵并把同样的活干 P 遍是纯浪费。
        if (rank == 0)
        {
            test_layout_condense(N);
            test_transpose(N);
            test_diag_jacobi(N);
            test_solve(N);
        }

#ifdef CPPNUM_HAVE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        test_mpi(N, rank, nprocs);                  // 唯一真正走 MPI 的用例
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // T6/T7 同样是纯局部重计算, 也只在 rank 0 上跑
        if (rank == 0)
        {
            test_block_ilu_pattern(N);
            test_block_ilu_solve(N);
        }

#ifdef CPPNUM_HAVE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

#ifdef CPPNUM_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
