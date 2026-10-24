#pragma once
// ============================================================================
// M1: Taylor-Hood (P2/P1) 稳态 Stokes 的装配与求解
// ----------------------------------------------------------------------------
//   方程 (无量纲, 常粘度 nu):
//       -nu Δu + ∇p = f
//        ∇·u        = 0
//   Galerkin 弱形式:
//       nu ∫∇u:∇v - ∫ p ∇·v = ∫ f·v        (∀v)
//       -∫ q ∇·u                 = 0        (∀q)      <- 连续性方程取了负号
//
//   【为什么连续性方程取负号】
//     记 D[k,(j,β)] = ∫ ψ_k ∂φ_j/∂x_β (散度), 则动量方程里压力项是 -Dᵀ p。
//     若把散度方程写成 +D u = 0, 系统就是 [A, -Dᵀ; D, 0] —— 非对称。
//     把散度方程取负、令 B = -D, 系统变成 [A, Bᵀ; B, 0] —— **对称不定**,
//     可以上 MINRES / 对称友好的 GMRES, 也让下面 Bᵀ 就是 B 的转置
//     (不用在 spmv 里额外记一个符号, 这种符号错在鞍点问题里极难查)。
//
//   ---------------------------------------------------------------------
//   【本文件最关键的一个简化: A = K ⊗ I₂, 所以速度块不需要 BlockMatrix】
//     Stokes 的速度算子是 nu ∫∇u:∇v, 它不含 ∂u_x/∂y 与 ∂u_y/∂x 的交叉项,
//     于是按 node-major (idx(i,α) = 2i+α) 排列时
//         A[(i,α),(j,β)] = δ_αβ · K_ij ,    K_ij = nu ∫∇φ_j·∇φ_i
//     即 A 精确等于"标量 P2 刚度矩阵 ⊗ 2x2 单位阵"。
//     推论:  A⁻¹ = K⁻¹ ⊗ I₂ —— **给速度块做预条件, 只要对标量 K 做一次 AMG,
//     再分别对两个分量切片各 apply 一次即可**, 不需要 BlockMatrix / BlockAMG。
//     好处: 直接复用 M0 里已经过收敛阶验证、并且刚修完 Dirichlet 强连接闸门的
//     标量 AMG; 而 BlockAMG (光滑聚合) 还没在 Stokes 上验证过。
//     【M2 注意】Navier-Stokes 的 Newton 线性化后 A 会带上对流项与 (∇u)ᵀ 项,
//     那时 A ≠ K ⊗ I₂, 必须换成 BlockMatrix<double,2> + BlockAMG。这里把
//     A_precond 做成 std::function 就是为了到时候只换这一个槽。
//
//   ---------------------------------------------------------------------
//   【Schur 补: 稳态 Stokes 下 S ≈ (1/nu) Mp】
//     连续层面 S = B A⁻¹ Bᵀ = (1/nu)·div(-Δ)⁻¹grad, 而 div(-Δ)⁻¹grad 在
//     周期/无界域上是恒等算子; 离散上"恒等"就是压力质量矩阵 Mp。所以
//         S⁻¹ ≈ nu · Mp⁻¹
//     这正是 Cahouet-Chabard 公式 S⁻¹ ≈ nu·Mp⁻¹ + alpha·Lp⁻¹ 在 **稳态
//     (alpha = 0)** 下的退化形式。Mp 的条件数 O(1), 用 lumped (行和) 对角逆
//     就足够谱等价 —— 这是 O(np) 的 apply, 比再建一个 AMG 划算得多。
//     【M2/瞬态】alpha = 1/dt 那一项需要 Lp (压力 Neumann Laplacian) 的逆,
//     到时候在这里加; 稳态先用不到, 不为它付 setup 成本。
//
//   【压力 null space】Stokes 的压力只差一个常数, 必须钉住。做法是固定
//     **一个**压力自由度 (其余保持零对角, 鞍点矩阵本来就没有压力对角块)。
//     注意 SMatrix 的 COO 构造会丢弃 0 值元素, 所以那个被钉住的位置要先塞
//     一个非零占位值, 随后 apply_dirichlet 会把该行整行清零、对角设 1 ——
//     与"原值是 0 再被设成 1"逐位等价。
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "fem/assembler.hpp"
#include "fem/dofmap.hpp"
#include "fem/element.hpp"
#include "fem/mesh.hpp"
#include "fem/quadrature.hpp"
#include "math/array.hpp"
#include "math/blockMatrix.hpp"

namespace fem
{

// ---------------------------------------------------------------------------
// 矩形稀疏模式: 行 dofmap x 列 dofmap (梯度/散度这类耦合算子用)
//   与 build_pattern 同理, 只是两个 dofmap 的每单元自由度数不同。
// ---------------------------------------------------------------------------
inline CsrPattern build_pattern_rect(const Mesh& m, const DofMap& dm_r, const DofMap& dm_c)
{
    const std::size_t nt  = m.n_tris();
    const std::size_t ndr = static_cast<std::size_t>(dm_r.ndof_per_elem());
    const std::size_t ndc = static_cast<std::size_t>(dm_c.ndof_per_elem());

    std::vector<std::pair<uint32_t, uint32_t>> pat;
    pat.reserve(nt * ndr * ndc);
    for (std::size_t e = 0; e < nt; ++e)
        for (std::size_t a = 0; a < ndr; ++a)
        {
            const uint32_t ga = dm_r.elem_dof[ndr * e + a];
            for (std::size_t b = 0; b < ndc; ++b)
                pat.emplace_back(ga, dm_c.elem_dof[ndc * e + b]);
        }
    std::sort(pat.begin(), pat.end());
    pat.erase(std::unique(pat.begin(), pat.end()), pat.end());

    CsrPattern P;
    P.n = static_cast<uint32_t>(dm_r.n_dof);
    P.row_ptr.assign(static_cast<std::size_t>(P.n) + 1, 0u);
    for (const auto& pr : pat) P.row_ptr[static_cast<std::size_t>(pr.first) + 1]++;
    for (std::size_t i = 0; i < P.n; ++i) P.row_ptr[i + 1] += P.row_ptr[i];
    P.col_idx.resize(pat.size());
    for (std::size_t k = 0; k < pat.size(); ++k) P.col_idx[k] = pat[k].second;
    return P;
}

// 矩形散射: 单元矩阵是 nr x nc, 不做对称镜像 (B 不是对称算子)
inline void scatter_rect(std::vector<double>& val, const uint32_t* pos,
                         const double* Ke, int nr, int nc)
{
    for (int a = 0; a < nr; ++a)
        for (int b = 0; b < nc; ++b)
            val[pos[static_cast<std::size_t>(a) * nc + b]] +=
                Ke[static_cast<std::size_t>(a) * nc + b];
}

// ---------------------------------------------------------------------------
// (负)散度算子 B:   Bx[k,j] = -∫ ψ_k ∂φ_j/∂x,  By[k,j] = -∫ ψ_k ∂φ_j/∂y
//   行 = 压力自由度 (dm_p, P1), 列 = 速度自由度 (dm_v, P2)。
//   两个矩阵共用同一份矩形模式 (耦合结构完全一致), 一次装配同时产出。
//
//   被积函数次数 = OrderP + (OrderV - 1), 求积精度默认给到 OrderP + OrderV
//   (留一档余量; P1 x P2 时是 3, 而理论只需 2)。
//
//   【性能】位置表 pos 每单元算一次 (np*nv 次二分), 不是每个求积点一次 ——
//     M0 里优化标量装配时踩过的同一个点。ElemPos 预计算表留到 M2 (那里
//     每个 Newton 步都要重装 B 的 Neumann 部分)。
// ---------------------------------------------------------------------------
template <int OrderV, int OrderP>
void assemble_divergence(const Mesh& m, const DofMap& dm_v, const DofMap& dm_p,
                         const CsrPattern& P,
                         std::vector<double>& val_x, std::vector<double>& val_y,
                         int qdeg = 0)
{
    if (qdeg <= 0) qdeg = OrderV + OrderP;
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int nv = TriElement<OrderV>::ndof;
    constexpr int np = TriElement<OrderP>::ndof;
    const std::size_t nt = m.n_tris();

    val_x.assign(P.col_idx.size(), 0.0);
    val_y.assign(P.col_idx.size(), 0.0);

    double   Jinv[4], area = 0.0;
    double   dNdx[nv], dNdy[nv], Psi[np];
    double   Bxl[np * nv], Byl[np * nv];
    uint32_t pos[np * nv];
    double   vx[3], vy[3];

    for (std::size_t e = 0; e < nt; ++e)
    {
        const uint32_t* dofv = &dm_v.elem_dof[static_cast<std::size_t>(nv) * e];
        const uint32_t* dofp = &dm_p.elem_dof[static_cast<std::size_t>(np) * e];

        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        tri_element_jac(vx, vy, Jinv, area);

        for (int i = 0; i < np * nv; ++i) { Bxl[i] = 0.0; Byl[i] = 0.0; }

        for (std::size_t q = 0; q < Q.size(); ++q)
        {
            const double l0 = Q.l0[q], l1 = Q.l1[q], l2 = Q.l2[q];

            double Gref[2 * nv];
            TriElement<OrderV>::grad_ref(l0, l1, l2, Gref);
            tri_apply_jac<OrderV>(Jinv, Gref, dNdx, dNdy);
            TriElement<OrderP>::shape(l0, l1, l2, Psi);

            const double wq = area * Q.w[q];
            for (int k = 0; k < np; ++k)
                for (int j = 0; j < nv; ++j)
                {
                    Bxl[k * nv + j] -= wq * Psi[k] * dNdx[j];
                    Byl[k * nv + j] -= wq * Psi[k] * dNdy[j];
                }
        }

        for (int k = 0; k < np; ++k)
            for (int j = 0; j < nv; ++j)
                pos[k * nv + j] =
                    static_cast<uint32_t>(pattern_index(P, dofp[k], dofv[j]));

        scatter_rect(val_x, pos, Bxl, np, nv);
        scatter_rect(val_y, pos, Byl, np, nv);
    }
}

// ---------------------------------------------------------------------------
// Taylor-Hood 系统的全部算子
// ---------------------------------------------------------------------------
struct TaylorHood
{
    std::size_t nv = 0;                 // 速度自由度数 (P2)
    std::size_t np = 0;                 // 压力自由度数 (P1)

    sparse::SMatrix K;                  // 速度标量刚度 (nv x nv), 已含 nu, **未**加约束
    sparse::SMatrix Kd;                 // K 施加速度 Dirichlet 后的版本, 只给预条件器用
    sparse::SMatrix Bx, By;             // 负散度    (np x nv)
    sparse::SMatrix BxT, ByT;           // 它们的转置 (nv x np) == Bᵀ 的两条分量
    sparse::SMatrix Mp;                 // 压力质量   (np x np)
    std::vector<double> Mp_lumped;      // 行和 lumping, 作 Mp⁻¹ 的谱等价近似

    sparse::SMatrix Kfull;              // 展平鞍点系统 (2nv+np, 对称不定)
    std::vector<double> rhs;            // 展平右端项
    std::size_t      p_pin = 0;         // 被钉住的压力自由度
};

// ---------------------------------------------------------------------------
// 把 (K, Bx, By) 展平成一个大对称不定 SMatrix
//     行 2i+α   : A 部分 (列 2j+α, 值 K_ij)  +  Bᵀ 部分 (列 2nv+k)
//     行 2nv+k  : B 部分 (列 2j+α)
//   node-major 布局: 速度分量 α 的自由度 i 落在 2i+α, 压力自由度 k 落在 2nv+k。
//
//   这里走 COO 构造 (SMatrix 内部排序 + 累加重复项), 而不是 M0 的两遍法:
//   混合阶 + 矩形块 + 展平这套组合下, 手写模式推导的错误面远大于一次
//   O(nnz log nnz) 排序。M1 只在 setup 阶段做一次, 可以接受;
//   【M2 TODO】非线性时要复用模式重装, 那时再改成两遍法 + ElemPos。
// ---------------------------------------------------------------------------
inline sparse::SMatrix build_saddle_matrix(const sparse::SMatrix& K,
                                           const sparse::SMatrix& Bx,
                                           const sparse::SMatrix& By,
                                           const sparse::SMatrix& BxT,
                                           const sparse::SMatrix& ByT,
                                           std::size_t p_pin)
{
    const uint32_t nv = K.rows();
    const uint32_t np = Bx.rows();
    if (Bx.cols() != nv || By.cols() != nv)
        throw std::invalid_argument("build_saddle_matrix: B 的列数必须等于速度自由度数");
    if (BxT.rows() != nv || ByT.rows() != nv)
        throw std::invalid_argument("build_saddle_matrix: Bᵀ 的行数必须等于速度自由度数");
    if (p_pin >= np)
        throw std::out_of_range("build_saddle_matrix: 被钉住的压力自由度越界");

    const uint32_t n = 2u * nv + np;

    // 上界估计: A 部分 2*nnz(K), Bᵀ 部分 nnz(BxT)+nnz(ByT) == nnz(Bx)+nnz(By),
    //           B 部分 nnz(Bx)+nnz(By), 再加 1 个压力对角占位。
    const std::size_t guess = 2ull * K.nnz() + 2ull * Bx.nnz() + 2ull * By.nnz() + 1ull;

    std::vector<double>   vals;  vals.reserve(guess);
    std::vector<uint32_t> rl;    rl.reserve(guess);
    std::vector<uint32_t> cl;    cl.reserve(guess);

    const auto& Krp = K.row_ptr();   const auto& Kci = K.col_idx();   const auto& Kv = K.values();
    const auto& Bxp = Bx.row_ptr();  const auto& Bxc = Bx.col_idx();  const auto& Bxv = Bx.values();
    const auto& Byp = By.row_ptr();  const auto& Byc = By.col_idx();  const auto& Byv = By.values();
    const auto& Bxtp = BxT.row_ptr(); const auto& Bxtc = BxT.col_idx(); const auto& Bxtv = BxT.values();
    const auto& Bytp = ByT.row_ptr(); const auto& Bytc = ByT.col_idx(); const auto& Bytv = ByT.values();

    // ---- A = K ⊗ I₂ ----
    for (uint32_t i = 0; i < nv; ++i)
        for (uint32_t p = Krp[i]; p < Krp[i + 1]; ++p)
        {
            const uint32_t j = Kci[p];
            const double   v = Kv[p];
            rl.push_back(2u * i + 0u); cl.push_back(2u * j + 0u); vals.push_back(v);
            rl.push_back(2u * i + 1u); cl.push_back(2u * j + 1u); vals.push_back(v);
        }

    // ---- Bᵀ (速度行 x 压力列) ----
    for (uint32_t i = 0; i < nv; ++i)
    {
        for (uint32_t p = Bxtp[i]; p < Bxtp[i + 1]; ++p)
        {
            rl.push_back(2u * i + 0u);
            cl.push_back(2u * nv + Bxtc[p]);
            vals.push_back(Bxtv[p]);
        }
        for (uint32_t p = Bytp[i]; p < Bytp[i + 1]; ++p)
        {
            rl.push_back(2u * i + 1u);
            cl.push_back(2u * nv + Bytc[p]);
            vals.push_back(Bytv[p]);
        }
    }

    // ---- B (压力行 x 速度列) ----
    for (uint32_t k = 0; k < np; ++k)
    {
        const uint32_t rk = 2u * nv + k;
        for (uint32_t p = Bxp[k]; p < Bxp[k + 1]; ++p)
        {
            rl.push_back(rk); cl.push_back(2u * Bxc[p] + 0u); vals.push_back(Bxv[p]);
        }
        for (uint32_t p = Byp[k]; p < Byp[k + 1]; ++p)
        {
            rl.push_back(rk); cl.push_back(2u * Byc[p] + 1u); vals.push_back(Byv[p]);
        }
    }

    // ---- 压力 pin 的对角占位 (COO 会丢 0 值, 所以给 1.0; 随后被消元设成 1) ----
    rl.push_back(2u * nv + static_cast<uint32_t>(p_pin));
    cl.push_back(2u * nv + static_cast<uint32_t>(p_pin));
    vals.push_back(1.0);

    return sparse::SMatrix(n, n, vals, rl, cl);
}

// ---------------------------------------------------------------------------
// 速度块预条件 (B): BlockAMG —— 光滑聚合版, 作用在 BlockMatrix<double,2>
//
//   A = K ⊗ I₂  ⇒  块 (i,j) = K_ij · I₂, 即**每个非零块都是 m·I₂**。
//   这正好命中 blockMatrix.hpp 里的标量快速路径 (blk_amg_scalarize 的判据),
//   所以运行时不会退化成 BCSR 块路径, 单价和标量 AMG 同量级。
//
//   与标量 AMG 版的差别**只在粗化策略**: 这里是光滑聚合 (aggregation),
//   标量版是 Ruge-Stüben C/F 分裂 + 直接插值。P2 刚度不是 M-矩阵时两种
//   粗化的健壮性不同 —— 这正是要实测对比的东西。
// ---------------------------------------------------------------------------
inline std::function<void(const std::vector<double>&, std::vector<double>&)>
make_velocity_preconditioner_block(const sparse::SMatrix& K,
                                   int max_agg = 8, bool verbose = false,
                                   double drop_tol = 0.02, uint32_t max_row_nnz = 40,
                                   int n_smooth = 2)
{
    const uint32_t nv = K.rows();
    const auto& rp = K.row_ptr();
    const auto& ci = K.col_idx();
    const auto& vv = K.values();
    const std::size_t nb = ci.size();

    // 块值布局: block[p][i*BS + j] = A(块行 i, 块列 j) —— 行主序
    std::vector<uint32_t> brp(rp.begin(), rp.end());
    std::vector<uint32_t> bci(ci.begin(), ci.end());
    std::vector<double>   bval(nb * 4, 0.0);
    for (std::size_t p = 0; p < nb; ++p)
    {
        const double k = vv[p];
        bval[4 * p + 0] = k;    // (0,0)
        bval[4 * p + 1] = 0.0;  // (0,1)
        bval[4 * p + 2] = 0.0;  // (1,0)
        bval[4 * p + 3] = k;    // (1,1)
    }
    block::BlockMatrix<double, 2> A(nv, nv, std::move(brp), std::move(bci), std::move(bval));

    // 返回的算子作用在长度 2*nv 的 node-major 向量上 —— 与 Stokes 的速度
    // 自由度布局 (2i+alpha) 逐位一致, 不需要任何重排。
    return block::make_block_amg_preconditioner<double, 2>(
        A, max_agg, verbose, drop_tol,
        (max_row_nnz > 0) ? max_row_nnz : uint32_t(40), n_smooth, true);
}

// ---------------------------------------------------------------------------
// 速度块预条件: A⁻¹ = K⁻¹ ⊗ I₂ —— 对两个分量切片各跑一次标量 AMG
//
//   【必须传 Kd, 不能传 K】K 是纯 Neumann 刚度: 没有 Dirichlet 约束时它有
//     一个常数向量零空间 (半正定、奇异)。AMG 吃下奇异矩阵会给出垃圾修正
//     (V-cycle 在最粗层上解的是奇异系统)。而消元后的 Kfull 里速度块其实是
//     "固定行列被清掉、对角置 1" 的版本, 所以预条件器要匹配那个版本 ——
//     也就是先对 K 单独施加一次速度 Dirichlet 得到的 Kd (SPD)。
//     (rhs 那一项是给整体系统用的, 这里丢弃; g 取什么都行, 它只进 rhs。)
//
//   为什么敢这么做见文件头: Stokes 的 A 精确等于 K ⊗ I₂, 不是近似。
//   两个分量的切片/回填是 O(nv) 的 gather/scatter, 与 AMG V-cycle 自身
//   同阶, 不会成为瓶颈; 缓冲区随闭包走, 不在每次 apply 里分配。
// ---------------------------------------------------------------------------
// 速度块预条件的三种实现 —— 全部基于同一个事实: A 精确等于 K ⊗ I₂
//   0 = 标量 AMG on K + 阻尼 Jacobi 光滑   (默认; 迄今全部实测都是它)
//   1 = 标量 AMG on K + Chebyshev 多项式光滑  (针对"P2 不是 M-矩阵")
//   2 = BlockAMG 光滑聚合 on BlockMatrix<double,2>  (换粗化策略)
//   对比这三组可以直接把"光滑子不行"和"粗化不行"分开 —— 这是 M1 里
//   唯一还悬着的代数问题, 光看迭代总数分不清是哪一环。
enum class VelPreKind { AMG_JACOBI = 0, AMG_CHEB = 1, BLOCK_AMG = 2 };

inline std::function<void(const std::vector<double>&, std::vector<double>&)>
make_velocity_preconditioner(const sparse::SMatrix& K,
                             double drop_tol = 0.02, uint32_t max_row_nnz = 60,
                             bool verbose = false, uint32_t coarse_thresh = 300,
                             int kind = 0, int cheb_deg = 2, double cheb_ratio = 30.0,
                             int blk_max_agg = 8, int blk_n_smooth = 2)
{
    const std::size_t nv = K.rows();

    // ---- kind 2: 光滑聚合 BlockAMG (走的完全是 blockMatrix.hpp 那套) ----
    if (kind == static_cast<int>(VelPreKind::BLOCK_AMG))
        return make_velocity_preconditioner_block(
            K, blk_max_agg, verbose, drop_tol, max_row_nnz, blk_n_smooth);

    auto amg = sparse::SMatrix::make_amg_preconditioner(
        K, drop_tol, max_row_nnz, verbose, coarse_thresh,
        (kind == static_cast<int>(VelPreKind::AMG_CHEB)) ? 1 : 0,
        cheb_deg, cheb_ratio);

    struct WS { std::vector<double> rx, ry, zx, zy; };
    auto ws = std::make_shared<WS>();
    ws->rx.resize(nv); ws->ry.resize(nv); ws->zx.resize(nv); ws->zy.resize(nv);

    return [amg, ws, nv](const std::vector<double>& r, std::vector<double>& z)
    {
        if (r.size() != 2u * nv)
            throw std::invalid_argument("速度预条件: 残差长度 != 2 * 速度自由度数");
        std::vector<double>& rx = ws->rx;   std::vector<double>& ry = ws->ry;
        std::vector<double>& zx = ws->zx;   std::vector<double>& zy = ws->zy;

        const std::ptrdiff_t nvi = static_cast<std::ptrdiff_t>(nv);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(nv > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < nvi; ++i)
        {
            rx[i] = r[2u * static_cast<std::size_t>(i) + 0u];
            ry[i] = r[2u * static_cast<std::size_t>(i) + 1u];
        }

        amg(rx, zx);
        amg(ry, zy);

        z.resize(2u * nv);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(nv > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < nvi; ++i)
        {
            z[2u * static_cast<std::size_t>(i) + 0u] = zx[i];
            z[2u * static_cast<std::size_t>(i) + 1u] = zy[i];
        }
    };
}

// ---------------------------------------------------------------------------
// Schur 预条件: S⁻¹ ≈ nu · Mp⁻¹   (Mp 用行和 lumping 求逆)
//   M2 需要 Cahouet-Chabard 的第二项 alpha·Lp⁻¹ (alpha = 1/dt) 时,
//   在这里加一个 Lp 的 AMG apply 并按 alpha 加权即可, 接口不用动。
// ---------------------------------------------------------------------------
inline std::function<void(const std::vector<double>&, std::vector<double>&)>
make_pressure_schur_preconditioner(const std::vector<double>& Mp_lumped, double nu,
                                   std::size_t p_pin = 0)
{
    auto ml = std::make_shared<std::vector<double>>(Mp_lumped);
    return [ml, nu, p_pin](const std::vector<double>& r, std::vector<double>& z)
    {
        const std::size_t n = ml->size();
        if (r.size() != n)
            throw std::invalid_argument("Schur 预条件: 残差长度 != 压力自由度数");
        z.resize(n);
        const double* m = ml->data();
        const std::ptrdiff_t ni = static_cast<std::ptrdiff_t>(n);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(n > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < ni; ++i)
        {
            // lumped 质量严格为正; 万一为 0 (空行) 就退化为 no-op, 不除零
            const double d = m[i];
            z[i] = (d > 0.0) ? nu * r[i] / d : 0.0;
        }
        // 【压力 pin 处必须与算子一致】被钉住的那一行在鞍点矩阵里是**单位行**
        //   (消元换来的), 而不是 G = B A^-1 B^T 的零行。Mp_lumped ~ h^2, 若这里
        //   照样除以它, 该分量会被放大 ~1/h^2 倍 (nx=32 时约 1000x), 再经 B^T
        //   反射成速度修正里的一个局部尖峰 —— GMRES 要额外迭代去抵消它。
        //   正确的做法就是照单位行办: 原样返回。
        if (p_pin < n) z[p_pin] = r[p_pin];
    };
}

// 把 TaylorHood 系统接成 block::SaddlePointSystem (给 make_schur_preconditioner 用)
inline block::SaddlePointSystem make_saddle_point_system(std::shared_ptr<TaylorHood> S)
{
    const std::size_t nv = S->nv;
    const std::size_t np = S->np;

    block::SaddlePointSystem sys;
    sys.n_vel  = 2u * nv;
    sys.n_pres = np;

    // y = A u,  A = K ⊗ I₂ —— 直接按 K 的 CSR 展开, 不构造 A_flat,
    // 省掉一份 2*nnz(K) 的副本 (M1 规模下不明显, M2 上就是实打实的内存)。
    sys.spmv_A = [S, nv](const std::vector<double>& u, std::vector<double>& y)
    {
        if (u.size() != 2u * nv)
            throw std::invalid_argument("spmv_A: 输入长度 != 2 * 速度自由度数");
        y.assign(2u * nv, 0.0);
        const auto& rp = S->K.row_ptr();
        const auto& ci = S->K.col_idx();
        const auto& vv = S->K.values();
        double* __restrict yp = y.data();
        const double* __restrict up = u.data();

        const std::ptrdiff_t nvi = static_cast<std::ptrdiff_t>(nv);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(nv > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < nvi; ++i)
        {
            double sx = 0.0, sy = 0.0;
            for (uint32_t p = rp[i]; p < rp[i + 1]; ++p)
            {
                const double a = vv[p];
                const double ujx = up[2u * static_cast<std::size_t>(ci[p]) + 0u];
                const double ujy = up[2u * static_cast<std::size_t>(ci[p]) + 1u];
                sx += a * ujx;
                sy += a * ujy;
            }
            yp[2u * static_cast<std::size_t>(i) + 0u] = sx;
            yp[2u * static_cast<std::size_t>(i) + 1u] = sy;
        }
    };

    // (B u)_k = Σ_j ( Bx[k,j] u_x[j] + By[k,j] u_y[j] )
    sys.spmv_B = [S, nv, np](const std::vector<double>& u, std::vector<double>& y)
    {
        if (u.size() != 2u * nv)
            throw std::invalid_argument("spmv_B: 输入长度 != 2 * 速度自由度数");
        y.assign(np, 0.0);
        const auto& Bxp = S->Bx.row_ptr(); const auto& Bxc = S->Bx.col_idx(); const auto& Bxv = S->Bx.values();
        const auto& Byp = S->By.row_ptr(); const auto& Byc = S->By.col_idx(); const auto& Byv = S->By.values();
        double* __restrict yp = y.data();
        const double* __restrict up = u.data();

        const std::ptrdiff_t npi = static_cast<std::ptrdiff_t>(np);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(np > 4096)
        #endif
        for (std::ptrdiff_t k = 0; k < npi; ++k)
        {
            const std::size_t kk = static_cast<std::size_t>(k);
            double s = 0.0;
            for (uint32_t p = Bxp[kk]; p < Bxp[kk + 1]; ++p)
                s += Bxv[p] * up[2u * static_cast<std::size_t>(Bxc[p]) + 0u];
            for (uint32_t p = Byp[kk]; p < Byp[kk + 1]; ++p)
                s += Byv[p] * up[2u * static_cast<std::size_t>(Byc[p]) + 1u];
            yp[kk] = s;
        }
    };

    // (Bᵀ p)[2i] = Σ_k Bx[k,i] p[k];  (Bᵀ p)[2i+1] = Σ_k By[k,i] p[k]
    //   用预先转置好的 BxT / ByT, 按行访问
    sys.spmv_Bt = [S, nv, np](const std::vector<double>& p, std::vector<double>& y)
    {
        if (p.size() != np)
            throw std::invalid_argument("spmv_Bt: 输入长度 != 压力自由度数");
        y.assign(2u * nv, 0.0);
        const auto& Bxtp = S->BxT.row_ptr(); const auto& Bxtc = S->BxT.col_idx(); const auto& Bxtv = S->BxT.values();
        const auto& Bytp = S->ByT.row_ptr(); const auto& Bytc = S->ByT.col_idx(); const auto& Bytv = S->ByT.values();
        double* __restrict yp = y.data();
        const double* __restrict pp = p.data();

        const std::ptrdiff_t nvi = static_cast<std::ptrdiff_t>(nv);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(nv > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < nvi; ++i)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            double sx = 0.0, sy = 0.0;
            for (uint32_t q = Bxtp[ii]; q < Bxtp[ii + 1]; ++q)
                sx += Bxtv[q] * pp[Bxtc[q]];
            for (uint32_t q = Bytp[ii]; q < Bytp[ii + 1]; ++q)
                sy += Bytv[q] * pp[Bytc[q]];
            yp[2u * ii + 0u] = sx;
            yp[2u * ii + 1u] = sy;
        }
    };

    return sys;
}

// ---------------------------------------------------------------------------
// 从展平解里拆出速度 / 压力
// ---------------------------------------------------------------------------
inline std::vector<double> extract_pressure(const std::vector<double>& x, std::size_t nv)
{
    return std::vector<double>(x.begin() + static_cast<std::ptrdiff_t>(2u * nv), x.end());
}

// ---------------------------------------------------------------------------
// 向量值场 (两个分量) 的真实 L2 / H1 误差
//   两个分量各自走 compute_errors 的逐点求积, 再按分量平方和合成:
//       ‖u - u_h‖² = ‖e_x‖² + ‖e_y‖²
//   压力单独走标量版本。
// ---------------------------------------------------------------------------
template <int Order, class UX, class GX, class UY, class GY>
std::pair<double, double> compute_errors_vec2(const Mesh& m, const DofMap& dm,
                                              const std::vector<double>& ux,
                                              const std::vector<double>& uy,
                                              UX ux_ex, GX gx_ex,
                                              UY uy_ex, GY gy_ex,
                                              int qdeg = 5)
{
    const auto ex = compute_errors<Order>(m, dm, ux, ux_ex, gx_ex, qdeg);
    const auto ey = compute_errors<Order>(m, dm, uy, uy_ex, gy_ex, qdeg);
    return { std::sqrt(ex.first * ex.first + ey.first * ey.first),
             std::sqrt(ex.second * ex.second + ey.second * ey.second) };
}

// ---------------------------------------------------------------------------
// 去均值误差  ‖e - mean(e)‖_L2,  mean(e) = ∫e / ∫1
//
//   【为什么需要它】压力 pin 之后, G = B A⁻¹Bᵀ 的零空间被换成一个"几乎还是
//   零空间"的方向 q = 1 - φ_pin: 它在 G 上的 Rayleigh 商极小 (O(h^?) ), 于是
//   GMRES (最小化残差) 在这一方向上几乎"看不到"误差 —— 残差已经 1e-12 了,
//   压力 L2 误差却还有 1e-6。这个方向还是**规范自由度**: Bᵀ·const = 0,
//   它完全不进入速度场, 对物理没有任何影响。
//
//   所以判读规则是:
//     p-L2 大但 p-L2* 小  -> 误差全在规范方向上, 压力"整体漂移"了一个常数,
//                            物理场是对的, 不是装配错误, 也不是预条件器失效;
//     p-L2 和 p-L2* 都大  -> 真的有离散化/代数问题, 得继续查。
// ---------------------------------------------------------------------------
template <int Order, class UExact>
double compute_error_mean_free(const Mesh& m, const DofMap& dm,
                               const std::vector<double>& u,
                               UExact u_ex, int qdeg = 5,
                               double* mean_out = nullptr)
{
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int nd = TriElement<Order>::ndof;
    const std::size_t nt = m.n_tris();

    double Jinv[4], area = 0.0, N[nd], vx[3], vy[3];
    double inte = 0.0, area_tot = 0.0;

    // 第一遍: ∫ e 与 ∫ 1
    for (std::size_t e = 0; e < nt; ++e)
    {
        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        tri_element_jac(vx, vy, Jinv, area);
        for (std::size_t q = 0; q < Q.size(); ++q)
        {
            const double l0 = Q.l0[q], l1 = Q.l1[q], l2 = Q.l2[q];
            TriElement<Order>::shape(l0, l1, l2, N);
            const double xq = l0 * vx[0] + l1 * vx[1] + l2 * vx[2];
            const double yq = l0 * vy[0] + l1 * vy[1] + l2 * vy[2];
            double uh = 0.0;
            for (int a = 0; a < nd; ++a)
                uh += u[dm.elem_dof[static_cast<std::size_t>(nd) * e + a]] * N[a];
            const double wq = area * Q.w[q];
            inte     += wq * (uh - u_ex(xq, yq));
            area_tot += wq;
        }
    }
    const double mu = (area_tot > 0.0) ? (inte / area_tot) : 0.0;
    if (mean_out) *mean_out = mu;

    // 第二遍: ∫ (e - mu)²
    double e2 = 0.0;
    for (std::size_t e = 0; e < nt; ++e)
    {
        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        tri_element_jac(vx, vy, Jinv, area);
        for (std::size_t q = 0; q < Q.size(); ++q)
        {
            const double l0 = Q.l0[q], l1 = Q.l1[q], l2 = Q.l2[q];
            TriElement<Order>::shape(l0, l1, l2, N);
            const double xq = l0 * vx[0] + l1 * vx[1] + l2 * vx[2];
            const double yq = l0 * vy[0] + l1 * vy[1] + l2 * vy[2];
            double uh = 0.0;
            for (int a = 0; a < nd; ++a)
                uh += u[dm.elem_dof[static_cast<std::size_t>(nd) * e + a]] * N[a];
            const double d = (uh - u_ex(xq, yq)) - mu;
            e2 += area * Q.w[q] * d * d;
        }
    }
    return std::sqrt(e2);
}

}   // namespace fem
