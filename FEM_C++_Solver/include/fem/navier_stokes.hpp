#pragma once
// ============================================================================
// M2: 稳态 Navier-Stokes (Taylor-Hood P2/P1) —— 对流项与 Newton 线性化
// ----------------------------------------------------------------------------
//   方程 (无量纲; nu = 1/Re):
//       (u·∇)u + ∇p - nu Δu = f
//       ∇·u                 = 0
//
//   弱形式。因为速度在**整个边界**上都是 Dirichlet (v|∂Ω = 0), 压力梯度项
//   分部积分后的边界项 ∫∂Ω p (v·n) 恒为 0, 于是
//       R_u(v) = nu∫∇u:∇v + ∫(u·∇)u·v - ∫p ∇·v - ∫f·v = 0
//       R_p(q) = -∫q ∇·u = 0                                (与 M1 同号)
//
//   ---------------------------------------------------------------------
//   【Newton 线性化: 这是 M2 与 M1 的本质差别】
//     d/dt [ ((u+tw)·∇)(u+tw) ] |_{t=0} = (w·∇)u + (u·∇)w
//   两项各对应一块:
//     (u·∇)w  ->  C_ij = ∫ (u_h·∇φ_j) φ_i      —— **分量对角** (w 的第 α 个
//                 分量只进第 α 个动量方程), 所以这一块是 C ⊗ I₂ 的形状;
//     (w·∇)u  ->  G^{αβ}_ij = ∫ φ_i φ_j ∂_β u_α —— **四个块** (α = 被微分的
//                 速度分量, β = 导数方向), 它把两个分量耦合起来。
//   因此速度块
//       A = nu·K + (C ⊗ I₂) + G,   G = [[Gxx, Gxy], [Gyx, Gyy]]
//   **不再**是"标量矩阵 ⊗ I₂" —— M1 里那条"给标量 K 做一次 AMG 就能当
//   速度块预条件"的捷径到这里失效 (只剩 C 那部分还是 ⊗ I₂ 的形状)。
//
//   符号自检 (最容易写反的地方): 线性系统解的是 J·δ = -R, 所以
//     R 里出现 +C(u)u, J 里就必须出现 +C(u) + G(u), 两者同号。
// ============================================================================
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fem/assembler.hpp"
#include "fem/dofmap.hpp"
#include "fem/element.hpp"
#include "fem/mesh.hpp"
#include "fem/quadrature.hpp"
#include "fem/stokes.hpp"        // scatter_rect / build_pattern_rect / TaylorHood
#include "math/array.hpp"

namespace fem
{

// ---------------------------------------------------------------------------
// 对流项 C 与 Newton 项 G 的一次装配
//
//   输入: 当前解的两个分量 ux, uy (长度 = dm_v.n_dof)
//   输出 (全部是 nv x nv 的标量 CSR, 与 P2 刚度矩阵**同一个模式** P):
//     C_ij       = ∫ (u_h·∇φ_j) φ_i
//     G^{αβ}_ij  = ∫ φ_i φ_j ∂_β u_α
//
//   求积阶: 被积函数次数 = φ(2) · u(2) · ∇φ(1) = 5 —— 必须给到 5 阶,
//   用 M1 散度那套 OrderV+OrderP=3 的阶会欠积分, 表现为"Newton 收敛到最后
//   卡在一个 1e-4 的残差上", 这种错极难与"预条件不够好"区分开。
//
//   【性能】pos 表每单元算一次 (nd² 次二分), 不是每求积点一次 —— 与 M0/M1
//   同一条准则。五个矩阵共享同一份 pos。
// ---------------------------------------------------------------------------
template <int OrderV>
void assemble_ns_terms(const Mesh& m, const DofMap& dm_v, const CsrPattern& P,
                       const std::vector<double>& ux,
                       const std::vector<double>& uy,
                       std::vector<double>& val_C,
                       std::vector<double>& val_Gxx,
                       std::vector<double>& val_Gxy,
                       std::vector<double>& val_Gyx,
                       std::vector<double>& val_Gyy,
                       int qdeg = 0)
{
    if (qdeg <= 0) qdeg = 2 * OrderV + 1;
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int nd = TriElement<OrderV>::ndof;
    const std::size_t nt = m.n_tris();

    const std::size_t nnz = P.col_idx.size();
    val_C.assign(nnz, 0.0);
    val_Gxx.assign(nnz, 0.0);
    val_Gxy.assign(nnz, 0.0);
    val_Gyx.assign(nnz, 0.0);
    val_Gyy.assign(nnz, 0.0);

    double   Jinv[4], area = 0.0;
    double   N[nd], Gref[2 * nd], dNdx[nd], dNdy[nd];
    double   Cl[nd * nd], Xl[nd * nd], Yl[nd * nd], Zl[nd * nd], Wl[nd * nd];
    uint32_t pos[nd * nd];
    double   vx[3], vy[3];

    for (std::size_t e = 0; e < nt; ++e)
    {
        const uint32_t* dof = &dm_v.elem_dof[static_cast<std::size_t>(nd) * e];

        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        tri_element_jac(vx, vy, Jinv, area);

        for (int i = 0; i < nd * nd; ++i)
        { Cl[i] = 0.0; Xl[i] = 0.0; Yl[i] = 0.0; Zl[i] = 0.0; Wl[i] = 0.0; }

        for (std::size_t q = 0; q < Q.size(); ++q)
        {
            const double l0 = Q.l0[q], l1 = Q.l1[q], l2 = Q.l2[q];

            TriElement<OrderV>::shape(l0, l1, l2, N);
            TriElement<OrderV>::grad_ref(l0, l1, l2, Gref);
            tri_apply_jac<OrderV>(Jinv, Gref, dNdx, dNdy);

            // 该求积点上的当前解与它的四个偏导
            double uhx = 0.0, uhy = 0.0;
            double duxdx = 0.0, duxdy = 0.0, duydx = 0.0, duydy = 0.0;
            for (int a = 0; a < nd; ++a)
            {
                const std::size_t gi = dof[a];
                uhx   += ux[gi] * N[a];
                uhy   += uy[gi] * N[a];
                duxdx += ux[gi] * dNdx[a];
                duxdy += ux[gi] * dNdy[a];
                duydx += uy[gi] * dNdx[a];
                duydy += uy[gi] * dNdy[a];
            }

            const double wq = area * Q.w[q];
            for (int i = 0; i < nd; ++i)
            {
                const double Ni = N[i], wi = wq * Ni;
                for (int j = 0; j < nd; ++j)
                {
                    const std::size_t ij = static_cast<std::size_t>(i) * nd + j;
                    const double wN = wi * N[j];
                    Cl[ij] += wi * (uhx * dNdx[j] + uhy * dNdy[j]);   // (u·∇φ_j) φ_i
                    Xl[ij] += wN * duxdx;                              // ∂u_x/∂x
                    Yl[ij] += wN * duxdy;                              // ∂u_x/∂y
                    Zl[ij] += wN * duydx;                              // ∂u_y/∂x
                    Wl[ij] += wN * duydy;                              // ∂u_y/∂y
                }
            }
        }

        for (int i = 0; i < nd; ++i)
            for (int j = 0; j < nd; ++j)
                pos[static_cast<std::size_t>(i) * nd + j] =
                    static_cast<uint32_t>(pattern_index(P, dof[i], dof[j]));

        scatter_rect(val_C,   pos, Cl, nd, nd);
        scatter_rect(val_Gxx, pos, Xl, nd, nd);
        scatter_rect(val_Gxy, pos, Yl, nd, nd);
        scatter_rect(val_Gyx, pos, Zl, nd, nd);
        scatter_rect(val_Gyy, pos, Wl, nd, nd);
    }
}

// ---------------------------------------------------------------------------
// 把 Newton 系统展平成 (2nv+np) 的鞍点矩阵
//
//   行 2i+0:  [Ad + Gxx]_ij (列 2j+0),  Gxy_ij (列 2j+1),  Bᵀ_x (列 2nv+k)
//   行 2i+1:  Gyx_ij (列 2j+0),  [Ad + Gyy]_ij (列 2j+1),  Bᵀ_y (列 2nv+k)
//   行 2nv+k: Bx (列 2j+0), By (列 2j+1)
//
//   Ad = nu·K + C —— 标量, 落在两个分量的对角块上。
//   与 M1 的 build_saddle_matrix 完全同构, 只是 A 部分从 1 个标量矩阵变成
//   "1 个对角块 + 4 个耦合块"。
//
//   同样走 COO (内部排序 + 累加重复项): 对角块与 Gxx 会写同一个位置
//   (2i+0, 2j+0), 必须靠累加合起来。
// ---------------------------------------------------------------------------
inline sparse::SMatrix build_saddle_matrix_ns(
    const sparse::SMatrix& Ad,
    const sparse::SMatrix& Gxx, const sparse::SMatrix& Gxy,
    const sparse::SMatrix& Gyx, const sparse::SMatrix& Gyy,
    const sparse::SMatrix& Bx,  const sparse::SMatrix& By,
    const sparse::SMatrix& BxT, const sparse::SMatrix& ByT,
    std::size_t p_pin)
{
    const uint32_t nv = Ad.rows();
    const uint32_t np = Bx.rows();
    if (Bx.cols() != nv || By.cols() != nv)
        throw std::invalid_argument("build_saddle_matrix_ns: B 的列数必须等于速度自由度数");
    if (p_pin >= np)
        throw std::out_of_range("build_saddle_matrix_ns: 被钉住的压力自由度越界");

    const uint32_t n = 2u * nv + np;
    const std::size_t guess = 2ull * Ad.nnz() + 2ull * Gxx.nnz() + 2ull * Gxy.nnz()
                            + 2ull * Gyx.nnz() + 2ull * Gyy.nnz()
                            + 2ull * Bx.nnz() + 2ull * By.nnz() + 1ull;

    std::vector<double>   vals;  vals.reserve(guess);
    std::vector<uint32_t> rl;    rl.reserve(guess);
    std::vector<uint32_t> cl;    cl.reserve(guess);

    const auto& Arp = Ad.row_ptr();  const auto& Aci = Ad.col_idx();  const auto& Avv = Ad.values();
    const auto& Xrp = Gxx.row_ptr(); const auto& Xci = Gxx.col_idx(); const auto& Xvv = Gxx.values();
    const auto& Yrp = Gxy.row_ptr(); const auto& Yci = Gxy.col_idx(); const auto& Yvv = Gxy.values();
    const auto& Zrp = Gyx.row_ptr(); const auto& Zci = Gyx.col_idx(); const auto& Zvv = Gyx.values();
    const auto& Wrp = Gyy.row_ptr(); const auto& Wci = Gyy.col_idx(); const auto& Wvv = Gyy.values();
    const auto& Bxp = Bx.row_ptr();  const auto& Bxc = Bx.col_idx();  const auto& Bxv = Bx.values();
    const auto& Byp = By.row_ptr();  const auto& Byc = By.col_idx();  const auto& Byv = By.values();
    const auto& Bxtp = BxT.row_ptr(); const auto& Bxtc = BxT.col_idx(); const auto& Bxtv = BxT.values();
    const auto& Bytp = ByT.row_ptr(); const auto& Bytc = ByT.col_idx(); const auto& Bytv = ByT.values();

    // ---- A = (nu K + C) ⊗ I₂ + G ----
    for (uint32_t i = 0; i < nv; ++i)
    {
        const uint32_t r0 = 2u * i + 0u, r1 = 2u * i + 1u;
        for (uint32_t p = Arp[i]; p < Arp[i + 1]; ++p)
        {
            const double v = Avv[p];
            const uint32_t c0 = 2u * Aci[p] + 0u, c1 = 2u * Aci[p] + 1u;
            rl.push_back(r0); cl.push_back(c0); vals.push_back(v);
            rl.push_back(r1); cl.push_back(c1); vals.push_back(v);
        }
        for (uint32_t p = Xrp[i]; p < Xrp[i + 1]; ++p)
        {
            const uint32_t c = 2u * Xci[p] + 0u;
            if (Xvv[p] != 0.0) { rl.push_back(r0); cl.push_back(c); vals.push_back(Xvv[p]); }
        }
        for (uint32_t p = Yrp[i]; p < Yrp[i + 1]; ++p)
        {
            const uint32_t c = 2u * Yci[p] + 1u;
            if (Yvv[p] != 0.0) { rl.push_back(r0); cl.push_back(c); vals.push_back(Yvv[p]); }
        }
        for (uint32_t p = Zrp[i]; p < Zrp[i + 1]; ++p)
        {
            const uint32_t c = 2u * Zci[p] + 0u;
            if (Zvv[p] != 0.0) { rl.push_back(r1); cl.push_back(c); vals.push_back(Zvv[p]); }
        }
        for (uint32_t p = Wrp[i]; p < Wrp[i + 1]; ++p)
        {
            const uint32_t c = 2u * Wci[p] + 1u;
            if (Wvv[p] != 0.0) { rl.push_back(r1); cl.push_back(c); vals.push_back(Wvv[p]); }
        }
    }

    // ---- Bᵀ 与 B (与 M1 逐位一致) ----
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

    // ---- 压力 pin 的对角占位 ----
    rl.push_back(2u * nv + static_cast<uint32_t>(p_pin));
    cl.push_back(2u * nv + static_cast<uint32_t>(p_pin));
    vals.push_back(1.0);

    return sparse::SMatrix(n, n, vals, rl, cl);
}

// ---------------------------------------------------------------------------
// 把速度块 A = (nuK + C) ⊗ I₂ + G 单独展平成 (2nv x 2nv)
//
//   与 build_saddle_matrix_ns 的 A 部分逐位一致, 只是不含 B / Bᵀ ——
//   ILU(0) 这类"必须拿到整块矩阵"的预条件器需要它 (AMG 那两条路径只用
//   标量 Ad, 不需要整块)。
//
//   同样走 COO: 对角块 Ad 与 Gxx 写在同一个位置 (2i+0, 2j+0), 必须靠
//   COO 构造的累加合起来。
// ---------------------------------------------------------------------------
inline sparse::SMatrix build_velocity_block_matrix(
    const sparse::SMatrix& Ad,
    const sparse::SMatrix& Gxx, const sparse::SMatrix& Gxy,
    const sparse::SMatrix& Gyx, const sparse::SMatrix& Gyy)
{
    const uint32_t nv = Ad.rows();
    const uint32_t n  = 2u * nv;

    const auto& Arp = Ad.row_ptr();  const auto& Aci = Ad.col_idx();  const auto& Avv = Ad.values();
    const auto& Xrp = Gxx.row_ptr(); const auto& Xci = Gxx.col_idx(); const auto& Xvv = Gxx.values();
    const auto& Yrp = Gxy.row_ptr(); const auto& Yci = Gxy.col_idx(); const auto& Yvv = Gxy.values();
    const auto& Zrp = Gyx.row_ptr(); const auto& Zci = Gyx.col_idx(); const auto& Zvv = Gyx.values();
    const auto& Wrp = Gyy.row_ptr(); const auto& Wci = Gyy.col_idx(); const auto& Wvv = Gyy.values();

    const std::size_t guess = 2ull * Ad.nnz() + Gxx.nnz() + Gxy.nnz()
                            + Gyx.nnz() + Gyy.nnz() + 1ull;
    std::vector<double>   vals; vals.reserve(guess);
    std::vector<uint32_t> rl;   rl.reserve(guess);
    std::vector<uint32_t> cl;   cl.reserve(guess);

    for (uint32_t i = 0; i < nv; ++i)
    {
        const uint32_t r0 = 2u * i + 0u, r1 = 2u * i + 1u;
        for (uint32_t q = Arp[i]; q < Arp[i + 1]; ++q)
        {
            const double v = Avv[q];
            rl.push_back(r0); cl.push_back(2u * Aci[q] + 0u); vals.push_back(v);
            rl.push_back(r1); cl.push_back(2u * Aci[q] + 1u); vals.push_back(v);
        }
        for (uint32_t q = Xrp[i]; q < Xrp[i + 1]; ++q)
            if (Xvv[q] != 0.0) { rl.push_back(r0); cl.push_back(2u * Xci[q] + 0u); vals.push_back(Xvv[q]); }
        for (uint32_t q = Yrp[i]; q < Yrp[i + 1]; ++q)
            if (Yvv[q] != 0.0) { rl.push_back(r0); cl.push_back(2u * Yci[q] + 1u); vals.push_back(Yvv[q]); }
        for (uint32_t q = Zrp[i]; q < Zrp[i + 1]; ++q)
            if (Zvv[q] != 0.0) { rl.push_back(r1); cl.push_back(2u * Zci[q] + 0u); vals.push_back(Zvv[q]); }
        for (uint32_t q = Wrp[i]; q < Wrp[i + 1]; ++q)
            if (Wvv[q] != 0.0) { rl.push_back(r1); cl.push_back(2u * Wci[q] + 1u); vals.push_back(Wvv[q]); }
    }
    return sparse::SMatrix(n, n, vals, rl, cl);
}

// ---------------------------------------------------------------------------
// SUPG (Streamline-Upwind / Petrov-Galerkin) 稳定项
// ----------------------------------------------------------------------------
//   动量方程的强形式残差 (单元内):
//       R_M = (u·∇)u + ∇p - nu Δu - f
//   SUPG 把检验函数 w 换成 w + tau (u·∇)w, 于是动量方程多出一项
//       Σ_K tau_K ∫_K (u·∇w)·R_M
//
//   【为什么 -nu Δu 与 -f 一项都不能少】只有用**完整**的强形式残差,
//   精确解才让 R_M ≡ 0, 这一项才在精确解处消失, 格式才是 consistent 的。
//   少任何一项都会给制造解验阶引入 O(tau) 的污染 —— 那不是"小扰动", 在
//   nu=1 的 MMS 上 tau~1e-4 但 Δu~O(10), 污染量级直接盖过离散误差本身。
//
//   【tau: Shakib / Codina 的双渐近形式】
//       h   = 2|u| / Σ_a |u·∇φ_a|      (Tezduyar 的流线长度; |u|≈0 时退回 sqrt(2A))
//       tau = [ (2|u|/h)² + (4 nu/h²)² ]^{-1/2}
//   对流占优 -> tau ≈ h/(2|u|);  扩散占优 -> tau ≈ h²/(4 nu)。
//   这样做的**副作用正是想要的**: 沿流线补上的扩散 tau|u|² 在高 Re 下远
//   大于 nu 本身, 速度块重新变成"椭圆主导", AMG 才有得可干 —— 实测
//   Re=400 时 C/(νK) ≈ 12, 加上 SUPG 后 C/(νK+Ssd) ≈ 1.7。
//
//   【Newton 线性化: 半线性化】
//     tau 与流线算子 (u·∇w) 都冻结在当前迭代步 (SUPG 的标准做法);
//     R_M 里只线性化 (u·∇)δu、(δu·∇)u 与 ∇δp 三项,
//     -nu Δδu 丢掉 (P2 的 Δu 是单元常数, 线性化会引出非物理的高阶耦合;
//     且它的量级只有 tau·|u|·nu/h, 在 Re>=100 时不到对流项的 5%)。
//     丢掉它只让 Jacobian 不那么精确 (Newton 仍收敛到同一个根), 不影响
//     残差的一致性。
//
//   【Δu_h 怎么算】P2 的一阶导是**精确的 P1**, 二阶导是单元常数。所以先取三个
//   顶点上的 ∇u_h, 再用重心坐标的梯度 ∇l_v 求导即可, 不需要 Hessian 接口:
//       ∂²u_x/∂x² = Σ_v (∂u_x/∂x)|_v · ∂l_v/∂x ,
//       ∇l_1 = (Jinv[0], Jinv[2]), ∇l_2 = (Jinv[1], Jinv[3]), ∇l_0 = -∇l_1-∇l_2
//
//   产出 (全是"增量", 调用方自行加到对应矩阵/向量上):
//     Ssd_ij  = tau ∫ (u·∇φ_i)(u·∇φ_j)      -> 加进 Ad (⊗I₂ 的对角块)
//     SGxx_ij = tau ∫ (u·∇φ_i) φ_j ∂_x u_x      -> 加进 Gxx (其余三个同理)
//     DxT_ik  = tau ∫ (u·∇φ_i) ∂_x ψ_k          -> 加进 Bᵀ_x (SUPG 独有的压力耦合)
//     rsx_i   = tau ∫ (u·∇φ_i) R_M,x            -> 加进动量残差
// ---------------------------------------------------------------------------
template <int OrderV, int OrderP, class FX, class FY>
void assemble_supg(const Mesh& m, const DofMap& dm_v, const DofMap& dm_p,
                   const CsrPattern& Pvv,      // nv x nv (速度-速度)
                   const CsrPattern& Pvp,      // nv x np (行=速度, 列=压力)
                   const std::vector<double>& ux,
                   const std::vector<double>& uy,
                   const std::vector<double>& pp,
                   double nu, FX fx, FY fy, double tau_scale,
                   std::vector<double>& val_Ssd,
                   std::vector<double>& val_Sxx, std::vector<double>& val_Sxy,
                   std::vector<double>& val_Syx, std::vector<double>& val_Syy,
                   std::vector<double>& val_DxT, std::vector<double>& val_DyT,
                   std::vector<double>& rsx, std::vector<double>& rsy,
                   int qdeg = 0)
{
    if (qdeg <= 0) qdeg = 2 * OrderV + 1;
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int ndv = TriElement<OrderV>::ndof;
    constexpr int ndp = TriElement<OrderP>::ndof;
    const std::size_t nt = m.n_tris();

    val_Ssd.assign(Pvv.col_idx.size(), 0.0);
    val_Sxx.assign(Pvv.col_idx.size(), 0.0);
    val_Sxy.assign(Pvv.col_idx.size(), 0.0);
    val_Syx.assign(Pvv.col_idx.size(), 0.0);
    val_Syy.assign(Pvv.col_idx.size(), 0.0);
    val_DxT.assign(Pvp.col_idx.size(), 0.0);
    val_DyT.assign(Pvp.col_idx.size(), 0.0);
    rsx.assign(dm_v.n_dof, 0.0);
    rsy.assign(dm_v.n_dof, 0.0);

    if (!(tau_scale > 0.0)) return;          // tau_scale = 0 -> 关闭 SUPG

    double   Jinv[4], area = 0.0;
    double   vx[3], vy[3];
    double   N[ndv], Gref[2 * ndv], dNdx[ndv], dNdy[ndv];
    double   Gpref[2 * ndp], dPdx[ndp], dPdy[ndp];
    double   Sg[ndv * ndv], Xg[ndv * ndv], Yg[ndv * ndv],
             Zg[ndv * ndv], Wg[ndv * ndv];
    double   Dxl[ndv * ndp], Dyl[ndv * ndp];
    double   rsxl[ndv], rsyl[ndv];
    uint32_t posvv[ndv * ndv], posvp[ndv * ndp];

    for (std::size_t e = 0; e < nt; ++e)
    {
        const uint32_t* dv = &dm_v.elem_dof[static_cast<std::size_t>(ndv) * e];
        const uint32_t* dp = &dm_p.elem_dof[static_cast<std::size_t>(ndp) * e];

        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        tri_element_jac(vx, vy, Jinv, area);

        // ---- 单元常数: 重心处的 u、流线长度 h、tau ----
        TriElement<OrderV>::shape(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, N);
        TriElement<OrderV>::grad_ref(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, Gref);
        tri_apply_jac<OrderV>(Jinv, Gref, dNdx, dNdy);
        double ucx = 0.0, ucy = 0.0;
        for (int a = 0; a < ndv; ++a)
        {
            const std::size_t ga = dv[a];
            ucx += ux[ga] * N[a];
            ucy += uy[ga] * N[a];
        }
        double ssum = 0.0;
        for (int a = 0; a < ndv; ++a)
            ssum += std::fabs(ucx * dNdx[a] + ucy * dNdy[a]);
        const double umag = std::sqrt(ucx * ucx + ucy * ucy);
        double hk = std::sqrt(2.0 * area);                  // 退化尺度
        if (umag > 1e-12 && ssum > 0.0) hk = 2.0 * umag / ssum;
        if (!(hk > 0.0)) hk = std::sqrt(2.0 * area);
        const double c1 = 2.0 * umag / hk;
        const double c2 = 4.0 * nu / (hk * hk);
        const double cd = c1 * c1 + c2 * c2;
        const double tau = (cd > 0.0) ? (tau_scale / std::sqrt(cd)) : 0.0;
        if (!(tau > 0.0)) continue;

        // ---- Δu_h (单元常数): 顶点梯度 + 重心坐标梯度 ----
        double gxx[3], gxy[3], gyx[3], gyy[3];
        for (int v = 0; v < 3; ++v)
        {
            const double l0 = (v == 0) ? 1.0 : 0.0;
            const double l1 = (v == 1) ? 1.0 : 0.0;
            const double l2 = (v == 2) ? 1.0 : 0.0;
            TriElement<OrderV>::grad_ref(l0, l1, l2, Gref);
            tri_apply_jac<OrderV>(Jinv, Gref, dNdx, dNdy);
            double ax = 0.0, ay = 0.0, bx = 0.0, by = 0.0;
            for (int a = 0; a < ndv; ++a)
            {
                const std::size_t ga = dv[a];
                ax += ux[ga] * dNdx[a];   ay += ux[ga] * dNdy[a];
                bx += uy[ga] * dNdx[a];   by += uy[ga] * dNdy[a];
            }
            gxx[v] = ax;  gxy[v] = ay;  gyx[v] = bx;  gyy[v] = by;
        }
        const double glx[3] = { -(Jinv[0] + Jinv[1]), Jinv[0], Jinv[1] };
        const double gly[3] = { -(Jinv[2] + Jinv[3]), Jinv[2], Jinv[3] };
        double lapux = 0.0, lapuy = 0.0;
        for (int v = 0; v < 3; ++v)
        {
            lapux += gxx[v] * glx[v] + gxy[v] * gly[v];
            lapuy += gyx[v] * glx[v] + gyy[v] * gly[v];
        }

        for (int i = 0; i < ndv * ndv; ++i)
        { Sg[i] = 0.0; Xg[i] = 0.0; Yg[i] = 0.0; Zg[i] = 0.0; Wg[i] = 0.0; }
        for (int i = 0; i < ndv * ndp; ++i) { Dxl[i] = 0.0; Dyl[i] = 0.0; }
        for (int i = 0; i < ndv; ++i)       { rsxl[i] = 0.0; rsyl[i] = 0.0; }

        for (std::size_t q = 0; q < Q.size(); ++q)
        {
            const double l0 = Q.l0[q], l1 = Q.l1[q], l2 = Q.l2[q];

            TriElement<OrderV>::shape(l0, l1, l2, N);
            TriElement<OrderV>::grad_ref(l0, l1, l2, Gref);
            tri_apply_jac<OrderV>(Jinv, Gref, dNdx, dNdy);
            TriElement<OrderP>::grad_ref(l0, l1, l2, Gpref);
            tri_apply_jac<OrderP>(Jinv, Gpref, dPdx, dPdy);

            double uhx = 0.0, uhy = 0.0;
            double duxdx = 0.0, duxdy = 0.0, duydx = 0.0, duydy = 0.0;
            for (int a = 0; a < ndv; ++a)
            {
                const std::size_t ga = dv[a];
                uhx   += ux[ga] * N[a];
                uhy   += uy[ga] * N[a];
                duxdx += ux[ga] * dNdx[a];
                duxdy += ux[ga] * dNdy[a];
                duydx += uy[ga] * dNdx[a];
                duydy += uy[ga] * dNdy[a];
            }
            double dpdx = 0.0, dpdy = 0.0;
            for (int b = 0; b < ndp; ++b)
            {
                const std::size_t gb = dp[b];
                dpdx += pp[gb] * dPdx[b];
                dpdy += pp[gb] * dPdy[b];
            }
            const double wx = vx[0] + (vx[1] - vx[0]) * l1 + (vx[2] - vx[0]) * l2;
            const double wy = vy[0] + (vy[1] - vy[0]) * l1 + (vy[2] - vy[0]) * l2;
            const double Rx = uhx * duxdx + uhy * duxdy + dpdx - nu * lapux - fx(wx, wy);
            const double Ry = uhx * duydx + uhy * duydy + dpdy - nu * lapuy - fy(wx, wy);

            const double wq = area * Q.w[q];
            for (int i = 0; i < ndv; ++i)
            {
                const double si = uhx * dNdx[i] + uhy * dNdy[i];   // (u·∇φ_i)
                const double wi = tau * wq * si;
                rsxl[i] += wi * Rx;
                rsyl[i] += wi * Ry;
                for (int j = 0; j < ndv; ++j)
                {
                    const std::size_t ij = static_cast<std::size_t>(i) * ndv + j;
                    const double sj = uhx * dNdx[j] + uhy * dNdy[j];
                    Sg[ij] += wi * sj;                 // 流线扩散 (对称半正定)
                    const double wN = wi * N[j];
                    Xg[ij] += wN * duxdx;              // x 动量行, 列 δu_x
                    Yg[ij] += wN * duxdy;              // x 动量行, 列 δu_y
                    Zg[ij] += wN * duydx;              // y 动量行, 列 δu_x
                    Wg[ij] += wN * duydy;              // y 动量行, 列 δu_y
                }
                for (int b = 0; b < ndp; ++b)
                {
                    const std::size_t ib = static_cast<std::size_t>(i) * ndp + b;
                    Dxl[ib] += wi * dPdx[b];
                    Dyl[ib] += wi * dPdy[b];
                }
            }
        }

        for (int i = 0; i < ndv; ++i)
            for (int j = 0; j < ndv; ++j)
                posvv[static_cast<std::size_t>(i) * ndv + j] =
                    static_cast<uint32_t>(pattern_index(Pvv, dv[i], dv[j]));
        for (int i = 0; i < ndv; ++i)
            for (int b = 0; b < ndp; ++b)
                posvp[static_cast<std::size_t>(i) * ndp + b] =
                    static_cast<uint32_t>(pattern_index(Pvp, dv[i], dp[b]));

        scatter_rect(val_Ssd, posvv, Sg, ndv, ndv);
        scatter_rect(val_Sxx, posvv, Xg, ndv, ndv);
        scatter_rect(val_Sxy, posvv, Yg, ndv, ndv);
        scatter_rect(val_Syx, posvv, Zg, ndv, ndv);
        scatter_rect(val_Syy, posvv, Wg, ndv, ndv);
        scatter_rect(val_DxT, posvp, Dxl, ndv, ndp);
        scatter_rect(val_DyT, posvp, Dyl, ndv, ndp);
        for (int i = 0; i < ndv; ++i)
        {
            rsx[dv[i]] += rsxl[i];
            rsy[dv[i]] += rsyl[i];
        }
    }
}

// ---------------------------------------------------------------------------
// 在任意物理点 (x, y) 上评估 Order 阶 Lagrange 场
//
//   定位方式是"遍历全部三角形 + 重心坐标判定": O(nt), 只在输出中心线剖面
//   这类"少量点"的场合用 (几十个点 * 几千单元, 一次几毫秒)。真要做大规模
//   插值/可视化再换成网格邻接搜索。
//
//   重心坐标: 记 (xi, eta) = (l1, l2), 则 [x;y] = [x0;y0] + J·[l1;l2],
//   所以 [l1;l2] = J^{-1}·[x-x0; y-y0]。tri_element_jac 给出的 Jinv 是
//   **梯度**变换矩阵 (J^{-1} 的转置排布), 对应关系是
//       l1 = Jinv[0]*dx + Jinv[2]*dy ;  l2 = Jinv[1]*dx + Jinv[3]*dy
//   (不是 Jinv[0]*dx + Jinv[1]*dy —— 那是梯度, 不是坐标)。
// ---------------------------------------------------------------------------
template <int Order>
bool eval_fe_at(const Mesh& m, const DofMap& dm, const std::vector<double>& u,
                double x, double y, double& out, double tol = 1e-10)
{
    constexpr int nd = TriElement<Order>::ndof;
    const std::size_t nt = m.n_tris();
    double vx[3], vy[3];

    for (std::size_t e = 0; e < nt; ++e)
    {
        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        double Jinv[4], area = 0.0;
        tri_element_jac(vx, vy, Jinv, area);

        const double dx = x - vx[0], dy = y - vy[0];
        const double l1 = Jinv[0] * dx + Jinv[2] * dy;
        const double l2 = Jinv[1] * dx + Jinv[3] * dy;
        const double l0 = 1.0 - l1 - l2;
        if (l0 < -tol || l1 < -tol || l2 < -tol) continue;

        double N[nd];
        TriElement<Order>::shape(l0, l1, l2, N);
        double s = 0.0;
        for (int a = 0; a < nd; ++a)
            s += u[dm.elem_dof[static_cast<std::size_t>(nd) * e + a]] * N[a];
        out = s;
        return true;
    }
    return false;
}

}   // namespace fem
