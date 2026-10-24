#pragma once
// ============================================================================
// 有限元装配
// ----------------------------------------------------------------------------
//   采用"两遍法"直接产出 CSR, 而不是常见的"收集三元组 (COO) 再排序去重":
//     第 1 遍  build_pattern()  只推导稀疏模式 -> row_ptr / col_idx
//     第 2 遍  assemble_*()     按模式累加数值
//   好处:
//     (1) 稀疏模式只算一次。非线性迭代 / 瞬态里 Jacobian 结构不变、只有数值变,
//         后续每一步都能直接复用 (这是后面做瞬态的前提)。
//     (2) 不产生 O(nnz * 单元重叠数) 的中间三元组, 省内存也省排序。
//     (3) 数值累加直接落在最终位置, 不需要归约重复项。
//
//   ---- 装配层的三个热点, 都已处理 ----
//
//   [1] 位置查找。原实现在每个求积点、对每对 (a,b) 都做一次 pattern_index
//       二分 —— P2 是 6 点 x 36 对 = 每单元 216 次二分。改成:
//         有预计算表 -> 直接查表; 没有 -> 每单元只算 nd*nd 次 (不是每求积点)。
//       非线性/瞬态会重复装配几十上百次, 建一次 ElemPos 表就能摊掉。
//
//   [2] 散射方式。原实现把 nd*nd 个贡献直接 += 到 val 的离散位置:
//       写地址运行时才算, 无法向量化, 而且每个求积点都走一遍全局内存。
//       改成: 先在栈上把单元矩阵 Ke[nd][nd] 累加完 (纯算术, 能进寄存器),
//       再一次散射 nd*nd 个值。顺带只算上三角 (b >= a) 再镜像 ——
//       运算量减半, 且保证矩阵严格对称 (对称形式下)。
//
//   [3] 雅可比。见 element.hpp: tri_element_jac 每单元一次, tri_apply_jac 每点一次。
//       另外质量矩阵与载荷向量**根本不需要梯度**, 原实现却跟着算了全套
//       dNdx/dNdy —— 这里直接去掉。
//
//   ---- 还没做的 (留给 M2) ----
//     * 并行装配: 需要按颜色分组 (相邻单元不共享自由度), 否则散射有竞争。
//       几千自由度时并行收益为负 (线程团队开销 > 计算量), 等 1e5 再加。
//     * build_pattern 的 pat 数组是 nt*nd*nd 个 pair, 百万单元 P2 ≈ 288 MB,
//       应改按行分桶 + 每行排序去重。
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fem/dofmap.hpp"
#include "fem/element.hpp"
#include "fem/mesh.hpp"
#include "fem/quadrature.hpp"

namespace fem
{

// 稀疏模式。布局与 sparse::SMatrix 的 CSR 完全一致, 可以直接交给它。
struct CsrPattern
{
    uint32_t             n = 0;
    std::vector<uint32_t> row_ptr;    // n + 1
    std::vector<uint32_t> col_idx;    // nnz
};

inline CsrPattern build_pattern(const Mesh& m, const DofMap& dm)
{
    const std::size_t nt = m.n_tris();
    const std::size_t nd = static_cast<std::size_t>(dm.ndof_per_elem());

    // 单元矩阵是稠密的 nd x nd, 所以模式的候选就是所有 (ga, gb) 对。
    std::vector<std::pair<uint32_t, uint32_t>> pat;
    pat.reserve(nt * nd * nd);
    for (std::size_t e = 0; e < nt; ++e)
        for (std::size_t a = 0; a < nd; ++a)
        {
            const uint32_t ga = dm.elem_dof[nd * e + a];
            for (std::size_t b = 0; b < nd; ++b)
                pat.emplace_back(ga, dm.elem_dof[nd * e + b]);
        }
    std::sort(pat.begin(), pat.end());
    pat.erase(std::unique(pat.begin(), pat.end()), pat.end());

    CsrPattern P;
    P.n = static_cast<uint32_t>(dm.n_dof);
    P.row_ptr.assign(static_cast<std::size_t>(P.n) + 1, 0u);
    for (const auto& pr : pat) P.row_ptr[static_cast<std::size_t>(pr.first) + 1]++;
    for (std::size_t i = 0; i < P.n; ++i) P.row_ptr[i + 1] += P.row_ptr[i];
    P.col_idx.resize(pat.size());
    for (std::size_t k = 0; k < pat.size(); ++k) P.col_idx[k] = pat[k].second;
    return P;
}

// 定位 (row, col) 在值数组中的下标。列号在每行内是升序的, 可以二分。
inline std::size_t pattern_index(const CsrPattern& P, uint32_t row, uint32_t col)
{
    const std::size_t lo = P.row_ptr[row];
    const std::size_t hi = P.row_ptr[static_cast<std::size_t>(row) + 1];
    auto it = std::lower_bound(P.col_idx.begin() + static_cast<std::ptrdiff_t>(lo),
                               P.col_idx.begin() + static_cast<std::ptrdiff_t>(hi), col);
    if (it == P.col_idx.begin() + static_cast<std::ptrdiff_t>(hi) || *it != col)
        throw std::runtime_error("装配: 稀疏模式里找不到该位置 (模式与单元不匹配?)");
    return static_cast<std::size_t>(it - P.col_idx.begin());
}

// ---------------------------------------------------------------------------
// 单元 -> CSR 位置 的预计算表
//
//   存 p[(e*nd + a)*nd + b] = val 数组里 (dof(e,a), dof(e,b)) 的下标。
//   一次性付出 nt*nd*nd 次二分 + nt*nd*nd 个 uint32 (P2 百万单元约 144 MB),
//   之后每次装配都只是查表。装配次数 >= 2 就回本 —— 非线性求解和瞬态推进
//   每步都要重装 Jacobian, 这张表是那时候的主要收益来源。
//
//   不想要这张表也可以: 给 assemble_* 传 nullptr, 就退化成"每单元算 nd*nd 次
//   二分" (而不是旧实现的"每求积点算 nd*nd 次"), 仍然快好几倍, 且零额外内存。
// ---------------------------------------------------------------------------
struct ElemPos
{
    int                   nd = 0;
    std::vector<uint32_t> p;      // nt * nd * nd
};

inline ElemPos build_elem_pos(const Mesh& m, const DofMap& dm, const CsrPattern& P)
{
    const std::size_t nt = m.n_tris();
    const std::size_t nd = static_cast<std::size_t>(dm.ndof_per_elem());

    if (P.col_idx.size() >= (static_cast<std::size_t>(1) << 32))
        throw std::runtime_error("装配: nnz 超过 uint32 上限, ElemPos 需要改用 64 位下标");

    ElemPos E;
    E.nd = static_cast<int>(nd);
    E.p.resize(nt * nd * nd);
    for (std::size_t e = 0; e < nt; ++e)
        for (std::size_t a = 0; a < nd; ++a)
        {
            const uint32_t ga = dm.elem_dof[nd * e + a];
            for (std::size_t b = 0; b < nd; ++b)
                E.p[(e * nd + a) * nd + b] =
                    static_cast<uint32_t>(pattern_index(P, ga, dm.elem_dof[nd * e + b]));
        }
    return E;
}

// 取某个单元的 nd*nd 个散射位置
template <int nd>
inline void fill_elem_pos(const CsrPattern& P, const uint32_t* dof,
                          const std::vector<uint32_t>* ep, std::size_t e,
                          uint32_t* pos)
{
    if (ep)
    {
        const std::size_t base = e * static_cast<std::size_t>(nd) * nd;
        for (int i = 0; i < nd * nd; ++i) pos[i] = (*ep)[base + static_cast<std::size_t>(i)];
    }
    else
    {
        for (int a = 0; a < nd; ++a)
            for (int b = 0; b < nd; ++b)
                pos[a * nd + b] =
                    static_cast<uint32_t>(pattern_index(P, dof[a], dof[b]));
    }
}

// 对称散射: 只存了上三角的 Ke, 这里把 (a,b) 与 (b,a) 两处都加上
template <int nd>
inline void scatter_sym(std::vector<double>& val, const uint32_t* pos, const double* Ke)
{
    for (int a = 0; a < nd; ++a)
        for (int b = a; b < nd; ++b)
        {
            const double v = Ke[a * nd + b];
            val[pos[a * nd + b]] += v;
            if (b != a) val[pos[b * nd + a]] += v;
        }
}

// ---------------------------------------------------------------------------
// 刚度矩阵:  A_ij = ∫ κ ∇φ_j · ∇φ_i
//   kappa(x, y) 返回扩散系数。常数 κ 时被积函数次数 = 2*(Order-1),
//   默认求积精度取 2*Order (P1 -> 2, P2 -> 4), 对变系数也够用。
// ---------------------------------------------------------------------------
template <int Order, class Kappa>
void assemble_stiffness(const Mesh& m, const DofMap& dm, const CsrPattern& P,
                        Kappa kappa, std::vector<double>& val, int qdeg = 0,
                        const std::vector<uint32_t>* elem_pos = nullptr)
{
    if (qdeg <= 0) qdeg = 2 * Order;
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int nd = TriElement<Order>::ndof;
    const std::size_t nt = m.n_tris();

    if (elem_pos && elem_pos->size() != nt * static_cast<std::size_t>(nd) * nd)
        throw std::runtime_error("装配: elem_pos 表的规模与 (单元数, ndof/单元) 不匹配");

    val.assign(P.col_idx.size(), 0.0);

    double   Jinv[4];
    double   area = 0.0;
    double   dNdx[nd], dNdy[nd];
    double   Ke[nd * nd];
    uint32_t pos[nd * nd];
    double   vx[3], vy[3];

    for (std::size_t e = 0; e < nt; ++e)
    {
        const uint32_t* dof = &dm.elem_dof[static_cast<std::size_t>(nd) * e];

        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        tri_element_jac(vx, vy, Jinv, area);

        for (int i = 0; i < nd * nd; ++i) Ke[i] = 0.0;
        fill_elem_pos<nd>(P, dof, elem_pos, e, pos);

        // P1 的参考梯度是常数 -> 物理梯度每单元只算一次, 与求积点数无关
        if constexpr (Order == 1)
        {
            double Gref[2 * nd];
            TriElement<Order>::grad_ref(0.0, 0.0, 0.0, Gref);
            tri_apply_jac<Order>(Jinv, Gref, dNdx, dNdy);
        }

        for (std::size_t q = 0; q < Q.size(); ++q)
        {
            const double l0 = Q.l0[q], l1 = Q.l1[q], l2 = Q.l2[q];

            if constexpr (Order != 1)
            {
                double Gref[2 * nd];
                TriElement<Order>::grad_ref(l0, l1, l2, Gref);
                tri_apply_jac<Order>(Jinv, Gref, dNdx, dNdy);
            }

            const double xq = l0 * vx[0] + l1 * vx[1] + l2 * vx[2];
            const double yq = l0 * vy[0] + l1 * vy[1] + l2 * vy[2];
            const double wq = area * Q.w[q] * kappa(xq, yq);

            // 只算上三角: 对称形式下 K_ab == K_ba, 运算量减半
            for (int a = 0; a < nd; ++a)
                for (int b = a; b < nd; ++b)
                    Ke[a * nd + b] += wq * (dNdx[a] * dNdx[b] + dNdy[a] * dNdy[b]);
        }

        scatter_sym<nd>(val, pos, Ke);
    }
}

// ---------------------------------------------------------------------------
// 质量矩阵:  M_ij = ∫ ρ φ_j φ_i
//   L2 误差范数 ‖e‖² = eᵀ M e 要用它, 所以它不是可选的。
//   (质量矩阵不需要梯度 —— 旧实现跟着 tri_physical_grad 算了全套 dNdx/dNdy,
//    纯属浪费, 这里只要面积。)
// ---------------------------------------------------------------------------
template <int Order, class Rho>
void assemble_mass(const Mesh& m, const DofMap& dm, const CsrPattern& P,
                   Rho rho, std::vector<double>& val, int qdeg = 0,
                   const std::vector<uint32_t>* elem_pos = nullptr)
{
    if (qdeg <= 0) qdeg = 2 * Order;
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int nd = TriElement<Order>::ndof;
    const std::size_t nt = m.n_tris();

    if (elem_pos && elem_pos->size() != nt * static_cast<std::size_t>(nd) * nd)
        throw std::runtime_error("装配: elem_pos 表的规模与 (单元数, ndof/单元) 不匹配");

    val.assign(P.col_idx.size(), 0.0);

    double   Jinv[4];
    double   area = 0.0;
    double   N[nd];
    double   Ke[nd * nd];
    uint32_t pos[nd * nd];
    double   vx[3], vy[3];

    for (std::size_t e = 0; e < nt; ++e)
    {
        const uint32_t* dof = &dm.elem_dof[static_cast<std::size_t>(nd) * e];

        for (int i = 0; i < 3; ++i)
        {
            const uint32_t g = m.tris[3 * e + i];
            vx[i] = m.coords[2 * g + 0];
            vy[i] = m.coords[2 * g + 1];
        }
        // 只要面积, 不需要梯度 -> 拿到的 Jinv 这里用不上, 但 tri_element_jac
        //   同时负责定向检查, 所以还是走它。
        tri_element_jac(vx, vy, Jinv, area);

        for (int i = 0; i < nd * nd; ++i) Ke[i] = 0.0;
        fill_elem_pos<nd>(P, dof, elem_pos, e, pos);

        for (std::size_t q = 0; q < Q.size(); ++q)
        {
            const double l0 = Q.l0[q], l1 = Q.l1[q], l2 = Q.l2[q];
            TriElement<Order>::shape(l0, l1, l2, N);
            const double xq = l0 * vx[0] + l1 * vx[1] + l2 * vx[2];
            const double yq = l0 * vy[0] + l1 * vy[1] + l2 * vy[2];
            const double wq = area * Q.w[q] * rho(xq, yq);

            for (int a = 0; a < nd; ++a)
                for (int b = a; b < nd; ++b)
                    Ke[a * nd + b] += wq * N[a] * N[b];
        }

        scatter_sym<nd>(val, pos, Ke);
    }
}

// ---------------------------------------------------------------------------
// 载荷向量:  b_i = ∫ f φ_i
//   默认求积精度比刚度高一档 (2*Order+1): f 一般不是多项式, 载荷项的求积误差
//   会直接进入相容性误差, 是 MMS 收敛阶测试里最容易把阶数"吃掉"的一项。
// ---------------------------------------------------------------------------
template <int Order, class F>
void assemble_load(const Mesh& m, const DofMap& dm, F f,
                   std::vector<double>& rhs, int qdeg = 0)
{
    if (qdeg <= 0) qdeg = 2 * Order + 1;
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int nd = TriElement<Order>::ndof;
    const std::size_t nt = m.n_tris();

    rhs.assign(dm.n_dof, 0.0);

    double Jinv[4];
    double area = 0.0;
    double N[nd];
    double vx[3], vy[3];

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
            const double wq = area * Q.w[q] * f(xq, yq);

            for (int a = 0; a < nd; ++a)
                rhs[dm.elem_dof[static_cast<std::size_t>(nd) * e + a]] += wq * N[a];
        }
    }
}

// ---------------------------------------------------------------------------
// 真实 L2 / H1 误差 (按求积直接积分, 与装配用同一套单元与求积)
//
//   【为什么不用 nodal 差值的 eᵀMe / eᵀAe】
//     e_i = u_i - u(x_i) 是"节点差值", 它张出的函数是 u_h - I_h u
//     (FE 解与插值之差), 而不是 u - u_h。前者是经典的超收敛量: 在结构网格上
//     它比真实误差高一阶 —— 实测 P1 的 eᵀAe 给出 1.99 阶、P2 的 eᵀMe 给出
//     3.9 阶, 看着"比理论还好", 实际是把离散误差的真实阶数掩盖掉了。
//     所以这里逐单元求积算
//        ‖u - u_h‖_{L2}²  = ∫ (u - Σ u_i φ_i)²
//        |u - u_h|_{H1}²  = ∫ |∇u - Σ u_i ∇φ_i|²
//     这两个才是对照理论阶 (P1: L2 2 阶 / H1 1 阶; P2: 3 / 2) 的量。
//
//   grad_u_ex(x, y, gx, gy) 用出参返回真解梯度。
// ---------------------------------------------------------------------------
template <int Order, class UExact, class GExact>
std::pair<double, double> compute_errors(const Mesh& m, const DofMap& dm,
                                         const std::vector<double>& u,
                                         UExact u_ex, GExact grad_u_ex,
                                         int qdeg = 5)
{
    const TriQuad Q = tri_quadrature(qdeg);
    constexpr int nd = TriElement<Order>::ndof;
    const std::size_t nt = m.n_tris();

    double e_l2 = 0.0, e_h1 = 0.0;

    double Jinv[4];
    double area = 0.0;
    double N[nd], dNdx[nd], dNdy[nd];
    double vx[3], vy[3];

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

            double Gref[2 * nd];
            TriElement<Order>::grad_ref(l0, l1, l2, Gref);
            tri_apply_jac<Order>(Jinv, Gref, dNdx, dNdy);
            TriElement<Order>::shape(l0, l1, l2, N);

            const double xq = l0 * vx[0] + l1 * vx[1] + l2 * vx[2];
            const double yq = l0 * vy[0] + l1 * vy[1] + l2 * vy[2];

            double uh = 0.0, ghx = 0.0, ghy = 0.0;
            for (int a = 0; a < nd; ++a)
            {
                const double ua = u[dm.elem_dof[static_cast<std::size_t>(nd) * e + a]];
                uh  += ua * N[a];
                ghx += ua * dNdx[a];
                ghy += ua * dNdy[a];
            }

            double egx = 0.0, egy = 0.0;
            grad_u_ex(xq, yq, egx, egy);
            const double du  = uh - u_ex(xq, yq);
            const double wq  = area * Q.w[q];
            e_l2 += wq * du * du;
            e_h1 += wq * ((ghx - egx) * (ghx - egx) + (ghy - egy) * (ghy - egy));
        }
    }
    return { std::sqrt(e_l2), std::sqrt(e_h1) };
}

}   // namespace fem
