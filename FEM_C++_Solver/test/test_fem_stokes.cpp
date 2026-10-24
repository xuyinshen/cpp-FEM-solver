// ============================================================================
// M1: Taylor-Hood (P2/P1) 稳态 Stokes —— 端到端验证
// ----------------------------------------------------------------------------
//   算例 A  【Poiseuille, 精确解检验】
//     Ω = [0,1]², nu = 1
//       u_x = y(1-y),  u_y = 0,  p = -2x,  f = (0,0)
//     验证: -Δu + ∇p = (2,0) + (-2,0) = 0 ✓   ∇·u = 0 ✓
//     边界: y=0/1 无滑移 (u=0); x=0/1 给抛物线入口/出口剖面。
//
//     【为什么这个算例几乎能"一击致命"地验出装配 bug】
//       u_x = y - y² 是二次多项式 ∈ P2 速度空间,
//       p = -2x     是一次多项式 ∈ P1 压力空间,
//       所以**解析解本身就落在离散空间里**, Galerkin 正交性意味着离散解
//       应当逐位等于插值 —— 误差应该只有舍入量级 (~1e-14)。
//       只要 A / B / Bᵀ 的装配、符号、Dirichlet 消元、压力 pin 里任何一处
//       差一个符号或错一个自由度, 误差立刻从 1e-14 掉到 1e-2 量级。
//       相比之下"看着像流场"的算例根本做不到这种敏感度。
//
//   算例 B  【MMS, 收敛阶检验】
//     用流函数构造一个无散、齐次边界的制造解, 检查 Taylor-Hood 的最优阶:
//       速度 L2 = 3 (P2), H1 = 2;   压力 L2 = 2 (P1)
//       ψ = sin²(πx) sin²(πy),  u = (∂ψ/∂y, -∂ψ/∂x)  => ∇·u ≡ 0, u|∂Ω = 0
//       p = cos(πx) cos(πy) (故意取非零, 顺便验压力右端项)
//       f = -nu Δu + ∇p
//
//   用法:
//     ./bin/stokes_app                      默认 Poiseuille, nx = 8,16,32
//     ./bin/stokes_app --case mms           跑 MMS 收敛阶
//     ./bin/stokes_app --nx 16 32 64 --restart 100
// ============================================================================
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "fem/assembler.hpp"
#include "fem/boundary.hpp"
#include "fem/dofmap.hpp"
#include "fem/mesh.hpp"
#include "fem/stokes.hpp"
#include "math/array.hpp"
#include "math/blockMatrix.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

// ===========================================================================
// 算例 A: Poiseuille
// ===========================================================================
struct Poiseuille
{
    static double ux(double /*x*/, double y)        { return y * (1.0 - y); }
    static double uy(double /*x*/, double /*y*/)    { return 0.0; }
    static double p (double x, double /*y*/)        { return -2.0 * x; }
    static void gux(double /*x*/, double y, double& gx, double& gy)
    {
        gx = 0.0;  gy = 1.0 - 2.0 * y;
    }
    static void guy(double /*x*/, double /*y*/, double& gx, double& gy)
    {
        gx = 0.0;  gy = 0.0;
    }
    static double fx(double /*x*/, double /*y*/)    { return 0.0; }
    static double fy(double /*x*/, double /*y*/)    { return 0.0; }
    static void gp(double /*x*/, double /*y*/, double& gx, double& gy)
    {
        gx = -2.0;  gy = 0.0;                        // p = -2x
    }
    static double nu()                              { return 1.0; }
};

// ===========================================================================
// 算例 B: MMS (流函数构造, 齐次边界)
//   ψ = sin²(πx) sin²(πy)
//   u_x = ∂ψ/∂y = π sin²(πx) sin(2πy)
//   u_y = -∂ψ/∂x = -π sin(2πx) sin²(πy)
// ===========================================================================
struct MMS
{
    static double ux(double x, double y)
    {
        const double s = std::sin(kPi * x);
        return kPi * s * s * std::sin(2.0 * kPi * y);
    }
    static double uy(double x, double y)
    {
        const double s = std::sin(kPi * y);
        return -kPi * std::sin(2.0 * kPi * x) * s * s;
    }
    static double p(double x, double y)
    {
        return std::cos(kPi * x) * std::cos(kPi * y);
    }
    // ∇u_x
    static void gux(double x, double y, double& gx, double& gy)
    {
        const double s = std::sin(kPi * x);
        gx = kPi * kPi * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
        gy = 2.0 * kPi * kPi * s * s * std::cos(2.0 * kPi * y);
    }
    // ∇u_y
    static void guy(double x, double y, double& gx, double& gy)
    {
        const double s = std::sin(kPi * y);
        gx = -2.0 * kPi * kPi * std::cos(2.0 * kPi * x) * s * s;
        gy = -kPi * kPi * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
    }
    // Δu_x = 2π³cos(2πx)sin(2πy) - 4π³sin²(πx)sin(2πy)
    static double lap_ux(double x, double y)
    {
        const double s = std::sin(kPi * x);
        return 2.0 * kPi * kPi * kPi * std::cos(2.0 * kPi * x) * std::sin(2.0 * kPi * y)
             - 4.0 * kPi * kPi * kPi * s * s * std::sin(2.0 * kPi * y);
    }
    // Δu_y = 4π³sin(2πx)sin²(πy) - 2π³sin(2πx)cos(2πy)
    static double lap_uy(double x, double y)
    {
        const double s = std::sin(kPi * y);
        return 4.0 * kPi * kPi * kPi * std::sin(2.0 * kPi * x) * s * s
             - 2.0 * kPi * kPi * kPi * std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y);
    }
    static void gp(double x, double y, double& gx, double& gy)
    {
        gx = -kPi * std::sin(kPi * x) * std::cos(kPi * y);
        gy = -kPi * std::cos(kPi * x) * std::sin(kPi * y);
    }
    static double fx(double x, double y)
    {
        return -nu() * lap_ux(x, y) - kPi * std::sin(kPi * x) * std::cos(kPi * y);
    }
    static double fy(double x, double y)
    {
        return -nu() * lap_uy(x, y) - kPi * std::cos(kPi * x) * std::sin(kPi * y);
    }
    static double nu() { return 1.0; }
};

// ===========================================================================
// 一次完整的 Stokes 求解
// ===========================================================================
struct Result
{
    std::size_t ndof = 0, nv = 0, np = 0;
    std::size_t nnz  = 0;
    int         iters = 0;
    double      t_asm = 0.0, t_pre = 0.0, t_it = 0.0;
    double      e_u_l2 = 0.0, e_u_h1 = 0.0, e_p_l2 = 0.0;
    double      e_p_l2_mf = 0.0;      // 去均值压力误差 ‖e - mean(e)‖_L2
    double      p_mean = 0.0;         // mean(e) —— 压力整体漂移了多少
    double      div_l2 = 0.0;         // ‖∇·u_h‖_L2, 应该 -> 0
    double      res    = 0.0;         // 真实相对残差 ‖b-Ax‖/‖b‖
    bool        converged = false;    // GMRES 是否真的收敛 (iters < max_iter)
};

// 速度块预条件的配置 (见 fem::VelPreKind)
struct Preconf
{
    int    kind       = 0;      // 0 = AMG+Jacobi, 1 = AMG+Chebyshev, 2 = BlockAMG 聚合
    int    cheb_deg   = 3;      // chebscan 实测甜点: deg=3, ratio=30
    double cheb_ratio = 30.0;
    int    blk_agg    = 8;
    int    blk_smooth = 2;

    std::string name() const
    {
        if (kind == 1)
        {
            std::string t = "标量 AMG on K + Chebyshev 光滑 (deg=";
            t += std::to_string(cheb_deg);
            t += ", alpha=beta/";
            t += std::to_string(static_cast<int>(cheb_ratio));
            t += ")";
            return t;
        }
        if (kind == 2)
        {
            std::string t = "BlockAMG 光滑聚合 on BlockMatrix<double,2> (agg<=";
            t += std::to_string(blk_agg);
            t += ", nu=";
            t += std::to_string(blk_smooth);
            t += ")";
            return t;
        }
        return "标量 AMG on K + 阻尼 Jacobi 光滑";
    }
};

template <class Case>
Result run_case(uint32_t nx, double drop, uint32_t cap, bool amg_verbose,
                uint32_t coarse_thresh, int restart, double tol, int max_iter,
                const Preconf& pc)
{
    const double nu = Case::nu();
    const auto T0 = std::chrono::steady_clock::now();

    // ---- 网格与自由度 ----
    const fem::Mesh   m    = fem::make_unit_square_mesh(nx, nx);
    const fem::DofMap dm_v = fem::make_dofmap(m, 2);      // 速度 P2
    const fem::DofMap dm_p = fem::make_dofmap(m, 1);      // 压力 P1

    const std::size_t nv = dm_v.n_dof;
    const std::size_t np = dm_p.n_dof;

    auto S = std::make_shared<fem::TaylorHood>();
    S->nv = nv;  S->np = np;  S->p_pin = 0;               // 钉住 (0,0) 处的压力

    // ---- 速度刚度 K = nu * ∫∇φ·∇φ (P2 标量) ----
    fem::CsrPattern pat_v = fem::build_pattern(m, dm_v);
    std::vector<double> val_K;
    fem::assemble_stiffness<2>(m, dm_v, pat_v, [nu](double, double) { return nu; }, val_K);
    S->K = sparse::SMatrix(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                           pat_v.row_ptr, pat_v.col_idx, val_K);

    // ---- 负散度 Bx / By (np x nv, 共享矩形模式) ----
    fem::CsrPattern pat_vp = fem::build_pattern_rect(m, dm_p, dm_v);
    std::vector<double> val_Bx, val_By;
    fem::assemble_divergence<2, 1>(m, dm_v, dm_p, pat_vp, val_Bx, val_By);
    S->Bx = sparse::SMatrix(static_cast<uint32_t>(np), static_cast<uint32_t>(nv),
                            pat_vp.row_ptr, pat_vp.col_idx, val_Bx);
    S->By = sparse::SMatrix(static_cast<uint32_t>(np), static_cast<uint32_t>(nv),
                            pat_vp.row_ptr, pat_vp.col_idx, val_By);
    S->BxT = S->Bx.transpose();
    S->ByT = S->By.transpose();

    // ---- 压力质量 Mp (给 Schur 预条件用) + 行和 lumping ----
    fem::CsrPattern pat_p = fem::build_pattern(m, dm_p);
    std::vector<double> val_Mp;
    fem::assemble_mass<1>(m, dm_p, pat_p, [](double, double) { return 1.0; }, val_Mp);
    S->Mp = sparse::SMatrix(static_cast<uint32_t>(np), static_cast<uint32_t>(np),
                            pat_p.row_ptr, pat_p.col_idx, val_Mp);
    S->Mp_lumped.assign(np, 0.0);
    {
        const auto& rp = S->Mp.row_ptr();
        const auto& vv = S->Mp.values();
        for (std::size_t k = 0; k < np; ++k)
        {
            double s = 0.0;
            for (uint32_t q = rp[k]; q < rp[k + 1]; ++q) s += vv[q];
            S->Mp_lumped[k] = s;
        }
    }

    // ---- 展平鞍点系统 + 右端项 ----
    S->Kfull = fem::build_saddle_matrix(S->K, S->Bx, S->By, S->BxT, S->ByT, S->p_pin);

    std::vector<double> fvx, fvy;
    fem::assemble_load<2>(m, dm_v, [](double x, double y) { return Case::fx(x, y); }, fvx);
    fem::assemble_load<2>(m, dm_v, [](double x, double y) { return Case::fy(x, y); }, fvy);
    S->rhs.assign(2u * nv + np, 0.0);
    for (std::size_t i = 0; i < nv; ++i)
    {
        S->rhs[2u * i + 0u] = fvx[i];
        S->rhs[2u * i + 1u] = fvy[i];
    }

    // ---- 预条件器专用的 Kd: K 先施加一次速度 Dirichlet ----
    //   纯 Neumann 的 K 有常数零空间, AMG 不能直接用 (见 stokes.hpp 的说明)。
    {
        const auto bd_v   = fem::boundary_dofs(m, dm_v);
        std::vector<double> val_Kd = S->K.values();
        std::vector<double> dummy_rhs(nv, 0.0);
        std::vector<double> gzero(bd_v.size(), 0.0);      // 只影响 rhs, 这里丢弃
        fem::apply_dirichlet(static_cast<uint32_t>(nv),
                             S->K.row_ptr(), S->K.col_idx(), val_Kd,
                             dummy_rhs, bd_v, gzero);
        S->Kd = sparse::SMatrix(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                S->K.row_ptr(), S->K.col_idx(), val_Kd);
    }

    // ---- Dirichlet: 速度全边界 + 压力 pin 一点 ----
    {
        const auto bd_v = fem::boundary_dofs(m, dm_v);
        const auto c_v  = fem::dof_coords(m, dm_v);
        const auto c_p  = fem::dof_coords(m, dm_p);

        std::vector<uint32_t> fixed;
        std::vector<double>   gval;
        fixed.reserve(2u * bd_v.size() + 1u);
        gval.reserve(2u * bd_v.size() + 1u);
        for (const uint32_t d : bd_v)
        {
            const double x = c_v[d].first, y = c_v[d].second;
            fixed.push_back(static_cast<uint32_t>(2u * d + 0u));
            gval.push_back(Case::ux(x, y));
            fixed.push_back(static_cast<uint32_t>(2u * d + 1u));
            gval.push_back(Case::uy(x, y));
        }
        fixed.push_back(static_cast<uint32_t>(2u * nv + S->p_pin));
        gval.push_back(Case::p(c_p[S->p_pin].first, c_p[S->p_pin].second));

        auto val = S->Kfull.values();          // 拷贝一份改, 再写回
        fem::apply_dirichlet(static_cast<uint32_t>(2u * nv + np),
                             S->Kfull.row_ptr(), S->Kfull.col_idx(), val, S->rhs,
                             fixed, gval);
        S->Kfull = sparse::SMatrix(S->Kfull.rows(), S->Kfull.cols(),
                                   S->Kfull.row_ptr(), S->Kfull.col_idx(), val);
    }

    const auto T1 = std::chrono::steady_clock::now();     // 装配结束

    // ---- 预条件: 块 Schur (速度块走标量 AMG, Schur 走 nu·Mp⁻¹) ----
    auto A_pre = fem::make_velocity_preconditioner(
        S->Kd, drop, cap, amg_verbose, coarse_thresh,
        pc.kind, pc.cheb_deg, pc.cheb_ratio, pc.blk_agg, pc.blk_smooth);
    auto S_pre = fem::make_pressure_schur_preconditioner(S->Mp_lumped, nu, S->p_pin);
    auto sys   = fem::make_saddle_point_system(S);
    auto pre   = block::make_schur_preconditioner(sys, A_pre, S_pre);

    const auto T2 = std::chrono::steady_clock::now();     // 预条件构造结束

    // ---- 求解 ----
    const auto [sol, iters] = sparse::SMatrix::gmres(
        S->Kfull, S->rhs, pre, restart, tol, max_iter);
    const auto T3 = std::chrono::steady_clock::now();         // 迭代结束

    // ---- 真实相对残差 ----
    //   没有这一列, "误差 6.6e-5" 无法区分是装配错还是 GMRES 没解完 ——
    //   这两者的排查方向完全相反, 混在一起会浪费大量时间。
    const bool gmres_ok = (iters < max_iter);
    double res = 0.0;
    {
        std::vector<double> Ax(S->Kfull.rows(), 0.0);
        sparse::SMatrix::spmv_into(S->Kfull, sol, Ax);
        double nb = 0.0, nr = 0.0;
        for (std::size_t i = 0; i < S->rhs.size(); ++i)
        {
            nb += S->rhs[i] * S->rhs[i];
            const double d = S->rhs[i] - Ax[i];
            nr += d * d;
        }
        res = (nb > 0.0) ? std::sqrt(nr / nb) : std::sqrt(nr);
    }
    auto ms = [](std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // ---- 拆出速度 / 压力, 算误差 ----
    std::vector<double> ux(nv), uy(nv);
    for (std::size_t i = 0; i < nv; ++i)
    {
        ux[i] = sol[2u * i + 0u];
        uy[i] = sol[2u * i + 1u];
    }
    const std::vector<double> pp = fem::extract_pressure(sol, nv);

    const auto eu = fem::compute_errors_vec2<2>(
        m, dm_v, ux, uy,
        [](double x, double y) { return Case::ux(x, y); },
        [](double x, double y, double& gx, double& gy) { Case::gux(x, y, gx, gy); },
        [](double x, double y) { return Case::uy(x, y); },
        [](double x, double y, double& gx, double& gy) { Case::guy(x, y, gx, gy); });

    const auto ep = fem::compute_errors<1>(
        m, dm_p, pp,
        [](double x, double y) { return Case::p(x, y); },
        [](double x, double y, double& gx, double& gy) { Case::gp(x, y, gx, gy); });

    // 去均值压力误差。压力 pin 引入的近零方向是**规范自由度** (Bᵀ·const = 0,
    //   不影响速度场), GMRES 最小化残差时在这一方向上"看不出"误差 —— 于是
    //   p-L2 可能很大而 p-L2* 很小。两者对比是判"物理场对不对"的关键。
    double p_mean = 0.0;
    const double ep_mf = fem::compute_error_mean_free<1>(
        m, dm_p, pp,
        [](double x, double y) { return Case::p(x, y); }, 5, &p_mean);

    // 离散散度 ‖∇·u_h‖_L2: 直接算 ∫(∂u_x/∂x + ∂u_y/∂y)²
    double div2 = 0.0;
    {
        const auto Q = fem::tri_quadrature(4);
        for (std::size_t e = 0; e < m.n_tris(); ++e)
        {
            double vx[3], vy[3];
            for (int i = 0; i < 3; ++i)
            {
                const uint32_t g = m.tris[3 * e + i];
                vx[i] = m.coords[2 * g + 0];  vy[i] = m.coords[2 * g + 1];
            }
            double Jinv[4], area = 0.0;
            fem::tri_element_jac(vx, vy, Jinv, area);
            for (std::size_t q = 0; q < Q.size(); ++q)
            {
                double dNdx[6], dNdy[6];
                fem::tri_physical_grad<2>(vx, vy, Q.l0[q], Q.l1[q], Q.l2[q], dNdx, dNdy, area);
                double gxx = 0.0, gyy = 0.0;
                for (int a = 0; a < 6; ++a)
                {
                    const std::size_t gi = dm_v.elem_dof[6 * e + a];
                    gxx += ux[gi] * dNdx[a];
                    gyy += uy[gi] * dNdy[a];
                }
                div2 += area * Q.w[q] * (gxx + gyy) * (gxx + gyy);
            }
        }
    }

    Result R;
    R.t_asm = ms(T0, T1);  R.t_pre = ms(T1, T2);  R.t_it = ms(T2, T3);
    R.ndof = 2u * nv + np;  R.nv = nv;  R.np = np;
    R.nnz  = S->Kfull.nnz();
    R.iters = iters;
    R.e_u_l2 = eu.first;  R.e_u_h1 = eu.second;  R.e_p_l2 = ep.first;
    R.e_p_l2_mf = ep_mf;  R.p_mean = p_mean;
    R.div_l2 = std::sqrt(div2);
    R.res    = res;  R.converged = gmres_ok;
    return R;
}

}   // namespace

int main(int argc, char** argv)
{
    std::string           casename = "poiseuille";
    std::vector<uint32_t> nxs { 8u, 16u, 32u };
    double   drop     = 0.02;
    uint32_t cap      = 60;
    uint32_t coarse   = 300;
    // 重启步数: 100 会在 nx>=64 上伪造 h 相关性 (实测 nx=64 在 restart=100 下
    //   477 次、restart=300 下 176 次 —— 差的那 300 次几乎全是重启惩罚)。
    //   200 已覆盖实测的所有收敛步数, 且不至于让 Krylov 基的存储翻倍。
    int      restart  = 200;
    double   tol      = 1e-12; // Poiseuille 期望误差 ~1e-14, 容差必须比它更紧
    int      max_iter = 2000;
    bool     amg_verbose = false;
    Preconf  pc;                 // 速度块预条件: 默认 AMG + 阻尼 Jacobi

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&](void) -> std::string
        {
            if (i + 1 >= argc) { std::cerr << "参数 " << a << " 缺少取值\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--case")          casename = next();
        else if (a == "--nx")
        {
            // 同时接受 "--nx 8 16 32" 与 "--nx 8,16,32"。
            //   后者写进了 FEMRUN/STOKESRUN 的示例。若只按空格切,
            //   std::stoul("8,16,32") 会静默取前缀 -> 只跑 nx=8 一档,
            //   于是 mms 判定打 SKIP, 看起来像"网格没给够"而不是参数没解析对。
            //   这种静默假象在收敛阶验证里代价极高, 必须堵掉。
            nxs.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                const std::string tok = next();
                std::size_t pos = 0;
                for (;;)
                {
                    const std::size_t   c   = tok.find(',', pos);
                    const std::string   one = (c == std::string::npos)
                                            ? tok.substr(pos)
                                            : tok.substr(pos, c - pos);
                    if (!one.empty())
                        nxs.push_back(static_cast<uint32_t>(std::stoul(one)));
                    if (c == std::string::npos) break;
                    pos = c + 1;
                }
            }
        }
        else if (a == "--amg-drop")     drop = std::stod(next());
        else if (a == "--amg-cap")      cap = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--amg-coarse")   coarse = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--restart")      restart = std::stoi(next());
        else if (a == "--tol")          tol = std::stod(next());
        else if (a == "--max-iter")     max_iter = std::stoi(next());
        else if (a == "--amg-verbose")  amg_verbose = true;
        else if (a == "--vel-precond")
        {
            const std::string v = next();
            //   amg     = 标量 AMG + 阻尼 Jacobi   (历史默认)
            //   cheb    = 标量 AMG + Chebyshev 多项式光滑
            //   blkamg  = BlockAMG 光滑聚合 (BlockMatrix<double,2>, 块 = K_ij·I₂)
            if      (v == "amg")     pc.kind = 0;
            else if (v == "cheb")    pc.kind = 1;
            else if (v == "blkamg")  pc.kind = 2;
            else { std::cerr << "--vel-precond 只接受 amg | cheb | blkamg, 收到: " << v << "\n"; return 2; }
        }
        else if (a == "--amg-cheb-deg")    pc.cheb_deg   = std::stoi(next());
        else if (a == "--amg-cheb-ratio")  pc.cheb_ratio = std::stod(next());
        else if (a == "--blk-agg")         pc.blk_agg    = std::stoi(next());
        else if (a == "--blk-smooth")      pc.blk_smooth = std::stoi(next());
        else { std::cerr << "未知参数: " << a << "\n"; return 2; }
    }

    if (nxs.empty()) nxs = { 8u, 16u, 32u };

    std::cout << "=== M1: Taylor-Hood (P2/P1) 稳态 Stokes ===\n";
    std::cout << "  算例 = " << casename << "\n";
    std::cout << "  速度 P2 / 压力 P1;  求解器 GMRES(" << restart << "), tol = " << tol
              << ", 预条件 = 块 Schur\n";
    std::cout << "    速度块: " << pc.name() << "\n";
    std::cout << "    Schur : nu·Mp⁻¹ (lumped)\n";
    if (casename == "poiseuille")
        std::cout << "  解析解 u = (y(1-y), 0), p = -2x  —— 离散误差为舍入量级; 可见误差受求解容差与压力 pin 放大控制\n";
    else
        std::cout << "  制造解 (流函数构造); 理论阶: 速度 L2 = 3 / H1 = 2, 压力 L2 = 2\n";
    std::cout << "\n";

    std::vector<Result> Rs;
    for (const uint32_t nx : nxs)
    {
        Result R = (casename == "mms")
            ? run_case<MMS>(nx, drop, cap, amg_verbose, coarse, restart, tol, max_iter, pc)
            : run_case<Poiseuille>(nx, drop, cap, amg_verbose, coarse, restart, tol, max_iter, pc);
        Rs.push_back(R);
    }

    std::cout << "  (残差列后的 '!' = GMRES 未收敛, 迭代数打满; 此时的误差是代数误差,\n"
                 "   不是离散化误差 —— 不要拿它去判装配)\n";
    std::cout << "   nx     ndof   nnz/row   iters   装配(ms)  预条件(ms)  迭代(ms)   ms/iter"
                 "        u-L2      rate        u-H1      rate        p-L2      rate      ‖∇·u‖      p-L2*      残差\n";
    std::cout << "  (p-L2* = 去均值压力误差 ‖e-mean(e)‖: 压力只差一个常数不改变物理,\n"
                 "   若 p-L2 大而 p-L2* 小 -> 误差全在规范方向上, 不是装配/预条件的问题)\n";
    for (std::size_t t = 0; t < Rs.size(); ++t)
    {
        const Result& R = Rs[t];
        const double mspit = (R.iters > 0) ? R.t_it / static_cast<double>(R.iters) : 0.0;
        std::cout << std::setw(5) << nxs[t]
                  << std::setw(9) << R.ndof
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << (static_cast<double>(R.nnz) / static_cast<double>(R.ndof))
                  << std::setw(9) << R.iters
                  << std::setw(11) << std::fixed << std::setprecision(1) << R.t_asm
                  << std::setw(12) << R.t_pre
                  << std::setw(11) << R.t_it
                  << std::setw(11) << std::fixed << std::setprecision(3) << mspit;
        std::cout << "   " << std::scientific << std::setprecision(4) << R.e_u_l2;
        if (t > 0) std::cout << "  " << std::fixed << std::setprecision(2)
                             << std::log2(Rs[t - 1].e_u_l2 / R.e_u_l2);
        else       std::cout << "     --";
        std::cout << "   " << std::scientific << std::setprecision(4) << R.e_u_h1;
        if (t > 0) std::cout << "  " << std::fixed << std::setprecision(2)
                             << std::log2(Rs[t - 1].e_u_h1 / R.e_u_h1);
        else       std::cout << "     --";
        std::cout << "   " << std::scientific << std::setprecision(4) << R.e_p_l2;
        if (t > 0) std::cout << "  " << std::fixed << std::setprecision(2)
                             << std::log2(Rs[t - 1].e_p_l2 / R.e_p_l2);
        else       std::cout << "     --";
        std::cout << "   " << std::scientific << std::setprecision(3) << R.div_l2
                  << "   " << std::scientific << std::setprecision(4) << R.e_p_l2_mf
                  << "   " << std::scientific << std::setprecision(3) << R.res
                  << (R.converged ? "  " : " !") << "\n";
    }
    std::cout << "\n";

    int rc = 0;
    if (casename == "poiseuille")
    {
        // 解析解在离散空间内 -> 任何一档的误差都应只是舍入量级
        // 【为什么阈值从 1e-10 放开】解析解虽然精确落在离散空间内, 但**能观察到
        //  的误差地板不是 1e-14, 而是"代数残差经 M^{-1} 放大后的量级"**。
        //  鞍点矩阵被"压力 pin"引入了一个近零方向: 取 q = 1 - e_pin (常向量挖掉
        //  被钉住的那一个分量), 它在 G 上的 Rayleigh 商只有 ~h^3~h^4 量级,
        //  于是 ||M^{-1}|| 随 h 增长。实测误差/残差放大倍数:
        //      nx=8/16/32 -> u-L2 约 1.5x / 2.2x / 85x, p-L2 约 1.3e3 / 8.8e3 / 1.2e6
        //  即双精度下压力误差根本到不了 1e-10 —— 那个阈值不可达, 不是装配有错。
        //  这里取"代数误差地板之上一到两个数量级"作阈值, 仍足以抓住真正的装配
        //  bug (装配错会把误差抬到 1e-2 量级, 比阈值高 4~6 个数量级)。
        const Result& R = Rs.back();
        // 【压力判据改用去均值误差 p-L2*】压力只差一个常数不改变物理场
        //   (Bᵀ·const = 0), 而 pin 引入的近零方向恰好就是"几乎常数"的方向,
        //   它把 p-L2 抬高了几个数量级却不代表任何物理错误。用 p-L2* 判,
        //   既不漏掉真正的装配 bug (那会把两个量一起抬到 1e-2), 也不会被
        //   规范自由度误伤。
        const bool ok = (R.e_u_l2 < 1e-8) && (R.e_u_h1 < 1e-6)
                     && (R.e_p_l2_mf < 1e-4) && (R.div_l2 < 1e-6);
        std::cout << "  判定: 最细档 u-L2 = " << std::scientific << R.e_u_l2
                  << ", p-L2 = " << R.e_p_l2
                  << ", p-L2* = " << R.e_p_l2_mf
                  << ", ‖∇·u‖ = " << R.div_l2
                  << "  (阈值 1e-8 / 1e-4 / 1e-6);  相对残差 = " << R.res << "\n";
        std::cout << "  压力整体漂移 mean(p_h - p) = " << std::scientific << R.p_mean
                  << "   (p-L2 与 p-L2* 的差 = 这一项被 ‖S⁻¹‖ 放大的量)\n";
        std::cout << "  误差/残差: u-L2 = " << std::scientific << std::setprecision(2)
                  << (R.e_u_l2 / R.res) << "x,  p-L2 = " << (R.e_p_l2 / R.res)
                  << "x   (用 ./STOKESRUN tolscan 可验证误差是否随容差线性下降)\n";
        if (!R.converged)
        {
            // 代数误差还没降下去就判误差阈值, 是在拿求解器的失败冒充装配的失败。
            std::cout << "  [代数未收敛: iters 打满]   ->  FAIL (SOLVER)\n";
            rc = 1;
        }
        else
        {
            std::cout << "  判定 ->  " << (ok ? "PASS" : "FAIL") << "\n";
            rc = ok ? 0 : 1;
        }
    }
    else
    {
        // 收敛阶: 用最后两档
        if (Rs.size() < 2)
        {
            std::cout << "  末档阶数: 只有一档网格, 无法算收敛阶   ->  SKIP\n";
        }
        else
        {
            const std::size_t L = Rs.size() - 1;
            const double r_u2 = std::log2(Rs[L - 1].e_u_l2 / Rs[L].e_u_l2);
            const double r_u1 = std::log2(Rs[L - 1].e_u_h1 / Rs[L].e_u_h1);
            const double r_p2 = std::log2(Rs[L - 1].e_p_l2 / Rs[L].e_p_l2);
            auto chk = [](double got, double want) { return std::fabs(got - want) < 0.25; };
            const bool ok = chk(r_u2, 3.0) && chk(r_u1, 2.0) && chk(r_p2, 2.0);
            std::cout << "  末档阶数: 速度 L2 = " << std::fixed << std::setprecision(2) << r_u2
                      << " (期望 3.00),  速度 H1 = " << r_u1 << " (期望 2.00),  压力 L2 = "
                      << r_p2 << " (期望 2.00)   ->  " << (ok ? "PASS" : "FAIL") << "\n";
            rc = ok ? 0 : 1;
        }
    }
    return rc;
}
