// ============================================================================
// M2: 稳态 Navier-Stokes (Taylor-Hood P2/P1) —— Newton + Re 延拓
// ----------------------------------------------------------------------------
//   算例 A  【cavity: 方腔驱动流, Ghia et al. (1982) 基准】
//     Ω = [0,1]², 顶盖 u = (1,0), 其余三边无滑移, f = 0, nu = 1/Re。
//     验证方式: 竖直中心线 x=0.5 上的 u 剖面与水平中心线 y=0.5 上的 v 剖面
//     逐点对比 Ghia 的 Table I / Table II (各 17 个点), 报 RMS 与最大偏差。
//     这是"没有解析解"的第一个算例 —— 只能靠基准数据判对错, 所以
//     Newton 是否真的收敛 (而不是停在某个残差上) 必须先看得见。
//
//   算例 B  【mms: 制造解收敛阶】
//     与 M1 同一个制造解 (流函数构造), 但右端项加上对流项:
//         f = -nu Δu + (u·∇)u + ∇p          (nu = 1)
//     目的: 在**没有基准数据**的情况下单独验证对流项 C 与 Newton 项 G 的
//     装配与符号 —— 若 C 或 G 差一个符号/错一个指标, 收敛阶会从 3/2/2 掉下来,
//     而 Ghia 那种"看着像流场"的对比根本抓不到这种错。
//
//   用法:
//     ./bin/ns_app                        cavity, nx=32, Re = 100,400,1000
//     ./bin/ns_app --case mms --nx 8,16,32
//     ./bin/ns_app --nx 64 --re 100,400,1000 --verbose
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "fem/assembler.hpp"
#include "fem/boundary.hpp"
#include "fem/dofmap.hpp"
#include "fem/element.hpp"
#include "fem/mesh.hpp"
#include "fem/navier_stokes.hpp"
#include "fem/quadrature.hpp"
#include "fem/stokes.hpp"
#include "math/array.hpp"
#include "math/blockMatrix.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

// ===========================================================================
// Ghia, Ghia & Shin (1982), J. Comput. Phys. 48(3), 387-411
//   Table I  : u 沿竖直中心线 x = 0.5 (17 点)
//   Table II : v 沿水平中心线 y = 0.5 (17 点)
//   两个表都按坐标升序存放。
//   【注意】Table II 里 Re=400 的 x=0.8594 (-0.44993) 与 x=0.9063 (-0.23827)
//   在原始论文里被普遍认为可疑 (与相邻点不单调), 下面的对比把它们排除。
// ===========================================================================
constexpr int kGhiaN = 17;

const double kGhiaY[kGhiaN] = {
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,
    0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000 };

const double kGhiaU100[kGhiaN] = {
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
    -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
    0.78871, 0.84123, 1.00000 };

const double kGhiaU400[kGhiaN] = {
    0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
    -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756,
    0.68439, 0.75837, 1.00000 };

const double kGhiaU1000[kGhiaN] = {
    0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
    -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117,
    0.57492, 0.65928, 1.00000 };

const double kGhiaX[kGhiaN] = {
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000,
    0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000 };

const double kGhiaV100[kGhiaN] = {
    0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
    0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
    -0.05906, 0.00000 };

const double kGhiaV400[kGhiaN] = {
    0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174,
    0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663,
    -0.12146, 0.00000 };

const double kGhiaV1000[kGhiaN] = {
    0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235,
    0.02526, -0.31966, -0.42665, -0.51500, -0.39188, -0.33714, -0.27669,
    -0.21388, 0.00000 };

// Re=400 这两个点被排除 (原始论文可疑值)
bool ghia_skip(int re_tag, double x)
{
    if (re_tag != 400) return false;
    return (std::fabs(x - 0.8594) < 1e-6) || (std::fabs(x - 0.9063) < 1e-6);
}

// ===========================================================================
// 算例 A: 方腔驱动流 (无解析解)
// ===========================================================================
struct Cavity
{
    static constexpr bool has_exact = false;
    static double nu_of(double re) { return 1.0 / re; }

    // 顶盖两端的过渡宽度 lid_delta:
    //   0   = 硬设置 (Ghia 的原始提法: 角点上速度从 0 直接跳到 1,
    //         于是角点附近单元里 |∇u| ~ 1/h, 而 G=∫φφ∂u 也被放大)
    //   > 0 = 两端各用一段三次 smoothstep 过渡到侧壁的 0。
    //   默认 2 个单元宽 —— 只改变角点附近, 中心线剖面不受影响。
    static double lid_delta;

    static double lid_ramp(double x)
    {
        const double d = lid_delta;
        if (!(d > 0.0)) return 1.0;
        if (x < d)       { const double t = x / d;         return t * t * (3.0 - 2.0 * t); }
        if (x > 1.0 - d) { const double t = (1.0 - x) / d; return t * t * (3.0 - 2.0 * t); }
        return 1.0;
    }

    // Dirichlet: 顶盖 (y = 1) 上 u = (lid_ramp(x), 0), 其余三边 u = 0。
    //   硬设置时角点 (0,1)/(1,1) 取顶盖值 (与 Ghia 一致);
    //   开了过渡之后它们自然变成 0, 与侧壁连续。
    static void bc(double x, double y, double& vx, double& vy)
    {
        vx = (y > 1.0 - 1e-12) ? lid_ramp(x) : 0.0;
        vy = 0.0;
    }
    static double fx(double, double) { return 0.0; }
    static double fy(double, double) { return 0.0; }
    static double p_pin_value()      { return 0.0; }
};
double Cavity::lid_delta = 0.0;

// ===========================================================================
// 算例 B: MMS (与 M1 同一个制造解, 右端项加上对流项), nu = 1
//   ψ = sin²(πx) sin²(πy)  =>  u = (∂ψ/∂y, -∂ψ/∂x),  ∇·u ≡ 0,  u|∂Ω = 0
//   p = cos(πx) cos(πy)
//   f = -Δu + (u·∇)u + ∇p
// ===========================================================================
struct MMSNS
{
    static constexpr bool has_exact = true;
    static double nu_of(double /*re*/) { return 1.0; }

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
    static void gux(double x, double y, double& gx, double& gy)
    {
        const double s = std::sin(kPi * x);
        gx = kPi * kPi * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
        gy = 2.0 * kPi * kPi * s * s * std::cos(2.0 * kPi * y);
    }
    static void guy(double x, double y, double& gx, double& gy)
    {
        const double s = std::sin(kPi * y);
        gx = -2.0 * kPi * kPi * std::cos(2.0 * kPi * x) * s * s;
        gy = -kPi * kPi * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
    }
    static double lap_ux(double x, double y)
    {
        const double s = std::sin(kPi * x);
        return 2.0 * kPi * kPi * kPi * std::cos(2.0 * kPi * x) * std::sin(2.0 * kPi * y)
             - 4.0 * kPi * kPi * kPi * s * s * std::sin(2.0 * kPi * y);
    }
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
    // f = -Δu + (u·∇)u + ∇p
    static double fx(double x, double y)
    {
        double a1 = 0.0, a2 = 0.0;
        gux(x, y, a1, a2);                       // ∇u_x = (a1, a2)
        return -lap_ux(x, y) + (ux(x, y) * a1 + uy(x, y) * a2)
             - kPi * std::sin(kPi * x) * std::cos(kPi * y);
    }
    static double fy(double x, double y)
    {
        double a1 = 0.0, a2 = 0.0;
        guy(x, y, a1, a2);                       // ∇u_y = (a1, a2)
        return -lap_uy(x, y) + (ux(x, y) * a1 + uy(x, y) * a2)
             - kPi * std::cos(kPi * x) * std::sin(kPi * y);
    }
    static void bc(double, double, double& vx, double& vy) { vx = 0.0; vy = 0.0; }
    static double p_pin_value() { return 1.0; }             // p(0,0) = 1
};

// ===========================================================================
// 小工具: CSR 矩阵-向量乘 (残差里要用 6 次, 不想每次都分配一份)
// ===========================================================================
void csr_mv(const sparse::SMatrix& A, const std::vector<double>& x,
            std::vector<double>& y)
{
    const auto& rp = A.row_ptr();
    const auto& ci = A.col_idx();
    const auto& vv = A.values();
    const uint32_t n = A.rows();
    y.assign(n, 0.0);
    for (uint32_t i = 0; i < n; ++i)
    {
        double s = 0.0;
        for (uint32_t p = rp[i]; p < rp[i + 1]; ++p) s += vv[p] * x[ci[p]];
        y[i] = s;
    }
}

// ===========================================================================
// 一次 Newton 求解的结果
// ===========================================================================
struct NSResult
{
    std::size_t ndof = 0;
    int    newton   = 0;      // Newton 步数
    int    lin_its  = 0;      // 累计 GMRES 迭代数
    double t_asm = 0.0, t_pre = 0.0, t_solve = 0.0;
    double res_final = 0.0;   // 最终非线性残差 (自由自由度上的 2-范数)
    bool   converged = false;

    std::vector<double> ux, uy, pp;

    double e_u_l2 = 0.0, e_u_h1 = 0.0, e_p_l2 = 0.0, e_p_l2_mf = 0.0;
    double div_l2 = 0.0;
};

struct NSOpt
{
    double   drop     = 0.02;
    uint32_t cap      = 60;
    uint32_t coarse   = 300;
    int      restart  = 200;
    int      max_newton = 30;
    double   newton_tol = 1e-10;
    double   lin_tol0   = 1e-2;    // 第一次线性求解的相对容差 (Eisenstat-Walker 起点)
    // 强制项下界。验阶对线性残差非常敏感：压强误差对代数残差的
    // 放大不可忽略，实测中 1e-6 的下界会让压强误差被污染约 50%，
    // 直接把收敛阶击坏 (MMS 出现 -0.24 阶)。所以默认回到 1e-10；
    // 高 Re 时若确实追不到，用 --lin-tol-min 1e-6 放松，但那时的
    // 结果不能拿来验阶。
    double   lin_tol_min = 1e-10;
    double   ew_alpha   = 1.618;   // Eisenstat-Walker 指数 α = (1+√5)/2
    // 上限放宽到 3000: 现在“打满”不再直接掐死 Newton (见 lin_usable 判据),
    // 打满只作为“本次方向精度不足”的信号, 不影响后续迭代。
    int      max_lin    = 3000;    // 单次线性求解的 GMRES 上限 (与 restart 解耦)
    bool     amg_verbose = false;
    bool     verbose = false;
    // 光滑子: 0 = 阻尼 Jacobi, 1 = Chebyshev, 2 = BlockAMG。
    // M1 在**同一个 P2 速度算子**上实测过: Jacobi 在 P2 上明显不行
    // (nx=32: Jacobi 348 次 vs Chebyshev 100 次, 而且 Jacobi 的迭代数随 h
    // 一路增长, Chebyshev 基本持平)。NS 里速度块就是那个算子加一个小扰动
    // C, 所以默认同样用 Chebyshev(deg=3, ratio=30)。
    // --vel-pre cd 会显式退回 Jacobi, 用来做 A/B。
    int      vel_kind = 1;         // 0 = AMG+Jacobi, 1 = AMG+Chebyshev, 2 = BlockAMG
    int      cheb_deg = 3;
    double   cheb_ratio = 30.0;
    bool     pre_on_cd = true;     // true: 速度块预条件建在 nuK+C 上; false: 只建在 nuK 上
    bool     vel_ilu    = false;   // true: 速度块改用 ILU(0) on 完整 A (含 C 与 G)
    // 默认用 Stokes 型 nu·Mp⁻¹：它在 MMS(3/2/2) 与 cavity Re=100 上都已验证。
    // LSC/BFBt 理论上是 Oseen 的精确 Schur 近似，但目前仍在实测阶段，
    // 所以保持可选 (--schur lsc)，不改默认行为。
    int      schur_kind = 0;       // 0 = nu·Mp⁻¹ (Stokes 型), 1 = LSC/BFBt (Oseen 型)
    int      lsc_cycles = 2;      // LSC 中 E⁻¹ 的 AMG V-cycle 次数 (迭代加细)
    // SUPG 稳定项的缩放系数; 0 = 关闭 (纯 Galerkin)。
    //   对高 Re 它不只是"稳定性"问题: 沿流线补上的扩散 tau·|u|² 在
    //   Re>=400 时比 nu 本身大一个量级, 速度块重新变成椭圆主导,
    //   AMG 才有得可干 (实测估算: C/νK ≈ 12 -> C/(νK+Ssd) ≈ 1.7)。
    double   tau_scale  = 1.0;
    bool     lid_smooth = true;   // 顶盖两端用 smoothstep 过渡 (消掉角点奇性)
};

// ===========================================================================
// Newton + 块 Schur 预条件的一次完整求解
//   ux/uy/pp 既是输入 (初值, 用于 Re 延拓) 也是输出。
// ===========================================================================
template <class Case>
NSResult run_ns(uint32_t nx, double re,
                std::vector<double>& ux, std::vector<double>& uy,
                std::vector<double>& pp, const NSOpt& op)
{
    const double nu = Case::nu_of(re);
    const auto T0 = std::chrono::steady_clock::now();

    const fem::Mesh   m    = fem::make_unit_square_mesh(nx, nx);
    const fem::DofMap dm_v = fem::make_dofmap(m, 2);
    const fem::DofMap dm_p = fem::make_dofmap(m, 1);
    const std::size_t nv = dm_v.n_dof;
    const std::size_t np = dm_p.n_dof;
    const std::size_t p_pin = 0;

    auto S = std::make_shared<fem::TaylorHood>();
    S->nv = nv;  S->np = np;  S->p_pin = p_pin;

    // ---- 速度刚度 K (含 nu) ----
    fem::CsrPattern pat_v = fem::build_pattern(m, dm_v);
    std::vector<double> val_K;
    fem::assemble_stiffness<2>(m, dm_v, pat_v, [nu](double, double) { return nu; }, val_K);
    S->K = sparse::SMatrix(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                           pat_v.row_ptr, pat_v.col_idx, val_K);

    // ---- 负散度 Bx / By ----
    fem::CsrPattern pat_vp  = fem::build_pattern_rect(m, dm_p, dm_v);
    // 转置形式: 行 = 速度 (检验函数), 列 = 压力。SUPG 的"压力 -> 动量"
    // 耦合块要加到 Bᵀ 上, 就需要这个方向的模式。
    fem::CsrPattern pat_vpT = fem::build_pattern_rect(m, dm_v, dm_p);
    std::vector<double> val_Bx, val_By;
    fem::assemble_divergence<2, 1>(m, dm_v, dm_p, pat_vp, val_Bx, val_By);
    S->Bx = sparse::SMatrix(static_cast<uint32_t>(np), static_cast<uint32_t>(nv),
                            pat_vp.row_ptr, pat_vp.col_idx, val_Bx);
    S->By = sparse::SMatrix(static_cast<uint32_t>(np), static_cast<uint32_t>(nv),
                            pat_vp.row_ptr, pat_vp.col_idx, val_By);
    S->BxT = S->Bx.transpose();
    S->ByT = S->By.transpose();

    // ---- 压力质量 Mp 与 lumping (Schur 预条件) ----
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

    // ---- 右端项 ----
    std::vector<double> fvx(nv, 0.0), fvy(nv, 0.0);
    fem::assemble_load<2>(m, dm_v, [](double x, double y) { return Case::fx(x, y); }, fvx);
    fem::assemble_load<2>(m, dm_v, [](double x, double y) { return Case::fy(x, y); }, fvy);

    // ---- Dirichlet 列表 + 边界值 ----
    const auto bd_v = fem::boundary_dofs(m, dm_v);
    const auto c_v  = fem::dof_coords(m, dm_v);
    std::vector<char> is_bd(nv, 0);
    for (const uint32_t d : bd_v) is_bd[d] = 1;

    // ---- 预条件器专用的 Kd: K 施加一次速度 Dirichlet ----
    //   ("只建在 nuK 上"那条对照路径 --vel-pre stokes 用; 纯 Neumann 的 K 有
    //    常数零空间, AMG 直接用会给出垃圾修正, 见 stokes.hpp 的说明)
    {
        std::vector<double> val_Kd = S->K.values();
        std::vector<double> dummy(nv, 0.0);
        std::vector<double> gz(bd_v.size(), 0.0);
        fem::apply_dirichlet(static_cast<uint32_t>(nv),
                             pat_v.row_ptr, pat_v.col_idx, val_Kd,
                             dummy, bd_v, gz);
        S->Kd = sparse::SMatrix(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                pat_v.row_ptr, pat_v.col_idx, val_Kd);
    }

    std::vector<uint32_t> fixed;
    std::vector<double>   gval;
    fixed.reserve(2u * bd_v.size() + 1u);
    gval.reserve(2u * bd_v.size() + 1u);
    for (const uint32_t d : bd_v)
    {
        double vx = 0.0, vy = 0.0;
        Case::bc(c_v[d].first, c_v[d].second, vx, vy);
        fixed.push_back(static_cast<uint32_t>(2u * d + 0u)); gval.push_back(vx);
        fixed.push_back(static_cast<uint32_t>(2u * d + 1u)); gval.push_back(vy);
    }
    fixed.push_back(static_cast<uint32_t>(2u * nv + p_pin));
    gval.push_back(Case::p_pin_value());

    // ---- 初值 ----
    if (ux.size() != nv) { ux.assign(nv, 0.0); uy.assign(nv, 0.0); }
    if (pp.size() != np) { pp.assign(np, 0.0); }
    for (const uint32_t d : bd_v)
    {
        double vx = 0.0, vy = 0.0;
        Case::bc(c_v[d].first, c_v[d].second, vx, vy);
        ux[d] = vx;  uy[d] = vy;
    }
    pp[p_pin] = Case::p_pin_value();

    const auto T1 = std::chrono::steady_clock::now();

    // ---- Schur 块: 与 M1 相同 (nu·Mp⁻¹) ----
    auto S_pre_mass = fem::make_pressure_schur_preconditioner(S->Mp_lumped, nu, p_pin);
    // make_schur_preconditioner 内部传给 S_pre 的是 B A^{-1}r_u - r_p。
    // pin 行在消元后是 +I，所以这里必须对该分量再取负，最终才得到 z_p=r_p。
    auto S_pre = [S_pre_mass, p_pin](const std::vector<double>& r,
                                     std::vector<double>& z)
    {
        S_pre_mass(r, z);
        if (p_pin < r.size()) z[p_pin] = -r[p_pin];
    };
    auto sys   = fem::make_saddle_point_system(S);

    // =======================================================================
    // LSC / BFBt 型 Schur 预条件
    //   Ŝ⁻¹ = E⁻¹ (B Mv⁻¹ A Mv⁻¹ Bᵀ) E⁻¹,   E = B Mv⁻¹ Bᵀ
    //   连续意义下它与真正的 S⁻¹ 精确相等: 记 L = nu(-Δ) + u·∇, 则
    //       S⁻¹   → h⁻² (nu k² + i u·k)/k²
    //       Ŝ⁻¹   → h⁻² (nu k² + i u·k)/k²          (两者逐项相同)
    //   而 nu·Mp⁻¹ 只给出 h⁻²·nu, 丢掉了 i u·k/k² 那一项 —— 高频端相差
    //   一个网格雷诺数 |u|h/nu (Re=1000, nx=32 时约 31 倍), 这正是高 Re
    //   GMRES 停滞的主因之一。
    //   E 与 u 无关, 所以只装配一次, AMG setup 在 Newton 外层摊销掉。
    // =======================================================================
    std::function<void(const std::vector<double>&, std::vector<double>&)> E_pre;
    std::vector<double> mv_inv;
    if (op.schur_kind == 1)
    {
        std::vector<double> val_Mv;
        fem::assemble_mass<2>(m, dm_v, pat_v,
                              [](double, double) { return 1.0; }, val_Mv);
        std::vector<double> mv(nv, 0.0);
        for (std::size_t i = 0; i < nv; ++i)
        {
            double acc = 0.0;
            for (uint32_t q = pat_v.row_ptr[i]; q < pat_v.row_ptr[i + 1]; ++q)
                acc += val_Mv[q];
            mv[i] = acc;
        }
        mv_inv.assign(nv, 0.0);
        for (std::size_t i = 0; i < nv; ++i)
            mv_inv[i] = (mv[i] > 0.0) ? (1.0 / mv[i]) : 0.0;

        // 列缩放 1/mv, 并去掉速度边界列 —— 与受约束的 B 保持一致
        auto scaled_B = [&](const sparse::SMatrix& B)
        {
            std::vector<double> v = B.values();
            const auto& ci = B.col_idx();
            for (std::size_t q = 0; q < v.size(); ++q)
            {
                const uint32_t j = ci[q];
                v[q] = (j < nv && !is_bd[j]) ? (v[q] * mv_inv[j]) : 0.0;
            }
            return sparse::SMatrix(B.rows(), B.cols(), B.row_ptr(), B.col_idx(), v);
        };
        const sparse::SMatrix Bxs = scaled_B(S->Bx);
        const sparse::SMatrix Bys = scaled_B(S->By);
        const sparse::SMatrix E1  = Bxs * S->BxT;
        const sparse::SMatrix E2  = Bys * S->ByT;

        // 两个乘积同模式, 但 SpGEMM 可能按数值裁剪 -> 用 COO 累加合并最稳
        std::vector<double>   eva; std::vector<uint32_t> erl, ecl;
        eva.reserve(E1.nnz() + E2.nnz());
        erl.reserve(E1.nnz() + E2.nnz());
        ecl.reserve(E1.nnz() + E2.nnz());
        for (uint32_t k = 0; k < E1.rows(); ++k)
        {
            for (uint32_t q = E1.row_ptr()[k]; q < E1.row_ptr()[k + 1]; ++q)
            { erl.push_back(k); ecl.push_back(E1.col_idx()[q]); eva.push_back(E1.values()[q]); }
            for (uint32_t q = E2.row_ptr()[k]; q < E2.row_ptr()[k + 1]; ++q)
            { erl.push_back(k); ecl.push_back(E2.col_idx()[q]); eva.push_back(E2.values()[q]); }
        }
        sparse::SMatrix E(E1.rows(), E1.cols(), eva, erl, ecl);

        // 压力 pin: 与全系统一致地把该行/列清零、对角置 1。去掉这一行之后
        // E 是可逆的 (E 的零空间是 span{1}, 而 1 在 pin 分量上非零)。
        {
            auto Ev = E.values();
            std::vector<double> dummy(E.rows(), 0.0);
            std::vector<uint32_t> pin_only { static_cast<uint32_t>(p_pin) };
            std::vector<double>   pin_zero { 0.0 };
            fem::apply_dirichlet(E.rows(), E.row_ptr(), E.col_idx(), Ev,
                                 dummy, pin_only, pin_zero);
            E = sparse::SMatrix(E.rows(), E.cols(), E.row_ptr(), E.col_idx(), Ev);
        }
        // E ≈ 压力 P1 Laplacian (SPD)。LSC 的 Ũ⁻¹ 里 E⁻¹ 出现两次，
        // 单次 V-cycle 的误差会被放大：实测中单 V-cycle 的 LSC 比
        // nu·Mp⁻¹ 还慢 6 倍。这里改成 n 次 V-cycle 的迭代加细 ——
        // 它仍然是 r 的线性算子，可以安全地嵌进 GMRES。
        // 光滑子用 M1 实测的 Chebyshev(deg=3, ratio=30)：E 虽然是 P1，
        // 但由 B M⁻¹ Bᵀ 生成，不是严格的 M-矩阵，Jacobi 不够用。
        auto Ep = std::make_shared<sparse::SMatrix>(std::move(E));
        auto E_vcycle = sparse::SMatrix::make_amg_preconditioner(
            *Ep, op.drop, op.cap, op.amg_verbose, op.coarse, 1, op.cheb_deg, op.cheb_ratio);
        E_pre = [Ep, E_vcycle, np, ncyc = op.lsc_cycles](
            const std::vector<double>& r, std::vector<double>& z)
        {
            std::vector<double> t(np), w(np);
            E_vcycle(r, z);
            for (int c = 1; c < ncyc; ++c)
            {
                sparse::SMatrix::spmv_into(*Ep, z, t);
                for (std::size_t i = 0; i < np; ++i) w[i] = r[i] - t[i];
                E_vcycle(w, t);
                for (std::size_t i = 0; i < np; ++i) z[i] += t[i];
            }
        };
    }

    // 速度块算子 A = Ad ⊗ I₂ + G 的 apply (LSC 的中间算子要用;
    // 每个 Newton 步重新装配后更新)
    std::function<void(const std::vector<double>&, std::vector<double>&)> vel_apply;

    // ---- Newton ----
    NSResult R;
    R.ndof = 2u * nv + np;
    double   nr0 = 0.0;       // 首个残差: Newton 相对收敛判据的固定基准
    double   nr_prev = 0.0;  // 上一步残差: 只给 Eisenstat-Walker 强制项使用
    // 强制项的动态下界。一旦某一步的 GMRES 没能做到要求的 η,
    // 就把下界抬到“实际做到的精度”, 否则每一步都会在同一堵
    // 墙上打满 max_lin, Newton 却毫无进展。
    double   eta_floor = op.lin_tol_min;
    int      n_stall = 0;     // 连续没有实质下降的 Newton 步数
    int      n_linfail = 0;   // 线性未达标 (但方向被接受) 的步数
    int      lin_total = 0;
    double   t_pre_total = 0.0;

    std::vector<double> rux(nv), ruy(nv), rp(np), tmp(nv);
    std::vector<double> val_C, val_Gxx, val_Gxy, val_Gyx, val_Gyy;

    for (int it = 0; it < op.max_newton; ++it)
    {
        // ---- (1) 装配对流项与 Newton 项 ----
        fem::assemble_ns_terms<2>(m, dm_v, pat_v, ux, uy,
                                  val_C, val_Gxx, val_Gxy, val_Gyx, val_Gyy);

        // ---- SUPG 稳定项 ----
        //   流线扩散 Ssd 进 Ad (⊗I₂ 的对角块);   G 那部分进 Gxx..Gyy;
        //   压力耦合块进 Bᵀ;   残差贡献进 rux/ruy。
        //   **注意**: Ssd 只能加进 Ad, 绝不能加进 C —— 非线性残差里要的
        //   是真实的对流项 C·u, 混进稳定项就不是同一个方程了。
        std::vector<double> val_Ssd, val_Sxx, val_Sxy, val_Syx, val_Syy;
        std::vector<double> val_DxT, val_DyT, rsx, rsy;
        if (op.tau_scale > 0.0)
        {
            fem::assemble_supg<2, 1>(m, dm_v, dm_p, pat_v, pat_vpT, ux, uy, pp, nu,
                                     [](double x, double y) { return Case::fx(x, y); },
                                     [](double x, double y) { return Case::fy(x, y); },
                                     op.tau_scale,
                                     val_Ssd, val_Sxx, val_Sxy, val_Syx, val_Syy,
                                     val_DxT, val_DyT, rsx, rsy);
            for (std::size_t p = 0; p < val_Gxx.size(); ++p)
            {
                val_Gxx[p] += val_Sxx[p];   val_Gxy[p] += val_Sxy[p];
                val_Gyx[p] += val_Syx[p];   val_Gyy[p] += val_Syy[p];
            }
        }

        // （必须在构造 C / Gxx..Gyy 之前完成: SMatrix 是按值拷贝的,
        //  先构造再改 val_Gxx 就加不进去了）

        const sparse::SMatrix C(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                pat_v.row_ptr, pat_v.col_idx, val_C);
        const sparse::SMatrix Gxx(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                  pat_v.row_ptr, pat_v.col_idx, val_Gxx);
        const sparse::SMatrix Gxy(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                  pat_v.row_ptr, pat_v.col_idx, val_Gxy);
        const sparse::SMatrix Gyx(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                  pat_v.row_ptr, pat_v.col_idx, val_Gyx);
        const sparse::SMatrix Gyy(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                  pat_v.row_ptr, pat_v.col_idx, val_Gyy);

        // Ad = nu·K + C (+ SUPG 流线扩散)   (K 已含 nu; 同模式, 逐元素相加)
        std::vector<double> val_Ad = S->K.values();
        {
            const auto& cv = C.values();
            for (std::size_t p = 0; p < val_Ad.size(); ++p) val_Ad[p] += cv[p];
            if (op.tau_scale > 0.0)
                for (std::size_t p = 0; p < val_Ad.size(); ++p) val_Ad[p] += val_Ssd[p];
        }
        const sparse::SMatrix Ad(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                                 pat_v.row_ptr, pat_v.col_idx, val_Ad);

        // Bᵀ + SUPG 的压力耦合块。两个矩阵的 (row, col) 集合相同,
        // 但 CSR 的列序不保证一致 -> 用 COO 合并, 不能逐元素相加。
        sparse::SMatrix BxT_u = S->BxT, ByT_u = S->ByT;
        if (op.tau_scale > 0.0)
        {
            auto merge_Bt = [&pat_vpT](const sparse::SMatrix& base,
                                       const std::vector<double>& add) -> sparse::SMatrix
            {
                std::vector<double>   cv;   std::vector<uint32_t> rl, cl;
                cv.reserve(base.nnz() + add.size());
                rl.reserve(base.nnz() + add.size());
                cl.reserve(base.nnz() + add.size());
                const uint32_t nr = base.rows();
                for (uint32_t i = 0; i < nr; ++i)
                {
                    for (uint32_t q = base.row_ptr()[i]; q < base.row_ptr()[i + 1]; ++q)
                    { rl.push_back(i); cl.push_back(base.col_idx()[q]); cv.push_back(base.values()[q]); }
                    for (uint32_t q = pat_vpT.row_ptr[i]; q < pat_vpT.row_ptr[i + 1]; ++q)
                    { rl.push_back(i); cl.push_back(pat_vpT.col_idx[q]); cv.push_back(add[q]); }
                }
                return sparse::SMatrix(base.rows(), base.cols(), cv, rl, cl);
            };
            BxT_u = merge_Bt(S->BxT, val_DxT);
            ByT_u = merge_Bt(S->ByT, val_DyT);
        }

        // 速度块 apply: y = (Ad ⊗ I₂) x + G x,  x/y 都是 2nv 的 node-major
        {
            struct VelWS { std::vector<double> inx, iny, t1, t2, t3; };
            auto vws = std::make_shared<VelWS>();
            vws->inx.resize(nv); vws->iny.resize(nv);
            vws->t1.resize(nv);  vws->t2.resize(nv);  vws->t3.resize(nv);
            vel_apply = [Ad, Gxx, Gxy, Gyx, Gyy, nv, vws](
                const std::vector<double>& x, std::vector<double>& y)
            {
                VelWS& w = *vws;
                for (std::size_t i = 0; i < nv; ++i)
                { w.inx[i] = x[2u * i + 0u];  w.iny[i] = x[2u * i + 1u]; }
                sparse::SMatrix::spmv_into(Ad,  w.inx, w.t1);
                sparse::SMatrix::spmv_into(Gxx, w.inx, w.t2);
                sparse::SMatrix::spmv_into(Gxy, w.iny, w.t3);
                for (std::size_t i = 0; i < nv; ++i)
                    y[2u * i + 0u] = w.t1[i] + w.t2[i] + w.t3[i];
                sparse::SMatrix::spmv_into(Gyx, w.inx, w.t1);
                sparse::SMatrix::spmv_into(Ad,  w.iny, w.t2);
                sparse::SMatrix::spmv_into(Gyy, w.iny, w.t3);
                for (std::size_t i = 0; i < nv; ++i)
                    y[2u * i + 1u] = w.t1[i] + w.t2[i] + w.t3[i];
            };
        }

        // ---- (2) 残差  R_u = K u + C u + Bᵀ p - f ;  R_p = B u ----
        csr_mv(S->K, ux, rux);
        csr_mv(S->K, uy, ruy);
        csr_mv(C, ux, tmp);  for (std::size_t i = 0; i < nv; ++i) rux[i] += tmp[i];
        csr_mv(C, uy, tmp);  for (std::size_t i = 0; i < nv; ++i) ruy[i] += tmp[i];
        csr_mv(S->BxT, pp, tmp); for (std::size_t i = 0; i < nv; ++i) rux[i] += tmp[i];
        csr_mv(S->ByT, pp, tmp); for (std::size_t i = 0; i < nv; ++i) ruy[i] += tmp[i];
        for (std::size_t i = 0; i < nv; ++i) { rux[i] -= fvx[i]; ruy[i] -= fvy[i]; }
        if (op.tau_scale > 0.0)                       // SUPG 残差项: tau (u·∇φ_i) R_M
            for (std::size_t i = 0; i < nv; ++i) { rux[i] += rsx[i]; ruy[i] += rsy[i]; }
        csr_mv(S->Bx, ux, rp);
        csr_mv(S->By, uy, tmp);
        for (std::size_t k = 0; k < np; ++k) rp[k] += tmp[k];

        // ---- (3) 残差范数 (只数自由自由度: 固定行在下面会被换成单位行) ----
        double nr2 = 0.0;
        for (std::size_t i = 0; i < nv; ++i)
            if (!is_bd[i]) { nr2 += rux[i] * rux[i] + ruy[i] * ruy[i]; }
        for (std::size_t k = 0; k < np; ++k)
            if (k != p_pin) { nr2 += rp[k] * rp[k]; }
        const double nr = std::sqrt(nr2);

        R.newton = it;      // 已接受的 Newton 更新数（当前只是第 it 次残差评估）
        R.res_final = nr;
        if (it == 0) { nr0 = nr; nr_prev = nr; }

        if (op.verbose)
            std::cout << "    [Newton " << std::setw(2) << it << "]  ‖R‖ = "
                      << std::scientific << std::setprecision(4) << nr
                      << "   相对首步 = " << std::scientific << std::setprecision(3)
                      << ((nr0 > 0.0) ? nr / nr0 : 0.0) << "\n";

        // 相对判据必须始终相对首步 nr0；若相对 nr_prev，就等价于要求某一步
        // 单独下降 1e-10，MMS 会一直跑到舍入误差仍被误报为未收敛。
        if (nr <= op.newton_tol * nr0 || nr < 1e-13)
        {
            R.converged = true;
            break;
        }

        // ---- 停滞判据 ----
        // 残差连续两步没有实质下降 => 已经撞上了该预条件器/该精度能达到的
        // 地板, 再继续只会白白烧掉线性求解。只要求两步是为了避免
        // 把高 Re 首步的正常过冲误判为停滞。
        if (it > 0 && nr_prev > 0.0 && nr > 0.9 * nr_prev) ++n_stall;
        else                                              n_stall = 0;
        if (n_stall >= 2)
        {
            if (op.verbose)
                std::cout << "              Newton 残差连续两步未实质下降 (比上一步 > 0.9x), "
                             "判定停滞, 停止本档\n";
            break;
        }

        // ---- (4) Eisenstat-Walker 型强制项 ----
        //   线性系统只要求"够用": η 太紧 -> 白白多迭代, 太松 -> Newton 不收敛。
        //   η_k = 0.9 (‖R_k‖/‖R_{k-1}‖)², 夹到 [1e-10, 0.1]; 首步用 lin_tol0。
        double eta = op.lin_tol0;
        if (it > 0 && nr_prev > 0.0)
        {
            const double ratio = nr / nr_prev;
            // 标准 Eisenstat-Walker: η = γ·ratio^α, α = (1+√5)/2 ≈ 1.618。
            // 用 α=2 会让 η 在 Newton 二次收敛区里塌得太快 —— Re=400 第 4 步
            // 的 η 掉到 2.4e-3, 而 GMRES 在那个预条件器下根本到不了。
            eta = 0.9 * std::pow(ratio, op.ew_alpha);
            if (eta > 0.1)   eta = 0.1;
            if (eta < eta_floor)      eta = eta_floor;   // eta_floor >= lin_tol_min
        }

        // ---- (5) 展平 Jacobian 与右端项, 施加约束 ----
        sparse::SMatrix J = fem::build_saddle_matrix_ns(
            Ad, Gxx, Gxy, Gyx, Gyy, S->Bx, S->By, BxT_u, ByT_u, p_pin);

        std::vector<double> Jrhs(2u * nv + np, 0.0);
        for (std::size_t i = 0; i < nv; ++i)
        {
            Jrhs[2u * i + 0u] = -rux[i];
            Jrhs[2u * i + 1u] = -ruy[i];
        }
        for (std::size_t k = 0; k < np; ++k) Jrhs[2u * nv + k] = -rp[k];

        {
            auto Jv = J.values();
            std::vector<double> gzero(fixed.size(), 0.0);
            fem::apply_dirichlet(static_cast<uint32_t>(2u * nv + np),
                                 J.row_ptr(), J.col_idx(), Jv, Jrhs, fixed, gzero);
            J = sparse::SMatrix(J.rows(), J.cols(), J.row_ptr(), J.col_idx(), Jv);
        }

        // ---- (6) 预条件 ----
        const auto TP0 = std::chrono::steady_clock::now();

        //   速度块: 标量 AMG 建在 (nuK + C) 的 Dirichlet 版上, 按分量各 apply 一次。
        //   这里**故意忽略 G** —— G 是 O(‖∇u‖) 的低阶扰动, 而 AMG 是针对椭圆
        //   算子的; 把 C 放进去已经能把对流方向的信息带给粗层。
        //   (--vel-pre stokes 可以退回到只建在 nuK 上做 A/B 对照。)
        std::vector<double> val_Ad_d = val_Ad;
        {
            std::vector<double> dummy(nv, 0.0);
            std::vector<uint32_t> bd_only = bd_v;
            std::vector<double>   gz(bd_only.size(), 0.0);
            fem::apply_dirichlet(static_cast<uint32_t>(nv),
                                 pat_v.row_ptr, pat_v.col_idx, val_Ad_d,
                                 dummy, bd_only, gz);
        }
        const sparse::SMatrix Amat = op.pre_on_cd
            ? sparse::SMatrix(static_cast<uint32_t>(nv), static_cast<uint32_t>(nv),
                              pat_v.row_ptr, pat_v.col_idx, val_Ad_d)
            : S->Kd;
        auto A_pre = fem::make_velocity_preconditioner(
            Amat, op.drop, op.cap, op.amg_verbose, op.coarse,
            op.vel_kind, op.cheb_deg, op.cheb_ratio);

        // 可选: 速度块改用 ILU(0) on 完整 A (含 C 与 G)。AMG 是针对椭圆算子
        // 设计的, 对流占优时粗化/光滑都不对路; ILU 至少把 G 也吃进去了。
        std::function<void(const std::vector<double>&, std::vector<double>&)> A_pre_use = A_pre;
        if (op.vel_ilu)
        {
            sparse::SMatrix V = fem::build_velocity_block_matrix(Ad, Gxx, Gxy, Gyx, Gyy);
            {
                auto Vv = V.values();
                std::vector<double> dummy(V.rows(), 0.0);
                std::vector<uint32_t> fv;  std::vector<double> gv;
                fv.reserve(2u * bd_v.size()); gv.reserve(2u * bd_v.size());
                for (const uint32_t d : bd_v)
                {
                    fv.push_back(static_cast<uint32_t>(2u * d + 0u)); gv.push_back(0.0);
                    fv.push_back(static_cast<uint32_t>(2u * d + 1u)); gv.push_back(0.0);
                }
                fem::apply_dirichlet(V.rows(), V.row_ptr(), V.col_idx(), Vv,
                                     dummy, fv, gv);
                V = sparse::SMatrix(V.rows(), V.cols(), V.row_ptr(), V.col_idx(), Vv);
            }
            A_pre_use = sparse::SMatrix::make_ilu0_preconditioner(V);
        }

        // 块 Schur apply 的速度求解由 A_pre 完成；SaddlePointSystem 当前
        // 实际只消费 B/B^T。不要在这里捕获五份 Jacobian 矩阵做无效 spmv_A。
        auto sys_ns = sys;

        // 全系统已经对速度边界与压力 pin 做了对称消元；Schur 预条件器里的
        // B/B^T 也必须使用同一个受约束算子。沿用原始 B 会让一次预条件 apply
        // 在边界自由度和 pin 列上重新注入非零量，高 Re 时会严重污染 Krylov 空间。
        sys_ns.spmv_B = [S, is_bd, np, p_pin](const std::vector<double>& u,
                                                  std::vector<double>& y)
        {
            y.assign(np, 0.0);
            const auto& Bxp = S->Bx.row_ptr(); const auto& Bxc = S->Bx.col_idx(); const auto& Bxv = S->Bx.values();
            const auto& Byp = S->By.row_ptr(); const auto& Byc = S->By.col_idx(); const auto& Byv = S->By.values();
            for (std::size_t k = 0; k < np; ++k)
            {
                if (k == p_pin) continue;
                double v = 0.0;
                for (uint32_t q = Bxp[k]; q < Bxp[k + 1]; ++q)
                    if (!is_bd[Bxc[q]]) v += Bxv[q] * u[2u * static_cast<std::size_t>(Bxc[q]) + 0u];
                for (uint32_t q = Byp[k]; q < Byp[k + 1]; ++q)
                    if (!is_bd[Byc[q]]) v += Byv[q] * u[2u * static_cast<std::size_t>(Byc[q]) + 1u];
                y[k] = v;
            }
        };
        sys_ns.spmv_Bt = [S, is_bd, nv, p_pin](const std::vector<double>& pvec,
                                                std::vector<double>& y)
        {
            y.assign(2u * nv, 0.0);
            const auto& Bxp = S->BxT.row_ptr(); const auto& Bxc = S->BxT.col_idx(); const auto& Bxv = S->BxT.values();
            const auto& Byp = S->ByT.row_ptr(); const auto& Byc = S->ByT.col_idx(); const auto& Byv = S->ByT.values();
            for (std::size_t i = 0; i < nv; ++i)
            {
                if (is_bd[i]) continue;
                double sx = 0.0, sy = 0.0;
                for (uint32_t q = Bxp[i]; q < Bxp[i + 1]; ++q)
                    if (Bxc[q] != p_pin) sx += Bxv[q] * pvec[Bxc[q]];
                for (uint32_t q = Byp[i]; q < Byp[i + 1]; ++q)
                    if (Byc[q] != p_pin) sy += Byv[q] * pvec[Byc[q]];
                y[2u * i + 0u] = sx;
                y[2u * i + 1u] = sy;
            }
        };

        std::function<void(const std::vector<double>&, std::vector<double>&)> S_pre_use = S_pre;
        if (op.schur_kind == 1 && E_pre && vel_apply)
        {
            struct LWS { std::vector<double> w, yu, yv, yp; };
            auto lw = std::make_shared<LWS>();
            lw->w.resize(np);  lw->yu.resize(2u * nv);
            lw->yv.resize(2u * nv); lw->yp.resize(np);
            S_pre_use = [E_pre, &vel_apply, S, is_bd, mv_inv, p_pin, nv, np, lw](
                const std::vector<double>& r, std::vector<double>& z)
            {
                LWS& W = *lw;
                // 1) w = E⁻¹ r
                E_pre(r, W.w);
                // 2) yu = Bᵀ w   (受约束: 边界速度行置 0, 跳过 pin 列)
                W.yu.assign(2u * nv, 0.0);
                {
                    const auto& Xp = S->BxT.row_ptr(); const auto& Xc = S->BxT.col_idx(); const auto& Xv = S->BxT.values();
                    const auto& Yp = S->ByT.row_ptr(); const auto& Yc = S->ByT.col_idx(); const auto& Yv = S->ByT.values();
                    for (std::size_t i = 0; i < nv; ++i)
                    {
                        if (is_bd[i]) continue;
                        double sx = 0.0, sy = 0.0;
                        for (uint32_t q = Xp[i]; q < Xp[i + 1]; ++q)
                            if (Xc[q] != p_pin) sx += Xv[q] * W.w[Xc[q]];
                        for (uint32_t q = Yp[i]; q < Yp[i + 1]; ++q)
                            if (Yc[q] != p_pin) sy += Yv[q] * W.w[Yc[q]];
                        W.yu[2u * i + 0u] = sx;  W.yu[2u * i + 1u] = sy;
                    }
                }
                // 3) yu = Mv⁻¹ yu
                for (std::size_t i = 0; i < nv; ++i)
                { W.yu[2u * i + 0u] *= mv_inv[i];  W.yu[2u * i + 1u] *= mv_inv[i]; }
                // 4) yv = A yu  (完整的速度块, 含 C 与 G)
                vel_apply(W.yu, W.yv);
                // 5) yv = Mv⁻¹ yv
                for (std::size_t i = 0; i < nv; ++i)
                { W.yv[2u * i + 0u] *= mv_inv[i];  W.yv[2u * i + 1u] *= mv_inv[i]; }
                // 6) yp = B yv  (受约束: 跳过 pin 行)
                W.yp.assign(np, 0.0);
                {
                    const auto& Xp = S->Bx.row_ptr(); const auto& Xc = S->Bx.col_idx(); const auto& Xv = S->Bx.values();
                    const auto& Yp = S->By.row_ptr(); const auto& Yc = S->By.col_idx(); const auto& Yv = S->By.values();
                    for (std::size_t k = 0; k < np; ++k)
                    {
                        if (k == p_pin) continue;
                        double acc = 0.0;
                        for (uint32_t q = Xp[k]; q < Xp[k + 1]; ++q)
                            if (!is_bd[Xc[q]]) acc += Xv[q] * W.yv[2u * Xc[q] + 0u];
                        for (uint32_t q = Yp[k]; q < Yp[k + 1]; ++q)
                            if (!is_bd[Yc[q]]) acc += Yv[q] * W.yv[2u * Yc[q] + 1u];
                        W.yp[k] = acc;
                    }
                }
                // 7) z = E⁻¹ yp
                E_pre(W.yp, z);
                // 8) pin 行是全系统的单位行 -> 与 mass 版同样的处理
                if (p_pin < z.size()) z[p_pin] = -r[p_pin];
            };
        }

        auto pre = block::make_schur_preconditioner(sys_ns, A_pre_use, S_pre_use);
        const auto TP1 = std::chrono::steady_clock::now();
        t_pre_total += std::chrono::duration<double, std::milli>(TP1 - TP0).count();

        // ---- (7) 线性求解 ----
        // 迭代上限必须与 restart 解耦: restart*20 意味着 restart 调大时
        // 上限也跟着涨, 掩盖了重启惩罚; 固定上限才能看出真实的停滞。
        const int max_iter = op.max_lin;
        const auto [delta, its] = sparse::SMatrix::gmres(
            J, Jrhs, pre, op.restart, eta, max_iter);
        lin_total += its;

        // GMRES 的迭代数等于 max_iter 不能区分“恰好末步收敛”和“打满”；
        // 用 J*delta-b 的真实相对残差作唯一判据，也避免把失败方向更新进 Newton。
        std::vector<double> Jdelta;
        sparse::SMatrix::spmv_into(J, delta, Jdelta);
        double lin_nb2 = 0.0, lin_nr2 = 0.0;
        for (std::size_t k = 0; k < Jrhs.size(); ++k)
        {
            lin_nb2 += Jrhs[k] * Jrhs[k];
            const double d = Jrhs[k] - Jdelta[k];
            lin_nr2 += d * d;
        }
        const double lin_rel = (lin_nb2 > 0.0) ? std::sqrt(lin_nr2 / lin_nb2) : 0.0;
        const bool lin_ok = lin_rel <= 1.05 * eta + 1e-14;
        // 未达标不一定要把 Newton 掐死。GMRES 打满上限所得到的方向, 只要真实
        // 相对残差还小于 0.1, 仍然是一个能下降的方向; 直接放弃会把
        // “预条件器不够好”误报成“非线性不收敛”。真正的兜底是
        // eta_floor (下一步不再要求更紧的精度) + 停滞判据。
        const bool lin_usable = lin_ok || (lin_rel < 0.1);

        if (op.verbose)
            std::cout << "              GMRES iters = " << std::setw(4) << its
                      << "   eta = " << std::scientific << std::setprecision(2) << eta
                      << "   true-rel = " << lin_rel
                      << (lin_ok ? ""
                                 : (lin_usable ? "   [未达标, 但方向可用]"
                                               : "   [未达标!]")) << "\n";

        if (!lin_usable)
        {
            // 方向已经不可靠: 保留当前非线性迭代点。高 Re 首档直接
            // 从零场起步时应改用 Re 延拓，而不是盲目增加 Newton 步数。
            if (op.verbose)
                std::cout << "              线性方向已不可靠，停止本档 Newton（解未更新）\n";
            break;
        }
        if (!lin_ok)
        {
            // 本步只做到了 lin_rel, 下一步至少不要求比它更紧的精度。
            const double f_new = std::max(op.lin_tol_min, 0.5 * lin_rel);
            ++n_linfail;
            if (f_new > eta_floor)
            {
                eta_floor = f_new;
                if (op.verbose)
                    std::cout << "              接受该方向; 后续 eta 下界抬到 "
                              << std::scientific << std::setprecision(2) << eta_floor << "\n";
            }
        }

        // ---- (8) 更新 ----
        for (std::size_t i = 0; i < nv; ++i)
        {
            ux[i] += delta[2u * i + 0u];
            uy[i] += delta[2u * i + 1u];
        }
        for (std::size_t k = 0; k < np; ++k) pp[k] += delta[2u * nv + k];

        R.newton = it + 1;
        nr_prev = nr;

        // 连续两步的线性都没达到强制项 => 已经在该预条件器的精度地板上。
        // 允许两步而不是一步，是因为“未达标但可用”的方向往往已经足以
        // 把残差再压一个量级 (不精确 Newton: ‖R_{k+1}‖ ≲ η‖R_k‖ + ‖R_k‖²),
        // 这一步常常就是跃过收敛阈值的那一步。
        if (n_linfail >= 2)
        {
            if (op.verbose)
                std::cout << "              连续两步线性求解未达到强制项, "
                             "已到该预条件器的精度地板, 停止本档\n";
            break;
        }
    }

    const auto T2 = std::chrono::steady_clock::now();

    // ---- 误差 / 散度 ----
    R.lin_its = lin_total;
    R.ux = ux;  R.uy = uy;  R.pp = pp;

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
    R.div_l2 = std::sqrt(div2);

    if constexpr (Case::has_exact)
    {
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
        R.e_u_l2 = eu.first;  R.e_u_h1 = eu.second;  R.e_p_l2 = ep.first;
        R.e_p_l2_mf = fem::compute_error_mean_free<1>(
            m, dm_p, pp, [](double x, double y) { return Case::p(x, y); }, 5);
    }

    auto ms = [](std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); };
    R.t_asm   = ms(T0, T1);
    R.t_pre   = t_pre_total;
    R.t_solve = std::max(0.0, ms(T1, T2) - t_pre_total);
    return R;
}

}   // namespace

int main(int argc, char** argv)
{
    std::string           casename = "cavity";
    std::vector<uint32_t> nxs { 32u };
    std::vector<double>   res { 100.0, 200.0, 400.0, 700.0, 1000.0 };
    NSOpt                 op;
    double                ghia_tol = 0.02;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&](void) -> std::string
        {
            if (i + 1 >= argc) { std::cerr << "参数 " << a << " 缺少取值\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--case")            casename = next();
        else if (a == "--nx")
        {
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
                    if (!one.empty()) nxs.push_back(static_cast<uint32_t>(std::stoul(one)));
                    if (c == std::string::npos) break;
                    pos = c + 1;
                }
            }
        }
        else if (a == "--re")
        {
            res.clear();
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
                    if (!one.empty()) res.push_back(std::stod(one));
                    if (c == std::string::npos) break;
                    pos = c + 1;
                }
            }
        }
        else if (a == "--amg-drop")       op.drop = std::stod(next());
        else if (a == "--amg-cap")        op.cap = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--amg-coarse")     op.coarse = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--restart")        op.restart = std::stoi(next());
        else if (a == "--max-newton")     op.max_newton = std::stoi(next());
        else if (a == "--newton-tol")     op.newton_tol = std::stod(next());
        else if (a == "--lin-tol0")       op.lin_tol0 = std::stod(next());
        else if (a == "--lin-tol-min")    op.lin_tol_min = std::stod(next());
        else if (a == "--ew-alpha")       op.ew_alpha = std::stod(next());
        else if (a == "--max-lin")        op.max_lin = std::stoi(next());
        else if (a == "--lsc-cycles")     op.lsc_cycles = std::stoi(next());
        else if (a == "--schur")
        {
            const std::string v = next();
            //   lsc  = BFBt/最小二乘交换子, Oseen 型 (默认)
            //   mass = nu·Mp⁻¹,          Stokes 型 (对照)
            if      (v == "lsc")  { op.schur_kind = 1; }
            else if (v == "mass") { op.schur_kind = 0; }
            else { std::cerr << "--schur 只接受 lsc | mass, 收到: " << v << "\n"; return 2; }
        }
        else if (a == "--ghia-tol")       ghia_tol = std::stod(next());
        else if (a == "--amg-verbose")    op.amg_verbose = true;
        else if (a == "--verbose")        op.verbose = true;
        else if (a == "--vel-pre")
        {
            const std::string v = next();
            //   cd     = AMG 建在 nuK + C 上 (默认)
            //   stokes = AMG 只建在 nuK 上 (对照: 看对流项进预条件值不值)
            //   cd   与 cheb 对应“光滑子 A/B” (算子都是 nuK + C);
            //   cheb 与 stokes 对应“算子 A/B” (光滑子都是 Chebyshev)。
            if      (v == "cd")     { op.pre_on_cd = true;  op.vel_kind = 0; }
            else if (v == "jac")    { op.pre_on_cd = true;  op.vel_kind = 0; }
            else if (v == "stokes") { op.pre_on_cd = false; }   // 光滑子保持默认
            else if (v == "cheb")   { op.pre_on_cd = true;  op.vel_kind = 1; }
            else if (v == "blk")    { op.pre_on_cd = true;  op.vel_kind = 2; }
            else if (v == "ilu")    { op.pre_on_cd = true;  op.vel_ilu = true; }
            else { std::cerr << "--vel-pre 只接受 cd | jac | stokes | cheb | blk | ilu, 收到: "
                             << v << "\n"; return 2; }
        }
        else if (a == "--tau-scale")      op.tau_scale  = std::stod(next());
        else if (a == "--supg")
        {
            const std::string v = next();
            if      (v == "on")  { op.tau_scale = 1.0; }
            else if (v == "off") { op.tau_scale = 0.0; }
            else { std::cerr << "--supg 只接受 on | off, 收到: " << v << "\n"; return 2; }
        }
        else if (a == "--lid")
        {
            const std::string v = next();
            if      (v == "smooth") { op.lid_smooth = true;  }
            else if (v == "hard")   { op.lid_smooth = false; }
            else { std::cerr << "--lid 只接受 smooth | hard, 收到: " << v << "\n"; return 2; }
        }
        else if (a == "--amg-cheb-deg")   op.cheb_deg   = std::stoi(next());
        else if (a == "--amg-cheb-ratio") op.cheb_ratio = std::stod(next());
        else { std::cerr << "未知参数: " << a << "\n"; return 2; }
    }

    if (nxs.empty()) nxs = { 32u };
    // cavity 默认做 Re 延拓: 100 -> 200 -> 400 -> 700 -> 1000。中间档只判 Newton
    // 收敛 (Ghia 没有对应数据), 但它们的解是下一档的初值 —— 没有
    // 这些中间档, Re=1000 会直接从仅满足边界条件的零内场起步,
    // 首步线性就会打满。（原来默认只有 100, 与 NSRUN 帮助不一致）
    if (res.empty()) res = { 100.0, 200.0, 400.0, 700.0, 1000.0 };
    if (casename == "mms") res = { 1.0 };

    std::cout << "=== M2: Taylor-Hood (P2/P1) 稳态 Navier-Stokes ===\n";
    std::cout << "  算例 = " << casename << "   (Newton + 块 Schur 预条件, GMRES("
              << op.restart << "))\n";
    std::cout << "  速度块预条件: "
              << (op.vel_ilu ? "ILU(0) on 完整 A = nuK + C + G"
                             : (op.vel_kind == 2 ? "BlockAMG on nuK + C"
                               : (std::string("标量 AMG on ") +
                                  (op.pre_on_cd ? "nuK + C" : "nuK (不含对流)") +
                                  (op.vel_kind == 1 ? " + Chebyshev 光滑"
                                                    : " + 阻尼 Jacobi 光滑"))))
              << "\n";
    std::cout << "  Schur 预条件: "
              << (op.schur_kind == 1 ? "LSC/BFBt  E⁻¹(B Mv⁻¹ A Mv⁻¹ Bᵀ)E⁻¹,  E = B Mv⁻¹ Bᵀ"
                                     : "nu·Mp⁻¹ (lumped) —— Stokes 型, 不含对流")
              << "\n";
    if (casename == "cavity")
        Cavity::lid_delta = op.lid_smooth ? (2.0 / static_cast<double>(nxs[0])) : 0.0;

    std::cout << "  稳定项: " << (op.tau_scale > 0.0
        ? (std::string("SUPG   tau = ") + std::to_string(op.tau_scale) +
           " / sqrt((2|u|/h)^2 + (4 nu/h^2)^2)   [流线扩散进 Ad, 压力耦合进 B^T]")
        : std::string("无 (纯 Galerkin)")) << "\n";
    if (casename == "cavity")
        std::cout << "  顶盖: " << (op.lid_smooth
            ? (std::string("两端 smoothstep 过渡, 宽度 = 2 个单元 = ")
               + std::to_string(Cavity::lid_delta))
            : std::string("硬设置 u=(1,0) (Ghia 原始提法, 角点有奇性)")) << "\n";

    std::cout << "  强制项: Eisenstat-Walker α = " << std::fixed << std::setprecision(3)
              << op.ew_alpha << ",  η ∈ [" << std::scientific << std::setprecision(1)
              << op.lin_tol_min << ", 1e-1];  单次线性上限 = " << op.max_lin << " 次\n";
    std::cout << "\n";

    int rc = 0;

    if (casename == "mms")
    {
        // ---------------- MMS: 收敛阶 ----------------
        struct Row { uint32_t nx; NSResult r; };
        std::vector<Row> rows;
        for (const uint32_t nx : nxs)
        {
            std::vector<double> ux, uy, pp;
            rows.push_back({ nx, run_ns<MMSNS>(nx, 1.0, ux, uy, pp, op) });
        }
        std::cout << "   nx     ndof   Newton   lin-its   装配(ms)  预条件(ms)  求解(ms)"
                     "        u-L2      rate        u-H1      rate        p-L2*     rate      ‖∇·u‖     ‖R‖\n";
        for (std::size_t t = 0; t < rows.size(); ++t)
        {
            const NSResult& R = rows[t].r;
            std::cout << std::setw(5) << rows[t].nx
                      << std::setw(9) << R.ndof
                      << std::setw(9) << R.newton
                      << std::setw(10) << R.lin_its
                      << std::setw(11) << std::fixed << std::setprecision(1) << R.t_asm
                      << std::setw(12) << R.t_pre
                      << std::setw(11) << R.t_solve;
            std::cout << "   " << std::scientific << std::setprecision(4) << R.e_u_l2;
            if (t > 0) std::cout << "  " << std::fixed << std::setprecision(2)
                                 << std::log2(rows[t-1].r.e_u_l2 / R.e_u_l2);
            else       std::cout << "     --";
            std::cout << "   " << std::scientific << std::setprecision(4) << R.e_u_h1;
            if (t > 0) std::cout << "  " << std::fixed << std::setprecision(2)
                                 << std::log2(rows[t-1].r.e_u_h1 / R.e_u_h1);
            else       std::cout << "     --";
            std::cout << "   " << std::scientific << std::setprecision(4) << R.e_p_l2_mf;
            if (t > 0) std::cout << "  " << std::fixed << std::setprecision(2)
                                 << std::log2(rows[t-1].r.e_p_l2_mf / R.e_p_l2_mf);
            else       std::cout << "     --";
            std::cout << "   " << std::scientific << std::setprecision(3) << R.div_l2
                      << "   " << std::scientific << std::setprecision(3) << R.res_final
                      << (R.converged ? "  " : " !") << "\n";
        }
        std::cout << "\n";
        if (rows.size() < 2)
        {
            std::cout << "  只有一档网格, 无法算收敛阶  ->  SKIP\n";
            return 0;
        }
        const std::size_t L = rows.size() - 1;
        const double r_u2 = std::log2(rows[L-1].r.e_u_l2    / rows[L].r.e_u_l2);
        const double r_u1 = std::log2(rows[L-1].r.e_u_h1    / rows[L].r.e_u_h1);
        const double r_p2 = std::log2(rows[L-1].r.e_p_l2_mf / rows[L].r.e_p_l2_mf);
        // Taylor-Hood P2/P1 的一般理论阶是 u-L2=3、u-H1=2、p-L2=2。
        // 验证的目标是抓住降阶，因此这里只设"理论阶 - 0.35"的下界；
        // 结构化、对称网格上的压力超收敛（如 3.42 -> 2.68）是允许的，
        // 不能因为观测阶高于 2 就判失败。非有限值仍必须失败。
        auto at_least = [](double got, double want)
        {
            return std::isfinite(got) && got >= want - 0.35;
        };
        const bool all_converged = std::all_of(
            rows.begin(), rows.end(), [](const Row& row) { return row.r.converged; });
        const bool ok = at_least(r_u2, 3.0) && at_least(r_u1, 2.0)
                      && at_least(r_p2, 2.0) && all_converged;
        std::cout << "  末档阶数: 速度 L2 = " << std::fixed << std::setprecision(2) << r_u2
                  << " (理论 3.00),  速度 H1 = " << r_u1 << " (理论 2.00),  压力 L2* = "
                  << r_p2 << " (理论 2.00; 允许超收敛)   ->  " << (ok ? "PASS" : "FAIL") << "\n";
        rc = ok ? 0 : 1;
        return rc;
    }

    // ---------------- cavity: Ghia 基准 ----------------
    if (casename != "cavity")
    {
        std::cerr << "--case 只接受 cavity | mms, 收到: " << casename << "\n";
        return 2;
    }
    if (!res.empty() && res.front() > 100.0 + 1e-12)
    {
        std::cout << "  [提示] 首档 Re=" << res.front()
                  << " 将从仅满足边界条件的零内场直接 Newton；高 Re 通常应使用"
                     " --re 100,400,1000 做延拓。\n\n";
    }
    std::vector<double> ux, uy, pp;         // Re 延拓: 上一档的解作下一档初值
    for (const double re : res)
    {
        const NSResult R = run_ns<Cavity>(nxs[0], re, ux, uy, pp, op);

        std::cout << "--- Re = " << std::fixed << std::setprecision(0) << re
                  << "   (nu = " << std::scientific << std::setprecision(3) << 1.0 / re
                  << ", nx = " << nxs[0] << ", ndof = " << R.ndof << ") ---\n";
        std::cout << "  Newton = " << R.newton << " 步,  累计 GMRES = " << R.lin_its
                  << " 次,  装配 " << std::fixed << std::setprecision(1) << R.t_asm
                  << " ms,  预条件 " << R.t_pre << " ms,  求解 " << R.t_solve
                  << " ms,  ‖R‖ = " << std::scientific << std::setprecision(3)
                  << R.res_final << (R.converged ? "" : "   [未收敛!]") << "\n";
        std::cout << "  ‖∇·u_h‖_L2 = " << std::scientific << std::setprecision(3)
                  << R.div_l2 << "\n";

        // ---- 与 Ghia 对比 ----
        const fem::Mesh   m    = fem::make_unit_square_mesh(nxs[0], nxs[0]);
        const fem::DofMap dm_v = fem::make_dofmap(m, 2);

        const int tag = (std::fabs(re - 100.0) < 1.0) ? 100
                      : (std::fabs(re - 400.0) < 1.0) ? 400 : 1000;
        const double* Uref = (tag == 100) ? kGhiaU100 : (tag == 400) ? kGhiaU400 : kGhiaU1000;
        const double* Vref = (tag == 100) ? kGhiaV100 : (tag == 400) ? kGhiaV400 : kGhiaV1000;
        // Re 延拓的中间档 (200 / 700 之类) 没有基准数据: 只判定 Newton 是否
        // 收敛。拿 Re=200 的解去比 Re=100 的表会得出一个"假 FAIL", 与求解器
        // 好坏无关 —— 那只是基准数据缺失。
        const bool do_ghia = (std::fabs(re - 100.0) < 1.0)
                           || (std::fabs(re - 400.0) < 1.0)
                           || (std::fabs(re - 1000.0) < 1.0);
        if (!do_ghia)
        {
            std::cout << "  (Re = " << re << " 是 Re 延拓的中间档, Ghia 没有该档数据:"
                         " 只判定 Newton 是否收敛, 不做剖面对比)\n";
            std::cout << "  判定 ->  " << (R.converged ? "PASS" : "FAIL") << "\n\n";
            if (!R.converged) rc = 1;
            continue;
        }

        double sum_u = 0.0, max_u = 0.0; int nu_cnt = 0;
        double sum_v = 0.0, max_v = 0.0; int nv_cnt = 0;

        std::cout << "      y       u(0.5,y)     Ghia      Δ        "
                     "      x       v(x,0.5)     Ghia      Δ\n";
        for (int i = 0; i < kGhiaN; ++i)
        {
            double uv = 0.0;
            const bool oku = fem::eval_fe_at<2>(m, dm_v, R.ux, 0.5, kGhiaY[i], uv);
            double vv = 0.0;
            const bool okv = fem::eval_fe_at<2>(m, dm_v, R.uy, kGhiaX[i], 0.5, vv);

            const bool skip_v = ghia_skip(tag, kGhiaX[i]);
            double du = 0.0, dv = 0.0;
            if (oku) { du = uv - Uref[i]; sum_u += du * du; ++nu_cnt;
                       if (std::fabs(du) > max_u) max_u = std::fabs(du); }
            if (okv && !skip_v) { dv = vv - Vref[i]; sum_v += dv * dv; ++nv_cnt;
                       if (std::fabs(dv) > max_v) max_v = std::fabs(dv); }

            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(8) << kGhiaY[i]
                      << std::setw(13) << uv << std::setw(11) << Uref[i]
                      << std::setw(11) << (oku ? du : 0.0)
                      << "   " << std::setw(8) << kGhiaX[i]
                      << std::setw(13) << vv << std::setw(11) << Vref[i]
                      << std::setw(11) << (okv && !skip_v ? dv : 0.0)
                      << (skip_v ? "  (排除)" : "") << "\n";
        }
        const double rms_u = (nu_cnt > 0) ? std::sqrt(sum_u / nu_cnt) : 0.0;
        const double rms_v = (nv_cnt > 0) ? std::sqrt(sum_v / nv_cnt) : 0.0;
        const double rms   = std::sqrt(0.5 * (rms_u * rms_u + rms_v * rms_v));
        std::cout << "  Ghia Re=" << tag << ": RMS(u) = " << std::scientific
                  << std::setprecision(3) << rms_u << ",  max|Δu| = " << max_u
                  << ";   RMS(v) = " << rms_v << ",  max|Δv| = " << max_v
                  << ";   合成 RMS = " << rms
                  << "   (阈值 " << ghia_tol << ")\n";

        const bool ok = R.converged && (rms < ghia_tol);
        std::cout << "  判定 ->  " << (ok ? "PASS" : "FAIL") << "\n\n";
        if (!ok) rc = 1;
    }
    return rc;
}
