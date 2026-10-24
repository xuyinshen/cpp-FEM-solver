// ============================================================================
// M0: FEM Poisson + 制造解 (MMS) 收敛阶验证
// ----------------------------------------------------------------------------
//   算例  -Δu = f  于单位正方形,  u = g 于四边
//   真解  u = sin(A x + C) * cos(B y + D),  A = 1.3π, B = 0.7π, C = 0.4, D = -0.2
//         f = (A² + B²) sin(A x + C) cos(B y + D)
//         ∇u = ( A cos(Ax+C) cos(By+D),  -B sin(Ax+C) sin(By+D) )
//
//   【为什么选这个真解, 而不是 sin(πx)sin(πy)】
//     (1) 边界值非零 —— 顺带把"非齐次 Dirichlet 消元"这条路径也验了。用齐次
//         真解时边界值全是 0, 消元里的 b(j) -= A(j,i)*g_i 这一项是恒等通过的,
//         bug 会被完全掩盖。
//     (2) 故意取非整数倍频率 + 相位偏移, 破坏解与网格的对称性。对称解在结构
//         网格上会出现超收敛, 实测阶数比理论高一阶, 看着"更好"实则掩盖问题。
//
//   【为什么第一个算例必须是 MMS 而不是方腔流】
//     方腔流没有解析解, 只能对 Ghia 基准数据。一旦离散 / 装配 / 边界条件有 bug,
//     它会和压力-速度耦合、非线性、Re 延拓混在一起, 定位成本极高。MMS 把离散层
//     单独隔离出来验: L2 与 H1 都达到理论阶 (P1: 2 / 1, P2: 3 / 2), 就说明
//     单元、等参映射、求积、P2 边自由度编号、Dirichlet 消元全部正确。
//     这一步过了, 后面每一层的 bug 都容易隔离。
//
//   用法:
//     ./bin/fem_test_app                       默认 AMG 预处理, P1, nx = 8,16,32
//     ./bin/fem_test_app --order 2             跑 P2
//     ./bin/fem_test_app --order 0             P1 与 P2 都跑
//     ./bin/fem_test_app ilu0                  换预处理器 (位置参数)
//     ./bin/fem_test_app --amg-drop 0 --amg-cap 0    关闭 Galerkin 稀疏化 (对照)
//     ./bin/fem_test_app --amg-verbose          打印 AMG 各层规模与复杂度
//     ./bin/fem_test_app --amg-coarse 50         调小最粗层阈值 (默认 300)
//     ./bin/fem_test_app --vtk                  输出 VTK
//     ./bin/fem_test_app --gmsh cavity.msh      试读 Gmsh 网格
// ============================================================================
#include "math/array.hpp"

#include "fem/assembler.hpp"
#include "fem/boundary.hpp"
#include "fem/dofmap.hpp"
#include "fem/mesh.hpp"
#include "io/vtk.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

// 制造解参数 (非整数倍频率 + 相位偏移, 见文件头注释)
constexpr double kA = 1.3 * kPi;
constexpr double kB = 0.7 * kPi;
constexpr double kC = 0.4;
constexpr double kD = -0.2;

double exact_u(double x, double y)
{
    return std::sin(kA * x + kC) * std::cos(kB * y + kD);
}

// -Δu = (A² + B²) sin(Ax+C) cos(By+D)
double exact_f(double x, double y)
{
    return (kA * kA + kB * kB) * std::sin(kA * x + kC) * std::cos(kB * y + kD);
}

void exact_grad(double x, double y, double& gx, double& gy)
{
    gx =  kA * std::cos(kA * x + kC) * std::cos(kB * y + kD);
    gy = -kB * std::sin(kA * x + kC) * std::sin(kB * y + kD);
}

struct Case
{
    int         nx    = 0;
    std::size_t ndof  = 0;
    std::size_t nnz   = 0;
    int         iters = 0;
    double      asm_ms = 0.0;      // 网格 + 自由度 + 稀疏模式 + 装配 + Dirichlet
    double      re_ms  = 0.0;      // 复用同一套 (稀疏模式, 位置表) 再装一遍的耗时
    double      pc_ms  = 0.0;      // 预处理器构造 (AMG setup); 非 AMG 时为 0
    double      it_ms  = 0.0;      // 纯 Krylov 迭代 (不含任何 setup)
    double      l2 = 0.0;
    double      h1 = 0.0;
    double      mass_area = 0.0;   // 1ᵀ M 1, 应等于区域面积 1
};

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <int Order>
Case run_case(int nx, const std::string& precond,
              double amg_drop, uint32_t amg_cap,
              bool amg_verbose, bool dump_vtk, uint32_t amg_coarse)
{
    using namespace fem;

    // 【进度探针】写 cerr: cerr 默认 unitbuf (无缓冲), 进程段错误时这条一定
    //   已经落到终端。sweep 一次跑六个算例、而 stdout 的报告又被攒到最后才
    //   打印, 没有它就只知道"段错误"而不知道崩在哪一档、哪个阶段。
    std::cerr << "[diag] P" << Order << " nx=" << nx << " precond=" << precond
              << " drop=" << amg_drop << " cap=" << amg_cap
              << " coarse=" << amg_coarse << " : start\n";

    const Clock::time_point t_begin = Clock::now();

    Mesh       m   = make_unit_square_mesh(static_cast<uint32_t>(nx),
                                           static_cast<uint32_t>(nx));
    DofMap     dm  = make_dofmap(m, Order);
    CsrPattern pat = build_pattern(m, dm);
    // 单元 -> CSR 位置的预计算表。建表一次, 之后每次装配只查表不再二分。
    //   非线性 (Newton) / 瞬态每一步都要重装 Jacobian, 这张表就是那时候的收益。
    ElemPos    ep  = build_elem_pos(m, dm, pat);

    std::vector<double> Aval, Mval, rhs;
    assemble_stiffness<Order>(m, dm, pat, [](double, double) { return 1.0; }, Aval, 0, &ep.p);
    assemble_mass<Order>(m, dm, pat, [](double, double) { return 1.0; }, Mval, 0, &ep.p);
    assemble_load<Order>(m, dm, exact_f, rhs);

    // Dirichlet: 四条边施加真解 (本例边界值非零, 消元路径被真正走到)
    const std::vector<std::pair<double, double>> co = dof_coords(m, dm);
    const std::vector<uint32_t>                  bd = boundary_dofs(m, dm, 0);
    std::vector<double> g(bd.size());
    for (std::size_t i = 0; i < bd.size(); ++i)
        g[i] = exact_u(co[bd[i]].first, co[bd[i]].second);

    apply_dirichlet(static_cast<uint32_t>(dm.n_dof), pat.row_ptr, pat.col_idx,
                    Aval, rhs, bd, g);

    const sparse::SMatrix A(pat.n, pat.n, pat.row_ptr, pat.col_idx, Aval);
    const sparse::SMatrix M(pat.n, pat.n, pat.row_ptr, pat.col_idx, Mval);

    const double setup_ms = ms_since(t_begin);

    // ---- 复用同一套 (稀疏模式, 位置表) 再装一遍 ----
    //   这正是 Newton 迭代 / 时间推进里每一步的代价: 模式与位置表都不用重算,
    //   只有数值变。它和"装配(ms)"的差就是建模式 + 建位置表的固定开销。
    const Clock::time_point t_re = Clock::now();
    {
        std::vector<double> A2, M2, b2;
        assemble_stiffness<Order>(m, dm, pat, [](double, double) { return 1.0; }, A2, 0, &ep.p);
        assemble_mass<Order>(m, dm, pat, [](double, double) { return 1.0; }, M2, 0, &ep.p);
        assemble_load<Order>(m, dm, exact_f, b2);
    }
    const double re_ms = ms_since(t_re);

    // ---- 求解 ----
    //   【计时必须三分: 装配 / 预处理器构造 / 纯迭代】
    //   AMG 的 setup (层次构造 + 最粗层稠密 LU 分解) 常常比迭代本身还贵。
    //   混成一个 "solve" 数字会让 ms/iter 完全失真 —— 上一版日志里
    //   P1 nx=8 报 "iters=0, solve=4.6ms" 就是这么来的: 4.6ms 全是 setup,
    //   一步迭代都没算进去。
    double pc_ms = 0.0, it_ms = 0.0;
    std::vector<double> u;
    int                 iters = 0;

    const Clock::time_point t_pc = Clock::now();
    if (precond == "amg")
    {
        // 走"外部预处理器"重载, 这样 drop/cap/coarse 能直接透传进去做 A/B 对照
        std::cerr << "[diag]   amg setup (n=" << pat.n << ") ...\n";
        auto pc = sparse::SMatrix::make_amg_preconditioner(A, amg_drop, amg_cap,
                                                          amg_verbose, amg_coarse);
        pc_ms = ms_since(t_pc);
        std::cerr << "[diag]   amg setup ok (" << pc_ms << " ms)\n";

        const Clock::time_point t_it = Clock::now();
        auto r  = sparse::SMatrix::pcg(A, rhs, pc, 1e-12, 8000);
        u     = std::get<0>(r);
        iters = std::get<1>(r);
        it_ms = ms_since(t_it);
        std::cerr << "[diag]   pcg done iters=" << iters << "\n";
    }
    else
    {
        std::cerr << "[diag]   pcg(" << precond << " n=" << pat.n << ") ...\n";
        auto r  = sparse::SMatrix::pcg(A, rhs, precond, 1e-12, 8000);
        u     = std::get<0>(r);
        iters = std::get<1>(r);
        it_ms = ms_since(t_pc);
    }

    // ---- 误差: 按求积算真实 L2 / H1, 不用 nodal 差值的 eᵀMe / eᵀAe ----
    const std::pair<double, double> err = compute_errors<Order>(m, dm, u, exact_u, exact_grad);

    // ---- 自检: 1ᵀ M 1 应等于区域面积 (单位正方形 = 1) ----
    const std::vector<double> ones(dm.n_dof, 1.0);
    const std::vector<double> Mones = sparse::SMatrix::spmv(M, ones);
    double area = 0.0;
    for (double v : Mones) area += v;

    if (dump_vtk)
        write_vtk("poisson_p" + std::to_string(Order) + "_" + std::to_string(nx) + ".vtk",
                  m, dm, u, "u");

    Case c;
    c.nx        = nx;
    c.ndof      = dm.n_dof;
    c.nnz       = pat.col_idx.size();
    c.iters     = iters;
    c.asm_ms    = setup_ms;
    c.re_ms     = re_ms;
    c.pc_ms     = pc_ms;
    c.it_ms     = it_ms;
    c.l2        = err.first;
    c.h1        = err.second;
    c.mass_area = area;
    return c;
}

std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    std::string cur;
    for (char ch : s)
    {
        if (ch == ',')
        {
            if (!cur.empty()) { out.push_back(std::stoi(cur)); cur.clear(); }
        }
        else cur.push_back(ch);
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
}

template <int Order>
void report(const std::vector<Case>& cs)
{
    const double expect_l2 = static_cast<double>(Order) + 1.0;
    const double expect_h1 = static_cast<double>(Order);

    std::cout << "  P" << Order << "  (理论阶: L2 = " << expect_l2
              << ", H1 = " << expect_h1 << ")\n";
    std::cout << "    nx      ndof   nnz/row   iters   装配(ms)  重装(ms)  预处理器(ms)"
              << "   迭代(ms)   ms/iter          L2      rate          H1      rate\n";
    std::cout << "      (装配 = 网格+模式+位置表+装配+约束; 重装 = 复用模式与位置表再装一遍,\n"
              << "       即 Newton / 时间步每一步的真实代价)\n";

    double prev_l2 = 0.0, prev_h1 = 0.0;
    double last_l2_rate = 0.0, last_h1_rate = 0.0;
    bool   first = true;

    for (const Case& c : cs)
    {
        const double rate_l2 = first ? 0.0 : std::log2(prev_l2 / c.l2);
        const double rate_h1 = first ? 0.0 : std::log2(prev_h1 / c.h1);
        if (!first) { last_l2_rate = rate_l2; last_h1_rate = rate_h1; }
        first = false;

        std::cout << "  " << std::setw(5) << c.nx
                  << "  " << std::setw(8) << c.ndof
                  << "  " << std::setw(7) << std::fixed << std::setprecision(2)
                  << (c.ndof ? static_cast<double>(c.nnz) / static_cast<double>(c.ndof) : 0.0)
                  << "  " << std::setw(7) << c.iters
                  << "  " << std::setw(9) << std::setprecision(1) << c.asm_ms
                  << "  " << std::setw(9) << c.re_ms
                  << "  " << std::setw(11) << c.pc_ms
                  << "  " << std::setw(9) << c.it_ms
                  << "  " << std::setw(8) << std::fixed << std::setprecision(3)
                  << c.it_ms / static_cast<double>(c.iters > 0 ? c.iters : 1)
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << c.l2
                  << "  " << std::setw(5) << std::fixed << std::setprecision(2)
                  << (rate_l2 > 0.0 ? rate_l2 : 0.0)
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << c.h1
                  << "  " << std::setw(5) << std::fixed << std::setprecision(2)
                  << (rate_h1 > 0.0 ? rate_h1 : 0.0)
                  << "\n";

        prev_l2 = c.l2;
        prev_h1 = c.h1;
    }

    // 只有一档网格时 rate 是构造出来的 0.00, 判它 FAIL 纯属误报
    //   (./FEMRUN raw --nx 32 就是这种情形)。这种情况明确跳过阶数判定。
    const bool have_rate = (cs.size() >= 2);
    const bool ok = have_rate
                 && std::fabs(last_l2_rate - expect_l2) < 0.30
                 && std::fabs(last_h1_rate - expect_h1) < 0.30;
    const bool area_ok = cs.empty() ? false
                       : std::fabs(cs.back().mass_area - 1.0) < 1e-8;

    if (have_rate)
        std::cout << "    末档阶数: L2 = " << std::fixed << std::setprecision(2)
                  << last_l2_rate << " (期望 " << expect_l2 << "),  H1 = "
                  << last_h1_rate << " (期望 " << expect_h1 << ")   ->  "
                  << (ok ? "PASS" : "FAIL") << "\n";
    else
        std::cout << "    末档阶数: 只有一档网格, 无法算收敛阶 (--nx 至少给 2 个值)"
                  << "   ->  SKIP\n";
    std::cout << "    自检: 1ᵀ M 1 = " << std::setprecision(10)
              << (cs.empty() ? 0.0 : cs.back().mass_area)
              << " (应为 1)  ->  " << (area_ok ? "OK" : "FAIL") << "\n\n";
}

}   // namespace

int main(int argc, char** argv)
{
    std::string     precond = "amg";
    int             order   = 1;
    std::vector<int> nxs{ 8, 16, 32 };
    double          drop    = 0.02;
    uint32_t        cap     = 60;
    uint32_t        coarse  = 300;
    bool            verbose = false;
    bool            dump    = false;
    std::string     gmsh_path;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string
        {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };

        if      (a == "--order")       order = std::stoi(next());
        else if (a == "--nx")          nxs   = parse_int_list(next());
        else if (a == "--precond")     precond = next();
        else if (a == "--amg-drop")    drop  = std::stod(next());
        else if (a == "--amg-cap")     cap   = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--amg-coarse")  coarse = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--amg-verbose") verbose = true;
        else if (a == "--vtk")         dump  = true;
        else if (a == "--gmsh")        gmsh_path = next();
        else if (!a.empty() && a[0] != '-') precond = a;     // 位置参数 = 预处理器
        else
        {
            std::cerr << "未知参数: " << a << "\n";
            return 1;
        }
    }

    if (!gmsh_path.empty())
    {
        try
        {
            const fem::Mesh gm = fem::read_gmsh(gmsh_path);
            std::cout << "[gmsh] " << gmsh_path << ": 节点 " << gm.n_nodes()
                      << ", 三角形 " << gm.n_tris()
                      << ", 边界边 " << gm.n_bedges() << "\n\n";
        }
        catch (const std::exception& ex)
        {
            std::cerr << "[gmsh] 读入失败: " << ex.what() << "\n";
            return 1;
        }
    }

    std::cout << "=== M0: FEM Poisson + MMS (单位正方形, 非零 Dirichlet) ===\n";
    std::cout << "  真解 u = sin(1.3π x + 0.4) cos(0.7π y - 0.2)\n";
    std::cout << "  求解器 PCG, 预处理器 = " << precond;
    if (precond == "amg")
        std::cout << " (drop = " << drop << ", cap = " << cap
                  << (drop == 0.0 && cap == 0 ? "  <- 关闭 Galerkin 稀疏化" : "") << ")";
    std::cout << "\n";
    std::cout << "  计时三分: 装配 / 预处理器构造 / 纯迭代;  ms/iter = 迭代(ms) / iters\n";
    std::cout << "  最粗层阈值 coarse_thresh = " << coarse
              << "  (ndof <= 它时 AMG 只有一层, 退化为最粗层稠密 LU 直接解,\n"
              << "   此时 iters 会是 1~2, 不代表迭代法的收敛速度)\n\n";

    if (order == 1 || order == 0)
    {
        std::vector<Case> cs;
        for (int nx : nxs) cs.push_back(run_case<1>(nx, precond, drop, cap, verbose, dump, coarse));
        report<1>(cs);
    }
    if (order == 2 || order == 0)
    {
        std::vector<Case> cs;
        for (int nx : nxs) cs.push_back(run_case<2>(nx, precond, drop, cap, verbose, dump, coarse));
        report<2>(cs);
    }

    if (dump) std::cout << "  VTK 已写出到当前目录\n";
    return 0;
}
