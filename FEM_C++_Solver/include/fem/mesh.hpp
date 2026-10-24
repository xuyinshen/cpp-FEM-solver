#pragma once
// ============================================================================
// 二维三角网格
// ----------------------------------------------------------------------------
//   只存直线边三角形 (P1 几何)。P2 的场变量自由度由 dofmap.hpp 在边的
//   层面上再加一层, 网格本身不需要知道它。
//
//   边界用"边 + 物理标签"表示, 而不是"节点 + 标记":
//     - 边界条件是施加在边上的 (Neumann/Robin 都要沿边积分);
//     - 节点标记在角点处有歧义 (角点属于两条不同标签的边)。
//   边界标签的具体含义由用户提供:
//     make_unit_square_mesh 约定 1=下 2=右 3=上 4=左;
//     read_gmsh 直接用 .msh 里的物理组编号。
// ============================================================================
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fem
{

struct Mesh
{
    std::vector<double>   coords;    // 2 * n_nodes   -> (x, y)
    std::vector<uint32_t> tris;      // 3 * n_tris    -> 顶点编号 (逆时针)
    std::vector<uint32_t> bedges;    // 2 * n_bedges  -> 边的两个顶点
    std::vector<int>      btag;      //   n_bedges    -> 物理标签

    std::size_t n_nodes()  const { return coords.size() / 2; }
    std::size_t n_tris()   const { return tris.size() / 3; }
    std::size_t n_bedges() const { return btag.size(); }
};

// ---------------------------------------------------------------------------
// 单位正方形上的结构三角网格
//   nx x ny 个矩形单元, 每个切成两个直角三角形。
//   边界标签: 1=下(y=0)  2=右(x=1)  3=上(y=1)  4=左(x=0)
//   这是 M0 的主力: 网格尺寸可以精确按 2 倍细化, 收敛阶才有意义
//   (非结构网格每次细化的 h 不严格减半, 阶数会被污染)。
// ---------------------------------------------------------------------------
inline Mesh make_unit_square_mesh(uint32_t nx, uint32_t ny)
{
    if (nx == 0 || ny == 0) throw std::invalid_argument("网格划分数必须为正");

    const uint32_t nx1 = nx + 1, ny1 = ny + 1;
    const double   hx  = 1.0 / static_cast<double>(nx);
    const double   hy  = 1.0 / static_cast<double>(ny);

    Mesh m;
    m.coords.resize(2 * static_cast<std::size_t>(nx1) * ny1);
    for (uint32_t j = 0; j < ny1; ++j)
        for (uint32_t i = 0; i < nx1; ++i)
        {
            const std::size_t id = static_cast<std::size_t>(j) * nx1 + i;
            m.coords[2 * id + 0] = static_cast<double>(i) * hx;
            m.coords[2 * id + 1] = static_cast<double>(j) * hy;
        }

    m.tris.reserve(6 * static_cast<std::size_t>(nx) * ny);
    for (uint32_t j = 0; j < ny; ++j)
        for (uint32_t i = 0; i < nx; ++i)
        {
            const uint32_t n00 = j * nx1 + i;
            const uint32_t n10 = n00 + 1;
            const uint32_t n01 = n00 + nx1;
            const uint32_t n11 = n01 + 1;
            // 两个三角形都取逆时针 (detJ > 0):
            //   (0,0)-(h,0)-(h,h)   与   (0,0)-(h,h)-(0,h)
            m.tris.push_back(n00); m.tris.push_back(n10); m.tris.push_back(n11);
            m.tris.push_back(n00); m.tris.push_back(n11); m.tris.push_back(n01);
        }

    auto push_edge = [&](uint32_t a, uint32_t b, int tag)
    {
        m.bedges.push_back(a); m.bedges.push_back(b); m.btag.push_back(tag);
    };
    for (uint32_t i = 0; i < nx; ++i) push_edge(i, i + 1, 1);                       // 下
    for (uint32_t j = 0; j < ny; ++j) push_edge(j * nx1 + nx, (j + 1) * nx1 + nx, 2); // 右
    for (uint32_t i = 0; i < nx; ++i) push_edge(ny * nx1 + i + 1, ny * nx1 + i, 3); // 上
    for (uint32_t j = 0; j < ny; ++j) push_edge((j + 1) * nx1, j * nx1, 4);         // 左

    return m;
}

// ---------------------------------------------------------------------------
// Gmsh .msh 读入 (ASCII, 2.2 与 4.1)
//   只取二维三角形单元 (type 2) 与一维边界单元 (type 1)。
//   4.1 里边界的物理标签要经 $Entities 的"曲线 -> 物理组"映射才能拿到,
//   所以 $Entities 必须解析; 2.2 里物理标签直接挂在元素的 tag 上。
//   节点编号在 gmsh 里未必连续, 读完统一压缩到 [0, n_nodes)。
// ---------------------------------------------------------------------------
inline int gmsh_nodes_per_elem(int type)
{
    switch (type)
    {
        case 1:  return 2;    // 2 节点线段
        case 2:  return 3;    // 3 节点三角形
        case 3:  return 4;    // 4 节点四边形
        case 4:  return 4;    // 4 节点四面体
        case 5:  return 8;    // 8 节点六面体
        case 6:  return 6;    // 6 节点棱柱
        case 7:  return 5;    // 5 节点金字塔
        case 8:  return 3;    // 3 节点二阶线段
        case 9:  return 6;    // 6 节点二阶三角形
        case 10: return 9;    // 9 节点二阶四边形
        case 11: return 10;   // 10 节点二阶四面体
        case 15: return 1;    // 点
        default: return -1;
    }
}

inline Mesh read_gmsh(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("无法打开 .msh 文件: " + path);

    double version = 2.2;
    std::string tok;

    std::unordered_map<int, int>            curve2phys;   // 曲线编号 -> 物理标签 (4.1)
    std::map<int, std::array<double, 2>>    node_xy;      // gmsh 节点号 -> 坐标 (有序)
    std::vector<int>                        raw_tris;     // 三角形, 存 gmsh 节点号
    std::vector<int>                        raw_bedges;   // 边界边, 存 gmsh 节点号
    std::vector<int>                        raw_btags;

    while (in >> tok)
    {
        if (tok == "$MeshFormat")
        {
            int ftype = 0; std::size_t dsize = 8;
            in >> version >> ftype >> dsize;
            if (ftype != 0)
                throw std::runtime_error("只支持 ASCII 格式的 .msh (二进制请用 gmsh 转存)");
            in >> tok;                                    // $EndMeshFormat
        }
        else if (tok == "$Entities" && version >= 4.0)
        {
            std::size_t npt, ncv, nsf, nvol;
            in >> npt >> ncv >> nsf >> nvol;

            for (std::size_t i = 0; i < npt; ++i)          // 点实体
            {
                int t; std::size_t nphys; double x, y, z;
                in >> t >> x >> y >> z >> nphys;
                for (std::size_t q = 0; q < nphys; ++q) { int p; in >> p; }
            }
            for (std::size_t i = 0; i < ncv; ++i)          // 曲线实体 (边界)
            {
                int t; std::size_t nphys, nbp; double b[6];
                in >> t >> b[0] >> b[1] >> b[2] >> b[3] >> b[4] >> b[5] >> nphys;
                int phys = 0;
                for (std::size_t q = 0; q < nphys; ++q)
                {
                    int p; in >> p;
                    if (q == 0) phys = p;
                }
                in >> nbp;
                for (std::size_t q = 0; q < nbp; ++q) { int p; in >> p; }
                curve2phys[t] = phys;
            }
            for (std::size_t i = 0; i < nsf; ++i)          // 曲面实体
            {
                int t; std::size_t nphys, nbc; double b[6];
                in >> t >> b[0] >> b[1] >> b[2] >> b[3] >> b[4] >> b[5] >> nphys;
                for (std::size_t q = 0; q < nphys; ++q) { int p; in >> p; }
                in >> nbc;
                for (std::size_t q = 0; q < nbc; ++q) { int c; in >> c; }
            }
            for (std::size_t i = 0; i < nvol; ++i)         // 体实体
            {
                int t; std::size_t nphys, nbs; double b[6];
                in >> t >> b[0] >> b[1] >> b[2] >> b[3] >> b[4] >> b[5] >> nphys;
                for (std::size_t q = 0; q < nphys; ++q) { int p; in >> p; }
                in >> nbs;
                for (std::size_t q = 0; q < nbs; ++q) { int s; in >> s; }
            }
            in >> tok;                                     // $EndEntities
        }
        else if (tok == "$Nodes")
        {
            if (version >= 4.0)
            {
                std::size_t nblk, nn; int mn, mx;
                in >> nblk >> nn >> mn >> mx;
                for (std::size_t b = 0; b < nblk; ++b)
                {
                    int edim, etag, parametric; std::size_t nnb;
                    in >> edim >> etag >> parametric >> nnb;
                    for (std::size_t i = 0; i < nnb; ++i)
                    {
                        int t; in >> t;
                        if (parametric) { double u, v, w; in >> u >> v >> w; }
                        double x, y, z; in >> x >> y >> z;
                        node_xy[t] = std::array<double, 2>{x, y};
                    }
                }
            }
            else
            {
                std::size_t nn; in >> nn;
                for (std::size_t i = 0; i < nn; ++i)
                {
                    int t; double x, y, z; in >> t >> x >> y >> z;
                    node_xy[t] = std::array<double, 2>{x, y};
                }
            }
            in >> tok;                                     // $EndNodes
        }
        else if (tok == "$Elements")
        {
            if (version >= 4.0)
            {
                std::size_t nblk, ne; int mn, mx;
                in >> nblk >> ne >> mn >> mx;
                for (std::size_t b = 0; b < nblk; ++b)
                {
                    int edim, etag, etype; std::size_t nne;
                    in >> edim >> etag >> etype >> nne;
                    const int nn = gmsh_nodes_per_elem(etype);
                    if (nn < 0) throw std::runtime_error("未知的 gmsh 单元类型: " + std::to_string(etype));
                    for (std::size_t i = 0; i < nne; ++i)
                    {
                        int et; in >> et;
                        std::vector<int> ns(static_cast<std::size_t>(nn));
                        for (int q = 0; q < nn; ++q) in >> ns[static_cast<std::size_t>(q)];
                        if (etype == 2 && edim == 2)
                        {
                            raw_tris.push_back(ns[0]); raw_tris.push_back(ns[1]); raw_tris.push_back(ns[2]);
                        }
                        else if (etype == 1)
                        {
                            auto it = curve2phys.find(etag);
                            raw_bedges.push_back(ns[0]); raw_bedges.push_back(ns[1]);
                            raw_btags.push_back(it == curve2phys.end() ? 0 : it->second);
                        }
                    }
                }
            }
            else
            {
                std::size_t ne; in >> ne;
                for (std::size_t i = 0; i < ne; ++i)
                {
                    int et, etype, ntags;
                    in >> et >> etype >> ntags;
                    std::vector<int> tags(static_cast<std::size_t>(ntags < 0 ? 0 : ntags));
                    for (int q = 0; q < ntags; ++q) in >> tags[static_cast<std::size_t>(q)];
                    const int nn = gmsh_nodes_per_elem(etype);
                    if (nn < 0) throw std::runtime_error("未知的 gmsh 单元类型: " + std::to_string(etype));
                    std::vector<int> ns(static_cast<std::size_t>(nn));
                    for (int q = 0; q < nn; ++q) in >> ns[static_cast<std::size_t>(q)];
                    if (etype == 2)
                    {
                        raw_tris.push_back(ns[0]); raw_tris.push_back(ns[1]); raw_tris.push_back(ns[2]);
                    }
                    else if (etype == 1)
                    {
                        raw_bedges.push_back(ns[0]); raw_bedges.push_back(ns[1]);
                        raw_btags.push_back(ntags > 0 ? tags[0] : 0);
                    }
                }
            }
            in >> tok;                                     // $EndElements
        }
        // 其余段 ($PhysicalNames / $PhysicalGroups / $NodeData ...) 不关心,
        // 逐个 token 读过去即可 —— 它们的内容里不会以 '$' 开头。
    }

    // 节点编号压缩
    std::unordered_map<int, int> g2i;
    g2i.reserve(node_xy.size());
    Mesh m;
    m.coords.reserve(2 * node_xy.size());
    for (const auto& kv : node_xy)
    {
        g2i[kv.first] = static_cast<int>(m.coords.size() / 2);
        m.coords.push_back(kv.second[0]);
        m.coords.push_back(kv.second[1]);
    }
    m.tris.resize(raw_tris.size());
    for (std::size_t i = 0; i < raw_tris.size(); ++i) m.tris[i] = static_cast<uint32_t>(g2i[raw_tris[i]]);
    m.bedges.resize(raw_bedges.size());
    for (std::size_t i = 0; i < raw_bedges.size(); ++i) m.bedges[i] = static_cast<uint32_t>(g2i[raw_bedges[i]]);
    m.btag = raw_btags;

    if (m.n_tris() == 0) throw std::runtime_error(".msh 里没有找到三角形单元: " + path);
    return m;
}

}   // namespace fem
