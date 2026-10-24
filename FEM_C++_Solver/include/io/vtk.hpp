#pragma once
// ============================================================================
// VTK 输出 (legacy ASCII, ParaView 可直接打开)
// ----------------------------------------------------------------------------
//   P1 -> CELL_TYPE 5  (VTK_TRIANGLE),            点 = 网格节点
//   P2 -> CELL_TYPE 22 (VTK_QUADRATIC_TRIANGLE),  点 = 顶点 + 边中点
//
//   【VTK 的节点顺序陷阱】二次三角形的 VTK 顺序是
//       v0 v1 v2  m(0,1) m(1,2) m(2,0)
//   而我们的局部自由度顺序是
//       v0 v1 v2  m(1,2) m(0,2) m(0,1)   = d0 d1 d2 d3 d4 d5
//   所以写出时重排为 d0 d1 d2 d5 d3 d4。顺序错了在 ParaView 里表现为
//   "单元形状正常但场扭曲", 是二阶单元最常见的可视化 bug。
//
//   坐标点表必须包含所有被引用的点 —— P2 时要把边中点也写进去, 否则
//   ParaView 会报点数不匹配。
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "fem/dofmap.hpp"
#include "fem/mesh.hpp"

namespace fem
{

inline void write_vtk(const std::string& path, const Mesh& m, const DofMap& dm,
                      const std::vector<double>& u, const std::string& field = "u")
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("无法写出 VTK 文件: " + path);

    const bool     p2  = (dm.order == 2);
    const std::size_t npts = p2 ? dm.n_dof : m.n_nodes();

    out << "# vtk DataFile Version 3.0\n";
    out << "fem solution\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << npts << " double\n";
    out << std::setprecision(12);
    for (std::size_t i = 0; i < dm.n_vertex; ++i)
        out << m.coords[2 * i + 0] << " " << m.coords[2 * i + 1] << " 0.0\n";
    if (p2)
    {
        for (std::size_t k = 0; k + 1 < dm.edge_nodes.size(); k += 2)
        {
            const uint32_t a = dm.edge_nodes[k], b = dm.edge_nodes[k + 1];
            out << 0.5 * (m.coords[2 * a + 0] + m.coords[2 * b + 0]) << " "
                << 0.5 * (m.coords[2 * a + 1] + m.coords[2 * b + 1]) << " 0.0\n";
        }
    }

    const std::size_t nt = m.n_tris();
    out << "\nCELLS " << nt << " " << ((p2 ? 7 : 4) * nt) << "\n";
    if (!p2)
    {
        for (std::size_t e = 0; e < nt; ++e)
            out << "3 " << m.tris[3 * e + 0] << " " << m.tris[3 * e + 1]
                << " " << m.tris[3 * e + 2] << "\n";
    }
    else
    {
        const std::size_t nd = 6;
        for (std::size_t e = 0; e < nt; ++e)
        {
            const uint32_t* d = &dm.elem_dof[nd * e];
            out << "6 " << d[0] << " " << d[1] << " " << d[2] << " "
                       << d[5] << " " << d[3] << " " << d[4] << "\n";
        }
    }

    out << "\nCELL_TYPES " << nt << "\n";
    const int ct = p2 ? 22 : 5;
    for (std::size_t e = 0; e < nt; ++e) out << ct << "\n";

    out << "\nPOINT_DATA " << npts << "\n";
    out << "SCALARS " << field << " double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < npts; ++i)
        out << (i < u.size() ? u[i] : 0.0) << "\n";
}

}   // namespace fem
