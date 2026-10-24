#pragma once
// ============================================================================
// 自由度编号
// ----------------------------------------------------------------------------
//   P1: 一个自由度 = 一个网格节点。
//   P2: 顶点自由度 + 每条边一个中点自由度。
//       边必须先全局编号, 否则相邻单元对同一条边会给出不同的自由度号,
//       装配出来的场在单元边界上不连续 —— 这个 bug 的表现是"解看着差不多但
//       收敛阶掉到 1 阶", 很难从图上看出, 所以边编号用 (min,max) 顶点对做键,
//       与单元的局部顶点顺序无关。
// ============================================================================
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fem/mesh.hpp"

namespace fem
{

inline uint64_t edge_key(uint32_t a, uint32_t b)
{
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

struct DofMap
{
    int                  order   = 1;
    std::size_t          n_vertex = 0;         // == mesh.n_nodes()
    std::size_t          n_dof    = 0;
    std::vector<uint32_t> elem_dof;            // ndof_per_elem * n_tris
    // 仅 order == 2 有意义
    std::unordered_map<uint64_t, uint32_t> edge_id;     // 边键 -> 边序号
    std::vector<uint32_t>                  edge_nodes;  // 2 * n_edge: 每条边的两个顶点

    int ndof_per_elem() const { return (order == 1) ? 3 : 6; }
};

inline DofMap make_dofmap(const Mesh& m, int order)
{
    if (order != 1 && order != 2)
        throw std::invalid_argument("make_dofmap: 目前只支持 order = 1 或 2");

    const std::size_t nt = m.n_tris();
    const int         nd = (order == 1) ? 3 : 6;

    DofMap dm;
    dm.order    = order;
    dm.n_vertex = m.n_nodes();
    dm.elem_dof.assign(static_cast<std::size_t>(nd) * nt, 0u);

    for (std::size_t e = 0; e < nt; ++e)
        for (int i = 0; i < 3; ++i)
            dm.elem_dof[static_cast<std::size_t>(nd) * e + i] = m.tris[3 * e + i];

    if (order == 1)
    {
        dm.n_dof = dm.n_vertex;
        return dm;
    }

    uint32_t n_edge = 0;
    dm.edge_id.reserve(3 * nt);
    for (std::size_t e = 0; e < nt; ++e)
    {
        const uint32_t v0 = m.tris[3 * e + 0];
        const uint32_t v1 = m.tris[3 * e + 1];
        const uint32_t v2 = m.tris[3 * e + 2];
        // 局部自由度 3,4,5 分别对应边 (1,2), (0,2), (0,1)
        const uint32_t ev[3][2] = { { v1, v2 }, { v0, v2 }, { v0, v1 } };

        for (int k = 0; k < 3; ++k)
        {
            const uint64_t key = edge_key(ev[k][0], ev[k][1]);
            auto it = dm.edge_id.find(key);
            uint32_t id;
            if (it == dm.edge_id.end())
            {
                id = n_edge++;
                dm.edge_id.emplace(key, id);
                dm.edge_nodes.push_back(ev[k][0]);
                dm.edge_nodes.push_back(ev[k][1]);
            }
            else
            {
                id = it->second;
            }
            dm.elem_dof[static_cast<std::size_t>(nd) * e + 3 + k]
                = static_cast<uint32_t>(dm.n_vertex) + id;
        }
    }
    dm.n_dof = dm.n_vertex + n_edge;
    return dm;
}

// 每个自由度的物理坐标。P2 的中点自由度取边的几何中点 —— 网格是直边三角形,
// 等参映射下中点自由度的节点位置就是几何中点 (不是参数中点, 两者在仿射映射
// 下一致; 这里直接取几何中点, 对非仿射情形也更稳)。
inline std::vector<std::pair<double, double>> dof_coords(const Mesh& m, const DofMap& dm)
{
    std::vector<std::pair<double, double>> c(dm.n_dof);
    for (std::size_t i = 0; i < dm.n_vertex; ++i)
        c[i] = { m.coords[2 * i + 0], m.coords[2 * i + 1] };
    for (std::size_t k = 0; k + 1 < dm.edge_nodes.size(); k += 2)
    {
        const uint32_t a = dm.edge_nodes[k], b = dm.edge_nodes[k + 1];
        c[dm.n_vertex + k / 2] = {
            0.5 * (m.coords[2 * a + 0] + m.coords[2 * b + 0]),
            0.5 * (m.coords[2 * a + 1] + m.coords[2 * b + 1])
        };
    }
    return c;
}

// 位于边界上的自由度。tag == 0 表示"所有边界"。
//   P2 时还要把边界边的中点自由度算进去 —— 漏掉它的后果是 P2 的 Dirichlet
//   条件只在顶点施加, 边界上仍有自由度在自由振荡。
inline std::vector<uint32_t> boundary_dofs(const Mesh& m, const DofMap& dm, int tag = 0)
{
    std::vector<char> mark(dm.n_dof, 0);
    for (std::size_t b = 0; b < m.n_bedges(); ++b)
    {
        if (tag != 0 && m.btag[b] != tag) continue;
        const uint32_t a = m.bedges[2 * b + 0];
        const uint32_t c = m.bedges[2 * b + 1];
        if (a < dm.n_dof) mark[a] = 1;
        if (c < dm.n_dof) mark[c] = 1;
        if (dm.order == 2)
        {
            auto it = dm.edge_id.find(edge_key(a, c));
            if (it != dm.edge_id.end())
            {
                const std::size_t d = dm.n_vertex + it->second;
                if (d < dm.n_dof) mark[d] = 1;
            }
        }
    }
    std::vector<uint32_t> out;
    for (std::size_t i = 0; i < dm.n_dof; ++i)
        if (mark[i]) out.push_back(static_cast<uint32_t>(i));
    return out;
}

}   // namespace fem
