#pragma once
// ============================================================================
// 三角形 Lagrange 单元 (P1 / P2)
// ----------------------------------------------------------------------------
//   参考三角形: 顶点 (0,0), (1,0), (0,1);  重心坐标
//       l0 = 1 - xi - eta,  l1 = xi,  l2 = eta
//   局部自由度顺序:
//       0,1,2          三个顶点 (与 Mesh::tris 的三个节点一一对应)
//       3,4,5 (P2)     三条边的中点:  3 -> 边(1,2),  4 -> 边(0,2),  5 -> 边(0,1)
//                      (即第 k 个中点自由度在"去掉顶点 k-3 的那条边"上)
//   这个顺序正是二阶单元的标准编号, dofmap.hpp 里按同样的次序给边分配全局号。
//
//   几何映射用三个顶点做等参 (P1 几何, 直边三角形); 场变量用 Order 阶基函数。
// ============================================================================
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "fem/quadrature.hpp"

namespace fem
{

template <int Order>
struct TriElement;

// ---------------------------- P1 ----------------------------
template <>
struct TriElement<1>
{
    static constexpr int ndof = 3;

    // 形函数值
    static void shape(double l0, double l1, double l2, double* N)
    {
        N[0] = l0;  N[1] = l1;  N[2] = l2;
    }

    // 参考域梯度, 行主序 ndof x 2:  [dN/dxi, dN/deta]
    //   P1 的梯度是常数, 与 (l0,l1,l2) 无关 —— 装配里利用这一点,
    //   让物理梯度每单元只算一次 (见 assembler.hpp 的 if constexpr)。
    static void grad_ref(double /*l0*/, double /*l1*/, double /*l2*/, double* G)
    {
        G[0] = -1.0;  G[1] = -1.0;     // N0 = 1 - xi - eta
        G[2] =  1.0;  G[3] =  0.0;     // N1 = xi
        G[4] =  0.0;  G[5] =  1.0;     // N2 = eta
    }
};

// ---------------------------- P2 ----------------------------
template <>
struct TriElement<2>
{
    static constexpr int ndof = 6;

    static void shape(double l0, double l1, double l2, double* N)
    {
        N[0] = l0 * (2.0 * l0 - 1.0);
        N[1] = l1 * (2.0 * l1 - 1.0);
        N[2] = l2 * (2.0 * l2 - 1.0);
        N[3] = 4.0 * l1 * l2;          // 边 (1,2) 中点
        N[4] = 4.0 * l0 * l2;          // 边 (0,2) 中点
        N[5] = 4.0 * l0 * l1;          // 边 (0,1) 中点
    }

    // d(l0)/dxi = -1, d(l0)/deta = -1;  d(l1)/dxi = 1, d(l1)/deta = 0;
    // d(l2)/dxi =  0, d(l2)/deta =  1
    static void grad_ref(double l0, double l1, double l2, double* G)
    {
        G[0]  = -(4.0 * l0 - 1.0);  G[1]  = -(4.0 * l0 - 1.0);
        G[2]  =  (4.0 * l1 - 1.0);  G[3]  = 0.0;
        G[4]  = 0.0;                G[5]  =  (4.0 * l2 - 1.0);
        G[6]  =  4.0 * l2;          G[7]  =  4.0 * l1;             // N3 = 4 l1 l2
        G[8]  = -4.0 * l2;          G[9]  =  4.0 * (l0 - l2);      // N4 = 4 l0 l2
        G[10] =  4.0 * (l0 - l1);   G[11] = -4.0 * l1;             // N5 = 4 l0 l1
    }
};

// ---------------------------------------------------------------------------
// 把参考域梯度变换到物理域
//
//   记 J = ∂(x,y)/∂(xi,eta)。由链式法则
//       [∂N/∂x]           [ ∂y/∂eta   -∂y/∂xi ] [∂N/∂xi ]
//       [∂N/∂y] = 1/detJ *[ -∂x/∂eta   ∂x/∂xi ] [∂N/∂eta]
//   注意是 J^{-1} 的转置 (不是 J^{-1} 本身), 这里是最容易写反的地方。
//
//   顺带返回单元面积 = |detJ| / 2。
//   定向为负 (节点顺序顺时针) 直接抛异常 —— 静默接受会让 detJ 变号、
//   刚度矩阵整体反号, 这种错误在收敛阶测试里表现为"结果看起来只是差个符号",
//   极难定位。
//
//   【优化】这套变换被拆成两步, 让 O(1) 的那部分每单元只做一次:
//     旧写法 tri_physical_grad 在**每个求积点**都重算 detJ、做一次 2x2 求逆、
//     并对 nd 个基函数各做 4 次乘法 2 次加法。但三角形是仿射单元: J 只与单元
//     顶点有关, 与求积点无关 —— P2 的 6 个求积点就是 6 遍完全一样的逆矩阵。
//     拆开后:
//       tri_element_jac()  每单元一次: detJ、定向检查、2x2 逆并入 4 个系数
//       tri_apply_jac()    每点一次:   nd 次 2x2 乘加 (无除法、无分支)
//     顺带把定向检查提到单元级 —— 原来退化单元要进求积循环才被发现。
// ---------------------------------------------------------------------------

// Jinv[4] 存的是"参考域梯度 -> 物理域梯度"的 2x2 系数, 1/detJ 已并入:
//     dNdx = Jinv[0] * dN/dxi + Jinv[1] * dN/deta
//     dNdy = Jinv[2] * dN/dxi + Jinv[3] * dN/deta
inline void tri_element_jac(const double* vx, const double* vy,
                            double* Jinv, double& area)
{
    const double dxdxi  = vx[1] - vx[0],  dxdeta = vx[2] - vx[0];
    const double dydxi  = vy[1] - vy[0],  dydeta = vy[2] - vy[0];
    const double detJ   = dxdxi * dydeta - dxdeta * dydxi;

    if (!(detJ > 0.0))
        throw std::runtime_error("三角形定向为负或退化 (单元节点顺序错误?)");

    area = 0.5 * detJ;

    Jinv[0] =  dydeta / detJ;   Jinv[1] = -dydxi  / detJ;
    Jinv[2] = -dxdeta / detJ;   Jinv[3] =  dxdxi  / detJ;
}

template <int Order>
inline void tri_apply_jac(const double* Jinv, const double* Gref,
                          double* dNdx, double* dNdy)
{
    constexpr int nd = TriElement<Order>::ndof;
    for (int i = 0; i < nd; ++i)
    {
        const double gxi  = Gref[2 * i + 0];
        const double geta = Gref[2 * i + 1];
        dNdx[i] = Jinv[0] * gxi + Jinv[1] * geta;
        dNdy[i] = Jinv[2] * gxi + Jinv[3] * geta;
    }
}

// 保留原接口 (一次调用同时拿到梯度与面积), 内部改走上面两步。
template <int Order>
inline void tri_physical_grad(const double* vx, const double* vy,
                              double l0, double l1, double l2,
                              double* dNdx, double* dNdy, double& area)
{
    constexpr int nd = TriElement<Order>::ndof;

    double Jinv[4];
    tri_element_jac(vx, vy, Jinv, area);

    double Gref[2 * nd];
    TriElement<Order>::grad_ref(l0, l1, l2, Gref);
    tri_apply_jac<Order>(Jinv, Gref, dNdx, dNdy);
}

}   // namespace fem
