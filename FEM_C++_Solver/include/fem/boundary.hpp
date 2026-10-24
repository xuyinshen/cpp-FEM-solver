#pragma once
// ============================================================================
// Dirichlet 边界条件的对称消元
// ----------------------------------------------------------------------------
//   做法是"消元"而不是"罚函数 / 对角线置大数":
//     对固定自由度 i (给定值 g_i):
//       (1) 对每个尚未固定的行 j 且 A(j,i) ≠ 0:
//              b(j) -= A(j,i) * g_i;   A(j,i) = 0
//       (2) 第 i 行整体换成单位行: A(i,i) = 1, 其余 = 0, b(i) = g_i
//   这样处理后矩阵仍然对称 (第 i 行与第 i 列同时被清空), 可以继续用 CG /
//   IC(0) / AMG 这些要求对称的求解器。罚函数法会把条件数平方级放大,
//   置大数法会破坏对称性, 都不如消元。
//
//   第 (1) 步需要"按列访问" A, 而 A 是 CSR (按行存), 所以先构造一份转置索引:
//   它只存 (行号, 原值数组下标), O(nnz), 不复制数值。
//   【注意】不要对每个固定自由度去扫全矩阵 (那是 O(n_fixed * nnz)); 边界自由度
//   在大网格上占 O(sqrt(N)), 那样做会让装配从 O(N) 退化到 O(N^1.5)。
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fem
{

inline void apply_dirichlet(uint32_t n,
                            const std::vector<uint32_t>& row_ptr,
                            const std::vector<uint32_t>& col_idx,
                            std::vector<double>& val,
                            std::vector<double>& rhs,
                            const std::vector<uint32_t>& fixed_dofs,
                            const std::vector<double>& fixed_vals)
{
    if (row_ptr.size() != static_cast<std::size_t>(n) + 1)
        throw std::invalid_argument("apply_dirichlet: row_ptr 长度与 n 不符");
    if (col_idx.size() != val.size())
        throw std::invalid_argument("apply_dirichlet: col_idx 与 val 长度不符");
    if (rhs.size() != n)
        throw std::invalid_argument("apply_dirichlet: rhs 长度与 n 不符");
    if (fixed_dofs.size() != fixed_vals.size())
        throw std::invalid_argument("apply_dirichlet: 自由度与取值数量不一致");

    std::vector<char> is_fixed(n, 0);
    for (std::size_t k = 0; k < fixed_dofs.size(); ++k)
    {
        if (fixed_dofs[k] >= n)
            throw std::out_of_range("apply_dirichlet: 固定自由度编号越界");
        is_fixed[fixed_dofs[k]] = 1;
    }

    // ---- 转置索引 ----
    const std::size_t nnz = col_idx.size();
    std::vector<uint32_t> trp(static_cast<std::size_t>(n) + 1, 0u);
    for (std::size_t p = 0; p < nnz; ++p) trp[col_idx[p] + 1]++;
    for (std::size_t i = 0; i < n; ++i)   trp[i + 1] += trp[i];

    std::vector<uint32_t> trow(nnz);      // 该非零元所在的行
    std::vector<std::size_t> tp(nnz);     // 该非零元在 val 中的下标
    {
        std::vector<uint32_t> off = trp;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
            {
                const uint32_t t = off[col_idx[p]]++;
                trow[t] = i;
                tp[t]   = p;
            }
    }

    // ---- (1) 把已知量移到右端项, 并清空固定列 ----
    for (std::size_t k = 0; k < fixed_dofs.size(); ++k)
    {
        const uint32_t i = fixed_dofs[k];
        const double   g = fixed_vals[k];
        for (uint32_t t = trp[i]; t < trp[i + 1]; ++t)
        {
            const uint32_t j = trow[t];
            if (is_fixed[j]) continue;                 // 固定行稍后整体处理
            rhs[j] -= val[tp[t]] * g;
            val[tp[t]] = 0.0;
        }
    }

    // ---- (2) 固定行换成单位行 ----
    for (std::size_t k = 0; k < fixed_dofs.size(); ++k)
    {
        const uint32_t i = fixed_dofs[k];
        const double   g = fixed_vals[k];
        bool has_diag = false;
        for (uint32_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
        {
            if (col_idx[p] == i) { val[p] = 1.0; has_diag = true; }
            else                 { val[p] = 0.0; }
        }
        if (!has_diag)
            throw std::runtime_error("apply_dirichlet: 该行缺少对角元, 无法消元");
        rhs[i] = g;
    }
}

}   // namespace fem
