#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>

#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>      // OpenMP 运行时 API
#endif

#include "math/integral.hpp"
#include "math/array.hpp"

using namespace array;
using namespace sparse;

void run_test(const std::string& precond_name, const SMatrix& A, const std::vector<double>& b, int restart) 
{
    std::cout << "----------------------------------------\n";
    std::cout << "Testing Preconditioner: " << precond_name << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 调用你的 bicgstab 函数 (专门用于非对称矩阵)
    auto [x, it] = SMatrix::gmres(A, b, precond_name, restart);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    // 计算最终真实残差 ||b - Ax|| / ||b||
    auto Ax = SMatrix::spmv(A, x);
    double normb = 0.0, normr = 0.0;
    for(uint32_t i = 0; i < b.size(); ++i) {
        double ri = b[i] - Ax[i];
        normb += b[i] * b[i];
        normr += ri * ri;
    }
    normb = std::sqrt(normb);
    normr = std::sqrt(normr);
    
    std::cout << "运行时间：" << std::fixed << std::setprecision(2) << duration.count() << " s\n";
    std::cout << "残差：" << std::scientific << normr / normb << "\n";
    std::cout << "预处理方法：" << precond_name << ", 重启数：" << restart << ", 迭代数：" << it + 1 << std::endl;
    std::cout << "----------------------------------------\n\n";
}

int main() {
    // 1. 设置网格规模
    const uint32_t N = 500; 
    const uint32_t n = N * N;
    
    // 2. 设置对流速度 (产生非对称性的核心参数)
    // vx, vy 越大，矩阵的非对称性越强。这里取 5.0 以产生显著的非对称效应。
    const double vx = 5.0;
    const double vy = 5.0;
    
    std::cout << "Generating 2D Convection-Diffusion matrix (Non-symmetric)...\n";
    std::cout << "Grid size: " << N << " x " << N << " (Total unknowns: " << n << ")\n";
    std::cout << "Convection velocity: vx = " << vx << ", vy = " << vy << "\n";
    
    std::vector<double> values;
    std::vector<uint32_t> rowlist;
    std::vector<uint32_t> collist;
    
    // 预留空间，每个点最多5个非零元，避免 push_back 频繁分配内存
    values.reserve(n * 5);
    rowlist.reserve(n * 5);
    collist.reserve(n * 5);
    
    // 3. 构造 COO 格式的非对称对流扩散矩阵
    // 物理方程: -Delta u + v \cdot \nabla u = f
    // 数值离散: 扩散项使用中心差分，对流项使用一阶迎风差分 (Upwind scheme)
    // 迎风格式保证了矩阵的对角占优性，同时引入了强烈的非对称性。
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            uint32_t idx = i * N + j;
            
            // 对角元: 扩散项系数(4) + 迎风对流项系数(vx + vy)
            rowlist.push_back(idx);
            collist.push_back(idx);
            values.push_back(4.0 + vx + vy);
            
            // 左 (j-1): 扩散(-1) + 迎风对流(-vx) -> 非对称来源
            if (j > 0) {
                rowlist.push_back(idx);
                collist.push_back(idx - 1);
                values.push_back(-1.0 - vx);
            }
            // 右 (j+1): 仅扩散(-1)
            if (j < N - 1) {
                rowlist.push_back(idx);
                collist.push_back(idx + 1);
                values.push_back(-1.0);
            }
            // 下 (i-1): 扩散(-1) + 迎风对流(-vy) -> 非对称来源
            if (i > 0) {
                rowlist.push_back(idx);
                collist.push_back(idx - N);
                values.push_back(-1.0 - vy);
            }
            // 上 (i+1): 仅扩散(-1)
            if (i < N - 1) {
                rowlist.push_back(idx);
                collist.push_back(idx + N);
                values.push_back(-1.0);
            }
        }
    }
    
    // 4. 实例化 SMatrix (内部会自动转为 CSR)
    SMatrix A(n, n, values, rowlist, collist);
    std::cout << "Matrix generated. NNZ = " << A.nnz() << "\n\n";
    
    // 5. 构造右端项 b (全 1 向量)
    std::vector<double> b(n, 1.0);
    
    // 6. 运行测试
    // 预期结果分析：
    // - none / diag: 迭代次数会非常多，因为对流项导致条件数变大。
    // - ic: IC(0) 假设矩阵是对称的 (U = L^T)，在非对称矩阵上强行应用会导致近似质量下降，效果不如 ILU。
    // - ilu: ILU(0) 完美保留了非对称结构 (U != L^T)，应该是迭代次数最少、表现最好的预处理。
    // run_test("none", A, b);
    // run_test("diag", A, b);
    // run_test("ic", A, b);   
    run_test("ilu", A, b, 10);  
    run_test("ilu", A, b, 20); 
    run_test("ilu", A, b, 30); 
    run_test("ilu", A, b, 50); 
    run_test("ilu", A, b, 80); 
    run_test("ilu", A, b, 100); 
    run_test("ilu", A, b, 140); 
    
    return 0;
}