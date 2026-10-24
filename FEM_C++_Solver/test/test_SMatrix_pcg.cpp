#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <chrono>

#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>      // OpenMP 运行时 API
#endif

#include "math/integral.hpp"
#include "math/array.hpp"

using namespace array;
using namespace sparse;

void run_test(const std::string& precond_name, const SMatrix& A, const std::vector<double>& b, double omega = 1.0) {
    std::cout << "----------------------------------------\n";
    std::cout << "Testing Preconditioner: " << precond_name << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 调用求解器
    auto [x, it] = SMatrix::pcg(A, b, precond_name, 1e-8, 2000);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    // 计算最终真实残差 ||b - Ax|| / ||b||
    auto Ax = SMatrix::spmv(A, x);
    double normb = 0.0, normr = 0.0;
    for(size_t i = 0; i < b.size(); ++i) {
        double ri = b[i] - Ax[i];
        normb += b[i] * b[i];
        normr += ri * ri;
    }
    normb = std::sqrt(normb);
    normr = std::sqrt(normr);
    
    std::cout << "运行时间：" << std::fixed << std::setprecision(2) << duration.count() << " s\n";
    std::cout << "残差：" << std::scientific << normr / normb << "\n";
    std::cout << "预处理方法：" << precond_name << ", 迭代数：" << it + 1 << std::endl;
    std::cout << "----------------------------------------\n\n";
}

int main() {
    // 1. 设置网格规模
    // N = 500 意味着未知数 n = 250,000
    // 这是一个典型的大规模稀疏矩阵，非零元约 125 万个
    const size_t N = 500; 
    const size_t n = N * N;
    
    std::cout << "Generating 2D Poisson matrix...\n";
    std::cout << "Grid size: " << N << " x " << N << " (Total unknowns: " << n << ")\n";
    
    std::vector<double> values;
    std::vector<size_t> rowlist;
    std::vector<size_t> collist;
    
    // 预留空间，每个点最多5个非零元，避免 push_back 频繁分配内存
    values.reserve(n * 5);
    rowlist.reserve(n * 5);
    collist.reserve(n * 5);
    
    // 2. 构造 COO 格式的五点差分拉普拉斯矩阵
    // 内部点方程: 4*u(i,j) - u(i-1,j) - u(i+1,j) - u(i,j-1) - u(i,j+1) = f
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t idx = i * N + j;
            
            // 对角元
            rowlist.push_back(idx);
            collist.push_back(idx);
            values.push_back(4.0);
            
            // 左
            if (j > 0) {
                rowlist.push_back(idx);
                collist.push_back(idx - 1);
                values.push_back(-1.0);
            }
            // 右
            if (j < N - 1) {
                rowlist.push_back(idx);
                collist.push_back(idx + 1);
                values.push_back(-1.0);
            }
            // 下
            if (i > 0) {
                rowlist.push_back(idx);
                collist.push_back(idx - N);
                values.push_back(-1.0);
            }
            // 上
            if (i < N - 1) {
                rowlist.push_back(idx);
                collist.push_back(idx + N);
                values.push_back(-1.0);
            }
        }
    }
    
    // 3. 实例化 SMatrix (内部会自动转为 CSR)
    SMatrix A(n, n, values, rowlist, collist);
    std::cout << "Matrix generated. NNZ = " << A.nnz() << "\n\n";
    
    // 4. 构造右端项 b (全 1 向量)
    std::vector<double> b(n, 1.0);
    
    // 5. 运行测试
    // run_test("none", A, b);
    // run_test("diag", A, b);
    // run_test("ic", A, b);
    // run_test("ilu", A, b);
    run_test("ssor", A, b);
    
    return 0;
}