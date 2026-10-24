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

int main() 
{
    Matrix A1(3, 3);
    A1(0,0) = 4.0; A1(0,1) = 2.0; A1(0,2) = 0.1;
    A1(1,0) = 2.0; A1(1,1) = 5.0; A1(1,2) = 3.0;
    A1(2,0) = 0.1; A1(2,1) = 3.0; A1(2,2) = 6.0;

    Matrix A2(3, 3);
    A2(0,0) = 0.0; A2(0,1) = 2.0; A2(0,2) = 1.0;
    A2(1,0) = -2.0; A2(1,1) = 0.0; A2(1,2) = -6.0;
    A2(2,0) = -1.0; A2(2,1) = 6.0; A2(2,2) = 0.0;

    Matrix A3(3, 1);
    A3(0,0) = 1.0;
    A3(1,0) = -1.0;
    A3(2,0) = 0.0;

    Matrix A4 = Matrix::identity(10000);

    std::vector<double> v5 = {2.5, 5.0, 7.5, 10.0};
    Matrix A5 = Matrix(v5);
 
    // 打印矩阵
    std::cout << "====================== 打印矩阵 ======================" << std::endl;
    // A1.print();
    // A2.print();
    // A3.print();
    // A4.print();
    A5.print();
    A5({1,2}).print();



    // 矩阵性质
    std::cout << "====================== 矩阵性质 ======================" << std::endl;
    // std::cout << A1.isSquare() << A2.isSquare() << A3.isSquare() << A4.isSquare() << std::endl;
    // std::cout << A1.isSymmetric() << A2.isSymmetric() << A3.isSymmetric() << A4.isSymmetric() << std::endl;
    // std::cout << A1.isDiag() << A2.isDiag() << A3.isDiag() << A4.isDiag() << std::endl;
    // std::cout << A1.isAntisym() << A2.isAntisym() << A3.isAntisym() << A4.isAntisym() << std::endl;
    // std::cout << A1.isSymPos() << A2.isSymPos() << A3.isSymPos() << A4.isSymPos() << std::endl;
    // std::cout << A1.isUpTri() << A2.isUpTri() << A3.isUpTri() << A4.isUpTri() << std::endl;
    // std::cout << A1.isDownTri() << A2.isDownTri() << A3.isDownTri() << A4.isDownTri() << std::endl;

    // auto start = std::chrono::high_resolution_clock::now();
    // std::cout << A4.isSymPos() << std::endl;
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double>  duration = (end - start);
    // std::cout << "耗时: " << duration.count() << " s" << std::endl;

    // 矩阵性质
    std::cout << "====================== 矩阵操作 ======================" << std::endl;
    // A1.transpose().print(); A2.transpose().print(); A3.transpose().print(); A4.transpose().print();
    
    // for (int i = 1; i < 5; i++)
    // {
    //     Matrix testmatrix = Matrix::identity(std::pow(10, i));
    //     auto start = std::chrono::high_resolution_clock::now();
    //     testmatrix.inverse();
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double>  duration = (end - start);
    //     std::cout << std::pow(10, i) << "阶矩阵求逆耗时: " << duration.count() << " s" << std::endl;
    // }

    std::cout << "====================== 静态工厂 ======================" << std::endl;
    // Matrix::random(10, 10).print();

    std::cout << "====================== 矩阵运算 ======================" << std::endl;
    // (A1 + A2).print();
    // (A1 - A2).print();
    // (A1 * A2).print();
    // (A1 * 2 - 2 * A1).print();
    // (A1 / 2 - 2 / A1).print();
    // (A1 + 2).print();
    // (A1 - 2).print();

    std::cout << "====================== 初等函数 ======================" << std::endl;
    // (A1.sin()).print();
    // (A1.cos()).print();
    // (A1.abs()).print();
    // (A1.exp()).print();
    // (A1.log()).print();
    // // (A2.log()).print();
    // (A1.sqrt()).print();
    // // (A2.sqrt()).print();
    // (A1.pow(2)).print();

    std::cout << "====================== 统计函数 ======================" << std::endl;
    std::cout << A1.sum() << " " << A1.mean() << std::endl;
    
    std::cout << "====================== 切片函数 ======================" << std::endl;
    A1({1,2}, {0,1}).print();
    A1.toTensor()({0,2}, {0,1}, {0, 0}).print();

    // 测试最大最小函数
    auto [max_val1, max_row1, max_col1] = A2.max();
    std::cout << "最大值: " << max_val1 
          << " (位置: [" << max_row1 << ", " << max_col1 << "])" << std::endl;

    auto [min_val1, min_row1, min_col1] = A1.min();
    std::cout << "最小值: " << min_val1 
          << " (位置: [" << min_row1 << ", " << min_col1 << "])" << std::endl;
}