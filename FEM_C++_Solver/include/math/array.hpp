#pragma once

#include <functional>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <random>
#include <mkl.h>
#include <memory>
#include <queue>
#include <cstdint>
#include <cstddef>

// ============ 边界检查宏 ============
// Debug 构建：越界抛 std::out_of_range，行为与原先一致。
// Release 构建(NDEBUG)：整条检查被彻底消除。
// 原因：Matrix/Tensor 的 operator() 位于所有热循环最内层，
// 只要里面留一个 throw 分支，编译器就无法证明它不会逃逸，
// 从而无法对循环做自动向量化(SIMD)。
#ifdef NDEBUG
    #define CPPNUM_ARRAY_BOUNDS(cond, msg) static_cast<void>(0)
#else
    #define CPPNUM_ARRAY_BOUNDS(cond, msg)                                  \
        do {                                                                \
            if (!(cond)) { throw std::out_of_range(msg); }                  \
        } while (false)
#endif

namespace array
{
    std::vector<double> linspace(double start, double stop, int num);

    std::vector<double> arange(double start, double stop, double step);

    struct Slice 
    {
        uint32_t start;
        uint32_t end;
        uint32_t size() const { return end - start + 1; }
    };

    class Matrix;

    class Tensor;

    class Matrix
    {
        private:
            std::vector<double> data;
            uint32_t rows_;
            uint32_t cols_;

        public:
            // 构造函数
            Matrix();
            Matrix(uint32_t rows, uint32_t cols);
            Matrix(uint32_t rows, uint32_t cols, double init_value);
            Matrix(const std::vector<std::vector<double>>& init_data);
            Matrix(const std::vector<double>& init_data);
            Matrix(uint32_t rows, uint32_t cols, const std::vector<double>& flat_data);

            // 转化为 Tensor 和 Vector
            Tensor toTensor() const;
            std::vector<std::vector<double>> toVector() const;
            
            // 拷贝构造函数
            Matrix(const Matrix&) = default;
            Matrix& operator=(const Matrix&) = default;

             // 移动构造函数和赋值运算符
            Matrix(Matrix&&) = default;
            Matrix& operator=(Matrix&&) = default;

            // 访问元素（内联定义；Release 下无分支，可被自动向量化）
            double& operator()(uint32_t row, uint32_t col)
            {
                CPPNUM_ARRAY_BOUNDS(row < rows_ && col < cols_, "Matrix: 索引超出范围");
                return data[static_cast<std::size_t>(row) * cols_ + col];
            }
            const double& operator()(uint32_t row, uint32_t col) const
            {
                CPPNUM_ARRAY_BOUNDS(row < rows_ && col < cols_, "Matrix: 索引超出范围");
                return data[static_cast<std::size_t>(row) * cols_ + col];
            }

            // 切片
            Matrix operator()(const Slice& r, const Slice& c = {0,0}) const;

            // 获取维度
            uint32_t rows() const { return rows_; }
            uint32_t cols() const { return cols_; }
            uint32_t size() const { return data.size(); }

            // 获取底层数据（用于与其他库交互）
            double* data_ptr() { return data.data(); }
            const double* data_ptr() const { return data.data(); }
            std::vector<double>& data_vector() { return data; }
            const std::vector<double>& data_vector() const { return data; }

            // 打印矩阵
            void print(uint32_t max_rows = 6, uint32_t max_cols = 6) const;

            // 矩阵性质判断
            bool isSquare() const;
            bool isSymmetric() const; // 仅适用于方阵
            bool isDiag() const; // 仅适用于方阵
            bool isAntisym() const; // 仅适用于方阵
            bool isSymPos() const; // 仅适用于方阵
            bool isUpTri() const; // 仅适用于方阵
            bool isDownTri() const; // 仅适用于方阵

            // 基本矩阵操作
            Matrix transpose() const;
            Matrix choleskyDecomposition() const; // 仅适用于方阵
            std::tuple<Matrix, Matrix, Matrix, int> lu() const; // 仅适用于方阵
            Matrix inverse() const;  // 仅适用于方阵
            double determinant() const;  // 仅适用于方阵

            // 静态工厂方法
            static Matrix identity(uint32_t n);
            static Matrix zeros(uint32_t rows, uint32_t cols);
            static Matrix ones(uint32_t rows, uint32_t cols);
            static Matrix random(uint32_t rows, uint32_t cols, double min = 0.0, double max = 1.0);

            // 运算符重载
            Matrix operator+(const Matrix& other) const;
            Matrix operator-(const Matrix& other) const;
            Matrix operator*(const Matrix& other) const;
            Matrix operator+(double scalar) const;
            Matrix operator-(double scalar) const;
            Matrix operator*(double scalar) const;
            Matrix operator/(double scalar) const;

            // 友元函数（用于运算符重载）
            friend Matrix operator*(double scalar, const Matrix& mat);
            friend Matrix operator+(double scalar, const Matrix& mat);
            friend Matrix operator-(double scalar, const Matrix& mat);
            friend Matrix operator/(double scalar, const Matrix& mat);

            // 元素级运算
            Matrix elementwise_multiply(const Matrix& other) const;
            Matrix elementwise_divide(const Matrix& other) const;

            // 向量化运算
            Matrix abs() const;
            Matrix sin() const;
            Matrix cos() const;
            Matrix exp() const;
            Matrix log() const;
            Matrix sqrt() const;
            Matrix pow(double exponent) const;
            template<typename Func> Matrix apply(Func func) const;

            // 统计函数
            double sum() const;
            double mean() const;
            std::tuple<double, uint32_t, uint32_t> max() const;
            std::tuple<double, uint32_t, uint32_t> min() const;
    };

    class Tensor
    {
        private:
            std::vector<double> data;
            uint32_t rows_;
            uint32_t cols_;
            uint32_t heis_;

        public:
            // 构造函数
            Tensor();
            Tensor(uint32_t rows, uint32_t cols, uint32_t heis_);
            Tensor(uint32_t rows, uint32_t cols, uint32_t heis_, double init_value);
            Tensor(const std::vector<std::vector<std::vector<double>>>& init_data);
            Tensor(const std::vector<std::vector<double>>& init_data);
            Tensor(const std::vector<double>& init_data);
            Tensor(uint32_t rows, uint32_t cols, uint32_t heis_, const std::vector<double>& flat_data);

            // 与 Matrix 和 vector 的转换
            Matrix toMatrix() const;
            std::vector<std::vector<std::vector<double>>> toVector() const;
            
            // 拷贝构造函数
            Tensor(const Tensor&) = default;
            Tensor& operator=(const Tensor&) = default;

             // 移动构造函数和赋值运算符
            Tensor(Tensor&&) = default;
            Tensor& operator=(Tensor&&) = default;

            // 访问元素（内联定义；Release 下无分支，可被自动向量化）
            double& operator()(uint32_t row, uint32_t col, uint32_t hei)
            {
                CPPNUM_ARRAY_BOUNDS(row < rows_ && col < cols_ && hei < heis_,
                                    "Tensor: 索引超出范围");
                return data[(static_cast<std::size_t>(row) * cols_ + col) * heis_ + hei];
            }
            const double& operator()(uint32_t row, uint32_t col, uint32_t hei) const
            {
                CPPNUM_ARRAY_BOUNDS(row < rows_ && col < cols_ && hei < heis_,
                                    "Tensor: 索引超出范围");
                return data[(static_cast<std::size_t>(row) * cols_ + col) * heis_ + hei];
            }
            Tensor operator()(const Slice& r = {0,0}, const Slice& c = {0,0}, const Slice& h = {0,0}) const;

            // 获取维度
            uint32_t rows() const { return rows_; }
            uint32_t cols() const { return cols_; }
            uint32_t heis() const { return heis_; }
            uint32_t size() const { return data.size(); }

            // 获取底层数据（用于与其他库交互）
            double* data_ptr() { return data.data(); }
            const double* data_ptr() const { return data.data(); }
            std::vector<double>& data_vector() { return data; }
            const std::vector<double>& data_vector() const { return data; }

            // 打印矩阵
            void print(uint32_t max_rows = 6, uint32_t max_cols = 6, uint32_t max_heis = 6) const;

            // 运算符重载
            Tensor operator+(const Tensor& other) const;
            Tensor operator-(const Tensor& other) const;
            Tensor operator+(double scalar) const;
            Tensor operator-(double scalar) const;
            Tensor operator*(double scalar) const;
            Tensor operator/(double scalar) const;

            // 友元函数（用于运算符重载）
            friend Tensor operator*(double scalar, const Tensor& mat);
            friend Tensor operator+(double scalar, const Tensor& mat);
            friend Tensor operator-(double scalar, const Tensor& mat);
            friend Tensor operator/(double scalar, const Tensor& mat);

            // 元素级运算
            Tensor elementwise_multiply(const Tensor& other) const;
            Tensor elementwise_divide(const Tensor& other) const;

            // 向量化运算
            Tensor abs() const;
            Tensor sin() const;
            Tensor cos() const;
            Tensor exp() const;
            Tensor log() const;
            Tensor sqrt() const;
            Tensor pow(double exponent) const;
            template<typename Func> Tensor apply(Func func) const;

            // 统计函数
            double sum() const;
            double mean() const;
            std::tuple<double, uint32_t, uint32_t, uint32_t> max() const;
            std::tuple<double, uint32_t, uint32_t, uint32_t> min() const;
    };

    template<typename Func> Matrix Matrix::apply(Func func) const 
    {
        Matrix result(rows_, cols_);
    
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++) 
        {
            for (uint32_t j = 0; j < cols_; j++) 
            {
                result(i, j) = func((*this)(i, j));
            }
        }
        return result;
    }

    template<typename Func> Tensor Tensor::apply(Func func) const 
    {
        Tensor result(rows_, cols_, heis_);
    
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++) 
        {
            for (uint32_t j = 0; j < cols_; j++) 
            {
                for (uint32_t k = 0; k < heis_; k++) 
                {
                    result(i, j, k) = func((*this)(i, j, k));
                }
            }
        }
        return result;
    }
}

namespace sparse
{
    class SMatrix
    {
        private:
            uint32_t rows_, cols_;

            std::vector<uint32_t> row_ptr_;      // 行指针，长度 rows_ + 1
            std::vector<uint32_t> col_idx_;      // 列指标，长度 nnz
            std::vector<double> values_;       // 非零元数值


        public:
            SMatrix();
            SMatrix(
                uint32_t rows, 
                uint32_t cols,
                const std::vector<double>& values,
                const std::vector<uint32_t>& rowlist,
                const std::vector<uint32_t>& collist
            );
            SMatrix(
                uint32_t rows, 
                uint32_t cols,
                std::vector<uint32_t> row_ptr,
                std::vector<uint32_t> col_idx,
                std::vector<double> values
            );
      

            static double dot(const std::vector<double>& a, const std::vector<double>& b);

            uint32_t rows() const;
            uint32_t cols() const;
            uint32_t nnz()  const;

            const std::vector<double>& values() const;
            const std::vector<uint32_t>& row_ptr() const;
            const std::vector<uint32_t>& col_idx() const;

            SMatrix transpose() const;

            SMatrix operator*(const SMatrix& B) const;

            // 预处理函数
            static std::function<void(const std::vector<double>&, std::vector<double>&)> make_diag_preconditioner(const SMatrix& A);
            static std::function<void(const std::vector<double>&, std::vector<double>&)> make_ic0_preconditioner(const SMatrix& A);
            static std::function<void(const std::vector<double>&, std::vector<double>&)> make_ilu0_preconditioner(const SMatrix& A);
            static std::function<void(const std::vector<double>&, std::vector<double>&)> make_ssor_preconditioner(const SMatrix& A, double omega = 1.2);
            // AMG 预处理器。
            //   drop_tol / max_row_nnz: Galerkin 粗算子的稀疏化判据, 见
            //     src/math/array.cpp 里 spgemm_prune 的注释。默认 0.02 / 60
            //     (块 AMG 上验证过的经验配置, 按标量元素个数换算)。
            //     传 drop_tol=0 且 max_row_nnz=0 可完全退回不裁剪的旧行为。
            //   verbose: 打印每层的规模与 operator complexity。
            //   smoother: 0 = 阻尼 Jacobi (默认), 1 = Chebyshev 多项式光滑。
            //     P2 刚度不是 M-矩阵 (39% 非对角为正), Jacobi 光滑性质差;
            //     Chebyshev 只要求 SPD + 谱区间, 是直击该根因的最小改动。
            //   cheb_deg / cheb_ratio: 仅 smoother=1 时生效。deg = 每次光滑的
            //     SpMV 次数 (与 deg 次 Jacobi sweep 同价), alpha = beta/cheb_ratio。
            static std::function<void(const std::vector<double>&, std::vector<double>&)> make_amg_preconditioner(
                const SMatrix& A, double drop_tol = 0.02, uint32_t max_row_nnz = 60,
                bool verbose = false, uint32_t coarse_thresh = 300,
                int smoother = 0, int cheb_deg = 2, double cheb_ratio = 30.0);

            // SpMV：矩阵向量乘法
            static std::vector<double> spmv(const SMatrix& A, const std::vector<double>& x); 
            // 出参版 spmv: 复用 y 的缓冲区，避免每次调用都重新分配（Krylov 迭代与预处理器内部热路径用）
            static void spmv_into(const SMatrix& A, const std::vector<double>& x, std::vector<double>& y);
            // CG：对称正定共轭梯度法+预处理
            static std::tuple<std::vector<double>, int> pcg(
                const SMatrix& A,
                const std::vector<double>& b,
                const std::string& precond = "none",
                double tol = 1e-10,
                int max_iter = 2000
            );
            // 同上, 但预处理器由调用方直接给出 (例如想给 AMG 传 drop/cap 参数,
            //   或者想把 Schur 补 / 块预处理器接进来时)。
            static std::tuple<std::vector<double>, int> pcg(
                const SMatrix& A,
                const std::vector<double>& b,
                const std::function<void(const std::vector<double>&, std::vector<double>&)>& solveM,
                double tol = 1e-10,
                int max_iter = 2000
            );
            // BiCGSTAB：非对称线性方程求解器·
            static std::tuple<std::vector<double>, int> bicgstab(
                const SMatrix& A, 
                const std::vector<double>& b, 
                const std::string& precond, 
                double tol = 1e-10, 
                int max_iter = 2000
            );
            // GMRES：非对称线性方程求解器
            static std::tuple<std::vector<double>, int> gmres(
                const SMatrix& A, 
                const std::vector<double>& b, 
                const std::string& precond, 
                int restart, 
                double tol = 1e-10,
                int max_iter = 2000
            );
            static std::tuple<std::vector<double>, int> gmres(
                const SMatrix& A,
                const std::vector<double>& b,
                const std::function<void(const std::vector<double>&, std::vector<double>&)>& solveM,
                int restart,
                double tol = 1e-10,
                int max_iter = 2000
            );
    
            // SOR
            static std::tuple<std::vector<double>, int> sor(
                const SMatrix& A, 
                const std::vector<double>& b,
                double omega, 
                double tol = 1e-10, 
                int max_iter = 2000
            );
            static std::tuple<std::vector<double>, int> sor_rb(
                const SMatrix& A, 
                const std::vector<double>& b, 
                double omega, 
                double tol = 1e-10, 
                int max_iter = 2000
            );
            // AMG
            static std::tuple<std::vector<double>, int> amg(
                const SMatrix& A, 
                const std::vector<double>& b, 
                double tol = 1e-10, 
                int max_iter = 2000
            );
    };
}
