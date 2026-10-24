#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif

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
#include <limits>

#include "math/array.hpp"

namespace
{
    // MKL 的整数参数是 MKL_INT（默认 LP64 下是 32 位 int）。
    // 规模越界时明确报错，避免静默截断成负数。
    inline MKL_INT to_mkl_int(std::size_t v)
    {
        if (v > static_cast<std::size_t>(std::numeric_limits<MKL_INT>::max()))
        {
            throw std::runtime_error("规模超出 MKL_INT 表示范围");
        }
        return static_cast<MKL_INT>(v);
    }

    // ------------------------------------------------------------------
    // GMRES Arnoldi 正交化: CGS2 (两遍经典 Gram-Schmidt)
    //
    // Vbuf 为行主序 (m+1) x n，Hbuf 为行主序 (m+1) x m。
    // 这里故意只使用 OpenMP 的规整 canonical loop：
    //   1) 每个 h_i 独占写入，按基向量并行做连续内积；
    //   2) 按向量下标 t 并行更新 w，每个线程只写自己的 w[t]。
    //
    // 旧版把 w 分成 2048 元素面板，再在 OpenMP 区域中嵌套跨行访问 V。
    // 该写法理论上无数据竞争，但在 GCC 15/libgomp 的多线程 Release 构建中
    // 会在第一次 GMRES 内触发段错误；标准规整循环同时消除了这个编译器/
    // 运行库敏感点。CGS2 的数学语义与旧版完全相同。
    // ------------------------------------------------------------------
    inline void arnoldi_orth(double* Vbuf, double* Hbuf, double* w,
                             std::size_t n, int m, int j, double* hcol)
    {
        const std::ptrdiff_t nn   = static_cast<std::ptrdiff_t>(n);
        const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(j) + 1;

        for (int pass = 0; pass < 2; ++pass)
        {
            // h_i = (V_i, w)。每个 i 写独立的 hcol[i]，无归约竞争；
            // V_i 和 w 都按连续内存读取，便于编译器自动向量化。
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(rows >= 4 && nn >= 8192)
            #endif
            for (std::ptrdiff_t i = 0; i < rows; ++i)
            {
                const double* Vi = Vbuf + static_cast<std::size_t>(i) * n;
                double acc = 0.0;
                for (std::ptrdiff_t t = 0; t < nn; ++t) acc += Vi[t] * w[t];
                hcol[i] = acc;
            }

            for (std::ptrdiff_t i = 0; i < rows; ++i)
                Hbuf[i * static_cast<std::ptrdiff_t>(m) + j] += hcol[i];

            // w -= V^T h。t 是完全独立的输出维度；使用单一规整循环，
            // 避免旧面板实现对特定 GCC/libgomp 组合的敏感行为。
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(nn >= 8192)
            #endif
            for (std::ptrdiff_t t = 0; t < nn; ++t)
            {
                double corr = 0.0;
                for (std::ptrdiff_t i = 0; i < rows; ++i)
                    corr += hcol[i] * Vbuf[static_cast<std::size_t>(i) * n + t];
                w[t] -= corr;
            }
        }
    }

    // 逐元素定义域检查。直接扫裸指针，不走 operator()，可自动向量化。
    // 检查与计算分离：先一次性判定，再交给 MKL VML 批量算。
    template <typename Pred>
    inline bool check_all(const std::vector<double>& v, Pred pred)
    {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(v.size());
        int ok = 1;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(*:ok) schedule(static) if(n > 4096)
        #endif
        for (std::ptrdiff_t i = 0; i < n; ++i)
        {
            if (!pred(v[i])) { ok = 0; }
        }
        return ok == 1;
    }
}

namespace array
{
    std::vector<double> linspace(double start, double stop, int num) 
    {
        std::vector<double> vec(num);
        if (num <= 1) 
        {
            vec[0] = stop;
            return vec;
        }
        double step = (stop - start) / (num - 1);
        // 生成器：使用捕获的变量，按索引计算值
        std::generate(
            vec.begin(), vec.end(), [n = 0, start, step]() mutable 
            { 
                return start + n++ * step; 
            }
        );
        return vec;
    }

    std::vector<double> arange(double start, double stop, double step) 
    {
        const int num = std::round((stop - start) / step) + 1;
        if (num < 0)
        {
            throw std::invalid_argument("arange: start, stop, step 参数无效，请确保 (stop - start) / step > 0");
        }
        std::vector<double> vec(num);

        std::generate(
            vec.begin(), vec.end(), [n = 0, start, step]() mutable 
            { 
                return start + n++ * step; 
            }
        );
        return vec;
    }

    // Matrix 构造函数
    Matrix::Matrix() : rows_(0), cols_(0) {}
    Matrix::Matrix(uint32_t rows, uint32_t cols) : data(rows * cols, 0.0), rows_(rows), cols_(cols) {}
    Matrix::Matrix(uint32_t rows, uint32_t cols, double init_value) : data(rows * cols, init_value), rows_(rows), cols_(cols) {}
    Matrix::Matrix(const std::vector<std::vector<double>>& init_data)
    {
        if (init_data.empty()) 
        {
            rows_ = 0;
            cols_ = 0;
            return;
        }
        rows_ = init_data.size();
        cols_ = init_data[0].size();
        data.resize(rows_ * cols_);
        for (uint32_t i = 0; i < rows_; ++i) 
        {
            if (init_data[i].size() != cols_) 
            {
                throw std::invalid_argument("矩阵的所有行必须具有相同列数");
            }
        }
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; ++i) 
        {
            for (uint32_t j = 0; j < cols_; ++j) 
            {
                data[i * cols_ + j] = init_data[i][j];
            }
        }
    }
    Matrix::Matrix(const std::vector<double>& init_data)
    {
        if (init_data.empty()) 
        {
            rows_ = 0;
            cols_ = 0;
            return;
        }
        rows_ = init_data.size();
        cols_ = 1;
        data.resize(rows_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ > 40)
        #endif
        for (uint32_t i = 0; i < rows_; ++i) 
        {
            data[i] = init_data[i];
        }
    }
    Matrix::Matrix(uint32_t rows, uint32_t cols, const std::vector<double>& flat_data) : data(flat_data), rows_(rows), cols_(cols) 
    {
        if (flat_data.size() != rows * cols) 
        {
            throw std::invalid_argument("Matrix: 需要矩阵化的向量的长度必须等同于行数乘以列数");
        }
    }

    // 转化为 Tensor 和 Vector
    Tensor Matrix::toTensor() const
    {
        return Tensor(rows_, cols_, 1, data);
    }
    std::vector<std::vector<double>> Matrix::toVector() const
    {
        std::vector<std::vector<double>> result(rows_,
            std::vector<double>(cols_, 0.0));

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t j = 0; j < rows_; ++j)
        {
            for (uint32_t k = 0; k < cols_; ++k)
            {
                result[j][k] = data[j * cols_ + k];
            }
        }

        return result;
    }

    // Matrix 访问元素
    Matrix Matrix::operator()(const Slice& r, const Slice& c) const
    {
        if (r.end > rows_ || c.end > cols_ || r.start > r.end || c.start > c.end)
        {
            throw std::out_of_range("Matrix::operator() 索引越界");
        }

        const uint32_t r_len = r.size();
        const uint32_t c_len = c.size();

        Matrix result;
        result.rows_ = r_len;
        result.cols_ = c_len;
        result.data.resize(r_len * c_len);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(r_len * c_len > 4096)
        #endif
        for (uint32_t j = 0; j < r_len; ++j)
        {
            const uint32_t src_j = r.start + j;
            const double* src = &data[src_j * cols_ + c.start];
            double* dst = &result.data[j * c_len];
            std::copy(src, src + c_len, dst);
        }

        return result;
    }


    // 打印矩阵
    void Matrix::print(uint32_t max_rows, uint32_t max_cols) const 
    {
        if (rows_ <= max_rows && cols_ <= max_cols) 
        {
            for (uint32_t i = 0; i < rows_; i++) 
            {
                for (uint32_t j = 0; j < cols_; j++) 
                {
                    std::cout << std::setw(12) << std::fixed << std::setprecision(6) << (*this)(i, j);
                }
                std::cout << std::endl;
            }
        }
        else
        {
            uint32_t half_rows = max_rows / 2;
            uint32_t half_cols = max_cols / 2;

            for (uint32_t i = 0; i < rows_; i++) 
            {
                bool row_skip = (i >= half_rows && i < rows_ - half_rows);
                if (row_skip) 
                {
                    if (i == half_rows) {
                        std::cout << "   ..." << std::endl;
                    }
                    continue;
                }

                for (uint32_t j = 0; j < cols_; j++) 
                {
                    bool col_skip = (j >= half_cols && j < cols_ - half_cols);
                    if (col_skip) 
                    {
                        if (j == half_cols) {
                            std::cout << std::setw(12) << "...";
                        }
                        continue;
                    }

                    std::cout << std::setw(12) << std::fixed << std::setprecision(6) << (*this)(i, j);
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Matrix 矩阵性质判断
    bool Matrix::isSquare() const
    {
        if (rows_ == cols_)
        {
            return true;
        }
        return false;
    }
    bool Matrix::isSymmetric() const
    {
        if (! isSquare())
        {
            return false;
        }
        if (rows_ == 1)
        {
            return true;
        }
        for (uint32_t i = 1; i < rows_; i++)
        {
            for (uint32_t j = 0; j < i; j++)
            {
                if ((*this)(i, j) != (*this)(j, i))
                {
                    return false;
                }
            }
        }
        return true;
    }
    bool Matrix::isDiag() const
    {
        if (! isSquare())
        {
            return false;
        }
        if (rows_ == 1)
        {
            return true;
        }
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                if (j != i)
                {
                    if ((*this)(i, j) != 0)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    bool Matrix::isAntisym() const
    {
        if (! isSquare())
        {
            return false;
        }
        if (rows_ == 1)
        {
            if (data[0] == 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        for (uint32_t i = 0; i < rows_; i++)
        {
            if ((*this)(i, i) != 0)
            {
                return false;
            }
        }
        for (uint32_t i = 1; i < rows_; i++)
        {
            for (uint32_t j = 0; j < i; j++)
            {
                if ((*this)(i, j) + (*this)(j, i) != 0)
                {
                    return false;
                }
            }
        }
        return true;
    }
    bool Matrix::isSymPos() const
    {
        if (! isSymmetric())
        {
            return false;
        }
        uint32_t n = rows_;
        Matrix inv = *this;
        lapack_int info;
        lapack_int nn = static_cast<lapack_int>(n);

        // 1. Cholesky 分解
        info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nn, inv.data.data(), nn);
        if (info != 0)
        {
            return false;
        } 
        else
        {
            return true;
        }
    }
    bool Matrix::isUpTri() const
    { 
        if (! isSquare())
        {
            return false;
        }
        if (rows_ == 1)
        {
            return true;
        }
        for (uint32_t j = 1; j < cols_; j++)
        {
            for (uint32_t i = 0; i < j; i++)
            {
                if ((*this)(i, j) != 0)
                {
                    return false;
                }
            }
        }
        return true;
    }
    bool Matrix::isDownTri() const
    { 
        if (! isSquare())
        {
            return false;
        }
        if (rows_ == 1)
        {
            return true;
        }
        for (uint32_t j = 1; j < cols_; j++)
        {
            for (uint32_t i = 0; i < j; i++)
            {
                if ((*this)(j, i) != 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    // Matrix 基本矩阵操作
    Matrix Matrix::transpose() const
    {
        Matrix trans(cols_, rows_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < cols_; i++)
        {
            for (uint32_t j = 0; j < rows_; j++)
            {
                trans(i, j) = (*this)(j, i);
            }
        }

        return trans;
    }
    // Matrix Matrix::choleskyDecomposition() const
    // {
    //     if (! isSymmetric())
    //     {
    //         throw std::invalid_argument("cholesky分解要求矩阵为对称方阵");
    //     }
    //     for (uint32_t i = 0; i < rows_; i++)
    //     {
    //         if ((*this)(i, i) <= 0)
    //         {
    //             throw std::invalid_argument("cholesky分解要求矩阵为正定方阵");
    //         }
    //     }

    //     Matrix L(rows_, cols_, 0.0);
        
    //     for (uint32_t i = 0; i < rows_; i++)
    //     {
    //         for (uint32_t j = 0; j <= i; j++)
    //         {

    //             double sum = (*this)(i, j);
    //             double minusum = 0.0;
    //             #ifdef CPPNUM_HAVE_OPENMP
    //                 #pragma omp parallel for reduction(+:minusum) schedule(static)
    //             #endif
    //             for (uint32_t k = 0; k < j; k++) 
    //             {
    //                 minusum += L(i, k) * L(j, k);
    //                 // sum -= L(i, k) * L(j, k);
    //             }

    //             sum -= minusum;

    //             if (i == j) 
    //             {
    //                 if (sum <= 0) 
    //                 {
    //                     throw std::invalid_argument("cholesky分解要求矩阵为正定方阵");
    //                 }
    //                 L(i, i) = std::sqrt(sum);
    //             } 
    //             else 
    //             {
    //                 L(i, j) = sum / L(j, j);
    //             }
    //         }
    //     }
    //     return L;
    // }
    // std::tuple<Matrix, Matrix, Matrix, int> Matrix::lu() const
    // {
    //     if (! isSquare())
    //     {
    //         throw std::invalid_argument("矩阵不是方阵");
    //     }
    //     uint32_t n = rows_;
    //     Matrix LU = *this;  // 拷贝
    //     Matrix P(n, n, 0.0);
    //     int swap_count = 0;  // 记录行交换次数
        
    //     // 初始化置换矩阵
    //     #ifdef CPPNUM_HAVE_OPENMP
    //         #pragma omp parallel for schedule(static)
    //     #endif
    //     for (uint32_t i = 0; i < n; i++) P(i, i) = 1.0;

    //     for (uint32_t k = 0; k < n - 1; k++) 
    //     {
    //         uint32_t max_row = k;
    //         double max_val = fabs(LU(k, k));

    //         for (uint32_t i = k + 1; i < n; i++) 
    //         {
    //             if (fabs(LU(i, k)) > max_val) 
    //             {
    //                 max_val = fabs(LU(i, k));
    //                 max_row = i;
    //             }
    //         }
            
    //         // 交换行
    //         if (max_row != k) 
    //         {
    //             for (uint32_t j = 0; j < n; j++) 
    //             {
    //                 std::swap(LU(k, j), LU(max_row, j));
    //                 std::swap(P(k, j), P(max_row, j));
    //             }
    //             swap_count++;  // 记录行交换
    //         }
            
    //         // 检查奇异性
    //         if (fabs(LU(k, k)) < 1e-15) 
    //         {
    //             throw std::runtime_error("矩阵是奇异的");
    //         }
            
    //         // 消去
    //         for (uint32_t i = k + 1; i < n; i++) 
    //         {
    //             LU(i, k) /= LU(k, k);
    //             for (uint32_t j = k + 1; j < n; j++) 
    //             {
    //                 LU(i, j) -= LU(i, k) * LU(k, j);
    //             }
    //         }
    //     }
        
    //     // 提取 L 和 U
    //     Matrix L(n, n, 0.0);
    //     Matrix U(n, n, 0.0);
        
    //     #ifdef CPPNUM_HAVE_OPENMP
    //         #pragma omp parallel for schedule(static)
    //     #endif
    //     for (uint32_t i = 0; i < n; i++) 
    //     {
    //         L(i, i) = 1.0;
    //         for (uint32_t j = 0; j < i; j++) 
    //         {
    //             L(i, j) = LU(i, j);
    //         }
    //         for (uint32_t j = i; j < n; j++) 
    //         {
    //             U(i, j) = LU(i, j);
    //         }
    //     }
        
    //     return {L, U, P, swap_count};
    // }
    Matrix Matrix::choleskyDecomposition() const 
    {
        if (!isSquare()) 
        {
            throw std::invalid_argument("Cholesky分解要求矩阵为方阵");
        }
        uint32_t n = rows_;
        Matrix L = *this; // 拷贝一份，因为 MKL 会原地修改数据

        lapack_int info = LAPACKE_dpotrf(
            LAPACK_ROW_MAJOR, 
            'L', 
            static_cast<lapack_int>(n), 
            L.data.data(), // 假设你的数据成员是 data，且是 std::vector
            static_cast<lapack_int>(n)
        );

        if (info < 0) 
        {
            throw std::runtime_error("MKL dpotrf: 第 " + std::to_string(-info) + " 个参数非法");
        } 
        else if (info > 0)
        {
            throw std::invalid_argument("矩阵不是正定矩阵，无法进行 Cholesky 分解");
        }

        Matrix L_result(n, n, 0.0);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < n; i++) 
        {
            for (uint32_t j = 0; j <= i; j++) 
            {
                L_result(i, j) = L(i, j);
            }
        }

        return L_result;
    }
    std::tuple<Matrix, Matrix, Matrix, int> Matrix::lu() const 
    {
        if (!isSquare()) 
        {
            throw std::invalid_argument("矩阵不是方阵");
        }
        
        uint32_t n = rows_;
        if (n == 0) 
        {
            throw std::invalid_argument("空矩阵无法进行 LU 分解");
        }
        
        // 2. 拷贝矩阵（MKL 会原地修改）
        Matrix LU = *this;
        std::vector<lapack_int> ipiv(n);
        
        // 调用 MKL 的 LU 分解
        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, LU.data.data(), n, ipiv.data());
        
        if (info < 0) 
        {
            throw std::runtime_error("MKL dgetrf: 第 " + std::to_string(-info) + " 个参数非法");
        } 
        else if (info > 0) 
        {
            throw std::runtime_error("矩阵是奇异的，U 的第 " + std::to_string(info) + " 个对角元为零");
        }
        
        int swap_count = 0;
        for (uint32_t i = 0; i < n; i++) 
        {
            if (ipiv[i] != static_cast<lapack_int>(i + 1)) 
            {
                swap_count++;
            }
        }
        
        Matrix L(n, n, 0.0);
        Matrix U(n, n, 0.0);
        
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < n; i++) {
            L(i, i) = 1.0;
            for (uint32_t j = 0; j < i; j++) {
                L(i, j) = LU(i, j);
            }
            for (uint32_t j = i; j < n; j++) {
                U(i, j) = LU(i, j);
            }
        }
        
        Matrix P = Matrix::identity(n);
        for (uint32_t i = 0; i < n; i++) {
            lapack_int pivot = ipiv[i] - 1;
            if (pivot != static_cast<lapack_int>(i)) {
                // 交换 P 的第 i 行和第 pivot 行
                for (uint32_t j = 0; j < n; j++) {
                    std::swap(P(i, j), P(pivot, j));
                }
            }
        }
        
        return {L, U, P, swap_count};
    }
    Matrix Matrix::inverse() const
    {
        if (! isSquare())
        {
            throw std::invalid_argument("矩阵不是方阵");
        }
        Matrix inv(rows_, cols_);
        
        try
        {
            uint32_t n = rows_;
            Matrix inv = *this;
            lapack_int info;
            lapack_int nn = static_cast<lapack_int>(n);

            // 1. Cholesky 分解
            info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nn, inv.data.data(), nn);
            if (info != 0) throw std::runtime_error("矩阵不是正定的");

            // 2. 用 Cholesky 因子求逆
            info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', nn, inv.data.data(), nn);
            if (info != 0) throw std::runtime_error("求逆失败");

            // 3. 对称填充（dpotri 只填充下三角）

            for (uint32_t i = 0; i < n; i++) 
            {
                for (uint32_t j = 0; j < i; j++) {
                    inv(i, j) = inv(j, i);
                }
            }

            return inv;
        }
        catch (const std::exception&) 
        {
            auto [L, U, P, swap_count] = lu();
            uint32_t n = rows_;
        
            
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
            #endif
            for (uint32_t col = 0; col < n; col++) 
            {
                std::vector<double> pb(n, 0.0);
                
                for (uint32_t i = 0; i < n; i++) 
                {
                    pb[i] = P(i, col);
                }
            
            
                std::vector<double> y(n, 0.0);
                for (uint32_t i = 0; i < n; i++) 
                {
                    double sum = pb[i];
                    for (uint32_t j = 0; j < i; j++) 
                    {
                        sum -= L(i, j) * y[j];
                    }
                    y[i] = sum;
                }
            
                std::vector<double> x(n, 0.0);
                for (int i = n - 1; i >= 0; i--) 
                {
                    double sum = y[i];
                    for (uint32_t j = i + 1; j < n; j++) 
                    {
                        sum -= U(i, j) * x[j];
                    }
                    x[i] = sum / U(i, i);
                }
            
                for (uint32_t i = 0; i < n; i++) 
                {
                    inv(i, col) = x[i];
                }
            }
        
            return inv;
        }
    }
    double Matrix::determinant() const
    {
        if (!isSquare()) 
        {
            throw std::invalid_argument("矩阵必须是方阵");
        }
        if (rows_ == 1) 
        {
            return (*this)(0, 0);
        }
        if (rows_ == 2) 
        {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        }
    
        auto [L, U, P, swap_count] = lu();
    
        double det_sign = (swap_count % 2 == 0) ? 1.0 : -1.0;
    
        double det_U = 1.0;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(*:det_U) schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; ++i) {
            det_U *= U(i, i);
        }
    
        return det_sign * det_U;
    }

    // 静态工厂方法
    Matrix Matrix::identity(uint32_t n)
    {
        Matrix I(n, n, 0.0);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(n > 4096)
        #endif
        for (uint32_t i = 0; i < n; i++)
        {
            I(i, i) = 1.0;
        }
        return I;
    }
    Matrix Matrix::zeros(uint32_t rows, uint32_t cols)
    {
        return Matrix(rows, cols, 0.0);
    }
    Matrix Matrix::ones(uint32_t rows, uint32_t cols)
    {
        return Matrix(rows, cols, 1.0);
    }
    Matrix Matrix::random(uint32_t rows, uint32_t cols, double min, double max)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min, max);
    
        Matrix result(rows, cols);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows * cols > 4096)
        #endif
        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++) 
            {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    // 运算符重载
    Matrix Matrix::operator+(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("相加矩阵的维度不匹配");
        }

        Matrix sum(rows_, cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return sum;
    }
    Matrix Matrix::operator-(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("相减矩阵的维度不匹配");
        }

        Matrix sum(rows_, cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) - other(i, j);
            }
        }
        return sum;
    }
    Matrix Matrix::operator*(const Matrix& other) const
    {
        // 修正原有实现的两处错误：
        //   1) 维度检查写成 cols_ != other.cols_（应为 cols_ != other.rows_）
        //   2) 结果矩阵建成 (rows_, cols_)（应为 (rows_, other.cols_)）
        // 这两处错误使非方阵乘法结果错误甚至越界。
        if (cols_ != other.rows_)
        {
            throw std::invalid_argument("相乘矩阵的维度不匹配");
        }

        Matrix C(rows_, other.cols_);
        if (rows_ == 0 || cols_ == 0 || other.cols_ == 0)
        {
            return C;
        }

        // 行主序 A(m×k)·B(k×n)=C(m×n)：
        //   A 的 ld = k = cols_，B 与 C 的 ld = n = other.cols_
        // 交给 MKL dgemm（自带多线程 + AVX2/FMA/AVX-512），比手写三重循环快一个量级。
        const MKL_INT m = to_mkl_int(rows_);
        const MKL_INT n = to_mkl_int(other.cols_);
        const MKL_INT k = to_mkl_int(cols_);
        const double alpha = 1.0;
        const double beta  = 0.0;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    alpha,
                    data.data(),       k,
                    other.data.data(), n,
                    beta,
                    C.data.data(),     n);

        return C;
    }
    Matrix Matrix::operator+(double scalar) const
    {
        Matrix sum(rows_, cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) + scalar;
            }
        }
        return sum;
    }
    Matrix Matrix::operator-(double scalar) const
    {
        Matrix sum(rows_, cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) - scalar;
            }
        }
        return sum;
    }
    Matrix Matrix::operator*(double scalar) const
    {
        Matrix sum(rows_, cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) * scalar;
            }
        }
        return sum;
    }
    Matrix Matrix::operator/(double scalar) const
    {
        if (std::fabs(scalar) < 1e-15) 
        {  // 使用容差
            throw std::invalid_argument("除数太小或为 0");
        }
        Matrix sum(rows_, cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) / scalar;
            }
        }
        return sum;
    }

    // 友元函数（运算符重载）
    Matrix operator*(double scalar, const Matrix& mat)
    {
        Matrix prod(mat.rows_, mat.cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(mat.rows_ * mat.cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                prod(i, j) = scalar * mat(i, j);
            }
        }

        return prod;
    }
    Matrix operator+(double scalar, const Matrix& mat)
    {
        Matrix prod(mat.rows_, mat.cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(mat.rows_ * mat.cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                prod(i, j) = scalar + mat(i, j);
            }
        }
        
        return prod;
    }
    Matrix operator-(double scalar, const Matrix& mat)
    {
        Matrix prod(mat.rows_, mat.cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(mat.rows_ * mat.cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                prod(i, j) = scalar - mat(i, j);
            }
        }
        
        return prod;
    }
    Matrix operator/(double scalar, const Matrix& mat)
    {
        Matrix prod(mat.rows_, mat.cols_);
        int ok = 1;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(*:ok) schedule(static) if(mat.rows_ * mat.cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                prod(i, j) = scalar / mat(i, j);
                if (std::fabs(mat(i, j)) < 1e-15)
                {
                    ok = 0;
                }
            }
        }
        if (ok == 0)
        {
            throw std::invalid_argument("除数矩阵含 0");
        }
        return prod;
    }

    // 元素级运算
    Matrix Matrix::elementwise_multiply(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("元素相乘矩阵的维度不匹配");
        }

        Matrix sum(rows_, cols_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) * other(i, j);
            }
        }
        return sum;
    }
    Matrix Matrix::elementwise_divide(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("元素相除矩阵的维度不匹配");
        }

        Matrix sum(rows_, cols_);
        int ok = 1;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(*:ok) schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum(i, j) = (*this)(i, j) / other(i, j);
                if (std::fabs(other(i, j)) < 1e-15)
                {
                    ok = 0;
                }
            }
        }

        if (ok == 0)
        {
            throw std::invalid_argument("除数矩阵包含接近 0 的元素");
        }
        return sum;
    }

    // 向量化运算
    Matrix Matrix::abs() const
    {
        Matrix out(rows_, cols_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdAbs(n, data.data(), out.data.data());
        return out;
    }
    Matrix Matrix::sin() const
    {
        Matrix out(rows_, cols_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdSin(n, data.data(), out.data.data());
        return out;
    }
    Matrix Matrix::cos() const
    {
        Matrix out(rows_, cols_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdCos(n, data.data(), out.data.data());
        return out;
    }
    Matrix Matrix::exp() const
    {
        Matrix out(rows_, cols_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdExp(n, data.data(), out.data.data());
        return out;
    }
    Matrix Matrix::log() const
    {
        Matrix out(rows_, cols_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        if (!check_all(data, [](double v) { return v > 0.0; }))
        {
            throw std::invalid_argument("对数运算不能作用在小于等于 0 的值上");
        }
        vdLn(n, data.data(), out.data.data());
        return out;
    }
    Matrix Matrix::sqrt() const
    {
        Matrix out(rows_, cols_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        if (!check_all(data, [](double v) { return v >= 0.0; }))
        {
            throw std::invalid_argument("根号运算不能作用在小于 0 的值上");
        }
        vdSqrt(n, data.data(), out.data.data());
        return out;
    }
    Matrix Matrix::pow(double exponent) const
    {
        Matrix out(rows_, cols_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        // vdPowx: r[i] = a[i] ^ b，b 为标量（省掉一份 n 长度的指数数组）
        vdPowx(n, data.data(), exponent, out.data.data());
        if (!check_all(out.data, [](double v) { return std::isfinite(v); }))
        {
            throw std::invalid_argument("幂运算不合理");
        }
        return out;
    }

    // 统计类函数
    double Matrix::sum() const
    {
        double sum = 0.0;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:sum) schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                sum += (*this)(i, j);
            }
        }

        return sum;
    }
    double Matrix::mean() const
    {
        return sum() / (rows_ * cols_);
    }
    std::tuple<double, uint32_t, uint32_t> Matrix::max() const
    {
        double max_val = data[0];
        uint32_t posrow = 0;
        uint32_t poscol = 0;

        const int max_threads = omp_get_max_threads();
        std::vector<uint32_t> posrowlist(max_threads, 0);
        std::vector<uint32_t> poscollist(max_threads, 0);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(max:max_val) schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                if (max_val < (*this)(i, j))
                {
                    max_val = (*this)(i, j);
                    posrowlist[omp_get_thread_num()] = i;
                    poscollist[omp_get_thread_num()] = j;
                }
            }
        }

        for (int i = 0; i < max_threads; i++)
        {
            if (max_val == (*this)(posrowlist[i], poscollist[i]))
            {
                posrow = posrowlist[i];
                poscol = poscollist[i];
                break;
            }
        }

        return {max_val, posrow, poscol};
    }
    std::tuple<double, uint32_t, uint32_t> Matrix::min() const
    {
        double min_val = data[0];
        uint32_t posrow = 0;
        uint32_t poscol = 0;

        const int max_threads = omp_get_max_threads();
        std::vector<uint32_t> posrowlist(max_threads, 0);
        std::vector<uint32_t> poscollist(max_threads, 0);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(min:min_val) schedule(static) if(rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                if (min_val > (*this)(i, j))
                {
                    min_val = (*this)(i, j);
                    posrowlist[omp_get_thread_num()] = i;
                    poscollist[omp_get_thread_num()] = j;
                }
            }
        }

        for (int i = 0; i < max_threads; i++)
        {
            if (min_val == (*this)(posrowlist[i], poscollist[i]))
            {
                posrow = posrowlist[i];
                poscol = poscollist[i];
                break;
            }
        }

        return {min_val, posrow, poscol};
    }

    Tensor::Tensor() : rows_(0), cols_(0), heis_(0) {}
    Tensor::Tensor(uint32_t rows, uint32_t cols, uint32_t heis) : data(rows * cols * heis, 0.0), rows_(rows), cols_(cols), heis_(heis) {}
    Tensor::Tensor(uint32_t rows, uint32_t cols, uint32_t heis, double init_value) : data(rows * cols * heis, init_value), rows_(rows), cols_(cols), heis_(heis) {}
    Tensor::Tensor(const std::vector<std::vector<std::vector<double>>>& init_data)
    {
        if (init_data.empty()) 
        {
            rows_ = 0;
            cols_ = 0;
            heis_ = 0;
            return;
        }
        heis_ = init_data.size();
        rows_ = init_data[0].size();
        cols_ = init_data[0][0].size();
        data.resize(rows_ * cols_ * heis_);
        for (uint32_t i = 0; i < heis_; ++i) 
        {
            if (init_data[i].size() != rows_) 
            {
                throw std::invalid_argument("张量的所有二维分量必须具有相同形状");
            }
            for (uint32_t j = 0; j < rows_; j++)
            {
                if (init_data[i][j].size() != cols_) 
                {
                    throw std::invalid_argument("张量的所有二维分量必须具有相同形状");
                }
            }
        }
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < heis_; ++i) 
        {
            for (uint32_t j = 0; j < rows_; ++j) 
            {
                for (uint32_t k = 0; k < cols_; ++k) 
                {
                    data[i * rows_ * cols_ + j * cols_ + k] = init_data[i][j][k];
                }
            }
        }
    }
    Tensor::Tensor(const std::vector<double>& init_data)
    {
        if (init_data.empty()) 
        {
            rows_ = 0;
            cols_ = 0;
            heis_ = 0;
            return;
        }
        heis_ = 1;
        rows_ = init_data.size();
        cols_ = 1;
        data.resize(rows_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t k = 0; k < rows_; ++k) 
        {
            data[k] = init_data[k];
        }
    }
    Tensor::Tensor(uint32_t rows, uint32_t cols, uint32_t heis, const std::vector<double>& flat_data) : data(flat_data), rows_(rows), cols_(cols), heis_(heis) 
    {
        if (flat_data.size() != rows * cols * heis) 
        {
            throw std::invalid_argument("Matrix: 需要矩阵化的向量的长度必须等同于行数乘以列数");
        }
    }

    // 与 Matrix 和 vector 的转换
    Matrix Tensor::toMatrix() const
    {
        if (heis_ == 1)
        {
            return Matrix(rows_, cols_, data);
        }
        else
        {
            throw std::invalid_argument("张量的高大于 1，无法转换成矩阵");
        }
    }
    std::vector<std::vector<std::vector<double>>> Tensor::toVector() const
    {
        std::vector<std::vector<std::vector<double>>> result(heis_,
        std::vector<std::vector<double>>(rows_,
            std::vector<double>(cols_, 0.0)));

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(heis_ * rows_ * cols_ > 4096)
        #endif
        for (uint32_t i = 0; i < heis_; ++i)
        {
            for (uint32_t j = 0; j < rows_; ++j)
            {
                for (uint32_t k = 0; k < cols_; ++k)
                {
                    result[i][j][k] = data[i * rows_ * cols_ + j * cols_ + k];
                }
            }
        }

        return result;
    }

    // Matrix 访问元素
    Tensor Tensor::operator()(const Slice& r, const Slice& c, const Slice& h) const
    {
        if (r.end > rows_ || c.end > cols_ || h.end > heis_ ||
            r.start > r.end || c.start > c.end || h.start > h.end)
        {
            throw std::out_of_range("Tensor::operator() 索引越界");
        }

        const uint32_t r_len = r.size();
        const uint32_t c_len = c.size();
        const uint32_t h_len = h.size();

        Tensor result;
        result.rows_ = r_len;
        result.cols_ = c_len;
        result.heis_ = h_len;
        result.data.resize(r_len * c_len * h_len);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(r_len * c_len * h_len > 4096)
        #endif
        for (uint32_t k = 0; k < h_len; ++k)          // 遍历 heis 维度
        {
            const uint32_t src_k = h.start + k;
            for (uint32_t j = 0; j < r_len; ++j)      // 遍历 rows 维度
            {
                const uint32_t src_j = r.start + j;

                const double* src = &data[src_k * rows_ * cols_
                                        + src_j * cols_
                                        + c.start];

                double* dst = &result.data[k * r_len * c_len
                                        + j * c_len];

                std::copy(src, src + c_len, dst);
            }
        }

        return result;
    }

    // 打印矩阵
    void Tensor::print(uint32_t max_rows, uint32_t max_cols, uint32_t max_heis) const 
    {
        if (rows_ <= max_rows && cols_ <= max_cols && heis_ <= max_heis) 
        {
            for (uint32_t k = 0; k < heis_; k++) 
            {
                std::cout << "[";
                for (uint32_t i = 0; i < rows_; i++) 
                {
                    for (uint32_t j = 0; j < cols_; j++) 
                    {
                    
                        std::cout << std::setw(12) << std::fixed << std::setprecision(6) << (*this)(i, j, k);
                    }
                    if (i == rows_ - 1)
                    {
                        std::cout << "]";
                    }
                    std::cout << std::endl;
                }
            }
        }
        else
        {
            uint32_t half_rows = max_rows / 2;
            uint32_t half_cols = max_cols / 2;

            for (uint32_t i = 0; i < rows_; i++) 
            {
                bool row_skip = (i >= half_rows && i < rows_ - half_rows);
                if (row_skip) 
                {
                    if (i == half_rows) {
                        std::cout << "   ..." << std::endl;
                    }
                    continue;
                }

                for (uint32_t j = 0; j < cols_; j++) 
                {
                    bool col_skip = (j >= half_cols && j < cols_ - half_cols);
                    if (col_skip) 
                    {
                        if (j == half_cols) {
                            std::cout << std::setw(12) << "...";
                        }
                        continue;
                    }

                    std::cout << std::setw(12) << std::fixed << std::setprecision(6) << (*this)(i, j, 1);
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // 运算符重载
    Tensor Tensor::operator+(const Tensor& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_ || heis_ != other.heis_)
        {
            throw std::invalid_argument("相加张量的维度不匹配");
        }

        Tensor sum(rows_, cols_, heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) + other(i, j, k);
                }
            }
        }
        return sum;
    }
    Tensor Tensor::operator-(const Tensor& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_ || heis_ != other.heis_)
        {
            throw std::invalid_argument("相减张量的维度不匹配");
        }

        Tensor sum(rows_, cols_, heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) - other(i, j, k);
                }
            }
        }
        return sum;
    }
    Tensor Tensor::operator+(double scalar) const
    {
        Tensor sum(rows_, cols_, heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) + scalar;
                }
            }
        }
        return sum;
    }
    Tensor Tensor::operator-(double scalar) const
    {
        Tensor sum(rows_, cols_, heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) - scalar;
                }
            }
        }
        return sum;
    }
    Tensor Tensor::operator*(double scalar) const
    {
        Tensor sum(rows_, cols_, heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) * scalar;
                }
            }
        }
        return sum;
    }
    Tensor Tensor::operator/(double scalar) const
    {
        if (std::fabs(scalar) < 1e-15) 
        {  // 使用容差
            throw std::invalid_argument("除数太小或为 0");
        }
        Tensor sum(rows_, cols_, heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) / scalar;
                }
            }
        }
        return sum;
    }

    // 友元函数（运算符重载）
    Tensor operator*(double scalar, const Tensor& mat)
    {
        Tensor prod(mat.rows_, mat.cols_, mat.heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                for (uint32_t k = 0; k < mat.heis_; k++)
                {
                    prod(i, j, k) = scalar * mat(i, j, k);
                }
            }
        }
        return prod;
    }
    Tensor operator+(double scalar, const Tensor& mat)
    {
        Tensor prod(mat.rows_, mat.cols_, mat.heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                for (uint32_t k = 0; k < mat.heis_; k++)
                {
                    prod(i, j, k) = scalar + mat(i, j, k);
                }
            }
        }
        return prod;
    }
    Tensor operator-(double scalar, const Tensor& mat)
    {
        Tensor prod(mat.rows_, mat.cols_, mat.heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                for (uint32_t k = 0; k < mat.heis_; k++)
                {
                    prod(i, j, k) = scalar - mat(i, j, k);
                }
            }
        }
        return prod;
    }
    Tensor operator/(double scalar, const Tensor& mat)
    {
        Tensor prod(mat.rows_, mat.cols_, mat.heis_);
        int ok = 1;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(*:ok) schedule(static)
        #endif
        for (uint32_t i = 0; i < mat.rows_; i++)
        {
            for (uint32_t j = 0; j < mat.cols_; j++)
            {
                for (uint32_t k = 0; k < mat.heis_; k++)
                {
                    prod(i, j, k) = scalar / mat(i, j, k);
                    if (std::fabs(mat(i, j, k)) < 1e-15)
                    {
                        ok = 0;
                    }
                }
                
            }
        }
        if (ok == 0)
        {
            throw std::invalid_argument("除数矩阵含 0");
        }
        return prod;
    }

    // 元素级运算
    Tensor Tensor::elementwise_multiply(const Tensor& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_ || heis_ != other.heis_)
        {
            throw std::invalid_argument("元素相乘矩阵的维度不匹配");
        }

        Tensor sum(rows_, cols_, heis_);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) * other(i, j, k);
                }
            }
        }
        return sum;
    }
    Tensor Tensor::elementwise_divide(const Tensor& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_ || heis_ != other.heis_)
        {
            throw std::invalid_argument("元素相除矩阵的维度不匹配");
        }

        Tensor sum(rows_, cols_, heis_);
        int ok = 1;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(*:ok) schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum(i, j, k) = (*this)(i, j, k) / other(i, j, k);
                    if (std::fabs(other(i, j, k)) < 1e-15)
                    {
                        ok = 0;
                    }
                }
            }
        }

        if (ok == 0)
        {
            throw std::invalid_argument("除数矩阵包含接近 0 的元素");
        }
        return sum;
    }

    // 向量化运算
    Tensor Tensor::abs() const
    {
        Tensor out(rows_, cols_, heis_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdAbs(n, data.data(), out.data.data());
        return out;
    }
    Tensor Tensor::sin() const
    {
        Tensor out(rows_, cols_, heis_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdSin(n, data.data(), out.data.data());
        return out;
    }
    Tensor Tensor::cos() const
    {
        Tensor out(rows_, cols_, heis_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdCos(n, data.data(), out.data.data());
        return out;
    }
    Tensor Tensor::exp() const
    {
        Tensor out(rows_, cols_, heis_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdExp(n, data.data(), out.data.data());
        return out;
    }
    Tensor Tensor::log() const
    {
        Tensor out(rows_, cols_, heis_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        if (!check_all(data, [](double v) { return v > 0.0; }))
        {
            throw std::invalid_argument("对数运算不能作用在小于等于 0 的值上");
        }
        vdLn(n, data.data(), out.data.data());
        return out;
    }
    Tensor Tensor::sqrt() const
    {
        Tensor out(rows_, cols_, heis_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        if (!check_all(data, [](double v) { return v >= 0.0; }))
        {
            throw std::invalid_argument("根号运算不能作用在小于 0 的值上");
        }
        vdSqrt(n, data.data(), out.data.data());
        return out;
    }
    Tensor Tensor::pow(double exponent) const
    {
        Tensor out(rows_, cols_, heis_);
        const MKL_INT n = to_mkl_int(data.size());
        if (n == 0) { return out; }
        vdPowx(n, data.data(), exponent, out.data.data());
        if (!check_all(out.data, [](double v) { return std::isfinite(v); }))
        {
            throw std::invalid_argument("幂运算不合理");
        }
        return out;
    }

    // 统计类函数
    double Tensor::sum() const
    {
        double sum = 0.0;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:sum) schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    sum += (*this)(i, j, k);
                }
            }
        }
        return sum;
    }
    double Tensor::mean() const
    {
        return sum() / (rows_ * cols_ * heis_);
    }
    std::tuple<double, uint32_t, uint32_t, uint32_t> Tensor::max() const
    {
        double max_val = data[0];
        uint32_t posrow = 0;
        uint32_t poscol = 0;
        uint32_t poshei = 0;

        const int max_threads = omp_get_max_threads();
        std::vector<uint32_t> posrowlist(max_threads, 0);
        std::vector<uint32_t> poscollist(max_threads, 0);
        std::vector<uint32_t> posheilist(max_threads, 0);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(max:max_val) schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    if (max_val < (*this)(i, j, k))
                    {
                        max_val = (*this)(i, j, k);
                        posrowlist[omp_get_thread_num()] = i;
                        poscollist[omp_get_thread_num()] = j;
                        posheilist[omp_get_thread_num()] = k;
                    }
                }
            }
        }

        for (int i = 0; i < max_threads; i++)
        {
            if (max_val == (*this)(posrowlist[i], poscollist[i], posheilist[i]))
            {
                posrow = posrowlist[i];
                poscol = poscollist[i];
                poshei = posheilist[i];
                break;
            }
        }

        return {max_val, posrow, poscol, poshei};
    }
    std::tuple<double, uint32_t, uint32_t, uint32_t> Tensor::min() const
    {
        double max_val = data[0];
        uint32_t posrow = 0;
        uint32_t poscol = 0;
        uint32_t poshei = 0;

        const int max_threads = omp_get_max_threads();
        std::vector<uint32_t> posrowlist(max_threads, 0);
        std::vector<uint32_t> poscollist(max_threads, 0);
        std::vector<uint32_t> posheilist(max_threads, 0);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(min:max_val) schedule(static)
        #endif
        for (uint32_t i = 0; i < rows_; i++)
        {
            for (uint32_t j = 0; j < cols_; j++)
            {
                for (uint32_t k = 0; k < heis_; k++)
                {
                    if (max_val > (*this)(i, j, k))
                    {
                        max_val = (*this)(i, j, k);
                        posrowlist[omp_get_thread_num()] = i;
                        poscollist[omp_get_thread_num()] = j;
                        posheilist[omp_get_thread_num()] = k;
                    }
                }
            }
        }

        for (int i = 0; i < max_threads; i++)
        {
            if (max_val == (*this)(posrowlist[i], poscollist[i], posheilist[i]))
            {
                posrow = posrowlist[i];
                poscol = poscollist[i];
                poshei = posheilist[i];
                break;
            }
        }

        return {max_val, posrow, poscol, poshei};
    }


}

namespace sparse
{
    SMatrix::SMatrix() : rows_(0), cols_(0) {}
    SMatrix::SMatrix(uint32_t rows, uint32_t cols, const std::vector<double>& values, const std::vector<uint32_t>& rowlist, const std::vector<uint32_t>& collist) : rows_(rows), cols_(cols)
    {
        if (values.size() != rowlist.size() || values.size() != collist.size())
        {
            throw std::invalid_argument("COO 输入错误：values, rowlist, collist 长度不一致");
        }

        const uint32_t input_nnz = values.size();

        using Entry = std::tuple<uint32_t, uint32_t, double>;
        std::vector<Entry> entries;
        entries.reserve(input_nnz);

        for (uint32_t k = 0; k < input_nnz; ++k)
        {
            const uint32_t i = rowlist[k];
            const uint32_t j = collist[k];

            if (i >= rows_ || j >= cols_)
            {
                throw std::out_of_range("COO 输入错误：行号或列号越界");
            }

            const double v = values[k];

            if (v != 0.0)
            {
                entries.emplace_back(i, j, v);
            }
        }

        // 按行、列排序
        std::sort(
            entries.begin(),
            entries.end(),
            [](const Entry& a, const Entry& b)
            {
                if (std::get<0>(a) != std::get<0>(b))
                {
                    return std::get<0>(a) < std::get<0>(b);
                }
                return std::get<1>(a) < std::get<1>(b);
            });

        // 合并重复的 (i,j)
        std::vector<Entry> merged;
        merged.reserve(entries.size());

        for (const auto& e : entries)
        {
            const uint32_t i = std::get<0>(e);
            const uint32_t j = std::get<1>(e);
            const double v = std::get<2>(e);

            if (!merged.empty() &&
                std::get<0>(merged.back()) == i &&
                std::get<1>(merged.back()) == j)
            {
                std::get<2>(merged.back()) += v;
            }
            else
            {
                merged.emplace_back(i, j, v);
            }
        }

        // 去掉合并后变成 0 的元素
        entries.clear();
        entries.reserve(merged.size());

        for (const auto& e : merged)
        {
            if (std::get<2>(e) != 0.0)
            {
                entries.push_back(e);
            }
        }

        const uint32_t nnz = entries.size();

        values_.assign(nnz, 0.0);
        col_idx_.assign(nnz, 0);
        row_ptr_.assign(rows_ + 1, 0);

        // 统计每一行非零元个数
        for (const auto& e : entries)
        {
            const uint32_t i = std::get<0>(e);
            row_ptr_[i + 1]++;
        }

        // 前缀和生成 row_ptr
        for (uint32_t i = 0; i < rows_; ++i)
        {
            row_ptr_[i + 1] += row_ptr_[i];
        }

        // 填充 values_ 和 col_idx_
        std::vector<uint32_t> offset = row_ptr_;

        for (const auto& e : entries)
        {
            const uint32_t i = std::get<0>(e);
            const uint32_t j = std::get<1>(e);
            const double v = std::get<2>(e);

            const uint32_t pos = offset[i]++;
            values_[pos] = v;
            col_idx_[pos] = j;
        }
    }
    SMatrix::SMatrix(uint32_t rows, uint32_t cols,std::vector<uint32_t> row_ptr,std::vector<uint32_t> col_idx,std::vector<double> values)
        : rows_(rows), cols_(cols),
          row_ptr_(std::move(row_ptr)),
          col_idx_(std::move(col_idx)),
          values_(std::move(values))
    {
    }

    uint32_t SMatrix::rows() const { return rows_; }
    uint32_t SMatrix::cols() const { return cols_; }
    uint32_t SMatrix::nnz()  const { return values_.size(); }

    const std::vector<double>& SMatrix::values() const { return values_; }

    const std::vector<uint32_t>& SMatrix::row_ptr() const { return row_ptr_; }

    const std::vector<uint32_t>& SMatrix::col_idx() const { return col_idx_; }

    // CSR 转置（setup 一次性，串行 O(nnz)）
    SMatrix SMatrix::transpose() const
    {
        const uint32_t n = rows(), m = cols();
        const auto& Ap = row_ptr();
        const auto& Ac = col_idx();
        const auto& Av = values();

        // 计数 -> 前缀和
        std::vector<uint32_t> rp(m + 1, 0);
        for (uint32_t p = 0; p < Ap[n]; ++p)
        {
            rp[Ac[p] + 1]++;
        }
        for (uint32_t i = 0; i < m; ++i)
        {
            rp[i + 1] += rp[i];
        }

        // 填充（每列区间不重叠，天然有序）
        std::vector<uint32_t> ci(Ap[n]);
        std::vector<double> vv(Ap[n]);
        std::vector<uint32_t> off = rp;
        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
            {
                const uint32_t j = Ac[p];
                ci[off[j]] = i;
                vv[off[j]++] = Av[p];
            }
        }

        return SMatrix(m, n, rp, ci, vv);   // CSR 直构
    }

    // 转为小写
    std::string to_lower(std::string s)
    {
        std::transform(
            s.begin(),
            s.end(),
            s.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            }
        );
        return s;
    }
    // 稠密 LU 分解（带部分主元），用于 AMG 最粗层直接求解
    static void dense_lu_factor(const SMatrix& A, std::vector<double>& mat, std::vector<uint32_t>& piv)
    {
        const uint32_t n = A.rows();
        mat.assign(n * n, 0.0);
        piv.resize(n);
        
        const auto& Ap = A.row_ptr();
        const auto& Ac = A.col_idx();
        const auto& Av = A.values();
        
        // 1. 提取稠密矩阵
        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
            {
                mat[i * n + Ac[p]] = Av[p];
            }
        }
        
        for (uint32_t i = 0; i < n; ++i) piv[i] = i;
        
        // 2. 高斯消元
        for (uint32_t k = 0; k < n; ++k)
        {
            // 寻找列主元
            uint32_t p_max = k;
            double v_max = std::fabs(mat[k * n + k]);
            for (uint32_t i = k + 1; i < n; ++i)
            {
                if (std::fabs(mat[i * n + k]) > v_max)
                {
                    v_max = std::fabs(mat[i * n + k]);
                    p_max = i;
                }
            }
            
            // 交换行
            if (p_max != k)
            {
                std::swap(piv[k], piv[p_max]);
                for (uint32_t j = 0; j < n; ++j) std::swap(mat[k * n + j], mat[p_max * n + j]);
            }
            
            // 防止除零（处理奇异或接近奇异的粗层矩阵）
            if (std::fabs(mat[k * n + k]) < 1e-14) 
            {
                mat[k * n + k] = 1e-14; 
            }
            
            // 消元
            for (uint32_t i = k + 1; i < n; ++i)
            {
                mat[i * n + k] /= mat[k * n + k];
                for (uint32_t j = k + 1; j < n; ++j)
                {
                    mat[i * n + j] -= mat[i * n + k] * mat[k * n + j];
                }
            }
        }
    }
    // 稠密 LU 求解
    static void dense_lu_solve(const std::vector<double>& mat, const std::vector<uint32_t>& piv, const std::vector<double>& b, std::vector<double>& x)
    {
        const uint32_t n = piv.size();
        x.resize(n);
        
        // 1. 应用行置换 (P * b)
        for (uint32_t i = 0; i < n; ++i) x[i] = b[piv[i]];
        
        // 2. 前代解 L * y = P * b
        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t j = 0; j < i; ++j) x[i] -= mat[i * n + j] * x[j];
        }
        
        // 3. 回代解 U * x = y
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i)
        {
            for (uint32_t j = i + 1; j < n; ++j) x[i] -= mat[i * n + j] * x[j];
            x[i] /= mat[i * n + i];
        }
    }

    // 运算符重载
    SMatrix SMatrix::operator*(const SMatrix& B) const
    {
        const uint32_t n = rows(), m = B.cols();

        if (cols() != B.rows())
        {
            throw std::invalid_argument("稀疏矩阵相乘：A 的列数必须等于 B 的行数");
        }
        if (n == 0 || m == 0)
        {
            return SMatrix(n, m, std::vector<uint32_t>(n + 1, 0),
                           std::vector<uint32_t>{}, std::vector<double>{});
        }

        const auto& Ap = row_ptr(); const auto& Ac = col_idx(); const auto& Av = values();
        const auto& Bp = B.row_ptr(); const auto& Bc = B.col_idx(); const auto& Bv = B.values();

        std::vector<uint32_t> rp(n + 1, 0);

        int num_threads = 1;
        #ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel
        {
            if (omp_get_thread_num() == 0) num_threads = omp_get_num_threads();
        }
        #endif
        if (num_threads < 1) { num_threads = 1; }

        // =====================================================================
        // 行分块 + 按“实际列跨度”分配 SPA（结果累加器）
        //
        // 旧实现：每个线程各持有一份长度为 m 的 marker(int32_t) + accum(double)，
        //   总量 O(T·m)。m=1e6、32 线程时仅这两个数组就额外常驻约 384 MB。
        // 新实现：把行切成若干块，每块只分配它真正触及的列区间 [cmin, cmax]，
        //   一次性拼进一条连续数组。块间由 OpenMP 并行、块内串行，因此同一时刻
        //   只有一个线程访问某块的 SPA —— 既不需要原子操作，也不会像“全局单 SPA”
        //   那样因 marker 写-写竞争而漏掉/重复列号。
        //   对带状/局部矩阵（FEM、有限差分），跨度≈带宽+块高，内存降到 O(n)；
        //   最坏情形（随机矩阵，每块跨度都是 m）退化为 O(T·m)，与旧实现持平。
        //
        // 另外修正了两趟共用的时间戳冲突：Pass1 用 +i，Pass2 用 -(i+1)，
        // 初始值取 INT32_MIN，三者互不相交，因此两趟之间无需重新清零 marker。
        // （旧实现两趟都用 i，若某行恰好是块内最后一个触及列 k 的行，
        //   Pass2 会误判为“已标记”而漏掉该列，导致结果与 rp 错位。）
        // =====================================================================
        const std::size_t budget = static_cast<std::size_t>(num_threads) * m;
        constexpr int32_t TS_FREE = std::numeric_limits<int32_t>::min();

        auto build_blocks = [&](uint32_t nblk,
                                std::vector<uint32_t>& begin,
                                std::vector<uint32_t>& cmin,
                                std::vector<uint32_t>& cmax) -> std::size_t
        {
            begin.assign(nblk + 1, 0);
            for (uint32_t b = 0; b <= nblk; ++b)
            {
                begin[b] = static_cast<uint32_t>((static_cast<uint64_t>(n) * b) / nblk);
            }

            // (1) 每块在 A 中引用到的 B 行区间 [jlo, jhi]
            std::vector<uint32_t> jlo(nblk), jhi(nblk), has(nblk);
            #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(nblk); ++b)
            {
                uint32_t lo = 0, hi = 0;
                bool any = false;
                for (uint32_t i = begin[b]; i < begin[b + 1]; ++i)
                    for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                    {
                        const uint32_t j = Ac[p];
                        if (!any) { lo = hi = j; any = true; }
                        else { if (j < lo) { lo = j; } if (j > hi) { hi = j; } }
                    }
                has[b] = any ? 1u : 0u;
                jlo[b] = any ? lo : 0u;
                jhi[b] = any ? hi : 0u;
            }

            // (2) 由 B 的这些行推出列跨度（CSR 行内列号有序，取首/尾元素即可）
            cmin.assign(nblk, 0);
            cmax.assign(nblk, 0);
            #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(nblk); ++b)
            {
                if (!has[b]) { continue; }
                uint32_t lo = m, hi = 0;
                for (uint32_t j = jlo[b]; j <= jhi[b]; ++j)
                {
                    if (Bp[j + 1] <= Bp[j]) { continue; }      // B 的第 j 行为空
                    const uint32_t f = Bc[Bp[j]];
                    const uint32_t l = Bc[Bp[j + 1] - 1];
                    if (f < lo) { lo = f; }
                    if (l > hi) { hi = l; }
                }
                if (lo > hi) { continue; }                     // 整个区间都是空行
                cmin[b] = lo;
                cmax[b] = hi;
            }

            std::size_t total = 0;
            for (uint32_t b = 0; b < nblk; ++b)
            {
                total += static_cast<std::size_t>(cmax[b] - cmin[b]) + 1;
            }
            return total;
        };

        // 先按 4×线程数分块（留出调度余量以平衡负载）；
        // 若列跨度总和反而超过旧实现的 O(T·m)，退回 T 个块（此时上界恰为 T·m）。
        std::vector<uint32_t> blk_begin, blk_cmin, blk_cmax;
        uint32_t nblk = static_cast<uint32_t>(4 * num_threads);
        if (nblk < 1) { nblk = 1; }
        if (nblk > n) { nblk = n; }

        std::size_t total_span = build_blocks(nblk, blk_begin, blk_cmin, blk_cmax);
        if (total_span > budget)
        {
            nblk = static_cast<uint32_t>(num_threads);
            if (nblk < 1) { nblk = 1; }
            if (nblk > n) { nblk = n; }
            total_span = build_blocks(nblk, blk_begin, blk_cmin, blk_cmax);
        }

        std::vector<std::size_t> blk_off(nblk + 1, 0);
        for (uint32_t b = 0; b < nblk; ++b)
        {
            blk_off[b + 1] = blk_off[b]
                           + static_cast<std::size_t>(blk_cmax[b] - blk_cmin[b]) + 1;
        }

        std::vector<int32_t> marker(total_span, TS_FREE);
        std::vector<double>  accum (total_span, 0.0);

        // ================= Pass 1: 符号乘法 (Symbolic) =================
        #ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(nblk); ++b)
        {
            int32_t* mk = marker.data() + blk_off[b];
            const uint32_t base = blk_cmin[b];

            for (uint32_t i = blk_begin[b]; i < blk_begin[b + 1]; ++i)
            {
                const int32_t ts = static_cast<int32_t>(i);
                uint32_t nnz_i = 0;
                for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                {
                    const uint32_t j = Ac[p];
                    for (uint32_t q = Bp[j]; q < Bp[j + 1]; ++q)
                    {
                        const uint32_t kk = Bc[q] - base;
                        if (mk[kk] != ts)
                        {
                            mk[kk] = ts;
                            nnz_i++;
                        }
                    }
                }
                rp[i + 1] = nnz_i;
            }
        }

        for (uint32_t i = 0; i < n; ++i)
        {
            rp[i + 1] += rp[i];
        }

        std::vector<uint32_t> ci(rp[n]);
        std::vector<double>   vv(rp[n]);

        // ================= Pass 2: 数值乘法 (Numeric) =================
        #ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(nblk); ++b)
        {
            int32_t* mk = marker.data() + blk_off[b];
            double*  ac = accum.data()  + blk_off[b];
            const uint32_t base = blk_cmin[b];

            // 复用同一个 scratch，避免旧实现“每行一次堆分配”
            std::vector<uint32_t> cols_in_row;
            cols_in_row.reserve(64);

            for (uint32_t i = blk_begin[b]; i < blk_begin[b + 1]; ++i)
            {
                const int32_t ts = -static_cast<int32_t>(i) - 1;
                cols_in_row.clear();

                for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                {
                    const uint32_t j = Ac[p];
                    const double   a = Av[p];
                    for (uint32_t q = Bp[j]; q < Bp[j + 1]; ++q)
                    {
                        const uint32_t kk = Bc[q] - base;
                        if (mk[kk] != ts)
                        {
                            mk[kk] = ts;
                            cols_in_row.push_back(kk);
                        }
                        ac[kk] += a * Bv[q];
                    }
                }

                std::sort(cols_in_row.begin(), cols_in_row.end());

                uint32_t w = rp[i];
                for (uint32_t kk : cols_in_row)
                {
                    ci[w] = kk + base;
                    vv[w] = ac[kk];
                    ac[kk] = 0.0;                 // 稀疏清零，供下一行复用
                    ++w;
                }
            }
        }

        return SMatrix(n, m, rp, ci, vv);
    }

    // 预处理函数
    std::function<void(const std::vector<double>&, std::vector<double>&)> SMatrix::make_diag_preconditioner(const SMatrix& A)
    {
        const uint32_t n = A.rows();

        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("diag 预处理要求矩阵是方阵");
        }

        std::vector<double> diag(n, 0.0);

        const auto& row_ptr = A.row_ptr();
        const auto& col_idx = A.col_idx();
        const auto& values = A.values();
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(n > 1000)
        #endif
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii)
        {
            const uint32_t i = static_cast<uint32_t>(ii);

            for (uint32_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            {
                if (col_idx[k] == i)
                {
                    diag[i] = values[k];
                    break;
                }
            }
        }

        const double tiny = 1e-14;

        for (uint32_t i = 0; i < n; ++i)
        {
            if (std::fabs(diag[i]) < tiny)
            {
                throw std::runtime_error("diag 预处理失败：矩阵存在过小或为 0 的对角元");
            }

            if (diag[i] <= 0.0)
            {
                throw std::runtime_error("diag 预处理失败：对 CG/PCG，diag(A) 应为正");
            }
        }

        return [diag](const std::vector<double>& rhs, std::vector<double>& z)
        {
            const uint32_t n = rhs.size();
            z.resize(n);
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                z[i] = rhs[i] / diag[i];
            }
        };
    }
    // 用于存储 IC(0) 分解后的数据结构
    struct IC0_Data
    {
        std::vector<uint32_t> L_row_ptr;
        std::vector<uint32_t> L_col_idx;
        std::vector<double> L_values;

        std::vector<uint32_t> LT_row_ptr;
        std::vector<uint32_t> LT_col_idx;
        std::vector<double> LT_values;

        std::vector<double> diag;
    };
    std::function<void(const std::vector<double>&, std::vector<double>&)> SMatrix::make_ic0_preconditioner(const SMatrix& A)
    {
        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("IC(0) 预处理要求矩阵是方阵");
        }

        const uint32_t n = A.rows();
        const auto& A_row_ptr = A.row_ptr();
        const auto& A_col_idx = A.col_idx();
        const auto& A_values = A.values();

        auto data = std::make_shared<IC0_Data>();

        // ---------------------------------------------------------
        // 1. 提取 A 的下三角部分，构建 L 的 CSR 格式
        // ---------------------------------------------------------
        data->L_row_ptr.assign(n + 1, 0);

        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
            {
                if (A_col_idx[p] <= i)
                {
                    data->L_row_ptr[i + 1]++;
                }
            }
        }

        for (uint32_t i = 0; i < n; ++i)
        {
            data->L_row_ptr[i + 1] += data->L_row_ptr[i];
        }

        uint32_t L_nnz = data->L_row_ptr[n];
        data->L_col_idx.resize(L_nnz);
        data->L_values.resize(L_nnz);

        std::vector<uint32_t> offset = data->L_row_ptr;
        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
            {
                uint32_t j = A_col_idx[p];
                if (j <= i)
                {
                    uint32_t dest = offset[i]++;
                    data->L_col_idx[dest] = j;
                    data->L_values[dest] = A_values[p];
                }
            }
        }

        // ---------------------------------------------------------
        // 2. IC(0) 分解 (使用双指针法，极快)
        // ---------------------------------------------------------
        data->diag.assign(n, 0.0);
        const double tiny = 1e-14;

        for (uint32_t i = 0; i < n; ++i)
        {
            uint32_t row_start = data->L_row_ptr[i];
            uint32_t row_end = data->L_row_ptr[i + 1];

            // 处理非对角元 (j < i)
            for (uint32_t p = row_start; p < row_end; ++p)
            {
                uint32_t j = data->L_col_idx[p];
                if (j >= i) break; // 列号有序，遇到对角元即停止

                double sum = 0.0;
                
                // 双指针寻找第 i 行和第 j 行的公共列 k < j
                uint32_t p_i = row_start;
                uint32_t p_j = data->L_row_ptr[j];
                uint32_t end_j = data->L_row_ptr[j + 1];

                while (p_i < p && p_j < end_j)
                {
                    uint32_t k_i = data->L_col_idx[p_i];
                    uint32_t k_j = data->L_col_idx[p_j];

                    if (k_i == k_j)
                    {
                        sum += data->L_values[p_i] * data->L_values[p_j];
                        ++p_i;
                        ++p_j;
                    }
                    else if (k_i < k_j)
                    {
                        ++p_i;
                    }
                    else
                    {
                        ++p_j;
                    }
                }

                data->L_values[p] = (data->L_values[p] - sum) / data->diag[j];
            }

            // 处理对角元
            uint32_t diag_p = row_end; // 初始为无效值
            for (uint32_t p = row_start; p < row_end; ++p)
            {
                if (data->L_col_idx[p] == i)
                {
                    diag_p = p;
                    break;
                }
            }

            if (diag_p == row_end)
            {
                throw std::runtime_error("IC(0) 失败：矩阵缺少对角元");
            }

            double sum_diag = 0.0;
            for (uint32_t p = row_start; p < diag_p; ++p)
            {
                sum_diag += data->L_values[p] * data->L_values[p];
            }

            double diag_val = data->L_values[diag_p] - sum_diag;
            if (diag_val <= tiny)
            {
                throw std::runtime_error("IC(0) 失败：矩阵可能不是 SPD，或需要 shifted IC");
            }

            data->L_values[diag_p] = std::sqrt(diag_val);
            data->diag[i] = data->L_values[diag_p];
        }

        // ---------------------------------------------------------
        // 3. 构建 L^T 的 CSR 格式 (用于回代)
        // ---------------------------------------------------------
        data->LT_row_ptr.assign(n + 1, 0);

        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t p = data->L_row_ptr[i]; p < data->L_row_ptr[i + 1]; ++p)
            {
                uint32_t j = data->L_col_idx[p];
                if (j < i) // 只转置严格下三角
                {
                    data->LT_row_ptr[j + 1]++;
                }
            }
        }

        for (uint32_t i = 0; i < n; ++i)
        {
            data->LT_row_ptr[i + 1] += data->LT_row_ptr[i];
        }

        uint32_t LT_nnz = data->LT_row_ptr[n];
        data->LT_col_idx.resize(LT_nnz);
        data->LT_values.resize(LT_nnz);

        std::vector<uint32_t> offset_LT = data->LT_row_ptr;
        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t p = data->L_row_ptr[i]; p < data->L_row_ptr[i + 1]; ++p)
            {
                uint32_t j = data->L_col_idx[p];
                if (j < i)
                {
                    uint32_t dest = offset_LT[j]++;
                    data->LT_col_idx[dest] = i; // 转置后列号变行号
                    data->LT_values[dest] = data->L_values[p];
                }
            }
        }

        // ---------------------------------------------------------
        // 4. 返回 Lambda 表达式 (按值捕获 shared_ptr)
        // ---------------------------------------------------------
        return [data](const std::vector<double>& rhs, std::vector<double>& z)
        {
            const uint32_t n = data->diag.size();
            std::vector<double> y(n, 0.0);
            z.resize(n);

            // 前代: L y = rhs
            for (uint32_t i = 0; i < n; ++i)
            {
                double sum = rhs[i];
                for (uint32_t p = data->L_row_ptr[i]; p < data->L_row_ptr[i + 1]; ++p)
                {
                    uint32_t j = data->L_col_idx[p];
                    if (j >= i) break;
                    sum -= data->L_values[p] * y[j];
                }
                y[i] = sum / data->diag[i];
            }

            // 回代: L^T z = y
            for (std::ptrdiff_t ii = static_cast<std::ptrdiff_t>(n) - 1; ii >= 0; --ii)
            {
                uint32_t i = static_cast<uint32_t>(ii);
                double sum = y[i];
                for (uint32_t p = data->LT_row_ptr[i]; p < data->LT_row_ptr[i + 1]; ++p)
                {
                    uint32_t j = data->LT_col_idx[p]; // 这里的 j 实际上 > i
                    sum -= data->LT_values[p] * z[j];
                }
                z[i] = sum / data->diag[i];
            }
        };
    }
    struct ILU0_Data 
    {
        std::vector<uint32_t> L_row_ptr;
        std::vector<uint32_t> L_col_idx;
        std::vector<double> L_values;

        std::vector<uint32_t> U_row_ptr;
        std::vector<uint32_t> U_col_idx;
        std::vector<double> U_values;
        std::vector<uint32_t> diag_U_idx; // 记录 U 中每一行对角元的索引
    };
    std::function<void(const std::vector<double>&, std::vector<double>&)> SMatrix::make_ilu0_preconditioner(const SMatrix& A)
    {
        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("ILU(0) 预处理要求矩阵是方阵");
        }

        const uint32_t n = A.rows();
        const auto& A_row_ptr = A.row_ptr();
        const auto& A_col_idx = A.col_idx();
        const auto& A_values = A.values();

        auto data = std::make_shared<ILU0_Data>();

        // ---------------------------------------------------------
        // 1. 统计 L (严格下三角) 和 U (对角及上三角) 每行的非零元个数
        // ---------------------------------------------------------
        data->L_row_ptr.assign(n + 1, 0);
        data->U_row_ptr.assign(n + 1, 0);

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(n > 1000)
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        {
            uint32_t count_L = 0;
            uint32_t count_U = 0;
            for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
            {
                if (A_col_idx[p] < static_cast<uint32_t>(i))
                    count_L++;
                else
                    count_U++;
            }
            data->L_row_ptr[i + 1] = count_L;
            data->U_row_ptr[i + 1] = count_U;
        }

        // 前缀和 (串行，O(N) 极快)
        for (uint32_t i = 0; i < n; ++i)
        {
            data->L_row_ptr[i + 1] += data->L_row_ptr[i];
            data->U_row_ptr[i + 1] += data->U_row_ptr[i];
        }

        uint32_t L_nnz = data->L_row_ptr[n];
        uint32_t U_nnz = data->U_row_ptr[n];

        data->L_col_idx.resize(L_nnz);
        data->L_values.resize(L_nnz);
        data->U_col_idx.resize(U_nnz);
        data->U_values.resize(U_nnz);
        data->diag_U_idx.resize(n);

        // ---------------------------------------------------------
        // 2. 提取 A 的元素，分别填充到 L 和 U 的 CSR 结构中 (OpenMP 加速)
        // ---------------------------------------------------------
        std::vector<uint32_t> offset_L = data->L_row_ptr;
        std::vector<uint32_t> offset_U = data->U_row_ptr;
        std::vector<int> has_diag(n, 0); // 用于线程安全地检查对角元

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(n > 1000)
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        {
            for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
            {
                uint32_t j = A_col_idx[p];
                if (j < static_cast<uint32_t>(i))
                {
                    uint32_t dest = offset_L[i]++;
                    data->L_col_idx[dest] = j;
                    data->L_values[dest] = A_values[p];
                }
                else
                {
                    uint32_t dest = offset_U[i]++;
                    data->U_col_idx[dest] = j;
                    data->U_values[dest] = A_values[p];
                    if (j == static_cast<uint32_t>(i))
                    {
                        data->diag_U_idx[i] = dest;
                        has_diag[i] = 1;
                    }
                }
            }
        }

        for (uint32_t i = 0; i < n; ++i)
        {
            if (!has_diag[i])
            {
                throw std::runtime_error("ILU(0) 失败：矩阵缺少对角元");
            }
        }

        // ---------------------------------------------------------
        // 3. ILU(0) 分解 (串行，因高斯消元存在强数据依赖)
        // ---------------------------------------------------------
        const double tiny = 1e-30;

        for (uint32_t i = 0; i < n; ++i)
        {
            uint32_t row_start_L = data->L_row_ptr[i];
            uint32_t row_end_L = data->L_row_ptr[i + 1];
            
            uint32_t row_start_U_i = data->U_row_ptr[i];
            uint32_t row_end_U_i = data->U_row_ptr[i + 1];
            
            for (uint32_t p_L = row_start_L; p_L < row_end_L; ++p_L)
            {
                uint32_t k = data->L_col_idx[p_L];
                
                uint32_t diag_k = data->diag_U_idx[k];
                if (std::fabs(data->U_values[diag_k]) < tiny)
                {
                    throw std::runtime_error("ILU(0) breakdown: U_{k,k} 接近 0，可能需要主元旋转 (Pivoting)");
                }
                
                // L_{i,k} = L_{i,k} / U_{k,k}
                data->L_values[p_L] /= data->U_values[diag_k];
                
                uint32_t row_start_U_k = data->U_row_ptr[k];
                uint32_t row_end_U_k = data->U_row_ptr[k + 1];
                
                // 双指针法寻找 U 的第 i 行和第 k 行的公共列，极快更新
                uint32_t p_i = row_start_U_i;
                uint32_t p_k = row_start_U_k;
                
                while (p_i < row_end_U_i && p_k < row_end_U_k)
                {
                    uint32_t j_i = data->U_col_idx[p_i];
                    uint32_t j_k = data->U_col_idx[p_k];
                    
                    if (j_i == j_k)
                    {
                        data->U_values[p_i] -= data->L_values[p_L] * data->U_values[p_k];
                        ++p_i;
                        ++p_k;
                    }
                    else if (j_i < j_k)
                    {
                        ++p_i;
                    }
                    else
                    {
                        // j_k < j_i，意味着 A 的 (i, j_k) 为 0，产生 fill-in
                        // ILU(0) 的核心：直接丢弃这个 fill-in
                        ++p_k;
                    }
                }
            }
        }

        // ---------------------------------------------------------
        // 4. 返回 Lambda 表达式 (按值捕获 shared_ptr)
        // ---------------------------------------------------------
        return [data](const std::vector<double>& rhs, std::vector<double>& z)
        {
            const uint32_t n = data->diag_U_idx.size();
            std::vector<double> y(n, 0.0);
            z.resize(n);

            // 前代: L y = rhs (L 为单位下三角，对角元为 1，无需除以 L_ii)
            for (uint32_t i = 0; i < n; ++i)
            {
                double sum = rhs[i];
                for (uint32_t p = data->L_row_ptr[i]; p < data->L_row_ptr[i + 1]; ++p)
                {
                    uint32_t j = data->L_col_idx[p];
                    sum -= data->L_values[p] * y[j];
                }
                y[i] = sum;
            }

            // 回代: U z = y
            for (std::ptrdiff_t ii = static_cast<std::ptrdiff_t>(n) - 1; ii >= 0; --ii)
            {
                uint32_t i = static_cast<uint32_t>(ii);
                double sum = y[i];
                for (uint32_t p = data->U_row_ptr[i]; p < data->U_row_ptr[i + 1]; ++p)
                {
                    uint32_t j = data->U_col_idx[p];
                    if (j == i) continue; // 跳过对角元，稍后统一除以 U_ii
                    sum -= data->U_values[p] * z[j];
                }
                z[i] = sum / data->U_values[data->diag_U_idx[i]];
            }
        };
    }
    // 红黑/多色 着色 + 重排数据（sor_redblack 与 SSOR 共用）
    struct RB_Data
    {
        SMatrix A2;                      // 重排后矩阵 P A P^T
        std::vector<uint32_t> color_ptr;   // 第 c 色的新编号区间 [color_ptr[c], color_ptr[c+1])
        std::vector<uint32_t> new_id;      // old -> new
        std::vector<uint32_t> perm;        // new -> old
        std::vector<double> diag2;       // 重排后对角元
        uint32_t ncolor = 0;
        // apply 的工作缓冲区（solveM 为串行调用，线程安全）
        std::vector<double> f, w, z2;
    };
    std::shared_ptr<RB_Data> make_redblack_data(const SMatrix& A)
    {
        const uint32_t n = A.rows();
        const uint32_t nnz = A.nnz();
        const auto& A_row_ptr = A.row_ptr();
        const auto& A_col_idx = A.col_idx();
        const auto& A_values  = A.values();

        auto data = std::make_shared<RB_Data>();

        // ---- 转置模式 (CSC)，取 A + A^T 的图结构保证着色正确 ----
        std::vector<uint32_t> AT_col_ptr(n + 1, 0);
        for (uint32_t p = 0; p < nnz; ++p) AT_col_ptr[A_col_idx[p] + 1]++;
        for (uint32_t i = 0; i < n; ++i) AT_col_ptr[i + 1] += AT_col_ptr[i];
        std::vector<uint32_t> AT_row_idx(nnz);
        {
            std::vector<uint32_t> off = AT_col_ptr;
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
                    AT_row_idx[off[A_col_idx[p]]++] = i;
        }

        // ---- 着色：BFS 2-色（二部图 => 红黑）；失败回退贪心多色 ----
        std::vector<int> color(n, -1);
        bool bipartite = true;
        std::vector<uint32_t> queue;
        for (uint32_t seed = 0; seed < n && bipartite; ++seed)
        {
            if (color[seed] != -1) continue;
            color[seed] = 0;
            queue.clear();
            queue.push_back(seed);
            for (uint32_t head = 0; head < queue.size() && bipartite; ++head)
            {
                const uint32_t i = queue[head];
                auto visit = [&](uint32_t j)
                {
                    if (j == i) return;
                    if (color[j] == -1) { color[j] = 1 - color[i]; queue.push_back(j); }
                    else if (color[j] == color[i]) bipartite = false;
                };
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) visit(A_col_idx[p]);
                for (uint32_t p = AT_col_ptr[i]; p < AT_col_ptr[i + 1]; ++p) visit(AT_row_idx[p]);
            }
        }

        if (!bipartite)
        {
            std::fill(color.begin(), color.end(), -1);
            std::vector<char> used(n + 1, 0);
            std::vector<uint32_t> touched;
            for (uint32_t i = 0; i < n; ++i)
            {
                touched.clear();
                auto mark = [&](uint32_t j)
                {
                    if (j == i) return;
                    const int c = color[j];
                    if (c >= 0 && !used[c]) { used[c] = 1; touched.push_back(c); }
                };
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) mark(A_col_idx[p]);
                for (uint32_t p = AT_col_ptr[i]; p < AT_col_ptr[i + 1]; ++p) mark(AT_row_idx[p]);
                int c = 0;
                while (used[c]) ++c;
                color[i] = c;
                for (uint32_t c2 : touched) used[c2] = 0;
            }
        }

        // ---- 统计颜色数 + 按色重排 ----
        for (uint32_t i = 0; i < n; ++i)
            if (static_cast<uint32_t>(color[i]) + 1 > data->ncolor)
                data->ncolor = color[i] + 1;

        data->color_ptr.assign(data->ncolor + 1, 0);
        for (uint32_t i = 0; i < n; ++i) data->color_ptr[color[i] + 1]++;
        for (uint32_t c = 0; c < data->ncolor; ++c) data->color_ptr[c + 1] += data->color_ptr[c];

        data->new_id.resize(n);
        {
            std::vector<uint32_t> off = data->color_ptr;
            for (uint32_t i = 0; i < n; ++i) data->new_id[i] = off[color[i]]++;
        }
        data->perm.resize(n);
        for (uint32_t i = 0; i < n; ++i) data->perm[data->new_id[i]] = i;

        // ---- 重排矩阵 A2 = P A P^T ----
        std::vector<double> values2; values2.reserve(nnz);
        std::vector<uint32_t> row2; row2.reserve(nnz);
        std::vector<uint32_t> col2; col2.reserve(nnz);
        for (uint32_t ni = 0; ni < n; ++ni)
        {
            const uint32_t i = data->perm[ni];
            for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
            {
                row2.push_back(ni);
                col2.push_back(data->new_id[A_col_idx[p]]);
                values2.push_back(A_values[p]);
            }
        }
        data->A2 = SMatrix(n, n, values2, row2, col2);

        // ---- 对角元 + 缓冲区 ----
        const auto& r2 = data->A2.row_ptr();
        const auto& c2 = data->A2.col_idx();
        const auto& v2 = data->A2.values();
        data->diag2.assign(n, 0.0);
        for (uint32_t ni = 0; ni < n; ++ni)
        {
            bool found = false;
            for (uint32_t p = r2[ni]; p < r2[ni + 1]; ++p)
            {
                if (c2[p] == ni) { data->diag2[ni] = v2[p]; found = true; break; }
            }
            if (!found || std::fabs(data->diag2[ni]) < 1e-300)
                throw std::runtime_error("SSOR/红黑 失败：缺少非零对角元");
        }
        data->f.resize(n); data->w.resize(n); data->z2.resize(n);

        return data;
    }
    std::function<void(const std::vector<double>&, std::vector<double>&)> SMatrix::make_ssor_preconditioner(const SMatrix& A, double omega)
    {
        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("SSOR 预处理要求矩阵是方阵");
        }
        if (!(omega > 0.0 && omega < 2.0))
        {
            throw std::invalid_argument("SSOR 错误：松弛因子必须满足 0 < omega < 2");
        }

        auto data = make_redblack_data(A);
        const double fw = omega * (2.0 - omega);   // 前代右端缩放 ω(2-ω)

        return [data, omega, fw](const std::vector<double>& rhs, std::vector<double>& z)
        {
            const uint32_t n = data->diag2.size();
            const uint32_t ncolor = data->ncolor;
            const auto& r2 = data->A2.row_ptr();
            const auto& c2 = data->A2.col_idx();
            const auto& v2 = data->A2.values();
            const auto& d2 = data->diag2;
            auto& f  = data->f;
            auto& w  = data->w;
            auto& z2 = data->z2;

            // 0. 重排右端：f = P r
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                f[i] = rhs[data->perm[i]];
            }

            // 1. 前代 (D + ωL) w = ω(2-ω) f ：按色正序，色内并行
            //    重排后 j < i 等价于 color[j] < color[i]（同色无耦合，已被独立集保证）
            for (uint32_t c = 0; c < ncolor; ++c)
            {
                const std::ptrdiff_t lo = static_cast<std::ptrdiff_t>(data->color_ptr[c]);
                const std::ptrdiff_t hi = static_cast<std::ptrdiff_t>(data->color_ptr[c + 1]);
                if (lo >= hi) continue;

                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t ii = lo; ii < hi; ++ii)
                {
                    double sigma = fw * f[ii];
                    for (uint32_t p = r2[ii]; p < r2[ii + 1]; ++p)
                    {
                        const uint32_t j = c2[p];
                        if (j < static_cast<uint32_t>(ii))
                        {
                            sigma -= omega * v2[p] * w[j];   // 下三角：色号更小，已算完
                        }
                    }
                    w[ii] = sigma / d2[ii];
                }
            }

            // 2. 回代 (D + ωU) z2 = D w ：按色逆序，色内并行
            for (std::ptrdiff_t cc = static_cast<std::ptrdiff_t>(ncolor) - 1; cc >= 0; --cc)
            {
                const std::ptrdiff_t lo = static_cast<std::ptrdiff_t>(data->color_ptr[cc]);
                const std::ptrdiff_t hi = static_cast<std::ptrdiff_t>(data->color_ptr[cc + 1]);
                if (lo >= hi) continue;

                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t ii = lo; ii < hi; ++ii)
                {
                    double sigma = 0.0;
                    for (uint32_t p = r2[ii]; p < r2[ii + 1]; ++p)
                    {
                        const uint32_t j = c2[p];
                        if (j > static_cast<uint32_t>(ii))
                        {
                            sigma += v2[p] * z2[j];          // 上三角：色号更大，已算完
                        }
                    }
                    z2[ii] = w[ii] - (omega / d2[ii]) * sigma;
                }
            }

            // 3. 反重排：z = P^T z2
            z.resize(n);
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                z[i] = z2[data->new_id[i]];
            }
        };
    }
    struct AMG_Level
    {
        SMatrix A;                  // 自动调用默认构造函数，安全初始化为空
        SMatrix P;                  // fine x coarse
        SMatrix R;                  // coarse x fine
        std::vector<double> diag;
        
        // V-cycle 工作缓冲
        mutable std::vector<double> tmp_new, rr, rc, uc, pc;
    };
    struct AMG_Data
    {
        std::vector<AMG_Level> lv;
        std::vector<double> lu_mat; std::vector<uint32_t> lu_piv;  // 最粗层稠密 LU
        double omega = 2.0 / 3.0;
        int nu1 = 1, nu2 = 1;
    };
    static std::shared_ptr<AMG_Data> make_amg_data(const SMatrix& A, double theta = 0.25, uint32_t coarse_thresh = 300, int max_levels = 25)
    {
        auto data = std::make_shared<AMG_Data>();
        data->lv.emplace_back();
        data->lv[0].A = A;

        for (int k = 0; ; ++k)
        {
            const SMatrix& Ak = data->lv[k].A;
            const uint32_t n = Ak.rows();
            const auto& Ap = Ak.row_ptr(); const auto& Ac = Ak.col_idx(); const auto& Av = Ak.values();

            // 对角元
            data->lv[k].diag.assign(n, 0.0);
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                    if (Ac[p] == i) { data->lv[k].diag[i] = Av[p]; break; }

            if (n <= coarse_thresh || k + 1 >= max_levels) break;

            // ---- (1) 强连接 S（行并行计数+填充）----
            std::vector<uint32_t> Sp(n + 1, 0);
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                double mx = 0.0;
                for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                    if (Ac[p] != (uint32_t)i) mx = std::max(mx, std::fabs(Av[p]));
                const double thr = theta * mx;
                uint32_t c = 0;
                for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                    if (Ac[p] != (uint32_t)i && std::fabs(Av[p]) >= thr) c++;
                Sp[i + 1] = c;
            }
            for (uint32_t i = 0; i < n; ++i) Sp[i + 1] += Sp[i];
            std::vector<uint32_t> Sc(Sp[n]);
            {
                std::vector<uint32_t> off = Sp;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                {
                    double mx = 0.0;
                    for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                        if (Ac[p] != (uint32_t)i) mx = std::max(mx, std::fabs(Av[p]));
                    const double thr = theta * mx;
                    for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                        if (Ac[p] != (uint32_t)i && std::fabs(Av[p]) >= thr)
                            Sc[off[i]++] = Ac[p];
                }
            }

            // ---- (2) C/F 贪心粗化（串行：贪心依赖）----
            std::vector<char> state(n, 0);              // 0=U 1=C 2=F
            std::vector<uint32_t> weight(n);
            for (uint32_t i = 0; i < n; ++i) weight[i] = Sp[i + 1] - Sp[i];
            std::priority_queue<std::pair<uint32_t, uint32_t>> pq;
            for (uint32_t i = 0; i < n; ++i) pq.push({weight[i], i});
            while (!pq.empty())
            {
                const uint32_t i = pq.top().second; pq.pop();
                if (state[i] != 0) continue;
                state[i] = 1;                            // i -> C
                for (uint32_t p = Sp[i]; p < Sp[i + 1]; ++p)
                {
                    const uint32_t j = Sc[p];
                    if (state[j] != 0) continue;
                    state[j] = 2;                        // j -> F
                    for (uint32_t q = Sp[j]; q < Sp[j + 1]; ++q)
                    {
                        const uint32_t kk = Sc[q];
                        if (state[kk] == 0) { weight[kk]++; pq.push({weight[kk], kk}); }
                    }
                }
            }
            for (uint32_t i = 0; i < n; ++i) if (state[i] == 0) state[i] = 1;

            // 第二遍修正：F 点若无强连接 C 邻居 -> 升为 C
            for (uint32_t i = 0; i < n; ++i)
            {
                if (state[i] != 2) continue;
                bool hasC = false;
                for (uint32_t p = Sp[i]; p < Sp[i + 1]; ++p)
                    if (state[Sc[p]] == 1) { hasC = true; break; }
                if (!hasC) state[i] = 1;
            }

            std::vector<uint32_t> mapC(n, 0); uint32_t nc = 0;
            for (uint32_t i = 0; i < n; ++i) { mapC[i] = nc; if (state[i] == 1) nc++; }
            if (nc == 0 || nc == n) break;               // 粗化失败/无压缩 -> 停止建层

            // ---- (3) 直接插值 P（行并行）----
            std::vector<uint32_t> Pp(n + 1, 0);
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                Pp[i + 1] = (state[i] == 1) ? 1 : (Sp[i + 1] - Sp[i]);
            for (uint32_t i = 0; i < n; ++i) Pp[i + 1] += Pp[i];
            std::vector<uint32_t> Pc(Pp[n]); std::vector<double> Pv(Pp[n]);
            {
                std::vector<uint32_t> off = Pp;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                {
                    if (state[i] == 1) { Pc[off[i]++] = mapC[i]; Pv[off[i] - 1] = 1.0; }
                    else
                    {
                        double denom = data->lv[k].diag[i];
                        for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)      // 弱连接 lump 到对角
                        {
                            const uint32_t j = Ac[p];
                            if (j == (uint32_t)i) continue;
                            bool strong = false;
                            for (uint32_t s = Sp[i]; s < Sp[i + 1]; ++s)
                                if (Sc[s] == j) { strong = true; break; }
                            if (!strong) denom += Av[p];
                        }
                        if (std::fabs(denom) < 1e-300) denom = data->lv[k].diag[i];
                        for (uint32_t s = Sp[i]; s < Sp[i + 1]; ++s)
                        {
                            const uint32_t j = Sc[s];
                            if (state[j] != 1) continue;
                            double aij = 0.0;
                            for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                                if (Ac[p] == j) { aij = Av[p]; break; }
                            Pc[off[i]] = mapC[j]; Pv[off[i]++] = -aij / denom;
                        }
                    }
                }
            }
            // P 由 COO 构造（行已有序）
            SMatrix P(n, nc, Pp, Pc, Pv);

            // ---- (4) Galerkin：A_{k+1} = R A P ----
            // SMatrix R  = P.transpose();
            SMatrix AP = Ak * P;
            SMatrix A_next = P.transpose() * AP;             // 修复：改名为 A_next 避免冲突

            data->lv[k].P = P;
            data->lv[k].R = P.transpose();
            data->lv.emplace_back();
            data->lv[k + 1].A = A_next;          // 修复：正确赋值 SMatrix
        }

        // 最粗层稠密 LU 分解
        dense_lu_factor(data->lv.back().A, data->lu_mat, data->lu_piv);
        // 缓冲分配
        for (auto& L : data->lv)
        {
            const uint32_t nf = L.A.rows();
            const uint32_t ncc = L.P.cols() ? L.P.cols() : nf;
            L.tmp_new.resize(nf); L.rr.resize(nf); L.pc.resize(nf);
            L.rc.resize(ncc); L.uc.resize(ncc);
        }
        return data;
    }
    static void amg_vcycle(const AMG_Data& D, uint32_t k, const std::vector<double>& f, std::vector<double>& u)
    {
        const AMG_Level& L = D.lv[k];
        const uint32_t n = L.A.rows();

        if (k + 1 == D.lv.size())                       // 最粗层：直接解
        {
            dense_lu_solve(D.lu_mat, D.lu_piv, f, u);
            return;
        }

        auto jacobi = [&](int nu)
        {
            for (int t = 0; t < nu; ++t)
            {
                std::vector<double> Ax = SMatrix::spmv(L.A, u);
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                    L.tmp_new[i] = u[i] + D.omega * (f[i] - Ax[i]) / L.diag[i];
                u = L.tmp_new;                          // 双缓冲交换
            }
        };

        jacobi(D.nu1);                                  // 预光滑

        std::vector<double> Ax = SMatrix::spmv(L.A, u); // 残差
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            L.rr[i] = f[i] - Ax[i];

        L.rc = SMatrix::spmv(L.R, L.rr);                // 限制
        L.uc.assign(L.uc.size(), 0.0);
        amg_vcycle(D, k + 1, L.rc, L.uc);               // 递归粗校正
        L.pc = SMatrix::spmv(L.P, L.uc);                // 插值回细层
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            u[i] += L.pc[i];

        jacobi(D.nu2);                                  // 后光滑
    }
    std::function<void(const std::vector<double>&, std::vector<double>&)> SMatrix::make_amg_preconditioner(const SMatrix& A)
    {
        auto data = make_amg_data(A);
        return [data](const std::vector<double>& rhs, std::vector<double>& z)
        {
            z.assign(rhs.size(), 0.0);
            amg_vcycle(*data, 0, rhs, z);               // 一次 V-cycle = 一次预处理 apply
        };
    }


    double SMatrix::dot(const std::vector<double>& a, const std::vector<double>& b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("点乘的向量长度不一致");
        }
        double sum = 0.0;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:sum) schedule(static) if(a.size() > 4000)
        #endif
        for (uint32_t i = 0; i < a.size(); i++)
        {
            sum += a[i]*b[i];
        }
        return sum;
    }

    std::vector<double> SMatrix::spmv(const SMatrix& A, const std::vector<double>& x)
    {
        if (A.cols_ != x.size())
        {
            throw std::invalid_argument("spmv 错误：矩阵列数与向量长度不一致");
        }

        std::vector<double> y(A.rows_, 0.0);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(A.rows_ > 1000)
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(A.rows_); ++i)
        {
            double sum = 0.0;

            for (uint32_t k = A.row_ptr_[i]; k < A.row_ptr_[i + 1]; ++k)
            {
                sum += A.values_[k] * x[A.col_idx_[k]];
            }

            y[i] = sum;
        }

        return y;
    }

    // 出参版 spmv: 复用 y 的缓冲区, 不再每次调用都 malloc/free + 填 0。
    // Krylov 求解里每一步都要 spmv, 几千次迭代就是几千次分配;
    // 预处理器内部 (Schur 补) 每次 apply 还要再来两次。
    void SMatrix::spmv_into(const SMatrix& A, const std::vector<double>& x,
                            std::vector<double>& y)
    {
        if (A.cols_ != x.size())
        {
            throw std::invalid_argument("spmv 错误：矩阵列数与向量长度不一致");
        }
        if (y.size() != A.rows_) { y.assign(A.rows_, 0.0); }

        double* __restrict yp = y.data();
        // 不给 xp 加 __restrict: 调用方可能让 x 与 y 指向同一块内存,
        // 同时对两者声明 restrict 属于未定义行为。
        const double* xp = x.data();
        const uint32_t* __restrict rp = A.row_ptr_.data();
        const uint32_t* __restrict ci = A.col_idx_.data();
        const double* __restrict av = A.values_.data();

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(A.rows_ > 1000)
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(A.rows_); ++i)
        {
            double sum = 0.0;
            const uint32_t end = rp[i + 1];
            // 【已移除 #pragma omp simd】内层循环长度就是矩阵的每行非零元数
            //   (本算例 7~9), 对这么短的循环做 SIMD 没有任何收益; 而
            //   "omp simd + reduction" 嵌在 "omp parallel for" 里是各版本 GCC 最
            //   容易出问题的组合之一。真正的并行度由外层 parallel for 提供。
            for (uint32_t k = rp[i]; k < end; ++k)
            {
                sum += av[k] * xp[ci[k]];
            }
            yp[i] = sum;
        }
    }

    // 稀疏矩阵线性系统求解器
    std::tuple<std::vector<double>, int> SMatrix::pcg(const SMatrix& A, const std::vector<double>& b, const std::string& precond,double tol,int max_iter)
    {
        if (A.rows_ != A.cols_)
        {
            throw std::invalid_argument("PCG 要求矩阵是方阵");
        }

        if (A.cols_ != b.size())
        {
            throw std::invalid_argument("PCG 错误：矩阵列数与向量长度不一致");
        }

        const uint32_t n = b.size();

        std::vector<double> x(n, 0.0);

        // ||b||
        const double normb = std::sqrt(SMatrix::dot(b, b));

        // 如果 b = 0，则 x = 0 就是解
        if (normb < 1e-300)
        {
            return {x, 0};
        }

        // 初始 x = 0，所以 r = b - A x = b
        std::vector<double> r = b;

        double normr = std::sqrt(SMatrix::dot(r, r));

        if (normr / normb < tol)
        {
            return {x, 0};
        }

        const std::string name = to_lower(precond);

        std::function<void(const std::vector<double>&, std::vector<double>&)> solveM;

        if (name == "none" || name == "identity" || name == "noprecond" || name == "cg")
        {
            solveM = [](const std::vector<double>& rhs,
                        std::vector<double>& z)
            {
                z = rhs;
            };
        }
        else if (name == "diag" || name == "diagonal" || name == "jacobi")
        {
            solveM = SMatrix::make_diag_preconditioner(A);
        }
        else if (name == "ic" || name == "ic0" || name == "ichol" || name == "incomplete_cholesky")
        {
            solveM = SMatrix::make_ic0_preconditioner(A);
        }
        else if (name == "ilu" || name == "ilu0")
        {
            solveM = SMatrix::make_ilu0_preconditioner(A);
        }
        else if (name == "sor" || name == "ssor")
        {
            solveM = SMatrix::make_ssor_preconditioner(A);
        }
        else if (name == "amg")
        {
            solveM = SMatrix::make_amg_preconditioner(A);
        }
        else
        {
            throw std::invalid_argument("未知预处理类型: " + precond);
        }

        std::vector<double> z(n, 0.0);
        std::vector<double> p(n, 0.0);

        // z0 = M^{-1} r0
        solveM(r, z);

        // p0 = z0
        p = z;

        // rho0 = r0^T z0
        double rz_old = SMatrix::dot(r, z);

        if (rz_old <= 0.0)
        {
            throw std::runtime_error("PCG 失败：r^T M^{-1} r <= 0，预处理器可能不是 SPD");
        }

        const double breakdown_tol = 1e-30;

        int it_end = 0;
        for (int it = 0; it < max_iter; ++it)
        {
            // Ap = A p
            std::vector<double> Ap = SMatrix::spmv(A, p);

            const double pAp = SMatrix::dot(p, Ap);

            if (std::fabs(pAp) < breakdown_tol)
            {
                throw std::runtime_error("PCG breakdown：p^T A p 过小");
            }

            if (pAp <= 0.0)
            {
                throw std::runtime_error("PCG 失败：p^T A p <= 0，矩阵 A 可能不是 SPD");
            }

            const double alpha = rz_old / pAp;

            // x = x + alpha p
            // r = r - alpha Ap
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            normr = std::sqrt(SMatrix::dot(r, r));

            if (normr / normb < tol)
            {
                break;
            }

            // z = M^{-1} r
            solveM(r, z);

            const double rz_new = SMatrix::dot(r, z);

            if (rz_new <= 0.0)
            {
                throw std::runtime_error("PCG 失败：r^T M^{-1} r <= 0，预处理器可能不是 SPD");
            }

            const double beta = rz_new / rz_old;

            // p = z + beta p
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                p[i] = z[i] + beta * p[i];
            }

            rz_old = rz_new;

            it_end = it;
        }

        return {x, it_end};
    }
    std::tuple<std::vector<double>, int> SMatrix::bicgstab(const SMatrix& A, const std::vector<double>& b, const std::string& precond, double tol, int max_iter)
    {
        if (A.rows_ != A.cols_)
        {
            throw std::invalid_argument("BiCGSTAB 要求矩阵是方阵");
        }

        if (A.cols_ != b.size())
        {
            throw std::invalid_argument("BiCGSTAB 错误：矩阵列数与向量长度不一致");
        }

        const uint32_t n = b.size();

        std::vector<double> x(n, 0.0);

        // ||b||
        const double normb = std::sqrt(SMatrix::dot(b, b));

        // 如果 b = 0，则 x = 0 就是解
        if (normb < 1e-300)
        {
            return {x, 0};
        }

        // 初始 x = 0，所以 r0 = b - A x0 = b
        std::vector<double> r = b;

        double normr = std::sqrt(SMatrix::dot(r, r));

        if (normr / normb < tol)
        {
            return {x, 0};
        }

        const std::string name = to_lower(precond);

        std::function<void(const std::vector<double>&, std::vector<double>&)> solveM;

        if (name == "none" || name == "identity" || name == "noprecond")
        {
            solveM = [](const std::vector<double>& rhs,
                        std::vector<double>& z)
            {
                z = rhs;
            };
        }
        else if (name == "diag" || name == "diagonal" || name == "jacobi")
        {
            solveM = SMatrix::make_diag_preconditioner(A);
        }
        else if (name == "ilu" || name == "ilu0")
        {
            solveM = SMatrix::make_ilu0_preconditioner(A);
        }
        else if (name == "ic" || name == "ic0" || name == "ichol" || name == "incomplete_cholesky")
        {
            // 允许但不推荐：IC0 一般只适合 SPD 矩阵
            solveM = SMatrix::make_ic0_preconditioner(A);
        }
        else if (name == "sor" || name == "ssor")
        {
            solveM = SMatrix::make_ssor_preconditioner(A);
        }
        else if (name == "amg")
        {
            solveM = SMatrix::make_amg_preconditioner(A);
        }
        else
        {
            throw std::invalid_argument("未知预处理类型: " + precond);
        }

        // 步骤 3：影子残差 r̃0，取 r̃0 = r0，保证 r̃0^T r0 = ||r0||^2 != 0
        const std::vector<double> rtilde = r;

        // 工作向量
        std::vector<double> p(n, 0.0);   // p0 = 0
        std::vector<double> v(n, 0.0);   // v0 = 0
        std::vector<double> y(n, 0.0);
        std::vector<double> z(n, 0.0);
        std::vector<double> s(n, 0.0);
        std::vector<double> t(n, 0.0);

        // 步骤 4：rho0 = alpha = omega0 = 1
        double rho_old = 1.0;
        double alpha   = 1.0;
        double omega   = 1.0;

        const double breakdown_tol = 1e-30;

        int it_end = 0;
        for (int it = 0; it < max_iter; ++it)
        {
            // (1) rho_{k+1} = r̃0^T r_k
            const double rho_new = SMatrix::dot(rtilde, r);

            if (std::fabs(rho_new) < breakdown_tol)
            {
                throw std::runtime_error("BiCGSTAB breakdown：rho_{k+1} = r̃0^T r_k 过小");
            }
            if (std::fabs(rho_old) < breakdown_tol)
            {
                throw std::runtime_error("BiCGSTAB breakdown：rho_k 过小");
            }
            if (std::fabs(omega) < breakdown_tol)
            {
                throw std::runtime_error("BiCGSTAB breakdown：omega_k = 0");
            }

            // (2) beta = (rho_{k+1}/rho_k) * (alpha/omega_k)
            //     此处 alpha、omega 为上一轮的值
            const double beta = (rho_new / rho_old) * (alpha / omega);

            // (3) p_{k+1} = r_k + beta (p_k - omega_k v_k)
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }

            // (4) 解 M y = p_{k+1}
            solveM(p, y);

            // (5) v_{k+1} = A y；alpha = rho_{k+1} / (r̃0^T v_{k+1})；s = r_k - alpha v_{k+1}
            v = SMatrix::spmv(A, y);

            const double rhat_v = SMatrix::dot(rtilde, v);

            if (std::fabs(rhat_v) < breakdown_tol)
            {
                throw std::runtime_error("BiCGSTAB breakdown：r̃0^T v_{k+1} 过小");
            }

            alpha = rho_new / rhat_v;

            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                s[i] = r[i] - alpha * v[i];
            }

            // 【已删除】基于 s 的提前退出逻辑，防止绕过真实残差校验

            // (6) 解 M z = s
            solveM(s, z);

            // (7) t = A z；omega_{k+1} = (t^T s)/(t^T t)
            t = SMatrix::spmv(A, z);

            const double tt = SMatrix::dot(t, t);

            if (tt < breakdown_tol)
            {
                throw std::runtime_error("BiCGSTAB breakdown：t^T t 过小");
            }

            omega = SMatrix::dot(t, s) / tt;

            // (8) x_{k+1} = x_k + alpha y + omega_{k+1} z
            // (9) r_{k+1} = s - omega_{k+1} t
            //     两个更新合并进同一个并行循环，减少同步开销
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                x[i] += alpha * y[i] + omega * z[i];
                r[i] = s[i] - omega * t[i];
            }

            normr = std::sqrt(SMatrix::dot(r, r));
            it_end = it;

            // =========================================================
            // 核心修改：收敛检查与虚假收敛处理 (Residual Replacement)
            // =========================================================
            bool converged = false;
            if (normr / normb < tol)
            {
                // 1. 发现递推残差达标，强制计算真实残差进行校验
                std::vector<double> Ax = SMatrix::spmv(A, x);
                double true_normr_sq = 0.0;
                
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for reduction(+:true_normr_sq) schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                    double diff = b[i] - Ax[i];
                    true_normr_sq += diff * diff;
                }
                double true_normr = std::sqrt(true_normr_sq);
                
                if (true_normr / normb < tol)
                {
                    // 真实残差也达标，真正收敛，准备退出
                    converged = true; 
                }
                else
                {
                    // 虚假收敛！递推残差 r 已经不可靠。
                    // 策略：用真实残差替换递推残差 r，并重新计算 rho，让算法继续迭代。
                    
                    #ifdef CPPNUM_HAVE_OPENMP
                        #pragma omp parallel for schedule(static) if(n > 1000)
                    #endif
                    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                        r[i] = b[i] - Ax[i];
                    }
                    normr = true_normr; // 更新显示的残差范数
                    
                    // 关键：因为 r 变了，必须重新计算下一轮需要的 rho_new (这里作为下一轮的 rho_old)
                    double rho_corrected = SMatrix::dot(rtilde, r);
                    
                    if (std::fabs(rho_corrected) < breakdown_tol) {
                        // 替换后内积为0，发生 breakdown，无法继续，强制退出
                        converged = true; 
                    } else {
                        // 传递修正后的 rho 到下一轮作为 rho_old
                        rho_old = rho_corrected;
                        continue; // 跳过本轮末尾的 rho_old = rho_new; 直接进入下一轮
                    }
                }
            }
            else
            {
                // 未收敛，正常传递 rho
                rho_old = rho_new;
            }

            if (converged) {
                break;
            }
        }

        return {x, it_end};
    }
    std::tuple<std::vector<double>, int> SMatrix::sor(const SMatrix& A, const std::vector<double>& b, double omega, double tol, int max_iter)
    {
        if (A.rows_ != A.cols_)
        {
            throw std::invalid_argument("SOR 要求矩阵是方阵");
        }

        if (A.cols_ != b.size())
        {
            throw std::invalid_argument("SOR 错误：矩阵列数与向量长度不一致");
        }

        const uint32_t n = b.size();

        std::vector<double> x(n, 0.0);

        // ||b||
        const double normb = std::sqrt(SMatrix::dot(b, b));

        if (normb < 1e-300)
        {
            return {x, 0};
        }

        // Kahan 定理：SOR 收敛的必要条件是 0 < omega < 2
        if (!(omega > 0.0 && omega < 2.0))
        {
            throw std::invalid_argument("SOR 错误：松弛因子必须满足 0 < omega < 2");
        }

        const auto& A_row_ptr = A.row_ptr();
        const auto& A_col_idx = A.col_idx();
        const auto& A_values = A.values();

        // 预提取对角元（一次性 O(nnz)，避免每步在行内重复查找）
        std::vector<double> diag(n, 0.0);
        for (uint32_t i = 0; i < n; ++i)
        {
            bool found = false;
            for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
            {
                if (A_col_idx[p] == i)
                {
                    diag[i] = A_values[p];
                    found = true;
                    break;
                }
            }
            if (!found || std::fabs(diag[i]) < 1e-300)
            {
                throw std::runtime_error("SOR 失败：第 " + std::to_string(i) + " 行缺少非零对角元");
            }
        }

        int it_end = 0;
        for (int it = 0; it < max_iter; ++it)
        {
            // ---------------------------------------------------------
            // SOR 扫掠：串行！第 i 行依赖 j < i 的本轮新值，不可 parallel for
            // in-place 更新：数组中 j < i 的位置天然存着 x^{k+1}
            // ---------------------------------------------------------
            for (uint32_t i = 0; i < n; ++i)
            {
                double sigma = b[i];
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
                {
                    const uint32_t j = A_col_idx[p];
                    if (j == i) continue;          // 对角元单独处理
                    sigma -= A_values[p] * x[j];   // j<i 为新值，j>i 为旧值
                }
                const double x_gs = sigma / diag[i];       // Gauss-Seidel 更新值
                x[i] += omega * (x_gs - x[i]);             // 松弛：x + ω·Δ
            }

            double err_sq = 0.0;
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for reduction(+:err_sq) schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                double ri = b[i];
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
                {
                    ri -= A_values[p] * x[A_col_idx[p]];
                }
                err_sq += ri * ri;
            }

            const double normr = std::sqrt(err_sq);

            if (!std::isfinite(normr))
            {
                throw std::runtime_error("SOR 发散：残差出现 NaN/Inf，请检查 omega 与矩阵性质");
            }

            it_end = it;

            if (normr / normb < tol)
            {
                break;
            }
        }

        return {x, it_end};
    }
    std::tuple<std::vector<double>, int> SMatrix::sor_rb(const SMatrix& A, const std::vector<double>& b, double omega, double tol, int max_iter)
    {
        if (A.rows_ != A.cols_)
        {
            throw std::invalid_argument("红黑 SOR 要求矩阵是方阵");
        }

        if (A.cols_ != b.size())
        {
            throw std::invalid_argument("红黑 SOR 错误：矩阵列数与向量长度不一致");
        }

        const uint32_t n = b.size();
        const uint32_t nnz = A.nnz();

        std::vector<double> x_out(n, 0.0);

        const double normb = std::sqrt(SMatrix::dot(b, b));
        if (normb < 1e-300)
        {
            return {x_out, 0};
        }

        // Kahan 定理：0 < omega < 2
        if (!(omega > 0.0 && omega < 2.0))
        {
            throw std::invalid_argument("红黑 SOR 错误：松弛因子必须满足 0 < omega < 2");
        }

        const auto& A_row_ptr = A.row_ptr();
        const auto& A_col_idx = A.col_idx();
        const auto& A_values  = A.values();

        // ---------------------------------------------------------
        // 1. 构建转置模式 (CSC)，用于获取"对称邻居"（保证着色正确性，
        //    即使矩阵数值非对称，图结构也取 A + A^T 的模式）
        // ---------------------------------------------------------
        std::vector<uint32_t> AT_col_ptr(n + 1, 0);
        for (uint32_t p = 0; p < nnz; ++p)
        {
            AT_col_ptr[A_col_idx[p] + 1]++;
        }
        for (uint32_t i = 0; i < n; ++i)
        {
            AT_col_ptr[i + 1] += AT_col_ptr[i];
        }
        std::vector<uint32_t> AT_row_idx(nnz);
        {
            std::vector<uint32_t> off = AT_col_ptr;
            for (uint32_t i = 0; i < n; ++i)
            {
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
                {
                    AT_row_idx[off[A_col_idx[p]]++] = i;
                }
            }
        }

        // ---------------------------------------------------------
        // 2. 着色：先 BFS 尝试 2-色化（二部图 => 标准红黑）；
        //    若发现奇环（非二部图），回退贪心多色化
        // ---------------------------------------------------------
        std::vector<int> color(n, -1);
        uint32_t ncolor = 0;
        bool bipartite = true;
        std::vector<uint32_t> queue;

        for (uint32_t seed = 0; seed < n && bipartite; ++seed)
        {
            if (color[seed] != -1) continue;
            color[seed] = 0;
            queue.clear();
            queue.push_back(seed);

            for (uint32_t head = 0; head < queue.size() && bipartite; ++head)
            {
                const uint32_t i = queue[head];
                auto visit = [&](uint32_t j)
                {
                    if (j == i) return;
                    if (color[j] == -1)
                    {
                        color[j] = 1 - color[i];
                        queue.push_back(j);
                    }
                    else if (color[j] == color[i])
                    {
                        bipartite = false;   // 奇环：非二部图
                    }
                };
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) visit(A_col_idx[p]);
                for (uint32_t p = AT_col_ptr[i]; p < AT_col_ptr[i + 1]; ++p) visit(AT_row_idx[p]);
            }
        }

        if (bipartite)
        {
            ncolor = 2;   // 标准红黑
        }
        else
        {
            // 贪心多色化（通用推广）
            std::fill(color.begin(), color.end(), -1);
            std::vector<char> used(n + 1, 0);
            std::vector<uint32_t> touched;
            for (uint32_t i = 0; i < n; ++i)
            {
                touched.clear();
                auto mark = [&](uint32_t j)
                {
                    if (j == i) return;
                    const int c = color[j];
                    if (c >= 0 && !used[c]) { used[c] = 1; touched.push_back(c); }
                };
                for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p) mark(A_col_idx[p]);
                for (uint32_t p = AT_col_ptr[i]; p < AT_col_ptr[i + 1]; ++p) mark(AT_row_idx[p]);

                int c = 0;
                while (used[c]) ++c;
                color[i] = c;
                if (static_cast<uint32_t>(c) + 1 > ncolor) ncolor = c + 1;
                for (uint32_t c2 : touched) used[c2] = 0;
            }
        }

        // ---------------------------------------------------------
        // 3. 按色重排：同色编号连续 => 并行区间连续、无分支、缓存友好
        //    重排后 A' = P A P^T 呈块结构，对角块为对角阵
        // ---------------------------------------------------------
        std::vector<uint32_t> color_ptr(ncolor + 1, 0);
        for (uint32_t i = 0; i < n; ++i) color_ptr[color[i] + 1]++;
        for (uint32_t c = 0; c < ncolor; ++c) color_ptr[c + 1] += color_ptr[c];

        std::vector<uint32_t> new_id(n);
        {
            std::vector<uint32_t> off = color_ptr;
            for (uint32_t i = 0; i < n; ++i) new_id[i] = off[color[i]]++;
        }
        std::vector<uint32_t> perm(n);           // perm[new_id] = old_id
        for (uint32_t i = 0; i < n; ++i) perm[new_id[i]] = i;

        std::vector<double> values2; values2.reserve(nnz);
        std::vector<uint32_t> row2; row2.reserve(nnz);
        std::vector<uint32_t> col2; col2.reserve(nnz);
        for (uint32_t ni = 0; ni < n; ++ni)
        {
            const uint32_t i = perm[ni];
            for (uint32_t p = A_row_ptr[i]; p < A_row_ptr[i + 1]; ++p)
            {
                row2.push_back(ni);
                col2.push_back(new_id[A_col_idx[p]]);
                values2.push_back(A_values[p]);
            }
        }
        SMatrix A2(n, n, values2, row2, col2);

        std::vector<double> b2(n);
        for (uint32_t i = 0; i < n; ++i) b2[new_id[i]] = b[i];

        const auto& r2 = A2.row_ptr();
        const auto& c2 = A2.col_idx();
        const auto& v2 = A2.values();

        // 预提取重排后的对角元
        std::vector<double> diag2(n, 0.0);
        for (uint32_t ni = 0; ni < n; ++ni)
        {
            bool found = false;
            for (uint32_t p = r2[ni]; p < r2[ni + 1]; ++p)
            {
                if (c2[p] == ni) { diag2[ni] = v2[p]; found = true; break; }
            }
            if (!found || std::fabs(diag2[ni]) < 1e-300)
            {
                throw std::runtime_error("红黑 SOR 失败：缺少非零对角元");
            }
        }

        // ---------------------------------------------------------
        // 4. 主迭代：逐色并行扫掠
        //    正确性核心：同色内无边（独立集），x2[j] (j 异色) 在本半步内不被写
        // ---------------------------------------------------------
        std::vector<double> x2(n, 0.0);
        int it_end = 0;

        for (int it = 0; it < max_iter; ++it)
        {
            for (uint32_t c = 0; c < ncolor; ++c)
            {
                const std::ptrdiff_t lo = static_cast<std::ptrdiff_t>(color_ptr[c]);
                const std::ptrdiff_t hi = static_cast<std::ptrdiff_t>(color_ptr[c + 1]);
                if (lo >= hi) continue;

                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t ii = lo; ii < hi; ++ii)
                {
                    double sigma = b2[ii];
                    for (uint32_t p = r2[ii]; p < r2[ii + 1]; ++p)
                    {
                        const uint32_t j = c2[p];
                        if (j == static_cast<uint32_t>(ii)) continue;
                        sigma -= v2[p] * x2[j];      // 邻居必异色，本半步只读不写
                    }
                    x2[ii] += omega * (sigma / diag2[ii] - x2[ii]);
                }
            }

            // 真实残差：并行归约（重排空间）
            double err_sq = 0.0;
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for reduction(+:err_sq) schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii)
            {
                double ri = b2[ii];
                for (uint32_t p = r2[ii]; p < r2[ii + 1]; ++p)
                {
                    ri -= v2[p] * x2[c2[p]];
                }
                err_sq += ri * ri;
            }
            const double normr = std::sqrt(err_sq);

            if (!std::isfinite(normr))
            {
                throw std::runtime_error("红黑 SOR 发散：残差出现 NaN/Inf");
            }

            it_end = it;
            if (normr / normb < tol)
            {
                break;
            }
        }

        // 反重排回原编号
        for (uint32_t i = 0; i < n; ++i) x_out[i] = x2[new_id[i]];

        return {x_out, it_end};
    }
    std::tuple<std::vector<double>, int> SMatrix::gmres(const SMatrix& A, const std::vector<double>& b, const std::string& precond, int restart, double tol, int max_iter)
    {
        if (A.rows_ != A.cols_)
        {
            throw std::invalid_argument("GMRES 要求矩阵是方阵");
        }
        if (A.cols_ != b.size())
        {
            throw std::invalid_argument("GMRES 错误：矩阵列数与向量长度不一致");
        }

        const uint32_t n = b.size();
        const int m = restart;
        if (m <= 0)
        {
            throw std::invalid_argument("GMRES 错误：restart 必须为正整数");
        }
        if (m <= 0)
        {
            throw std::invalid_argument("GMRES 错误：restart 必须为正整数");
        }

        std::vector<double> x(n, 0.0);

        const double normb = std::sqrt(SMatrix::dot(b, b));
        if (normb < 1e-300)
        {
            return {x, 0};
        }

        // ---- 字符串预处理分发（与 bicgstab 完全相同）----
        const std::string name = to_lower(precond);
        std::function<void(const std::vector<double>&, std::vector<double>&)> solveM;
        if (name == "none" || name == "identity" || name == "noprecond")
        {
            solveM = [](const std::vector<double>& rhs, std::vector<double>& z) { z = rhs; };
        }
        else if (name == "diag" || name == "diagonal" || name == "jacobi")
        {
            solveM = SMatrix::make_diag_preconditioner(A);
        }
        else if (name == "ilu" || name == "ilu0")
        {
            solveM = SMatrix::make_ilu0_preconditioner(A);
        }
        else if (name == "ic" || name == "ic0" || name == "ichol")
        {
            solveM = SMatrix::make_ic0_preconditioner(A);
        }
        else if (name == "ssor" || name == "sor")
        {
            solveM = SMatrix::make_ssor_preconditioner(A, 1.2);
        }
        else if (name == "amg")
        {
            solveM = SMatrix::make_amg_preconditioner(A);
        }
        else
        {
            throw std::invalid_argument("未知预处理类型: " + precond);
        }

        const double breakdown_tol = 1e-30;

        // ---- Krylov 基 V 与 Hessenberg 矩阵 H ----
        // 旧写法 vector<vector<double>>(m+1, vector<double>(n)) 会做 m+1 次独立
        // 堆分配且彼此不相邻；Arnoldi 正交化时要在 m+1 个互不相干的内存区间
        // 之间来回跳，n 稍大就是几十上百 MB 且 cache 局部性极差。
        // 改成单条连续缓冲区，用 V(i)/H(i) 取行首指针。
        std::vector<double> Vbuf(static_cast<std::size_t>(m + 1) * n, 0.0);
        std::vector<double> Hbuf(static_cast<std::size_t>(m + 1) * m, 0.0);
        auto V = [&](int i) -> double*
        { return Vbuf.data() + static_cast<std::size_t>(i) * n; };
        auto H = [&](int i) -> double*
        { return Hbuf.data() + static_cast<std::size_t>(i) * m; };

        // 裸指针内积：避免为适配 std::vector 接口而反复构造临时向量
        auto dot_raw = [&](const double* p, const double* q) -> double
        {
            double s = 0.0;
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for reduction(+:s) schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
            { s += p[t] * q[t]; }
            return s;
        };

        std::vector<double> cs(m, 0.0), sn(m, 0.0);   // Givens 旋转
        std::vector<double> g(m + 1, 0.0);            // 右端（旋转后）
        std::vector<double> y(m, 0.0);                // 最小二乘解
        std::vector<double> w(n, 0.0), wtmp(n, 0.0), z(n, 0.0), dz(n, 0.0);
        std::vector<double> Ax(n, 0.0);           // 重启时的真实残差工作向量
        std::vector<double> hcol(m + 1, 0.0);              // arnoldi_orth 的工作缓冲
        std::vector<double> vj(n, 0.0);               // V[j] 的 vector 视图，喂给 solveM

        int total = 0;        // 总 Arnoldi 步数
        int it_end = 0;
        bool converged = false;

        while (total < max_iter && !converged)
        {
            double beta;
            double* V0 = V(0);
            if (total == 0)
            {
                beta = normb;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                { V0[i] = b[i] / beta; }
            }
            else
            {
                SMatrix::spmv_into(A, x, Ax);
                double rr = 0.0;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for reduction(+:rr) schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                {
                    const double ri = b[i] - Ax[i];
                    V0[i] = ri;
                    rr += ri * ri;
                }
                beta = std::sqrt(rr);
                if (beta / normb < tol) { converged = true; break; }
                const double inv_beta = 1.0 / beta;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                { V0[i] *= inv_beta; }
            }

            g.assign(m + 1, 0.0);
            g[0] = beta;
            std::fill(Hbuf.begin(), Hbuf.end(), 0.0);

            int j_used = 0;

            for (int j = 0; j < m && total < max_iter; ++j)
            {
                // (3)a 右预处理：w = A M^{-1} v_j
                vj.assign(V(j), V(j) + n);
                solveM(vj, wtmp);
                SMatrix::spmv_into(A, wtmp, w);

                // (3)b Arnoldi 正交化: CGS2 + BLAS-2 dgemv (见 arnoldi_orth 注释)
                //     原来是 MGS: 每个 i 都要一次全长度点积 + 一次全长度 axpy,
                //     访存 ~5n/次, 是 GMRES 在大 n 下真正的瓶颈。
                arnoldi_orth(Vbuf.data(), Hbuf.data(), w.data(), n, m, j, hcol.data());

                // (3)c Hessenberg 次对角元与下一个基向量
                H(j + 1)[j] = std::sqrt(dot_raw(w.data(), w.data()));
                if (H(j + 1)[j] > breakdown_tol)
                {
                    const double inv = 1.0 / H(j + 1)[j];
                    const double* ww = w.data();
                    double* Vn = V(j + 1);
                    #ifdef CPPNUM_HAVE_OPENMP
                        #pragma omp parallel for schedule(static) if(n > 1000)
                    #endif
                    for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
                    { Vn[t] = ww[t] * inv; }
                }

                // (3)d 应用旧 Givens 旋转于第 j 列，再构造新 Givens 消去 H[j+1][j]
                for (int i = 0; i < j; ++i)
                {
                    const double t1 =  cs[i] * H(i)[j] + sn[i] * H(i + 1)[j];
                    const double t2 = -sn[i] * H(i)[j] + cs[i] * H(i + 1)[j];
                    H(i)[j]     = t1;
                    H(i + 1)[j] = t2;
                }
                const double den = std::hypot(H(j)[j], H(j + 1)[j]);
                if (den < breakdown_tol)
                {
                    throw std::runtime_error("GMRES breakdown：Hessenberg 主元过小");
                }
                cs[j] = H(j)[j] / den;
                sn[j] = H(j + 1)[j] / den;
                H(j)[j] = cs[j] * H(j)[j] + sn[j] * H(j + 1)[j];
                H(j + 1)[j] = 0.0;
                const double g1 =  cs[j] * g[j] + sn[j] * g[j + 1];
                const double g2 = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = g1;
                g[j + 1] = g2;

                j_used = j + 1;
                it_end = total;
                ++total;

                // (3)e 残差监测：右预处理下 |g_{j+1}| 即真实残差
                if (std::fabs(g[j + 1]) / normb < tol)
                {
                    converged = true;
                    break;
                }
            }

            // ---- (4) 回代解 R y = g ----
            for (int i = j_used - 1; i >= 0; --i)
            {
                const double* Hi = H(i);
                double sum = g[i];
                for (int k = i + 1; k < j_used; ++k)
                { sum -= Hi[k] * y[k]; }
                if (std::fabs(Hi[i]) < breakdown_tol)
                {
                    throw std::runtime_error("GMRES breakdown：上三角主元过小");
                }
                y[i] = sum / Hi[i];
            }

            // ---- (5) x += M^{-1} (V y) ----
            // 交换内外循环：外层遍历基向量，内层对一段连续内存做 axpy。
            // 旧写法外层 t、内层 i，每次都要跨 n 个元素在 m 个基之间跳。
            std::fill(z.begin(), z.end(), 0.0);
            for (int i = 0; i < j_used; ++i)
            {
                const double yi = y[i];
                const double* Vi = V(i);
                double* zz = z.data();
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
                { zz[t] += yi * Vi[t]; }
            }
            solveM(z, dz);
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
            { x[t] += dz[t]; }
        }

        return {x, it_end};
    }
    std::tuple<std::vector<double>, int> SMatrix::gmres(const SMatrix& A, const std::vector<double>& b, const std::function<void(const std::vector<double>&, std::vector<double>&)>& solveM, int restart, double tol, int max_iter)
    {
        const size_t n = b.size();
        const int m = restart;
        if (m <= 0)
        {
            throw std::invalid_argument("GMRES 错误：restart 必须为正整数");
        }
        if (m <= 0)
        {
            throw std::invalid_argument("GMRES 错误：restart 必须为正整数");
        }

        std::vector<double> x(n, 0.0);

        const double normb = std::sqrt(SMatrix::dot(b, b));
        if (normb < 1e-300)
        {
            return {x, 0};
        }

        const double breakdown_tol = 1e-30;

        // ---- Krylov 基 V 与 Hessenberg 矩阵 H ----
        // 旧写法 vector<vector<double>>(m+1, vector<double>(n)) 会做 m+1 次独立
        // 堆分配且彼此不相邻；Arnoldi 正交化时要在 m+1 个互不相干的内存区间
        // 之间来回跳，n 稍大就是几十上百 MB 且 cache 局部性极差。
        // 改成单条连续缓冲区，用 V(i)/H(i) 取行首指针。
        std::vector<double> Vbuf(static_cast<std::size_t>(m + 1) * n, 0.0);
        std::vector<double> Hbuf(static_cast<std::size_t>(m + 1) * m, 0.0);
        auto V = [&](int i) -> double*
        { return Vbuf.data() + static_cast<std::size_t>(i) * n; };
        auto H = [&](int i) -> double*
        { return Hbuf.data() + static_cast<std::size_t>(i) * m; };

        // 裸指针内积：避免为适配 std::vector 接口而反复构造临时向量
        auto dot_raw = [&](const double* p, const double* q) -> double
        {
            double s = 0.0;
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for reduction(+:s) schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
            { s += p[t] * q[t]; }
            return s;
        };

        std::vector<double> cs(m, 0.0), sn(m, 0.0);   // Givens 旋转
        std::vector<double> g(m + 1, 0.0);            // 右端（旋转后）
        std::vector<double> y(m, 0.0);                // 最小二乘解
        std::vector<double> w(n, 0.0), wtmp(n, 0.0), z(n, 0.0), dz(n, 0.0);
        std::vector<double> Ax(n, 0.0);           // 重启时的真实残差工作向量
        std::vector<double> hcol(m + 1, 0.0);              // arnoldi_orth 的工作缓冲
        std::vector<double> vj(n, 0.0);               // V[j] 的 vector 视图，喂给 solveM

        int total = 0;        // 总 Arnoldi 步数
        int it_end = 0;
        bool converged = false;

        while (total < max_iter && !converged)
        {
            double beta;
            double* V0 = V(0);
            if (total == 0)
            {
                beta = normb;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                { V0[i] = b[i] / beta; }
            }
            else
            {
                SMatrix::spmv_into(A, x, Ax);
                double rr = 0.0;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for reduction(+:rr) schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                {
                    const double ri = b[i] - Ax[i];
                    V0[i] = ri;
                    rr += ri * ri;
                }
                beta = std::sqrt(rr);
                if (beta / normb < tol) { converged = true; break; }
                const double inv_beta = 1.0 / beta;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                { V0[i] *= inv_beta; }
            }

            g.assign(m + 1, 0.0);
            g[0] = beta;
            std::fill(Hbuf.begin(), Hbuf.end(), 0.0);

            int j_used = 0;

            for (int j = 0; j < m && total < max_iter; ++j)
            {
                // (3)a 右预处理：w = A M^{-1} v_j
                vj.assign(V(j), V(j) + n);
                solveM(vj, wtmp);
                SMatrix::spmv_into(A, wtmp, w);

                // (3)b Arnoldi 正交化: CGS2 + BLAS-2 dgemv (见 arnoldi_orth 注释)
                //     原来是 MGS: 每个 i 都要一次全长度点积 + 一次全长度 axpy,
                //     访存 ~5n/次, 是 GMRES 在大 n 下真正的瓶颈。
                arnoldi_orth(Vbuf.data(), Hbuf.data(), w.data(), n, m, j, hcol.data());

                // (3)c Hessenberg 次对角元与下一个基向量
                H(j + 1)[j] = std::sqrt(dot_raw(w.data(), w.data()));
                if (H(j + 1)[j] > breakdown_tol)
                {
                    const double inv = 1.0 / H(j + 1)[j];
                    const double* ww = w.data();
                    double* Vn = V(j + 1);
                    #ifdef CPPNUM_HAVE_OPENMP
                        #pragma omp parallel for schedule(static) if(n > 1000)
                    #endif
                    for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
                    { Vn[t] = ww[t] * inv; }
                }

                // (3)d 应用旧 Givens 旋转于第 j 列，再构造新 Givens 消去 H[j+1][j]
                for (int i = 0; i < j; ++i)
                {
                    const double t1 =  cs[i] * H(i)[j] + sn[i] * H(i + 1)[j];
                    const double t2 = -sn[i] * H(i)[j] + cs[i] * H(i + 1)[j];
                    H(i)[j]     = t1;
                    H(i + 1)[j] = t2;
                }
                const double den = std::hypot(H(j)[j], H(j + 1)[j]);
                if (den < breakdown_tol)
                {
                    throw std::runtime_error("GMRES breakdown：Hessenberg 主元过小");
                }
                cs[j] = H(j)[j] / den;
                sn[j] = H(j + 1)[j] / den;
                H(j)[j] = cs[j] * H(j)[j] + sn[j] * H(j + 1)[j];
                H(j + 1)[j] = 0.0;
                const double g1 =  cs[j] * g[j] + sn[j] * g[j + 1];
                const double g2 = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = g1;
                g[j + 1] = g2;

                j_used = j + 1;
                it_end = total;
                ++total;

                // (3)e 残差监测：右预处理下 |g_{j+1}| 即真实残差
                if (std::fabs(g[j + 1]) / normb < tol)
                {
                    converged = true;
                    break;
                }
            }

            // ---- (4) 回代解 R y = g ----
            for (int i = j_used - 1; i >= 0; --i)
            {
                const double* Hi = H(i);
                double sum = g[i];
                for (int k = i + 1; k < j_used; ++k)
                { sum -= Hi[k] * y[k]; }
                if (std::fabs(Hi[i]) < breakdown_tol)
                {
                    throw std::runtime_error("GMRES breakdown：上三角主元过小");
                }
                y[i] = sum / Hi[i];
            }

            // ---- (5) x += M^{-1} (V y) ----
            // 交换内外循环：外层遍历基向量，内层对一段连续内存做 axpy。
            // 旧写法外层 t、内层 i，每次都要跨 n 个元素在 m 个基之间跳。
            std::fill(z.begin(), z.end(), 0.0);
            for (int i = 0; i < j_used; ++i)
            {
                const double yi = y[i];
                const double* Vi = V(i);
                double* zz = z.data();
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(n > 1000)
                #endif
                for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
                { zz[t] += yi * Vi[t]; }
            }
            solveM(z, dz);
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static) if(n > 1000)
            #endif
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n); ++t)
            { x[t] += dz[t]; }
        }

        return {x, it_end};
    }
    std::tuple<std::vector<double>, int> SMatrix::amg(const SMatrix& A, const std::vector<double>& b, double tol, int max_iter)
    {
        const uint32_t n = b.size();
        std::vector<double> x(n, 0.0);
        const double normb = std::sqrt(SMatrix::dot(b, b));
        if (normb < 1e-300) return {x, 0};

        auto data = make_amg_data(A);
        std::vector<double> r(n), d(n);
        int it_end = 0;
        for (int it = 0; it < max_iter; ++it)
        {
            std::vector<double> Ax = SMatrix::spmv(A, x);
            double rr = 0.0;
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for reduction(+:rr) schedule(static)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            { r[i] = b[i] - Ax[i]; rr += r[i] * r[i]; }
            it_end = it;
            if (std::sqrt(rr) / normb < tol) break;
            d.assign(n, 0.0);
            amg_vcycle(*data, 0, r, d);                 // x += V-cycle(r)
            #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp parallel for schedule(static)
            #endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) x[i] += d[i];
        }
        return {x, it_end};
    }
}


