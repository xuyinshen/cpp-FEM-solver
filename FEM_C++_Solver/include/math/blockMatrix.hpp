#pragma once
#include <vector>
#include <cstring>
#include <cstdio>
#include <array>
#include <cstdint>
#include <algorithm>
#include <map>
#include <memory>       // make_shared (Schur/BlockILU0 工作区)
#include <stdexcept>    // invalid_argument / runtime_error
#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef CPPNUM_HAVE_MPI
#include <mpi.h>
#endif
#include "math/array.hpp"   // SMatrix

using namespace sparse;

namespace block 
{

    #ifdef CPPNUM_HAVE_MPI

    // MPI 数据类型映射 Traits
    template <typename T> struct MPI_Type_Traits;
    template <> struct MPI_Type_Traits<double> { static MPI_Datatype type() { return MPI_DOUBLE; } };
    template <> struct MPI_Type_Traits<float>  { static MPI_Datatype type() { return MPI_FLOAT; } };
    template <> struct MPI_Type_Traits<int>    { static MPI_Datatype type() { return MPI_INT; } };
    template <> struct MPI_Type_Traits<unsigned int> { static MPI_Datatype type() { return MPI_UNSIGNED; } };

    // 便捷调用函数
    template <typename T>
    inline MPI_Datatype mpi_type() 
    { 
        return MPI_Type_Traits<T>::type(); 
    }
    #endif

    template <int BS>
    inline void blk_matmat_accum(const double* A, const double* B, double* C)
    {
        for (int r = 0; r < BS; ++r)
            for (int c = 0; c < BS; ++c)
            {
                double s = 0.0;
                for (int k = 0; k < BS; ++k) s += A[r*BS+k] * B[k*BS+c];
                C[r*BS+c] -= s;
            }
    }
    // C := C*B   (右乘, 不可交换性的体现)
    template <int BS>
    inline void blk_right_mul(double* C, const double* B)
    {
        double tmp[BS*BS];
        for (int r = 0; r < BS; ++r)
            for (int c = 0; c < BS; ++c)
            {
                double s = 0.0;
                for (int k = 0; k < BS; ++k) s += C[r*BS+k] * B[k*BS+c];
                tmp[r*BS+c] = s;
            }
        for (int t = 0; t < BS*BS; ++t) C[t] = tmp[t];
    }
    // y = M*x
    template <int BS>
    inline void blk_matvec(const double* M, const double* x, double* y)
    {
        for (int r = 0; r < BS; ++r)
        {
            double s = 0.0;
            for (int c = 0; c < BS; ++c) s += M[r*BS+c] * x[c];
            y[r] = s;
        }
    }
    // Minv = M^{-1} (高斯-部分主元); 失败返回 false
    template <int BS>
    inline bool blk_invert(const double* M, double* Minv)
    {
        double aug[BS][2*BS];
        for (int i = 0; i < BS; ++i)
            for (int j = 0; j < BS; ++j)
            { aug[i][j] = M[i*BS+j]; aug[i][BS+j] = (i==j) ? 1.0 : 0.0; }
        for (int k = 0; k < BS; ++k)
        {
            int piv = k;
            for (int i = k+1; i < BS; ++i)
                if (std::fabs(aug[i][k]) > std::fabs(aug[piv][k])) piv = i;
            if (std::fabs(aug[piv][k]) < 1e-300) return false;
            if (piv != k)
                for (int j = 0; j < 2*BS; ++j) std::swap(aug[k][j], aug[piv][j]);
            for (int i = k+1; i < BS; ++i)
            {
                const double f = aug[i][k] / aug[k][k];
                for (int j = k; j < 2*BS; ++j) aug[i][j] -= f * aug[k][j];
            }
        }
        for (int i = BS-1; i >= 0; --i)
        {
            for (int j = i+1; j < BS; ++j)
                for (int c = 0; c < BS; ++c) aug[i][BS+c] -= aug[i][j] * aug[j][BS+c];
            for (int c = 0; c < BS; ++c) aug[i][BS+c] /= aug[i][i];
        }
        for (int i = 0; i < BS; ++i)
            for (int j = 0; j < BS; ++j) Minv[i*BS+j] = aug[i][BS+j];
        return true;
    }
    // 行内二分查找块列; 未找到返回 UINT32_MAX (行内块列升序前提)
    inline uint32_t blk_find_col(const std::vector<uint32_t>& rp,
                                const std::vector<uint32_t>& ci,
                                uint32_t row, uint32_t col)
    {
        uint32_t lo = rp[row], hi = rp[row+1];
        while (lo < hi)
        {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (ci[mid] == col) return mid;
            if (ci[mid] < col) lo = mid + 1; else hi = mid;
        }
        return UINT32_MAX;
    }

    // Block-ILU(0) 数据: L (单位块下三角, 对角隐含), U (块上三角含对角), inv(U_KK)
    template <int BS>
    struct BlockILU0_Data
    {
        std::vector<uint32_t> L_row_ptr, L_col_idx;
        std::vector<double>   L_values;
        std::vector<uint32_t> U_row_ptr, U_col_idx;
        std::vector<double>   U_values;
        std::vector<uint32_t> U_diag_ptr;    // 每块行对角块的块偏移
        std::vector<double>   inv_diag;      // nb * BS*BS
        mutable std::vector<double> y_buf;   // apply 工作缓冲
    };
// =====================================================================
// BlockMatrix<T, BS> : Block-CSR (BAIJ) 分布式块稀疏矩阵
//   T  : 标量类型 (double)
//   BS : 块边长 (编译期: 2D NS 同阶元=3, 3D=4)
// 向量布局: node-major (块内 BS 个分量连续), 与 FEM 组装自然顺序一致,
//           因此 std::vector<double> 块向量可零改动接入现有 Krylov 求解器
// =====================================================================
    template <typename T, int BS>
    class BlockMatrix
    {
        private:
            uint32_t nbr_, nbc_;
            std::vector<uint32_t> row_ptr_;      // 块行指针 (长度 nbr+1)
            std::vector<uint32_t> col_idx_;      // 块列索引 (全局编号)
            std::vector<T>        values_;       // 连续块值 (nnz_blocks * BS*BS)
            std::vector<uint32_t> diag_ptr_;     // 每块行对角块的块偏移 (UINT32_MAX=缺)

            // ---- MPI 成员 ----
            int      rank_ = 0, nprocs_ = 1;
            uint32_t n_owned_ = 0, n_ghost_ = 0;          // 本地 owned / ghost 块行数
            std::vector<uint32_t> local_col_idx_;         // 全局块列 -> x_full 本地索引
            std::vector<int>      recv_procs_, recv_cnts_;
            std::vector<uint32_t> recv_global_;           // 要接收的全局块行
            std::vector<int>      send_procs_, send_cnts_;
            std::vector<uint32_t> send_local_;            // x_owned 中要发送的索引
            mutable std::vector<T> x_full_;               // owned+ghost 缓冲 (预分配)
            mutable std::vector<T> send_buf_;
            #ifdef CPPNUM_HAVE_MPI
                mutable std::vector<MPI_Request> reqs_;
            #endif


        public:
            static constexpr int kBlock     = BS;
            static constexpr int kBlockVals = BS * BS;

            // 构造函数
            BlockMatrix() : nbr_(0), nbc_(0), row_ptr_{0} {}
            BlockMatrix(
                uint32_t nbr, uint32_t nbc, 
                std::vector<uint32_t> row_ptr,
                std::vector<uint32_t> col_idx,
                std::vector<T> values
            ) : nbr_(nbr), nbc_(nbc),
                row_ptr_(std::move(row_ptr)),
                col_idx_(std::move(col_idx)),
                values_(std::move(values))
            { 
                cache_diag(); 
            }
            
            // ---- 访问器 ----
            uint32_t block_rows() const { return nbr_; }
            uint32_t block_cols() const { return nbc_; }
            uint32_t rows() const { return nbr_ * BS; }      // 标量自由度行数
            uint32_t cols() const { return nbc_ * BS; }
            size_t   nnz_blocks() const { return col_idx_.size(); }
            size_t   nnz() const { return col_idx_.size() * (size_t)kBlockVals; }

            const std::vector<uint32_t>& block_row_ptr() const { return row_ptr_; }
            const std::vector<uint32_t>& block_col_idx() const { return col_idx_; }
            const std::vector<T>&        values()        const { return values_; }

            const T* block(uint32_t p) const { return &values_[(size_t)p * kBlockVals]; }
            const T* diag_block(uint32_t bi) const { return &values_[(size_t)diag_ptr_[bi] * kBlockVals]; }

            // ---- 核心运算 ----
            void cache_diag()
            {
                diag_ptr_.assign(nbr_, UINT32_MAX);
                for (uint32_t bi = 0; bi < nbr_; ++bi)
                    for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                        if (col_idx_[p] == bi) { diag_ptr_[bi] = p; break; }
            }

            // 矩阵向量乘法
            void matvec(const std::vector<T>& x, std::vector<T>& y) const
            {
                y.assign(rows(), T(0));
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nbr_; ++bi)
                {
                    T* yi = &y[(size_t)bi * BS];
                    for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                    {
                        const T* B  = block(p);
                        const T* xj = &x[(size_t)col_idx_[p] * BS];
                        // 块内 GEMV: yi += B * xj  (BS 编译期展开 + simd)
                        for (int r = 0; r < BS; ++r)
                        {
                            T acc = T(0);
                            #ifdef CPPNUM_HAVE_OPENMP
                                #pragma omp simd reduction(+:acc)
                            #endif
                            for (int c = 0; c < BS; ++c)
                                acc += B[r * BS + c] * xj[c];
                            yi[r] += acc;
                        }
                    }
                }
            }

            // 转置
            BlockMatrix<T, BS> transpose() const
            {
                // 把块当"标量"做 CSR 转置, 得到块转置的结构
                std::vector<uint32_t> trp(nbc_ + 1, 0);
                for (size_t p = 0; p < col_idx_.size(); ++p) trp[col_idx_[p] + 1]++;
                for (uint32_t i = 0; i < nbc_; ++i) trp[i + 1] += trp[i];
                std::vector<uint32_t> tci(col_idx_.size());
                std::vector<uint32_t> off = trp;
                for (uint32_t i = 0; i < nbr_; ++i)
                    for (uint32_t p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p)
                        tci[off[col_idx_[p]]++] = i;

                std::vector<T> tv(values_.size());
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t j = 0; j < (std::ptrdiff_t)nbc_; ++j)      // 转置后的块行
                    for (uint32_t q = trp[j]; q < trp[j + 1]; ++q)
                    {
                        const uint32_t i = tci[q];                              // 原块行
                        // 找原矩阵中块 (i, j) 的位置 p
                        uint32_t p = row_ptr_[i];
                        while (col_idx_[p] != j) ++p;
                        const T* B = block(p);
                        T* C = &tv[(size_t)q * kBlockVals];
                        for (int r = 0; r < BS; ++r)
                            for (int c = 0; c < BS; ++c)
                                C[r * BS + c] = B[c * BS + r];                  // 块内转置
                    }
                return BlockMatrix(nbc_, nbr_, std::move(trp), std::move(tci), std::move(tv));
            }

            static BlockMatrix<T, BS> condense(const SMatrix& A, uint32_t nbr, uint32_t nbc)
            {
                const auto& Ap = A.row_ptr(); const auto& Ac = A.col_idx(); const auto& Av = A.values();

                // Pass 1: 每块行收集去重块列 (行并行, 局部去重)
                std::vector<std::vector<uint32_t>> blkcols(nbr);
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(dynamic, 64)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nbr; ++bi)
                {
                    auto& out = blkcols[bi];
                    for (uint32_t i = bi * BS; i < (bi + 1) * BS; ++i)
                        for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                            out.push_back(Ac[p] / BS);
                    std::sort(out.begin(), out.end());
                    out.erase(std::unique(out.begin(), out.end()), out.end());
                }

                std::vector<uint32_t> rp(nbr + 1, 0);
                for (uint32_t bi = 0; bi < nbr; ++bi) rp[bi + 1] = blkcols[bi].size();
                for (uint32_t bi = 0; bi < nbr; ++bi) rp[bi + 1] += rp[bi];
                std::vector<uint32_t> ci(rp[nbr]);
                std::vector<T> vv((size_t)rp[nbr] * kBlockVals, T(0));

                // Pass 2: 填值 (行并行, 各块行写自己的块值区间, 无竞争)
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(dynamic, 64)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nbr; ++bi)
                {
                    for (uint32_t k = 0; k < blkcols[bi].size(); ++k) ci[rp[bi] + k] = blkcols[bi][k];
                    for (uint32_t i = bi * BS; i < (bi + 1) * BS; ++i)
                    {
                        const int r = i % BS;
                        for (uint32_t p = Ap[i]; p < Ap[i + 1]; ++p)
                        {
                            const uint32_t bj = Ac[p] / BS, c = Ac[p] % BS;
                            // 块行 bi 内定位块 bj 的偏移 (块数小, 二分)
                            auto it = std::lower_bound(blkcols[bi].begin(), blkcols[bi].end(), bj);
                            const uint32_t boff = rp[bi] + (uint32_t)(it - blkcols[bi].begin());
                            vv[(size_t)boff * kBlockVals + (size_t)r * BS + c] += Av[p];
                        }
                    }
                }
                return BlockMatrix(nbr, nbc, std::move(rp), std::move(ci), std::move(vv));
            }

            // ---- MPI 分布 ----
            void set_halo(std::vector<int> recv_procs, std::vector<int> recv_cnts,
                  std::vector<uint32_t> recv_global,
                  std::vector<int> send_procs, std::vector<int> send_cnts,
                  std::vector<uint32_t> send_local,
                  uint32_t owned_global_start = 0)
            {
                recv_procs_ = std::move(recv_procs); recv_cnts_ = std::move(recv_cnts);
                recv_global_ = std::move(recv_global);
                send_procs_ = std::move(send_procs); send_cnts_ = std::move(send_cnts);
                send_local_ = std::move(send_local);
                n_ghost_ = (uint32_t)recv_global_.size();
                n_owned_ = nbr_;                                  // 本地行即 owned 行

                // 全局块列 -> x_full 本地索引 (owned 区 + ghost 区), 预计算避免每步查表
                local_col_idx_.resize(col_idx_.size());
                std::vector<uint32_t> ghost_pos(nbc_, UINT32_MAX);
                for (uint32_t k = 0; k < n_ghost_; ++k) ghost_pos[recv_global_[k]] = n_owned_ + k;
                for (size_t p = 0; p < col_idx_.size(); ++p)
                {
                    const uint32_t g = col_idx_[p];
                    if (g >= owned_global_start && g < owned_global_start + n_owned_)
                        local_col_idx_[p] = g - owned_global_start;          // owned 列 -> 本地偏移
                    else
                        local_col_idx_[p] = ghost_pos[g];                    // ghost 列 -> ghost 区
                }
                x_full_.resize(((size_t)n_owned_ + n_ghost_) * BS);
                send_buf_.resize(send_local_.size() * BS);
                #ifdef CPPNUM_HAVE_MPI
                    reqs_.resize(2 * (recv_procs_.size() + send_procs_.size()) + 4);
                #endif
            }
            uint32_t owned_rows() const { return n_owned_; }

            void distributed_matvec(const std::vector<T>& x_owned, std::vector<T>& y_owned) const
            {
                #ifdef CPPNUM_HAVE_MPI
                    int my_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // 获取当前进程的 Rank
                    
                    const size_t owned_vals = (size_t)n_owned_ * BS;
                    
                    // 1. 拷贝 owned 区 + 打包 send
                    for (size_t i = 0; i < owned_vals; ++i) x_full_[i] = x_owned[i];
                    for (size_t i = 0; i < send_local_.size(); ++i)
                        for (int c = 0; c < BS; ++c)
                            send_buf_[i * BS + c] = x_owned[(size_t)send_local_[i] * BS + c];

                    // 2. 非阻塞启动通信
                    int rq = 0; 
                    size_t off = owned_vals;
                    
                    // 接收 Ghost 数据
                    for (size_t k = 0; k < recv_procs_.size(); ++k)
                    {
                        int src = recv_procs_[k]; // 源进程的 Rank
                        // 【核心】接收方 Tag = 1000 + 源Rank
                        MPI_Irecv(&x_full_[off], recv_cnts_[k] * BS, mpi_type<T>(),
                                src, 1000 + src, MPI_COMM_WORLD, &reqs_[rq++]);
                        off += (size_t)recv_cnts_[k] * BS;
                    }
                    
                    off = 0;
                    // 发送 Owned 数据
                    for (size_t k = 0; k < send_procs_.size(); ++k)
                    {
                        int dst = send_procs_[k]; // 目标进程的 Rank
                        // 【核心】发送方 Tag = 1000 + 自己的Rank (即源Rank)
                        // 这样无论对方是谁，双方算出的 Tag 永远一致！
                        MPI_Isend(&send_buf_[off], send_cnts_[k] * BS, mpi_type<T>(),
                                dst, 1000 + my_rank, MPI_COMM_WORLD, &reqs_[rq++]);
                        off += (size_t)send_cnts_[k] * BS;
                    }

                    // 3. 通信-计算重叠: 算 owned 列
                    y_owned.assign(owned_vals, T(0));
                    #ifdef CPPNUM_HAVE_OPENMP
                        #pragma omp parallel for schedule(static)
                    #endif
                    for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)n_owned_; ++bi)
                    {
                        T* yi = &y_owned[(size_t)bi * BS];
                        for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                        {
                            if (local_col_idx_[p] >= n_owned_) continue;
                            const T* B = block(p);
                            const T* xj = &x_full_[(size_t)local_col_idx_[p] * BS];
                            for (int r = 0; r < BS; ++r)
                            {
                                T acc = 0;
                                for (int c = 0; c < BS; ++c) acc += B[r * BS + c] * xj[c];
                                yi[r] += acc;
                            }
                        }
                    }

                    // 4. 等待通信完成 (防御性判断)
                    if (rq > 0) 
                    {
                        MPI_Waitall(rq, reqs_.data(), MPI_STATUSES_IGNORE);
                    }

                    // 5. 补算 ghost 列
                    #ifdef CPPNUM_HAVE_OPENMP
                        #pragma omp parallel for schedule(static)
                    #endif
                    for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)n_owned_; ++bi)
                    {
                        T* yi = &y_owned[(size_t)bi * BS];
                        for (uint32_t p = row_ptr_[bi]; p < row_ptr_[bi + 1]; ++p)
                        {
                            if (local_col_idx_[p] < n_owned_) continue;
                            const T* B = block(p);
                            const T* xj = &x_full_[(size_t)local_col_idx_[p] * BS];
                            for (int r = 0; r < BS; ++r)
                            {
                                T acc = 0;
                                for (int c = 0; c < BS; ++c) acc += B[r * BS + c] * xj[c];
                                yi[r] += acc;
                            }
                        }
                    }
                #else
                    matvec(x_owned, y_owned);
                #endif
            }

            static std::tuple<std::vector<T>, int> gmres(
                const BlockMatrix& A,
                const std::vector<T>& b,
                const std::string& precond,
                double tol,
                int max_iter,
                int restart
            );
    };

    // ---- Block-ilu 算法 ----
    template <typename T, int BS>
    std::function<void(const std::vector<T>&, std::vector<T>&)> make_block_ilu0_preconditioner(const BlockMatrix<T, BS>& A) // 【修复1】正确的双模板参数
    {
        if (A.block_rows() != A.block_cols())
            throw std::invalid_argument("Block-ILU(0) 要求块矩阵为方阵");

        const uint32_t nb = A.block_rows();
        const int BV = BS * BS;
        const auto& Arp = A.block_row_ptr();
        const auto& Aci = A.block_col_idx();
        const auto& Avv = A.values();

        // 【修复2】必须加上 <BS>
        auto data = std::make_shared<BlockILU0_Data<BS>>();

        // 1. 计数: L (J<I) 与 U (J>=I)
        data->L_row_ptr.assign(nb + 1, 0);
        data->U_row_ptr.assign(nb + 1, 0);
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (std::ptrdiff_t I = 0; I < (std::ptrdiff_t)nb; ++I)
        {
            uint32_t cL = 0, cU = 0;
            for (uint32_t p = Arp[I]; p < Arp[I+1]; ++p)
            { if (Aci[p] < (uint32_t)I) ++cL; else ++cU; }
            data->L_row_ptr[I+1] = cL;
            data->U_row_ptr[I+1] = cU;
        }
        for (uint32_t I = 0; I < nb; ++I)
        { data->L_row_ptr[I+1] += data->L_row_ptr[I];
        data->U_row_ptr[I+1] += data->U_row_ptr[I]; }

        data->L_col_idx.resize(data->L_row_ptr[nb]);
        data->L_values.resize((size_t)data->L_row_ptr[nb] * BV);
        data->U_col_idx.resize(data->U_row_ptr[nb]);
        data->U_values.resize((size_t)data->U_row_ptr[nb] * BV);
        data->U_diag_ptr.assign(nb, UINT32_MAX);
        std::vector<char> has_diag(nb, 0);

        // 2. 填充
        // 【修复3】补全 offU
        std::vector<uint32_t> offL = data->L_row_ptr, offU = data->U_row_ptr;
        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static)
        #endif
        for (std::ptrdiff_t I = 0; I < (std::ptrdiff_t)nb; ++I)
        {
            for (uint32_t p = Arp[I]; p < Arp[I+1]; ++p)
            {
                const uint32_t J = Aci[p];
                const T* src = &Avv[(size_t)p * BV];
                if (J < (uint32_t)I)
                {
                    const uint32_t d = offL[I]++;
                    data->L_col_idx[d] = J;
                    std::copy(src, src + BV, &data->L_values[(size_t)d * BV]);
                }
                else
                {
                    const uint32_t d = offU[I]++;
                    data->U_col_idx[d] = J;
                    std::copy(src, src + BV, &data->U_values[(size_t)d * BV]);
                    if (J == (uint32_t)I) { data->U_diag_ptr[I] = d; has_diag[I] = 1; }
                }
            }
        }
        for (uint32_t I = 0; I < nb; ++I)
            if (!has_diag[I])
                throw std::runtime_error("Block-ILU(0) 失败：块行 " + std::to_string(I) + " 缺少对角块");

        // 3. 分解
        data->inv_diag.resize((size_t)nb * BV);
        for (uint32_t I = 0; I < nb; ++I)
        {
            const uint32_t Ls = data->L_row_ptr[I];
            const uint32_t Le = data->L_row_ptr[I+1]; // 【修复4】补全 Le
            const uint32_t Us = data->U_row_ptr[I];
            const uint32_t Ue = data->U_row_ptr[I+1]; // 【修复5】补全 Ue

            for (uint32_t pL = Ls; pL < Le; ++pL)
            {
                const uint32_t K = data->L_col_idx[pL];
                T* LIK = &data->L_values[(size_t)pL * BV];

                // 左和式
                for (uint32_t q = Ls; q < pL; ++q)
                {
                    const uint32_t J = data->L_col_idx[q];
                    const uint32_t pJK = blk_find_col(data->U_row_ptr, data->U_col_idx, J, K);
                    if (pJK != UINT32_MAX)
                        blk_matmat_accum<BS>(&data->L_values[(size_t)q * BV],
                                            &data->U_values[(size_t)pJK * BV], LIK);
                }
                
                // 右乘逆
                blk_right_mul<BS>(LIK, &data->inv_diag[(size_t)K * BV]);

                // 双指针更新 Urow(I)
                uint32_t pi = Us, pk = data->U_row_ptr[K];
                const uint32_t ek = data->U_row_ptr[K+1];
                while (pi < Ue && pk < ek)
                {
                    if (data->U_col_idx[pi] == data->U_col_idx[pk])
                    {
                        blk_matmat_accum<BS>(LIK, &data->U_values[(size_t)pk * BV],
                                            &data->U_values[(size_t)pi * BV]);
                        ++pi; ++pk;
                    }
                    else if (data->U_col_idx[pi] < data->U_col_idx[pk]) ++pi;
                    else ++pk;
                }
            }

            // 缓存对角块之逆
            if (!(blk_invert<BS>(&data->U_values[(size_t)data->U_diag_ptr[I] * BV],
                                &data->inv_diag[(size_t)I * BV])))
                throw std::runtime_error("Block-ILU(0) breakdown：对角块 U_" +
                                        std::to_string(I) + std::to_string(I) + " 奇异");
        }

        // 4. 返回 apply (前代 + 回代)
        // 修复：因为 data 是 shared_ptr，按值捕获是完美且安全的
        return [data, nb](const std::vector<T>& rhs, std::vector<T>& z)
        {
            auto& y = data->y_buf;
            y.resize((size_t)nb * BS);
            z.resize((size_t)nb * BS);

            // 前代
            for (uint32_t I = 0; I < nb; ++I)
            {
                T* yI = &y[(size_t)I * BS];
                for (int c = 0; c < BS; ++c) yI[c] = rhs[(size_t)I * BS + c];
                for (uint32_t p = data->L_row_ptr[I]; p < data->L_row_ptr[I+1]; ++p)
                {
                    const T* LIJ = &data->L_values[(size_t)p * BV];
                    const T* yJ = &y[(size_t)data->L_col_idx[p] * BS];
                    for (int r = 0; r < BS; ++r)
                    {
                        T s = 0;
                        for (int c = 0; c < BS; ++c) s += LIJ[r*BS+c] * yJ[c];
                        yI[r] -= s;
                    }
                }
            }

            // 回代
            for (uint32_t II = nb; II-- > 0; )
            {
                T tmp[BS];
                for (int c = 0; c < BS; ++c) tmp[c] = y[(size_t)II * BS + c];
                for (uint32_t p = data->U_row_ptr[II]; p < data->U_row_ptr[II+1]; ++p)
                {
                    const uint32_t J = data->U_col_idx[p];
                    if (J == II) continue;
                    const T* UIJ = &data->U_values[(size_t)p * BV];
                    const T* zJ = &z[(size_t)J * BS];
                    for (int r = 0; r < BS; ++r)
                    {
                        T s = 0;
                        for (int c = 0; c < BS; ++c) s += UIJ[r*BS+c] * zJ[c];
                        tmp[r] -= s;
                    }
                }
                blk_matvec<BS>(&data->inv_diag[(size_t)II * BV], tmp, &z[(size_t)II * BS]);
            }
        };
    }

    // ---- 舒尔补算法 ----

    // 1. 鞍点系统结构体 (持有子矩阵的 spmv 接口)
    struct SaddlePointSystem {
        size_t n_vel;
        size_t n_pres;
        std::function<void(const std::vector<double>&, std::vector<double>&)> spmv_A;
        std::function<void(const std::vector<double>&, std::vector<double>&)> spmv_B;
        std::function<void(const std::vector<double>&, std::vector<double>&)> spmv_Bt;
    };

    // 2. Schur 补预处理工厂函数
    //
    //   【必须 inline】本函数定义在头文件里, 而 blockMatrix.hpp 被多个 TU 包含。
    //   非 inline 的命名空间级函数会在每个 TU 生成强符号 (ODR 违规)。
    //
    //   【工作缓冲区: 不再用 static thread_local】
    //   原实现把 9 个 std::vector 声明成 lambda 体内的 static thread_local, 有三个
    //   问题, 其中第 1 个正是 8 线程段错误的头号嫌疑:
    //     1) 同一个 lambda 表达式只产生 *一个* 闭包类型, 于是 static 局部变量在
    //        该类型的所有实例间共享 —— 测试里 schur_I 与 schur_AMG 其实共用同一
    //        批缓冲。更关键的是 thread_local 的初始化/析构由运行时按线程管理,
    //        与 OpenMP 工作线程池的生命周期交织, 是典型的多线程崩溃源。
    //     2) 9 个 vector x T 个线程把内存长期钉住, 尺寸变化时还要反复 resize。
    //     3) 两个不同尺寸的 Schur 预处理器交替使用会反复重分配。
    //   改成随闭包走的状态块 (shared_ptr 按值捕获): 每个预处理器独立一份, 生命
    //   周期与闭包一致; apply 内部多线程只读尺寸、写各自区间, 无共享可变状态。
    inline std::function<void(const std::vector<double>&, std::vector<double>&)>
    make_schur_preconditioner(
        const SaddlePointSystem& sys,
        std::function<void(const std::vector<double>&, std::vector<double>&)> A_precond,
        std::function<void(const std::vector<double>&, std::vector<double>&)> S_precond)
    {
        // 工作缓冲区: 每次 apply 复用, 只在尺寸变化时 resize。
        struct Workspace {
            std::vector<double> r_u, r_p, w_u, w_p, r_p_mod, z_p, w_bt, r_u_mod, z_u;
            void ensure(std::size_t nv, std::size_t np)
            {
                if (r_u.size()     != nv) r_u.resize(nv);
                if (r_p.size()     != np) r_p.resize(np);
                if (w_u.size()     != nv) w_u.resize(nv);
                if (w_p.size()     != np) w_p.resize(np);
                if (r_p_mod.size() != np) r_p_mod.resize(np);
                if (z_p.size()     != np) z_p.resize(np);
                if (w_bt.size()    != nv) w_bt.resize(nv);
                if (r_u_mod.size() != nv) r_u_mod.resize(nv);
                if (z_u.size()     != nv) z_u.resize(nv);
            }
        };
        auto ws = std::make_shared<Workspace>();

        return [sys, A_precond, S_precond, ws](
                   const std::vector<double>& r, std::vector<double>& z) mutable
        {
            Workspace& w = *ws;                 // 引用, 不改变所有权
            const std::size_t n_vel  = sys.n_vel;
            const std::size_t n_pres = sys.n_pres;

            // 【防御】长度不符就直接抛异常, 而不是让 memcpy 越界读写成段错误。
            if (r.size() != n_vel + n_pres)
                throw std::invalid_argument("Schur 预处理器: 右端项长度 != n_vel + n_pres");

            w.ensure(n_vel, n_pres);

            // 拆分 r: 两段都是连续区间, 直接用 memcpy (编译器展开成向量化块拷贝)。
            // 原来是两条逐元素串行循环, 既不可向量化也不并行。
            if (n_vel)  std::memcpy(w.r_u.data(), r.data(),         n_vel  * sizeof(double));
            if (n_pres) std::memcpy(w.r_p.data(), r.data() + n_vel, n_pres * sizeof(double));

            // 1. w_u = A^{-1} r_u
            A_precond(w.r_u, w.w_u);

            // 2. w_p = B * w_u
            sys.spmv_B(w.w_u, w.w_p);

            // 3. r_p_mod = w_p - r_p   (并行 + 自动向量化)
            {
                const std::ptrdiff_t np_ = (std::ptrdiff_t)n_pres;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(np_ > 4096)
                #endif
                for (std::ptrdiff_t i = 0; i < np_; ++i)
                    w.r_p_mod[i] = w.w_p[i] - w.r_p[i];
            }

            // 4. z_p = S^{-1} r_p_mod
            S_precond(w.r_p_mod, w.z_p);

            // 5. w_bt = B^T * z_p
            sys.spmv_Bt(w.z_p, w.w_bt);

            // 6. r_u_mod = r_u - w_bt   (并行 + 自动向量化)
            {
                const std::ptrdiff_t nv_ = (std::ptrdiff_t)n_vel;
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static) if(nv_ > 4096)
                #endif
                for (std::ptrdiff_t i = 0; i < nv_; ++i)
                    w.r_u_mod[i] = w.r_u[i] - w.w_bt[i];
            }

            // 7. z_u = A^{-1} r_u_mod
            A_precond(w.r_u_mod, w.z_u);

            // 组装 z: 同样是两段连续拷贝
            z.resize(n_vel + n_pres);
            if (n_vel)  std::memcpy(z.data(),         w.z_u.data(), n_vel  * sizeof(double));
            if (n_pres) std::memcpy(z.data() + n_vel, w.z_p.data(), n_pres * sizeof(double));
        };
    }

    // Block 版本的 gmres
    template <typename T, int BS>
    std::tuple<std::vector<T>, int> BlockMatrix<T, BS>::gmres(
        const BlockMatrix& A,
        const std::vector<T>& b,
        const std::string& precond,
        double tol,
        int max_iter,
        int restart)
    {
        const size_t n = b.size();
        const int m = restart;
        std::vector<T> x(n, T(0));
        
        // 计算 ||b||
        T normb_sq = T(0);
        for (size_t i = 0; i < n; ++i) normb_sq += b[i] * b[i];
        const T normb = std::sqrt(normb_sq);
        if (normb < 1e-300) return {x, 0};

        // ====== 字符串分发 ======
        std::function<void(const std::vector<T>&, std::vector<T>&)> solveM;
        
        if (precond == "none" || precond == "identity")
        {
            solveM = [](const std::vector<T>& r, std::vector<T>& z) { z = r; };
        }
        else if (precond == "blockjacobi")
        {
            // 构造块 Jacobi 预处理器
            solveM = [A](const std::vector<T>& r, std::vector<T>& z) {
                z.resize(r.size());
                const uint32_t nb = A.block_rows();
                #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp parallel for schedule(static)
                #endif
                for (std::ptrdiff_t bi = 0; bi < (std::ptrdiff_t)nb; ++bi)
                {
                    const T* D = A.diag_block(bi);
                    // 这里需要调用块求逆，简化起见直接用高斯消元
                    // 实际应调用 blk_invert<BS>
                    T tmp[BS];
                    for (int c = 0; c < BS; ++c) tmp[c] = r[(size_t)bi * BS + c];
                    T Minv[BS * BS];
                    blk_invert<BS>(D, Minv);
                    blk_matvec<BS>(Minv, tmp, &z[(size_t)bi * BS]);
                }
            };
        }
        else if (precond == "blockilu" || precond == "blockilu0")
        {
            // 构造 Block-ILU(0) 预处理器
            solveM = make_block_ilu0_preconditioner(A);
        }
        else
        {
            throw std::invalid_argument("BlockMatrix::gmres 未知预处理类型: " + precond);
        }

        // ====== GMRES 核心循环（与 SMatrix::gmres 完全相同，只是 spmv 改为 A.matvec）======
        const T breakdown_tol = 1e-30;
        std::vector<std::vector<T>> V(m + 1, std::vector<T>(n, T(0)));
        std::vector<std::vector<T>> H(m + 1, std::vector<T>(m, T(0)));
        std::vector<T> cs(m, T(0)), sn(m, T(0));
        std::vector<T> g(m + 1, T(0));
        std::vector<T> y(m, T(0));
        std::vector<T> w(n), wtmp(n), z(n), dz(n);

        int total = 0;
        int it_end = 0;
        bool converged = false;

        while (total < max_iter && !converged)
        {
            //std::cout << "第" << total << "次迭代" << std::endl;
            T beta;
            if (total == 0)
            {
                beta = normb;
                for (size_t i = 0; i < n; ++i) V[0][i] = b[i] / beta;
            }
            else
            {
                std::vector<T> Ax;
                A.matvec(x, Ax);
                T rr = T(0);
                for (size_t i = 0; i < n; ++i)
                {
                    T ri = b[i] - Ax[i];
                    V[0][i] = ri;
                    rr += ri * ri;
                }
                beta = std::sqrt(rr);
                if (beta / normb < tol) { converged = true; break; }
                for (size_t i = 0; i < n; ++i) V[0][i] /= beta;
            }

            g.assign(m + 1, T(0));
            g[0] = beta;
            for (int i = 0; i < m + 1; ++i) H[i].assign(m, T(0));

            int j_used = 0;

            for (int j = 0; j < m && total < max_iter; ++j)
            {
                solveM(V[j], wtmp);
                A.matvec(wtmp, w);  // w = A * M^{-1} * v_j

                for (int i = 0; i <= j; ++i)
                {
                    T hij = T(0);
                    for (size_t t = 0; t < n; ++t) hij += w[t] * V[i][t];
                    H[i][j] = hij;
                    for (size_t t = 0; t < n; ++t) w[t] -= hij * V[i][t];
                }

                T wnorm_sq = T(0);
                for (size_t t = 0; t < n; ++t) wnorm_sq += w[t] * w[t];
                H[j + 1][j] = std::sqrt(wnorm_sq);
                
                if (H[j + 1][j] > breakdown_tol)
                {
                    const T inv = T(1) / H[j + 1][j];
                    for (size_t t = 0; t < n; ++t) V[j + 1][t] = w[t] * inv;
                }

                for (int i = 0; i < j; ++i)
                {
                    const T t1 =  cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                    const T t2 = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                    H[i][j] = t1;
                    H[i + 1][j] = t2;
                }
                const T den = std::hypot(H[j][j], H[j + 1][j]);
                if (den < breakdown_tol)
                    throw std::runtime_error("GMRES breakdown: Hessenberg pivot too small");
                cs[j] = H[j][j] / den;
                sn[j] = H[j + 1][j] / den;
                H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j];
                H[j + 1][j] = T(0);
                const T g1 =  cs[j] * g[j] + sn[j] * g[j + 1];
                const T g2 = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = g1;
                g[j + 1] = g2;

                j_used = j + 1;
                it_end = total;
                ++total;

                if (std::fabs(g[j + 1]) / normb < tol)
                {
                    converged = true;
                    break;
                }
            }

            for (int i = j_used - 1; i >= 0; --i)
            {
                T sum = g[i];
                for (int k = i + 1; k < j_used; ++k)
                    sum -= H[i][k] * y[k];
                if (std::fabs(H[i][i]) < breakdown_tol)
                    throw std::runtime_error("GMRES breakdown: upper triangular pivot too small");
                y[i] = sum / H[i][i];
            }

            for (size_t t = 0; t < n; ++t)
            {
                T acc = T(0);
                for (int i = 0; i < j_used; ++i)
                    acc += y[i] * V[i][t];
                z[t] = acc;
            }
            solveM(z, dz);
            for (size_t t = 0; t < n; ++t)
                x[t] += dz[t];
        }

        return {x, it_end};
    }

    // =====================================================================
    // Block AMG (块代数多重网格) — 光滑聚合方法
    // 需要在文件顶部添加: #include <map>  #include <memory>
    // =====================================================================

    // ---- 辅助: 块 Frobenius 范数 ----
    template <typename T, int BS>
    inline T blk_fro_norm(const T* blk) 
    {
        T s = 0;
        for (int i = 0; i < BS * BS; ++i) s += blk[i] * blk[i];
        return std::sqrt(s);
    }

    // ---- 辅助: 块矩阵乘法 C = A * B (密集累加器 + 稀疏化) ----
    //
    //   drop_tol   : 相对丢弃阈值。每行取 maxn = max_j ||C_ij||_F,
    //                凡是 ||C_ij||_F < drop_tol * maxn 的块直接丢弃。
    //                drop_tol = 0 (默认) 表示不丢弃, 保持原有行为。
    //   max_row_nnz: 每行最多保留多少个块 (0 = 不限)。超出时只保留范数最大的
    //                max_row_nnz 个 —— 即 AMG 里标准的「行容量截断」。
    //
    //   对角块 (j == i) 永远保留: 丢掉它会让 Jacobi 光滑除零、粗层 LU 奇异。
    //
    //   【为什么必须稀疏化】光滑聚合的 Galerkin 乘子 R*A*P 会让模板逐层膨胀。
    //     N=32 实测 nnz/row: 7 -> 34.7 -> 257 -> 稠密,
    //     operator complexity 达 3.20, setup 时间超线性增长; 而且 V-cycle
    //     每一步都要在这些稠密粗层上做 matvec, 把每次 apply 的成本也抬高。
    //     截断后 nnz/row 稳定在几十, complexity 回到 ~1.5。
    template <typename T, int BS>
    BlockMatrix<T, BS> blk_matmul(const BlockMatrix<T, BS>& A, const BlockMatrix<T, BS>& B,
                                  double drop_tol = 0.0, uint32_t max_row_nnz = 0)
    {
        const int BV = BS * BS;
        const uint32_t nr = A.block_rows();
        const uint32_t nc = B.block_cols();

        std::vector<uint32_t> rp(nr + 1, 0);
        std::vector<uint32_t> ci;
        std::vector<T> val;

        std::vector<T> accum((size_t)nc * BV);
        // 【内存/带宽】原来是 std::vector<char> hit(nc,0) 并在每行开头
        //   std::fill(hit.begin(), hit.end(), 0)  —— 光清零就是 O(nr*nc)。
        //   AMG 第 0 层 nr=4.2e5, nc=1.6e4~1.4e5, 一次 Galerkin 就是几十 GB
        //   的纯写流量, 而且 A*P 和 R*(AP) 各来一遍。
        // 改成"时间戳"标记: 每行换一个 stamp, seen[j]==stamp 才算命中,
        //   清零成本从 O(nr*nc) 降到 O(nc)(只在回绕时清一次)。
        std::vector<uint32_t> seen(nc, 0);
        uint32_t stamp = 0;
        std::vector<uint32_t> hit_list;
        // 稀疏化用的范数缓冲。提到行循环外复用 —— 否则每行两次堆分配,
        //   nr=3e4 就是 6 万次 malloc/free, 反而把稀疏化省下的时间吃掉。
        std::vector<T> nrm, srt;

        for (uint32_t i = 0; i < nr; ++i) {
            hit_list.clear();
            if (++stamp == 0) { std::fill(seen.begin(), seen.end(), 0u); stamp = 1; }

            for (uint32_t pa = A.block_row_ptr()[i]; pa < A.block_row_ptr()[i+1]; ++pa) {
                uint32_t k = A.block_col_idx()[pa];
                const T* ab = &A.values()[(size_t)pa * BV];
                for (uint32_t pb = B.block_row_ptr()[k]; pb < B.block_row_ptr()[k+1]; ++pb) {
                    uint32_t j = B.block_col_idx()[pb];
                    const T* bb = &B.values()[(size_t)pb * BV];
                    if (seen[j] != stamp) {
                        seen[j] = stamp;
                        hit_list.push_back(j);
                        std::fill(accum.begin() + (size_t)j*BV,
                                accum.begin() + (size_t)(j+1)*BV, T(0));
                    }
                    T* cb = &accum[(size_t)j * BV];
                    for (int r = 0; r < BS; ++r)
                        for (int c = 0; c < BS; ++c) {
                            T s = 0;
                            for (int m = 0; m < BS; ++m) s += ab[r*BS+m] * bb[m*BS+c];
                            cb[r*BS+c] += s;
                        }
                }
            }
            std::sort(hit_list.begin(), hit_list.end());

            // ---- 稀疏化: 相对阈值 + 行容量截断 ----
            //   两个判据取「更严的那个」(thr = max(relthr, capthr)):
            //     relthr = drop_tol * max_j ||C_ij||_F       —— 丢掉相对噪声
            //     capthr = 第 max_row_nnz 大的范数           —— 硬顶行长度
            //   nth_element 是 O(n) 的第 k 大选择, 不需要全排序。
            //   对角块 (j == i) 无条件保留。
            if ((drop_tol > 0.0 || max_row_nnz > 0) && hit_list.size() > 1)
            {
                const std::size_t nz = hit_list.size();
                nrm.resize(nz);
                T maxn = T(0);
                for (std::size_t q = 0; q < nz; ++q)
                {
                    nrm[q] = blk_fro_norm<T, BS>(&accum[(size_t)hit_list[q] * BV]);
                    if (nrm[q] > maxn) maxn = nrm[q];
                }

                const T relthr = (drop_tol > 0.0) ? (T)(drop_tol * (double)maxn) : T(0);
                T capthr = T(0);
                if (max_row_nnz > 0 && nz > max_row_nnz)
                {
                    srt.assign(nrm.begin(), nrm.end());
                    std::nth_element(srt.begin(),
                                     srt.begin() + (std::ptrdiff_t)(max_row_nnz - 1),
                                     srt.end(), [](T a, T b) { return a > b; });
                    capthr = srt[(std::size_t)(max_row_nnz - 1)];
                }
                const T thr = (relthr > capthr) ? relthr : capthr;

                // 原地压缩, 保持升序
                std::size_t keep = 0;
                for (std::size_t q = 0; q < nz; ++q)
                {
                    const uint32_t j = hit_list[q];
                    if (j == i || nrm[q] >= thr) hit_list[keep++] = j;
                }
                hit_list.resize(keep);
            }

            rp[i+1] = (uint32_t)hit_list.size();
            for (uint32_t j : hit_list) {
                ci.push_back(j);
                for (int v = 0; v < BV; ++v) val.push_back(accum[(size_t)j*BV + v]);
            }
        }
        for (uint32_t i = 0; i < nr; ++i) rp[i+1] += rp[i];
        return BlockMatrix<T, BS>(nr, nc, rp, ci, val);
    }

    // ---- 层级数据 ----
    template <typename T, int BS>
    struct BlockAMGLevel 
    {
        BlockMatrix<T, BS> A;      // 当前层算子
        BlockMatrix<T, BS> P;      // 延拓 (细→粗)
        BlockMatrix<T, BS> R;      // 限制 (粗→细), R = P^T
        std::vector<T> diag_inv;   // 对角块逆 (nrows × BS²)
        uint32_t nrows;

        // ---- 单次 apply 复用的工作缓冲 (见 blk_amg_vcycle) ----
        //   随层次活到预处理器析构, 不再每个 V-cycle malloc/free。
        //   w_sol 只在 lvl>0 用到 (第 0 层直接用调用方传入的 sol)。
        std::vector<T> w_sol, w_tmp, w_r, w_rc, w_pec;

        // ---- 最粗层稠密 LU 缓存 (setup 时分解一次, 每个 V-cycle 只回代) ----
        std::vector<T>   c_lu;      // sz*sz, 原地 LU, 行序已被主元交换打乱
        std::vector<int> c_piv;     // 第 k 步选中的主元行号
        bool c_ready = false;       // 已分解
        bool c_pinv  = false;       // 撞过主元下限 => 该方向是零空间, 解完去均值

        // ---- 标量快速路径 (每个非零块都是 m*I_BS) ----
        //   只有 blk_amg_scalarize 在 setup 时判定通过 (L.scalar == true) 才有效。
        //   命中时块版的 A/P/R/diag_inv 全部留空 —— 反正 V-cycle 不会再看它们。
        bool scalar = false;
        std::vector<uint32_t> s_rp, s_ci;   // 标量算子 CSR (n x n)
        std::vector<T>        s_val;
        std::vector<T>        s_diag_inv;   // 1/m_ii  (Jacobi 光滑用)
        std::vector<uint32_t> p_rp, p_ci;   // 标量延拓 P (n x n_c)
        std::vector<T>        p_val;
        std::vector<uint32_t> r_rp, r_ci;   // 标量限制 R = P^T (n_c x n)
        std::vector<T>        r_val;
        // 最粗层标量 LU: A_c = M (x) I_BS, 只需分解 n_c x n_c 的 M,
        //   回代时按分量展开 —— 分解的 flops 比块版少 BS^3 倍 (BS=3 即 27 倍)。
        std::vector<T>   c_lu_s;
        std::vector<int> c_piv_s;
        bool c_ready_s = false;
        bool c_pinv_s  = false;
    };

    // ---- 最粗层稠密 LU 分解 (只在 setup 调一次) ----
    //   【原来】这段分解被写在 V-cycle 里, 于是每个 V-cycle 都要重算一遍:
    //     sz = 132 (BS=3, 50 块) 时 132^3/3 ≈ 1.5 MFlop, 完全串行, 还是
    //     三重循环里最内层长度递减的 axpy —— 占了单次 apply 相当可观的一块。
    //     粗算子在 apply 之间是不变的, 这份工纯属重复劳动。
    //   【数值安全网】主元下限改成「相对本层算子量级」, 不再用绝对量 1e-14:
    //     粗算子近奇异 (周期边界 Laplace + 对角正则化, 零空间 = 常数向量),
    //     最后一个主元约等于 lambda_min; Galerkin 会把算子逐级放大, 绝对
    //     1e-14 的 clamp 在放大后等价于 1e14 倍放大 —— 原来 1e+47 的来源。
    //     撞到下限就记 c_pinv, 解完减掉均值 (伪逆), 这正是 Schur 预处理器
    //     真正想要的 (真 Schur 补本身也是奇异的)。
    template <typename T, int BS>
    void blk_amg_factor_coarse(BlockAMGLevel<T, BS>& L)
    {
        const int BV = BS * BS;
        const uint32_t n = L.nrows;
        const int sz = (int)(n * BS);

        std::vector<T>& dense = L.c_lu;
        dense.assign((size_t)sz * sz, T(0));
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t p = L.A.block_row_ptr()[i]; p < L.A.block_row_ptr()[i+1]; ++p) {
                uint32_t j = L.A.block_col_idx()[p];
                const T* blk = &L.A.values()[(size_t)p * BV];
                for (int r = 0; r < BS; ++r)
                    for (int c = 0; c < BS; ++c)
                        dense[(size_t)(i*BS+r)*sz + j*BS+c] = blk[r*BS+c];
            }

        double dscale = 0.0;
        for (size_t q = 0; q < dense.size(); ++q)
            if (std::fabs(dense[q]) > dscale) dscale = std::fabs(dense[q]);
        if (dscale == 0.0) dscale = 1.0;
        const T dmin = (T)(1e-13 * dscale);
        bool hit_floor = false;

        // 高斯消元 (部分主元)
        L.c_piv.assign((size_t)sz, 0);
        for (int k = 0; k < sz; ++k) {
            int pk = k;
            for (int i = k+1; i < sz; ++i)
                if (std::fabs(dense[(size_t)i*sz+k]) > std::fabs(dense[(size_t)pk*sz+k])) pk = i;
            L.c_piv[(size_t)k] = pk;
            if (pk != k)
                for (int j = 0; j < sz; ++j)
                    std::swap(dense[(size_t)k*sz+j], dense[(size_t)pk*sz+j]);
            T d = dense[(size_t)k*sz+k];
            if (std::fabs(d) < dmin)
            { d = (d >= 0 ? dmin : -dmin); hit_floor = true; }
            dense[(size_t)k*sz+k] = d;      // 存回被 clamp 的主元, 回代时不再判断
            for (int i = k+1; i < sz; ++i) {
                T f = dense[(size_t)i*sz+k] / d;
                for (int j = k; j < sz; ++j)
                    dense[(size_t)i*sz+j] -= f * dense[(size_t)k*sz+j];
            }
        }
        L.c_pinv  = hit_floor;
        L.c_ready = true;
    }

    // ---- V-cycle 前向声明 (标量版 vcycle 要递归回通用版) ----
    template <typename T, int BS>
    void blk_amg_vcycle(std::vector<BlockAMGLevel<T, BS>>& levels,
                        int lvl, int n_smooth, T omega,
                        const std::vector<T>& rhs, std::vector<T>& sol);

    // =====================================================================
    // 标量快速路径 (块 = m*I_BS)
    // =====================================================================
    //
    //   【为什么值得做】BAMG 的单价一直卡在带宽墙上 (N=32 实测 ~10 ms/次 apply,
    //   是 ILU(0)+标量 AMG 的 1.78 倍), 而 BS=3 的块格式把这份带宽又放大了:
    //     * 块 matvec 每个非零块要读 BS^2 = 9 个 double, 但只算 3 个自由度的贡献;
    //       算子里真正的信息只有 1 个标量 m。矩阵值的流量被白白放大 9 倍。
    //     * 块 Jacobi 要读 9 个 double 的 D_i^{-1}, 做 9 次乘加 —— 而 D_i = m_ii*I
    //       的逆就是 (1/m_ii)*I, 一个标量就够。
    //     * 最粗层稠密 LU 是 (n_c*BS)^2 的矩阵, 而它其实是 M (x) I_BS,
    //       分解 n_c x n_c 的 M 就够, flops 少 27 倍。
    //   于是"每个块都是 m*I"这个结构信息, 本质上是把 BS^3 的算子压回 BS^1。
    //
    //   【什么时候会用上】三维 Stokes / 弹性 / 各向同性扩散等"同一标量算子按
    //   分量复制 BS 遍"的场合 —— 本项目 test_BlockMatrix_schur3d 的速度块
    //   A = (6+alpha)*I - sum(邻居) 正是这种 (块值 = M[p][q] * I_3)。
    //   各向异性 / 分量耦合的块 (比如真正的 Stokes 速度-压力耦合块、旋转坐标
    //   下的弹性算子) 检测会失败, 自动回退到通用 BCSR 路径 —— 这是正确性底线。
    //
    //   【数值上不是近似】块是 m*I 时, 标量路径与块路径做的是同一件事:
    //     * 块 Jacobi 修正: D_i^{-1} = (1/m_ii)*I  (blk_invert 对 m*I 精确给出
    //       三个相同的对角元、非对角精确为 0), 与标量版逐位一致;
    //     * Galerkin: (P^T A P) 的块 = (P_s^T M P_s)_{ij} * I, 同样闭合;
    //     * drop 判据: ||m*I||_F = sqrt(BS)*|m|, 相对阈值与 |m| 等价,
    //       所以稀疏化的结构与块版一致。
    //   差异只剩浮点结合顺序 (1e-16 量级), 迭代数应当基本一致。
    //
    //   【向量布局不变】所有层的向量仍是 node-major (bi*BS + c), 标量 spmv 一次
    //   遍历同时累加 BS 个分量 (acc[BS] 在寄存器里), 所以 x 的访存与块版相同
    //   (每行读 BS 个连续 double), 而矩阵值流量降到 1/BS^2。

    // ---- 运行时检测 + 提取: 每个非零块是否都是 m*I_BS ? ----
    //   通过就把标量 m 抽成 CSR 写进 rp/ci/val, 返回 true; 否则返回 false,
    //   调用方走通用 BCSR 路径。
    //   判据是「相对量级」而不是绝对量: 容差 = 1e-12 * (全矩阵对角元的最大绝对值)。
    //   真正的耦合块, 非对角元是对角量级的 O(1) 倍, 再松的容差也盖不住。
    template <typename T, int BS>
    bool blk_amg_scalarize(const BlockMatrix<T, BS>& A,
                           std::vector<uint32_t>& rp,
                           std::vector<uint32_t>& ci,
                           std::vector<T>& val)
    {
        // BS=1 时块路径本身就在做标量运算, 另起一条路径只会白增分支
        if (BS == 1) return false;
        const int BV = BS * BS;
        const std::size_t nnz = A.nnz_blocks();
        if (nnz == 0) return false;

        const T* Av = A.values().data();
        T scale = T(0);
        for (std::size_t p = 0; p < nnz; ++p) {
            const T* b = Av + p * BV;
            for (int r = 0; r < BS; ++r)
                scale = std::max(scale, std::fabs(b[r*BS+r]));
        }
        if (scale == T(0)) return false;
        const T tol = (T)(1e-12) * scale;

        rp = A.block_row_ptr();
        ci = A.block_col_idx();
        val.resize(nnz);
        for (std::size_t p = 0; p < nnz; ++p) {
            const T* b = Av + p * BV;
            T m = T(0);
            for (int r = 0; r < BS; ++r) m += b[r*BS+r];
            m /= (T)BS;
            for (int r = 0; r < BS; ++r) {
                if (std::fabs(b[r*BS+r] - m) > tol) return false;
                for (int c = 0; c < BS; ++c)
                    if (r != c && std::fabs(b[r*BS+c]) > tol) return false;
            }
            val[p] = m;
        }
        return true;
    }

    // ---- 标量 CSR 转置 ----
    template <typename T>
    void blk_amg_scalar_transpose(uint32_t nr, uint32_t nc,
                                  const std::vector<uint32_t>& rp,
                                  const std::vector<uint32_t>& ci,
                                  const std::vector<T>& val,
                                  std::vector<uint32_t>& trp,
                                  std::vector<uint32_t>& tci,
                                  std::vector<T>& tval)
    {
        trp.assign((size_t)nc + 1, 0u);
        for (std::size_t p = 0; p < ci.size(); ++p) trp[ci[p] + 1]++;
        for (uint32_t i = 0; i < nc; ++i) trp[i+1] += trp[i];
        tci.resize(ci.size());
        tval.resize(val.size());
        std::vector<uint32_t> off = trp;
        for (uint32_t i = 0; i < nr; ++i)
            for (uint32_t p = rp[i]; p < rp[i+1]; ++p) {
                const uint32_t q = off[ci[p]]++;
                tci[q]  = i;
                tval[q] = val[p];
            }
    }

    // ---- 标量 SpGEMM + 稀疏化 (Galerkin 用) ----
    //   与 blk_matmul 同样的两个判据 (相对阈值 + 行容量), 只是范数从块
    //   Frobenius 换成 |m| —— 因为 ||m*I||_F = sqrt(BS)*|m|, 相对阈值下等价。
    template <typename T>
    void blk_amg_scalar_matmul(uint32_t nr, uint32_t nc,
                               const std::vector<uint32_t>& a_rp,
                               const std::vector<uint32_t>& a_ci,
                               const std::vector<T>& a_v,
                               const std::vector<uint32_t>& b_rp,
                               const std::vector<uint32_t>& b_ci,
                               const std::vector<T>& b_v,
                               double drop_tol, uint32_t max_row_nnz,
                               std::vector<uint32_t>& c_rp,
                               std::vector<uint32_t>& c_ci,
                               std::vector<T>& c_v)
    {
        std::vector<T> accum((size_t)nc, T(0));
        std::vector<uint32_t> seen((size_t)nc, 0u);   // 时间戳, 避免每行 O(nc) 清零
        uint32_t stamp = 0;
        std::vector<uint32_t> hit;
        std::vector<T> nrm, srt;

        c_rp.assign((size_t)nr + 1, 0u);
        c_ci.clear();
        c_v.clear();

        for (uint32_t i = 0; i < nr; ++i) {
            hit.clear();
            if (++stamp == 0) { std::fill(seen.begin(), seen.end(), 0u); stamp = 1; }

            for (uint32_t pa = a_rp[i]; pa < a_rp[i+1]; ++pa) {
                const uint32_t k = a_ci[pa];
                const T av = a_v[pa];
                for (uint32_t pb = b_rp[k]; pb < b_rp[k+1]; ++pb) {
                    const uint32_t j = b_ci[pb];
                    if (seen[j] != stamp) { seen[j] = stamp; hit.push_back(j); accum[j] = T(0); }
                    accum[j] += av * b_v[pb];
                }
            }
            std::sort(hit.begin(), hit.end());

            if ((drop_tol > 0.0 || max_row_nnz > 0) && hit.size() > 1) {
                const std::size_t nz = hit.size();
                nrm.resize(nz);
                T maxn = T(0);
                for (std::size_t q = 0; q < nz; ++q) {
                    nrm[q] = std::fabs(accum[hit[q]]);
                    if (nrm[q] > maxn) maxn = nrm[q];
                }
                const T relthr = (drop_tol > 0.0) ? (T)(drop_tol * (double)maxn) : T(0);
                T capthr = T(0);
                if (max_row_nnz > 0 && nz > max_row_nnz) {
                    srt.assign(nrm.begin(), nrm.end());
                    std::nth_element(srt.begin(),
                                     srt.begin() + (std::ptrdiff_t)(max_row_nnz - 1),
                                     srt.end(), [](T x, T y) { return x > y; });
                    capthr = srt[(std::size_t)(max_row_nnz - 1)];
                }
                const T thr = (relthr > capthr) ? relthr : capthr;
                std::size_t keep = 0;
                for (std::size_t q = 0; q < nz; ++q)
                    if (hit[q] == i || nrm[q] >= thr) hit[keep++] = hit[q];
                hit.resize(keep);
            }

            c_rp[i+1] = (uint32_t)hit.size();
            for (uint32_t j : hit) { c_ci.push_back(j); c_v.push_back(accum[j]); }
        }
        for (uint32_t i = 0; i < nr; ++i) c_rp[i+1] += c_rp[i];
    }

    // ---- 标量 -> 块矩阵的膨胀 (m -> m*I_BS) ----
    //   标量 Galerkin 得到的粗算子要变回 BlockMatrix, 下一层才能继续做
    //   m*I 检测 —— 检测不过就自动回退到通用 BCSR 路径。
    template <typename T, int BS>
    BlockMatrix<T, BS> blk_amg_scalar_expand(const std::vector<uint32_t>& rp,
                                             const std::vector<uint32_t>& ci,
                                             const std::vector<T>& val,
                                             uint32_t n)
    {
        const std::size_t nnz = ci.size();
        std::vector<T> vv(nnz * (size_t)(BS * BS), T(0));
        for (std::size_t p = 0; p < nnz; ++p)
            for (int r = 0; r < BS; ++r)
                vv[p * (size_t)(BS * BS) + (size_t)r * BS + r] = val[p];
        return BlockMatrix<T, BS>(n, n, rp, ci, std::move(vv));
    }

    // ---- 标量 CSR x node-major 多分量向量 ----
    //   一次遍历同时累加 BS 个分量: acc[BS] 留在寄存器里, 于是
    //   * 矩阵值流量 = nnz * 1   (块版是 nnz * BS^2)
    //   * x 的流量   = 每行 BS 个连续 double (与块版完全相同)
    //   * flops      = nnz * BS  (块版是 nnz * BS^2)
    template <typename T, int BS>
    void blk_amg_scalar_matvec(const std::vector<uint32_t>& rp,
                               const std::vector<uint32_t>& ci,
                               const std::vector<T>& vv, uint32_t nr,
                               const std::vector<T>& x, std::vector<T>& y)
    {
        const std::ptrdiff_t pn = (std::ptrdiff_t)nr;
        y.assign((size_t)nr * BS, T(0));
        #ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static) if(pn > 1024)
        #endif
        for (std::ptrdiff_t i = 0; i < pn; ++i) {
            T acc[BS];
            for (int c = 0; c < BS; ++c) acc[c] = T(0);
            for (uint32_t p = rp[(size_t)i]; p < rp[(size_t)i+1]; ++p) {
                const T m  = vv[p];
                const T* xj = &x[(size_t)ci[p] * BS];
                #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp simd
                #endif
                for (int c = 0; c < BS; ++c) acc[c] += m * xj[c];
            }
            T* yi = &y[(size_t)i * BS];
            for (int c = 0; c < BS; ++c) yi[c] = acc[c];
        }
    }

    // y += A x  (延拓修正用; 就地累加, 不额外过一遍 ndof 长的缓冲)
    template <typename T, int BS>
    void blk_amg_scalar_matvec_add(const std::vector<uint32_t>& rp,
                                   const std::vector<uint32_t>& ci,
                                   const std::vector<T>& vv, uint32_t nr,
                                   const std::vector<T>& x, std::vector<T>& y)
    {
        const std::ptrdiff_t pn = (std::ptrdiff_t)nr;
        #ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static) if(pn > 1024)
        #endif
        for (std::ptrdiff_t i = 0; i < pn; ++i) {
            T acc[BS];
            for (int c = 0; c < BS; ++c) acc[c] = T(0);
            for (uint32_t p = rp[(size_t)i]; p < rp[(size_t)i+1]; ++p) {
                const T m  = vv[p];
                const T* xj = &x[(size_t)ci[p] * BS];
                #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp simd
                #endif
                for (int c = 0; c < BS; ++c) acc[c] += m * xj[c];
            }
            T* yi = &y[(size_t)i * BS];
            for (int c = 0; c < BS; ++c) yi[c] += acc[c];
        }
    }

    // ---- 最粗层: 只分解 n_c x n_c 的 M (不是 (n_c*BS)^2 的块矩阵) ----
    //   主元下限同样取「相对本层量级」, 与块版 blk_amg_factor_coarse 一致。
    //   【后续项】分解便宜了 BS^3 倍之后, coarse_max 其实有了往上调的余量
    //     (现在 BS=3 只放到 50 块 = 150 自由度, 是按块版 LU 的开销定的)。
    //     放粗层意味着 V-cycle 更准、迭代数更少 —— 但那会改变收敛行为,
    //     要和 n_smooth 一样单独做一轮实测, 别顺手改。
    template <typename T, int BS>
    void blk_amg_factor_coarse_scalar(BlockAMGLevel<T, BS>& L)
    {
        const uint32_t n  = L.nrows;
        const int      sz = (int)n;

        std::vector<T>& dense = L.c_lu_s;
        dense.assign((size_t)sz * sz, T(0));
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t p = L.s_rp[i]; p < L.s_rp[i+1]; ++p)
                dense[(size_t)i * sz + L.s_ci[p]] = L.s_val[p];

        double dscale = 0.0;
        for (size_t q = 0; q < dense.size(); ++q)
            if (std::fabs(dense[q]) > dscale) dscale = std::fabs(dense[q]);
        if (dscale == 0.0) dscale = 1.0;
        const T dmin = (T)(1e-13 * dscale);
        bool hit_floor = false;

        L.c_piv_s.assign((size_t)sz, 0);
        for (int k = 0; k < sz; ++k) {
            int pk = k;
            for (int i = k+1; i < sz; ++i)
                if (std::fabs(dense[(size_t)i*sz+k]) > std::fabs(dense[(size_t)pk*sz+k])) pk = i;
            L.c_piv_s[(size_t)k] = pk;
            if (pk != k)
                for (int j = 0; j < sz; ++j)
                    std::swap(dense[(size_t)k*sz+j], dense[(size_t)pk*sz+j]);
            T d = dense[(size_t)k*sz+k];
            if (std::fabs(d) < dmin)
            { d = (d >= 0 ? dmin : -dmin); hit_floor = true; }
            dense[(size_t)k*sz+k] = d;
            for (int i = k+1; i < sz; ++i) {
                T f = dense[(size_t)i*sz+k] / d;
                for (int j = k; j < sz; ++j)
                    dense[(size_t)i*sz+j] -= f * dense[(size_t)k*sz+j];
            }
        }
        L.c_pinv_s  = hit_floor;
        L.c_ready_s = true;
    }

    // ---- 标量路径的 V-cycle ----
    //   与通用版结构一一对应, 三处不同:
    //     1. 光滑: 一次遍历算完 BS 个分量并就地修正, 不落 tmp/r 缓冲
    //        (块版要过 tmp 和 r 两个 ndof 长的向量, 每次 sweep 多 4 趟 ndof 流量);
    //     2. 延拓: sol += P e_c 就地累加, 不需要 w_pec;
    //     3. 最粗层: 对 M 做一次 LU, 三个分量共用, 回代按分量展开。
    template <typename T, int BS>
    void blk_amg_vcycle_scalar(std::vector<BlockAMGLevel<T, BS>>& levels,
                               int lvl, int n_smooth, T omega,
                               const std::vector<T>& rhs, std::vector<T>& sol)
    {
        auto& L = levels[(size_t)lvl];
        const uint32_t n    = L.nrows;
        const uint32_t ndof = n * BS;
        const std::ptrdiff_t pn = (std::ptrdiff_t)n;

        // V-cycle 从 x0 = 0 起步: 深层缓冲是复用的, 必须显式清零
        sol.assign(ndof, T(0));

        // ======== 最粗层: 标量 LU (setup 时已分解好) ========
        if (lvl == (int)levels.size() - 1) {
            if (!L.c_ready_s) blk_amg_factor_coarse_scalar(L);
            const int sz = (int)n;
            const std::vector<T>& lu = L.c_lu_s;

            // sol 在入口被清零；粗层 LU 必须先装入 rhs，否则永返回零修正。
            for (uint32_t i = 0; i < ndof; ++i) sol[(size_t)i] = rhs[(size_t)i];

            // 分解时的行交换必须同步作用在右端项上 —— 三个分量各自换
            for (int k = 0; k < sz; ++k) {
                const int pk = L.c_piv_s[(size_t)k];
                if (pk != k)
                    for (int c = 0; c < BS; ++c)
                        std::swap(sol[(size_t)k*BS+c], sol[(size_t)pk*BS+c]);
            }
            // 前代 L y = P b
            for (int k = 0; k < sz; ++k) {
                const T d = lu[(size_t)k * sz + k];
                for (int i = k + 1; i < sz; ++i) {
                    const T f = lu[(size_t)i * sz + k] / d;
                    if (f == T(0)) continue;
                    for (int c = 0; c < BS; ++c)
                        sol[(size_t)i*BS+c] -= f * sol[(size_t)k*BS+c];
                }
            }
            // 回代 U x = y
            for (int i = sz - 1; i >= 0; --i) {
                for (int j = i + 1; j < sz; ++j) {
                    const T a = lu[(size_t)i * sz + j];
                    if (a == T(0)) continue;
                    for (int c = 0; c < BS; ++c)
                        sol[(size_t)i*BS+c] -= a * sol[(size_t)j*BS+c];
                }
                const T d = lu[(size_t)i * sz + i];
                for (int c = 0; c < BS; ++c) sol[(size_t)i*BS+c] /= d;
            }
            if (L.c_pinv_s) {
                // 撞过主元下限 => 常数向量在零空间里。块版是整体去一次均值,
                // 标量版按分量各去一次 —— 这才是 M (x) I_BS 真正的伪逆
                // (零空间是 span{1 (x) e_c}, 共 BS 维)。
                for (int c = 0; c < BS; ++c) {
                    T m = 0;
                    for (int i = 0; i < sz; ++i) m += sol[(size_t)i*BS+c];
                    m /= (T)sz;
                    for (int i = 0; i < sz; ++i) sol[(size_t)i*BS+c] -= m;
                }
            }
            return;
        }

        std::vector<T>& r = L.w_r; if (r.size() != ndof) r.resize(ndof);

        const std::vector<uint32_t>& rp = L.s_rp;
        const std::vector<uint32_t>& ci = L.s_ci;
        const std::vector<T>&       vv  = L.s_val;

        // ======== 光滑: 阻尼 Jacobi ========
        //   acc[BS] 一次累加完, 直接用 rhs - A x 修正, 不落中间向量。
        auto jacobi_sweep = [&]() {
            #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(pn > 1024)
            #endif
            for (std::ptrdiff_t i = 0; i < pn; ++i) {
                T acc[BS];
                for (int c = 0; c < BS; ++c) acc[c] = T(0);
                for (uint32_t p = rp[(size_t)i]; p < rp[(size_t)i+1]; ++p) {
                    const T m  = vv[p];
                    const T* xj = &sol[(size_t)ci[p] * BS];
                    #ifdef CPPNUM_HAVE_OPENMP
                    #pragma omp simd
                    #endif
                    for (int c = 0; c < BS; ++c) acc[c] += m * xj[c];
                }
                const T ow  = omega * L.s_diag_inv[(size_t)i];
                const T* rb = &rhs[(size_t)i * BS];
                T* si = &sol[(size_t)i * BS];
                for (int c = 0; c < BS; ++c) si[c] += ow * (rb[c] - acc[c]);
            }
        };

        for (int s = 0; s < n_smooth; ++s) jacobi_sweep();      // 预光滑

        // ======== 残差 -> 限制 -> 递归 -> 延拓修正 ========
        #ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static) if(pn > 1024)
        #endif
        for (std::ptrdiff_t i = 0; i < pn; ++i) {
            T acc[BS];
            for (int c = 0; c < BS; ++c) acc[c] = T(0);
            for (uint32_t p = rp[(size_t)i]; p < rp[(size_t)i+1]; ++p) {
                const T m  = vv[p];
                const T* xj = &sol[(size_t)ci[p] * BS];
                #ifdef CPPNUM_HAVE_OPENMP
                #pragma omp simd
                #endif
                for (int c = 0; c < BS; ++c) acc[c] += m * xj[c];
            }
            const T* rb = &rhs[(size_t)i * BS];
            T* ri = &r[(size_t)i * BS];
            for (int c = 0; c < BS; ++c) ri[c] = rb[c] - acc[c];
        }

        std::vector<T>& rc = L.w_rc;                       // r_c = R r
        blk_amg_scalar_matvec<T, BS>(L.r_rp, L.r_ci, L.r_val,
                                     levels[(size_t)(lvl + 1)].nrows, r, rc);

        std::vector<T>& ec = levels[(size_t)(lvl + 1)].w_sol;
        blk_amg_vcycle(levels, lvl + 1, n_smooth, omega, rc, ec);

        blk_amg_scalar_matvec_add<T, BS>(L.p_rp, L.p_ci, L.p_val, n, ec, sol);

        for (int s = 0; s < n_smooth; ++s) jacobi_sweep();      // 后光滑
    }

    // ---- V-cycle 递归 ----
    //   【性能】AMG 值不值, 取决于「迭代数降的倍数」÷「单次 apply 涨的倍数」。
    //   实测 BAMG 的单价一度是 ILU(0) 的 2.1~2.7 倍, 迭代数只降 1.0~3.1 倍,
    //   于是 alpha>=1 时净亏 1.9~6.5 倍。下面三条都是冲着单价去的硬要求:
    //     1. 每层的工作缓冲挂在 BlockAMGLevel 上复用 —— 原来每次 apply 在每层
    //        malloc 4~5 个 ndof 长的 vector, level-0 就是 786KB x 5, 一次
    //        GMRES(22 步 x 2 个 V-cycle) 要反复 malloc/free 上百 MB;
    //     2. 光滑器里的向量循环全部 OpenMP 并行 —— 原来只有 spmv 并行, 残差 /
    //        块 Jacobi 修正 / 延拓修正都是串行, 8 线程下这是 Amdahl 主项
    //        (level-0 约 25~30 MB 串行访存 / V-cycle);
    //     3. 最粗层走 setup 时缓存好的稠密 LU (见 blk_amg_factor_coarse),
    //        不再每个 V-cycle 重算一遍 132^3/3 的串行分解。
    //   数值上与改动前完全等价 —— 可以用来验证: 迭代数应当一模一样。
    template <typename T, int BS>
    void blk_amg_vcycle(std::vector<BlockAMGLevel<T, BS>>& levels,
                        int lvl, int n_smooth, T omega,
                        const std::vector<T>& rhs, std::vector<T>& sol)
    {
        // ---- 路径分派 ----
        //   该层的块都是 m*I_BS 就走标量快速路径 (见 blk_amg_scalarize 注释)。
        //   层次可以混着走: 父层用标量限制/延拓, 子层是通用的就递归回这里 ——
        //   两种路径的向量布局都是 node-major, 接口兼容。
        if (levels[(size_t)lvl].scalar) {
            blk_amg_vcycle_scalar(levels, lvl, n_smooth, omega, rhs, sol);
            return;
        }

        const int BV = BS * BS;
        auto& L = levels[lvl];
        const uint32_t n    = L.nrows;
        const uint32_t ndof = n * BS;
        const std::ptrdiff_t pn  = (std::ptrdiff_t)n;
        const std::ptrdiff_t pnd = (std::ptrdiff_t)ndof;

        // V-cycle 从 x0 = 0 起步: 深层缓冲是复用的, 必须显式清零
        sol.assign(ndof, T(0));

        // ======== 最粗层: 稠密 LU (setup 时已分解好) ========
        if (lvl == (int)levels.size() - 1) {
            const int sz = (int)ndof;
            if (!L.c_ready) blk_amg_factor_coarse(L);
            for (int i = 0; i < sz; ++i) sol[(size_t)i] = rhs[(size_t)i];
            // 分解时的行交换必须同步作用在右端项上。原来漏了这一步, 只是靠
            // "粗算子 SPD / 对角占优, 实际不会换主元" 才没暴露出来。
            for (int k = 0; k < sz; ++k) {
                const int pk = L.c_piv[(size_t)k];
                if (pk != k) std::swap(sol[(size_t)k], sol[(size_t)pk]);
            }
            // 前代 L y = P b     (L 单位下三角, 乘数 l_ik = c_lu[i][k]/c_lu[k][k])
            for (int k = 0; k < sz; ++k) {
                const T d = L.c_lu[(size_t)k * sz + k];
                for (int i = k + 1; i < sz; ++i)
                    sol[(size_t)i] -= (L.c_lu[(size_t)i * sz + k] / d) * sol[(size_t)k];
            }
            // 回代 U x = y
            for (int i = sz - 1; i >= 0; --i) {
                for (int j = i + 1; j < sz; ++j)
                    sol[(size_t)i] -= L.c_lu[(size_t)i * sz + j] * sol[(size_t)j];
                sol[(size_t)i] /= L.c_lu[(size_t)i * sz + i];
            }
            if (L.c_pinv) {
                // 撞过主元下限 => 该方向是零空间, 用伪逆: 去掉常数分量
                T m = 0;
                for (int i = 0; i < sz; ++i) m += sol[(size_t)i];
                m /= (T)sz;
                for (int i = 0; i < sz; ++i) sol[(size_t)i] -= m;
            }
            return;
        }

        std::vector<T>& tmp = L.w_tmp; if (tmp.size() != ndof) tmp.resize(ndof);
        std::vector<T>& r   = L.w_r;   if (r.size()   != ndof) r.resize(ndof);

        // ======== 光滑: 阻尼块 Jacobi (预 / 后各 n_smooth 次) ========
        auto jacobi_sweep = [&]() {
            L.A.matvec(sol, tmp);
#ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(pnd > 4096)
#endif
            for (std::ptrdiff_t i = 0; i < pnd; ++i) r[i] = rhs[i] - tmp[i];
#ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for schedule(static) if(pn > 1024)
#endif
            for (std::ptrdiff_t bi = 0; bi < pn; ++bi) {
                const T* dinv = &L.diag_inv[(size_t)bi * BV];
                for (int c = 0; c < BS; ++c) {
                    T corr = 0;
                    for (int rr = 0; rr < BS; ++rr)
                        corr += dinv[c*BS+rr] * r[(size_t)bi*BS+rr];
                    sol[(size_t)bi*BS+c] += omega * corr;
                }
            }
        };

        for (int s = 0; s < n_smooth; ++s) jacobi_sweep();      // 预光滑

        // ======== 残差 -> 限制 -> 递归 -> 延拓修正 ========
        L.A.matvec(sol, tmp);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static) if(pnd > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < pnd; ++i) r[i] = rhs[i] - tmp[i];

        std::vector<T>& rc = L.w_rc;                       // r_c = R r
        L.R.matvec(r, rc);

        std::vector<T>& ec = levels[lvl + 1].w_sol;        // 子层的解写进它自己的缓冲
        blk_amg_vcycle(levels, lvl + 1, n_smooth, omega, rc, ec);

        std::vector<T>& pec = L.w_pec;                     // P e_c
        L.P.matvec(ec, pec);
#ifdef CPPNUM_HAVE_OPENMP
        #pragma omp parallel for schedule(static) if(pnd > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < pnd; ++i) sol[i] += pec[i];

        for (int s = 0; s < n_smooth; ++s) jacobi_sweep();      // 后光滑
    }

    // =====================================================================
    // 主函数: 创建 Block AMG 预处理器
    // =====================================================================
    template <typename T, int BS>
    std::function<void(const std::vector<T>&, std::vector<T>&)>
    // max_agg: 单个聚合最多吸收多少个块行。
    //   三维 7 点模板下每个节点只有 6 个强邻居, 所以 max_agg >= 7 才能凑出
    //   完整的 2x2x2 团块; 默认取 8 (留一格余量)。
    //   注意: 粗化比主要由「两遍聚合」保证 (见下面第 2 步), 而不是靠放大
    //   max_agg —— 单遍贪心即使 max_agg=8 在三维下也只做到 2.00。
    // drop_tol / max_row_nnz: 原样传给 blk_matmul, 用于 Galerkin 稀疏化
    //   (见 blk_matmul 的注释)。默认值是光滑聚合的经验配置。
    // n_smooth: 前 / 后光滑各做几次, 默认 2 (光滑聚合的标准配置)。
    //   它同时控制两件事, 而且方向相反 —— 这是 AMG 调参的核心权衡:
    //     单价   每个 V-cycle 在*每一层*要做 2*n_smooth + 1 次 matvec
    //            (预光滑 n 次 + 粗化残差 1 次 + 后光滑 n 次), 所以
    //            n_smooth 2->1 直接把每层的 matvec 从 5 次降到 3 次;
    //     收敛   光滑越弱, V-cycle 收敛因子越差, GMRES 迭代数上升。
    //   1 是下限 (0 会让 V-cycle 退化成纯粗网格校正, 高频误差完全压不掉)。
    //
    //   【实测结论: n_smooth 2->1 是负收益, 默认保持 2】
    //     (N=16/24/32 x alpha=0.1/1/10, 9 例, best-of-3, 两次重复取平均)
    //     单价    BAMG/(ILU0+AMG) 从 1.78 降到 1.49 (-16%)。方向对, 但幅度只有
    //             16% 而不是"省掉 2/5 matvec"对应的 40% —— 因为每次迭代里还有
    //             约 45% 的固定成本 (K 的 spmv + Arnoldi 正交化) 不随 n_smooth 变。
    //     迭代数  BAMG 反而涨 14%~46% (N=32: alpha=0.1 22->25, alpha=1 17->21,
    //             alpha=10 14->18), 涨得比单价降得多。
    //     净收益  alpha=0.1 持平 (1.73->1.63), alpha=1 变差 (0.84->0.75),
    //             alpha=10 持平 (0.59->0.59)。九例无一变好。
    //   体检指标其实提前预警了: 残差因子 0.033 -> 0.107 (差 3.2 倍), 在 GMRES
    //   开跑前就把"迭代数要涨"说出来了 —— 以后调这类参数先看它, 能省一轮实测。
    // allow_scalar: 允许走标量快速路径 (默认开)。设 false 可强制回退到通用
    //   BCSR 路径, 用来做 A/B 对照: 两条路径的迭代数应当一致, 差的是单价。
    //   检测是逐层做的 —— 一旦某层的块不再是 m*I_BS, 就永久切回块路径。
    make_block_amg_preconditioner(const BlockMatrix<T, BS>& A, int max_agg = 8,
                                  bool verbose = false,
                                  double drop_tol = 0.02, uint32_t max_row_nnz = 40,
                                  int n_smooth = 2, bool allow_scalar = true)
    {
        if (A.block_rows() != A.block_cols())
            throw std::invalid_argument("Block AMG 要求方阵");
        if (n_smooth < 1) n_smooth = 1;
        // 标量快速路径只对 BS>1 有意义 (见 blk_amg_scalarize)
        bool try_scalar = allow_scalar && (BS > 1);

        const int BV = BS * BS;
        const T theta        = 0.25;            // 强度连接阈值
        const T omega_smooth = T(2) / T(3);     // Jacobi 阻尼
        // n_smooth 见参数表处的注释: 2*n_smooth+1 次 matvec/层 是单价的主项,
        //   而光滑强度又决定 V-cycle 收敛因子, 两者反向。
        const int max_levels = 20;              // 最大层数
        // 最粗层用稠密 LU 直接解, 开销 O((n*BS)^3) 且是串行的 —— 所以上限必须
        //   按「标量自由度」而不是块数来定。BS=1 保持原来调好的 100 个自由度;
        //   BS=3 换算成 50 个块 = 150 个自由度, 否则 300^3/3 = 9e6 flop 的串行
        //   LU 会占满每一个 V-cycle。
        const uint32_t coarse_max = (BS == 1) ? 100u : (uint32_t)(150u / (uint32_t)BS);
        // operator complexity 闸门: 当前层算子的 nnz 一旦超过 finest 的
        //   2.0 倍, 再粗化下去只会让模板继续变密、粗层稠密 LU 更病态,
        //   直接停下来把当前层当最粗层解掉。
        //   (有了 Galerkin 稀疏化之后这个闸门基本不会触发, 只作安全网。)
        const double max_complexity = 2.0;
        const double nnz0 = (A.nnz_blocks() > 0) ? (double)A.nnz_blocks() : 1.0;

        auto lv_ptr = std::make_shared<std::vector<BlockAMGLevel<T, BS>>>();
        auto& levels = *lv_ptr;
        BlockMatrix<T, BS> cur_A = A;

        for (int lvl = 0; lvl < max_levels; ++lvl) {
            uint32_t n = cur_A.block_rows();

            BlockAMGLevel<T, BS> level;
            level.nrows = n;

            // ---- 运行时路径选择 (必须在分层诊断之前) ----
            //   块全是 m*I_BS 就走标量快速路径; 否则回退通用 BCSR。
            //   一旦回退就永久关掉 (后面的粗算子由块 Galerkin 产生, 不会再
            //   突然变回各向同性), 省掉每层的重复扫描。
            const bool sc = try_scalar &&
                blk_amg_scalarize(cur_A, level.s_rp, level.s_ci, level.s_val);
            level.scalar = sc;
            if (!sc) try_scalar = false;
            // 标量路径只认 s_*, 块算子压根不会读 —— 不拷贝, 省 BS^2 倍内存。
            if (!sc) level.A = cur_A;

            // ---- 分层诊断 ----
            // minRowSum: 块行和的最小绝对值。对 Laplace 类算子它近似等于
            //   lambda_min 的量级 (行和 = 对角 + 非对角 = 正则化量)。
            //   如果它掉到机器精度, 说明粗算子已经数值奇异, 最粗层那个
            //   50x50 稠密 LU 就会撞上 1e-14 的 pivot clamp -> 1e14 倍放大。
            if (verbose)
            {
                double minrs = 0.0; bool first = true;
                for (uint32_t i = 0; i < n; ++i)
                {
                    double rs = 0.0;
                    if (sc) {
                        // 块行和 = 块内全部元素之和 = BS * sum_j m_ij
                        for (uint32_t p = level.s_rp[i]; p < level.s_rp[i+1]; ++p)
                            rs += (double)level.s_val[p];
                        rs *= (double)BS;
                    } else {
                        for (uint32_t p = cur_A.block_row_ptr()[i]; p < cur_A.block_row_ptr()[i+1]; ++p)
                        {
                            const T* b = &cur_A.values()[(size_t)p * BV];
                            for (int r = 0; r < BS; ++r)
                                for (int c = 0; c < BS; ++c) rs += b[r*BS+c];
                        }
                    }
                    const double a = std::fabs(rs);
                    if (first || a < minrs) { minrs = a; first = false; }
                }
                const double nnzpr = (n > 0) ? (double)cur_A.nnz_blocks() / (double)n : 0.0;
                std::printf("[BlockAMG] level %2d: n=%8u nnz/row=%7.1f minRowSum=%.3e%s\n",
                            lvl, n, nnzpr, minrs,
                            (lvl == (int)levels.size() && n <= coarse_max) ? "  (最粗层)" : "");
                std::fflush(stdout);
            }

            // 对角逆: 标量路径只要 1/m_ii, 块路径才要 BS x BS 的逆
            //   (sc 已在上面判定)
            if (sc) {
                level.s_diag_inv.assign(n, T(0));
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t p = level.s_rp[i]; p < level.s_rp[i+1]; ++p)
                        if (level.s_ci[p] == i)
                        { level.s_diag_inv[i] = T(1) / level.s_val[p]; break; }
            } else {
                level.diag_inv.resize((size_t)n * BV);
                for (uint32_t i = 0; i < n; ++i)
                    blk_invert<BS>(cur_A.diag_block(i), &level.diag_inv[(size_t)i * BV]);
            }

            // 停止粗化的两个条件: (a) 已经足够小; (b) 粗算子 nnz 膨胀到
            //   finest 的 1.6 倍以上 (模板正在变密, 再粗化是负收益)。
            if (n <= coarse_max ||
                (double)cur_A.nnz_blocks() > max_complexity * nnz0)
            { levels.push_back(std::move(level)); break; }

            // ---- 1. 强度连接 (Frobenius 范数) ----
            std::vector<std::vector<uint32_t>> strong(n);
            if (sc) {
                // ||m*I||_F = sqrt(BS)*|m|, 相对阈值下与 |m| 等价 —— 结果一致
                for (uint32_t i = 0; i < n; ++i) {
                    T mx = 0;
                    for (uint32_t p = level.s_rp[i]; p < level.s_rp[i+1]; ++p)
                        if (level.s_ci[p] != i)
                            mx = std::max(mx, std::fabs(level.s_val[p]));
                    // 【mx>0 闸门】对称 Dirichlet 消元是"置 0 但保留结构":
                    //   固定行的非对角全是 0.0, 于是 mx = 0 -> thr = 0 ->
                    //   "|0| >= 0" 把**全部 0 值邻居**都判成强连接, 聚合被毒化。
                    //   这里让 mx=0 的行没有强连接 (当孤立根往下传),
                    //   与标量 AMG (array.cpp) 的修法保持一致。
                    if (mx > T(0)) {
                        const T thr = theta * mx;
                        for (uint32_t p = level.s_rp[i]; p < level.s_rp[i+1]; ++p)
                            if (level.s_ci[p] != i && std::fabs(level.s_val[p]) >= thr)
                                strong[i].push_back(level.s_ci[p]);
                    }
                }
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    T mx = 0;
                    for (uint32_t p = cur_A.block_row_ptr()[i]; p < cur_A.block_row_ptr()[i+1]; ++p) {
                        uint32_t j = cur_A.block_col_idx()[p];
                        if (j != i)
                            mx = std::max(mx, blk_fro_norm<T, BS>(&cur_A.values()[(size_t)p*BV]));
                    }
                    // 同上: mx=0 的行 (Dirichlet 置 0 保结构) 不给任何强连接。
                    if (mx > T(0)) {
                        T thr = theta * mx;
                        for (uint32_t p = cur_A.block_row_ptr()[i]; p < cur_A.block_row_ptr()[i+1]; ++p) {
                            uint32_t j = cur_A.block_col_idx()[p];
                            if (j != i && blk_fro_norm<T, BS>(&cur_A.values()[(size_t)p*BV]) >= thr)
                                strong[i].push_back(j);
                        }
                    }
                }
            }

            // ---- 2. 两遍聚合 (Mandel-Brezina 风格) ----
            //   原来的"单遍贪心"在结构化三维网格上会退化: 种子 i 一次把
            //   +/-x, +/-y, +/-z 六个方向的邻居全抓走, 后面轮到的节点只剩
            //   一个空闲邻居可抓 —— 16^3 上实测 1664 个聚合大小=2、
            //   256 个大小=1, 粗化比只有 2.00 (N=24/32 同样是精确的一半)。
            //
            //   粗化比 2 是后面所有灾难的源头:
            //     * Galerkin 后粗算子模板每级膨胀 7 -> 66 -> 292 -> 稠密
            //     * operator complexity 爆炸, setup 时间按 n^2 涨
            //     * 粗算子被逐级放大 (minRowSum 1e-6 -> 1e54)
            //     * 最粗层稠密 LU 撞上近奇异主元 -> 1e14 倍放大 -> GMRES 发散
            //
            //   两遍法修的正是这一点:
            //     Pass1 只有 i 的强邻居「全部」无归属时才成团 —— 保证团是
            //           紧凑的局部块 (三维 7 点模板下近似 2x2x2);
            //     Pass2 把 Pass1 剩下的点并入「还有空位」的相邻团,
            //           没有空位就自己新开一个团。
            //   实测 16^3 / 32^3 粗化比 = 7.0 / 7.1, 聚合大小 1..8。
            std::vector<int> agg(n, -1);
            std::vector<int> agg_sz;                 // 每个团的当前大小
            int n_agg = 0;

            // 【孤立块行(Dirichlet 单位行)必须先标记丢弃】
            //   strong[i] 为空 -> Pass1 会把它做成 singleton 团, 而 singleton
            //   团**永不缩减**, 于是 n_agg 的下界被钉在边界块行数上。停止条件
            //   n <= coarse_max (BS=2 时 = 75) 在 nx>=16 (边界 128) 时永远不可能
            //   满足 -> 层级一路堆到 max_levels=20, 粗算子逐级退化 -> GMRES 停滞。
            //   实测失败位置与 64<=75<128 的分界完全吻合 (nx=8 PASS, 16/32 FAIL)。
            //   标记 agg=-2: 不建团、不进 P_t、不进粗算子。它自己的方程
            //   x_i = b_i 由光滑子一次阻尼 Jacobi 解掉, 无需粗层表示。
            uint32_t n_iso = 0;
            for (uint32_t i = 0; i < n; ++i)
                if (strong[i].empty()) { agg[i] = -2; ++n_iso; }

            // Pass 1: 只吸收邻居全空闲的节点, 得到紧凑团块
            for (uint32_t i = 0; i < n; ++i) {
                if (agg[i] != -1) continue;
                bool all_free = true;
                for (uint32_t j : strong[i])
                    if (agg[j] != -1) { all_free = false; break; }
                if (!all_free) continue;              // 交给 Pass 2
                agg[i] = n_agg;
                int sz = 1;
                for (uint32_t j : strong[i])
                    if (agg[j] == -1 && sz < max_agg) { agg[j] = n_agg; ++sz; }
                agg_sz.push_back(sz);
                ++n_agg;
            }
            // Pass 2: 剩余点并入还有空位的相邻团; 实在没有就自己开一个
            for (uint32_t i = 0; i < n; ++i) {
                if (agg[i] != -1) continue;
                int best = -1;
                for (uint32_t j : strong[i]) {
                    const int g = agg[j];
                    // g >= 0 (不能写 g != -1): 丢弃行是 -2, 会索引 agg_sz[巨数]
                    if (g >= 0 && agg_sz[(size_t)g] < max_agg) { best = g; break; }
                }
                if (best >= 0) { agg[i] = best; agg_sz[(size_t)best] += 1; continue; }
                agg[i] = n_agg;
                int sz = 1;
                for (uint32_t j : strong[i])
                    if (agg[j] == -1 && sz < max_agg) { agg[j] = n_agg; ++sz; }
                agg_sz.push_back(sz);
                ++n_agg;
            }

            if (verbose)
            {
                std::printf("[BlockAMG]          -> 聚合数 = %u (粗化比 %.2f)\n",
                            (uint32_t)n_agg, (n_agg > 0) ? (double)n / (double)n_agg : 0.0);
                if (n_iso > 0)
                    std::printf("[BlockAMG]          -> 丢弃孤立块行 %u (不占粗层自由度)\n",
                                n_iso);
                std::fflush(stdout);
            }

            // ---- 3+4. 光滑延拓 P = (I − ω D⁻¹ A) P_t ----
            //   块是 m*I 时, D_i⁻¹ A[i][k] = (m_ik / m_ii) * I —— P 的块同样是
            //   标量乘 I, 所以 P 只要存标量 CSR, 块结构由 I 隐含。
            if (sc) {
                // 每行一个临时 map (列升序) -> 行末搬进 vector, 与块版同样的
                //   内存节奏: 不同时持有 n 个 map 对象。
                std::vector<std::vector<std::pair<uint32_t, T>>> p_rows(n);
                for (uint32_t i = 0; i < n; ++i) {
                    if (agg[i] < 0) continue;             // 孤立块行: P 该行为空
                    std::map<uint32_t, T> row;
                    row[(uint32_t)agg[i]] = T(1);                 // 恒等项
                    const T inv = level.s_diag_inv[i];            // 1/m_ii
                    for (uint32_t p = level.s_rp[i]; p < level.s_rp[i+1]; ++p) {
                        // 被丢弃的孤立块行值全为 0, 但仍留在 CSR 结构里 ——
                        //   必须跳过, 否则 (uint32_t)(-2) 会变成巨大列号写进 p_ci。
                        const int ga = agg[level.s_ci[p]];
                        if (ga < 0) continue;
                        const uint32_t g = (uint32_t)ga;
                        auto it = row.find(g);
                        if (it == row.end()) it = row.emplace(g, T(0)).first;
                        it->second -= omega_smooth * inv * level.s_val[p];
                    }
                    p_rows[i].assign(row.begin(), row.end());
                }
                level.p_rp.assign(n + 1, 0);
                level.p_ci.clear();
                level.p_val.clear();
                for (uint32_t i = 0; i < n; ++i) {
                    level.p_rp[i+1] = (uint32_t)p_rows[i].size();
                    for (auto& kv : p_rows[i]) {
                        level.p_ci.push_back(kv.first);
                        level.p_val.push_back(kv.second);
                    }
                }
                for (uint32_t i = 0; i < n; ++i) level.p_rp[i+1] += level.p_rp[i];
                blk_amg_scalar_transpose(n, (uint32_t)n_agg,
                                         level.p_rp, level.p_ci, level.p_val,
                                         level.r_rp, level.r_ci, level.r_val);
                // 块版的 A/P/R 留空: 标量路径用不到它们
                levels.push_back(std::move(level));
            } else {
                // ---- 3. 分片常数延拓 P_t:  P_t[i][agg[i]] = I ----
                // 孤立块行不占列 -> 行长可为 0, 改用前缀和构造
                std::vector<uint32_t> pt_rp(n + 1, 0);
                for (uint32_t i = 0; i < n; ++i)
                    pt_rp[i + 1] = pt_rp[i] + ((agg[i] >= 0) ? 1u : 0u);
                std::vector<uint32_t> pt_ci(pt_rp[n]);
                std::vector<T> pt_val((size_t)pt_rp[n] * BV, T(0));
                for (uint32_t i = 0; i < n; ++i) {
                    if (agg[i] < 0) continue;
                    const uint32_t q = pt_rp[i];
                    pt_ci[q] = (uint32_t)agg[i];
                    for (int r = 0; r < BS; ++r) pt_val[(size_t)q*BV + r*BS + r] = T(1);
                }
                BlockMatrix<T, BS> Pt(n, (uint32_t)n_agg, pt_rp, pt_ci, pt_val);

                // ---- 4. 光滑延拓 P = (I − ω D⁻¹ A) P_t ----
                std::vector<std::vector<std::pair<uint32_t, std::vector<T>>>> p_rows(n);
                for (uint32_t i = 0; i < n; ++i) {
                    if (agg[i] < 0) continue;             // 孤立块行: P 该行为空
                    std::map<uint32_t, std::vector<T>> row_blks;

                    // 恒等项
                    row_blks[(uint32_t)agg[i]] = std::vector<T>(BV, T(0));
                    for (int r = 0; r < BS; ++r)
                        row_blks[(uint32_t)agg[i]][r*BS+r] = T(1);

                    // 光滑项: −ω D_i⁻¹ A[i][k]
                    const T* dinv = &level.diag_inv[(size_t)i * BV];
                    for (uint32_t p = cur_A.block_row_ptr()[i]; p < cur_A.block_row_ptr()[i+1]; ++p) {
                        uint32_t k = cur_A.block_col_idx()[p];
                        const int ga = agg[k];
                        if (ga < 0) continue;                 // 丢弃的孤立块行
                        uint32_t g = (uint32_t)ga;
                        const T* ablk = &cur_A.values()[(size_t)p * BV];
                        if (!row_blks.count(g)) row_blks[g] = std::vector<T>(BV, T(0));
                        for (int r = 0; r < BS; ++r)
                            for (int c = 0; c < BS; ++c) {
                                T s = 0;
                                for (int m = 0; m < BS; ++m) s += dinv[r*BS+m] * ablk[m*BS+c];
                                row_blks[g][r*BS+c] -= omega_smooth * s;
                            }
                    }
                    for (auto& kv : row_blks) p_rows[i].push_back(kv);
                }

                std::vector<uint32_t> p_rp(n + 1, 0);
                std::vector<uint32_t> p_ci;
                std::vector<T> p_val;
                for (uint32_t i = 0; i < n; ++i) {
                    p_rp[i+1] = (uint32_t)p_rows[i].size();
                    for (auto& kv : p_rows[i]) {
                        p_ci.push_back(kv.first);
                        for (int v = 0; v < BV; ++v) p_val.push_back(kv.second[v]);
                    }
                }
                for (uint32_t i = 0; i < n; ++i) p_rp[i+1] += p_rp[i];

                level.P = BlockMatrix<T, BS>(n, (uint32_t)n_agg, p_rp, p_ci, p_val);
                level.R = level.P.transpose();
                levels.push_back(std::move(level));
            }

            // ---- 5. Galerkin: A_c = P^T A P ----
            if (sc) {
                std::vector<uint32_t> ap_rp, ap_ci; std::vector<T> ap_v;
                blk_amg_scalar_matmul(n, (uint32_t)n_agg,
                                      levels.back().s_rp, levels.back().s_ci, levels.back().s_val,
                                      levels.back().p_rp, levels.back().p_ci, levels.back().p_val,
                                      drop_tol, max_row_nnz, ap_rp, ap_ci, ap_v);
                std::vector<uint32_t> ac_rp, ac_ci; std::vector<T> ac_v;
                blk_amg_scalar_matmul((uint32_t)n_agg, (uint32_t)n_agg,
                                      levels.back().r_rp, levels.back().r_ci, levels.back().r_val,
                                      ap_rp, ap_ci, ap_v,
                                      drop_tol, max_row_nnz, ac_rp, ac_ci, ac_v);
                // 膨胀回块矩阵: 下一层还要拿它做 m*I 检测, 检测不过就自动回退
                cur_A = blk_amg_scalar_expand<T, BS>(ac_rp, ac_ci, ac_v, (uint32_t)n_agg);
            } else {
                BlockMatrix<T, BS> AP = blk_matmul(cur_A, levels.back().P, drop_tol, max_row_nnz);
                cur_A = blk_matmul(levels.back().R, AP, drop_tol, max_row_nnz);
            }
        }

        // operator complexity = (所有层的 nnz 之和) / (finest 层的 nnz)。
        //   这是判断 AMG 层次是否健康的核心指标: 1.2~1.8 正常,
        //   > 2.5 说明 Galerkin 在把模板填满 (原来 N=32 是 3.20)。
        if (verbose)
        {
            double tot = 0.0;
            for (const auto& L : levels)
                tot += (double)(L.scalar ? L.s_ci.size() : L.A.nnz_blocks());
            std::printf("[BlockAMG]          -> 层数 %zu | operator complexity = %.2f "
                        "(drop=%.3g, cap=%u, smooth=%d, path=%s)\n",
                        levels.size(), (nnz0 > 0.0) ? (tot / nnz0) : 0.0,
                        drop_tol, max_row_nnz, n_smooth,
                        (!levels.empty() && levels.front().scalar) ? "scalar" : "block");
            std::fflush(stdout);
        }

        // ---- 最粗层稠密 LU 在 setup 时分解一次 ----
        //   (原来每个 V-cycle 重算一遍: sz=132 时 132^3/3 ≈ 1.5 MFlop 串行)
        if (!levels.empty()) {
            auto& Lc = levels.back();
            if (Lc.scalar) blk_amg_factor_coarse_scalar(Lc);
            else           blk_amg_factor_coarse(Lc);
        }

        // ---- 返回 V-cycle lambda ----
        //   注意: 这份 lambda 的所有 std::function 副本共享同一份层次与工作
        //   缓冲 —— apply 不可重入 (同一层次不能被嵌套调用), 串行调用是安全的。
        return [lv_ptr, n_smooth, omega_smooth](const std::vector<T>& rhs,
                                                std::vector<T>& sol) {
            blk_amg_vcycle(*lv_ptr, 0, n_smooth, omega_smooth, rhs, sol);
        };
    }
}