# C++ 数值分析平台（构建中）

这是一个用 C++17 编写的**通用数值计算 / 微分方程求解**基础库。当前代码完成了**线性代数层**：稠密矩阵与三阶张量、CSR 稀疏矩阵及其迭代求解器与预条件子、Block-CSR（BAIJ）分布式块稀疏矩阵及其块 ILU(0) / 块 smoothed-aggregation AMG / Schur 补预条件子。


- 编译产物：静态库 `math_lib` + `bin/` 下的可执行文件（demo 与各测试用例）。
- 命名空间：`array`（稠密）、`sparse`（标量稀疏）、`block`（块稀疏）、`math`（积分）。
- 标识符允许中文（UTF-8），测试代码里会使用中文明命名。

---

## 1. 目录结构

```
.
├── CMakeLists.txt          # 根：C++17、OpenMP/MPI 探测、MKLROOT、enable_testing()
├── src/
│   ├── CMakeLists.txt      # 找 MKL、编 math_lib、挂 OpenMP/MPI 宏
│   ├── math/array.cpp      # Matrix / Tensor / SMatrix 实现（4.5k 行）
│   └── math/integrate.cpp  # 梯形 / Simpson 积分
├── include/math/
│   ├── array.hpp           # 稠密 + 标量稀疏的声明（含 #include <mkl.h>）
│   ├── blockMatrix.hpp     # BlockMatrix 全部实现（header-only，1.3k 行）
│   └── integral.hpp        # 积分接口
├── apps/
│   ├── CMakeLists.txt
│   └── main.cpp            # demo：simplified_serial_app，调积分接口
├── test/
│   ├── CMakeLists.txt      # 对 test/*.cpp 自动成一个 ctest 用例
│   └── test_*.cpp          # 12 个用例（见 §4）
├── RUN                     # 串行开发入口（增量配置+编译+ctest -L serial）
├── MPIRUN                  # 并行入口（默认 mpicxx + build-mpi + ctest -L mpi）
└── bin/                    # 所有可执行文件的统一输出目录
```

`build/`、`build-mpi/`、`build-verify/`、`bin/` 都在 `.gitignore` 里。`.gitattributes` 强制 `eol=lf`，否则 `RUN`/`MPIRUN` 在 Windows 侧 checkout 后带 `\r`，WSL 里无法执行。

---

## 2. 依赖

| 依赖 | 版本 / 要求 | 用途 | 缺失后果 |
|---|---|---|---|
| CMake | ≥ 3.16 | 构建 | 无法配置 |
| Ninja | 任意 | 生成器（`RUN`/`MPIRUN` 用 `-G Ninja`） | 改用 Makefiles 亦可 |
| C++ 编译器 | 支持 C++17（GCC ≥ 9） | — | — |
| **Intel MKL** | 需含 `include/mkl.h` 与 `lib/libmkl_rt` | `Matrix` 的 Cholesky / LU / 求逆 / 行列式走 `LAPACKE_dpotrf`、`LAPACKE_dgetrf`、`LAPACKE_dpotri`（`LAPACK_ROW_MAJOR`） | 配置阶段 `FATAL_ERROR`，见 §3.1 |
| OpenMP | libgomp / libiomp5 | 几乎所有循环与 `apply()` 的并行化 | 自动降级为串行（`CPPNUM_HAVE_OPENMP` 未定义） |
| MPI | OpenMPI / MPICH | `BlockMatrix` 的 halo 交换与 `distributed_matvec` | 自动降级（`CPPNUM_HAVE_MPI` 未定义），MPI 用例不注册 |

MKL 是**必须要安装的**，OpenMP 与 MPI 的功能可以通过 根 `CMakeLists.txt` 里的两个开关调整：

```cmake
option(CPPNUM_ENABLE_OPENMP "Enable OpenMP" ON)
option(CPPNUM_ENABLE_MPI    "Enable MPI"    ON)
```

源码里能用的预处理宏（由 `src/CMakeLists.txt` 以 `PUBLIC` 传递给所有链接者）：

| 宏 | 含义 |
|---|---|
| `CPPNUM_HAVE_OPENMP` | 可以写 `#pragma omp ...` |
| `CPPNUM_HAVE_MPI` | 可以 `#include <mpi.h>`，块矩阵的分布式成员存在 |

---

## 3. 配置与构建

### 3.1 相关库安装

安装 MKL 的两种方式：

```bash
# 方式 A：conda（本项目实际用的方式，环境名 cfd7）
conda create -n cfd7 -c conda-forge mkl mkl-devel openmpi cmake ninja gxx_linux-64
conda activate cfd7
ls "$CONDA_PREFIX/include/mkl.h" "$CONDA_PREFIX/lib/libmkl_rt.so"   # 两个都在即 OK

# 方式 B：Intel 官方 oneAPI HPC Toolkit
source /opt/intel/oneAPI/setvars.sh    # 导出 MKLROOT，之后 -DMKLROOT=$MKLROOT
```


### 3.2 脚本运行 （RUN 和 MPI_RUN）

```bash
./RUN              # 首次 cmake 配置到 build/，

./MPIRUN           # mpi 并行

```

### 3.4 .exe文件

统一落在 `bin/` 文件夹里。

---

## 4. 测试体系

### 用例清单

| 用例 | 内容 | 参考耗时* |
|---|---|---|
| `test_matrix_test` | `Matrix`/`Tensor` 全接口语法演示（含 10000 阶单位阵） | ~1 s（demo） |
| `test_omp` | OpenMP 并行加速比演示 | ~18 s（demo） |
| `test_integrate` | 梯形 / Simpson 积分收敛阶演示 | ~0.4 s（demo） |
| `test_SMatrix_pcg` | Poisson 型 SPD 稀疏阵，CG + 各预条件子（diag/ic0/ilu0/ssor/amg） | ~2 s |
| `test_SMatrix_bicgstab` | 非对称系统 BiCGSTAB + 预条件对比 | ~42 s |
| `test_SMatrix_gmres` | 非对称系统 GMRES（带 restart）+ 预条件对比 | ~10 s |
| `test_SMatrix_sor` | SOR / SOR-RB 松弛因子扫描 | ~7 s |
| `test_SMatrix_amg` | 标量 AMG 的 V-cycle 收敛性 | ~3 s |
| `test_BlockMatrix_schur` | 1D Stokes 鞍点系统：ILU0 + Schur 补 + GMRES | 秒级 |
| `test_BlockMatrix_mpi` | BlockMatrix 主回归网：布局/condense/转置/块 Jacobi/mini-CG/分布式 matvec/块 ILU0/预条件对比（N=128,512,1024） | ~22 s，峰值 RSS ~1.9 GB |
| `test_BlockMatrix_fixes` | halo 参数校验、迭代计数语义、分布式误用防护 | 秒级 |

每次使用时，修改 `test/CMakelists.txt` 里的测试用例的名称即可。

---

## 5. `array::Matrix` —— 稠密矩阵

行主序、`std::vector<double>` 连续存储、`uint32_t` 下标。

```cpp
#include "math/array.hpp"
using namespace array;

Matrix A(3, 3);                              // 3x3，未初始化元素为 0
Matrix B(3, 3, 1.0);                         // 全 1
Matrix C{{1.0, 2.0}, {3.0, 4.0}};            // 嵌套 vector 初始化（行优先）
Matrix D(2, 3, std::vector<double>{1,2,3,4,5,6});  // 展平数据 + 指定形状
auto   E = Matrix::identity(4);
auto   F = Matrix::zeros(3, 5);
auto   G = Matrix::ones(3, 5);
auto   H = Matrix::random(3, 3, -1.0, 1.0);  // 均匀分布

A(0, 1) = 2.0;                               // 读写
double v = A(1, 0);                          // 只读版本带边界检查
A.rows(); A.cols(); A.size();                // 维度 / 元素总数
A.data_ptr();  A.data_vector();              // 与 MKL / 外部库交互
A.print();                                   // 默认只打印 6x6，可给上限
```

**切片**：`Slice{start, end}` 是**双端闭区间**，`size() = end - start + 1`。默认参数 `{0,0}` 表示"只取第 0 个"，不是"取全部"：

```cpp
Matrix 子 = A({0, 1}, {1, 2});        // 2x2：行 0..1、列 1..2
Matrix 首列 = A({0, A.rows()-1}, {0, 0});
```

**性质判断**（均要求方阵）：`isSquare()`、`isSymmetric()`、`isDiag()`、`isAntisym()`、`isSymPos()`（先要求对称，再用 Cholesky 是否成功判定）、`isUpTri()`、`isDownTri()`。

**分解与求逆**（MKL 后端）：

```cpp
Matrix L = A.choleskyDecomposition();   // 只返回下三角 L（A = L Lᵀ），非正定抛 invalid_argument
auto [Lu, U, P, 换行数] = A.lu();        // LAPACKE_dgetrf，P·A = L·U；奇异抛 runtime_error
Matrix Ainv = A.inverse();               // 先试 Cholesky 快速路径(dpotrf+dpotri)，失败自动回退 LU 逐列求解
double det  = A.determinant();           // 基于 LU 的上三角对角元乘积 × 置换符号
```

**运算**：

```cpp
A + B;  A - B;
A * B;                                  // 仅支持"同尺寸方阵 × 方阵"，见下方警告
A * 2.0;  2.0 * A;  A / 2.0;  2.0 - A;  // 标量（左右两侧都有友元）
A.elementwise_multiply(B);              // Hadamard 积
A.elementwise_divide(B);
A.transpose();
```

**向量化逐元素运算**（内部 `#pragma omp parallel for`）：`abs() sin() cos() exp() log() sqrt() pow(e)`，以及自定义（示例）：

```cpp
Matrix g = A.apply([](double x){ return std::tanh(x) + 1.0; });
```

**统计**：`sum()`、`mean()`，以及返回 `{值, 行, 列}` 的 `max()` / `min()`（`Tensor` 版本返回 `{值, 行, 列, 层}`）。

`Matrix` ↔ `Tensor` 互转：`A.toTensor()`、`T.toMatrix()`；降回标准容器：`A.toVector()`。

---

## 6. `array::Tensor` —— 三阶张量

形状 `rows x cols x heis`（第三个维度叫 **heis**，按"层/高度"理解），行主序展平存储。适合存"每个网格点上有若干分量"这类数据。

```cpp
Tensor T(4, 5, 3);                              // 未初始化
Tensor U(4, 5, 3, 0.0);                         // 指定初值
Tensor V{{{1.0, 2.0}, {3.0, 4.0}}};             // 嵌套 vector
Tensor W(4, 5, 3, 展平vector);                  // 扁平数据 + 形状

T(2, 3, 1) = 7.0;
T.rows(); T.cols(); T.heis(); T.size();

Tensor 块 = T({0, 1}, {2, 4}, {0, 2});           // 三方向闭区间切片
T.print();                                      // print(最大行, 最大列, 最大层)
```

运算符：`+` `-`（张量间，形状必须一致）、与标量的 `+ - * /`（左右两侧都有友元）、`elementwise_multiply/divide`、向量化 `abs/sin/cos/exp/log/sqrt/pow/apply`、统计 `sum/mean/max/min`。

`Tensor` 只为三维数据存储、网格绘制、可视化等工作而设置，因此**没有**矩阵乘法与转置。对于矩阵操作，请使用 `Matrix`/`SMatrix`/`BlockMatrix`。

---

## 7. `sparse::SMatrix` —— CSR 标量稀疏矩阵

标准 CSR 三元组 `row_ptr_(rows+1) / col_idx_(nnz) / values_(nnz)`。

```cpp
#include "math/array.hpp"
using namespace sparse;

// 方式一：COO（三元组列表）—— 构造时自动排序、合并重复 (i,j)、丢掉合并后的 0
SMatrix A(/*rows*/3, /*cols*/3,
          {/*values   */ { 4.0, -1.0, -1.0,  4.0, -1.0, -1.0,  4.0 },
           {/*rowlist */ { 0,   0,     0,     1,     1,     2,     2  },
           {/*collist */ { 0,   1,     2,     0,     1,     0,     2  }});

// 方式二：直接给 CSR 数组（不做合并，最快）
SMatrix B(3, 3, std::vector<uint32_t>{0,3,6,9},
              std::vector<uint32_t>{0,1,2, 0,1,2, 0,1,2},
              std::vector<double> { /* 9 个值 */ });

A.rows(); A.cols(); A.nnz();
A.row_ptr(); A.col_idx(); A.values();      // 全部是 const 引用
// SMatrix 没有 print()：要看得逐个元素走 row_ptr/col_idx，或先转成 Matrix
```

> COO 构造的 `rowlist/collist` 必须是 `std::vector<uint32_t>`，写成 `vector<size_t>` 会没有匹配重载。

**基本运算**：

```cpp
std::vector<double> y = SMatrix::spmv(A, x);   // y = A·x
SMatrix At = A.transpose();                    // Aᵀ（CSR）
SMatrix C  = A * B;                            // 稀疏 × 稀疏
double  d  = SMatrix::dot(u, v);               // 向量内积
```

**预条件子工厂**（都返回 `std::function<void(const std::vector<double>&, std::vector<double>&)>`，即 `apply(r, z)` 表示 `z ≈ M⁻¹r`）：

```cpp
auto Mdiag = SMatrix::make_diag_preconditioner(A);
auto Mic0  = SMatrix::make_ic0_preconditioner(A);      // 不完全 Cholesky（SPD）
auto Milu  = SMatrix::make_ilu0_preconditioner(A);
auto Mssor = SMatrix::make_ssor_preconditioner(A, 1.2); // omega 默认 1.2
auto Mamg  = SMatrix::make_amg_preconditioner(A);       // 标量 smoothed-aggregation AMG
```

**迭代求解器**（`static`）：

```cpp
auto [x, it] = SMatrix::pcg(A, b, "ilu", 1e-10, 2000);          // 对称正定
auto [x, it] = SMatrix::bicgstab(A, b, "ilu", 1e-10, 2000);     // 非对称
auto [x, it] = SMatrix::gmres(A, b, "ilu", /*restart*/30, 1e-10, 2000);
auto [x, it] = SMatrix::gmres(A, b, Milu, 30, 1e-10, 2000);     // 传自定义预条件子
auto [x, it] = SMatrix::sor   (A, b, /*omega*/1.5, 1e-10, 2000);
auto [x, it] = SMatrix::sor_rb(A, b, 1.5, 1e-10, 2000);          // 红黑排序
auto [x, it] = SMatrix::amg   (A, b, 1e-10, 2000);              // AMG 独立求解
```

`precond` 字符串（大小写不敏感，内部 `to_lower`）可用别名：

| 归一化 | 可写 |
|---|---|
| 无预条件 | `none` `identity` `noprecond` |
| Jacobi | `diag` `diagonal` `jacobi` |
| IC(0) | `ic` `ic0` `ichol` `incomplete_cholesky` |
| ILU(0) | `ilu` `ilu0` |
| SSOR | `sor` `ssor` |
| AMG | `amg` |

注意求解器输出的除了解 `x` 之外，还有 `it`，即求解器的迭代次数。

---

## 8. 未来更新计划

- 块矩阵 `blockMatrix` 还没有完全构建完成，敬请期待
