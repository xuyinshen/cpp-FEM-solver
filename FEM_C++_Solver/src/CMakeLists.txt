# ============== 寻找 MKL =================

set(MKLROOT "$ENV{CONDA_PREFIX}")
if(NOT MKLROOT)
    set(MKLROOT "/home/kazeyoubi/miniconda3/envs/cfd7")
endif()

find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKLROOT}/include NO_DEFAULT_PATH)
find_library(MKL_RT_LIB mkl_rt PATHS ${MKLROOT}/lib NO_DEFAULT_PATH)

if(NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIB)
    message(FATAL_ERROR "MKL not found in ${MKLROOT}")
endif()

message(STATUS "[src] MKL_INCLUDE_DIR = ${MKL_INCLUDE_DIR}")
message(STATUS "[src] MKL_RT_LIB = ${MKL_RT_LIB}")

# ============== 编译静态库 =================

# 把 src 下的源码编译成静态库
add_library(math_lib STATIC
    math/integrate.cpp
    math/array.cpp
)

# =============== 链接 include ===============

# 把项目根目录的 include/ 作为 PUBLIC 头文件目录暴露出去：
# - integrate.cpp 自己可以 #include "math/integral.hpp"
# - 链接了 math_lib 的目标(如 main.cpp)也自动获得该头文件路径

target_include_directories(math_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${MKL_INCLUDE_DIRS}
)

target_link_libraries(math_lib PUBLIC cppnum_warnings)

# =============== 链接并行库 ===================

if(CPPNUM_ENABLE_OPENMP)
    target_link_libraries(math_lib PUBLIC OpenMP::OpenMP_CXX)   # 真正的编译/链接旗标
    target_compile_definitions(math_lib PUBLIC CPPNUM_HAVE_OPENMP)  # 源码里可用的宏
endif()

if(CPPNUM_ENABLE_MPI)
    target_link_libraries(math_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(math_lib PUBLIC CPPNUM_HAVE_MPI)
endif()

# mkl_rt 放在 OpenMP 之后链接：GCC 构建必须优先装载 libgomp，运行时再由
# MKL_THREADING_LAYER=GNU 选择同一套线程运行库，避免 libgomp/libiomp5 混用。
target_link_libraries(math_lib PUBLIC ${MKL_RT_LIB})

# =============== 原生指令集 ===================
# 默认 x86-64 基线只到 SSE2；不开 -march=native 就完全拿不到 AVX2/FMA，
# 稠密内核与 VML 的收益会大打折扣。用 PUBLIC 传播，保证 apps/test 与库同 ISA。
include(CheckCXXCompilerFlag)
option(CPPNUM_ENABLE_NATIVE_ARCH "Enable -march=native" ON)
if(CPPNUM_ENABLE_NATIVE_ARCH)
    check_cxx_compiler_flag("-march=native" CPPNUM_HAVE_MARCH_NATIVE)
    if(CPPNUM_HAVE_MARCH_NATIVE)
        target_compile_options(math_lib PUBLIC -march=native)
        message(STATUS "[src] 启用 -march=native")
    else()
        message(WARNING "[src] 编译器不支持 -march=native，回退到默认指令集")
    endif()
endif()

