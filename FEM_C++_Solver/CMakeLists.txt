cmake_minimum_required(VERSION 3.16)

project(simplified_serial)

# =============== 全局编译设置 =================

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CPPNUM_ENABLE_OPENMP "Enable OpenMP" ON)
option(CPPNUM_ENABLE_MPI    "Enable MPI"    ON)

# =============== 文件生成 ====================

# 可执行文件统一输出到项目根目录下的 bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 生成 compile_commands.json 给 clangd 用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ================ 警告接口 =====================
add_library(cppnum_warnings INTERFACE)
target_compile_options(cppnum_warnings INTERFACE -Wall -Wextra -Wpedantic)

# ================ 并行依赖 =====================
if(CPPNUM_ENABLE_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
        message(WARNING "OpenMP requested but not found -> disabled")
        set(CPPNUM_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP" FORCE)
    endif()
endif()

if(CPPNUM_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(NOT MPI_FOUND)
        message(WARNING "编译需要 MPI 但未能找到")
        set(CPPNUM_ENABLE_MPI OFF CACHE BOOL "Enable MPI" FORCE)
    endif()
endif()

# ================ 子目录 =======================
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(test)

# ================ 寻找 MKL =======================


set(MKLROOT "/home/kazeyoubi/miniconda3/envs/cfd7")
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKLROOT}/include NO_DEFAULT_PATH)

# 使用 mkl_rt（运行时单库，最简单）
find_library(MKL_RT_LIB mkl_rt PATHS ${MKLROOT}/lib NO_DEFAULT_PATH)

# 如果找不到 mkl_rt，则分别链接各个库
if(NOT MKL_RT_LIB)
    find_library(MKL_INTEL_LIB mkl_intel_ilp64 PATHS ${MKLROOT}/lib NO_DEFAULT_PATH)
    find_library(MKL_THREAD_LIB mkl_intel_thread PATHS ${MKLROOT}/lib NO_DEFAULT_PATH)
    find_library(MKL_CORE_LIB mkl_core PATHS ${MKLROOT}/lib NO_DEFAULT_PATH)
    find_library(MKL_IOMP_LIB iomp5 PATHS ${MKLROOT}/lib NO_DEFAULT_PATH)
    
    if(NOT MKL_INTEL_LIB OR NOT MKL_THREAD_LIB OR NOT MKL_CORE_LIB OR NOT MKL_IOMP_LIB)
        message(FATAL_ERROR "MKL libraries not found in ${MKLROOT}/lib")
    endif()
    
    set(MKL_LIBRARIES ${MKL_INTEL_LIB} ${MKL_THREAD_LIB} ${MKL_CORE_LIB} ${MKL_IOMP_LIB})
else()
    set(MKL_LIBRARIES ${MKL_RT_LIB})
endif()

if(NOT MKL_INCLUDE_DIR)
    message(FATAL_ERROR "MKL header (mkl.h) not found in ${MKLROOT}/include")
endif()

message(STATUS "MKL_INCLUDE_DIR = ${MKL_INCLUDE_DIR}")
message(STATUS "MKL_LIBRARIES = ${MKL_LIBRARIES}")