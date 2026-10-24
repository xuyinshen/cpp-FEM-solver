add_executable(smatrix_test_app
    test_BlockMatrix_schur3d.cpp
)

# 1D Stokes + Schur 补预处理 + GMRES (纯共享内存, 不依赖 MPI)
add_executable(schur_test_app
    test_BlockMatrix_schur.cpp
)

# M0: FEM Poisson + 制造解 (MMS) 收敛阶验证
#   —— 装配 / 单元 / Dirichlet 消元这条链路的第一次端到端自检
add_executable(fem_test_app
    test_fem_poisson.cpp
)

# 链接静态库；因为 include 目录是 PUBLIC 的，头文件路径会自动传递过来
target_link_libraries(smatrix_test_app PRIVATE math_lib)
target_link_libraries(schur_test_app  PRIVATE math_lib)
target_link_libraries(fem_test_app    PRIVATE math_lib)

# M1: Taylor-Hood (P2/P1) 稳态 Stokes —— Poiseuille 精确解 + MMS 收敛阶
add_executable(stokes_app
    test_fem_stokes.cpp
)
target_link_libraries(stokes_app      PRIVATE math_lib)

# M2: 稳态 Navier-Stokes (P2/P1) —— Newton + Re 延拓 + Ghia 方腔基准
add_executable(ns_app
    test_fem_ns.cpp
)
target_link_libraries(ns_app          PRIVATE math_lib)

# BlockMatrix + MPI 分布式 matvec 往返一致性测试
if(CPPNUM_ENABLE_MPI)
    add_executable(blockmatrix_mpi_test_app
        test_BlockMatrix_mpi.cpp
    )
    target_link_libraries(blockmatrix_mpi_test_app PRIVATE math_lib)
endif()
