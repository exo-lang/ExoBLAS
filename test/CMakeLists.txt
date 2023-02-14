find_package(BLAS REQUIRED)

if (APPLE)
  add_executable(test_gemv_accelerate level2/test_gemv.cpp)
  target_link_libraries(test_gemv_accelerate PRIVATE EXO_BLAS::gemv "-framework Accelerate")
  target_compile_features(test_gemv_accelerate PRIVATE cxx_std_11)

  add_test(NAME gemv_1000 COMMAND test_gemv_accelerate 1000)


  add_executable(test_syrk_accelerate level3/test_syrk.cpp)
  target_link_libraries(test_syrk_accelerate PRIVATE EXO_BLAS::syrk "-framework Accelerate")
  target_compile_features(test_syrk_accelerate PRIVATE cxx_std_11)

  add_test(NAME syrk_10 COMMAND test_syrk_accelerate 10)

endif()

if (HAVE_NEON)
  add_executable(test_sgemm level3/test_gemm.cpp)
  target_link_libraries(test_sgemm PRIVATE EXO_BLAS::sgemm BLAS::BLAS)
endif ()
