if (APPLE)
  add_executable(test_gemv_accelerate level2/test_gemv.cpp)
  target_link_libraries(test_gemv_accelerate PRIVATE EXO_BLAS::gemv "-framework Accelerate")
  target_compile_features(test_gemv_accelerate PRIVATE cxx_std_11)

  add_test(NAME gemv_1000 COMMAND test_gemv_accelerate 1000)
endif()
