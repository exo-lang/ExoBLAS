foreach(kernel gemm mscal symm syr2k syrk trmm)
  add_blas_library("exo_${kernel}" "${kernel}.py")
endforeach()
