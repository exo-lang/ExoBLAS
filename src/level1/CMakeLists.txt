foreach(kernel asum dot rot rotm scal axpy swap dsdot)
  add_blas_library("exo_${kernel}" "${kernel}.py")
endforeach()

add_blas_library(exo_copy exo_copy.py)
