foreach(kernel gemm syrk syr2k symm)
    add_exo_blas_test(level3 ${kernel} "" EXO_BLAS::exo_mscal)
endforeach()

add_exo_blas_test(level3 trmm s)
