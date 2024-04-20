foreach(kernel asum dot rot rotm scal axpy swap dsdot)
    add_exo_blas_test(level1 ${kernel} "")
endforeach()

add_exo_blas_test(level1 dsdot s)
