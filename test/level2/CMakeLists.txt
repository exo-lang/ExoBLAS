foreach(kernel gemv ger trmv symv trsv syr syr2)
    add_exo_blas_test(level2 ${kernel} "")
endforeach()

foreach(kernel gbmv tbmv tbsv sbmv)
    add_exo_blas_test(level2 ${kernel} s)
    add_exo_blas_test(level2 ${kernel} d)
endforeach()
