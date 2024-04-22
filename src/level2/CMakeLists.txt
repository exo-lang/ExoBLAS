foreach(kernel gemv gbmv ger trmv tbmv symv trsv tbsv sbmv syr syr2)
  add_blas_library("exo_${kernel}" "${kernel}.py")
endforeach()
