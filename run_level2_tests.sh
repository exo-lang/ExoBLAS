#!/bin/bash
KERNELS=("syr" "syr2" "ger" "gemv" "gbmv" "trmv")

for bla_vendor in "OpenBLAS" "Intel10_64lp_seq"
do
  cmake --preset avx2 -DCMAKE_PREFIX_PATH=~/benchmark/build/ -DBLA_VENDOR=$bla_vendor
  for kernel in ${KERNELS[@]}; do
    cmake --build build/avx2/ --target exo_${kernel}
    for precision in "s" "d"; do
      cmake --build build/avx2/ --target ${precision}${kernel}_bench
      ctest --test-dir build/avx2 -R cblas_${precision}${kernel}_bench
    done
  done
done

for kernel in ${KERNELS[@]}; do
  for precision in "s" "d"; do
    ctest --test-dir build/avx2 -R exo_${precision}${kernel}_bench
    ctest --test-dir build/avx2 -R ${precision}${kernel}_graph
  done
done
