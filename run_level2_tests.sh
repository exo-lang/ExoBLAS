#!/bin/bash
#KERNELS=("syr" "syr2" "ger" "gemv" "gbmv" "trmv")
KERNELS=("ger")

#PRESET=avx2
PRESET=apple-silicon

#for bla_vendor in "OpenBLAS" "Intel10_64lp_seq"
for bla_vendor in "OpenBLAS" "Apple"
do
  # cmake --preset $PRESET -DCMAKE_PREFIX_PATH=~/benchmark/build/ -DBLA_VENDOR=$bla_vendor
  cmake --preset $PRESET -DCMAKE_PREFIX_PATH=~/benchmark/build/ -DBLA_VENDOR=$bla_vendor -DBLAS_openblas_LIBRARY=/opt/homebrew/opt/openblas/lib/libopenblas.dylib 
  for kernel in ${KERNELS[@]}; do
    cmake --build build/${PRESET}/ --target exo_${kernel}
    for precision in "s" "d"; do
      cmake --build build/${PRESET}/ --target ${precision}${kernel}_bench
      ctest --test-dir build/${PRESET} -R cblas_${precision}${kernel}_bench
    done
  done
done

for kernel in ${KERNELS[@]}; do
  for precision in "s" "d"; do
    ctest --test-dir build/${PRESET} -R exo_${precision}${kernel}_bench
    ctest --test-dir build/${PRESET} -R ${precision}${kernel}_graph
  done
done
