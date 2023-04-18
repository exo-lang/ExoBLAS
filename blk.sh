#!/usr/bin/env bash
file="sgemm_params6"
touch $file.out
: > $file.out

n_blk=(768 1536 3072 6144)
m_blk=(192 384 768 1536)
k_blk=(384 768 1536 3072)

for k in "${k_blk[@]}"
do
    for m in "${m_blk[@]}"
    do
        for n in "${n_blk[@]}"
        do
              echo "=============================="
              echo "M=$(($m)), K=$(($k)), N=$(($n))"
              echo "=============================="
              cmake --preset avx2 -DGEMM_M_BLK=$(($m)) -DGEMM_K_BLK=$(($k)) -DGEMM_N_BLK=$(($n))
              cmake --build build/avx2/ --target sgemm_bench
              echo "BLOCKS: M=$(($m)), K=$(($k)), N=$(($n))" >> $file.out
              ctest --test-dir build/avx2/test -R exo_sgemm_bench -V | grep G/s  >> $file.out
        done
    done
done
