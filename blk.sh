#!/usr/bin/env bash
file="sgemm_params4"
touch $file.out
: > $file.out

n_blk=(5 6 7 8 9 10 11 12)
m_blk=(5 6 7 8 9 10 11 12)
k_blk=(5 6 7 8 9 10 11 12)

for k in "${k_blk[@]}"
do
    for m in "${m_blk[@]}"
    do
        for n in "${n_blk[@]}"
        do
              echo "=============================="
              echo "M=$((2**$m)), K=$((2**$k)), N=$((2**$n))"
              echo "=============================="
              cmake --preset avx2 -DGEMM_M_BLK=$((2**$m)) -DGEMM_K_BLK=$((2**$k)) -DGEMM_N_BLK=$((2**$n))
              cmake --build build/avx2/ --target sgemm_bench
              echo "BLOCKS: M=$((2**$m)), K=$((2**$k)), N=$((2**$n))" >> $file.out
              ctest --test-dir build/avx2/test -R exo_sgemm_bench -V | grep G/s  >> $file.out
        done
    done
done
