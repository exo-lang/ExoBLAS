#!/usr/bin/env bash
file="sgemm_params7"
touch $file.out
: > $file.out

n_blk=(768 1024 1536 3072)
m_blk=(192 256 384 512 768)
k_blk=(256 384 512 768 1024 1536)

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
