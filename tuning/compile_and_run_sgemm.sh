#!/usr/bin/zsh

#EDIT THIS SCRIPT TO THE CORRECT PATH FOR YOUR BLAS
cd $HOME/BLAS
date >> $HOME/tuning_compile_log.txt
cmake --preset avx2 -DGEMM_M_BLK=${2:?"need m blk"} -DGEMM_N_BLK=${3:?"need n blk"} -DGEMM_K_BLK=${4:?"need k blk"} >> ./tuning_compile_log.txt
cmake --build build/avx2 --target sgemm_bench >> ./tuning_compile_log.txt
#add flag --clean-first if issues happen.
build/avx2/test/level3/sgemm_bench --benchmark_format=json --benchmark_filter=BM_SGEMM_EXO/n:${1:?"Error: matrix size not specified as arg"} | tee -a ./bench_results.log
