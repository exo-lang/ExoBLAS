#!/bin/bash
str="h"
if [ "$1" == "$str" ] 
then
    echo Usage: ./run_test [OP] [K_BLK] [M_BLK] [M_REG] [N_REG] [PROBLEM SIZE]
else
    python3 $1.py --kc $2 --mc $3 --mr $4 --nr $5
    cd c/$1
    export OPENBLAS_NUM_THREADS=1
    g++ test_$1.cpp -lopenblas -o $1 -O3 -fpermissive -ffast-math -ffp-contract=fast
    ./$1 $6
fi