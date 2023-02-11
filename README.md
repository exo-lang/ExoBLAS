```
cd level2/
exocc --stem gbmv -o test gbmv.py
```

## Compiling SGEMV Code

Compile Exo to C:
```
exocc -o tests/neon/gemv --stem sgemv level2/gemv.py
```

To run the tests, from within `tests/neon/gemv`, run
```
clang -c -O3 -ffast-math -ffp-contract=fast -arch arm64 -o sgemv.o sgemv.c
clang++ -std=c++17 -O3 -ffast-math -ffp-contract=fast -framework Accelerate -arch arm64 -o test.o test.cpp sgemv.o
./test.o 10000
```
If there is an error, try `export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)`.
