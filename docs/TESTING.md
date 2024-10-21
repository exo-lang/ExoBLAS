# Testing ExoBLAS

## Building ExoBLAS

We use CMake and its presets feature to ease configuration. There
are currently three presets, but `cmake --list-presets` will always
show the current list:

```
$ cmake --list-presets
Available configure presets:

  "apple-silicon" - Apple M1 or M2 mac
  "linux-arm64"   - Linux AArch64
  "avx2"          - Intel AVX2
  "avx512"        - Intel AVX512
```

For example, to use `avx2`, you can setup a build directory using:

```
cmake --preset avx2
```
You can build the whole library with tests using:
```
cmake --build build/avx2
```

If you want to compile an individual target you can run:
```
cmake --build build/avx2 --target exo_[KERNEL] # to build the binary for this kernel
cmake --build build/avx2 --target [KERNEL]_correctness # to build the correctness test for this kernel
cmake --build build/avx2 --target [KERNEL]_bench # to build the bench test for this kernel
```

If you want to benchmark against a different BLAS library (e.g., MKL), run:
```
cmake --preset avx2 -DBLA_VENDOR=[OTHER_LIB]
```
For example,
```
$ cmake --preset avx2 -DBLA_VENDOR=OpenBLAS # use OpenBLAS as a reference
$ cmake --preset avx2 -DBLA_VENDOR=Intel10_64lp_seq # Use MKL as a reference
```


Then, recompile the benchmark test:
```
cmake --build build/avx2 --target [KERNEL]_bench
```

## Running the correctness tests
To run the correctness test for all kernels:
```
ctest --test-dir ./build/avx2 -V -R correctness
```
To run the correctness for an individual kernel:
```
ctest --test-dir ./build/avx2 -V -R [KERNEL]_correctness
```

## Running the benchmark tests
To run the benchmark test for all kernels:
```
ctest --test-dir ./build/avx2 -V -R bench
```
To run the benchmark for an individual kernel:
```
ctest --test-dir ./build/avx2 -V -R [KERNEL]_bench
```
To run the benchmark for ExoBLAS only:
```
ctest --test-dir ./build/avx2 -V -R exo_[KERNEL]_bench # Run ExoBLAS benchmark for [KERNEL]
ctest --test-dir ./build/avx2 -V -R exo_ # Run ExoBLAS benchmark for all kernels
```
To run the benchmark for the reference BLAS library only:
```
ctest --test-dir ./build/avx2 -V -R cblas_[KERNEL]_bench # Run reference benchmark for [KERNEL]
ctest --test-dir ./build/avx2 -V -R cblas_ # Run the reference benchmark for all kernels
```
Note that benchmark results are cached, so you can run the reference benchmarks once when you start development. Remember to rerun them if you change the inputs being passed by the benchmark.

## Checking codegen consistency
It is valuable when doing development to be able to compare against the code-generation at the commit before you started you made any changes.

Before you start development, make sure to build the entire library using:
```
cmake --build build/avx2
```
Then, run the following command to generate a cache of the generated C code:
```
ctest --test-dir ./build/avx2 -V -R update_
```
This will create a local untracked copy of all the generated C sources in `test/codegen/reference/sources/`.

As you change the scheduling code or the library code, you can run the following command to check that the codegen is still the same (after rebuilding the library):
```
ctest --test-dir ./build/avx2 -V -R codegen_nonverbose
```
This will fail if the new files are different from the cached source and print the diff.

If you want to check an individual kernel's codegen, you can run:
```
ctest --test-dir ./build/avx2 -V -R [KERNEL]_codegen_nonverbose
```

If a kernel's codegen has changed, but this is expected. You can update its codegen again by running:
```
ctest --test-dir ./build/avx2 -V -R [KERNEL]_update_reference
```
This will update the cached sources, and the hash of the sources in the json stored in `test/codegen/reference/sha256/`. The hashes jsons are tracked files and they are what we use to verify the codegen output. The cached sources are only relevant so that we can show the diff when the hash is inconsistent.

## Running the graphing script
To plot one kernel results, you can run:
```
python analytics_tools/graphing/graph.py [KERNEL]
```
To plot summary results for all kernels, you can run:
```
python analytics_tools/graphing/graph.py all
```
You will be able to find the results in:
```
python analytics_tools/graphing/graphs
```

## Lines of code summary
To get a summary of the lines of code in `src/`, you can run:
```
python analytics_tools/loc/count_loc.py
```
