# ExoBLAS

ExoBLAS is a BLAS (Basic Linear Algebra Subprograms) library implemented using the [Exo](https://github.com/exo-lang/exo) programming language.

The library design documentation can be found in Section 6 of [our ASPLOS '25 paper](.) and in [Samir Droubi's master's thesis](https://dspace.mit.edu/handle/1721.1/156752).

We collected performance graphs comparing ExoBLAS against OpenBLAS, BLIS, and MKL on two machines, which can be found in the following Google Drive folders:
- [Intel i7-1185G7](https://drive.google.com/drive/folders/102lOlilyndcNjh9ncNso_nw9FCjhOuXE?usp=sharing)
- [AWS m7i.large instance with Intel(R) Xeon(R) Platinum 8488C](https://drive.google.com/drive/folders/19cH-7wkl9RasBU6XF5kRrqpIWwYFfZiw?usp=sharing).

## Supported Kernels and Limitations

### BLAS Level 1

We developed a scheduling operator (`optimize_level_1`) that optimizes the entire BLAS level 1 kernels, including `asum`, `axpy`, `dot`, `rot`, `rotm`, `scale`, `swap`, `copy`, and `dsdot`, for a total of 24 kernel variants. However, due to limitations in Exo's object code, which lacks support for value-dependent control, we were unable to support the `nrm2` and `iamax` kernels.

All the implementations of kernels can be found in the `src/level1` directory, and the implementation of `optimize_level_1` is in `src/common/blaslib.py`.
Performance graphs can be found in the Google folder linked above under the `level_1` directories.
Our optimized kernels match the performance of OpenBLAS, BLIS, and MKL across the board.

### BLAS Level 2

We optimized 50 kernel variants, supporting most BLAS level-2 operations (excluding banded operations and packed-triangular formats) across all configurations. This includes operations such as `gemv`, `ger`, `symv`, `syr`, `syr2`, `trmv`, and `trsv`, precisions like float (`s`) and double (`d`), and operational parameters such as transpose (`t`), non-transpose (`n`), lower (`l`), upper (`u`) triangular, unit (`u`), and non-unit (`n`).
Our implementation achieved competitive performance with OpenBLAS and MKL on both AVX2 and AVX512 platforms.

### BLAS Level 3

We optimized `gemm`, `symm`, and `syrk` kernels supporting float (`s`) and double (`d`) precisions, transpose (`t`), non-transpose (`n`), lower (`l`), and upper (`u`) triangular operational parameters.
Our implementation is better than existing libraries for small sizes, but is worse for larger matrix sizes, probably due to less optimal tile and microkernel sizes.


## Install dependencies

As always, create a Python virtual environment.

```
$ python3 -m venv ~/.venv/exo-blas
$ source ~/.venv/exo-blas/bin/activate
$ python -m pip install -U pip setuptools wheel build
$ python -m pip install -Ur requirements.txt
$ pre-commit install
```

We suggest using the latest version of Exo, following its build
instructions to install Exo into this virtual environment.

```
$ git clone https://github.com/exo-lang/exo
$ cd exo
$ python -m build
$ python -m pip install dist/*.whl
$ cd ..
$ rm -rf exo
```

You will also need to install Google benchmark to run the tests:

```
$ git clone https://github.com/google/benchmark
$ cmake -S benchmark -B benchmark/build -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=NO
$ cmake --build benchmark/build
$ cmake --install benchmark/build --prefix ~/.local
```

If you prefer to keep your `~/.local` clean, you can install
elsewhere and set `CMAKE_PREFIX_PATH` to that directory in your
environment.

Additionally, the following dependencies are required:
- CMake version 3.23 or higher
- Ninja (on Ubuntu, install with `apt install ninja-build`)
- A CBLAS reference implementation, such as OpenBLAS (on Ubuntu, install with `apt install libopenblas-dev`) or MKL

## Building ExoBLAS

We use CMake and its presets feature to ease configuration. We currently support four presets, but `cmake --list-presets` will always show the current list:
```
$ cmake --list-presets
Available configure presets:

  "apple-silicon" - Apple M1 or M2 mac
  "linux-arm64"   - Linux AArch64
  "avx2"          - Intel AVX2
  "avx512"        - Intel AVX512
```

For example, to build ExoBLAS targeting AVX512 instructions, run:
```
cmake --preset avx512
cmake --build build/avx512
```

To target Apple M series with Neon, run:
```
cmake --preset apple-silicon
cmake --build build/apple-silicon
```

If there is an error on apple-silicon, try running:
```
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
```

## Testing ExoBLAS
For more detailed on building and testing ExoBLAS, please read [here](docs/TESTING.md).
