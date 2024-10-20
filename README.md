# ExoBLAS

ExoBLAS is a BLAS library implemented with [Exo](https://github.com/exo-lang/exo).
The library design documentation and the performance graphs compared against OpenBLAS and MKL can be found in [Samir Droubi's thesis](https://dspace.mit.edu/handle/1721.1/156752).

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

## Building Exo BLAS

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

## Testing ExoBLAS
For more detailed on building and testing ExoBLAS, please read [here](docs/TESTING.md).

## Troubleshooting
Some troubleshooting instructions can be found [here](docs/TROUBLESHOOTING.md).
