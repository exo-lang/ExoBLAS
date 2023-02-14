# Exo BLAS

## Install dependencies

As always, create a Python virtual environment.

```
$ python3 -m venv ~/.venv/exo-blas
$ source ~/.venv/exo-blas/bin/activate
$ python -m pip install -U pip setuptools wheel build
$ python -m pip install -Ur requirements.txt
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

## Building Exo BLAS

We use CMake and its presets feature to ease configuration. There
are currently three presets, but `cmake --list-presets` will always
show the current list:

```
$ cmake --list-presets
Available configure presets:

  "apple-silicon" - Apple M1 or M2 mac
  "linux-arm64"   - Linux AArch64
  "avx2"          - Intel AVX2
```

For example, to use `apple-silicon`, you would run:

```
cmake --preset apple-silicon
cmake --build build/apple-silicon
ctest --test-dir build/apple-silicon
```

## Troubleshooting

If there is an error on apple-silicon, try running

```
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
```
