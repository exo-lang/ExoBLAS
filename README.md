## How to build

List of supported architectures (put them in <arch> in the command below)
- linux-arm64
- apple-silicon
- avx2

```
cmake --preset <arch>
cmake --build build/<arch>
ctest --test-dir build/<arch>
```

If there is an error on apple-silicon, try `export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)`.
