#pragma once

#include <stdint.h>
#include <vector>
#include <random>
#include <algorithm>

template <typename T>
static void randomize(std::vector<T> &buffer) {
    static std::random_device rd;
    static std::mt19937 rng{rd()};
    std::uniform_real_distribution<> rv{-1.0f, 1.0f};
    std::generate(buffer.begin(), buffer.end(), [&]() { return rv(rng); });
}

static std::vector<float> generate1d_sbuffer(int n, int inc) {
    std::vector<float> buffer(1 + (n - 1) * abs(inc));
    randomize(buffer);
    return buffer;
}

static std::vector<double> generate1d_dbuffer(int n, int inc) {
    std::vector<double> buffer(1 + (n - 1) * abs(inc));
    randomize(buffer);
    return buffer;
}
