#pragma once

#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

template <typename T>
class AlignedBuffer {
 public:
  AlignedBuffer(size_t N, int inc, size_t alignment = 64) {
    size_ = 1 + (N - 1) * abs(inc);
    alignment_ = alignment;
    alloc_alligned();
    randomize();
  }

  AlignedBuffer(size_t N, int inc, T value, size_t alignment) {
    size_ = 1 + (N - 1) * abs(inc);
    alignment_ = alignment;
    alloc_alligned();
    memset(buffer_, value, size_ * sizeof(T));
  }

  AlignedBuffer(const AlignedBuffer<T> &other) {
    size_ = other.size_;
    alignment_ = other.alignment_;
    alloc_alligned();
    memcpy(buffer_, other.buffer_, size_ * sizeof(T));
  }

  AlignedBuffer<T> &operator=(const AlignedBuffer<T> &other) {
    free(buffer_);

    size_ = other.size_;
    alignment_ = other.alignment_;
    alloc_alligned();
    memcpy(buffer_, other.buffer_, size_ * sizeof(T));

    return *this;
  }

  ~AlignedBuffer() { free(buffer_); }

  size_t size() { return size_; }

  size_t alignment() { return alignment_; }

  T *data() { return buffer_; }

  T &operator[](std::size_t i) { return buffer_[i]; }

 private:
  void randomize() {
    static std::random_device rd;
    static std::mt19937 rng{rd()};
    std::uniform_real_distribution<> rv{-1.0f, 1.0f};
    for (size_t i = 0; i < size_; ++i) {
      buffer_[i] = rv(rng);
    }
  }

  void alloc_alligned() {
    buffer_ = (T *)aligned_alloc(
        alignment_, sizeof(T) * (size_ + alignment_ - (size_ % alignment_)));
    if (buffer_ == NULL) {
      throw "AlignedBuffer allocation Failed";
    }
  }

  size_t size_;
  size_t alignment_;
  T *buffer_;
};

template <typename T>
class AlignedBuffer2D : public AlignedBuffer<T> {
 public:
  AlignedBuffer2D(size_t M, size_t N, size_t alignment = 64)
      : AlignedBuffer<T>(M * N, 1, alignment) {}

  AlignedBuffer2D(size_t M, size_t N, T value, size_t alignment)
      : AlignedBuffer<T>(M * N, 1, value, alignment) {}
};
