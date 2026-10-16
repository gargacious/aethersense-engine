#pragma once
#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace aether {
// Complex float64 radix-2 CPU plan. Forward sign=-, inverse sign=+ and1/N.
// Thread confined: one caller owns this plan's mutable scratch. Separate plans
// may run concurrently. No allocations after construction on successful calls.
class Fft64Plan {
  size_t size_, max_batches_;
  std::vector<size_t> reversal_;
  std::vector<std::complex<double>> twiddles_, scratch_;
public:
  static constexpr size_t maximum_size=65536, maximum_batches=256;
  static constexpr size_t maximum_elements=1048576;
  explicit Fft64Plan(size_t size,size_t max_batches=1);
  Fft64Plan(const Fft64Plan&)=delete;
  Fft64Plan& operator=(const Fft64Plan&)=delete;
  Fft64Plan(Fft64Plan&&) noexcept=default;
  Fft64Plan& operator=(Fft64Plan&&) noexcept=default;
  size_t size() const noexcept {return size_;}
  size_t max_batches() const noexcept {return max_batches_;}
  // Element storage only: excludes allocator bookkeeping and object metadata.
  size_t storage_bytes() const noexcept;
  void execute(std::span<std::complex<double>>,bool inverse=false);
  // Contiguous batches, each size(). Rejects invalid dimensions/nonfinite input
  // or nonrepresentable output WITHOUT modifying ANY part of caller input.
  // Scratch is reusable after failure. Empty batches are invalid.
  void execute_batch(std::span<std::complex<double>>,size_t batches,bool inverse=false);
};
} // namespace aether
