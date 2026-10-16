#pragma once
#include <complex>
#include <cstddef>
#include <span>
#include <vector>
namespace aether {
// Row-major power grid; full rectangular training band surrounds guard+CUT.
// A skipped edge cell has threshold -1. Power must be finite and nonnegative.
// Max axes 1024, product 262144. Training extents count cells beyond guard.
struct Cfar2DConfig {
  std::size_t training_rows{2}, training_columns{3};
  std::size_t guard_rows{1}, guard_columns{1};
  double pfa{0.001};
};
std::vector<double> ca_cfar_2d(std::span<const double> power, std::size_t rows, std::size_t columns,
                               const Cfar2DConfig &);
// Explicit CPU precision ports; radix 2, 1..65536 samples, unnormalized forward,
// inverse divided by length. Inputs/results finite; invalid input is unchanged.
void fft_float32(std::span<std::complex<float>>, bool inverse = false);
void fft_float64(std::span<std::complex<double>>, bool inverse = false);
} // namespace aether
