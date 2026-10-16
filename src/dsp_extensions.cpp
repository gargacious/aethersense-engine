#include <aethersense/dsp_extensions.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace aether {
namespace {
void require(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(message);
}
template <class T> void transform(std::span<std::complex<T>> input, bool inverse) {
  const auto n = input.size();
  require(n > 0 && n <= 65536 && !(n & (n - 1)), "FFT length must be power of two in [1,65536]");
  for (auto x : input)
    require(std::isfinite(x.real()) && std::isfinite(x.imag()), "Nonfinite FFT input");
  // Scratch bounds failure/overflow without partially modifying caller-owned input.
  std::vector<std::complex<T>> values(input.begin(), input.end());
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    auto bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(values[i], values[j]);
  }
  for (std::size_t length = 2; length <= n; length <<= 1) {
    const auto omega =
        std::polar(T{1}, (inverse ? T{2} : T{-2}) * std::numbers::pi_v<T> / static_cast<T>(length));
    for (std::size_t start = 0; start < n; start += length) {
      std::complex<T> w{1, 0};
      for (std::size_t k = 0; k < length / 2; ++k) {
        auto a = values[start + k], b = values[start + k + length / 2] * w;
        values[start + k] = a + b;
        values[start + k + length / 2] = a - b;
        w *= omega;
      }
    }
  }
  for (auto &v : values) {
    if (inverse)
      v /= static_cast<T>(n);
    require(std::isfinite(v.real()) && std::isfinite(v.imag()), "FFT numerical overflow");
  }
  std::copy(values.begin(), values.end(), input.begin());
}
} // namespace
void fft_float32(std::span<std::complex<float>> input, bool inverse) { transform(input, inverse); }
void fft_float64(std::span<std::complex<double>> input, bool inverse) { transform(input, inverse); }
std::vector<double> ca_cfar_2d(std::span<const double> power, std::size_t rows, std::size_t columns,
                               const Cfar2DConfig &c) {
  require(rows > 0 && rows <= 1024 && columns > 0 && columns <= 1024 && rows <= 262144 / columns &&
              power.size() == rows * columns,
          "CFAR 2D invalid or excessive grid shape");
  require(c.training_rows > 0 && c.training_rows <= 511 && c.training_columns > 0 &&
              c.training_columns <= 511 && c.guard_rows <= 511 && c.guard_columns <= 511 &&
              std::isfinite(c.pfa) && c.pfa > 0 && c.pfa < 1,
          "CFAR 2D invalid training/guard/Pfa");
  const auto ry = c.training_rows + c.guard_rows, rx = c.training_columns + c.guard_columns;
  require(2 * ry + 1 <= rows && 2 * rx + 1 <= columns, "CFAR 2D window exceeds grid");
  for (auto v : power)
    require(std::isfinite(v) && v >= 0, "CFAR 2D power must be finite nonnegative");
  const auto count =
      (2 * ry + 1) * (2 * rx + 1) - (2 * c.guard_rows + 1) * (2 * c.guard_columns + 1);
  const auto alpha = static_cast<long double>(count) *
                     std::expm1(-std::log(static_cast<long double>(c.pfa)) / count);
  // Summed area table accumulates in extended precision, bounded to ~4 MiB.
  std::vector<long double> prefix((rows + 1) * (columns + 1));
  const auto stride = columns + 1;
  for (std::size_t y = 0; y < rows; ++y) {
    long double row_sum = 0;
    for (std::size_t x = 0; x < columns; ++x) {
      row_sum += power[y * columns + x];
      prefix[(y + 1) * stride + x + 1] = prefix[y * stride + x + 1] + row_sum;
      require(std::isfinite(prefix[(y + 1) * stride + x + 1]),
              "CFAR 2D accumulated power overflow");
    }
  }
  auto rectangle = [&](std::size_t y0, std::size_t x0, std::size_t y1, std::size_t x1) {
    return prefix[y1 * stride + x1] - prefix[y0 * stride + x1] - prefix[y1 * stride + x0] +
           prefix[y0 * stride + x0];
  };
  std::vector<double> threshold(power.size(), -1);
  for (std::size_t y = ry; y < rows - ry; ++y)
    for (std::size_t x = rx; x < columns - rx; ++x) {
      auto sum = rectangle(y - ry, x - rx, y + ry + 1, x + rx + 1) -
                 rectangle(y - c.guard_rows, x - c.guard_columns, y + c.guard_rows + 1,
                           x + c.guard_columns + 1);
      auto value = alpha * std::max(0.L, sum) / count;
      require(std::isfinite(value) && value <= std::numeric_limits<double>::max(),
              "CFAR 2D threshold overflow");
      threshold[y * columns + x] = static_cast<double>(value);
    }
  return threshold;
}
} // namespace aether
