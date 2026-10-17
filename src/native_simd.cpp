#include <aethersense/compute_backend.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#if !defined(AETHER_DISABLE_NATIVE_SIMD) && (defined(__aarch64__) || defined(__arm64__)) &&        \
    defined(__ARM_NEON)
#include <arm_neon.h>
#define AETHER_SIMD_NEON 1
#elif !defined(AETHER_DISABLE_NATIVE_SIMD) && defined(__x86_64__) && defined(__SSE2__)
#include <emmintrin.h>
#define AETHER_SIMD_SSE2 1
#endif
namespace aether {
bool native_simd_available() noexcept {
#if defined(AETHER_SIMD_NEON) || defined(AETHER_SIMD_SSE2)
  return true;
#else
  return false;
#endif
}
std::string_view native_simd_instruction_set() noexcept {
#if defined(AETHER_SIMD_NEON)
  return "arm64_neon";
#elif defined(AETHER_SIMD_SSE2)
  return "x86_64_sse2";
#else
  return "unsupported";
#endif
}
NativeSimdBackend::NativeSimdBackend() {
  if (!native_simd_available())
    throw std::invalid_argument("native_simd unavailable on this compile target");
}
bool NativeSimdBackend::supports(ComputePrecision precision) const noexcept {
  return precision == ComputePrecision::float64 && native_simd_available();
}
ComputeCapabilities NativeSimdBackend::capabilities() const noexcept {
  return {true, false, native_simd_available(), false, native_simd_available() ? 65536u : 0u, 0, 0};
}
void NativeSimdBackend::fft32(std::span<std::complex<float>>, bool) const {
  throw std::invalid_argument("native_simd supports float64 FFT only");
}
std::vector<double> NativeSimdBackend::ca_cfar_2d(std::span<const double>, std::size_t, std::size_t,
                                                  const Cfar2DConfig &) const {
  throw std::invalid_argument("native_simd has no CFAR implementation");
}
namespace {
#if defined(AETHER_SIMD_NEON) || defined(AETHER_SIMD_SSE2)
inline void butterfly(std::complex<double> &a, std::complex<double> &b, std::complex<double> w) {
  // Lanes [real,imag]. b*w = b*wr + swap(b)*[-wi,+wi].
#if defined(AETHER_SIMD_NEON)
  const auto av = vsetq_lane_f64(a.imag(), vdupq_n_f64(a.real()), 1);
  const auto bv = vsetq_lane_f64(b.imag(), vdupq_n_f64(b.real()), 1);
  const auto signs = vsetq_lane_f64(w.imag(), vdupq_n_f64(-w.imag()), 1);
  const auto product = vaddq_f64(vmulq_n_f64(bv, w.real()), vmulq_f64(vextq_f64(bv, bv, 1), signs));
  const auto sum = vaddq_f64(av, product), difference = vsubq_f64(av, product);
  a = {vgetq_lane_f64(sum, 0), vgetq_lane_f64(sum, 1)};
  b = {vgetq_lane_f64(difference, 0), vgetq_lane_f64(difference, 1)};
#else
  const auto av = _mm_set_pd(a.imag(), a.real()), bv = _mm_set_pd(b.imag(), b.real());
  const auto product =
      _mm_add_pd(_mm_mul_pd(bv, _mm_set1_pd(w.real())),
                 _mm_mul_pd(_mm_shuffle_pd(bv, bv, 1), _mm_set_pd(w.imag(), -w.imag())));
  const auto sum = _mm_add_pd(av, product), difference = _mm_sub_pd(av, product);
  a = {_mm_cvtsd_f64(sum), _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum))};
  b = {_mm_cvtsd_f64(difference), _mm_cvtsd_f64(_mm_unpackhi_pd(difference, difference))};
#endif
}
#endif
} // namespace
void NativeSimdBackend::fft64(std::span<std::complex<double>> input, bool inverse) const {
#if defined(AETHER_SIMD_NEON) || defined(AETHER_SIMD_SSE2)
  const auto n = input.size();
  if (n == 0 || n > 65536 || (n & (n - 1)))
    throw std::invalid_argument("SIMD FFT length must be power of two in [1,65536]");
  for (auto value : input)
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
      throw std::invalid_argument("Nonfinite SIMD FFT input");
  std::vector<std::complex<double>> values(input.begin(), input.end());
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
        std::polar(1., (inverse ? 2. : -2.) * std::numbers::pi / static_cast<double>(length));
    for (std::size_t start = 0; start < n; start += length) {
      std::complex<double> w{1, 0};
      for (std::size_t k = 0; k < length / 2; ++k) {
        butterfly(values[start + k], values[start + k + length / 2], w);
        w *= omega;
      }
    }
  }
  for (auto &value : values) {
    if (inverse)
      value /= static_cast<double>(n);
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
      throw std::invalid_argument("SIMD FFT numerical overflow");
  }
  std::copy(values.begin(), values.end(), input.begin());
#else
  (void)input;
  (void)inverse;
  throw std::invalid_argument("native_simd unavailable on this compile target");
#endif
}
} // namespace aether
