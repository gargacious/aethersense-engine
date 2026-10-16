#include <aethersense/compute_backend.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
using namespace aether;
void check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  check(caught, "Expected backend rejection");
}
int main() {
  try {
    auto backend = make_compute_backend("reference_cpu");
    check(backend->name() == "reference_cpu", "Backend name");
    check(backend->supports(ComputePrecision::float32) &&
              backend->supports(ComputePrecision::float64),
          "CPU precision capability");
    check(!backend->supports(static_cast<ComputePrecision>(999)), "Unknown precision advertised");
    auto caps = backend->capabilities();
    check(caps.cpu_only && caps.fft_float32 && caps.fft_float64 && caps.cfar_float64,
          "CPU-only capability flags");
    check(caps.max_fft_samples == 65536 && caps.max_cfar_axis == 1024 &&
              caps.max_cfar_cells == 262144,
          "Explicit numerical bounds");
    for (auto name : {"", "cpu", "cuda", "fftw", "reference_cpu ", "REFERENCE_CPU"})
      rejects([&] { make_compute_backend(name); });
    constexpr size_t n = 64;
    std::vector<std::complex<double>> original(n);
    for (size_t i = 0; i < n; ++i)
      original[i] = {std::sin(.13 * i), std::cos(.71 * i)};
    std::vector<std::complex<double>> expected(n);
    for (size_t k = 0; k < n; ++k)
      for (size_t j = 0; j < n; ++j)
        expected[k] += original[j] * std::polar(1., -2 * std::numbers::pi * k * j / n);
    auto d = original;
    std::vector<std::complex<float>> f(original.begin(), original.end());
    backend->fft64(d);
    backend->fft32(f);
    double double_error = 0, float_error = 0;
    for (size_t i = 0; i < n; ++i) {
      double_error = std::max(double_error, std::abs(d[i] - expected[i]));
      float_error = std::max(float_error, std::abs(std::complex<double>(f[i]) - expected[i]));
    }
    check(double_error < 1e-10 && float_error < 2e-4, "Backend independent direct DFT parity");
    backend->fft64(d, true);
    backend->fft32(f, true);
    for (size_t i = 0; i < n; ++i) {
      check(std::abs(d[i] - original[i]) < 1e-12, "Double inverse");
      check(std::abs(std::complex<double>(f[i]) - original[i]) < 1e-5, "Float inverse");
    }
    Cfar2DConfig c{1, 2, 1, 1, .01};
    std::vector<double> power(16 * 24, 1);
    auto threshold = backend->ca_cfar_2d(power, 16, 24, c);
    const double count = 5 * 7 - 3 * 3,
                 expected_threshold = count * std::expm1(-std::log(.01) / count);
    for (size_t y = 0; y < 16; ++y)
      for (size_t x = 0; x < 24; ++x) {
        bool edge = y < 2 || y >= 14 || x < 3 || x >= 21;
        check(edge ? threshold[y * 24 + x] == -1
                   : std::abs(threshold[y * 24 + x] - expected_threshold) < 1e-12,
              "Backend CFAR count/edge contract");
      }
    std::vector<std::complex<double>> empty;
    rejects([&] { backend->fft64(empty); });
    std::vector<std::complex<float>> badshape(7);
    rejects([&] { backend->fft32(badshape); });
    badshape.resize(caps.max_fft_samples * 2);
    rejects([&] { backend->fft32(badshape); });
    std::vector<std::complex<double>> nonfinite{{NAN, 0}};
    rejects([&] { backend->fft64(nonfinite); });
    std::vector<std::complex<float>> overflow{{std::numeric_limits<float>::max(), 0},
                                              {std::numeric_limits<float>::max(), 0}};
    auto saved = overflow;
    rejects([&] { backend->fft32(overflow); });
    check(saved == overflow, "Backend failure atomicity");
    rejects([&] { backend->ca_cfar_2d(power, 1024, 1024, c); });
    power[0] = INFINITY;
    rejects([&] { backend->ca_cfar_2d(power, 16, 24, c); });
    std::cout << "PASS CPU backend explicit capabilities/selection/bounds/finite/atomicity/CFAR; "
                 "DFT double_error="
              << double_error << " float_error=" << float_error << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
