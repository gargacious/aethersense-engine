#include <aethersense/dsp_extensions.hpp>
#include <aethersense/engine.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <sys/utsname.h>
using namespace aether;
using Clock = std::chrono::steady_clock;
namespace {
volatile double sink = 0;
template <class F> json measure(F operation, size_t items) {
  auto call = [&] {
    auto start = Clock::now();
    double sum = operation();
    auto end = Clock::now();
    sink = sum;
    if (!std::isfinite(sum))
      throw std::runtime_error("Nonfinite benchmark checksum");
    return std::chrono::duration<double, std::micro>(end - start).count();
  };
  double cold = call();
  for (int i = 0; i < 5; ++i)
    call();
  std::vector<double> times;
  auto batch_start = Clock::now();
  for (int i = 0; i < 40; ++i)
    times.push_back(call());
  double elapsed = std::chrono::duration<double>(Clock::now() - batch_start).count();
  std::sort(times.begin(), times.end());
  auto percentile = [&](double p) {
    return times[static_cast<size_t>(std::ceil(p * times.size())) - 1];
  };
  return {{"cold_us", cold},
          {"warmup_trials", 5},
          {"timed_trials", 40},
          {"quantile_method", "nearest_rank"},
          {"p50_us", percentile(.50)},
          {"p95_us", percentile(.95)},
          {"p99_us", percentile(.99)},
          {"max_us", times.back()},
          {"timed_batch_elapsed_s", elapsed},
          {"items_per_trial", items},
          {"throughput_items_per_s", items * 40 / elapsed},
          {"checksum", static_cast<double>(sink)},
          {"allocations", "not_measured"}};
}
std::vector<std::complex<double>> direct_dft(const std::vector<std::complex<double>> &input) {
  std::vector<std::complex<double>> output(input.size());
  for (size_t k = 0; k < input.size(); ++k)
    for (size_t n = 0; n < input.size(); ++n)
      output[k] += input[n] * std::polar(1., -2 * std::numbers::pi * k * n / input.size());
  return output;
}
std::vector<double> nested_cfar(const std::vector<double> &p, size_t rows, size_t cols,
                                const Cfar2DConfig &c) {
  auto ry = c.training_rows + c.guard_rows, rx = c.training_columns + c.guard_columns;
  auto count = (2 * ry + 1) * (2 * rx + 1) - (2 * c.guard_rows + 1) * (2 * c.guard_columns + 1);
  double alpha = count * std::expm1(-std::log(c.pfa) / count);
  std::vector<double> out(p.size(), -1);
  for (size_t y = ry; y < rows - ry; ++y)
    for (size_t x = rx; x < cols - rx; ++x) {
      double sum = 0;
      for (size_t sy = y - ry; sy <= y + ry; ++sy)
        for (size_t sx = x - rx; sx <= x + rx; ++sx)
          if (sy < y - c.guard_rows || sy > y + c.guard_rows || sx < x - c.guard_columns ||
              sx > x + c.guard_columns)
            sum += p[sy * cols + sx];
      out[y * cols + x] = alpha * sum / count;
    }
  return out;
}
template <class T> double checksum(const std::vector<std::complex<T>> &values) {
  double result = 0;
  for (size_t i = 0; i < values.size(); ++i)
    result += (i + 1) * (static_cast<double>(values[i].real()) + .3 * values[i].imag());
  return result;
}
double checksum(const std::vector<double> &values) {
  double result = 0;
  for (size_t i = 0; i < values.size(); ++i)
    result += (i + 1) * values[i];
  return result;
}
} // namespace
int main() {
  try {
    constexpr uint32_t seed = 20261016;
    Random rng(seed);
    std::vector<std::complex<double>> input(128);
    json input_json = json::array();
    for (auto &v : input) {
      v = {rng.normal(), rng.normal()};
      input_json.push_back({v.real(), v.imag()});
    }
    std::vector<double> power(64 * 96);
    for (auto &v : power)
      v = -std::log(rng.uniform());
    Cfar2DConfig c{2, 3, 1, 1, .01};
    json measurements;
    measurements["fft_float64"] = measure(
        [&] {
          auto v = input;
          fft_float64(v);
          return checksum(v);
        },
        input.size());
    measurements["fft_float32"] = measure(
        [&] {
          std::vector<std::complex<float>> v(input.begin(), input.end());
          fft_float32(v);
          return checksum(v);
        },
        input.size());
    measurements["scalar_direct_dft_float64"] =
        measure([&] { return checksum(direct_dft(input)); }, input.size());
    measurements["cfar_summed_area_float64"] =
        measure([&] { return checksum(ca_cfar_2d(power, 64, 96, c)); }, power.size());
    measurements["scalar_nested_cfar_float64"] =
        measure([&] { return checksum(nested_cfar(power, 64, 96, c)); }, power.size());
    auto reference = direct_dft(input), actual = input;
    fft_float64(actual);
    std::vector<std::complex<float>> f(input.begin(), input.end());
    fft_float32(f);
    double error64 = 0, error32 = 0;
    for (size_t i = 0; i < input.size(); ++i) {
      error64 = std::max(error64, std::abs(reference[i] - actual[i]));
      error32 = std::max(error32, std::abs(reference[i] - std::complex<double>(f[i])));
    }
    auto slow = nested_cfar(power, 64, 96, c), fast = ca_cfar_2d(power, 64, 96, c);
    double error_cfar = 0;
    for (size_t i = 0; i < slow.size(); ++i)
      error_cfar = std::max(error_cfar, std::abs(slow[i] - fast[i]));
    if (error64 > 1e-9 || error32 > 5e-4 || error_cfar > 1e-10)
      throw std::runtime_error("Benchmark parity failure");
    utsname host{};
    if (uname(&host) != 0)
      throw std::runtime_error("uname failed");
    json results = {
        {"schema_version", 1},
        {"benchmark", "cpu_dsp_baseline"},
        {"clock", "steady_clock"},
        {"host", {{"os", host.sysname}, {"release", host.release}, {"architecture", host.machine}}},
        {"compiler", __VERSION__},
        {"build_flags", "provided by runner; see build metadata"},
        {"backend", "cpu"},
        {"seed", seed},
        {"input_fingerprint", fingerprint(json{{"fft", input_json}, {"power", power}}.dump())},
        {"fingerprint_algorithm", "FNV-1a diagnostic identifier; not integrity checksum"},
        {"parity",
         {{"float64_max_complex_error", error64},
          {"float64_tolerance", 1e-9},
          {"float32_max_complex_error", error32},
          {"float32_tolerance", 5e-4},
          {"cfar_max_absolute_error", error_cfar},
          {"cfar_tolerance", 1e-10}}},
        {"workload",
         {{"fft_samples", 128},
          {"cfar_rows", 64},
          {"cfar_columns", 96},
          {"training_rows", 2},
          {"training_columns", 3},
          {"guard_rows", 1},
          {"guard_columns", 1},
          {"pfa", .01},
          {"input_units", "FFT complex amplitudes; CFAR nonnegative linear power"},
          {"fft_item_unit", "complex samples"},
          {"cfar_item_unit", "grid cells including skipped edges"}}},
        {"notes",
         {"cold means first workload invocation in this process; caches are not flushed and "
          "process startup is excluded",
          "per-call allocation and input copying included; allocation count not measured",
          "single-thread software workload; not real-time scheduling or end-to-end service SLA"}}};
    results["measurements"] = std::move(measurements);
    std::cout << results.dump(2) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
