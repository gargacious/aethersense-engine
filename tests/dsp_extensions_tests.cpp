#include <aethersense/dsp_extensions.hpp>
#include <aethersense/engine.hpp>
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
  bool v = false;
  try {
    f();
  } catch (const std::exception &) {
    v = true;
  }
  check(v, "Expected rejection");
}
int main(int argc, char **argv) {
  try {
    Cfar2DConfig c{2, 3, 1, 2, .01};
    const size_t rows = 24, cols = 32;
    std::vector<double> power(rows * cols);
    for (size_t i = 0; i < power.size(); ++i)
      power[i] = .25 + (i * 17 % 101) / 31.;
    auto threshold = ca_cfar_2d(power, rows, cols, c);
    std::vector<std::complex<double>> input(64);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = {std::sin(.37 * i) + .25 * std::cos(.91 * i),
                  std::cos(.13 * i) - .1 * std::sin(.47 * i)};
    auto d = input;
    fft_float64(d);
    std::vector<std::complex<float>> f(input.begin(), input.end());
    fft_float32(f);
    if (argc == 2 && std::string(argv[1]) == "--oracle") {
      json dj = json::array(), fj = json::array(), ij = json::array();
      for (size_t i = 0; i < input.size(); ++i) {
        ij.push_back({input[i].real(), input[i].imag()});
        dj.push_back({d[i].real(), d[i].imag()});
        fj.push_back({f[i].real(), f[i].imag()});
      }
      std::cout << json{{"power", power},
                        {"rows", rows},
                        {"columns", cols},
                        {"threshold", threshold},
                        {"training_rows", c.training_rows},
                        {"training_columns", c.training_columns},
                        {"guard_rows", c.guard_rows},
                        {"guard_columns", c.guard_columns},
                        {"pfa", c.pfa},
                        {"fft_input", ij},
                        {"float64", dj},
                        {"float32", fj}}
                       .dump()
                << '\n';
      return 0;
    }
    const size_t count = 7 * 11 - 3 * 5;
    const double expected = count * std::expm1(-std::log(.01) / count);
    auto uniform = ca_cfar_2d(std::vector<double>(rows * cols, 1), rows, cols, c);
    for (size_t y = 0; y < rows; ++y)
      for (size_t x = 0; x < cols; ++x) {
        const bool edge = y < 3 || y >= rows - 3 || x < 5 || x >= cols - 5;
        check(edge ? uniform[y * cols + x] == -1
                   : std::abs(uniform[y * cols + x] - expected) < 1e-12,
              "Rectangle training count/edge");
      }
    auto high = std::vector<double>(rows * cols, 1);
    high[12 * cols + 16] = 1e8;
    auto with_peak = ca_cfar_2d(high, rows, cols, c);
    check(std::abs(with_peak[12 * cols + 16] - expected) < 1e-9, "CUT excluded");
    high[13 * cols + 17] = 1e8;
    with_peak = ca_cfar_2d(high, rows, cols, c);
    check(std::abs(with_peak[12 * cols + 16] - expected) < 1e-9, "Guard rectangle excluded");
    Cfar2DConfig bad = c;
    bad.pfa = 0;
    rejects([&] { ca_cfar_2d(power, rows, cols, bad); });
    bad.pfa = NAN;
    rejects([&] { ca_cfar_2d(power, rows, cols, bad); });
    bad = c;
    bad.guard_rows = SIZE_MAX;
    rejects([&] { ca_cfar_2d(power, rows, cols, bad); });
    bad = c;
    bad.training_columns = 0;
    rejects([&] { ca_cfar_2d(power, rows, cols, bad); });
    rejects([&] { ca_cfar_2d(power, 0, cols, c); });
    rejects([&] { ca_cfar_2d(power, 1024, 1024, c); });
    rejects([&] { ca_cfar_2d(power, rows - 1, cols, c); });
    auto invalid = power;
    invalid[0] = -1;
    rejects([&] { ca_cfar_2d(invalid, rows, cols, c); });
    invalid[0] = INFINITY;
    rejects([&] { ca_cfar_2d(invalid, rows, cols, c); });
    rejects([&] { ca_cfar_2d(std::vector<double>(9), 3, 3, c); });
    for (size_t i = 0; i < f.size(); ++i)
      check(std::abs(std::complex<double>(f[i]) - d[i]) < 2e-4, "Float/double parity");
    fft_float32(f, true);
    fft_float64(d, true);
    for (size_t i = 0; i < f.size(); ++i) {
      check(std::abs(std::complex<double>(f[i]) - input[i]) < 1e-5, "Float inverse");
      check(std::abs(d[i] - input[i]) < 1e-12, "Double inverse");
    }
    for (size_t length : {size_t{1}, size_t{2}, size_t{128}, size_t{1024}, size_t{65536}}) {
      std::vector<std::complex<float>> tone(length);
      for (size_t i = 0; i < length; ++i)
        tone[i] = std::polar(1.f, 2 * std::numbers::pi_v<float> * static_cast<float>(i) / length);
      auto original = tone;
      fft_float32(tone);
      fft_float32(tone, true);
      for (size_t i = 0; i < length; ++i)
        check(std::abs(tone[i] - original[i]) < .002, "Bounded float large FFT inverse");
    }
    std::vector<std::complex<float>> malformed(3);
    rejects([&] { fft_float32(malformed); });
    malformed.resize(131072);
    rejects([&] { fft_float32(malformed); });
    malformed.resize(2);
    malformed[0] = {NAN, 0};
    rejects([&] { fft_float32(malformed); });
    malformed = {{std::numeric_limits<float>::max(), 0}, {std::numeric_limits<float>::max(), 0}};
    auto before = malformed;
    rejects([&] { fft_float32(malformed); });
    check(before == malformed, "Overflow mutated caller input");
    Cfar2DConfig overflow = c;
    overflow.pfa = std::numeric_limits<double>::min();
    rejects([&] {
      ca_cfar_2d(std::vector<double>(rows * cols, std::numeric_limits<double>::max()), rows, cols,
                 overflow);
    });
    std::vector<std::complex<double>> empty;
    rejects([&] { fft_float64(empty); });
    std::vector<std::complex<double>> bad_double{{INFINITY, 0}};
    rejects([&] { fft_float64(bad_double); });
    Random rng(93);
    uint64_t alarms = 0, total = 0;
    for (int trial = 0; trial < 180; ++trial) {
      for (auto &v : power)
        v = -std::log(rng.uniform());
      auto thresholds = ca_cfar_2d(power, rows, cols, c);
      for (size_t i = 0; i < power.size(); ++i)
        if (thresholds[i] >= 0) {
          ++total;
          if (power[i] > thresholds[i])
            ++alarms;
        }
    }
    double observed = static_cast<double>(alarms) / total;
    check(observed > .006 && observed < .015, "Seeded 2D Pfa outside broad acceptance band");
    std::cout << "PASS bounded 2D CFAR rectangle/guard/edge/input/overflow; float32/float64 FFT "
                 "forward/inverse parity; seeded exponential Pfa="
              << observed << " alarms=" << alarms << " n=" << total << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
