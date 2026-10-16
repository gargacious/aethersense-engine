#include <aethersense/dsp_extensions.hpp>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::uint64_t next(std::uint64_t &s) {
  auto z = (s += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
}
int main() {
  try {
    std::cout << std::setprecision(17) << "{\"schema_version\":1,\"family_alpha\":0.001,\"cases\":[";
    for (int k = 0; k < 6; ++k) {
      const auto geometry = k / 3;
      const double pfa = k % 3 == 0 ? .1 : (k % 3 == 1 ? .01 : .001);
      const std::size_t trials = k % 3 == 2 ? 200000 : 50000;
      const aether::Cfar2DConfig c = geometry == 0 ? aether::Cfar2DConfig{1, 2, 0, 0, pfa}
                                                   : aether::Cfar2DConfig{2, 1, 1, 1, pfa};
      const auto rows = 2 * (c.training_rows + c.guard_rows) + 1;
      const auto cols = 2 * (c.training_columns + c.guard_columns) + 1;
      const auto cut = (rows / 2) * cols + cols / 2;
      const auto training = rows * cols - (2 * c.guard_rows + 1) * (2 * c.guard_columns + 1);
      const std::uint64_t seed = UINT64_C(0x5347303400000000) + static_cast<std::uint64_t>(k);
      auto state = seed;
      std::size_t detections = 0;
      if (k) std::cout << ',';
      std::cout << "{\"index\":" << k << ",\"seed\":\"" << seed << "\",\"pfa\":" << pfa
                << ",\"trials\":" << trials << ",\"rows\":" << rows << ",\"columns\":" << cols
                << ",\"training_rows\":" << c.training_rows << ",\"training_columns\":" << c.training_columns
                << ",\"guard_rows\":" << c.guard_rows << ",\"guard_columns\":" << c.guard_columns
                << ",\"training_count\":" << training << ",\"samples\":[";
      for (std::size_t t = 0; t < trials; ++t) {
        std::vector<double> power(rows * cols);
        for (auto &v : power) {
          const double u = (static_cast<double>(next(state) >> 12) + .5) * 0x1p-52;
          v = -std::log(u);
        }
        const auto threshold = aether::ca_cfar_2d(power, rows, cols, c);
        for (std::size_t i = 0; i < threshold.size(); ++i)
          if (i != cut && threshold[i] != -1) throw std::runtime_error("edge sentinel");
        detections += power[cut] > threshold[cut];
        if (t < 8) {
          if (t) std::cout << ',';
          std::cout << "{\"power\":[";
          for (std::size_t i = 0; i < power.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << power[i];
          }
          std::cout << "],\"threshold\":" << threshold[cut] << '}';
        }
      }
      const auto log_tail = std::log(12000.);
      const double bound = std::sqrt(2 * static_cast<double>(trials) * pfa * (1-pfa) * log_tail) + 2 * log_tail / 3;
      if (std::abs(static_cast<double>(detections) - static_cast<double>(trials) * pfa) > bound)
        throw std::runtime_error("predeclared Bernstein acceptance failed");
      std::cout << "],\"detections\":" << detections << ",\"count_bound\":" << bound << '}';
    }
    std::cout << "],\"status\":\"PASS\"}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
