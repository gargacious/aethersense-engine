#include <aethersense/fixtures.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace aether;
void check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
template <class F> void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception &) {
    rejected = true;
  }
  check(rejected, "Expected rejection");
}
void psd(const Track &t) {
  // Independent Cholesky test, stronger than checking only diagonal/symmetry.
  double l[4][4]{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j <= i; ++j) {
      check(std::abs(t.covariance[i * 4 + j] - t.covariance[j * 4 + i]) < 1e-10,
            "Covariance symmetry");
      double v = t.covariance[i * 4 + j];
      for (int k = 0; k < j; ++k)
        v -= l[i][k] * l[j][k];
      if (i == j) {
        check(v > 0, "Covariance Cholesky positive pivot");
        l[i][j] = std::sqrt(v);
      } else
        l[i][j] = v / l[j][j];
    }
}
bool golden_equal(const json &a, const json &b) {
  if (a.is_number() && b.is_number()) {
    if (a.is_number_integer() && b.is_number_integer())
      return a == b;
    double x = a.get<double>(), y = b.get<double>();
    return std::abs(x - y) <= 1e-12 * std::max({1., std::abs(x), std::abs(y)});
  }
  if (a.type() != b.type() || a.size() != b.size())
    return false;
  if (a.is_array()) {
    for (size_t i = 0; i < a.size(); ++i)
      if (!golden_equal(a[i], b[i]))
        return false;
    return true;
  }
  if (a.is_object()) {
    for (auto it = a.begin(); it != a.end(); ++it)
      if (!b.contains(it.key()) || !golden_equal(it.value(), b.at(it.key())))
        return false;
    return true;
  }
  return a == b;
}
int main(int argc, char **argv) {
  try {
    auto raw = analytic_tone_fixture();
    auto j = serialize_raw_block(raw);
    check(serialize_raw_block(parse_raw_block(j.dump())) == j, "Raw roundtrip");
    auto malformed = [&](auto f) {
      auto b = j;
      f(b);
      rejects([&] { import_raw_block(b); });
    };
    malformed([](json &b) { b["unknown"] = 1; });
    malformed([](json &b) { b.erase("sample_rate_hz"); });
    malformed([](json &b) { b["packing"] = "sample_channel_chirp"; });
    malformed([](json &b) { b["channels"] = 3; });
    malformed([](json &b) { b["schema_version"] = 2; });
    malformed([](json &b) { b["sample_format"] = "int16"; });
    malformed([](json &b) { b["chirps"] = 8.0; });
    malformed([](json &b) { b["samples_per_chirp"] = 4294967328ULL; });
    malformed([](json &b) {
      b["samples_per_chirp"] = 1024;
      b["chirps"] = 64;
    });
    malformed([](json &b) { b["iq"].erase(b["iq"].end() - 1); });
    malformed([](json &b) { b["iq"][0] = {1}; });
    malformed([](json &b) { b["iq"][0][0] = NAN; });
    malformed([](json &b) { b["chirp_period_s"] = 1e-9; });
    malformed([](json &b) { b["sample_rate_hz"] = "1000"; });
    malformed([](json &b) { b["receive_steady_ns"] = "9223372036854775808"; });
    malformed([](json &b) { b["receive_steady_ns"] = 123; });
    malformed([](json &b) { b["timestamp_ns"] = -1; });
    rejects([&] { parse_raw_block(j.dump().substr(0, 100)); });
    rejects([&] { parse_raw_block("{\"schema_version\":1,\"schema_version\":1}"); });
    auto tone = std::vector<std::complex<double>>(raw.iq.begin(), raw.iq.begin() + 32);
    fft(tone);
    check(std::abs(tone[5] - std::complex<double>(32, 0)) < 1e-10, "Analytic FFT bin 5");
    std::vector<std::complex<double>> slow;
    for (int m = 0; m < 8; ++m)
      slow.push_back(raw.iq[m * 32]);
    fft(slow);
    check(std::abs(slow[2] - std::complex<double>(8, 0)) < 1e-10, "Analytic slow-time bin 2");
    std::vector<std::complex<double>> pulse{{1, 0}, {0, 1}, {-1, 0}}, delayed(16);
    std::copy(pulse.begin(), pulse.end(), delayed.begin() + 7);
    auto corr = matched_filter(delayed, pulse);
    check(std::abs(corr[9] - std::complex<double>(3, 0)) < 1e-12, "Analytic matched delay 7");
    for (const char *name : {"single_stationary", "crossing_paths", "intermittent_observation"}) {
      auto fixture = observation_fixture(name);
      check(fixture.dump() == observation_fixture(name).dump(), "Same seed bytes");
      check(fixture["frames"] != observation_fixture(name, 43)["frames"], "Different seed samples");
      Config c = Config::from_json(fixture["config"]);
      Tracker tracker;
      for (const auto &frame : fixture["frames"]) {
        std::vector<Detection> observations;
        for (const auto &d : frame["observations"]) {
          check(!d.contains("object_id") && !d.contains("truth_id"), "Truth ID leak");
          observations.push_back({d["observation_id"].get<uint64_t>(), d["x_m"], d["y_m"],
                                  d["covariance_m2"][0], d["covariance_m2"][3]});
        }
        tracker.update(observations, frame["timestamp_ns"], c);
        std::set<uint64_t> matched;
        for (const auto &t : tracker.tracks()) {
          psd(t);
          check(t.matched.size() <= 1, "Multiple observations per track");
          for (auto id : t.matched)
            check(matched.insert(id).second, "Observation assigned twice");
        }
      }
      rejects([&] { tracker.update({}, 5400000000LL, c); });
      rejects([&] { tracker.update({}, 5300000000LL, c); });
      if (argc == 3 && std::string(argv[1]) == "--write")
        std::ofstream(std::string(argv[2]) + "/" + name + ".json") << fixture.dump(2) << '\n';
      if (argc == 3 && std::string(argv[1]) == "--check") {
        json stored;
        std::ifstream file(std::string(argv[2]) + "/" + name + ".json");
        file >> stored;
        if (stored.contains("config"))
          stored["config"] = Config::from_json(stored["config"]).to_json();
        check(golden_equal(stored, fixture), "Persisted observation golden differs");
      }
    }
    if (argc == 3 && std::string(argv[1]) == "--write")
      std::ofstream(std::string(argv[2]) + "/analytic_tone.json") << j.dump(2) << '\n';
    if (argc == 3 && std::string(argv[1]) == "--check") {
      json stored;
      std::ifstream file(std::string(argv[2]) + "/analytic_tone.json");
      file >> stored;
      check(golden_equal(stored, j), "Persisted raw golden differs");
    }
    std::cout << "PASS raw strict import/roundtrip/negative cases; analytic FFT/delay; 3 golden "
                 "scenarios; Cholesky PSD; one-to-one; late/duplicate rejection\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
