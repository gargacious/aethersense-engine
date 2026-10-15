#include <aethersense/engine.hpp>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
using namespace aether;
void check(bool x, const char *m) {
  if (!x)
    throw std::runtime_error(m);
}
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  check(caught, "Expected rejection");
}
int main() {
  try {
    Config c;
    c.validate();
    rejects([&] { Config::from_json({{"seed", -1}}); });
    for (const auto *key : {"objects", "schema_version", "fft_size", "chirps", "cfar_training",
                            "cfar_guard", "confirmation_hits", "coast_frames"})
      rejects([&] { Config::from_json({{key, 4294967297ULL}}); });
    rejects([&] { Config::from_json({{"objects", 1.5}}); });
    rejects([&] { Config::from_json({{"surprise", 1}}); });
    rejects([&] {
      Config bad = c;
      bad.noise_m = 0;
      bad.validate();
    });
    rejects([&] {
      Config bad = c;
      bad.fft_size = 63;
      bad.validate();
    });
    Random r1(42), r2(42);
    for (int i = 0; i < 100; ++i)
      check(r1.normal() == r2.normal(), "Seeded RNG differs");
    std::vector<std::complex<double>> tone(128);
    for (int i = 0; i < 128; ++i)
      tone[i] = std::polar(1., 2 * std::numbers::pi * 13 * i / 128);
    auto original = tone;
    fft(tone);
    check(std::abs(tone[13] - std::complex<double>(128, 0)) < 1e-10, "FFT peak");
    fft(tone, true);
    for (int i = 0; i < 128; ++i)
      check(std::abs(tone[i] - original[i]) < 1e-12, "FFT roundtrip");
    rejects([&] {
      std::vector<std::complex<double>> v(13);
      fft(v);
    });
    std::vector<std::complex<double>> pulse{{1, 0}, {0, 1}, {-1, 0}}, input(12);
    std::copy(pulse.begin(), pulse.end(), input.begin() + 4);
    auto filtered = matched_filter(input, pulse);
    check(std::abs(filtered[6] - std::complex<double>(3, 0)) < 1e-12, "Matched-filter delay");
    auto thresholds = ca_cfar(std::vector<double>(32, 1), 3, 1, .001);
    check(thresholds[0] == -1 && thresholds[31] == -1, "CFAR edge policy");
    check(std::abs(thresholds[10] - 6 * (std::pow(1000., 1. / 6) - 1)) < 1e-12, "CFAR scaling");
    rejects([&] { ca_cfar(std::vector<double>{-1, 1, 1}, 1, 0, .01); });
    c.objects = 1;
    c.scenario = "single_stationary";
    Random r(42);
    Envelope e;
    auto truth = truth_at(c, 0);
    auto raw = synthesize_iq(c, truth, r, e);
    raw.validate();
    auto result = process_iq(raw, c);
    check(!result.detections.empty(), "Raw-to-observation path found no target");
    double expected = std::hypot(truth[0].x, truth[0].y), best = 1e9;
    for (auto &d : result.detections)
      best = std::min(best, std::abs(d.range_m - expected));
    check(best < result.range_bin_m, "Range reconstruction > one bin");
    auto different_rate = raw;
    different_rate.sample_rate_hz = 500000;
    auto different_signal = process_iq(different_rate, c).to_json();
    check(different_signal["sample_rate_hz"] == 500000, "Signal metadata lost raw sample rate");
    auto bad = raw;
    bad.iq.pop_back();
    rejects([&] { bad.validate(); });
    bad = raw;
    bad.sample_rate_hz = 0;
    rejects([&] { bad.validate(); });
    bad = raw;
    bad.iq[0] = {NAN, 0};
    rejects([&] { bad.validate(); });
    Tracker tracker;
    c.noise_m = .2;
    c.dropout = 0;
    c.false_rate = 0;
    c.scenario = "crossing_paths";
    double err = 0;
    for (int f = 0; f < 200; ++f) {
      auto gt = truth_at(c, f * .1);
      std::vector<Detection> ds{{static_cast<uint64_t>(f), gt[0].x, gt[0].y, .04, .04}};
      tracker.update(ds, f * 100000000ll, c);
      if (f > 20) {
        auto &t = tracker.tracks().at(0);
        err += std::hypot(t.state[0] - gt[0].x, t.state[1] - gt[0].y);
        check(t.confirmed, "Track not confirmed");
        for (int i = 0; i < 4; ++i) {
          check(t.covariance[i * 4 + i] >= 0, "Negative covariance");
          for (int j = 0; j < 4; ++j)
            check(std::abs(t.covariance[i * 4 + j] - t.covariance[j * 4 + i]) < 1e-10,
                  "Covariance asymmetric");
        }
      }
    }
    check(err < .1, "Tracking analytic constant velocity");
    rejects([&] { tracker.update({}, 1, c); });
    auto before = tracker.tracks().size();
    rejects([&] {
      std::vector<Detection> bad_d{{1, 0, 0, -1, 1}};
      tracker.update(bad_d, 20000000000ll, c);
    });
    check(tracker.tracks().size() == before, "Invalid covariance mutated tracks");
    for (int f = 200; f < 208; ++f)
      tracker.update({}, f * 100000000ll, c);
    check(tracker.tracks().empty(), "Track deletion after coasting");
    // Seeded empirical Pfa on exponential noise, complete-window cells only.
    Random noise(17);
    uint64_t alarms = 0, total = 0;
    for (int trial = 0; trial < 300; ++trial) {
      std::vector<double> power(128);
      for (auto &v : power)
        v = -std::log(noise.uniform());
      auto th = ca_cfar(power, 6, 2, .01);
      for (size_t i = 0; i < power.size(); ++i)
        if (th[i] >= 0) {
          ++total;
          if (power[i] > th[i])
            ++alarms;
        }
    }
    double p = static_cast<double>(alarms) / total;
    check(p > .006 && p < .015, "Empirical false-alarm rate outside seeded confidence band");
    std::cout << "PASS config, RNG, FFT, matched filter, CFAR, raw metadata, range, tracking, "
                 "covariance, lifecycle; Pfa="
              << p << " n=" << total << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
