#include <aethersense/contracts.hpp>
#include <aethersense/engine.hpp>
#include <cmath>
#include <iostream>
#include <limits>
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
    // Multiprocess admission is explicit; unsupported configurations must fail
    // before spawning, while the established in-process limits stay unchanged.
    Config multiprocess = c;
    multiprocess.executor = "multiprocess";
    multiprocess.queue_capacity = 1;
    multiprocess.recording_options = RecordingOptions{};
    multiprocess.fft_size = 256;
    multiprocess.chirps = 16;
    for (const char *path : {"synthetic_detections", "synthetic_iq"}) {
      multiprocess.input = path;
      multiprocess.validate();
      check(Config::from_json(multiprocess.to_json()).to_json() == multiprocess.to_json(),
            "Multiprocess config roundtrip changed explicit settings");
    }
    auto missing_recording = multiprocess;
    missing_recording.recording_options.reset();
    rejects([&] { missing_recording.validate(); });
    auto omitted_recording = multiprocess.to_json();
    omitted_recording.erase("recording_options");
    rejects([&] { Config::from_json(omitted_recording); });
    for (int capacity : {0, 2, 8, 64}) {
      auto invalid = multiprocess;
      invalid.queue_capacity = capacity;
      rejects([&] { invalid.validate(); });
    }
    auto default_queue = multiprocess.to_json();
    default_queue.erase("queue_capacity");
    rejects([&] { Config::from_json(default_queue); });
    for (int capacity : {0, 9}) {
      auto invalid = multiprocess;
      invalid.recording_options->queue_capacity = capacity;
      rejects([&] { invalid.validate(); });
    }
    auto invalid_recording = multiprocess;
    invalid_recording.recording_options->mode = "synchronous";
    rejects([&] { invalid_recording.validate(); });
    invalid_recording = multiprocess;
    invalid_recording.recording_options->version = 2;
    rejects([&] { invalid_recording.validate(); });
    auto paced = multiprocess;
    paced.source_options = SourceOptions{};
    rejects([&] { paced.validate(); });
    for (bool samples : {false, true}) {
      auto oversized = multiprocess;
      if (samples) oversized.fft_size = 512;
      else oversized.chirps = 32;
      rejects([&] { oversized.validate(); });
      rejects([&] { Config::from_json(oversized.to_json()); });
      oversized.input = "synthetic_detections";
      oversized.validate(); // Inactive IQ dimensions retain their legacy bounds.
    }
    for (const char *executor : {"single_thread", "staged_threads"}) {
      Config legacy = c;
      legacy.executor = executor;
      legacy.input = "synthetic_iq";
      legacy.fft_size = 1024;
      legacy.chirps = 64;
      legacy.validate();
      check(!legacy.recording_options && legacy.queue_capacity == 8,
            "Legacy defaults were silently converted to multiprocess requirements");
      check(Config::from_json(legacy.to_json()).to_json() == legacy.to_json(),
            "Legacy executor maximum dimensions changed");
    }
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
    // Public synthesis validates dimensions before signed products, allocation,
    // or RNG consumption, including callers that bypass Config::from_json.
    for (bool samples : {true, false}) {
      for (int invalid : {0, -1, std::numeric_limits<int>::min(),
                          std::numeric_limits<int>::max(), samples ? 2048 : 128}) {
        Config invalid_config = c;
        (samples ? invalid_config.fft_size : invalid_config.chirps) = invalid;
        Random attempted(71), untouched(71);
        bool rejected = false;
        try {
          (void)synthesize_iq(invalid_config, truth, attempted, e);
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        check(rejected, "Synthesis must reject invalid dimensions before arithmetic");
        check(attempted.normal() == untouched.normal(), "Rejected synthesis consumed RNG");
      }
    }
    for (const auto dimensions : {std::pair{32, 8}, std::pair{1024, 64}}) {
      Config boundary = c;
      boundary.fft_size = dimensions.first;
      boundary.chirps = dimensions.second;
      Random first(72), second(72);
      auto block = synthesize_iq(boundary, truth, first, e);
      auto repeat = synthesize_iq(boundary, truth, second, e);
      block.validate();
      check(block.iq.size() == std::size_t{2} * dimensions.first * dimensions.second,
            "Boundary synthesis shape");
      check(block.iq == repeat.iq, "Boundary synthesis seed determinism");
    }
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
