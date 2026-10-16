#include <aethersense/engine.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
using namespace aether;
namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
json tracks(const Tracker &tracker) {
  json result = json::array();
  for (const auto &t : tracker.tracks())
    result.push_back(serialize(t));
  return result;
}
void positive_covariance(const Track &track) {
  double lower[4][4]{};
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col <= row; ++col) {
      double value = track.covariance[row * 4 + col];
      require(std::isfinite(value), "Nonfinite covariance");
      require(std::abs(value - track.covariance[col * 4 + row]) < 1e-9, "Asymmetric covariance");
      for (int k = 0; k < col; ++k)
        value -= lower[row][k] * lower[col][k];
      if (row == col) {
        require(value > 0, "Nonpositive covariance pivot");
        lower[row][col] = std::sqrt(value);
      } else
        lower[row][col] = value / lower[col][col];
    }
}
Config configured() {
  Config c;
  c.objects = 1;
  c.duration_s = 6;
  c.noise_m = 1;
  c.dropout = 0;
  c.false_rate = 0;
  return c;
}
json profile() {
  return {{"schema_version", 1},
          {"motion",
           {{"mode", "stationary_linear_turn"},
            {"stationary_s", .5},
            {"linear_s", .5},
            {"turn_rate_rad_s", .2}}},
          {"occlusion", {{"start_frame", 10}, {"frame_count", 5}}},
          {"false_burst", {{"start_frame", 20}, {"frame_count", 3}, {"false_rate", 10}}},
          {"high_noise", {{"start_frame", 30}, {"frame_count", 5}, {"noise_m", 4}}}};
}
json simulate(const Config &config) {
  Random rng(config.seed);
  Tracker tracker;
  json frames = json::array();
  double squared_error = 0;
  int matched = 0, missing_original = 0;
  uint64_t previous_matched = 0;
  int transitions = 0;
  for (uint64_t frame = 0; frame < 60; ++frame) {
    const auto truth = truth_at(config, frame * config.dt_s);
    const auto observations = simulate_detections(config, truth, rng, frame);
    tracker.update(observations, static_cast<int64_t>(frame * 100000000), config);
    const bool occluded = frame >= 10 && frame < 15;
    const bool burst = frame >= 20 && frame < 23;
    const bool noisy = frame >= 30 && frame < 35;
    require(observations.size() == (occluded ? 0u : 1u) + (burst ? 10u : 0u),
            "Disturbance window boundary/count");
    if (!observations.empty() && !occluded) {
      require(observations.front().var_x == (noisy ? 16 : 1), "Absolute noise covariance override");
      require(observations.front().var_y == observations.front().var_x,
              "Modeled diagonal variance");
    }
    for (const auto &track : tracker.tracks()) {
      for (double value : track.state)
        require(std::isfinite(value), "Nonfinite tracker state");
      positive_covariance(track);
    }
    if (std::none_of(tracker.tracks().begin(), tracker.tracks().end(),
                     [](const auto &t) { return t.id == 1; }))
      ++missing_original;
    // Evaluation only: no truth value is an argument to Tracker::update.
    const Track *nearest = nullptr;
    double best = 100;
    for (const auto &track : tracker.tracks()) {
      const double distance = std::hypot(track.state[0] - truth[0].x, track.state[1] - truth[0].y);
      if (distance < best) {
        nearest = &track;
        best = distance;
      }
    }
    if (nearest && best <= 10) {
      squared_error += best * best;
      ++matched;
      if (previous_matched && previous_matched != nearest->id)
        ++transitions;
      previous_matched = nearest->id;
    }
    json obs = json::array();
    for (const auto &observation : observations)
      obs.push_back(serialize(observation));
    frames.push_back({{"sequence_no", frame},
                      {"evaluation_truth", serialize(truth[0])},
                      {"observations", obs},
                      {"tracks", tracks(tracker)}});
  }
  return {
      {"frames", frames},
      {"conditional_nearest_position_rmse_m",
       matched ? json(std::sqrt(squared_error / matched)) : json(nullptr)},
      {"matched_evaluation_frames", matched},
      {"original_id_missing_frames", missing_original},
      {"nearest_track_identity_transitions", transitions},
      {"evaluation_scope",
       "nearest truth within 10m, conditional; not association purity or uncertainty calibration"}};
}
} // namespace
int main() {
  try {
    json report = {{"schema_version", 1}, {"status", "passed"}};
    Config legacy = configured();
    require(!legacy.to_json().contains("scenario_overrides"), "Legacy optional omission changed");
    require(Config::from_json(legacy.to_json()).to_json() == legacy.to_json(),
            "Legacy default config roundtrip");
    auto full = legacy.to_json();
    full["scenario_overrides"] = profile();
    Config c = Config::from_json(full);
    require(c.to_json() == full, "Scenario full roundtrip changed shape");
    require(c.scenario_overrides.has_value(), "Typed optional overrides absent");
    auto reject = [&](const json &value) {
      try {
        Config::from_json(value);
      } catch (const std::exception &) {
        return;
      }
      throw std::runtime_error("Malformed scenario config accepted");
    };
    for (const auto &bad :
         json::array({nullptr, 4, json::array(), json::object(), json{{"schema_version", 2}},
                      json{{"schema_version", 1}, {"unknown", 1}}})) {
      auto value = full;
      value["scenario_overrides"] = bad;
      reject(value);
    }
    for (const char *group : {"motion", "occlusion", "false_burst", "high_noise"}) {
      auto value = full;
      value["scenario_overrides"][group]["unknown"] = 1;
      reject(value);
      value = full;
      auto begin = value["scenario_overrides"][group].begin();
      value["scenario_overrides"][group].erase(begin.key());
      reject(value);
      value = full;
      value["scenario_overrides"][group] = nullptr;
      reject(value);
    }
    for (const auto &bad : json::array({-1, 0, 180001, 4294967296ULL, 1.5, "2", nullptr, true})) {
      auto value = full;
      value["scenario_overrides"]["occlusion"]["frame_count"] = bad;
      reject(value);
    }
    for (const auto &bad : json::array({-1, 180000, 4294967296ULL, .5, "1", nullptr, true})) {
      auto value = full;
      value["scenario_overrides"]["occlusion"]["start_frame"] = bad;
      reject(value);
    }
    for (const auto &bad : json::array({-.1, 3601, "1", nullptr, true})) {
      auto value = full;
      value["scenario_overrides"]["motion"]["stationary_s"] = bad;
      reject(value);
    }
    for (const auto &bad : json::array({-.51, .51, "0", nullptr, true})) {
      auto value = full;
      value["scenario_overrides"]["motion"]["turn_rate_rad_s"] = bad;
      reject(value);
    }
    {
      auto value = full;
      value["input"] = "synthetic_iq";
      reject(value);
      value = full;
      value["scenario_overrides"]["motion"]["mode"] = "other";
      reject(value);
      value = full;
      value["scenario_overrides"]["motion"]["linear_s"] = 6;
      reject(value);
      value = full;
      value["scenario_overrides"]["occlusion"]["start_frame"] = 59;
      value["scenario_overrides"]["occlusion"]["frame_count"] = 2;
      reject(value);
      value = full;
      value["scenario_overrides"]["false_burst"]["false_rate"] = 10.1;
      reject(value);
      value = full;
      value["scenario_overrides"]["high_noise"]["noise_m"] = 20.1;
      reject(value);
      value = full;
      value["scenario_overrides"]["high_noise"]["noise_m"] = 0;
      reject(value);
      auto typed = c;
      typed.scenario_overrides->motion->turn_rate_rad_s = std::numeric_limits<double>::infinity();
      try {
        typed.validate();
        throw std::runtime_error("Infinity accepted");
      } catch (const std::invalid_argument &) {
      }
      typed = c;
      typed.scenario_overrides->occlusion->start_frame = std::numeric_limits<uint32_t>::max();
      try {
        typed.validate();
        throw std::runtime_error("Overflow window accepted");
      } catch (const std::invalid_argument &) {
      }
    }
    {
      Config maximum;
      maximum.duration_s = 3600;
      maximum.dt_s = .02;
      maximum.scenario_overrides = ScenarioOverrides{};
      maximum.scenario_overrides->occlusion = ScenarioWindow{0, 180000};
      maximum.validate();
      maximum.scenario_overrides->occlusion = ScenarioWindow{179999, 1};
      maximum.validate();
      maximum.scenario_overrides->occlusion = ScenarioWindow{179999, 2};
      try {
        maximum.validate();
        throw std::runtime_error("Window sum overflow accepted");
      } catch (const std::invalid_argument &) {
      }
    }
    for (const auto &entry :
         std::vector<std::pair<const char *, const char *>>{{"motion", "linear_s"},
                                                            {"motion", "turn_rate_rad_s"},
                                                            {"false_burst", "false_rate"},
                                                            {"high_noise", "noise_m"}}) {
      for (double bad :
           {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        auto value = full;
        value["scenario_overrides"][entry.first][entry.second] = bad;
        reject(value);
      }
    }
    for (double bad :
         {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      try {
        truth_at(c, bad);
        throw std::runtime_error("Invalid source time accepted");
      } catch (const std::invalid_argument &) {
      }
    }
    report["motion_samples"] = json::array();
    for (const double rate : {.2, -.2, 0., 1e-12, -1e-12, std::numeric_limits<double>::denorm_min(),
                              -std::numeric_limits<double>::denorm_min()}) {
      Config motion = configured();
      motion.duration_s = 30;
      motion.scenario_overrides = ScenarioOverrides{};
      motion.scenario_overrides->motion = ScenarioMotion{"stationary_linear_turn", 1, 2, rate};
      motion.validate();
      for (double t :
           {0., .5, 1., 2.5, 3., 3. - 1e-9, 3. + 1e-9, 3. + std::numbers::pi / (2 * .2), 15.}) {
        const auto truth = truth_at(motion, t)[0];
        report["motion_samples"].push_back({{"scenario", motion.scenario},
                                            {"t_s", t},
                                            {"stationary_s", 1},
                                            {"linear_s", 2},
                                            {"turn_rate_rad_s", rate},
                                            {"truth", serialize(truth)}});
        require(std::isfinite(truth.x) && std::isfinite(truth.y) && std::isfinite(truth.vx) &&
                    std::isfinite(truth.vy),
                "Nonfinite truth");
      }
      motion.scenario = "single_stationary";
      const auto truth = truth_at(motion, 15)[0];
      require(truth.x == 10 && truth.y == 45 && truth.vx == 0 && truth.vy == 0,
              "Stationary nominal velocity must stay zero");
    }
    {
      Config ordinary = configured(), noisy = ordinary;
      noisy.scenario_overrides = ScenarioOverrides{};
      noisy.scenario_overrides->high_noise = ScenarioHighNoise{{0, 1}, 4};
      noisy.validate();
      Random first(42), second(42);
      auto truth = truth_at(ordinary, 0);
      auto a = simulate_detections(ordinary, truth, first, 0),
           b = simulate_detections(noisy, truth, second, 0);
      require(std::abs((b[0].x - truth[0].x) / 4 - (a[0].x - truth[0].x)) < 1e-12,
              "Noise scales seeded x draw");
      require(std::abs((b[0].y - truth[0].y) / 4 - (a[0].y - truth[0].y)) < 1e-12,
              "Noise scales seeded y draw");
      noisy.scenario_overrides->high_noise.reset();
      noisy.scenario_overrides->occlusion = ScenarioWindow{0, 1};
      noisy.scenario_overrides->false_burst = ScenarioFalseBurst{{0, 1}, 10};
      Random third(42);
      auto clutter = simulate_detections(noisy, truth, third, 0);
      require(clutter.size() == 10, "Occlusion must not suppress false observations");
    }
    {
      const Truth valid{1, 10, 20, 1, 0};
      for (const auto &invalid : std::vector<Truth>{
               {2, 0, 0, 1, 0},
               {2, std::numeric_limits<double>::quiet_NaN(), 1, 1, 0},
               {2, 1, 2, std::numeric_limits<double>::infinity(), 0},
               {2, std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 1, 0},
               {2, 1e308, 1, 1e308, 0}}) {
        Random actual(42), untouched(42);
        const std::array<Truth, 2> input{valid, invalid};
        try {
          simulate_detections(c, input, actual, 0);
          throw std::runtime_error("Invalid radial truth accepted");
        } catch (const std::invalid_argument &) {
        }
        require(actual.uniform() == untouched.uniform(), "Truth rejection consumed RNG state");
      }
    }
    report["scenario_run"] = simulate(c);
    require(simulate(c) == report["scenario_run"], "Same seed scenario must repeat exactly");
    auto changed = c;
    changed.seed = 43;
    require(simulate(changed) != report["scenario_run"], "Different seed must change observations");
    // A test-only bounded delivery schedule preserves original timestamps; no production
    // retiming/reorder knob.
    {
      Config delivery = configured();
      delivery.scenario = "single_stationary";
      delivery.noise_m = .01;
      Random rng(7);
      std::vector<std::vector<Detection>> batches;
      for (int i = 0; i < 6; ++i)
        batches.push_back(simulate_detections(delivery, truth_at(delivery, i * .1), rng, i));
      require(batches.size() <= 8, "Delivery fixture exceeds bound");
      Tracker scheduled, reference;
      int rejected = 0, accepted = 0;
      json rejection_frames = json::array();
      for (int frame : {0, 2, 1, 2, 4, 3, 5}) {
        const auto before = tracks(scheduled);
        try {
          scheduled.update(batches[frame], int64_t(frame) * 100000000, delivery);
          ++accepted;
        } catch (const std::invalid_argument &) {
          ++rejected;
          rejection_frames.push_back(frame);
          require(tracks(scheduled) == before, "Rejected old batch mutated tracker");
        }
      }
      for (int frame : {0, 2, 4, 5})
        reference.update(batches[frame], int64_t(frame) * 100000000, delivery);
      require(accepted == 4 && rejected == 3 && tracks(scheduled) == tracks(reference),
              "Strict delayed delivery rejection/recovery");
      report["delivery"] = {{"capacity", 8},
                            {"accepted", accepted},
                            {"rejected", rejected},
                            {"rejected_source_frames", rejection_frames},
                            {"policy", "reject timestamp <= last accepted; state unchanged"},
                            {"production_delivery_scheduler", false}};
    }
    std::cout << report.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
