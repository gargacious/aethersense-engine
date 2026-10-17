#include <aethersense/engine.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
using namespace aether;
void require(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
using Matrix = std::array<double, 16>;
double nees(const Track &t, const std::array<double, 4> &truth) {
  double l[4][4]{}, z[4]{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j <= i; ++j) {
      require(std::isfinite(t.covariance[i * 4 + j]), "Nonfinite covariance");
      require(std::abs(t.covariance[i * 4 + j] - t.covariance[j * 4 + i]) < 1e-10,
              "Asymmetric covariance");
      double v = t.covariance[i * 4 + j];
      for (int k = 0; k < j; ++k)
        v -= l[i][k] * l[j][k];
      if (i == j) {
        require(v > 0, "Covariance not positive definite");
        l[i][j] = std::sqrt(v);
      } else
        l[i][j] = v / l[j][j];
    }
  for (int i = 0; i < 4; ++i) {
    double v = t.state[i] - truth[i];
    for (int k = 0; k < i; ++k)
      v -= l[i][k] * z[k];
    z[i] = v / l[i][i];
  }
  double result = 0;
  for (auto v : z)
    result += v * v;
  require(std::isfinite(result), "Nonfinite NEES");
  return result;
}
void one_to_one(const Tracker &t) {
  std::set<uint64_t> ids;
  for (const auto &track : t.tracks()) {
    require(track.matched.size() <= 1, "Track matched multiple observations");
    for (auto id : track.matched)
      require(ids.insert(id).second, "Observation matched multiple tracks");
  }
}
json compact_frame(int frame, const Tracker &tracker) {
  require(tracker.tracks().size() <= 8, "Regression fixture exceeds eight tracks per frame");
  json tracks = json::array();
  for (const auto &track : tracker.tracks())
    tracks.push_back({{"track_id", track.id},
                      {"confirmed", track.confirmed},
                      {"matched_observation_ids", track.matched}});
  return {{"frame_index", frame}, {"tracks", std::move(tracks)}};
}
json crossing(uint32_t seed, json *compact_frames = nullptr) {
  Config c;
  c.confirmation_hits = 3;
  c.coast_frames = 5;
  Random random(seed);
  Tracker tracker;
  json frames = json::array();
  std::map<uint64_t, int> previous_truth;
  size_t switches = 0, associations = 0, correct_majority = 0;
  std::map<uint64_t, std::array<size_t, 2>> counts;
  for (int frame = 0; frame < 150; ++frame) {
    double time = frame * .1;
    std::vector<Detection> observations;
    std::map<uint64_t, int> identity;
    for (int object = 0; object < 2; ++object) {
      double vx = object ? -.7 : .7, x = object ? 5. : -5.;
      uint64_t id = frame * 2 + object;
      observations.push_back(
          {id, x + vx * time + .1 * random.normal(), 30. + .1 * random.normal(), .01, .01});
      identity[id] = object;
    }
    tracker.update(observations, frame * 100000000LL, c);
    one_to_one(tracker);
    json tracks = json::array();
    for (const auto &t : tracker.tracks()) {
      nees(t, t.state);
      tracks.push_back(serialize(t));
      for (auto id : t.matched) {
        int truth = identity.at(id);
        if (previous_truth.contains(t.id) && previous_truth[t.id] != truth)
          ++switches;
        previous_truth[t.id] = truth;
        ++counts[t.id][truth];
        ++associations;
      }
    }
    frames.push_back(tracks);
    if (compact_frames)
      compact_frames->push_back(compact_frame(frame, tracker));
  }
  for (const auto &[id, count] : counts) {
    (void)id;
    correct_majority += std::max(count[0], count[1]);
  }
  return {{"frames", frames},
          {"association_count", associations},
          {"identity_switches", switches},
          {"majority_association_purity", static_cast<double>(correct_majority) / associations}};
}
json diagnose_loss() {
  Config c;
  c.confirmation_hits = 3;
  c.coast_frames = 5;
  for (int seed = 9000; seed < 9256; ++seed) {
    Random random(seed);
    Tracker tracker;
    json history = json::array();
    for (int frame = 0; frame < 160; ++frame) {
      double time = frame * .1;
      Detection observation{static_cast<uint64_t>(frame), 10. + .8 * time + .5 * random.normal(),
                            30. - .3 * time + .5 * random.normal(), .25, .25};
      json before = json::array();
      for (const auto &t : tracker.tracks())
        before.push_back(serialize(t));
      tracker.update(std::span<const Detection>(&observation, 1), frame * 100000000LL, c);
      json after = json::array();
      bool original = false;
      for (const auto &t : tracker.tracks()) {
        after.push_back(serialize(t));
        if (t.id == 1)
          original = true;
      }
      history.push_back({{"frame", frame},
                         {"observation", serialize(observation)},
                         {"before", before},
                         {"after", after}});
      if (!original)
        return {{"seed", seed}, {"loss_frame", frame}, {"history", history}};
    }
  }
  return {{"loss", "none"}};
}
json cascade_regression() {
  Config c;
  Random random(9003);
  Tracker tracker;
  json fixture;
  for (int frame = 0; frame <= 26; ++frame) {
    double time = frame * .1;
    Detection observation{static_cast<uint64_t>(frame), 10. + .8 * time + .5 * random.normal(),
                          30. - .3 * time + .5 * random.normal(), .25, .25};
    json before = json::array();
    for (const auto &t : tracker.tracks())
      before.push_back(serialize(t));
    tracker.update(std::span<const Detection>(&observation, 1), frame * 100000000LL, c);
    if (frame == 22) {
      json after = json::array();
      bool reacquired = false;
      for (const auto &t : tracker.tracks()) {
        after.push_back(serialize(t));
        if (t.id == 1 && !t.matched.empty())
          reacquired = true;
      }
      require(reacquired, "Seed9003 established track not reacquired at frame22");
      fixture = {{"seed", 9003},
                 {"frame", 22},
                 {"before", before},
                 {"observation", serialize(observation)},
                 {"after", after}};
    }
    if (frame == 26)
      require(std::any_of(tracker.tracks().begin(), tracker.tracks().end(),
                          [](const Track &t) { return t.id == 1 && t.confirmed; }),
              "Seed9003 original lost at former deletion frame26");
  }
  return fixture;
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--diagnose") {
      std::cout << diagnose_loss().dump(2) << '\n';
      return 0;
    }
    auto cascade = cascade_regression();
    if (argc == 2 && std::string(argv[1]) == "--cascade-fixture") {
      std::cout << cascade.dump(2) << '\n';
      return 0;
    }
    const bool regression_fixture = argc == 2 && std::string(argv[1]) == "--regression-fixture";
    Config c;
    c.objects = 1;
    c.confirmation_hits = 3;
    c.coast_frames = 5;
    c.noise_m = .5;
    c.dropout = 0;
    c.false_rate = 0;
    constexpr int seeds = 256, frames = 160;
    double sum_nees = 0, sum_position_error = 0;
    size_t completed = 0, lost = 0, confirmed_delay = 0, psd_checks = 0, identity_transitions = 0,
           confirmed_covered = 0, coverage_frames = 0, replacements = 0, final_associated = 0,
           original_confirmed = 0;
    double all_position_error = 0;
    std::vector<double> final_nees;
    json records = json::array(), trials = json::array();
    for (int seed = 0; seed < seeds; ++seed) {
      Random random(9000 + seed);
      Tracker tracker;
      json trial_frames = json::array(), final_tracks = json::array();
      std::array<double, 4> final_truth{};
      int first_confirmation = -1;
      uint64_t previous_associated = 0;
      for (int frame = 0; frame < frames; ++frame) {
        double time = frame * .1;
        std::array<double, 4> truth{10. + .8 * time, 30. - .3 * time, .8, -.3};
        // Truth is used to create sensor noise here; update sees only position/covariance.
        std::vector<Detection> observations{{static_cast<uint64_t>(frame),
                                             truth[0] + .5 * random.normal(),
                                             truth[1] + .5 * random.normal(), .25, .25}};
        tracker.update(observations, frame * 100000000LL, c);
        one_to_one(tracker);
        if (regression_fixture) {
          trial_frames.push_back(compact_frame(frame, tracker));
          if (frame == frames - 1) {
            final_truth = truth;
            for (const auto &track : tracker.tracks())
              final_tracks.push_back(serialize(track));
          }
        }
        for (const auto &t : tracker.tracks()) {
          nees(t, truth);
          ++psd_checks;
          if (t.id == 1 && t.confirmed && first_confirmation < 0)
            first_confirmation = frame;
        }
        for (const auto &t : tracker.tracks())
          if (!t.matched.empty()) {
            if (previous_associated && previous_associated != t.id)
              ++identity_transitions;
            previous_associated = t.id;
            if (frame >= 2) {
              ++coverage_frames;
              if (t.confirmed)
                ++confirmed_covered;
            }
            if (frame == frames - 1) {
              double dx = t.state[0] - truth[0], dy = t.state[1] - truth[1];
              all_position_error += dx * dx + dy * dy;
              ++final_associated;
            }
          }
        if (frame == frames - 1) {
          auto original = std::find_if(tracker.tracks().begin(), tracker.tracks().end(),
                                       [](const Track &t) { return t.id == 1; });
          if (original == tracker.tracks().end()) {
            ++lost;
            for (const auto &t : tracker.tracks())
              if (t.confirmed && !t.matched.empty())
                ++replacements;
            continue;
          }
          double value = nees(*original, truth);
          sum_nees += value;
          final_nees.push_back(value);
          ++completed;
          double dx = original->state[0] - truth[0], dy = original->state[1] - truth[1];
          sum_position_error += dx * dx + dy * dy;
          records.push_back({{"seed", 9000 + seed},
                             {"truth", truth},
                             {"state", original->state},
                             {"covariance", original->covariance},
                             {"nees", value}});
        }
      }
      if (first_confirmation >= 0) {
        confirmed_delay += first_confirmation;
        ++original_confirmed;
      }
      if (regression_fixture)
        trials.push_back({{"seed", 9000 + seed},
                          {"frames", std::move(trial_frames)},
                          {"final_truth", final_truth},
                          {"final_tracks", std::move(final_tracks)}});
    }
    require(completed > 0, "No original tracks survived for conditional NEES evaluation");
    double mean = sum_nees / completed;
    // Constant truth has zero process noise while filter assumes acceleration noise;
    // expected NEES is conservative, not exactly chi-square4. Broad upper alarm only.
    require(mean > .1 && mean < 6, "Gross constant-velocity NEES inconsistency");
    std::sort(final_nees.begin(), final_nees.end());
    auto quantile = [&](double p) {
      return final_nees[static_cast<size_t>(std::ceil(p * final_nees.size())) - 1];
    };
    Tracker lifecycle;
    std::vector<Detection> observation{{1, 10, 30, .25, .25}};
    lifecycle.update(observation, 0, c);
    require(!lifecycle.tracks()[0].confirmed, "Confirmed before hit threshold");
    lifecycle.update(observation, 100000000, c);
    require(!lifecycle.tracks()[0].confirmed, "Confirmed before hit3");
    lifecycle.update(observation, 200000000, c);
    require(lifecycle.tracks()[0].confirmed, "No confirmation at third hit");
    lifecycle.update({}, 300000000, c);
    require(lifecycle.tracks()[0].misses == 1, "Coast missed count");
    lifecycle.update(observation, 400000000, c);
    require(lifecycle.tracks()[0].id == 1 && lifecycle.tracks()[0].misses == 0,
            "Coast reacquisition lost identity");
    for (int miss = 1; miss <= 5; ++miss) {
      lifecycle.update({}, 400000000 + miss * 100000000LL, c);
      require(lifecycle.tracks().size() == 1, "Deleted before coast limit");
    }
    lifecycle.update({}, 1000000000, c);
    require(lifecycle.tracks().empty(), "Failed deletion after coast limit");
    Tracker tentative;
    tentative.update(observation, 0, c);
    tentative.update({}, 100000000, c);
    require(tentative.tracks().size() == 1, "Tentative premature deletion");
    tentative.update({}, 200000000, c);
    require(tentative.tracks().empty(), "Tentative loss not deleted");
    json crossing_frames = json::array();
    auto cross = crossing(456, regression_fixture ? &crossing_frames : nullptr);
    require(cross.dump() == crossing(456).dump(), "Crossing association not deterministic");
    json report = {
        {"schema_version", 2},
        {"status", "PASS_WITH_LIMITATIONS"},
        {"scenario", "no-dropout known constant velocity; Gaussian sigma0.5m"},
        {"seeds", seeds},
        {"frames_per_seed", frames},
        {"completed_original_tracks", completed},
        {"original_track_losses", lost},
        {"lost_original_with_final_confirmed_replacement", replacements},
        {"observation_track_identity_transitions", identity_transitions},
        {"confirmed_observation_coverage_after_warmup",
         coverage_frames ? json(static_cast<double>(confirmed_covered) / coverage_frames)
                         : json(nullptr)},
        {"eligible_post_warmup_frames", seeds * (frames - 2)},
        {"associated_post_warmup_frames", coverage_frames},
        {"confirmed_associated_post_warmup_frames", confirmed_covered},
        {"confirmed_coverage_all_eligible_frames",
         static_cast<double>(confirmed_covered) / (seeds * (frames - 2))},
        {"final_associated_count", final_associated},
        {"final_unassociated_count", seeds - final_associated},
        {"final_associated_position_rmse_m",
         final_associated ? json(std::sqrt(all_position_error / final_associated)) : json(nullptr)},
        {"covariance_cholesky_checks", psd_checks},
        {"final_state_mean_nees", mean},
        {"final_nees_p50", quantile(.5)},
        {"final_nees_p95", quantile(.95)},
        {"final_position_rmse_m", std::sqrt(sum_position_error / completed)},
        {"original_confirmed_count", original_confirmed},
        {"original_never_confirmed_count", seeds - original_confirmed},
        {"mean_original_confirmation_delay_s",
         original_confirmed ? json(.1 * confirmed_delay / original_confirmed) : json(nullptr)},
        {"nees_test_bounds", {.1, 6.}},
        {"nees_interpretation", "conditional on original-track survival; conservative "
                                "deterministic CV model; not chi-square calibration acceptance"},
        {"crossing",
         {{"associations", cross["association_count"]},
          {"identity_switches", cross["identity_switches"]},
          {"majority_association_purity", cross["majority_association_purity"]},
          {"deterministic_repeat", true}}},
        {"lifecycle", "third hit confirmation; five misses coast; sixth delete; tentative second "
                      "miss delete; reacquire identity preserved"}};
    if (argc == 2 && std::string(argv[1]) == "--oracle")
      report["records"] = records;
    if (regression_fixture) {
      report["regression_fixture"] = {
          {"schema_version", 1},
          {"profile", "cv_tracking_regression_v1"},
          {"trials", std::move(trials)},
          {"crossing", {{"seed", 456}, {"frames", std::move(crossing_frames)}}}};
      const auto serialized = report.dump();
      require(serialized.size() <= 16 * 1024 * 1024, "Regression fixture exceeds 16 MiB");
      std::cout << serialized << '\n';
    } else
      std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
