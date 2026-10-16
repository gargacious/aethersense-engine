#include <aethersense/engine.hpp>
#include <aethersense/tracking_diagnostics.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace aether;
namespace {
int checks{};
void require(bool yes, const char *message) {
  ++checks;
  if (!yes)
    throw std::runtime_error(message);
}
Config enabled() {
  Config c;
  c.tracking_diagnostics = TrackingDiagnosticsOptions{};
  return c;
}
Detection d(uint64_t id, double x, double y = 0, double variance = 1) {
  return {id, x, y, variance, variance};
}
json tracks(const Tracker &t) {
  json a = json::array();
  for (const auto &v : t.tracks())
    a.push_back(serialize(v));
  return a;
}
template <class Fn> void rejects(Fn fn, const char *m) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, m);
}
TrackingDiagnostics report(Tracker &t, std::vector<Detection> obs, int64_t time, const Config &c) {
  std::optional<TrackingDiagnostics> out;
  t.update(obs, time, c, &out);
  require(out.has_value(), "enabled report");
  const auto encoded = out->to_json();
  require(TrackingDiagnostics::from_json(encoded).to_json() == encoded,
          "strict typed report roundtrip");
  return *out;
}
void config_cases() {
  Config c;
  auto old = c.to_json();
  require(!old.contains("tracking_diagnostics"), "legacy omission");
  require(Config::from_json(old).to_json() == old, "legacy roundtrip bytes");
  c.tracking_diagnostics = TrackingDiagnosticsOptions{};
  auto j = c.to_json();
  require(Config::from_json(j).to_json() == j, "enabled config roundtrip");
  for (json bad : {json{{"schema_version", 1}, {"enabled", false}},
                   json{{"schema_version", 2}, {"enabled", true}},
                   json{{"schema_version", 1}, {"enabled", 1}}, json{{"schema_version", 1}},
                   json{{"schema_version", 1}, {"enabled", true}, {"unknown", 0}}}) {
    auto v = old;
    v["tracking_diagnostics"] = bad;
    rejects([&] { (void)Config::from_json(v); }, "strict diagnostic options");
  }
}
void no_effect() {
  Config off, on = enabled();
  Random random(42);
  Tracker a, b;
  std::optional<TrackingDiagnostics> out;
  for (int i = 0; i < 300; ++i) {
    auto truth = truth_at(off, i * off.dt_s);
    auto obs = simulate_detections(off, truth, random, i);
    a.update(obs, i * 100000000LL, off);
    b.update(obs, i * 100000000LL, on, &out);
    require(out.has_value(), "every successful enabled frame report");
    require(tracks(a).dump() == tracks(b).dump(), "on/off exact track serialization");
  }
  a.update({}, 30000000000LL, off, &out);
  require(!out, "disabled clears stale report");
}
void lifecycle() {
  auto c = enabled();
  c.coast_frames = 0;
  Tracker t;
  auto first = report(t, {d(1, 0)}, 0, c);
  require(first.prior_tracks == 0 && first.born_tracks == 1 && first.matches.empty(),
          "birth has no fake innovation");
  auto second = report(t, {d(2, 1, .5)}, 100000000, c);
  require(second.matches.size() == 1 &&
              second.matches[0].innovation.residual_m == std::array<double, 2>{1, .5},
          "actual pre-update residual");
  require(t.tracks()[0].state[0] != 0 &&
              second.matches[0].innovation.residual_m[0] != 1 - t.tracks()[0].state[0],
          "postupdate residual differs");
  auto empty = report(t, {}, 200000000, c);
  require(empty.deleted_tracks == 1 && empty.unmatched_prior_tracks == 1 &&
              empty.evaluated_pairs == 0,
          "empty/deletion report");
  std::optional<TrackingDiagnostics> stale = second;
  auto before = tracks(t);
  rejects([&] { t.update({}, 200000000, c, &stale); }, "late timestamp rejected");
  require(!stale && tracks(t) == before, "late clears report/state unchanged");
  Tracker fresh;
  std::vector<Detection> excess(129, d(1, 0));
  stale = second;
  rejects([&] { fresh.update(excess, 0, c, &stale); }, "upfront capacity");
  require(!stale && fresh.tracks().empty(), "capacity no mutation");
  rejects(
      [&] {
        fresh.update(std::vector<Detection>{d(1, 0), d(1, 1)}, 0, c, &stale);
      },
      "upfront unique IDs");
  require(fresh.tracks().empty(), "duplicate no mutation");
  report(fresh, {d(1, 0)}, 0, c);
  auto invalid = d(2, std::numeric_limits<double>::infinity());
  rejects([&] { fresh.update(std::vector<Detection>{invalid}, 100000000, c, &stale); },
          "nonfinite input");
  require(!stale, "invalid clears report");
  Tracker overflow;
  report(overflow, {d(1, 0)}, 0, c);
  rejects([&] { overflow.update(std::vector<Detection>{d(2, 1e308)}, 100000000, c, &stale); },
          "overflow distance fails visibly");
  require(!stale, "overflow no fake report");
}
void association() {
  auto c = enabled();
  c.confirmation_hits = 2;
  Tracker t;
  report(t, {d(1, 0)}, 0, c);
  report(t, {d(2, 0), d(3, 2)}, 100000000, c);
  auto r = report(t, {d(4, 2)}, 200000000, c);
  require(r.matches.size() == 1 && r.matches[0].track_id == 1 &&
              r.matches[0].confirmed_at_association,
          "confirmed-first actual farther match");
  require(r.skipped_observation_only == 1 && r.conflict_examples[0].track_id == 2,
          "tentative competition counted");
  Tracker ties;
  report(ties, {d(1, 0), d(2, 0)}, 0, c);
  auto tie = report(ties, {d(3, 0), d(4, 0)}, 100000000, c);
  require(tie.matches[0].track_id == 1 && tie.matches[0].observation_id == 3 &&
              tie.matches[1].track_id == 2 && tie.matches[1].observation_id == 4,
          "stable equal-distance ordering");
  require(tie.skipped_track_only == 1 && tie.skipped_observation_only == 1 && tie.skipped_both == 0,
          "disjoint conflict partition");
  Tracker dense;
  std::vector<Detection> many;
  for (uint64_t i = 0; i < 128; ++i)
    many.push_back(d(i, 0));
  report(dense, many, 0, c);
  auto full = report(dense, many, 100000000, c);
  require(full.evaluated_pairs == 16384 && full.matches.size() == 128 &&
              full.conflict_examples.size() == 32,
          "bounded full association");
  require(full.gate_pass_pairs == full.selected_associations + full.skipped_track_only +
                                      full.skipped_observation_only + full.skipped_both,
          "partition count identity");
  require(full.to_json().dump().size() <= 128 * 1024, "encoded cap");
  Tracker rejected;
  report(rejected, many, 0, c);
  for (auto &v : many)
    v.x = 1000;
  // Existing128 unmatched tracks would exceed birth capacity, so use coast0 to delete them.
  c.coast_frames = 0;
  auto out = report(rejected, many, 100000000, c);
  require(out.gate_reject_pairs == 16384 && out.gate_reject_examples.size() == 32 &&
              out.to_json()["gate_reject_examples_omitted"] == 16352,
          "bounded rejected examples");
}
void boundary_math() {
  auto correlated = tracking_innovation({1, 2}, {4, 1.5, 1.5, 3}, {3, -1}, {.5, 2});
  require(correlated.residual_m == std::array<double, 2>{2, -3} &&
              correlated.covariance_m2 == std::array<double, 4>{4.5, 1.5, 1.5, 5},
          "correlated S capture");
  rejects(
      [&] {
        (void)tracking_innovation({0, 0}, {1, 2, 2, 1}, {0, 0}, {1, 1});
      },
      "non PSD prediction");
  rejects(
      [&] {
        (void)tracking_innovation({0, 0}, {1, 0, 0, 1}, {1, 0}, {0, 1});
      },
      "nonpositive R");
  auto c = enabled();
  const double p = 1 + 25 * c.dt_s * c.dt_s + .25 * std::pow(c.dt_s, 4);
  double edge = std::sqrt(9.21034 * (p + 1));
  while (tracking_innovation({0, 0}, {p, 0, 0, p}, {edge, 0}, {1, 1}).mahalanobis_squared < 9.21034)
    edge = std::nextafter(edge, INFINITY);
  Tracker outside;
  report(outside, {d(1, 0)}, 0, c);
  auto rejected = report(outside, {d(2, edge)}, 100000000, c);
  require(rejected.gate_reject_pairs == 1, "boundary equality-or-above is rejected");
  double inside = std::nextafter(edge, 0.0);
  while (tracking_innovation({0, 0}, {p, 0, 0, p}, {inside, 0}, {1, 1}).mahalanobis_squared >=
         9.21034)
    inside = std::nextafter(inside, 0.0);
  Tracker within;
  report(within, {d(1, 0)}, 0, c);
  auto accepted = report(within, {d(2, inside)}, 100000000, c);
  require(accepted.gate_pass_pairs == 1, "adjacent below boundary accepted");
}
void strict_import() {
  auto c = enabled();
  Tracker t;
  report(t, {d(1, 0)}, 0, c);
  auto r = report(t, {d(2, 1, .5)}, 100000000, c);
  auto good = r.to_json();
  auto reject = [&](json bad) {
    rejects([&] { (void)TrackingDiagnostics::from_json(bad); },
            "strict diagnostics import rejection");
  };
  auto bad = good;
  bad["unknown"] = 1;
  reject(bad);
  bad = good;
  bad.erase("comparison");
  reject(bad);
  bad = good;
  bad["schema_version"] = 1.0;
  reject(bad);
  bad = good;
  bad["schema_version"] = 2;
  reject(bad);
  bad = good;
  bad["timestamp_ns"] = 9007199254740992ULL;
  reject(bad);
  bad = good;
  bad["gate_pass_pairs"] = 1.0;
  reject(bad);
  bad = good;
  bad["gate_pass_pairs"] = 16385;
  reject(bad);
  bad = good;
  bad["selected_associations"] = 0;
  reject(bad);
  bad = good;
  bad["gate_reject_examples_omitted"] = 1;
  reject(bad);
  bad = good;
  bad["matches"][0]["extra"] = 1;
  reject(bad);
  bad = good;
  bad["matches"][0]["track_id"] = 0;
  reject(bad);
  bad = good;
  bad["matches"][0]["observation_id"] = 9007199254740992ULL;
  reject(bad);
  bad = good;
  bad["matches"][0]["confirmed_at_association"] = 1;
  reject(bad);
  bad = good;
  bad["matches"][0]["innovation_m"] = json::array({1});
  reject(bad);
  bad = good;
  bad["matches"][0]["mahalanobis_squared"] = std::numeric_limits<double>::infinity();
  reject(bad);
  bad = good;
  bad["matches"][0]["mahalanobis_squared"] = 0;
  reject(bad);
  bad = good;
  bad["matches"][0]["mahalanobis_squared"] = 9.21034;
  reject(bad);
  bad = good;
  bad["matches"][0]["innovation_covariance_m2"] = json::array({1, 2, 2, 1});
  reject(bad);
  bad = good;
  bad["matches"][0]["disposition"] = std::string(200000, 'x');
  reject(bad);
  bad = good;
  bad["gate_reject_examples"] = json::array();
  for (int i = 0; i < 33; ++i)
    bad["gate_reject_examples"].push_back(good["matches"][0]);
  reject(bad);
  // Hand-constructed counter-consistent forgeries isolate relationships across arrays.
  auto forged = r;
  forged.prior_tracks = 3;
  forged.observations = 3;
  forged.evaluated_pairs = 9;
  forged.gate_pass_pairs = 2;
  forged.gate_reject_pairs = 7;
  forged.unmatched_prior_tracks = 2;
  forged.born_tracks = 2;
  forged.skipped_track_only = 1;
  auto example = forged.matches[0];
  example.track_id = 2;
  example.observation_id = 3;
  example.disposition = "track_used";
  forged.conflict_examples = {example};
  rejects([&] { (void)forged.to_json(); }, "track_used requires matched track");
  forged.skipped_track_only = 0;
  forged.skipped_observation_only = 1;
  forged.conflict_examples[0].disposition = "observation_used";
  rejects([&] { (void)forged.to_json(); }, "observation_used requires matched observation");
  forged.skipped_observation_only = 0;
  forged.skipped_both = 1;
  forged.conflict_examples[0].disposition = "both_used";
  rejects([&] { (void)forged.to_json(); }, "both_used requires both matched endpoints");
  forged.skipped_both = 0;
  forged.skipped_observation_only = 1;
  forged.conflict_examples[0].track_id = 1;
  forged.conflict_examples[0].disposition = "track_used";
  rejects([&] { (void)forged.to_json(); }, "per-disposition example count bounded");
  forged.skipped_observation_only = 0;
  forged.skipped_track_only = 1;
  auto valid_partial = forged.to_json();
  auto duplicate = valid_partial;
  duplicate["conflict_examples"].push_back(duplicate["conflict_examples"][0]);
  duplicate["skipped_track_only"] = 2;
  duplicate["gate_pass_pairs"] = 3;
  duplicate["gate_reject_pairs"] = 6;
  duplicate["conflict_examples_omitted"] = 0;
  reject(duplicate);
  forged.conflict_examples.clear();
  forged.skipped_track_only = 0;
  forged.gate_pass_pairs = 1;
  forged.gate_reject_pairs = 8;
  auto rejected_pair = forged.matches[0];
  rejected_pair.innovation.residual_m = {100, 0};
  const auto &S = rejected_pair.innovation.covariance_m2;
  rejected_pair.innovation.mahalanobis_squared = 10000 * S[3] / (S[0] * S[3] - S[1] * S[2]);
  rejected_pair.disposition = "gate_rejected";
  rejected_pair.observation_id = 1;
  for (uint64_t track = 2; track <= 4; ++track) {
    rejected_pair.track_id = track;
    forged.gate_reject_examples.push_back(rejected_pair);
  }
  rejects([&] { (void)forged.to_json(); }, "distinct prior IDs cannot exceed prior count");
  std::optional<TrackingDiagnostics> output = r;
  Tracker fresh;
  rejects([&] { fresh.update(std::vector<Detection>{d(9007199254740992ULL, 0)}, 0, c, &output); },
          "safe ID enforced before mutation");
  require(!output && fresh.tracks().empty(), "unsafe ID no stale report/mutation");
  rejects([&] { fresh.update({}, 9007199254740992LL, c, &output); }, "safe timestamp enforced");
  require(!output && fresh.tracks().empty(), "unsafe timestamp no stale report/mutation");
}
json oracle() {
  auto c = enabled();
  Tracker t;
  report(t, {d(1, 0)}, 0, c);
  auto r = report(t, {d(2, 1, .5)}, 100000000, c);
  auto v = tracking_innovation({1, 2}, {4, 1.5, 1.5, 3}, {3, -1}, {.5, 2});
  return {{"report", r.to_json()},
          {"correlated",
           {{"residual", v.residual_m}, {"S", v.covariance_m2}, {"d2", v.mahalanobis_squared}}}};
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--oracle") {
      std::cout << oracle().dump() << '\n';
      return 0;
    }
    config_cases();
    no_effect();
    lifecycle();
    association();
    boundary_math();
    strict_import();
    std::cout << "PASS " << checks << " tracking diagnostics assertions\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
