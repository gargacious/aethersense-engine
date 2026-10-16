#include <aethersense/tracking_diagnostics.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
namespace aether {
TrackingInnovation tracking_innovation(std::array<double, 2> position, std::array<double, 4> p,
                                       std::array<double, 2> observation,
                                       std::array<double, 2> variance) {
  auto finite = [](const auto &a) {
    return std::all_of(a.begin(), a.end(), [](double v) { return std::isfinite(v); });
  };
  if (!finite(position) || !finite(p) || !finite(observation) || !finite(variance) || p[0] < 0 ||
      p[3] < 0 || p[1] != p[2] || variance[0] <= 0 || variance[1] <= 0)
    throw std::invalid_argument("Invalid tracking innovation inputs");
  const double pd = p[0] * p[3] - p[1] * p[1];
  if (!std::isfinite(pd) || pd < 0)
    throw std::invalid_argument("Invalid predicted position covariance");
  double a = p[0] + variance[0], b = p[1], cc = p[3] + variance[1], det = a * cc - b * b,
         dx = observation[0] - position[0], dy = observation[1] - position[1];
  if (!std::isfinite(a) || !std::isfinite(cc) || !std::isfinite(det) || det <= 0 ||
      !std::isfinite(dx) || !std::isfinite(dy))
    throw std::runtime_error("Nonfinite/nonpositive innovation covariance or residual");
  double distance = (cc * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
  if (!std::isfinite(distance) || distance < 0)
    throw std::runtime_error("Nonfinite/negative innovation distance");
  return {{dx, dy}, {a, b, b, cc}, distance};
}
nlohmann::json TrackingDiagnostics::to_json() const {
  using nlohmann::json;
  if (gate_pass_pairs > 16384 || gate_reject_pairs > 16384 || selected_associations > 128 ||
      skipped_track_only > 16384 || skipped_observation_only > 16384 || skipped_both > 16384 ||
      timestamp_ns < 0 || timestamp_ns > 9007199254740991LL || prior_tracks > 128 ||
      observations > 128 || evaluated_pairs != prior_tracks * observations ||
      evaluated_pairs != gate_pass_pairs + gate_reject_pairs ||
      gate_pass_pairs !=
          selected_associations + skipped_track_only + skipped_observation_only + skipped_both ||
      selected_associations > prior_tracks || selected_associations > observations ||
      born_tracks != observations - selected_associations ||
      unmatched_prior_tracks != prior_tracks - selected_associations ||
      deleted_tracks > unmatched_prior_tracks || matches.size() != selected_associations ||
      matches.size() > 128 || gate_reject_examples.size() > 32 || conflict_examples.size() > 32 ||
      gate_reject_examples.size() > gate_reject_pairs ||
      conflict_examples.size() > gate_pass_pairs - selected_associations)
    throw std::invalid_argument("Invalid tracking diagnostics counts/bounds");
  auto encode = [](const AssociationDiagnostic &d, int kind) {
    const auto &s = d.innovation.covariance_m2;
    const auto &r = d.innovation.residual_m;
    const double determinant = s[0] * s[3] - s[1] * s[2];
    auto finite = [](const auto &a) {
      return std::all_of(a.begin(), a.end(), [](double v) { return std::isfinite(v); });
    };
    if (d.track_id == 0 || d.track_id > 9007199254740991ULL ||
        d.observation_id > 9007199254740991ULL || !finite(d.predicted_position_m) || !finite(r) ||
        !finite(s) || s[0] <= 0 || s[3] <= 0 || s[1] != s[2] || !std::isfinite(determinant) ||
        determinant <= 0 || !std::isfinite(d.innovation.mahalanobis_squared) ||
        d.innovation.mahalanobis_squared < 0 ||
        (kind == 0 &&
         (d.disposition != "selected" || d.innovation.mahalanobis_squared >= 9.21034)) ||
        (kind == 1 &&
         (d.disposition != "gate_rejected" || d.innovation.mahalanobis_squared < 9.21034)) ||
        (kind == 2 && ((d.disposition != "track_used" && d.disposition != "observation_used" &&
                        d.disposition != "both_used") ||
                       d.innovation.mahalanobis_squared >= 9.21034)))
      throw std::invalid_argument("Invalid tracking diagnostics pair");
    const double derived =
        (s[3] * r[0] * r[0] - 2 * s[1] * r[0] * r[1] + s[0] * r[1] * r[1]) / determinant;
    if (!std::isfinite(derived) || derived < 0 ||
        std::abs(derived - d.innovation.mahalanobis_squared) > 1e-12 + 1e-10 * std::abs(derived))
      throw std::invalid_argument("Inconsistent tracking innovation distance");
    return json{{"track_id", d.track_id},
                {"observation_id", d.observation_id},
                {"confirmed_at_association", d.confirmed_at_association},
                {"predicted_position_m", d.predicted_position_m},
                {"innovation_m", r},
                {"innovation_covariance_m2", s},
                {"mahalanobis_squared", d.innovation.mahalanobis_squared},
                {"disposition", d.disposition}};
  };
  json selected = json::array(), rejected = json::array(), conflicts = json::array();
  std::set<uint64_t> tracks, observations_used, prior_ids, observation_ids;
  std::set<std::pair<uint64_t, uint64_t>> pair_ids;
  auto register_pair = [&](const AssociationDiagnostic &d) {
    if (!pair_ids.emplace(d.track_id, d.observation_id).second)
      throw std::invalid_argument("Duplicate tracking diagnostic pair");
    prior_ids.insert(d.track_id);
    observation_ids.insert(d.observation_id);
  };
  for (const auto &d : matches) {
    if (!tracks.insert(d.track_id).second || !observations_used.insert(d.observation_id).second)
      throw std::invalid_argument("Non one-to-one tracking diagnostics");
    register_pair(d);
    selected.push_back(encode(d, 0));
  }
  for (const auto &d : gate_reject_examples) {
    register_pair(d);
    rejected.push_back(encode(d, 1));
  }
  uint32_t track_examples = 0, observation_examples = 0, both_examples = 0;
  for (const auto &d : conflict_examples) {
    register_pair(d);
    if (d.disposition == "track_used") {
      ++track_examples;
      if (!tracks.contains(d.track_id))
        throw std::invalid_argument("Conflict track not selected");
    } else if (d.disposition == "observation_used") {
      ++observation_examples;
      if (!observations_used.contains(d.observation_id))
        throw std::invalid_argument("Conflict observation not selected");
    } else if (d.disposition == "both_used") {
      ++both_examples;
      if (!tracks.contains(d.track_id) || !observations_used.contains(d.observation_id))
        throw std::invalid_argument("Conflict endpoints not selected");
    }
    conflicts.push_back(encode(d, 2));
  }
  if (prior_ids.size() > prior_tracks || observation_ids.size() > observations ||
      track_examples > skipped_track_only || observation_examples > skipped_observation_only ||
      both_examples > skipped_both)
    throw std::invalid_argument("Diagnostics example identity/count contradiction");
  json result = {{"schema_version", 1},
                 {"timestamp_ns", timestamp_ns},
                 {"coordinate_frame", "world_cartesian_2d"},
                 {"position_unit", "m"},
                 {"covariance_unit", "m2"},
                 {"distance_kind", "squared_mahalanobis_2d"},
                 {"gate_threshold_d2", 9.21034},
                 {"comparison", "strict_less_than"},
                 {"association_policy", "confirmed_first_greedy_stable"},
                 {"prior_tracks", prior_tracks},
                 {"observations", observations},
                 {"evaluated_pairs", evaluated_pairs},
                 {"gate_pass_pairs", gate_pass_pairs},
                 {"gate_reject_pairs", gate_reject_pairs},
                 {"selected_associations", selected_associations},
                 {"skipped_track_only", skipped_track_only},
                 {"skipped_observation_only", skipped_observation_only},
                 {"skipped_both", skipped_both},
                 {"born_tracks", born_tracks},
                 {"deleted_tracks", deleted_tracks},
                 {"unmatched_prior_tracks", unmatched_prior_tracks},
                 {"matches", std::move(selected)},
                 {"gate_reject_examples", std::move(rejected)},
                 {"conflict_examples", std::move(conflicts)},
                 {"gate_reject_examples_omitted", gate_reject_pairs - gate_reject_examples.size()},
                 {"conflict_examples_omitted",
                  gate_pass_pairs - selected_associations - conflict_examples.size()}};
  if (result.dump().size() > 128 * 1024)
    throw std::runtime_error("Tracking diagnostics exceed 128 KiB");
  return result;
}
TrackingDiagnostics TrackingDiagnostics::from_json(const nlohmann::json &value) {
  using nlohmann::json;
  auto fields = [](const json &v, const json &model) {
    if (!v.is_object() || v.size() != model.size())
      throw std::invalid_argument("Diagnostics fields");
    for (const auto &[key, item] : v.items()) {
      (void)item;
      if (!model.contains(key))
        throw std::invalid_argument("Unknown diagnostics field");
    }
  };
  const auto model = TrackingDiagnostics{}.to_json();
  fields(value, model);
  for (const char *name :
       {"schema_version", "coordinate_frame", "position_unit", "covariance_unit", "distance_kind",
        "gate_threshold_d2", "comparison", "association_policy"})
    if (value.at(name) != model.at(name))
      throw std::invalid_argument("Diagnostics constants/version");
  if (!value.at("schema_version").is_number_integer())
    throw std::invalid_argument("Diagnostics version type");
  auto integer = [](const json &v, uint64_t max) {
    if (!v.is_number_integer() || (!v.is_number_unsigned() && v.get<int64_t>() < 0))
      throw std::invalid_argument("Diagnostics unsigned integer type");
    auto result = v.get<uint64_t>();
    if (result > max)
      throw std::invalid_argument("Diagnostics integer bound");
    return result;
  };
  TrackingDiagnostics result;
  result.timestamp_ns =
      static_cast<int64_t>(integer(value.at("timestamp_ns"), 9007199254740991ULL));
#define COUNT(name, cap) result.name = static_cast<uint32_t>(integer(value.at(#name), cap))
  COUNT(prior_tracks, 128);
  COUNT(observations, 128);
  COUNT(evaluated_pairs, 16384);
  COUNT(gate_pass_pairs, 16384);
  COUNT(gate_reject_pairs, 16384);
  COUNT(selected_associations, 128);
  COUNT(skipped_track_only, 16384);
  COUNT(skipped_observation_only, 16384);
  COUNT(skipped_both, 16384);
  COUNT(born_tracks, 128);
  COUNT(deleted_tracks, 128);
  COUNT(unmatched_prior_tracks, 128);
#undef COUNT
  auto number = [](const json &v) {
    if (!v.is_number())
      throw std::invalid_argument("Diagnostics numerical type");
    auto n = v.get<double>();
    if (!std::isfinite(n))
      throw std::invalid_argument("Diagnostics nonfinite value");
    return n;
  };
  auto array = [&](const json &v, auto &out) {
    if (!v.is_array() || v.size() != out.size())
      throw std::invalid_argument("Diagnostics numeric shape");
    for (size_t i = 0; i < out.size(); ++i)
      out[i] = number(v[i]);
  };
  auto pairs = [&](const char *key, size_t maximum, std::vector<AssociationDiagnostic> &out) {
    const auto &list = value.at(key);
    if (!list.is_array() || list.size() > maximum)
      throw std::invalid_argument("Diagnostics list bound");
    const json shape = {{"track_id", 0},
                        {"observation_id", 0},
                        {"confirmed_at_association", false},
                        {"predicted_position_m", nullptr},
                        {"innovation_m", nullptr},
                        {"innovation_covariance_m2", nullptr},
                        {"mahalanobis_squared", 0},
                        {"disposition", ""}};
    for (const auto &v : list) {
      fields(v, shape);
      AssociationDiagnostic item;
      item.track_id = integer(v.at("track_id"), 9007199254740991ULL);
      item.observation_id = integer(v.at("observation_id"), 9007199254740991ULL);
      if (!v.at("confirmed_at_association").is_boolean())
        throw std::invalid_argument("Diagnostics confirmed type");
      item.confirmed_at_association = v.at("confirmed_at_association").get<bool>();
      array(v.at("predicted_position_m"), item.predicted_position_m);
      array(v.at("innovation_m"), item.innovation.residual_m);
      array(v.at("innovation_covariance_m2"), item.innovation.covariance_m2);
      item.innovation.mahalanobis_squared = number(v.at("mahalanobis_squared"));
      if (!v.at("disposition").is_string() ||
          v.at("disposition").get_ref<const std::string &>().size() > 24)
        throw std::invalid_argument("Diagnostics disposition bound/type");
      item.disposition = v.at("disposition").get<std::string>();
      out.push_back(std::move(item));
    }
  };
  pairs("matches", 128, result.matches);
  pairs("gate_reject_examples", 32, result.gate_reject_examples);
  pairs("conflict_examples", 32, result.conflict_examples);
  const auto canonical = result.to_json();
  for (const char *name : {"gate_reject_examples_omitted", "conflict_examples_omitted"})
    if (integer(value.at(name), 16384) != canonical.at(name).get<uint64_t>())
      throw std::invalid_argument("Diagnostics omitted count");
  // Every accepted shape/string/numeric field has already been bounded; dumping
  // this input now cannot allocate an unbounded tree/string or recurse deeply.
  if (value.dump().size() > 128 * 1024)
    throw std::invalid_argument("Diagnostics encoded size");
  return result;
}
} // namespace aether
