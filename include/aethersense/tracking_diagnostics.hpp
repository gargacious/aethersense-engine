#pragma once
#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace aether {
struct TrackingDiagnosticsOptions {
  int version{1};
  bool enabled{true};
};
struct TrackingInnovation {
  std::array<double, 2> residual_m;
  std::array<double, 4> covariance_m2;
  double mahalanobis_squared{};
};
// Same pre-update calculation used by Tracker; diagonal observation covariance.
// Predicted covariance may be correlated; inputs/S and distance must be finite/PD.
TrackingInnovation tracking_innovation(std::array<double, 2> predicted_position,
                                       std::array<double, 4> predicted_covariance,
                                       std::array<double, 2> observation_position,
                                       std::array<double, 2> observation_variance);
struct AssociationDiagnostic {
  uint64_t track_id{}, observation_id{};
  bool confirmed_at_association{};
  std::array<double, 2> predicted_position_m{};
  TrackingInnovation innovation;
  std::string disposition;
};
struct TrackingDiagnostics {
  int64_t timestamp_ns{};
  uint32_t prior_tracks{}, observations{}, evaluated_pairs{}, gate_pass_pairs{},
      gate_reject_pairs{};
  uint32_t selected_associations{}, skipped_track_only{}, skipped_observation_only{},
      skipped_both{};
  uint32_t born_tracks{}, deleted_tracks{}, unmatched_prior_tracks{};
  std::vector<AssociationDiagnostic> matches, gate_reject_examples, conflict_examples;
  nlohmann::json to_json() const;
  static TrackingDiagnostics from_json(const nlohmann::json &);
};
} // namespace aether
