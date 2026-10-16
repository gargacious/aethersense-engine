#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
namespace aether {
struct Config;
struct SourceDeliveryProfile { uint32_t every_n{}, delay_frames{}; };
struct SourceOptions {
  int version{1};
  std::string mode{"independently_paced"};
  double rate_hz{};
  uint32_t queue_capacity{};
  std::string overflow;
  std::optional<SourceDeliveryProfile> delivery;
  void validate() const;
  nlohmann::json to_json() const;
  static SourceOptions from_json(const nlohmann::json&);
};
struct SourceLedger {
  int version{1};
  std::string scope, config_sha256;
  uint64_t generated{}, offered{}, accepted{}, overflow_rejected{}, failed_admission{};
  uint64_t cancelled_before_offer{}, failed_after_generation{}, processed_frames{};
  uint64_t late_rejected{}, cancelled_after_accept{}, failed_processing{};
  uint64_t ready_depth{}, delayed_depth{}, compute_active{};
  std::optional<uint64_t> last_generated_sequence, last_processed_sequence;
  std::optional<int64_t> last_processed_timestamp_ns;
  // Present only for run_cumulative_commit_fence, required there.
  std::optional<uint64_t> source_gap_before, late_rejected_since_previous_commit;
  void validate(const Config&, std::string_view expected_config_sha256) const;
  nlohmann::json to_json() const;
  static SourceLedger from_json(const nlohmann::json&, const Config&, std::string_view expected_hash);
};
struct SourceFrameCheckpoint {
  SourceLedger ledger;
  uint64_t sequence{};
  int64_t timestamp_ns{};
};
// Returns nullopt only when source_options absent and extension absent. Validates
// Frame envelope/quality bindings but does not replace full structural Frame checks.
std::optional<SourceFrameCheckpoint> source_frame_checkpoint(
    const nlohmann::json&, const Config&, std::string_view expected_hash);
void validate_source_frame_progress(const SourceFrameCheckpoint& previous,
                                   const SourceFrameCheckpoint& current);
void validate_source_ledger_progress(const SourceLedger& previous, const SourceLedger& current);
void validate_source_capture_first(const SourceLedger& capture_start, const SourceFrameCheckpoint& first);
struct SourceCaptureClose { SourceLedger ledger; std::string reason; };
struct SourceCaptureSummary {
  int version{1};
  std::string summary_status{"unavailable"};
  std::optional<SourceLedger> capture_start, capture_end;
  std::string close_reason;
  std::optional<uint64_t> first_recorded_sequence, last_recorded_sequence;
  uint64_t recorded_processed_frames{}, uncaptured_processed_at_close{};
  nlohmann::json to_json() const;
  static SourceCaptureSummary from_json(const nlohmann::json&, const Config&, std::string_view hash);
  void validate(const Config&, std::string_view hash, uint64_t frames, bool closed,
                const std::optional<SourceFrameCheckpoint>& first,
                const std::optional<SourceFrameCheckpoint>& last) const;
};
} // namespace aether
