#pragma once
#include <array>
#include <complex>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace aether {
using json = nlohmann::json;
constexpr int schema_version = 1;
struct Envelope {
  int version{1};
  std::string run_id, source_id{"synthetic-1"}, clock_domain{"simulated"};
  uint64_t sequence_no{};
  int64_t timestamp_ns{}, receive_steady_ns{};
};
struct Config {
  std::string scenario{"crossing_paths"}, input{"synthetic_detections"}, window{"hann"};
  uint32_t seed{42};
  int objects{3}, fft_size{128}, chirps{32}, confirmation_hits{3}, coast_frames{5};
  double duration_s{60}, dt_s{0.1}, noise_m{0.6}, dropout{0.05}, false_rate{0.1};
  int cfar_training{6}, cfar_guard{2};
  double cfar_pfa{0.001};
  void validate() const;
  json to_json() const;
  static Config from_json(const json &);
};
struct Truth {
  int id;
  double x, y, vx, vy;
};
struct Detection {
  uint64_t id;
  double x, y, var_x, var_y;
  double range_m{}, radial_velocity_mps{}, confidence{1};
};
struct Track {
  uint64_t id{};
  std::array<double, 4> state{};
  std::array<double, 16> covariance{};
  int hits{1}, misses{}, age{1};
  bool confirmed{};
  int64_t first_ns{}, last_ns{};
  std::vector<uint64_t> matched;
};
struct RawBlock {
  Envelope envelope;
  int samples_per_chirp{}, chirps{}, channels{2};
  double sample_rate_hz{1e6}, chirp_period_s{0.001}, slope_hz_s{1e12}, wavelength_m{0.004};
  std::vector<std::complex<double>> iq; // [channel][chirp][sample], complex floating-point I+jQ
  void validate() const;
};
struct SignalResult {
  int bins{}, chirps{};
  double range_bin_m{}, velocity_bin_mps{}, sample_rate_hz{}, chirp_period_s{};
  std::vector<double> i, q, spectrum, phase, range_profile, threshold, range_doppler;
  std::vector<Detection> detections;
  json to_json() const;
};
class Random {
  std::mt19937 engine_;

public:
  explicit Random(uint32_t seed) : engine_(seed){};
  double uniform();
  double normal();
};
std::vector<Truth> truth_at(const Config &, double t);
std::vector<Detection> simulate_detections(const Config &, std::span<const Truth>, Random &,
                                           uint64_t sequence);
RawBlock synthesize_iq(const Config &, std::span<const Truth>, Random &, const Envelope &);
void fft(std::span<std::complex<double>>, bool inverse = false);
std::vector<std::complex<double>> matched_filter(std::span<const std::complex<double>>,
                                                 std::span<const std::complex<double>>);
std::vector<double> ca_cfar(std::span<const double>, int training, int guard, double pfa);
SignalResult process_iq(const RawBlock &, const Config &);
class Tracker {
  std::vector<Track> tracks_;
  uint64_t next_id_{1};
  int64_t previous_ns_{-1};

public:
  void update(std::span<const Detection>, int64_t timestamp_ns, const Config &);
  const std::vector<Track> &tracks() const { return tracks_; }
};
json serialize(const Track &);
json serialize(const Detection &);
json serialize(const Truth &);
std::string
    fingerprint(std::string_view); // FNV-1a config identifier, not an integrity/security checksum
} // namespace aether
