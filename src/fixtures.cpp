#include <aethersense/fixtures.hpp>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
namespace aether {
namespace {
void require(bool v, const char *why) {
  if (!v)
    throw std::invalid_argument(why);
}
uint64_t integer(const json &j, uint64_t max) {
  require(j.is_number_integer(), "Raw integer field has wrong type");
  require(!j.is_number_integer() || j.is_number_unsigned() || j.get<int64_t>() >= 0,
          "Raw integer field is negative");
  auto v = j.get<uint64_t>();
  require(v <= max, "Raw integer field exceeds bounds");
  return v;
}
double positive(const json &j) {
  require(j.is_number(), "Raw metadata must be numeric");
  double x = j.get<double>();
  require(std::isfinite(x) && x > 0, "Raw metadata must be finite and positive");
  return x;
}
void envelope_valid(const Envelope &e) {
  require(e.version == 1 && e.clock_domain == "simulated" && e.timestamp_ns >= 0 &&
              e.receive_steady_ns >= 0,
          "Invalid raw envelope version or clock metadata");
}
} // namespace
json serialize_raw_block(const RawBlock &b) {
  b.validate();
  envelope_valid(b.envelope);
  json pairs = json::array();
  for (auto v : b.iq)
    pairs.push_back({v.real(), v.imag()});
  return {{"schema_version", 1},
          {"run_id", b.envelope.run_id},
          {"source_id", b.envelope.source_id},
          {"sequence_no", b.envelope.sequence_no},
          {"clock_domain", "simulated"},
          {"timestamp_ns", b.envelope.timestamp_ns},
          {"timestamp_origin", "simulation_epoch_zero"},
          {"receive_steady_ns", std::to_string(b.envelope.receive_steady_ns)},
          {"sample_format", "complex_float64"},
          {"packing", "channel_chirp_sample_I_plus_jQ"},
          {"samples_per_chirp", b.samples_per_chirp},
          {"chirps", b.chirps},
          {"channels", b.channels},
          {"sample_rate_hz", b.sample_rate_hz},
          {"chirp_period_s", b.chirp_period_s},
          {"slope_hz_s", b.slope_hz_s},
          {"wavelength_m", b.wavelength_m},
          {"iq", std::move(pairs)}};
}
RawBlock import_raw_block(const json &j) {
  static const std::set<std::string> keys{"schema_version",    "run_id",
                                          "source_id",         "sequence_no",
                                          "clock_domain",      "timestamp_ns",
                                          "timestamp_origin",  "receive_steady_ns",
                                          "sample_format",     "packing",
                                          "samples_per_chirp", "chirps",
                                          "channels",          "sample_rate_hz",
                                          "chirp_period_s",    "slope_hz_s",
                                          "wavelength_m",      "iq"};
  require(j.is_object() && j.size() == keys.size(), "Missing or unknown raw fields");
  for (auto it = j.begin(); it != j.end(); ++it)
    require(keys.contains(it.key()), "Unknown raw field");
  require(integer(j.at("schema_version"), 1) == 1, "Unsupported raw version");
  require(j.at("clock_domain") == "simulated" &&
              j.at("timestamp_origin") == "simulation_epoch_zero" &&
              j.at("sample_format") == "complex_float64" &&
              j.at("packing") == "channel_chirp_sample_I_plus_jQ",
          "Unsupported raw format, packing or clock");
  RawBlock b;
  require(j.at("run_id").is_string() && j.at("source_id").is_string(), "Raw IDs must be strings");
  b.envelope.run_id = j.at("run_id").get<std::string>();
  b.envelope.source_id = j.at("source_id").get<std::string>();
  b.envelope.sequence_no = integer(j.at("sequence_no"), UINT64_MAX);
  b.envelope.timestamp_ns = static_cast<int64_t>(integer(j.at("timestamp_ns"), INT64_MAX));
  require(j.at("receive_steady_ns").is_string(), "Raw receive clock must be decimal string");
  auto clock = j.at("receive_steady_ns").get<std::string>();
  require(!clock.empty() &&
              std::all_of(clock.begin(), clock.end(), [](char c) { return c >= '0' && c <= '9'; }),
          "Invalid raw receive clock");
  auto parsed =
      std::from_chars(clock.data(), clock.data() + clock.size(), b.envelope.receive_steady_ns);
  require(parsed.ec == std::errc{} && parsed.ptr == clock.data() + clock.size(),
          "Raw receive clock overflow");
  b.samples_per_chirp = static_cast<int>(integer(j.at("samples_per_chirp"), 1024));
  b.chirps = static_cast<int>(integer(j.at("chirps"), 64));
  b.channels = static_cast<int>(integer(j.at("channels"), 2));
  require(b.channels == 2 && b.samples_per_chirp >= 32 &&
              !(b.samples_per_chirp & (b.samples_per_chirp - 1)) && b.chirps >= 8 &&
              !(b.chirps & (b.chirps - 1)),
          "Unsupported raw shape");
  b.sample_rate_hz = positive(j.at("sample_rate_hz"));
  b.chirp_period_s = positive(j.at("chirp_period_s"));
  b.slope_hz_s = positive(j.at("slope_hz_s"));
  b.wavelength_m = positive(j.at("wavelength_m"));
  require(b.chirp_period_s >= b.samples_per_chirp / b.sample_rate_hz, "Inconsistent raw timing");
  const size_t size = static_cast<size_t>(b.samples_per_chirp) * b.chirps * b.channels;
  const auto &iq = j.at("iq");
  require(size <= 131072 && iq.is_array() && iq.size() == size, "Raw product/length mismatch");
  // Validate the complete document before allocating the numerical sample buffer.
  for (const auto &pair : iq) {
    require(pair.is_array() && pair.size() == 2, "Malformed I/Q pair");
    for (const auto &v : pair)
      require(v.is_number() && std::isfinite(v.get<double>()), "Nonfinite I/Q");
  }
  b.iq.reserve(size);
  for (const auto &pair : iq)
    b.iq.emplace_back(pair[0].get<double>(), pair[1].get<double>());
  b.validate();
  return b;
}
RawBlock parse_raw_block(std::string_view text) {
  require(text.size() <= 12 * 1024 * 1024, "Raw JSON exceeds 12 MiB limit");
  // Reject duplicate keys rather than accepting JSON's last-value-wins ambiguity.
  std::vector<std::set<std::string>> objects;
  auto callback = [&](int depth, json::parse_event_t event, json &value) {
    require(depth <= 8, "Raw JSON nesting exceeds limit");
    if (event == json::parse_event_t::object_start)
      objects.emplace_back();
    if (event == json::parse_event_t::key)
      require(objects.back().insert(value.get<std::string>()).second, "Duplicate raw JSON field");
    if (event == json::parse_event_t::object_end)
      objects.pop_back();
    return true;
  };
  return import_raw_block(json::parse(text, callback));
}
json observation_fixture(std::string_view scenario, uint32_t seed) {
  require(scenario == "single_stationary" || scenario == "crossing_paths" ||
              scenario == "intermittent_observation",
          "Unknown golden scenario");
  Config c;
  c.scenario = scenario;
  c.seed = seed;
  c.objects = scenario == "single_stationary" ? 1 : 2;
  Random rng(seed);
  json frames = json::array();
  for (uint64_t frame = 0; frame < 55; ++frame) {
    auto truth = truth_at(c, frame * c.dt_s);
    auto obs = simulate_detections(c, truth, rng, frame);
    json truth_json = json::array(), observations = json::array();
    for (const auto &t : truth)
      truth_json.push_back(serialize(t));
    for (const auto &d : obs)
      observations.push_back(serialize(d));
    frames.push_back({{"timestamp_ns", static_cast<int64_t>(std::llround(frame * c.dt_s * 1e9))},
                      {"observations", observations},
                      {"evaluation_truth", truth_json}});
  }
  return {{"schema_version", 1},
          {"fixture_type", "normalized_observation_golden"},
          {"world_frame", "x_right_y_forward"},
          {"clock_domain", "simulated"},
          {"config", c.to_json()},
          {"frames", frames}};
}
RawBlock analytic_tone_fixture() {
  RawBlock b;
  b.samples_per_chirp = 32;
  b.chirps = 8;
  b.envelope.run_id = "analytic-tone-v1";
  for (int ch = 0; ch < 2; ++ch)
    for (int m = 0; m < 8; ++m)
      for (int n = 0; n < 32; ++n)
        b.iq.push_back(
            std::polar(1., 2 * std::numbers::pi * (5. * n / 32 + 2. * m / 8) + ch * .25));
  b.validate();
  return b;
}
} // namespace aether
