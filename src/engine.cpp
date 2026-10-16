#include <aethersense/engine.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <set>
#include <stdexcept>
#include <tuple>

namespace aether {
constexpr double pi = std::numbers::pi;
namespace {
void scenario_keys(const json &value, std::initializer_list<std::string_view> allowed,
                   std::initializer_list<std::string_view> required) {
  if (!value.is_object())
    throw std::invalid_argument("Scenario override must be an object");
  for (auto it = value.begin(); it != value.end(); ++it)
    if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end())
      throw std::invalid_argument("Unknown scenario override field: " + it.key());
  for (auto key : required)
    if (!value.contains(key))
      throw std::invalid_argument("Missing scenario override field: " + std::string(key));
}
double scenario_number(const json &value, const char *key) {
  if (!value.at(key).is_number())
    throw std::invalid_argument("Scenario field must be numeric");
  const auto number = value.at(key).get<double>();
  if (!std::isfinite(number))
    throw std::invalid_argument("Scenario field must be finite");
  return number;
}
uint32_t scenario_integer(const json &value, const char *key, uint32_t maximum) {
  const auto &number = value.at(key);
  if (!number.is_number_integer() || number.get<double>() < 0 || number.get<double>() > maximum)
    throw std::invalid_argument("Scenario window integer out of bounds");
  return number.get<uint32_t>();
}
ScenarioWindow scenario_window(const json &value) {
  return {scenario_integer(value, "start_frame", 179999),
          scenario_integer(value, "frame_count", 180000)};
}
bool scenario_active(const ScenarioWindow &window, uint64_t sequence) {
  return sequence >= window.start_frame &&
         sequence < uint64_t(window.start_frame) + window.frame_count;
}
ScenarioOverrides parse_scenario(const json &value) {
  scenario_keys(value, {"schema_version", "motion", "occlusion", "false_burst", "high_noise"},
                {"schema_version"});
  if (!value.at("schema_version").is_number_integer() || value.at("schema_version") != 1)
    throw std::invalid_argument("Unsupported scenario override version");
  ScenarioOverrides result;
  if (value.contains("motion")) {
    const auto &m = value.at("motion");
    scenario_keys(m, {"mode", "stationary_s", "linear_s", "turn_rate_rad_s"},
                  {"mode", "stationary_s", "linear_s", "turn_rate_rad_s"});
    if (!m.at("mode").is_string())
      throw std::invalid_argument("Motion mode must be a string");
    result.motion =
        ScenarioMotion{m.at("mode").get<std::string>(), scenario_number(m, "stationary_s"),
                       scenario_number(m, "linear_s"), scenario_number(m, "turn_rate_rad_s")};
  }
  if (value.contains("occlusion")) {
    const auto &w = value.at("occlusion");
    scenario_keys(w, {"start_frame", "frame_count"}, {"start_frame", "frame_count"});
    result.occlusion = scenario_window(w);
  }
  if (value.contains("false_burst")) {
    const auto &w = value.at("false_burst");
    scenario_keys(w, {"start_frame", "frame_count", "false_rate"},
                  {"start_frame", "frame_count", "false_rate"});
    result.false_burst = ScenarioFalseBurst{scenario_window(w), scenario_number(w, "false_rate")};
  }
  if (value.contains("high_noise")) {
    const auto &w = value.at("high_noise");
    scenario_keys(w, {"start_frame", "frame_count", "noise_m"},
                  {"start_frame", "frame_count", "noise_m"});
    result.high_noise = ScenarioHighNoise{scenario_window(w), scenario_number(w, "noise_m")};
  }
  return result;
}
json serialize_scenario(const ScenarioOverrides &value) {
  json result = {{"schema_version", value.version}};
  if (value.motion) {
    const auto &m = *value.motion;
    result["motion"] = {{"mode", m.mode},
                        {"stationary_s", m.stationary_s},
                        {"linear_s", m.linear_s},
                        {"turn_rate_rad_s", m.turn_rate_rad_s}};
  }
  if (value.occlusion) {
    const auto &w = *value.occlusion;
    result["occlusion"] = {{"start_frame", w.start_frame}, {"frame_count", w.frame_count}};
  }
  if (value.false_burst) {
    const auto &w = *value.false_burst;
    result["false_burst"] = {{"start_frame", w.start_frame},
                             {"frame_count", w.frame_count},
                             {"false_rate", w.false_rate}};
  }
  if (value.high_noise) {
    const auto &w = *value.high_noise;
    result["high_noise"] = {
        {"start_frame", w.start_frame}, {"frame_count", w.frame_count}, {"noise_m", w.noise_m}};
  }
  return result;
}
} // namespace
void Config::validate() const {
  if (source_options) {
    source_options->validate();
    if (executor != "staged_threads")
      throw std::invalid_argument("Independently paced source requires staged_threads");
  }
  if (executor != "single_thread" && executor != "staged_threads")
    throw std::invalid_argument("Executor must be single_thread or staged_threads");
  if (queue_capacity < 1 || queue_capacity > 64)
    throw std::invalid_argument("Queue capacity must be 1..64");
  if (scenario != "single_stationary" && scenario != "crossing_paths" &&
      scenario != "intermittent_observation" && scenario != "noise_sweep" &&
      scenario != "overload_test")
    throw std::invalid_argument("Unknown scenario");
  if (input != "synthetic_detections" && input != "synthetic_iq")
    throw std::invalid_argument("Only synthetic_detections and synthetic_iq are available");
  if (objects < 1 || objects > 32 || !std::isfinite(duration_s) || duration_s <= 0 ||
      duration_s > 3600 || !std::isfinite(dt_s) || dt_s < 0.02 || dt_s > 1 ||
      !std::isfinite(noise_m) || noise_m < 0.01 || noise_m > 20 || !std::isfinite(dropout) ||
      dropout < 0 || dropout > 1 || !std::isfinite(false_rate) || false_rate < 0 || false_rate > 10)
    throw std::invalid_argument(
        "Invalid scenario dimensions, duration, noise, dropout or false rate");
  if (fft_size < 32 || fft_size > 1024 || (fft_size & (fft_size - 1)) || chirps < 8 ||
      chirps > 64 || (chirps & (chirps - 1)))
    throw std::invalid_argument(
        "FFT size must be power-of-two 32..1024; chirps power-of-two 8..64");
  if (window != "hann" && window != "rectangular")
    throw std::invalid_argument("Window must be hann or rectangular");
  if (cfar_training < 1 || cfar_training > 32 || cfar_guard < 0 || cfar_guard > 8 ||
      2 * (cfar_training + cfar_guard) + 1 >= fft_size || !std::isfinite(cfar_pfa) ||
      cfar_pfa < 1e-6 || cfar_pfa > 0.1)
    throw std::invalid_argument("Invalid CA-CFAR cell sizes or Pfa (1e-6..0.1)");
  if (confirmation_hits < 1 || confirmation_hits > 20 || coast_frames < 0 || coast_frames > 100)
    throw std::invalid_argument("Invalid track lifecycle thresholds");
  if (tracking_diagnostics && (tracking_diagnostics->version != 1 || !tracking_diagnostics->enabled))
    throw std::invalid_argument("Tracking diagnostics require version1 and enabled true");
  if (recording_options && (recording_options->version != 1 ||
      recording_options->mode != "async_worker" || recording_options->queue_capacity < 1 ||
      recording_options->queue_capacity > 8))
    throw std::invalid_argument("Invalid recording options version/mode/capacity");
  if (iq_processing) {
    if (input != "synthetic_iq")
      throw std::invalid_argument("IQ processing options require synthetic_iq");
    iq_processing->validate(fft_size, chirps);
  }
  if (scenario_overrides) {
    const auto &o = *scenario_overrides;
    if (o.version != 1 || input != "synthetic_detections")
      throw std::invalid_argument("Scenario overrides v1 support synthetic_detections only");
    if (o.motion) {
      const auto &m = *o.motion;
      if (m.mode != "stationary_linear_turn" || !std::isfinite(m.stationary_s) ||
          m.stationary_s < 0 || m.stationary_s > 3600 || !std::isfinite(m.linear_s) ||
          m.linear_s < 0 || m.linear_s > 3600 || m.stationary_s + m.linear_s > duration_s ||
          !std::isfinite(m.turn_rate_rad_s) || std::abs(m.turn_rate_rad_s) > .5)
        throw std::invalid_argument("Invalid stationary/linear/turn scenario phases");
    }
    const auto total_frames = static_cast<uint64_t>(std::ceil(duration_s / dt_s));
    const auto check_window = [&](const ScenarioWindow &w) {
      if (w.start_frame > 179999 || w.frame_count < 1 || w.frame_count > 180000 ||
          uint64_t(w.start_frame) + w.frame_count > 180000 ||
          uint64_t(w.start_frame) + w.frame_count > total_frames)
        throw std::invalid_argument("Scenario window extends outside the configured run");
    };
    if (o.occlusion)
      check_window(*o.occlusion);
    if (o.false_burst) {
      check_window(*o.false_burst);
      if (!std::isfinite(o.false_burst->false_rate) || o.false_burst->false_rate < 0 ||
          o.false_burst->false_rate > 10)
        throw std::invalid_argument("Invalid burst false rate");
    }
    if (o.high_noise) {
      check_window(*o.high_noise);
      if (!std::isfinite(o.high_noise->noise_m) || o.high_noise->noise_m < .01 ||
          o.high_noise->noise_m > 20)
        throw std::invalid_argument("Invalid scenario noise sigma");
    }
  }
}
json Config::to_json() const {
  json result = {{"schema_version", 1},
                 {"executor", executor},
                 {"queue_capacity", queue_capacity},
                 {"scenario", scenario},
                 {"input", input},
                 {"seed", seed},
                 {"objects", objects},
                 {"duration_s", duration_s},
                 {"dt_s", dt_s},
                 {"noise_m", noise_m},
                 {"dropout", dropout},
                 {"false_rate", false_rate},
                 {"fft_size", fft_size},
                 {"chirps", chirps},
                 {"window", window},
                 {"cfar_training", cfar_training},
                 {"cfar_guard", cfar_guard},
                 {"cfar_pfa", cfar_pfa},
                 {"confirmation_hits", confirmation_hits},
                 {"coast_frames", coast_frames}};
  if (scenario_overrides)
    result["scenario_overrides"] = serialize_scenario(*scenario_overrides);
  if (iq_processing)
    result["iq_processing"] = iq_processing->to_json();
  if (tracking_diagnostics)
    result["tracking_diagnostics"]={{"schema_version",tracking_diagnostics->version},{"enabled",tracking_diagnostics->enabled}};
  if (recording_options)
    result["recording_options"] = {{"schema_version",recording_options->version},
      {"mode",recording_options->mode},{"queue_capacity",recording_options->queue_capacity}};
  if (source_options) result["source_options"] = source_options->to_json();
  return result;
}
Config Config::from_json(const json &j) {
  if (!j.is_object())
    throw std::invalid_argument("Configuration must be an object");
  Config c;
  for (const auto *key :
       {"schema_version", "seed", "objects", "fft_size", "chirps", "cfar_training", "cfar_guard",
        "confirmation_hits", "coast_frames", "queue_capacity"}) {
    if (j.contains(key) && !j.at(key).is_number_integer())
      throw std::invalid_argument(std::string(key) + " must be an integer");
    if (j.contains(key) && std::string_view(key) != "seed" &&
        (j.at(key).get<double>() < std::numeric_limits<int>::min() ||
         j.at(key).get<double>() > std::numeric_limits<int>::max()))
      throw std::invalid_argument(std::string(key) + " exceeds signed integer range");
  }
  if (j.contains("seed") &&
      (j.at("seed").get<double>() < 0 || j.at("seed").get<double>() > 4294967295.))
    throw std::invalid_argument("Seed must be an unsigned 32-bit integer");
  auto allowed = c.to_json();
  allowed["scenario_overrides"] = json::object();
  allowed["iq_processing"] = json::object();
  allowed["recording_options"] = json::object();
  allowed["tracking_diagnostics"] = json::object();
  allowed["source_options"] = json::object();
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!allowed.contains(it.key()))
      throw std::invalid_argument("Unknown configuration field: " + it.key());
  if (j.value("schema_version", 1) != 1)
    throw std::invalid_argument("Unsupported schema version");
#define FIELD(x)                                                                                   \
  if (j.contains(#x))                                                                              \
  j.at(#x).get_to(c.x)
  FIELD(executor);
  FIELD(queue_capacity);
  FIELD(scenario);
  FIELD(input);
  FIELD(seed);
  FIELD(objects);
  FIELD(duration_s);
  FIELD(dt_s);
  FIELD(noise_m);
  FIELD(dropout);
  FIELD(false_rate);
  FIELD(fft_size);
  FIELD(chirps);
  FIELD(window);
  FIELD(cfar_training);
  FIELD(cfar_guard);
  FIELD(cfar_pfa);
  FIELD(confirmation_hits);
  FIELD(coast_frames);
  if (j.contains("scenario_overrides"))
    c.scenario_overrides = parse_scenario(j.at("scenario_overrides"));
  if (j.contains("iq_processing"))
    c.iq_processing = IQProcessingConfig::from_json(j.at("iq_processing"));
  if (j.contains("tracking_diagnostics")) {
    const auto &d=j.at("tracking_diagnostics");
    scenario_keys(d,{"schema_version","enabled"},{"schema_version","enabled"});
    if (!d.at("enabled").is_boolean() || d.at("enabled") != true)
      throw std::invalid_argument("Tracking diagnostics enabled must be true");
    TrackingDiagnosticsOptions options;
    options.version=static_cast<int>(scenario_integer(d,"schema_version",1));
    c.tracking_diagnostics=options;
  }
  if (j.contains("source_options")) c.source_options = SourceOptions::from_json(j.at("source_options"));
  if (j.contains("recording_options")) {
    const auto &r=j.at("recording_options");
    scenario_keys(r,{"schema_version","mode","queue_capacity"},{"schema_version","mode","queue_capacity"});
    RecordingOptions o;
    o.version=static_cast<int>(scenario_integer(r,"schema_version",1));
    o.queue_capacity=static_cast<int>(scenario_integer(r,"queue_capacity",8));
    if (!r.at("mode").is_string()) throw std::invalid_argument("Recording mode must be a string");
    o.mode=r.at("mode").get<std::string>();c.recording_options=std::move(o);
  }
#undef FIELD
  c.validate();
  return c;
}
double Random::uniform() { return (static_cast<double>(engine_()) + 0.5) / 4294967296.; }
double Random::normal() {
  return std::sqrt(-2 * std::log(uniform())) * std::cos(2 * pi * uniform());
}
std::string fingerprint(std::string_view s) {
  uint64_t h = 14695981039346656037ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  std::ostringstream o;
  o << std::hex << std::setw(16) << std::setfill('0') << h;
  return o.str();
}
std::vector<Truth> truth_at(const Config &c, double t) {
  if (!std::isfinite(t) || t < 0 || t > 3600)
    throw std::invalid_argument("Invalid scenario source time");
  std::vector<Truth> out;
  for (int k = 0; k < c.objects; ++k) {
    double x = -25. + k * 18., y = 35. + (k % 3) * 12., vx = k % 2 ? -1.1 : 1.2,
           vy = (k % 3 - 1) * 0.15;
    if (c.scenario == "single_stationary") {
      x = 10. + k * 8;
      y = 45;
      vx = vy = 0;
    }
    // The existing scenario supplies the initial position and nominal velocity.
    // In particular single_stationary has zero nominal velocity in every phase.
    if (c.scenario_overrides && c.scenario_overrides->motion) {
      const auto &m = *c.scenario_overrides->motion;
      if (t < m.stationary_s) {
        vx = vy = 0;
      } else if (t < m.stationary_s + m.linear_s) {
        const auto elapsed = t - m.stationary_s;
        x += vx * elapsed;
        y += vy * elapsed;
      } else {
        x += vx * m.linear_s;
        y += vy * m.linear_s;
        const auto elapsed = t - m.stationary_s - m.linear_s, rate = m.turn_rate_rad_s;
        if (rate == 0) {
          x += vx * elapsed;
          y += vy * elapsed;
        } else {
          const auto angle = rate * elapsed, sine = std::sin(angle), cosine = std::cos(angle);
          // Ratios use the angle, not tiny rate, preserving the linear limit even
          // when rate is subnormal. The half-angle identity avoids 1-cos loss.
          const auto half_angle = angle / 2;
          const auto sinc = angle == 0 ? 1. : sine / angle;
          const auto half_sinc = half_angle == 0 ? 1. : std::sin(half_angle) / half_angle;
          const auto along = elapsed * sinc;
          const auto across = elapsed * half_angle * half_sinc * half_sinc;
          x += vx * along - vy * across;
          y += vx * across + vy * along;
          const auto rotated_x = vx * cosine - vy * sine;
          vy = vx * sine + vy * cosine;
          vx = rotated_x;
        }
      }
      out.push_back({k + 1, x, y, vx, vy});
    } else
      out.push_back({k + 1, x + vx * t, y + vy * t, vx, vy});
  }
  return out;
}
std::vector<Detection> simulate_detections(const Config &c, std::span<const Truth> truth,
                                           Random &rng, uint64_t seq) {
  // Check the complete truth batch before drawing any noise. Radial velocity at
  // the exact sensor origin is undefined, so reject rather than inventing zero.
  for (const auto &t : truth) {
    const auto radius = std::hypot(t.x, t.y);
    if (!std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(t.vx) ||
        !std::isfinite(t.vy) || !std::isfinite(radius) || radius <= 0 ||
        !std::isfinite((t.x * t.vx + t.y * t.vy) / radius))
      throw std::invalid_argument("Undefined or nonfinite truth radial geometry");
  }
  std::vector<Detection> out;
  uint64_t id = seq * 128;
  double noise = c.noise_m * (c.scenario == "noise_sweep" ? 1. + (seq % 200) / 50. : 1.);
  bool occluded = false;
  double false_rate = c.false_rate;
  if (c.scenario_overrides) {
    const auto &o = *c.scenario_overrides;
    occluded = o.occlusion && scenario_active(*o.occlusion, seq);
    if (o.high_noise && scenario_active(*o.high_noise, seq))
      noise = o.high_noise->noise_m;
    if (o.false_burst && scenario_active(*o.false_burst, seq))
      false_rate = o.false_burst->false_rate;
  }
  for (auto &t : truth) {
    if (rng.uniform() < c.dropout ||
        (c.scenario == "intermittent_observation" && seq % 100 >= 35 && seq % 100 < 50) || occluded)
      continue;
    double x = t.x + noise * rng.normal(), y = t.y + noise * rng.normal();
    out.push_back({++id, x, y, noise * noise, noise * noise, std::hypot(x, y),
                   (t.x * t.vx + t.y * t.vy) / std::hypot(t.x, t.y), 0.95});
  }
  for (int k = 0; k < 10; ++k)
    if (rng.uniform() < false_rate / 10.) {
      double x = (rng.uniform() - .5) * 140, y = 10 + rng.uniform() * 100;
      out.push_back({++id, x, y, 4, 4, std::hypot(x, y), 0, 0.3});
    }
  return out;
}
void RawBlock::validate() const {
  if (samples_per_chirp < 32 || samples_per_chirp > 1024 ||
      (samples_per_chirp & (samples_per_chirp - 1)) || chirps < 8 || chirps > 64 ||
      (chirps & (chirps - 1)) || channels != 2 ||
      iq.size() != static_cast<size_t>(samples_per_chirp * chirps * channels))
    throw std::invalid_argument("Malformed raw shape: requires two known ULA channels");
  if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0 || !std::isfinite(chirp_period_s) ||
      chirp_period_s <= 0 || !std::isfinite(slope_hz_s) || slope_hz_s <= 0 ||
      !std::isfinite(wavelength_m) || wavelength_m <= 0 ||
      sample_rate_hz * chirp_period_s < samples_per_chirp)
    throw std::invalid_argument("Missing or invalid raw sample timing/calibration metadata");
  for (auto x : iq)
    if (!std::isfinite(x.real()) || !std::isfinite(x.imag()))
      throw std::invalid_argument("Nonfinite I/Q sample");
}
RawBlock synthesize_iq(const Config &c, std::span<const Truth> truth, Random &rng,
                       const Envelope &e) {
  RawBlock b;
  b.envelope = e;
  b.samples_per_chirp = c.fft_size;
  b.chirps = c.chirps;
  b.chirp_period_s = std::max(.001, c.fft_size / 1e6);
  b.iq.resize(2 * c.fft_size * c.chirps);
  for (int ch = 0; ch < 2; ++ch)
    for (int m = 0; m < c.chirps; ++m)
      for (int n = 0; n < c.fft_size; ++n) {
        std::complex<double> v{.07 * rng.normal(), .07 * rng.normal()};
        for (auto &t : truth) {
          double r = std::hypot(t.x, t.y), vr = (t.x * t.vx + t.y * t.vy) / r;
          double phase = 2 * pi *
                             (2 * b.slope_hz_s * r / 299792458. * n / b.sample_rate_hz +
                              2 * vr / b.wavelength_m * m * b.chirp_period_s) +
                         ch * pi * t.x / r;
          v += std::polar(1., phase);
        }
        b.iq[(ch * c.chirps + m) * c.fft_size + n] = v;
      }
  return b;
}
void fft(std::span<std::complex<double>> a, bool inverse) {
  size_t n = a.size();
  if (n == 0 || (n & (n - 1)) || n > standalone_fft_max_elements)
    throw std::invalid_argument("FFT length must be power of two in [1, 2^20]");
  const auto finite = [](std::complex<double> v) {
    return std::isfinite(v.real()) && std::isfinite(v.imag());
  };
  for (const auto v : a)
    if (!finite(v))
      throw std::invalid_argument("FFT input must be finite");
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len *= 2) {
    auto wlen = std::polar(1., (inverse ? 2 : -2) * pi / len);
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w = 1;
      for (size_t j = 0; j < len / 2; ++j) {
        auto u = a[i + j], v = a[i + j + len / 2] * w;
        const auto sum = u + v, difference = u - v;
        if (!finite(v) || !finite(sum) || !finite(difference))
          throw std::overflow_error("FFT arithmetic is nonfinite; discard buffer");
        a[i + j] = sum;
        a[i + j + len / 2] = difference;
        w *= wlen;
        if (!finite(w))
          throw std::overflow_error("FFT twiddle arithmetic is nonfinite; discard buffer");
      }
    }
    if (len == n)
      break; // Never shift/multiply a terminal size, even on a narrow size_t.
  }
  if (inverse)
    for (auto &v : a) {
      v /= n;
      if (!finite(v))
        throw std::overflow_error("FFT normalization is nonfinite; discard buffer");
    }
}
std::vector<std::complex<double>> matched_filter(std::span<const std::complex<double>> a,
                                                 std::span<const std::complex<double>> ref) {
  if (a.empty() || ref.empty() || a.size() > matched_filter_max_input_elements ||
      ref.size() > matched_filter_max_input_elements)
    throw std::invalid_argument("Matched-filter input length must be in [1, 2^20]");
  // Subtraction/division avoid overflowing size arithmetic before allocating output.
  if (a.size() > matched_filter_max_output_elements - (ref.size() - 1) ||
      a.size() > matched_filter_max_operations / ref.size())
    throw std::invalid_argument("Matched-filter output or operation budget exceeded");
  const auto finite = [](std::complex<double> v) {
    return std::isfinite(v.real()) && std::isfinite(v.imag());
  };
  for (const auto input : {a, ref})
    for (const auto value : input)
      if (!finite(value))
        throw std::invalid_argument("Matched-filter inputs must be finite");
  std::vector<std::complex<double>> out(a.size() + ref.size() - 1);
  for (size_t n = 0; n < a.size(); ++n)
    for (size_t k = 0; k < ref.size(); ++k) {
      const auto product = a[n] * std::conj(ref[ref.size() - 1 - k]);
      const auto sum = out[n + k] + product;
      if (!finite(product) || !finite(sum))
        throw std::overflow_error("Matched-filter arithmetic is nonfinite");
      out[n + k] = sum;
    }
  return out;
}
std::vector<double> ca_cfar(std::span<const double> p, int training, int guard, double pfa) {
  if (training < 1 || guard < 0 || !std::isfinite(pfa) || pfa <= 0 || pfa >= 1 ||
      p.empty() || p.size() > standalone_cfar_max_elements)
    throw std::invalid_argument("Invalid CFAR configuration");
  const auto training_size = static_cast<std::size_t>(training);
  const auto guard_size = static_cast<std::size_t>(guard);
  const auto half_available = (p.size() - 1) / 2;
  // Validate by subtraction before adding margins or doubling training counts.
  if (training_size > half_available || guard_size > half_available - training_size)
    throw std::invalid_argument("CFAR training/guard margin exceeds input");
  const auto margin = training_size + guard_size;
  const auto training_count = 2 * training_size;
  for (auto v : p)
    if (!std::isfinite(v) || v < 0)
      throw std::invalid_argument("CFAR input must be finite nonnegative power");
  const double alpha = training_count * (std::pow(pfa, -1. / training_count) - 1);
  if (!std::isfinite(alpha))
    throw std::overflow_error("CFAR alpha is nonfinite");
  std::vector<double> out(p.size(), -1);
  for (std::size_t i = margin; i < p.size() - margin; ++i) {
    double sum = 0;
    for (std::size_t k = guard_size + 1; k <= margin; ++k) {
      sum += p[i - k] + p[i + k];
      if (!std::isfinite(sum))
        throw std::overflow_error("CFAR training sum is nonfinite");
    }
    const double scaled = alpha * sum;
    const double threshold = scaled / training_count;
    if (!std::isfinite(scaled) || !std::isfinite(threshold))
      throw std::overflow_error("CFAR threshold is nonfinite");
    out[i] = threshold;
  }
  return out;
}
json SignalResult::to_json() const {
  json result = {{"schema_version", 1},
          {"samples", bins},
          {"chirps", chirps},
          {"range_bin_m", range_bin_m},
          {"velocity_bin_mps", velocity_bin_mps},
          {"sample_rate_hz", sample_rate_hz},
          {"chirp_period_s", chirp_period_s},
          {"i", i},
          {"q", q},
          {"spectrum", spectrum},
          {"phase", phase},
          {"range_profile_power", range_profile},
          {"threshold_power", threshold},
          {"range_doppler_power", range_doppler},
          {"normalization",
           "DC removed; window coherent-gain corrected; FFT/N; power=|complex amplitude|^2"},
          {"cfar_edge_policy", "skip; threshold -1 means unavailable"},
          {"provenance", "SIMULATED educational FMCW; two calibrated half-wavelength ULA channels; "
                         "uncalibrated model"}};
  if (processing) result["processing"] = processing->to_json();
  if (!range_doppler_threshold.empty())
    result["range_doppler_threshold_power"] = range_doppler_threshold;
  return result;
}

void Tracker::update(std::span<const Detection> obs, int64_t ns, const Config &c) {
  update(obs, ns, c, nullptr);
}
void Tracker::update(std::span<const Detection> obs, int64_t ns, const Config &c,
                     std::optional<TrackingDiagnostics> *output) {
  if (output)
    output->reset();
  if (c.tracking_diagnostics &&
      (c.tracking_diagnostics->version != 1 || !c.tracking_diagnostics->enabled))
    throw std::invalid_argument("Invalid tracking diagnostics options");
  if (ns < 0 || obs.size() > 128)
    throw std::invalid_argument("Invalid observation timestamp/capacity");
  std::set<uint64_t> ids;
  for (const auto &d : obs)
    if (!ids.insert(d.id).second)
      throw std::invalid_argument("Duplicate observation identifier");
  const bool collect = output && c.tracking_diagnostics && c.tracking_diagnostics->enabled;
  if (collect && ns > 9007199254740991LL)
    throw std::invalid_argument("Diagnostics timestamp exceeds safe JSON integer");
  if (collect)
    for (const auto &d : obs)
      if (d.id > 9007199254740991ULL)
        throw std::invalid_argument("Diagnostics observation identifier exceeds safe JSON integer");
  TrackingDiagnostics report;
  report.timestamp_ns = ns;
  report.prior_tracks = static_cast<uint32_t>(tracks_.size());
  report.observations = static_cast<uint32_t>(obs.size());
  report.evaluated_pairs = report.prior_tracks * report.observations;
  if (previous_ns_ >= 0 && ns <= previous_ns_)
    throw std::invalid_argument("Out-of-order/duplicate observation timestamp rejected");
  for (auto &d : obs)
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.var_x) ||
        !std::isfinite(d.var_y) || d.var_x <= 0 || d.var_y <= 0)
      throw std::invalid_argument("Invalid observation position/covariance");
  double dt = previous_ns_ < 0 ? c.dt_s : (ns - previous_ns_) * 1e-9;
  if (!std::isfinite(dt) || dt <= 0)
    throw std::invalid_argument("Invalid tracking time interval");
  previous_ns_ = ns;
  for (auto &t : tracks_) {
    t.state[0] += dt * t.state[2];
    t.state[1] += dt * t.state[3];
    std::array<double, 16> f{1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1}, fp{}, p{};
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
          fp[i * 4 + j] += f[i * 4 + k] * t.covariance[k * 4 + j];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
          p[i * 4 + j] += fp[i * 4 + k] * f[j * 4 + k];
    for (int axis = 0; axis < 2; ++axis) {
      p[axis * 4 + axis] += .25 * dt * dt * dt * dt;
      p[axis * 4 + axis + 2] += .5 * dt * dt * dt;
      p[(axis + 2) * 4 + axis] += .5 * dt * dt * dt;
      p[(axis + 2) * 4 + axis + 2] += dt * dt;
    }
    t.covariance = p;
    ++t.age;
    ++t.misses;
    t.matched.clear();
  }
  struct Candidate {
    double distance;
    size_t t, d, report_index;
  };
  std::vector<Candidate> candidates;
  std::vector<AssociationDiagnostic> pair_reports;
  for (size_t i = 0; i < tracks_.size(); ++i)
    for (size_t j = 0; j < obs.size(); ++j) {
      auto &t = tracks_[i];
      auto &d = obs[j];
      const auto innovation =
          tracking_innovation({t.state[0], t.state[1]},
                              {t.covariance[0], t.covariance[1], t.covariance[4], t.covariance[5]},
                              {d.x, d.y}, {d.var_x, d.var_y});
      const double dist = innovation.mahalanobis_squared;
      if (dist < 9.21034) {
        size_t index = 0;
        if (collect) {
          ++report.gate_pass_pairs;
          index = pair_reports.size();
          pair_reports.push_back(
              {t.id, d.id, t.confirmed, {t.state[0], t.state[1]}, innovation, "selected"});
        }
        candidates.push_back({dist, i, j, index});
      } else if (collect) {
        ++report.gate_reject_pairs;
        if (report.gate_reject_examples.size() < 32)
          report.gate_reject_examples.push_back(
              {t.id, d.id, t.confirmed, {t.state[0], t.state[1]}, innovation, "gate_rejected"});
      }
    }
  // Reacquire established hypotheses before tentative births can take their
  // in-gate observations. Within each lifecycle class retain nearest-neighbor
  // distance and stable tie order; this favors continuity over tentative capture.
  std::stable_sort(candidates.begin(), candidates.end(), [&](const auto &a, const auto &b) {
    if (tracks_[a.t].confirmed != tracks_[b.t].confirmed)
      return tracks_[a.t].confirmed;
    return a.distance < b.distance;
  });
  std::vector<bool> used_t(tracks_.size()), used_d(obs.size());
  for (auto cand : candidates) {
    if (used_t[cand.t] || used_d[cand.d]) {
      if (collect) {
        auto pair = pair_reports[cand.report_index];
        if (used_t[cand.t] && used_d[cand.d]) {
          ++report.skipped_both;
          pair.disposition = "both_used";
        } else if (used_t[cand.t]) {
          ++report.skipped_track_only;
          pair.disposition = "track_used";
        } else {
          ++report.skipped_observation_only;
          pair.disposition = "observation_used";
        }
        if (report.conflict_examples.size() < 32)
          report.conflict_examples.push_back(std::move(pair));
      }
      continue;
    }
    if (collect) {
      ++report.selected_associations;
      report.matches.push_back(pair_reports[cand.report_index]);
    }
    used_t[cand.t] = used_d[cand.d] = true;
    auto &t = tracks_[cand.t];
    auto &d = obs[cand.d];
    auto p = t.covariance;
    double a = p[0] + d.var_x, b = p[1], cc = p[5] + d.var_y, det = a * cc - b * b;
    std::array<double, 8> k{};
    for (int i = 0; i < 4; ++i) {
      k[i * 2] = (p[i * 4] * cc - p[i * 4 + 1] * b) / det;
      k[i * 2 + 1] = (p[i * 4 + 1] * a - p[i * 4] * b) / det;
    }
    double dx = d.x - t.state[0], dy = d.y - t.state[1];
    for (int i = 0; i < 4; ++i)
      t.state[i] += k[2 * i] * dx + k[2 * i + 1] * dy;
    std::array<double, 16> ikh{}, tmp{}, updated{};
    for (int i = 0; i < 4; ++i) {
      ikh[i * 4 + i] = 1;
      ikh[i * 4] -= k[2 * i];
      ikh[i * 4 + 1] -= k[2 * i + 1];
    }
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int v = 0; v < 4; ++v)
          tmp[i * 4 + j] += ikh[i * 4 + v] * p[v * 4 + j];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        for (int v = 0; v < 4; ++v)
          updated[i * 4 + j] += tmp[i * 4 + v] * ikh[j * 4 + v];
        updated[i * 4 + j] += k[2 * i] * d.var_x * k[2 * j] + k[2 * i + 1] * d.var_y * k[2 * j + 1];
      }
    t.covariance = updated;
    t.misses = 0;
    ++t.hits;
    t.confirmed = t.confirmed || t.hits >= c.confirmation_hits;
    t.last_ns = ns;
    t.matched = {d.id};
  }
  const auto before_deletion = tracks_.size();
  std::erase_if(tracks_, [&](const Track &t) {
    return t.misses > c.coast_frames || (!t.confirmed && t.misses > 1);
  });
  report.deleted_tracks = static_cast<uint32_t>(before_deletion - tracks_.size());
  for (size_t j = 0; j < obs.size(); ++j)
    if (!used_d[j]) {
      if (tracks_.size() >= 128)
        throw std::runtime_error("Track capacity 128 exceeded: fail_run");
      auto &d = obs[j];
      Track t;
      t.id = next_id_++;
      t.state = {d.x, d.y, 0, 0};
      t.covariance = {d.var_x, 0, 0, 0, 0, d.var_y, 0, 0, 0, 0, 25, 0, 0, 0, 0, 25};
      t.first_ns = t.last_ns = ns;
      t.confirmed = c.confirmation_hits <= 1;
      t.matched = {d.id};
      tracks_.push_back(t);
    }
  for (const auto &track : tracks_) {
    for (double value : track.state)
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite tracking state");
    for (double value : track.covariance)
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite tracking covariance");
  }
  if (collect) {
    report.born_tracks = report.observations - report.selected_associations;
    report.unmatched_prior_tracks = report.prior_tracks - report.selected_associations;
    (void)report.to_json(); // Validate bounded report before publishing it transactionally.
    *output = std::move(report);
  }
}
json serialize(const Truth &t) {
  return {{"schema_version", 1}, {"object_id", t.id}, {"x_m", t.x},
          {"y_m", t.y},          {"vx_mps", t.vx},    {"vy_mps", t.vy}};
}
json serialize(const Detection &d) {
  return {{"schema_version", 1},
          {"observation_id", d.id},
          {"sensor_id", "synthetic-1"},
          {"x_m", d.x},
          {"y_m", d.y},
          {"covariance_m2", {d.var_x, 0, 0, d.var_y}},
          {"range_m", d.range_m},
          {"radial_velocity_mps", d.radial_velocity_mps},
          {"confidence", d.confidence},
          {"quality_flags", {"modeled_covariance", "simulated"}}};
}
json serialize(const Track &t) {
  return {{"schema_version", 1},
          {"track_id", t.id},
          {"state", t.state},
          {"covariance", t.covariance},
          {"state_units", {"m", "m", "m/s", "m/s"}},
          {"filter", "constant_velocity_kalman_2d"},
          {"lifecycle", t.misses      ? "coasting"
                        : t.confirmed ? "confirmed"
                                      : "tentative"},
          {"hits", t.hits},
          {"misses", t.misses},
          {"first_timestamp_ns", t.first_ns},
          {"last_timestamp_ns", t.last_ns},
          {"matched_observations", t.matched}};
}
} // namespace aether
