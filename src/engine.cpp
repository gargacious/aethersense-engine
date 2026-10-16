#include <aethersense/engine.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace aether {
constexpr double pi = std::numbers::pi;
void Config::validate() const {
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
}
json Config::to_json() const {
  return {{"schema_version", 1},
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
  std::vector<Truth> out;
  for (int k = 0; k < c.objects; ++k) {
    double x = -25. + k * 18., y = 35. + (k % 3) * 12., vx = k % 2 ? -1.1 : 1.2,
           vy = (k % 3 - 1) * 0.15;
    if (c.scenario == "single_stationary") {
      x = 10. + k * 8;
      y = 45;
      vx = vy = 0;
    }
    out.push_back({k + 1, x + vx * t, y + vy * t, vx, vy});
  }
  return out;
}
std::vector<Detection> simulate_detections(const Config &c, std::span<const Truth> truth,
                                           Random &rng, uint64_t seq) {
  std::vector<Detection> out;
  uint64_t id = seq * 128;
  double noise = c.noise_m * (c.scenario == "noise_sweep" ? 1. + (seq % 200) / 50. : 1.);
  for (auto &t : truth) {
    if (rng.uniform() < c.dropout ||
        (c.scenario == "intermittent_observation" && seq % 100 >= 35 && seq % 100 < 50))
      continue;
    double x = t.x + noise * rng.normal(), y = t.y + noise * rng.normal();
    out.push_back({++id, x, y, noise * noise, noise * noise, std::hypot(x, y),
                   (t.x * t.vx + t.y * t.vy) / std::hypot(t.x, t.y), 0.95});
  }
  for (int k = 0; k < 10; ++k)
    if (rng.uniform() < c.false_rate / 10.) {
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
  if (n == 0 || (n & (n - 1)))
    throw std::invalid_argument("FFT length must be nonzero power of two");
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    auto wlen = std::polar(1., (inverse ? 2 : -2) * pi / len);
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w = 1;
      for (size_t j = 0; j < len / 2; ++j) {
        auto u = a[i + j], v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (inverse)
    for (auto &v : a)
      v /= n;
}
std::vector<std::complex<double>> matched_filter(std::span<const std::complex<double>> a,
                                                 std::span<const std::complex<double>> ref) {
  if (a.empty() || ref.empty())
    throw std::invalid_argument("Empty matched-filter input");
  std::vector<std::complex<double>> out(a.size() + ref.size() - 1);
  for (size_t n = 0; n < a.size(); ++n)
    for (size_t k = 0; k < ref.size(); ++k)
      out[n + k] += a[n] * std::conj(ref[ref.size() - 1 - k]);
  return out;
}
std::vector<double> ca_cfar(std::span<const double> p, int training, int guard, double pfa) {
  if (training < 1 || guard < 0 || !std::isfinite(pfa) || pfa <= 0 || pfa >= 1 ||
      p.size() < static_cast<size_t>(2 * (training + guard) + 1))
    throw std::invalid_argument("Invalid CFAR configuration");
  for (auto v : p)
    if (!std::isfinite(v) || v < 0)
      throw std::invalid_argument("CFAR input must be finite nonnegative power");
  std::vector<double> out(p.size(), -1);
  double alpha = 2 * training * (std::pow(pfa, -1. / (2 * training)) - 1);
  for (int i = training + guard; i < static_cast<int>(p.size()) - training - guard; ++i) {
    double sum = 0;
    for (int k = guard + 1; k <= guard + training; ++k)
      sum += p[i - k] + p[i + k];
    out[i] = alpha * sum / (2 * training);
  }
  return out;
}
SignalResult process_iq(const RawBlock &b, const Config &c) {
  b.validate();
  if (c.fft_size != b.samples_per_chirp || c.chirps != b.chirps)
    throw std::invalid_argument("Raw dimensions differ from immutable config");
  int n = b.samples_per_chirp, m = b.chirps;
  SignalResult r;
  r.sample_rate_hz = b.sample_rate_hz;
  r.chirp_period_s = b.chirp_period_s;
  r.bins = n;
  r.chirps = m;
  r.range_bin_m = 299792458. * b.sample_rate_hz / (2 * b.slope_hz_s * n);
  r.velocity_bin_mps = b.wavelength_m / (2 * b.chirp_period_s * m);
  r.i.resize(n);
  r.q.resize(n);
  r.spectrum.resize(n);
  r.phase.resize(n);
  r.range_profile.resize(n);
  r.range_doppler.resize(n * m);
  for (int k = 0; k < n; ++k) {
    r.i[k] = b.iq[k].real();
    r.q[k] = b.iq[k].imag();
  }
  auto a = b.iq;
  double gain = c.window == "hann" ? (n - 1.) / (2 * n) : 1.;
  for (int ch = 0; ch < 2; ++ch)
    for (int j = 0; j < m; ++j) {
      auto row = std::span(a).subspan((ch * m + j) * n, n);
      std::complex<double> dc = 0;
      for (auto v : row)
        dc += v;
      dc /= n;
      for (int k = 0; k < n; ++k)
        row[k] =
            (row[k] - dc) * (c.window == "hann" ? .5 - .5 * std::cos(2 * pi * k / (n - 1)) : 1.);
      fft(row);
      for (auto &v : row)
        v /= n * gain;
    }
  for (int k = 0; k < n; ++k) {
    r.spectrum[k] = std::abs(a[k]);
    r.phase[k] = std::arg(a[k]);
    for (int j = 0; j < m; ++j)
      r.range_profile[k] += std::norm(a[j * n + k]) / m;
  }
  r.threshold = ca_cfar(r.range_profile, c.cfar_training, c.cfar_guard, c.cfar_pfa);
  std::vector<std::complex<double>> col(m);
  double dg = (m - 1.) / (2 * m);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < m; ++j)
      col[j] = a[j * n + k] * (.5 - .5 * std::cos(2 * pi * j / (m - 1)));
    fft(col);
    for (int j = 0; j < m; ++j)
      r.range_doppler[((j + m / 2) % m) * n + k] = std::norm(col[j] / (m * dg));
  }
  for (int k = 1; k < n / 2 - 1; ++k)
    if (r.threshold[k] >= 0 && r.range_profile[k] > r.threshold[k] &&
        r.range_profile[k] > r.range_profile[k - 1] &&
        r.range_profile[k] > r.range_profile[k + 1]) {
      int dbin = 0;
      for (int j = 1; j < m; ++j)
        if (r.range_doppler[j * n + k] > r.range_doppler[dbin * n + k])
          dbin = j;
      std::complex<double> cross = 0;
      for (int j = 0; j < m; ++j)
        cross += a[(m + j) * n + k] * std::conj(a[j * n + k]);
      double sx = std::clamp(std::arg(cross) / pi, -1., 1.), range = k * r.range_bin_m;
      double variance = std::max(1., range * range * .0025);
      r.detections.push_back({b.envelope.sequence_no * 2048 + static_cast<uint64_t>(k), range * sx,
                              range * std::sqrt(1 - sx * sx), variance, variance, range,
                              (dbin - m / 2) * r.velocity_bin_mps, .8});
    }
  return r;
}
json SignalResult::to_json() const {
  return {{"schema_version", 1},
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
}

void Tracker::update(std::span<const Detection> obs, int64_t ns, const Config &c) {
  if (previous_ns_ >= 0 && ns <= previous_ns_)
    throw std::invalid_argument("Out-of-order/duplicate observation timestamp rejected");
  for (auto &d : obs)
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.var_x) ||
        !std::isfinite(d.var_y) || d.var_x <= 0 || d.var_y <= 0)
      throw std::invalid_argument("Invalid observation position/covariance");
  double dt = previous_ns_ < 0 ? c.dt_s : (ns - previous_ns_) * 1e-9;
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
    size_t t, d;
  };
  std::vector<Candidate> candidates;
  for (size_t i = 0; i < tracks_.size(); ++i)
    for (size_t j = 0; j < obs.size(); ++j) {
      auto &t = tracks_[i];
      auto &d = obs[j];
      double a = t.covariance[0] + d.var_x, b = t.covariance[1], cc = t.covariance[5] + d.var_y,
             det = a * cc - b * b, dx = d.x - t.state[0], dy = d.y - t.state[1];
      if (det <= 0)
        throw std::runtime_error("Nonpositive innovation covariance");
      double dist = (cc * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
      if (dist < 9.21034)
        candidates.push_back({dist, i, j});
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
    if (used_t[cand.t] || used_d[cand.d])
      continue;
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
  std::erase_if(tracks_, [&](const Track &t) {
    return t.misses > c.coast_frames || (!t.confirmed && t.misses > 1);
  });
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
