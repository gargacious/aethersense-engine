#include "../src/tracking_kalman.hpp"
#include <aethersense/engine.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace aether;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Cholesky solve is independent of the Python oracle's generic matrix inverse.
// Called at each step so a failed/nonfinite/non-SPD trial cannot be omitted.
double nees(const Track &track, const std::array<double, 4> &truth) {
  double lower[4][4]{}, whitened[4]{};
  for (int i = 0; i < 4; ++i) {
    require(std::isfinite(track.state[i]) && std::isfinite(truth[i]), "Nonfinite state/truth");
    for (int j = 0; j < 4; ++j) {
      require(std::isfinite(track.covariance[i * 4 + j]), "Nonfinite covariance");
      require(std::abs(track.covariance[i * 4 + j] - track.covariance[j * 4 + i]) < 1e-10,
              "Asymmetric covariance");
    }
    for (int j = 0; j <= i; ++j) {
      double value = track.covariance[i * 4 + j];
      for (int k = 0; k < j; ++k)
        value -= lower[i][k] * lower[j][k];
      if (i == j) {
        require(value > 0, "Covariance not positive definite");
        lower[i][j] = std::sqrt(value);
      } else
        lower[i][j] = value / lower[j][j];
    }
    double error = track.state[i] - truth[i];
    for (int j = 0; j < i; ++j)
      error -= lower[i][j] * whitened[j];
    whitened[i] = error / lower[i][i];
  }
  double value = 0;
  for (double component : whitened)
    value += component * component;
  require(std::isfinite(value), "Nonfinite NEES");
  return value;
}

json fixture() {
  constexpr double dt = .1;
  json trials = json::array();
  for (uint32_t seed = 30000; seed < 31024; ++seed) {
    Random random(seed);
    Track estimate;
    estimate.state = {10., 30., .8, -.3};
    estimate.covariance = {.25, 0, 0, 0, 0, .25, 0, 0, 0, 0, 25, 0, 0, 0, 0, 25};
    std::array<double, 4> truth = estimate.state;
    constexpr std::array<double, 4> prior_sigma{.5, .5, 5., 5.};
    for (size_t i = 0; i < truth.size(); ++i)
      truth[i] += prior_sigma[i] * random.normal();
    const auto initial_truth = truth;
    json steps = json::array();
    double final_nees = nees(estimate, truth), final_nis = 0;
    for (int step = 1; step <= 20; ++step) {
      const double ax = random.normal(), ay = random.normal();
      const double nx = .5 * random.normal(), ny = .5 * random.normal();
      // Withheld truth and samples generate measurements; neither enters the kernel.
      truth[0] += dt * truth[2] + .5 * dt * dt * ax;
      truth[1] += dt * truth[3] + .5 * dt * dt * ay;
      truth[2] += dt * ax;
      truth[3] += dt * ay;
      const Detection observation{static_cast<uint64_t>(step), truth[0] + nx, truth[1] + ny, .25,
                                  .25};
      require(std::isfinite(observation.x) && std::isfinite(observation.y),
              "Nonfinite observation");
      detail::kalman_predict(estimate, dt);
      nees(estimate, truth);
      const auto innovation = tracking_innovation({estimate.state[0], estimate.state[1]},
                                                  {estimate.covariance[0], estimate.covariance[1],
                                                   estimate.covariance[4], estimate.covariance[5]},
                                                  {observation.x, observation.y},
                                                  {observation.var_x, observation.var_y});
      final_nis = innovation.mahalanobis_squared;
      require(std::isfinite(final_nis) && final_nis >= 0, "Invalid NIS");
      detail::kalman_correct(estimate, observation);
      final_nees = nees(estimate, truth);
      steps.push_back({{"step_index", step},
                       {"acceleration", {ax, ay}},
                       {"measurement_noise", {nx, ny}},
                       {"observation", {observation.x, observation.y}}});
    }
    trials.push_back({{"seed", seed},
                      {"initial_truth", initial_truth},
                      {"steps", std::move(steps)},
                      {"final_truth", truth},
                      {"state", estimate.state},
                      {"covariance", estimate.covariance},
                      {"nees", final_nees},
                      {"nis", final_nis}});
  }
  return {{"schema_version", 1}, {"profile", "model_matched_cv_v1"}, {"trials", std::move(trials)}};
}
} // namespace

int main(int argc, char **) {
  try {
    require(argc == 1, "Consistency fixture accepts no arguments");
    const auto serialized = fixture().dump();
    require(serialized.size() + 1 <= 16 * 1024 * 1024, "Consistency fixture exceeds 16 MiB");
    std::cout << serialized << '\n';
    require(static_cast<bool>(std::cout), "Failed to write consistency fixture");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
