#pragma once
#include <aethersense/engine.hpp>

namespace aether::detail {
// Internal numerical kernel shared with the model-matched consistency harness.
// Caller supplies finite state/SPD covariance, positive dt and measurement variances.
// Lifecycle, association, timestamps and input validation remain with Tracker.
inline void kalman_predict(Track &t, double dt) {
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
}

inline void kalman_correct(Track &t, const Detection &d) {
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
}
} // namespace aether::detail
