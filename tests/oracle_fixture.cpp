#include <aethersense/engine.hpp>
#include <iostream>
#include <numbers>
using namespace aether;
int main() {
  Config c;
  c.fft_size = 32;
  c.chirps = 8;
  RawBlock b;
  b.samples_per_chirp = 32;
  b.chirps = 8;
  b.iq.resize(512);
  json iq = json::array();
  for (int ch = 0; ch < 2; ++ch)
    for (int m = 0; m < 8; ++m)
      for (int n = 0; n < 32; ++n) {
        auto v = std::polar(1., 2 * std::numbers::pi * (10. * n / 32 + 1. * m / 8) + ch * .3) +
                 std::complex<double>(.2, -.1);
        b.iq[(ch * 8 + m) * 32 + n] = v;
        iq.push_back({v.real(), v.imag()});
      }
  auto signal = process_iq(b, c);
  auto values = b.iq;
  values.resize(32);
  auto transformed = values;
  fft(transformed);
  json ff = json::array();
  for (auto v : transformed)
    ff.push_back({v.real(), v.imag()});
  Tracker tracker;
  json steps = json::array();
  for (int f = 0; f < 40; ++f) {
    double x = 5 + .2 * f + .1 * std::sin(f), y = 30 - .1 * f + .15 * std::cos(f);
    Detection d{static_cast<uint64_t>(f), x, y, .25, .36};
    tracker.update(std::span(&d, 1), f * 100000000ll, c);
    steps.push_back({{"measurement", {x, y}}, {"track", serialize(tracker.tracks().at(0))}});
  }
  std::cout
      << json{{"iq", iq}, {"fft", ff}, {"signal", signal.to_json()}, {"tracking", steps}}.dump();
}
