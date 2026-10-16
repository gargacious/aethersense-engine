#include <aethersense/compute_backend.hpp>
#include <stdexcept>
namespace aether {
bool ReferenceCpuBackend::supports(ComputePrecision precision) const noexcept {
  return precision == ComputePrecision::float32 || precision == ComputePrecision::float64;
}
ComputeCapabilities ReferenceCpuBackend::capabilities() const noexcept {
  return {true, true, true, true, 65536, 1024, 262144};
}
void ReferenceCpuBackend::fft32(std::span<std::complex<float>> input, bool inverse) const {
  fft_float32(input, inverse);
}
void ReferenceCpuBackend::fft64(std::span<std::complex<double>> input, bool inverse) const {
  fft_float64(input, inverse);
}
std::vector<double> ReferenceCpuBackend::ca_cfar_2d(std::span<const double> input, std::size_t rows,
                                                    std::size_t columns,
                                                    const Cfar2DConfig &config) const {
  return aether::ca_cfar_2d(input, rows, columns, config);
}
std::unique_ptr<IComputeBackend> make_compute_backend(std::string_view name) {
  if (name == "reference_cpu")
    return std::make_unique<ReferenceCpuBackend>();
  throw std::invalid_argument("Unknown or unavailable compute backend");
}
} // namespace aether
