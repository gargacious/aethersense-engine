#pragma once
#include <aethersense/dsp_extensions.hpp>
#include <memory>
#include <string_view>
namespace aether {
enum class ComputePrecision { float32, float64 };
struct ComputeCapabilities {
  bool cpu_only{true};
  bool fft_float32{}, fft_float64{}, cfar_float64{};
  std::size_t max_fft_samples{}, max_cfar_axis{}, max_cfar_cells{};
};
// Numerical CPU extension port only. No implicit device discovery or fallback.
// Mutable FFT buffers retain the validation/atomic failure contract of extensions.
class IComputeBackend {
public:
  virtual ~IComputeBackend() = default;
  virtual std::string_view name() const noexcept = 0;
  virtual bool supports(ComputePrecision) const noexcept = 0;
  virtual ComputeCapabilities capabilities() const noexcept = 0;
  virtual void fft32(std::span<std::complex<float>>, bool inverse = false) const = 0;
  virtual void fft64(std::span<std::complex<double>>, bool inverse = false) const = 0;
  virtual std::vector<double> ca_cfar_2d(std::span<const double>, std::size_t rows,
                                         std::size_t columns, const Cfar2DConfig &) const = 0;
};
class ReferenceCpuBackend final : public IComputeBackend {
public:
  std::string_view name() const noexcept override { return "reference_cpu"; }
  bool supports(ComputePrecision) const noexcept override;
  ComputeCapabilities capabilities() const noexcept override;
  void fft32(std::span<std::complex<float>>, bool inverse = false) const override;
  void fft64(std::span<std::complex<double>>, bool inverse = false) const override;
  std::vector<double> ca_cfar_2d(std::span<const double>, std::size_t rows, std::size_t columns,
                                 const Cfar2DConfig &) const override;
};
// Compile-target capability only: ARM64 NEON or x86-64 SSE2. Unsupported targets
// reject explicit selection; no scalar fallback is labeled SIMD.
bool native_simd_available() noexcept;
std::string_view native_simd_instruction_set() noexcept;
class NativeSimdBackend final : public IComputeBackend {
public:
  NativeSimdBackend();
  std::string_view name() const noexcept override { return "native_simd"; }
  bool supports(ComputePrecision) const noexcept override;
  ComputeCapabilities capabilities() const noexcept override;
  void fft32(std::span<std::complex<float>>, bool inverse = false) const override;
  void fft64(std::span<std::complex<double>>, bool inverse = false) const override;
  std::vector<double> ca_cfar_2d(std::span<const double>, std::size_t rows,
                                std::size_t columns, const Cfar2DConfig &) const override;
};
// Exact, explicit local selection; unknown/unavailable backend names reject.
std::unique_ptr<IComputeBackend> make_compute_backend(std::string_view name);
} // namespace aether
