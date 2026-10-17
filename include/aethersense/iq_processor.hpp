#pragma once
#include <aethersense/engine.hpp>
#include <memory>
namespace aether {
// Thread-confined, immutable configuration, owned reusable numerical workspace.
// process_into commits output only after successful validation/numerics. Prepared
// output retains capacity across calls. Returned owned outputs/JSON may allocate.
class IQProcessor {
  struct Impl;
  std::unique_ptr<Impl> impl_;

public:
  explicit IQProcessor(const Config &);
  ~IQProcessor();
  IQProcessor(IQProcessor &&) noexcept;
  IQProcessor &operator=(IQProcessor &&) noexcept;
  IQProcessor(const IQProcessor &) = delete;
  IQProcessor &operator=(const IQProcessor &) = delete;
  SignalResult make_output() const;
  void process_into(const RawBlock &, SignalResult &);
  SignalResult process(const RawBlock &);
  // Owned numerical element storage; excludes result vectors, metadata and allocator overhead.
  size_t workspace_bytes() const noexcept;
};
} // namespace aether
