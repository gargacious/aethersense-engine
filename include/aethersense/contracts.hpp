#pragma once
#include <aethersense/engine.hpp>
#include <chrono>
#include <optional>
#include <stop_token>
#include <variant>

namespace aether::contracts {
// Versioned ownership ports, not a claim every optional backend/adapter exists.
struct ObservationBatch {
  Envelope envelope;
  std::vector<Detection> observations;
};
using SourceBatch = std::variant<RawBlock, ObservationBatch>;
enum class OverloadPolicy {
  block_with_timeout,
  drop_oldest_noncritical,
  drop_newest_with_metric,
  fail_run
};
enum class Delivery { accepted, dropped_oldest, dropped_newest, timed_out, stopped, failed };
struct DeliveryResult {
  Delivery status;
  uint64_t drops;
  std::string diagnostic;
};
class ISource {
public:
  virtual ~ISource() = default;
  // Returned batch owns its buffers. nullopt means orderly end/cancellation; errors throw.
  virtual std::optional<SourceBatch> next(std::stop_token) = 0;
};
class IProcessor {
public:
  virtual ~IProcessor() = default;
  virtual SignalResult process(const RawBlock &, const Config &) = 0;
};
class IRecorder {
public:
  virtual ~IRecorder() = default;
  virtual DeliveryResult append(const Envelope &, const json &,
                                std::chrono::milliseconds timeout) = 0;
  virtual void finalize() = 0; // write failures must be observable
};
class IPublisher {
public:
  virtual ~IPublisher() = default;
  // Copies or owns the snapshot before returning; never retains caller-owned references.
  virtual DeliveryResult publish(json snapshot, std::chrono::milliseconds timeout) = 0;
};
} // namespace aether::contracts
