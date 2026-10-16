#pragma once
#include <aethersense/engine.hpp>
namespace aether {
// Version 1 portable raw wire form. Parsing is bounded to 12 MiB before JSON allocation.
json serialize_raw_block(const RawBlock &);
RawBlock import_raw_block(const json &);
RawBlock parse_raw_block(std::string_view);
// Deterministic regression artifacts, with truth held separately from observations.
json observation_fixture(std::string_view scenario, uint32_t seed = 42);
RawBlock analytic_tone_fixture();
} // namespace aether
