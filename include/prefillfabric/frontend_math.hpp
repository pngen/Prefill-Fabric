// Prefill Fabric - shared deterministic frontend-work digest spec.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
// Both the CPU executor and the CUDA executor must produce the SAME digest
// for identical (request, attempt, generation, token_start, token_count,
// token payload). The digest is a parallel-reducible integer sum so a
// sequential host computation and a device reduction agree exactly.
#pragma once
#include <cstdint>
#include "prefillfabric/types.hpp"

namespace prefillfabric {

constexpr std::uint64_t kFfM1 = 0x5851F42D4C957F2DULL;
constexpr std::uint64_t kFfM2 = 0x94D049BB133111EBULL;
constexpr std::uint64_t kFfM3 = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kFfM4 = 0xBF58476D1CE4E5B9ULL;

// Pure host-side computation of the verifiable per-member digest over a token
// range. position is the absolute token offset (token_start + local index).
inline std::uint64_t ff_token_position_value(std::uint64_t token, std::uint64_t position) noexcept {
  return (token * kFfM1) ^ (position * kFfM2) ^ ((position << 7) & ~0ULL);
}

inline std::uint64_t ff_member_digest(RequestId r, AttemptId a, Generation g,
                                      const std::uint32_t* tokens, std::uint64_t token_start,
                                      std::uint64_t count) noexcept {
  std::uint64_t acc = 0;
  for (std::uint64_t i = 0; i < count; ++i) {
    const std::uint64_t pos = token_start + i;
    acc += ff_token_position_value(static_cast<std::uint64_t>(tokens[i]), pos);
  }
  std::uint64_t h = kFfM3 ^ r.value();
  h ^= a.value() + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= g.value() + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= count + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h ^ (acc * kFfM4);
}

}  // namespace prefillfabric
