// Prefill Fabric - deterministic chunk planning.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <vector>
#include "prefillfabric/types.hpp"
#include "prefillfabric/request.hpp"
#include "prefillfabric/policy.hpp"

namespace prefillfabric {

// Builds an ordered PrefillPlan for a request's REMAINING prefill work.
// Chunk boundaries are deterministic for a given request and policy state.
// The same request + config always yields the same chunking, which is what
// enables preemption at chunk boundaries and cross-worker recovery.
class ChunkPlanner {
 public:
  explicit ChunkPlanner(const SchedulerConfig& cfg) : cfg_(cfg) {}

  // Plan chunks covering exactly request.remaining_prefill_tokens().
  PrefillPlan plan(const PrefillRequest& req, AttemptId attempt,
                   Generation seed_generation) const;

  // Sizing helpers used by the scheduler for budget accounting.
  std::uint64_t effective_chunk_token_count(const PrefillRequest& req) const;

 private:
  SchedulerConfig cfg_;
};

}  // namespace prefillfabric
