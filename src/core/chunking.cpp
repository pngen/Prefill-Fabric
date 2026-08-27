// Prefill Fabric - deterministic chunk planning implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/chunking.hpp"

namespace prefillfabric {

std::uint64_t ChunkPlanner::effective_chunk_token_count(const PrefillRequest& req) const {
  const std::uint64_t remaining = req.remaining_prefill_tokens();
  if (remaining == 0) return 0;

  std::uint64_t chunk = cfg_.max_tokens_per_chunk;
  if (chunk == 0) chunk = 1;

  if (req.compute_estimate > 0) {
    // Work per chunk is proportional to chunk tokens. Bound chunk so the
    // resulting chunk compute does not exceed max_work_per_chunk.
    const long double ratio = static_cast<long double>(cfg_.max_work_per_chunk) /
                              static_cast<long double>(req.compute_estimate);
    std::uint64_t limit = static_cast<std::uint64_t>(ratio * static_cast<long double>(remaining));
    if (limit > remaining) limit = remaining;
    if (limit < chunk) chunk = limit;
  }

  if (req.memory_estimate_bytes > 0) {
    const long double ratio = static_cast<long double>(cfg_.max_memory_per_chunk) /
                              static_cast<long double>(req.memory_estimate_bytes);
    std::uint64_t limit = static_cast<std::uint64_t>(ratio * static_cast<long double>(remaining));
    if (limit > remaining) limit = remaining;
    if (limit < chunk) chunk = limit;
  }

  if (chunk == 0) chunk = 1;
  if (chunk > remaining) chunk = remaining;
  if (chunk == 0) chunk = 1;
  return chunk;
}

PrefillPlan ChunkPlanner::plan(const PrefillRequest& req, AttemptId attempt,
                               Generation seed_generation) const {
  PrefillPlan plan;
  plan.request = req;
  const std::uint64_t remaining = req.remaining_prefill_tokens();
  if (remaining == 0) {
    plan.total_chunk_tokens = 0;
    plan.total_chunk_count = 0;
    return plan;
  }

  const std::uint64_t chunk_size = effective_chunk_token_count(req);
  std::uint64_t offset = 0;
  std::uint32_t idx = 0;
  const bool reuse = req.has_valid_reusable_prefix();

  while (offset < remaining) {
    std::uint64_t n = remaining - offset;
    if (n > chunk_size) n = chunk_size;

    PrefillChunk c;
    c.request_id = req.request_id;
    c.attempt_id = attempt;
    c.token_start = offset;
    c.token_count = n;
    c.chunk_index = idx;
    c.generation = Generation(seed_generation.value() + static_cast<std::uint64_t>(idx) + 1);
    c.parent_generation = Generation(seed_generation.value());
    c.work_kind = (reuse ? WorkKind::reused_suffix
                    : (offset + n >= remaining && idx == 0 ? WorkKind::full_prompt
                       : WorkKind::chunked));
    if (req.deadline) c.deadline = req.deadline;

    // Proportional estimates, capped at cfg limits.
    if (req.compute_estimate > 0) {
      long double frac = static_cast<long double>(n) / static_cast<long double>(remaining);
      c.compute_estimate = static_cast<std::uint64_t>(frac * static_cast<long double>(req.compute_estimate));
      if (c.compute_estimate > cfg_.max_work_per_chunk) c.compute_estimate = cfg_.max_work_per_chunk;
    }
    if (req.memory_estimate_bytes > 0) {
      long double frac = static_cast<long double>(n) / static_cast<long double>(remaining);
      c.memory_estimate_bytes = static_cast<std::uint64_t>(frac * static_cast<long double>(req.memory_estimate_bytes));
      if (c.memory_estimate_bytes > cfg_.max_memory_per_chunk) c.memory_estimate_bytes = cfg_.max_memory_per_chunk;
    }

    plan.chunks.push_back(std::move(c));
    plan.total_chunk_tokens += n;
    offset += n;
    ++idx;
  }
  plan.total_chunk_count = static_cast<std::uint32_t>(plan.chunks.size());
  return plan;
}

}  // namespace prefillfabric
