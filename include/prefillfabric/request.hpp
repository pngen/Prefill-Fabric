// Prefill Fabric - prefill request, reusable-prefix metadata, plan, chunk.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include "prefillfabric/types.hpp"
#include "prefillfabric/result.hpp"
#include "prefillfabric/clock.hpp"

namespace prefillfabric {

// Validated reusable-prefix metadata. Prefill Fabric does NOT implement the
// reusable state cache itself; it accepts metadata describing a validated
// reusable prefix and governs the REMAINING prefill execution.
struct ReusablePrefixMetadata {
  // Number of leading prompt tokens already represented by valid reusable
  // state. Invariant: 0 <= token_count <= prompt_token_count of the request.
  std::uint64_t token_count = 0;
  // Opaque fingerprint of the cache entry. Not interpreted by the runtime;
  // it is carried for lineage/explain and must match the caller's cache key.
  std::string fingerprint;
  // Whether the metadata claims the cache entry is verifiably valid.
  bool validated = false;
};

// Caller-supplied shape/layout that the executor layer may require.
struct InputShape {
  std::uint64_t seq_len = 0;
  std::uint64_t hidden_dim = 0;
  std::uint64_t num_heads = 0;
  std::uint64_t head_dim = 0;
  bool operator==(const InputShape& o) const noexcept {
    return seq_len == o.seq_len && hidden_dim == o.hidden_dim &&
           num_heads == o.num_heads && head_dim == o.head_dim;
  }
  bool operator!=(const InputShape& o) const noexcept { return !(*this == o); }
};

// A prefill submission. Thread-safe to construct; the scheduler takes it by
// value/copy and never mutates the caller's object.
struct PrefillRequest {
  RequestId request_id;
  TenantId tenant_id;
  ModelKey model;

  // Adapter identity (optional). AdapterRelation describes how the adapter
  // relates to the base model for compatibility purposes.
  AdapterId adapter_id;
  AdapterRelation adapter_relation = AdapterRelation::none;

  // Tokenizer / vocabulary compatibility.
  VocabSpec vocab;

  // Total prompt length and supplied reusable-prefix extent.
  std::uint64_t prompt_token_count = 0;
  ReusablePrefixMetadata reusable_prefix;

  // Input shape / layout / numeric mode.
  InputShape input_shape;
  InputLayout input_layout = InputLayout::seq_major;
  NumericMode numeric_mode = NumericMode::fp32;

  // Backend / device preferences.
  std::string backend = kBackendAny;          // "", "cpu", "cuda", "*", ...
  DeviceId device;                             // nil = any
  std::string device_match;                    // exact device name or empty

  // Scheduling semantics.
  LatencyClass latency_class = LatencyClass::standard;
  std::optional<Nanoseconds> deadline;         // absolute monotonic deadline
  int priority = 0;                            // higher = more urgent
  double tenant_weight = 1.0;                  // relative service weight
  Nanoseconds arrival_time = 0;                // supplied by caller
  std::optional<Nanoseconds> max_queue_delay;

  // Resource estimates.
  std::uint64_t memory_estimate_bytes = 0;     // peak prefill memory
  std::uint64_t compute_estimate = 0;          // abstract compute units
  int max_retries = 2;

  // Provenance: who/where the request came from (used for explainability).
  std::string requester;

  // The amount of ACTUAL prefill work remaining after reusable-prefix
  // accounting. This is the quantity the scheduler governs, not the raw
  // prompt length. Returns 0 when the prompt is fully covered by reuse.
  std::uint64_t remaining_prefill_tokens() const noexcept {
    if (prompt_token_count <= reusable_prefix.token_count) return 0;
    return prompt_token_count - reusable_prefix.token_count;
  }

  bool has_valid_reusable_prefix() const noexcept {
    return reusable_prefix.validated && reusable_prefix.token_count > 0;
  }

  // Validate caller-supplied fields that must hold before admission.
  // Returns an ErrorInfo describing the first detected problem, or ok.
  ErrorInfo validate() const noexcept {
    if (request_id.is_nil()) return ErrorInfo(ErrorCode::invalid_argument, "request_id is nil");
    if (tenant_id.is_nil()) return ErrorInfo(ErrorCode::invalid_argument, "tenant_id is nil");
    if (prompt_token_count == 0) return ErrorInfo(ErrorCode::invalid_argument, "prompt_token_count must be > 0");
    if (reusable_prefix.token_count > prompt_token_count)
      return ErrorInfo(ErrorCode::invalid_reusable_prefix, "reusable prefix exceeds prompt length");
    if (reusable_prefix.validated && reusable_prefix.token_count == 0)
      return ErrorInfo(ErrorCode::invalid_reusable_prefix, "validated reusable prefix has zero tokens");
    if (tenant_weight <= 0.0)
      return ErrorInfo(ErrorCode::invalid_argument, "tenant_weight must be > 0");
    if (max_retries < 0) return ErrorInfo(ErrorCode::invalid_argument, "max_retries must be >= 0");
    if (deadline && *deadline < arrival_time)
      return ErrorInfo(ErrorCode::deadline_impossible, "deadline before arrival");
    if (max_queue_delay && *max_queue_delay < 0)
      return ErrorInfo(ErrorCode::invalid_argument, "max_queue_delay must be >= 0");
    return ErrorInfo(ErrorCode::ok, "");
  }

  bool operator==(const PrefillRequest& o) const noexcept {
    if (request_id != o.request_id || tenant_id != o.tenant_id || model != o.model) return false;
    if (prompt_token_count != o.prompt_token_count) return false;
    if (reusable_prefix.token_count != o.reusable_prefix.token_count) return false;
    if (numeric_mode != o.numeric_mode) return false;
    return true;
  }
};

// A single chunk of prefill execution. A request's remaining prefill is
// decomposed into an ordered sequence of PrefillChunk objects (a PrefillPlan).
struct PrefillChunk {
  RequestId request_id;
  AttemptId attempt_id;        // the attempt this chunk belongs to
  GroupId group_id;            // assigned when formed into a group
  std::uint64_t token_start = 0;   // offset within the prompt (0-based)
  std::uint64_t token_count = 0;   // number of uncached tokens in this chunk
  std::uint32_t chunk_index = 0;   // 0-based within the current attempt plan
  Generation generation;            // monotonically increasing per attempt
  Generation parent_generation;     // lineage: generation that produced this
  WorkKind work_kind = WorkKind::full_prompt;
  std::uint64_t memory_estimate_bytes = 0;
  std::uint64_t compute_estimate = 0;
  std::optional<Nanoseconds> deadline;

  // Validate a chunk against its request.
  ErrorInfo validate(const PrefillRequest& req) const noexcept {
    if (request_id != req.request_id)
      return ErrorInfo(ErrorCode::invalid_argument, "chunk request_id mismatch");
    if (token_count == 0) return ErrorInfo(ErrorCode::invalid_argument, "chunk token_count must be > 0");
    const auto remaining = req.remaining_prefill_tokens();
    if (token_start + token_count > remaining)
      return ErrorInfo(ErrorCode::out_of_range, "chunk exceeds remaining prefill extent");
    return ErrorInfo(ErrorCode::ok, "");
  }
};

// An ordered plan of chunks covering a request's remaining prefill.
struct PrefillPlan {
  PrefillRequest request;
  std::vector<PrefillChunk> chunks;
  std::uint64_t total_chunk_tokens = 0;   // sum of uncached token_count
  std::uint32_t total_chunk_count = 0;

  // Total remaining prefill tokens this plan covers (== request.remaining).
  std::uint64_t covered_tokens() const noexcept { return request.remaining_prefill_tokens(); }
};

}  // namespace prefillfabric
