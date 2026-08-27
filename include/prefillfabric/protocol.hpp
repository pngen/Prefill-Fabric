// Prefill Fabric - framed TCP protocol.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "prefillfabric/types.hpp"
#include "prefillfabric/result.hpp"
#include "prefillfabric/request.hpp"
#include "prefillfabric/executor.hpp"
#include "prefillfabric/observability.hpp"
#include "prefillfabric/explain.hpp"
#include "prefillfabric/persistence.hpp"
#include "prefillfabric/scheduler.hpp"

namespace prefillfabric {

// Framing: [u32 little-endian length][u8 version][u8 type][payload (length-2 bytes)].
inline constexpr std::uint32_t kFrameLengthBytes = 4;
inline constexpr std::uint32_t kFrameMaxBytes = 64u * 1024u * 1024u;   // 64 MiB
inline constexpr std::uint8_t kProtocolVersion = 1;

// Message kinds carried in the type octet.
enum class MsgKind : std::uint8_t {
  hello = 1,
  worker_register = 2,
  worker_register_ack = 3,
  submit = 4,
  submit_ack = 5,
  cancel = 6,
  cancel_ack = 7,
  query_state = 8,
  state = 9,
  query_explain = 10,
  explain_resp = 11,
  query_stats = 12,
  stats_resp = 13,
  dispatch = 14,
  completion = 15,
  shutdown = 16,
  shutdown_ack = 17,
  ping = 18,
  pong = 19,
  roll_epoch = 20,
  error = 21
};

inline const char* to_string(MsgKind k) noexcept {
  switch (k) {
    case MsgKind::hello: return "hello";
    case MsgKind::worker_register: return "worker_register";
    case MsgKind::worker_register_ack: return "worker_register_ack";
    case MsgKind::submit: return "submit";
    case MsgKind::submit_ack: return "submit_ack";
    case MsgKind::cancel: return "cancel";
    case MsgKind::cancel_ack: return "cancel_ack";
    case MsgKind::query_state: return "query_state";
    case MsgKind::state: return "state";
    case MsgKind::query_explain: return "query_explain";
    case MsgKind::explain_resp: return "explain_resp";
    case MsgKind::query_stats: return "query_stats";
    case MsgKind::stats_resp: return "stats_resp";
    case MsgKind::dispatch: return "dispatch";
    case MsgKind::completion: return "completion";
    case MsgKind::shutdown: return "shutdown";
    case MsgKind::shutdown_ack: return "shutdown_ack";
    case MsgKind::ping: return "ping";
    case MsgKind::pong: return "pong";
    case MsgKind::roll_epoch: return "roll_epoch";
    case MsgKind::error: return "error";
    default: return "unknown";
  }
}

// A semantic protocol message. Fields are populated per-kind; unused fields
// are ignored on encode/decode. Integers are always serialized losslessly as
// little-endian 64-bit values -- never routed through floating-point JSON.
struct Message {
  MsgKind kind = MsgKind::hello;

  // Worker register / state / workers listing.
  WorkerDescriptor worker;
  Epoch epoch;
  std::vector<WorkerDescriptor> workers;

  // Submit.
  PrefillRequest request;
  std::vector<std::uint32_t> tokens;   // full prompt payload for submit

  // Acks / responses.
  ErrorInfo error;
  RequestId request_id;
  AttemptId attempt_id;
  Generation generation;
  Lifecycle lifecycle = Lifecycle::submitted;
  bool applied = false;
  bool accepted = false;

  // Dispatch.
  ExecutionGroup group;
  // Per-member token payload concatenated + offsets so the worker can run.
  std::vector<std::uint32_t> member_tokens;
  std::vector<std::uint64_t> member_token_offsets;  // size = group.size()+1

  // Completion.
  std::vector<MemberResult> member_results;

  // Stats / explain.
  Stats stats;
  Explain explain;

  // Cancel.
  CancelResult cancel_result;
};

// A decoded frame (length-prefixed packet).
struct Frame {
  std::uint8_t version = kProtocolVersion;
  MsgKind kind = MsgKind::hello;
  std::vector<std::uint8_t> payload;
};

// Encode a semantic message into a frame (with length prefix).
Result<std::vector<std::uint8_t>> encode_frame(const Message& msg);

// Decode a frame from a full buffer (length prefix + body). Validates
// version, length, kind and bounds.
Result<Message> decode_frame(const std::vector<std::uint8_t>& frame_bytes);

// Encode / decode the frame BODY (without the 4-byte length prefix).
Result<std::vector<std::uint8_t>> encode_body(const Message& msg);
Result<Message> decode_body(std::uint8_t version, MsgKind kind, const std::uint8_t* payload, std::size_t len);

// Validate a length prefix against the frame maximum.
bool valid_frame_length(std::uint32_t length) noexcept;

}  // namespace prefillfabric
