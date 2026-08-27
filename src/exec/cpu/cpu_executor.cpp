// Prefill Fabric - CPU executor implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/exec/cpu_executor.hpp"
#include "prefillfabric/frontend_math.hpp"

namespace prefillfabric {

CpuExecutor::CpuExecutor()
    : device_(DeviceDescriptor(DeviceId(1), "cpu0", kBackendCpu, 0, 0, "", true)) {}

CpuExecutor::CpuExecutor(DeviceDescriptor device) : device_(std::move(device)) {}

Result<ExecutorResult> CpuExecutor::execute(const ExecutionGroup& group,
                                            const TokenResolver& tokens) {
  ExecutorResult out;
  out.executor_family = "cpu";
  out.group_succeeded = true;
  out.members.reserve(group.members.size());

  for (const auto& m : group.members) {
    MemberResult mr;
    mr.request_id = m.request_id;
    mr.attempt_id = m.attempt_id;
    mr.generation = m.generation;
    mr.tokens_completed = m.token_count;
    mr.next_token_start = m.token_start + m.token_count;
    mr.requires_next_chunk = false;
    mr.outcome = MemberOutcome::success;

    if (m.token_count == 0) {
      mr.outcome = MemberOutcome::non_retryable_failure;
      mr.failure_code = ErrorCode::invalid_argument;
      mr.message = "zero-token member";
      out.group_succeeded = false;
      out.members.push_back(std::move(mr));
      continue;
    }

    const auto tok = tokens.resolve_tokens(m.request_id, m.attempt_id, m.generation,
                                            m.token_start, m.token_count);
    if (!tok) {
      mr.outcome = MemberOutcome::non_retryable_failure;
      mr.failure_code = tok.error().code;
      mr.message = tok.error().message;
      out.group_succeeded = false;
      out.members.push_back(std::move(mr));
      continue;
    }

    // Real bounded numerical work: a full pass over the chunk token payload.
    mr.digest = ff_member_digest(m.request_id, m.attempt_id, m.generation, tok.value(),
                                 m.token_start, m.token_count);

    out.members.push_back(std::move(mr));
  }
  return Result<ExecutorResult>::ok(std::move(out));
}

}  // namespace prefillfabric
