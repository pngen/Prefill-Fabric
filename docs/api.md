# Public API / Use Guide

The public API is header-only strong typing under `include/prefillfabric`. No exceptions for ordinary
control flow; functions return `Result<T>` or `Result<void>` with structured `ErrorCode`.

## Key types

- Strong IDs: `RequestId`, `AttemptId`, `TenantId`, `WorkerId`, `WorkerBootId`, `Epoch`, `Generation`,
  `GroupId`, `DeviceId`, `AdapterId`. They are distinct types and cannot be silently interchanged.
- `PrefillRequest`: request/tenant/model/revision/adapter identity, `prompt_token_count`,
  `ReusablePrefixMetadata`, shape/layout/numeric-mode/backend/device, latency class, deadline,
  priority, tenant weight, arrival time, max queue delay, memory/compute estimates, `max_retries`.
  `remaining_prefill_tokens()` returns the actual uncached work; `validate()` performs input checks.
- `PrefillChunk` and `PrefillPlan`: ordered chunks covering the *remaining* prefill.
- `CompatibilityKey` / `CompatibilityDecision`: typed, explainable packing compatibility.
- `ExecutionGroup` / `ExecutableMember` / `MemberResult`: the unit of execution and per-member outcome.
- `Executor`: vendor-neutral; `CpuExecutor` and `CudaExecutor` implement it.
- `SchedulerConfig`: every scheduling policy knob, inspectable independently.
- `PrefillScheduler`: the core scheduling runtime.
- `Coordinator`, `Worker`, `Client`: the distributed control plane.

## PrefillScheduler usage

```cpp
SchedulerConfig cfg; cfg.max_tokens_per_chunk = 1024;
PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
PrefillRequest r; r.request_id = RequestId(1); r.tenant_id = TenantId(1);
r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 4096;
sch.submit_with_tokens(r, std::vector<std::uint32_t>(4096, 1));
sch.drive_until_quiescent();
bool done = sch.is_terminal(RequestId(1));
Stats st = sch.stats();
auto ex = sch.explain(RequestId(1));   // text and JSON trace
Snapshot sn = sch.snapshot();
```

## Errors

`ErrorCode` enumerates structured conditions for admission, scheduling, identity/lineage/staleness,
execution/device, persistence/protocol, and concurrency. `Result<T>::error()` returns an `ErrorInfo`.

## Thread safety

`PrefillScheduler` is internally synchronized. Concurrent `submit`/`cancel`/`snapshot`/`explain`/
`stats` are safe; executor calls happen outside the internal lock.
