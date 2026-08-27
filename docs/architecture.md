# Architecture

## Layering

1. **Foundation**: `result.hpp` (`Result<T>`, `ErrorCode`), `types.hpp` (strong IDs, model/vocab,
   latency classes, work kinds, lifecycles), `clock.hpp` (injectable monotonic clock).
2. **Model**: `request.hpp` (`PrefillRequest`, `ReusablePrefixMetadata`, `PrefillPlan`,
   `PrefillChunk`), `compatibility.hpp` (`CompatibilityKey`, `CompatibilityDecision`).
3. **Execution**: `executor.hpp` (`Executor`, `ExecutionGroup`, `MemberResult`, `TokenResolver`), plus
   concrete `CpuExecutor` and `CudaExecutor`. The runtime core never depends on a specific backend.
4. **Scheduling**: `policy.hpp` (explicit `SchedulerConfig`), `chunking.hpp` (`ChunkPlanner`),
   `memory.hpp` (`MemoryGovernor`), `scheduler.hpp`/`.cpp` (`PrefillScheduler`).
5. **Observability/Explain**: `observability.hpp` (`Stats`, `Snapshot`, `EventLog`), `explain.hpp`.
6. **Persistence**: `persistence.hpp`/`.cpp` (versioned, checksummed, bounds-validated).
7. **Distributed**: `network.hpp`/`.cpp` (Winsock framing), `protocol.hpp`/`.cpp` (codec),
   `coordinator.hpp`/`.cpp`, `worker.hpp`/`.cpp`, `client.hpp`/`.cpp`.

## Scheduling core

`PrefillScheduler` is a single thread-safe object. Public entry points (`submit`, `run_cycle`,
`report_completion`, `cancel`, `worker_lost`, `roll_epoch`, `persist`, `recover`, `snapshot`,
`explain`) each take an internal mutex, operate on authoritative state, and release the lock before
any I/O or executor call. The in-process driver `drive_until_quiescent` calls `run_cycle`, executes
groups via the attached executor **outside** the lock, then calls `report_completion`.

`run_cycle()` performs: (1) a deadline and max-queue-delay expiry sweep; (2) candidate selection using
a lexicographic multi-criteria ordering (deadline urgency, deadline, priority, latency class, weighted
normalized-service fairness deficit, starvation aging, arrival, identifier) rather than a single
score; (3) compatibility-aware formation into `ExecutionGroup`s; (4) per-member memory reservation
with rollback on failure; (5) deterministic worker assignment. It never blocks on I/O under the lock.

`report_completion()` validates authority in order: epoch, current worker boot id, attempt, chunk/work
generation, terminal/duplicate state, then applies per-member terminal-or-continuation outcomes.
Stale authority is rejected with an explicit error result and never mutates state.

## Thread-safety contract

- `PrefillScheduler` is safe for concurrent `submit`/`cancel`/`snapshot`/`explain`/`stats`.
- No method performs blocking I/O while holding the scheduler lock; executor calls run outside it.
- No callback is invoked while holding an internal lock.
- Internal state transitions are re-entrancy-free: helpers assume the lock is held and never
  re-acquire it, and no public method invokes another public method while holding the lock.

## CUDA

`CudaExecutor` allocates device memory, transfers tokens, launches a parallel-reduction kernel that
computes the same verifiable per-member digest as the CPU executor, synchronizes, and frees all
resources. It verifies that device memory returns to the pre-group baseline. It is compiled by a
dedicated `nvcc` invocation because CMake/Ninja/MSVC PDB flags break `nvcc` on this toolchain.
