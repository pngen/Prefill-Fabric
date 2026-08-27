# Validation / Testing Methodology

Tests live under `tests/` and run through CTest. There are no test timeouts; tests run to natural
completion. The suites are:

- `tests/unit`: chunk planner determinism, reusable-prefix remainder, CPU executor verifiable digest,
  scheduler completion, cached-prefix no-work, and CUDA validation (single request, packed group,
  long-prompt multi-chunk, memory cleanup) when CUDA is present. A required CUDA test is a hard
  failure if CUDA is unavailable; it never silently skips.
- `tests/concurrency`: multithreaded submit/cancel/snapshot/explain under pressure, with invariant
  checks (no reservation leak, no underflow, zero queue/running/remaining after closure, exact
  terminal accounting, finite per-tenant service).
- `tests/property`: randomized heavy workload with a fixed printed seed, continuous invariant checks,
  deadline expiry, random cancellation, multiple tenants/weights/latency classes.
- `tests/adversarial`: zero-token input, cached-prefix > prompt, duplicate id, overflow boundaries,
  expired deadline, double completion, double cancellation (idempotent), retry after terminal,
  unknown backend/worker/model deferral.
- `tests/persistence`: round-trip recovery, corrupt checksum, truncation, unknown version rejection.
- `tests/multiprocess`: real coordinator + two worker OS processes, heterogeneous requests, dispatch,
  completion, packing, fairness, zero leaks, exact token accounting.
- `tests/atomic_closure`: the atomic restart/epoch/stale-authority proof. The SAME scenario launches a
  coordinator and two worker OS processes, submits heterogeneous work, terminates one worker as an
  OS process, restarts it with a NEW WorkerBootId, rolls the epoch, transmits preserved pre-restart
  authority over the real protocol proving deterministic stale-epoch / stale-worker / stale-attempt /
  stale-generation rejection, then proves fresh work completes and all reservations are released.

## Automated closure gate

All of the following are required: clean Release and Debug configure/build, `/W4 /WX`, no warnings,
all CTest targets passing, at least three clean Release test-suite runs and three clean Debug runs,
substantial randomized/property checks, the dedicated adversarial/persistence/concurrency suites, a
real external `find_package` consumer, the real framed-TCP multiprocess test, the atomic closure
scenario repeated in Release and Debug, the real CUDA proof with output verification and memory
cleanup, CLI smoke coverage, examples compiling, install/export validation, no test timeouts, and a
clean git status.
