# Prefill Fabric

Prefill Fabric is an open-source, vendor-neutral runtime for scheduling, packing, executing,
measuring, and governing prompt-prefill work across heterogeneous AI inference infrastructure.

> **Governing systems question:** How should prompt-prefill work be formed, scheduled,
> partitioned, executed, and governed so that large and heterogeneous prompts make efficient
> forward progress without destroying latency, fairness, memory headroom, or downstream serving
> capacity?

This is a real runtime, not a toy scheduler, benchmark shell, API sketch, or simulation. It
implements the control semantics, an execution layer, persistence, observability, a framed TCP
multiprocess protocol, adversarial validation, examples, benchmarks, install/export, and
release-quality documentation.

## What it does

Prefill Fabric reasons about **prefill as a first-class inference phase** (not generic anonymous
work), modelling request/attempt/tenant/model/revision identity, reusable-prefix remainder,
chunking, compatible packing, fairness, deadlines, memory governance, and stale-authority
rejection across a distributed control plane. The distinction between **total prompt length** and
**actual prefill work remaining after reusable-prefix accounting** is fundamental: a request with a
32K-token prompt whose first 28K tokens are represented by valid reusable state is scheduled by its
4K-token *uncached* burden, not as 32K tokens of new computation.

## Architecture (top level)

```mermaid
flowchart LR
  subgraph Backends
    CPU[CPU Executor]
    CUDA[CUDA Executor (Blackwell sm_120)]
  end
  subgraph Core
    S[PrefillScheduler]
    M[MemoryGovernor]
    P[ChunkPlanner]
    C[Compatibility / Packing]
    F[Fairness/Deadline]
    X[Explain/Stats/Events]
  end
  subgraph Distributed
    Coord[Coordinator]
    W1[Worker 1]
    W2[Worker 2]
    Cli[Client/CLI]
  end
  S --> M; S --> P; S --> C; S --> F; S --> X;
  Coord --> S;
  Coord -->|framed TCP dispatch| W1;
  Coord -->|framed TCP dispatch| W2;
  W1 --> CPU; W2 --> CUDA;
  Cli -->|framed TCP submit/query| Coord;
```

The runtime core (`PrefillScheduler`) is thread-safe and never performs network or blocking I/O
while holding its internal lock. CUDA is one concrete accelerator backend behind a vendor-neutral
`Executor` interface; the core never refers to CUDA types.

## Key semantics

- **Admission**: validates every request (zero-token inputs, cached-prefix > prompt, duplicate ids,
  impossible deadlines) and enforces per-tenant outstanding/token limits.
- **Chunking**: deterministic chunk boundaries from max-tokens / max-work / max-memory / policy,
  enabling **chunk-boundary preemption** (not arbitrary kernel-level preemption).
- **Packing**: typed, explainable `CompatibilityKey` / `CompatibilityDecision`; compatible work
  packs into groups up to max-requests/tokens/compute/memory and per-tenant limits.
- **Fairness**: weighted tenant service accounting by scheduled work (not request count), plus
  starvation aging and deterministic tie-breaks; heterogeneous prompt sizes make forward progress.
- **Deadlines/latency**: deadline-aware ordering, urgent-work elevation, deterministic expiry that
  never consumes capacity, and distinction between admission rejection and later expiry.
- **Memory governance**: reservations before dispatch, released on completion/cancel/failure/
  worker-loss/retry; no double-release, no leak, underflow guards.
- **Stale authority**: rejects old epoch, stale worker boot id, stale attempt, stale chunk/work
  generation, duplicate completion, and completion for cancelled terminal work, using explicit
  error/result types rather than silent drops.
- **Persistence/recovery**: deterministic, versioned, checksummed state with strict bounds
  validation and conservative reconciliation of in-flight-at-death work.
- **Explainability**: structured `Explain` trace per request, in text and JSON.
- **Observability**: snapshot, stats, event stream with bounded retention, per-tenant service,
  per-device work, memory reservation state, worker health.

See `docs/architecture.md` for the full design, `docs/api.md` for the public API,
`docs/protocol.md` for the distributed protocol, `docs/persistence.md` for the persistence format,
`docs/testing.md` for the validation methodology, `docs/benchmarks.md` for measured results, and
`docs/limitations.md` for proven limitations.

## Building

Requirements: C++20, CMake ≥ 3.24, Ninja, MSVC 2022 (Windows x64); CUDA 13.1 + a Blackwell device
(RTX 5090, sm_120) optionally for the accelerator backend. The CPU path builds without CUDA.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Using the installed package

```cmake
find_package(PrefillFabric CONFIG REQUIRED)
target_link_libraries(app PRIVATE PrefillFabric::PrefillFabric)
```

## License

Apache License 2.0. Copyright 2026 Summon Software Labs. No telemetry transmission.
