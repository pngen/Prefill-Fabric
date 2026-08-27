# Benchmark Methodology

`bench/benchmark.cpp` measures completed useful runtime work, never empty loops or counter-only
iterations. It records the exact configuration (prompt/token sizes, chunk size, queue depth) that
 produced each figure. A representative run on the target machine (AMD Ryzen 7 9800X3D, NVIDIA GeForce
 RTX 5090, CUDA 13.1, MSVC 2022, Release) produced:

| Benchmark | Result | Workload |
|-----------|--------|----------|
| Submission throughput | ~611k req/s | 2,000 512-token requests |
| Schedule + complete | ~2.7k req/s | 2,000 requests, 1.02M tokens |
| CPU executor | ~1.4M tok/s | CPU executor over scheduled tokens |
| Chunk-plan creation | ~3.1M plans/s | 100k plans of a 10k prompt |
| Packed group formation | per-cycle | 2,000 compatible requests |
| Snapshot | ~36k sn/s | 100 snapshots of 2,000 requests |
| Persistence serialize | ~758 states/s | 2,000 request state |
| Persistence deserialize | ~945 states/s | same blob |
| Queue-10k submission | ~870k req/s | 10,000 queued requests |

Correctness checks are not deleted from the runtime path for benchmarks unless a benchmark is
explicitly identified as measuring a lower-level primitive (e.g., chunk plan creation, digest
computation). Scheduling, memory governance, and stale-authority validation remain enabled.
