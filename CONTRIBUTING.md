# Contributing to Prefill Fabric

Contributions from individuals and organizations are accepted on the terms of the Apache License 2.0
without requiring a Contributor License Agreement (CLA) (see NOTICE).

## Process

1. Open an issue describing the change and its motivation. This is a runtime with strict semantic
   invariants; design changes are reviewed for safety and determinism.
2. Fork, branch, implement, and add/remove tests under `tests/`. Every behavioral change needs a
   regression test and, where appropriate, a property/adversarial case.
3. Keep the build clean: `/W4 /WX` (MSVC) with zero warnings in Release and Debug. The CUDA executor
   is compiled by `nvcc` with `/W4 /WX` host flags.
4. Run the full suite: `ctest --test-dir build --output-on-failure`. Do not introduce test timeouts.
5. Submit a pull request. Maintainers do not add Co-authored-by trailers unless you opt in.

## Code conventions

- C++20, no exceptions for ordinary control flow; use `Result<T>` and structured error codes.
- Strongly typed IDs for identities; never interchange request/attempt/worker IDs.
- Determinism is a core requirement: same input and same policy produce same chunks and ordering.
- Never perform network or blocking I/O while holding a scheduler lock; never invoke callbacks while
  holding an internal lock.
- The vendor-neutral core must not reference CUDA types; accelerators live behind `Executor`.

## Reporting defects

Include the build config, environment (MSVC/CUDA/driver/GPU), a minimal reproduction, and the exact
check that failed. Prefer a property or adversarial test that reproduces deterministically with a
printed seed.
