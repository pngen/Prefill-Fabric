# Limitations

The following limitations are proven by the implementation and test suite, and are stated plainly.

1. **Distributed validation is loopback/single-host.** The multiprocess and atomic-closure proofs use
   real OS processes over the real framed TCP protocol on a single host (127.0.0.1). Network
   partitions, cross-host clocks, and real-failure network behavior are not exercised.
2. **CUDA backend is proven on one NVIDIA device.** It is validated on the RTX 5090 (sm_120, CUDA 13.1).
   No multi-GPU or non-Blackwell device validation is performed here.
3. **Preemption is chunk-boundary, not arbitrary kernel-level.** A dispatched chunk runs to completion;
   cancellation and yield become authoritative between explicit prefill chunks. This is a documented
   design choice, not an omission.
4. **Packing requires an exact typed `CompatibilityKey` match.** Only requests with identical model,
   revision, base model, adapter relation, backend, device, layout, numeric mode, vocabulary, executor
   family, and operator policy fingerprint pack together. This is deliberate and explainable.
5. **Te?lemetry is local only** and never leaves the machine; all observability is written to operator
   -chosen local files and is not transmitted.
6. **Reservations are dispatched/running only.** Queued work is represented by lifecycle state, not a
   real memory reservation; actual device memory is reserved before dispatch and released on terminal,
   cancellation, retry, or worker loss.
