# Persistence Format

`PrefillScheduler` persists deterministic authoritative state and recovers it. The format is a
self-contained binary blob: `[magic u32][version u32][flags u32][payload_len u64][checksum u64][payload]`.

- `magic = 0x50524631` ("PRF1").
- `version = 1`. Any other version is rejected as `unknown_version`.
- `payload` is a flat, deterministic little-endian encoding of the epoch, generation/attempt counters,
  request records, worker records, and tenant records.
- `checksum` is the FNV-1a 64-bit hash over the payload. A mismatch is rejected as `checksum_mismatch`.
  Declared `payload_len` that exceeds the blob, trailing bytes, unreasonable counts, and out-of-range
  enums are rejected as `truncation` or `corrupt_state`. Serialized lengths are never trusted blindly.

## Recoverable state

- Policies/configuration needed for semantic recovery is reconstructed from the persisted records.
- Admitted work, waiting/queue assignment (by lifecycle), request/attempt lineage, chunk progress,
  fairness/service state (`normalized_service`, per-tenant scheduled tokens, weight), epoch/generation
  state, worker metadata, and terminal outcomes are persisted. Reservation state is reconstructed
  conservatively (all active work is re-queued, not double-counted).
- Requests recorded as in-flight at process death are reconciled: the in-flight chunk is re-run,
  completed chunks are preserved, and a new generation is assigned, so late completions are stale.

## Determinism

Given identical inputs and policy, `serialize_state` produces identical bytes, and `deserialize_state`
plus `PrefillScheduler::recover` reconstructs a semantically equivalent authoritative state. Chunk
plans are deterministic, so `next_chunk_token_start` re-maps to the same chunk index.
