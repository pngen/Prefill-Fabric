# Distributed Protocol (framed TCP)

The control plane is a framed TCP protocol over real OS processes. Two kinds of process connect to a
`Coordinator`: **workers** (announce `worker_register`) and **clients** (announce `hello`).

## Framing

- `[u32 little-endian length][u8 version][u8 kind][payload (length-2 bytes)]`.
- `kFrameMaxBytes = 64 MiB`. Zero-length and 1-byte frames, oversized frames, wrong version, unknown
  kinds, truncated bodies, and semantically invalid payloads are rejected with explicit error codes.
- Integers are serialized losslessly as little-endian 64-bit values; identity never passes through a
  floating-point JSON representation.

## Roles / message kinds

- Worker (`worker_register`, `worker_register_ack` carrying epoch, `dispatch` coordinator -> worker,
  `completion` worker -> coordinator, `ping`/`pong`, `shutdown`).
- Client (`hello`, `submit`/`submit_ack`, `cancel`/`cancel_ack`, `query_state`/`state`,
  `query_explain`/`explain_resp`, `query_stats`/`stats_resp`, `roll_epoch`, `shutdown`).

Dispatch carries the formed `ExecutionGroup` (per-member request/attempt/generation/token range), the
per-member token payload and offsets, the coordinator epoch, and the worker boot id. Completion
carries per-member results with the epoch, worker id, and boot id so the coordinator can enforce
authority.

## Stale-authority rejection

`report_completion` rejects (with explicit error results, never silent drops): old epoch
(`stale_epoch`); correct epoch but old worker boot id (`stale_worker`); correct epoch and boot but
obsolete attempt (`stale_attempt`); correct attempt but obsolete chunk/work generation
(`stale_generation`); duplicate completion; completion for cancelled/terminal work
(`terminal_cancelled`). The dispatcher also validates the *currently registered* worker boot id, so a
completion from a restarted worker with an old boot id is rejected even if the worker id is reused.
