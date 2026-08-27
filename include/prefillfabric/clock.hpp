// Prefill Fabric - injectable monotonic clock.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <chrono>

namespace prefillfabric {

// Nanoseconds since an arbitrary (process-local or monotonic) origin.
// The runtime never interprets these as wall-clock timestamps; they are
// used only for ordering and delay accounting.
using Nanoseconds = std::int64_t;

// Abstraction over the clock so tests can inject a deterministic clock to
// prove wait, starvation, and deadline behavior without real time.
class Clock {
 public:
  virtual ~Clock() = default;
  virtual Nanoseconds now() const noexcept = 0;
};

// A process-wall monotonic ("steady") clock.
class SteadyClock : public Clock {
 public:
  Nanoseconds now() const noexcept override {
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<Nanoseconds>(std::chrono::duration_cast<std::chrono::nanoseconds>(t).count());
  }
};

}  // namespace prefillfabric
