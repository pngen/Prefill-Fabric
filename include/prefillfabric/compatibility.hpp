// Prefill Fabric - typing packing compatibility with explainable decisions.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include "prefillfabric/types.hpp"
#include "prefillfabric/request.hpp"

namespace prefillfabric {

// Deterministic 64-bit FNV-1a over an octet stream. Used for keys and
// persistence checksums. Stable across platforms for identical inputs.
inline std::uint64_t fnv1a64(std::uint64_t h, const void* data, std::size_t n) noexcept {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= static_cast<std::uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

inline std::uint64_t fnv1a64_str(std::uint64_t h, const std::string& s) noexcept {
  return fnv1a64(h, s.data(), s.size());
}

// The compatibility-relevant attributes for packing. Two requests may be
// packed together exactly when their CompatibilityKey compares equal, AND
// policy-level formation constraints are otherwise satisfiable. The fields
// are typed (not ad hoc string comparisons) and explainable.
struct CompatibilityKey {
  ModelKey model;
  std::string base_model;            // shared base identity among adapters
  AdapterRelation relation = AdapterRelation::none;
  std::string backend = kBackendAny;
  DeviceId device;
  InputLayout layout = InputLayout::seq_major;
  NumericMode numeric_mode = NumericMode::fp32;
  VocabSpec vocab;
  std::string executor_family;       // kernel/executor compatibility family
  std::string policy_fingerprint;    // operator-configured policy constraints

  bool operator==(const CompatibilityKey& o) const noexcept {
    return model == o.model && base_model == o.base_model && relation == o.relation &&
           backend == o.backend && device == o.device && layout == o.layout &&
           numeric_mode == o.numeric_mode && vocab == o.vocab &&
           executor_family == o.executor_family && policy_fingerprint == o.policy_fingerprint;
  }
  bool operator!=(const CompatibilityKey& o) const noexcept { return !(*this == o); }

  // Deterministic 64-bit hash of the typed key.
  std::uint64_t hash() const noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    h = fnv1a64_str(h, model.name);
    h = fnv1a64_str(h, model.revision);
    h = fnv1a64_str(h, base_model);
    h = fnv1a64(h, &relation, sizeof(relation));
    h = fnv1a64_str(h, backend);
    h = fnv1a64(h, &device, sizeof(device));
    h = fnv1a64(h, &layout, sizeof(layout));
    h = fnv1a64(h, &numeric_mode, sizeof(numeric_mode));
    h = fnv1a64_str(h, vocab.tokenizer);
    h = fnv1a64(h, &vocab.vocab_size, sizeof(vocab.vocab_size));
    h = fnv1a64_str(h, executor_family);
    h = fnv1a64_str(h, policy_fingerprint);
    return h;
  }
};

struct HashCompatibilityKey {
  std::size_t operator()(const CompatibilityKey& k) const noexcept {
    return static_cast<std::size_t>(k.hash());
  }
};

// A compact reason code plus a human-readable explanation for a compatibility
// decision. Codes are stable for programmatic handling; text is for operators.
struct CompatibilityReason {
  int code = 0;
  std::string text;
};

// Explains whether two requests (or a request vs. an existing group) may be
// packed together and, if not, precisely why.
struct CompatibilityDecision {
  bool compatible = false;
  std::vector<CompatibilityReason> reasons;
  CompatibilityKey key;
  int non_compat_reason_count = 0;

  bool can_pack() const noexcept { return compatible; }

  std::string to_text() const {
    std::string s = compatible ? "compatible" : "not compatible";
    if (reasons.empty()) return s;
    s += ":";
    for (const auto& r : reasons) {
      s += " [" + std::to_string(r.code) + "] " + r.text;
    }
    return s;
  }
};

}  // namespace prefillfabric
