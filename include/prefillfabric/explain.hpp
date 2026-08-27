// Prefill Fabric - structured explainability for requests and cycles.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include "prefillfabric/types.hpp"

namespace prefillfabric {

// Minimal JSON string escaping for explain/observability output.
inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          const auto n = static_cast<std::size_t>(std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(static_cast<unsigned char>(c))));
          out.append(buf, n);
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

// Structured explanation of why a request transitioned the way it did.
struct Explain {
  RequestId request_id;
  AttemptId attempt_id;
  std::string lifecycle;
  std::vector<std::string> trace;   // ordered reasons/notes

  void add(std::string line) { trace.push_back(std::move(line)); }
  bool empty() const noexcept { return trace.empty(); }

  std::size_t trace_size() const noexcept { return trace.size(); }

  // Human-readable text.
  std::string to_text() const {
    std::string out = "request ";
    out += std::to_string(request_id.value());
    out += "/attempt " + std::to_string(attempt_id.value()) + " lifecycle=" + lifecycle;
    for (const auto& t : trace) { out += "\n  - "; out += t; }
    return out;
  }

  // JSON output.
  std::string to_json() const {
    std::string out = "{";
    out += "\"request_id\":" + std::to_string(request_id.value()) + ",";
    out += "\"attempt_id\":" + std::to_string(attempt_id.value()) + ",";
    out += "\"lifecycle\":\"" + json_escape(lifecycle) + "\",";
    out += "\"trace\":[";
    for (std::size_t i = 0; i < trace.size(); ++i) {
      if (i) out += ",";
      out += "\"" + json_escape(trace[i]) + "\"";
    }
    out += "]}";
    return out;
  }
};

}  // namespace prefillfabric
