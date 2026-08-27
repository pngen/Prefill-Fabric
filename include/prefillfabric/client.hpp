// Prefill Fabric - client / driver for the distributed control plane.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include "prefillfabric/network.hpp"
#include "prefillfabric/protocol.hpp"
#include "prefillfabric/request.hpp"
#include "prefillfabric/observability.hpp"

namespace prefillfabric {

// Thin client that speaks the framed protocol to a Coordinator.
class Client {
 public:
  Client(const std::string& host, std::uint16_t port);
  ~Client();
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  Result<void> connect();
  void close();

  // Submit; returns request_id and whether accepted. On rejection, error carries code.
  Result<RequestId> submit(const PrefillRequest& req, const std::vector<std::uint32_t>& tokens);
  Result<CancelResult> cancel(RequestId req);
  Result<Message> query_state();
  Result<Message> query_explain(RequestId req);
  Result<Stats> query_stats();
  Result<Epoch> roll_epoch();
  Result<void> shutdown();

 private:
  Result<Message> roundtrip(const Message& msg, MsgKind expect);
  std::string host_;
  std::uint16_t port_;
  Socket sock_;
};

}  // namespace prefillfabric
