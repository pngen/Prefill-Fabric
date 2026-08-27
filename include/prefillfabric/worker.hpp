// Prefill Fabric - worker (distributed execution).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include "prefillfabric/executor.hpp"
#include "prefillfabric/network.hpp"
#include "prefillfabric/protocol.hpp"

namespace prefillfabric {

// A worker OS process. Connects to a Coordinator, registers with a fresh
// WorkerBootId, receives Dispatch frames, executes them with a concrete
// Executor, and returns Completion frames carrying its authority.
class Worker {
 public:
  Worker(WorkerId id, const std::string& host, std::uint16_t port,
         std::shared_ptr<Executor> ex, const std::string& backend);

  WorkerBootId boot_id() const { return boot_id_; }
  Result<void> run();       // blocks until shutdown
  void request_shutdown();  // thread-safe signal

 private:
  WorkerId id_;
  std::string host_;
  std::uint16_t port_;
  std::shared_ptr<Executor> ex_;
  std::string backend_;
  WorkerBootId boot_id_;
  Epoch epoch_;
  std::atomic<bool> shutdown_{false};
};

}  // namespace prefillfabric
