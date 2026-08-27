// Prefill Fabric - coordinator (distributed control plane).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <map>
#include <mutex>
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/network.hpp"
#include "prefillfabric/protocol.hpp"

namespace prefillfabric {

// Coordinates distributed prefill across real worker OS processes over a
// framed TCP control plane. One listener accepts both workers and clients.
class Coordinator {
 public:
  Coordinator(SchedulerConfig cfg, std::shared_ptr<Clock> clock, std::uint16_t port);
  ~Coordinator();
  Coordinator(const Coordinator&) = delete;
  Coordinator& operator=(const Coordinator&) = delete;

  Result<void> start();
  void shutdown();
  std::uint16_t port() const;
  PrefillScheduler& scheduler() { return *scheduler_; }

  // Number of currently registered ready workers.
  std::size_t worker_count() const;

 private:
  void accept_loop();
  void handle_connection(std::shared_ptr<Socket> sock);
  void schedule_loop();
  bool dispatch_group(const DispatchedGroup& dg);
  WorkerDescriptor worker_for(WorkerId id) const;

  SchedulerConfig cfg_;
  std::shared_ptr<Clock> clock_;
  Listener listener_;
  std::shared_ptr<PrefillScheduler> scheduler_;
  std::atomic<bool> stop_{false};
  std::thread accept_thread_;
  std::thread schedule_thread_;
  mutable std::mutex conn_mtx_;
  std::map<WorkerId, std::shared_ptr<Socket>, std::less<WorkerId>> worker_socks_;
  std::map<WorkerId, WorkerDescriptor, std::less<WorkerId>> workers_;
  std::map<RequestId, std::vector<std::uint32_t>, std::less<RequestId>> tokens_;
};

}  // namespace prefillfabric
