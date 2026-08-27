// Prefill Fabric - worker implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/worker.hpp"
#include "prefillfabric/protocol.hpp"
#include <map>
#include <thread>
#include <chrono>

namespace prefillfabric {

namespace {

// Resolves token slices carried in a dispatch frame for an executor.
struct DispatchResolver : TokenResolver {
  struct Slice { std::uint64_t base = 0; std::vector<std::uint32_t> data; };
  std::map<RequestId, Slice, std::less<RequestId>> slices;
  Result<const std::uint32_t*> resolve_tokens(RequestId r, AttemptId, Generation,
                                               std::uint64_t offset,
                                               std::uint64_t count) const override {
    auto it = slices.find(r);
    if (it == slices.end())
      return Result<const std::uint32_t*>::err(ErrorCode::unknown_request, "no dispatch slice");
    const auto& s = it->second;
    if (offset < s.base || (offset + count) > (s.base + s.data.size()))
      return Result<const std::uint32_t*>::err(ErrorCode::out_of_range, "dispatch slice range");
    return Result<const std::uint32_t*>::ok(s.data.data() + static_cast<std::size_t>(offset - s.base));
  }
};

}  // namespace

Worker::Worker(WorkerId id, const std::string& host, std::uint16_t port,
               std::shared_ptr<Executor> ex, const std::string& backend)
    : id_(id), host_(host), port_(port), ex_(std::move(ex)), backend_(backend),
      boot_id_(WorkerBootId(std::uint64_t(0x9E3779B97F4A7C15ULL ^
          static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())))) {}

void Worker::request_shutdown() { shutdown_.store(true); }

Result<void> Worker::run() {
  auto sock = Socket::connect(host_, port_);
  if (!sock) return Result<void>::err(sock.error());
  Socket& s = sock.value();

  // Register.
  Message reg;
  reg.kind = MsgKind::worker_register;
  reg.worker.worker_id = id_;
  reg.worker.boot_id = boot_id_;
  reg.worker.host = host_;
  reg.worker.backend = backend_;
  reg.worker.state = WorkerState::ready;
  reg.worker.capacity_units = 1;
  auto rb = encode_frame(reg);
  if (!rb) return Result<void>::err(rb.error());
  auto r = s.send_frame(rb.value().data(), rb.value().size());
  if (!r) return r;

  // Read register ack.
  auto af = s.recv_frame();
  if (!af) return Result<void>::err(af.error());
  auto ack = decode_frame(af.value());
  if (!ack) return Result<void>::err(ack.error());
  epoch_ = ack.value().epoch;

  while (!shutdown_.load()) {
    auto f = s.recv_frame();
    if (!f) return Result<void>::err(ErrorCode::io_failure, "coordinator closed connection");
    auto msg = decode_frame(f.value());
    if (!msg) continue;
    const Message& m = msg.value();
    if (m.kind == MsgKind::dispatch) {
      DispatchResolver resolver;
      const auto& offs = m.member_token_offsets;
      for (std::size_t i = 0; i < m.group.members.size(); ++i) {
        const auto& mem = m.group.members[i];
        DispatchResolver::Slice slice;
        slice.base = mem.token_start;
        const std::uint64_t begin = (i + 1 < offs.size()) ? offs[i] : 0;
        const std::uint64_t end = (i + 1 < offs.size()) ? offs[i + 1] : offs.empty() ? 0 : offs.back();
        if (i + 1 >= offs.size()) continue;
        if (end > m.member_tokens.size()) continue;
        slice.data.assign(m.member_tokens.begin() + static_cast<std::ptrdiff_t>(begin),
                          m.member_tokens.begin() + static_cast<std::ptrdiff_t>(end));
        resolver.slices[mem.request_id] = std::move(slice);
      }
      auto er = ex_->execute(m.group, resolver);
      Message comp;
      comp.kind = MsgKind::completion;
      if (er) {
        for (auto& mr : er.value().members) {
          mr.epoch = m.epoch;
          mr.worker = id_;
          mr.boot = boot_id_;
          comp.member_results.push_back(std::move(mr));
        }
      } else {
        for (const auto& mem : m.group.members) {
          MemberResult mr;
          mr.request_id = mem.request_id;
          mr.attempt_id = mem.attempt_id;
          mr.generation = mem.generation;
          mr.outcome = MemberOutcome::retryable_failure;
          mr.failure_code = er.error().code;
          mr.message = er.error().message;
          mr.epoch = m.epoch;
          mr.worker = id_;
          mr.boot = boot_id_;
          comp.member_results.push_back(std::move(mr));
        }
      }
      auto cb = encode_frame(comp);
      if (cb) s.send_frame(cb.value().data(), cb.value().size());
    } else if (m.kind == MsgKind::shutdown) {
      Message a; a.kind = MsgKind::shutdown_ack;
      auto ab = encode_frame(a); if (ab) s.send_frame(ab.value().data(), ab.value().size());
      return Result<void>::ok();
    }
  }
  return Result<void>::ok();
}

}  // namespace prefillfabric
