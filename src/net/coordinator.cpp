// Prefill Fabric - coordinator implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/coordinator.hpp"
#include <thread>
#include <chrono>

namespace prefillfabric {

Coordinator::Coordinator(SchedulerConfig cfg, std::shared_ptr<Clock> clock, std::uint16_t port)
    : cfg_(cfg), clock_(std::move(clock)), listener_(port),
      scheduler_(std::make_shared<PrefillScheduler>(cfg_, clock_)) {}

Coordinator::~Coordinator() { shutdown(); }

std::uint16_t Coordinator::port() const { return listener_.port(); }
std::size_t Coordinator::worker_count() const {
  std::lock_guard<std::mutex> lk(conn_mtx_);
  return worker_socks_.size();
}

Result<void> Coordinator::start() {
  scheduler_->unregister_worker(WorkerId(0));
  auto r = listener_.start();
  if (!r) return r;
  stop_.store(false);
  accept_thread_ = std::thread([this] { accept_loop(); });
  schedule_thread_ = std::thread([this] { schedule_loop(); });
  return Result<void>::ok();
}

void Coordinator::shutdown() {
  if (stop_.exchange(true)) return;
  listener_.close();
  if (accept_thread_.joinable()) accept_thread_.join();
  if (schedule_thread_.joinable()) schedule_thread_.join();
}

void Coordinator::accept_loop() {
  while (!stop_.load()) {
    auto s = listener_.accept();
    if (!s) { if (stop_.load()) break; std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
    auto sp = std::make_shared<Socket>(std::move(s.value()));
    std::thread(&Coordinator::handle_connection, this, sp).detach();
  }
}

void Coordinator::handle_connection(std::shared_ptr<Socket> sock) {
  auto fb = sock->recv_frame();
  if (!fb) return;
  auto first = decode_frame(fb.value());
  if (!first) return;

  bool is_worker = (first.value().kind == MsgKind::worker_register);
  bool is_client = (first.value().kind == MsgKind::hello);
  if (!is_worker && !is_client) {
    Message err; err.kind = MsgKind::error;
    err.error = ErrorInfo(ErrorCode::protocol_violation, "expected hello or worker_register");
    auto eb = encode_frame(err); if (eb) sock->send_frame(eb.value().data(), eb.value().size());
    return;
  }

  if (is_worker) {
    const WorkerDescriptor wd = first.value().worker;
    if (wd.worker_id.is_nil()) return;
    {
      std::lock_guard<std::mutex> lk(conn_mtx_);
      workers_[wd.worker_id] = wd;
      worker_socks_[wd.worker_id] = sock;
    }
    scheduler_->register_worker(wd);
    Message ack; ack.kind = MsgKind::worker_register_ack;
    ack.epoch = scheduler_->current_epoch();
    ack.worker = wd;
    auto ab = encode_frame(ack); if (ab) sock->send_frame(ab.value().data(), ab.value().size());
    while (!stop_.load()) {
      auto rb = sock->recv_frame();
      if (!rb) {
        std::lock_guard<std::mutex> lk(conn_mtx_);
        auto it = worker_socks_.find(wd.worker_id);
        if (it != worker_socks_.end() && it->second == sock) worker_socks_.erase(it);
        scheduler_->worker_lost(wd.worker_id);
        return;
      }
      auto msg = decode_frame(rb.value());
      if (!msg) continue;
      if (msg.value().kind == MsgKind::completion) {
        for (const auto& mr : msg.value().member_results) {
          auto res = scheduler_->report_completion(mr);
          if (!res) {
            Message e; e.kind = MsgKind::error; e.error = res.error(); e.request_id = mr.request_id;
            auto eb = encode_frame(e); if (eb) sock->send_frame(eb.value().data(), eb.value().size());
            break;
          }
        }
      } else if (msg.value().kind == MsgKind::ping) {
        Message p; p.kind = MsgKind::pong;
        auto pb = encode_frame(p); if (pb) sock->send_frame(pb.value().data(), pb.value().size());
      } else if (msg.value().kind == MsgKind::shutdown) {
        Message a; a.kind = MsgKind::shutdown_ack;
        auto ab2 = encode_frame(a); if (ab2) sock->send_frame(ab2.value().data(), ab2.value().size());
        return;
      }
    }
  } else {
    while (!stop_.load()) {
      auto rb = sock->recv_frame();
      if (!rb) return;
      auto msg = decode_frame(rb.value());
      if (!msg) continue;
      const Message& m = msg.value();
      switch (m.kind) {
        case MsgKind::submit: {
          std::vector<std::uint32_t> toks = m.tokens;
          { std::lock_guard<std::mutex> lk(conn_mtx_); tokens_[m.request.request_id] = std::move(toks); }
          auto res = scheduler_->submit(m.request);
          Message a; a.kind = MsgKind::submit_ack; a.request_id = m.request.request_id;
          if (res) a.accepted = true; else a.error = res.error();
          auto ab = encode_frame(a); if (ab) sock->send_frame(ab.value().data(), ab.value().size());
          break;
        }
        case MsgKind::cancel: {
          auto cr = scheduler_->cancel(m.request_id);
          Message a; a.kind = MsgKind::cancel_ack; a.request_id = m.request_id;
          a.applied = cr.applied; a.cancel_result = cr; a.error = ErrorInfo(cr.code, cr.message);
          auto ab = encode_frame(a); if (ab) sock->send_frame(ab.value().data(), ab.value().size());
          break;
        }
        case MsgKind::query_state: {
          Message a; a.kind = MsgKind::state; a.epoch = scheduler_->current_epoch();
          for (const auto& w : scheduler_->workers()) if (!w.worker_id.is_nil()) a.workers.push_back(w);
          auto ab = encode_frame(a); if (ab) sock->send_frame(ab.value().data(), ab.value().size());
          break;
        }
        case MsgKind::query_explain: {
          auto ex = scheduler_->explain(m.request_id);
          Message a; a.kind = MsgKind::explain_resp; a.request_id = m.request_id;
          if (ex) { a.explain = ex.value(); a.attempt_id = ex.value().attempt_id; a.lifecycle = scheduler_->lifecycle(m.request_id).value_or(Lifecycle::submitted); a.generation = scheduler_->current_generation(m.request_id); }
          else a.error = ex.error();
          auto ab = encode_frame(a); if (ab) sock->send_frame(ab.value().data(), ab.value().size());
          break;
        }
        case MsgKind::completion: {
          for (const auto& mr : m.member_results) {
            auto res = scheduler_->report_completion(mr);
            if (!res) {
              Message e; e.kind = MsgKind::error; e.error = res.error(); e.request_id = mr.request_id;
              auto eb = encode_frame(e); if (eb) sock->send_frame(eb.value().data(), eb.value().size());
              break;
            }
          }
          break;
        }
        case MsgKind::roll_epoch: {
          auto rr = scheduler_->roll_epoch();
          if (rr) { Message a; a.kind = MsgKind::state; a.epoch = scheduler_->current_epoch(); auto ab = encode_frame(a); if (ab) sock->send_frame(ab.value().data(), ab.value().size()); }
          break;
        }
        case MsgKind::query_stats: {
          Message a; a.kind = MsgKind::stats_resp; a.stats = scheduler_->stats();
          auto ab = encode_frame(a); if (ab) sock->send_frame(ab.value().data(), ab.value().size());
          break;
        }
        case MsgKind::shutdown: {
          Message a; a.kind = MsgKind::shutdown_ack;
          auto ab = encode_frame(a); if (ab) sock->send_frame(ab.value().data(), ab.value().size());
          return;
        }
        default: break;
      }
    }
  }
}

WorkerDescriptor Coordinator::worker_for(WorkerId id) const {
  std::lock_guard<std::mutex> lk(conn_mtx_);
  auto it = workers_.find(id);
  return it == workers_.end() ? WorkerDescriptor() : it->second;
}

bool Coordinator::dispatch_group(const DispatchedGroup& dg) {
  std::shared_ptr<Socket> sock;
  { std::lock_guard<std::mutex> lk(conn_mtx_);
    auto it = worker_socks_.find(dg.assigned_worker);
    if (it == worker_socks_.end()) return false;
    sock = it->second;
  }
  Message d;
  d.kind = MsgKind::dispatch;
  d.group = dg.group;
  d.epoch = scheduler_->current_epoch();
  d.worker.boot_id = worker_for(dg.assigned_worker).boot_id;
  std::uint64_t total = 0;
  for (const auto& m : dg.group.members) {
    std::vector<std::uint32_t> slice;
    { std::lock_guard<std::mutex> lk(conn_mtx_);
      auto it = tokens_.find(m.request_id);
      if (it == tokens_.end()) return false;
      const auto& tvec = it->second;
      if (tvec.size() < m.token_start + m.token_count) return false;
      slice.assign(tvec.begin() + static_cast<std::ptrdiff_t>(m.token_start),
                   tvec.begin() + static_cast<std::ptrdiff_t>(m.token_start + m.token_count));
    }
    for (auto t : slice) d.member_tokens.push_back(t);
    total += m.token_count;
    d.member_token_offsets.push_back(total);
  }
  d.member_token_offsets.insert(d.member_token_offsets.begin(), 0);
  auto fb = encode_frame(d);
  if (!fb) return false;
  return sock->send_frame(fb.value().data(), fb.value().size()).is_ok();
}

void Coordinator::schedule_loop() {
  while (!stop_.load()) {
    auto batch = scheduler_->run_cycle();
    if (batch) {
      for (const auto& dg : batch.value().groups) {
        if (!dispatch_group(dg)) scheduler_->worker_lost(dg.assigned_worker);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace prefillfabric
