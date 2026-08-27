// Prefill Fabric - prefill scheduler core implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/chunking.hpp"
#include "prefillfabric/frontend_math.hpp"
#include <mutex>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdlib>

namespace prefillfabric {

struct PrefillScheduler::Impl {
  SchedulerConfig cfg;
  std::shared_ptr<Clock> clock;
  std::shared_ptr<Executor> attached_executor;
  std::shared_ptr<InMemoryTokenResolver> token_resolver = std::make_shared<InMemoryTokenResolver>();
  std::unique_ptr<ChunkPlanner> planner;
  MemoryGovernor mem;
  mutable std::mutex mtx;
  EventLog events;
  Stats stats;

  Epoch epoch = Epoch(1);
  std::uint64_t next_generation_value = 1;
  std::uint64_t next_attempt_value = 1;
  std::uint64_t next_group_value = 1;
  std::uint64_t next_reservation_value = 1;
  std::uint64_t next_boot_value = 1;
  WorkerId next_worker_value{1};

  struct Rs {
    PrefillRequest request;
    AttemptId attempt;
    Lifecycle lifecycle = Lifecycle::submitted;
    PrefillPlan plan;
    std::uint32_t next_chunk_index = 0;
    std::uint64_t completed_uncached = 0;
    std::uint64_t completed_token_total = 0;
    Generation current_generation;
    ReservationId reservation;
    std::uint64_t reserved_bytes = 0;
    bool in_flight = false;
    GroupId current_group;
    WorkerId dispatched_worker;
    Nanoseconds dispatch_time = 0;
    WorkKind last_work_kind = WorkKind::full_prompt;
    int retries_used = 0;
    double normalized_service = 0.0;
    std::uint64_t aging_cycles = 0;
    Nanoseconds admit_time = 0;
    Nanoseconds first_dispatch_time = 0;
    std::uint64_t chunks_done = 0;
    bool cancel_pending = false;
    bool recovered_work = false;
    std::vector<std::string> trace;
  };

  struct Gs {
    ExecutionGroup group;
    WorkerId worker;
    std::set<RequestId, std::less<RequestId>> outstanding;
  };

  struct Ws {
    WorkerDescriptor desc;
    std::uint64_t busy_units = 0;
    std::shared_ptr<Executor> local_exec;
  };

  std::unordered_map<RequestId, Rs, HashStrongId> requests_;
  std::unordered_map<GroupId, Gs, HashStrongId> groups_;
  std::unordered_map<WorkerId, Ws, HashStrongId> workers_;
  std::map<TenantId, TenantService, std::less<TenantId>> tenants_;

  explicit Impl(SchedulerConfig c, std::shared_ptr<Clock> cl)
      : cfg(c), clock(std::move(cl)), planner(std::make_unique<ChunkPlanner>(c)),
        mem(c.usable_memory_bytes(), c.memory_headroom_bytes) {
    WorkerDescriptor w;
    w.worker_id = WorkerId(0);
    w.boot_id = WorkerBootId(next_boot_value++);
    w.host = "local";
    w.backend = kBackendCpu;
    w.capacity_units = 1;
    w.state = WorkerState::ready;
    Ws ws;
    ws.desc = w;
    workers_[w.worker_id] = ws;
  }

  std::unique_lock<std::mutex> lock() { return std::unique_lock<std::mutex>(mtx); }
  Nanoseconds now() const noexcept { return clock->now(); }

  std::uint32_t index_for_token_start(const PrefillPlan& plan, std::uint64_t start) const noexcept {
    for (std::uint32_t i = 0; i < plan.chunks.size(); ++i) {
      if (plan.chunks[i].token_start == start) return i;
    }
    return 0;
  }

  bool has_pending_chunk(const Rs& s) const noexcept {
    return s.next_chunk_index < s.plan.chunks.size();
  }

  std::uint64_t pending_token_count(const Rs& s) const noexcept {
    if (s.next_chunk_index >= s.plan.chunks.size()) return 0;
    return s.plan.chunks[s.next_chunk_index].token_count;
  }

  ReservationId alloc_reservation() noexcept { return ReservationId(next_reservation_value++); }
  GroupId alloc_group() noexcept { return GroupId(next_group_value++); }
  Generation alloc_generation() noexcept { return Generation(next_generation_value++); }
  AttemptId alloc_attempt() noexcept { return AttemptId(next_attempt_value++); }

  void release_reservation(Rs& s) noexcept {
    if (!s.reservation.is_nil()) {
      mem.release(s.reservation);
      s.reservation = ReservationId();
      s.reserved_bytes = 0;
    }
  }

  std::uint64_t queued_count_locked() const noexcept {
    std::uint64_t n = 0;
    for (const auto& kv : requests_) {
      if (kv.second.lifecycle == Lifecycle::queued && !kv.second.in_flight) ++n;
    }
    return n;
  }

  std::uint64_t waiting_tokens_locked() const noexcept {
    std::uint64_t n = 0;
    for (const auto& kv : requests_) {
      const auto& s = kv.second;
      if (s.lifecycle == Lifecycle::queued && !s.in_flight) n += pending_token_count(s);
    }
    return n;
  }

  std::uint64_t remaining_tokens_locked() const noexcept {
    std::uint64_t n = 0;
    for (const auto& kv : requests_) {
      const auto& s = kv.second;
      if (s.lifecycle == Lifecycle::completed || s.lifecycle == Lifecycle::cancelled ||
          s.lifecycle == Lifecycle::expired || s.lifecycle == Lifecycle::failed_non_retryable ||
          s.lifecycle == Lifecycle::rejected) continue;
      n += (s.request.remaining_prefill_tokens() - s.completed_uncached);
    }
    return n;
  }
};

PrefillScheduler::PrefillScheduler(SchedulerConfig cfg, std::shared_ptr<Clock> clock)
    : impl_(std::make_unique<Impl>(cfg, std::move(clock))) {}

PrefillScheduler::~PrefillScheduler() = default;

const SchedulerConfig& PrefillScheduler::config() const noexcept { return impl_->cfg; }
MemoryGovernor& PrefillScheduler::memory() { return impl_->mem; }
EventLog& PrefillScheduler::events() { return impl_->events; }
const EventLog& PrefillScheduler::events() const { return impl_->events; }
Epoch PrefillScheduler::current_epoch() const { return impl_->epoch; }
std::uint64_t PrefillScheduler::next_generation_value() const { return impl_->next_generation_value; }
std::uint64_t PrefillScheduler::next_attempt_value() const { return impl_->next_attempt_value; }
void PrefillScheduler::set_ring_note(bool) noexcept {}
namespace {

CompatibilityKey make_compat_key(const PrefillRequest& req, const SchedulerConfig& cfg) {
  CompatibilityKey k;
  k.model = req.model;
  k.base_model = req.model.name;
  k.relation = req.adapter_relation;
  k.backend = req.backend.empty() ? kBackendAny : req.backend;
  k.device = req.device;
  k.layout = req.input_layout;
  k.numeric_mode = req.numeric_mode;
  k.vocab = req.vocab;
  k.executor_family = k.backend == kBackendAny ? "default" : k.backend;
  k.policy_fingerprint = std::to_string(cfg.max_requests_per_group) + "|" +
                          std::to_string(cfg.max_tokens_per_group) + "|" +
                          std::to_string(cfg.max_memory_per_chunk);
  return k;
}

}  // namespace


Result<void> PrefillScheduler::submit_with_tokens(const PrefillRequest& req,
                                                  std::vector<std::uint32_t> tokens) {
  auto lk = impl_->lock();
  const auto err = req.validate();
  if (err.code != ErrorCode::ok) return Result<void>::err(err.code, err.message);

  if (impl_->requests_.find(req.request_id) != impl_->requests_.end())
    return Result<void>::err(ErrorCode::duplicate_request, "duplicate request id");

  auto tr = impl_->token_resolver->register_request(req, std::move(tokens));
  if (!tr) return Result<void>::err(tr.error());

  if (impl_->requests_.size() >= impl_->cfg.max_admitted_requests) {
    impl_->stats.rejected++;
    impl_->stats.rejection_by_reason[ErrorCode::capacity_exceeded]++;
    return Result<void>::err(ErrorCode::capacity_exceeded, "admission capacity exceeded");
  }
  auto& ts = impl_->tenants_[req.tenant_id];
  if (ts.weight == 0.0) ts.weight = req.tenant_weight;
  if (ts.outstanding >= impl_->cfg.per_tenant_outstanding_limit) {
    impl_->stats.rejected++;
    impl_->stats.rejection_by_reason[ErrorCode::capacity_exceeded]++;
    return Result<void>::err(ErrorCode::capacity_exceeded, "tenant outstanding limit exceeded");
  }
  const std::uint64_t rem = req.remaining_prefill_tokens();
  if (ts.admitted_tokens + rem > impl_->cfg.per_tenant_admitted_token_limit) {
    impl_->stats.rejected++;
    impl_->stats.rejection_by_reason[ErrorCode::capacity_exceeded]++;
    return Result<void>::err(ErrorCode::capacity_exceeded, "tenant admitted-token limit exceeded");
  }

  auto& s = impl_->requests_[req.request_id];
  s.request = req;
  s.attempt = impl_->alloc_attempt();
  s.lifecycle = Lifecycle::queued;
  s.admit_time = impl_->now();
  s.plan = impl_->planner->plan(req, s.attempt, Generation(0));
  s.next_chunk_index = 0;
  s.completed_uncached = 0;
  s.completed_token_total = 0;
  s.current_generation = Generation(0);
  s.retries_used = 0;
  s.cancel_pending = false;
  s.recovered_work = false;
  s.normalized_service = 0.0;
  s.chunks_done = 0;
  s.trace.clear();
  s.trace.push_back("admitted; remaining_prefill=" + std::to_string(rem));

  impl_->stats.submitted++;
  impl_->stats.admitted++;
  ts.outstanding++;
  ts.admitted_tokens += rem;

  if (rem == 0) {
    // Fully covered by a valid reusable prefix: no prefill execution is needed.
    s.lifecycle = Lifecycle::completed;
    s.completed_uncached = 0;
    s.completed_token_total = 0;
    impl_->stats.completed++;
    s.trace.push_back("fully-cached prompt; no prefill work remains");
    ts.completed_requests++;
    ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
    impl_->events.record(impl_->now(), "submit", req.request_id, s.attempt, "complete-immediate");
    return Result<void>::ok();
  }

  impl_->events.record(impl_->now(), "submit", req.request_id, s.attempt, "queued");
  return Result<void>::ok();
}

Result<void> PrefillScheduler::submit(const PrefillRequest& req) {
  std::vector<std::uint32_t> tokens(req.prompt_token_count);
  std::uint64_t seed = 0x9E3779B97F4A7C15ULL ^ req.request_id.value();
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
    tokens[i] = static_cast<std::uint32_t>(seed & 0xFFFFFFFFu);
  }
  return submit_with_tokens(req, std::move(tokens));
}

Result<DispatchBatch> PrefillScheduler::run_cycle() {
  auto lk = impl_->lock();
  const Nanoseconds now = impl_->now();
  DispatchBatch batch;

  // ---- Deadline / queue-delay expiry sweep ----
  for (auto& kv : impl_->requests_) {
    auto& s = kv.second;
    if (s.lifecycle == Lifecycle::completed || s.lifecycle == Lifecycle::cancelled ||
        s.lifecycle == Lifecycle::expired || s.lifecycle == Lifecycle::failed_non_retryable ||
        s.lifecycle == Lifecycle::rejected)
      continue;
    bool exp = false;
    if (s.request.deadline && now > *s.request.deadline) exp = true;
    else if (!s.in_flight && s.request.max_queue_delay &&
             (now - s.admit_time) > *s.request.max_queue_delay) {
      impl_->stats.queue_delay_exceeded++;
      exp = true;
    }
    if (exp) {
      s.lifecycle = Lifecycle::expired;
      if (s.in_flight) {
        // Remove from outstanding; a late completion will be rejected.
        auto git = impl_->groups_.find(s.current_group);
        if (git != impl_->groups_.end()) {
          git->second.outstanding.erase(s.request.request_id);
          if (git->second.outstanding.empty()) {
            auto wit = impl_->workers_.find(git->second.worker);
            if (wit != impl_->workers_.end() && wit->second.busy_units > 0) wit->second.busy_units--;
            impl_->groups_.erase(git);
          }
        }
      }
      impl_->release_reservation(s);
      impl_->stats.expired++;
      impl_->stats.deadline_missed++;
      auto& tsv = impl_->tenants_[s.request.tenant_id];
      tsv.outstanding = tsv.outstanding > 0 ? tsv.outstanding - 1 : 0;
      s.trace.push_back("expired; deadline=" + std::to_string(s.request.deadline ? std::int64_t(*s.request.deadline) : 0));
      impl_->events.record(now, "expire", s.request.request_id, s.attempt, "deadline");
    }
  }

  // ---- Candidate selection (fairness / priority / deadline / aging) ----
  std::vector<RequestId> cands;
  for (auto& kv : impl_->requests_) {
    auto& s = kv.second;
    if (s.lifecycle == Lifecycle::queued && !s.in_flight && impl_->has_pending_chunk(s))
      cands.push_back(kv.first);
  }

  auto sorter = [&](RequestId a, RequestId b) -> bool {
    const auto& sa = impl_->requests_[a];
    const auto& sb = impl_->requests_[b];
    const auto& ta = impl_->tenants_[sa.request.tenant_id];
    const auto& tb = impl_->tenants_[sb.request.tenant_id];
    auto urgent = [&](const PrefillScheduler::Impl::Rs& s) -> bool {
      return s.request.deadline.has_value() && (now + impl_->cfg.deadline_slack_ns >= *s.request.deadline);
    };
    const bool ua = urgent(sa), ub = urgent(sb);
    if (ua != ub) return ua;   // urgent first
    std::int64_t da = sa.request.deadline ? std::int64_t(*sa.request.deadline) : std::numeric_limits<std::int64_t>::max();
    std::int64_t db = sb.request.deadline ? std::int64_t(*sb.request.deadline) : std::numeric_limits<std::int64_t>::max();
    if (da != db) return da < db;
    if (sa.request.priority != sb.request.priority) return sa.request.priority > sb.request.priority;
    if (sa.request.latency_class != sb.request.latency_class)
      return static_cast<int>(sa.request.latency_class) > static_cast<int>(sb.request.latency_class);
    const double wta = ta.weight > 0 ? ta.weight : 1.0;
    const double wtb = tb.weight > 0 ? tb.weight : 1.0;
    const double fa = ta.normalized_service / wta;
    const double fb = tb.normalized_service / wtb;
    if (fa != fb) return fa < fb;
    if (sa.aging_cycles != sb.aging_cycles) return sa.aging_cycles > sb.aging_cycles;
    if (sa.admit_time != sb.admit_time) return sa.admit_time < sb.admit_time;
    return a.value() < b.value();
  };
  std::sort(cands.begin(), cands.end(), sorter);

  // ---- Formation ----
  struct OpenGroup {
    CompatibilityKey key;
    ExecutionGroup group;
    std::vector<RequestId> rids;
    std::uint64_t total_tokens = 0;
    std::uint64_t total_compute = 0;
    std::uint64_t max_mem = 0;
    std::map<TenantId, std::uint64_t, std::less<TenantId>> tenant_tokens;
  };
  std::vector<OpenGroup> open;

  auto compatible_with_group = [&](const OpenGroup& g, const PrefillChunk& ch,
                                    const CompatibilityKey& k) -> CompatibilityDecision {
    CompatibilityDecision d;
    d.key = g.key;
    if (!(g.key == k)) {
      d.compatible = false;
      d.non_compat_reason_count = 1;
      d.reasons.push_back({1, "different CompatibilityKey"});
      return d;
    }
    if (g.group.size() >= impl_->cfg.max_requests_per_group) {
      d.reasons.push_back({2, "max requests per group reached"}); return d;
    }
    if (g.total_tokens + ch.token_count > impl_->cfg.max_tokens_per_group) {
      d.reasons.push_back({3, "max tokens per group exceeded"}); return d;
    }
    if (g.total_compute + ch.compute_estimate > impl_->cfg.max_compute_per_group) {
      d.reasons.push_back({4, "max compute per group exceeded"}); return d;
    }
    const std::uint64_t newmem = std::max(g.max_mem, ch.memory_estimate_bytes);
    if (newmem > impl_->cfg.max_memory_per_group) {
      d.reasons.push_back({5, "max memory per group exceeded"}); return d;
    }
    d.compatible = true;
    d.reasons.push_back({0, "compatible"});
    return d;
  };

  for (const RequestId rid : cands) {
    auto& s = impl_->requests_[rid];
    if (s.lifecycle != Lifecycle::queued || s.in_flight || !impl_->has_pending_chunk(s)) continue;
    const PrefillChunk& ch = s.plan.chunks[s.next_chunk_index];
    const CompatibilityKey k = make_compat_key(s.request, impl_->cfg);

    bool placed = false;
    for (auto& g : open) {
      const auto dec = compatible_with_group(g, ch, k);
      if (dec.compatible) {
        ExecutableMember m;
        m.request_id = rid;
        m.attempt_id = s.attempt;
        m.generation = Generation(0);  // assigned at dispatch
        m.parent_generation = Generation(0);
        m.token_start = ch.token_start;
        m.token_count = ch.token_count;
        m.work_kind = s.recovered_work ? WorkKind::worker_recovered : ch.work_kind;
        m.memory_estimate_bytes = ch.memory_estimate_bytes;
        m.compute_estimate = ch.compute_estimate;
        m.member_index = g.group.size();
        g.group.members.push_back(m);
        g.group.total_token_count += ch.token_count;
        g.group.sum_compute_estimate += ch.compute_estimate;
        g.group.max_memory_estimate_bytes = std::max(g.group.max_memory_estimate_bytes, ch.memory_estimate_bytes);
        g.rids.push_back(rid);
        g.total_tokens += ch.token_count;
        g.total_compute += ch.compute_estimate;
        g.max_mem = std::max(g.max_mem, ch.memory_estimate_bytes);
        g.tenant_tokens[s.request.tenant_id] += ch.token_count;
        placed = true;
        s.trace.push_back("packed into group (compatible)");
        break;
      }
    }
    if (!placed) {
      OpenGroup g;
      g.key = k;
      g.group.key = k;
      g.group.group_id = impl_->alloc_group();
      g.group.backend = k.backend;
      g.group.device = k.device;
      ExecutableMember m;
      m.request_id = rid;
      m.attempt_id = s.attempt;
      m.generation = Generation(0);
      m.parent_generation = Generation(0);
      m.token_start = ch.token_start;
      m.token_count = ch.token_count;
      m.work_kind = s.recovered_work ? WorkKind::worker_recovered : ch.work_kind;
      m.memory_estimate_bytes = ch.memory_estimate_bytes;
      m.compute_estimate = ch.compute_estimate;
      g.group.members.push_back(m);
      g.group.total_token_count += ch.token_count;
      g.group.sum_compute_estimate += ch.compute_estimate;
      g.group.max_memory_estimate_bytes = ch.memory_estimate_bytes;
      g.rids.push_back(rid);
      g.total_tokens += ch.token_count;
      g.total_compute += ch.compute_estimate;
      g.max_mem = ch.memory_estimate_bytes;
      g.tenant_tokens[s.request.tenant_id] += ch.token_count;
      open.push_back(std::move(g));
    }
  }

  // ---- Reserve + assign workers + build batch ----
  for (auto& g : open) {
    if (g.group.members.empty()) continue;
    // Build the group-id key for worker selection.
    std::string backend = g.group.backend;
    // Choose worker deterministically (lowest id).
    WorkerId chosen;
    bool have = false;
    for (auto& wk : impl_->workers_) {
      auto& w = wk.second;
      if (w.desc.state != WorkerState::ready && w.desc.state != WorkerState::registered) continue;
      if (w.busy_units >= w.desc.capacity_units) continue;
      const bool backend_match = (backend == kBackendAny) || (w.desc.backend == kBackendAny) || (w.desc.backend == backend);
      if (!backend_match) continue;
      if (!have || wk.first.value() < chosen.value()) { chosen = wk.first; have = true; }
    }
    if (!have) {
      batch.notes.push_back("no compatible worker available; deferred");
      for (const auto rid : g.rids) impl_->requests_[rid].trace.push_back("deferred: no compatible worker");
      continue;
    }

    // Reserve per-member memory; if any fails, roll back and defer.
    std::vector<ReservationId> reserved;
    std::vector<RequestId> reserved_rids;
    bool ok = true;
    for (const auto rid : g.rids) {
      auto& s = impl_->requests_[rid];
      const std::uint64_t mb = s.plan.chunks[s.next_chunk_index].memory_estimate_bytes;
      const ReservationId rv = impl_->alloc_reservation();
      auto rres = impl_->mem.reserve(rv, rid, mb, ReservationKind::reserved);
      if (!rres) { ok = false; break; }
      reserved.push_back(rv);
      reserved_rids.push_back(rid);
      s.reservation = rv;
      s.reserved_bytes = mb;
    }
    if (!ok) {
      // Roll back reservations of this group.
      for (const auto& rv : reserved) impl_->mem.release(rv);
      for (auto& kv : impl_->requests_) {
        for (const auto rid : reserved_rids) {
          if (kv.first == rid) { kv.second.reservation = ReservationId(); kv.second.reserved_bytes = 0; }
        }
      }
      batch.notes.push_back("memory reservation failed; group deferred");
      for (const auto rid : g.rids) impl_->requests_[rid].trace.push_back("deferred: memory unavailable");
      continue;
    }

    // Commit dispatch: establish generation + authority + in-flight state.
    const GroupId gid = g.group.group_id;
    PrefillScheduler::Impl::Gs gs;
    gs.worker = chosen;
    gs.group = g.group;
    std::uint64_t group_reserved_bytes = 0;
    for (std::size_t i = 0; i < g.rids.size(); ++i) {
      const RequestId rid = g.rids[i];
      auto& s = impl_->requests_[rid];
      const Generation gen = impl_->alloc_generation();
      s.current_generation = gen;
      s.in_flight = true;
      s.lifecycle = Lifecycle::dispatched;
      s.current_group = gid;
      s.dispatched_worker = chosen;
      s.dispatch_time = now;
      if (s.first_dispatch_time == 0) {
        s.first_dispatch_time = now;
        impl_->stats.queue_latency_total_ns += (now - s.admit_time);
        impl_->stats.queue_latency_samples++;
      }
      gs.group.members[i].generation = gen;
      gs.group.members[i].assigned_worker = chosen;
      gs.outstanding.insert(rid);
      group_reserved_bytes += gs.group.members[i].memory_estimate_bytes;
      impl_->stats.scheduled_tokens += gs.group.members[i].token_count;
    }
    impl_->workers_[chosen].busy_units++;
    impl_->groups_[gid] = std::move(gs);
    impl_->stats.groups_formed++;
    impl_->stats.group_size_distribution[static_cast<std::uint32_t>(g.rids.size())]++;

    DispatchedGroup dg;
    dg.group_id = gid;
    dg.group = impl_->groups_[gid].group;
    dg.assigned_worker = chosen;
    dg.reservation_id = impl_->requests_[g.rids[0]].reservation;
    dg.reserved_bytes = group_reserved_bytes;
    batch.groups.push_back(std::move(dg));
    batch.cycle_scheduled_tokens += g.group.total_token_count;
  }

  // Aging: bump every queued candidate that was not dispatched this cycle.
  for (auto& kv : impl_->requests_) {
    auto& s = kv.second;
    if (s.lifecycle == Lifecycle::queued && !s.in_flight) s.aging_cycles++;
  }

  impl_->stats.queue_depth = static_cast<std::uint64_t>(impl_->queued_count_locked());
  impl_->stats.waiting_tokens = impl_->waiting_tokens_locked();
  impl_->stats.remaining_prefill_tokens = impl_->remaining_tokens_locked();
  return Result<DispatchBatch>::ok(std::move(batch));
}

bool is_terminal_lifecycle(Lifecycle l) noexcept {
  return l == Lifecycle::completed || l == Lifecycle::cancelled || l == Lifecycle::expired ||
         l == Lifecycle::failed_non_retryable || l == Lifecycle::rejected;
}

Result<void> PrefillScheduler::report_completion(const MemberResult& result) {
  auto lk = impl_->lock();
  if (result.epoch.is_nil())
    return Result<void>::err(ErrorCode::illegal_state, "completion carries no epoch");
  if (result.epoch != impl_->epoch) {
    impl_->stats.stale_authority_rejected++;
    impl_->events.record(impl_->now(), "stale", result.request_id, result.attempt_id, "epoch");
    return Result<void>::err(ErrorCode::stale_epoch, "stale scheduler epoch");
  }

  auto it = impl_->requests_.find(result.request_id);
  if (it == impl_->requests_.end())
    return Result<void>::err(ErrorCode::unknown_request, "unknown request");
  auto& s = it->second;

  // Stale worker boot validation against the CURRENTLY registered worker.
  if (!result.worker.is_nil() || !result.boot.is_nil()) {
    auto wit = impl_->workers_.find(result.worker);
    if (wit == impl_->workers_.end() || wit->second.desc.boot_id != result.boot) {
      impl_->stats.stale_authority_rejected++;
      impl_->events.record(impl_->now(), "stale", result.request_id, result.attempt_id, "worker-boot");
      return Result<void>::err(ErrorCode::stale_worker, "stale worker boot id");
    }
  }

  if (result.attempt_id != s.attempt) {
    impl_->stats.stale_authority_rejected++;
    impl_->events.record(impl_->now(), "stale", result.request_id, result.attempt_id, "attempt");
    return Result<void>::err(ErrorCode::stale_attempt, "stale attempt");
  }
  if (result.generation != s.current_generation) {
    impl_->stats.stale_authority_rejected++;
    impl_->events.record(impl_->now(), "stale", result.request_id, result.attempt_id, "generation");
    return Result<void>::err(ErrorCode::stale_generation, "stale chunk/work generation");
  }
  if (is_terminal_lifecycle(s.lifecycle)) {
    impl_->stats.duplicate_completion_rejected++;
    impl_->events.record(impl_->now(), "stale", result.request_id, result.attempt_id, "terminal");
    return Result<void>::err(ErrorCode::terminal_cancelled, "completion for terminal request");
  }
  if (!s.in_flight) {
    impl_->stats.duplicate_completion_rejected++;
    impl_->events.record(impl_->now(), "stale", result.request_id, result.attempt_id, "no-inflight");
    return Result<void>::err(ErrorCode::duplicate_completion, "no in-flight work for this generation");
  }
  if (!result.worker.is_nil() && result.worker != s.dispatched_worker) {
    impl_->stats.stale_authority_rejected++;
    return Result<void>::err(ErrorCode::stale_worker, "completion from a different worker");
  }

  // ---- Authoritative application ----
  const PrefillChunk& ch = s.plan.chunks[s.next_chunk_index];
  s.in_flight = false;
  s.chunks_done++;
  s.completed_uncached += ch.token_count;
  s.completed_token_total += ch.token_count;
  impl_->stats.completed_tokens += ch.token_count;
  impl_->stats.execution_latency_total_ns += (impl_->now() - s.dispatch_time);
  impl_->stats.execution_latency_samples++;
  auto& ts = impl_->tenants_[s.request.tenant_id];
  ts.scheduled_tokens += ch.token_count;
  s.normalized_service += static_cast<double>(ch.token_count);

  if (result.outcome == MemberOutcome::stale_rejected) {
    impl_->stats.stale_authority_rejected++;
  } else if (result.outcome == MemberOutcome::retryable_failure) {
    if (s.retries_used < s.request.max_retries) {
      s.retries_used++;
      s.attempt = impl_->alloc_attempt();
      s.plan = impl_->planner->plan(s.request, s.attempt, Generation(0));
      s.next_chunk_index = 0;
      s.completed_uncached = 0;
      s.completed_token_total = 0;
      s.current_generation = Generation(0);
      s.lifecycle = Lifecycle::queued;
      s.last_work_kind = WorkKind::retried;
      s.recovered_work = false;
      impl_->stats.retried++;
      s.trace.push_back("retryable failure; new attempt " + std::to_string(s.attempt.value()));
      impl_->events.record(impl_->now(), "retry", s.request.request_id, s.attempt, "attempt " + std::to_string(s.attempt.value()));
    } else {
      s.lifecycle = Lifecycle::failed_non_retryable;
      impl_->stats.failed_non_retryable++;
      impl_->stats.completed_tokens += 0;
      ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
      if (ts.admitted_tokens >= s.request.remaining_prefill_tokens())
        ts.admitted_tokens -= s.request.remaining_prefill_tokens();
      s.trace.push_back("retry budget exhausted; non-retryable failure");
      impl_->events.record(impl_->now(), "fail", s.request.request_id, s.attempt, "non-retryable");
    }
  } else if (result.outcome == MemberOutcome::non_retryable_failure) {
    s.lifecycle = Lifecycle::failed_non_retryable;
    impl_->stats.failed_non_retryable++;
    ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
    if (ts.admitted_tokens >= s.request.remaining_prefill_tokens())
      ts.admitted_tokens -= s.request.remaining_prefill_tokens();
    s.trace.push_back("non-retryable failure: " + result.message);
    impl_->events.record(impl_->now(), "fail", s.request.request_id, s.attempt, result.message);
  } else if (result.outcome == MemberOutcome::cancelled) {
    s.lifecycle = Lifecycle::cancelled;
    s.cancel_pending = false;
    impl_->stats.cancelled++;
    ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
    if (ts.admitted_tokens >= s.request.remaining_prefill_tokens())
      ts.admitted_tokens -= s.request.remaining_prefill_tokens();
    s.trace.push_back("cancelled");
    impl_->events.record(impl_->now(), "cancel", s.request.request_id, s.attempt, "completion");
  } else if (result.outcome == MemberOutcome::expired) {
    s.lifecycle = Lifecycle::expired;
    impl_->stats.expired++;
    impl_->stats.deadline_missed++;
    ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
    if (ts.admitted_tokens >= s.request.remaining_prefill_tokens())
      ts.admitted_tokens -= s.request.remaining_prefill_tokens();
    s.trace.push_back("expired during execution");
  } else {  // success
    if (s.cancel_pending) {
      s.lifecycle = Lifecycle::cancelled;
      s.cancel_pending = false;
      impl_->stats.cancelled++;
      ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
      if (ts.admitted_tokens >= s.request.remaining_prefill_tokens())
        ts.admitted_tokens -= s.request.remaining_prefill_tokens();
      s.trace.push_back("cancelled at chunk boundary");
      impl_->events.record(impl_->now(), "cancel", s.request.request_id, s.attempt, "chunk-boundary");
    } else if (s.next_chunk_index + 1 < s.plan.chunks.size()) {
      s.next_chunk_index++;
      s.lifecycle = Lifecycle::queued;
      s.trace.push_back("chunk complete; advancing to next chunk");
    } else {
      s.lifecycle = Lifecycle::completed;
      impl_->stats.completed++;
      ts.completed_requests++;
      ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
      if (ts.admitted_tokens >= s.request.remaining_prefill_tokens())
        ts.admitted_tokens -= s.request.remaining_prefill_tokens();
      s.trace.push_back("prefill completed");
      impl_->events.record(impl_->now(), "complete", s.request.request_id, s.attempt, "ok");
    }
  }

  impl_->release_reservation(s);
  // Remove completed member from its group and free the worker if the group finished.
  if (!s.current_group.is_nil()) {
    auto git = impl_->groups_.find(s.current_group);
    if (git != impl_->groups_.end()) {
      git->second.outstanding.erase(s.request.request_id);
      if (git->second.outstanding.empty()) {
        auto wit = impl_->workers_.find(git->second.worker);
        if (wit != impl_->workers_.end() && wit->second.busy_units > 0) wit->second.busy_units--;
        impl_->groups_.erase(git);
      }
    }
    s.current_group = GroupId();
  }

  impl_->stats.remaining_prefill_tokens = impl_->remaining_tokens_locked();
  return Result<void>::ok();
}

CancelResult PrefillScheduler::cancel(RequestId req) {
  auto lk = impl_->lock();
  auto it = impl_->requests_.find(req);
  if (it == impl_->requests_.end())
    return CancelResult{false, false, ErrorCode::not_found, "unknown request"};
  auto& s = it->second;
  if (is_terminal_lifecycle(s.lifecycle))
    return CancelResult{false, true, ErrorCode::terminal_cancelled, "request already terminal"};
  if (s.in_flight) {
    s.cancel_pending = true;
    s.trace.push_back("cancel requested; authoritative at next safe chunk boundary");
    impl_->events.record(impl_->now(), "cancel", req, s.attempt, "pending chunk boundary");
    return CancelResult{true, false, ErrorCode::ok, "cancellation pending at chunk boundary"};
  }
  s.lifecycle = Lifecycle::cancelled;
  impl_->release_reservation(s);
  impl_->stats.cancelled++;
  auto& ts = impl_->tenants_[s.request.tenant_id];
  ts.outstanding = ts.outstanding > 0 ? ts.outstanding - 1 : 0;
  if (ts.admitted_tokens >= s.request.remaining_prefill_tokens())
    ts.admitted_tokens -= s.request.remaining_prefill_tokens();
  if (!s.current_group.is_nil()) {
    auto git = impl_->groups_.find(s.current_group);
    if (git != impl_->groups_.end()) {
      git->second.outstanding.erase(req);
      if (git->second.outstanding.empty()) {
        auto wit = impl_->workers_.find(git->second.worker);
        if (wit != impl_->workers_.end() && wit->second.busy_units > 0) wit->second.busy_units--;
        impl_->groups_.erase(git);
      }
    }
    s.current_group = GroupId();
  }
  s.trace.push_back("cancelled while waiting");
  impl_->events.record(impl_->now(), "cancel", req, s.attempt, "waiting");
  return CancelResult{true, false, ErrorCode::ok, "cancelled"};
}

CancelResult PrefillScheduler::cancel_generation(RequestId req, Generation only_gen) {
  auto lk = impl_->lock();
  auto it = impl_->requests_.find(req);
  if (it == impl_->requests_.end())
    return CancelResult{false, false, ErrorCode::not_found, "unknown request"};
  auto& s = it->second;
  if (is_terminal_lifecycle(s.lifecycle))
    return CancelResult{false, true, ErrorCode::terminal_cancelled, "request already terminal"};
  if (s.in_flight && s.current_generation == only_gen) {
    s.cancel_pending = true;
    s.trace.push_back("generation cancel requested; authoritative at chunk boundary");
    return CancelResult{true, false, ErrorCode::ok, "generation cancellation pending"};
  }
  return CancelResult{false, false, ErrorCode::not_found, "generation not in-flight"};
}

Result<void> PrefillScheduler::force_retry(RequestId req) {
  auto lk = impl_->lock();
  auto it = impl_->requests_.find(req);
  if (it == impl_->requests_.end())
    return Result<void>::err(ErrorCode::not_found, "unknown request");
  auto& s = it->second;
  if (is_terminal_lifecycle(s.lifecycle))
    return Result<void>::err(ErrorCode::illegal_state, "cannot retry a terminal request");
  if (s.in_flight) {
    // Release the in-flight reservation and detach from any group.
    impl_->release_reservation(s);
    s.in_flight = false;
    if (!s.current_group.is_nil()) {
      auto git = impl_->groups_.find(s.current_group);
      if (git != impl_->groups_.end()) {
        auto wit = impl_->workers_.find(git->second.worker);
        if (wit != impl_->workers_.end() && wit->second.busy_units > 0) wit->second.busy_units--;
        impl_->groups_.erase(git);
      }
      s.current_group = GroupId();
    }
  }
  if (s.retries_used >= s.request.max_retries)
    return Result<void>::err(ErrorCode::capacity_exceeded, "retry budget exhausted");
  s.retries_used++;
  s.attempt = impl_->alloc_attempt();
  s.plan = impl_->planner->plan(s.request, s.attempt, Generation(0));
  s.next_chunk_index = 0;
  s.completed_uncached = 0;
  s.completed_token_total = 0;
  s.current_generation = Generation(0);
  s.lifecycle = Lifecycle::queued;
  s.last_work_kind = WorkKind::retried;
  s.recovered_work = false;
  impl_->stats.retried++;
  s.trace.push_back("forced retry; new attempt " + std::to_string(s.attempt.value()));
  impl_->events.record(impl_->now(), "retry", req, s.attempt, "forced");
  return Result<void>::ok();
}

Result<void> PrefillScheduler::register_worker(const WorkerDescriptor& w) {
  auto lk = impl_->lock();
  if (w.worker_id.is_nil())
    return Result<void>::err(ErrorCode::invalid_argument, "worker id is nil");
  auto& ws = impl_->workers_[w.worker_id];
  if (ws.desc.boot_id.is_nil() && !w.boot_id.is_nil()) {
    ws.desc = w;
  } else if (!w.boot_id.is_nil()) {
    ws.desc = w;
  } else {
    WorkerDescriptor d = w;
    d.boot_id = WorkerBootId(impl_->next_boot_value++);
    ws.desc = d;
  }
  ws.busy_units = 0;
  if (ws.desc.state == WorkerState::unknown) ws.desc.state = WorkerState::ready;
  ws.local_exec.reset();
  impl_->events.record(impl_->now(), "worker-register", RequestId(), AttemptId(),
                       "id=" + std::to_string(w.worker_id.value()));
  return Result<void>::ok();
}

Result<void> PrefillScheduler::unregister_worker(WorkerId w) {
  auto lk = impl_->lock();
  auto it = impl_->workers_.find(w);
  if (it == impl_->workers_.end()) return Result<void>::err(ErrorCode::not_found, "unknown worker");
  it->second.desc.state = WorkerState::retired;
  impl_->events.record(impl_->now(), "worker-retire", RequestId(), AttemptId(),
                       "id=" + std::to_string(w.value()));
  return Result<void>::ok();
}

Result<void> PrefillScheduler::worker_lost(WorkerId w) {
  auto lk = impl_->lock();
  for (auto git = impl_->groups_.begin(); git != impl_->groups_.end();) {
    if (git->second.worker != w) { ++git; continue; }
    for (const RequestId rid : git->second.outstanding) {
      auto it = impl_->requests_.find(rid);
      if (it != impl_->requests_.end()) {
        auto& s = it->second;
        if (s.in_flight && s.lifecycle == Lifecycle::dispatched) {
          s.in_flight = false;
          s.recovered_work = true;
          impl_->release_reservation(s);
          s.current_generation = Generation(0);
          s.lifecycle = Lifecycle::queued;
          // Keep the same attempt and same next_chunk_index: redo the in-flight chunk.
          s.trace.push_back("recovered after worker loss");
          impl_->events.record(impl_->now(), "worker-lost", rid, s.attempt, "recovered");
        }
      }
    }
    impl_->groups_.erase(git++);
  }
  auto wit = impl_->workers_.find(w);
  if (wit != impl_->workers_.end()) {
    wit->second.busy_units = 0;
    wit->second.desc.state = WorkerState::lost;
  }
  impl_->events.record(impl_->now(), "worker-lost", RequestId(), AttemptId(),
                       "id=" + std::to_string(w.value()));
  return Result<void>::ok();
}

Result<void> PrefillScheduler::roll_epoch() {
  auto lk = impl_->lock();
  impl_->epoch = Epoch(impl_->epoch.value() + 1);
  impl_->events.record(impl_->now(), "epoch-roll", RequestId(), AttemptId(),
                       "epoch=" + std::to_string(impl_->epoch.value()));
  return Result<void>::ok();
}

void PrefillScheduler::attach_executor(std::shared_ptr<Executor> ex, const std::string& backend) {
  auto lk = impl_->lock();
  impl_->attached_executor = ex;
  auto it = impl_->workers_.find(WorkerId(0));
  if (it == impl_->workers_.end()) {
    PrefillScheduler::Impl::Ws ws;
    ws.desc.worker_id = WorkerId(0);
    ws.desc.boot_id = WorkerBootId(impl_->next_boot_value++);
    ws.desc.host = "local";
    ws.desc.backend = backend.empty() ? (ex ? ex->backend() : kBackendCpu) : backend;
    ws.desc.capacity_units = 1;
    ws.desc.state = WorkerState::ready;
    ws.desc.devices.push_back(ex ? ex->device() : DeviceDescriptor());
    ws.local_exec = ex;
    impl_->workers_[WorkerId(0)] = ws;
  } else {
    it->second.local_exec = ex;
    it->second.desc.backend = backend.empty() ? (ex ? ex->backend() : kBackendCpu) : backend;
    it->second.desc.state = WorkerState::ready;
    it->second.desc.devices.clear();
    if (ex) it->second.desc.devices.push_back(ex->device());
  }
}

std::shared_ptr<Executor> PrefillScheduler::executor() const {
  auto lk = impl_->lock();
  return impl_->attached_executor;
}

WorkerId PrefillScheduler::local_worker_id() const { return WorkerId(0); }

std::vector<WorkerDescriptor> PrefillScheduler::workers() const {
  auto lk = impl_->lock();
  std::vector<WorkerDescriptor> out;
  for (const auto& kv : impl_->workers_) out.push_back(kv.second.desc);
  return out;
}

std::optional<WorkerDescriptor> PrefillScheduler::worker(WorkerId w) const {
  auto lk = impl_->lock();
  auto it = impl_->workers_.find(w);
  if (it == impl_->workers_.end()) return std::nullopt;
  return it->second.desc;
}

Result<void> PrefillScheduler::drive_until_quiescent(std::size_t max_cycles) {
  for (std::size_t c = 0; c < max_cycles; ++c) {
    auto bres = run_cycle();
    if (!bres) return Result<void>::err(bres.error());
    auto& batch = bres.value();
    if (batch.groups.empty()) break;
    for (auto& dg : batch.groups) {
      std::shared_ptr<Executor> ex;
      std::shared_ptr<TokenResolver> tr;
      Epoch e;
      WorkerBootId boot;
      {
        auto lk = impl_->lock();
        auto wit = impl_->workers_.find(dg.assigned_worker);
        if (wit != impl_->workers_.end()) ex = wit->second.local_exec;
        tr = impl_->token_resolver;
        e = impl_->epoch;
        if (wit != impl_->workers_.end()) boot = wit->second.desc.boot_id;
      }
      if (!ex) {
        for (const auto& m : dg.group.members) {
          MemberResult mr;
          mr.request_id = m.request_id;
          mr.attempt_id = m.attempt_id;
          mr.generation = m.generation;
          mr.outcome = MemberOutcome::retryable_failure;
          mr.failure_code = ErrorCode::no_compatible_worker;
          mr.message = "no in-process executor for worker";
          mr.epoch = e;
          mr.worker = dg.assigned_worker;
          mr.boot = boot;
          auto rr = report_completion(mr);
          (void)rr;
        }
        continue;
      }
      auto erres = ex->execute(dg.group, *tr);
      if (!erres) {
        for (const auto& m : dg.group.members) {
          MemberResult mr;
          mr.request_id = m.request_id;
          mr.attempt_id = m.attempt_id;
          mr.generation = m.generation;
          mr.outcome = MemberOutcome::retryable_failure;
          mr.failure_code = erres.error().code;
          mr.message = erres.error().message;
          mr.epoch = e;
          mr.worker = dg.assigned_worker;
          mr.boot = boot;
          auto rr = report_completion(mr);
          (void)rr;
        }
      } else {
        auto& er = erres.value();
        for (auto& mr : er.members) {
          mr.epoch = e;
          mr.worker = dg.assigned_worker;
          mr.boot = boot;
          auto rr = report_completion(mr);
          (void)rr;
        }
      }
    }
  }
  return Result<void>::ok();
}

Stats PrefillScheduler::stats() const {
  auto lk = impl_->lock();
  return impl_->stats;
}

std::uint64_t PrefillScheduler::memory_reserved() const { return impl_->mem.reserved_bytes(); }
std::uint64_t PrefillScheduler::memory_available() const { return impl_->mem.available_bytes(); }

Snapshot PrefillScheduler::snapshot() const {
  auto lk = impl_->lock();
  Snapshot sn;
  sn.stats = impl_->stats;
  sn.queue_depth = impl_->queued_count_locked();
  sn.waiting_tokens = impl_->waiting_tokens_locked();
  sn.remaining_prefill_tokens = impl_->remaining_tokens_locked();
  sn.reserved_memory_bytes = impl_->mem.reserved_bytes();
  sn.completed_requests = impl_->stats.completed;
  std::uint64_t running = 0;
  for (const auto& kv : impl_->requests_) {
    const auto& s = kv.second;
    if (s.lifecycle == Lifecycle::queued && !s.in_flight) sn.waiting_request_ids.push_back(kv.first);
    if (s.in_flight) { sn.running_request_ids.push_back(kv.first); ++running; }
  }
  sn.running_requests = running;
  return sn;
}

Result<Explain> PrefillScheduler::explain(RequestId req) const {
  auto lk = impl_->lock();
  auto it = impl_->requests_.find(req);
  if (it == impl_->requests_.end())
    return Result<Explain>::err(ErrorCode::not_found, "unknown request");
  const auto& s = it->second;
  Explain ex;
  ex.request_id = req;
  ex.attempt_id = s.attempt;
  ex.lifecycle = to_string(s.lifecycle);
  ex.trace = s.trace;
  return Result<Explain>::ok(std::move(ex));
}

std::size_t PrefillScheduler::request_count() const { auto lk = impl_->lock(); return impl_->requests_.size(); }
std::size_t PrefillScheduler::queued_count() const { auto lk = impl_->lock(); return impl_->queued_count_locked(); }

std::size_t PrefillScheduler::running_count() const {
  auto lk = impl_->lock();
  std::size_t n = 0;
  for (const auto& kv : impl_->requests_) if (kv.second.in_flight) ++n;
  return n;
}

bool PrefillScheduler::is_terminal(RequestId req) const {
  auto lk = impl_->lock();
  auto it = impl_->requests_.find(req);
  if (it == impl_->requests_.end()) return false;
  return is_terminal_lifecycle(it->second.lifecycle);
}

std::optional<Lifecycle> PrefillScheduler::lifecycle(RequestId req) const {
  auto lk = impl_->lock();
  auto it = impl_->requests_.find(req);
  if (it == impl_->requests_.end()) return std::nullopt;
  return it->second.lifecycle;
}

Generation PrefillScheduler::current_generation(RequestId req) const {
  auto lk = impl_->lock();
  auto it = impl_->requests_.find(req);
  if (it == impl_->requests_.end()) return Generation();
  return it->second.current_generation;
}

std::vector<PrefillScheduler::InFlightInfo> PrefillScheduler::in_flight() const {
  auto lk = impl_->lock();
  std::vector<InFlightInfo> out;
  for (const auto& kv : impl_->groups_) {
    InFlightInfo info;
    info.group_id = kv.first;
    info.worker = kv.second.worker;
    info.requests.assign(kv.second.outstanding.begin(), kv.second.outstanding.end());
    info.lifecycle = Lifecycle::dispatched;
    out.push_back(std::move(info));
  }
  return out;
}

PersistedState PrefillScheduler::export_state() const {
  auto lk = impl_->lock();
  PersistedState st;
  st.epoch = impl_->epoch;
  st.next_generation_value = impl_->next_generation_value;
  st.next_attempt_value = impl_->next_attempt_value;
  st.was_clean_shutdown = (impl_->groups_.empty());
  for (const auto& kv : impl_->requests_) {
    const auto& s = kv.second;
    PersistedRequestRecord r;
    r.request_id = kv.first;
    r.attempt_id = s.attempt;
    r.tenant_id = s.request.tenant_id;
    r.model = s.request.model;
    r.adapter_id = s.request.adapter_id;
    r.relation = s.request.adapter_relation;
    r.prompt_token_count = s.request.prompt_token_count;
    r.cached_tokens = s.request.reusable_prefix.token_count;
    r.completed_uncached_tokens = s.completed_uncached;
    r.current_generation = s.current_generation;
    r.parent_generation = s.plan.chunks.empty() ? Generation(0) : s.plan.chunks[0].parent_generation;
    r.next_chunk_token_start = s.next_chunk_index < s.plan.chunks.size()
        ? s.plan.chunks[s.next_chunk_index].token_start : s.request.remaining_prefill_tokens();
    r.work_kind = s.last_work_kind;
    r.lifecycle = s.lifecycle;
    r.priority = s.request.priority;
    r.tenant_weight = s.request.tenant_weight;
    r.normalized_service = s.normalized_service;
    r.completed_token_total = s.completed_token_total;
    r.needs_recovery_full_prefill = s.in_flight;
    r.vocab = s.request.vocab;
    r.numeric_mode = s.request.numeric_mode;
    r.latency_class = s.request.latency_class;
    r.deadline_ns = s.request.deadline ? std::int64_t(*s.request.deadline) : 0;
    r.arrival_ns = s.admit_time;
    r.has_deadline = s.request.deadline.has_value();
    r.has_queue_delay = s.request.max_queue_delay.has_value();
    r.max_queue_delay_ns = s.request.max_queue_delay ? std::int64_t(*s.request.max_queue_delay) : 0;
    st.requests.push_back(std::move(r));
  }
  for (const auto& kv : impl_->workers_) {
    PersistedWorkerRecord rec;
    rec.worker_id = kv.first;
    rec.boot_id = kv.second.desc.boot_id;
    rec.host = kv.second.desc.host;
    rec.port = kv.second.desc.port;
    rec.backend = kv.second.desc.backend;
    rec.state = kv.second.desc.state;
    rec.capacity_units = kv.second.desc.capacity_units;
    st.workers.push_back(std::move(rec));
  }
  for (const auto& kv : impl_->tenants_) {
    PersistedTenantRecord rec;
    rec.tenant_id = kv.first;
    rec.weight = kv.second.weight;
    rec.scheduled_tokens = kv.second.scheduled_tokens;
    rec.completed_requests = kv.second.completed_requests;
    rec.wait_total_ns = kv.second.wait_total_ns;
    rec.outstanding = kv.second.outstanding;
    rec.normalized_service = kv.second.normalized_service;
    st.tenants.push_back(std::move(rec));
  }
  return st;
}

Result<void> PrefillScheduler::persist(const std::string& path) const {
  const PersistedState st = export_state();
  FilePersistence fp;
  return fp.save(st, path);
}

Result<void> PrefillScheduler::recover(const std::string& path) {
  FilePersistence fp;
  auto st = fp.load(path);
  if (!st) return Result<void>::err(st.error());
  return import_state(st.value());
}

Result<void> PrefillScheduler::import_state(const PersistedState& st) {
  auto lk = impl_->lock();
  impl_->epoch = st.epoch;
  if (st.next_generation_value > impl_->next_generation_value) impl_->next_generation_value = st.next_generation_value;
  if (st.next_attempt_value > impl_->next_attempt_value) impl_->next_attempt_value = st.next_attempt_value;
  impl_->requests_.clear();
  impl_->groups_.clear();
  impl_->tenants_.clear();

  for (const auto& rec : st.requests) {
    PrefillRequest r;
    r.request_id = rec.request_id;
    r.tenant_id = rec.tenant_id;
    r.model = rec.model;
    r.adapter_id = rec.adapter_id;
    r.adapter_relation = rec.relation;
    r.prompt_token_count = rec.prompt_token_count;
    r.reusable_prefix.token_count = rec.cached_tokens;
    r.vocab = rec.vocab;
    r.numeric_mode = rec.numeric_mode;
    r.latency_class = rec.latency_class;
    r.priority = rec.priority;
    r.tenant_weight = rec.tenant_weight;
    r.arrival_time = rec.arrival_ns;
    if (rec.has_deadline) r.deadline = Nanoseconds(rec.deadline_ns);
    if (rec.has_queue_delay) r.max_queue_delay = Nanoseconds(rec.max_queue_delay_ns);

    auto& s = impl_->requests_[rec.request_id];
    s.request = r;
    s.attempt = rec.attempt_id;
    s.lifecycle = rec.lifecycle;
    s.plan = impl_->planner->plan(r, rec.attempt_id, Generation(0));
    s.completed_token_total = rec.completed_token_total;
    s.current_generation = rec.current_generation;
    s.normalized_service = rec.normalized_service;
    s.admit_time = rec.arrival_ns;
    s.last_work_kind = rec.work_kind;
    s.retries_used = 0;
    s.recovered_work = false;
    s.completed_uncached = 0;
    s.trace.clear();
    s.trace.push_back("recovered from persistence");

    if (rec.next_chunk_token_start >= r.remaining_prefill_tokens()) {
      s.next_chunk_index = static_cast<std::uint32_t>(s.plan.chunks.size());
      s.completed_uncached = r.remaining_prefill_tokens();
    } else {
      s.next_chunk_index = impl_->index_for_token_start(s.plan, rec.next_chunk_token_start);
    }

    // Conservative reconciliation of work that was in-flight at process death.
    if (s.lifecycle == Lifecycle::dispatched || s.lifecycle == Lifecycle::running) {
      s.lifecycle = Lifecycle::queued;
      s.in_flight = false;
      s.current_generation = Generation(0);
      s.recovered_work = true;
      s.trace.push_back("reconciled in-flight-at-death work conservatively (redo chunk)");
    }

    auto& ts = impl_->tenants_[r.tenant_id];
    if (ts.weight <= 0.0) ts.weight = r.tenant_weight;
    if (s.lifecycle != Lifecycle::completed && s.lifecycle != Lifecycle::cancelled &&
        s.lifecycle != Lifecycle::expired && s.lifecycle != Lifecycle::failed_non_retryable &&
        s.lifecycle != Lifecycle::rejected) {
      ts.outstanding++;
      ts.admitted_tokens += r.remaining_prefill_tokens();
    }
  }

  for (const auto& rec : st.workers) {
    auto& ws = impl_->workers_[rec.worker_id];
    ws.desc.worker_id = rec.worker_id;
    ws.desc.boot_id = rec.boot_id;
    ws.desc.host = rec.host;
    ws.desc.port = rec.port;
    ws.desc.backend = rec.backend;
    ws.desc.state = rec.state;
    ws.desc.capacity_units = rec.capacity_units;
    ws.busy_units = 0;
    ws.local_exec.reset();
  }
  for (const auto& rec : st.tenants) {
    TenantService t;
    t.weight = rec.weight;
    t.scheduled_tokens = rec.scheduled_tokens;
    t.completed_requests = rec.completed_requests;
    t.wait_total_ns = rec.wait_total_ns;
    t.outstanding = rec.outstanding;
    t.normalized_service = rec.normalized_service;
    impl_->tenants_[rec.tenant_id] = t;
  }
  impl_->stats.remaining_prefill_tokens = impl_->remaining_tokens_locked();
  impl_->events.record(impl_->now(), "recover", RequestId(), AttemptId(),
                       "epoch=" + std::to_string(impl_->epoch.value()));
  return Result<void>::ok();
}

}  // namespace prefillfabric

