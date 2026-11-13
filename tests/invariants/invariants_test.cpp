// Prefill Fabric - invariant regression tests.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
//
// Each test pins one invariant of the scheduler, the persistence format, the
// wire codec or the coordinator. Run without arguments to run all of them, or
// with a test name to run that one alone (--list prints the names).
#include "tests/common/TestHarness.hpp"
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/persistence.hpp"
#include "prefillfabric/protocol.hpp"
#include "prefillfabric/network.hpp"
#include "prefillfabric/coordinator.hpp"
#include "prefillfabric/client.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include <psapi.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace prefillfabric;

namespace {

class ManualClock : public Clock {
 public:
  Nanoseconds now() const noexcept override { return t_.load(); }
  void adv(Nanoseconds n) noexcept { t_.fetch_add(n); }
 private:
  std::atomic<Nanoseconds> t_{0};
};

PrefillRequest mk(std::uint64_t id, std::uint64_t tenant, std::uint64_t prompt) {
  PrefillRequest r;
  r.request_id = RequestId(id);
  r.tenant_id = TenantId(tenant);
  r.model.name = "m";
  r.model.revision = "1";
  r.prompt_token_count = prompt;
  r.tenant_weight = 1.0;
  r.compute_estimate = prompt;
  r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt;
  return r;
}

// Token payload base, base+1, ... so a test can tell which tokens were used.
std::vector<std::uint32_t> toks(std::uint64_t n, std::uint32_t base = 100) {
  std::vector<std::uint32_t> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = base + static_cast<std::uint32_t>(i);
  return v;
}

SchedulerConfig base_cfg() {
  SchedulerConfig c;
  c.device_memory_bytes = 8ULL << 30;
  return c;
}

// The completion a worker would send for member i of a dispatched group,
// stamped with the authority the scheduler currently holds for that worker.
MemberResult done(const PrefillScheduler& sch, const DispatchedGroup& dg, std::size_t i,
                  MemberOutcome outcome = MemberOutcome::success) {
  const ExecutableMember& m = dg.group.members[i];
  MemberResult mr;
  mr.request_id = m.request_id;
  mr.attempt_id = m.attempt_id;
  mr.generation = m.generation;
  mr.outcome = outcome;
  mr.tokens_completed = m.token_count;
  mr.epoch = sch.current_epoch();
  mr.worker = dg.assigned_worker;
  const auto w = sch.worker(dg.assigned_worker);
  mr.boot = w ? w->boot_id : WorkerBootId();
  return mr;
}

void remote_worker(PrefillScheduler& sch, std::uint64_t id, std::uint64_t boot) {
  WorkerDescriptor w;
  w.worker_id = WorkerId(id);
  w.boot_id = WorkerBootId(boot);
  w.host = "remote";
  w.backend = kBackendAny;
  w.state = WorkerState::ready;
  w.capacity_units = 1;
  CHECK(sch.register_worker(w).is_ok());
}

DispatchBatch cycle(PrefillScheduler& sch) {
  auto b = sch.run_cycle();
  CHECK(b.is_ok());
  return b ? b.value() : DispatchBatch();
}

// Dispatches the best-ranked request of a cycle, completes it and returns its
// id (0 when the cycle did not dispatch exactly one request).
std::uint64_t serve_one(PrefillScheduler& sch) {
  const DispatchBatch b = cycle(sch);
  if (b.groups.size() != 1 || b.groups[0].group.members.size() != 1) return 0;
  const std::uint64_t id = b.groups[0].group.members[0].request_id.value();
  CHECK(sch.report_completion(done(sch, b.groups[0], 0)).is_ok());
  return id;
}

// An executor that succeeds for every member and records the token range it
// was handed, together with the first and last token of that range.
struct RecordingExecutor : Executor {
  struct Seen {
    std::uint64_t request = 0;
    std::uint64_t start = 0;
    std::uint64_t count = 0;
    bool resolved = false;
    std::uint32_t first = 0;
    std::uint32_t last = 0;
  };
  std::vector<Seen> seen;

  Result<ExecutorResult> execute(const ExecutionGroup& group, const TokenResolver& tokens) override {
    ExecutorResult out;
    out.group_succeeded = true;
    for (const auto& m : group.members) {
      Seen sn;
      sn.request = m.request_id.value();
      sn.start = m.token_start;
      sn.count = m.token_count;
      const auto t = tokens.resolve_tokens(m.request_id, m.attempt_id, m.generation,
                                           m.token_start, m.token_count);
      if (t && m.token_count > 0) {
        sn.resolved = true;
        sn.first = t.value()[0];
        sn.last = t.value()[m.token_count - 1];
      }
      seen.push_back(sn);
      MemberResult mr;
      mr.request_id = m.request_id;
      mr.attempt_id = m.attempt_id;
      mr.generation = m.generation;
      mr.outcome = MemberOutcome::success;
      mr.tokens_completed = m.token_count;
      out.members.push_back(mr);
    }
    return Result<ExecutorResult>::ok(out);
  }
  std::string name() const override { return "recording"; }
  bool available() const override { return true; }
  DeviceDescriptor device() const override { return DeviceDescriptor(); }
  std::string backend() const override { return kBackendCpu; }
};

void expect_refused(PrefillScheduler& sch, const PersistedState& bad) {
  const auto r = sch.import_state(bad);
  CHECK(!r.is_ok());
  CHECK(r.error().code == ErrorCode::corrupt_state);
  CHECK_EQ(sch.request_count(), 1u);
  CHECK(sch.current_epoch() == Epoch(1));
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::queued);
}

std::uint64_t commit_peak() {
  PROCESS_MEMORY_COUNTERS pmc;
  std::memset(&pmc, 0, sizeof(pmc));
  pmc.cb = sizeof(pmc);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
  return static_cast<std::uint64_t>(pmc.PeakPagefileUsage);
}

void patch_u64(std::vector<std::uint8_t>& body, std::size_t at, std::uint64_t v) {
  for (std::size_t i = 0; i < 8; ++i)
    body[at + i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu);
}

// ---- wire helpers for the coordinator tests ----

bool send_msg(Socket& s, const Message& m) {
  auto b = encode_frame(m);
  return b.is_ok() && s.send_frame(b.value().data(), b.value().size()).is_ok();
}

// True once the socket has something to read. Bounded, so a frame that never
// comes fails the test instead of blocking it.
bool readable(Socket& s, int tries = 500) {
  for (int i = 0; i < tries; ++i) {
    fd_set rd;
    FD_ZERO(&rd);
    FD_SET(s.native(), &rd);
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 10000;
    if (::select(0, &rd, nullptr, nullptr, &tv) > 0) return true;
  }
  return false;
}

bool recv_msg(Socket& s, Message& out) {
  if (!readable(s)) return false;
  auto f = s.recv_frame();
  if (!f) return false;
  auto m = decode_frame(f.value());
  if (!m) return false;
  out = m.value();
  return true;
}

// Next dispatch frame on a worker connection; error frames are skipped.
bool next_dispatch(Socket& s, Message& out) {
  for (int i = 0; i < 64; ++i) {
    Message m;
    if (!recv_msg(s, m)) return false;
    if (m.kind == MsgKind::dispatch) {
      out = m;
      return true;
    }
  }
  return false;
}

bool join_as_worker(Socket& s, std::uint16_t port, std::uint64_t id, std::uint64_t boot) {
  auto c = Socket::connect("127.0.0.1", port);
  if (!c) return false;
  s = std::move(c.value());
  Message reg;
  reg.kind = MsgKind::worker_register;
  reg.worker.worker_id = WorkerId(id);
  reg.worker.boot_id = WorkerBootId(boot);
  reg.worker.host = "127.0.0.1";
  reg.worker.backend = kBackendCpu;
  reg.worker.state = WorkerState::ready;
  reg.worker.capacity_units = 1;
  if (!send_msg(s, reg)) return false;
  Message ack;
  return recv_msg(s, ack) && ack.kind == MsgKind::worker_register_ack;
}

Message completion_of(const Message& dispatch, std::uint64_t id, std::uint64_t boot) {
  Message c;
  c.kind = MsgKind::completion;
  for (const auto& mem : dispatch.group.members) {
    MemberResult mr;
    mr.request_id = mem.request_id;
    mr.attempt_id = mem.attempt_id;
    mr.generation = mem.generation;
    mr.outcome = MemberOutcome::success;
    mr.tokens_completed = mem.token_count;
    mr.epoch = dispatch.epoch;
    mr.worker = WorkerId(id);
    mr.boot = WorkerBootId(boot);
    c.member_results.push_back(mr);
  }
  return c;
}

bool wait_lifecycle(Client& c, std::uint64_t id, Lifecycle want, int tries = 600) {
  for (int i = 0; i < tries; ++i) {
    auto ex = c.query_explain(RequestId(id));
    if (ex && ex.value().lifecycle == want) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Epoch, worker restart, retry
// ---------------------------------------------------------------------------

TEST(epoch_roll_requeues_in_flight_work) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;
  const MemberResult old_completion = done(sch, b1.groups[0], 0);

  CHECK(sch.roll_epoch().is_ok());
  // The work dispatched under the old epoch holds nothing and waits again.
  CHECK_EQ(sch.running_count(), 0u);
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::queued);
  CHECK_EQ(sch.memory_reserved(), 0u);
  const auto stale = sch.report_completion(old_completion);
  CHECK(!stale.is_ok());
  CHECK(stale.error().code == ErrorCode::stale_epoch);

  const DispatchBatch b2 = cycle(sch);
  CHECK_EQ(b2.groups.size(), 1u);
  if (b2.groups.size() == 1) CHECK(sch.report_completion(done(sch, b2.groups[0], 0)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);
  CHECK_EQ(sch.memory().unreleased_count(), 0u);
}

TEST(worker_restart_requeues_in_flight_work) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.unregister_worker(WorkerId(0)).is_ok());
  remote_worker(sch, 5, 100);
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;
  CHECK(b1.groups[0].assigned_worker == WorkerId(5));
  const MemberResult old_completion = done(sch, b1.groups[0], 0);

  remote_worker(sch, 5, 200);   // the worker process restarted
  CHECK_EQ(sch.running_count(), 0u);
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::queued);
  CHECK_EQ(sch.memory_reserved(), 0u);
  const auto stale = sch.report_completion(old_completion);
  CHECK(!stale.is_ok());
  CHECK(stale.error().code == ErrorCode::stale_worker);

  const DispatchBatch b2 = cycle(sch);
  CHECK_EQ(b2.groups.size(), 1u);
  if (b2.groups.size() == 1) CHECK(sch.report_completion(done(sch, b2.groups[0], 0)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);
}

TEST(same_boot_registration_keeps_work_and_capacity) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.unregister_worker(WorkerId(0)).is_ok());
  remote_worker(sch, 5, 100);
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;

  remote_worker(sch, 5, 100);   // the same process registers again
  CHECK_EQ(sch.running_count(), 1u);
  CHECK(sch.submit(mk(2, 1, 100)).is_ok());
  CHECK(cycle(sch).groups.empty());   // the worker's only unit is still in use
  CHECK(sch.report_completion(done(sch, b1.groups[0], 0)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);
}

TEST(refused_force_retry_leaves_work_completable) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  PrefillRequest r = mk(1, 1, 100);
  r.max_retries = 0;
  CHECK(sch.submit(r).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;

  CHECK(!sch.force_retry(RequestId(1)).is_ok());   // no retry budget
  CHECK_EQ(sch.running_count(), 1u);
  CHECK(sch.report_completion(done(sch, b1.groups[0], 0)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);
  CHECK_EQ(sch.memory().unreleased_count(), 0u);
}

TEST(force_retry_leaves_group_members_running) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.unregister_worker(WorkerId(0)).is_ok());
  remote_worker(sch, 5, 100);
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  CHECK(sch.submit(mk(2, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;
  CHECK_EQ(b1.groups[0].group.members.size(), 2u);

  CHECK(sch.force_retry(RequestId(1)).is_ok());
  // Request 2 still runs on the worker's only unit.
  CHECK(cycle(sch).groups.empty());
  CHECK(sch.worker_lost(WorkerId(5)).is_ok());
  CHECK(sch.lifecycle(RequestId(2)) == Lifecycle::queued);
  CHECK_EQ(sch.running_count(), 0u);
  CHECK_EQ(sch.memory_reserved(), 0u);
}

// ---------------------------------------------------------------------------
// Completion authority and outcomes
// ---------------------------------------------------------------------------

TEST(completion_needs_the_assigned_worker) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.unregister_worker(WorkerId(0)).is_ok());
  remote_worker(sch, 5, 100);
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;

  MemberResult anonymous = done(sch, b1.groups[0], 0);
  anonymous.worker = WorkerId();
  anonymous.boot = WorkerBootId();
  const auto refused = sch.report_completion(anonymous);
  CHECK(!refused.is_ok());
  CHECK(refused.error().code == ErrorCode::stale_worker);
  CHECK_EQ(sch.running_count(), 1u);

  CHECK(sch.report_completion(done(sch, b1.groups[0], 0)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);
}

TEST(unknown_outcome_is_refused) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;

  MemberResult odd = done(sch, b1.groups[0], 0);
  odd.outcome = static_cast<MemberOutcome>(42);
  CHECK(!sch.report_completion(odd).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::dispatched);
  CHECK_EQ(sch.running_count(), 1u);

  CHECK(sch.report_completion(done(sch, b1.groups[0], 0)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);
}

TEST(stale_rejected_outcome_requeues_the_chunk) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;

  CHECK(sch.report_completion(done(sch, b1.groups[0], 0, MemberOutcome::stale_rejected)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::queued);
  CHECK_EQ(sch.stats().completed_tokens, 0u);

  const DispatchBatch b2 = cycle(sch);
  CHECK_EQ(b2.groups.size(), 1u);
  if (b2.groups.size() == 1) CHECK(sch.report_completion(done(sch, b2.groups[0], 0)).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);
}

TEST(failed_chunk_is_not_completed_work) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;
  CHECK(sch.report_completion(done(sch, b1.groups[0], 0, MemberOutcome::retryable_failure)).is_ok());
  CHECK_EQ(sch.stats().completed_tokens, 0u);

  const DispatchBatch b2 = cycle(sch);
  CHECK_EQ(b2.groups.size(), 1u);
  if (b2.groups.size() == 1) CHECK(sch.report_completion(done(sch, b2.groups[0], 0)).is_ok());
  const Stats st = sch.stats();
  CHECK_EQ(st.completed, 1u);
  CHECK_EQ(st.completed_tokens, 100u);
  CHECK_EQ(st.chunk_count, 1u);
}

TEST(pending_cancel_wins_over_retry_and_recovery) {
  {
    // The in-flight chunk fails retryably after the cancel was requested.
    PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
    CHECK(sch.submit(mk(1, 1, 100)).is_ok());
    const DispatchBatch b1 = cycle(sch);
    CHECK_EQ(b1.groups.size(), 1u);
    if (b1.groups.size() != 1) return;
    CHECK(sch.cancel(RequestId(1)).applied);
    CHECK(sch.report_completion(done(sch, b1.groups[0], 0, MemberOutcome::retryable_failure)).is_ok());
    CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::cancelled);
    CHECK_EQ(sch.stats().cancelled, 1u);
    CHECK(cycle(sch).groups.empty());
    CHECK_EQ(sch.memory().unreleased_count(), 0u);
  }
  {
    // The worker is lost after the cancel was requested.
    PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
    CHECK(sch.unregister_worker(WorkerId(0)).is_ok());
    remote_worker(sch, 5, 100);
    CHECK(sch.submit(mk(1, 1, 100)).is_ok());
    CHECK_EQ(cycle(sch).groups.size(), 1u);
    CHECK(sch.cancel(RequestId(1)).applied);
    CHECK(sch.worker_lost(WorkerId(5)).is_ok());
    CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::cancelled);
  }
  {
    // The epoch rolls after the cancel was requested.
    PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
    CHECK(sch.submit(mk(1, 1, 100)).is_ok());
    CHECK_EQ(cycle(sch).groups.size(), 1u);
    CHECK(sch.cancel(RequestId(1)).applied);
    CHECK(sch.roll_epoch().is_ok());
    CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::cancelled);
  }
}

// ---------------------------------------------------------------------------
// Expiry
// ---------------------------------------------------------------------------

TEST(expired_in_flight_request_stops_running) {
  auto clock = std::make_shared<ManualClock>();
  PrefillScheduler sch(base_cfg(), clock);
  PrefillRequest r = mk(1, 1, 100);
  r.deadline = Nanoseconds(1000);
  CHECK(sch.submit(r).is_ok());
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;
  const MemberResult late = done(sch, b1.groups[0], 0);

  clock->adv(2000);
  (void)cycle(sch);
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::expired);
  CHECK_EQ(sch.running_count(), 0u);
  CHECK_EQ(sch.snapshot().running_requests, 0u);
  CHECK_EQ(sch.memory_reserved(), 0u);
  CHECK(!sch.report_completion(late).is_ok());

  // The worker's unit is free for other work.
  CHECK(sch.submit(mk(2, 1, 100)).is_ok());
  CHECK_EQ(cycle(sch).groups.size(), 1u);
}

TEST(expiry_returns_the_tenant_token_budget) {
  SchedulerConfig cfg = base_cfg();
  cfg.per_tenant_admitted_token_limit = 1000;
  auto clock = std::make_shared<ManualClock>();
  PrefillScheduler sch(cfg, clock);
  PrefillRequest a = mk(1, 1, 800);
  a.deadline = Nanoseconds(100);
  CHECK(sch.submit(a).is_ok());
  clock->adv(200);
  (void)cycle(sch);
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::expired);
  // The expired request no longer counts against the tenant's token limit.
  CHECK(sch.submit(mk(2, 1, 800)).is_ok());
}

TEST(queue_delay_expiry_is_not_a_deadline_miss) {
  auto clock = std::make_shared<ManualClock>();
  PrefillScheduler sch(base_cfg(), clock);
  PrefillRequest r = mk(1, 1, 100);
  r.max_queue_delay = Nanoseconds(100);
  CHECK(sch.submit(r).is_ok());
  clock->adv(200);
  (void)cycle(sch);
  const Stats st = sch.stats();
  CHECK_EQ(st.expired, 1u);
  CHECK_EQ(st.queue_delay_exceeded, 1u);
  CHECK_EQ(st.deadline_missed, 0u);
}

// ---------------------------------------------------------------------------
// Fairness, admission, memory
// ---------------------------------------------------------------------------

TEST(fairness_follows_weighted_tenant_service) {
  SchedulerConfig cfg = base_cfg();
  cfg.max_requests_per_group = 1;   // one request per cycle on the local worker
  {
    auto clock = std::make_shared<ManualClock>();
    PrefillScheduler sch(cfg, clock);
    CHECK(sch.submit(mk(1, 1, 1000)).is_ok());
    clock->adv(1);
    CHECK(sch.submit(mk(2, 1, 1000)).is_ok());
    clock->adv(1);
    CHECK(sch.submit(mk(3, 2, 1000)).is_ok());
    CHECK_EQ(serve_one(sch), 1u);
    CHECK_EQ(serve_one(sch), 3u);   // tenant 2 has had no service yet
    CHECK_EQ(serve_one(sch), 2u);
  }
  {
    auto clock = std::make_shared<ManualClock>();
    PrefillScheduler sch(cfg, clock);
    PrefillRequest a1 = mk(1, 1, 1000);
    a1.tenant_weight = 4.0;
    PrefillRequest a2 = mk(4, 1, 1000);
    a2.tenant_weight = 4.0;
    CHECK(sch.submit(a1).is_ok());
    clock->adv(1);
    CHECK(sch.submit(mk(2, 2, 1000)).is_ok());
    clock->adv(1);
    CHECK(sch.submit(mk(3, 2, 1000)).is_ok());
    clock->adv(1);
    CHECK(sch.submit(a2).is_ok());
    CHECK_EQ(serve_one(sch), 1u);
    CHECK_EQ(serve_one(sch), 2u);
    // Both tenants have had 1000 tokens; tenant 1 weighs four times as much.
    CHECK_EQ(serve_one(sch), 4u);
    CHECK_EQ(serve_one(sch), 3u);
  }
}

TEST(admission_bound_counts_outstanding_requests) {
  SchedulerConfig cfg = base_cfg();
  cfg.max_admitted_requests = 2;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  CHECK(sch.submit_with_tokens(mk(1, 1, 100), toks(100)).is_ok());
  CHECK(sch.submit_with_tokens(mk(2, 1, 100), toks(100)).is_ok());
  const auto full = sch.submit_with_tokens(mk(3, 1, 100), toks(100));
  CHECK(!full.is_ok());
  CHECK(full.error().code == ErrorCode::capacity_exceeded);

  CHECK(sch.drive_until_quiescent().is_ok());
  CHECK_EQ(sch.stats().completed, 2u);
  // Finished requests no longer hold admission capacity.
  CHECK(sch.submit_with_tokens(mk(3, 1, 100), toks(100)).is_ok());
}

TEST(queue_bound_applies_backpressure) {
  SchedulerConfig cfg = base_cfg();
  cfg.max_queued_requests = 2;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  CHECK(sch.submit(mk(2, 1, 100)).is_ok());
  const auto full = sch.submit(mk(3, 1, 100));
  CHECK(!full.is_ok());
  CHECK(full.error().code == ErrorCode::backpressure);
}

TEST(memory_budget_is_device_memory_minus_headroom) {
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 1ULL << 30;
  cfg.memory_headroom_bytes = 512ULL << 20;
  cfg.memory_headroom_fraction = 0.0;
  CHECK_EQ(cfg.usable_memory_bytes(), 512ULL << 20);
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  CHECK_EQ(sch.memory().budget(), cfg.usable_memory_bytes());

  PrefillRequest r = mk(1, 1, 100);
  r.memory_estimate_bytes = 300ULL << 20;
  CHECK(sch.submit(r).is_ok());
  CHECK_EQ(cycle(sch).groups.size(), 1u);
  CHECK_EQ(sch.memory_reserved(), 300ULL << 20);
}

TEST(headroom_fraction_is_clamped) {
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 8ULL << 30;
  cfg.memory_headroom_bytes = 256ULL << 20;
  cfg.memory_headroom_fraction = std::numeric_limits<double>::quiet_NaN();
  CHECK_EQ(cfg.usable_memory_bytes(), (8ULL << 30) - (256ULL << 20));
  cfg.memory_headroom_fraction = -1.0;
  CHECK_EQ(cfg.usable_memory_bytes(), (8ULL << 30) - (256ULL << 20));
  cfg.memory_headroom_fraction = 2.0;
  CHECK_EQ(cfg.usable_memory_bytes(), 0u);
}

TEST(group_larger_than_budget_dispatches_what_fits) {
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 1ULL << 30;
  cfg.memory_headroom_bytes = 0;
  cfg.memory_headroom_fraction = 0.0;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  for (std::uint64_t id = 1; id <= 3; ++id) {
    PrefillRequest r = mk(id, 1, 100);
    r.memory_estimate_bytes = 400ULL << 20;   // three together exceed 1 GiB
    CHECK(sch.submit(r).is_ok());
  }
  const DispatchBatch b1 = cycle(sch);
  CHECK_EQ(b1.groups.size(), 1u);
  if (b1.groups.size() != 1) return;
  CHECK_EQ(b1.groups[0].group.members.size(), 2u);
  CHECK_EQ(sch.memory_reserved(), 800ULL << 20);
  for (std::size_t i = 0; i < b1.groups[0].group.members.size(); ++i)
    CHECK(sch.report_completion(done(sch, b1.groups[0], i)).is_ok());

  const DispatchBatch b2 = cycle(sch);
  CHECK_EQ(b2.groups.size(), 1u);
  if (b2.groups.size() == 1) CHECK(sch.report_completion(done(sch, b2.groups[0], 0)).is_ok());
  CHECK_EQ(sch.stats().completed, 3u);
  CHECK_EQ(sch.memory().unreleased_count(), 0u);
}

TEST(chunk_larger_than_the_budget_is_refused) {
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 1ULL << 30;
  cfg.memory_headroom_bytes = 0;
  cfg.memory_headroom_fraction = 0.0;
  cfg.max_memory_per_chunk = 2ULL << 30;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  PrefillRequest r = mk(1, 1, 100);
  r.memory_estimate_bytes = 1536ULL << 20;   // one chunk, larger than the budget
  const auto res = sch.submit(r);
  CHECK(!res.is_ok());
  CHECK(res.error().code == ErrorCode::memory_unavailable);
  CHECK_EQ(sch.request_count(), 0u);
}

// ---------------------------------------------------------------------------
// Request validation and token ranges
// ---------------------------------------------------------------------------

TEST(reused_prefix_runs_the_uncached_suffix) {
  SchedulerConfig cfg = base_cfg();
  cfg.max_tokens_per_chunk = 2;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  auto exec = std::make_shared<RecordingExecutor>();
  sch.attach_executor(exec, kBackendCpu);
  PrefillRequest r = mk(1, 1, 10);
  r.reusable_prefix.token_count = 6;
  r.reusable_prefix.validated = true;
  CHECK(sch.submit_with_tokens(r, toks(10)).is_ok());   // tokens 100..109
  CHECK(sch.drive_until_quiescent().is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::completed);

  // The four uncached tokens are positions 6..9 of the prompt.
  CHECK_EQ(exec->seen.size(), 2u);
  if (exec->seen.size() != 2) return;
  CHECK(exec->seen[0].resolved);
  CHECK_EQ(exec->seen[0].start, 6u);
  CHECK_EQ(exec->seen[0].count, 2u);
  CHECK_EQ(exec->seen[0].first, 106u);
  CHECK_EQ(exec->seen[0].last, 107u);
  CHECK(exec->seen[1].resolved);
  CHECK_EQ(exec->seen[1].start, 8u);
  CHECK_EQ(exec->seen[1].first, 108u);
  CHECK_EQ(exec->seen[1].last, 109u);
}

TEST(unvalidated_prefix_is_refused) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  PrefillRequest r = mk(1, 1, 1000);
  r.reusable_prefix.token_count = 900;
  r.reusable_prefix.validated = false;
  const auto res = sch.submit(r);
  CHECK(!res.is_ok());
  CHECK(res.error().code == ErrorCode::invalid_reusable_prefix);
}

TEST(invalid_weight_and_enums_are_refused) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  PrefillRequest nan_weight = mk(1, 1, 100);
  nan_weight.tenant_weight = std::numeric_limits<double>::quiet_NaN();
  CHECK(!sch.submit(nan_weight).is_ok());
  PrefillRequest inf_weight = mk(2, 1, 100);
  inf_weight.tenant_weight = std::numeric_limits<double>::infinity();
  CHECK(!sch.submit(inf_weight).is_ok());
  PrefillRequest bad_class = mk(3, 1, 100);
  bad_class.latency_class = static_cast<LatencyClass>(99);
  CHECK(!sch.submit(bad_class).is_ok());
  PrefillRequest bad_mode = mk(4, 1, 100);
  bad_mode.numeric_mode = static_cast<NumericMode>(77);
  CHECK(!sch.submit(bad_mode).is_ok());
  PrefillRequest bad_layout = mk(5, 1, 100);
  bad_layout.input_layout = static_cast<InputLayout>(-3);
  CHECK(!sch.submit(bad_layout).is_ok());
  PrefillRequest bad_relation = mk(6, 1, 100);
  bad_relation.adapter_relation = static_cast<AdapterRelation>(12);
  CHECK(!sch.submit(bad_relation).is_ok());
  CHECK_EQ(sch.request_count(), 0u);
}

TEST(oversized_prompt_is_a_rejection) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  PrefillRequest r = mk(1, 1, 1ULL << 62);
  r.memory_estimate_bytes = 0;
  r.compute_estimate = 0;
  const auto res = sch.submit(r);   // must come back as a result, not as an exception
  CHECK(!res.is_ok());
  CHECK_EQ(sch.request_count(), 0u);
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

TEST(stats_are_current) {
  SchedulerConfig cfg = base_cfg();
  cfg.max_tokens_per_chunk = 1000;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 7, 2500)).is_ok());
  CHECK_EQ(sch.stats().queue_depth, 1u);   // before any cycle has run
  CHECK_EQ(sch.stats().remaining_prefill_tokens, 2500u);

  DispatchBatch b = cycle(sch);
  CHECK_EQ(b.groups.size(), 1u);
  {
    const Stats st = sch.stats();
    CHECK_EQ(st.running_requests, 1u);
    CHECK_EQ(st.queue_depth, 0u);
    CHECK(st.reserved_memory_bytes > 0);
    CHECK_EQ(st.reserved_memory_bytes, sch.memory_reserved());
    CHECK_EQ(st.available_memory_bytes, sch.memory_available());
    CHECK_EQ(st.per_tenant.size(), 1u);
    const auto t = st.per_tenant.find(TenantId(7));
    CHECK(t != st.per_tenant.end());
    if (t != st.per_tenant.end()) {
      CHECK_EQ(t->second.outstanding, 1u);
      CHECK_EQ(t->second.scheduled_tokens, 1000u);
    }
  }
  for (int guard = 0; guard < 8 && b.groups.size() == 1; ++guard) {
    CHECK(sch.report_completion(done(sch, b.groups[0], 0)).is_ok());
    b = cycle(sch);
  }
  {
    const Stats st = sch.stats();
    CHECK_EQ(st.running_requests, 0u);
    CHECK_EQ(st.completed, 1u);
    CHECK_EQ(st.completed_tokens, 2500u);
    CHECK_EQ(st.chunk_count, 3u);
    CHECK_EQ(st.remaining_prefill_tokens, 0u);
    const auto t = st.per_tenant.find(TenantId(7));
    CHECK(t != st.per_tenant.end());
    if (t != st.per_tenant.end()) {
      CHECK_EQ(t->second.outstanding, 0u);
      CHECK_EQ(t->second.completed_requests, 1u);
      CHECK_EQ(t->second.scheduled_tokens, 2500u);
    }
    CHECK_EQ(st.per_device.size(), 1u);
    if (st.per_device.size() == 1) {
      CHECK_EQ(st.per_device.begin()->second.groups_run, 3u);
      CHECK_EQ(st.per_device.begin()->second.tokens_processed, 2500u);
    }
  }
}

TEST(explain_trace_is_bounded) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  PrefillRequest r = mk(1, 1, 100);
  r.backend = "nosuchbackend";   // no worker runs it, so it is deferred every cycle
  CHECK(sch.submit(r).is_ok());
  for (int i = 0; i < 5000; ++i) (void)sch.run_cycle();
  const auto ex = sch.explain(RequestId(1));
  CHECK(ex.is_ok());
  CHECK(ex.value().trace_size() <= 1000u);
}

TEST(released_reservations_are_forgotten) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  for (std::uint64_t id = 1; id <= 50; ++id)
    CHECK(sch.submit_with_tokens(mk(id, 1, 64), toks(64)).is_ok());
  CHECK(sch.drive_until_quiescent().is_ok());
  CHECK_EQ(sch.stats().completed, 50u);
  CHECK_EQ(sch.memory().records().size(), 0u);
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

TEST(import_refuses_bad_state_and_keeps_live_state) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const PersistedState good = sch.export_state();
  CHECK_EQ(good.requests.size(), 1u);
  CHECK_EQ(good.tenants.size(), 1u);
  if (good.requests.size() != 1 || good.tenants.size() != 1) return;

  PersistedState bad = good;
  bad.epoch = Epoch(0);
  expect_refused(sch, bad);

  bad = good;
  bad.requests.push_back(bad.requests[0]);   // the same request id twice
  expect_refused(sch, bad);

  bad = good;
  bad.requests[0].request_id = RequestId(0);
  expect_refused(sch, bad);

  bad = good;
  bad.requests[0].tenant_weight = std::numeric_limits<double>::quiet_NaN();
  expect_refused(sch, bad);

  bad = good;
  bad.tenants[0].normalized_service = std::numeric_limits<double>::quiet_NaN();
  expect_refused(sch, bad);

  bad = good;   // unfinished, yet nothing left to do
  bad.requests[0].next_chunk_token_start = bad.requests[0].prompt_token_count;
  expect_refused(sch, bad);

  CHECK(sch.import_state(good).is_ok());
  CHECK(sch.lifecycle(RequestId(1)) == Lifecycle::queued);
}

TEST(import_gives_back_live_reservations_and_capacity) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  CHECK_EQ(cycle(sch).groups.size(), 1u);
  CHECK(sch.memory_reserved() > 0);

  PersistedState empty;
  empty.epoch = Epoch(7);
  CHECK(sch.import_state(empty).is_ok());
  CHECK_EQ(sch.memory_reserved(), 0u);
  CHECK_EQ(sch.memory().unreleased_count(), 0u);
  // The local worker's unit is free for the work that follows.
  CHECK(sch.submit(mk(2, 1, 100)).is_ok());
  CHECK_EQ(cycle(sch).groups.size(), 1u);
}

TEST(recovery_keeps_the_attached_executor) {
  PersistedState st;
  {
    PrefillScheduler src(base_cfg(), std::make_shared<ManualClock>());
    CHECK(src.submit(mk(1, 1, 1000)).is_ok());
    st = src.export_state();
  }
  PrefillScheduler dst(base_cfg(), std::make_shared<ManualClock>());
  auto exec = std::make_shared<RecordingExecutor>();
  dst.attach_executor(exec, kBackendCpu);
  CHECK(dst.import_state(st).is_ok());
  CHECK(dst.drive_until_quiescent().is_ok());
  CHECK(dst.lifecycle(RequestId(1)) == Lifecycle::completed);
  CHECK_EQ(exec->seen.size(), 1u);
}

TEST(recovery_recomputes_tenant_accounting) {
  {
    SchedulerConfig cfg = base_cfg();
    cfg.per_tenant_admitted_token_limit = 1000;
    PersistedState st;
    {
      PrefillScheduler src(cfg, std::make_shared<ManualClock>());
      CHECK(src.submit(mk(1, 1, 800)).is_ok());
      st = src.export_state();
    }
    PrefillScheduler dst(cfg, std::make_shared<ManualClock>());
    CHECK(dst.import_state(st).is_ok());
    // 800 of the tenant's 1000 tokens are still admitted after recovery.
    CHECK(!dst.submit(mk(2, 1, 800)).is_ok());
  }
  {
    SchedulerConfig cfg = base_cfg();
    cfg.per_tenant_outstanding_limit = 1;
    PersistedState st;
    {
      PrefillScheduler src(cfg, std::make_shared<ManualClock>());
      CHECK(src.submit(mk(1, 1, 100)).is_ok());
      st = src.export_state();
    }
    CHECK_EQ(st.tenants.size(), 1u);
    if (st.tenants.size() != 1) return;
    st.tenants[0].outstanding = 0;   // a counter that disagrees with the requests
    PrefillScheduler dst(cfg, std::make_shared<ManualClock>());
    CHECK(dst.import_state(st).is_ok());
    CHECK(!dst.submit(mk(2, 1, 100)).is_ok());
  }
}

TEST(recovery_requeues_every_unfinished_request) {
  PersistedState st;
  {
    PrefillScheduler src(base_cfg(), std::make_shared<ManualClock>());
    CHECK(src.submit(mk(1, 1, 1000)).is_ok());
    st = src.export_state();
  }
  CHECK_EQ(st.requests.size(), 1u);
  if (st.requests.size() != 1) return;
  st.requests[0].lifecycle = Lifecycle::awaiting_retry;
  PrefillScheduler dst(base_cfg(), std::make_shared<ManualClock>());
  auto exec = std::make_shared<RecordingExecutor>();
  dst.attach_executor(exec, kBackendCpu);
  CHECK(dst.import_state(st).is_ok());
  CHECK(dst.lifecycle(RequestId(1)) == Lifecycle::queued);
  CHECK(dst.drive_until_quiescent().is_ok());
  CHECK(dst.lifecycle(RequestId(1)) == Lifecycle::completed);
}

TEST(recovery_keeps_completed_chunks) {
  SchedulerConfig cfg = base_cfg();
  cfg.max_tokens_per_chunk = 1000;
  PersistedState st;
  {
    PrefillScheduler src(cfg, std::make_shared<ManualClock>());
    CHECK(src.submit(mk(1, 1, 2500)).is_ok());
    const DispatchBatch b1 = cycle(src);
    CHECK_EQ(b1.groups.size(), 1u);
    if (b1.groups.size() != 1) return;
    CHECK(src.report_completion(done(src, b1.groups[0], 0)).is_ok());
    st = src.export_state();
  }
  PrefillScheduler dst(cfg, std::make_shared<ManualClock>());
  CHECK(dst.import_state(st).is_ok());
  CHECK_EQ(dst.snapshot().remaining_prefill_tokens, 1500u);
}

TEST(recovery_never_reuses_identifiers) {
  PersistedState st;
  {
    PrefillScheduler src(base_cfg(), std::make_shared<ManualClock>());
    CHECK(src.submit(mk(1, 1, 100)).is_ok());
    CHECK_EQ(cycle(src).groups.size(), 1u);
    st = src.export_state();
  }
  CHECK_EQ(st.requests.size(), 1u);
  if (st.requests.size() != 1) return;
  // Counters that lag behind the identifiers the records carry.
  st.next_attempt_value = 1;
  st.next_generation_value = 1;
  PrefillScheduler dst(base_cfg(), std::make_shared<ManualClock>());
  CHECK(dst.import_state(st).is_ok());
  CHECK(dst.next_attempt_value() > st.requests[0].attempt_id.value());
  CHECK(dst.next_generation_value() > st.requests[0].current_generation.value());
}

TEST(recovery_keeps_placement_estimates_and_retry_budget) {
  PersistedState st;
  {
    PrefillScheduler src(base_cfg(), std::make_shared<ManualClock>());
    PrefillRequest r = mk(1, 1, 100);
    r.backend = kBackendCuda;
    r.memory_estimate_bytes = 300ULL << 20;
    r.max_retries = 0;
    CHECK(src.submit(r).is_ok());
    // Through the on-disk format, as recover() would read it.
    auto blob = serialize_state(src.export_state());
    CHECK(blob.is_ok());
    auto back = deserialize_state(blob.value());
    CHECK(back.is_ok());
    if (!back.is_ok()) return;
    st = back.value();
  }
  PrefillScheduler dst(base_cfg(), std::make_shared<ManualClock>());
  CHECK(dst.import_state(st).is_ok());
  // The local worker runs the cpu backend; the request still asks for cuda.
  CHECK(cycle(dst).groups.empty());
  WorkerDescriptor w;
  w.worker_id = WorkerId(9);
  w.boot_id = WorkerBootId(1);
  w.backend = kBackendCuda;
  w.state = WorkerState::ready;
  w.capacity_units = 1;
  CHECK(dst.register_worker(w).is_ok());
  const DispatchBatch b = cycle(dst);
  CHECK_EQ(b.groups.size(), 1u);
  if (b.groups.size() != 1) return;
  CHECK(b.groups[0].assigned_worker == WorkerId(9));
  CHECK_EQ(dst.memory_reserved(), 300ULL << 20);
  // The request was admitted without a retry budget.
  CHECK(dst.report_completion(done(dst, b.groups[0], 0, MemberOutcome::retryable_failure)).is_ok());
  CHECK(dst.lifecycle(RequestId(1)) == Lifecycle::failed_non_retryable);

  // A snapshot written in the version-1 layout, before these fields existed,
  // still loads.
  ByteWriter p;
  p.u64(3);    // epoch
  p.u64(10);   // next generation
  p.u64(10);   // next attempt
  p.u8(1);     // clean shutdown
  p.u64(1);    // one request record
  p.u64(1); p.u64(1); p.u64(1);       // request, attempt, tenant
  p.str("m"); p.str("1");             // model name, revision
  p.u64(0); p.u8(0);                  // adapter, relation
  p.u64(100); p.u64(0); p.u64(0);     // prompt, cached, completed uncached
  p.u64(0); p.u64(0); p.u64(0);       // current and parent generation, next chunk start
  p.u8(0); p.u8(static_cast<std::uint8_t>(Lifecycle::queued));
  p.u32(0); p.f64(1.0); p.f64(0.0);   // priority, tenant weight, service
  p.u64(0); p.u64(0); p.u8(0);        // last chunk, completed total, needs recovery
  p.str(""); p.u64(0);                // vocabulary
  p.u8(0); p.u8(static_cast<std::uint8_t>(LatencyClass::standard));
  p.i64(0); p.i64(0); p.u8(0); p.u8(0); p.i64(0);   // deadline, arrival, flags, queue delay
  p.u64(0);    // workers
  p.u64(0);    // tenants
  ByteWriter v1;
  v1.u32(kPersistenceMagic);
  v1.u32(1);
  v1.u32(0);
  v1.u64(p.size());
  v1.u64(fnv1a64(1469598103934665603ULL, p.bytes().data(), p.size()));
  v1.raw(p.bytes().data(), p.size());
  const auto old_state = deserialize_state(v1.bytes());
  CHECK(old_state.is_ok());
  if (!old_state.is_ok()) return;
  CHECK_EQ(old_state.value().requests.size(), 1u);
  PrefillScheduler old_sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(old_sch.import_state(old_state.value()).is_ok());
  CHECK(old_sch.lifecycle(RequestId(1)) == Lifecycle::queued);
}

TEST(recovery_keeps_the_chunk_plan) {
  PersistedState st;
  {
    PrefillScheduler src(base_cfg(), std::make_shared<ManualClock>());
    PrefillRequest r = mk(1, 1, 8000);
    r.memory_estimate_bytes = 2ULL << 30;   // bounds a chunk to 2000 tokens
    CHECK(src.submit(r).is_ok());
    const DispatchBatch b1 = cycle(src);
    CHECK_EQ(b1.groups.size(), 1u);
    if (b1.groups.size() != 1) return;
    CHECK_EQ(b1.groups[0].group.members[0].token_count, 2000u);
    CHECK(src.report_completion(done(src, b1.groups[0], 0)).is_ok());
    auto blob = serialize_state(src.export_state());
    CHECK(blob.is_ok());
    auto back = deserialize_state(blob.value());
    CHECK(back.is_ok());
    if (!back.is_ok()) return;
    st = back.value();
  }
  PrefillScheduler dst(base_cfg(), std::make_shared<ManualClock>());
  CHECK(dst.import_state(st).is_ok());
  // The first chunk stays done and the next chunk is the one planned before.
  CHECK_EQ(dst.snapshot().remaining_prefill_tokens, 6000u);
  const DispatchBatch b2 = cycle(dst);
  CHECK_EQ(b2.groups.size(), 1u);
  if (b2.groups.size() != 1) return;
  CHECK_EQ(b2.groups[0].group.members[0].token_start, 2000u);
  CHECK_EQ(b2.groups[0].group.members[0].token_count, 2000u);
}

// ---------------------------------------------------------------------------
// Persistence format and wire codec
// ---------------------------------------------------------------------------

TEST(persisted_enums_are_range_checked) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  const PersistedState good = sch.export_state();
  CHECK(!good.requests.empty());
  CHECK(!good.workers.empty());
  if (good.requests.empty() || good.workers.empty()) return;
  {
    auto blob = serialize_state(good);
    CHECK(blob.is_ok());
    CHECK(deserialize_state(blob.value()).is_ok());
  }
  {
    PersistedState bad = good;
    bad.requests[0].work_kind = static_cast<WorkKind>(200);
    auto blob = serialize_state(bad);
    CHECK(blob.is_ok());
    const auto dec = deserialize_state(blob.value());
    CHECK(!dec.is_ok());
    CHECK(dec.error().code == ErrorCode::corrupt_state);
  }
  {
    PersistedState bad = good;
    bad.requests[0].latency_class = static_cast<LatencyClass>(77);
    auto blob = serialize_state(bad);
    CHECK(blob.is_ok());
    CHECK(!deserialize_state(blob.value()).is_ok());
  }
  {
    PersistedState bad = good;
    bad.workers[0].state = static_cast<WorkerState>(99);
    auto blob = serialize_state(bad);
    CHECK(blob.is_ok());
    const auto dec = deserialize_state(blob.value());
    CHECK(!dec.is_ok());
    CHECK(dec.error().code == ErrorCode::corrupt_state);
  }
}

TEST(bytes_after_the_payload_are_refused) {
  PrefillScheduler sch(base_cfg(), std::make_shared<ManualClock>());
  CHECK(sch.submit(mk(1, 1, 100)).is_ok());
  auto blob = serialize_state(sch.export_state());
  CHECK(blob.is_ok());
  std::vector<std::uint8_t> bytes = blob.value();
  CHECK(deserialize_state(bytes).is_ok());
  bytes.push_back(0);
  bytes.push_back(1);
  bytes.push_back(2);
  bytes.push_back(3);
  const auto dec = deserialize_state(bytes);
  CHECK(!dec.is_ok());
  CHECK(dec.error().code == ErrorCode::corrupt_state);
}

TEST(frame_counts_are_checked_before_allocation) {
  const std::uint64_t slack = 64ULL << 20;
  {
    Message m;
    m.kind = MsgKind::submit;
    m.request = mk(1, 1, 10);
    auto body = encode_frame(m);
    CHECK(body.is_ok());
    std::vector<std::uint8_t> bytes = body.value();
    CHECK(bytes.size() > 8);
    if (bytes.size() <= 8) return;
    patch_u64(bytes, bytes.size() - 8, 1ULL << 26);   // claims 2^26 tokens, carries none
    const std::uint64_t before = commit_peak();
    const auto dec = decode_frame(bytes);
    const std::uint64_t after = commit_peak();
    CHECK(!dec.is_ok());
    CHECK(after - before < slack);
  }
  {
    Message m;
    m.kind = MsgKind::dispatch;
    auto body = encode_frame(m);
    CHECK(body.is_ok());
    std::vector<std::uint8_t> bytes = body.value();
    CHECK(bytes.size() > 32);
    if (bytes.size() <= 32) return;
    patch_u64(bytes, bytes.size() - 32, 1ULL << 26);   // the member token count
    const std::uint64_t before = commit_peak();
    const auto dec = decode_frame(bytes);
    const std::uint64_t after = commit_peak();
    CHECK(!dec.is_ok());
    CHECK(after - before < slack);
  }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

TEST(coordinator_refuses_mismatched_token_payload) {
  Coordinator coord(base_cfg(), std::make_shared<ManualClock>(), 0);
  CHECK(coord.start().is_ok());
  Socket worker;
  CHECK(join_as_worker(worker, coord.port(), 1, 100));
  Client c("127.0.0.1", coord.port());
  CHECK(c.connect().is_ok());

  // A payload shorter than the prompt can never be dispatched.
  const auto bad = c.submit(mk(1, 1, 1000), toks(10));
  CHECK(!bad.is_ok());

  // The worker stays in service for a well-formed request.
  CHECK(c.submit(mk(2, 1, 500), toks(500)).is_ok());
  Message d;
  const bool got = next_dispatch(worker, d);
  CHECK(got);
  if (got) {
    CHECK_EQ(d.group.members.size(), 1u);
    CHECK(send_msg(worker, completion_of(d, 1, 100)));
    CHECK(wait_lifecycle(c, 2, Lifecycle::completed));
  }
  const auto st = c.query_state();
  CHECK(st.is_ok());
  if (st.is_ok()) {
    CHECK_EQ(st.value().workers.size(), 1u);
    if (st.value().workers.size() == 1) CHECK(st.value().workers[0].state == WorkerState::ready);
  }
  coord.shutdown();
}

TEST(coordinator_refused_submit_keeps_the_live_payload) {
  SchedulerConfig cfg = base_cfg();
  cfg.max_tokens_per_chunk = 4;
  Coordinator coord(cfg, std::make_shared<ManualClock>(), 0);
  CHECK(coord.start().is_ok());
  Socket worker;
  CHECK(join_as_worker(worker, coord.port(), 1, 100));
  Client c("127.0.0.1", coord.port());
  CHECK(c.connect().is_ok());

  CHECK(c.submit(mk(1, 1, 8), toks(8, 100)).is_ok());   // 100..107 in two chunks
  Message d1;
  const bool got1 = next_dispatch(worker, d1);
  CHECK(got1);
  if (got1) {
    CHECK(d1.member_tokens == toks(4, 100));
    // A second submit under the same id is refused...
    CHECK(!c.submit(mk(1, 1, 8), toks(8, 900)).is_ok());
    // ...and the next chunk still carries the tokens of the admitted request.
    CHECK(send_msg(worker, completion_of(d1, 1, 100)));
    Message d2;
    const bool got2 = next_dispatch(worker, d2);
    CHECK(got2);
    if (got2) {
      CHECK(d2.member_tokens == toks(4, 104));
      CHECK(send_msg(worker, completion_of(d2, 1, 100)));
      CHECK(wait_lifecycle(c, 1, Lifecycle::completed));
    }
  }
  coord.shutdown();
}

TEST(coordinator_applies_every_member_result) {
  auto clock = std::make_shared<ManualClock>();
  Coordinator coord(base_cfg(), clock, 0);
  CHECK(coord.start().is_ok());
  Client c("127.0.0.1", coord.port());
  CHECK(c.connect().is_ok());
  PrefillRequest a = mk(1, 1, 50);
  a.deadline = Nanoseconds(1000);
  CHECK(c.submit(a, toks(50)).is_ok());
  CHECK(c.submit(mk(2, 1, 50), toks(50)).is_ok());

  // The worker joins once both requests are queued, so they form one group.
  Socket worker;
  CHECK(join_as_worker(worker, coord.port(), 1, 100));
  Message d;
  const bool got = next_dispatch(worker, d);
  CHECK(got);
  if (got) {
    CHECK_EQ(d.group.members.size(), 2u);
    if (d.group.members.size() == 2) CHECK(d.group.members[0].request_id == RequestId(1));
    // The first member's deadline passes while the group is in flight.
    clock->adv(5000);
    CHECK(wait_lifecycle(c, 1, Lifecycle::expired));
    // Its result is refused; the result of the second member still applies.
    CHECK(send_msg(worker, completion_of(d, 1, 100)));
    CHECK(wait_lifecycle(c, 2, Lifecycle::completed));
  }
  coord.shutdown();
}

TEST(coordinator_stale_connection_leaves_the_restarted_worker) {
  Coordinator coord(base_cfg(), std::make_shared<ManualClock>(), 0);
  CHECK(coord.start().is_ok());
  Client c("127.0.0.1", coord.port());
  CHECK(c.connect().is_ok());
  Socket old_conn;
  CHECK(join_as_worker(old_conn, coord.port(), 1, 100));
  Socket new_conn;
  CHECK(join_as_worker(new_conn, coord.port(), 1, 200));   // the worker restarted
  old_conn.close();   // the connection of the previous process ends afterwards

  bool stayed_ready = true;
  for (int i = 0; i < 60 && stayed_ready; ++i) {
    const auto st = c.query_state();
    if (!st.is_ok() || st.value().workers.size() != 1 ||
        st.value().workers[0].state != WorkerState::ready ||
        st.value().workers[0].boot_id != WorkerBootId(200)) {
      stayed_ready = false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  CHECK(stayed_ready);

  CHECK(c.submit(mk(1, 1, 50), toks(50)).is_ok());
  Message d;
  CHECK(next_dispatch(new_conn, d));
  coord.shutdown();
}

TEST(coordinator_shutdown_ends_open_connections) {
  Coordinator coord(base_cfg(), std::make_shared<ManualClock>(), 0);
  CHECK(coord.start().is_ok());
  Client c("127.0.0.1", coord.port());
  CHECK(c.connect().is_ok());
  CHECK(c.query_state().is_ok());
  coord.shutdown();
  // No handler is left to serve the connection once shutdown has returned.
  CHECK(!c.query_state().is_ok());
}

int main(int argc, char** argv) {
  if (!net_init().is_ok()) {
    std::fprintf(stderr, "net_init failed\n");
    return 2;
  }
  int rc = 0;
  if (argc < 2) {
    rc = ::pf_test::run_all();
  } else {
    const std::string want = argv[1];
    bool found = false;
    for (const auto& e : ::pf_test::TestRegistry::entries()) {
      if (want == "--list") {
        std::printf("%s\n", e.name);
        found = true;
        continue;
      }
      if (want != e.name) continue;
      found = true;
      try {
        e.fn();
      } catch (const std::exception& ex) {
        std::cerr << "[EXCEPTION] " << e.name << ": " << ex.what() << "\n";
        ++::pf_test::fail_counter();
      } catch (...) {
        std::cerr << "[EXCEPTION] " << e.name << ": unknown\n";
        ++::pf_test::fail_counter();
      }
    }
    if (!found) {
      std::fprintf(stderr, "unknown test: %s\n", want.c_str());
      rc = 2;
    } else if (want != "--list") {
      std::printf("%s: %d check(s), %d failed\n", want.c_str(), ::pf_test::check_counter(),
                  ::pf_test::fail_counter());
      rc = ::pf_test::fail_counter() == 0 ? 0 : 1;
    }
  }
  net_cleanup();
  return rc;
}
