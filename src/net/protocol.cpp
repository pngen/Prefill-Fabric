// Prefill Fabric - protocol codec implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/protocol.hpp"
#include "prefillfabric/persistence.hpp"

namespace prefillfabric {

namespace {

void enc_worker(ByteWriter& w, const WorkerDescriptor& wd) {
  w.strong(wd.worker_id);
  w.strong(wd.boot_id);
  w.str(wd.host);
  w.u32(wd.port);
  w.str(wd.backend);
  w.u8(static_cast<std::uint8_t>(wd.state));
  w.u64(wd.capacity_units);
  w.strong(wd.devices.empty() ? DeviceId() : wd.devices[0].id);
  w.str(wd.devices.empty() ? "" : wd.devices[0].name);
}

Result<WorkerDescriptor> dec_worker(ByteReader& r) {
  WorkerDescriptor wd;
  auto v = r.strong<WorkerIdTag>(); if (!v) return Result<WorkerDescriptor>::err(v.error()); wd.worker_id = v.value();
  auto bv = r.strong<WorkerBootIdTag>(); if (!bv) return Result<WorkerDescriptor>::err(bv.error()); wd.boot_id = bv.value();
  auto s = r.str(); if (!s) return Result<WorkerDescriptor>::err(s.error()); wd.host = s.value();
  auto u32 = r.u32(); if (!u32) return Result<WorkerDescriptor>::err(u32.error()); wd.port = u32.value();
  s = r.str(); if (!s) return Result<WorkerDescriptor>::err(s.error()); wd.backend = s.value();
  auto u8 = r.u8(); if (!u8) return Result<WorkerDescriptor>::err(u8.error()); wd.state = static_cast<WorkerState>(u8.value());
  auto u64 = r.u64(); if (!u64) return Result<WorkerDescriptor>::err(u64.error()); wd.capacity_units = u64.value();
  auto did = r.strong<DeviceIdTag>(); if (!did) return Result<WorkerDescriptor>::err(did.error());
  s = r.str(); if (!s) return Result<WorkerDescriptor>::err(s.error());
  if (!did.value().is_nil()) {
    DeviceDescriptor d; d.id = did.value(); d.name = s.value(); d.backend = wd.backend; d.available = true;
    wd.devices.push_back(d);
  }
  return Result<WorkerDescriptor>::ok(wd);
}

void enc_req(ByteWriter& w, const PrefillRequest& r) {
  w.strong(r.request_id);
  w.strong(r.tenant_id);
  w.str(r.model.name);
  w.str(r.model.revision);
  w.strong(r.adapter_id);
  w.u8(static_cast<std::uint8_t>(r.adapter_relation));
  w.str(r.vocab.tokenizer);
  w.u64(r.vocab.vocab_size);
  w.u64(r.prompt_token_count);
  w.u64(r.reusable_prefix.token_count);
  w.u8(r.reusable_prefix.validated ? 1u : 0u);
  w.str(r.reusable_prefix.fingerprint);
  w.u64(r.input_shape.seq_len);
  w.u64(r.input_shape.hidden_dim);
  w.u64(r.input_shape.num_heads);
  w.u64(r.input_shape.head_dim);
  w.u8(static_cast<std::uint8_t>(r.input_layout));
  w.u8(static_cast<std::uint8_t>(r.numeric_mode));
  w.str(r.backend);
  w.strong(r.device);
  w.str(r.device_match);
  w.u8(static_cast<std::uint8_t>(r.latency_class));
  w.u8(r.deadline.has_value() ? 1u : 0u);
  w.i64(r.deadline ? std::int64_t(*r.deadline) : 0);
  w.u32(static_cast<std::uint32_t>(r.priority));
  w.f64(r.tenant_weight);
  w.i64(r.arrival_time);
  w.u8(r.max_queue_delay.has_value() ? 1u : 0u);
  w.i64(r.max_queue_delay ? std::int64_t(*r.max_queue_delay) : 0);
  w.u64(r.memory_estimate_bytes);
  w.u64(r.compute_estimate);
  w.u32(static_cast<std::uint32_t>(r.max_retries));
  w.str(r.requester);
}

Result<PrefillRequest> dec_req(ByteReader& r) {
  PrefillRequest q;
  auto a = r.strong<RequestIdTag>(); if (!a) return Result<PrefillRequest>::err(a.error()); q.request_id = a.value();
  auto tn = r.strong<TenantIdTag>(); if (!tn) return Result<PrefillRequest>::err(tn.error()); q.tenant_id = tn.value();
  auto s = r.str(); if (!s) return Result<PrefillRequest>::err(s.error()); q.model.name = s.value();
  s = r.str(); if (!s) return Result<PrefillRequest>::err(s.error()); q.model.revision = s.value();
  auto ad = r.strong<AdapterIdTag>(); if (!ad) return Result<PrefillRequest>::err(ad.error()); q.adapter_id = ad.value();
  auto u8 = r.u8(); if (!u8) return Result<PrefillRequest>::err(u8.error()); q.adapter_relation = static_cast<AdapterRelation>(u8.value());
  s = r.str(); if (!s) return Result<PrefillRequest>::err(s.error()); q.vocab.tokenizer = s.value();
  auto u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.vocab.vocab_size = u64.value();
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.prompt_token_count = u64.value();
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.reusable_prefix.token_count = u64.value();
  u8 = r.u8(); if (!u8) return Result<PrefillRequest>::err(u8.error()); q.reusable_prefix.validated = (u8.value() != 0);
  s = r.str(); if (!s) return Result<PrefillRequest>::err(s.error()); q.reusable_prefix.fingerprint = s.value();
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.input_shape.seq_len = u64.value();
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.input_shape.hidden_dim = u64.value();
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.input_shape.num_heads = u64.value();
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.input_shape.head_dim = u64.value();
  u8 = r.u8(); if (!u8) return Result<PrefillRequest>::err(u8.error()); q.input_layout = static_cast<InputLayout>(u8.value());
  u8 = r.u8(); if (!u8) return Result<PrefillRequest>::err(u8.error()); q.numeric_mode = static_cast<NumericMode>(u8.value());
  s = r.str(); if (!s) return Result<PrefillRequest>::err(s.error()); q.backend = s.value();
  auto dv = r.strong<DeviceIdTag>(); if (!dv) return Result<PrefillRequest>::err(dv.error()); q.device = dv.value();
  s = r.str(); if (!s) return Result<PrefillRequest>::err(s.error()); q.device_match = s.value();
  u8 = r.u8(); if (!u8) return Result<PrefillRequest>::err(u8.error()); q.latency_class = static_cast<LatencyClass>(u8.value());
  u8 = r.u8(); if (!u8) return Result<PrefillRequest>::err(u8.error()); const bool has_dl = (u8.value() != 0);
  auto i64 = r.i64(); if (!i64) return Result<PrefillRequest>::err(i64.error()); if (has_dl) q.deadline = Nanoseconds(i64.value());
  auto u32 = r.u32(); if (!u32) return Result<PrefillRequest>::err(u32.error()); q.priority = static_cast<int>(u32.value());
  auto f = r.f64(); if (!f) return Result<PrefillRequest>::err(f.error()); q.tenant_weight = f.value();
  i64 = r.i64(); if (!i64) return Result<PrefillRequest>::err(i64.error()); q.arrival_time = i64.value();
  u8 = r.u8(); if (!u8) return Result<PrefillRequest>::err(u8.error()); const bool has_qd = (u8.value() != 0);
  i64 = r.i64(); if (!i64) return Result<PrefillRequest>::err(i64.error()); if (has_qd) q.max_queue_delay = Nanoseconds(i64.value());
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.memory_estimate_bytes = u64.value();
  u64 = r.u64(); if (!u64) return Result<PrefillRequest>::err(u64.error()); q.compute_estimate = u64.value();
  u32 = r.u32(); if (!u32) return Result<PrefillRequest>::err(u32.error()); q.max_retries = static_cast<int>(u32.value());
  s = r.str(); if (!s) return Result<PrefillRequest>::err(s.error()); q.requester = s.value();
  return Result<PrefillRequest>::ok(q);
}

}  // namespace


void enc_member(ByteWriter& w, const ExecutableMember& m) {
  w.strong(m.request_id); w.strong(m.attempt_id); w.strong(m.generation); w.strong(m.parent_generation);
  w.u64(m.token_start); w.u64(m.token_count); w.u8(static_cast<std::uint8_t>(m.work_kind));
  w.u64(m.memory_estimate_bytes); w.u64(m.compute_estimate); w.strong(m.assigned_worker); w.u32(m.member_index);
}

Result<ExecutableMember> dec_member(ByteReader& r) {
  ExecutableMember m;
  auto v = r.strong<RequestIdTag>(); if (!v) return Result<ExecutableMember>::err(v.error()); m.request_id = v.value();
  auto at = r.strong<AttemptIdTag>(); if (!at) return Result<ExecutableMember>::err(at.error()); m.attempt_id = at.value();
  auto ge = r.strong<GenerationTag>(); if (!ge) return Result<ExecutableMember>::err(ge.error()); m.generation = ge.value();
  auto pg = r.strong<GenerationTag>(); if (!pg) return Result<ExecutableMember>::err(pg.error()); m.parent_generation = pg.value();
  auto u64 = r.u64(); if (!u64) return Result<ExecutableMember>::err(u64.error()); m.token_start = u64.value();
  u64 = r.u64(); if (!u64) return Result<ExecutableMember>::err(u64.error()); m.token_count = u64.value();
  auto u8 = r.u8(); if (!u8) return Result<ExecutableMember>::err(u8.error()); m.work_kind = static_cast<WorkKind>(u8.value());
  u64 = r.u64(); if (!u64) return Result<ExecutableMember>::err(u64.error()); m.memory_estimate_bytes = u64.value();
  u64 = r.u64(); if (!u64) return Result<ExecutableMember>::err(u64.error()); m.compute_estimate = u64.value();
  auto w = r.strong<WorkerIdTag>(); if (!w) return Result<ExecutableMember>::err(w.error()); m.assigned_worker = w.value();
  auto u32 = r.u32(); if (!u32) return Result<ExecutableMember>::err(u32.error()); m.member_index = u32.value();
  return Result<ExecutableMember>::ok(m);
}

void enc_mres(ByteWriter& w, const MemberResult& r) {
  w.strong(r.request_id); w.strong(r.attempt_id); w.strong(r.generation);
  w.u8(static_cast<std::uint8_t>(r.outcome));
  w.u64(r.tokens_completed); w.u64(r.digest);
  w.u8(static_cast<std::uint8_t>(r.failure_code)); w.str(r.message);
  w.u64(r.next_token_start); w.u8(r.requires_next_chunk ? 1u : 0u);
  w.strong(r.epoch); w.strong(r.worker); w.strong(r.boot);
}

Result<MemberResult> dec_mres(ByteReader& r) {
  MemberResult m;
  auto v = r.strong<RequestIdTag>(); if (!v) return Result<MemberResult>::err(v.error()); m.request_id = v.value();
  auto at = r.strong<AttemptIdTag>(); if (!at) return Result<MemberResult>::err(at.error()); m.attempt_id = at.value();
  auto ge = r.strong<GenerationTag>(); if (!ge) return Result<MemberResult>::err(ge.error()); m.generation = ge.value();
  auto u8 = r.u8(); if (!u8) return Result<MemberResult>::err(u8.error()); m.outcome = static_cast<MemberOutcome>(u8.value());
  auto u64 = r.u64(); if (!u64) return Result<MemberResult>::err(u64.error()); m.tokens_completed = u64.value();
  u64 = r.u64(); if (!u64) return Result<MemberResult>::err(u64.error()); m.digest = u64.value();
  u8 = r.u8(); if (!u8) return Result<MemberResult>::err(u8.error()); m.failure_code = static_cast<ErrorCode>(u8.value());
  auto s = r.str(); if (!s) return Result<MemberResult>::err(s.error()); m.message = s.value();
  u64 = r.u64(); if (!u64) return Result<MemberResult>::err(u64.error()); m.next_token_start = u64.value();
  u8 = r.u8(); if (!u8) return Result<MemberResult>::err(u8.error()); m.requires_next_chunk = (u8.value() != 0);
  auto e = r.strong<EpochTag>(); if (!e) return Result<MemberResult>::err(e.error()); m.epoch = e.value();
  auto wk = r.strong<WorkerIdTag>(); if (!wk) return Result<MemberResult>::err(wk.error()); m.worker = wk.value();
  auto b = r.strong<WorkerBootIdTag>(); if (!b) return Result<MemberResult>::err(b.error()); m.boot = b.value();
  return Result<MemberResult>::ok(m);
}

void enc_group(ByteWriter& w, const ExecutionGroup& g) {
  w.strong(g.group_id); w.str(g.backend); w.strong(g.device);
  w.u64(g.key.hash()); w.u32(g.size());
  for (const auto& m : g.members) enc_member(w, m);
  w.u64(g.total_token_count); w.u64(g.max_memory_estimate_bytes); w.u64(g.sum_compute_estimate);
}

Result<ExecutionGroup> dec_group(ByteReader& r) {
  ExecutionGroup g;
  auto v = r.strong<GroupIdTag>(); if (!v) return Result<ExecutionGroup>::err(v.error()); g.group_id = v.value();
  auto s = r.str(); if (!s) return Result<ExecutionGroup>::err(s.error()); g.backend = s.value();
  auto dv = r.strong<DeviceIdTag>(); if (!dv) return Result<ExecutionGroup>::err(dv.error()); g.device = dv.value();
  auto h = r.u64(); if (!h) return Result<ExecutionGroup>::err(h.error()); g.key.policy_fingerprint = std::to_string(h.value());
  auto u32 = r.u32(); if (!u32) return Result<ExecutionGroup>::err(u32.error());
  for (std::uint32_t i = 0; i < u32.value(); ++i) { auto m = dec_member(r); if (!m) return Result<ExecutionGroup>::err(m.error()); g.members.push_back(m.value()); }
  auto u64 = r.u64(); if (!u64) return Result<ExecutionGroup>::err(u64.error()); g.total_token_count = u64.value();
  u64 = r.u64(); if (!u64) return Result<ExecutionGroup>::err(u64.error()); g.max_memory_estimate_bytes = u64.value();
  u64 = r.u64(); if (!u64) return Result<ExecutionGroup>::err(u64.error()); g.sum_compute_estimate = u64.value();
  return Result<ExecutionGroup>::ok(g);
}

void enc_stats(ByteWriter& w, const Stats& st) {
  w.u64(st.submitted); w.u64(st.admitted); w.u64(st.rejected); w.u64(st.cancelled); w.u64(st.retried);
  w.u64(st.completed); w.u64(st.failed_non_retryable); w.u64(st.expired); w.u64(st.deadline_missed); w.u64(st.groups_formed);
  w.u64(st.scheduled_tokens); w.u64(st.completed_tokens); w.u64(st.chunk_count); w.u64(st.stale_authority_rejected);
  w.u64(st.duplicate_completion_rejected);
  w.u64(st.queue_depth); w.u64(st.waiting_tokens); w.u64(st.running_requests); w.u64(st.remaining_prefill_tokens);
  w.i64(st.queue_latency_total_ns); w.i64(st.execution_latency_total_ns); w.u64(st.queue_latency_samples); w.u64(st.execution_latency_samples);
}

Stats dec_stats(ByteReader& r) {
  Stats st;
  auto u64 = r.u64(); if (u64) st.submitted = u64.value();
  u64 = r.u64(); if (u64) st.admitted = u64.value();
  u64 = r.u64(); if (u64) st.rejected = u64.value();
  u64 = r.u64(); if (u64) st.cancelled = u64.value();
  u64 = r.u64(); if (u64) st.retried = u64.value();
  u64 = r.u64(); if (u64) st.completed = u64.value();
  u64 = r.u64(); if (u64) st.failed_non_retryable = u64.value();
  u64 = r.u64(); if (u64) st.expired = u64.value();
  u64 = r.u64(); if (u64) st.deadline_missed = u64.value();
  u64 = r.u64(); if (u64) st.groups_formed = u64.value();
  u64 = r.u64(); if (u64) st.scheduled_tokens = u64.value();
  u64 = r.u64(); if (u64) st.completed_tokens = u64.value();
  u64 = r.u64(); if (u64) st.chunk_count = u64.value();
  u64 = r.u64(); if (u64) st.stale_authority_rejected = u64.value();
  u64 = r.u64(); if (u64) st.duplicate_completion_rejected = u64.value();
  u64 = r.u64(); if (u64) st.queue_depth = u64.value();
  u64 = r.u64(); if (u64) st.waiting_tokens = u64.value();
  u64 = r.u64(); if (u64) st.running_requests = u64.value();
  u64 = r.u64(); if (u64) st.remaining_prefill_tokens = u64.value();
  auto i = r.i64(); if (i) st.queue_latency_total_ns = i.value();
  i = r.i64(); if (i) st.execution_latency_total_ns = i.value();
  u64 = r.u64(); if (u64) st.queue_latency_samples = u64.value();
  u64 = r.u64(); if (u64) st.execution_latency_samples = u64.value();
  return st;
}


Result<std::vector<std::uint8_t>> encode_body(const Message& msg) {
  ByteWriter w;
  w.u8(kProtocolVersion);
  w.u8(static_cast<std::uint8_t>(msg.kind));
  switch (msg.kind) {
    case MsgKind::hello: break;
    case MsgKind::worker_register: enc_worker(w, msg.worker); break;
    case MsgKind::worker_register_ack: w.strong(msg.epoch); enc_worker(w, msg.worker); break;
    case MsgKind::submit: enc_req(w, msg.request); w.u64(msg.tokens.size()); for (auto t : msg.tokens) w.u32(t); break;
    case MsgKind::submit_ack:
      w.strong(msg.request_id); w.u8(msg.accepted ? 1u : 0u);
      w.u32(static_cast<std::uint32_t>(msg.error.code)); w.str(msg.error.message);
      w.strong(msg.attempt_id); break;
    case MsgKind::cancel: w.strong(msg.request_id); break;
    case MsgKind::cancel_ack:
      w.strong(msg.request_id); w.u8(msg.applied ? 1u : 0u); w.u8(msg.cancel_result.was_terminal ? 1u : 0u);
      w.u32(static_cast<std::uint32_t>(msg.error.code)); w.str(msg.error.message); break;
    case MsgKind::query_state: break;
    case MsgKind::state: w.strong(msg.epoch); w.u64(msg.workers.size()); for (const auto& x : msg.workers) enc_worker(w, x); break;
    case MsgKind::query_explain: w.strong(msg.request_id); break;
    case MsgKind::explain_resp:
      w.strong(msg.request_id); w.strong(msg.attempt_id); w.u8(static_cast<std::uint8_t>(msg.lifecycle));
      w.strong(msg.generation); w.u64(msg.explain.trace.size()); for (const auto& t : msg.explain.trace) w.str(t); break;
    case MsgKind::query_stats: break;
    case MsgKind::stats_resp: enc_stats(w, msg.stats); break;
    case MsgKind::dispatch:
      enc_group(w, msg.group);
      w.u64(msg.member_tokens.size()); for (auto t : msg.member_tokens) w.u32(t);
      w.u64(msg.member_token_offsets.size()); for (auto o : msg.member_token_offsets) w.u64(o);
      w.strong(msg.epoch); w.strong(msg.worker.boot_id); break;
    case MsgKind::completion: w.u64(msg.member_results.size()); for (const auto& mr : msg.member_results) enc_mres(w, mr); break;
    case MsgKind::shutdown: break;
    case MsgKind::shutdown_ack: break;
    case MsgKind::ping: break;
    case MsgKind::pong: break;
    case MsgKind::roll_epoch: break;
    case MsgKind::error: w.u32(static_cast<std::uint32_t>(msg.error.code)); w.str(msg.error.message); break;
    default: return Result<std::vector<std::uint8_t>>::err(ErrorCode::protocol_violation, "unknown message kind");
  }
  return Result<std::vector<std::uint8_t>>::ok(w.bytes());
}

Result<Message> decode_body(std::uint8_t version, MsgKind kind, const std::uint8_t* payload, std::size_t len) {
  if (version != kProtocolVersion)
    return Result<Message>::err(ErrorCode::protocol_violation, "unsupported protocol version");
  ByteReader r(payload, len);
  Message m;
  m.kind = kind;
  auto fail = [&](const ErrorInfo& e) { return Result<Message>::err(e); };
  switch (kind) {
    case MsgKind::hello: break;
    case MsgKind::worker_register: { auto w = dec_worker(r); if (!w) return fail(w.error()); m.worker = w.value(); break; }
    case MsgKind::worker_register_ack: { auto e = r.strong<EpochTag>(); if (!e) return fail(e.error()); m.epoch = e.value(); auto w = dec_worker(r); if (!w) return fail(w.error()); m.worker = w.value(); break; }
    case MsgKind::submit: { auto q = dec_req(r); if (!q) return fail(q.error()); m.request = q.value(); auto n = r.u64(); if (!n) return fail(n.error()); if (n.value() > (1ULL << 26)) return Result<Message>::err(ErrorCode::protocol_violation, "token count unreasonable"); m.tokens.resize(static_cast<std::size_t>(n.value())); for (auto& t : m.tokens) { auto x = r.u32(); if (!x) return fail(x.error()); t = x.value(); } break; }
    case MsgKind::submit_ack: { auto id = r.strong<RequestIdTag>(); if (!id) return fail(id.error()); m.request_id = id.value(); auto acc = r.u8(); if (!acc) return fail(acc.error()); m.accepted = (acc.value() != 0); auto c = r.u32(); if (!c) return fail(c.error()); m.error.code = static_cast<ErrorCode>(c.value()); auto s = r.str(); if (!s) return fail(s.error()); m.error.message = s.value(); auto at = r.strong<AttemptIdTag>(); if (!at) return fail(at.error()); m.attempt_id = at.value(); break; }
    case MsgKind::cancel: { auto id = r.strong<RequestIdTag>(); if (!id) return fail(id.error()); m.request_id = id.value(); break; }
    case MsgKind::cancel_ack: { auto id = r.strong<RequestIdTag>(); if (!id) return fail(id.error()); m.request_id = id.value(); auto a = r.u8(); if (!a) return fail(a.error()); m.applied = (a.value() != 0); auto w = r.u8(); if (!w) return fail(w.error()); m.cancel_result.was_terminal = (w.value() != 0); auto c = r.u32(); if (!c) return fail(c.error()); m.error.code = static_cast<ErrorCode>(c.value()); auto s = r.str(); if (!s) return fail(s.error()); m.error.message = s.value(); break; }
    case MsgKind::query_state: break;
    case MsgKind::state: { auto e = r.strong<EpochTag>(); if (!e) return fail(e.error()); m.epoch = e.value(); auto n = r.u64(); if (!n) return fail(n.error()); if (n.value() > (1ULL << 20)) return Result<Message>::err(ErrorCode::protocol_violation, "worker count unreasonable"); for (std::uint64_t i = 0; i < n.value(); ++i) { auto w = dec_worker(r); if (!w) return fail(w.error()); m.workers.push_back(w.value()); } break; }
    case MsgKind::query_explain: { auto id = r.strong<RequestIdTag>(); if (!id) return fail(id.error()); m.request_id = id.value(); break; }
    case MsgKind::explain_resp: { auto id = r.strong<RequestIdTag>(); if (!id) return fail(id.error()); m.request_id = id.value(); auto at = r.strong<AttemptIdTag>(); if (!at) return fail(at.error()); m.attempt_id = at.value(); auto lc = r.u8(); if (!lc) return fail(lc.error()); m.lifecycle = static_cast<Lifecycle>(lc.value()); auto g = r.strong<GenerationTag>(); if (!g) return fail(g.error()); m.generation = g.value(); auto n = r.u64(); if (!n) return fail(n.error()); if (n.value() > (1ULL << 22)) return Result<Message>::err(ErrorCode::protocol_violation, "trace length unreasonable"); for (std::uint64_t i = 0; i < n.value(); ++i) { auto s = r.str(); if (!s) return fail(s.error()); m.explain.trace.push_back(s.value()); } break; }
    case MsgKind::query_stats: break;
    case MsgKind::stats_resp: m.stats = dec_stats(r); break;
    case MsgKind::dispatch: { auto g = dec_group(r); if (!g) return fail(g.error()); m.group = g.value(); auto n = r.u64(); if (!n) return fail(n.error()); if (n.value() > (1ULL << 28)) return Result<Message>::err(ErrorCode::protocol_violation, "member_tokens too large"); m.member_tokens.resize(static_cast<std::size_t>(n.value())); for (auto& t : m.member_tokens) { auto x = r.u32(); if (!x) return fail(x.error()); t = x.value(); } auto n2 = r.u64(); if (!n2) return fail(n2.error()); if (n2.value() > (1ULL << 20)) return Result<Message>::err(ErrorCode::protocol_violation, "member offsets too large"); m.member_token_offsets.resize(static_cast<std::size_t>(n2.value())); for (auto& o : m.member_token_offsets) { auto x = r.u64(); if (!x) return fail(x.error()); o = x.value(); } auto e = r.strong<EpochTag>(); if (!e) return fail(e.error()); m.epoch = e.value(); auto b = r.strong<WorkerBootIdTag>(); if (!b) return fail(b.error()); m.worker.boot_id = b.value(); break; }
    case MsgKind::completion: { auto n = r.u64(); if (!n) return fail(n.error()); if (n.value() > (1ULL << 22)) return Result<Message>::err(ErrorCode::protocol_violation, "completion count unreasonable"); for (std::uint64_t i = 0; i < n.value(); ++i) { auto mr = dec_mres(r); if (!mr) return fail(mr.error()); m.member_results.push_back(mr.value()); } break; }
    case MsgKind::shutdown: break;
    case MsgKind::shutdown_ack: break;
    case MsgKind::ping: break;
    case MsgKind::pong: break;
    case MsgKind::roll_epoch: break;
    case MsgKind::error: { auto c = r.u32(); if (!c) return fail(c.error()); m.error.code = static_cast<ErrorCode>(c.value()); auto s = r.str(); if (!s) return fail(s.error()); m.error.message = s.value(); break; }
    default: return Result<Message>::err(ErrorCode::malformed_frame, "unknown message kind");
  }
  return Result<Message>::ok(std::move(m));
}

bool valid_frame_length(std::uint32_t length) noexcept {
  return length >= 2 && length <= kFrameMaxBytes;
}

// encode_frame returns the frame BODY (version + kind + payload). The socket
// layer adds the 4-byte little-endian length prefix.
Result<std::vector<std::uint8_t>> encode_frame(const Message& msg) { return encode_body(msg); }

// decode_frame expects the frame BODY (as returned by Socket::recv_frame).
Result<Message> decode_frame(const std::vector<std::uint8_t>& body) {
  if (body.size() < 2) return Result<Message>::err(ErrorCode::malformed_frame, "frame body too small");
  const std::uint8_t version = body[0];
  const std::uint8_t kind_byte = body[1];
  const MsgKind kind = static_cast<MsgKind>(kind_byte);
  if (kind_byte < static_cast<std::uint8_t>(MsgKind::hello) ||
      kind_byte > static_cast<std::uint8_t>(MsgKind::error))
    return Result<Message>::err(ErrorCode::malformed_frame, "unknown message kind byte");
  return decode_body(version, kind, body.data() + 2, body.size() - 2);
}

}  // namespace prefillfabric
