// Prefill Fabric - persistence implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/persistence.hpp"
#include "prefillfabric/compatibility.hpp"
#include <fstream>
#include <cstring>

namespace prefillfabric {

namespace {

void write_string_field(ByteWriter& w, const std::string& s) { w.str(s); }

void write_req(ByteWriter& w, const PersistedRequestRecord& r) {
  w.strong(r.request_id);
  w.strong(r.attempt_id);
  w.strong(r.tenant_id);
  w.str(r.model.name);
  w.str(r.model.revision);
  w.strong(r.adapter_id);
  w.u8(static_cast<std::uint8_t>(r.relation));
  w.u64(r.prompt_token_count);
  w.u64(r.cached_tokens);
  w.u64(r.completed_uncached_tokens);
  w.strong(r.current_generation);
  w.strong(r.parent_generation);
  w.u64(r.next_chunk_token_start);
  w.u8(static_cast<std::uint8_t>(r.work_kind));
  w.u8(static_cast<std::uint8_t>(r.lifecycle));
  w.u32(static_cast<std::uint32_t>(r.priority));
  w.f64(r.tenant_weight);
  w.f64(r.normalized_service);
  w.u64(r.last_chunk_token_count);
  w.u64(r.completed_token_total);
  w.u8(r.needs_recovery_full_prefill ? 1u : 0u);
  w.str(r.vocab.tokenizer);
  w.u64(r.vocab.vocab_size);
  w.u8(static_cast<std::uint8_t>(r.numeric_mode));
  w.u8(static_cast<std::uint8_t>(r.latency_class));
  w.i64(r.deadline_ns);
  w.i64(r.arrival_ns);
  w.u8(r.has_deadline ? 1u : 0u);
  w.u8(r.has_queue_delay ? 1u : 0u);
  w.i64(r.max_queue_delay_ns);
}

Result<PersistedRequestRecord> read_req(ByteReader& r) {
  PersistedRequestRecord rec;
  auto rd = [&]() -> Result<void> {
    auto ridv = r.strong<RequestIdTag>(); if (!ridv) return Result<void>::err(ridv.error()); rec.request_id = ridv.value();
    auto atv = r.strong<AttemptIdTag>(); if (!atv) return Result<void>::err(atv.error()); rec.attempt_id = atv.value();
    auto tndv = r.strong<TenantIdTag>(); if (!tndv) return Result<void>::err(tndv.error()); rec.tenant_id = tndv.value();
    auto s = r.str(); if (!s) return Result<void>::err(s.error()); rec.model.name = s.value();
    s = r.str(); if (!s) return Result<void>::err(s.error()); rec.model.revision = s.value();
    auto a = r.strong<AdapterIdTag>(); if (!a) return Result<void>::err(a.error()); rec.adapter_id = a.value();
    auto u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.relation = static_cast<AdapterRelation>(u8.value());
    auto u64 = r.u64(); if (!u64) return Result<void>::err(u64.error()); rec.prompt_token_count = u64.value();
    u64 = r.u64(); if (!u64) return Result<void>::err(u64.error()); rec.cached_tokens = u64.value();
    u64 = r.u64(); if (!u64) return Result<void>::err(u64.error()); rec.completed_uncached_tokens = u64.value();
    auto g = r.strong<GenerationTag>(); if (!g) return Result<void>::err(g.error()); rec.current_generation = g.value();
    g = r.strong<GenerationTag>(); if (!g) return Result<void>::err(g.error()); rec.parent_generation = g.value();
    u64 = r.u64(); if (!u64) return Result<void>::err(u64.error()); rec.next_chunk_token_start = u64.value();
    u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.work_kind = static_cast<WorkKind>(u8.value());
    u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.lifecycle = static_cast<Lifecycle>(u8.value());
    auto u32 = r.u32(); if (!u32) return Result<void>::err(u32.error()); rec.priority = static_cast<int>(u32.value());
    auto f = r.f64(); if (!f) return Result<void>::err(f.error()); rec.tenant_weight = f.value();
    f = r.f64(); if (!f) return Result<void>::err(f.error()); rec.normalized_service = f.value();
    u64 = r.u64(); if (!u64) return Result<void>::err(u64.error()); rec.last_chunk_token_count = u64.value();
    u64 = r.u64(); if (!u64) return Result<void>::err(u64.error()); rec.completed_token_total = u64.value();
    u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.needs_recovery_full_prefill = u8.value() != 0;
    s = r.str(); if (!s) return Result<void>::err(s.error()); rec.vocab.tokenizer = s.value();
    u64 = r.u64(); if (!u64) return Result<void>::err(u64.error()); rec.vocab.vocab_size = u64.value();
    u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.numeric_mode = static_cast<NumericMode>(u8.value());
    u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.latency_class = static_cast<LatencyClass>(u8.value());
    auto i = r.i64(); if (!i) return Result<void>::err(i.error()); rec.deadline_ns = i.value();
    i = r.i64(); if (!i) return Result<void>::err(i.error()); rec.arrival_ns = i.value();
    u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.has_deadline = u8.value() != 0;
    u8 = r.u8(); if (!u8) return Result<void>::err(u8.error()); rec.has_queue_delay = u8.value() != 0;
    i = r.i64(); if (!i) return Result<void>::err(i.error()); rec.max_queue_delay_ns = i.value();
    return Result<void>::ok();
  };
  auto res = rd();
  if (!res) return Result<PersistedRequestRecord>::err(res.error());
  return Result<PersistedRequestRecord>::ok(rec);
}

void write_worker(ByteWriter& w, const PersistedWorkerRecord& a) {
  w.strong(a.worker_id);
  w.strong(a.boot_id);
  w.str(a.host);
  w.u32(a.port);
  w.str(a.backend);
  w.u8(static_cast<std::uint8_t>(a.state));
  w.u64(a.capacity_units);
}

Result<PersistedWorkerRecord> read_worker(ByteReader& r) {
  PersistedWorkerRecord a;
  auto wv = r.strong<WorkerIdTag>(); if (!wv) return Result<PersistedWorkerRecord>::err(wv.error()); a.worker_id = wv.value();
  auto bv = r.strong<WorkerBootIdTag>(); if (!bv) return Result<PersistedWorkerRecord>::err(bv.error()); a.boot_id = bv.value();
  auto s = r.str(); if (!s) return Result<PersistedWorkerRecord>::err(s.error()); a.host = s.value();
  auto u32 = r.u32(); if (!u32) return Result<PersistedWorkerRecord>::err(u32.error()); a.port = u32.value();
  s = r.str(); if (!s) return Result<PersistedWorkerRecord>::err(s.error()); a.backend = s.value();
  auto u8 = r.u8(); if (!u8) return Result<PersistedWorkerRecord>::err(u8.error()); a.state = static_cast<WorkerState>(u8.value());
  auto u64 = r.u64(); if (!u64) return Result<PersistedWorkerRecord>::err(u64.error()); a.capacity_units = u64.value();
  return Result<PersistedWorkerRecord>::ok(a);
}

void write_tenant(ByteWriter& w, const PersistedTenantRecord& t) {
  w.strong(t.tenant_id);
  w.f64(t.weight);
  w.u64(t.scheduled_tokens);
  w.u64(t.completed_requests);
  w.i64(t.wait_total_ns);
  w.u64(t.outstanding);
  w.f64(t.normalized_service);
}

Result<PersistedTenantRecord> read_tenant(ByteReader& r) {
  PersistedTenantRecord t;
  auto v = r.strong<TenantIdTag>(); if (!v) return Result<PersistedTenantRecord>::err(v.error()); t.tenant_id = v.value();
  auto f = r.f64(); if (!f) return Result<PersistedTenantRecord>::err(f.error()); t.weight = f.value();
  auto u64 = r.u64(); if (!u64) return Result<PersistedTenantRecord>::err(u64.error()); t.scheduled_tokens = u64.value();
  u64 = r.u64(); if (!u64) return Result<PersistedTenantRecord>::err(u64.error()); t.completed_requests = u64.value();
  auto i = r.i64(); if (!i) return Result<PersistedTenantRecord>::err(i.error()); t.wait_total_ns = i.value();
  u64 = r.u64(); if (!u64) return Result<PersistedTenantRecord>::err(u64.error()); t.outstanding = u64.value();
  f = r.f64(); if (!f) return Result<PersistedTenantRecord>::err(f.error()); t.normalized_service = f.value();
  return Result<PersistedTenantRecord>::ok(t);
}

}  // namespace

Result<std::vector<std::uint8_t>> serialize_state(const PersistedState& state) {
  ByteWriter payload;
  payload.strong(state.epoch);
  payload.u64(state.next_generation_value);
  payload.u64(state.next_attempt_value);
  payload.u8(state.was_clean_shutdown ? 1u : 0u);
  payload.u64(static_cast<std::uint64_t>(state.requests.size()));
  for (const auto& r : state.requests) write_req(payload, r);
  payload.u64(static_cast<std::uint64_t>(state.workers.size()));
  for (const auto& w : state.workers) write_worker(payload, w);
  payload.u64(static_cast<std::uint64_t>(state.tenants.size()));
  for (const auto& t : state.tenants) write_tenant(payload, t);

  const auto& pl = payload.bytes();
  std::uint64_t checksum = 1469598103934665603ULL;
  checksum = fnv1a64(checksum, pl.data(), pl.size());

  ByteWriter w;
  w.u32(state.magic);
  w.u32(state.version);
  w.u32(0u);  // flags (reserved)
  w.u64(static_cast<std::uint64_t>(pl.size()));
  w.u64(checksum);
  w.raw(pl.data(), pl.size());
  return Result<std::vector<std::uint8_t>>::ok(w.bytes());
}

Result<PersistedState> deserialize_state(const std::vector<std::uint8_t>& blob) {
  ByteReader r(blob);
  auto magic = r.u32(); if (!magic) return Result<PersistedState>::err(magic.error());
  auto version = r.u32(); if (!version) return Result<PersistedState>::err(version.error());
  auto flags = r.u32(); if (!flags) return Result<PersistedState>::err(flags.error());
  auto plen = r.u64(); if (!plen) return Result<PersistedState>::err(plen.error());
  auto stored_checksum = r.u64(); if (!stored_checksum) return Result<PersistedState>::err(stored_checksum.error());

  if (magic.value() != kPersistenceMagic)
    return Result<PersistedState>::err(ErrorCode::corrupt_state, "bad magic");
  if (version.value() != kPersistenceVersion)
    return Result<PersistedState>::err(ErrorCode::unknown_version, "unsupported persistence version");
  const std::size_t blob_len = blob.size();
  if (std::size_t(plen.value()) > blob_len - r.offset())
    return Result<PersistedState>::err(ErrorCode::truncation, "payload length exceeds blob");

  std::uint64_t checksum = 1469598103934665603ULL;
  checksum = fnv1a64(checksum, blob.data() + r.offset(), static_cast<std::size_t>(plen.value()));
  if (checksum != stored_checksum.value())
    return Result<PersistedState>::err(ErrorCode::checksum_mismatch, "checksum mismatch");

  // Exactly consume the payload for the declared length.
  ByteReader pr(blob.data() + r.offset(), static_cast<std::size_t>(plen.value()));
  PersistedState s;
  s.magic = magic.value();
  s.version = version.value();
  auto e = pr.strong<EpochTag>(); if (!e) return Result<PersistedState>::err(e.error()); s.epoch = e.value();
  auto u64 = pr.u64(); if (!u64) return Result<PersistedState>::err(u64.error()); s.next_generation_value = u64.value();
  u64 = pr.u64(); if (!u64) return Result<PersistedState>::err(u64.error()); s.next_attempt_value = u64.value();
  auto u8 = pr.u8(); if (!u8) return Result<PersistedState>::err(u8.error()); s.was_clean_shutdown = u8.value() != 0;

  auto nreq = pr.u64(); if (!nreq) return Result<PersistedState>::err(nreq.error());
  if (nreq.value() > (1ULL << 24)) return Result<PersistedState>::err(ErrorCode::corrupt_state, "unreasonable request count");
  for (std::uint64_t i = 0; i < nreq.value(); ++i) {
    auto rec = read_req(pr); if (!rec) return Result<PersistedState>::err(rec.error());
    s.requests.push_back(std::move(rec.value()));
  }
  auto nwk = pr.u64(); if (!nwk) return Result<PersistedState>::err(nwk.error());
  if (nwk.value() > (1ULL << 20)) return Result<PersistedState>::err(ErrorCode::corrupt_state, "unreasonable worker count");
  for (std::uint64_t i = 0; i < nwk.value(); ++i) {
    auto rec = read_worker(pr); if (!rec) return Result<PersistedState>::err(rec.error());
    s.workers.push_back(std::move(rec.value()));
  }
  auto ntenant = pr.u64(); if (!ntenant) return Result<PersistedState>::err(ntenant.error());
  if (ntenant.value() > (1ULL << 20)) return Result<PersistedState>::err(ErrorCode::corrupt_state, "unreasonable tenant count");
  for (std::uint64_t i = 0; i < ntenant.value(); ++i) {
    auto rec = read_tenant(pr); if (!rec) return Result<PersistedState>::err(rec.error());
    s.tenants.push_back(std::move(rec.value()));
  }
  if (pr.remaining() != 0)
    return Result<PersistedState>::err(ErrorCode::corrupt_state, "trailing bytes in payload");

  // Semantic validation: lifecycle and work_kind must be in range.
  for (const auto& rec : s.requests) {
    if (static_cast<int>(rec.lifecycle) < static_cast<int>(Lifecycle::submitted) ||
        static_cast<int>(rec.lifecycle) > static_cast<int>(Lifecycle::rejected))
      return Result<PersistedState>::err(ErrorCode::corrupt_state, "bad lifecycle in record");
    if (rec.cached_tokens > rec.prompt_token_count)
      return Result<PersistedState>::err(ErrorCode::corrupt_state, "cached_tokens exceeds prompt length");
  }
  return Result<PersistedState>::ok(std::move(s));
}

Result<void> FilePersistence::save(const PersistedState& state, const std::string& path) const {
  auto blob = serialize_state(state);
  if (!blob) return Result<void>::err(blob.error());
  std::ofstream out(path, std::ios::binary);
  if (!out) return Result<void>::err(ErrorCode::io_failure, "cannot open file for write");
  auto& bytes = blob.value();
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!out) return Result<void>::err(ErrorCode::io_failure, "file write failed");
  return Result<void>::ok();
}

Result<PersistedState> FilePersistence::load(const std::string& path) const {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Result<PersistedState>::err(ErrorCode::io_failure, "cannot open file for read");
  in.seekg(0, std::ios::end);
  const std::streamsize len = in.tellg();
  if (len < 0) return Result<PersistedState>::err(ErrorCode::io_failure, "cannot determine file size");
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> blob(static_cast<std::size_t>(len));
  if (len > 0) in.read(reinterpret_cast<char*>(blob.data()), len);
  if (!in && len > 0) return Result<PersistedState>::err(ErrorCode::io_failure, "file read failed");
  return deserialize_state(blob);
}

}  // namespace prefillfabric
