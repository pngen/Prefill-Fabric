// Prefill Fabric - client implementation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/client.hpp"

namespace prefillfabric {

Client::Client(const std::string& host, std::uint16_t port) : host_(host), port_(port) {}
Client::~Client() { close(); }

void Client::close() { sock_.close(); }

Result<void> Client::connect() {
  auto s = Socket::connect(host_, port_);
  if (!s) return Result<void>::err(s.error());
  sock_ = std::move(s.value());
  // Send hello to announce client role.
  Message h; h.kind = MsgKind::hello;
  auto hb = encode_frame(h);
  if (!hb) return Result<void>::err(hb.error());
  return sock_.send_frame(hb.value().data(), hb.value().size());
}

Result<Message> Client::roundtrip(const Message& msg, MsgKind expect) {
  if (!sock_.valid()) return Result<Message>::err(ErrorCode::io_failure, "client not connected");
  auto eb = encode_frame(msg);
  if (!eb) return Result<Message>::err(eb.error());
  auto sr = sock_.send_frame(eb.value().data(), eb.value().size());
  if (!sr) return Result<Message>::err(sr.error());
  auto fb = sock_.recv_frame();
  if (!fb) return Result<Message>::err(fb.error());
  auto dm = decode_frame(fb.value());
  if (!dm) return Result<Message>::err(dm.error());
  if (dm.value().kind == MsgKind::error)
    return Result<Message>::err(dm.value().error);
  if (dm.value().kind != expect)
    return Result<Message>::err(ErrorCode::protocol_violation, "unexpected reply kind");
  return Result<Message>::ok(std::move(dm.value()));
}

Result<RequestId> Client::submit(const PrefillRequest& req, const std::vector<std::uint32_t>& tokens) {
  Message m;
  m.kind = MsgKind::submit;
  m.request = req;
  m.tokens = tokens;
  auto r = roundtrip(m, MsgKind::submit_ack);
  if (!r) return Result<RequestId>::err(r.error());
  if (!r.value().accepted) return Result<RequestId>::err(r.value().error);
  return Result<RequestId>::ok(r.value().request_id);
}

Result<CancelResult> Client::cancel(RequestId req) {
  Message m; m.kind = MsgKind::cancel; m.request_id = req;
  auto r = roundtrip(m, MsgKind::cancel_ack);
  if (!r) return Result<CancelResult>::err(r.error());
  return Result<CancelResult>::ok(r.value().cancel_result);
}

Result<Message> Client::query_state() {
  Message m; m.kind = MsgKind::query_state;
  return roundtrip(m, MsgKind::state);
}

Result<Message> Client::query_explain(RequestId req) {
  Message m; m.kind = MsgKind::query_explain; m.request_id = req;
  return roundtrip(m, MsgKind::explain_resp);
}

Result<Stats> Client::query_stats() {
  Message m; m.kind = MsgKind::query_stats;
  auto r = roundtrip(m, MsgKind::stats_resp);
  if (!r) return Result<Stats>::err(r.error());
  return Result<Stats>::ok(r.value().stats);
}

Result<Epoch> Client::roll_epoch() {
  Message m; m.kind = MsgKind::roll_epoch;
  auto r = roundtrip(m, MsgKind::state);
  if (!r) return Result<Epoch>::err(r.error());
  return Result<Epoch>::ok(r.value().epoch);
}

Result<void> Client::shutdown() {
  Message m; m.kind = MsgKind::shutdown;
  auto r = roundtrip(m, MsgKind::shutdown_ack);
  if (!r) return Result<void>::err(r.error());
  return Result<void>::ok();
}

}  // namespace prefillfabric
