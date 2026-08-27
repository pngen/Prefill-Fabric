// Prefill Fabric - structured Result<T> and error codes.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <string>
#include <utility>
#include <type_traits>
#include <optional>
#include <variant>

namespace prefillfabric {

// Structured error codes. Ordinary control flow is expressed through
// Result<T> and these codes, not through throwing exceptions.
// (The library never throws for expected control-flow outcomes.)
enum class ErrorCode : int {
  ok = 0,

  // Generic / precondition failures.
  invalid_argument = 100,
  out_of_range = 101,
  not_found = 102,
  already_exists = 103,
  illegal_state = 104,
  precondition_failed = 105,
  capacity_exceeded = 106,
  not_implemented = 107,
  unsupported = 108,
  overflow = 109,

  // Admission and scheduling.
  rejected = 200,
  admission_rejected_terminal = 201,
  deadline_expired = 202,
  deadline_impossible = 203,
  memory_unavailable = 204,
  no_compatible_worker = 205,
  no_capacity = 206,
  backpressure = 207,
  queued_delay_exceeded = 208,

  // Request / work identity and lineage.
  unknown_request = 300,
  unknown_attempt = 301,
  unknown_tenant = 302,
  unknown_model = 303,
  duplicate_request = 304,
  invalid_reusable_prefix = 305,
  stale_epoch = 306,
  stale_worker = 307,
  stale_attempt = 308,
  stale_generation = 309,
  duplicate_completion = 310,
  terminal_cancelled = 311,
  superseded_retry = 312,

  // Execution / member outcomes.
  retryable_failure = 400,
  non_retryable_failure = 401,
  member_failed = 402,
  executor_failure = 403,
  kernel_failure = 404,
  device_failure = 405,
  cuda_unavailable = 406,
  cuda_error = 407,

  // Persistence / protocol.
  corrupt_state = 500,
  checksum_mismatch = 501,
  truncation = 502,
  unknown_version = 503,
  malformed_frame = 504,
  oversized_frame = 505,
  protocol_violation = 506,
  io_failure = 507,

  // Concurrency / internal.
  service_unavailable = 600,
  already_dispatching = 601,

  max_code = 700
};

inline const char* to_string(ErrorCode c) noexcept {
  switch (c) {
    case ErrorCode::ok: return "ok";
    case ErrorCode::invalid_argument: return "invalid_argument";
    case ErrorCode::out_of_range: return "out_of_range";
    case ErrorCode::not_found: return "not_found";
    case ErrorCode::already_exists: return "already_exists";
    case ErrorCode::illegal_state: return "illegal_state";
    case ErrorCode::precondition_failed: return "precondition_failed";
    case ErrorCode::capacity_exceeded: return "capacity_exceeded";
    case ErrorCode::not_implemented: return "not_implemented";
    case ErrorCode::unsupported: return "unsupported";
    case ErrorCode::overflow: return "overflow";
    case ErrorCode::rejected: return "rejected";
    case ErrorCode::admission_rejected_terminal: return "admission_rejected_terminal";
    case ErrorCode::deadline_expired: return "deadline_expired";
    case ErrorCode::deadline_impossible: return "deadline_impossible";
    case ErrorCode::memory_unavailable: return "memory_unavailable";
    case ErrorCode::no_compatible_worker: return "no_compatible_worker";
    case ErrorCode::no_capacity: return "no_capacity";
    case ErrorCode::backpressure: return "backpressure";
    case ErrorCode::queued_delay_exceeded: return "queued_delay_exceeded";
    case ErrorCode::unknown_request: return "unknown_request";
    case ErrorCode::unknown_attempt: return "unknown_attempt";
    case ErrorCode::unknown_tenant: return "unknown_tenant";
    case ErrorCode::unknown_model: return "unknown_model";
    case ErrorCode::duplicate_request: return "duplicate_request";
    case ErrorCode::invalid_reusable_prefix: return "invalid_reusable_prefix";
    case ErrorCode::stale_epoch: return "stale_epoch";
    case ErrorCode::stale_worker: return "stale_worker";
    case ErrorCode::stale_attempt: return "stale_attempt";
    case ErrorCode::stale_generation: return "stale_generation";
    case ErrorCode::duplicate_completion: return "duplicate_completion";
    case ErrorCode::terminal_cancelled: return "terminal_cancelled";
    case ErrorCode::superseded_retry: return "superseded_retry";
    case ErrorCode::retryable_failure: return "retryable_failure";
    case ErrorCode::non_retryable_failure: return "non_retryable_failure";
    case ErrorCode::member_failed: return "member_failed";
    case ErrorCode::executor_failure: return "executor_failure";
    case ErrorCode::kernel_failure: return "kernel_failure";
    case ErrorCode::device_failure: return "device_failure";
    case ErrorCode::cuda_unavailable: return "cuda_unavailable";
    case ErrorCode::cuda_error: return "cuda_error";
    case ErrorCode::corrupt_state: return "corrupt_state";
    case ErrorCode::checksum_mismatch: return "checksum_mismatch";
    case ErrorCode::truncation: return "truncation";
    case ErrorCode::unknown_version: return "unknown_version";
    case ErrorCode::malformed_frame: return "malformed_frame";
    case ErrorCode::oversized_frame: return "oversized_frame";
    case ErrorCode::protocol_violation: return "protocol_violation";
    case ErrorCode::io_failure: return "io_failure";
    case ErrorCode::service_unavailable: return "service_unavailable";
    case ErrorCode::already_dispatching: return "already_dispatching";
    default: return "unknown_error";
  }
}

struct ErrorInfo {
  ErrorCode code = ErrorCode::ok;
  std::string message;
  ErrorInfo() = default;
  ErrorInfo(ErrorCode c, std::string m) : code(c), message(std::move(m)) {}
  explicit operator bool() const noexcept { return code != ErrorCode::ok; }
};

// Result<std::monostate> is used as Result<void>-equivalent. To keep
// the API ergonomic we provide a dedicated Result<void> specialization.
template <typename T>
class Result {
 public:
  Result() = default;
  Result(const T& val) : value_(val), err_(ErrorCode::ok, "") {}
  Result(T&& val) : value_(std::move(val)), err_(ErrorCode::ok, "") {}
  Result(ErrorInfo err) : err_(std::move(err)) {}
  Result(ErrorCode c, std::string m) : err_(c, std::move(m)) {}

  static Result ok(T val) { return Result(std::move(val)); }
  static Result err(ErrorCode c, std::string m) { return Result(ErrorInfo(c, std::move(m))); }
  static Result err(const ErrorInfo& e) { return Result(e); }

  bool has_value() const noexcept { return err_.code == ErrorCode::ok; }
  bool is_ok() const noexcept { return has_value(); }
  bool is_error() const noexcept { return !has_value(); }
  explicit operator bool() const noexcept { return has_value(); }

  const T& value() const noexcept { return value_; }
  T& value() noexcept { return value_; }
  T&& move_value() noexcept { return std::move(value_); }

  const ErrorInfo& error() const noexcept { return err_; }
  const T& operator*() const noexcept { return value_; }
  const T* operator->() const noexcept { return &value_; }

  T value_or(T def) const & { return has_value() ? value_ : std::move(def); }
  T value_or(T def) && { return has_value() ? std::move(value_) : std::move(def); }

 private:
  T value_{};
  ErrorInfo err_;
};

template <>
class Result<void> {
 public:
  Result() = default;
  Result(ErrorInfo err) : err_(std::move(err)) {}
  Result(ErrorCode c, std::string m) : err_(c, std::move(m)) {}

  static Result ok() { return Result(); }
  static Result err(ErrorCode c, std::string m) { return Result(ErrorInfo(c, std::move(m))); }
  static Result err(const ErrorInfo& e) { return Result(e); }

  bool has_value() const noexcept { return err_.code == ErrorCode::ok; }
  bool is_ok() const noexcept { return has_value(); }
  bool is_error() const noexcept { return !has_value(); }
  explicit operator bool() const noexcept { return has_value(); }
  const ErrorInfo& error() const noexcept { return err_; }

 private:
  ErrorInfo err_;
};

// Helper: construct a Result<T> from a value.
template <typename T>
inline Result<T> make_ok(T val) { return Result<T>::ok(std::move(val)); }

inline Result<void> make_ok_void() { return Result<void>::ok(); }
inline Result<void> make_err(ErrorCode c, std::string m) { return Result<void>::err(c, std::move(m)); }

}  // namespace prefillfabric
