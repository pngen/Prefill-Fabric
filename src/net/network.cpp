// Prefill Fabric - socket layer implementation (Winsock2).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/network.hpp"
#include <cstring>

namespace prefillfabric {

Result<void> net_init() {
#ifdef _WIN32
  WSADATA d;
  if (WSAStartup(MAKEWORD(2, 2), &d) != 0)
    return Result<void>::err(ErrorCode::io_failure, "WSAStartup failed");
  return Result<void>::ok();
#else
  return Result<void>::ok();
#endif
}

void net_cleanup() {
#ifdef _WIN32
  WSACleanup();
#endif
}

Socket::~Socket() { close(); }
Socket::Socket(Socket&& o) noexcept : s_(o.s_) { o.s_ = INVALID_SOCKET; }
Socket& Socket::operator=(Socket&& o) noexcept {
  if (this != &o) { close(); s_ = o.s_; o.s_ = INVALID_SOCKET; }
  return *this;
}

void Socket::close() {
  if (s_ != INVALID_SOCKET) {
#ifdef _WIN32
    ::closesocket(s_);
#else
    ::close(s_);
#endif
    s_ = INVALID_SOCKET;
  }
}

Result<void> Socket::send_all(const std::uint8_t* data, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
    const int sent = ::send(s_, reinterpret_cast<const char*>(data + off),
                            static_cast<int>(n - off), 0);
    if (sent == SOCKET_ERROR)
      return Result<void>::err(ErrorCode::io_failure, "socket send failed");
    if (sent == 0)
      return Result<void>::err(ErrorCode::io_failure, "socket closed during send");
    off += static_cast<std::size_t>(sent);
  }
  return Result<void>::ok();
}

Result<void> Socket::recv_exact(std::uint8_t* data, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
    const int r = ::recv(s_, reinterpret_cast<char*>(data + off),
                         static_cast<int>(n - off), 0);
    if (r == SOCKET_ERROR)
      return Result<void>::err(ErrorCode::io_failure, "socket recv failed");
    if (r == 0)
      return Result<void>::err(ErrorCode::io_failure, "peer closed the connection");
    off += static_cast<std::size_t>(r);
  }
  return Result<void>::ok();
}

Result<Socket> Socket::connect(const std::string& host, std::uint16_t port) {
  const pf_socket_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET)
    return Result<Socket>::err(ErrorCode::io_failure, "socket() failed");
  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(host.c_str());
  if (addr.sin_addr.s_addr == INADDR_NONE) {
    // Resolve hostname.
    hostent* he = ::gethostbyname(host.c_str());
    if (he) std::memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    else { ::closesocket(s == INVALID_SOCKET ? -1 : s); return Result<Socket>::err(ErrorCode::io_failure, "bad host"); }
  }
  if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
    ::closesocket(s);
    return Result<Socket>::err(ErrorCode::io_failure, "connect failed");
  }
  return Result<Socket>::ok(Socket(s));
}

Result<void> Socket::send_frame(const std::uint8_t* body, std::size_t n) {
  if (n > (64u * 1024u * 1024u))
    return Result<void>::err(ErrorCode::oversized_frame, "frame too large");
  std::uint32_t len = static_cast<std::uint32_t>(n);
  // Little-endian 4-byte length prefix.
  std::uint8_t hdr[4] = {
      static_cast<std::uint8_t>(len & 0xffu),
      static_cast<std::uint8_t>((len >> 8) & 0xffu),
      static_cast<std::uint8_t>((len >> 16) & 0xffu),
      static_cast<std::uint8_t>((len >> 24) & 0xffu)};
  auto r = send_all(hdr, 4);
  if (!r) return r;
  return send_all(body, n);
}

Result<std::vector<std::uint8_t>> Socket::recv_frame() {
  std::uint8_t hdr[4];
  auto r = recv_exact(hdr, 4);
  if (!r) return Result<std::vector<std::uint8_t>>::err(r.error());
  std::uint32_t len = static_cast<std::uint32_t>(hdr[0]) |
      (static_cast<std::uint32_t>(hdr[1]) << 8) |
      (static_cast<std::uint32_t>(hdr[2]) << 16) |
      (static_cast<std::uint32_t>(hdr[3]) << 24);
  if (len > (64u * 1024u * 1024u))
    return Result<std::vector<std::uint8_t>>::err(ErrorCode::oversized_frame, "incoming frame too large");
  // Zero-length frames are invalid for this protocol (every frame has a version+type).
  if (len < 2)
    return Result<std::vector<std::uint8_t>>::err(ErrorCode::malformed_frame, "frame too short (zero or one byte)");
  std::vector<std::uint8_t> body(static_cast<std::size_t>(len));
  auto rr = recv_exact(body.data(), body.size());
  if (!rr) return Result<std::vector<std::uint8_t>>::err(rr.error());
  return Result<std::vector<std::uint8_t>>::ok(std::move(body));
}

Listener::Listener(std::uint16_t port) : s_(INVALID_SOCKET), port_(port) {}
Listener::~Listener() { close(); }

Result<void> Listener::start() {
  const pf_socket_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET)
    return Result<void>::err(ErrorCode::io_failure, "socket() failed");
  const int opt = 1;
  ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port_);
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
    ::closesocket(s);
    return Result<void>::err(ErrorCode::io_failure, "bind failed");
  }
  // Discover the actual ephemeral port if port 0 was requested.
  sockaddr_in bound;
  int blen = static_cast<int>(sizeof(bound));
  ::getsockname(s, reinterpret_cast<sockaddr*>(&bound), &blen);
  port_ = ntohs(bound.sin_port);
  if (::listen(s, SOMAXCONN) == SOCKET_ERROR) {
    ::closesocket(s);
    return Result<void>::err(ErrorCode::io_failure, "listen failed");
  }
  s_ = s;
  return Result<void>::ok();
}

Result<Socket> Listener::accept() {
  if (s_ == INVALID_SOCKET) return Result<Socket>::err(ErrorCode::io_failure, "listener not started");
  const pf_socket_t c = ::accept(s_, nullptr, nullptr);
  if (c == INVALID_SOCKET) return Result<Socket>::err(ErrorCode::io_failure, "accept failed");
  return Result<Socket>::ok(Socket(c));
}

void Listener::close() {
  if (s_ != INVALID_SOCKET) { ::closesocket(s_); s_ = INVALID_SOCKET; }
}

}  // namespace prefillfabric
