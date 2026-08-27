// Prefill Fabric - TCP socket layer (Winsock2).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "prefillfabric/result.hpp"

#ifdef _WIN32
#define NOMINMAX
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET pf_socket_t;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
typedef int pf_socket_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close
#endif

namespace prefillfabric {

// Initialize the socket subsystem (calls WSAStartup on Windows). Safe to call
// multiple times.
Result<void> net_init();
void net_cleanup();

// RAII socket wrapper. Movable, not copyable.
class Socket {
 public:
  Socket() : s_(INVALID_SOCKET) {}
  explicit Socket(pf_socket_t s) : s_(s) {}
  ~Socket();
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;
  Socket(Socket&& o) noexcept;
  Socket& operator=(Socket&& o) noexcept;

  bool valid() const noexcept { return s_ != INVALID_SOCKET; }
  pf_socket_t native() const noexcept { return s_; }
  void close();

  // Reliable send of all bytes.
  Result<void> send_all(const std::uint8_t* data, std::size_t n);
  // Reliable receive of exactly n bytes.
  Result<void> recv_exact(std::uint8_t* data, std::size_t n);

  static Result<Socket> connect(const std::string& host, std::uint16_t port);
  // Send a full framed packet (length prefix + body).
  Result<void> send_frame(const std::uint8_t* body, std::size_t n);
  // Receive one full framed packet (length prefix + body).
  Result<std::vector<std::uint8_t>> recv_frame();

 private:
  pf_socket_t s_;
};

// Listening socket.
class Listener {
 public:
  explicit Listener(std::uint16_t port = 0);
  ~Listener();
  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;
  Result<void> start();
  Result<Socket> accept();
  std::uint16_t port() const noexcept { return port_; }
  void close();

 private:
  pf_socket_t s_;
  std::uint16_t port_;
};

}  // namespace prefillfabric
