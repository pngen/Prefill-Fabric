// Prefill Fabric - OS process helper (Windows) for multiprocess tests.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <string>
#include <vector>
#include <cstdio>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace pf_test {

// RAII wrapper over a Win32 process with stdin/stdout as a pipe (in), and the
// ability to terminate and reap a child (an actual OS process).
class Proc {
 public:
  Proc() = default;
  ~Proc() { kill(); }
  Proc(const Proc&) = delete;
  Proc& operator=(const Proc&) = delete;

  // Spawn an executable with args. Optionally capture stdout via a pipe.
  bool spawn(const std::string& exe, const std::string& args, bool capture_stdout) {
    std::string cmd = "\"" + exe + "\" " + args;
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    HANDLE rd = nullptr, wr = nullptr;
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE; sa.lpSecurityDescriptor = nullptr;
    if (capture_stdout) {
      if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
      SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
      si.dwFlags |= STARTF_USESTDHANDLES;
      si.hStdOutput = wr;
      si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    }
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));
    std::vector<char> cmdline(cmd.begin(), cmd.end());
    cmdline.push_back('\0');
    if (!CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, capture_stdout ? TRUE : FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
      if (wr) CloseHandle(wr); if (rd) CloseHandle(rd);
      return false;
    }
    h_ = pi.hProcess;
    if (pi.hThread) CloseHandle(pi.hThread);
    if (wr) CloseHandle(wr);
    in_ = rd;
    return h_ != nullptr;
  }

  // Terminate the process (actual OS termination).
  void kill() {
    if (h_) { TerminateProcess(h_, 0); WaitForSingleObject(h_, 5000); }
    CloseHandle(h_); h_ = nullptr;
    if (in_) { CloseHandle(in_); in_ = nullptr; }
  }

  bool running() const {
    if (!h_) return false;
    DWORD code = 0;
    return GetExitCodeProcess(h_, &code) && code == STILL_ACTIVE;
  }

  // Read up to n bytes from the child stdout (non-blocking-ish with timeout).
  std::string read_stdout(int max_bytes = 4096) {
    std::string out;
    if (!in_) return out;
    DWORD avail = 0;
    PeekNamedPipe(in_, nullptr, 0, nullptr, &avail, nullptr);
    if (avail == 0) return out;
    if (avail > (DWORD)max_bytes) avail = (DWORD)max_bytes;
    std::vector<char> buf(avail);
    DWORD read = 0;
    if (ReadFile(in_, buf.data(), avail, &read, nullptr)) out.assign(buf.data(), read);
    return out;
  }

 private:
  HANDLE h_ = nullptr;
  HANDLE in_ = nullptr;
};

}  // namespace pf_test
