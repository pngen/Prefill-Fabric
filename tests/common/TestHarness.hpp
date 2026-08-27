// Prefill Fabric - minimal, dependency-free test harness.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace pf_test {

inline int& fail_counter() { static int c = 0; return c; }
inline int& check_counter() { static int c = 0; return c; }

struct TestRegistry {
  struct Entry { const char* name; void (*fn)(); };
  static std::vector<Entry>& entries() { static std::vector<Entry> e; return e; }
};

struct TestRegistrar {
  TestRegistrar(const char* name, void (*fn)()) { TestRegistry::entries().push_back({name, fn}); }
};

inline int run_all() {
  int failures = 0;
  for (const auto& e : TestRegistry::entries()) {
    const int before = fail_counter();
    try {
      e.fn();
    } catch (const std::exception& ex) {
      std::cerr << "[EXCEPTION] " << e.name << ": " << ex.what() << "\n";
      ++fail_counter();
    } catch (...) {
      std::cerr << "[EXCEPTION] " << e.name << ": unknown\n";
      ++fail_counter();
    }
    const int after = fail_counter();
    if (after == before) std::printf("[PASS] %s\n", e.name);
    else { ++failures; std::printf("[FAIL] %s (%d check(s) failed)\n", e.name, after - before); }
  }
  std::printf("\n%d test(s), %d failure(s)\n",
              static_cast<int>(TestRegistry::entries().size()), failures);
  std::printf("total checks: %d, failed checks: %d\n", check_counter(), fail_counter());
  return failures == 0 && fail_counter() == 0 ? 0 : 1;
}

}  // namespace pf_test

#define TEST(name) \
  static void test_##name(); \
  static ::pf_test::TestRegistrar pf_reg_##name(#name, &test_##name); \
  static void test_##name()

#define CHECK(cond) \
  do { ++::pf_test::check_counter(); if (!(cond)) { ++::pf_test::fail_counter(); \
       std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__ << "  CHECK(" << #cond << ")\n"; } } while (0)

#define CHECK_EQ(a, b) \
  do { ++::pf_test::check_counter(); auto pf_a = (a); auto pf_b = (b); \
       if (!(pf_a == pf_b)) { ++::pf_test::fail_counter(); \
         std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__ << "  CHECK_EQ( " << #a << ", " << #b << " )\n"; } } while (0)

#define CHECK_GE(a, b) \
  do { ++::pf_test::check_counter(); auto pf_a = (a); auto pf_b = (b); \
       if (!(pf_a >= pf_b)) { ++::pf_test::fail_counter(); \
         std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__ << "  CHECK_GE( " << #a << ", " << #b << " )\n"; } } while (0)

#define CHECK_THAT(expr) CHECK(expr)
