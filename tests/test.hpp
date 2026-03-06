// tests/test.hpp — Lightweight single-header test framework for the tensor library
//
// Usage:
//   Each test file defines:
//       void run_my_tests(tl::TestContext& ctx);
//   and uses the macros CHECK / CHECK_NEAR / CHECK_THROWS inside it.
//   run_all_tests.cpp includes every header and calls all suites.
//
// Build command (from project root):
//   g++ -std=c++17 tests/run_all_tests.cpp -o run_tests && ./run_tests

#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace tl {

// ─── ANSI colour helpers (disabled automatically on non-tty if needed) ────────
namespace detail {
    inline const char* GREEN  = "\033[32m";
    inline const char* RED    = "\033[31m";
    inline const char* YELLOW = "\033[33m";
    inline const char* BOLD   = "\033[1m";
    inline const char* RESET  = "\033[0m";
    inline const char* CYAN   = "\033[36m";
    inline const char* GREY   = "\033[90m";
}

// ─── TestContext: tracks results across the entire run ───────────────────────
struct TestContext {
    int passed  = 0;
    int failed  = 0;
    std::string current_suite;
    std::string current_test;

    void begin_suite(const std::string& name) {
        current_suite = name;
        std::cout << "\n" << detail::BOLD << detail::CYAN
                  << "[ Suite ] " << name
                  << detail::RESET << "\n";
    }

    void begin_test(const std::string& name) {
        current_test = name;
    }

    void record_pass(const std::string& expr, const char* file, int line) {
        ++passed;
        std::cout << "  " << detail::GREEN << "  PASS " << detail::RESET
                  << detail::GREY << expr << detail::RESET << "\n";
        (void)file; (void)line;
    }

    void record_fail(const std::string& expr, const char* file, int line,
                     const std::string& note = "") {
        ++failed;
        std::cout << "  " << detail::RED << "  FAIL " << detail::RESET
                  << detail::BOLD << expr << detail::RESET;
        if (!note.empty()) std::cout << "  →  " << detail::RED << note << detail::RESET;
        std::cout << "\n"
                  << detail::GREY << "       at " << file << ":" << line
                  << detail::RESET << "\n";
    }

    // Print the grand total and return 0 (all pass) or 1 (any fail)
    int summary() const {
        int total = passed + failed;
        std::cout << "\n" << detail::BOLD
                  << "─────────────────────────────────────\n"
                  << " Results: " << total << " checks";
        if (failed == 0) {
            std::cout << detail::GREEN << "  All passed ✓" << detail::RESET << "\n";
        } else {
            std::cout << "\n"
                      << detail::GREEN << "  Passed: " << passed << detail::RESET << "\n"
                      << detail::RED   << "  Failed: " << failed << detail::RESET << "\n";
        }
        std::cout << detail::BOLD << "─────────────────────────────────────"
                  << detail::RESET << "\n";
        return failed > 0 ? 1 : 0;
    }
};

} // namespace tl


// ─── Macros ───────────────────────────────────────────────────────────────────

// CHECK(expr) — fails if expr is false
#define CHECK(ctx, expr)                                                       \
    do {                                                                       \
        if (expr) { (ctx).record_pass(#expr, __FILE__, __LINE__); }            \
        else       { (ctx).record_fail(#expr, __FILE__, __LINE__); }           \
    } while(0)

// CHECK_EQ(a, b) — fails if a != b, prints both values
#define CHECK_EQ(ctx, a, b)                                                    \
    do {                                                                       \
        auto _a = (a); auto _b = (b);                                          \
        if (_a == _b) { (ctx).record_pass(#a " == " #b, __FILE__, __LINE__); } \
        else {                                                                  \
            std::ostringstream _msg;                                            \
            _msg << "got " << _a << ", expected " << _b;                        \
            (ctx).record_fail(#a " == " #b, __FILE__, __LINE__, _msg.str());    \
        }                                                                       \
    } while(0)

// CHECK_NEAR(a, b, eps) — fails if |a - b| >= eps
#define CHECK_NEAR(ctx, a, b, eps)                                             \
    do {                                                                       \
        auto _a = (a); auto _b = (b); auto _e = (eps);                         \
        double _diff = std::abs(static_cast<double>(_a) - static_cast<double>(_b)); \
        if (_diff < static_cast<double>(_e)) {                                  \
            (ctx).record_pass(#a " ≈ " #b, __FILE__, __LINE__);                \
        } else {                                                                \
            std::ostringstream _msg;                                            \
            _msg << "|" << _a << " - " << _b << "| = " << _diff               \
                 << " (eps=" << _e << ")";                                      \
            (ctx).record_fail(#a " ≈ " #b, __FILE__, __LINE__, _msg.str());    \
        }                                                                       \
    } while(0)

// CHECK_THROWS(ctx, ExceptionType, expr) — fails if expr does NOT throw ExceptionType
#define CHECK_THROWS(ctx, ExcType, expr)                                       \
    do {                                                                       \
        bool _threw = false;                                                   \
        try { (void)(expr); }                                                  \
        catch (const ExcType&) { _threw = true; }                              \
        catch (...) {}                                                          \
        if (_threw) { (ctx).record_pass("throws " #ExcType ": " #expr, __FILE__, __LINE__); } \
        else        { (ctx).record_fail("throws " #ExcType ": " #expr, __FILE__, __LINE__, "no exception thrown"); } \
    } while(0)

// CHECK_NO_THROW(ctx, expr) — fails if expr throws anything
#define CHECK_NO_THROW(ctx, expr)                                              \
    do {                                                                       \
        bool _threw = false;                                                   \
        try { (void)(expr); } catch (...) { _threw = true; }                   \
        if (!_threw) { (ctx).record_pass("no throw: " #expr, __FILE__, __LINE__); } \
        else         { (ctx).record_fail("no throw: " #expr, __FILE__, __LINE__, "unexpected exception"); } \
    } while(0)

// SUITE(ctx, name) — marks a new named group within a test file
#define SUITE(ctx, name) (ctx).begin_suite(name)

// ─── sstream needed by CHECK_EQ / CHECK_NEAR ─────────────────────────────────
#include <sstream>
