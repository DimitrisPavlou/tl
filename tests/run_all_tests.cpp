// tests/run_all_tests.cpp — Single entry-point that runs every test suite.
//
// To add a new suite:
//   1. Create tests/test_myfeature.cpp with a function:
//          void run_myfeature_tests(tl::TestContext& ctx);
//   2. Forward-declare it below and call it in main().
//
// Build & run (from project root):
//   g++ -std=c++17 -O2 tests/run_all_tests.cpp -o run_tests && ./run_tests

#include "test.hpp"
#include "../tl/tl.hpp"
#include <iostream>

// ── Forward declarations (one per test file) ──────────────────────────────────
void run_tensor_core_tests     (tl::TestContext& ctx);
void run_linalg_tests          (tl::TestContext& ctx);
void run_functional_tests      (tl::TestContext& ctx);
void run_dot_product_tests     (tl::TestContext& ctx);
void run_elementary_tests      (tl::TestContext& ctx);
void run_broadcasting_tests    (tl::TestContext& ctx);
void run_view_semantics_tests  (tl::TestContext& ctx);
void run_ownership_tests       (tl::TestContext& ctx);
void run_phase3_tests          (tl::TestContext& ctx);
void run_phase4_tests          (tl::TestContext& ctx);
void run_phase5_tests          (tl::TestContext& ctx);

// ── Include test translation units ───────────────────────────────────────────
// (Each file defines the function declared above.)
#include "test_tensor_core.cpp"
#include "test_linalg.cpp"
#include "test_functional.cpp"
#include "test_dot_product.cpp"
#include "test_elementary_functions.cpp"
#include "test_broadcasting.cpp"
#include "test_view_semantics.cpp"
#include "test_ownership.cpp"
#include "test_phase3.cpp"
#include "test_phase4.cpp"
#include "test_phase5.cpp"


// ── Runner ────────────────────────────────────────────────────────────────────
int main() {
    using namespace tl::detail;

    std::cout << BOLD << CYAN
              << "╔══════════════════════════════════════╗\n"
              << "║     Tensor Library — Test Runner     ║\n"
              << "╚══════════════════════════════════════╝"
              << RESET << "\n";

    tl::TestContext ctx;

    run_tensor_core_tests(ctx);
    run_linalg_tests(ctx);
    run_functional_tests(ctx);
    run_dot_product_tests(ctx);
    run_elementary_tests(ctx);
    run_broadcasting_tests(ctx);
    run_view_semantics_tests(ctx);
    run_ownership_tests(ctx);
    run_phase3_tests(ctx);
    run_phase4_tests(ctx);
    run_phase5_tests(ctx);

    return ctx.summary();   // exits 0 if all pass, 1 if any failed
}
