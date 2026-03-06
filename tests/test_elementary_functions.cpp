// tests/test_elementary_functions.cpp — Tests for tl::functional elementary functions
// (This file focuses on edge cases and numerical accuracy)
#include "test.hpp"
#include "../tl/tl.hpp"
#include <cmath>

void run_elementary_tests(tl::TestContext& ctx) {

    // ── exp edge cases ────────────────────────────────────────────────────────
    SUITE(ctx, "Elementary — exp");

    {
        tl::Tensor<float> t({3}, {0.0f, 1.0f, -1.0f});
        auto e = tl::functional::exp(t);
        CHECK_NEAR(ctx, e.data[0], 1.0f,         1e-6f);
        CHECK_NEAR(ctx, e.data[1], 2.71828182f,  1e-4f);
        CHECK_NEAR(ctx, e.data[2], 0.36787944f,  1e-5f);
    }

    // ── relu ──────────────────────────────────────────────────────────────────
    SUITE(ctx, "Elementary — relu");

    {
        tl::Tensor<float> t({4}, {-10.0f, 0.0f, 5.0f, -2.0f});
        auto r = tl::functional::relu(t);
        CHECK_EQ(ctx, r.data[0], 0.0f);
        CHECK_EQ(ctx, r.data[1], 0.0f);
        CHECK_EQ(ctx, r.data[2], 5.0f);
        CHECK_EQ(ctx, r.data[3], 0.0f);
    }

    // ── abs ───────────────────────────────────────────────────────────────────
    SUITE(ctx, "Elementary — abs");

    {
        tl::Tensor<float> t({4}, {-5.5f, 3.2f, 0.0f, -0.001f});
        auto a = tl::functional::abs(t);
        CHECK_NEAR(ctx, a.data[0], 5.5f,   1e-6f);
        CHECK_NEAR(ctx, a.data[1], 3.2f,   1e-6f);
        CHECK_EQ(ctx,   a.data[2], 0.0f);
        CHECK_NEAR(ctx, a.data[3], 0.001f, 1e-8f);
    }

    // ── clip ──────────────────────────────────────────────────────────────────
    SUITE(ctx, "Elementary — clip");

    {
        tl::Tensor<float> t({3}, {1.0f, 5.0f, 10.0f});
        auto c = tl::functional::clip(t, 2.0f, 8.0f);
        CHECK_EQ(ctx, c.data[0], 2.0f);  // clamped to min
        CHECK_EQ(ctx, c.data[1], 5.0f);  // unchanged
        CHECK_EQ(ctx, c.data[2], 8.0f);  // clamped to max
    }

    // ── tanh ──────────────────────────────────────────────────────────────────
    SUITE(ctx, "Elementary — tanh");

    {
        tl::Tensor<float> t({3}, {0.0f, 1.0f, -1.0f});
        auto th = tl::functional::tanh(t);
        CHECK_NEAR(ctx, th.data[0],  0.0f,        1e-6f);
        CHECK_NEAR(ctx, th.data[1],  0.76159415f, 1e-5f);
        CHECK_NEAR(ctx, th.data[2], -0.76159415f, 1e-5f);
    }

    // ── integer input → float output ─────────────────────────────────────────
    SUITE(ctx, "Elementary — integer input promotion");

    {
        tl::Tensor<int> ti({3}, {0, 1, 4});
        auto sq = tl::functional::sqrt(ti);
        static_assert(std::is_same_v<decltype(sq)::value_type, float>,
                      "sqrt(Tensor<int>) must return Tensor<float>");
        CHECK_NEAR(ctx, sq.data[1], 1.0f, 1e-6f);
        CHECK_NEAR(ctx, sq.data[2], 2.0f, 1e-6f);
    }
}