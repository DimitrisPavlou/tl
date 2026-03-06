// tests/test_functional.cpp — Tests for tl::functional (math, activations, clamping)
#include "test.hpp"
#include "../tl/tl.hpp"
#include <cmath>

void run_functional_tests(tl::TestContext& ctx) {

    // ── Elementary functions ──────────────────────────────────────────────────
    SUITE(ctx, "Functional — exp / log / sqrt");

    {
        tl::Tensor<float> t({4}, {0.0f, 1.0f, 2.0f, 4.0f});

        auto e = tl::functional::exp(t);
        CHECK_NEAR(ctx, e.data[0], 1.0f,    1e-5f);
        CHECK_NEAR(ctx, e.data[1], 2.71828f, 1e-4f);

        auto l = tl::functional::log(tl::Tensor<float>({3}, {1.0f, std::exp(1.0f), std::exp(2.0f)}));
        CHECK_NEAR(ctx, l.data[0], 0.0f, 1e-6f);
        CHECK_NEAR(ctx, l.data[1], 1.0f, 1e-6f);
        CHECK_NEAR(ctx, l.data[2], 2.0f, 1e-6f);

        auto s = tl::functional::sqrt(tl::Tensor<float>({3}, {1.0f, 4.0f, 9.0f}));
        CHECK_NEAR(ctx, s.data[0], 1.0f, 1e-6f);
        CHECK_NEAR(ctx, s.data[1], 2.0f, 1e-6f);
        CHECK_NEAR(ctx, s.data[2], 3.0f, 1e-6f);
    }

    // Double precision must be preserved (was always downcast to float before the fix)
    {
        tl::Tensor<double> td({2}, {0.0, 1.0});
        auto ed = tl::functional::exp(td);
        static_assert(std::is_same_v<decltype(ed)::value_type, double>,
                      "exp(Tensor<double>) must return Tensor<double>");
        CHECK_NEAR(ctx, ed.data[1], std::exp(1.0), 1e-12);
    }

    // ── Trigonometric ─────────────────────────────────────────────────────────
    SUITE(ctx, "Functional — sin / cos / tan");

    {
        const float pi = 3.14159265358979f;
        tl::Tensor<float> t({3}, {0.0f, pi / 2.0f, pi});

        auto sn = tl::functional::sin(t);
        CHECK_NEAR(ctx, sn.data[0], 0.0f,  1e-6f);
        CHECK_NEAR(ctx, sn.data[1], 1.0f,  1e-6f);
        CHECK_NEAR(ctx, sn.data[2], 0.0f,  1e-5f);

        auto cs = tl::functional::cos(t);
        CHECK_NEAR(ctx, cs.data[0],  1.0f, 1e-6f);
        CHECK_NEAR(ctx, cs.data[1],  0.0f, 1e-6f);
        CHECK_NEAR(ctx, cs.data[2], -1.0f, 1e-5f);
    }

    // ── Abs, ceil, floor, round ───────────────────────────────────────────────
    SUITE(ctx, "Functional — abs / ceil / floor / round");

    {
        tl::Tensor<float> t({4}, {-5.5f, 3.2f, -1.0f, 2.7f});

        auto a = tl::functional::abs(t);
        CHECK_EQ(ctx, a.data[0], 5.5f);
        CHECK_EQ(ctx, a.data[1], 3.2f);

        auto c = tl::functional::ceil(t);
        CHECK_EQ(ctx, c.data[0], -5.0f);
        CHECK_EQ(ctx, c.data[3],  3.0f);

        auto f = tl::functional::floor(t);
        CHECK_EQ(ctx, f.data[0], -6.0f);
        CHECK_EQ(ctx, f.data[3],  2.0f);

        auto r = tl::functional::round(t);
        CHECK_EQ(ctx, r.data[0], -6.0f);
        CHECK_EQ(ctx, r.data[3],  3.0f);
    }

    // ── Activation functions ──────────────────────────────────────────────────
    SUITE(ctx, "Functional — ReLU / Leaky ReLU / Sigmoid");

    {
        tl::Tensor<float> t({4}, {-2.0f, 0.0f, 1.0f, 3.0f});

        auto relu = tl::functional::relu(t);
        CHECK_EQ(ctx, relu.data[0], 0.0f);
        CHECK_EQ(ctx, relu.data[1], 0.0f);
        CHECK_EQ(ctx, relu.data[2], 1.0f);
        CHECK_EQ(ctx, relu.data[3], 3.0f);

        auto lrelu = tl::functional::leaky_relu(t, 0.1f);
        CHECK_NEAR(ctx, lrelu.data[0], -0.2f, 1e-6f);   // alpha * -2.0
        CHECK_EQ(ctx,   lrelu.data[2],  1.0f);

        auto sig = tl::functional::sigmoid(tl::Tensor<float>({2}, {0.0f, 100.0f}));
        CHECK_NEAR(ctx, sig.data[0], 0.5f,  1e-6f);
        CHECK_NEAR(ctx, sig.data[1], 1.0f,  1e-5f);
    }

    // ── clip ─────────────────────────────────────────────────────────────────
    SUITE(ctx, "Functional — clip");

    {
        tl::Tensor<float> t({5}, {-5.0f, 1.0f, 5.0f, 8.0f, 15.0f});
        auto c = tl::functional::clip(t, 0.0f, 10.0f);
        CHECK_EQ(ctx, c.data[0],  0.0f);   // clamped to min
        CHECK_EQ(ctx, c.data[2],  5.0f);   // unchanged
        CHECK_EQ(ctx, c.data[4], 10.0f);   // clamped to max
    }

    // ── power / square ────────────────────────────────────────────────────────
    SUITE(ctx, "Functional — power / square");

    {
        tl::Tensor<float> t({3}, {2.0f, 3.0f, 4.0f});

        auto sq = tl::functional::square(t);
        CHECK_EQ(ctx, sq.data[0],  4.0f);
        CHECK_EQ(ctx, sq.data[1],  9.0f);

        auto pw = tl::functional::power(t, 3.0f);
        CHECK_NEAR(ctx, pw.data[0],  8.0f, 1e-5f);
        CHECK_NEAR(ctx, pw.data[2], 64.0f, 1e-4f);
    }

    // ── sqrt of negatives gives NaN ───────────────────────────────────────────
    SUITE(ctx, "Functional — sqrt NaN behaviour");

    {
        tl::Tensor<float> t({2}, {-1.0f, -4.0f});
        auto s = tl::functional::sqrt(t);
        CHECK(ctx, std::isnan(s.data[0]));
        CHECK(ctx, std::isnan(s.data[1]));
    }
}