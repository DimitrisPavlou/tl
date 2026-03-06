// tests/test_dot_product.cpp — Tests for tl::dot and tl::sum/mean/max/min
#include "test.hpp"
#include "../tl/tl.hpp"
#include <cmath>
#include <stdexcept>

void run_dot_product_tests(tl::TestContext& ctx) {

    // ── dot product ───────────────────────────────────────────────────────────
    SUITE(ctx, "Utils — dot product");

    {
        tl::Tensor<double> a({3}, {1.5, 2.5, 3.5});
        tl::Tensor<double> b({3}, {2.0, 3.0, 4.0});
        double result = tl::dot(a, b);   // 1.5*2 + 2.5*3 + 3.5*4 = 24.5
        CHECK_NEAR(ctx, result, 24.5, 1e-10);
    }

    {
        tl::Tensor<int> a({3}, {1, 2, 3});
        tl::Tensor<int> b({3}, {4, 5, 6});
        int result = tl::dot(a, b);      // 1*4 + 2*5 + 3*6 = 32
        CHECK_EQ(ctx, result, 32);
    }

    // Length mismatch must throw
    CHECK_THROWS(ctx, std::runtime_error, ({
        tl::Tensor<float> a({3}, {1,2,3});
        tl::Tensor<float> b({4}, {1,2,3,4});
        tl::dot(a, b);
    }));

    // Non-1D must throw
    CHECK_THROWS(ctx, std::runtime_error, ({
        tl::Tensor<float> a({2,2}, {1,2,3,4});
        tl::Tensor<float> b({2,2}, {1,2,3,4});
        tl::dot(a, b);
    }));

    // ── sum / mean / max / min ────────────────────────────────────────────────
    SUITE(ctx, "Utils — reductions");

    {
        tl::Tensor<float> t({4}, {1.0f, 2.0f, 3.0f, 4.0f});
        CHECK_NEAR(ctx, tl::sum(t),   10.0f, 1e-6f);
        CHECK_NEAR(ctx, tl::mean(t),   2.5f, 1e-6f);
        CHECK_EQ(ctx,   tl::max(t),    4.0f);
        CHECK_EQ(ctx,   tl::min(t),    1.0f);
    }

    // mean preserves double precision
    {
        tl::Tensor<double> td({2}, {1.0, 3.0});
        double m = tl::mean(td);
        static_assert(std::is_same_v<decltype(m), double>,
                      "mean(Tensor<double>) must return double");
        CHECK_NEAR(ctx, m, 2.0, 1e-12);
    }

    // ── factory functions ─────────────────────────────────────────────────────
    SUITE(ctx, "Utils — zeros / ones / full / reshape");

    {
        auto z = tl::zeros<float>({2, 3});
        CHECK_EQ(ctx, z.data[0], 0.0f);
        CHECK_EQ(ctx, z.data[5], 0.0f);

        auto o = tl::ones<int>({3});
        CHECK_EQ(ctx, o.data[0], 1);
        CHECK_EQ(ctx, o.data[2], 1);

        auto f = tl::full<int>({2, 3}, 7);
        CHECK_EQ(ctx, f.data[0], 7);
        CHECK_EQ(ctx, f.data[5], 7);
    }

    {   // reshape
        tl::Tensor<int> t({6}, {1, 2, 3, 4, 5, 6});
        auto r = tl::reshape(t, {2, 3});
        CHECK_EQ(ctx, r.shape[0], 2u);
        CHECK_EQ(ctx, r.shape[1], 3u);
        CHECK_EQ(ctx, r.data[3],  4);

        // Bad reshape must throw
        CHECK_THROWS(ctx, std::runtime_error, tl::reshape(t, {2, 4}));
    }
}