// tests/test_broadcasting.cpp — Tests for NumPy-style broadcasting
#include "test.hpp"
#include "../tl/tl.hpp"
#include <stdexcept>

void run_broadcasting_tests(tl::TestContext& ctx) {

    // ── Same-shape fast path is unchanged ────────────────────────────────────
    SUITE(ctx, "Broadcasting — same-shape fast path");

    {
        tl::Tensor<float> a({3}, {1.0f, 2.0f, 3.0f});
        tl::Tensor<float> b({3}, {10.0f, 20.0f, 30.0f});
        auto c = a + b;
        CHECK_EQ(ctx, c.data[0], 11.0f);
        CHECK_EQ(ctx, c.data[1], 22.0f);
        CHECK_EQ(ctx, c.data[2], 33.0f);
    }

    // ── Scalar (1,) + Vector (3,) ─────────────────────────────────────────────
    SUITE(ctx, "Broadcasting — scalar + vector");

    {
        tl::Tensor<float> scalar({1}, {5.0f});
        tl::Tensor<float> vec({3}, {1.0f, 2.0f, 3.0f});
        auto r = scalar + vec;
        CHECK_EQ(ctx, r.shape[0], 3u);
        CHECK_EQ(ctx, r.data[0], 6.0f);
        CHECK_EQ(ctx, r.data[1], 7.0f);
        CHECK_EQ(ctx, r.data[2], 8.0f);
    }

    // ── Vector (3,) broadcast to Matrix (2, 3) ───────────────────────────────
    // This is the classic "add bias to every row" operation.
    SUITE(ctx, "Broadcasting — vector + matrix (bias addition)");

    {
        tl::Tensor<float> bias({3}, {1.0f, 2.0f, 3.0f});
        tl::Tensor<float> matrix({2, 3}, {10.0f, 20.0f, 30.0f,
                                          40.0f, 50.0f, 60.0f});
        auto r = matrix + bias;

        // Output shape must be {2, 3}
        CHECK_EQ(ctx, r.shape[0], 2u);
        CHECK_EQ(ctx, r.shape[1], 3u);

        // Row 0: [10+1, 20+2, 30+3] = [11, 22, 33]
        CHECK_EQ(ctx, r.data[0], 11.0f);
        CHECK_EQ(ctx, r.data[1], 22.0f);
        CHECK_EQ(ctx, r.data[2], 33.0f);

        // Row 1: [40+1, 50+2, 60+3] = [41, 52, 63]
        CHECK_EQ(ctx, r.data[3], 41.0f);
        CHECK_EQ(ctx, r.data[4], 52.0f);
        CHECK_EQ(ctx, r.data[5], 63.0f);
    }

    // ── Column vector (2, 1) broadcast to Matrix (2, 3) ──────────────────────
    // This is the classic "add a per-row scalar" operation.
    SUITE(ctx, "Broadcasting — column vector * matrix");

    {
        tl::Tensor<float> col({2, 1}, {2.0f, 3.0f});
        tl::Tensor<float> mat({2, 3}, {1.0f, 2.0f, 3.0f,
                                       4.0f, 5.0f, 6.0f});
        auto r = mat * col;

        CHECK_EQ(ctx, r.shape[0], 2u);
        CHECK_EQ(ctx, r.shape[1], 3u);

        // Row 0 * 2: [2, 4, 6]
        CHECK_EQ(ctx, r.data[0], 2.0f);
        CHECK_EQ(ctx, r.data[1], 4.0f);
        CHECK_EQ(ctx, r.data[2], 6.0f);

        // Row 1 * 3: [12, 15, 18]
        CHECK_EQ(ctx, r.data[3], 12.0f);
        CHECK_EQ(ctx, r.data[4], 15.0f);
        CHECK_EQ(ctx, r.data[5], 18.0f);
    }

    // ── All operators work with broadcasting ──────────────────────────────────
    SUITE(ctx, "Broadcasting — all operators");

    {
        tl::Tensor<float> a({1}, {10.0f});
        tl::Tensor<float> b({3}, {1.0f, 2.0f, 4.0f});

        auto sub = a - b;
        CHECK_EQ(ctx, sub.data[0], 9.0f);
        CHECK_EQ(ctx, sub.data[1], 8.0f);
        CHECK_EQ(ctx, sub.data[2], 6.0f);

        auto div = a / b;
        CHECK_EQ(ctx, div.data[0], 10.0f);
        CHECK_EQ(ctx, div.data[1],  5.0f);
        CHECK_EQ(ctx, div.data[2],  2.5f);
    }

    // ── compute_broadcast_shape utility ───────────────────────────────────────
    SUITE(ctx, "Broadcasting — shape computation");

    {
        auto s = tl::compute_broadcast_shape({3}, {1, 3});
        CHECK_EQ(ctx, s.size(), 2u);
        CHECK_EQ(ctx, s[0], 1u);
        CHECK_EQ(ctx, s[1], 3u);

        auto s2 = tl::compute_broadcast_shape({2, 1}, {2, 3});
        CHECK_EQ(ctx, s2[0], 2u);
        CHECK_EQ(ctx, s2[1], 3u);
    }

    // ── Incompatible shapes must still throw ──────────────────────────────────
    SUITE(ctx, "Broadcasting — incompatible shapes throw");

    {
        CHECK_THROWS(ctx, std::runtime_error, ({
            tl::Tensor<float> a({2, 3});
            tl::Tensor<float> b({2, 4});
            auto r = a + b;   // 3 vs 4, neither is 1 → error
        }));

        CHECK_THROWS(ctx, std::runtime_error, ({
            tl::Tensor<float> a({5});
            tl::Tensor<float> b({3});
            auto r = a + b;   // 5 vs 3, neither is 1 → error
        }));
    }
}
