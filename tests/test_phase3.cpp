// tests/test_phase3.cpp — Phase 3: axis reductions, random factories, slicing,
// batched matmul, and the Numeric concept.
#include "test.hpp"
#include "../tl/tl.hpp"
#include <cmath>
#include <cstddef>

void run_phase3_tests(tl::TestContext& ctx) {

    // ── Axis reductions ──────────────────────────────────────────────────────
    SUITE(ctx, "Phase3 — sum/mean/max/min along an axis");
    {
        tl::Tensor<float> m({2, 3}, {1, 2, 3,
                                     4, 5, 6});

        auto s0 = tl::sum(m, 0);            // collapse rows → {5, 7, 9}
        CHECK_EQ(ctx, s0.shape.size(), 1u);
        CHECK_EQ(ctx, s0.shape[0], 3u);
        CHECK_EQ(ctx, s0.data[0], 5.0f);
        CHECK_EQ(ctx, s0.data[1], 7.0f);
        CHECK_EQ(ctx, s0.data[2], 9.0f);

        auto s1 = tl::sum(m, 1);            // collapse cols → {6, 15}
        CHECK_EQ(ctx, s1.shape[0], 2u);
        CHECK_EQ(ctx, s1.data[0], 6.0f);
        CHECK_EQ(ctx, s1.data[1], 15.0f);

        auto s1k = tl::sum(m, 1, /*keepdims=*/true);   // shape {2,1}
        CHECK_EQ(ctx, s1k.shape.size(), 2u);
        CHECK_EQ(ctx, s1k.shape[0], 2u);
        CHECK_EQ(ctx, s1k.shape[1], 1u);

        auto me = tl::mean(m, 1);           // {2, 5}
        CHECK_EQ(ctx, me.data[0], 2.0f);
        CHECK_EQ(ctx, me.data[1], 5.0f);

        auto mx = tl::max(m, 0);            // {4, 5, 6}
        CHECK_EQ(ctx, mx.data[0], 4.0f);
        CHECK_EQ(ctx, mx.data[2], 6.0f);

        auto mn = tl::min(m, 0);            // {1, 2, 3}
        CHECK_EQ(ctx, mn.data[0], 1.0f);
        CHECK_EQ(ctx, mn.data[2], 3.0f);

        // Whole-tensor reductions still work (overload coexistence).
        CHECK_EQ(ctx, tl::sum(m), 21.0f);
        CHECK_EQ(ctx, tl::max(m), 6.0f);

        // Negative axis
        auto sneg = tl::sum(m, -1);         // same as axis 1
        CHECK_EQ(ctx, sneg.data[0], 6.0f);
    }

    SUITE(ctx, "Phase3 — argmax along an axis");
    {
        tl::Tensor<float> m({2, 3}, {1, 5, 3,
                                     4, 2, 6});
        auto am = tl::argmax(m, 1);         // row0 → idx 1, row1 → idx 2
        CHECK_EQ(ctx, am.shape[0], 2u);
        CHECK_EQ(ctx, am.data[0], 1u);
        CHECK_EQ(ctx, am.data[1], 2u);
    }

    // ── Random factories ─────────────────────────────────────────────────────
    SUITE(ctx, "Phase3 — random reproducibility & ranges");
    {
        tl::seed(42);
        auto a = tl::randn<float>({500});
        tl::seed(42);
        auto b = tl::randn<float>({500});
        bool identical = true;
        for (std::size_t i = 0; i < a.data.size(); ++i)
            if (a.data[i] != b.data[i]) identical = false;
        CHECK(ctx, identical);   // same seed → same sequence

        auto u = tl::uniform<float>({1000}, -2.0f, 5.0f);
        bool in_range = true;
        for (std::size_t i = 0; i < u.data.size(); ++i)
            if (u.data[i] < -2.0f || u.data[i] >= 5.0f) in_range = false;
        CHECK(ctx, in_range);

        // Gaussian sample mean should be near 0 for a large sample.
        auto g = tl::randn<double>({20000}, 0.0, 1.0);
        double m = tl::mean(g);
        CHECK_NEAR(ctx, m, 0.0, 0.05);
    }

    // ── Slicing ──────────────────────────────────────────────────────────────
    SUITE(ctx, "Phase3 — slicing (minibatch & block)");
    {
        tl::Tensor<float> X({4, 3}, { 0,  1,  2,
                                      3,  4,  5,
                                      6,  7,  8,
                                      9, 10, 11});

        auto rows = tl::slice(X, {tl::Slice(1, 3)});    // rows 1..2 → shape {2,3}
        CHECK_EQ(ctx, rows.shape[0], 2u);
        CHECK_EQ(ctx, rows.shape[1], 3u);
        CHECK_EQ(ctx, rows.data[0], 3.0f);
        CHECK_EQ(ctx, rows.data[5], 8.0f);

        auto block = tl::slice(X, {tl::Slice(1, 3), tl::Slice(0, 2)});  // {2,2}
        CHECK_EQ(ctx, block.shape[0], 2u);
        CHECK_EQ(ctx, block.shape[1], 2u);
        CHECK_EQ(ctx, block.data[0], 3.0f);
        CHECK_EQ(ctx, block.data[1], 4.0f);
        CHECK_EQ(ctx, block.data[2], 6.0f);
        CHECK_EQ(ctx, block.data[3], 7.0f);

        auto strided = tl::slice(X, {tl::Slice(0, 4, 2)});   // rows 0, 2 → {2,3}
        CHECK_EQ(ctx, strided.shape[0], 2u);
        CHECK_EQ(ctx, strided.data[0], 0.0f);
        CHECK_EQ(ctx, strided.data[3], 6.0f);
    }

    // ── Batched matmul ───────────────────────────────────────────────────────
    SUITE(ctx, "Phase3 — batched matmul");
    {
        // 3D @ 2D (shared weight)
        tl::Tensor<float> A({2, 2, 3}, {1, 2, 3, 4, 5, 6,      // batch 0
                                        7, 8, 9, 10, 11, 12}); // batch 1
        tl::Tensor<float> W({3, 2}, {1, 0,
                                     0, 1,
                                     1, 1});
        auto C = tl::linalg::matmul(A, W);   // → {2, 2, 2}
        CHECK_EQ(ctx, C.shape.size(), 3u);
        CHECK_EQ(ctx, C.shape[0], 2u);
        CHECK_EQ(ctx, C.shape[1], 2u);
        CHECK_EQ(ctx, C.shape[2], 2u);

        // Verify batch 0 against a manual 2D matmul of the same slice.
        tl::Tensor<float> A0({2, 3}, {1, 2, 3, 4, 5, 6});
        auto C0 = tl::linalg::matmul(A0, W);
        CHECK_EQ(ctx, C.data[0], C0.data[0]);
        CHECK_EQ(ctx, C.data[1], C0.data[1]);
        CHECK_EQ(ctx, C.data[2], C0.data[2]);
        CHECK_EQ(ctx, C.data[3], C0.data[3]);

        // 3D @ 3D (per-batch weights)
        tl::Tensor<float> B3({2, 3, 2}, {1, 0, 0, 1, 1, 1,
                                         2, 0, 0, 2, 2, 2});
        auto C3 = tl::linalg::matmul(A, B3);
        CHECK_EQ(ctx, C3.shape[0], 2u);
        CHECK_EQ(ctx, C3.shape[2], 2u);
        // batch 1 uses weights scaled by 2 relative to batch-0 identity-ish
        CHECK_EQ(ctx, C3.data[4], C.data[4] * 2.0f);
    }
}
