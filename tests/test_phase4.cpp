// tests/test_phase4.cpp — Phase 4: fused map + broadcast odometer correctness.
#include "test.hpp"
#include "../tl/tl.hpp"
#include <type_traits>
#include <cstddef>

void run_phase4_tests(tl::TestContext& ctx) {

    // ── Fused map matches the eager expression, element for element ──────────
    SUITE(ctx, "Phase4 — fused map == eager a*b+c");
    {
        tl::Tensor<float> A({2, 3}, {1, 2, 3, 4, 5, 6});
        tl::Tensor<float> B({2, 3}, {6, 5, 4, 3, 2, 1});
        tl::Tensor<float> C({2, 3}, {1, 1, 1, 1, 1, 1});

        auto eager = (A * B) + C;
        auto fused = tl::functional::map(
            [](float a, float b, float c) { return a * b + c; }, A, B, C);

        CHECK_EQ(ctx, fused.shape[0], 2u);
        CHECK_EQ(ctx, fused.shape[1], 3u);
        bool same = true;
        for (std::size_t i = 0; i < eager.data.size(); ++i)
            if (eager.data[i] != fused.data[i]) same = false;
        CHECK(ctx, same);
        CHECK_EQ(ctx, fused.data[0], 7.0f);   // 1*6+1
        CHECK_EQ(ctx, fused.data[5], 7.0f);   // 6*1+1
    }

    SUITE(ctx, "Phase4 — map result type deduction & unary");
    {
        tl::Tensor<int> A({3}, {1, 2, 3});
        // lambda returns double → Tensor<double>
        auto d = tl::functional::map([](int a) { return a * 0.5; }, A);
        static_assert(std::is_same_v<decltype(d)::value_type, double>);
        CHECK_EQ(ctx, d.data[0], 0.5);
        CHECK_EQ(ctx, d.data[2], 1.5);
    }

    SUITE(ctx, "Phase4 — map shape mismatch throws");
    {
        CHECK_THROWS(ctx, std::runtime_error, ({
            tl::Tensor<float> A({2, 3});
            tl::Tensor<float> B({3, 2});
            auto r = tl::functional::map([](float a, float b) { return a + b; }, A, B);
        }));
    }

    // ── Broadcast odometer: multi-dimensional carry (2,1,3)+(1,4,1)=(2,4,3) ──
    SUITE(ctx, "Phase4 — 3D broadcast (odometer) correctness");
    {
        tl::Tensor<float> A({2, 1, 3}, {1, 2, 3,
                                        4, 5, 6});
        tl::Tensor<float> B({1, 4, 1}, {10, 20, 30, 40});
        auto R = A + B;                         // → {2, 4, 3}
        CHECK_EQ(ctx, R.shape[0], 2u);
        CHECK_EQ(ctx, R.shape[1], 4u);
        CHECK_EQ(ctx, R.shape[2], 3u);

        // R[i][j][k] = A[i][0][k] + B[0][j][0]
        // R[0][0][0] = 1 + 10 = 11
        CHECK_EQ(ctx, R.data[0], 11.0f);
        // R[0][1][0] = 1 + 20 = 21  (index = 1*3 + 0 = 3)
        CHECK_EQ(ctx, R.data[3], 21.0f);
        // R[0][3][2] = 3 + 40 = 43  (index = 3*3 + 2 = 11)
        CHECK_EQ(ctx, R.data[11], 43.0f);
        // R[1][0][0] = 4 + 10 = 14  (index = 1*(4*3) = 12)
        CHECK_EQ(ctx, R.data[12], 14.0f);
        // R[1][3][2] = 6 + 40 = 46  (index = 12 + 11 = 23)
        CHECK_EQ(ctx, R.data[23], 46.0f);
    }
}
