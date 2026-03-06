// tests/test_tensor_core.cpp — Tests for Tensor construction, indexing, operators
#include "test.hpp"
#include "../tl/tl.hpp"
#include <stdexcept>

void run_tensor_core_tests(tl::TestContext& ctx) {

    // ── Construction ──────────────────────────────────────────────────────────
    SUITE(ctx, "Tensor Core — Construction");

    {
        tl::Tensor<float> t({{1, 2}, {3, 4}});
        CHECK_EQ(ctx, t.shape[0], 2u);
        CHECK_EQ(ctx, t.shape[1], 2u);
        CHECK_EQ(ctx, t.data[0], 1.0f);
        CHECK_EQ(ctx, t.data[3], 4.0f);
    }

    {
        tl::Tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
        CHECK_EQ(ctx, t.data[0], 1);
        CHECK_EQ(ctx, t.data[5], 6);
    }

    // Shape/data mismatch must throw
    CHECK_THROWS(ctx, std::runtime_error,
        tl::Tensor<int>({2, 2}, {1, 2, 3}));   // 3 elements, shape needs 4

    // ── Strides ───────────────────────────────────────────────────────────────
    SUITE(ctx, "Tensor Core — Strides");

    {
        tl::Tensor<int> t({3, 4, 5});
        // Row-major: strides = {4*5, 5, 1} = {20, 5, 1}
        CHECK_EQ(ctx, t.strides[0], 20u);
        CHECK_EQ(ctx, t.strides[1],  5u);
        CHECK_EQ(ctx, t.strides[2],  1u);
    }

    // ── Indexing & bounds checking ────────────────────────────────────────────
    SUITE(ctx, "Tensor Core — Indexing");

    {
        tl::Tensor<int> mat({2, 3}, {1, 2, 3, 4, 5, 6});
        CHECK_EQ(ctx, (int)mat[0][0], 1);
        CHECK_EQ(ctx, (int)mat[1][2], 6);
    }

    // 0D tensor indexing must throw
    CHECK_THROWS(ctx, std::out_of_range,
        tl::Tensor<int>({})[0]);

    // Out-of-range index must throw
    CHECK_THROWS(ctx, std::out_of_range, ({
        tl::Tensor<int> t({3}, {1, 2, 3});
        auto v = t[5];
    }));

    // ── Element-wise operators ────────────────────────────────────────────────
    SUITE(ctx, "Tensor Core — Element-wise Operators");

    {
        tl::Tensor<float> a({3}, {1.0f, 2.0f, 3.0f});
        tl::Tensor<float> b({3}, {4.0f, 5.0f, 6.0f});

        auto add = a + b;
        CHECK_EQ(ctx, add.data[0], 5.0f);
        CHECK_EQ(ctx, add.data[2], 9.0f);

        auto sub = b - a;
        CHECK_EQ(ctx, sub.data[0], 3.0f);

        auto mul = a * b;
        CHECK_EQ(ctx, mul.data[1], 10.0f);

        auto div = b / a;
        CHECK_NEAR(ctx, div.data[0], 4.0f, 1e-6);
    }

    // ── Scalar operators ──────────────────────────────────────────────────────
    SUITE(ctx, "Tensor Core — Scalar Operators");

    {
        tl::Tensor<float> t({3}, {1.0f, 2.0f, 3.0f});

        auto ts = t * 2.0f;
        CHECK_EQ(ctx, ts.data[1], 4.0f);

        auto st = 10.0f - t;           // scalar - tensor (was the buggy operator)
        CHECK_EQ(ctx, st.data[0], 9.0f);
        CHECK_EQ(ctx, st.data[2], 7.0f);

        auto sd = 6.0f / t;            // scalar / tensor
        CHECK_NEAR(ctx, sd.data[0], 6.0f, 1e-6);
        CHECK_NEAR(ctx, sd.data[2], 2.0f, 1e-6);
    }

    // ── In-place scalar operators ─────────────────────────────────────────────
    SUITE(ctx, "Tensor Core — In-place Operators");

    {
        tl::Tensor<float> t({2}, {2.0f, 4.0f});
        t *= 3.0f;
        CHECK_EQ(ctx, t.data[0],  6.0f);
        CHECK_EQ(ctx, t.data[1], 12.0f);

        t /= 2.0f;
        CHECK_EQ(ctx, t.data[0], 3.0f);
        CHECK_EQ(ctx, t.data[1], 6.0f);

        t += 1.0f;
        CHECK_EQ(ctx, t.data[0], 4.0f);

        t -= 4.0f;
        CHECK_EQ(ctx, t.data[0], 0.0f);
    }
}