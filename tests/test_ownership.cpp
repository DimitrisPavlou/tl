// tests/test_ownership.cpp — Phase 2: shared-buffer ownership, lifetime safety,
// stride-aware view assignment, and value-semantics independence.
#include "test.hpp"
#include "../tl/tl.hpp"
#include <cstddef>

void run_ownership_tests(tl::TestContext& ctx) {

    // ── A View outlives its source Tensor (dangling fix) ─────────────────────
    // The Tensor is destroyed at the end of the lambda; the returned View keeps
    // the underlying buffers alive via shared ownership.
    SUITE(ctx, "Ownership — view outlives its tensor");
    {
        tl::View<float> escaped = [] {
            tl::Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
            return t.view();     // t dies here
        }();
        CHECK_EQ(ctx, (float)escaped[0][0], 1.0f);
        CHECK_EQ(ctx, (float)escaped[1][2], 6.0f);
    }

    // ── A row view from a destroyed tensor is still readable ─────────────────
    SUITE(ctx, "Ownership — escaped row view stays valid");
    {
        tl::View<float> row = [] {
            tl::Tensor<float> t({2, 3}, {10, 20, 30, 40, 50, 60});
            return t[1];         // row 1; t dies here
        }();
        CHECK_EQ(ctx, (float)row[0], 40.0f);
        CHECK_EQ(ctx, (float)row[2], 60.0f);
    }

    // ── View from a temporary expression result survives ─────────────────────
    SUITE(ctx, "Ownership — view from temporary expression");
    {
        auto v = (tl::ones<float>({3}) + tl::ones<float>({3})).view();
        CHECK_EQ(ctx, (float)v[0], 2.0f);
        CHECK_EQ(ctx, (float)v[2], 2.0f);
    }

    // ── Stride-aware copy_from on NON-contiguous (column) views ──────────────
    // Columns of a row-major 2x3 matrix have stride 3 — not contiguous. The old
    // std::copy-based assignment would have copied the wrong elements.
    SUITE(ctx, "Ownership — stride-aware column copy");
    {
        tl::Tensor<float> m({2, 3}, {1, 2, 3,
                                     4, 5, 6});
        std::size_t col_shape[1]  = {2};
        std::size_t col_stride[1] = {3};

        tl::View<float> col0{ &m.data[0], col_shape, col_stride, 1 };
        tl::View<float> col2{ &m.data[2], col_shape, col_stride, 1 };

        col0.copy_from(col2);    // copy column 2 (values 3, 6) into column 0

        // Column 0 positions are data[0] and data[3].
        CHECK_EQ(ctx, m.data[0], 3.0f);
        CHECK_EQ(ctx, m.data[3], 6.0f);
        // Everything else untouched.
        CHECK_EQ(ctx, m.data[1], 2.0f);
        CHECK_EQ(ctx, m.data[2], 3.0f);
        CHECK_EQ(ctx, m.data[4], 5.0f);
        CHECK_EQ(ctx, m.data[5], 6.0f);
    }

    // ── Tensor copy / clone are independent (value semantics preserved) ──────
    SUITE(ctx, "Ownership — copy & clone independence");
    {
        tl::Tensor<int> a({3}, {1, 2, 3});

        auto b = a;              // copy-construct
        b.data[0] = 99;
        CHECK_EQ(ctx, a.data[0], 1);   // original unchanged

        auto c = a.clone();      // explicit deep copy
        c.data[1] = 77;
        CHECK_EQ(ctx, a.data[1], 2);   // original unchanged

        tl::Tensor<int> d({3});
        d = a;                   // copy-assign
        d.data[2] = 55;
        CHECK_EQ(ctx, a.data[2], 3);   // original unchanged
    }
}
