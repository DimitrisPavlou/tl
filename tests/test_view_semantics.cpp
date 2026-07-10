// tests/test_view_semantics.cpp — Phase 1: View const-correctness & value semantics
#include "test.hpp"
#include "../tl/tl.hpp"
#include <stdexcept>
#include <type_traits>

void run_view_semantics_tests(tl::TestContext& ctx) {

    // ── Fully-indexed view converts to a scalar element ──────────────────────
    SUITE(ctx, "View — scalar conversion (dims_left == 0)");
    {
        tl::Tensor<int> m({2, 3}, {1, 2, 3, 4, 5, 6});
        CHECK_EQ(ctx, (int)m[0][0], 1);
        CHECK_EQ(ctx, (int)m[1][2], 6);
    }

    // ── Converting a NON-fully-indexed view to a scalar now throws ───────────
    // Previously this silently returned element 0.
    SUITE(ctx, "View — partial view -> scalar throws");
    {
        CHECK_THROWS(ctx, std::out_of_range, ({
            tl::Tensor<int> m({2, 3}, {1, 2, 3, 4, 5, 6});
            int x = m[0];            // m[0] still has 1 dimension left → invalid
            (void)x;
        }));
    }

    // ── NumPy-style write-through assignment (view = view) ───────────────────
    SUITE(ctx, "View — write-through row assignment");
    {
        tl::Tensor<float> m({2, 3}, {1, 2, 3, 4, 5, 6});
        m[0] = m[1];                 // copy row 1 into row 0
        CHECK_EQ(ctx, m.data[0], 4.0f);
        CHECK_EQ(ctx, m.data[1], 5.0f);
        CHECK_EQ(ctx, m.data[2], 6.0f);
        // Source row is untouched
        CHECK_EQ(ctx, m.data[3], 4.0f);
    }

    // ── Explicit .copy_from() is equivalent and self-documenting ─────────────
    SUITE(ctx, "View — copy_from()");
    {
        tl::Tensor<float> m({2, 2}, {1, 2, 3, 4});
        m[0].copy_from(m[1]);
        CHECK_EQ(ctx, m.data[0], 3.0f);
        CHECK_EQ(ctx, m.data[1], 4.0f);
    }

    // ── Size-mismatched view assignment throws ───────────────────────────────
    SUITE(ctx, "View — size-mismatch assignment throws");
    {
        CHECK_THROWS(ctx, std::runtime_error, ({
            tl::Tensor<float> a({2, 3});
            tl::Tensor<float> b({2, 2});
            a[0] = b[0];             // 3 elements vs 2 → mismatch
        }));
    }

    // ── Const-correctness: indexing a const view yields View<const T> ────────
    SUITE(ctx, "View — const indexing yields read-only view");
    {
        tl::Tensor<int> m({2, 2}, {1, 2, 3, 4});
        const tl::View<int> cv = m.view();
        // Reading through a const view still works.
        CHECK_EQ(ctx, (int)cv[1][1], 4);
        // The const overload returns a read-only handle.
        using Elem = decltype(cv[0]);
        static_assert(std::is_same_v<Elem, tl::View<const int>>,
                      "const View<T>::operator[] must return View<const T>");
        CHECK(ctx, (std::is_same_v<Elem, tl::View<const int>>));
    }
}
