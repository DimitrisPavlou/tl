// tests/test_linalg.cpp — Tests for tl::linalg (matmul, transpose, eye, trace, norms)
#include "test.hpp"
#include "../tl/tl.hpp"
#include <cmath>
#include <stdexcept>

void run_linalg_tests(tl::TestContext& ctx) {

    // ── Matrix multiplication ─────────────────────────────────────────────────
    SUITE(ctx, "Linalg — matmul");

    {
        tl::Tensor<float> A({{1, 2}, {3, 4}});
        tl::Tensor<float> B({{5, 6}, {7, 8}});
        auto C = tl::linalg::matmul(A, B);
        // C = {{1*5+2*7, 1*6+2*8}, {3*5+4*7, 3*6+4*8}} = {{19,22},{43,50}}
        CHECK_EQ(ctx, C.data[0], 19.0f);
        CHECK_EQ(ctx, C.data[1], 22.0f);
        CHECK_EQ(ctx, C.data[2], 43.0f);
        CHECK_EQ(ctx, C.data[3], 50.0f);
    }

    // All-ones matmul: 100x100 of 1s → each element should be 100
    {
        tl::Tensor<float> A({10, 10});
        tl::Tensor<float> B({10, 10});
        std::fill(A.data.begin(), A.data.end(), 1.0f);
        std::fill(B.data.begin(), B.data.end(), 1.0f);
        auto C = tl::linalg::matmul(A, B);
        CHECK_EQ(ctx, C.data[0], 10.0f);
        CHECK_EQ(ctx, C.data[99], 10.0f);
    }

    // Dimension mismatch must throw
    CHECK_THROWS(ctx, std::runtime_error, ({
        tl::Tensor<float> A({2, 3});
        tl::Tensor<float> B({4, 2});
        tl::linalg::matmul(A, B);
    }));

    // ── Transpose ─────────────────────────────────────────────────────────────
    SUITE(ctx, "Linalg — transpose");

    {
        tl::Tensor<float> M({2, 3}, {1, 2, 3, 4, 5, 6});
        auto T = tl::linalg::transpose(M);
        CHECK_EQ(ctx, T.shape[0], 3u);
        CHECK_EQ(ctx, T.shape[1], 2u);
        // T[0][0]=1, T[1][0]=2, T[2][0]=3, T[0][1]=4 ...
        CHECK_EQ(ctx, T.data[0], 1.0f);   // T[0][0]
        CHECK_EQ(ctx, T.data[1], 4.0f);   // T[0][1]
        CHECK_EQ(ctx, T.data[2], 2.0f);   // T[1][0]
    }

    // ── Identity matrix ───────────────────────────────────────────────────────
    SUITE(ctx, "Linalg — eye");

    {
        auto I = tl::linalg::eye<float>(3);
        CHECK_EQ(ctx, I.data[0], 1.0f);   // [0,0]
        CHECK_EQ(ctx, I.data[1], 0.0f);   // [0,1]
        CHECK_EQ(ctx, I.data[4], 1.0f);   // [1,1]
        CHECK_EQ(ctx, I.data[8], 1.0f);   // [2,2]
    }

    // ── Trace ─────────────────────────────────────────────────────────────────
    SUITE(ctx, "Linalg — trace");

    {
        auto I = tl::linalg::eye<double>(4);
        CHECK_NEAR(ctx, tl::linalg::trace(I), 4.0, 1e-10);

        tl::Tensor<float> M({{1, 2}, {3, 4}});
        CHECK_NEAR(ctx, tl::linalg::trace(M), 5.0f, 1e-6);
    }

    // Non-square must throw
    CHECK_THROWS(ctx, std::runtime_error, ({
        tl::Tensor<float> M({2, 3});
        tl::linalg::trace(M);
    }));

    // ── Matrix norms ──────────────────────────────────────────────────────────
    SUITE(ctx, "Linalg — matrix_norm");

    {
        tl::Tensor<double> mat({2, 2}, {3.0, 4.0, 0.0, 0.0});
        CHECK_NEAR(ctx, tl::linalg::matrix_norm(mat, "frob"), 5.0,  1e-10);
        CHECK_NEAR(ctx, tl::linalg::matrix_norm(mat, "1"),    4.0,  1e-10);
        CHECK_NEAR(ctx, tl::linalg::matrix_norm(mat, "inf"),  7.0,  1e-10);
    }

    // Unknown norm type must throw
    CHECK_THROWS(ctx, std::runtime_error,
        tl::linalg::matrix_norm(tl::Tensor<double>({2,2}), "bad"));
}