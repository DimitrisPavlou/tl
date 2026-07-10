// tests/test_phase5.cpp — Phase 5: autograd (gradient-checked) + RK4 ODE solver.
#include "test.hpp"
#include "../tl/tl.hpp"
#include <cmath>
#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void run_phase5_tests(tl::TestContext& ctx) {
    using namespace tl::autograd;

    // ── Basic gradients: z = sum(a * b) ⇒ dz/da = b, dz/db = a ────────────────
    SUITE(ctx, "Phase5 — autograd elementwise grads");
    {
        tl::Tensor<double> av({3}, {1, 2, 3});
        tl::Tensor<double> bv({3}, {4, 5, 6});
        Var a(av, true), b(bv, true);
        Var z = sum(a * b);
        z.backward();
        // dz/da == b
        CHECK_EQ(ctx, a.grad().data[0], 4.0);
        CHECK_EQ(ctx, a.grad().data[2], 6.0);
        // dz/db == a
        CHECK_EQ(ctx, b.grad().data[0], 1.0);
        CHECK_EQ(ctx, b.grad().data[2], 3.0);
        // z value == 1*4 + 2*5 + 3*6 = 32
        CHECK_EQ(ctx, z.value().data[0], 32.0);
    }

    // ── Gradient check: loss = sum(relu(X @ W)) vs finite differences ────────
    SUITE(ctx, "Phase5 — autograd gradient check (linear+relu)");
    {
        tl::Tensor<double> X({2, 3}, {0.5, -1.0, 2.0,
                                      1.5,  0.7, -0.3});
        tl::Tensor<double> Wv({3, 2}, {0.2, -0.4,
                                       0.9,  0.1,
                                      -0.5,  0.6});

        auto loss_of = [&](const tl::Tensor<double>& Wt) {
            Var Xv(X), W(Wt, true);
            Var loss = sum(relu(matmul(Xv, W)));
            return loss;
        };

        // Analytic gradient via autograd.
        Var loss = loss_of(Wv);
        loss.backward();
        // Recover the W node's grad: rebuild to grab the same handle.
        Var Xv(X), W(Wv, true);
        Var l2 = sum(relu(matmul(Xv, W)));
        l2.backward();
        const auto& gW = W.grad();

        // Numerical gradient (central differences).
        const double eps = 1e-6;
        bool all_close = true;
        double max_err = 0.0;
        for (std::size_t i = 0; i < Wv.data.size(); ++i) {
            tl::Tensor<double> Wp = Wv, Wm = Wv;
            Wp.data[i] += eps;
            Wm.data[i] -= eps;
            double fp = loss_of(Wp).value().data[0];
            double fm = loss_of(Wm).value().data[0];
            double num = (fp - fm) / (2 * eps);
            double err = std::abs(num - gW.data[i]);
            if (err > max_err) max_err = err;
            if (err > 1e-4) all_close = false;
        }
        CHECK(ctx, all_close);
        CHECK_NEAR(ctx, max_err, 0.0, 1e-4);
    }

    // ── RK4: scalar decay y' = -y, y(0)=1 ⇒ y(1) = e^-1 ──────────────────────
    SUITE(ctx, "Phase5 — RK4 scalar decay");
    {
        auto f = [](double, tl::SmallTensor<double, 1> y) {
            tl::SmallTensor<double, 1> dy; dy[0] = -y[0]; return dy;
        };
        tl::SmallTensor<double, 1> y0; y0[0] = 1.0;
        auto traj = tl::ode::integrate(f, y0, 0.0, 1.0, 0.001);
        double y_end = traj.back()[0];
        CHECK_NEAR(ctx, y_end, std::exp(-1.0), 1e-6);
    }

    // ── RK4: harmonic oscillator x'' = -x  ⇒  x(t)=cos t, v(t)=-sin t ────────
    SUITE(ctx, "Phase5 — RK4 harmonic oscillator");
    {
        auto f = [](double, tl::SmallTensor<double, 2> y) {
            tl::SmallTensor<double, 2> dy;
            dy[0] = y[1];      // x' = v
            dy[1] = -y[0];     // v' = -x
            return dy;
        };
        tl::SmallTensor<double, 2> y0; y0[0] = 1.0; y0[1] = 0.0;

        // Quarter period: t = π/2 ⇒ x≈0, v≈-1
        auto traj = tl::ode::integrate(f, y0, 0.0, M_PI / 2, 0.0005);
        CHECK_NEAR(ctx, traj.back()[0], 0.0,  1e-5);
        CHECK_NEAR(ctx, traj.back()[1], -1.0, 1e-5);

        // Full period: energy conserved, returns to start
        auto full = tl::ode::integrate(f, y0, 0.0, 2 * M_PI, 0.0005);
        CHECK_NEAR(ctx, full.back()[0], 1.0, 1e-4);
        CHECK_NEAR(ctx, full.back()[1], 0.0, 1e-4);
    }
}
