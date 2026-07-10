#pragma once

#include <vector>
#include <cstddef>

// Generic ODE integrators. State-type-agnostic: they work with tl::SmallTensor
// (heap-free, ideal for small systems), tl::Tensor, or any type supporting
// `State + State` and `State * scalar`.
//
//   dy/dt = f(t, y)
//
//   auto step = tl::ode::rk4_step(f, t, y, h);
//   auto traj = tl::ode::integrate(f, y0, t0, t1, h);   // full trajectory
namespace tl {
namespace ode {

// One classical 4th-order Runge–Kutta step.
template <typename State, typename F, typename Scalar = double>
State rk4_step(F f, Scalar t, const State& y, Scalar h) {
    State k1 = f(t, y);
    State k2 = f(t + h / 2, y + k1 * (h / 2));
    State k3 = f(t + h / 2, y + k2 * (h / 2));
    State k4 = f(t + h, y + k3 * h);
    // y + (h/6) (k1 + 2k2 + 2k3 + k4)
    return y + (k1 + k2 * Scalar{2} + k3 * Scalar{2} + k4) * (h / 6);
}

// Integrate from t0 to t1 with fixed step h; returns the sampled trajectory
// (including the initial state).
template <typename State, typename F, typename Scalar = double>
std::vector<State> integrate(F f, State y0, Scalar t0, Scalar t1, Scalar h) {
    std::vector<State> traj;
    traj.push_back(y0);
    State y = y0;
    Scalar t = t0;
    while (t < t1) {
        if (t + h > t1) h = t1 - t;   // last partial step
        y = rk4_step<State>(f, t, y, h);
        t += h;
        traj.push_back(y);
    }
    return traj;
}

} // namespace ode
} // namespace tl
