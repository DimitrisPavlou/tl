#pragma once

#include <array>
#include <cstddef>
#include <cmath>
#include <type_traits>

// SmallTensor<T, N> — a fixed-size, stack-allocated vector for small state
// vectors (ODE / control-systems state). No heap allocation per step, so tight
// integrator loops (RK4) don't pay allocation overhead — the specialised fast
// path for numerical simulation.
namespace tl {

template <typename T, std::size_t N>
struct SmallTensor {
    static_assert(std::is_floating_point_v<T> || std::is_arithmetic_v<T>,
                  "SmallTensor requires a numeric type");
    std::array<T, N> data{};

    static constexpr std::size_t size() { return N; }

    T&       operator[](std::size_t i)       { return data[i]; }
    const T& operator[](std::size_t i) const { return data[i]; }

    SmallTensor operator+(const SmallTensor& o) const {
        SmallTensor r;
        for (std::size_t i = 0; i < N; ++i) r.data[i] = data[i] + o.data[i];
        return r;
    }
    SmallTensor operator-(const SmallTensor& o) const {
        SmallTensor r;
        for (std::size_t i = 0; i < N; ++i) r.data[i] = data[i] - o.data[i];
        return r;
    }
    SmallTensor operator*(T s) const {
        SmallTensor r;
        for (std::size_t i = 0; i < N; ++i) r.data[i] = data[i] * s;
        return r;
    }
    SmallTensor& operator+=(const SmallTensor& o) {
        for (std::size_t i = 0; i < N; ++i) data[i] += o.data[i];
        return *this;
    }

    // Euclidean norm — handy for step-size control / convergence checks.
    T norm() const {
        T s{0};
        for (std::size_t i = 0; i < N; ++i) s += data[i] * data[i];
        return std::sqrt(s);
    }
};

template <typename T, std::size_t N>
SmallTensor<T, N> operator*(T s, const SmallTensor<T, N>& v) { return v * s; }

} // namespace tl
