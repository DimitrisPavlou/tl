#pragma once

#include "tensor.hpp"
#include <random>
#include <cstdint>
#include <type_traits>

// Random tensor factories built on <random> (replaces hand-rolled LCG init).
//   tl::seed(1234);                         // reproducibility
//   auto W = tl::randn<float>({784, 64});   // Gaussian weights
//   auto U = tl::uniform<float>({3, 3}, -1.0f, 1.0f);
namespace tl {

// Process-wide engine. Seed it for reproducible runs.
inline std::mt19937_64& global_rng() {
    static std::mt19937_64 engine{std::random_device{}()};
    return engine;
}

inline void seed(std::uint64_t s) { global_rng().seed(s); }

// Uniform in [low, high).  Integral T uses an inclusive integer distribution.
template <Numeric T = double>
Tensor<T> uniform(const std::vector<std::size_t>& shape, T low = T{0}, T high = T{1}) {
    Tensor<T> t(shape);
    auto& g = global_rng();
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(low, high);
        for (std::size_t i = 0; i < t.data.size(); ++i) t.data[i] = dist(g);
    } else {
        std::uniform_int_distribution<T> dist(low, high);
        for (std::size_t i = 0; i < t.data.size(); ++i) t.data[i] = dist(g);
    }
    return t;
}

// Standard-normal-scaled Gaussian.  Floating-point only.
template <typename T = double>
Tensor<T> randn(const std::vector<std::size_t>& shape, T mean = T{0}, T stddev = T{1}) {
    static_assert(std::is_floating_point_v<T>, "randn requires a floating-point type");
    Tensor<T> t(shape);
    std::normal_distribution<T> dist(mean, stddev);
    auto& g = global_rng();
    for (std::size_t i = 0; i < t.data.size(); ++i) t.data[i] = dist(g);
    return t;
}

// Convenience: uniform [0, 1).
template <typename T = double>
Tensor<T> rand(const std::vector<std::size_t>& shape) {
    return uniform<T>(shape, T{0}, T{1});
}

// Xavier/Glorot uniform initialisation: range ±sqrt(6 / (fan_in + fan_out)).
template <typename T = double>
Tensor<T> xavier_uniform(std::size_t fan_in, std::size_t fan_out) {
    static_assert(std::is_floating_point_v<T>, "xavier_uniform requires a floating-point type");
    const T limit = std::sqrt(static_cast<T>(6) /
                              static_cast<T>(fan_in + fan_out));
    return uniform<T>({fan_in, fan_out}, -limit, limit);
}

} // namespace tl
