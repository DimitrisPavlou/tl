#pragma once

#include "../tensor_core/tensor.hpp"
#include <cmath>
#include <algorithm>

// Note: #pragma omp simd requires compiling with -fopenmp (GCC/Clang) or /openmp (MSVC).
// Without that flag the pragma is silently ignored; all loops remain correct.

namespace tl {
namespace functional {

    // apply_unary: applies a unary op element-wise, preserving the input type T.
    // FIX: The output type Tout is now a separate template parameter so that:
    //   - apply_unary<double>(t, std::exp) returns Tensor<double> (no precision loss)
    //   - apply_unary<float>(t, std::exp)  returns Tensor<float>
    // Callers that want integer->float promotion can explicitly pass Tout=float.
    template <typename Tout, typename T, typename Op>
    Tensor<Tout> apply_unary(const Tensor<T>& t, Op op) {
        Tensor<Tout> res(t.shape);
        const T* src = t.data.data();
        Tout* dst = res.data.data();
        const std::size_t n = t.data.size();

        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = op(static_cast<Tout>(src[i]));
        }
        return res;
    }

    // Helper: deduce output type.
    // For floating-point T: output is T (preserves float or double).
    // For integral T: output is float (sensible default for math functions on ints).
    template <typename T>
    using math_result_t = std::conditional_t<std::is_floating_point_v<T>, T, float>;

    // Convenience wrapper that applies the type promotion rule above.
    template <typename T, typename Op>
    Tensor<math_result_t<T>> apply_unary_math(const Tensor<T>& t, Op op) {
        return apply_unary<math_result_t<T>>(t, op);
    }


    // --- Elementary Functions ---

    template <typename T>
    Tensor<math_result_t<T>> abs(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::abs(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> exp(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::exp(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> log(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::log(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> sqrt(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::sqrt(v); });
    }


    // --- Trigonometric & Hyperbolic Functions ---

    template <typename T>
    Tensor<math_result_t<T>> sin(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::sin(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> cos(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::cos(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> tan(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::tan(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> sinh(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::sinh(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> cosh(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::cosh(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> tanh(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::tanh(v); });
    }


    // --- Inverse Hyperbolic Functions ---
    // acosh is only defined for x >= 1; atanh for |x| < 1.

    template <typename T>
    Tensor<math_result_t<T>> asinh(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::asinh(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> acosh(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::acosh(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> atanh(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::atanh(v); });
    }


    // --- Additional Numerical Functions ---

    template <typename T>
    Tensor<math_result_t<T>> ceil(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::ceil(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> floor(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::floor(v); });
    }

    template <typename T>
    Tensor<math_result_t<T>> round(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return std::round(v); });
    }


    // --- Power and Square Root ---

    template <typename T>
    Tensor<math_result_t<T>> square(const Tensor<T>& t) {
        return apply_unary_math(t, [](math_result_t<T> v) { return v * v; });
    }

    // Element-wise power: result[i] = t[i]^p
    template <typename T>
    Tensor<math_result_t<T>> power(const Tensor<T>& t, math_result_t<T> p) {
        return apply_unary_math(t, [p](math_result_t<T> v) { return std::pow(v, p); });
    }


    // --- Activation Functions ---

    template <typename T>
    Tensor<math_result_t<T>> relu(const Tensor<T>& t) {
        using R = math_result_t<T>;
        return apply_unary_math(t, [](R v) { return (v > R{0}) ? v : R{0}; });
    }

    template <typename T>
    Tensor<math_result_t<T>> leaky_relu(const Tensor<T>& t, math_result_t<T> alpha = 0.01f) {
        using R = math_result_t<T>;
        return apply_unary_math(t, [alpha](R v) { return (v > R{0}) ? v : alpha * v; });
    }

    template <typename T>
    Tensor<math_result_t<T>> sigmoid(const Tensor<T>& t) {
        using R = math_result_t<T>;
        return apply_unary_math(t, [](R v) { return R{1} / (R{1} + std::exp(-v)); });
    }


    // --- Clamping ---

    template <typename T>
    Tensor<math_result_t<T>> clip(const Tensor<T>& t, math_result_t<T> min_val, math_result_t<T> max_val) {
        using R = math_result_t<T>;
        return apply_unary_math(t, [min_val, max_val](R v) {
            return std::max(min_val, std::min(max_val, v));
        });
    }

} // namespace functional
} // namespace tl