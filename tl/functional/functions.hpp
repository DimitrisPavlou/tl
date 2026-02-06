#pragma once

#include "../tensor_core/tensor.hpp"
#include <cmath>
#include <algorithm>

namespace tl {
namespace functional {

    // Helper: Maps Tensor<T> to Tensor<float>, applying a unary operation in the process.
    // This handles the transition from possible integer data to floating point results for cases like sqrt(int) -> float. 
    // Should higher accuracy be needed (e.g., double), we can template the output type as well, or statically cast to double. 
    // For now we assume float32 is sufficient for these operations.

    template <typename T, typename Op>
    Tensor<float> apply_unary_float(const Tensor<T>& t, Op op) {
        Tensor<float> res(t.shape);
        const T* src = t.data.data();
        float* dst = res.data.data();
        std::size_t n = t.data.size();

        // Using SIMD for the conversion/operation loop where possible
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = op(static_cast<float>(src[i]));
        }
        return res;
    }

    // --- Elementary Functions (Returning Float) ---
    // Used Lambda functions to avoid code duplication.


    template <typename T>
    Tensor<float> abs(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::abs(v); });
    }

    template <typename T>
    Tensor<float> exp(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::exp(v); });
    }

    template <typename T>
    Tensor<float> log(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::log(v); });
    }

    template <typename T>
    Tensor<float> sqrt(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::sqrt(v); });
    }

    // --- Trigonometric & Hyperbolic Functions ---

    template <typename T>
    Tensor<float> sin(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::sin(v); });
    }

    template <typename T>
    Tensor<float> cos(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::cos(v); });
    }

    template <typename T>
    Tensor<float> tan(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::tan(v); });
    }

    template <typename T>
    Tensor<float> sinh(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::sinh(v); });
    }

    template <typename T>
    Tensor<float> cosh(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::cosh(v); });
    }

    template <typename T>
    Tensor<float> tanh(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::tanh(v); });
    }

    // --- Inverse Hyperbolic Functions ---

    template <typename T>
    Tensor<float> asinh(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::asinh(v); });
    }

    template <typename T>
    Tensor<float> acosh(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { 
            // acosh is only defined for x >= 1
            return std::acosh(v); 
        });
    }

    template <typename T>
    Tensor<float> atanh(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { 
            // atanh is only defined for |x| < 1
            return std::atanh(v); 
        });
    }

    // --- Additional Numerical Functions ---

    template <typename T>
    Tensor<float> ceil(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::ceil(v); });
    }

    template <typename T>
    Tensor<float> floor(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::floor(v); });
    }

    template <typename T>
    Tensor<float> round(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return std::round(v); });
    }

    // --- Power and Square Root ---

    template <typename T>
    Tensor<float> square(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { return v * v; });
    }

    // Element-wise power: result = base^p
    template <typename T>
    Tensor<float> power(const Tensor<T>& t, float p) {
        return apply_unary_float(t, [p](float v) { return std::pow(v, p); });
    }

    // --- Activation Functions ---

    template <typename T>
    Tensor<float> relu(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { 
            return (v > 0.0f) ? v : 0.0f; 
        });
    }

    template <typename T>
    Tensor<float> leaky_relu(const Tensor<T>& t, float alpha = 0.01f) {
        return apply_unary_float(t, [alpha](float v) { 
            return (v > 0.0f) ? v : alpha * v; 
        });
    }

    template <typename T>
    Tensor<float> sigmoid(const Tensor<T>& t) {
        return apply_unary_float(t, [](float v) { 
            return 1.0f / (1.0f + std::exp(-v)); 
        });
    }

    // --- Comparison/Clamping ---

    template <typename T>
    Tensor<float> clip(const Tensor<T>& t, float min_val, float max_val) {
        return apply_unary_float(t, [min_val, max_val](float v) {
            return std::max(min_val, std::min(max_val, v));
        });
    }

} // namespace functional
} // namespace tl