#pragma once

#include "../tensor_core/tensor.hpp"
#include <type_traits>
#include <stdexcept>
#include <cstddef>

// Fused element-wise map / zip_with.
//
// The eager operators (a * b + c) allocate an intermediate tensor per operation
// and traverse memory once per op. `map` fuses an arbitrary element-wise
// expression into a SINGLE pass with ZERO intermediates — the key throughput win
// for neural-net forward/backward passes and ODE right-hand-side evaluations:
//
//   auto y = tl::functional::map(
//       [](float a, float b, float c) { return a * b + c; }, A, B, C);
//
// All inputs must share the same shape. The result element type is deduced from
// the callable's return type.
namespace tl {
namespace functional {

template <typename F, typename T0, typename... Tn>
auto map(F f, const Tensor<T0>& t0, const Tensor<Tn>&... tn)
    -> Tensor<std::decay_t<std::invoke_result_t<F, T0, Tn...>>>
{
    // Every operand must have the same shape (checked at fold-expression time).
    const std::vector<std::size_t>& s0 = t0.shape;
    bool ok = true;
    ((ok = ok && (tn.shape == s0)), ...);
    if (!ok) throw std::runtime_error("map: all tensors must share the same shape");

    using R = std::decay_t<std::invoke_result_t<F, T0, Tn...>>;
    Tensor<R> out(t0.shape);

    const std::size_t n = t0.data.size();
    R* dst = out.data.data();
    const T0* p0 = t0.data.data();
    // Pack of input pointers, expanded per element below.
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = f(p0[i], tn.data[i]...);
    }
    return out;
}

} // namespace functional
} // namespace tl
