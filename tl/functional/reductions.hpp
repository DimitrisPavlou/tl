#pragma once

#include "../tensor_core/tensor.hpp"
#include <vector>
#include <limits>
#include <stdexcept>
#include <cstddef>

// Axis-wise reductions (NumPy-style): sum / mean / max / min / argmax along a
// single axis, with optional keepdims. These are prerequisites for softmax,
// cross-entropy loss, batch-norm, etc.
//
// The whole-tensor reductions (tl::sum(t), tl::mean(t), ...) live in
// tensor_utils.hpp; these axis overloads are distinguished by the extra `axis`
// argument, so both coexist without ambiguity.

namespace tl {
namespace reduce_detail {

    inline std::size_t normalize_axis(int axis, std::size_t rank) {
        if (rank == 0) throw std::runtime_error("Cannot reduce a 0-dimensional tensor along an axis");
        int a = axis;
        if (a < 0) a += static_cast<int>(rank);
        if (a < 0 || a >= static_cast<int>(rank)) {
            throw std::out_of_range("axis out of range for tensor rank");
        }
        return static_cast<std::size_t>(a);
    }

    // Output shape with `axis` removed (or set to 1 when keepdims).
    inline std::vector<std::size_t> reduced_shape(
        const std::vector<std::size_t>& in_shape, std::size_t ax, bool keepdims) {
        std::vector<std::size_t> out;
        for (std::size_t d = 0; d < in_shape.size(); ++d) {
            if (d == ax) { if (keepdims) out.push_back(1); }
            else         { out.push_back(in_shape[d]); }
        }
        return out;   // may be empty (scalar) — caller handles that
    }

    // For each input dim d, the stride within `out` of the corresponding output
    // dim (0 along the reduced axis, since it collapses / becomes size 1).
    template <typename OutT>
    std::vector<std::size_t> in_to_out_strides(
        const std::vector<std::size_t>& in_shape, std::size_t ax, bool keepdims,
        const Tensor<OutT>& out) {
        const std::size_t rank = in_shape.size();
        std::vector<std::size_t> map(rank, 0);
        std::size_t out_dim = 0;
        for (std::size_t d = 0; d < rank; ++d) {
            if (d == ax) { if (keepdims) ++out_dim; continue; }   // reduced → 0
            map[d] = out.strides[out_dim];
            ++out_dim;
        }
        return map;
    }

    // Generic elementwise combine reduction (used by sum/max/min).
    template <typename T, typename Combine>
    Tensor<T> reduce_axis(const Tensor<T>& t, int axis, bool keepdims, T init, Combine comb) {
        const std::size_t rank = t.shape.size();
        const std::size_t ax = normalize_axis(axis, rank);

        auto out_shape = reduced_shape(t.shape, ax, keepdims);
        Tensor<T> out(out_shape.empty() ? std::vector<std::size_t>{1} : out_shape);
        std::fill(out.data.begin(), out.data.end(), init);

        auto map = in_to_out_strides(t.shape, ax, keepdims, out);

        const std::size_t total = t.data.size();
        for (std::size_t i = 0; i < total; ++i) {
            // Decompose contiguous index i into coords, accumulate into out cell.
            std::size_t out_off = 0, rem = i;
            for (std::size_t d = 0; d < rank; ++d) {
                std::size_t coord = (rem / t.strides[d]) % t.shape[d];
                out_off += coord * map[d];
            }
            out.data[out_off] = comb(out.data[out_off], t.data[i]);
        }
        return out;
    }

} // namespace reduce_detail

template <typename T>
Tensor<T> sum(const Tensor<T>& t, int axis, bool keepdims = false) {
    return reduce_detail::reduce_axis<T>(t, axis, keepdims, T{0},
                                         [](T a, T b) { return a + b; });
}

template <typename T>
Tensor<T> max(const Tensor<T>& t, int axis, bool keepdims = false) {
    return reduce_detail::reduce_axis<T>(t, axis, keepdims,
        std::numeric_limits<T>::lowest(), [](T a, T b) { return a > b ? a : b; });
}

template <typename T>
Tensor<T> min(const Tensor<T>& t, int axis, bool keepdims = false) {
    return reduce_detail::reduce_axis<T>(t, axis, keepdims,
        std::numeric_limits<T>::max(), [](T a, T b) { return a < b ? a : b; });
}

template <typename T>
Tensor<T> mean(const Tensor<T>& t, int axis, bool keepdims = false) {
    const std::size_t ax = reduce_detail::normalize_axis(axis, t.shape.size());
    Tensor<T> s = sum(t, axis, keepdims);
    const T n = static_cast<T>(t.shape[ax]);
    for (std::size_t i = 0; i < s.data.size(); ++i) s.data[i] /= n;
    return s;
}

// argmax along an axis → indices (Tensor<std::size_t>).
template <typename T>
Tensor<std::size_t> argmax(const Tensor<T>& t, int axis, bool keepdims = false) {
    const std::size_t rank = t.shape.size();
    const std::size_t ax = reduce_detail::normalize_axis(axis, rank);

    auto out_shape = reduce_detail::reduced_shape(t.shape, ax, keepdims);
    Tensor<std::size_t> out(out_shape.empty() ? std::vector<std::size_t>{1} : out_shape);

    // Track the best value seen per output cell alongside its axis index.
    std::vector<T> best(out.data.size(), std::numeric_limits<T>::lowest());
    auto map = reduce_detail::in_to_out_strides(t.shape, ax, keepdims, out);

    const std::size_t total = t.data.size();
    for (std::size_t i = 0; i < total; ++i) {
        std::size_t out_off = 0, axis_coord = 0, rem = i;
        for (std::size_t d = 0; d < rank; ++d) {
            std::size_t coord = (rem / t.strides[d]) % t.shape[d];
            if (d == ax) axis_coord = coord;
            out_off += coord * map[d];
        }
        if (t.data[i] > best[out_off]) {
            best[out_off] = t.data[i];
            out.data[out_off] = axis_coord;
        }
    }
    return out;
}

} // namespace tl
