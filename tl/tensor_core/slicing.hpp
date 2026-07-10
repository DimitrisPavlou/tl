#pragma once

#include "tensor.hpp"
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <limits>

// Range-based slicing (roadmap item). Returns a MATERIALISED contiguous tensor
// (a copy), which is always correct regardless of how downstream ops iterate.
// A zero-copy strided-view slice is a future optimisation that depends on the
// eager ops becoming stride-aware.
//
//   auto minibatch = tl::slice(X, {tl::Slice(0, 32)});      // rows 0..31
//   auto block     = tl::slice(M, {tl::Slice(1, 3), tl::Slice(0, 2)});
namespace tl {

struct Slice {
    std::size_t start;
    std::size_t stop;
    std::size_t step;
    explicit Slice(std::size_t start_ = 0,
                   std::size_t stop_  = std::numeric_limits<std::size_t>::max(),
                   std::size_t step_  = 1)
        : start(start_), stop(stop_), step(step_) {}
};

template <typename T>
Tensor<T> slice(const Tensor<T>& t, std::vector<Slice> slices) {
    const std::size_t rank = t.shape.size();
    if (slices.size() > rank)
        throw std::runtime_error("slice: more slice specs than tensor dimensions");

    // Pad missing trailing dimensions with full-range slices.
    while (slices.size() < rank) slices.emplace_back(0, t.shape[slices.size()], 1);

    std::vector<std::size_t> out_shape(rank);
    for (std::size_t d = 0; d < rank; ++d) {
        std::size_t stop = std::min(slices[d].stop, t.shape[d]);
        std::size_t start = slices[d].start;
        std::size_t step = slices[d].step == 0 ? 1 : slices[d].step;
        if (start > t.shape[d]) start = t.shape[d];
        out_shape[d] = (stop > start) ? (stop - start + step - 1) / step : 0;
        slices[d] = Slice(start, stop, step);
    }

    Tensor<T> out(out_shape);
    if (out.data.empty()) return out;

    const std::size_t total = out.data.size();
    for (std::size_t j = 0; j < total; ++j) {
        // Decompose the output index into coords, map back to input coords.
        std::size_t in_off = 0, rem = j;
        for (std::size_t d = 0; d < rank; ++d) {
            std::size_t out_coord = (rem / out.strides[d]) % out_shape[d];
            std::size_t in_coord  = slices[d].start + out_coord * slices[d].step;
            in_off += in_coord * t.strides[d];
        }
        out.data[j] = t.data[in_off];
    }
    return out;
}

} // namespace tl
