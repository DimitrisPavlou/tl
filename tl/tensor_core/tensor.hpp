#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include "view.hpp"
#include "broadcasting.hpp"

namespace tl {

template <typename T>
class Tensor {
public:
    using value_type = T;   // enables decltype(tensor)::value_type in tests and generic code

    std::vector<T> data;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;

    // Shape-only constructor: allocates zero-initialized data.
    Tensor(std::vector<std::size_t> s) : shape(std::move(s)) {
        std::size_t total_size = 1;
        for (auto dim : shape) total_size *= dim;
        data.resize(total_size);
        recalculate_strides();
    }

    // NumPy-style 2D nested init: Tensor t({{1,2}, {3,4}});
    Tensor(std::initializer_list<std::initializer_list<T>> list) {
        std::size_t rows = list.size();
        std::size_t cols = (rows > 0) ? list.begin()->size() : 0;
        shape = {rows, cols};
        data.reserve(rows * cols);
        for (auto& row : list) {
            if (row.size() != cols) throw std::runtime_error("Inconsistent row lengths");
            data.insert(data.end(), row.begin(), row.end());
        }
        recalculate_strides();
    }

    // Flat data + shape constructor: Tensor t({2,2}, {1,2,3,4});
    // FIX: validates that the number of data elements matches the shape.
    Tensor(std::vector<std::size_t> s, std::initializer_list<T> d)
        : data(d), shape(std::move(s)) {
        std::size_t expected = 1;
        for (auto dim : shape) expected *= dim;
        if (data.size() != expected) {
            throw std::runtime_error(
                "Shape/data mismatch: shape implies " + std::to_string(expected) +
                " elements, but " + std::to_string(data.size()) + " were provided.");
        }
        recalculate_strides();
    }

    void recalculate_strides() {
        strides.resize(shape.size());
        std::size_t stride = 1;
        for (std::size_t i = shape.size(); i-- > 0; ) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // Indexing with bounds checking (FIX: throws for 0D tensors and out-of-range)
    View<T> operator[](std::size_t i) {
        if (shape.empty()) {
            throw std::out_of_range("Cannot index a 0-dimensional tensor (scalar)");
        }
        if (i >= shape[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(i) +
                " out of range for dimension of size " + std::to_string(shape[0]));
        }
        return View<T>{ &data[i * strides[0]], &shape[1], &strides[1], shape.size() - 1 };
    }

    const View<const T> operator[](std::size_t i) const {
        if (shape.empty()) {
            throw std::out_of_range("Cannot index a 0-dimensional tensor (scalar)");
        }
        if (i >= shape[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(i) +
                " out of range for dimension of size " + std::to_string(shape[0]));
        }
        return View<const T>{ const_cast<T*>(&data[i * strides[0]]), &shape[1], &strides[1], shape.size() - 1 };
    }

    // --- Element-wise tensor operators (with broadcasting) ---
    // Fast path: identical shapes → direct loop, no overhead.
    // Broadcast path: different shapes → stride-based multi-dimensional loop.

    Tensor operator+(const Tensor& other) const {
        return broadcast_apply(other, [](T a, T b){ return a + b; });
    }

    Tensor operator-(const Tensor& other) const {
        return broadcast_apply(other, [](T a, T b){ return a - b; });
    }

    Tensor operator*(const Tensor& other) const {
        return broadcast_apply(other, [](T a, T b){ return a * b; });
    }

    Tensor operator/(const Tensor& other) const {
        return broadcast_apply(other, [](T a, T b){ return a / b; });
    }

    // --- Element-wise scalar operators ---

    Tensor operator+(T scalar) const {
        Tensor res(shape);
        T* r = res.data.data();
        const T* a = this->data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) r[i] = a[i] + scalar;
        return res;
    }

    Tensor operator*(T scalar) const {
        Tensor res(shape);
        T* r = res.data.data();
        const T* a = this->data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) r[i] = a[i] * scalar;
        return res;
    }

    Tensor operator-(T scalar) const {
        Tensor res(shape);
        T* r = res.data.data();
        const T* a = this->data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) r[i] = a[i] - scalar;
        return res;
    }

    Tensor operator/(T scalar) const {
        Tensor res(shape);
        T* r = res.data.data();
        const T* a = this->data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) r[i] = a[i] / scalar;
        return res;
    }

    // --- In-place tensor operators ---

    Tensor& operator+=(const Tensor& other) {
        check_shape(other);
        T* a = this->data.data();
        const T* b = other.data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) a[i] += b[i];
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        check_shape(other);
        T* a = this->data.data();
        const T* b = other.data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) a[i] -= b[i];
        return *this;
    }

    Tensor& operator*=(const Tensor& other) {
        check_shape(other);
        T* a = this->data.data();
        const T* b = other.data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) a[i] *= b[i];
        return *this;
    }

    Tensor& operator/=(const Tensor& other) {
        check_shape(other);
        T* a = this->data.data();
        const T* b = other.data.data();
        const std::size_t n = data.size();
        for (std::size_t i = 0; i < n; ++i) a[i] /= b[i];
        return *this;
    }

    // --- In-place scalar operators ---
    // Note: #pragma omp simd requires compiling with -fopenmp (GCC/Clang) or /openmp (MSVC).
    // Without that flag the pragma is silently ignored; the loops are still correct.

    Tensor& operator+=(T scalar) {
        T* a = this->data.data();
        const std::size_t n = data.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) a[i] += scalar;
        return *this;
    }

    Tensor& operator-=(T scalar) {
        T* a = this->data.data();
        const std::size_t n = data.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) a[i] -= scalar;
        return *this;
    }

    Tensor& operator*=(T scalar) {
        T* a = this->data.data();
        const std::size_t n = data.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) a[i] *= scalar;
        return *this;
    }

    Tensor& operator/=(T scalar) {
        T* a = this->data.data();
        const std::size_t n = data.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) a[i] /= scalar;
        return *this;
    }

    // --- Rule of Five ---
    ~Tensor() = default;
    Tensor(const Tensor& other) : data(other.data), shape(other.shape), strides(other.strides) {}
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            data = other.data;
            shape = other.shape;
            strides = other.strides;
        }
        return *this;
    }
    Tensor(Tensor&& other) noexcept
        : data(std::move(other.data)), shape(std::move(other.shape)), strides(std::move(other.strides)) {}

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            shape = std::move(other.shape);
            strides = std::move(other.strides);
        }
        return *this;
    }

    // Implicit conversion to View (used by linalg and print utilities)
    operator View<T>() {
        return View<T>{ data.data(), shape.data(), strides.data(), shape.size() };
    }

    View<T> view() {
        return static_cast<View<T>>(*this);
    }

    operator View<const T>() const {
        return View<const T>{ const_cast<T*>(data.data()), shape.data(), strides.data(), shape.size() };
    }

private:
    void check_shape(const Tensor& other) const {
        if (shape != other.shape) throw std::runtime_error("Shape mismatch");
    }

    // Core broadcasting engine.
    // Op is a binary functor (T, T) -> T.
    template <typename Op>
    Tensor broadcast_apply(const Tensor& other, Op op) const {
        // Fast path: identical shapes — original direct loop, zero overhead.
        if (shape == other.shape) {
            Tensor res(shape);
            T* r = res.data.data();
            const T* a = data.data();
            const T* b = other.data.data();
            const std::size_t n = data.size();
            for (std::size_t i = 0; i < n; ++i) r[i] = op(a[i], b[i]);
            return res;
        }

        // Broadcast path: compute output shape and per-tensor broadcasted strides.
        std::vector<std::size_t> out_shape = compute_broadcast_shape(shape, other.shape);
        std::vector<std::size_t> str_a = get_broadcast_strides(shape, strides, out_shape);
        std::vector<std::size_t> str_b = get_broadcast_strides(other.shape, other.strides, out_shape);

        Tensor res(out_shape);
        const std::size_t rank = out_shape.size();

        // Compute total output elements
        std::size_t total = 1;
        for (auto d : out_shape) total *= d;

        for (std::size_t flat = 0; flat < total; ++flat) {
            // Convert flat index to multi-dimensional coordinates, then
            // compute the linear offset into each input using its (possibly 0) strides.
            std::size_t off_a = 0, off_b = 0;
            std::size_t remaining = flat;
            for (std::size_t d = rank; d-- > 0; ) {
                std::size_t coord = remaining % out_shape[d];
                remaining /= out_shape[d];
                off_a += coord * str_a[d];
                off_b += coord * str_b[d];
            }
            res.data[flat] = op(data[off_a], other.data[off_b]);
        }
        return res;
    }
};

} // namespace tl
