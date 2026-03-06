#pragma once

#include <iostream>
#include <string>
#include "tensor.hpp"

// Note: #pragma omp simd hints require compiling with -fopenmp (GCC/Clang) or /openmp (MSVC).
// Without that flag the pragmas are silently ignored; all loops remain correct.

namespace tl {

// --- Commutative Scalar Math (scalar OP tensor) ---
// FIX: operator+ and operator* are commutative, so we can delegate.
// FIX: operator- and operator/ are NOT commutative — each element is computed correctly.

template <typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& t) { return t + scalar; }

template <typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& t) { return t * scalar; }

// scalar - tensor: result[i] = scalar - t[i]  (NOT t[i] - scalar)
template <typename T>
Tensor<T> operator-(T scalar, const Tensor<T>& t) {
    Tensor<T> res(t.shape);
    T* r = res.data.data();
    const T* a = t.data.data();
    const std::size_t n = t.data.size();
    for (std::size_t i = 0; i < n; ++i) r[i] = scalar - a[i];
    return res;
}

// scalar / tensor: result[i] = scalar / t[i]  (NOT t[i] / scalar)
template <typename T>
Tensor<T> operator/(T scalar, const Tensor<T>& t) {
    Tensor<T> res(t.shape);
    T* r = res.data.data();
    const T* a = t.data.data();
    const std::size_t n = t.data.size();
    for (std::size_t i = 0; i < n; ++i) r[i] = scalar / a[i];
    return res;
}


// --- Factories ---

// FIX: shape parameter is now const& so temporaries like zeros<float>({2,3}) compile.
template <typename T>
Tensor<T> zeros(const std::vector<std::size_t>& shape) {
    Tensor<T> t(shape);
    std::fill(t.data.begin(), t.data.end(), static_cast<T>(0));
    return t;
}

template <typename T>
Tensor<T> ones(const std::vector<std::size_t>& shape) {
    Tensor<T> t(shape);
    std::fill(t.data.begin(), t.data.end(), static_cast<T>(1));
    return t;
}

template <typename T>
Tensor<T> full(const std::vector<std::size_t>& shape, T value) {
    Tensor<T> t(shape);
    std::fill(t.data.begin(), t.data.end(), value);
    return t;
}


// --- Reshape ---
template <typename T>
Tensor<T> reshape(const Tensor<T>& item, std::vector<std::size_t> new_shape) {
    std::size_t old_vol = 1, new_vol = 1;
    for (auto s : item.shape) old_vol *= s;
    for (auto s : new_shape) new_vol *= s;

    if (old_vol != new_vol) {
        throw std::runtime_error("Cannot reshape: total size must remain constant.");
    }

    Tensor<T> new_tensor = item;
    new_tensor.shape = new_shape;
    new_tensor.recalculate_strides();
    return new_tensor;
}


// --- Dot Product ---
template <typename T>
T dot(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape.size() != 1 || b.shape.size() != 1) {
        throw std::runtime_error("dot product requires 1D tensors.");
    }
    if (a.shape[0] != b.shape[0]) {
        throw std::runtime_error("Vectors must be the same length.");
    }

    T result = static_cast<T>(0);
    const T* ptr_a = a.data.data();
    const T* ptr_b = b.data.data();
    const std::size_t n = a.data.size();

    for (std::size_t i = 0; i < n; ++i) {
        result += ptr_a[i] * ptr_b[i];
    }
    return result;
}


// --- Reductions ---

// FIX: sum returns T (was already correct, kept as-is).
template <typename T>
T sum(const Tensor<T>& t) {
    T total = static_cast<T>(0);
    const T* ptr = t.data.data();
    const std::size_t n = t.data.size();

    #pragma omp simd reduction(+:total)
    for (std::size_t i = 0; i < n; ++i) {
        total += ptr[i];
    }
    return total;
}

// FIX: mean now returns T instead of always float, preserving double precision.
template <typename T>
T mean(const Tensor<T>& t) {
    if (t.data.empty()) {
        throw std::runtime_error("Cannot compute mean of empty tensor");
    }
    return sum(t) / static_cast<T>(t.data.size());
}

template <typename T>
T max(const Tensor<T>& t) {
    if (t.data.empty()) {
        throw std::runtime_error("Cannot compute max of empty tensor");
    }
    return *std::max_element(t.data.begin(), t.data.end());
}

template <typename T>
T min(const Tensor<T>& t) {
    if (t.data.empty()) {
        throw std::runtime_error("Cannot compute min of empty tensor");
    }
    return *std::min_element(t.data.begin(), t.data.end());
}


// --- Recursive Print ---
template <typename T>
void print_recursive(View<T> view, int indent = 0) {
    std::string space(indent, ' ');
    if (view.dims_left == 0) {
        std::cout << (T)view;
        return;
    }

    std::cout << "[";
    if (view.dims_left == 1) {
        for (std::size_t i = 0; i < view.shape_ptr[0]; ++i)
            std::cout << view[i] << (i == view.shape_ptr[0] - 1 ? "" : " ");
    } else {
        std::cout << "\n";
        for (std::size_t i = 0; i < view.shape_ptr[0]; ++i) {
            std::cout << space << "  ";
            print_recursive(view[i], indent + 2);
            if (i != view.shape_ptr[0] - 1) std::cout << ",\n";
        }
        std::cout << "\n" << space;
    }
    std::cout << "]";
}

template <typename T>
void print(const Tensor<T>& tensor) {
    View<const T> root{
        tensor.data.data(),
        tensor.shape.data(),
        tensor.strides.data(),
        tensor.shape.size()
    };
    print_recursive(root);
    std::cout << std::endl;
}

} // namespace tl
