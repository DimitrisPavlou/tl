#pragma once

#include <iostream>
#include <string>
#include "tensor.hpp"

namespace tl {

// --- Commutative Scalar Math ---
template <typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& t) { return t * scalar; }

template <typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& t) { return t + scalar; }

// --- Factories ---
template <typename T>
Tensor<T> zeros(std::vector<int> shape) {
    Tensor<T> t(shape);
    std::fill(t.data.begin(), t.data.end(), static_cast<T>(0));
    return t;
}

template <typename T>
Tensor<T> ones(std::vector<int> shape) {
    Tensor<T> t(shape);
    std::fill(t.data.begin(), t.data.end(), static_cast<T>(1));
    return t;
}

// --- Reshape ---
template <typename T>
Tensor<T> reshape(const Tensor<T>& item, std::vector<int> new_shape) {
    // Check if volumes match
    int old_vol = 1, new_vol = 1;
    for (int s : item.shape) old_vol *= s;
    for (int s : new_shape) new_vol *= s;

    if (old_vol != new_vol) {
        throw std::runtime_error("Cannot reshape: total size must remain constant.");
    }

    Tensor<T> new_tensor = item; // Copy data
    new_tensor.shape = new_shape;
    new_tensor.recalculate_strides();
    return new_tensor;
}

// --- Print Logic ---
template <typename T>
void print_recursive(View<T> view, int indent = 0) {
    std::string space(indent, ' ');
    if (view.dims_left == 0) { std::cout << (T)view; return; }

    std::cout << "[";
    if (view.dims_left == 1) {
        for (int i = 0; i < view.shape[0]; ++i) 
            std::cout << view[i] << (i == view.shape[0] - 1 ? "" : " ");
    } else {
        for (int i = 0; i < view.shape[0]; ++i) {
            if (i > 0) std::cout << "\n" << space << " ";
            print_recursive(view[i], indent + 1);
        }
    }
    std::cout << "]";
}

template <typename T>
void print(const Tensor<T>& tensor) {
    if (tensor.data.empty()) { std::cout << "[]" << std::endl; return; }
    View<T> root{ const_cast<T*>(tensor.data.data()), tensor.shape, tensor.strides, (int)tensor.shape.size() };
    print_recursive(root);
    std::cout << std::endl;
}

} // namespace tl