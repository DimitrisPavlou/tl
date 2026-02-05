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
Tensor<T> zeros(std::vector<std::size_t> shape) { // Changed int to std::size_t
    Tensor<T> t(shape);
    std::fill(t.data.begin(), t.data.end(), static_cast<T>(0));
    return t;
}

template <typename T>
Tensor<T> ones(std::vector<std::size_t> shape) { // Changed int to std::size_t
    Tensor<T> t(shape);
    std::fill(t.data.begin(), t.data.end(), static_cast<T>(1));
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

template <typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B) {
    // 1. Validation
    if (A.shape.size() != 2 || B.shape.size() != 2) {
        throw std::runtime_error("matmul currently supports 2D matrices only.");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::runtime_error("Inner dimensions must match for matmul.");
    }

    std::size_t M = A.shape[0];
    std::size_t K = A.shape[1];
    std::size_t N = B.shape[1];

    // 2. Result allocation (initialized to zero)
    Tensor<T> C({M, N}); 
    
    // 3. Optimized Triple Loop
    // We use i-k-j order to maximize cache hits
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            // A[i][k] is constant for the entire inner loop over j
            T val_a = (T)A[i][k]; 
            
            // Raw pointers for the fastest possible inner loop
            T* res_row = &C.data[i * N];
            const T* b_row = &B.data[k * N];

            for (std::size_t j = 0; j < N; ++j) {
                res_row[j] += val_a * b_row[j];
            }
        }
    }
    return C;
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
        for (std::size_t i = 0; i < view.shape_ptr[0]; ++i) // Use shape_ptr[0]
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
    // CHANGE: Added .data() to shape and strides
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