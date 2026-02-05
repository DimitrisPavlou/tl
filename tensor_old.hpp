#pragma once

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace tl {

// --- Forward Declarations ---
template <typename T> struct View;

// --- Tensor Class Definition ---
template <typename T>
class Tensor {
    public:
        std::vector<T> data;
        std::vector<int> shape;
        std::vector<int> strides;

        Tensor(std::vector<int> s) : shape(s) {
            int total_size = 1;
            for (auto dim : shape) total_size *= dim;
            data.resize(total_size);
            
            recalculate_strides();
        }

        void recalculate_strides() {
            strides.resize(shape.size());
            int stride = 1;
            for (int i = (int)shape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }

        // High-level access (original)
        T& operator()(int i, int j) {
            return data[i * strides[0] + j * strides[1]];
        }

        const T& operator()(int i, int j) const {
            return data[i * strides[0] + j * strides[1]];
        }

        // Entry point for array[i][j][k][l]
        View<T> operator[](int i);
        const View<const T> operator[](int i) const;

        // Basic Math
        Tensor operator+(const Tensor& other) const {
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }

        Tensor operator*(T scalar) const {
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] * scalar;
            }
            return result;
        }

        // Addition by scalar: tensor + 5
        Tensor operator+(T scalar) const {
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] + scalar;
            }
            return result;
        }

        // --- Element-Wise Operations (Tensor vs Tensor) ---

        // Subtraction: A - B
        Tensor operator-(const Tensor& other) const {
            check_shape(other);
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }

        // Element-wise Multiplication (Hadamard Product): A * B
        Tensor operator*(const Tensor& other) const {
            check_shape(other);
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] * other.data[i];
            }
            return result;
        }

        // Element-wise Division: A / B
        Tensor operator/(const Tensor& other) const {
            check_shape(other);
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] / other.data[i];
            }
            return result;
        }

        // In-place Addition: A += B for efficiency
        Tensor& operator+=(const Tensor& other) {
            check_shape(other);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] += other.data[i];
            }
            return *this;
        }

    private:
        void check_shape(const Tensor& other) const {
            if (shape != other.shape) {
                throw std::runtime_error("Tensor shapes must match for element-wise operations.");
            }
        }
};

// --- View Struct Definition ---
// This handles the recursive [i][j] indexing
template <typename T>
struct View {
    T* data_ptr;
    std::vector<int> shape;   // Added to track sub-geometry
    std::vector<int> strides; // Added to track sub-geometry
    int dims_left;

    // Chaining operator
    View operator[](int index) {
        // Create a new view with one less dimension
        std::vector<int> next_shape(shape.begin() + 1, shape.end());
        std::vector<int> next_strides(strides.begin() + 1, strides.end());
        
        return View{ 
            data_ptr + (index * strides[0]), 
            next_shape, 
            next_strides, 
            dims_left - 1 
        };
    }

    operator T&() const { return *data_ptr; }
    T& operator=(const T& val) { *data_ptr = val; return *data_ptr; }
};

// --- Implement Tensor Indexing (must be after View definition) ---
template <typename T>
View<T> Tensor<T>::operator[](int i) {
    return View<T>{ &data[i * strides[0]], &strides[1], (int)shape.size() - 1 };
}

template <typename T>
const View<const T> Tensor<T>::operator[](int i) const {
    return View<const T>{ &data[i * strides[0]], &strides[1], (int)shape.size() - 1 };
}

// --- Namespace Factory Functions ---

template <typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& tensor) {
    return tensor * scalar; // Reuses the member operator
}

template <typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& tensor) {
    return tensor + scalar;
}

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

// Generic print for 2D slices (can be extended for ND)
template <typename T>
void print_recursive(View<T> view, int indent = 0) {
    std::string space(indent, ' ');
    
    if (view.dims_left == 0) {
        // Base case: This is a single scalar value
        std::cout << (T)view;
        return;
    }

    std::cout << "[";
    
    // If we are at the last dimension (1D row), print elements separated by spaces
    if (view.dims_left == 1) {
        for (int i = 0; i < view.shape[0]; ++i) {
            std::cout << view[i] << (i == view.shape[0] - 1 ? "" : " ");
        }
    } else {
        // For higher dimensions, recurse and add newlines
        for (int i = 0; i < view.shape[0]; ++i) {
            if (i > 0) std::cout << "\n" << space << " ";
            print_recursive(view[i], indent + 1);
        }
    }
    
    std::cout << "]";
}

// Entry point for the user
template <typename T>
void print(const Tensor<T>& tensor) {
    if (tensor.data.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    // Initialize root_view with all metadata from the tensor
    View<T> root_view{ 
        const_cast<T*>(tensor.data.data()), 
        tensor.shape, 
        tensor.strides, 
        (int)tensor.shape.size() 
    };
    
    print_recursive(root_view);
    std::cout << std::endl;
}

} // namespace tl