#pragma once
#include <vector>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace tl {

template <typename T>
struct View {
    T* data_ptr;
    const std::size_t* shape_ptr;   // Points into the owning Tensor's shape array
    const std::size_t* strides_ptr; // Points into the owning Tensor's strides array
    std::size_t dims_left;          // Dimensions remaining in this view

    View operator[](std::size_t index) {
        if (dims_left == 0) {
            throw std::out_of_range("Cannot index a 0-dimensional view (scalar)");
        }
        if (index >= shape_ptr[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(index) +
                " out of range for dimension of size " + std::to_string(shape_ptr[0]));
        }
        return View{
            data_ptr + (index * strides_ptr[0]),
            shape_ptr + 1,
            strides_ptr + 1,
            dims_left - 1
        };
    }

    View operator[](std::size_t index) const {
        if (dims_left == 0) {
            throw std::out_of_range("Cannot index a 0-dimensional view (scalar)");
        }
        if (index >= shape_ptr[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(index) +
                " out of range for dimension of size " + std::to_string(shape_ptr[0]));
        }
        return View{
            data_ptr + (index * strides_ptr[0]),
            shape_ptr + 1,
            strides_ptr + 1,
            dims_left - 1
        };
    }

    // Implicit conversion to element reference (used when dims_left == 0)
    operator T&() const { return *data_ptr; }

    // Scalar assignment: tensor[i][j] = value
    T& operator=(const T& val) { *data_ptr = val; return *data_ptr; }

    // View-to-view assignment: copies all elements from other into this view's memory.
    // Validates that both views cover the same total number of elements.
    View& operator=(const View& other) {
        std::size_t my_size = 1;
        for (std::size_t i = 0; i < dims_left; ++i) my_size *= shape_ptr[i];

        std::size_t other_size = 1;
        for (std::size_t i = 0; i < other.dims_left; ++i) other_size *= other.shape_ptr[i];

        if (my_size != other_size) {
            throw std::runtime_error(
                "View assignment size mismatch: destination has " + std::to_string(my_size) +
                " elements, source has " + std::to_string(other_size));
        }
        std::copy(other.data_ptr, other.data_ptr + my_size, this->data_ptr);
        return *this;
    }
};

} // namespace tl
