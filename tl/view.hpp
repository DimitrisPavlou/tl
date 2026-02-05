/*#pragma once
#include <vector>

namespace tl {

template <typename T>
struct View {
    T* data_ptr;
    std::vector<int> shape;
    std::vector<int> strides;
    int dims_left;

    // Chaining operator [i][j]...
    View operator[](int index) {
        std::vector<int> next_shape(shape.begin() + 1, shape.end());
        std::vector<int> next_strides(strides.begin() + 1, strides.end());
        
        return View{ 
            data_ptr + (index * strides[0]), 
            next_shape, 
            next_strides, 
            dims_left - 1 
        };
    }

    // Accessors
    operator T&() const { return *data_ptr; }
    T& operator=(const T& val) { *data_ptr = val; return *data_ptr; }
};

} // namespace tl */


//#pragma once
//#include <vector>
//#include <algorithm> // For std::copy

/*namespace tl {

template <typename T>
struct View {
    T* data_ptr;
    std::vector<int> shape;
    std::vector<int> strides;
    int dims_left;

    // Helper to get total number of elements in this view
    size_t size() const {
        if (shape.empty()) return 0;
        size_t total = 1;
        for (int s : shape) total *= s;
        return total;
    }

    // Chaining operator [i][j]...
    View operator[](int index) {
        std::vector<int> next_shape(shape.begin() + 1, shape.end());
        std::vector<int> next_strides(strides.begin() + 1, strides.end());
        
        return View{ 
            data_ptr + (index * strides[0]), 
            next_shape, 
            next_strides, 
            dims_left - 1 
        };
    }

    // --- New Feature: Assignment from another View ---
    // This performs a data copy from the source view to this view's memory
    View& operator=(const View& other) {
        // Simple safety check: total elements must match
        if (this->size() != other.size()) {
            throw std::runtime_error("View assignment dimension mismatch");
        }
        
        // Since your current library only creates contiguous slices, 
        // we can use a simple pointer-based copy.
        std::copy(other.data_ptr, other.data_ptr + other.size(), this->data_ptr);
        return *this;
    }

    // Existing Accessors
    operator T&() const { return *data_ptr; }
    T& operator=(const T& val) { *data_ptr = val; return *data_ptr; }
};

} // namespace tl */

#pragma once
#include <vector>
#include <algorithm>
#include <cstddef> // Required for std::size_t

namespace tl {

template <typename T>
struct View {
    T* data_ptr;
    const std::size_t* shape_ptr;   // Use size_t for dimensions
    const std::size_t* strides_ptr; // Use size_t for strides
    std::size_t dims_left;          // Dimensions remaining in this view

    View operator[](std::size_t index) {
        // No heap allocations! Just pointer arithmetic.
        return View{ 
            data_ptr + (index * strides_ptr[0]), 
            shape_ptr + 1, 
            strides_ptr + 1, 
            dims_left - 1 
        };
    }

    // We return a View<T> but the T will be const if the original was const
    View operator[](std::size_t index) const {
        return View{ 
            data_ptr + (index * strides_ptr[0]), 
            shape_ptr + 1, 
            strides_ptr + 1, 
            dims_left - 1 
        };
    }

    operator T&() const { return *data_ptr; }
    T& operator=(const T& val) { *data_ptr = val; return *data_ptr; }
    
    // Assignment from another View (for tensor[i] = other)
    View& operator=(const View& other) {
        std::size_t size = 1;
        for(std::size_t i = 0; i < dims_left; ++i) size *= shape_ptr[i];
        std::copy(other.data_ptr, other.data_ptr + size, this->data_ptr);
        return *this;
    }
};

} // namespace tl