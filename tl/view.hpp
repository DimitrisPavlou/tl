#pragma once
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

} // namespace tl