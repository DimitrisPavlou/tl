#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include "view.hpp"

namespace tl {

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

    // Indexing
    View<T> operator[](int i) {
        // We pass the slices of shape/strides starting from the 2nd element
        std::vector<int> next_shape(shape.begin() + 1, shape.end());
        std::vector<int> next_strides(strides.begin() + 1, strides.end());
        return View<T>{ &data[i * strides[0]], next_shape, next_strides, (int)shape.size() - 1 };
    }

    const View<const T> operator[](int i) const {
        std::vector<int> next_shape(shape.begin() + 1, shape.end());
        std::vector<int> next_strides(strides.begin() + 1, strides.end());
        return View<const T>{ const_cast<T*>(&data[i * strides[0]]), next_shape, next_strides, (int)shape.size() - 1 };
    }

    // element wise math Operators
    Tensor operator+(const Tensor& other) const { check_shape(other); Tensor res(shape); 
        for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] + other.data[i]; return res; }
    
    Tensor operator-(const Tensor& other) const { check_shape(other); Tensor res(shape); 
        for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] - other.data[i]; return res; }

    Tensor operator*(const Tensor& other) const { check_shape(other); Tensor res(shape); 
        for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] * other.data[i]; return res; }

    Tensor operator/(const Tensor& other) const { check_shape(other); Tensor res(shape); 
        for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] / other.data[i]; return res; }

    //element-wise with scalar
    Tensor operator*(T scalar) const { Tensor res(shape); 
        for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] * scalar; return res; }

    Tensor operator+(T scalar) const { Tensor res(shape); 
        for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] + scalar; return res; }
    
    // in-place addition for efficiency
    Tensor& operator+=(const Tensor& other) {
        check_shape(other);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        check_shape(other);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }
   

private:
    void check_shape(const Tensor& other) const {
        if (shape != other.shape) throw std::runtime_error("Shape mismatch");
    }
};

} // namespace tl