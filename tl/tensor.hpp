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
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;

    Tensor(std::vector<std::size_t> s) : shape(s) {
        std::size_t total_size = 1;
        for (auto dim : shape) total_size *= dim;
        data.resize(total_size);
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

    // Indexing
    View<T> operator[](std::size_t i) {
        return View<T>{ &data[i * strides[0]], &shape[1], &strides[1], shape.size() - 1 };
    }

    const View<const T> operator[](std::size_t i) const {
        return View<const T>{ const_cast<T*>(&data[i * strides[0]]), &shape[1], &strides[1], shape.size() - 1 };
    }


    // element wise math Operators
    //Tensor operator+(const Tensor& other) const { check_shape(other); Tensor res(shape); 
    //    for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] + other.data[i]; return res; }
    
    //Tensor operator-(const Tensor& other) const { check_shape(other); Tensor res(shape); 
    //    for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] - other.data[i]; return res; }
    
    //Tensor operator*(const Tensor& other) const { check_shape(other); Tensor res(shape); 
    //    for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] * other.data[i]; return res; }

    //Tensor operator/(const Tensor& other) const { check_shape(other); Tensor res(shape); 
    //    for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] / other.data[i]; return res; }
    
    //element-wise with scalar
    //Tensor operator*(T scalar) const { Tensor res(shape); 
    //    for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] * scalar; return res; }

    //Tensor operator+(T scalar) const { Tensor res(shape); 
    //    for(size_t i=0; i<data.size(); ++i) res.data[i] = data[i] + scalar; return res; }


    //faster implementation of element-wise addition using raw pointers and potential for vectorization
    Tensor operator+(const Tensor& other) const {
        check_shape(other);
        Tensor res(shape); // Still has zero-init overhead
        T* res_ptr = res.data.data();
        const T* a_ptr = this->data.data();
        const T* b_ptr = other.data.data();
        std::size_t n = data.size();

        //#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            res_ptr[i] = a_ptr[i] + b_ptr[i];
        }
        return res;
    }

    Tensor operator-(const Tensor& other) const {
        check_shape(other);
        Tensor res(shape); // Still has zero-init overhead
        T* res_ptr = res.data.data();
        const T* a_ptr = this->data.data();
        const T* b_ptr = other.data.data();
        std::size_t n = data.size();

        //#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            res_ptr[i] = a_ptr[i] - b_ptr[i];
        }
        return res;
    }


    Tensor operator*(const Tensor& other) const {
        check_shape(other);
        Tensor res(shape); // Still has zero-init overhead
        T* res_ptr = res.data.data();
        const T* a_ptr = this->data.data();
        const T* b_ptr = other.data.data();
        std::size_t n = data.size();

        //#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            res_ptr[i] = a_ptr[i] * b_ptr[i];
        }
        return res;
    }

    Tensor operator/(const Tensor& other) const {
        check_shape(other);
        Tensor res(shape); // Still has zero-init overhead
        T* res_ptr = res.data.data();
        const T* a_ptr = this->data.data();
        const T* b_ptr = other.data.data();
        std::size_t n = data.size();

        //#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            res_ptr[i] = a_ptr[i] / b_ptr[i];
        }
        return res;
    }
    
    //element-wise with scalar
    Tensor operator+(T scalar) const {
        Tensor res(shape);         
        // 2. Extract raw pointers to help the compiler vectorize
        T* r = res.data.data();
        const T* a = this->data.data();
        std::size_t n = data.size();

        // 3. Simple loop that the compiler can easily turn into AVX instructions
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = a[i] + scalar;
        }
        return res;
    }

    Tensor operator*(T scalar) const {
        // 1. Allocate without zero-filling the memory
        Tensor res(shape); 
        
        // 2. Extract raw pointers to help the compiler vectorize
        T* r = res.data.data();
        const T* a = this->data.data();
        std::size_t n = data.size();

        // 3. Simple loop that the compiler can easily turn into AVX instructions
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = a[i] * scalar;
        }
        return res;
    }

    // in-place addition for efficiency
    Tensor& operator+=(const Tensor& other) {
        check_shape(other);
        T* a = this->data.data();
        const T* b = other.data.data();
        std::size_t n = data.size();
        //#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) a[i] += b[i];
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        check_shape(other);
        T* a = this->data.data();
        const T* b = other.data.data();
        std::size_t n = data.size();
        
        for (std::size_t i = 0; i < n; ++i) a[i] -= b[i];
        return *this;
    }


    // Rule of Five 
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

    // This allows Tensor to be assigned to a View: tensor[i] = other_tensor
    operator View<T>() {
        return View<T>{ data.data(), shape, strides, (int)shape.size() };
    }

    // Explicitly get a view if needed
    View<T> view() {
        return static_cast<View<T>>(*this);
    }

    // Also provide a const version
    operator View<const T>() const {
        return View<const T>{ const_cast<T*>(data.data()), shape, strides, (int)shape.size() };
    }

    
private:
    void check_shape(const Tensor& other) const {
        if (shape != other.shape) throw std::runtime_error("Shape mismatch");
    }
};

} // namespace tl