#pragma once 

#include "../tensor_core/tensor.hpp"
#include <cmath>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace tl {
namespace linalg {

    // Matrix multiplication (optimized version)
    template <typename T>
    Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B) {
        if (A.shape.size() != 2 || B.shape.size() != 2) {
            throw std::runtime_error("matmul currently supports 2D matrices only.");
        }
        if (A.shape[1] != B.shape[0]) {
            throw std::runtime_error("Inner dimensions must match for matmul.");
        }
        
        const std::size_t M = A.shape[0];
        const std::size_t K = A.shape[1];
        const std::size_t N = B.shape[1];
        
        Tensor<T> C({M, N});
        std::fill(C.data.begin(), C.data.end(), static_cast<T>(0));

        // Optimized i-k-j order for cache efficiency
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const T a_ik = A.data[i * K + k];
                T* c_row = &C.data[i * N];
                const T* b_row = &B.data[k * N];
                
                // Vectorizable inner loop
                #pragma omp simd
                for (std::size_t j = 0; j < N; ++j) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
        return C;
    }

    // Matrix norm (optimized)
    template <typename T>
    double matrix_norm(const Tensor<T>& A, const std::string& type = "frob") {
        if (A.shape.size() != 2) {
            throw std::runtime_error("matrix_norm currently supports 2D matrices only.");
        }
        
        const std::size_t rows = A.shape[0];
        const std::size_t cols = A.shape[1];
        
        if (type == "frob" || type == "fro") {
            // Frobenius norm: sqrt(sum of squared elements)
            double sum_squares = 0.0;
            const T* data_ptr = A.data.data();
            const std::size_t n = A.data.size();
            
            for (std::size_t i = 0; i < n; ++i) {
                double val = static_cast<double>(data_ptr[i]);
                sum_squares += val * val;
            }
            return std::sqrt(sum_squares);

        } else if (type == "1") {
            // 1-norm: max column sum
            double max_sum = 0.0;
            for (std::size_t j = 0; j < cols; ++j) {
                double col_sum = 0.0;
                for (std::size_t i = 0; i < rows; ++i) {
                    col_sum += std::abs(static_cast<double>(A.data[i * cols + j]));
                }
                if (col_sum > max_sum) {
                    max_sum = col_sum;
                }
            }
            return max_sum;

        } else if (type == "inf") {
            // Infinity norm: max row sum
            double max_sum = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                double row_sum = 0.0;
                for (std::size_t j = 0; j < cols; ++j) {
                    row_sum += std::abs(static_cast<double>(A.data[i * cols + j]));
                }
                if (row_sum > max_sum) {
                    max_sum = row_sum;
                }
            }
            return max_sum;

        } else {
            throw std::runtime_error("Unsupported norm type: " + type + 
                                   ". Supported types: 'frob', '1', 'inf'");
        }
    }

    // Transpose
    template <typename T>
    Tensor<T> transpose(const Tensor<T>& A) {
        if (A.shape.size() != 2) {
            throw std::runtime_error("transpose currently supports 2D matrices only.");
        }
        
        const std::size_t rows = A.shape[0];
        const std::size_t cols = A.shape[1];
        Tensor<T> result({cols, rows});
        
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result.data[j * rows + i] = A.data[i * cols + j];
            }
        }
        
        return result;
    }

    // Identity matrix
    template <typename T>
    Tensor<T> eye(std::size_t n) {
        Tensor<T> result({n, n});
        std::fill(result.data.begin(), result.data.end(), static_cast<T>(0));
        
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i * n + i] = static_cast<T>(1);
        }
        
        return result;
    }

    // Trace (sum of diagonal elements)
    template <typename T>
    T trace(const Tensor<T>& A) {
        if (A.shape.size() != 2) {
            throw std::runtime_error("trace requires a 2D matrix.");
        }
        if (A.shape[0] != A.shape[1]) {
            throw std::runtime_error("trace requires a square matrix.");
        }
        
        T sum = static_cast<T>(0);
        const std::size_t n = A.shape[0];
        
        for (std::size_t i = 0; i < n; ++i) {
            sum += A.data[i * n + i];
        }
        
        return sum;
    }

} // namespace linalg
} // namespace tl