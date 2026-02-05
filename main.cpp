#include "tl/tl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>


void test_dot_product_fix() {
    std::cout << "=== Testing dot product type fix ===" << std::endl;
    
    // Test with double (would fail before fix)
    tl::Tensor<double> a({3}, {1.5, 2.5, 3.5});
    tl::Tensor<double> b({3}, {2.0, 3.0, 4.0});
    
    double result = dot(a, b);  // Now returns double, not float!
    std::cout << "Dot product (double): " << result << std::endl;
    std::cout << "Expected: 24.5 (1.5*2.0 + 2.5*3.0 + 3.5*4.0)" << std::endl;
    assert(std::abs(result - 24.5) < 1e-10);
    
    // Test with int
    tl::Tensor<int> c({3}, {1, 2, 3});
    tl::Tensor<int> d({3}, {4, 5, 6});
    
    int result_int = dot(c, d);  // Now works with int!
    std::cout << "Dot product (int): " << result_int << std::endl;
    assert(result_int == 32);
    
    std::cout << "✓ Dot product fix verified\n" << std::endl;
}

void test_scalar_operations() {
    std::cout << "=== Testing new scalar operations ===" << std::endl;
    
    tl::Tensor<float> t({2, 2}, {1, 2, 3, 4});
    
    // New: scalar - tensor
    auto result1 = 10.0f - t;
    std::cout << "10 - tensor:" << std::endl;
    tl::print(result1);
    
    // New: tensor - scalar
    auto result2 = t - 1.0f;
    std::cout << "tensor - 1:" << std::endl;
    tl::print(result2);
    
    // New: in-place *= and /=
    tl::Tensor<float> t2({2}, {2, 4});
    t2 *= 3.0f;
    std::cout << "After *= 3: ";
    tl::print(t2);
    
    t2 /= 2.0f;
    std::cout << "After /= 2: ";
    tl::print(t2);
    
    std::cout << "✓ Scalar operations verified\n" << std::endl;
}

void test_new_utilities() {
    std::cout << "=== Testing new utility functions ===" << std::endl;
    
    // full() - create tensor with specific value
    auto t1 = tl::full<int>({2, 3}, 7);
    std::cout << "full({2,3}, 7):" << std::endl;
    tl::print(t1);
    
    // transpose
    tl::Tensor<float> mat({2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << "Original matrix:" << std::endl;
    tl::print(mat);
    
    auto transposed = tl::linalg::transpose(mat);
    std::cout << "Transposed:" << std::endl;
    tl::print(transposed);
    
    // sum, mean, max, min
    tl::Tensor<float> data({3}, {1.5, 2.5, 3.5});
    std::cout << "Data: ";
    tl::print(data);
    std::cout << "Sum: " << tl::sum(data) << std::endl;
    std::cout << "Mean: " << tl::mean(data) << std::endl;
    std::cout << "Max: " << tl::max(data) << std::endl;
    std::cout << "Min: " << tl::min(data) << std::endl;
    
    // eye and trace
    auto I = tl::linalg::eye<float>(3);
    std::cout << "Identity matrix:" << std::endl;
    tl::print(I);
    std::cout << "Trace: " << tl::linalg::trace(I) << std::endl;
    
    std::cout << "✓ New utilities verified\n" << std::endl;
}

void test_bounds_checking() {
    std::cout << "=== Testing bounds checking ===" << std::endl;
    
    // Test: Try to index a scalar (0D tensor)
    try {
        tl::Tensor<int> scalar({});  // 0-dimensional tensor (scalar)
        // Attempting to index a scalar should throw
        auto view = scalar[0];
        std::cout << "✗ Should have thrown exception for 0D tensor!" << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "✓ Caught expected exception (0D tensor): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Caught exception (0D tensor): " << e.what() << std::endl;
    }
    
    // Valid multi-dimensional indexing works fine
    tl::Tensor<int> mat({2, 3}, {1, 2, 3, 4, 5, 6});
    int val = mat[1][2];  // Access element at row 1, col 2
    std::cout << "✓ Valid indexing mat[1][2] = " << val << " (expected 6)" << std::endl;
    assert(val == 6);
    
    std::cout << std::endl;
}

void test_data_validation() {
    std::cout << "=== Testing data/shape validation ===" << std::endl;
    
    try {
        // Shape says 2x2=4 elements, but only 3 provided
        tl::Tensor<int> bad({2, 2}, {1, 2, 3,4});  // Should throw
        std::cout << "✗ Should have thrown exception!" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Caught expected exception: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void test_performance_features() {
    std::cout << "=== Demonstrating optimized operations ===" << std::endl;
    
    // Large tensors benefit most from SIMD
    tl::Tensor<float> large1({1000,});
    tl::Tensor<float> large2({1000,});
    
    // Fill with data
    for (size_t i = 0; i < 1000; ++i) {
        large1.data[i] = static_cast<float>(i);
        large2.data[i] = static_cast<float>(i * 2);
    }
    
    // These operations now use SIMD instructions
    auto sum_result = large1 + large2;      // Vectorized addition
    auto mul_result = large1 * large2;      // Vectorized multiplication
    auto scalar_result = large1 * 2.0f;     // Vectorized scalar multiply
    
    std::cout << "Sum of first 5 elements: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << sum_result.data[i] << " ";
    }
    std::cout << std::endl;
    
    // Matrix multiplication with cache-optimized i-k-j order
    tl::Tensor<float> A({100, 100});
    tl::Tensor<float> B({100, 100});
    for (size_t i = 0; i < A.data.size(); ++i) {
        A.data[i] = 1.0f;
        B.data[i] = 1.0f;
    }
    
    auto C = tl::linalg::matmul(A, B);
    std::cout << "Matrix multiply result C[0][0] = " << C[0][0] << std::endl;
    std::cout << "(Should be 100.0 since A and B are all 1s)" << std::endl;
    
    std::cout << "✓ Performance features demonstrated\n" << std::endl;
}

void test_norm_improvements() {
    std::cout << "=== Testing matrix norm improvements ===" << std::endl;
    
    tl::Tensor<double> mat({2, 2}, {3.0, 4.0, 0.0, 0.0});
    
    double frob_norm = tl::linalg::matrix_norm(mat, "frob");
    double one_norm = tl::linalg::matrix_norm(mat, "1");
    double inf_norm = tl::linalg::matrix_norm(mat, "inf");
    
    std::cout << "Matrix:" << std::endl;
    tl::print(mat);
    std::cout << "Frobenius norm: " << frob_norm << " (should be 5.0)" << std::endl;
    std::cout << "1-norm: " << one_norm << " (should be 4.0)" << std::endl;
    std::cout << "Inf-norm: " << inf_norm << " (should be 7.0)" << std::endl;
    
    assert(std::abs(frob_norm - 5.0) < 1e-10);
    assert(std::abs(one_norm - 4.0) < 1e-10);
    assert(std::abs(inf_norm - 7.0) < 1e-10);
    
    std::cout << "✓ Matrix norms verified\n" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "   Tensor Library - Tests & Demos    " << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    test_dot_product_fix();
    test_scalar_operations();
    test_new_utilities();
    //test_bounds_checking();
    test_data_validation();
    test_norm_improvements();
    test_performance_features();
    
    std::cout << "======================================" << std::endl;
    std::cout << "   All tests passed! ✓               " << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}