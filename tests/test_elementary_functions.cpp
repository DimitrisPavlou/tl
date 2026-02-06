#include "../tl/tl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>


void test_elementary_functions() {
    // 1. Test Exp
    tl::Tensor<float> t1({2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
    auto t_exp = tl::functional::exp(t1);
    assert(std::abs(t_exp.data[0] - 1.0f) < 1e-5);
    assert(std::abs(t_exp.data[1] - 2.71828f) < 1e-5);
    std::cout << "Exp test passed!" << std::endl;

    // 2. Test ReLU (Using SIMD-optimized loop)
    tl::Tensor<float> t2({1, 4}, {-10.0f, 0.0f, 5.0f, -2.0f});
    auto t_relu = tl::functional::relu(t2);
    assert(t_relu.data[0] == 0.0f);
    assert(t_relu.data[1] == 0.0f);
    assert(t_relu.data[2] == 5.0f);
    assert(t_relu.data[3] == 0.0f);
    std::cout << "ReLU test passed!" << std::endl;

    // 3. Test Abs
    tl::Tensor<float> t3({2}, {-5.5f, 3.2f});
    auto t_abs = tl::functional::abs(t3);
    assert(t_abs.data[0] == 5.5f);
    assert(t_abs.data[1] == 3.2f);
    std::cout << "Abs test passed!" << std::endl;

    // 4. Test Clip
    tl::Tensor<float> t4({1, 3}, {1.0f, 5.0f, 10.0f});
    auto t_clip = tl::functional::clip(t4, 2.0f, 8.0f);
    assert(t_clip.data[0] == 2.0f); // Clamped to min
    assert(t_clip.data[1] == 5.0f); // Unchanged
    assert(t_clip.data[2] == 8.0f); // Clamped to max
    std::cout << "Clip test passed!" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "   Tensor Library - Tests & Demos    " << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    
    test_elementary_functions();
    std::cout << "======================================" << std::endl;
    std::cout << "   All tests passed! ✓               " << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}