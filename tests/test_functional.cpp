#include "../tl/tl.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_activations() {
    tl::Tensor<float> t({1, 4}, {-2.0f, -1.0f, 1.0f, 2.0f});
    auto relu_out = tl::functional::relu(t);
    
    assert(relu_out.data[0] == 0.0f);
    assert(relu_out.data[2] == 1.0f);
    std::cout << "Functional ReLU: PASSED\n";
}

void test_math_unary() {
    tl::Tensor<float> t({4}, {1.0f, 4.0f, -2.0f, -1.0f});
    auto sqrt_out = tl::functional::sqrt(t);
    assert(sqrt_out.data[0] == 1.0f);
    assert(sqrt_out.data[1] == 2.0f);
    assert(std::isnan(sqrt_out.data[2]));  // sqrt of negative number is NaN
    assert(std::isnan(sqrt_out.data[3]));  // sqrt of negative number is NaN



    std::cout << "Functional Sqrt: PASSED\n";
}

int main() {
    test_activations();
    test_math_unary();
    return 0;
}