#include "../tl/tl.hpp"
#include <cassert>
#include <iostream>

void test_initialization() {
    // Test 2D Init
    tl::Tensor<float> t({{1, 2}, {3, 4}});
    assert(t.shape[0] == 2);
    assert(t.shape[1] == 2);
    assert(t.data[0] == 1.0f);
    assert(t.data[3] == 4.0f);
    std::cout << "Tensor Initialization: PASSED\n";
}

void test_strides() {
    tl::Tensor<int> t({3, 4, 5});
    // For a row-major tensor, strides should be {20, 5, 1}
    assert(t.strides[0] == 20);
    assert(t.strides[1] == 5);
    assert(t.strides[2] == 1);
    std::cout << "Stride Calculation: PASSED\n";
}

int main() {
    test_initialization();
    test_strides();
    return 0;
}