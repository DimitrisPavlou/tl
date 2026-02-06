#include "../tl/tl.hpp"
#include <cassert>
#include <iostream>

void test_matmul() {
    tl::Tensor<float> A({{1, 2}, {3, 4}});
    tl::Tensor<float> B({{5, 6}, {7, 8}});
    
    // Result should be {{19, 22}, {43, 50}}
    auto C = tl::linalg::matmul(A, B);
    
    assert(C.data[0] == 19.0f);
    assert(C.data[3] == 50.0f);
    std::cout << "Linalg Matmul: PASSED\n";
}

void test_identity() {
    auto I = tl::linalg::eye<float>(3);
    assert(I.data[0] == 1.0f);
    assert(I.data[1] == 0.0f);
    assert(I.data[4] == 1.0f);
    std::cout << "Linalg Eye: PASSED\n";
}

int main() {
    test_matmul();
    test_identity();
    return 0;
}