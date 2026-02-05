#include <iostream>
#include <iomanip>
#include "tl/tl.hpp"


int main() {
    auto A = tl::ones<float>({2, 3}); // 2x3 of 1s
    auto B = tl::ones<float>({3, 4}); // 3x4 of 1s

    auto C = tl::matmul(A, B); // Result should be 2x4, all values = 3.0f
    tl::print(C);
    return 0;
}