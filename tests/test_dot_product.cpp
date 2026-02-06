#include "../tl/tl.hpp"
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

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "   Tensor Library - Tests & Demos    " << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    test_dot_product_fix();

    std::cout << "======================================" << std::endl;
    std::cout << "   All tests passed! ✓               " << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}