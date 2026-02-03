# tl (Tensor Library) 🚀

`tl` is a lightweight, header-only C++ library designed to provide a multidimensional array experience similar to **NumPy** and **PyTorch**. 

> **Project Status:** ⚠️ Under active development.

The core of `tl` is built on a **stride-based memory mapping** system, allowing for efficient $N$-dimensional indexing and manipulation while maintaining a contiguous memory footprint.

---

## 🎯 The Vision
The ultimate goal of this project is to create a modular C++ ecosystem for numerical computing and scientific simulation. While it begins as a tensor manipulation tool, it is evolving into a suite for solving complex mathematical problems numerically, specifically:

* **Ordinary Differential Equations (ODEs):** Solvers for initial value problems.
* **Partial Differential Equations (PDEs):** Frameworks for finite difference and element methods.
* **Machine Learning Algorithms:** Fast implementations of classic algorithms like KNN and Decision Trees
* **Deep Learning:** Back Propagation Implementation, Feed Forward NNs, Convolutional NNs, ...

---

## ✨ Features (until now)
- [x] **N-Dimensional Support:** Create tensors of any rank (2D, 3D, ..., ND).
- [x] **Recursive Indexing:** Natural C++ syntax for deep access: `tensor[i][j][k][l]`.
- [x] **Contiguous Memory:** Data is stored in a flat `std::vector` for cache-friendly performance.
- [x] **Operator Overloading:** Support for element-wise arithmetic (`+`, `-`, `*`, `/`) and scalar operations.
- [x] **NumPy-Style Printing:** Recursive formatting that mirrors Python’s nested bracket style.
- [x] **Header-Only:** No complex build systems; just include the `tl/` directory.

---

## 🛠️ Project Structure (until now)
```text
tl/
├── view.hpp          # Lightweight window/slice into tensor data
├── tensor.hpp        # Main Tensor class & memory management
├── tensor_utils.hpp  # Factories (zeros, ones) and utility functions
└── tl.hpp            # Master include header
```


🚀 Quick Start
Basic Usage
```C++

#include "tl/tl.hpp"

int main() {
    // Create a 4D tensor: [Batch, Channels, Height, Width]
    auto batch = tl::ones<float>({2, 3, 4, 4});

    // Deep indexing through the View system
    batch[1][1][2][3] = 5.5f;

    // Scalar and Element-wise math
    auto result = (batch * 2.0f) + 1.0f;

    // Python-style recursive print
    tl::print(result);

    return 0;
}
```
Compilation

Ensure the tl directory is in your include path. Since it uses modern C++ features, C++20 or later is recommended.
Bash

g++ -std=c++23 main.cpp -o main
./main

🗺️ Roadmap
```text
    [ ] Slicing: Support for range-based sub-views (e.g., tensor(tl::Slice(0, 5))).

    [ ] Randomization: Uniform and Gaussian distribution factories.

    [ ] Linear Algebra: Matrix multiplication (GEMM) and transposition.

    [ ] ODE Solvers: Implementation of Runge-Kutta (RK4) methods.

    [ ] PDE Solvers: Laplace and Heat equation numerical approximations.
```
🤝 Contributing

Contributions are welcome! If you're interested in numerical stability, stride optimization, or adding new mathematical solvers, feel free to open a Pull Request.
