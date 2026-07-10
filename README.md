# tl (Tensor Library) 🚀

`tl` is a lightweight, header-only C++ library designed to provide a multidimensional array experience similar to **NumPy** and **PyTorch**.

> **Project Status:** ⚠️ Under active development.

The core of `tl` is built on a **stride-based memory mapping** system, allowing for efficient $N$-dimensional indexing and manipulation while maintaining a contiguous memory footprint. Storage is **shared-owned**, so the lightweight `View` handles returned by indexing stay valid even if the source tensor is destroyed.

---

## 🎯 The Vision
The ultimate goal of this project is to create a modular C++ ecosystem for numerical computing and scientific simulation. While it begins as a tensor manipulation tool, it is evolving into a suite for solving complex mathematical problems numerically, specifically:

* **Machine Learning & Deep Learning:** reverse-mode autodifferentiation, feed-forward networks, and classic ML algorithms.
* **Ordinary Differential Equations (ODEs):** Runge–Kutta solvers for initial value problems (useful for control-systems research).
* **Partial Differential Equations (PDEs):** frameworks for finite difference / element methods.

---

## ✨ Features
- [x] **N-Dimensional Support:** Create tensors of any rank (2D, 3D, ..., ND).
- [x] **Recursive Indexing:** Natural C++ syntax for deep access: `tensor[i][j][k][l]`.
- [x] **Contiguous Memory:** Data is stored in a flat buffer for cache-friendly performance.
- [x] **Lifetime-Safe Views:** Shared-ownership storage — a `View` keeps its buffer alive even if the parent tensor dies (no dangling pointers).
- [x] **Operator Overloading:** Element-wise arithmetic (`+`, `-`, `*`, `/`) and scalar operations.
- [x] **NumPy-Style Broadcasting:** `matrix + bias`, column/row broadcasting, etc.
- [x] **Linear Algebra:** matmul (2D + **batched** 3D), transpose, trace, identity, matrix norms.
- [x] **Elementwise Math & Activations:** `exp`, `log`, `sqrt`, trig/hyperbolic, `relu`, `sigmoid`, `tanh`, `clip`, ...
- [x] **Axis-Wise Reductions:** `sum`, `mean`, `max`, `min`, `argmax` along any axis (with `keepdims`).
- [x] **Fused Element-Wise `map`:** single-pass, zero-intermediate expression evaluation.
- [x] **Random Factories:** `randn`, `uniform`, `xavier_uniform`, seedable RNG.
- [x] **Slicing:** range-based sub-tensors, `tl::slice(t, {tl::Slice(0, 5)})`.
- [x] **Reverse-Mode Autograd:** define-by-run tape (`Var`, `.backward()`), gradient-checked.
- [x] **ODE Solvers:** generic RK4 (`rk4_step`, `integrate`) + heap-free `SmallTensor<T, N>`.
- [x] **Concept-Constrained:** `Tensor<T>` requires `tl::Numeric<T>` for clear diagnostics.
- [x] **Optional OpenMP:** multi-threaded GEMM and element-wise loops.
- [x] **Header-Only:** No complex build systems; just include the `tl/` directory.

---

## 🛠️ Project Structure
```text
tl/
├── tl.hpp                     # Master include header
├── tensor_core/
│   ├── concepts.hpp           # tl::Numeric concept
│   ├── shared_vec.hpp         # Shared-ownership vector (lifetime-safe views)
│   ├── view.hpp               # Lightweight window/slice into tensor data
│   ├── tensor.hpp             # Main Tensor class & memory management
│   ├── broadcasting.hpp       # NumPy-style broadcast shape/stride rules
│   ├── tensor_utils.hpp       # Factories (zeros/ones/full), reshape, dot, print
│   ├── random.hpp             # randn / uniform / xavier_uniform / seed
│   └── slicing.hpp            # tl::Slice + slice()
├── linalg/
│   └── linalg_utils.hpp       # matmul (2D + batched), transpose, norm, eye, trace
├── functional/
│   ├── functions.hpp          # elementwise math + activations
│   ├── reductions.hpp         # axis-wise sum/mean/max/min/argmax
│   └── map.hpp                # fused N-ary element-wise map
├── autograd/
│   └── autograd.hpp           # reverse-mode automatic differentiation
└── ode/
    ├── small_tensor.hpp       # heap-free fixed-size state vector
    └── rk4.hpp                # Runge–Kutta integrators
```

---

## 🚀 Quick Start

### Basic Usage
```cpp
#include "tl/tl.hpp"

int main() {
    // Create a 4D tensor: [Batch, Channels, Height, Width]
    auto batch = tl::ones<float>({2, 3, 4, 4});

    // Deep indexing through the View system
    batch[1][1][2][3] = 5.5f;

    // Scalar and element-wise math (broadcasting supported)
    auto result = (batch * 2.0f) + 1.0f;

    // Python-style recursive print
    tl::print(result);
    return 0;
}
```

### Linear Algebra & Reductions
```cpp
auto A = tl::randn<float>({64, 128});
auto W = tl::randn<float>({128, 32});
auto Z = tl::linalg::matmul(A, W);      // [64, 32]

auto col_sums = tl::sum(Z, /*axis=*/0);      // shape [32]
auto row_max  = tl::max(Z, /*axis=*/1, /*keepdims=*/true); // shape [64, 1]
auto preds    = tl::argmax(Z, 1);            // shape [64]
```

### Fused Element-Wise Evaluation
```cpp
// (A * B) + C in a SINGLE pass, no intermediate tensors:
auto Y = tl::functional::map(
    [](float a, float b, float c) { return a * b + c; }, A, B, C);
```

### Reverse-Mode Autograd
```cpp
using namespace tl::autograd;

Var X(x_tensor);
Var W(w_tensor, /*requires_grad=*/true);
Var loss = sum(relu(matmul(X, W)));

loss.backward();
const auto& dW = W.grad();     // ∂loss/∂W
```

### ODE Solving (RK4) — Control-Systems Friendly
```cpp
// Harmonic oscillator  x'' = -x   as a first-order system [x, v]
auto f = [](double, tl::SmallTensor<double, 2> y) {
    tl::SmallTensor<double, 2> dy;
    dy[0] = y[1];     // x' = v
    dy[1] = -y[0];    // v' = -x
    return dy;
};

tl::SmallTensor<double, 2> y0; y0[0] = 1.0; y0[1] = 0.0;
auto traj = tl::ode::integrate(f, y0, 0.0, 2 * 3.14159265, 0.001);
```

---

## ⚙️ Building

Requires **C++20 or later** (the library uses concepts).

```bash
# Single-file program
g++ -std=c++20 -O3 -march=native -I. main.cpp -o main
./main

# Enable OpenMP (multi-threaded GEMM / map)
g++ -std=c++20 -O3 -march=native -fopenmp -I. main.cpp -o main
```

### CMake
```bash
cmake -S . -B build
cmake --build build --config Release

./build/main                 # demo        (./build/Release/main.exe on MSVC)
./build/run_all_tests        # test suite
./build/bench                # benchmarks
ctest --test-dir build --output-on-failure
```
OpenMP is linked automatically when `find_package(OpenMP)` succeeds. Configure with `-DTL_SANITIZE=ON` (GCC/Clang) to build with AddressSanitizer + UBSan.

---

## ✅ Testing

A lightweight single-header framework (`tests/test.hpp`) drives **272 checks** across
core tensors, views, ownership/lifetime, broadcasting, linalg, functional math,
reductions, random, slicing, batched matmul, fused `map`, autograd (gradient-checked),
and RK4 (validated against analytic solutions).

```bash
g++ -std=c++20 -O2 -I. tests/run_all_tests.cpp -o run_tests && ./run_tests
```

---

## 📊 Performance

Micro-benchmarks (`bench/bench.cpp`, single machine, `-O3 -march=native`):

| Benchmark                                   | Single-thread | OpenMP        |
|---------------------------------------------|--------------:|--------------:|
| `matmul` 512×512 @ 512×512                   | ~50 GFLOP/s   | ~187 GFLOP/s  |
| element-wise `a*b+c` (eager vs fused `map`)  | baseline      | **1.6–1.8×**  |

`#pragma omp simd` / `parallel for` hints are active only when compiled with
`-fopenmp` (GCC/Clang) or `/openmp` (MSVC); the code is correct either way.

---

## 🗺️ Roadmap
```text
    [x] Slicing:        range-based sub-views (tl::slice(t, {tl::Slice(0, 5)})).
    [x] Randomization:  uniform and Gaussian distribution factories.
    [x] Linear Algebra: matrix multiplication (GEMM), batched matmul, transpose.
    [x] Autograd:       reverse-mode automatic differentiation.
    [x] ODE Solvers:    Runge-Kutta (RK4) methods.
    [ ] Zero-copy strided views for slice/transpose (needs stride-aware ops).
    [ ] Expression templates for whole-expression fusion.
    [ ] Broadcasting-aware autograd + optimizers (SGD/Adam) and NN layer API.
    [ ] PDE Solvers:    Laplace and Heat equation numerical approximations.
    [ ] GPU / BLAS backend.
```

---

## 🤝 Contributing

Contributions are welcome! If you're interested in numerical stability, stride
optimization, autograd operators, or adding new mathematical solvers, feel free
to open a Pull Request.
