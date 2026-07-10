// bench/bench.cpp — micro-benchmarks for the hot paths.
//
// Build (standalone):
//   g++ -std=c++20 -O3 -march=native -I. bench/bench.cpp -o bench
//   # add -fopenmp to enable multi-threaded GEMM / map
//   ./bench
//
// Or via CMake:  cmake --build build --target bench && ./build/bench
#include "tl/tl.hpp"
#include <chrono>
#include <cstdio>
#include <string>

using clk = std::chrono::high_resolution_clock;

template <typename F>
double time_ms(int iters, F&& f) {
    // Warm-up
    f();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    tl::seed(1);
    std::printf("%-42s %12s\n", "benchmark", "ms/iter");
    std::printf("--------------------------------------------------------\n");

    // ── Matmul 512x512 ──────────────────────────────────────────────────────
    {
        auto A = tl::randn<float>({512, 512});
        auto B = tl::randn<float>({512, 512});
        double ms = time_ms(5, [&] { volatile auto C = tl::linalg::matmul(A, B); (void)C; });
        double gflops = (2.0 * 512 * 512 * 512) / (ms * 1e6);
        std::printf("%-42s %12.3f   (%.1f GFLOP/s)\n", "matmul 512x512 @ 512x512", ms, gflops);
    }

    // ── Batched matmul 32 x (128x256 @ 256x64) ──────────────────────────────
    {
        auto A = tl::randn<float>({32, 128, 256});
        auto W = tl::randn<float>({256, 64});
        double ms = time_ms(20, [&] { volatile auto C = tl::linalg::matmul(A, W); (void)C; });
        std::printf("%-42s %12.3f\n", "batched matmul 32x[128x256 @ 256x64]", ms);
    }

    // ── Element-wise: eager chain vs fused map ──────────────────────────────
    {
        const std::size_t N = 4'000'000;
        auto A = tl::randn<float>({N});
        auto B = tl::randn<float>({N});
        auto C = tl::randn<float>({N});

        double eager = time_ms(20, [&] {
            volatile auto r = (A * B) + C;   // 2 passes + 2 allocations
            (void)r;
        });
        double fused = time_ms(20, [&] {
            volatile auto r = tl::functional::map(
                [](float a, float b, float c) { return a * b + c; }, A, B, C);  // 1 pass
            (void)r;
        });
        std::printf("%-42s %12.3f\n", "elementwise a*b+c  (eager, 4M)", eager);
        std::printf("%-42s %12.3f   (%.2fx)\n", "elementwise a*b+c  (fused map, 4M)",
                    fused, eager / fused);
    }

    // ── Broadcasting: bias add [1024,1024] + [1024] ─────────────────────────
    {
        auto M = tl::randn<float>({1024, 1024});
        auto bias = tl::randn<float>({1024});
        double ms = time_ms(50, [&] { volatile auto r = M + bias; (void)r; });
        std::printf("%-42s %12.3f\n", "broadcast bias add [1024x1024]+[1024]", ms);
    }

    // ── Axis reduction: sum over axis 1 of [2048,2048] ──────────────────────
    {
        auto M = tl::randn<float>({2048, 2048});
        double ms = time_ms(20, [&] { volatile auto r = tl::sum(M, 1); (void)r; });
        std::printf("%-42s %12.3f\n", "sum axis=1 [2048x2048]", ms);
    }

    return 0;
}
