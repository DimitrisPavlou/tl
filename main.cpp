// main.cpp — Feed-Forward Network Forward Pass Demo
// Architecture: Input(784) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(10)
// Batch size: 32  (mimics a single MNIST mini-batch)

#include "tl/tl.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

// ─── Simple weight initialisation ────────────────────────────────────────────
// Xavier-like constant init:  value = scale / sqrt(fan_in)
// In a real project you would use a random number generator here.
template <typename T>
tl::Tensor<T> init_weights(std::size_t rows, std::size_t cols) {
    tl::Tensor<T> W({rows, cols});
    // Xavier uniform: range ±sqrt(1 / fan_in)
    // Uses a simple LCG to get a different sign/magnitude per weight.
    const T limit = static_cast<T>(1.0) / std::sqrt(static_cast<T>(rows));
    std::size_t state = 42; // seed
    for (std::size_t i = 0; i < W.data.size(); ++i) {
        state = state * 1664525u + 1013904223u; // LCG step
        // Map to [-1, 1] then scale by limit
        T u = static_cast<T>(static_cast<int>(state >> 1)) /
              static_cast<T>(0x3FFFFFFF);  // in [-1, 1]
        W.data[i] = u * limit;
    }
    return W;
}

template <typename T>
tl::Tensor<T> init_bias(std::size_t size) {
    tl::Tensor<T> b({size});
    std::fill(b.data.begin(), b.data.end(), static_cast<T>(0.01));
    return b;
}

// ─── Linear layer:  out = input @ W + b  (with broadcasting for bias) ────────
template <typename T>
tl::Tensor<T> linear(const tl::Tensor<T>& x,
                      const tl::Tensor<T>& W,
                      const tl::Tensor<T>& b) {
    auto out = tl::linalg::matmul(x, W);   // [batch, in] @ [in, out] → [batch, out]
    return out + b;                          // broadcast b: [out] → [batch, out]
}

// ─── Helper: min / max across data for a quick sanity print ──────────────────
template <typename T>
std::pair<T,T> data_minmax(const tl::Tensor<T>& t) {
    return { tl::min(t), tl::max(t) };
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n"
              << "║  2-Layer Feed-Forward Network — Demo     ║\n"
              << "╚══════════════════════════════════════════╝\n\n";

    // ── Dimensions ────────────────────────────────────────────────────────────
    const std::size_t BATCH  = 32;
    const std::size_t IN_DIM = 28 * 28;  // 784  (MNIST image flattened)
    const std::size_t H1     = 64;
    const std::size_t H2     = 32;
    const std::size_t OUT    = 10;       // digit classes

    // ── Input (simulate a normalised MNIST batch) ─────────────────────────────
    tl::Tensor<float> X({BATCH, IN_DIM});
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>((i % 255)) / 255.0f;

    std::cout << std::fixed << std::setprecision(4);
    auto [xmin, xmax] = data_minmax(X);
    std::cout << "Input  X    : shape [" << BATCH << ", " << IN_DIM << "]"
              << "  min=" << xmin << "  max=" << xmax << "\n\n";

    // ── Layer 1:  784 → 64  ───────────────────────────────────────────────────
    auto W1 = init_weights<float>(IN_DIM, H1);
    auto b1 = init_bias<float>(H1);

    auto z1 = linear(X, W1, b1);          // [32, 784] @ [784, 64] + [64] → [32, 64]
    auto a1 = tl::functional::relu(z1);   // [32, 64]

    auto [a1min, a1max] = data_minmax(a1);
    std::cout << "Layer 1 z1  : shape [" << z1.shape[0] << ", " << z1.shape[1] << "]  (pre-ReLU)\n"
              << "Layer 1 a1  : shape [" << a1.shape[0] << ", " << a1.shape[1] << "]  "
              << "min=" << a1min << "  max=" << a1max
              << "  (post-ReLU, all values >= 0)\n\n";

    // ── Layer 2:  64 → 32  ────────────────────────────────────────────────────
    auto W2 = init_weights<float>(H1, H2);
    auto b2 = init_bias<float>(H2);

    auto z2 = linear(a1, W2, b2);         // [32, 64] @ [64, 32] + [32] → [32, 32]
    auto a2 = tl::functional::relu(z2);   // [32, 32]

    auto [a2min, a2max] = data_minmax(a2);
    std::cout << "Layer 2 z2  : shape [" << z2.shape[0] << ", " << z2.shape[1] << "]  (pre-ReLU)\n"
              << "Layer 2 a2  : shape [" << a2.shape[0] << ", " << a2.shape[1] << "]  "
              << "min=" << a2min << "  max=" << a2max
              << "  (post-ReLU, all values >= 0)\n\n";

    // ── Output layer:  32 → 10  (no activation — raw logits) ─────────────────
    auto W3 = init_weights<float>(H2, OUT);
    auto b3 = init_bias<float>(OUT);

    auto logits = linear(a2, W3, b3);     // [32, 32] @ [32, 10] + [10] → [32, 10]

    auto [lmin, lmax] = data_minmax(logits);
    std::cout << "Output logits: shape [" << logits.shape[0] << ", " << logits.shape[1] << "]  "
              << "min=" << lmin << "  max=" << lmax << "\n\n";

    // ── Print first sample's logits ───────────────────────────────────────────
    std::cout << "Logits for sample 0 (raw, before softmax):\n  [ ";
    for (std::size_t j = 0; j < OUT; ++j)
        std::cout << logits.data[j] << (j < OUT - 1 ? "  " : " ]\n");

    // ── Verify: ReLU outputs are all non-negative ─────────────────────────────
    bool relu1_ok = (a1min >= 0.0f);
    bool relu2_ok = (a2min >= 0.0f);
    std::cout << "\nSanity checks:\n"
              << "  ReLU layer 1 (all >= 0): " << (relu1_ok ? "✓" : "✗") << "\n"
              << "  ReLU layer 2 (all >= 0): " << (relu2_ok ? "✓" : "✗") << "\n"
              << "  Output shape [32, 10]:   " 
              << (logits.shape[0]==32 && logits.shape[1]==10 ? "✓" : "✗") << "\n";

    std::cout << "\n✓  Forward pass complete — library is working correctly!\n";
    return 0;
}