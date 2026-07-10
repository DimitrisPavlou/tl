#pragma once

#include "../tensor_core/tensor.hpp"
#include "../tensor_core/tensor_utils.hpp"
#include "../linalg/linalg_utils.hpp"
#include "../functional/functions.hpp"
#include <memory>
#include <functional>
#include <vector>
#include <unordered_set>

// Reverse-mode automatic differentiation (define-by-run tape), PyTorch-style.
//
//   using namespace tl::autograd;
//   Var X(x_tensor), W(w_tensor, /*requires_grad=*/true);
//   Var loss = sum(relu(matmul(X, W)));
//   loss.backward();
//   auto dW = W.grad();          // ∂loss/∂W
//
// Each operation records a small backward closure onto the graph. backward()
// walks the graph in reverse topological order, accumulating gradients. All
// values are double for numerical robustness.
namespace tl {
namespace autograd {

using T = double;

struct Node {
    Tensor<T> value;
    Tensor<T> grad;                                    // accumulated gradient
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(const Tensor<T>&)> backward_fn; // propagate grad to parents
    bool requires_grad = false;

    explicit Node(Tensor<T> v, bool rg)
        : value(std::move(v)), grad(value.shape), requires_grad(rg) {
        std::fill(grad.data.begin(), grad.data.end(), T{0});
    }
};

inline std::shared_ptr<Node> make_node(Tensor<T> v, bool rg) {
    return std::make_shared<Node>(std::move(v), rg);
}

// Zero-add helper: dst += src (element-wise, same shape).
inline void add_into(Tensor<T>& dst, const Tensor<T>& src) {
    for (std::size_t i = 0; i < dst.data.size(); ++i) dst.data[i] += src.data[i];
}

class Var {
public:
    std::shared_ptr<Node> node;

    Var() = default;
    explicit Var(Tensor<T> value, bool requires_grad = false)
        : node(make_node(std::move(value), requires_grad)) {}
    explicit Var(std::shared_ptr<Node> n) : node(std::move(n)) {}

    const Tensor<T>& value() const { return node->value; }
    const Tensor<T>& grad()  const { return node->grad; }

    // Seed the output gradient (ones for a scalar) and walk the tape backwards.
    void backward() {
        // Build reverse-topological order via post-order DFS.
        std::vector<Node*> topo;
        std::unordered_set<Node*> seen;
        std::function<void(const std::shared_ptr<Node>&)> dfs =
            [&](const std::shared_ptr<Node>& n) {
                if (!n || seen.count(n.get())) return;
                seen.insert(n.get());
                for (auto& p : n->parents) dfs(p);
                topo.push_back(n.get());
            };
        dfs(node);

        // Seed: d(out)/d(out) = 1 for every element of the output.
        std::fill(node->grad.data.begin(), node->grad.data.end(), T{1});

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Node* n = *it;
            if (n->backward_fn) n->backward_fn(n->grad);
        }
    }

    // Zero all gradients reachable from this node (call between backward passes).
    void zero_grad() {
        std::unordered_set<Node*> seen;
        std::function<void(const std::shared_ptr<Node>&)> dfs =
            [&](const std::shared_ptr<Node>& n) {
                if (!n || seen.count(n.get())) return;
                seen.insert(n.get());
                std::fill(n->grad.data.begin(), n->grad.data.end(), T{0});
                for (auto& p : n->parents) dfs(p);
            };
        dfs(node);
    }
};

// ── Element-wise addition ────────────────────────────────────────────────────
inline Var operator+(const Var& a, const Var& b) {
    Var out(a.value() + b.value(), a.node->requires_grad || b.node->requires_grad);
    out.node->parents = {a.node, b.node};
    auto pa = a.node, pb = b.node;
    out.node->backward_fn = [pa, pb](const Tensor<T>& g) {
        add_into(pa->grad, g);
        add_into(pb->grad, g);
    };
    return out;
}

// ── Element-wise subtraction ─────────────────────────────────────────────────
inline Var operator-(const Var& a, const Var& b) {
    Var out(a.value() - b.value(), a.node->requires_grad || b.node->requires_grad);
    out.node->parents = {a.node, b.node};
    auto pa = a.node, pb = b.node;
    out.node->backward_fn = [pa, pb](const Tensor<T>& g) {
        add_into(pa->grad, g);
        for (std::size_t i = 0; i < pb->grad.data.size(); ++i) pb->grad.data[i] -= g.data[i];
    };
    return out;
}

// ── Element-wise (Hadamard) multiplication ───────────────────────────────────
inline Var operator*(const Var& a, const Var& b) {
    Var out(a.value() * b.value(), a.node->requires_grad || b.node->requires_grad);
    out.node->parents = {a.node, b.node};
    auto pa = a.node, pb = b.node;
    out.node->backward_fn = [pa, pb](const Tensor<T>& g) {
        for (std::size_t i = 0; i < g.data.size(); ++i) {
            pa->grad.data[i] += pb->value.data[i] * g.data[i];
            pb->grad.data[i] += pa->value.data[i] * g.data[i];
        }
    };
    return out;
}

// ── Matrix multiply (2D) ─────────────────────────────────────────────────────
inline Var matmul(const Var& a, const Var& b) {
    Var out(tl::linalg::matmul(a.value(), b.value()),
            a.node->requires_grad || b.node->requires_grad);
    out.node->parents = {a.node, b.node};
    auto pa = a.node, pb = b.node;
    out.node->backward_fn = [pa, pb](const Tensor<T>& g) {
        // dA = g · Bᵀ ; dB = Aᵀ · g
        add_into(pa->grad, tl::linalg::matmul(g, tl::linalg::transpose(pb->value)));
        add_into(pb->grad, tl::linalg::matmul(tl::linalg::transpose(pa->value), g));
    };
    return out;
}

// ── ReLU ─────────────────────────────────────────────────────────────────────
inline Var relu(const Var& a) {
    Var out(tl::functional::relu(a.value()), a.node->requires_grad);
    out.node->parents = {a.node};
    auto pa = a.node;
    out.node->backward_fn = [pa](const Tensor<T>& g) {
        for (std::size_t i = 0; i < g.data.size(); ++i)
            pa->grad.data[i] += (pa->value.data[i] > T{0}) ? g.data[i] : T{0};
    };
    return out;
}

// ── Sum-to-scalar reduction ──────────────────────────────────────────────────
inline Var sum(const Var& a) {
    Tensor<T> s({1});
    s.data[0] = tl::sum(a.value());
    Var out(std::move(s), a.node->requires_grad);
    out.node->parents = {a.node};
    auto pa = a.node;
    out.node->backward_fn = [pa](const Tensor<T>& g) {
        for (std::size_t i = 0; i < pa->grad.data.size(); ++i) pa->grad.data[i] += g.data[0];
    };
    return out;
}

} // namespace autograd
} // namespace tl
